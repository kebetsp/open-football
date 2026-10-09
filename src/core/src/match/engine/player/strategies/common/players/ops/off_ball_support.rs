//! Off-ball support spots — a continuously re-evaluated choice instead of
//! a fixed point (2026-10-08, Pavel: "central midfielders get stuck passing
//! to each other ... the rest of the team stays static ... standing still
//! is bad overall, and should only happen at set pieces ... teammates
//! should come and help their CMs").
//!
//! Why this exists. `attacking_support.rs` gives every off-ball teammate a
//! target that is a constant per role and phase (plus 25% of the ball's y).
//! Once a player has arrived nothing ever moves him again, so while the
//! ball is passed back and forth in one zone the team freezes (measured:
//! inside middle-third possession loops of 15s or more, 85-100% of the ball
//! team's back line and strikers stood still). Real off-ball movement is a
//! response to the ball, the defenders and the other teammates: a player
//! keeps adjusting so the carrier has a clear, useful, unmarked option.
//!
//! What it does. For each off-ball teammate, starting from the old target
//! (the role/phase anchor), it scores a small set of nearby spots and takes
//! the best one:
//!   * a clear passing lane from the carrier (opponents away from the line),
//!   * space around the spot (nearest opponent),
//!   * a support distance band (10-25m from the carrier; closer when the
//!     team is stuck),
//!   * not on top of a teammate (including spots already taken this tick),
//!   * a tether to the old target (role discipline, widened for helpers),
//!   * not beyond the opponent's last line (offside),
//!   * stickiness to the previously chosen spot, switched off once he has
//!     stood in the same place too long (dwell limit) — that is what makes
//!     movement never stop: a spot held too long loses its value.
//! When a side's possession has not progressed for `HELP_AFTER_MS`, the
//! nearest forward and the nearest wide player become helpers: their tether
//! is relaxed and their distance band moves close to the carrier, so someone
//! comes to make the numerical advantage in the carrier's zone.
//!
//! Process check (decision-from-null): uses only raw positions, pitch size
//! and time; calls no earlier decision function.

use nalgebra::Vector3;

/// Angles tried around the old target.
const ANGLES: usize = 8;
/// Ring radii (units, 8 per metre): 6m, 12m, 18m.
const RADII: [f32; 3] = [48.0, 96.0, 144.0];
/// Extra rings tried by helpers: 24m, 30m.
const HELPER_EXTRA_RADII: [f32; 2] = [192.0, 240.0];
/// Tether: how far from the old target a normal player may roam (15m) and a helper (35m).
const LEASH: f32 = 120.0;
const HELPER_LEASH: f32 = 280.0;
/// Passing-lane clearance: no credit under 2m of margin, full credit at 7m.
const LANE_MIN: f32 = 16.0;
const LANE_FULL: f32 = 56.0;
/// Space around the spot: full credit at 8m from the nearest opponent.
const SPACE_FULL: f32 = 64.0;
/// Support band from the carrier, normal: 10-25m. Helper: 8-16m.
const BAND: (f32, f32) = (80.0, 200.0);
const HELPER_BAND: (f32, f32) = (64.0, 128.0);
/// Teammates closer than 7m to a spot make it worse.
const CROWD_RADIUS: f32 = 56.0;
/// Standing within 5m of one point for this long forces a relocation.
pub const DWELL_LIMIT_MS: u64 = 2500;
const DWELL_RADIUS: f32 = 40.0;
/// A spot within 3m of the previous choice keeps a small bonus.
const STICKY_RADIUS: f32 = 24.0;
/// Possession without progress for this long makes help arrive.
pub const HELP_AFTER_MS: u64 = 3000;
/// A spot this close to the opponent's last line (or beyond it) is offside.
const OFFSIDE_MARGIN: f32 = 8.0;

pub struct SupportWorld<'a> {
    pub carrier: Vector3<f32>,
    pub attack_dir: f32,
    pub opponents: &'a [Vector3<f32>],
    /// Every other outfield teammate (id, current position), carrier included.
    pub teammates: &'a [(u32, Vector3<f32>)],
    /// Absolute x of the opponent's last outfield defender.
    pub opp_last_line_x: Option<f32>,
    pub width: f32,
    pub height: f32,
    pub now_ms: u64,
}

pub struct SupportPlayer {
    pub id: u32,
    pub position: Vector3<f32>,
    pub base: Vector3<f32>,
    pub prev_spot: Option<Vector3<f32>>,
    pub anchor: Vector3<f32>,
    pub anchor_since_ms: u64,
    pub helper: bool,
}

fn seg_dist(p: Vector3<f32>, a: Vector3<f32>, b: Vector3<f32>) -> f32 {
    let ab = b - a;
    let l2 = ab.x * ab.x + ab.y * ab.y;
    if l2 < 1.0 {
        return ((p.x - a.x).powi(2) + (p.y - a.y).powi(2)).sqrt();
    }
    let t = (((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / l2).clamp(0.0, 1.0);
    let cx = a.x + ab.x * t;
    let cy = a.y + ab.y * t;
    ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt()
}

fn dist2d(a: Vector3<f32>, b: Vector3<f32>) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// Whether the player has stood in the same place for too long.
pub fn dwell_expired(p: &SupportPlayer, now_ms: u64) -> bool {
    dist2d(p.position, p.anchor) < DWELL_RADIUS
        && now_ms.saturating_sub(p.anchor_since_ms) > DWELL_LIMIT_MS
}

/// Pick the best support spot for one player.
pub fn choose_spot(w: &SupportWorld, p: &SupportPlayer, taken: &[Vector3<f32>]) -> Vector3<f32> {
    let expired = dwell_expired(p, w.now_ms);
    let leash = if p.helper { HELPER_LEASH } else { LEASH };
    let band = if p.helper { HELPER_BAND } else { BAND };
    let clamp = |v: Vector3<f32>| {
        Vector3::new(v.x.clamp(24.0, w.width - 24.0), v.y.clamp(24.0, w.height - 24.0), 0.0)
    };

    let mut cands: Vec<Vector3<f32>> = vec![clamp(p.base)];
    let mut radii: Vec<f32> = RADII.to_vec();
    if p.helper {
        radii.extend_from_slice(&HELPER_EXTRA_RADII);
    }
    for &r in &radii {
        for k in 0..ANGLES {
            let a = (k as f32) * std::f32::consts::TAU / ANGLES as f32;
            cands.push(clamp(Vector3::new(p.base.x + r * a.cos(), p.base.y + r * a.sin(), 0.0)));
        }
    }

    let mut best = cands[0];
    let mut best_score = f32::MIN;
    for c in cands {
        // Passing lane from the carrier.
        let lane_gap = w
            .opponents
            .iter()
            .map(|o| seg_dist(*o, w.carrier, c))
            .fold(f32::MAX, f32::min);
        let lane = ((lane_gap - LANE_MIN) / (LANE_FULL - LANE_MIN)).clamp(0.0, 1.0);
        // Space around the spot.
        let opp_gap = w.opponents.iter().map(|o| dist2d(*o, c)).fold(f32::MAX, f32::min);
        let space = (opp_gap / SPACE_FULL).clamp(0.0, 1.0);
        // Support band from the carrier.
        let d = dist2d(c, w.carrier);
        let band_v = if d < band.0 {
            ((d - 24.0) / (band.0 - 24.0)).clamp(0.0, 1.0)
        } else if d > band.1 {
            (1.0 - (d - band.1) / 160.0).clamp(0.0, 1.0)
        } else {
            1.0
        };
        // Mild preference for spots ahead of the ball.
        let ahead = (((c.x - w.carrier.x) * w.attack_dir) / 160.0).clamp(-1.0, 1.0);
        // Crowding: nearest teammate (not himself) or already-taken spot.
        let mut crowd_gap = f32::MAX;
        for (id, pos) in w.teammates {
            if *id != p.id {
                crowd_gap = crowd_gap.min(dist2d(*pos, c));
            }
        }
        for t in taken {
            crowd_gap = crowd_gap.min(dist2d(*t, c));
        }
        let crowd = if crowd_gap < CROWD_RADIUS { (CROWD_RADIUS - crowd_gap) / CROWD_RADIUS } else { 0.0 };
        // Tether to the old target.
        let tether = dist2d(c, p.base) / leash;
        // Offside.
        let offside = match w.opp_last_line_x {
            Some(lx) => ((c.x - lx) * w.attack_dir > -OFFSIDE_MARGIN) as i32 as f32,
            None => 0.0,
        };
        // Stickiness and dwell.
        let mut sticky = 0.0;
        if let Some(prev) = p.prev_spot {
            if !expired && dist2d(c, prev) < STICKY_RADIUS {
                sticky = 0.25;
            }
        }
        let stale = if expired && dist2d(c, p.anchor) < DWELL_RADIUS { 0.9 } else { 0.0 };

        let score = 1.0 * lane + 0.6 * space + 0.8 * band_v + 0.3 * ahead - 1.0 * crowd - 0.8 * tether
            - 1.0 * offside
            + sticky
            - stale;
        if score > best_score {
            best_score = score;
            best = c;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Defensive side: the same idea for a defending player who is not pressing,
// marking or covering — he keeps adjusting his spot inside the shape so the
// carrier's options are cut, instead of standing on a fixed point.
// ---------------------------------------------------------------------------

/// Defensive ring radii (4m, 8m, 12m) and tether (12m): small adjustments
/// inside the shape, never a different job.
const DEF_RADII: [f32; 3] = [32.0, 64.0, 96.0];
const DEF_LEASH: f32 = 96.0;
/// Only opponents between 5m and 35m from the carrier count as options.
const OPTION_MIN: f32 = 40.0;
const OPTION_MAX: f32 = 280.0;
/// Lane cover: full credit within 2m of the carrier-to-option line, none at 8m.
const COVER_FULL: f32 = 16.0;
const COVER_NONE: f32 = 64.0;
/// Tight-but-not-stuck: best distance to the nearest opponent is 5m.
const MARK_IDEAL: f32 = 40.0;
const MARK_SPREAD: f32 = 64.0;

pub struct DefWorld<'a> {
    pub carrier: Vector3<f32>,
    pub own_goal: Vector3<f32>,
    /// Every opposing outfield player except the carrier.
    pub options: &'a [Vector3<f32>],
    pub teammates: &'a [(u32, Vector3<f32>)],
    pub width: f32,
    pub height: f32,
    pub now_ms: u64,
}

pub fn choose_defensive_spot(w: &DefWorld, p: &SupportPlayer, taken: &[Vector3<f32>]) -> Vector3<f32> {
    let expired = dwell_expired(p, w.now_ms);
    let clamp = |v: Vector3<f32>| {
        Vector3::new(v.x.clamp(24.0, w.width - 24.0), v.y.clamp(24.0, w.height - 24.0), 0.0)
    };
    let mut cands: Vec<Vector3<f32>> = vec![clamp(p.base)];
    for &r in &DEF_RADII {
        for k in 0..ANGLES {
            let a = (k as f32) * std::f32::consts::TAU / ANGLES as f32;
            cands.push(clamp(Vector3::new(p.base.x + r * a.cos(), p.base.y + r * a.sin(), 0.0)));
        }
    }
    let mut best = cands[0];
    let mut best_score = f32::MIN;
    for c in cands {
        // Lane cover: stand on the line between the carrier and a dangerous option.
        let mut cover = 0.0f32;
        for o in w.options {
            let d = dist2d(*o, w.carrier);
            if !(OPTION_MIN..=OPTION_MAX).contains(&d) {
                continue;
            }
            let gap = seg_dist(c, w.carrier, *o);
            let v = 1.0 - ((gap - COVER_FULL) / (COVER_NONE - COVER_FULL)).clamp(0.0, 1.0);
            // An option nearer our goal is more dangerous.
            let danger = (1.3 - dist2d(*o, w.own_goal) / 600.0).clamp(0.4, 1.2);
            cover = cover.max(v * danger);
        }
        // Tight-but-not-stuck distance to the nearest opponent.
        let nearest = w.options.iter().map(|o| dist2d(*o, c)).fold(f32::MAX, f32::min);
        let mark = 1.0 - ((nearest - MARK_IDEAL).abs() / MARK_SPREAD).clamp(0.0, 1.0);
        // Crowding with teammates / spots already taken.
        let mut crowd_gap = f32::MAX;
        for (id, pos) in w.teammates {
            if *id != p.id {
                crowd_gap = crowd_gap.min(dist2d(*pos, c));
            }
        }
        for t in taken {
            crowd_gap = crowd_gap.min(dist2d(*t, c));
        }
        let crowd = if crowd_gap < CROWD_RADIUS { (CROWD_RADIUS - crowd_gap) / CROWD_RADIUS } else { 0.0 };
        let tether = dist2d(c, p.base) / DEF_LEASH;
        let mut sticky = 0.0;
        if let Some(prev) = p.prev_spot {
            if !expired && dist2d(c, prev) < STICKY_RADIUS {
                sticky = 0.25;
            }
        }
        let stale = if expired && dist2d(c, p.anchor) < DWELL_RADIUS { 0.9 } else { 0.0 };
        let score = 1.0 * cover + 0.4 * mark - 1.0 * crowd - 0.8 * tether + sticky - stale;
        if score > best_score {
            best_score = score;
            best = c;
        }
    }
    best
}
