//! Off-ball support spacing (§10.3, wishlist #17).
//!
//! A shared refinement pass over a state's own heuristic support target:
//! score candidate positions around the proposed target so support runs
//! spread out (maximise collectively covered space) and favour positions
//! that open a longer, cleaner passing lane from the ball carrier — rather
//! than bunching a trivial couple of metres from the ball. The proven
//! forward `find_optimal_free_zone` scoring terms are reused here in a
//! role-neutral form; the caller's heuristic target anchors the search so
//! each role's zone discipline and tactical intent stay intact.

use crate::r#match::{MatchPlayerLite, StateProcessingContext};
use nalgebra::Vector3;

/// Radius inside which a candidate is penalised per nearby teammate —
/// same 90u repulsion radius the forward zone scorer uses.
const TEAMMATE_REPULSION_RADIUS: f32 = 90.0;
/// Weight of the summed linear teammate repulsion (forward scorer value).
const TEAMMATE_REPULSION_WEIGHT: f32 = 50.0;
/// Support closer to the holder than this is "short and redundant" —
/// deprioritised unless a genuine combination (link play) is in progress.
const SHORT_SUPPORT_RADIUS: f32 = 28.0;
/// Candidates further from the proposed target pay a mild tether cost so
/// the refinement repositions within the state's intent, never wanders.
const TETHER_WEIGHT: f32 = 0.15;

/// §11.9 hard exclusion radius (~3m at the ~8u/m conversion): no two
/// teammates may target the same spot or the same small area around it.
/// Unlike the soft repulsion above, a candidate inside this radius of a
/// claimed point is EXCLUDED outright, not merely down-scored — the
/// penalty approach alone demonstrably failed to stop actual overlap.
pub const EXCLUSION_RADIUS: f32 = 24.0;
/// Run-destination projection horizon in engine ticks (~0.8s at 10ms/tick;
/// velocities are u/tick). A moving player "owns" the space he is running
/// into, not just the spot he stands on.
const PROJECTION_TICKS: f32 = 80.0;

/// Points already claimed by the rest of the team (§11.9): the ball
/// carrier's projected run destination, plus every other teammate's
/// current position AND projected destination.
pub fn claimed_points(ctx: &StateProcessingContext) -> Vec<Vector3<f32>> {
    let mut pts = Vec::with_capacity(24);
    if let Some(h) = ball_holder(ctx) {
        pts.push(h.position + h.velocity(ctx) * PROJECTION_TICKS);
    }
    for t in ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
    {
        let v = t.velocity(ctx);
        pts.push(t.position);
        pts.push(t.position + v * PROJECTION_TICKS);
    }
    pts
}

/// Diagnostic counters for the §11.9 exclusion (match-logs builds only) —
/// same opt-in pattern as `tackle_stats`. Proves the hard rule actually
/// fires at the mechanism level, since aggregate movement proxies are
/// dominated by paths this rule deliberately doesn't touch.
#[cfg(feature = "match-logs")]
pub mod spacing_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Candidates dropped by the hard exclusion.
    pub static EXCLUDED: AtomicU64 = AtomicU64::new(0);
    /// RunningInBehind targets shifted out of the carrier's space.
    pub static CARRIER_SHIFTS: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        EXCLUDED.store(0, Ordering::Relaxed);
        CARRIER_SHIFTS.store(0, Ordering::Relaxed);
    }

    pub fn snapshot() -> [u64; 2] {
        [
            EXCLUDED.load(Ordering::Relaxed),
            CARRIER_SHIFTS.load(Ordering::Relaxed),
        ]
    }
}

/// True when `candidate` falls inside the hard exclusion radius of any
/// claimed point.
pub fn violates_exclusion(candidate: Vector3<f32>, claimed: &[Vector3<f32>]) -> bool {
    let r_sq = EXCLUSION_RADIUS * EXCLUSION_RADIUS;
    let hit = claimed
        .iter()
        .any(|c| (c - candidate).norm_squared() < r_sq);
    #[cfg(feature = "match-logs")]
    if hit {
        spacing_stats::EXCLUDED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    hit
}

/// §11.9 carrier-space guard for direct run targets (e.g. RunningInBehind,
/// which doesn't go through candidate scoring): if `target` converges on
/// the space the ball carrier is already running into, shift it laterally
/// to just outside the exclusion radius. The carrier owns that space.
pub fn avoid_carrier_space(
    ctx: &StateProcessingContext,
    target: Vector3<f32>,
) -> Vector3<f32> {
    let Some(h) = ball_holder(ctx) else {
        return target;
    };
    if ctx.player.link_target == Some(h.id) {
        return target; // a one-two deliberately plays into the carrier's space
    }
    let carrier_dest = h.position + h.velocity(ctx) * PROJECTION_TICKS;
    let d = target - carrier_dest;
    let dist_sq = d.x * d.x + d.y * d.y;
    if dist_sq >= EXCLUSION_RADIUS * EXCLUSION_RADIUS {
        return target;
    }
    #[cfg(feature = "match-logs")]
    spacing_stats::CARRIER_SHIFTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let field_height = ctx.context.field_size.height as f32;
    let away: f32 = if target.y >= carrier_dest.y { 1.0 } else { -1.0 };
    let y = (carrier_dest.y + away * (EXCLUSION_RADIUS + 4.0)).clamp(15.0, field_height - 15.0);
    Vector3::new(target.x, y, 0.0)
}

/// Refine a state's proposed off-ball support target by scoring candidates
/// around it. Returns the best-scoring position (possibly `proposed`
/// itself). Callers pass the target their own heuristics produced.
///
/// Skipped (returns `proposed` unchanged) when the player is in a genuine
/// combination with the ball holder (`link_target` pins them together) —
/// a one-two needs close support, and that case must not be broken.
pub fn refine_support_position(
    ctx: &StateProcessingContext,
    proposed: Vector3<f32>,
) -> Vector3<f32> {
    let holder = ball_holder(ctx);

    if let Some(h) = &holder {
        if ctx.player.link_target == Some(h.id) {
            return proposed;
        }
    }

    let field_width = ctx.context.field_size.width as f32;
    let field_height = ctx.context.field_size.height as f32;
    let goal_pos = ctx.player().opponent_goal_position();

    let opponents: Vec<Vector3<f32>> = ctx
        .players()
        .opponents()
        .all()
        .map(|o| o.position)
        .collect();
    let teammates: Vec<(u32, Vector3<f32>)> = ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
        .map(|t| (t.id, t.position))
        .collect();

    // §11.9 hard exclusion: candidates inside ~3m of the carrier's run
    // destination or any teammate's position/destination are dropped
    // outright before scoring. The soft repulsion below still shapes
    // preferences among the survivors.
    let claimed = claimed_points(ctx);

    // Candidate ring around the proposed target (25 positions), plus the
    // player's current position so "stay put" competes fairly.
    let mut best = proposed;
    let mut best_score = f32::MIN;
    // Fallback if EVERY candidate is excluded (dense box, all space
    // claimed): best-scoring candidate regardless of exclusion beats
    // returning an unvetted `proposed`.
    let mut best_any = proposed;
    let mut best_any_score = f32::MIN;
    let mut found_valid = false;
    let offsets: [f32; 5] = [-50.0, -25.0, 0.0, 25.0, 50.0];
    let mut consider = |candidate: Vector3<f32>| {
        let clamped = Vector3::new(
            candidate.x.clamp(20.0, field_width - 20.0),
            candidate.y.clamp(20.0, field_height - 20.0),
            0.0,
        );
        let score = score_candidate(
            clamped, proposed, &holder, goal_pos, &opponents, &teammates,
        );
        if score > best_any_score {
            best_any_score = score;
            best_any = clamped;
        }
        if violates_exclusion(clamped, &claimed) {
            return;
        }
        found_valid = true;
        if score > best_score {
            best_score = score;
            best = clamped;
        }
    };
    for dx in offsets {
        for dy in offsets {
            consider(proposed + Vector3::new(dx, dy, 0.0));
        }
    }
    consider(ctx.player.position);

    if found_valid { best } else { best_any }
}

fn score_candidate(
    position: Vector3<f32>,
    proposed: Vector3<f32>,
    holder: &Option<MatchPlayerLite>,
    goal_pos: Vector3<f32>,
    opponents: &[Vector3<f32>],
    teammates: &[(u32, Vector3<f32>)],
) -> f32 {
    let mut score = 0.0f32;

    // Open space: congestion from opponents within 30u (forward scorer term).
    let mut congestion = 0.0f32;
    for opp in opponents {
        let d = (opp - position).magnitude();
        if d < 30.0 {
            congestion += (30.0 - d) / 30.0;
        }
    }
    score += (10.0 - congestion.min(10.0)) * 3.0;

    // Universal teammate repulsion — crowding ANY teammate wastes the
    // team's collective coverage (forward scorer weight).
    let crowd: f32 = teammates
        .iter()
        .map(|(_, t)| (t - position).magnitude())
        .filter(|&d| d < TEAMMATE_REPULSION_RADIUS)
        .map(|d| (TEAMMATE_REPULSION_RADIUS - d) / TEAMMATE_REPULSION_RADIUS)
        .sum::<f32>();
    score -= crowd * TEAMMATE_REPULSION_WEIGHT;

    if let Some(h) = holder {
        let holder_dist = (position - h.position).magnitude();

        // Short redundant support: a candidate a couple of metres from the
        // carrier offers a trivial pass — deprioritise it firmly.
        if holder_dist < SHORT_SUPPORT_RADIUS {
            score -= (SHORT_SUPPORT_RADIUS - holder_dist) * 2.5;
        }

        // Passing-lane clearance from the holder, and lane QUALITY: a clear
        // lane scores, and a clear lane that progresses the ball toward goal
        // (longer, through the defence) scores more than a sideways one.
        let lane = position - h.position;
        let lane_len = lane.magnitude();
        if lane_len > 1.0 {
            let dir = lane / lane_len;
            let blocked = opponents.iter().any(|opp| {
                let to_opp = opp - h.position;
                let proj = to_opp.dot(&dir);
                if proj <= 0.0 || proj >= lane_len {
                    return false;
                }
                let lateral = to_opp - dir * proj;
                lateral.norm_squared() < 4.0 * 4.0
            });
            if blocked {
                score -= 10.0;
            } else {
                score += 15.0;
                let progress = (h.position - goal_pos).magnitude()
                    - (position - goal_pos).magnitude();
                score += (progress / 120.0).clamp(0.0, 1.0) * 20.0;
            }
        }

        // Width vs the holder stretches the shape (forward scorer term,
        // reduced — mids/defenders also stretch vertically via progress).
        let lateral_distance = (position.y - h.position.y).abs();
        if lateral_distance > 80.0 {
            score += 15.0;
        } else if lateral_distance > 40.0 {
            score += 8.0;
        }
    }

    // Tether: stay recognisably within the caller's tactical intent.
    score -= (position - proposed).magnitude() * TETHER_WEIGHT;

    score
}

fn ball_holder(ctx: &StateProcessingContext) -> Option<MatchPlayerLite> {
    let owner_id = ctx.ball().owner_id()?;
    ctx.players()
        .teammates()
        .all()
        .find(|t| t.id == owner_id)
}
