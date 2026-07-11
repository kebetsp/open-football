//! Corner set-piece coordination.
//!
//! Pure functions — no shared mutable state. Every player independently
//! runs the same deterministic assignment each tick, so the framework
//! needs no message-passing or `MatchContext` mutations.

use crate::r#match::{MatchPlayerLite, StateProcessingContext};
use nalgebra::Vector3;

// ── Zone geometry ─────────────────────────────────────────────────────────────

/// The four delivery zones for a corner kick.
#[derive(Debug, Clone, Copy)]
pub struct CornerZones {
    /// Same lateral side as the taker, ~30u from the goal line.
    pub near_post: Vector3<f32>,
    /// Opposite lateral side from the taker, ~30u from the goal line.
    pub far_post: Vector3<f32>,
    /// Penalty-spot area — classic central aerial target.
    pub penalty_spot: Vector3<f32>,
    /// Edge of the penalty area — second-ball / late runner zone.
    pub edge_of_box: Vector3<f32>,
}

/// Compute the four corner delivery zones from the taker's position, the
/// attacking goal, and the field dimensions.
pub fn corner_zones(
    taker_pos: Vector3<f32>,
    goal_pos: Vector3<f32>,
    field_w: f32,
    field_h: f32,
) -> CornerZones {
    let taker_near_top = taker_pos.y < field_h * 0.5;
    let is_right_goal = goal_pos.x > field_w * 0.5;
    // `inward_x` points away from the goal line toward the centre of the pitch.
    let inward_x: f32 = if is_right_goal { -1.0 } else { 1.0 };
    let gx = goal_pos.x;
    let cy = field_h * 0.5; // lateral centre ≈ 272.5

    let near_y = if taker_near_top { cy - 52.0 } else { cy + 52.0 };
    let far_y  = if taker_near_top { cy + 52.0 } else { cy - 52.0 };

    CornerZones {
        near_post:    Vector3::new(gx + inward_x * 30.0,  near_y, 0.0),
        far_post:     Vector3::new(gx + inward_x * 30.0,  far_y,  0.0),
        penalty_spot: Vector3::new(gx + inward_x * 88.0,  cy,     0.0),
        edge_of_box:  Vector3::new(gx + inward_x * 132.0, near_y, 0.0),
    }
}

// ── Deterministic assignment ──────────────────────────────────────────────────

/// Assign corner zones to runners.  For each zone (priority order: near-post,
/// far-post, penalty-spot, edge-of-box) pick the nearest unassigned eligible
/// teammate.  Excludes the taker and the goalkeeper.
///
/// Returns `(player_id, zone_target)` pairs.  Candidates are sorted by id
/// before assignment so every player computes the same result every tick.
pub fn assign_corner_runners(
    taker_id: u32,
    taker_pos: Vector3<f32>,
    goal_pos: Vector3<f32>,
    field_w: f32,
    field_h: f32,
    teammates: impl Iterator<Item = MatchPlayerLite>,
) -> Vec<(u32, Vector3<f32>)> {
    let zones = corner_zones(taker_pos, goal_pos, field_w, field_h);
    let zone_list = [
        zones.near_post,
        zones.far_post,
        zones.penalty_spot,
        zones.edge_of_box,
    ];

    let mut candidates: Vec<MatchPlayerLite> = teammates
        .filter(|t| t.id != taker_id && !t.tactical_positions.is_goalkeeper())
        .collect();
    candidates.sort_by_key(|t| t.id);

    let mut used: Vec<u32> = Vec::with_capacity(4);
    let mut assignments: Vec<(u32, Vector3<f32>)> = Vec::with_capacity(4);

    for zone_target in &zone_list {
        if let Some(best) = candidates
            .iter()
            .filter(|t| !used.contains(&t.id))
            .min_by(|a, b| {
                let da = (a.position - zone_target).magnitude_squared();
                let db = (b.position - zone_target).magnitude_squared();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            used.push(best.id);
            assignments.push((best.id, *zone_target));
        }
    }

    assignments
}

// ── Per-player helpers ────────────────────────────────────────────────────────

/// Return this player's assigned corner zone if a corner is being taken by
/// their team, or `None` otherwise.
pub fn player_corner_zone(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
    if !ctx.ball().is_team_attacking_corner() {
        return None;
    }
    // Wishlist #20 issue 2: keep the zone assignment alive through the
    // FLIGHT, not just while the taker holds the ball. current_owner goes
    // None the instant the corner is struck, so keying on it made every
    // runner abandon their zone exactly when the ball was in the air. Fall
    // back to last_owner (the taker) and anchor the geometry to the ball's
    // strike point (the corner spot, stable) rather than the taker's live
    // position (he's now running off).
    let taker_id = ctx
        .tick_context
        .ball
        .current_owner
        .or(ctx.tick_context.ball.last_owner)?;
    let taker_pos = ctx
        .tick_context
        .ball
        .pass_origin_position
        .unwrap_or_else(|| ctx.tick_context.positions.players.position(taker_id));
    let goal_pos = ctx.player().opponent_goal_position();
    let field_w = ctx.context.field_size.width as f32;
    let field_h = ctx.context.field_size.height as f32;

    let assignments = assign_corner_runners(
        taker_id,
        taker_pos,
        goal_pos,
        field_w,
        field_h,
        ctx.players().teammates().all(),
    );

    assignments
        .into_iter()
        .find(|(pid, _)| *pid == ctx.player.id)
        .map(|(_, zone)| zone)
}

/// `true` once ≥2 runners are within 20u of their assigned zones — the
/// signal for the corner taker to deliver.
pub fn corner_box_loaded(ctx: &StateProcessingContext) -> bool {
    let taker_id = match ctx.tick_context.ball.current_owner {
        Some(id) => id,
        None => return false,
    };
    let taker_pos = ctx.tick_context.positions.players.position(taker_id);
    let goal_pos = ctx.player().opponent_goal_position();
    let field_w = ctx.context.field_size.width as f32;
    let field_h = ctx.context.field_size.height as f32;

    let assignments = assign_corner_runners(
        taker_id,
        taker_pos,
        goal_pos,
        field_w,
        field_h,
        ctx.players().teammates().all(),
    );

    let ready = assignments.iter().filter(|(pid, zone_target)| {
        let pos = ctx.tick_context.positions.players.position(*pid);
        (pos - zone_target).magnitude() < 20.0
    }).count();

    ready >= 2
}

/// Choose a delivery zone and return `(receiver_id, zone_target)`.
///
/// Picks the runner who is **closest to their assigned zone** — delivering to
/// a runner who is already in position rather than one who is still running.
/// `zone_rotation` adds a 30u bias toward a preferred zone so consecutive
/// corners vary when distances are similar.
pub fn pick_corner_delivery(
    ctx: &StateProcessingContext,
    zone_rotation: u8,
) -> Option<(u32, Vector3<f32>)> {
    let taker_id = ctx.player.id;
    let taker_pos = ctx.player.position;
    let goal_pos = ctx.player().opponent_goal_position();
    let field_w = ctx.context.field_size.width as f32;
    let field_h = ctx.context.field_size.height as f32;

    // §9.4.2 short corner — a distinct decision branch, not a zone. Only
    // available when a teammate is ACTUALLY standing in the near zone
    // (the band between the corner arc and the six-yard box) at the
    // moment of the kick: then the delivery may be a precise, deliberate
    // pass at that teammate's real position. An empty near zone is never
    // a target for any delivery mode.
    if let Some(mate) = ctx
        .players()
        .teammates()
        .all()
        .filter(|t| {
            t.id != taker_id
                && (t.position - taker_pos).magnitude() < 70.0
                // Still outside the box mouth — a runner already at
                // the near post is a cross target, not a short option.
                && (t.position.y - field_h * 0.5).abs() > 84.0
        })
        .min_by(|a, b| {
            let da = (a.position - taker_pos).magnitude();
            let db = (b.position - taker_pos).magnitude();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        // The taker MAY go short — usually does when an option was
        // staged, but a cross stays possible so defences can't sit on
        // the short ball.
        if ctx.context.rng.bernoulli(0.7) {
            return Some((mate.id, mate.position));
        }
    }

    let assignments = assign_corner_runners(
        taker_id,
        taker_pos,
        goal_pos,
        field_w,
        field_h,
        ctx.players().teammates().all(),
    );

    if assignments.is_empty() {
        return None;
    }

    let n = assignments.len();
    let preferred_idx = (zone_rotation as usize) % n;

    // Score each assignment: distance to zone minus a bias for the preferred
    // rotation index.  Lower score = deliver here.
    assignments
        .iter()
        .enumerate()
        .min_by(|(i, (pid_a, zone_a)), (j, (pid_b, zone_b))| {
            let pos_a = ctx.tick_context.positions.players.position(*pid_a);
            let pos_b = ctx.tick_context.positions.players.position(*pid_b);
            let da = (pos_a - zone_a).magnitude()
                - if *i == preferred_idx { 30.0 } else { 0.0 };
            let db = (pos_b - zone_b).magnitude()
                - if *j == preferred_idx { 30.0 } else { 0.0 };
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, &(pid, zone))| {
            (pid, clamp_cross_target_out_of_short_band(zone, goal_pos, field_w, field_h))
        })
}

/// §9.4.2 hard clamp, cross branch only: a cross-style corner target
/// must be at or beyond the six-yard box line — never in the band
/// between the corner arc and the box (near the byline, outside the
/// box mouth). The six-yard area is ~55u deep and its mouth spans
/// goal-centre ± 84u (29u half-goal + 55u). Zone geometry already
/// respects this; the clamp guarantees it against any future retune.
fn clamp_cross_target_out_of_short_band(
    target: Vector3<f32>,
    goal_pos: Vector3<f32>,
    field_w: f32,
    field_h: f32,
) -> Vector3<f32> {
    const SIX_YARD_DEPTH: f32 = 55.0;
    const SIX_YARD_HALF_MOUTH: f32 = 84.0;
    let byline_x = if goal_pos.x > field_w * 0.5 { field_w } else { 0.0 };
    let cy = field_h * 0.5;
    let depth = (target.x - byline_x).abs();
    let lateral = (target.y - cy).abs();
    if depth < SIX_YARD_DEPTH && lateral > SIX_YARD_HALF_MOUTH {
        // Pull the target laterally to the six-yard mouth edge — the
        // nearest legal point for a genuine cross.
        let clamped_y = if target.y > cy {
            cy + SIX_YARD_HALF_MOUTH
        } else {
            cy - SIX_YARD_HALF_MOUTH
        };
        return Vector3::new(target.x, clamped_y, target.z);
    }
    target
}
