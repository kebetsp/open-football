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

use crate::PlayerPositionType;
use crate::r#match::player::strategies::common::players::ops::on_ball_value;
use crate::r#match::{BallSideZone, MatchPlayerLite, StateProcessingContext};
use nalgebra::Vector3;

/// Milestone 7 (possession-decision-intelligence PRD) — which flank a
/// wide-slotted role belongs to. Central roles (including GK/sweeper/
/// defensive mid) return `None` — they don't participate in flank
/// rotation, only genuinely wide-slotted positions do.
#[derive(PartialEq, Clone, Copy)]
enum Side {
    Left,
    Right,
}

fn flank_side(pt: PlayerPositionType) -> Option<Side> {
    match pt {
        PlayerPositionType::DefenderLeft
        | PlayerPositionType::MidfielderLeft
        | PlayerPositionType::AttackingMidfielderLeft
        | PlayerPositionType::WingbackLeft
        | PlayerPositionType::ForwardLeft => Some(Side::Left),
        PlayerPositionType::DefenderRight
        | PlayerPositionType::MidfielderRight
        | PlayerPositionType::AttackingMidfielderRight
        | PlayerPositionType::WingbackRight
        | PlayerPositionType::ForwardRight => Some(Side::Right),
        _ => None,
    }
}

/// The teammate who shares my flank (e.g. the winger ahead of a wide
/// fullback, or vice versa) — `None` if I'm not in a wide-slotted role
/// myself, or if nobody else on the team is on the same flank. Nearest
/// by distance is the tie-break when a formation happens to have more
/// than one same-side wide role.
fn flank_partner(ctx: &StateProcessingContext) -> Option<MatchPlayerLite> {
    let my_side = flank_side(ctx.player.tactical_position.current_position)?;
    ctx.players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
        .filter(|t| flank_side(t.tactical_positions) == Some(my_side))
        .min_by(|a, b| {
            let da = (a.position - ctx.player.position).magnitude();
            let db = (b.position - ctx.player.position).magnitude();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Milestone 5 (possession-decision-intelligence PRD) — weight for the
/// reachability term in `score_candidate`. `pass_value_from`'s documented
/// ~[0,1.2] range times this weight lands in the same tens-scale ballpark
/// as the existing lane-clearance-plus-progress combination it
/// complements (~[-10,+35]) — a reasoned starting estimate, calibrated
/// against real match-logs traces before trusting it, same discipline as
/// every other unsourced weight in this file.
const REACHABILITY_WEIGHT: f32 = 40.0;

/// Milestone 6 — weight for the decoy-value term. `danger * marking_pressure`
/// is already a product of two [0,1] terms (typically smaller than
/// reachability alone), so this starts at the same tens-scale target.
const DECOY_WEIGHT: f32 = 35.0;

/// Milestone 7 — weight for the flank-rotation term: same tens-scale
/// target as Milestones 5/6's new terms. The "take the vacant wide role"
/// bonus is deliberately half this magnitude (a nudge, not a mandate) —
/// see `rotation_adjustment`.
const ROTATION_WEIGHT: f32 = 20.0;

/// Milestone 12 — weight for weak-side width-alignment. Smaller than
/// reachability/decoy (this is a positioning-readiness nudge, not a
/// primary value term) but still large enough to matter against the
/// tether/repulsion terms it competes with — same tens-scale reasoning
/// as the other milestone weights in this file.
const WEAK_SIDE_PATIENCE_WEIGHT: f32 = 18.0;

/// Pressure-relief support ("show for the ball" — wishlist item
/// "pressure-sensitive spread distance", implemented here on top of the
/// possession-decision-intelligence architecture rather than the old
/// dead-code `is_ball_holder_under_pressure`/`CheckToFeet` scaffolding it
/// originally lived in). Weighted well above the other milestone terms —
/// measured via a match-logs score trace that a lower (tens-scale, ~30)
/// weight left close candidates scoring a mean of ~7-8 under genuine
/// pressure against ~27-34 for 80-200u candidates, i.e. structurally
/// unable to win even when a genuinely open near pocket existed.
const PRESSURE_RELIEF_WEIGHT: f32 = 70.0;

/// Range within which a close option can plausibly act as an immediate
/// out-ball. Deliberately wider than `SHORT_SUPPORT_RADIUS` (28u) so the
/// two terms overlap smoothly: inside 28u the redundancy penalty is
/// being relaxed at the same time the relief bonus ramps up, rather than
/// the bonus only starting exactly where the penalty stops.
const PRESSURE_RELIEF_RADIUS: f32 = 45.0;

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

/// §12.5 — universal off-ball separation, every phase (extends §11.9,
/// which only guarded attacking-support target selection). A gentle
/// repulsion velocity between same-team outfielders standing inside the
/// exclusion radius of each other, applied at the processor level so it
/// covers defensive shape, out-of-possession shifts, and every other
/// off-ball path without touching each state's own logic.
///
/// Deliberately NOT applied to: restarts/set-piece shapes (walls stand
/// 9u apart by design — gated on an OpenPlay origin), the ball carrier
/// or anyone within 60u of the ball (receivers, duels, pressing),
/// goalkeepers, and link-play pairs (a one-two needs close support).
/// Returns `(separation_velocity, blend_weight)`. The caller blends
/// `state_velocity * (1-w) + separation * w` — an additive-only nudge
/// demonstrably loses to the state's own convergence pull (two players
/// walking to near-identical anchors equilibrate a few units apart, or
/// fully overlapped), so at close range the separation must progressively
/// REPLACE the state's movement, not fight it. Asymmetric by id: the
/// higher-id player yields (full weight), the lower-id player mostly
/// holds — one of the two owns the contested zone, mirroring how real
/// players resolve a shared space.
pub fn separation_nudge(ctx: &StateProcessingContext) -> Option<(Vector3<f32>, f32)> {
    use crate::r#match::PassOriginRestart;

    let ball_meta = &ctx.tick_context.ball;
    // Skip only while a set piece is STAGED (restart origin + a taker
    // holding the ball): walls and blocks stand deliberately tight. A
    // restart origin alone isn't enough to skip — origins persist through
    // the delivery flight and loose scrambles for many seconds, which
    // disabled separation exactly when post-set-piece clusters form.
    if ball_meta.pass_origin_restart != PassOriginRestart::OpenPlay && ball_meta.is_owned {
        return None;
    }
    if ctx
        .player
        .tactical_position
        .current_position
        .is_goalkeeper()
    {
        return None;
    }
    if ball_meta.current_owner == Some(ctx.player.id) {
        return None;
    }
    if ctx.ball().distance() < 60.0 {
        return None;
    }

    /// Outward speed at full blend (~comfortable jog; velocities are u/tick).
    const SEPARATION_SPEED: f32 = 0.30;

    let mut dir_sum = Vector3::zeros();
    let mut w_max = 0.0f32;
    for t in ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
    {
        if ctx.player.link_target == Some(t.id) {
            continue;
        }
        if ball_meta.current_owner == Some(t.id) {
            continue;
        }
        let d = ctx.player.position - t.position;
        let dist = (d.x * d.x + d.y * d.y).sqrt();
        if dist >= EXCLUSION_RADIUS {
            continue;
        }
        let dir = if dist > 0.5 {
            d / dist
        } else if ctx.player.id > t.id {
            // Deterministic split for exact overlap.
            Vector3::new(0.0, 1.0, 0.0)
        } else {
            Vector3::new(0.0, -1.0, 0.0)
        };
        // 1.5× slope: full takeover inside ~8u, tapering to zero at the
        // exclusion radius. Equilibrium against a typical 0.3 u/tick
        // state pull lands around 16u (~2m) — outside visual overlap.
        let closeness = (1.5 * (1.0 - dist / EXCLUSION_RADIUS)).clamp(0.0, 1.0);
        let yield_factor = if ctx.player.id > t.id { 1.0 } else { 0.4 };
        let w = closeness * yield_factor;
        dir_sum += dir * w;
        w_max = w_max.max(w);
    }
    if w_max < 1e-3 || dir_sum.norm_squared() < 1e-6 {
        return None;
    }
    Some((dir_sum.normalize() * SEPARATION_SPEED, w_max))
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
            ctx, clamped, proposed, &holder, goal_pos, &opponents, &teammates,
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

    // Pressure-relief candidates: the caller's own baseline `proposed`
    // target is a SUPPORT/SPACING point (typically 80-180u from the ball
    // holder for these states — build-up outlets, box runs, wide
    // stretch), so the ±50u ring above almost never produces a candidate
    // close enough to the holder for the pressure-relief terms in
    // `score_candidate` to ever matter — confirmed empirically via a
    // match-logs trace (selected distance-to-holder was statistically
    // flat across every pressure bucket before this fix, ~130-140u
    // regardless of pressure). Mirrors `carry_candidates`'s own
    // byline-candidate precedent (2026-07-25): when a real pattern has
    // no candidate representing it, add one explicitly rather than hope
    // the general ring reaches it. Genuinely inert (loop doesn't run)
    // whenever the holder isn't under real pressure.
    if let Some(h) = &holder {
        let holder_pressure = ctx.player().pressure().pressure_intensity_for(h.id);
        // Same 0.5 engagement floor as `score_candidate` — mild pressure
        // is common and shouldn't trigger extra candidate injection that
        // would only ever score at the (also-gated) baseline anyway.
        if holder_pressure > 0.5 {
            // Two radii, not one: a "pressured" holder by definition has
            // opponents within ~30u (`pressure_intensity_for`'s own
            // scan radius), so a single fixed ring often lands inside
            // that same crowd too — measured via a match-logs trace
            // showing close
            // candidates scoring far below the eventual winner even with
            // the pressure-relief bonus applied. A tighter ring (genuine
            // one-touch range) and a wider one (just beyond the crowd,
            // the real "show on the far side of the marker" pocket) give
            // the scorer real open options to choose between instead of
            // one coarse sample per direction.
            const RELIEF_RING_DEG: [f32; 8] = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
            for &radius_frac in &[0.45, 0.9] {
                let relief_radius = PRESSURE_RELIEF_RADIUS * radius_frac;
                for &deg in &RELIEF_RING_DEG {
                    let rad = deg.to_radians();
                    let offset = Vector3::new(rad.cos(), rad.sin(), 0.0) * relief_radius;
                    consider(h.position + offset);
                }
            }
        }
    }

    if found_valid { best } else { best_any }
}

fn score_candidate(
    ctx: &StateProcessingContext,
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

    // Pressure-relief support ("show for the ball" — wishlist item
    // "pressure-sensitive spread distance"): how surrounded the HOLDER
    // currently is, reusing the same crowding primitive Milestone 6's
    // `marking_pressure` already normalizes below — one formula, another
    // consumer, rather than inventing new proximity math. Computed once,
    // unconditionally (0.0 when there's no holder), so it can also drive
    // the tether blend at the bottom of this function, not just the
    // holder-relative terms below.
    let holder_pressure = holder
        .as_ref()
        .map(|h| ctx.player().pressure().pressure_intensity_for(h.id))
        .unwrap_or(0.0);

    // Engagement, not raw pressure: mild pressure (a single opponent
    // loosely tracking, `holder_pressure` in roughly 0.1-0.3) is by far
    // the MOST common reading in ordinary play — nudging the scoring
    // landscape on nearly every tick at that level measurably suppressed
    // goals in a regression gate (isolated via scoped git-stash: ~3.67
    // baseline vs. ~2.89 with an unconditional linear blend, consistent
    // across two independent batch pairs) even though it rarely flipped
    // the actual winning candidate — it just nudged close decisions.
    // `PRESSURE_ENGAGEMENT_FLOOR` keeps every term below it byte-for-byte
    // identical to no-pressure behaviour; only genuinely elevated
    // pressure (the same range the match-logs distance trace actually
    // showed moving selected positions) engages the mechanism at all.
    const PRESSURE_ENGAGEMENT_FLOOR: f32 = 0.5;
    let relief_engagement =
        ((holder_pressure - PRESSURE_ENGAGEMENT_FLOOR) / (1.0 - PRESSURE_ENGAGEMENT_FLOOR))
            .clamp(0.0, 1.0);

    if let Some(h) = holder {
        let holder_dist = (position - h.position).magnitude();

        // Short redundant support: a candidate a couple of metres from the
        // carrier offers a trivial pass — deprioritise it firmly. Relaxed
        // under genuine pressure: a close option stops being a redundant
        // trivial pass and becomes the actual point once the holder is
        // truly surrounded (matches the wishlist's own framing: "one
        // thing if your partner is surrounded... needs help... another
        // thing if there is space").
        if holder_dist < SHORT_SUPPORT_RADIUS {
            score -= (SHORT_SUPPORT_RADIUS - holder_dist) * 2.5 * (1.0 - relief_engagement);
        }

        // Pressure-relief bonus: reward a close, genuinely CLEAN outlet
        // specifically when the holder needs one — gated on both the
        // holder's own pressure and the candidate spot itself being
        // clear to receive at (`congestion`, already computed above for
        // the open-space term), so this never rewards converging into
        // another marked/crowded spot just because the holder is under
        // pressure elsewhere. Additive, never gates the reachability/
        // decoy/rotation/weak-side terms below — same "don't let one
        // term zero out another" discipline Milestone 6 already
        // established for reachability vs. decoy value. Zero whenever
        // pressure is below the engagement floor, so ordinary unpressed
        // circulation is untouched.
        if relief_engagement > 0.0 && holder_dist < PRESSURE_RELIEF_RADIUS {
            let closeness =
                ((PRESSURE_RELIEF_RADIUS - holder_dist) / PRESSURE_RELIEF_RADIUS).clamp(0.0, 1.0);
            let candidate_openness = (1.0 - congestion.min(10.0) / 10.0).clamp(0.0, 1.0);
            score += relief_engagement * closeness * candidate_openness * PRESSURE_RELIEF_WEIGHT;
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

    // Milestone 5 (possession-decision-intelligence PRD) — reachability:
    // would the ACTUAL current ball holder's pass to a receiver AT THIS
    // POSITION genuinely be a good option, using the real completion-
    // probability-and-terminal-value reasoning from `on_ball_value`
    // (Milestone 3) rather than the binary 4u-corridor check above. This
    // is additive alongside that check, not a replacement — the corridor
    // check stays as a cheap hard signal, this adds the real probabilistic
    // one on top.
    if let Some(h) = holder {
        let hypothetical = MatchPlayerLite {
            id: ctx.player.id,
            position,
            tactical_positions: ctx.player.tactical_position.current_position,
        };
        let reachability = on_ball_value::pass_value_from(ctx, h.position, &hypothetical);
        score += reachability * REACHABILITY_WEIGHT;
    }

    // Milestone 6 — decoy value: credits a position for being a genuine
    // attacking threat that would force a nearby opponent to engage,
    // INDEPENDENT of whether a pass here is realistic right now (the
    // reachability term above) — "drag a defender away, even though I
    // won't receive it" as a legitimate value in its own right. Summed
    // into `score`, never gated by reachability, so a position with
    // near-zero reachability but genuine danger + marking pressure still
    // gets its full decoy contribution — this is deliberately the
    // opposite construction from `carry_value`'s MAX-not-SUM (Milestone
    // 3): here the two terms must NOT gate each other, or shipping
    // reachability scoring would mechanically zero out decoy-run value,
    // which is exactly the risk the PRD calls out.
    let gk_pos = ctx
        .players()
        .opponents()
        .goalkeeper()
        .next()
        .map(|g| g.position);
    let danger = (on_ball_value::effective_open_angle(ctx, position, gk_pos) / 1.31).clamp(0.0, 1.0);
    let marking_pressure =
        (on_ball_value::congestion_risk(ctx, position) / on_ball_value::CONGESTION_CAP).clamp(0.0, 1.0);
    score += danger * marking_pressure * DECOY_WEIGHT;

    // Milestone 7 — flank rotation: two teammates on the same flank (e.g.
    // a wide fullback and the winger ahead of him) should recognise which
    // of them currently holds the wide role, rather than both
    // independently reacting to ball-side and potentially stacking on the
    // same touchline. Scoped to genuinely wide candidates only — this is
    // purely about the wide-vs-tuck decision, not overall shape.
    score += rotation_adjustment(ctx, position);

    // Milestone 12 — weak-side off-ball patience: a candidate on the
    // opposite lateral side from the ball isn't judged on immediate
    // reachability (already near-zero there via the Milestone 5 term
    // above); its job is positioning for a LATER switch of play.
    score += weak_side_patience_adjustment(ctx, position);

    // Tether: stay recognisably within the caller's tactical intent —
    // blended toward the HOLDER's own position as pressure rises PAST the
    // engagement floor. Below the floor this is exactly `proposed`
    // (unchanged behaviour for every already-verified milestone: ordinary
    // spread/reachability/decoy/rotation/weak-side scoring never sees
    // this term move for mild, common pressure readings). Past the floor,
    // "the caller's tactical intent" itself shifts from "hold attacking
    // shape" to "get open near the outlet" — real football doesn't treat
    // an emergency out-ball as a deviation from shape, it treats it as
    // the actual priority. Necessary because the pressure-relief bonus
    // above was measured (match-logs trace) to be too small on its own to
    // overcome the tether pull toward a distant `proposed` target — the
    // tether itself has to relax, not just gain a competing bonus.
    let tether_target = match holder {
        Some(h) => proposed + (h.position - proposed) * relief_engagement,
        None => proposed,
    };
    score -= (position - tether_target).magnitude() * TETHER_WEIGHT;

    score
}

/// Milestone 7 — see `score_candidate`'s own call-site comment. `0.0` when
/// `candidate` isn't in the wide band (Milestone 4's established
/// `field_h*0.30/0.70` threshold), when the player has no flank partner,
/// or when the partner is on the opposite half of the pitch (a
/// side-mismatch guard — rotation only makes sense between two players
/// genuinely sharing the same flank). Otherwise: the partner already
/// holding width there is a real penalty (don't stack); the partner
/// having tucked in is a smaller bonus (a nudge to take the vacant wide
/// role, not a mandate — the existing width/repulsion terms still decide
/// the rest).
fn rotation_adjustment(ctx: &StateProcessingContext, candidate: Vector3<f32>) -> f32 {
    let field_h = ctx.context.field_size.height as f32;
    let is_wide_candidate = candidate.y < field_h * 0.30 || candidate.y > field_h * 0.70;
    if !is_wide_candidate {
        return 0.0;
    }
    let Some(partner) = flank_partner(ctx) else {
        return 0.0;
    };
    let candidate_left = candidate.y < field_h * 0.5;
    let partner_left = partner.position.y < field_h * 0.5;
    if candidate_left != partner_left {
        return 0.0; // partner is on the opposite flank right now — not a rotation conflict
    }
    let partner_is_wide = partner.position.y < field_h * 0.30 || partner.position.y > field_h * 0.70;
    if partner_is_wide {
        -ROTATION_WEIGHT
    } else {
        ROTATION_WEIGHT * 0.5
    }
}

/// Milestone 12 (possession-decision-intelligence PRD) — weak-side
/// off-ball patience. A player positioned on the opposite lateral side
/// from the ball ("weak side") is, by construction, rarely a genuine
/// pass option right now — Milestone 5's reachability term already and
/// correctly scores that near zero at this range. Their real job in
/// real football is positioning for a LATER switch of play: hold
/// genuine width if the team wants a stretched picture for a diagonal
/// switch, or stay compact and ready to combine quickly if not —
/// following the team's own already-computed `team_width_target`
/// (0 = narrow, 1 = full width) rather than a fixed universal
/// preference. Returns `0.0` when the ball is central (no meaningful
/// "weak side" exists) or the candidate is on the ball's own side —
/// strong-side positioning is already densely handled by the
/// reachability/decoy/rotation terms above; this is additive, not a
/// replacement for any of them.
fn weak_side_patience_adjustment(ctx: &StateProcessingContext, candidate: Vector3<f32>) -> f32 {
    let field_h = ctx.context.field_size.height as f32;
    let ball_side = ctx.team().tactical().ball_side;
    if matches!(ball_side, BallSideZone::Center) {
        return 0.0;
    }
    let candidate_side = BallSideZone::for_y(field_h, candidate.y);
    let is_weak_side = matches!(
        (ball_side, candidate_side),
        (BallSideZone::Left, BallSideZone::Right) | (BallSideZone::Right, BallSideZone::Left)
    );
    if !is_weak_side {
        return 0.0;
    }

    let width_target = ctx.team().tactical().team_width_target.clamp(0.0, 1.0);
    let center_y = field_h / 2.0;
    let half_h = field_h / 2.0;
    let dist_from_center = ((candidate.y - center_y).abs() / half_h).clamp(0.0, 1.0);
    // How well this candidate's width matches the team's current width
    // preference — a wide candidate scores when width_target is high, a
    // central one scores when it's low.
    let alignment = (1.0 - (dist_from_center - width_target).abs()).max(0.0);
    alignment * WEAK_SIDE_PATIENCE_WEIGHT
}

fn ball_holder(ctx: &StateProcessingContext) -> Option<MatchPlayerLite> {
    let owner_id = ctx.ball().owner_id()?;
    ctx.players()
        .teammates()
        .all()
        .find(|t| t.id == owner_id)
}
