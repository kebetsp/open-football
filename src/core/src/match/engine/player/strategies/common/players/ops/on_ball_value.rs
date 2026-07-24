//! Option B — unified on-ball value function (see `docs/on-ball-decision-
//! logic-spec-optionB.md` in The Gaffer repo). Shared primitives consumed
//! by the shot decision (Component A, B2), carry-target selection
//! (Component B, B1 — this stage), and pass evaluation (Component C, B3).
//!
//! B1 wired up `time_to_intercept` / `control_prob` / `carry_value` /
//! `carry_candidates`. B2 added `effective_open_angle` / `angle_xg_correction`
//! (Component A). B3 (this stage) adds `pass_value` (Component C).
//!
//! Milestone 3 (possession-decision-intelligence PRD) generalizes
//! `pass_value` to an arbitrary hypothetical passer position
//! (`pass_value_from`), folds a genuine "what if I pass from here" term
//! into `carry_value` (`best_pass_value_from`), adds real crowding-based
//! risk (`congestion_risk`), and gives the receiver's own terminal value
//! one real level of recursion (`shallow_best_pass_value_from`) — the
//! deferred `best_pass_value_from(P)` term the original spec called for
//! and B3 explicitly left out, per that function's own docstring.

use crate::r#match::player::strategies::spacing;
use crate::r#match::{MatchPlayerLite, StateProcessingContext};
use nalgebra::Vector3;

/// Human reaction-time floor (~150ms), converted directly to ticks at the
/// engine's 10ms/tick match-time step — a duration, not a rate constant,
/// so it does NOT need the 9x match-time-compression conversion that
/// applies to rate-based constants (see CLAUDE.md's third core rule).
const REACTION_TICKS: f32 = 15.0;

/// Candidate re-evaluation / reach horizon: 60-120 ticks (0.6-1.2s) per
/// the spec. Mid-point used to size the candidate ring's reach radius.
const REACH_TICKS: f32 = 90.0;

/// Sigmoid steepness for `control_prob`: a `REACTION_TICKS`-sized timing
/// advantage should already meaningfully swing the probability.
const CONTROL_PROB_SCALE: f32 = 15.0;

/// Ball travel speed used by every lane-contest estimate in this file —
/// matches the engine's own `MAX_PASS_VELOCITY` (players.rs) as a flat
/// approximation. Real passes travel below that cap, but a constant
/// upper-bound speed is a defensible simplification for a lane-contest
/// estimate, not a trajectory simulation.
const BALL_SPEED: f32 = 3.2;

/// Radius (game units) within which an opponent contributes to
/// `congestion_risk` — mirrors `ForwardCreatingSpaceState`'s own
/// congestion term (30u), widened slightly since this is a risk COST
/// applied at an arbitrary candidate/passer point, not a position-
/// selection score.
const CONGESTION_RADIUS: f32 = 35.0;

/// Cap on summed `congestion_risk`, matching `risk_penalty`'s own 0.3
/// cap so the two cost terms stay comparable in magnitude.
pub(crate) const CONGESTION_CAP: f32 = 0.3;

/// Ticks for `opponent` to reach `point`: their real, condition-adjusted
/// max speed (u/tick) plus the reaction-time floor. This is the single
/// primitive Component B (carry) and Component C (pass completion, B3)
/// both consume — one formula, two call sites, per the spec.
pub fn time_to_intercept(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    point: Vector3<f32>,
) -> f32 {
    let dist = (point - opponent.position).magnitude();
    let max_speed = ctx
        .context
        .players
        .by_id(opponent.id)
        .map(|p| p.skills.max_speed_with_condition(p.player_attributes.condition))
        .unwrap_or(0.45);
    REACTION_TICKS + dist / max_speed.max(0.05)
}

/// Spearman-style control probability at `point`: how favourably the
/// carrier's own time-to-reach compares against the fastest-arriving
/// opponent's time-to-intercept. ~1.0 = uncontested, ~0.5 = contested,
/// ~0.0 = covered. Only opponents within 120u are considered — a
/// centre-back sitting the other side of the pitch isn't part of this
/// tick's decision.
pub fn control_prob(ctx: &StateProcessingContext, point: Vector3<f32>) -> f32 {
    let carrier_max_speed = ctx
        .player
        .skills
        .max_speed_with_condition(ctx.player.player_attributes.condition);
    let carrier_dist = (point - ctx.player.position).magnitude();
    let carrier_time = carrier_dist / carrier_max_speed.max(0.05);

    let mut min_opp_time = f32::MAX;
    for opp in ctx.players().opponents().nearby(120.0) {
        let t = time_to_intercept(ctx, &opp, point);
        if t < min_opp_time {
            min_opp_time = t;
        }
    }
    if min_opp_time == f32::MAX {
        return 1.0; // no opponent within range — fully open
    }
    let advantage = min_opp_time - carrier_time; // ticks; positive = carrier wins the race
    1.0 / (1.0 + (-advantage / CONTROL_PROB_SCALE).exp())
}

/// Small backward/exposure penalty: carrying into a point further from
/// goal than the carrier's current position generalises the §12.6
/// chance-retreat penalty to carrying. Deliberately small relative to
/// `control_prob * shot_value` so it discourages retreat without
/// overriding a genuinely open outlet.
fn risk_penalty(ctx: &StateProcessingContext, point: Vector3<f32>) -> f32 {
    let goal = ctx.player().opponent_goal_position();
    let current_dist = (goal - ctx.player.position).magnitude();
    let point_dist = (goal - point).magnitude();
    let retreat = (point_dist - current_dist).max(0.0);
    (retreat / 300.0).min(0.3)
}

/// Milestone 3 — genuine crowding cost at `point`, replacing "risk" that
/// only ever measured "did I retreat from goal" with a real local-
/// density term: a linear falloff summed over every opponent within
/// `CONGESTION_RADIUS`, capped at `CONGESTION_CAP` so it stays
/// comparable in magnitude to `risk_penalty`. Reuses
/// `ForwardCreatingSpaceState`'s own congestion-scoring shape rather
/// than reinventing it.
pub fn congestion_risk(ctx: &StateProcessingContext, point: Vector3<f32>) -> f32 {
    let mut total = 0.0f32;
    for opp in ctx.players().opponents().nearby_at(point, CONGESTION_RADIUS) {
        let dist = (point - opp.position).magnitude();
        total += ((CONGESTION_RADIUS - dist) / CONGESTION_RADIUS).max(0.0);
    }
    total.min(CONGESTION_CAP)
}

/// Carry value at a single candidate point — Component B's scoring
/// function. `carry_shot_value` supplies Component A's keeper-aware
/// angle term; `best_pass_value_from(P)` (Component C, Milestone 3) is
/// now folded in as the MAX alternative to shooting, not a sum — a
/// carrier occupying point P will do one or the other with the same
/// touch, so summing would inflate positions that are mediocre at both
/// over one that's excellent at exactly one (this is a deliberate
/// correction to the original spec text, which wrote `+`). Real
/// congestion cost (`congestion_risk`) is subtracted alongside the
/// existing retreat-based `risk_penalty` — a candidate can be penalised
/// for being crowded independently of whether it's also a retreat.
pub fn carry_value(ctx: &StateProcessingContext, point: Vector3<f32>) -> f32 {
    let cp = control_prob(ctx, point);
    let sv = carry_shot_value(ctx, point);
    let pv = best_pass_value_from(ctx, point);
    let risk = risk_penalty(ctx, point);
    let congestion = congestion_risk(ctx, point);
    cp * sv.max(pv) - risk - congestion
}

/// Half-width of the goal in game units — same constant `flow/goal.rs`
/// and `player.rs::shot_clarity()` already use (real 7.32m goal, 8u/m).
const GOAL_HALF_WIDTH: f32 = 29.0;

/// Estimated keeper blocking half-width along the goal line (game
/// units). No sourced real-world figure exists for a keeper's effective
/// reach/set-position footprint at a given range — flagged as an
/// estimate per the spec's own instruction; calibrate visually against
/// replay before trusting it numerically at a finer grain than "roughly
/// right."
const GK_BLOCK_HALF_WIDTH: f32 = 6.0;

/// Component A — occluded open angle at `shot_pos`, in radians. Uses
/// true SIGNED post angles (angle-to-post-A minus angle-to-post-B),
/// NOT `player.rs::shot_clarity()`'s abs-collapsed near/far-offset
/// approximation — that formula treats "shooter stands laterally within
/// the goal's projected width" (a central shooter) and "shooter stands
/// outside it" (a wide shooter) asymmetrically, silently HALVING the
/// true angle for central positions (verified by direct hand-calculation
/// during B2's own verification pass: a dead-centre shooter's true
/// visible span is ~2×atan2(GOAL_HALF_WIDTH, depth), but the abs-
/// collapsed formula computes only ~1×atan2(GOAL_HALF_WIDTH, depth)).
/// That's a fine approximation for `shot_clarity()`'s own calibrated
/// purpose (it was tuned end-to-end against it) but wrong to reuse for a
/// from-scratch angle measure — this is therefore a genuinely separate
/// computation, not a mirror, despite sharing the same `GOAL_HALF_WIDTH`
/// constant and conceptual goal.
///
/// `gk_pos = None` returns the plain unoccluded angle (no keeper to
/// account for). With a keeper, the occluded angular segment is
/// intersected against the visible span before subtracting.
pub fn effective_open_angle(
    ctx: &StateProcessingContext,
    shot_pos: Vector3<f32>,
    gk_pos: Option<Vector3<f32>>,
) -> f32 {
    let goal_position = ctx.player().opponent_goal_position();
    let x_offset = (goal_position.x - shot_pos.x).abs().max(1.0);
    let post_a_y = goal_position.y - GOAL_HALF_WIDTH;
    let post_b_y = goal_position.y + GOAL_HALF_WIDTH;
    let angle_a = (post_a_y - shot_pos.y).atan2(x_offset);
    let angle_b = (post_b_y - shot_pos.y).atan2(x_offset);
    let (lo, hi) = if angle_a < angle_b {
        (angle_a, angle_b)
    } else {
        (angle_b, angle_a)
    };

    let Some(gk) = gk_pos else {
        return hi - lo;
    };
    let gk_angle = (gk.y - shot_pos.y).atan2(x_offset);
    let gk_half_angle = GK_BLOCK_HALF_WIDTH.atan2(x_offset);
    let overlap_lo = (gk_angle - gk_half_angle).max(lo);
    let overlap_hi = (gk_angle + gk_half_angle).min(hi);
    if overlap_hi <= overlap_lo {
        return hi - lo;
    }
    ((hi - lo) - (overlap_hi - overlap_lo)).max(0.0)
}

/// Ratio of the shooter's ACTUAL effective open angle (against the real,
/// current keeper position) to what it would be against a "doctrine"
/// keeper — one standing on the real keeper's own depth but on the line
/// from the shooter to goal centre (the standard "imaginary line from
/// goal centre to the ball" positioning cited in the spec's sourcing).
/// >1.0 = the keeper is caught genuinely out of position for THIS shot
/// (exploitable); ~1.0 = well-positioned; <1.0 = somehow even more
/// square-on than doctrine (rare).
///
/// This is a revised design from B2's first cut, changed after direct
/// numeric verification (see the B2 implementation notes in
/// docs/on-ball-decision-logic-spec-optionB.md's decisions-log entry)
/// found that comparing against a hypothetical CENTRAL shot position
/// (rather than against a hypothetical KEEPER position at the same shot
/// spot) conflated two separate, real effects: (1) a wide-angle shot is
/// objectively harder than a central one purely by geometry, keeper
/// aside — real xG models agree, and this is already implicit in
/// `shooting.rs::expected_xg`'s distance-based curve; (2) whether THIS
/// keeper is well-positioned for THIS specific shot, which is the
/// actual thing this correction should isolate. A central-reference
/// design made effect (1) dominate every test case, so wide positions
/// almost never scored above 1.0 even against a badly out-of-position
/// keeper — the opposite of the spec's intent. This design instead
/// holds the shot position fixed and asks only "is the keeper where a
/// well-drilled keeper would be," which correctly returns ~1.0 for
/// normal play (keepers are usually reasonably positioned) and >1.0
/// specifically when a carry/pass has genuinely dragged them out of
/// position — verified by direct hand-calculation before implementing.
///
/// Clamped to a moderate band so it nudges `shooting.rs::expected_xg`'s
/// Opta-sourced distance curve rather than overwhelming it — that curve
/// is real, calibrated data and must stay the dominant term. Applied
/// only at the shot-DECISION call site (`forward_shot_decision.rs`), not
/// inside `ShotSkillProfile::expected_xg` itself — that shared function
/// is also called from `handle_shoot_event` with no
/// `StateProcessingContext` available (raw in-flight inputs only, see
/// its own docstring), so changing its signature would require touching
/// that second call site too. Scoped here instead: stat-time xG
/// (`last_shot_xg`, xG-chain/buildup credit, GK xG-prevented ledger)
/// stays on the plain distance-based value; only the shoot/pass/hold
/// DECISION becomes angle-sensitive, which is what the spec's carrier
/// behaviour actually needs.
pub fn angle_xg_correction(ctx: &StateProcessingContext) -> f32 {
    let shot_pos = ctx.player.position;
    let goal_position = ctx.player().opponent_goal_position();
    let Some(actual_gk) = ctx
        .players()
        .opponents()
        .goalkeeper()
        .next()
        .map(|g| g.position)
    else {
        return 1.0; // no keeper found (shouldn't happen mid-match) — no correction
    };

    let actual = effective_open_angle(ctx, shot_pos, Some(actual_gk));

    // Doctrine keeper: same real depth (x) as the actual keeper — real
    // keepers don't roam far off their line — but y re-derived as the
    // point on the shooter-to-goal-centre line at that depth.
    let dx = goal_position.x - shot_pos.x;
    let t = if dx.abs() > 1.0 {
        (actual_gk.x - shot_pos.x) / dx
    } else {
        1.0
    };
    let doctrine_y = shot_pos.y + t * (goal_position.y - shot_pos.y);
    let doctrine_gk = Vector3::new(actual_gk.x, doctrine_y, 0.0);
    let reference = effective_open_angle(ctx, shot_pos, Some(doctrine_gk));

    if reference < 0.02 {
        return 1.0; // reference itself near-zero (extreme edge case) — don't divide by ~0
    }
    (actual / reference).clamp(0.5, 1.6)
}

/// PRD: docs/attacker-angle-seeking-and-gk-drag (Option B completion,
/// Milestone 1). Projects where the keeper could realistically be by the
/// time the carrier reaches `candidate` — the `gk_projected_pos` term the
/// original spec called for so "hold width, drag the keeper" falls out
/// of `carry_value` instead of needing to be hand-coded. Reuses
/// `angle_xg_correction`'s own doctrine-line derivation (same real-
/// keeper-depth, y re-derived onto the shooter-to-goal-centre line) but
/// evaluated from `candidate` rather than the carrier's actual shot
/// position, then moves the keeper from his CURRENT position toward that
/// doctrine point by however far he can cover in `REACTION_TICKS` at his
/// own real max speed — never past the doctrine point, so a slow keeper
/// is genuinely caught out of position and a fast one recovers in time.
fn projected_gk_position(ctx: &StateProcessingContext, candidate: Vector3<f32>) -> Vector3<f32> {
    let goal_position = ctx.player().opponent_goal_position();
    let Some(gk) = ctx.players().opponents().goalkeeper().next() else {
        return candidate; // no keeper found (shouldn't happen mid-match)
    };
    let actual_gk = gk.position;

    let dx = goal_position.x - candidate.x;
    let t = if dx.abs() > 1.0 {
        (actual_gk.x - candidate.x) / dx
    } else {
        1.0
    };
    let doctrine_y = candidate.y + t * (goal_position.y - candidate.y);
    let doctrine = Vector3::new(actual_gk.x, doctrine_y, 0.0);

    let to_doctrine = doctrine - actual_gk;
    let dist_to_doctrine = to_doctrine.magnitude();
    if dist_to_doctrine < 0.5 {
        return actual_gk; // already there
    }

    let gk_max_speed = ctx
        .context
        .players
        .by_id(gk.id)
        .map(|p| p.skills.max_speed_with_condition(p.player_attributes.condition))
        .unwrap_or(0.40);
    let reach = (gk_max_speed * REACTION_TICKS).min(dist_to_doctrine);

    actual_gk + to_doctrine.normalize() * reach
}

/// Component A's keeper-aware replacement for the old flat-distance
/// `shot_value_placeholder` (removed). Normalizes `effective_open_angle`
/// against the PROJECTED keeper by the same reference constant
/// `pass_value` already uses (`/ 1.31`) so `carry_value`'s overall
/// magnitude stays comparable to what the placeholder produced — a wide
/// candidate that would drag a well-positioned keeper out of line scores
/// higher than a central one at the same distance, which is the concrete
/// mechanism the reported "keeper never gets dragged out" gap needed.
fn carry_shot_value(ctx: &StateProcessingContext, point: Vector3<f32>) -> f32 {
    let gk = projected_gk_position(ctx, point);
    (effective_open_angle(ctx, point, Some(gk)) / 1.31).clamp(0.0, 1.0)
}

/// Generate and score carry candidates within the carrier's own reach
/// horizon (`REACH_TICKS`), argmax by `carry_value`. Structurally the
/// same "score a ring of candidates around a target" shape as
/// `ForwardCreatingSpaceState::find_optimal_free_zone`.
///
/// Exclusion (`spacing::claimed_points`/`violates_exclusion`) is applied
/// ONLY when the caller does not currently have the ball — i.e. an
/// off-ball run (RunningInBehind) must not converge on a gap another
/// teammate (including the ball carrier) is already occupying or running
/// into, which is exactly the §11.9 exclusion's purpose. When the caller
/// IS the ball carrier (Dribbling), `claimed_points` would include the
/// carrier's own current position/projected point (`ball_holder(ctx)`
/// resolves to `ctx.player` in that case), so applying it would be
/// self-defeating — the carrier's own search is left unconstrained.
///
/// This split was found necessary empirically, not assumed up front:
/// B1's first cut applied no exclusion anywhere and passed each of
/// Dribbling-only and RunningInBehind-only in isolation (~2.7 goals/
/// match each) but the COMBINATION produced 4.17 goals/match with
/// repeated 8-10 goal blowouts — both states independently converging on
/// the identical best-value gap every attacking sequence, a systematic
/// overload rather than an occasional good chance. Bisected via a
/// scoped `git stash` on each file individually before concluding this
/// was an interaction effect, not a bug in either state alone.
///
/// Returns `(best_point, best_value)`; falls back to the carrier's
/// current position if every candidate scores below holding still (walks
/// out as "the best carry candidate is roughly where I already am" per
/// the spec's decision-rule section — no separate Hold branch needed).
pub fn carry_candidates(ctx: &StateProcessingContext) -> (Vector3<f32>, f32) {
    let player_pos = ctx.player.position;
    let goal = ctx.player().opponent_goal_position();
    let forward = (goal - player_pos).normalize();
    let lateral = Vector3::new(-forward.y, forward.x, 0.0);

    let carrier_max_speed = ctx
        .player
        .skills
        .max_speed_with_condition(ctx.player.player_attributes.condition);
    let reach = (carrier_max_speed * REACH_TICKS).clamp(15.0, 70.0);

    let field_width = ctx.context.field_size.width as f32;
    let field_height = ctx.context.field_size.height as f32;

    let mut candidates: Vec<Vector3<f32>> = Vec::with_capacity(16);
    candidates.push(player_pos); // "hold" candidate — current position
    for &fwd_frac in &[0.3, 0.6, 1.0] {
        for &lat_frac in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let candidate = player_pos + forward * (reach * fwd_frac) + lateral * (reach * lat_frac);
            candidates.push(candidate);
        }
    }

    let on_ball = ctx.player.has_ball(ctx);
    let claimed = if on_ball {
        Vec::new()
    } else {
        spacing::claimed_points(ctx)
    };

    let mut best_point = player_pos;
    let mut best_value = f32::MIN;
    let mut best_any_point = player_pos;
    let mut best_any_value = f32::MIN;
    for candidate in candidates {
        let clamped = Vector3::new(
            candidate.x.clamp(15.0, field_width - 15.0),
            candidate.y.clamp(15.0, field_height - 15.0),
            0.0,
        );
        let value = carry_value(ctx, clamped);
        if value > best_any_value {
            best_any_value = value;
            best_any_point = clamped;
        }
        if !on_ball && spacing::violates_exclusion(clamped, &claimed) {
            continue;
        }
        if value > best_value {
            best_value = value;
            best_point = clamped;
        }
    }

    // Every candidate excluded (packed box) — fall back to the best
    // scorer regardless of claim, same pattern as find_optimal_free_zone.
    if best_value == f32::MIN {
        (best_any_point, best_any_value)
    } else {
        (best_point, best_value)
    }
}

/// Shared lane-contest completion-probability estimate: the minimum
/// margin (an opponent's `time_to_intercept` minus the ball's own
/// travel time at `BALL_SPEED`) to the closest point on the straight
/// `from_pos`→`to_pos` lane, over every opponent within range, converted
/// to a probability via the same sigmoid `control_prob` uses. Extracted
/// so `pass_value_from` (the real passer's evaluation) and
/// `shallow_best_pass_value_from` (Milestone 3's non-recursive
/// third-player evaluation) share one lane-contest computation instead
/// of duplicating the opponent-margin loop — same "extract, don't
/// duplicate" discipline `carry_shot_value` already uses for angle math.
fn lane_completion_prob(
    ctx: &StateProcessingContext,
    from_pos: Vector3<f32>,
    to_pos: Vector3<f32>,
) -> f32 {
    let dist = (to_pos - from_pos).magnitude();
    let dir = if dist > 1.0 {
        (to_pos - from_pos) / dist
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };

    let mut min_margin = f32::MAX;
    for opp in ctx
        .players()
        .opponents()
        .nearby_at(from_pos, dist.max(20.0) + 30.0)
    {
        let to_opp = opp.position - from_pos;
        let proj = (to_opp.x * dir.x + to_opp.y * dir.y).clamp(0.0, dist);
        let closest_point = from_pos + dir * proj;
        let ball_time_to_point = proj / BALL_SPEED;
        let opp_time = time_to_intercept(ctx, &opp, closest_point);
        let margin = opp_time - ball_time_to_point;
        if margin < min_margin {
            min_margin = margin;
        }
    }
    if min_margin == f32::MAX {
        0.95
    } else {
        (1.0 / (1.0 + (-min_margin / CONTROL_PROB_SCALE).exp())).clamp(0.05, 0.95)
    }
}

/// Non-recursive receiver-terminal-value estimate: the same shot-value-
/// or-distance-proxy `pass_value` always used, extracted verbatim (not
/// re-derived) so `shallow_best_pass_value_from` can reuse it for a
/// hypothetical THIRD player without opening a second recursion level.
fn shallow_terminal_value_at(
    ctx: &StateProcessingContext,
    pos: Vector3<f32>,
    gk_pos: Option<Vector3<f32>>,
) -> f32 {
    let goal_position = ctx.player().opponent_goal_position();
    let goal_dist = (goal_position - pos).magnitude();
    if goal_dist < 220.0 {
        (effective_open_angle(ctx, pos, gk_pos) / 1.31).clamp(0.0, 1.0)
    } else {
        (1.0 - goal_dist / 500.0).clamp(0.0, 1.0)
    }
}

/// Milestone 3's one-level "what happens next" term: the best value a
/// hypothetical THIRD player (a teammate of the receiver, excluding the
/// receiver themself) could offer if the receiver looked to combine
/// immediately rather than shoot/hold. Deliberately does NOT call
/// `pass_value_from` or `best_pass_value_from` — recursion is
/// structurally impossible (a separate function, not a depth counter),
/// per the plan's own requirement: each candidate is scored with the
/// shared lane-contest probability (`lane_completion_prob`) but the
/// NON-recursive `shallow_terminal_value_at`, so this is capped at
/// exactly one level by construction, not by convention.
fn shallow_best_pass_value_from(
    ctx: &StateProcessingContext,
    from_pos: Vector3<f32>,
    exclude_id: u32,
) -> f32 {
    let gk_pos = ctx
        .players()
        .opponents()
        .goalkeeper()
        .next()
        .map(|g| g.position);

    let mut best = 0.0f32;
    for teammate in ctx.players().teammates().nearby_at(from_pos, 90.0) {
        if teammate.id == exclude_id {
            continue;
        }
        let completion = lane_completion_prob(ctx, from_pos, teammate.position);
        let terminal = shallow_terminal_value_at(ctx, teammate.position, gk_pos);
        let value = completion * terminal;
        if value > best {
            best = value;
        }
    }
    best
}

/// Component C — net expected-goal-difference value of passing to
/// `receiver` from `passer_pos`:
/// `pass_completion_prob * receiver_terminal_value - (1 - pass_completion_prob) * turnover_risk`.
///
/// Generalized (Milestone 3) from the original `pass_value`, which only
/// ever evaluated from `ctx.player`'s actual position — `passer_pos` is
/// now an explicit parameter, following the same "position as a
/// parameter, not implicit `ctx.player`" pattern `effective_open_angle`/
/// `time_to_intercept` already use, so `best_pass_value_from` (Component
/// B's missing term) can evaluate a hypothetical carry/pass from any
/// candidate point without synthesizing a second `StateProcessingContext`.
///
/// `receiver_terminal_value` now gets a genuine one-level recursive
/// term: `shallow_terminal_value_at` (the receiver's own immediate shot/
/// distance value, exactly as before) MAX `shallow_best_pass_value_from`
/// (Milestone 3 — a real teammate the receiver could combine with,
/// scored non-recursively so depth is capped at one level by
/// construction, not convention). This finishes the spec's own
/// deferred "what happens next" term, previously left as a shallow
/// proxy per this function's earlier docstring.
///
/// `turnover_risk` now also reflects real local crowding at the
/// passer's own position (`congestion_risk`), not just distance to the
/// passer's own goal — losing the ball in a crowd is genuinely riskier
/// than losing it in space at the same field position.
///
/// Returns a value in roughly [0, 1.2], NOT pre-weighted for any call
/// site — each of the three consumers (evaluator.rs, forwards passing,
/// midfielders breakthrough) applies its own weight sized to that
/// scorer's native scale; see the comments at each wiring point. Kept
/// deliberately modest in weight everywhere it's used — CLAUDE.md's own
/// Phase 12 lesson: a broad forward-passing reward both inflates goals
/// and dilutes the surgically-tuned link/supply/intercept features if
/// it's not kept tightly scoped.
pub fn pass_value_from(
    ctx: &StateProcessingContext,
    passer_pos: Vector3<f32>,
    receiver: &MatchPlayerLite,
) -> f32 {
    let receiver_pos = receiver.position;
    let completion_prob = lane_completion_prob(ctx, passer_pos, receiver_pos);

    let gk_pos = ctx
        .players()
        .opponents()
        .goalkeeper()
        .next()
        .map(|g| g.position);
    let terminal_value = shallow_terminal_value_at(ctx, receiver_pos, gk_pos)
        .max(shallow_best_pass_value_from(ctx, receiver_pos, receiver.id));

    // Turnover risk scoped to the PASSER's own exposure if intercepted
    // (a give-and-go lost near your own box is far costlier than one
    // lost near the opponent's corner flag) — generalises the spirit of
    // §12.6's chance-retreat penalty without a second full context.
    // Milestone 3 adds real local congestion at the passer's position
    // on top of the existing distance-based term.
    let goal_position = ctx.player().opponent_goal_position();
    let own_goal_position = Vector3::new(
        ctx.context.field_size.width as f32 - goal_position.x,
        goal_position.y,
        0.0,
    );
    let passer_own_goal_dist = (own_goal_position - passer_pos).magnitude();
    let base_turnover_risk = (1.0 - passer_own_goal_dist / 500.0).clamp(0.10, 0.60);
    let turnover_risk =
        (base_turnover_risk + congestion_risk(ctx, passer_pos) * 0.5).clamp(0.10, 0.60);

    (completion_prob * terminal_value - (1.0 - completion_prob) * turnover_risk).max(0.0)
}

/// Thin wrapper preserving the original call signature — a pure
/// delegation to `pass_value_from` at the passer's actual current
/// position, so the three existing call sites (`evaluator.rs`,
/// `forwarders/states/passing`, `midfielders/states/passing`) need zero
/// changes.
pub fn pass_value(ctx: &StateProcessingContext, receiver: &MatchPlayerLite) -> f32 {
    pass_value_from(ctx, ctx.player.position, receiver)
}

/// Component B's missing "what if I pass from here" term (Milestone 3):
/// the best `pass_value_from(point, teammate)` across teammates within
/// reach of `point`, treating `point` as a hypothetical carrier position
/// — NOT `ctx.player`'s actual position. This is the concrete mechanism
/// that lets a byline-adjacent candidate score well when a teammate is
/// genuinely in a receivable crossing position, even though the carrier
/// hasn't physically arrived there yet. Bounded to a 110u radius
/// (mirrors `carry_candidates`'s own reach horizon) rather than scanning
/// every teammate on the pitch. Returns 0.0 if nobody is in range — never
/// a spurious positive.
pub fn best_pass_value_from(ctx: &StateProcessingContext, point: Vector3<f32>) -> f32 {
    let mut best = 0.0f32;
    for teammate in ctx.players().teammates().nearby_at(point, 110.0) {
        let value = pass_value_from(ctx, point, &teammate);
        if value > best {
            best = value;
        }
    }
    best
}
