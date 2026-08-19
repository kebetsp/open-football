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

/// "Genuinely open net" fast-path signal (2026-07-28, Pavel's "wide open
/// goal, GK dragged wide, no shot" report). Used ONLY by two narrow
/// consumers — the long-range willingness floor in
/// `forward_shot_decision.rs` and the team-cooldown exemption in
/// `TeamOperationsImpl::can_shoot` — never by `shot_clarity()` /
/// `has_clear_shot()` itself, which is deliberately GK-blind by design
/// (see that function's own doc comment on why tying the general
/// clear-shot gate to keeper position would create a bad incentive for
/// the AI to "wait for the keeper to be out of position" on ordinary
/// shots). This is a separate, additional, stricter signal for the
/// specific case real players treat as basically automatic regardless
/// of range: nobody in the shot lane AND the keeper barely covers the
/// goal-mouth angle from here.
///
/// Real-world grounding: no public dataset isolates "shot-taking rate
/// with a genuinely open net, any range" — this is a reasoned estimate
/// (category (d) per the realism framework), not a sourced target. The
/// qualitative case is uncontroversial (a real player very rarely
/// declines a clean look at an unguarded goal at any sensible range);
/// the previous willingness model had no mechanism honouring that past
/// 36u at all.
///
/// Returns `(is_open, gk_occlusion)`. `gk_occlusion` is how much of the
/// unoccluded goal-mouth angle the keeper's position currently blocks,
/// in `[0, 1]` — 0 means the keeper isn't in the way at all. `is_open`
/// additionally requires the outfield lane to be clear (`has_clear_shot`)
/// and the shot to be within the engine's own absolute shot-distance
/// ceiling (220u — matches the hard cap in `forward_shot_decision.rs`;
/// duplicated here as a guard since this is also consulted from
/// `team.rs`, which has no equivalent prior distance check of its own).
pub fn open_net_signal(ctx: &StateProcessingContext) -> (bool, f32) {
    let shot_pos = ctx.player.position;
    let goal_position = ctx.player().opponent_goal_position();
    let distance = (goal_position - shot_pos).magnitude();
    if distance > 220.0 {
        return (false, 0.0);
    }

    let gk_pos = ctx
        .players()
        .opponents()
        .goalkeeper()
        .next()
        .map(|g| g.position);

    let full_angle = effective_open_angle(ctx, shot_pos, None);
    let occluded_angle = match gk_pos {
        Some(gp) => effective_open_angle(ctx, shot_pos, Some(gp)),
        None => full_angle,
    };
    let gk_occlusion = if full_angle > 0.02 {
        (1.0 - occluded_angle / full_angle).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Keeper blocking less than ~15% of the visible angle is "not
    // meaningfully in the way" — genuinely dragged out of position, not
    // just favouring a side while still covering the frame.
    const OPEN_NET_OCCLUSION_MAX: f32 = 0.15;
    let is_open = ctx.player().has_clear_shot() && gk_occlusion < OPEN_NET_OCCLUSION_MAX;
    (is_open, gk_occlusion)
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

    let mut candidates: Vec<Vector3<f32>> = Vec::with_capacity(18);
    candidates.push(player_pos); // "hold" candidate — current position
    for &fwd_frac in &[0.3, 0.6, 1.0] {
        for &lat_frac in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let candidate = player_pos + forward * (reach * fwd_frac) + lateral * (reach * lat_frac);
            candidates.push(candidate);
        }
    }

    // realism-bug (2026-07-25): `forward` points at the OPPONENT GOAL
    // CENTER, so for a wide-channel carrier every candidate above is
    // biased toward cutting infield (`forward` itself pulls diagonally
    // inward for anyone starting wide) — there was never a candidate
    // representing "keep hugging the touchline and advance straight
    // toward the byline/corner," the actual real-football pattern this
    // was supposed to enable. Measured baseline (25-match external
    // position check): ~0.04 genuine byline-reaching carries per match
    // in default/undirected play. Sourced target: no public stat
    // isolates "byline-run rate" specifically (flagged as an open gap
    // during research), but open-play crosses run ~19-24/team/match in
    // top leagues (Soccerment/Premier League-Opta data) — even a
    // conservative fraction of those genuinely originating from a
    // byline carry implies a per-match rate far above the measured
    // ~0.04. Fix: for a carrier already in a wide channel (the same
    // 0.30/0.70 field-height band this codebase already uses for wide
    // classification elsewhere — Milestone 4/7), add explicit
    // byline-directed candidates — straight toward the goal LINE
    // (x-axis only, preserving the carrier's own y so he stays in his
    // channel), not toward goal centre — so the existing scoring
    // (`carry_value`, already correctly rewarding a genuine cutback via
    // `best_pass_value_from`) actually gets the option to select this
    // pattern instead of it being structurally absent from the
    // candidate set.
    let is_wide_channel = player_pos.y < field_height * 0.30 || player_pos.y > field_height * 0.70;
    let mut byline_candidate_points: Vec<Vector3<f32>> = Vec::new();
    if is_wide_channel {
        let byline_x = if forward.x >= 0.0 { field_width } else { 0.0 };
        let to_byline = Vector3::new(byline_x - player_pos.x, 0.0, 0.0);
        if to_byline.magnitude() > 1.0 {
            let byline_dir = to_byline.normalize();
            for &frac in &[0.5, 1.0] {
                let c = player_pos + byline_dir * (reach * frac);
                candidates.push(c);
                byline_candidate_points.push(c);
            }
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
        let mut value = carry_value(ctx, clamped);
        if byline_candidate_points
            .iter()
            .any(|c| (c - candidate).magnitude() < 1.0)
        {
            value += byline_isolation_bonus(ctx, clamped) + momentum_bonus(ctx, player_pos, clamped);
        }
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

/// realism-bug (2026-07-25), byline-run frequency follow-up. The
/// byline-directed candidates added above only won `carry_candidates`'s
/// argmax 5.9% of the time even when eligible (measured, internal
/// diagnostic) — the real-football trigger for a winger actually
/// committing to a byline run is genuine separation from his direct
/// marker (the classic winger-vs-isolated-fullback 1v1), and the run's
/// payoff is largely what it creates a few seconds later (a cutback, a
/// corner, forcing the back line deep) — value a single-tick lookahead
/// scorer structurally can't see. Modeled as an explicit doctrine
/// credit on the byline candidates specifically (not general carry
/// candidates, which already have their own geometry-driven value): the
/// fewer opponents near the candidate point, the stronger the bonus.
/// Magnitude (0.12 max) is sized against the measured mean value gap
/// (0.098) when the byline candidate lost — enough to flip genuinely
/// isolated cases without overriding a clearly-better central option.
/// Trimmed 0.12 → 0.08 after a first cut (combined with the momentum
/// bonus below) pushed the per-tick byline win rate to 81.6% — far
/// stronger than intended — and a fresh regression batch measured
/// 2.67-2.38 goals/match, below the 3.0-4.5 band the possession-
/// decision-intelligence work established. See the momentum-bonus
/// scoping note below for the larger contributor.
fn byline_isolation_bonus(ctx: &StateProcessingContext, point: Vector3<f32>) -> f32 {
    const ISOLATION_RADIUS_MIN: f32 = 10.0;
    const ISOLATION_RADIUS_MAX: f32 = 40.0;
    const MAX_BONUS: f32 = 0.08;
    let nearest = ctx
        .players()
        .opponents()
        .nearby_at(point, ISOLATION_RADIUS_MAX)
        .map(|o| (o.position - point).magnitude())
        .fold(f32::MAX, f32::min);
    let frac = ((nearest - ISOLATION_RADIUS_MIN) / (ISOLATION_RADIUS_MAX - ISOLATION_RADIUS_MIN))
        .clamp(0.0, 1.0);
    frac * MAX_BONUS
}

/// realism-bug (2026-07-25), byline-run frequency follow-up. `carry_
/// candidates` re-evaluates from scratch on effectively every tick a
/// player is on the ball, so a candidate that wins by a small margin
/// one tick can lose to a different candidate the next as the geometry
/// shifts by a few units — aborting a run before it completes. Real
/// players don't flip-flop like this; once committed to a direction
/// they hold it unless the alternative becomes clearly better.
///
/// Originally applied to EVERY carry candidate (any defender/
/// midfielder/forward carry, not just byline runs) — measured as the
/// larger contributor to a real regression: applying it universally
/// creates a self-reinforcing feedback loop (winning a tick means
/// moving that way, which earns MORE momentum bonus next tick), which
/// pushed the byline candidates' per-tick win rate to 81.6% and,
/// applied everywhere, made carry decisions broadly stickier than
/// intended — a regression batch measured 2.67-2.38 goals/match,
/// below the 3.0-4.5 band. Rescoped to the byline candidates only
/// (alongside `byline_isolation_bonus`, same call site) — the same
/// "don't reverse for a marginal EV difference" principle, but
/// confined to the actual mechanism being fixed instead of touching
/// every carry decision in the match.
fn momentum_bonus(ctx: &StateProcessingContext, from: Vector3<f32>, candidate: Vector3<f32>) -> f32 {
    const MAX_BONUS: f32 = 0.04;
    const MIN_SPEED: f32 = 0.5;
    let velocity = ctx.player.velocity;
    let speed = velocity.magnitude();
    if speed < MIN_SPEED {
        return 0.0;
    }
    let to_candidate = candidate - from;
    let dist = to_candidate.magnitude();
    if dist < 1.0 {
        return 0.0;
    }
    let alignment = (velocity.normalize().dot(&to_candidate.normalize())).max(0.0);
    alignment * MAX_BONUS
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

    // Realism-bug 2026-07-26 (passing follow-up, turnover-risk-by-
    // location): a real central turnover hands the opponent a direct
    // transitional lane toward goal; a wide turnover forces them to
    // work the ball inside first, buying the defence time to
    // reorganise — real transition doctrine (unlike a central-lane
    // INTERCEPTION penalty, which a same-day measurement pass ruled
    // out: central-destination passes did NOT face more opponents in
    // the direct passing corridor than wide ones, 2026-07-26 investig-
    // ation — this term is about the COST of losing the ball, not the
    // PROBABILITY, so it doesn't duplicate or contradict that finding).
    // Scoped to the receiver's lateral position (where a completed-
    // looking pass is actually heading, i.e. the type of pass being
    // made) rather than the passer's, since `base_turnover_risk` above
    // already covers the passer's own exposure. Capped at the same
    // order of magnitude as the existing `congestion_risk` term (max
    // +0.15) so it nudges the already-clamped 0.10-0.60 band rather
    // than dominating it.
    let field_height = ctx.context.field_size.height as f32;
    let lateral_center = field_height / 2.0;
    let receiver_width_ratio =
        ((receiver_pos.y - lateral_center).abs() / (field_height / 2.0)).clamp(0.0, 1.0);
    let receiver_centrality = 1.0 - receiver_width_ratio; // 1.0 dead-center, 0.0 touchline
    let lateral_turnover_bonus = receiver_centrality * 0.15;

    let turnover_risk = (base_turnover_risk
        + congestion_risk(ctx, passer_pos) * 0.5
        + lateral_turnover_bonus)
        .clamp(0.10, 0.60);

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

// ─────────────────────────────────────────────────────────────────────────
// Unified dribble/pass/shoot/safe-restart/recycle comparison (2026-08
// realism-bug session — see the offside/TakeBall work earlier the same
// session for the architectural precedent this follows). `ForwardRunningState`
// previously decided pass vs. dribble via two independent, sequential
// skill-curve gates that never compared against each other, and forced a
// Pass as the fallback when both failed regardless of whether one existed.
// These three functions let that fallback tier become a genuine value
// comparison instead.
// ─────────────────────────────────────────────────────────────────────────

/// Real value of shooting from the carrier's CURRENT position — reuses the
/// exact same xG computation `evaluate_forward_shot_decision`
/// (forward_shot_decision.rs) already trusts (distance-banded
/// `expected_xg` × this file's own `angle_xg_correction`), so the raw
/// magnitude is directly comparable to what that function would compute,
/// just queryable as a pure value without committing to
/// `ShotDecision::Shoot`. Naturally near-zero from a hopeless distance/
/// angle, naturally high for a genuine chance — no separate distance/
/// angle gate is layered on top.
///
/// `SHOOT_COMPARISON_SCALE`: raw xG lives on a "true probability this
/// exact attempt scores" scale (typical realistic in-game values ~0.02-
/// 0.15, per `expected_xg`'s own calibration comment — real shot
/// populations average ~0.10). `pass_val`/`dribble_val` at this same
/// decision point live on a *different* currency — `pass_value_from`'s
/// terminal value is `shallow_terminal_value_at`, an angle-subtended
/// POSITIONAL-danger proxy (`effective_open_angle`-based, ~[0,1], no
/// steep distance decay), not a converted goal probability — the same
/// currency `carry_value`'s own internal shot term (`carry_shot_value`)
/// already uses. Comparing raw xG against that proxy directly is
/// comparing different units: measured against ~11,800 real in-match
/// decision points reaching this comparison (2026-08 diagnostic,
/// VALCMP), raw xG's mean (0.018) sat roughly 15-20x below pass/dribble's
/// means (0.26/0.36) and shoot won 0% of comparisons — including cases
/// with a plainly reasonable close-range attempt on offer. This constant
/// is the documented conversion between the two currencies, not a hard
/// override: it does not touch the underlying xG shape (a hopeless long
/// shot still scores far below a genuine chance, just as before), it
/// only rescales the whole curve up so a *good* chance can plausibly
/// outweigh a *mediocre* pass/dribble option, matching Pavel's framing
/// ("shot gets a strong situational preference via genuine xG magnitude,
/// not a hard override"). Tuned against that same diagnostic data (not
/// sourced — no dataset publishes a "how often should a forward shoot
/// vs. pass at this specific decision tier" rate), iterating 8/10/15:
/// 15 pushed goals/match to a repeatable ~4.6-5.2 across two 20-30-match
/// batches (too high); 8 pulled it to ~2.6-2.9; 10 landed at 2.53-3.17
/// across three batches (pooled ~2.9/75 matches), statistically
/// indistinguishable from the SAME-day, unmodified, already-deployed
/// baseline measured the identical way (2.53, then 3.13 — pooled
/// ~2.83/60 matches) — i.e. 10 sits within this project's own
/// well-documented batch-to-batch noise band rather than shifting goals/
/// match at all attributably. At scale=10, the dispatched
/// `FWD_RUN_VALUE_SHOOT` event (previously 0/match, the reported bug)
/// fires ~10-13/match, roughly half of all shot attempts in a match —
/// this call site only runs AFTER the eager high-priority shot chain
/// (FWD_ROUND_KEEPER/FWD_RUN_SHOOT_ON_SIGHT/FWD_SNAPSHOT_PRESSED/
/// FWD_RUN_POINT_BLANK/FWD_RUN_PRIO05_CLEAR/FWD_RUN_PRIO06_BOX) has
/// already declined, so the population reaching here is deliberately
/// skewed toward mediocre-to-poor chances — a real, non-trivial win
/// rate here (not a rare one) is consistent with this being the exact
/// gap the wider deflection/corner investigation flagged: too few shot
/// attempts from anything but a clean chance, suppressing blocks/
/// deflections/corners league-wide. `shoot_value()` has exactly one call
/// site (this comparison) as of 2026-08, so this scaling lives here
/// rather than at the call site.
const SHOOT_COMPARISON_SCALE: f32 = 10.0;

pub fn shoot_value(ctx: &StateProcessingContext) -> f32 {
    // Same hard gates evaluate_forward_shot_decision checks first — a
    // shot that's mechanically impossible right now (cooldown, absurd
    // range) must never win the comparison just because its xG looks
    // fine in isolation.
    if !ctx.team().can_shoot() || !ctx.player().can_shoot() {
        return 0.0;
    }
    let distance = ctx.ball().distance_to_opponent_goal();
    if distance > 220.0 {
        return 0.0;
    }
    let profile = ctx.player().shooting().shot_profile();
    let has_clear = ctx.player().has_clear_shot();
    let mut xg = profile.expected_xg(distance, has_clear);
    xg *= angle_xg_correction(ctx);
    (xg * SHOOT_COMPARISON_SCALE).max(0.0)
}

/// A candidate "put it into the nearest defender and concede a controlled
/// restart rather than risk losing the ball badly" action.
pub struct SafeRestartCandidate {
    pub value: f32,
    pub target_defender_id: u32,
    pub aim_point: Vector3<f32>,
}

/// Distance margins for `safe_restart_value` — reasoned estimates (no
/// sourced dataset exists for this niche a behaviour), not sourced. Tune
/// from observed batch behaviour rather than treating as calibrated.
const SAFE_RESTART_LINE_MARGIN: f32 = 40.0; // ~5m from a byline/touchline
const SAFE_RESTART_PRESS_MARGIN: f32 = 24.0; // ~3m from the nearest defender
const SAFE_RESTART_BASE_VALUE: f32 = 0.30;

/// Value of deliberately sending the ball into the nearest opponent to
/// concede a corner/throw-in/goal-kick instead of risking a turnover.
/// Deliberately a smooth, continuous falloff on both the line-proximity
/// and press-proximity terms (a hard cutoff was found, earlier this same
/// investigation, to create an unrealistic discontinuity in the analogous
/// `shot_clarity` angle formula) — so the value tapers to zero rather than
/// jumping, and is ~0 from mid-pitch by construction, with no separate
/// "don't try this from the middle of the field" gate needed.
///
/// Returns `None` when no opponent is close enough to plausibly be the
/// target of the deflection at all.
pub fn safe_restart_value(ctx: &StateProcessingContext) -> Option<SafeRestartCandidate> {
    let pos = ctx.player.position;
    let field_width = ctx.context.field_size.width as f32;
    let field_height = ctx.context.field_size.height as f32;

    let dist_to_line = pos
        .x
        .min(field_width - pos.x)
        .min(pos.y)
        .min(field_height - pos.y);
    let line_factor = (1.0 - dist_to_line / SAFE_RESTART_LINE_MARGIN).clamp(0.0, 1.0);
    if line_factor <= 0.0 {
        return None;
    }

    let nearest_defender = ctx
        .players()
        .opponents()
        .all()
        .filter(|p| !p.tactical_positions.is_goalkeeper())
        .map(|p| (p.id, p.position, (p.position - pos).magnitude()))
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))?;
    let (defender_id, defender_pos, defender_dist) = nearest_defender;
    let press_factor = (1.0 - defender_dist / SAFE_RESTART_PRESS_MARGIN).clamp(0.0, 1.0);
    if press_factor <= 0.0 {
        return None;
    }

    let value = SAFE_RESTART_BASE_VALUE * line_factor * press_factor;
    if value <= 0.0 {
        return None;
    }
    Some(SafeRestartCandidate {
        value,
        target_defender_id: defender_id,
        aim_point: defender_pos,
    })
}

/// Modest, deliberately unglamorous "hold shape / turn back / lay it off
/// safely" floor value. Reasoned starting constant, not sourced — should
/// only ever win the comparison because every other candidate scored
/// lower, never because it's hardcoded as a default fallback (that was
/// the exact bug this whole comparison replaces).
const RECYCLE_VALUE: f32 = 0.15;

pub fn recycle_value(_ctx: &StateProcessingContext) -> f32 {
    RECYCLE_VALUE
}
