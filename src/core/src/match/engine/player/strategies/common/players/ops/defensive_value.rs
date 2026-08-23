//! Defensive decision logic (see `docs/defensive-decision-logic-spec.md`
//! in The Gaffer repo). Mirrors `on_ball_value.rs`'s MAX-comparison
//! architecture for the other side of the ball: instead of the current
//! hardwired "always attempt a tackle once in range" behavior, a
//! ball-facing defender gets a genuine three-way comparison between
//! tackling, containing (jockey without committing), and deliberately
//! conceding a restart — mirroring real defensive doctrine (show the
//! attacker outside, don't dive in unless the moment is right).
//!
//! **D1 (this stage, 2026-08-20): the primitives below, built and
//! hand-verified in isolation. NOT yet wired into any state** —
//! `DefenderTacklingState` still always attempts a tackle once in range;
//! wiring the actual MAX comparison into `DefenderPressingState`/
//! `DefenderTacklingState` is D2. `tackle_win_probability` is the one
//! exception to "new code": it's a behavior-preserving EXTRACTION of the
//! success-chance sigmoid that already lived inline in
//! `DefenderTacklingState::attempt_sliding_tackle` (tackling/mod.rs) —
//! same formula, same constants, same 0.06-0.55 clamp, now queryable
//! without re-deriving or re-tuning it, exactly as the spec requires.
//!
//! Scale warning, flagged honestly rather than assumed solved: `tackle_
//! value`/`contain_value`/`concede_out_value`'s relative weights
//! (`TACKLE_WIN_VALUE`, `CONTAIN_BASE_VALUE`, `CONCEDE_BASE_VALUE`) are
//! reasoned starting points, NOT yet cross-calibrated against each other
//! with real match data. This is exactly the trap `on_ball_value.rs`'s
//! own `SHOOT_COMPARISON_SCALE` fix (2026-08-20, same day) found and
//! fixed for the attacker side: two genuinely different value currencies
//! can look comparable in isolation and still be wildly mismatched once
//! actually compared. Reconciling these three constants against
//! real match-log data (the way `SHOOT_COMPARISON_SCALE` was iterated
//! 8/10/15 against real batch outcomes) is explicitly a D2 task.

use crate::r#match::player::strategies::common::players::ops::defender_skill::DefenderSkillProfile;
use crate::r#match::player::strategies::common::players::ops::on_ball_value::{
    SAFE_RESTART_LINE_MARGIN, effective_open_angle_at_goal,
};
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::{MatchPlayerLite, PlayerSide, StateProcessingContext};
use nalgebra::Vector3;

/// Signed margin of `defender_pos` along the `attacker_pos -> own_goal_pos`
/// direction, relative to the attacker. Positive means the defender is
/// genuinely between the attacker and goal (goal-side); negative means
/// the defender is trailing/beaten — further from goal than the attacker
/// along that same axis, even if nearby in absolute distance. Zero means
/// level with the attacker.
///
/// Pure geometry, no `StateProcessingContext` needed — this is the one
/// primitive everything else in this module depends on. This exact
/// formula was already externally verified against 25 real matches
/// before being written here: `d0_goalside.mjs` (scratchpad, 2026-08-20
/// D0 diagnostic) computed this same signed-margin quantity from raw
/// exported position data and found defenders trailing more often than
/// not near the byline (mean margin -3.0, only 38.4% goal-side when
/// engaged, n=73) — this Rust implementation reproduces an
/// already-validated measurement, not a fresh, unverified formula.
pub fn is_goal_side(
    defender_pos: Vector3<f32>,
    attacker_pos: Vector3<f32>,
    own_goal_pos: Vector3<f32>,
) -> f32 {
    let to_goal = own_goal_pos - attacker_pos;
    let len = to_goal.magnitude();
    if len < 0.01 {
        return 0.0;
    }
    let dir = to_goal / len;
    let to_defender = defender_pos - attacker_pos;
    to_defender.dot(&dir)
}

/// D1 (`docs/attacking-defensive-decision-rebuild-plan.md` §2.3) —
/// jockey/"show outside" lateral bias magnitude (engine units). Real
/// doctrine (research doc §9.3): "the defender's job is to position
/// their body between the attacker and the danger area... and
/// physically steer the attacker toward the touchline" — a lateral
/// bias to a presser's own closing target, distinct from `is_goal_side`
/// above (goal-line axis) and the D3/D4 tackle/contain/concede-out
/// comparison this deliberately does NOT touch (§0.3: this is HOW a
/// defender closes down, D3/D4 is WHAT he decides once engaged).
/// Reasoned estimate — no sourced real-world figure for this specific
/// lateral offset, same status as every other new constant this
/// session — kept modest relative to the existing goal-side bias
/// (2-12u in the pressing states) so this nudges the closing ANGLE
/// rather than dominating the direct chase.
pub const JOCKEY_SHOW_OUTSIDE_BIAS: f32 = 10.0;

/// Signed lateral (y-axis) direction a presser's own intercept target
/// should lean to show the carrier outside: away from whichever
/// touchline `carrier_y` already sits nearer to (the carrier's own
/// "outside"), toward the pitch centre-line (the carrier's "inside") —
/// so the defender's body naturally ends up between the carrier and
/// the central/goal-scoring area, and the only route left open is the
/// one toward the touchline. A hard binary split on which half the
/// carrier occupies, matching this codebase's existing convention for
/// side-relative signs (e.g. `PlayerSide`/attacking-direction flips)
/// rather than a continuous gradient with no reference doctrine to
/// calibrate its shape against.
/// `show_outside` — P6 AI lever (`defensiveBias.showSide`): `true` (the
/// real-doctrine default) keeps the base direction; `false` ("show
/// inside") flips it, biasing the presser's body to the outside lane
/// instead and inviting the carrier to cut infield.
pub fn jockey_show_outside_sign(carrier_y: f32, field_height: f32, show_outside: bool) -> f32 {
    let base = if carrier_y < field_height / 2.0 {
        1.0
    } else {
        -1.0
    };
    if show_outside { base } else { -base }
}

/// Existing, already-calibrated tackle-success sigmoid, extracted
/// verbatim from `DefenderTacklingState::attempt_sliding_tackle`
/// (tackling/mod.rs) so it can be queried without re-deriving or
/// re-tuning it. See that function's own doc comments for the
/// calibration history behind the 2.4 steepness and 0.06-0.55 clamp
/// (weak-team-competitiveness rationale) — untouched here, and
/// `attempt_sliding_tackle` itself now calls this function instead of
/// duplicating the computation, so behavior is byte-identical to before.
pub fn tackle_win_probability(ctx: &StateProcessingContext, opponent: &MatchPlayerLite) -> f32 {
    let minute = sc::minute_from_ms(ctx.context.total_match_time);
    let def_profile = DefenderSkillProfile::from_ctx(ctx);
    let attacker_score = if let Some(att) = ctx.context.players.by_id(opponent.id) {
        sc::dribble_attack(att, minute)
    } else {
        let dribbling;
        let agility;
        {
            let players = ctx.player();
            let s = players.skills(opponent.id);
            dribbling = sc::n(s.technical.dribbling);
            agility = sc::n(s.physical.agility);
        }
        (dribbling + agility) * 0.5
    };
    let raw_diff = def_profile.tackle_profile - attacker_score;
    (1.0 / (1.0 + (-raw_diff * 2.4).exp())).clamp(0.06, 0.55)
}

/// The tackle sigmoid's own upper clamp (see `tackle_win_probability`) —
/// named here so `contain_value`/`concede_out_value`'s "is the tackle
/// already clearly favourable" checks read against the real ceiling
/// rather than a re-guessed number.
const TACKLE_WIN_PROB_CEILING: f32 = 0.55;

/// Reference figures for `value_conceded_if_beaten` — reasoned estimates
/// (a distance/angle danger PROXY, not literal xG; building a real
/// skill-aware xG for an opposing attacker from the defender's own ctx
/// would need constructing a full `ShotSkillProfile` for that opponent —
/// extra plumbing not justified for this isolated D1 primitive). Flagged
/// per this project's own Realism Diagnosis Framework category (d) — an
/// explicitly-flagged estimate, not silently treated as ground truth.
const DANGER_DISTANCE_HORIZON: f32 = 350.0; // beyond this, being beaten barely matters
const DANGER_MAX_OPEN_ANGLE: f32 = 1.2; // radians (~69°) treated as "fully dangerous"

/// How costly it would be to get beaten by this attacker right now, in
/// [0, 1]. Built on the SAME occluded-angle geometry `on_ball_value.rs`'s
/// `effective_open_angle` already uses for shot decisions, generalized
/// via `effective_open_angle_at_goal` to take an explicit goal position
/// — `effective_open_angle` itself hardcodes `ctx.player().
/// opponent_goal_position()`, correct when `ctx.player` IS the shooter,
/// wrong here where `ctx.player` is the DEFENDER and the relevant goal
/// is the defender's OWN goal (`ctx.ball().direction_to_own_goal()`).
pub fn value_conceded_if_beaten(ctx: &StateProcessingContext, attacker_pos: Vector3<f32>) -> f32 {
    let own_goal = ctx.ball().direction_to_own_goal();
    let dist = (attacker_pos - own_goal).magnitude();
    let gk_pos = ctx
        .players()
        .teammates()
        .all()
        .find(|p| p.tactical_positions.is_goalkeeper())
        .map(|g| g.position);
    let angle = effective_open_angle_at_goal(attacker_pos, own_goal, gk_pos);

    let distance_factor = (1.0 - (dist / DANGER_DISTANCE_HORIZON).clamp(0.0, 1.0)).powf(1.5);
    let angle_factor = (angle / DANGER_MAX_OPEN_ANGLE).clamp(0.0, 1.0);
    (distance_factor * (0.4 + 0.6 * angle_factor)).clamp(0.0, 1.0)
}

const TACKLE_WIN_VALUE: f32 = 0.5;
const TACKLE_FOUL_RISK_COST_BASE: f32 = 0.15;
const TACKLE_FOUL_RISK_COST_OWN_BOX_MULT: f32 = 2.0; // a foul in the box risks a penalty

/// `tackle_value = win_prob * WIN_VALUE - (1 - win_prob) * (foul_risk +
/// value_conceded_if_beaten)` — the existing coin-flip, reframed as a
/// comparable value rather than the only outcome that can happen. Does
/// NOT touch `tackle_win_probability`'s calibration; only adds cost
/// terms alongside it.
pub fn tackle_value(ctx: &StateProcessingContext, opponent: &MatchPlayerLite) -> f32 {
    let p = tackle_win_probability(ctx, opponent);
    let in_own_box = ctx
        .context
        .penalty_area(ctx.player.side == Some(PlayerSide::Left))
        .contains(&ctx.tick_context.positions.ball.position);
    let foul_risk_cost = if in_own_box {
        TACKLE_FOUL_RISK_COST_BASE * TACKLE_FOUL_RISK_COST_OWN_BOX_MULT
    } else {
        TACKLE_FOUL_RISK_COST_BASE
    };
    let conceded = value_conceded_if_beaten(ctx, opponent.position);
    p * TACKLE_WIN_VALUE - (1.0 - p) * (foul_risk_cost + conceded)
}

/// Range within which a teammate's converging movement counts as genuine
/// cover for this duel, and the alignment threshold for "actively closing
/// in" vs. merely nearby — mirrors `DefensiveOperationsImpl::
/// is_opponent_being_engaged`'s own velocity-alignment check
/// (defensive.rs), applied here to "is someone else backing ME up"
/// rather than "is someone else already marking this opponent."
const COVER_CONVERGENCE_RADIUS: f32 = 60.0;
const COVER_CONVERGENCE_ALIGNMENT: f32 = 0.3;
const COVER_STATIONARY_CLOSE: f32 = 15.0;

/// Is a teammate (other than this defender) genuinely converging on the
/// attacker as cover — close enough, and moving toward them with real
/// alignment, not just standing nearby or drifting elsewhere.
pub fn has_covering_teammate(ctx: &StateProcessingContext, attacker_pos: Vector3<f32>) -> bool {
    ctx.players()
        .teammates()
        .defenders()
        .filter(|d| d.id != ctx.player.id)
        .any(|teammate| {
            let dist = (teammate.position - attacker_pos).magnitude();
            if dist > COVER_CONVERGENCE_RADIUS {
                return false;
            }
            let velocity = teammate.velocity(ctx);
            let speed = velocity.norm();
            if speed > 0.5 {
                let to_attacker = (attacker_pos - teammate.position).normalize();
                velocity.normalize().dot(&to_attacker) > COVER_CONVERGENCE_ALIGNMENT
            } else {
                dist < COVER_STATIONARY_CLOSE
            }
        })
}

const CONTAIN_BASE_VALUE: f32 = 0.35;
/// How far into a comfortably-goal-side margin (game units) before
/// `goal_side_factor` saturates at 1.0 — ~2.5m, a reasoned estimate.
const CONTAIN_GOAL_SIDE_SATURATION: f32 = 20.0;
/// Ticks before containment's urgency term meaningfully erodes its
/// value (~1.5s at 10ms/tick) — a reasoned estimate flagged per the
/// spec's own risk note: real jockeying has a clock on it, or defenders
/// will contain indefinitely and never engage.
const CONTAIN_URGENCY_DECAY_TICKS: f32 = 150.0;

/// The value of NOT committing to a tackle yet. High when containment is
/// actually working (comfortably goal-side, not just nearby), a
/// teammate is genuinely converging as cover, and the tackle itself
/// isn't yet clearly favourable. Decays with `contain_duration_ticks` so
/// jockeying isn't a free, indefinite stall.
///
/// `contain_duration_ticks` is an explicit PARAMETER, not derived
/// internally — no persistent "how long have I been containing" field
/// exists on `MatchPlayer` yet (D1 keeps this function pure/testable in
/// isolation). Wiring a real per-player counter (mirroring the
/// established `kickoff_pass_pending`/`throw_in_pass_pending` pattern)
/// is D2's job, alongside the actual state wiring.
pub fn contain_value(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    contain_duration_ticks: f32,
) -> f32 {
    let own_goal = ctx.ball().direction_to_own_goal();
    let margin = is_goal_side(ctx.player.position, opponent.position, own_goal);
    if margin <= 0.0 {
        // Not actually goal-side — containing isn't working, nothing to
        // reward. This is the concrete mechanism behind the spec's own
        // framing: "high when containment IS working, not failing."
        return 0.0;
    }
    let goal_side_factor = (margin / CONTAIN_GOAL_SIDE_SATURATION).clamp(0.0, 1.0);
    let cover_bonus = if has_covering_teammate(ctx, opponent.position) {
        1.0
    } else {
        0.5
    };
    let p = tackle_win_probability(ctx, opponent);
    // Not-yet-favourable tackle odds make containing MORE attractive —
    // no reason to jockey once the tackle is already clearly winning.
    let non_favorable_factor = (1.0 - (p / TACKLE_WIN_PROB_CEILING)).clamp(0.0, 1.0);
    let urgency = (1.0 - contain_duration_ticks / CONTAIN_URGENCY_DECAY_TICKS).clamp(0.0, 1.0);
    // D2 real-match measurement (2026-08-21, docs/defensive-decision-
    // logic-spec.md): the first cut of this function had no notion that
    // continuing to contain still lets a genuinely dangerous attacker
    // keep the ball — contain buys TIME, it doesn't neutralize the
    // threat the way a won tackle or a conceded restart does. Isolated
    // regression (D1+D2 vs D1-only, two 24-30 match rounds each, same
    // pre-existing unrelated WIP on both sides) showed a real, repeated
    // goals/match increase (pooled ~4.70 vs ~3.91) with foul-rate flat
    // (~15.4 both sides — the spec's foul-coupling risk did NOT
    // materialize, only this one did). Discounting by how dangerous the
    // attacker's position already is fixes the actual mechanism: a
    // defender in the six-yard box facing a striker doesn't patiently
    // contain in real football, they commit — this factor makes that
    // the model's own behavior rather than tuning an unrelated constant.
    let danger_discount = 1.0 - value_conceded_if_beaten(ctx, opponent.position);
    CONTAIN_BASE_VALUE
        * goal_side_factor
        * cover_bonus
        * (0.5 + 0.5 * non_favorable_factor)
        * urgency
        * danger_discount
}

const CONCEDE_BASE_VALUE: f32 = 0.30;
/// Same saturation distance as `contain_value`'s goal-side factor —
/// conceding only makes sense as the payoff of containment that's
/// already working, so both use the same "comfortably goal-side" bar.
const CONCEDE_GOAL_SIDE_SATURATION: f32 = 20.0;

/// The value of deliberately conceding a restart — high near a boundary
/// line (reuses `on_ball_value.rs`'s own `SAFE_RESTART_LINE_MARGIN` for
/// the same geometry the attacker-side `safe_restart_value` uses), when
/// the defender is genuinely goal-side and has shown the attacker toward
/// that line (conceding as the outcome of a working plan, not a panic
/// response — gated on `is_goal_side` being positive, same mechanism
/// `contain_value` uses), and when the tackle itself is low-to-mediocre
/// (if it's already clearly favourable, just tackle).
///
/// D1 scope: this returns a comparable SCORE only. The actual execution
/// mechanism (which of `DefenderClearingState`'s byline-aim extension or
/// a dedicated block/deflection action actually strikes the ball) is
/// explicitly D3's job, per the spec's own staging — not attempted here.
pub fn concede_out_value(ctx: &StateProcessingContext, opponent: &MatchPlayerLite) -> f32 {
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
        return 0.0;
    }

    let own_goal = ctx.ball().direction_to_own_goal();
    let margin = is_goal_side(pos, opponent.position, own_goal);
    let goal_side_factor = (margin / CONCEDE_GOAL_SIDE_SATURATION).clamp(0.0, 1.0);
    if goal_side_factor <= 0.0 {
        return 0.0;
    }

    let p = tackle_win_probability(ctx, opponent);
    let non_favorable_factor = (1.0 - (p / TACKLE_WIN_PROB_CEILING)).clamp(0.0, 1.0);

    CONCEDE_BASE_VALUE * line_factor * goal_side_factor * non_favorable_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defender_directly_between_attacker_and_goal_scores_positive() {
        let attacker = Vector3::new(700.0, 272.5, 0.0);
        let goal = Vector3::new(840.0, 272.5, 0.0);
        let defender = Vector3::new(770.0, 272.5, 0.0); // 70u closer to goal than attacker
        let margin = is_goal_side(defender, attacker, goal);
        assert!(
            margin > 0.0,
            "expected positive (goal-side) margin, got {margin}"
        );
        assert!((margin - 70.0).abs() < 0.01, "expected margin ~70, got {margin}");
    }

    #[test]
    fn defender_trailing_behind_attacker_scores_negative() {
        let attacker = Vector3::new(700.0, 272.5, 0.0);
        let goal = Vector3::new(840.0, 272.5, 0.0);
        let defender = Vector3::new(650.0, 272.5, 0.0); // 50u further from goal — beaten
        let margin = is_goal_side(defender, attacker, goal);
        assert!(
            margin < 0.0,
            "expected negative (trailing) margin, got {margin}"
        );
        assert!((margin + 50.0).abs() < 0.01, "expected margin ~-50, got {margin}");
    }

    #[test]
    fn defender_level_with_attacker_scores_near_zero() {
        let attacker = Vector3::new(700.0, 272.5, 0.0);
        let goal = Vector3::new(840.0, 272.5, 0.0);
        let defender = Vector3::new(700.0, 290.0, 0.0); // pure lateral offset, no x progress
        let margin = is_goal_side(defender, attacker, goal);
        assert!(margin.abs() < 0.01, "expected ~0 margin, got {margin}");
    }

    #[test]
    fn is_goal_side_handles_non_axis_aligned_goal_direction() {
        // Attacker wide near the right byline; goal at the near post.
        // Defender tucked slightly inside and behind — should still read
        // as goal-side despite the attacker->goal line not being
        // horizontal.
        let attacker = Vector3::new(800.0, 450.0, 0.0);
        let goal = Vector3::new(840.0, 272.5, 0.0);
        let defender = Vector3::new(790.0, 430.0, 0.0);
        let margin = is_goal_side(defender, attacker, goal);
        assert!(
            margin > 0.0,
            "expected a defender tucked inside/behind to read goal-side, got {margin}"
        );
    }
}
