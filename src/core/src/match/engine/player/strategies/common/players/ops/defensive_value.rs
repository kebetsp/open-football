//! Defensive threat primitives shared by the unified tackle system
//! (`tackle.rs`, 2026-09-29): goal-side geometry, the tackle-win
//! sigmoid, the "how costly is it to be beaten here" threat value and
//! the covering-teammate check.
//!
//! History: this module used to hold the tackle/contain/concede-out
//! value comparison (D1-D6 of `docs/defensive-decision-logic-spec.md`)
//! and the desperate-slide table. Those were replaced wholesale by the
//! unified tackle system's threat-driven hold / challenge / slide
//! decision and its physical contact model; only the reusable
//! primitives below survive.


use crate::r#match::player::strategies::common::players::ops::on_ball_value::effective_open_angle_at_goal;
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::{MatchPlayerLite, StateProcessingContext};
use nalgebra::Vector3;

/// Goal-side margin (game units) at which "genuinely goal-side" saturates.
/// `pub(crate)`: `on_ball_value.rs`'s attacker-side `safe_restart_value`
/// uses the same distance for its "is the nearest defender genuinely
/// goal-side of me" gate, so both sides of the interaction agree on what
/// goal-side means. (Survivor of the removed `concede_out_value`.)
pub(crate) const CONCEDE_GOAL_SIDE_SATURATION: f32 = 10.0;

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
///
/// `tackle_profile` (D5): the caller's own role-appropriate composite —
/// `DefenderSkillProfile::from_ctx(ctx).tackle_profile` for a defender,
/// `MidfielderSkillProfile::from_ctx(ctx).tackle_profile` for a
/// midfielder — passed in explicitly rather than derived internally, so
/// this same sigmoid works for either role without duplicating it.
pub fn tackle_win_probability(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    tackle_profile: f32,
) -> f32 {
    let minute = sc::minute_from_ms(ctx.context.total_match_time);
    // docs/positional-play-master-design.md, Follow-up #6 (2026-09-11,
    // target: >=8 open-play byline-reaching carries/match): an internal
    // funnel diagnostic found ~98% of byline_commitment_ticks episodes
    // end (ball changes possession) well before the carrier's depth
    // crosses the near-byline threshold, while only ~31% of episodes
    // ever trigger the specific close_defenders/single_blocker
    // containment check -- meaning the dominant attrition is ordinary
    // tackle duels along the way, not the containment/forced-pass
    // mechanism (already fixed above, in `carrier_contained_pass_
    // override`). A real winger fully committed to a byline run is
    // playing with heightened focus and tight ball control specifically
    // for that duel -- not a blanket dribbling buff, only active while
    // `byline_commitment_ticks` is armed, so every other tackle duel in
    // the match is untouched.
    const COMMITTED_RUN_ATTACKER_BONUS: f32 = 0.20;
    let attacker_score = if let Some(att) = ctx.context.players.by_id(opponent.id) {
        let base = sc::dribble_attack(att, minute);
        if att.byline_commitment_ticks > 0 {
            (base + COMMITTED_RUN_ATTACKER_BONUS).min(1.0)
        } else {
            base
        }
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
    let raw_diff = tackle_profile - attacker_score;
    (1.0 / (1.0 + (-raw_diff * 2.4).exp())).clamp(0.06, 0.55)
}

/// The tackle sigmoid's own upper clamp (see `tackle_win_probability`) —
/// named here so `contain_value`/`concede_out_value`'s "is the tackle
/// already clearly favourable" checks read against the real ceiling
/// rather than a re-guessed number.

/// Reference figures for `value_conceded_if_beaten` — reasoned estimates
/// (a distance/angle danger PROXY, not literal xG; building a real
/// skill-aware xG for an opposing attacker from the defender's own ctx
/// would need constructing a full `ShotSkillProfile` for that opponent —
/// extra plumbing not justified for this isolated D1 primitive). Flagged
/// per this project's own Realism Diagnosis Framework category (d) — an
/// explicitly-flagged estimate, not silently treated as ground truth.
const DANGER_DISTANCE_HORIZON: f32 = 350.0; // beyond this, being beaten barely matters
const DANGER_MAX_OPEN_ANGLE: f32 = 1.2; // radians (~69°) treated as "fully dangerous"
/// 2026-08-22 redesign (docs/defensive-decision-logic-spec.md, "walk me
/// through the formula" conversation): how far a delivery (pass/cross/
/// cutback) to a supporting teammate is still considered plausible from
/// the attacker's current position. Category (d), reasoned not sourced.
const DELIVERY_RANGE: f32 = 200.0; // ~25m, a generous cross/cutback range
/// Cover-multiplier shape: 0 covering teammates (last man) amplifies
/// danger toward `COVER_MULTIPLIER_CEIL`; more covering teammates only
/// ever relaxes it back toward neutral (`COVER_MULTIPLIER_FLOOR = 1.0`),
/// NEVER below the raw `attacking_threat` reading. 2026-08-22, Pavel's
/// own correction: cover reduces the probability a dangerous event
/// SCORES, not the probability it HAPPENS at all — a crowded-box header
/// off a cross is still a genuine chance regardless of how many bodies
/// are there, so numbers back should add urgency when absent, but must
/// never claim a genuinely dangerous moment is safe just because
/// teammates are nearby. Category (d), reasoned not sourced.
const COVER_MULTIPLIER_BASE: f32 = 1.5;
const COVER_MULTIPLIER_PER_TEAMMATE: f32 = 0.35;
const COVER_MULTIPLIER_FLOOR: f32 = 1.0;
const COVER_MULTIPLIER_CEIL: f32 = 1.5;

/// Pure "how dangerous is a shot from exactly this spot" — the original
/// (pre-2026-08-22) `value_conceded_if_beaten` body verbatim, now a
/// private helper reused for both the attacker himself and any
/// supporting teammate he could realistically deliver to. Same occluded-
/// angle geometry `on_ball_value.rs`'s `effective_open_angle` uses for
/// real shot decisions, via `effective_open_angle_at_goal` (goal-
/// parameterized so it works from the DEFENDER's ctx).
fn spot_shot_danger(
    pos: Vector3<f32>,
    own_goal: Vector3<f32>,
    gk_pos: Option<Vector3<f32>>,
) -> f32 {
    let dist = (pos - own_goal).magnitude();
    let angle = effective_open_angle_at_goal(pos, own_goal, gk_pos);
    let distance_factor = (1.0 - (dist / DANGER_DISTANCE_HORIZON).clamp(0.0, 1.0)).powf(1.5);
    let angle_factor = (angle / DANGER_MAX_OPEN_ANGLE).clamp(0.0, 1.0);
    (distance_factor * (0.4 + 0.6 * angle_factor)).clamp(0.0, 1.0)
}

/// 2026-08-22 redesign: replaces the old self-only angle_factor with the
/// MAX of (a) the attacker's own shot danger from where he stands, and
/// (b) the best danger any of his nearby teammates would pose if he
/// delivered to them instead (a cross/cutback threat), discounted by how
/// plausible that delivery is at this range. This is what fixes the
/// byline case Pavel raised: `spot_shot_danger(self)`≈0 there (angle is
/// nearly 0 from the byline), but a teammate waiting in the box can
/// still dominate the MAX — the old formula had no way to represent
/// that at all.
fn attacking_threat(
    ctx: &StateProcessingContext,
    attacker: &MatchPlayerLite,
    own_goal: Vector3<f32>,
    gk_pos: Option<Vector3<f32>>,
) -> f32 {
    let self_danger = spot_shot_danger(attacker.position, own_goal, gk_pos);
    let attacker_id = attacker.id;
    let best_teammate = ctx
        .players()
        .opponents()
        .all()
        .filter(|p| p.id != attacker_id)
        .map(|teammate| {
            let delivery_dist = (teammate.position - attacker.position).magnitude();
            let deliverability = (1.0 - delivery_dist / DELIVERY_RANGE).clamp(0.0, 1.0);
            deliverability * spot_shot_danger(teammate.position, own_goal, gk_pos)
        })
        .fold(0.0f32, f32::max);
    self_danger.max(best_teammate)
}

/// 2026-08-22 redesign: counts DEFENDING outfield teammates (not the
/// duelling defender himself, not the GK — a GK-only 1v1 is already
/// represented by `spot_shot_danger`'s own distance/angle terms once the
/// attacker closes in, not double-counted here) who are goal-side of the
/// attacker right now. Zero cover means this defender IS the last line —
/// a missed duel is a clean break, amplify. Two or more covering
/// teammates means a missed duel is a recoverable mistake, dampen.
fn cover_multiplier(
    ctx: &StateProcessingContext,
    attacker_pos: Vector3<f32>,
    own_goal: Vector3<f32>,
) -> f32 {
    let my_id = ctx.player.id;
    let teammates = ctx.players().teammates();
    let cover_count = teammates
        .defenders()
        .chain(teammates.midfielders())
        .filter(|t| t.id != my_id)
        .filter(|t| is_goal_side(t.position, attacker_pos, own_goal) > 0.0)
        .count();
    (COVER_MULTIPLIER_BASE - COVER_MULTIPLIER_PER_TEAMMATE * cover_count as f32)
        .clamp(COVER_MULTIPLIER_FLOOR, COVER_MULTIPLIER_CEIL)
}

/// How costly it would be to get beaten by this attacker right now, in
/// [0, 1]. 2026-08-22 redesign (docs/defensive-decision-logic-spec.md):
/// `attacking_threat` (self shot danger OR best-teammate delivery
/// danger) times `cover_multiplier` (this defender's own team shape) —
/// see both functions' own doc comments for why the original self-only
/// angle-based formula missed each of these.
pub fn value_conceded_if_beaten(ctx: &StateProcessingContext, opponent: &MatchPlayerLite) -> f32 {
    let own_goal = ctx.ball().direction_to_own_goal();
    let gk_pos = ctx
        .players()
        .teammates()
        .all()
        .find(|p| p.tactical_positions.is_goalkeeper())
        .map(|g| g.position);
    let threat = attacking_threat(ctx, opponent, own_goal, gk_pos);
    let cover = cover_multiplier(ctx, opponent.position, own_goal);
    (threat * cover).clamp(0.0, 1.0)
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
