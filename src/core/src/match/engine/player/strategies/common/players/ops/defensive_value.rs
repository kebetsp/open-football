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

/// Box-edge exit from the jockey hold (2026-10-25). The hold-at-2.5 m
/// delay (docs/defensive-delay-and-engagement-timing.md) is right in the
/// middle third, but its exits (xG danger >= 0.15, touchline, heavy touch,
/// cover ready) know nothing about the penalty area: a carrier 16.5-30 m
/// out has far less than 0.15 xG-danger, so the presser kept holding until
/// the carrier was already in the box. Measured 2026-10-25 (30 matches):
/// presser trails at a median 3.7-4.1 m with ~0 closing speed, within
/// lunge reach only 24-29% of the time. Once the carrier is this close to
/// our goal (190u = 23.75 m, i.e. ~7 m outside the 132u box) the presser
/// commits and meets him. Reasoned estimate (category d).
/// ON by default; `BOXEDGE_COMMIT_OFF=1` restores the old behaviour.
/// Measured 2026-10-25 (60-match arms, replicated): with this plus the
/// geometric hold cost in `tackle::choose`, dribbled shots fell ~34%
/// (268/250 -> 177/167 per 60 matches) and their median distance rose from
/// ~9.1 to ~10.8 m, tackle contacts 11.9 -> 13.5 per match. Costs: total
/// shots -18%, penalties 0.27 -> 0.43 per match, defenders in the box at a
/// shot 5.8 -> 5.5. Alone (without the geometric hold cost) the commit
/// pieces gave most of the effect.
pub const BOX_EDGE_COMMIT_UNITS: f32 = 190.0;

/// Canonical penalty area (16.5 m deep, 40.3 m wide at 8u/m) as the engine's
/// own `penalty_area()` uses it.
const BOX_DEPTH_UNITS: f32 = 132.0;
const BOX_HALF_WIDTH_UNITS: f32 = 161.3;
/// The first defender holds the entrance ~1 m inside the box edge, not deeper,
/// while the carrier is still outside the box (2026-10-25: engine carries into
/// the box were a median 25 m long from 37 m out, real 14 m from 21 m,
/// because the presser backpedalled into the box with the carrier).
pub const ENTRANCE_HOLD_UNITS: f32 = 140.0;

/// Is this position inside the penalty area we defend?
pub fn carrier_in_box(carrier_pos: Vector3<f32>, own_goal: Vector3<f32>) -> bool {
    (carrier_pos.x - own_goal.x).abs() <= BOX_DEPTH_UNITS
        && (carrier_pos.y - own_goal.y).abs() <= BOX_HALF_WIDTH_UNITS
}

/// Keep a presser's target from retreating deeper than the entrance line.
pub fn clamp_to_entrance(target: Vector3<f32>, own_goal: Vector3<f32>) -> Vector3<f32> {
    let dir = if target.x >= own_goal.x { 1.0 } else { -1.0 };
    let depth = (target.x - own_goal.x).abs();
    if depth >= ENTRANCE_HOLD_UNITS {
        return target;
    }
    Vector3::new(own_goal.x + dir * ENTRANCE_HOLD_UNITS, target.y, target.z)
}

/// True when the carrier is OUTSIDE the box but close enough to our goal
/// that a presser must stop delaying and engage. Inside the box the old
/// (careful) behaviour is kept: a foul there is a penalty.
pub fn box_edge_commit(carrier_pos: Vector3<f32>, own_goal: Vector3<f32>) -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    if *OFF.get_or_init(|| std::env::var_os("BOXEDGE_COMMIT_OFF").is_some()) {
        return false;
    }
    if carrier_in_box(carrier_pos, own_goal) {
        return false;
    }
    (carrier_pos - own_goal).magnitude() <= BOX_EDGE_COMMIT_UNITS
}

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
