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
//!
//! **D5 (2026-08-22): role-generalized.** Every function that needs a
//! tackle-skill composite (`tackle_win_probability`, `slide_connect_
//! quality`, and everything downstream of them — `tackle_value`,
//! `contain_value`, `concede_out_value`, `slide_outcome_probs`,
//! `desperate_slide_value`) now takes `tackle_profile: f32` as an
//! explicit parameter instead of deriving it internally via
//! `DefenderSkillProfile::from_ctx(ctx)`. This is what makes the module
//! usable from `MidfielderTacklingState`/`MidfielderPressingState` (D5)
//! as well as the defender states it was built for — the caller supplies
//! its own role-appropriate profile's `tackle_profile` field
//! (`DefenderSkillProfile`/`MidfielderSkillProfile` both expose one,
//! same name, same 0..1 scale). Behavior-preserving for every existing
//! defender call site: each one now passes
//! `DefenderSkillProfile::from_ctx(ctx).tackle_profile` explicitly where
//! it used to be computed inside these functions — same value, same
//! formula, just threaded through instead of hidden. Forwards are
//! deliberately NOT wired to this module (see D5 results in the spec
//! doc) — their tackling model is a separately-tuned, purpose-built
//! counter-press system, not a gap this generalization is meant to fill.
//!
//! **D6 (2026-08-22): "push it out" is now a genuine, gated decision, not
//! a distance cliff.** Two fixes, both raised directly by Pavel reading
//! a raw zone breakdown of 220 real push-out events: (1) `concede_out_
//! value` never checked whether the attacker was actually dangerous —
//! a defender near a boundary against a harmless attacker could still
//! "decide" to concede a corner/throw-in instead of pressing/tackling
//! normally. Fixed with a new `danger_factor` (reuses `value_conceded_
//! if_beaten`, the same danger primitive `contain_value` and the D4.6
//! slide's own trigger already use — not a new concept, applied where
//! it was missing). (2) The "close enough to a boundary" question was a
//! flat `SAFE_RESTART_LINE_MARGIN` distance, which the raw data showed
//! was both too generous in places (76/113 pushes from just outside the
//! box still got reclaimed) and structurally wrong for D4.6's slide,
//! whose SUCCESS outcome fired `nearest_boundary_edge` unconditionally
//! with no distance check at all — 82/84 slides from 17-33% of the
//! pitch length out (well outside the box, dead central) got reclaimed
//! because the target was never reachable in the first place. Replaced
//! with `boundary_push_viability`, a genuine race check (ball's real
//! strike speed vs. the fastest-closing opponent's `time_to_intercept`
//! to the SAME point) — reusing `on_ball_value.rs`'s already-proven
//! "who wins the race to a point" primitive (the same one `control_prob`
//! is built on) rather than inventing a second reachability model.

use crate::r#match::player::strategies::common::players::ops::on_ball_value::{
    SAFE_RESTART_LINE_MARGIN, effective_open_angle_at_goal, time_to_intercept,
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
    let raw_diff = tackle_profile - attacker_score;
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

const TACKLE_WIN_VALUE: f32 = 0.5;
const TACKLE_FOUL_RISK_COST_BASE: f32 = 0.15;
const TACKLE_FOUL_RISK_COST_OWN_BOX_MULT: f32 = 2.0; // a foul in the box risks a penalty

/// `tackle_value = win_prob * WIN_VALUE - (1 - win_prob) * (foul_risk +
/// value_conceded_if_beaten)` — the existing coin-flip, reframed as a
/// comparable value rather than the only outcome that can happen. Does
/// NOT touch `tackle_win_probability`'s calibration; only adds cost
/// terms alongside it. `tackle_profile`: see `tackle_win_probability`.
pub fn tackle_value(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    tackle_profile: f32,
) -> f32 {
    let p = tackle_win_probability(ctx, opponent, tackle_profile);
    let in_own_box = ctx
        .context
        .penalty_area(ctx.player.side == Some(PlayerSide::Left))
        .contains(&ctx.tick_context.positions.ball.position);
    let foul_risk_cost = if in_own_box {
        TACKLE_FOUL_RISK_COST_BASE * TACKLE_FOUL_RISK_COST_OWN_BOX_MULT
    } else {
        TACKLE_FOUL_RISK_COST_BASE
    };
    let conceded = value_conceded_if_beaten(ctx, opponent);
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
    tackle_profile: f32,
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
    let p = tackle_win_probability(ctx, opponent, tackle_profile);
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
    let danger_discount = 1.0 - value_conceded_if_beaten(ctx, opponent);
    CONTAIN_BASE_VALUE
        * goal_side_factor
        * cover_bonus
        * (0.5 + 0.5 * non_favorable_factor)
        * urgency
        * danger_discount
}

// D3 recalibration (2026-08-21): raised 0.30 -> 0.40 after raw match-log
// measurement showed concede_out_value structurally could not win the
// comparison even in its own best-case scenario. At the closest observed
// real "near boundary + comfortably goal-side" duel (line_factor=0.43,
// goal_side_factor=0.27, tackle not favourable), contain_value still
// scored ~1.9x higher — not because containing was truly better here,
// but because contain_value has no boundary-proximity requirement at
// all (concede pays that tax alone) while its own "tackle not yet
// favourable" term floors at 0.5 rather than crushing toward 0 the way
// concede's did before the floor fix just below. 0.40 was sized so
// concede's product in that exact measured scenario (line_factor=0.43 *
// goal_side_factor=0.27 * floored_non_favorable=0.785 ~= 0.091) reaches
// roughly the same order of magnitude contain scored there (~0.037,
// with no covering teammate) — not a precise cross-calibration, an
// isolated first pass sized against one real observed case, flagged for
// further tuning once concede-out fires enough to measure its actual
// hit rate.
const CONCEDE_BASE_VALUE: f32 = 0.40;
/// D3 recalibration: deliberately DECOUPLED from
/// `CONTAIN_GOAL_SIDE_SATURATION` (was a shared 20.0). Concede's own
/// `line_factor` already imposes the harder, boundary-proximity bar
/// this branch needs; requiring the SAME "comfortably goal-side" margin
/// contain uses on top compounds two independently-rare conditions (D0's
/// own measurement: only 8.2% of real byline duels reach margin>10u at
/// all) into an almost-never-both-at-once joint requirement. A lower
/// bar here is also doctrinally defensible on its own terms: once the
/// ball is genuinely near the line, being MODESTLY goal-side (a few
/// units) is already "showing him outside" — you don't need maximal
/// goal-side margin to justify putting a ball out that's basically at
/// the touchline already.
/// `pub(crate)`: D4 (`on_ball_value.rs`'s `safe_restart_value`) reuses
/// this exact saturation distance for the attacker-side "is the nearest
/// defender genuinely goal-side of me" gate -- the same convention D1
/// already established the other direction with `SAFE_RESTART_LINE_
/// MARGIN`, so both sides of the interaction agree on what "goal-side"
/// means rather than drifting to two independently-tuned numbers.
pub(crate) const CONCEDE_GOAL_SIDE_SATURATION: f32 = 10.0;

// ── D6: reachability + danger primitives, shared by concede_out_value
// below and the D4.6 slide's success-outcome direction choice
// (pressing/mod.rs) ────────────────────────────────────────────────────

/// Nearest of the four boundary edges from `ball_pos` — moved here from
/// `player/events/players.rs` (D6) so the DECISION side (this module)
/// and the EXECUTION side (players.rs) target the exact same point
/// instead of two independently-derived copies. `players.rs` now calls
/// this instead of a private `Self::` copy.
pub fn nearest_boundary_edge(
    ball_pos: Vector3<f32>,
    field_width: f32,
    field_height: f32,
) -> Vector3<f32> {
    let candidates = [
        (ball_pos.x, Vector3::new(0.0, ball_pos.y, 0.0)),
        (
            field_width - ball_pos.x,
            Vector3::new(field_width, ball_pos.y, 0.0),
        ),
        (ball_pos.y, Vector3::new(ball_pos.x, 0.0, 0.0)),
        (
            field_height - ball_pos.y,
            Vector3::new(ball_pos.x, field_height, 0.0),
        ),
    ];
    candidates
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|c| c.1)
        .unwrap_or(ball_pos)
}

/// Strike-speed constants for a CONTROLLED push (the `DEF_CONCEDE_OUT`
/// clearance and the new `DEF_SLIDE_SUCCESS_LAYOFF` pass) — shared
/// between the decision side (here, `boundary_push_viability`'s caller)
/// and the execution side (`players.rs`) so the two can't silently
/// disagree about how fast the ball is actually struck, the same
/// "one source of truth" discipline `slide_outcome_probs` already uses
/// for the RNG roll. Values unchanged from the pre-D6 inline literals.
pub const CONCEDE_STRIKE_SPEED_BASE: f32 = 1.0;
pub const CONCEDE_STRIKE_SPEED_RANGE: f32 = 0.6;
/// A committed slide's lunge — faster than the controlled push above.
pub const SLIDE_STRIKE_SPEED_BASE: f32 = 1.4;
pub const SLIDE_STRIKE_SPEED_RANGE: f32 = 0.8;

/// Same sigmoid steepness `on_ball_value.rs`'s `control_prob` uses for
/// its own time-advantage logistic (`CONTROL_PROB_SCALE`, private there)
/// — reused here rather than re-derived, since this is the identical
/// kind of "who wins the race" comparison, just applied to a boundary
/// point instead of a pass-reception point.
const BOUNDARY_PUSH_VIABILITY_SCALE: f32 = 15.0;

/// The engine's OWN ground-friction constant for a struck, non-shot
/// ball (`ball/ball/motion.rs::update_velocity`'s
/// `GROUND_FRICTION_COEFFICIENT` — reused by value, not re-derived; the
/// redirects this module drives all clear `cached_shot_target`, so the
/// lighter shot-specific friction never applies here). Velocity decays
/// multiplicatively by this fraction every tick once the ball is
/// rolling — this is WHY a naive `distance / launch_speed` estimate is
/// wrong for anything past a modest range, caught by raw verification
/// (Core Rule 1), not assumed on the first pass: a ball launched at
/// `speed` never actually travels at `speed` for long, so "133 ticks to
/// cover 200u at 1.5u/tick" was fiction — under pure exponential decay
/// the ball's total LIFETIME distance is bounded at `speed / k`
/// regardless of how much time passes. For the D6 strike speeds
/// (1.0-2.2u/tick) that caps out around 65-145 units — strikingly close
/// to the pre-D6 `SAFE_RESTART_LINE_MARGIN` (130u), suggesting that flat
/// distance was itself an implicit, empirically-arrived-at proxy for
/// this exact physical limit, not an arbitrary number.
const GROUND_FRICTION_COEFFICIENT: f32 = 0.015;

/// Real ticks for a ball launched at `speed` to cover `dist`, under the
/// engine's own exponential ground-friction decay — not a naive
/// constant-speed division. `None` means the ball can never physically
/// reach `dist` at all (it's beyond the friction-bounded lifetime
/// travel distance, `speed / GROUND_FRICTION_COEFFICIENT`), regardless
/// of whether any opponent is even nearby.
fn friction_aware_travel_ticks(dist: f32, speed: f32) -> Option<f32> {
    let k = GROUND_FRICTION_COEFFICIENT;
    let max_dist = speed / k;
    if dist >= max_dist {
        return None;
    }
    // position(t) = (speed/k) * (1 - (1-k)^t)  =>  solve for t at
    // position(t) = dist.
    let remaining_fraction = (1.0 - dist * k / speed).max(1e-6);
    Some(remaining_fraction.ln() / (1.0 - k).ln())
}

/// Genuine reachability, not a fixed distance: (1) can the ball
/// physically cover the ground to `target` at all, under its own real
/// friction decay — most long pushes simply can't, independent of any
/// opponent; and if it can, (2) how favourably its real friction-aware
/// travel time compares against the fastest-closing opponent's
/// `time_to_intercept` to that SAME point. ~1.0 = the ball clearly
/// wins the race (push it out for real); ~0.0 = it can't get there at
/// all, or an opponent would clearly beat it. Mirrors `on_ball_
/// value.rs::control_prob`'s exact shape (a logistic on a time-
/// advantage) for the race half, just scored against a boundary point
/// instead of a pass-reception point — same idea, not a new one.
///
/// Scans `opponents().all()`, not `.nearby(radius)` — a first cut used
/// `.nearby()`, caught wrong by the same raw-verification discipline:
/// `.nearby()` filters by distance to the DEFENDER's own position, not
/// to `target`, so an opponent genuinely standing near a far-off target
/// was routinely invisible to the scan. With only ~10 opponents on the
/// pitch, scanning all of them and letting `time_to_intercept` do the
/// distance weighting is both correct and cheap.
pub fn boundary_push_viability(
    ctx: &StateProcessingContext,
    target: Vector3<f32>,
    ball_speed: f32,
) -> f32 {
    let ball_pos = ctx.tick_context.positions.ball.position;
    let dist = (target - ball_pos).magnitude();
    let ball_ticks = match friction_aware_travel_ticks(dist, ball_speed) {
        Some(t) => t,
        None => return 0.0, // physically unreachable at this speed, full stop
    };

    let mut min_opp_ticks = f32::MAX;
    for opp in ctx.players().opponents().all() {
        let t = time_to_intercept(ctx, &opp, target);
        if t < min_opp_ticks {
            min_opp_ticks = t;
        }
    }
    if min_opp_ticks == f32::MAX {
        return 1.0; // nobody around to contest it — trivially viable
    }
    let advantage = min_opp_ticks - ball_ticks; // positive = the ball wins
    (1.0 / (1.0 + (-advantage / BOUNDARY_PUSH_VIABILITY_SCALE).exp())).clamp(0.0, 1.0)
}

/// The value of deliberately conceding a restart — high near a boundary
/// line (reuses `on_ball_value.rs`'s own `SAFE_RESTART_LINE_MARGIN` for
/// the same geometry the attacker-side `safe_restart_value` uses), when
/// the defender is genuinely goal-side and has shown the attacker toward
/// that line (conceding as the outcome of a working plan, not a panic
/// response — gated on `is_goal_side` being positive, same mechanism
/// `contain_value` uses), and when the tackle itself is low-to-mediocre
/// (if it's already clearly favourable, just tackle).
///
/// D3 fix (2026-08-21, docs/defensive-decision-logic-spec.md): the first
/// cut measured `dist_to_line` from the DEFENDER's own position — wrong.
/// "Near a boundary line" should ask whether the BALL (the thing that
/// would actually go out) is close to the edge, not the defender. A
/// defender doing exactly what the spec describes — goal-side AND
/// slightly infield, showing the attacker outside — sits systematically
/// FURTHER from the line than the ball/attacker he's containing, which
/// silenced this branch almost entirely (verified via a temporary raw
/// trace across 1363 real duel-decision points: only 9 ever had
/// defender-position dist_to_line < 40, and even those scored a near-zero
/// line_factor since the defender, by design, doesn't stand ON the
/// line). Switched to the ball's live position — the literal object
/// being conceded out of play, and what a human watching the replay
/// would judge "near the touchline" by.
///
/// D6 fix (2026-08-22): this function had two real gaps, both surfaced
/// by Pavel reading a raw zone breakdown, not from a self-reported
/// aggregate. (1) It never checked whether the attacker was actually
/// dangerous — a defender near a boundary against a harmless attacker
/// could still "concede" instead of pressing/tackling normally, which
/// is not real defensive doctrine. Fixed with `danger_factor`, reusing
/// the same `value_conceded_if_beaten` primitive `contain_value`
/// already uses. (2) `line_factor` alone (a flat distance) doesn't
/// capture whether the push can actually WIN the race to the boundary —
/// even inside that distance, an opponent standing right there still
/// reclaims it. Fixed with `boundary_push_viability` (a real race
/// check), multiplied in alongside `line_factor` rather than replacing
/// it — `line_factor` stays as the cheap pre-filter (skip the opponent
/// scan entirely when nowhere near a boundary at all), viability does
/// the real "is this actually reachable" work.
pub fn concede_out_value(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    tackle_profile: f32,
) -> f32 {
    let ball_pos = ctx.tick_context.positions.ball.position;
    let field_width = ctx.context.field_size.width as f32;
    let field_height = ctx.context.field_size.height as f32;
    let dist_to_line = ball_pos
        .x
        .min(field_width - ball_pos.x)
        .min(ball_pos.y)
        .min(field_height - ball_pos.y);
    let line_factor = (1.0 - dist_to_line / SAFE_RESTART_LINE_MARGIN).clamp(0.0, 1.0);
    if line_factor <= 0.0 {
        return 0.0;
    }

    let own_goal = ctx.ball().direction_to_own_goal();
    let margin = is_goal_side(ctx.player.position, opponent.position, own_goal);
    let goal_side_factor = (margin / CONCEDE_GOAL_SIDE_SATURATION).clamp(0.0, 1.0);
    if goal_side_factor <= 0.0 {
        return 0.0;
    }

    // D6: is this attacker actually dangerous? If not, conceding a
    // restart is a bad trade (you're giving the ball away for no real
    // reason) — a real defender presses/tackles instead. Multiplicative,
    // matching every other term here, so a low-danger situation
    // collapses this toward zero on its own rather than needing a
    // separate hard cutoff.
    let danger_factor = value_conceded_if_beaten(ctx, opponent);
    if danger_factor <= 0.0 {
        return 0.0;
    }

    // D6: can the push actually win the race to the boundary, or would
    // a nearby opponent just reclaim it? Only worth the (relatively
    // cheap) opponent scan once the two gates above already passed.
    let strength01 = (ctx.player.skills.physical.strength / 20.0).clamp(0.0, 1.0);
    let strike_speed = CONCEDE_STRIKE_SPEED_BASE + strength01 * CONCEDE_STRIKE_SPEED_RANGE;
    let target = nearest_boundary_edge(ball_pos, field_width, field_height);
    let viability = boundary_push_viability(ctx, target, strike_speed);
    if viability <= 0.0 {
        return 0.0;
    }

    let p = tackle_win_probability(ctx, opponent, tackle_profile);
    let non_favorable_factor = (1.0 - (p / TACKLE_WIN_PROB_CEILING)).clamp(0.0, 1.0);
    // Floored 0.5-1.0, same shape as contain_value's own
    // "(0.5 + 0.5*non_favorable_factor)" term — the D3 recalibration
    // above: letting this go all the way to 0 as the tackle become
    // merely MODERATELY favourable (not clearly favourable) crushed
    // concede far harder than contain's equivalent term did for the
    // same duel, which was half of the asymmetry that kept concede from
    // ever winning. A tackle only needs to be genuinely clearly
    // favourable (p near TACKLE_WIN_PROB_CEILING) to meaningfully
    // discourage conceding.
    let non_favorable_floored = 0.5 + 0.5 * non_favorable_factor;

    CONCEDE_BASE_VALUE
        * line_factor
        * goal_side_factor
        * non_favorable_floored
        * danger_factor
        * viability
}

// ── D4.6 (docs/defensive-decision-logic-spec.md): last-ditch sliding
// block ──────────────────────────────────────────────────────────────
//
// Architecturally distinct from `tackle_value`/`contain_value`/
// `concede_out_value` above: those three assume the defender has
// ALREADY arrived (goal-side, in control, calmly weighing a jockey vs.
// a committed challenge). This covers the opposite moment — still
// chasing, trailing or level, deciding whether a slide's extra reach
// is worth the foul/miss/own-risk it carries against just letting the
// gap stay open. Real doctrine (a slide covers ground a stride can't),
// corroborated by physics (extended reach at the cost of being grounded
// afterward) and the Laws of the Game (a mistimed slide inside the box
// is a real penalty risk, not a free action).

/// Reach window beyond a normal standing engagement
/// (`TACKLE_DISTANCE_THRESHOLD`, 10u, tackling/mod.rs) that a committed
/// slide can plausibly cover before the attacker's already gone. Real
/// doctrine, not a sourced figure — no dataset isolates slide-tackle
/// reach specifically (Realism Diagnosis Framework category (d)),
/// reasoned as roughly a further ~3m of committed lunge.
pub const SLIDE_REACH_DISTANCE: f32 = 34.0;
/// Inside this distance it's a normal tackle-range engagement — D1-D4's
/// own territory (`DefenderTacklingState`'s tackle/contain/concede-out
/// comparison), not a desperate slide from further out.
const SLIDE_MIN_DISTANCE: f32 = 10.0;
/// `value_conceded_if_beaten` must clear this before a slide is even a
/// candidate — a low-stakes trailing moment doesn't need a lunge, a
/// dangerous one does.
const SLIDE_DANGER_THRESHOLD: f32 = 0.25;

/// Whether a desperate slide is a live candidate this tick: the
/// defender is trailing/level (`is_goal_side` <= 0 — the complement of
/// `contain_value`/`concede_out_value`'s own precondition, which both
/// require a POSITIVE margin), within slide reach but beyond a normal
/// standing engagement, and the moment is genuinely dangerous if the
/// attacker isn't stopped now.
pub fn desperate_slide_candidate(ctx: &StateProcessingContext, opponent: &MatchPlayerLite) -> bool {
    let own_goal = ctx.ball().direction_to_own_goal();
    let margin = is_goal_side(ctx.player.position, opponent.position, own_goal);
    if margin > 0.0 {
        return false;
    }
    let dist = opponent.distance(ctx);
    if !(SLIDE_MIN_DISTANCE..=SLIDE_REACH_DISTANCE).contains(&dist) {
        return false;
    }
    value_conceded_if_beaten(ctx, opponent) > SLIDE_DANGER_THRESHOLD
}

/// Pure connect-quality estimate in [0,1] — how likely the slide is to
/// make meaningful contact at all, driven by how much of the reach
/// window is already closed (nearer the normal engagement range is an
/// easier connect) and tackle skill. No RNG — mirrors the existing
/// split between `tackle_win_probability` (pure) and
/// `attempt_sliding_tackle` (the RNG-rolling caller); the real outcome
/// roll happens at the execution site (`pressing/mod.rs`), using
/// `slide_outcome_probs` below so the decision-time value and the
/// execution-time roll can't drift apart.
pub fn slide_connect_quality(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    tackle_profile: f32,
) -> f32 {
    let dist = opponent.distance(ctx);
    let reach_ratio =
        (1.0 - (dist - SLIDE_MIN_DISTANCE) / (SLIDE_REACH_DISTANCE - SLIDE_MIN_DISTANCE))
            .clamp(0.0, 1.0);
    (0.30 + reach_ratio * 0.45 + tackle_profile * 0.25).clamp(0.05, 0.90)
}

/// The five outcomes the spec calls for, as a weighted probability
/// table (mirrors `try_block_shot`'s own cumulative-threshold idiom,
/// `ball/ball/interactions.rs`) — remainder after summing the four
/// named fields is the fifth, Miss.
pub struct SlideOutcomeProbs {
    pub p_success: f32,
    pub p_partial: f32,
    pub p_foul: f32,
    pub p_own_risk: f32,
}

impl SlideOutcomeProbs {
    pub fn p_miss(&self) -> f32 {
        (1.0 - self.p_success - self.p_partial - self.p_foul - self.p_own_risk).max(0.0)
    }
}

/// Shared by both the decision-time EV (`desperate_slide_value`) and
/// the execution-time RNG roll (`pressing/mod.rs::attempt_desperate_
/// slide`) — one source of truth for the outcome split so the two
/// can't silently disagree.
pub fn slide_outcome_probs(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    tackle_profile: f32,
) -> SlideOutcomeProbs {
    let connect = slide_connect_quality(ctx, opponent, tackle_profile);
    let in_own_box = ctx
        .context
        .penalty_area(ctx.player.side == Some(PlayerSide::Left))
        .contains(&ctx.tick_context.positions.ball.position);
    let p_success = connect * 0.50;
    let p_partial = connect * 0.20;
    // A mistimed (low-connect) slide risks a foul more than a
    // well-timed one; a genuine connect can still catch the man as well
    // as the ball (the tackle-state equivalent: "won-ball fouls exist").
    // Own-box multiplier smaller than a standing tackle's own-box
    // restraint (`TACKLE_FOUL_RISK_COST_OWN_BOX_MULT`'s cost side) since
    // this is a probability, not a cost — a committed slide is already
    // less likely to happen calmly inside the box at all (danger and
    // urgency crowd it out), so the extra multiplier here is modest.
    let p_foul = (((1.0 - connect) * 0.28 + connect * 0.08) * if in_own_box { 1.2 } else { 1.0 })
        .clamp(0.0, 0.5);
    // Rarer, and more likely from a mistimed (low-connect) lunge — a
    // clean, well-timed slide rarely turns into a genuine own-goal risk.
    let p_own_risk = ((1.0 - connect) * 0.10).clamp(0.0, 0.25);
    SlideOutcomeProbs {
        p_success,
        p_partial,
        p_foul,
        p_own_risk,
    }
}

const SLIDE_SUCCESS_VALUE: f32 = 0.55; // slightly above TACKLE_WIN_VALUE (0.5) — denying the attacker outright is at least as valuable here, since the alternative is estimated to lose the race
const SLIDE_FOUL_COST_BASE: f32 = 0.20;
const SLIDE_FOUL_COST_OWN_BOX_MULT: f32 = 2.2; // a mistimed slide in the box is real penalty risk — slightly above the standing tackle's own equivalent (TACKLE_FOUL_RISK_COST_OWN_BOX_MULT, 2.0) since a slide commits with more force
const SLIDE_OWN_RISK_COST: f32 = 0.25;

/// EV of committing to the slide right now, compared against the
/// IMPLICIT value of just letting the ordinary chase continue — 0, the
/// baseline this is measured against (per the spec: NOT tackle_value/
/// contain_value, which both assume the defender is already
/// positioned). Miss and Partial contribute nothing beyond that
/// baseline, matching the spec's own framing ("the same outcome as not
/// sliding at all" / "genuinely contestable"); only Success, Foul, and
/// OwnRisk move the value away from it.
pub fn desperate_slide_value(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    tackle_profile: f32,
) -> f32 {
    if !desperate_slide_candidate(ctx, opponent) {
        return 0.0;
    }
    let probs = slide_outcome_probs(ctx, opponent, tackle_profile);
    let danger = value_conceded_if_beaten(ctx, opponent);
    let in_own_box = ctx
        .context
        .penalty_area(ctx.player.side == Some(PlayerSide::Left))
        .contains(&ctx.tick_context.positions.ball.position);
    let foul_cost = if in_own_box {
        SLIDE_FOUL_COST_BASE * SLIDE_FOUL_COST_OWN_BOX_MULT
    } else {
        SLIDE_FOUL_COST_BASE
    };
    probs.p_success * SLIDE_SUCCESS_VALUE
        - probs.p_foul * foul_cost
        - probs.p_own_risk * SLIDE_OWN_RISK_COST * danger
}

/// D6 (2026-08-22): whether a SUCCESSFUL slide should push the ball
/// toward a boundary, or lay it off to the nearest teammate instead.
/// `desperate_slide_candidate`'s own danger gate already decided the
/// slide itself was worth attempting — this is a SEPARATE question,
/// asked only once the slide has already connected: "tackle to get the
/// ball" and "tackle to push the ball away" are two different
/// decisions, not one. Before this fix, a successful slide unconditionally
/// aimed at `nearest_boundary_edge` regardless of how far away that edge
/// actually was — raw data showed 82/84 slides from 17-33% of the pitch
/// length out (dead central, well outside the box) got reclaimed before
/// the ball ever arrived, because the target was never reachable in the
/// first place. Reuses the same `boundary_push_viability` race check
/// `concede_out_value` now uses, at the slide's own (faster, committed)
/// strike speed — not a re-guessed threshold.
pub fn slide_should_push_to_boundary(ctx: &StateProcessingContext) -> bool {
    let ball_pos = ctx.tick_context.positions.ball.position;
    let field_width = ctx.context.field_size.width as f32;
    let field_height = ctx.context.field_size.height as f32;
    let target = nearest_boundary_edge(ball_pos, field_width, field_height);
    let strength01 = (ctx.player.skills.physical.strength / 20.0).clamp(0.0, 1.0);
    let strike_speed = SLIDE_STRIKE_SPEED_BASE + strength01 * SLIDE_STRIKE_SPEED_RANGE;
    boundary_push_viability(ctx, target, strike_speed) > SLIDE_BOUNDARY_VIABILITY_THRESHOLD
}

/// Require the ball to clearly win the race (not just a coin flip) —
/// matches `control_prob`'s own framing of ~0.5 as "contested," and a
/// genuine push-it-out decision should be more confident than a mere
/// contested 50/50 before committing to it over the safer lay-off.
const SLIDE_BOUNDARY_VIABILITY_THRESHOLD: f32 = 0.6;

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
