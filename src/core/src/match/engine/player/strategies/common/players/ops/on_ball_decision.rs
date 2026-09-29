//! On-ball attacking decisions — designed from null.
//!
//! `docs/decision-from-null.md` (domain 2) /
//! `docs/on-ball-attacking-decisions-from-null.md` (this domain's own
//! football reasoning, spec, and verification log — read that first).
//!
//! A single, per-player decision (shoot / pass-to-T / carry / dribble /
//! clear / hold), re-evaluated every tick for whichever outfield player
//! currently, genuinely holds the ball. Deliberately calls no
//! pre-existing decision function from this engine (no
//! `common::passing::evaluator::PassEvaluator`,
//! `common::players::ops::on_ball_value::{shoot_value, carry_value,
//! pass_value_from, ...}`, and none of the old per-role
//! `Running`/`Dribbling`/`CreatingSpace`/`Passing`/`Clearing` state
//! bodies) — every score, threshold, and target formula here is derived
//! fresh from the football reasoning in the design doc, using only raw
//! `StateProcessingContext` data plus genuine measurements
//! (`max_speed_with_condition`, goal positions, `opponent_goal_position`,
//! `pass_teammate_power`/`shoot_goal_power` — real skill-based physics,
//! not judgments — and plain Euclidean geometry).
//!
//! Process check (`docs/decision-from-null.md`'s required self-audit):
//! this module's call graph touches only `StateProcessingContext`
//! field/method accessors that expose raw position/velocity/skill data
//! (`ctx.player`, `ctx.context.goal_positions`, `ctx.context.players.by_id`,
//! `ctx.players().teammates()/.opponents().nearby(...)`,
//! `PlayerSkills::max_speed_with_condition`,
//! `PlayerOperationsImpl::{opponent_goal_position, has_ball,
//! pass_teammate_power, shoot_goal_power}`), `nalgebra::Vector3`
//! geometry, and the real-ballistics physics conversion
//! (`engine::physics::z_launch_velocity_for_height`) — every one of
//! these is a measurement (a fact given the same inputs, two
//! different-minded coaches would compute it identically). Zero
//! pre-existing decision-function dependencies.

use crate::PlayerFieldPositionGroup;
use crate::r#match::player::PlayerSide;
use crate::r#match::{MatchPlayerLite, StateProcessingContext};
use nalgebra::Vector3;

// ---- Reasoned constants (measurement-scale references, not judgments).
// Every one is category (d) per this project's own sourcing hierarchy —
// no dataset publishes "how far a real player scans for a pass option"
// or similar — flagged as such, and a target for the raw-trace
// verification pass, not treated as final. Kept in one place so a future
// recalibration doesn't have to hunt through the scoring functions.

/// Beyond this distance from the attacking goal, a shot is never worth
/// evaluating at all (a cheap pre-filter — the overwhelming majority of
/// touches are nowhere near this range, and paying the cost of a full
/// clarity scan for a shot that will never plausibly happen is wasted
/// work). ~27.5m.
const SHOOT_EVAL_RADIUS: f32 = 220.0;
/// How close an outfield opponent must sit to the direct line from
/// shooter to goal-centre to meaningfully obstruct the sight of goal —
/// a real jockeying/blocking distance. ~1.75m.
const SHOOT_BLOCK_CORRIDOR: f32 = 14.0;
/// Beyond this distance, a pass isn't a realistic option this tick. ~40m.
const PASS_MAX_RANGE: f32 = 320.0;
/// Nearest-opponent distance at which a receiver, or a carry/dribble
/// destination, reads as "fully open" (openness saturates at 1.0 beyond
/// this). ~7.5m — enough room to receive and turn unpressured.
const OPEN_REFERENCE_DIST: f32 = 60.0;
/// How close to the direct passing line an opponent must sit to be
/// treated as a real interception threat at all.  ~3m.
const INTERCEPT_CORRIDOR: f32 = 24.0;
/// Real human reaction-time floor, in ticks at this engine's 100-tick/s
/// simulation rate (~150ms) — this project's own established
/// physiological anchor, re-derived locally rather than importing the
/// old decision layer's copy.
const REACTION_TICKS: f32 = 15.0;
/// A reasoned average pass speed for the decision-time intercept race —
/// the ball's real execution speed varies with the passer's own skill
/// and the chosen force (kept execution physics); this is only used to
/// judge WHETHER a pass is interceptable, not to move the ball.
const NOMINAL_PASS_SPEED: f32 = 2.2;
/// How far ahead a carry/dribble open-space scan looks before giving up
/// on finding more room. ~17.5m.
const CARRY_LOOKAHEAD: f32 = 140.0;
/// Domain-2 carry's own longer scan (~30m) for the fair-carry scoring.
const CARRY_RUN_LOOKAHEAD: f32 = 240.0;
/// Same 0.20 retention floor a pass gets (`PASS_RETENTION_BASE`).
const CARRY_RETENTION_BASE: f32 = 0.20;

/// Same-binary A/B toggle (mirrors `OF_SCORE_BLIND`/`MOBILITY_ABLATE`):
/// set `CARRY_FAIR_ABLATE` (any value) to restore the old carry scoring.
fn carry_fair_scoring() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CARRY_FAIR_ABLATE").is_none())
}
/// How close to a candidate running lane an opponent must sit to count
/// as contesting it — a real defender's own reach/step-across distance.
const LANE_CONTEST_RADIUS: f32 = 20.0;
/// Number of candidate headings fanned out around the attacking
/// direction when scanning for the best carry lane.
const CARRY_HEADING_COUNT: usize = 7;
/// Half-width of the heading fan scanned for a carry, in degrees either
/// side of straight toward the attacking goal — wide enough to find a
/// genuinely open wide lane, not so wide it considers running backward.
const CARRY_FAN_HALF_ANGLE_DEG: f32 = 70.0;
/// How close a specific opponent must sit, directly ahead, to count as
/// "blocking" the immediate route (the trigger for evaluating a dribble
/// at all, distinct from an open carry). ~4.5m.
const DRIBBLE_BLOCK_RADIUS: f32 = 36.0;
/// Lateral tolerance for "directly ahead" when scanning for a blocker.
const DRIBBLE_BLOCK_CORRIDOR: f32 = 16.0;
/// Full pitch length — the reference scale for how a turnover's danger
/// grows the closer it happens to one's own goal.
const TURNOVER_REFERENCE_DIST: f32 = 840.0;
/// Genuine own-third crisis zone — inside this radius of our own goal, a
/// clearance becomes a live option at all. ~22m, independently reasoned
/// at the same magnitude domain 1's own crisis radius uses (the same
/// real football concept, not a call into that file's constant).
const CRISIS_OWN_GOAL_RADIUS: f32 = 176.0;
/// How close an opponent must be, even inside the crisis zone, for
/// clearing to actually be the right call (real, immediate pressure —
/// not just "the ball happens to be deep"). ~2.5m.
const CRISIS_PRESSURE_RADIUS: f32 = 20.0;

// Value-function weights. Deliberately built on a shared, comparable
// scale across every action (see the design doc's own "scale
// consistency treated as a first-class property" note) — each action's
// maximum plausible score sits in a similar rough band, with Shoot able
// to clearly dominate a genuinely clear chance, Carry preferred over
// Dribble when both are live (open ground beats a contested duel), and
// Hold a low floor beneath any real alternative.
//
// `CARRY_SPACE_WEIGHT`/`DRIBBLE_BASE_WEIGHT` corrected 0.65/0.55 →
// 0.45/0.38 (2026-09-23), found by raw-tracing the actual decisions
// this domain made against real football doctrine (verification plan
// step 3 — "does risk scale with field position"), not tuned against
// goals/match or any other informational metric. Baseline (defenders'
// action mix, bucketed by distance from own goal): the innermost
// crisis zone [0,150u) already read correctly (85.7% Pass, 14.3%
// Clear, 0% Carry — matches real safety-first defending near one's own
// box). But the middle zones [150,450u), where the crisis gate doesn't
// apply, read 80.8%/98.4% Carry — real centre-backs mostly progress
// the ball by passing, carrying forward only some of the time even
// with open space ahead; this was Carry dominating almost completely,
// not "sometimes." A real, plausible confound exists too (off-ball
// attacking positioning/support is a separate, deferred domain, so
// teammates' default positions may not offer a genuinely progressive
// passing lane, inflating Carry's relative advantage) — not fully
// disentangled from a real weight miscalibration; this correction
// narrows the gap without claiming to have isolated the cause, and is
// exactly the kind of property that needs watching on the pitch, not
// further blind constant-tuning, to settle definitively.
const SHOOT_BASE_WEIGHT: f32 = 0.90;
const PASS_RETENTION_BASE: f32 = 0.20;
const PASS_PROGRESS_WEIGHT: f32 = 0.55;
const PASS_OPENNESS_WEIGHT: f32 = 0.30;
const PASS_RISK_WEIGHT: f32 = 0.75;
const CARRY_SPACE_WEIGHT: f32 = 0.45;
const CARRY_RISK_MULT: f32 = 0.35;
const DRIBBLE_BASE_WEIGHT: f32 = 0.38;
const DRIBBLE_RISK_MULT: f32 = 0.45;
const CLEAR_VALUE: f32 = 0.55;
const HOLD_VALUE: f32 = 0.15;
/// Real football: don't just give the ball straight back to whoever
/// passed it to you, unless doing so represents genuine forward
/// progress (a real give-and-go/wall-pass, where the passer has moved
/// into meaningfully better space). Found via raw match-trace
/// verification (2026-09-23), not assumed: without this, two forwards
/// positioned near each other fell into a repeated, static mutual pass
/// loop — 66-74% of their own pass involvement was with each other in
/// traced matches, a real and severe deviation from real circulation,
/// not a minor imbalance.
const PASS_BACK_TO_PASSER_PENALTY: f32 = 0.25;
/// Below this forward-progress threshold, a pass back to the immediate
/// previous passer reads as "just give it back," not a genuine
/// advancing combination — the penalty only applies below this bar.
const PASS_BACK_PROGRESS_EXEMPT: f32 = 0.15;

/// The decision this domain produces for the current ball carrier. A
/// discrete action (Shoot/Pass/Clear) is dispatched as a real event the
/// same tick it's chosen; Carry/Dribble are steering targets consumed by
/// a velocity override and re-evaluated fresh the following tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Shoot,
    Pass(u32),
    Carry(Vector3<f32>),
    Dribble(Vector3<f32>),
    Clear,
    Hold,
    /// Domain 4 (`box_attack.rs`): a pass to a runner aimed at his
    /// assigned box zone (a cross or cut-back), not at where he stands.
    Deliver { to: u32, target: Vector3<f32> },
    /// Off-ball attacking runs, from null (`off_ball_runs.rs`): a pass
    /// aimed at a teammate's frozen, committed run target (`.1`), not at
    /// his current position — a real through ball, played into the
    /// space he is running into rather than at his feet. The conflict
    /// check's own item 5: "a run pays off only if he can play it."
    Through { to: u32, target: Vector3<f32> },
}

/// Step A's gate: true when a manager-issued directive already governs
/// this player's on-ball behaviour, or he's inside a kept, Law-driven
/// restart-procedure window. Both cases mean this domain's default,
/// undirected decision-making must back off entirely — see the design
/// doc's "manager instructions always win" section.
fn is_directed_or_restart_pending(ctx: &StateProcessingContext) -> bool {
    let player = ctx.player;
    if player.behavioral_directive.is_some()
        || player.link_target.is_some()
        || player.supply_target.is_some()
        || player.kickoff_pass_pending > 0
        || player.throw_in_pass_pending > 0
        || player.free_kick_pass_pending > 0
    {
        return true;
    }
    // Broad, robust net for every other restart type (corner, goal kick,
    // penalty, generic/direct/indirect free kick) — `PassOriginRestart`
    // doesn't cover kickoff, which is why the three specific `_pending`
    // fields above are also checked directly. Set-piece PLAYER decisions
    // (taker selection, shoot-vs-cross-vs-short) are confirmed in-scope
    // for the eventual rewrite but explicitly deferred — this domain's
    // decision-making must not compete with the kept staging/delivery
    // machinery for a player who's genuinely mid-restart.
    use crate::r#match::PassOriginRestart;
    ctx.tick_context.ball.pass_origin_restart != PassOriginRestart::OpenPlay
}

/// `pub(crate)` (2026-09-24) so `processor.rs`'s `on_ball_ai_velocity`
/// can reuse this exact measurement for the `Action::Hold` shielding fix
/// (see that call site's own doc comment) rather than re-deriving the
/// identical side-to-goal lookup a second time.
pub(crate) fn own_goal_position(ctx: &StateProcessingContext) -> Vector3<f32> {
    match ctx.player.side {
        Some(PlayerSide::Left) => ctx.context.goal_positions.left,
        Some(PlayerSide::Right) => ctx.context.goal_positions.right,
        None => Vector3::zeros(),
    }
}

/// A player's real, condition-adjusted top speed, looked up by id via
/// the match-start skill snapshot — the same measurement pattern used
/// throughout this engine's kept physics/geometry layer (position drifts
/// tick to tick; skills and condition don't, or drift slowly enough that
/// this snapshot is an honest approximation, matching existing usage
/// elsewhere in the codebase for the identical lookup).
pub(crate) fn real_max_speed(ctx: &StateProcessingContext, id: u32) -> f32 {
    ctx.context
        .players
        .by_id(id)
        .map(|p| p.skills.max_speed_with_condition(p.player_attributes.condition))
        .unwrap_or(0.45)
}

pub(crate) fn rotate_xy(v: Vector3<f32>, angle_rad: f32) -> Vector3<f32> {
    let (s, c) = angle_rad.sin_cos();
    Vector3::new(v.x * c - v.y * s, v.x * s + v.y * c, 0.0)
}

/// Worst-case interception risk along the straight segment from `from`
/// to `to`, against every opponent within `INTERCEPT_CORRIDOR` of the
/// segment. A genuine race-in-time fact: does the opponent's own real
/// max speed (plus a real reaction-time floor) get him to the
/// nearest-approach point before the ball — travelling at
/// `NOMINAL_PASS_SPEED` — would arrive there. 0.0 = no real threat,
/// 1.0 = an opponent comfortably wins the race. Takes the single worst
/// threat, not a sum — one well-placed defender kills a passing lane; a
/// handful of distant ones don't compound linearly the way a naive sum
/// would suggest.
pub(crate) fn intercept_risk(
    ctx: &StateProcessingContext,
    from: Vector3<f32>,
    to: Vector3<f32>,
    opponents: &[MatchPlayerLite],
) -> f32 {
    let seg = to - from;
    let seg_len = seg.magnitude();
    if seg_len < 1.0 {
        return 0.0;
    }
    let seg_dir = seg / seg_len;
    let mut worst: f32 = 0.0;
    for opp in opponents {
        let to_opp = opp.position - from;
        let along = to_opp.dot(&seg_dir).clamp(0.0, seg_len);
        let closest_point = from + seg_dir * along;
        let perp_dist = (opp.position - closest_point).magnitude();
        if perp_dist > INTERCEPT_CORRIDOR {
            continue;
        }
        let ball_time = along / NOMINAL_PASS_SPEED;
        let opp_dist = (closest_point - opp.position).magnitude();
        let opp_speed = real_max_speed(ctx, opp.id);
        let opp_time = REACTION_TICKS + opp_dist / opp_speed.max(0.05);
        let margin = opp_time - ball_time;
        let risk = if margin <= 0.0 {
            1.0
        } else {
            // Risk fades out over a real ~30-tick (~0.3s) margin window —
            // a comfortable race isn't a threat; a close one still is.
            (1.0 - margin / 30.0).clamp(0.0, 1.0)
        };
        worst = worst.max(risk);
    }
    worst
}

/// How dangerous a turnover would be if the ball were lost near `point`
/// — real football's own turnover-risk doctrine: a loss near one's own
/// goal can concede immediately, the identical loss in the opponent's
/// final third is comparatively cheap. 0.05 (safe floor — even a
/// forward-third turnover carries some cost) to 1.0 (catastrophic).
fn turnover_cost(own_goal: Vector3<f32>, point: Vector3<f32>) -> f32 {
    let dist = (point - own_goal).magnitude();
    (1.0 - dist / TURNOVER_REFERENCE_DIST).clamp(0.05, 1.0)
}

pub(crate) fn nearest_opponent_dist(point: Vector3<f32>, opponents: &[MatchPlayerLite]) -> f32 {
    opponents
        .iter()
        .map(|o| (o.position - point).magnitude())
        .fold(f32::MAX, f32::min)
}

pub(crate) fn openness(point: Vector3<f32>, opponents: &[MatchPlayerLite]) -> f32 {
    if opponents.is_empty() {
        return 1.0;
    }
    (nearest_opponent_dist(point, opponents) / OPEN_REFERENCE_DIST).clamp(0.0, 1.0)
}

/// Fraction of the remaining distance to the attacking goal covered by
/// moving from `from` to `to` — positive for genuine forward progress,
/// negative for backward. Real, simple geometry.
fn progress(from: Vector3<f32>, to: Vector3<f32>, attacking_goal: Vector3<f32>) -> f32 {
    let d0 = (from - attacking_goal).magnitude();
    let d1 = (to - attacking_goal).magnitude();
    if d0 < 1.0 {
        return 0.0;
    }
    ((d0 - d1) / d0).clamp(-1.0, 1.0)
}

/// Real sight of goal: 1.0 if no outfield opponent meaningfully
/// obstructs the direct line from `from` to `goal_centre`, degrading
/// toward 0.0 the closer an obstructing opponent sits to that line.
/// Deliberately excludes the goalkeeper — he's the normal, expected
/// obstacle a shot is aimed past (handled by the kept execution/save
/// mechanic), not a reason to avoid shooting at all.
fn shot_clarity(from: Vector3<f32>, goal_centre: Vector3<f32>, opponents: &[MatchPlayerLite]) -> f32 {
    let seg = goal_centre - from;
    let seg_len = seg.magnitude();
    if seg_len < 1.0 {
        return 1.0;
    }
    let seg_dir = seg / seg_len;
    let mut worst_clarity = 1.0f32;
    for opp in opponents {
        if opp.tactical_positions.position_group() == PlayerFieldPositionGroup::Goalkeeper {
            continue;
        }
        let to_opp = opp.position - from;
        let along = to_opp.dot(&seg_dir);
        if along < 0.0 || along > seg_len {
            continue;
        }
        let closest_point = from + seg_dir * along;
        let perp_dist = (opp.position - closest_point).magnitude();
        if perp_dist < SHOOT_BLOCK_CORRIDOR {
            worst_clarity = worst_clarity.min((perp_dist / SHOOT_BLOCK_CORRIDOR).clamp(0.0, 1.0));
        }
    }
    worst_clarity
}

fn shoot_value(from: Vector3<f32>, goal_centre: Vector3<f32>, opponents: &[MatchPlayerLite]) -> f32 {
    let dist = (goal_centre - from).magnitude();
    let clarity = shot_clarity(from, goal_centre, opponents);
    let distance_factor = (1.0 - dist / SHOOT_EVAL_RADIUS).clamp(0.0, 1.0);
    SHOOT_BASE_WEIGHT * clarity * distance_factor
}

/// Real football's phase-of-possession doctrine, corrected 2026-09-23
/// after Pavel's own follow-up clarification: "I do not say that
/// midfielders should carry, or that forward should carry in a certain
/// zone... it is about who is doing what as a group of players." The
/// prior `carry_priority_mult` (still kept, still real — see its own
/// doc comment) discourages an isolated player from personally
/// dribbling outside his line's zone, but the actual mechanism Pavel
/// described is about PASSING: the ball should flow through and between
/// the members of whichever line is currently responsible for
/// progressing it (defenders circulating in the own third, midfielders
/// combining with each other through the middle third), only reaching
/// the next line once that line has genuinely done its job — not that
/// any individual player needs to carry it there himself. This is a
/// pass-TARGET preference, not an action preference.
fn responsible_line_for_zone(own_goal_dist: f32) -> PlayerFieldPositionGroup {
    if own_goal_dist < OWN_THIRD_BOUNDARY {
        PlayerFieldPositionGroup::Defender
    } else if own_goal_dist < FINAL_THIRD_BOUNDARY {
        PlayerFieldPositionGroup::Midfielder
    } else {
        PlayerFieldPositionGroup::Forward
    }
}

/// Modest, reasoned bonus (category d, no sourced dataset) for passing
/// to a teammate in the line currently responsible for this zone —
/// small enough that a genuinely excellent, safe, progressive pass to a
/// different line still wins on its own merits (this is a preference,
/// not a veto), large enough to matter when several options are
/// otherwise close.
const LINE_PRIORITY_BONUS: f32 = 0.18;

fn pass_value(
    ctx: &StateProcessingContext,
    from: Vector3<f32>,
    teammate: &MatchPlayerLite,
    own_goal: Vector3<f32>,
    attacking_goal: Vector3<f32>,
    opponents: &[MatchPlayerLite],
) -> f32 {
    let to = teammate.position;
    let prog = progress(from, to, attacking_goal);
    let open = openness(to, opponents);
    let risk = intercept_risk(ctx, from, to, opponents);
    let cost = turnover_cost(own_goal, to);
    let mut value = PASS_RETENTION_BASE + PASS_PROGRESS_WEIGHT * prog + PASS_OPENNESS_WEIGHT * open
        - PASS_RISK_WEIGHT * risk * cost;
    let responsible = responsible_line_for_zone((from - own_goal).magnitude());
    if teammate.tactical_positions.position_group() == responsible {
        value += LINE_PRIORITY_BONUS;
    }
    if ctx.tick_context.ball.last_owner == Some(teammate.id) && prog < PASS_BACK_PROGRESS_EXEMPT {
        value -= PASS_BACK_TO_PASSER_PENALTY;
    }
    value
}

/// How far the player could advance in `heading` before the nearest
/// opponent's own realistic reach into that lane would contest it,
/// capped at `CARRY_LOOKAHEAD`.
pub(crate) fn open_distance_in_heading(
    from: Vector3<f32>,
    heading: Vector3<f32>,
    opponents: &[MatchPlayerLite],
) -> f32 {
    open_distance_capped(from, heading, opponents, CARRY_LOOKAHEAD)
}

/// Same scan with an explicit cap. Domain 4 (`box_attack.rs`) shares
/// `open_distance_in_heading` at the old 140u cap; domain 2's carry uses
/// the longer `CARRY_RUN_LOOKAHEAD` so a long open run is worth more than
/// a short one (2026-09-29 fair-carry change).
fn open_distance_capped(
    from: Vector3<f32>,
    heading: Vector3<f32>,
    opponents: &[MatchPlayerLite],
    cap: f32,
) -> f32 {
    let mut best = cap;
    for opp in opponents {
        let to_opp = opp.position - from;
        let along = to_opp.dot(&heading);
        if along < 0.0 || along > cap {
            continue;
        }
        let closest_point = from + heading * along;
        let perp_dist = (opp.position - closest_point).magnitude();
        if perp_dist < LANE_CONTEST_RADIUS {
            best = best.min(along.max(0.0));
        }
    }
    best
}

fn carry_best_heading(
    from: Vector3<f32>,
    forward: Vector3<f32>,
    opponents: &[MatchPlayerLite],
) -> (Vector3<f32>, f32) {
    let half = CARRY_FAN_HALF_ANGLE_DEG.to_radians();
    let mut best_heading = forward;
    let mut best_open = 0.0f32;
    for i in 0..CARRY_HEADING_COUNT {
        let t = if CARRY_HEADING_COUNT > 1 {
            i as f32 / (CARRY_HEADING_COUNT - 1) as f32
        } else {
            0.5
        };
        let angle = -half + 2.0 * half * t;
        let heading = rotate_xy(forward, angle);
        let open = open_distance_in_heading(from, heading, opponents);
        if open > best_open {
            best_open = open;
            best_heading = heading;
        }
    }
    (best_heading, best_open)
}

/// Real football's phase-of-possession doctrine, added 2026-09-23 after
/// Pavel watched an isolated forward try to dribble through a packed
/// defense 30m out instead of recycling to midfield: WHO is primarily
/// responsible for carrying the ball forward changes by zone. Defenders
/// build out through their own third; midfielders progress the ball
/// through the middle third; forwards (and advanced wide players) carry
/// and take players on in the final third, where an individual action
/// actually threatens goal. Outside a role's own zone, carrying/
/// dribbling is discounted, not forbidden (a genuinely wide-open lane
/// still carries some value) -- suppressed enough that a real
/// pass-back/recycle option (already retained by `pass_value`'s own
/// `PASS_RETENTION_BASE`) wins the comparison instead, exactly the
/// "turn back and play with midfield" pattern Pavel described. Zone
/// boundaries and multipliers are reasoned estimates (category d, real
/// thirds-of-the-pitch coaching doctrine, no sourced numeric dataset) --
/// a target for the raw-trace verification pass, not treated as final.
///
/// `pub(crate)` (2026-09-24) so `attacking_support.rs` (domain 3, off-ball
/// attacking positioning) can key its own possession-phase logic
/// (build-up/progression/final-third) off the SAME real pitch-thirds
/// boundary as this module's own on-ball carrying priority, rather than
/// independently re-deriving what is physically the identical fact about
/// the pitch -- exactly the kind of constant drift between two places
/// that mean the same thing this project's own history has been bitten
/// by before. The value/meaning here is unchanged; only visibility moved.
pub(crate) const OWN_THIRD_BOUNDARY: f32 = 280.0; // ~35m
pub(crate) const FINAL_THIRD_BOUNDARY: f32 = 560.0; // ~70m

fn carry_priority_mult(group: PlayerFieldPositionGroup, own_goal_dist: f32) -> f32 {
    let zone = if own_goal_dist < OWN_THIRD_BOUNDARY {
        0
    } else if own_goal_dist < FINAL_THIRD_BOUNDARY {
        1
    } else {
        2
    };
    match (group, zone) {
        (PlayerFieldPositionGroup::Defender, 0) => 1.0,
        (PlayerFieldPositionGroup::Defender, 1) => 0.55,
        (PlayerFieldPositionGroup::Defender, _) => 0.35,
        (PlayerFieldPositionGroup::Midfielder, 0) => 0.75,
        (PlayerFieldPositionGroup::Midfielder, 1) => 1.0,
        (PlayerFieldPositionGroup::Midfielder, _) => 0.85,
        (PlayerFieldPositionGroup::Forward, 0) => 0.35,
        (PlayerFieldPositionGroup::Forward, 1) => 0.45,
        (PlayerFieldPositionGroup::Forward, _) => 1.0,
        _ => 1.0,
    }
}

/// How many opponents genuinely stand between `from` and the opponent's
/// goal, in the direction of the run -- a real measurement of "is this
/// a clear break or a packed wall," not an assumption from zone alone.
/// Pavel's own follow-up clarification (2026-09-23): "a forward on a
/// counterattack may receive the ball in the central circle and make a
/// run towards the goal if there is no (or just one) defender ahead of
/// him. But if there is a clear wall of opponents, and he is far from
/// the goal, there is no reason to be a hero — just play with your
/// teammates." A flat zone discount alone can't tell these two
/// situations apart; this measurement can.
fn defenders_ahead_count(
    from: Vector3<f32>,
    forward: Vector3<f32>,
    opponent_goal: Vector3<f32>,
    opponents: &[MatchPlayerLite],
) -> usize {
    let to_goal_dist = (opponent_goal - from).magnitude();
    opponents
        .iter()
        .filter(|o| {
            let to_opp = o.position - from;
            let along = to_opp.dot(&forward);
            along > 0.0 && along < to_goal_dist
        })
        .count()
}

/// A genuine break (0-1 defenders between the carrier and goal) bypasses
/// the zone-based discount entirely — running at goal is exactly the
/// right decision there, regardless of which third of the pitch it
/// starts from. A real "wall" (this many defenders or more ahead) gets
/// the full, zone-based discount. Between the two, the discount scales
/// linearly rather than snapping — reasoned estimates (category d), not
/// sourced.
const CLEAR_BREAK_MAX_DEFENDERS: usize = 1;
const WALL_DEFENDERS: usize = 4;

fn effective_carry_priority(base_priority: f32, defenders_ahead: usize) -> f32 {
    if defenders_ahead <= CLEAR_BREAK_MAX_DEFENDERS {
        1.0
    } else if defenders_ahead >= WALL_DEFENDERS {
        base_priority
    } else {
        let t = (defenders_ahead - CLEAR_BREAK_MAX_DEFENDERS) as f32
            / (WALL_DEFENDERS - CLEAR_BREAK_MAX_DEFENDERS) as f32;
        1.0 * (1.0 - t) + base_priority * t
    }
}

fn carry_value(
    from: Vector3<f32>,
    forward: Vector3<f32>,
    own_goal: Vector3<f32>,
    opponent_goal: Vector3<f32>,
    opponents: &[MatchPlayerLite],
    group: PlayerFieldPositionGroup,
) -> (f32, Vector3<f32>) {
    let pressure = 1.0 - openness(from, opponents);
    let risk = turnover_cost(own_goal, from) * pressure * CARRY_RISK_MULT;
    let base_priority = carry_priority_mult(group, (from - own_goal).magnitude());
    let defenders_ahead = defenders_ahead_count(from, forward, opponent_goal, opponents);
    let priority = effective_carry_priority(base_priority, defenders_ahead);

    if carry_fair_scoring() {
        // Fair-carry scoring (2026-09-29): a carry is scored on the SAME
        // currency as a pass (`pass_value`): the 0.20 retention floor, plus
        // 0.55 x progress toward goal measured at the carry's END POINT
        // (so a longer open run earns proportionally more), plus 0.30 x
        // the openness of that end point (a pass's own destination-
        // openness term; a first cut that kept the old space term as well
        // overshot to 97.8% carry). Heading is chosen by this score, not by raw
        // open distance (which tied at the cap and kept the first, -70deg,
        // heading). The floor is scaled by openness so a blocked lane
        // earns no retention credit. The role/zone priority still scales
        // the progress and space terms (the "responsible line" doctrine).
        let half = CARRY_FAN_HALF_ANGLE_DEG.to_radians();
        let mut best = (f32::MIN, forward, 0.0f32);
        for i in 0..CARRY_HEADING_COUNT {
            let t = if CARRY_HEADING_COUNT > 1 {
                i as f32 / (CARRY_HEADING_COUNT - 1) as f32
            } else {
                0.5
            };
            let heading = rotate_xy(forward, -half + 2.0 * half * t);
            let open = open_distance_capped(from, heading, opponents, CARRY_RUN_LOOKAHEAD);
            let end = from + heading * open.max(20.0);
            let prog = progress(from, end, opponent_goal);
            let short_open = (open / CARRY_LOOKAHEAD).clamp(0.0, 1.0);
            let end_open = openness(end, opponents);
            let v = CARRY_RETENTION_BASE * short_open
                + (PASS_PROGRESS_WEIGHT * prog + PASS_OPENNESS_WEIGHT * end_open) * priority
                - risk;
            if v > best.0 {
                best = (v, heading, open);
            }
        }
        let target = from + best.1 * best.2.max(20.0);
        return (best.0, target);
    }

    let (heading, open_dist) = carry_best_heading(from, forward, opponents);
    let openness_frac = (open_dist / CARRY_LOOKAHEAD).clamp(0.0, 1.0);
    let align = ((heading.dot(&forward) + 1.0) / 2.0).clamp(0.0, 1.0);
    let value = CARRY_SPACE_WEIGHT * openness_frac * (0.4 + 0.6 * align) * priority - risk;
    let target = from + heading * open_dist.max(20.0);
    (value, target)
}

pub(crate) fn nearest_blocker<'a>(
    from: Vector3<f32>,
    forward: Vector3<f32>,
    opponents: &'a [MatchPlayerLite],
) -> Option<&'a MatchPlayerLite> {
    opponents
        .iter()
        .filter(|o| {
            let to_opp = o.position - from;
            let along = to_opp.dot(&forward);
            if along <= 0.0 || along > DRIBBLE_BLOCK_RADIUS {
                return false;
            }
            let perp = (to_opp - forward * along).magnitude();
            perp < DRIBBLE_BLOCK_CORRIDOR
        })
        .min_by(|a, b| {
            let da = (a.position - from).magnitude();
            let db = (b.position - from).magnitude();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn dribble_value(
    ctx: &StateProcessingContext,
    from: Vector3<f32>,
    forward: Vector3<f32>,
    own_goal: Vector3<f32>,
    opponent_goal: Vector3<f32>,
    opponents: &[MatchPlayerLite],
    group: PlayerFieldPositionGroup,
) -> Option<(f32, Vector3<f32>)> {
    let blocker = nearest_blocker(from, forward, opponents)?;
    let beyond = blocker.position + forward * 30.0;
    let payoff_open = open_distance_in_heading(beyond, forward, opponents);
    let payoff = (payoff_open / CARRY_LOOKAHEAD).clamp(0.0, 1.0);
    let skill = (ctx.player.skills.technical.dribbling / 20.0).clamp(0.0, 1.0);
    let risk = turnover_cost(own_goal, from) * DRIBBLE_RISK_MULT;
    let base_priority = carry_priority_mult(group, (from - own_goal).magnitude());
    // The blocker himself is always one of the counted defenders ahead —
    // beating just him (nobody else in the way) is a genuine 1v1, not a
    // wall; several defenders stacked behind him is.
    let defenders_ahead = defenders_ahead_count(from, forward, opponent_goal, opponents);
    let priority = effective_carry_priority(base_priority, defenders_ahead);
    let value = DRIBBLE_BASE_WEIGHT * skill * (0.3 + 0.7 * payoff) * priority - risk;
    let lateral = Vector3::new(-forward.y, forward.x, 0.0);
    let blocker_lateral_offset = (blocker.position - from).dot(&lateral);
    let side = if blocker_lateral_offset >= 0.0 { -1.0 } else { 1.0 };
    let target = beyond + lateral * side * 14.0;
    Some((value, target))
}

fn clear_value(own_goal_dist: f32, nearest_opp_dist: f32) -> f32 {
    if own_goal_dist < CRISIS_OWN_GOAL_RADIUS && nearest_opp_dist < CRISIS_PRESSURE_RADIUS {
        CLEAR_VALUE
    } else {
        0.0
    }
}

/// The Step A-D decision: which action scores highest for the current
/// ball carrier, right now. `None` if this domain doesn't govern this
/// player this tick (no ball, directed, goalkeeper, mid-restart-window).
pub fn decide_on_ball_action(ctx: &StateProcessingContext) -> Option<Action> {
    if !ctx.player.has_ball(ctx) {
        return None;
    }
    #[cfg(feature = "match-logs")]
    {
        let p = ctx.player;
        crate::match_log_info!(
            "ONBALLDIAG t={} id={} state={:?} directive={} link={} supply={} kickoff={} throwin={} fk={} restart={:?}",
            ctx.context.total_match_time,
            p.id,
            p.state,
            p.behavioral_directive.is_some(),
            p.link_target.is_some(),
            p.supply_target.is_some(),
            p.kickoff_pass_pending,
            p.throw_in_pass_pending,
            p.free_kick_pass_pending,
            ctx.tick_context.ball.pass_origin_restart
        );
    }
    if is_directed_or_restart_pending(ctx) {
        return None;
    }
    // A shot this player has just struck is already in flight (the ball
    // is released a tick later): he decides nothing further. Without this
    // one action was dispatched as several Shoot events 40ms apart (a third
    // of all shots in a 10-match sample, 2026-09-26).
    if ctx.tick_context.ball.cached_shot_target.is_some() {
        return Some(Action::Hold);
    }
    let group = ctx.player.tactical_position.current_position.position_group();
    if group == PlayerFieldPositionGroup::Goalkeeper {
        return None;
    }

    let from = ctx.player.position;
    let own_goal = own_goal_position(ctx);
    let attacking_goal = ctx.player().opponent_goal_position();
    let forward = {
        let d = attacking_goal - from;
        if d.magnitude() > 1.0 {
            let n = d.normalize();
            Vector3::new(n.x, n.y, 0.0)
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        }
    };

    let opponents: Vec<MatchPlayerLite> = ctx.players().opponents().nearby(300.0).collect();

    // Domain 4 owns the whole final third (ball >= FINAL_THIRD_BOUNDARY from
    // our own goal): the progress-based pass value and the angle-less
    // shoot_value below never run for a carrier there.
    if super::box_attack::governs((from - own_goal).magnitude()) {
        return Some(super::box_attack::decide_final_third(ctx, from, attacking_goal, &opponents));
    }

    let mut best_action = Action::Hold;
    let mut best_value = HOLD_VALUE;

    let dist_to_goal = (attacking_goal - from).magnitude();
    if dist_to_goal <= SHOOT_EVAL_RADIUS {
        let v = shoot_value(from, attacking_goal, &opponents);
        if v > best_value {
            best_value = v;
            best_action = Action::Shoot;
        }
    }

    for tm in ctx.players().teammates().nearby(PASS_MAX_RANGE) {
        let dist = (tm.position - from).magnitude();
        if dist < 5.0 {
            continue;
        }
        let v = pass_value(ctx, from, &tm, own_goal, attacking_goal, &opponents);
        if v > best_value {
            best_value = v;
            best_action = Action::Pass(tm.id);
        }
        // Off-ball attacking runs, from null (`off_ball_runs.rs`): a
        // genuine through ball into a teammate's frozen run target,
        // valued on this domain's own progress/openness/risk currency
        // (unchanged philosophy — `pass_value`'s own terms), just aimed
        // at where he is running into instead of where he stands. Reads
        // `ctx.tick_context.run_commitments` (a fresh-every-tick cache),
        // NOT `ctx.context.players` — the latter is a one-time snapshot
        // that never sees this field (see `GameTickContext::run_
        // commitments`'s own doc comment for why).
        if let Some(&(_, target)) = ctx.tick_context.run_commitments.get(&tm.id) {
            if (target - from).magnitude() >= 5.0 {
                let prog = progress(from, target, attacking_goal);
                let open = openness(target, &opponents);
                let risk = intercept_risk(ctx, from, target, &opponents);
                let cost = turnover_cost(own_goal, target);
                let tv = PASS_RETENTION_BASE + PASS_PROGRESS_WEIGHT * prog
                    + PASS_OPENNESS_WEIGHT * open
                    - PASS_RISK_WEIGHT * risk * cost;
                if tv > best_value {
                    best_value = tv;
                    best_action = Action::Through { to: tm.id, target };
                }
            }
        }
    }

    let (cv, ctarget) = carry_value(from, forward, own_goal, attacking_goal, &opponents, group);
    if cv > best_value {
        best_value = cv;
        best_action = Action::Carry(ctarget);
    }

    if let Some((dv, dtarget)) =
        dribble_value(ctx, from, forward, own_goal, attacking_goal, &opponents, group)
    {
        if dv > best_value {
            best_value = dv;
            best_action = Action::Dribble(dtarget);
        }
    }

    let own_goal_dist = (from - own_goal).magnitude();
    let nearest_opp = nearest_opponent_dist(from, &opponents);
    let clv = clear_value(own_goal_dist, nearest_opp);
    if clv > best_value {
        best_action = Action::Clear;
    }

    #[cfg(feature = "match-logs")]
    {
        let (kind, tx, ty) = match best_action {
            Action::Shoot => ("Shoot", 0.0, 0.0),
            Action::Pass(id) => ("Pass", id as f32, 0.0),
            Action::Carry(t) => ("Carry", t.x, t.y),
            Action::Dribble(t) => ("Dribble", t.x, t.y),
            Action::Clear => ("Clear", 0.0, 0.0),
            Action::Hold => ("Hold", 0.0, 0.0),
            Action::Deliver { to, .. } => ("Deliver", to as f32, 0.0),
            Action::Through { to, .. } => ("Through", to as f32, 0.0),
        };
        crate::match_log_info!(
            "ONBALLACT t={} id={} group={:?} action={} value={:.3} own_goal_dist={:.1} tx={:.1} ty={:.1} fx={:.2} fy={:.2}",
            ctx.context.total_match_time,
            ctx.player.id,
            group,
            kind,
            best_value,
            own_goal_dist,
            tx,
            ty,
            forward.x,
            forward.y
        );
    }

    Some(best_action)
}

/// Fresh, deliberately blunt clearance velocity — Step 7's "genuine last
/// resort... blunt, not a refined score" carried through to execution
/// too, per the design doc's own scope note that the old
/// `DefenderClearingState`'s skill-scaled jitter/targeting logic is
/// decision-adjacent and not called here. Aims roughly upfield, biased
/// toward the nearer touchline (a real turnover-avoidance choice — clear
/// away from the danger of a central recycle, not just far), using a
/// real skill-scaled strength for speed and the shared, kept ballistic
/// height conversion for a plausible hoofed-clearance arc.
pub fn build_clear_velocity(ctx: &StateProcessingContext) -> Vector3<f32> {
    let ball_pos = ctx.player.position;
    let field_width = ctx.context.field_size.width as f32;
    let field_height = ctx.context.field_size.height as f32;
    let attack_dir_sign: f32 = match ctx.player.side {
        Some(PlayerSide::Left) => 1.0,
        _ => -1.0,
    };
    let target_x = (ball_pos.x + attack_dir_sign * 260.0).clamp(20.0, field_width - 20.0);
    let nearer_touchline_y = if ball_pos.y < field_height / 2.0 {
        20.0
    } else {
        field_height - 20.0
    };
    let target_y = (ball_pos.y * 0.4 + nearer_touchline_y * 0.6).clamp(20.0, field_height - 20.0);
    let target = Vector3::new(target_x, target_y, 0.0);
    let to_target = target - ball_pos;
    let dist = to_target.magnitude().max(1.0);
    let dir = to_target / dist;
    let strength = (ctx.player.skills.physical.strength / 20.0).clamp(0.3, 1.0);
    let speed = 3.5 + strength * 1.5;
    let horiz = dir * speed;
    let height_m = 5.0 + strength * 1.0;
    let z = crate::r#match::engine::physics::z_launch_velocity_for_height(height_m);
    Vector3::new(horiz.x, horiz.y, z)
}
