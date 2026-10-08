//! Off-ball defensive positioning — designed from null.
//!
//! `docs/decision-from-null.md` (domain 1) /
//! `docs/off-ball-defensive-positioning-from-null.md` (this domain's own
//! football reasoning, spec, and verification log — read that first).
//!
//! A single, team-wide, per-tick allocation of duty (press / mark /
//! cover / hold-shape / scramble) across every out-of-possession
//! defender and midfielder, run once per full tick from
//! `game_tick_inner` before player AI dispatch. Deliberately calls no
//! pre-existing decision function from this engine (no
//! `defensive_role_for_ball_carrier_unified`, `find_unmarked_opponent`,
//! `find_help_target`, `calculate_opponent_danger_score`,
//! `cover_target_position`, or any of their siblings in
//! `common/players/ops/defensive.rs`) — every ranking, threshold, and
//! target-position formula here is derived fresh from the football
//! reasoning in the design doc, using only raw `MatchField`/
//! `MatchContext`/`MatchPlayer` data plus genuine measurements
//! (`max_speed_with_condition`, `goal_positions`, Euclidean geometry).
//!
//! Process check (`docs/decision-from-null.md`'s required self-audit):
//! this module's call graph touches only `MatchField`/`MatchContext`/
//! `MatchPlayer` struct fields, `nalgebra::Vector3` geometry, and
//! `PlayerSkills::max_speed_with_condition` — a measurement (a player's
//! real, condition-adjusted top speed; encodes no judgment, two
//! different-minded coaches would compute the identical number). Zero
//! pre-existing decision-function dependencies.

use crate::PlayerFieldPositionGroup;
use crate::r#match::defenders::states::DefenderState;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::player::DefensiveDuty;
use crate::r#match::player::state::PlayerState;
use crate::r#match::{MatchContext, MatchField, PlayerSide};
use nalgebra::Vector3;

/// Real jockeying distance — the presser stands this far off the
/// carrier, on the goal side, showing him away from danger rather than
/// diving in. 1.5m is the standard coaching cue for "close enough to
/// pressure, not so close he can be turned or nutmegged" (arm's length
/// plus a step). 8 units/metre, this project's established scale.
const JOCKEY_DISTANCE_UNITS: f32 = 12.0;

/// A press is only assigned if the best available closing time is
/// under this many ticks (~2.2s at 100 ticks/s) — a real defender
/// doesn't set off on a chase he can't realistically win; a hopeless
/// sprint just drags him out of position for nothing. Reasoned
/// estimate (category d, no sourced dataset for "how long a real
/// defender will chase before giving up") — a verification target for
/// the raw-trace pass, not treated as final.
const MAX_PRESS_CLOSING_TICKS: f32 = 220.0;

/// Opponents beyond this distance from the ball aren't ranked as
/// current threats at all — real marking doesn't extend to the far
/// side of the pitch. ~35m.
const THREAT_SCAN_RADIUS_UNITS: f32 = 280.0;

/// Below this danger score, an opponent isn't worth pulling a defender
/// out of shape to mark individually — real teams don't man-mark every
/// opponent on the pitch, only the currently dangerous ones.
const DANGER_FLOOR: f32 = 0.30;

/// Attacker value (xG, `box_attack::threat_value_at`) that counts as full
/// danger: a central player ~10m from goal.
const THREAT_FULL_VALUE: f32 = 0.10;

/// Real defensive-zone doctrine: a centre-back or full-back stepping up
/// to press is normal inside the defensive third plus a margin; a
/// carrier further out than this is a midfielder's job, not a
/// defender's. Reasoned estimate (category d, real pitch thirds —
/// defensive/middle/attacking each ~35m of a 105m pitch — but no
/// sourced dataset for the exact defender-press cutoff). Verified
/// necessary by raw trace (Pavel's "defenders press midfielders"
/// report, 2026-09-22): without this gate, 27% of all defender-press
/// assignments had the carrier beyond 35m from goal (up to 44m at the
/// 90th percentile) — genuinely midfield territory, not a defender's
/// zone — because Step C's pure closing-time race has no concept of
/// "whose job this is," only "who gets there fastest."
const PRESS_DEFENDER_ZONE_UNITS: f32 = 320.0; // 40.0m

/// The forward-side mirror of `PRESS_DEFENDER_ZONE_UNITS`. Real doctrine:
/// a forward's off-ball defensive job is triggering pressure high up the
/// pitch — closing down the opposition's goalkeeper/centre-backs/pivot
/// while they're building out — not chasing back to help a settled
/// defensive shape. A carrier this far from the OPPONENT's goal (i.e.
/// already progressed well past the forward line) is a defender's or
/// midfielder's job, not a forward's. Same magnitude and same reasoned-
/// estimate status as the defender gate (category d, real pitch-thirds
/// doctrine, no sourced exact cutoff) — deliberately kept symmetric
/// rather than independently guessed, since it's the same underlying
/// "whose job is this zone" question from the opposite end of the pitch.
const PRESS_FORWARD_ZONE_UNITS: f32 = 320.0; // 40.0m

/// A hard cap on how far from OUR OWN GOAL an opponent can be and
/// still be worth pulling a defender out of shape to individually
/// mark — distinct from `THREAT_SCAN_RADIUS_UNITS` (distance from the
/// BALL, which only checks "could this opponent plausibly receive a
/// pass," not "is he actually near enough to our goal to be a real
/// defensive concern"). Verified necessary by raw trace (same report):
/// without this gate, marked opponents sat a median 46.2m from goal,
/// and 48.1% were beyond 50m — nearly half of all individual marking
/// assignments were tracking a player who wasn't a genuine threat to
/// our goal at all. Real teams hold a compact shape against a deep-
/// lying opponent rather than chasing him individually that far out.
const MARK_MAX_DIST_UNITS: f32 = 320.0; // 40.0m, matches the press zone

/// Marking tightness bounds — touch-tight at the edge of our box,
/// several metres of breathing room in the middle third (enough to see
/// the ball and cut the passing lane, not so close it matters yet).
const MARK_TIGHT_UNITS: f32 = 8.0; // 1.0m
const MARK_LOOSE_UNITS: f32 = 40.0; // 5.0m

/// Real cover stand-off distance — a covering defender sits this far
/// behind the ball on the ball-to-goal line, deep enough to react if
/// the presser is beaten, not level with him (that would just be a
/// second presser, leaving nobody behind).
const COVER_STANDOFF_UNITS: f32 = 64.0; // 8.0m

/// Wide-play plan (2026-10-04): the presser aims where the carrier will be
/// this many ticks ahead (not where he is), so he can meet a cut-in instead
/// of chasing a point 1.5 m goal-side of the carrier's current spot at the
/// same top speed forever (measured: 7.6 m away at +0.3 s, still 4.9 m at
/// +3 s). ~0.6 s at the engine's 100 ticks/s. Category (d) estimate.
const PRESS_LEAD_TICKS: f32 = 60.0;
/// How far the covering defender looks ahead along the carrier's path.
/// ~1.0 s. Category (d) estimate.
const COVER_LEAD_TICKS: f32 = 100.0;
/// A carrier further than this from the centre line counts as "wide" for
/// the presser's show-outside bias (100u = 12.5 m, the same wide-lane cut
/// the measurement script uses), full effect 120u further out.
const WIDE_START_UNITS: f32 = 100.0;
const WIDE_RAMP_UNITS: f32 = 120.0;
/// Share of the inward direction mixed into the presser's stand-off
/// direction for a wide carrier (0 = straight goal-side as before).
const SHOW_OUTSIDE_WEIGHT: f32 = 0.8;

/// Defensive-line depth bounds and how much the ball's own distance
/// from our goal pulls the line forward. Real doctrine: push up when
/// the ball is contained and far away, sit deep when it's dangerously
/// advanced — bounded so the line never sits inside our own box nor
/// pushes all the way to halfway regardless of ball position.
const LINE_PUSH_FACTOR: f32 = 0.55;
const LINE_MIN_DEPTH_UNITS: f32 = 96.0; // 12.0m from goal
/// Lowered from 320u/40m (2026-09-22, Pavel's "why do they run
/// forward, stay in your line" report). 40m is not a realistic ceiling
/// for a default defensive line depth — that's essentially the
/// halfway line, and real doctrine puts even an aggressive, organised
/// high defensive line at roughly 30-35m from goal at its most
/// advanced, not as a routine baseline. Reasoned estimate (category
/// d), not a fix for a specific measured bug at this exact constant —
/// the actual kickoff collision was traced to a separate, now-fixed
/// issue (the universal separation correction being overwritten by
/// this system, see processor.rs's `separation_nudge` repositioning,
/// same date) — but 40m was too generous regardless and is tightened
/// here too, since it directly answers "nobody needs them that high."
const LINE_MAX_DEPTH_UNITS: f32 = 260.0; // 32.5m from goal

/// Real compactness doctrine: the midfield line holds roughly this far
/// ahead of the back line, both moving together as the ball's own
/// depth changes — real teams keep a compact ~25-35m block from back
/// line to front line, of which the back-to-midfield gap is a portion.
/// Added 2026-09-22, after Pavel asked directly why defenders were
/// visibly running forward while midfielders stood still: Step F
/// previously computed ONE shared depth for the whole leftover pool,
/// with no distinction between a defender and a midfielder at all —
/// when that single target happened to land near where midfielders
/// already naturally sit, defenders had to visibly close a real gap to
/// reach it while midfielders barely moved, looking exactly like two
/// unrelated groups with "different instincts" rather than one
/// synchronised shape. Reasoned estimate (category d, no sourced exact
/// figure for this specific gap).
const MIDFIELD_LINE_GAP_UNITS: f32 = 120.0; // 15.0m ahead of the back line

/// How much a hold-shape player's lateral (width) position is pulled
/// toward the ball's own side — real compactness doctrine (shift as a
/// block toward the ball, don't spread evenly across the width of the
/// pitch).
const WIDTH_PULL_FRACTION: f32 = 0.30;

/// How far ahead of the ball carrier's CURRENT position the line-depth
/// calculation looks, extrapolated from the carrier's own live
/// velocity. Raw-traced before adding this (Pavel's "defenders don't
/// drop into the box" report, 2026-09-22): the un-anticipated line
/// target tracked the ball's instantaneous position correctly, but a
/// player's realized depth during a genuinely fast-developing attack
/// lagged the target by a median 19-36m through the first 1-3 seconds
/// of a sustained possession, only closing (to <1m) after 4-5+ seconds
/// — i.e. the line was purely reactive and never got a head start,
/// exactly the class of bug this project's `shot_lane_cover_
/// override_velocity` already solved once for an individual defender's
/// cover target ("a target that keeps moving away at up to the
/// carrier's own full pace only closes by the DIFFERENCE in speeds,
/// not full speed"). 80 ticks (0.8s) is a reasoned estimate (category
/// d, no sourced dataset for "how far ahead a real back line reads an
/// attack") — long enough to give the whole line a meaningful head
/// start on a real sprint-speed break, short enough that a carrier
/// changing direction doesn't send the line lurching toward a stale
/// prediction.
const LINE_LEAD_TICKS: f32 = 80.0;

/// Real off-ball spacing from a nearby attacker — Pavel's own coaching
/// reference point ("3-6m between them and attackers"); this is the
/// hard floor of that band (below 3m, push away; within the 3-6m band
/// itself no correction is needed, real defending doesn't need to be
/// exactly one specific distance, just not glued to an attacker).
/// Below this, a hold-shape player's target gets pushed directly away
/// from his own single nearest opponent (not a shared point — each
/// player reacts to HIS OWN nearest attacker, which is naturally
/// spread across different attackers for different defenders, so this
/// is anti-clustering by construction, not by explicit fan-out logic
/// like Step G's). Scales with the same danger-based tightness idea as
/// Step D's marking bounds, but applies universally (every hold-shape
/// player, not just the top-ranked marked threats).
const ATTACKER_SPACING_UNITS: f32 = 24.0; // 3.0m — never closer than this

/// Ball within this distance of our own goal is a genuine box crisis —
/// real defenders stop tracking individual assignments and just get
/// bodies between the danger and the goal. Roughly the penalty area
/// plus a margin. Reasoned estimate (category d) — a first-pass proxy
/// for "shot/cross imminent," not a literal check of ball height or an
/// in-flight shot (that measurement doesn't exist at this call site).
const CRISIS_RADIUS_UNITS: f32 = 176.0; // 22.0m

/// Lateral spacing between neighbouring scramble responders once
/// they're spread evenly across the box (see the Step G rewrite below)
/// — real marking distance for a crisis moment, not a tight huddle.
/// Widened 2026-09-23 (Pavel's direct report: "defenders and attackers
/// are all grouped somewhere around penalty spot — nobody is covering
/// areas to the left or to the right"): the previous 24u/3m increment,
/// applied as a small fixed step out from a single central point,
/// meant even a genuine crisis with 4-6 responders only ever spanned
/// ~9-15m of a real ~40m-wide box — a tight central clump by
/// construction, regardless of how many bodies were actually
/// available to cover it.
const SCRAMBLE_SPREAD_STEP_UNITS: f32 = 56.0; // ~7.0m between neighbours

/// Hard cap on how far from the danger point's own centre line the
/// scramble spread can reach on either side — stays inside the real
/// box width (a real box is ~40m wide, so ~161u half-width at this
/// project's scale) rather than sending a defender to stand outside
/// the box chasing a phantom lane, however many responders there are.
const SCRAMBLE_MAX_DEPTH_UNITS: f32 = 80.0; // 10m
const SCRAMBLE_MAX_HALF_SPAN_UNITS: f32 = 140.0; // ~17.5m either side

/// Box defence (2026-10-08): when the ball is wide in the crossing area of
/// OUR goal (>= 12.5m off the centre line, within 25m of the goal line), the
/// players on hold-shape duty take positions that cover the box instead of
/// holding the line outside it: near-post guard, penalty spot, far post,
/// box edge, and two back-side cover positions. Measured at the moment of a
/// cross (StatsBomb 360, 221 crosses vs 60 engine matches): defenders in
/// the box 5.43 vs 3.30, their lateral spread 6.6m vs 4.8m, and 2.8 hold-
/// shape players per delivery stood a median 5m outside the box.
/// Slot geometry mirrors `box_attack::zone_geometry` (units: depth from the
/// goal line, lateral toward the carrier's side). `BOX_DEFENCE_OFF` (any
/// value) disables (same-binary A/B).
const BOX_DEF_MIN_LATERAL: f32 = 100.0;
const BOX_DEF_MAX_DEPTH: f32 = 200.0;
const BOX_DEF_SLOTS: [(f32, f32); 6] =
    [(40.0, 30.0), (88.0, 0.0), (48.0, -37.0), (140.0, 0.0), (112.0, 72.0), (112.0, -72.0)];
const BOX_MARK_DEPTH: f32 = 132.0;
const BOX_MARK_HALF_WIDTH: f32 = 161.0;
const BOX_MARK_MIN_UNITS: f32 = 20.0; // 2.5m
fn box_defence_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BOX_DEFENCE_OFF").is_none())
}

/// A tackle attempt is only live once the presser is genuinely this
/// close AND cover has arrived (missing is no longer fatal) — or, if
/// no cover exists this tick, only once much closer still (a real
/// defender sometimes has to commit anyway rather than let the
/// carrier escape, but only as a last resort).
const ENGAGE_WITH_COVER_UNITS: f32 = 16.0; // 2.0m
const ENGAGE_WITHOUT_COVER_UNITS: f32 = 9.6; // 1.2m

/// Stickiness — a real marker doesn't swap onto a different opponent
/// just because a teammate is fractionally closer this exact tick; he
/// stays with his man through ordinary movement and only hands off
/// when a teammate is genuinely, clearly better placed. Applied as a
/// discount on the incumbent's effective ranking distance in Steps
/// C/D/E. Verified necessary, not just theorized: an undiscounted
/// nearest-defender-wins recompute (no memory at all) measured a
/// median marker-assignment run length of only 2 ticks in a real raw
/// trace — the incumbent must be a real 40% farther before a
/// challenger takes over, not a marginal few units.
const STICKY_FACTOR: f32 = 0.6;

/// Lightweight, cheap-to-clone snapshot of everything the assignment
/// pass needs about one outfield player. Collected once per tick from
/// `field.players` before any mutation, so the algorithm never has to
/// fight the borrow checker over reading opponents while writing
/// duties.
#[derive(Clone, Copy)]
struct Cand {
    id: u32,
    team_id: u32,
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    max_speed: f32,
    position_group: PlayerFieldPositionGroup,
    start_position: Vector3<f32>,
    side: Option<PlayerSide>,
    /// This player's duty from the *previous* tick — read before this
    /// tick's reset, purely to bias continuity (Steps C/D/E). Never
    /// used as a source of new information, only as a tie-break toward
    /// "keep the existing assignment."
    prev_duty: DefensiveDuty,
    /// True while this player is in the `TakeBall` state — the
    /// established, already-proven mechanism (`should_force_takeball`)
    /// that commits the single genuinely-closest teammate to chase a
    /// loose ball. Used only to exclude that one player from this
    /// system's own loose-ball assignments below, so the two mechanisms
    /// never fight over the same player.
    is_taking_ball: bool,
}

/// One player's resolved assignment for this tick.
struct Assignment {
    id: u32,
    duty: DefensiveDuty,
    target: Vector3<f32>,
    engage: bool,
}

/// Entry point — called once per full tick from `game_tick_inner`,
/// before player AI dispatch. Resets every player's
/// `ai_defensive_duty` to `None` each call, then assigns real duties.
/// Two cases: (1) an opponent clearly holds the ball — the original
/// single-team path, unchanged; (2) the ball is loose (a genuine
/// contest, 50/50, or post-block scramble) — BOTH teams organise
/// around the ball's own position, since either side could plausibly
/// end up with it. A team's own in-possession phase (its own clear
/// carrier) still leaves this system fully inactive for THAT team —
/// on-ball decisions are a separate, not-yet-rewritten domain.
///
/// Case (2) was added 2026-09-22, after Pavel watched a genuine 5-a-6
/// scramble just outside the box resolve with the back four already
/// abandoning it — the ORIGINAL design deliberately scoped loose balls
/// out entirely ("a different, not-yet-rewritten domain"), which meant
/// this whole system went fully dark exactly during real football's
/// most chaotic, highest-stakes moments: a contested ball near either
/// box. Whichever player is already the established, genuinely-
/// closest chaser (`TakeBall` state, the separate, working
/// `should_force_takeball` mechanism) is excluded from these new
/// assignments so the two systems never fight over the same player —
/// this system only governs everyone ELSE, using the same Mark/Cover/
/// HoldShape/Scramble machinery already proven for the clear-
/// possession case, anchored on the ball's own position instead of a
/// specific carrier's. Press/Engage (Steps C/H) are skipped for a
/// loose ball by design — closing on a ball nobody owns yet is exactly
/// what `TakeBall` already does; duplicating it here would only
/// create a second, uncoordinated chaser.
pub fn assign_defensive_duties(field: &mut MatchField, context: &MatchContext) {
    // Reset-and-return helper for the early-exit paths below (no clear
    // carrier, carrier not found, defending side unknown) — a genuine
    // loose-ball / possession-transition moment, where starting fresh
    // next tick is the correct behaviour, not a stickiness concern.
    fn reset_all(field: &mut MatchField) {
        for p in field.players.iter_mut() {
            p.ai_defensive_duty = DefensiveDuty::None;
            p.ai_defensive_engage = false;
        }
    }

    // Captures each player's PREVIOUS tick's duty (`prev_duty`) before
    // any reset happens this tick — the only source of continuity
    // Steps C/D/E's stickiness bias reads from. Built unconditionally
    // (needed by both cases below) before any reset happens.
    let cands: Vec<Cand> = field
        .players
        .iter()
        .filter(|p| !p.is_sent_off)
        .map(|p| Cand {
            id: p.id,
            team_id: p.team_id,
            position: p.position,
            velocity: p.velocity,
            max_speed: p.skills.max_speed_with_condition(p.player_attributes.condition),
            position_group: p.tactical_position.current_position.position_group(),
            start_position: p.start_position,
            side: p.side,
            prev_duty: p.ai_defensive_duty,
            is_taking_ball: matches!(
                p.state,
                PlayerState::Defender(DefenderState::TakeBall)
                    | PlayerState::Midfielder(MidfielderState::TakeBall)
                    | PlayerState::Forward(ForwardState::TakeBall)
            ),
        })
        .collect();

    let goal_for_side = |side: PlayerSide| match side {
        PlayerSide::Left => context.goal_positions.left,
        PlayerSide::Right => context.goal_positions.right,
    };
    let opposite_side = |side: PlayerSide| match side {
        PlayerSide::Left => PlayerSide::Right,
        PlayerSide::Right => PlayerSide::Left,
    };

    #[cfg(feature = "match-logs")]
    crate::match_log_info!(
        "CASEDIAG t={} owner={:?} pending_passer={:?} previous_owner={:?} pass_target={:?}",
        context.total_match_time,
        field.ball.current_owner,
        field.ball.pending_pass_passer,
        field.ball.previous_owner,
        field.ball.pass_target_player_id
    );
    let assignments: Vec<Assignment> = if let Some(carrier_id) = field.ball.current_owner {
        // Case 1: an opponent clearly holds the ball — the original,
        // single-team path.
        let Some(carrier) = cands.iter().find(|c| c.id == carrier_id).copied() else {
            reset_all(field);
            return;
        };
        let defending_team = if carrier.team_id == field.home_team_id {
            field.away_team_id
        } else {
            field.home_team_id
        };
        let Some(defending_side) = cands
            .iter()
            .find(|c| c.team_id == defending_team)
            .and_then(|c| c.side)
        else {
            reset_all(field);
            return;
        };
        assign_team(
            &cands,
            defending_team,
            carrier.team_id,
            carrier.position,
            carrier.velocity,
            Some(carrier.id),
            goal_for_side(defending_side),
            goal_for_side(opposite_side(defending_side)),
            defending_side,
        )
    } else if let Some(passer_id) = field.ball.pending_pass_passer {
        // Case 2a (added 2026-09-23, folded in from domain 2's carry-over
        // list — docs/decision-from-null.md's carry-over section,
        // docs/on-ball-attacking-decisions-from-null.md's own scope
        // note): an ordinary directed pass in flight, NOT a genuine loose
        // ball. `current_owner` goes `None` the instant a pass is struck
        // and stays that way until the receiver claims it — but the
        // PASSING team is still, in every real football sense,
        // comfortably in possession for that window; only the DEFENDING
        // team should organise, exactly as Case 1 already does for a
        // known carrier. Diagnosed and quantified before this fix
        // (`DEFDUTY_CASE2`, still logged below): ~25-26% of what used to
        // be genuine Case-2 firings coincided with this exact signal,
        // needlessly pulling the passing team's own players back into
        // defensive duty mid-pass. Routes through the SAME single-team
        // `assign_team` Case 1 uses, anchored on the ball's own live
        // position/velocity (no single "carrier" mid-flight) rather than
        // a specific player's position.
        let Some(passer) = cands.iter().find(|c| c.id == passer_id) else {
            reset_all(field);
            return;
        };
        let defending_team = if passer.team_id == field.home_team_id {
            field.away_team_id
        } else {
            field.home_team_id
        };
        let Some(defending_side) = cands
            .iter()
            .find(|c| c.team_id == defending_team)
            .and_then(|c| c.side)
        else {
            reset_all(field);
            return;
        };
        assign_team(
            &cands,
            defending_team,
            passer.team_id,
            field.ball.position,
            field.ball.velocity,
            None,
            goal_for_side(defending_side),
            goal_for_side(opposite_side(defending_side)),
            defending_side,
        )
    } else if let Some(prev_owner_id) = field
        .ball
        .previous_owner
        // 2026-09-25: `previous_owner` alone left ~31% of ticks in the
        // loose-ball branch below; 97% of those were a moving ball within
        // 3s of a CONTROLLED touch (a pass/clearance/carry that outran
        // its owner) — still that touch's team's ball. Only an
        // uncontrolled or stale last touch is genuinely loose.
        .or_else(|| field.ball.possessing_touch())
    {
        // Case 2c (added 2026-09-24, "team follows the ball" investigation
        // continued): a settled dribble/hold, or a pass whose completion
        // was already credited (stats-wise) before the ball has
        // physically reached the receiver. Neither `current_owner` nor
        // `pending_pass_passer` represents this — `current_owner` is set
        // for only a single tick at the instant of a touch, and the
        // pass-completion credit path (`credit_completed_pass`) clears
        // `pending_pass_passer` the moment the pass is deemed to succeed,
        // which can run well before the ball's own physics simulation
        // (`motion.rs`) actually gets it there — a raw-traced example
        // showed a 2.25s gap between that credit and the receiver's real
        // claim. `previous_owner`, by contrast, is kept truthful by
        // `check_ball_ownership` itself on real ball-distance grounds
        // (cleared only once the previous owner is more than
        // `BALL_DISTANCE_THRESHOLD`/5 units from the ball's CURRENT
        // position) — so `Some(_)` here means someone was genuinely,
        // physically in possession moments ago and the ball hasn't
        // meaningfully separated from them since. Same single-team
        // routing as Case 2a: only the OTHER team organises.
        let Some(prev_owner) = cands.iter().find(|c| c.id == prev_owner_id) else {
            reset_all(field);
            return;
        };
        let defending_team = if prev_owner.team_id == field.home_team_id {
            field.away_team_id
        } else {
            field.home_team_id
        };
        let Some(defending_side) = cands
            .iter()
            .find(|c| c.team_id == defending_team)
            .and_then(|c| c.side)
        else {
            reset_all(field);
            return;
        };
        assign_team(
            &cands,
            defending_team,
            prev_owner.team_id,
            field.ball.position,
            field.ball.velocity,
            None,
            goal_for_side(defending_side),
            goal_for_side(opposite_side(defending_side)),
            defending_side,
        )
    } else {
        // Case 2b: a genuine loose ball — both teams organise around
        // the ball's own position (see this function's own doc
        // comment for why). If a team has no players left to derive a
        // side from (e.g. every outfielder sent off — a degenerate
        // edge case), that team is simply skipped rather than aborting
        // the whole tick for the other team too.
        let ball_pos = field.ball.position;
        let ball_vel = field.ball.velocity;
        // Diagnostic, kept permanently (feature-gated, zero cost without
        // match-logs): added 2026-09-23, Pavel's "defense/midfield never
        // cross halfway" question — the finding that motivated the Case
        // 2a branch above. Now only fires for genuine loose balls (the
        // in-flight-pass case is routed away by the `pending_pass_passer`
        // branch above), so this counter's own "coincides with a pass
        // signal" share should read near zero going forward; kept to
        // verify that directly rather than assumed.
        #[cfg(feature = "match-logs")]
        crate::match_log_info!(
            "DEFDUTY_CASE2 t={} has_pass_target={} has_pending_passer={} owner_prev={:?}",
            context.total_match_time,
            field.ball.pass_target_player_id.is_some(),
            field.ball.pending_pass_passer.is_some(),
            field.ball.previous_owner
        );
        #[cfg(feature = "match-logs")]
        crate::match_log_info!(
            "DARKCLASS t={} dead={} pending_taker={} lt_team={:?} lt_ctrl={} lt_age={} speed={:.2} pass_origin={:?} home={}",
            context.total_match_time,
            context.total_match_time < context.dead_ball_until_ms,
            field.ball.restart_pending_taker.is_some(),
            field.ball.last_touch_team_id,
            field.ball.last_touch_was_controlled,
            field.ball.current_tick_cached.saturating_sub(field.ball.last_touch_tick),
            field.ball.velocity.norm(),
            field.ball.pass_origin_restart,
            field.home_team_id
        );
        let mut all = Vec::new();
        for (team_id, opponent_team) in [
            (field.home_team_id, field.away_team_id),
            (field.away_team_id, field.home_team_id),
        ] {
            let Some(side) = cands.iter().find(|c| c.team_id == team_id).and_then(|c| c.side)
            else {
                continue;
            };
            all.extend(assign_team(
                &cands,
                team_id,
                opponent_team,
                ball_pos,
                ball_vel,
                None,
                goal_for_side(side),
                goal_for_side(opposite_side(side)),
                side,
            ));
        }
        all
    };

    reset_all(field);

    // Diagnostic-only helpers: the ball's own current position (a valid
    // stand-in for "the danger point" in both cases — it's where the
    // carrier is when there is one) and each logged player's OWN
    // team's goal (since Case 2 can log assignments from both teams in
    // the same pass, there's no single shared `own_goal` any more).
    #[cfg(feature = "match-logs")]
    let ball_pos_for_log = field.ball.position;
    #[cfg(feature = "match-logs")]
    let own_goal_for = |player_id: u32| -> Vector3<f32> {
        cands
            .iter()
            .find(|c| c.id == player_id)
            .and_then(|c| c.side)
            .map(goal_for_side)
            .unwrap_or(Vector3::zeros())
    };
    #[cfg(feature = "match-logs")]
    let opponent_goal_for = |player_id: u32| -> Vector3<f32> {
        cands
            .iter()
            .find(|c| c.id == player_id)
            .and_then(|c| c.side)
            .map(|s| goal_for_side(opposite_side(s)))
            .unwrap_or(Vector3::zeros())
    };

    #[cfg(feature = "match-logs")]
    for a in &assignments {
        match a.duty {
            DefensiveDuty::Mark { opponent_id } => {
                if let Some(marker) = cands.iter().find(|c| c.id == a.id) {
                    let opp_dist_goal = cands
                        .iter()
                        .find(|c| c.id == opponent_id)
                        .map(|o| (o.position - own_goal_for(a.id)).magnitude())
                        .unwrap_or(-1.0);
                    crate::match_log_info!(
                        "DEFDUTY_MARK t={} marker={} marker_group={:?} marker_dist_start={:.1} opp={} opp_dist_goal={:.1}",
                        context.total_match_time,
                        a.id,
                        marker.position_group,
                        (marker.position - marker.start_position).magnitude(),
                        opponent_id,
                        opp_dist_goal
                    );
                }
            }
            DefensiveDuty::Cover => {
                let dist = (a.target - ball_pos_for_log).magnitude();
                crate::match_log_info!(
                    "DEFDUTY_COVER t={} cover={} ball_dist_from_target={:.1}",
                    context.total_match_time,
                    a.id,
                    dist
                );
            }
            // Diagnostic, kept permanently (feature-gated, zero cost
            // without match-logs): confirms `PRESS_DEFENDER_ZONE_UNITS`
            // stays respected as this system evolves. Added for Pavel's
            // "defenders press midfielders" report, 2026-09-22.
            DefensiveDuty::Press { opponent_id } => {
                if let Some(presser) = cands.iter().find(|c| c.id == a.id) {
                    crate::match_log_info!(
                        "DEFDUTY_PRESS t={} presser={} presser_group={:?} presser_dist_start={:.1} carrier={} own_goal_dist={:.1} opp_goal_dist={:.1}",
                        context.total_match_time,
                        a.id,
                        presser.position_group,
                        (presser.position - presser.start_position).magnitude(),
                        opponent_id,
                        (ball_pos_for_log - own_goal_for(a.id)).magnitude(),
                        (ball_pos_for_log - opponent_goal_for(a.id)).magnitude()
                    );
                }
            }
            DefensiveDuty::Scramble => {
                // target_x/target_y added 2026-09-23 (Pavel's "grouped
                // around penalty spot, nobody covering left/right"
                // report) — the original line only logged ball distance,
                // which can't show whether the scramble spread actually
                // widened.
                crate::match_log_info!(
                    "DEFDUTY_SCRAMBLE t={} p={} ball_dist_goal={:.1} target_x={:.1} target_y={:.1}",
                    context.total_match_time,
                    a.id,
                    (ball_pos_for_log - own_goal_for(a.id)).magnitude(),
                    a.target.x,
                    a.target.y
                );
            }
            // Diagnostic, kept permanently (feature-gated, zero cost
            // without match-logs): HoldShape was never logged at all
            // before Pavel's "defenders don't space/drop" report,
            // 2026-09-22.
            DefensiveDuty::HoldShape => {
                let og = own_goal_for(a.id);
                crate::match_log_info!(
                    "DEFDUTY_HOLDSHAPE t={} p={} ball_dist_goal={:.1} target_dist_goal={:.1} player_dist_goal={:.1}",
                    context.total_match_time,
                    a.id,
                    (ball_pos_for_log - og).magnitude(),
                    (a.target - og).magnitude(),
                    (cands.iter().find(|c| c.id == a.id).map(|c| c.position).unwrap_or(a.target) - og).magnitude()
                );
            }
            _ => {}
        }
    }

    for a in assignments {
        if let Some(idx) = field.player_index(a.id) {
            field.players[idx].ai_defensive_duty = a.duty;
            field.players[idx].ai_defensive_target = a.target;
            field.players[idx].ai_defensive_engage = a.engage;
        }
    }
}

/// Steps C-H of the design doc's spec, for one defending team against
/// one danger point — either a known ball carrier (`carrier_id: Some`,
/// `focal_pos`/`focal_vel` are his own) or, for a genuine loose ball,
/// the ball's own position/velocity with `carrier_id: None`. Pure
/// function of the snapshot — no field mutation happens here, keeping
/// the allocation logic simple and independently reasoned-about.
fn assign_team(
    cands: &[Cand],
    defending_team: u32,
    opponent_team: u32,
    focal_pos: Vector3<f32>,
    focal_vel: Vector3<f32>,
    carrier_id: Option<u32>,
    own_goal: Vector3<f32>,
    opponent_goal: Vector3<f32>,
    defending_side: PlayerSide,
) -> Vec<Assignment> {
    // Forwards are included in the pool (needed so they're eligible for
    // Step C's press-trigger race and Step G's crisis scramble, both
    // added 2026-09-23 — see those steps' own doc comments) but, per
    // this pass's deliberately narrow scope, are excluded from Steps
    // D/E/F (mark/cover/hold-shape) below. Their off-ball defensive job
    // is genuinely different from a defender's or midfielder's — mainly
    // triggering pressure high up the pitch, not holding a defensive
    // line or individually marking — and building that properly (a real
    // screening/passing-lane-blocking target for the non-pressing case)
    // is left for its own follow-up rather than folded in here. A
    // forward who isn't pressing and isn't in a genuine crisis simply
    // gets no duty this tick (`DefensiveDuty::None`) and falls through
    // to his own, separate, not-yet-rewritten off-ball attacking
    // positioning logic — a different domain, deliberately untouched.
    let pool: Vec<Cand> = cands
        .iter()
        .filter(|c| c.team_id == defending_team)
        .filter(|c| {
            matches!(
                c.position_group,
                PlayerFieldPositionGroup::Defender
                    | PlayerFieldPositionGroup::Midfielder
                    | PlayerFieldPositionGroup::Forward
            )
        })
        // Loose-ball case only: whoever's already the established,
        // genuinely-closest chaser (`TakeBall`) is excluded from these
        // assignments so this system and `should_force_takeball` never
        // fight over the same player. Irrelevant for the clear-
        // possession case (nobody on the DEFENDING side is ever in
        // TakeBall then, since that state only fires for a side that
        // doesn't yet have a clear carrier).
        .filter(|c| carrier_id.is_some() || !c.is_taking_ball)
        .copied()
        .collect();
    if pool.is_empty() {
        return Vec::new();
    }

    let mut results: Vec<Assignment> = Vec::new();
    let mut assigned: Vec<u32> = Vec::new();

    // --- Step C: exactly one presser, chosen by real closing time. ---
    // Skipped entirely for a loose ball (`carrier_id: None`) — closing
    // on a ball nobody owns yet is exactly what `should_force_takeball`
    // already does; duplicating it here would only create a second,
    // uncoordinated chaser. Resolved but not pushed yet — engagement
    // (Step H) depends on whether Step E finds real cover, so the
    // final Assignment is pushed once that's known.
    let mut presser: Option<Cand> = None;
    if let Some(carrier_id) = carrier_id {
        // Real defensive-zone doctrine: a defender is only eligible to
        // press when the carrier is genuinely in his zone (see
        // `PRESS_DEFENDER_ZONE_UNITS`'s own doc comment) — UNLESS no
        // midfielder exists in the pool at all, the genuine last-resort
        // case. A closing-time race with no zone awareness was sending
        // defenders 35-44m upfield to press players who were plainly a
        // midfielder's job.
        let ball_dist_to_goal_for_zone = (focal_pos - own_goal).magnitude();
        let any_midfielder_available = pool
            .iter()
            .any(|c| c.position_group == PlayerFieldPositionGroup::Midfielder);
        // Forward-side mirror of the defender zone-gate above (see
        // `PRESS_FORWARD_ZONE_UNITS`'s own doc comment): a forward is
        // only eligible to press when the carrier is genuinely deep in
        // the opponent's own territory, UNLESS no defender or midfielder
        // exists as an alternative — same last-resort structure as the
        // defender gate, just from the opposite end of the pitch.
        let ball_dist_to_opp_goal_for_zone = (focal_pos - opponent_goal).magnitude();
        let any_def_or_mid_available = pool.iter().any(|c| {
            matches!(
                c.position_group,
                PlayerFieldPositionGroup::Defender | PlayerFieldPositionGroup::Midfielder
            )
        });
        let mut best: Option<(Cand, f32, f32)> = None; // (cand, real closing_ticks, effective)
        for c in &pool {
            if c.position_group == PlayerFieldPositionGroup::Defender
                && ball_dist_to_goal_for_zone > PRESS_DEFENDER_ZONE_UNITS
                && any_midfielder_available
            {
                continue;
            }
            if c.position_group == PlayerFieldPositionGroup::Forward
                && ball_dist_to_opp_goal_for_zone > PRESS_FORWARD_ZONE_UNITS
                && any_def_or_mid_available
            {
                continue;
            }
            let dist = (c.position - focal_pos).magnitude();
            let closing_ticks = dist / c.max_speed.max(0.01);
            let sticky = matches!(
                c.prev_duty,
                DefensiveDuty::Press { opponent_id } if opponent_id == carrier_id
            );
            let effective = closing_ticks * if sticky { STICKY_FACTOR } else { 1.0 };
            if best.is_none_or(|(_, _, be)| effective < be) {
                best = Some((*c, closing_ticks, effective));
            }
        }
        if let Some((candidate, closing_ticks, _)) = best {
            if closing_ticks <= MAX_PRESS_CLOSING_TICKS {
                presser = Some(candidate);
                assigned.push(candidate.id);
            }
        }
    }

    // --- Step D: rank non-carrier opponents by danger, assign markers. ---
    let mut threats: Vec<(Cand, f32)> = cands
        .iter()
        .filter(|c| c.team_id == opponent_team && Some(c.id) != carrier_id)
        .filter(|c| (c.position - focal_pos).magnitude() <= THREAT_SCAN_RADIUS_UNITS)
        // Distinct from the scan radius above (distance from the BALL —
        // "could he plausibly receive a pass"): this is distance from
        // OUR OWN GOAL — "is he actually a threat to it right now."
        // See `MARK_MAX_DIST_UNITS`'s own doc comment.
        .filter(|c| (c.position - own_goal).magnitude() <= MARK_MAX_DIST_UNITS)
        .map(|c| {
            // Threat is what the attacker can PRODUCE from where he
            // stands, in the attack's own xG currency (2026-10-03, wide
            // play plan): a corner-flag winger (cross EV ~0.03) no longer
            // outranks a central player at the box edge just because he
            // is close to our goal in a straight line.
            let danger = (crate::r#match::player::strategies::common::players::ops::box_attack::threat_value_at(
                c.position, own_goal,
            ) / THREAT_FULL_VALUE)
                .clamp(0.0, 1.0);
            (*c, danger)
        })
        .filter(|(_, danger)| *danger >= DANGER_FLOOR)
        .collect();
    threats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (opponent, danger) in threats {
        // Forwards excluded from marker selection this pass (see the
        // pool-construction comment above) — individual marking isn't a
        // forward's off-ball job here.
        let marker = pool
            .iter()
            .filter(|c| {
                !assigned.contains(&c.id) && c.position_group != PlayerFieldPositionGroup::Forward
            })
            .min_by(|a, b| {
                let sticky_a = matches!(
                    a.prev_duty,
                    DefensiveDuty::Mark { opponent_id } if opponent_id == opponent.id
                );
                let sticky_b = matches!(
                    b.prev_duty,
                    DefensiveDuty::Mark { opponent_id } if opponent_id == opponent.id
                );
                let da = (a.position - opponent.position).magnitude()
                    * if sticky_a { STICKY_FACTOR } else { 1.0 };
                let db = (b.position - opponent.position).magnitude()
                    * if sticky_b { STICKY_FACTOR } else { 1.0 };
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied();
        let Some(marker) = marker else {
            break;
        };
        let mut tightness = MARK_LOOSE_UNITS - (MARK_LOOSE_UNITS - MARK_TIGHT_UNITS) * danger.min(1.0);
        // Inside our own box a marker keeps the real typical distance, not
        // touch-tight: StatsBomb 360 at a cross, attacker to nearest defender
        // mean 3.05m, median ~2.4m (engine with box defence: 1.8m).
        if box_defence_on()
            && (opponent.position.x - own_goal.x).abs() <= BOX_MARK_DEPTH
            && (opponent.position.y - own_goal.y).abs() <= BOX_MARK_HALF_WIDTH
        {
            tightness = tightness.max(BOX_MARK_MIN_UNITS);
        }
        let to_goal = own_goal - opponent.position;
        let dir = if to_goal.magnitude() > 1.0 {
            to_goal.normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let target = opponent.position + dir * tightness;
        results.push(Assignment {
            id: marker.id,
            duty: DefensiveDuty::Mark {
                opponent_id: opponent.id,
            },
            target,
            engage: false,
        });
        assigned.push(marker.id);
    }

    // --- Step E: cover, from whoever's left — closest to the real
    // stand-off point, not "closest to goal overall". A prior version
    // selected by requiring the candidate to already be closer to goal
    // than the ball, falling back to "deepest overall" otherwise — that
    // primary condition becomes nearly impossible to satisfy exactly
    // when it matters most (the ball already deep in our own box
    // requires standing almost on the goal line to qualify), so the
    // fallback dominated there and picked whoever was deepest with no
    // regard for whether they were anywhere near the ball at all.
    // Verified with raw position data before fixing: cover assignments
    // with the ball in the box were genuinely goal-side of it only
    // 24.1% of the time (vs 49.0% in the defensive third), and the
    // realized cover-to-ball distance in the box bucket (median 43.4m)
    // was *worse* than in calmer zones (14.2m) — backwards from real
    // defending, where cover should tighten as danger increases.
    // Selecting by proximity to the intended target point instead
    // degrades gracefully (always picks whoever's genuinely closest to
    // where cover should stand) rather than falling through to an
    // unrelated "deepest overall" pick.
    let ball_dist_to_goal = (focal_pos - own_goal).magnitude();
    // Cover stands on the carrier's projected path, goal-side of it, not
    // 8 m behind the ball on the ball-to-goal line (a winger cutting in
    // then meets someone).
    let cover_from = focal_pos + focal_vel * COVER_LEAD_TICKS;
    let to_goal = own_goal - cover_from;
    let dir = if to_goal.magnitude() > 1.0 {
        to_goal.normalize()
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let target = cover_from + dir * COVER_STANDOFF_UNITS;
    // Forwards excluded from cover selection this pass, same reasoning
    // as Step D above — covering isn't a forward's off-ball job here.
    let cover = pool
        .iter()
        .filter(|c| {
            !assigned.contains(&c.id) && c.position_group != PlayerFieldPositionGroup::Forward
        })
        .min_by(|a, b| {
            let sticky_a = matches!(a.prev_duty, DefensiveDuty::Cover);
            let sticky_b = matches!(b.prev_duty, DefensiveDuty::Cover);
            let da = (a.position - target).magnitude() * if sticky_a { STICKY_FACTOR } else { 1.0 };
            let db = (b.position - target).magnitude() * if sticky_b { STICKY_FACTOR } else { 1.0 };
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied();
    if let Some(cover) = cover {
        results.push(Assignment {
            id: cover.id,
            duty: DefensiveDuty::Cover,
            target,
            engage: false,
        });
        assigned.push(cover.id);
    }
    let cover_arrived = cover.is_some();

    // --- Step H: engagement, now that cover is known. Only reachable
    // when Step C actually ran (a real carrier existed) — `presser` is
    // never `Some` for a loose ball. ---
    if let (Some(presser), Some(carrier_id)) = (presser, carrier_id) {
        let dist_to_carrier = (presser.position - focal_pos).magnitude();
        let engage = (cover_arrived && dist_to_carrier <= ENGAGE_WITH_COVER_UNITS)
            || dist_to_carrier <= ENGAGE_WITHOUT_COVER_UNITS;
        // Stand at the carrier's anticipated position, goal-side AND (for a
        // wide carrier) on the inside, so the carrier is shown toward the
        // touchline instead of being allowed to cut in.
        let predicted = focal_pos + focal_vel * PRESS_LEAD_TICKS;
        let to_goal = own_goal - predicted;
        let mut dir = if to_goal.magnitude() > 1.0 {
            to_goal.normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let lateral = predicted.y - own_goal.y; // goal centre y == pitch centre y
        let wide = ((lateral.abs() - WIDE_START_UNITS) / WIDE_RAMP_UNITS).clamp(0.0, 1.0);
        if wide > 0.0 {
            let inward = Vector3::new(0.0, -lateral.signum(), 0.0);
            let mixed = dir + inward * (SHOW_OUTSIDE_WEIGHT * wide);
            if mixed.magnitude() > 0.01 {
                dir = mixed.normalize();
            }
        }
        let target = predicted + dir * JOCKEY_DISTANCE_UNITS;
        results.push(Assignment {
            id: presser.id,
            duty: DefensiveDuty::Press {
                opponent_id: carrier_id,
            },
            target,
            engage,
        });
    }

    // --- Step F: everyone else holds shape. ---
    let attack_dir: f32 = match defending_side {
        PlayerSide::Left => 1.0,
        PlayerSide::Right => -1.0,
    };
    // Anticipated carrier position, not the raw instantaneous one — see
    // `LINE_LEAD_TICKS`' own doc comment for why: a purely reactive
    // target measurably lagged a fast-developing attack by 19-36m
    // through the first 1-3 seconds, raw-traced this session.
    let predicted_carrier_pos = focal_pos + focal_vel * LINE_LEAD_TICKS;
    let predicted_ball_dist_to_goal = (predicted_carrier_pos - own_goal).magnitude();
    // Back-line depth is the existing formula, unchanged; the midfield
    // line holds a real, fixed gap ahead of it (see
    // `MIDFIELD_LINE_GAP_UNITS`'s own doc comment) — both driven by the
    // SAME ball-distance signal, so they move together rather than one
    // line reacting and the other sitting still.
    let back_line_depth = (predicted_ball_dist_to_goal * LINE_PUSH_FACTOR)
        .clamp(LINE_MIN_DEPTH_UNITS, LINE_MAX_DEPTH_UNITS);
    let mid_line_depth = back_line_depth + MIDFIELD_LINE_GAP_UNITS;

    // Forwards excluded from ordinary hold-shape too (see the pool-
    // construction comment above) — they don't hold the back/mid line.
    // A forward who isn't pressing gets no duty here; Step G below still
    // pulls him into a genuine box crisis separately.
    let mut hold_shape_ids: Vec<u32> = Vec::new();
    for c in pool
        .iter()
        .filter(|c| !assigned.contains(&c.id) && c.position_group != PlayerFieldPositionGroup::Forward)
    {
        let line_depth = match c.position_group {
            PlayerFieldPositionGroup::Midfielder => mid_line_depth,
            _ => back_line_depth,
        };
        let target_x = own_goal.x + attack_dir * line_depth;
        let target_y = c.start_position.y * (1.0 - WIDTH_PULL_FRACTION)
            + predicted_carrier_pos.y * WIDTH_PULL_FRACTION;
        let mut target = Vector3::new(target_x, target_y, 0.0);
        // Real spacing from the single nearest attacker (any opponent —
        // not just the carrier or the small set of top-ranked marked
        // threats Step D already covers). Each hold-shape player reacts
        // to HIS OWN nearest opponent from HIS OWN current position, so
        // this is anti-clustering by construction: different defenders
        // naturally have different nearest attackers, there's no shared
        // point for multiple of them to converge on.
        if let Some(nearest_opp) = cands
            .iter()
            .filter(|o| o.team_id == opponent_team)
            .min_by(|a, b| {
                let da = (a.position - c.position).magnitude();
                let db = (b.position - c.position).magnitude();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            let gap = (c.position - nearest_opp.position).magnitude();
            if gap < ATTACKER_SPACING_UNITS {
                let push_dir = if gap > 1.0 {
                    (c.position - nearest_opp.position) / gap
                } else {
                    Vector3::new(-attack_dir, 0.0, 0.0)
                };
                target += push_dir * (ATTACKER_SPACING_UNITS - gap);
            }
        }
        results.push(Assignment {
            id: c.id,
            duty: DefensiveDuty::HoldShape,
            target,
            engage: false,
        });
        hold_shape_ids.push(c.id);
    }

    // --- Box defence: hold-shape players cover the box when the ball is
    // wide in the crossing area (see BOX_DEF_SLOTS). ---
    if box_defence_on()
        && (focal_pos.y - own_goal.y).abs() >= BOX_DEF_MIN_LATERAL
        && (focal_pos.x - own_goal.x).abs() <= BOX_DEF_MAX_DEPTH
    {
        let side = (focal_pos.y - own_goal.y).signum();
        let mut free: Vec<u32> = hold_shape_ids.clone();
        for (depth, lat) in BOX_DEF_SLOTS {
            if free.is_empty() {
                break;
            }
            let pt = Vector3::new(own_goal.x + attack_dir * depth, own_goal.y + side * lat, 0.0);
            let (fi, _) = free
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let pos = pool.iter().find(|c| c.id == *id).map(|c| c.position).unwrap_or(pt);
                    (i, (pos - pt).magnitude())
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            let id = free.remove(fi);
            if let Some(a) = results.iter_mut().find(|a| a.id == id) {
                a.target = pt;
            }
        }
    }

    // --- Step G: crisis override. Relabels the hold-shape pool
    // (defenders/midfielders), then also pulls in any forward who isn't
    // already pressing (added 2026-09-23) — a genuine goalmouth crisis
    // is exactly the moment real football stops asking "whose job is
    // this" and gets every available body between the danger and the
    // goal, forwards included. Both groups share the same `rank`
    // counter so the fan-out spacing accounts for everyone converging,
    // not just the hold-shape half. ---
    if ball_dist_to_goal <= CRISIS_RADIUS_UNITS {
        // Stand where a block stands: ~40% of the way to the ball, at most
        // ~10m off the goal line (real back four sit ~9.9m from goal at a
        // shot, StatsBomb 360), not 60% of the way out to the ball.
        let to_ball = focal_pos - own_goal;
        let ball_len = to_ball.magnitude();
        let danger_point = if ball_len > 1.0 {
            own_goal + to_ball / ball_len * (ball_len * 0.4).min(SCRAMBLE_MAX_DEPTH_UNITS)
        } else {
            own_goal
        };
        // Full responder list gathered FIRST so the spread below can be
        // spaced evenly across the whole group in one pass, rather than
        // fanning outward one at a time from a fixed small step (the
        // previous design, which bunched everyone near the centre
        // regardless of how many responders there actually were — see
        // `SCRAMBLE_SPREAD_STEP_UNITS`'s own doc comment). Real crisis
        // defending covers the WIDTH of the danger area (near-post,
        // central, far-post) rather than clustering on one spot.
        let mut scramble_ids: Vec<u32> = hold_shape_ids.clone();
        for c in pool.iter().filter(|c| {
            c.position_group == PlayerFieldPositionGroup::Forward && !assigned.contains(&c.id)
        }) {
            scramble_ids.push(c.id);
        }
        let n = scramble_ids.len();
        for (i, id) in scramble_ids.into_iter().enumerate() {
            let offset_y = if n <= 1 {
                0.0
            } else {
                let half_span =
                    ((n - 1) as f32 * SCRAMBLE_SPREAD_STEP_UNITS / 2.0).min(SCRAMBLE_MAX_HALF_SPAN_UNITS);
                let step = (2.0 * half_span) / (n - 1) as f32;
                -half_span + i as f32 * step
            };
            let target = danger_point + Vector3::new(0.0, offset_y, 0.0);
            if let Some(existing) = results.iter_mut().find(|a| a.id == id) {
                existing.duty = DefensiveDuty::Scramble;
                existing.target = target;
            } else {
                results.push(Assignment {
                    id,
                    duty: DefensiveDuty::Scramble,
                    target,
                    engage: false,
                });
            }
        }
    }

    results
}
