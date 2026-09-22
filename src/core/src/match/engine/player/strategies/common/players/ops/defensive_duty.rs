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
use crate::r#match::player::DefensiveDuty;
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
const DANGER_FLOOR: f32 = 0.15;

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

/// Defensive-line depth bounds and how much the ball's own distance
/// from our goal pulls the line forward. Real doctrine: push up when
/// the ball is contained and far away, sit deep when it's dangerously
/// advanced — bounded so the line never sits inside our own box nor
/// pushes all the way to halfway regardless of ball position.
const LINE_PUSH_FACTOR: f32 = 0.55;
const LINE_MIN_DEPTH_UNITS: f32 = 96.0; // 12.0m from goal
const LINE_MAX_DEPTH_UNITS: f32 = 320.0; // 40.0m from goal

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

/// Lateral fan-out spacing between scramble responders, so a crisis
/// converges bodies on the danger without literally stacking them.
const SCRAMBLE_FAN_SPACING_UNITS: f32 = 24.0; // 3.0m

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
/// `ai_defensive_duty` to `None` each call, then assigns real duties
/// to the out-of-possession team's defenders/midfielders when an
/// opponent clearly holds the ball. A loose ball, or the team itself
/// holding it, leaves this system fully inactive for that team this
/// tick (loose-ball recovery and on-ball decisions are separate,
/// not-yet-rewritten domains — untouched by design).
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

    let Some(carrier_id) = field.ball.current_owner else {
        reset_all(field);
        return;
    };

    // Captures each player's PREVIOUS tick's duty (`prev_duty`) before
    // any reset happens this tick — the only source of continuity
    // Steps C/D/E's stickiness bias reads from.
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
        })
        .collect();

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
    let own_goal = match defending_side {
        PlayerSide::Left => context.goal_positions.left,
        PlayerSide::Right => context.goal_positions.right,
    };

    let assignments = assign_team(&cands, defending_team, carrier, own_goal, defending_side);
    reset_all(field);

    #[cfg(feature = "match-logs")]
    for a in &assignments {
        match a.duty {
            DefensiveDuty::Mark { opponent_id } => {
                if let Some(marker) = cands.iter().find(|c| c.id == a.id) {
                    let opp_dist_goal = cands
                        .iter()
                        .find(|c| c.id == opponent_id)
                        .map(|o| (o.position - own_goal).magnitude())
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
                let dist = (a.target - carrier.position).magnitude();
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
                        "DEFDUTY_PRESS t={} presser={} presser_group={:?} presser_dist_start={:.1} carrier={} own_goal_dist={:.1}",
                        context.total_match_time,
                        a.id,
                        presser.position_group,
                        (presser.position - presser.start_position).magnitude(),
                        opponent_id,
                        (carrier.position - own_goal).magnitude()
                    );
                }
            }
            DefensiveDuty::Scramble => {
                crate::match_log_info!(
                    "DEFDUTY_SCRAMBLE t={} p={} ball_dist_goal={:.1}",
                    context.total_match_time,
                    a.id,
                    (carrier.position - own_goal).magnitude()
                );
            }
            // Diagnostic, kept permanently (feature-gated, zero cost
            // without match-logs): HoldShape was never logged at all
            // before Pavel's "defenders don't space/drop" report,
            // 2026-09-22.
            DefensiveDuty::HoldShape => {
                crate::match_log_info!(
                    "DEFDUTY_HOLDSHAPE t={} p={} ball_dist_goal={:.1} target_dist_goal={:.1} player_dist_goal={:.1}",
                    context.total_match_time,
                    a.id,
                    (carrier.position - own_goal).magnitude(),
                    (a.target - own_goal).magnitude(),
                    (cands.iter().find(|c| c.id == a.id).map(|c| c.position).unwrap_or(a.target) - own_goal).magnitude()
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
/// one known ball carrier. Pure function of the snapshot — no field
/// mutation happens here, keeping the allocation logic simple and
/// independently reasoned-about.
fn assign_team(
    cands: &[Cand],
    defending_team: u32,
    carrier: Cand,
    own_goal: Vector3<f32>,
    defending_side: PlayerSide,
) -> Vec<Assignment> {
    let pool: Vec<Cand> = cands
        .iter()
        .filter(|c| c.team_id == defending_team)
        .filter(|c| {
            matches!(
                c.position_group,
                PlayerFieldPositionGroup::Defender | PlayerFieldPositionGroup::Midfielder
            )
        })
        .copied()
        .collect();
    if pool.is_empty() {
        return Vec::new();
    }

    let mut results: Vec<Assignment> = Vec::new();
    let mut assigned: Vec<u32> = Vec::new();

    // --- Step C: exactly one presser, chosen by real closing time. ---
    // Resolved but not pushed yet — engagement (Step H) depends on
    // whether Step E finds real cover, so the final Assignment is
    // pushed once that's known.
    let mut presser: Option<Cand> = None;
    {
        // Real defensive-zone doctrine: a defender is only eligible to
        // press when the carrier is genuinely in his zone (see
        // `PRESS_DEFENDER_ZONE_UNITS`'s own doc comment) — UNLESS no
        // midfielder exists in the pool at all, the genuine last-resort
        // case. A closing-time race with no zone awareness was sending
        // defenders 35-44m upfield to press players who were plainly a
        // midfielder's job.
        let ball_dist_to_goal_for_zone = (carrier.position - own_goal).magnitude();
        let any_midfielder_available = pool
            .iter()
            .any(|c| c.position_group == PlayerFieldPositionGroup::Midfielder);
        let mut best: Option<(Cand, f32, f32)> = None; // (cand, real closing_ticks, effective)
        for c in &pool {
            if c.position_group == PlayerFieldPositionGroup::Defender
                && ball_dist_to_goal_for_zone > PRESS_DEFENDER_ZONE_UNITS
                && any_midfielder_available
            {
                continue;
            }
            let dist = (c.position - carrier.position).magnitude();
            let closing_ticks = dist / c.max_speed.max(0.01);
            let sticky = matches!(
                c.prev_duty,
                DefensiveDuty::Press { opponent_id } if opponent_id == carrier.id
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
    let opponent_team = carrier.team_id;
    let mut threats: Vec<(Cand, f32)> = cands
        .iter()
        .filter(|c| c.team_id == opponent_team && c.id != carrier.id)
        .filter(|c| (c.position - carrier.position).magnitude() <= THREAT_SCAN_RADIUS_UNITS)
        // Distinct from the scan radius above (distance from the BALL —
        // "could he plausibly receive a pass"): this is distance from
        // OUR OWN GOAL — "is he actually a threat to it right now."
        // See `MARK_MAX_DIST_UNITS`'s own doc comment.
        .filter(|c| (c.position - own_goal).magnitude() <= MARK_MAX_DIST_UNITS)
        .map(|c| {
            let dist_to_goal = (c.position - own_goal).magnitude();
            let proximity = 1.0 - (dist_to_goal / THREAT_SCAN_RADIUS_UNITS).clamp(0.0, 1.0);
            let nearby_defenders = pool
                .iter()
                .filter(|d| (d.position - c.position).magnitude() <= 100.0)
                .count();
            let openness = 1.0 - (nearby_defenders as f32 / 3.0).clamp(0.0, 1.0);
            let danger = (0.7 * proximity + 0.3 * openness).max(proximity * 0.7);
            (*c, danger)
        })
        .filter(|(_, danger)| *danger >= DANGER_FLOOR)
        .collect();
    threats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (opponent, danger) in threats {
        let marker = pool
            .iter()
            .filter(|c| !assigned.contains(&c.id))
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
        let tightness = MARK_LOOSE_UNITS - (MARK_LOOSE_UNITS - MARK_TIGHT_UNITS) * danger.min(1.0);
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
    let ball_dist_to_goal = (carrier.position - own_goal).magnitude();
    let to_goal = own_goal - carrier.position;
    let dir = if to_goal.magnitude() > 1.0 {
        to_goal.normalize()
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let target = carrier.position + dir * COVER_STANDOFF_UNITS;
    let cover = pool
        .iter()
        .filter(|c| !assigned.contains(&c.id))
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

    // --- Step H: engagement, now that cover is known. ---
    if let Some(presser) = presser {
        let dist_to_carrier = (presser.position - carrier.position).magnitude();
        let engage = (cover_arrived && dist_to_carrier <= ENGAGE_WITH_COVER_UNITS)
            || dist_to_carrier <= ENGAGE_WITHOUT_COVER_UNITS;
        let to_goal = own_goal - carrier.position;
        let dir = if to_goal.magnitude() > 1.0 {
            to_goal.normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let target = carrier.position + dir * JOCKEY_DISTANCE_UNITS;
        results.push(Assignment {
            id: presser.id,
            duty: DefensiveDuty::Press {
                opponent_id: carrier.id,
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
    let predicted_carrier_pos = carrier.position + carrier.velocity * LINE_LEAD_TICKS;
    let predicted_ball_dist_to_goal = (predicted_carrier_pos - own_goal).magnitude();
    let line_depth = (predicted_ball_dist_to_goal * LINE_PUSH_FACTOR)
        .clamp(LINE_MIN_DEPTH_UNITS, LINE_MAX_DEPTH_UNITS);
    let target_x = own_goal.x + attack_dir * line_depth;

    let mut hold_shape_ids: Vec<u32> = Vec::new();
    for c in pool.iter().filter(|c| !assigned.contains(&c.id)) {
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

    // --- Step G: crisis override, hold-shape pool only. ---
    if ball_dist_to_goal <= CRISIS_RADIUS_UNITS {
        let danger_point = own_goal + (carrier.position - own_goal) * 0.6;
        let mut rank = 0i32;
        for a in results.iter_mut() {
            if !hold_shape_ids.contains(&a.id) {
                continue;
            }
            a.duty = DefensiveDuty::Scramble;
            let fan_side = if rank % 2 == 0 { 1.0 } else { -1.0 };
            let fan_step = ((rank + 1) / 2) as f32;
            let offset = Vector3::new(0.0, fan_side * fan_step * SCRAMBLE_FAN_SPACING_UNITS, 0.0);
            a.target = danger_point + offset;
            rank += 1;
        }
    }

    results
}
