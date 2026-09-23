//! Off-ball attacking positioning/support — designed from null.
//!
//! `docs/decision-from-null.md` (domain 3) /
//! `docs/off-ball-attacking-positioning-from-null.md` (this domain's own
//! football reasoning, spec, and verification log — read that first).
//!
//! A single, team-wide, per-tick allocation of attacking duty
//! (rest-defense / width / advance / penetrate) across every outfield
//! teammate of the current ball carrier — the carrier's own decision
//! stays domain 2's (`on_ball_decision`). Deliberately calls no
//! pre-existing decision function (no `ForwardCreatingSpaceState`,
//! `MidfielderRunningState`'s off-ball branch, `DefenderPushingUpState`,
//! or any sibling) — every target formula here is derived fresh from
//! the football reasoning in the design doc, using only raw
//! `MatchField`/`MatchContext` data plus genuine measurements
//! (`max_speed_with_condition`, goal positions, Euclidean geometry).
//!
//! Process check (`docs/decision-from-null.md`'s required self-audit):
//! this module's call graph touches only `MatchField`/`MatchContext`/
//! `MatchPlayer` struct fields, `MatchContext::penalty_area` (real box
//! geometry, already whitelisted as a measurement), and
//! `nalgebra::Vector3` geometry — none of it encodes a judgment. Zero
//! pre-existing decision-function dependencies. `MatchContext::
//! attack_reference_depth` (read/written by this module) is likewise a
//! measurement in the sense that matters here — a computed fact about
//! how far this side's attack has recently reached, not itself a
//! ranking or a choice; the actual judgment (how that fact turns into a
//! target position) stays entirely in this file's own Step C/D
//! formulas.

use crate::PlayerFieldPositionGroup;
use crate::club::player::ability::position::PlayerPositionType;
use crate::r#match::player::AttackDuty;
use crate::r#match::{MatchContext, MatchField, PlayerSide};
use nalgebra::Vector3;

/// Number of the deepest-positioned defenders held back for balance,
/// regardless of ball advancement — real rest-defense doctrine.
/// Reasoned estimate (category d): a small, fixed floor rather than a
/// fraction that could round to zero.
const REST_DEFENSE_COUNT: usize = 2;

/// Width target depth bounds (distance from own goal) and how strongly
/// it tracks the ball's own advancement. A wide player should already
/// hold a fairly advanced base position even during build-up (real
/// wingers don't retreat deep), and reach genuinely high, close to the
/// opponent's box, once the team is attacking. Reasoned estimate
/// (category d, no sourced exact figure) — ~50m min, ~92.5m max on a
/// 105m pitch.
const WIDTH_MIN_DEPTH: f32 = 400.0;
const WIDTH_MAX_DEPTH: f32 = 740.0;
const WIDTH_PUSH_FACTOR: f32 = 0.85;
/// How close to the touchline a width target sits.
const WIDTH_MARGIN: f32 = 24.0;

/// Advancing-line depth bounds for central players, the direct
/// attacking-phase counterpart of domain 1's own defensive-line
/// mechanism (`LINE_PUSH_FACTOR`/`LINE_MIN_DEPTH_UNITS`/
/// `LINE_MAX_DEPTH_UNITS` in `defensive_duty.rs`) — same idea, much
/// higher cap, since this governs players supporting an attack, not
/// holding a defensive shape. Reasoned estimate (category d) — ~37.5m
/// min, ~77.5m max: genuinely into the opponent's half, not stopping
/// just past halfway (the specific symptom Pavel reported).
const ADVANCE_MIN_DEPTH: f32 = 300.0;
const ADVANCE_MAX_DEPTH: f32 = 620.0;
const ADVANCE_PUSH_FACTOR: f32 = 0.80;

/// How much a central player's lateral position stays anchored to his
/// own formation slot vs. drifting — anti-clustering by construction
/// (different players have different slots).
const ADVANCE_WIDTH_PULL: f32 = 0.25;

/// Beyond this distance from a target, the nearest opponent doesn't
/// meaningfully contest it — genuine open space worth pushing further
/// into.
const PENETRATION_OPEN_THRESHOLD: f32 = 90.0;
/// How far a penetration bias pushes an already-advanced, genuinely
/// open target further toward the opponent's goal.
const PENETRATION_PUSH: f32 = 40.0;

/// How quickly `MatchContext::attack_reference_depth` forgets a recent
/// advance once the ball stops matching it — applied once per tick, so
/// `ATTACK_REF_DECAY_PER_TICK.powf(ticks)` gives the fraction remaining
/// after that many ticks (100 ticks/s, this project's established tick
/// rate). Reasoned estimate (category d, no sourced dataset for "how
/// long a real team holds an advanced shape through a recycling pass")
/// — tuned for a roughly 8-second half-life: a single backward lay-off
/// and a brief hold barely moves the reference (a few percent over 1-3
/// real seconds), while a genuinely sustained deep rebuild (10+
/// seconds) gradually pulls it back down to match reality. A
/// verification target for the raw-trace pass, not treated as final.
const ATTACK_REF_DECAY_PER_TICK: f32 = 0.999134;

/// Real box-crowding scenario gate (Pavel's direct report, 2026-09-23:
/// "when a forward has a ball in the box, defenders and attackers are
/// all grouped somewhere around penalty spot — nobody is covering
/// areas to the left or to the right"). The ball must be genuinely
/// inside, or just outside, the opponent's box before this system
/// starts redirecting overflow players to open flank pockets — a real
/// six-yard-box-ish margin past the box's own real depth (measured via
/// `MatchContext::penalty_area`, pure geometry) so a player converging
/// on an edge-of-box cutback is covered too, not just literal box
/// occupants.
const BOX_CROWD_MARGIN_UNITS: f32 = 44.0; // ~5.5m past the box's own real depth

/// How wide a central band around the box's own centre line counts as
/// "the same tiny zone" for the crowding check — roughly a penalty-
/// spot-sized cluster, not the whole box. Expressed as a fraction of
/// the box's real half-width so it scales correctly with field size.
const BOX_CENTRAL_BAND_FRACTION: f32 = 0.35;

/// How many Advance-duty players may legitimately occupy that central
/// band before any further ones get redirected to an open flank pocket
/// instead — a real box moment plausibly has two central runners (e.g.
/// a striker plus an arriving midfielder), not zero; a third and
/// beyond piling on the same small area, while genuine flank space sits
/// empty, is the reported bug. Reasoned estimate (category d).
const BOX_CENTRAL_OCCUPANCY_CAP: usize = 2;

/// How far toward the box's own edge a redirected "expose the open
/// area" pocket sits, as a fraction of the box's real half-width —
/// close enough to the flank to genuinely stretch the defense, not so
/// close it stands outside the box chasing a phantom lane.
const BOX_POCKET_Y_FRACTION: f32 = 0.75;

/// How much a pocket's effective openness is discounted per teammate
/// already redirected there this tick — found necessary by raw trace,
/// 2026-09-23: without it, every overflow player independently scores
/// both pockets from the SAME opponent geometry and picks the identical
/// "more open" one, so a third/fourth overflow attacker all collapsed
/// onto one single pocket point instead of spreading across both. A
/// genuinely lopsided real gap (one flank truly empty, the other
/// covered) still wins outright — this only breaks a near-tie toward
/// the less-occupied side. Sized relative to `PENETRATION_OPEN_
/// THRESHOLD` (90u) so one existing occupant meaningfully tips a close
/// call. Reasoned estimate (category d).
const POCKET_OCCUPANCY_PENALTY: f32 = 60.0;

#[derive(Clone, Copy)]
struct Cand {
    id: u32,
    team_id: u32,
    position: Vector3<f32>,
    position_type: PlayerPositionType,
    position_group: PlayerFieldPositionGroup,
    start_position: Vector3<f32>,
    side: Option<PlayerSide>,
    is_directed: bool,
}

struct Assignment {
    id: u32,
    duty: AttackDuty,
    target: Vector3<f32>,
    /// True only for an Advance-duty player redirected to an open box
    /// flank pocket by the box-crowding rule (Step D). Diagnostic field
    /// — lets the match-logs trace directly confirm the redirect fires,
    /// rather than inferring it from `target.y` alone.
    redirected: bool,
}

fn is_wide(pt: PlayerPositionType) -> bool {
    matches!(
        pt,
        PlayerPositionType::DefenderLeft
            | PlayerPositionType::DefenderRight
            | PlayerPositionType::MidfielderLeft
            | PlayerPositionType::MidfielderRight
            | PlayerPositionType::AttackingMidfielderLeft
            | PlayerPositionType::AttackingMidfielderRight
            | PlayerPositionType::WingbackLeft
            | PlayerPositionType::WingbackRight
            | PlayerPositionType::ForwardLeft
            | PlayerPositionType::ForwardRight
    )
}

fn is_left_flank(pt: PlayerPositionType) -> bool {
    matches!(
        pt,
        PlayerPositionType::DefenderLeft
            | PlayerPositionType::MidfielderLeft
            | PlayerPositionType::AttackingMidfielderLeft
            | PlayerPositionType::WingbackLeft
            | PlayerPositionType::ForwardLeft
    )
}

/// Entry point — called once per full tick from `game_tick_inner`,
/// right after `assign_defensive_duties`. Resets every player's
/// `ai_attack_duty` to `None` each call, then, when a team clearly
/// holds the ball, assigns real duties to that team's non-carrier
/// outfielders. A loose ball leaves this system fully inactive (a
/// deliberate, deferred scope boundary — see the design doc).
pub fn assign_attacking_support(field: &mut MatchField, context: &mut MatchContext) {
    for p in field.players.iter_mut() {
        p.ai_attack_duty = AttackDuty::None;
    }

    // The genuine current owner (domain 2 governs him, so he's excluded
    // from this pool below) OR, during an ordinary directed pass in
    // flight, the passer -- used only to identify the team/side this
    // pass belongs to. Real-data bug found 2026-09-23 (Pavel's direct
    // report: midfielders advance the ball then stand still while it's
    // in the air to a forward, leaving him isolated): `current_owner`
    // goes `None` for the ball's entire flight time on every single
    // pass, and this whole system previously required it to be `Some`,
    // going fully dark during that window -- exactly the same class of
    // gap domain 1's own Case 2a fix (`defensive_duty.rs`) already
    // closed once this session, just not carried over here at the time.
    // The passer himself is deliberately NOT excluded from the pool
    // once the ball has left his feet (`current_owner` is already
    // `None`) -- he's genuinely off-ball now and should be free to
    // continue a real run/support movement, matching Pavel's own
    // explicit ask ("run without the ball... to offer themselves for a
    // pass").
    let reference_id = field.ball.current_owner.or(field.ball.pending_pass_passer);
    let Some(reference_id) = reference_id else {
        return;
    };

    let cands: Vec<Cand> = field
        .players
        .iter()
        .filter(|p| !p.is_sent_off)
        .map(|p| Cand {
            id: p.id,
            team_id: p.team_id,
            position: p.position,
            position_type: p.tactical_position.current_position,
            position_group: p.tactical_position.current_position.position_group(),
            start_position: p.start_position,
            side: p.side,
            is_directed: p.behavioral_directive.is_some()
                || p.link_target.is_some()
                || p.supply_target.is_some(),
        })
        .collect();

    let Some(carrier) = cands.iter().find(|c| c.id == reference_id).copied() else {
        return;
    };
    let Some(carrier_side) = carrier.side else {
        return;
    };

    let goal_for_side = |side: PlayerSide| match side {
        PlayerSide::Left => context.goal_positions.left,
        PlayerSide::Right => context.goal_positions.right,
    };
    let opposite_side = |side: PlayerSide| match side {
        PlayerSide::Left => PlayerSide::Right,
        PlayerSide::Right => PlayerSide::Left,
    };
    let own_goal = goal_for_side(carrier_side);
    let opponent_goal = goal_for_side(opposite_side(carrier_side));
    let attack_dir: f32 = match carrier_side {
        PlayerSide::Left => 1.0,
        PlayerSide::Right => -1.0,
    };
    // Tried a domain-1-style ball-velocity lead term here (2026-09-23,
    // mirroring `LINE_LEAD_TICKS`) and REVERTED it after raw measurement
    // showed it made Width's own target-arrival gap WORSE, not better
    // (144.8u mean -> 170.5u). Root cause, confirmed by reasoning through
    // the actual numbers rather than re-guessing: the ball's own
    // velocity (up to ~3.2 u/tick during a genuine pass/shot) is 5-8x a
    // player's own real max speed (~0.4-0.6 u/tick) -- domain 1's
    // `LINE_LEAD_TICKS` extrapolates a PLAYER's bounded velocity, a very
    // different, much smaller quantity than the ball's own burst speed;
    // applying the identical lead-tick count to the ball massively
    // overshoots into a target position the ball itself won't sustain
    // (a pass is typically caught/intercepted well before it travels
    // that far at that speed). A real, useful negative result, not
    // chased further this pass -- see the design doc's own status log.
    let ball_dist_to_own_goal = (field.ball.position - own_goal).magnitude();

    // Ratcheted "how far forward has this side's attack genuinely
    // established itself" reference — see `MatchContext::
    // attack_reference_depth`'s own doc comment for the full reasoning.
    // Decays every tick this system runs (both sides, so a side that's
    // been out of possession for a while doesn't carry a stale high
    // value into its next spell of possession), then ratchets up to at
    // least this tick's raw ball depth for whichever side currently
    // holds it. Read in place of `ball_dist_to_own_goal` by the Width/
    // Advance depth formulas below.
    for v in context.attack_reference_depth.iter_mut() {
        *v *= ATTACK_REF_DECAY_PER_TICK;
    }
    let side_idx = match carrier_side {
        PlayerSide::Left => 0,
        PlayerSide::Right => 1,
    };
    context.attack_reference_depth[side_idx] =
        context.attack_reference_depth[side_idx].max(ball_dist_to_own_goal);
    let attack_ref_depth = context.attack_reference_depth[side_idx];

    let opponents: Vec<Vector3<f32>> = cands
        .iter()
        .filter(|c| c.team_id != carrier.team_id)
        .map(|c| c.position)
        .collect();
    let nearest_opp_dist = |p: Vector3<f32>| -> f32 {
        opponents
            .iter()
            .map(|o| (o - p).magnitude())
            .fold(f32::MAX, f32::min)
    };

    // The reference player (the genuine owner, or the passer during an
    // ordinary pass in flight) stays excluded from the assignable pool
    // -- narrowed 2026-09-23 after a real, severe performance regression
    // was found and isolated: an earlier version of this fix instead
    // included a mid-flight passer in the pool (reasoning that he's
    // genuinely off-ball once the ball leaves his feet and could
    // legitimately continue a run). A clean isolated A/B (git-stash
    // isolated, 15+15 matches) showed 0/15 slow-match outliers on the
    // build WITHOUT that inclusion vs. 4 outliers (660-953s instead of
    // ~2s) across 35 matches WITH it. The exact mechanism wasn't fully
    // root-caused given time cost -- reverted to the safer, already-
    // proven-clean exclusion rather than ship a severe, poorly-
    // understood hang risk. The actual reported problem (teammates OTHER
    // than the passer going dark during his pass's flight) is still
    // fixed by `reference_id` above; only "does the passer himself also
    // get a duty mid-flight" is deferred, a real, honestly-scoped
    // narrower fix than originally attempted.
    let pool: Vec<Cand> = cands
        .iter()
        .filter(|c| {
            c.team_id == carrier.team_id
                && c.id != carrier.id
                && c.position_group != PlayerFieldPositionGroup::Goalkeeper
                && !c.is_directed
        })
        .copied()
        .collect();

    let mut results: Vec<Assignment> = Vec::new();
    let mut assigned: Vec<u32> = Vec::new();

    // --- Step A.5: the designated pass RECEIVER, if any, gets a
    // dedicated target -- the ball's own real flight destination
    // (`Ball.pending_pass_target`), not whatever generic Width/Advance
    // team-line formula would otherwise apply to him. Runs BEFORE every
    // other step and excludes him from all of them via `assigned`.
    //
    // Real bug (Pavel, 2026-09-23, unhedged -- ground truth per this
    // project's own Realism Diagnosis Framework, not a hypothesis to
    // second-guess): "#10 passes to #6 but #6 retracts so the pass
    // lands short... the opponent came into the space between them and
    // took the ball." Root cause: Steps C/D's target formulas have no
    // concept of "a pass is already live and aimed at me" -- they
    // recompute a generic team-shape position every tick regardless,
    // which can sit BEHIND a receiver who had drifted forward to offer
    // himself, dragging him backward mid-flight and shortening the
    // pass into the exact gap an opponent then steps into. A real
    // receiver's job is to adjust to the ball coming to him, not
    // retreat because an unrelated formula recomputed his line
    // position this tick.
    if let Some(receiver_id) = field.ball.pass_target_player_id {
        if let Some(receiver) = pool.iter().find(|c| c.id == receiver_id) {
            let target = field.ball.pending_pass_target.unwrap_or(receiver.position);
            results.push(Assignment {
                id: receiver.id,
                duty: AttackDuty::Receive,
                target,
                redirected: false,
            });
            assigned.push(receiver.id);
        }
    }

    // --- Step B: rest-defense floor -- the N deepest-positioned
    // defenders hold their own anchor, real football's deliberate
    // balance so a turnover doesn't instantly expose an empty defense.
    let mut defenders: Vec<Cand> = pool
        .iter()
        .filter(|c| c.position_group == PlayerFieldPositionGroup::Defender)
        .copied()
        .collect();
    defenders.sort_by(|a, b| {
        let da = (a.position - own_goal).magnitude();
        let db = (b.position - own_goal).magnitude();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    for d in defenders.iter().take(REST_DEFENSE_COUNT) {
        results.push(Assignment {
            id: d.id,
            duty: AttackDuty::RestDefense,
            target: d.start_position,
            redirected: false,
        });
        assigned.push(d.id);
    }

    // --- Step C: width -- any remaining wide-slotted player holds the
    // flank, tracking the ball's own advancement.
    let target_x_from_depth = |depth: f32| own_goal.x + attack_dir * depth;
    let wide_cands: Vec<Cand> = pool
        .iter()
        .filter(|c| !assigned.contains(&c.id) && is_wide(c.position_type))
        .copied()
        .collect();
    // Real football staggers multiple wide players on the SAME flank by
    // depth (a fullback overlaps/underlaps BEHIND the winger, doesn't
    // stand on top of him) -- without this, e.g. DefenderLeft and
    // ForwardLeft both wide-slotted collapsed onto the literal identical
    // target point (verified via raw trace, 2026-09-23: mean distance
    // from realized position to target was 167.8u for Width duty, far
    // worse than RestDefense/Advance's 37-43u, traced to exactly this
    // collision), the same anti-clustering violation domain 1's own
    // 2026-09-12 rollback exists to prevent.
    let base_depth = (attack_ref_depth * WIDTH_PUSH_FACTOR).clamp(WIDTH_MIN_DEPTH, WIDTH_MAX_DEPTH);
    for c in wide_cands.iter() {
        let (offset, min_d, max_d) = match c.position_group {
            PlayerFieldPositionGroup::Forward => (0.0, WIDTH_MIN_DEPTH, WIDTH_MAX_DEPTH),
            PlayerFieldPositionGroup::Midfielder => {
                (-90.0, WIDTH_MIN_DEPTH - 60.0, WIDTH_MAX_DEPTH - 90.0)
            }
            _ => (-180.0, WIDTH_MIN_DEPTH - 120.0, WIDTH_MAX_DEPTH - 180.0),
        };
        let depth = (base_depth + offset).clamp(min_d, max_d);
        let field_height = context.field_size.height as f32;
        let y = if is_left_flank(c.position_type) {
            WIDTH_MARGIN
        } else {
            field_height - WIDTH_MARGIN
        };
        let mut target = Vector3::new(target_x_from_depth(depth), y, 0.0);
        if nearest_opp_dist(target) > PENETRATION_OPEN_THRESHOLD {
            let to_goal = opponent_goal - target;
            let dist = to_goal.magnitude();
            if dist > 1.0 {
                target += (to_goal / dist) * PENETRATION_PUSH;
            }
        }
        results.push(Assignment {
            id: c.id,
            duty: AttackDuty::Width,
            target,
            redirected: false,
        });
        assigned.push(c.id);
    }

    // --- Step D: advancing line -- every remaining central player
    // (midfielders, and defenders beyond the rest-defense floor) tracks
    // the ball's depth as a coordinated line, the direct attacking-phase
    // counterpart of domain 1's own defensive-line formula.
    let depth = (attack_ref_depth * ADVANCE_PUSH_FACTOR).clamp(ADVANCE_MIN_DEPTH, ADVANCE_MAX_DEPTH);
    let target_x = target_x_from_depth(depth);
    // Real box geometry (a pure measurement, `MatchContext::
    // penalty_area`) for the box-crowding redirect below (Pavel's
    // direct report, 2026-09-23: "defenders and attackers are all
    // grouped somewhere around penalty spot -- nobody is covering areas
    // to the left or to the right"). `penalty_area`'s own `is_home_team`
    // flag really means "is this the box on the LEFT side of the pitch"
    // -- true for the box we're attacking whenever the opponent sits on
    // the Left.
    let attacking_box = context.penalty_area(opposite_side(carrier_side) == PlayerSide::Left);
    let box_half_width = (attacking_box.max.y - attacking_box.min.y) / 2.0;
    let box_center_y = (attacking_box.max.y + attacking_box.min.y) / 2.0;
    let box_depth = (attacking_box.max.x - attacking_box.min.x).abs();
    let box_crowd_gate = box_depth + BOX_CROWD_MARGIN_UNITS;
    let box_crowding_active = (field.ball.position - opponent_goal).magnitude() <= box_crowd_gate;
    let box_central_band = box_half_width * BOX_CENTRAL_BAND_FRACTION;
    let box_pocket_offset = box_half_width * BOX_POCKET_Y_FRACTION;
    let mut central_box_count = 0usize;
    // Per-pocket occupancy this tick -- see the redirect comment below
    // for why this exists (found via raw trace, 2026-09-23: without it,
    // every overflow player independently picks the SAME "more open"
    // pocket, since openness only depends on opponent positions, which
    // look identical to every teammate asking -- trading one central
    // crowd for a lopsided single-pocket crowd, not the genuine spread
    // Pavel asked for).
    let mut pocket_occupancy = [0i32, 0i32]; // [left, right]
    for c in pool.iter().filter(|c| !assigned.contains(&c.id)) {
        let y = c.start_position.y * (1.0 - ADVANCE_WIDTH_PULL) + field.ball.position.y * ADVANCE_WIDTH_PULL;
        let mut target = Vector3::new(target_x, y, 0.0);
        let mut redirected_to_pocket = false;
        // Box-crowding redirect: once the central band already has its
        // fair share of Advance-duty players, further ones expose an
        // open flank pocket instead of piling onto the same small zone
        // -- "attackers should expose those empty areas... going to
        // those open spaces in the box" (Pavel, verbatim). Each pocket's
        // effective openness is discounted by how many teammates this
        // tick have already been sent there (`POCKET_OCCUPANCY_PENALTY`)
        // -- a genuinely more-open pocket still wins outright, but a
        // second overflow player no longer piles onto the exact same
        // spot as the first purely because the opponent geometry looks
        // identical to both of them.
        if box_crowding_active && (target.y - box_center_y).abs() <= box_central_band {
            if central_box_count < BOX_CENTRAL_OCCUPANCY_CAP {
                central_box_count += 1;
            } else {
                let left_pocket = Vector3::new(target.x, box_center_y - box_pocket_offset, 0.0);
                let right_pocket = Vector3::new(target.x, box_center_y + box_pocket_offset, 0.0);
                let left_open = nearest_opp_dist(left_pocket)
                    - pocket_occupancy[0] as f32 * POCKET_OCCUPANCY_PENALTY;
                let right_open = nearest_opp_dist(right_pocket)
                    - pocket_occupancy[1] as f32 * POCKET_OCCUPANCY_PENALTY;
                if left_open >= right_open {
                    target.y = left_pocket.y;
                    pocket_occupancy[0] += 1;
                } else {
                    target.y = right_pocket.y;
                    pocket_occupancy[1] += 1;
                }
                redirected_to_pocket = true;
            }
        }
        // Skipped for a pocket-redirected target -- pushing back toward
        // the fixed goal-centre point would undo the redirect that just
        // deliberately moved this player off it.
        if !redirected_to_pocket && nearest_opp_dist(target) > PENETRATION_OPEN_THRESHOLD {
            let to_goal = opponent_goal - target;
            let dist = to_goal.magnitude();
            if dist > 1.0 {
                target += (to_goal / dist) * PENETRATION_PUSH;
            }
        }
        results.push(Assignment {
            id: c.id,
            duty: AttackDuty::Advance,
            target,
            redirected: redirected_to_pocket,
        });
    }

    // Diagnostic (temporary for this investigation, kept feature-gated
    // at zero cost otherwise): ball_dist_to_own_goal vs. the ratcheted
    // attack_ref_depth, once per tick — directly proves the ratchet
    // holds through a backward pass instead of collapsing with it.
    #[cfg(feature = "match-logs")]
    crate::match_log_info!(
        "ATTACKREF t={} side={:?} ball_dist={:.1} ref_depth={:.1}",
        context.total_match_time,
        carrier_side,
        ball_dist_to_own_goal,
        attack_ref_depth
    );

    #[cfg(feature = "match-logs")]
    for a in &results {
        if let Some(c) = pool.iter().find(|c| c.id == a.id) {
            crate::match_log_info!(
                "ATTACKSUPPORT t={} id={} pt={:?} duty={:?} pos_x={:.1} pos_y={:.1} target_x={:.1} target_y={:.1} redirected={}",
                context.total_match_time,
                a.id,
                c.position_type,
                a.duty,
                c.position.x,
                c.position.y,
                a.target.x,
                a.target.y,
                a.redirected
            );
        }
    }

    for a in results {
        if let Some(idx) = field.player_index(a.id) {
            field.players[idx].ai_attack_duty = a.duty;
            field.players[idx].ai_attack_target = a.target;
        }
    }
}
