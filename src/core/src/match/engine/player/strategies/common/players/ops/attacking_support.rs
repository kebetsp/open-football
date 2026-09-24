//! Off-ball attacking positioning/support — designed from null.
//!
//! `docs/decision-from-null.md` (domain 3) /
//! `docs/off-ball-attacking-positioning-from-null.md` (this domain's own
//! original football reasoning, spec, and first two implementation
//! rounds) / `docs/off-ball-attacking-positioning-phase-redesign.md`
//! (the possession-phase redesign this file now implements — read that
//! one first for WHY the earlier ball-distance-ratchet design was
//! replaced).
//!
//! **What changed, 2026-09-24, and why.** The previous version of this
//! module drove every off-ball teammate's target depth off a single,
//! continuous "how far forward has this side's attack genuinely
//! established itself" number (`MatchContext::attack_reference_depth`,
//! a ratchet over the ball's raw distance from goal). Pavel reported a
//! live symptom that survived a same-day patch to that ratchet (a
//! backward pass still visibly pulled the whole team back), then
//! directly challenged the underlying mechanism as not real football
//! logic — a shared "depth from goal" ceiling that every central
//! attacking player gets capped to is genuine defensive-line doctrine
//! (that IS what defending is), but it is not how an attacking player's
//! job actually works; his job depends on WHICH PHASE of possession the
//! team is in, not on a raw continuous distance. His own framing (the
//! design doc's own verbatim quote) is what this file now implements:
//! identify who has the ball, and — while we do — derive each line's
//! job from WHICH THIRD OF THE PITCH the ball currently occupies
//! (build-up / progression / final-third), not from a continuous
//! number. `MatchContext::attack_reference_depth` and its ratchet/decay
//! logic are deleted entirely (not kept alongside this as a fallback).
//! `REST_DEFENSE_COUNT` (a fixed "N deepest defenders always hold back"
//! headcount rule) is also deleted, not replaced with an equivalent —
//! the "someone stays back for balance" property now emerges from
//! central defenders simply having a much shallower depth cap than
//! every other role/phase (a real centre-back doesn't join the box in
//! open play, at any phase), per Pavel's own explicit framing: "we do
//! not need 'stay back' — you play as the line, we go back as the line
//! when the opponents are in the attack."
//!
//! `AttackDuty::Receive` (the pass-receiver-target fix, same file,
//! earlier the same week) is untouched — it solves a different,
//! correctly-diagnosed problem (a specific player ignoring a live pass
//! aimed at him) and still runs first, before any of the phase logic
//! below. The box-crowding anti-clustering redirect and the
//! same-flank/same-role stagger are also kept (re-derived onto the new
//! phase gate, not copied verbatim) — anti-clustering is this project's
//! own repeated, hard-won lesson (its founding case is domain 1's own
//! 2026-09-12 rollback), not part of what was rejected here; only the
//! CONTINUOUS DEPTH FORMULA the old system used is what's being
//! replaced.
//!
//! Deliberately calls no pre-existing decision function from before the
//! from-null rewrite (no `ForwardCreatingSpaceState`,
//! `MidfielderRunningState`'s off-ball branch, `DefenderPushingUpState`,
//! or any sibling) — every target formula here is derived fresh from
//! the football reasoning in the two design docs above, using only raw
//! `MatchField`/`MatchContext` data plus genuine measurements
//! (`max_speed_with_condition`, goal positions, real box geometry,
//! Euclidean geometry).
//!
//! Process check (`docs/decision-from-null.md`'s required self-audit):
//! this module's call graph touches only `MatchField`/`MatchContext`/
//! `MatchPlayer` struct fields, `MatchContext::penalty_area` (real box
//! geometry, already whitelisted as a measurement), the pitch-thirds
//! boundary constants shared with domain 2's `on_ball_decision` module
//! (`OWN_THIRD_BOUNDARY`/`FINAL_THIRD_BOUNDARY` — a physical fact about
//! the pitch, identical in both domains by construction rather than
//! independently re-derived and risking drift; the constants themselves
//! encode no judgment, only where real thirds-of-a-105m-pitch sit), and
//! `nalgebra::Vector3` geometry. Zero pre-existing decision-function
//! dependencies, and zero dependency on domain 2's own on-ball
//! DECISIONS (`carry_priority_mult`, `pass_value`, etc. are not called
//! here — only the two shared boundary constants are).

use crate::PlayerFieldPositionGroup;
use crate::club::player::ability::position::PlayerPositionType;
use crate::r#match::player::AttackDuty;
use crate::r#match::player::strategies::common::players::ops::on_ball_decision::{
    FINAL_THIRD_BOUNDARY, OWN_THIRD_BOUNDARY,
};
use crate::r#match::{MatchContext, MatchField, PlayerSide};
use nalgebra::Vector3;

/// Which third of the pitch OUR OWN attack currently occupies — the
/// replacement for the old continuous "how far forward" ratchet. Real
/// football's phase-of-possession doctrine (same real-world concept
/// domain 2's own `carry_priority_mult` already names in its doc
/// comment): who's responsible for progressing the ball, and where
/// everyone else should be, changes discretely by zone, not smoothly by
/// exact distance. Recomputed fresh every tick from the ball's real,
/// current position — no persistence/hysteresis (open question #1 in
/// the design doc: ship the simple version first, verify with real
/// match data whether a zone-boundary really dithers, before adding any
/// debounce to guess around a problem that might not exist).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Ball in our own defensive third.
    BuildUp,
    /// Ball in the middle third.
    Progression,
    /// Ball in the opponent's third.
    Final,
}

fn phase_for(own_goal_dist: f32) -> Phase {
    if own_goal_dist < OWN_THIRD_BOUNDARY {
        Phase::BuildUp
    } else if own_goal_dist < FINAL_THIRD_BOUNDARY {
        Phase::Progression
    } else {
        Phase::Final
    }
}

/// Width-duty target depth (distance from own goal), by role and
/// current phase. Reasoned estimates (category d, no sourced exact
/// figures) grounded in the design doc's own football reasoning: a wide
/// player already holds a fairly advanced base position even during
/// build-up (real wingers don't retreat deep to help), reaching genuinely
/// high, width-stretching positions once the team is established higher
/// up. Wingbacks/full-backs (the `Defender` row) sit deliberately behind
/// wide midfielders/forwards at every phase — a real full-back overlaps
/// BEHIND the winger ahead of him, not level with or in front of him —
/// and only reach a genuinely high position in the Final phase, matching
/// Pavel's own "especially wingbacks" framing for who's freed to join
/// the attack once it's properly established.
fn width_depth(group: PlayerFieldPositionGroup, phase: Phase) -> f32 {
    match (group, phase) {
        (PlayerFieldPositionGroup::Forward, Phase::BuildUp) => 460.0,
        (PlayerFieldPositionGroup::Forward, Phase::Progression) => 560.0,
        (PlayerFieldPositionGroup::Forward, Phase::Final) => 700.0,
        (PlayerFieldPositionGroup::Midfielder, Phase::BuildUp) => 380.0,
        (PlayerFieldPositionGroup::Midfielder, Phase::Progression) => 520.0,
        (PlayerFieldPositionGroup::Midfielder, Phase::Final) => 680.0,
        (PlayerFieldPositionGroup::Defender, Phase::BuildUp) => 230.0,
        (PlayerFieldPositionGroup::Defender, Phase::Progression) => 420.0,
        (PlayerFieldPositionGroup::Defender, Phase::Final) => 620.0,
        // Goalkeepers never reach this pool (excluded below); kept for
        // exhaustiveness only.
        _ => 400.0,
    }
}

/// Advance-duty (central player) target depth, by role and current
/// phase — the direct replacement for the old ratcheted-ball-distance
/// formula. This table IS where "someone stays back for balance" lives
/// now: a central `Defender`'s cap stays shallow even in the Final
/// phase (a real centre-back does not join the box in open play), while
/// `Midfielder`/`Forward` push meaningfully higher each phase — no
/// fixed headcount rule needed. Reasoned estimates (category d).
fn advance_depth(group: PlayerFieldPositionGroup, phase: Phase) -> f32 {
    match (group, phase) {
        (PlayerFieldPositionGroup::Defender, Phase::BuildUp) => 190.0,
        (PlayerFieldPositionGroup::Defender, Phase::Progression) => 300.0,
        (PlayerFieldPositionGroup::Defender, Phase::Final) => 380.0,
        (PlayerFieldPositionGroup::Midfielder, Phase::BuildUp) => 340.0,
        (PlayerFieldPositionGroup::Midfielder, Phase::Progression) => 480.0,
        (PlayerFieldPositionGroup::Midfielder, Phase::Final) => 620.0,
        // Forwards hold the highest available central line even during
        // our own build-up (a real forward doesn't drop back to help —
        // he stays the long out-ball option), a fixed Progression-phase
        // baseline that the dedicated come-short/run-in-behind branch
        // below then adjusts per-moment, and a Final-phase baseline the
        // box anti-crowding redirect (below) then refines.
        (PlayerFieldPositionGroup::Forward, Phase::BuildUp) => 460.0,
        (PlayerFieldPositionGroup::Forward, Phase::Progression) => 480.0,
        (PlayerFieldPositionGroup::Forward, Phase::Final) => 660.0,
        _ => 400.0,
    }
}

/// How much a same-flank, same-role-group collision (a real but rare
/// formation overlap — e.g. `MidfielderRight` and
/// `AttackingMidfielderRight` both flagged wide-right) staggers the
/// second player's depth behind the first, so two width-duty players
/// who'd otherwise get the literal identical target from `width_depth`
/// don't collapse onto the same point. Reasoned estimate (category d).
const WIDTH_SAME_GROUP_STAGGER: f32 = 60.0;

/// How much a marked/covered forward's Progression-phase target pulls
/// back from its baseline depth to "come short" and offer a foot pass,
/// instead of holding a line position a defender already has covered.
/// Reasoned estimate (category d) — roughly the same real-world
/// magnitude as domain 1's own `MIDFIELD_LINE_GAP_UNITS` (15m), applied
/// here as a genuine drop-off distance rather than a line gap.
const FORWARD_COME_SHORT_PULLBACK: f32 = 120.0; // ~15m

/// How much a central player's lateral position stays anchored to his
/// own formation slot vs. drifting toward the ball — anti-clustering by
/// construction (different players have different slots, so this alone
/// already spreads same-phase, same-role central players apart).
const ADVANCE_WIDTH_PULL: f32 = 0.25;

/// Beyond this distance from a target, the nearest opponent doesn't
/// meaningfully contest it — genuine open space worth pushing further
/// into (Step E: "advanced players look for real space, not just a line
/// position"). Applied only in the Progression/Final phases — pushing
/// into open space during our own build-up is exactly the "sprint up
/// the pitch" behaviour the design doc explicitly says build-up doesn't
/// need.
const PENETRATION_OPEN_THRESHOLD: f32 = 90.0;
/// How far a penetration bias pushes an already-advanced, genuinely
/// open target further toward the opponent's goal.
const PENETRATION_PUSH: f32 = 40.0;

/// How far to the touchline a width target sits.
const WIDTH_MARGIN: f32 = 24.0;

/// Real box-crowding scenario gate (Pavel's direct report, 2026-09-23:
/// "when a forward has a ball in the box, defenders and attackers are
/// all grouped somewhere around penalty spot — nobody is covering areas
/// to the left or to the right"), kept from the prior round and
/// re-derived onto the phase system: box-crowding redirect now requires
/// BOTH the Final phase AND the ball genuinely near the opponent's own
/// box (this margin, past the box's own real depth via
/// `MatchContext::penalty_area`) — a player shouldn't be redirected to
/// an "open flank pocket of the box" the moment the ball merely crosses
/// into the attacking third at its very edge, 35m from the opponent's
/// goal.
const BOX_CROWD_MARGIN_UNITS: f32 = 44.0; // ~5.5m past the box's own real depth

/// How wide a central band around the box's own centre line counts as
/// "the same tiny zone" for the crowding check — roughly a penalty-
/// spot-sized cluster, not the whole box. Expressed as a fraction of
/// the box's real half-width so it scales correctly with field size.
const BOX_CENTRAL_BAND_FRACTION: f32 = 0.35;

/// How many Advance-duty players may legitimately occupy that central
/// band before any further ones get redirected to an open flank pocket
/// instead — a real box moment plausibly has two central runners (e.g.
/// a striker plus an arriving midfielder), not zero. Reasoned estimate
/// (category d).
const BOX_CENTRAL_OCCUPANCY_CAP: usize = 2;

/// How far toward the box's own edge a redirected "expose the open
/// area" pocket sits, as a fraction of the box's real half-width.
const BOX_POCKET_Y_FRACTION: f32 = 0.75;

/// How much a pocket's effective openness is discounted per teammate
/// already redirected there this tick — prevents every overflow player
/// independently scoring both pockets from the SAME opponent geometry
/// and collapsing onto the identical "more open" one (found by raw
/// trace, 2026-09-23; kept, unrelated to this redesign).
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
    /// flank pocket by the Final-phase box-crowding rule. Diagnostic
    /// field — lets the match-logs trace directly confirm the redirect
    /// fires, rather than inferring it from `target.y` alone.
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
/// holds the ball (Step 1 of the design doc: a known carrier, or an
/// ordinary directed pass genuinely still in flight), assigns real
/// duties to that team's non-carrier outfielders, keyed on which THIRD
/// of the pitch the ball currently occupies. A genuine loose ball
/// (Step 4, "transition") leaves this system fully inactive — the
/// design doc names this honestly as the least-resolved piece, not
/// decided here; domain 1's own loose-ball machinery already organises
/// both teams reactively around a contested ball. When the OPPONENT
/// clearly has the ball, this system is likewise inactive for our own
/// team — Step 3's "drop back as a line" reaction is domain 1's job,
/// not rebuilt here (see the design doc's own "what already exists and
/// can likely be reused" section).
pub fn assign_attacking_support(field: &mut MatchField, context: &mut MatchContext) {
    for p in field.players.iter_mut() {
        p.ai_attack_duty = AttackDuty::None;
    }

    // The genuine current owner OR, during an ordinary directed pass in
    // flight, the passer -- used only to identify the team/side that
    // clearly, currently holds the ball this tick (Step 1's Team-A/
    // Team-B distinction). `current_owner` goes `None` for the ball's
    // entire flight time on every single pass, so without the
    // `pending_pass_passer` fallback this whole system would go fully
    // dark during that window (found 2026-09-23, fixed the same day --
    // kept here unchanged, this redesign only replaces the phase/depth
    // formulas downstream of this identification step).
    //
    // realism-bug (2026-09-24, "team follows the ball" investigation,
    // continued): `pending_pass_passer` itself doesn't cover a settled
    // dribble/hold, or the window between a pass being credited
    // complete (stats-wise, `credit_completed_pass`) and the ball
    // physically reaching the receiver -- a raw-traced example showed
    // a 2.25s gap where BOTH fields were `None` while a player was
    // plainly, continuously in control. `previous_owner` is kept
    // truthful by `check_ball_ownership` on real ball-distance grounds
    // (cleared only once its owner is genuinely >5 units from the
    // ball's live position), so it's a reliable last fallback here --
    // same fix, same reasoning, applied to both from-null domains
    // (see the matching Case 2c added to `defensive_duty.rs` the same
    // session).
    let reference_id = field
        .ball
        .current_owner
        .or(field.ball.pending_pass_passer)
        .or(field.ball.previous_owner);
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

    #[cfg(feature = "match-logs")]
    crate::match_log_info!(
        "CARRIERDIAG t={} carrier_id={} carrier_team={}",
        context.total_match_time,
        carrier.id,
        carrier.team_id
    );

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

    // Step 2 of the design doc: the phase is which third of the pitch
    // the ball CURRENTLY occupies -- not a ratcheted/smoothed number.
    // Recomputed fresh every tick from the ball's real position.
    let ball_dist_to_own_goal = (field.ball.position - own_goal).magnitude();
    let phase = phase_for(ball_dist_to_own_goal);

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
    // ordinary pass in flight) stays excluded from the assignable pool.
    // The passer himself is NOT included even once the ball has left
    // his feet -- a real, severe performance regression was found and
    // isolated when this was tried (2026-09-23, see
    // `docs/off-ball-attacking-positioning-from-null.md`'s own log), so
    // this stays the narrower, proven-safe exclusion.
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
    // dedicated target -- the ball's own real flight destination, not
    // whatever generic Width/Advance target would otherwise apply to
    // him.
    //
    // realism-bug (2026-09-24, passing-direction investigation): the
    // target here used to be `pending_pass_target` alone -- a STATIC
    // point equal to the receiver's own position at the exact instant
    // the ball was struck, with zero regard for his own momentum.
    // `handle_pass_to_event` (players.rs) already computes a strictly
    // better target for this -- `ideal_target`, which leads the ball
    // toward wherever the receiver's own current run will carry him,
    // and stores it on the player as `incoming_pass_target` -- but
    // nothing here ever read it, so a receiver already mid-run got
    // frozen at his kick-instant position instead of continuing to
    // meet the ball in stride. Raw match data (10 matches, 3011
    // classified forward/backward passes) showed this specific gap is
    // NOT what was driving Pavel's reported "players run away from a
    // back pass" symptom -- the divergence this fixes is actually
    // larger for forward passes than backward ones, and the intended
    // receiver's real drift-from-target at true pass resolution was
    // already ~0.2-0.7% for every dispatch path, both directions --
    // but it's a real, separate correctness gap (a receiver's own run
    // should count, whichever way it's headed) worth closing on its
    // own merits regardless.
    if let Some(receiver_id) = field.ball.pass_target_player_id {
        if let Some(receiver) = pool.iter().find(|c| c.id == receiver_id) {
            let target = field
                .get_player(receiver_id)
                .and_then(|p| p.incoming_pass_target)
                .or(field.ball.pending_pass_target)
                .unwrap_or(receiver.position);
            results.push(Assignment {
                id: receiver.id,
                duty: AttackDuty::Receive,
                target,
                redirected: false,
            });
            assigned.push(receiver.id);
        }
    }

    // --- Step C: width -- any remaining wide-slotted player holds the
    // flank, at a depth keyed to role + current phase (`width_depth`).
    let target_x_from_depth = |depth: f32| own_goal.x + attack_dir * depth;
    let wide_cands: Vec<Cand> = pool
        .iter()
        .filter(|c| !assigned.contains(&c.id) && is_wide(c.position_type))
        .copied()
        .collect();
    let field_height = context.field_size.height as f32;
    let mut flank_group_seen: Vec<(bool, PlayerFieldPositionGroup, i32)> = Vec::new();
    for c in wide_cands.iter() {
        let is_left = is_left_flank(c.position_type);
        let stagger_index = match flank_group_seen
            .iter_mut()
            .find(|(l, g, _)| *l == is_left && *g == c.position_group)
        {
            Some((_, _, n)) => {
                *n += 1;
                *n - 1
            }
            None => {
                flank_group_seen.push((is_left, c.position_group, 1));
                0
            }
        };
        let depth =
            width_depth(c.position_group, phase) - stagger_index as f32 * WIDTH_SAME_GROUP_STAGGER;
        let y = if is_left { WIDTH_MARGIN } else { field_height - WIDTH_MARGIN };
        let mut target = Vector3::new(target_x_from_depth(depth), y, 0.0);
        let mut duty = AttackDuty::Width;
        // Step E: push into genuinely open space -- only once the team
        // has genuinely progressed (Progression/Final); build-up doesn't
        // need width players sprinting into space ahead of the ball.
        if matches!(phase, Phase::Progression | Phase::Final)
            && nearest_opp_dist(target) > PENETRATION_OPEN_THRESHOLD
        {
            let to_goal = opponent_goal - target;
            let dist = to_goal.magnitude();
            if dist > 1.0 {
                target += (to_goal / dist) * PENETRATION_PUSH;
                duty = AttackDuty::Penetrate;
            }
        }
        results.push(Assignment {
            id: c.id,
            duty,
            target,
            redirected: false,
        });
        assigned.push(c.id);
    }

    // --- Step D: advancing line -- every remaining central player
    // (midfielders, and defenders, forwards) gets a target depth from
    // `advance_depth`, keyed to role + current phase. Real box geometry
    // (`MatchContext::penalty_area`) governs the Final-phase
    // anti-crowding redirect below.
    let attacking_box = context.penalty_area(opposite_side(carrier_side) == PlayerSide::Left);
    let box_half_width = (attacking_box.max.y - attacking_box.min.y) / 2.0;
    let box_center_y = (attacking_box.max.y + attacking_box.min.y) / 2.0;
    let box_depth = (attacking_box.max.x - attacking_box.min.x).abs();
    let box_crowd_gate = box_depth + BOX_CROWD_MARGIN_UNITS;
    let box_crowding_active = matches!(phase, Phase::Final)
        && (field.ball.position - opponent_goal).magnitude() <= box_crowd_gate;
    let box_central_band = box_half_width * BOX_CENTRAL_BAND_FRACTION;
    let box_pocket_offset = box_half_width * BOX_POCKET_Y_FRACTION;
    let mut central_box_count = 0usize;
    let mut pocket_occupancy = [0i32, 0i32]; // [left, right]

    for c in pool.iter().filter(|c| !assigned.contains(&c.id)) {
        let y = c.start_position.y * (1.0 - ADVANCE_WIDTH_PULL) + field.ball.position.y * ADVANCE_WIDTH_PULL;

        // Step 2's forward-specific choice: during Progression, a
        // forward chooses between coming short to combine or making a
        // run in behind, based on whether genuine space exists -- not a
        // fixed formula depth. Every other player/phase combination
        // uses the flat per-role, per-phase `advance_depth` table.
        let (mut target, mut duty) = if c.position_group == PlayerFieldPositionGroup::Forward
            && matches!(phase, Phase::Progression)
        {
            let base_depth = advance_depth(PlayerFieldPositionGroup::Forward, phase);
            let candidate = Vector3::new(target_x_from_depth(base_depth), y, 0.0);
            if nearest_opp_dist(candidate) > PENETRATION_OPEN_THRESHOLD {
                // Genuine space beyond the last defender -- commit to
                // the run in behind.
                let to_goal = opponent_goal - candidate;
                let dist = to_goal.magnitude();
                let pushed = if dist > 1.0 {
                    candidate + (to_goal / dist) * PENETRATION_PUSH
                } else {
                    candidate
                };
                (pushed, AttackDuty::Penetrate)
            } else {
                // Marked/covered -- come short to offer a foot pass
                // instead of holding a line position a defender already
                // has covered.
                let come_short_depth = (base_depth - FORWARD_COME_SHORT_PULLBACK).max(0.0);
                (
                    Vector3::new(target_x_from_depth(come_short_depth), y, 0.0),
                    AttackDuty::Advance,
                )
            }
        } else {
            let depth = advance_depth(c.position_group, phase);
            (Vector3::new(target_x_from_depth(depth), y, 0.0), AttackDuty::Advance)
        };

        // Final-phase box anti-crowding redirect (Pavel's direct report,
        // 2026-09-23; kept from the prior round, re-derived onto the
        // Final-phase gate instead of a raw ball-distance-to-box check).
        let mut redirected_to_pocket = false;
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
        } else if !redirected_to_pocket
            && matches!(phase, Phase::Progression | Phase::Final)
            && c.position_group != PlayerFieldPositionGroup::Defender
            && duty != AttackDuty::Penetrate
        {
            // Step E, general case (forwards already resolved their own
            // space question above in Progression; this covers
            // midfielders in Progression/Final, and forwards in Final
            // when the box-crowding branch above didn't apply). Central
            // defenders never get this push -- a real centre-back
            // doesn't make forward penetrating runs regardless of
            // phase, which is exactly what keeps "someone back for
            // balance" true without a headcount rule.
            if nearest_opp_dist(target) > PENETRATION_OPEN_THRESHOLD {
                let to_goal = opponent_goal - target;
                let dist = to_goal.magnitude();
                if dist > 1.0 {
                    target += (to_goal / dist) * PENETRATION_PUSH;
                }
                duty = AttackDuty::Penetrate;
            }
        }

        results.push(Assignment {
            id: c.id,
            duty,
            target,
            redirected: redirected_to_pocket,
        });
    }

    // Diagnostic (kept feature-gated at zero cost otherwise): the
    // resolved phase + ball depth once per tick -- lets a raw trace
    // directly check for boundary flicker (open question #1 in the
    // design doc) before deciding whether any debounce is warranted.
    #[cfg(feature = "match-logs")]
    crate::match_log_info!(
        "ATTACKPHASE t={} side={:?} phase={:?} ball_dist={:.1}",
        context.total_match_time,
        carrier_side,
        phase,
        ball_dist_to_own_goal
    );

    #[cfg(feature = "match-logs")]
    for a in &results {
        if let Some(c) = pool.iter().find(|c| c.id == a.id) {
            crate::match_log_info!(
                "ATTACKSUPPORT t={} id={} pt={:?} duty={:?} phase={:?} pos_x={:.1} pos_y={:.1} target_x={:.1} target_y={:.1} redirected={}",
                context.total_match_time,
                a.id,
                c.position_type,
                a.duty,
                phase,
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
