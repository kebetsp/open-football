//! Match-engine ball model, split by concern. The `Ball` struct lives
//! here together with the per-tick orchestrator (`update` / `update_light`)
//! and the simple state queries the rest of the engine reads. The
//! heavier domain passes are sibling modules:
//!
//! | Submodule       | Concern                                                      |
//! |-----------------|--------------------------------------------------------------|
//! | [`ownership`]   | Pass-target claims, deadlock resolution, stall safety nets, ball-ownership claim flow |
//! | [`interactions`]| Intercept / shot-block / shot-save resolution                |
//! | [`goal`]        | Goal / over-the-bar / wide-of-goal handling                  |
//! | [`motion`]      | Velocity integration, owner tracking, boundary inset         |
//! | [`stall`]       | Position-anchor stall detector + snapshot diagnostics        |

pub mod contact;
mod goal;
pub mod interactions;
mod motion;
mod ownership;
mod restart;
mod save;
mod stall;

use crate::r#match::engine::ball::events::BallEvent;
use crate::r#match::engine::set_pieces::CornerRoutine;
use crate::r#match::events::EventCollection;
use crate::r#match::{GameTickContext, MatchContext, MatchPlayer, PlayerSide};
use nalgebra::Vector3;
use std::collections::VecDeque;

/// Origin of the most recent live pass / restart. Read by the offside
/// resolver: only goal kicks, throw-ins, and corners are exempt from
/// offside; free kicks (direct/indirect) and penalties are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassOriginRestart {
    OpenPlay,
    GoalKick,
    Corner,
    ThrowIn,
    /// Generic free kick (legacy / offside fallback). Treated like a
    /// direct free kick by the offside resolver.
    FreeKick,
    /// Foul outside the penalty area, severity Normal+: ball can be shot
    /// at goal directly.
    DirectFreeKick,
    /// Offside or technical infringement: cannot be shot directly into
    /// goal — needs a touch from a second player first.
    IndirectFreeKick,
    /// Foul inside defending penalty area: ball at penalty spot.
    Penalty,
}

impl Default for PassOriginRestart {
    fn default() -> Self {
        PassOriginRestart::OpenPlay
    }
}

impl PassOriginRestart {
    /// Set-piece restarts that exempt the receiver from offside.
    pub fn is_offside_exempt(self) -> bool {
        matches!(
            self,
            PassOriginRestart::GoalKick | PassOriginRestart::Corner | PassOriginRestart::ThrowIn
        )
    }

    /// True for any free-kick-style restart (direct/indirect/legacy).
    /// Penalties and corners are NOT free kicks for routine selection.
    pub fn is_free_kick(self) -> bool {
        matches!(
            self,
            PassOriginRestart::FreeKick
                | PassOriginRestart::DirectFreeKick
                | PassOriginRestart::IndirectFreeKick
        )
    }
}

/// Snapshot of the offside-relevant geometry at the moment a pass is
/// kicked. Stored on the ball for the duration of an in-flight pass so
/// the offside check can fire on receiver involvement (touch / claim /
/// active challenge) instead of at pass start.
#[derive(Debug, Clone, Copy)]
pub struct OffsideSnapshot {
    pub origin: PassOriginRestart,
    pub passer_id: u32,
    pub passer_side: PlayerSide,
    pub receiver_id: u32,
    pub ball_x_at_kick: f32,
    pub second_last_defender_x: f32,
    pub receiver_x_at_kick: f32,
    pub receiver_y_at_kick: f32,
    pub set_tick: u64,
}

impl OffsideSnapshot {
    /// Decide whether the snapshot represents an offside position.
    /// Tolerance 1.5u absorbs foot-vs-shoulder ambiguity.
    pub fn is_offside(&self) -> bool {
        const TOLERANCE: f32 = 1.5;
        match self.passer_side {
            PlayerSide::Left => {
                if self.receiver_x_at_kick <= self.ball_x_at_kick + TOLERANCE {
                    return false;
                }
                self.receiver_x_at_kick > self.second_last_defender_x + TOLERANCE
            }
            PlayerSide::Right => {
                if self.receiver_x_at_kick >= self.ball_x_at_kick - TOLERANCE {
                    return false;
                }
                self.receiver_x_at_kick < self.second_last_defender_x - TOLERANCE
            }
        }
    }
}

pub struct Ball {
    pub start_position: Vector3<f32>,
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub center_field_position: f32,

    pub field_width: f32,
    pub field_height: f32,

    pub flags: BallFlags,

    pub previous_owner: Option<u32>,
    pub current_owner: Option<u32>,
    pub take_ball_notified_players: Vec<u32>,
    pub notification_cooldown: u32,
    pub notification_timeout: u32,
    pub last_boundary_position: Option<Vector3<f32>>,
    pub unowned_stopped_ticks: u32,
    pub ownership_duration: u32,
    pub claim_cooldown: u32,
    /// (player, until tick): a carrier who lost a duel cannot reclaim the
    /// ball he was just dispossessed of until then (contest model).
    pub claim_barred: Option<(u32, u64)>,
    pub pass_target_player_id: Option<u32>,
    /// Passer id of the most-recent live pass. Set on pass emit,
    /// cleared on any opponent touch or when the pass's natural
    /// window (150 ticks ≈ 1.5 s) expires. The pass-completion stat
    /// uses this as the source of truth for "was this claim a pass
    /// reception?" — `pass_target_player_id` gets cleared in too
    /// many unrelated paths to serve that role. None outside an
    /// active pass window.
    pub pending_pass_passer: Option<u32>,
    pub pending_pass_set_tick: u64,
    pub recent_passers: VecDeque<u32>,
    pub contested_claim_count: u32,
    pub unowned_ticks: u32,
    /// Snapshot captured at the moment the ball became uncontrolled — ball
    /// kinematics plus every player's state/position/velocity. Held until
    /// the stall resolves, then attached to the resolution log (only if
    /// the stall was long enough to log). Provides the "what did the
    /// pitch look like when this got stuck" context in the same line as
    /// the duration. Cleared on ownership resume.
    pub stall_start_snapshot: Option<String>,
    pub goal_scored: bool,
    pub kickoff_team_side: Option<PlayerSide>,
    pub cached_landing_position: Vector3<f32>,
    /// When a set-piece (corner, goal kick) rewrites ownership to a
    /// specific player, the ball can only mutate itself here — player
    /// teleport requires &mut field.players which lives one layer up.
    /// Populated inside `check_wide_of_goal` and drained by the engine
    /// after `ball.update` returns, so the owner is on the ball before
    /// the next `move_to` distance check can null their ownership.
    pub pending_set_piece_teleport: Option<(u32, Vector3<f32>)>,
    /// Players that must step/lunge onto the ball after a contact
    /// resolved inside `Ball::update` (which only sees `&[MatchPlayer]`).
    /// Drained by `apply_pending_set_piece_teleport`. The BALL never
    /// moves for a contact; the player does (see `contact.rs`).
    pub pending_player_jumps: Vec<(u32, Vector3<f32>)>,
    /// Attacking centre-backs to teleport into the box when a corner is
    /// awarded — the dead-ball set-up (in real football the big men walk
    /// up during the stoppage). Populated in the corner branch of
    /// `check_wide_of_goal`, drained by the engine alongside the taker
    /// teleport. Each entry is (player_id, box_target_position). Without
    /// this the CBs cannot cover the length of the pitch before the cross
    /// is delivered, so defenders never get to attack corners.
    pub pending_corner_teleports: Vec<(u32, Vector3<f32>)>,
    /// Bodies to place for a foul restart — the free-kick wall and
    /// retreating defenders, or the box-clear for a penalty. Populated
    /// by `award_restart_for_foul`, drained by the engine alongside the
    /// taker teleport. Unlike corner teleports these do NOT override the
    /// player's state: the wall holds only through the restart window
    /// (claim cooldown), then normal positioning resumes.
    pub pending_restart_teleports: Vec<(u32, Vector3<f32>)>,
    /// §13.3: true only for the exact duration a struck direct free-kick
    /// shot is in flight after clearing the wall check. `pass_origin_restart`
    /// cannot be reused for this — `resolve_free_kick`'s DirectShot branch
    /// resets it to OpenPlay on the same tick the shot is dispatched (so a
    /// rebound doesn't spuriously re-trigger a wall check), which happens
    /// before `try_intercept`/`try_block_shot` ever run on the now-flying
    /// ball. Set fresh by every `handle_shoot_event` call; inert once the
    /// ball is claimed/saved/out since those interaction checks already
    /// early-return in that case.
    pub direct_fk_shot_in_flight: bool,
    /// realism-bug (2026-08-26, corner/deflection investigation): tick
    /// until which a defender's touch (block, tackle, push-out, aerial
    /// clearance) is still sending the ball over a boundary. Set by
    /// `release_after_touch` and the aerial-contest resolver; read by
    /// `try_intercept`, which otherwise treats the deflection as an
    /// interceptable pass from the shooter's team (via `previous_owner`,
    /// untouched by the block) and lets the very defender who just
    /// deflected it re-claim it within 1-2 ticks — measured at 3/4
    /// corner-branch firings across 40 matches before this fix.
    pub corner_deflection_guard_until_tick: u64,
    /// Defenders who already had their one block attempt at the current
    /// shot (same idiom as `intercept_attempted` for passes). Without it
    /// `try_block_shot` re-rolled every tick a defender stayed near the
    /// shot line: measured median 4 rolls per shot, which stacked a
    /// 10-15% per-event chance into a 47-61% block rate (real 26%).
    pub block_rolled: Vec<u32>,
    /// `ShotTarget::dispatch_tick` of the shot `block_rolled` belongs to;
    /// a different value means a new shot (or a deflection) and the list
    /// is cleared.
    pub block_roll_shot_tick: u64,
    /// Defenders who already had their one attempt at the current pass.
    pub intercept_attempted: Vec<u32>,
    /// The resolved contest waiting for its contact tick.
    pub pending_intercept: Option<PendingIntercept>,
    /// Fire-once guard for the discrete corner aerial contest. A played-out
    /// lofted corner can't thread the congested box to a specific runner, so
    /// once the cross is struck the engine resolves a single skill-weighted
    /// aerial contest (attacking headers vs the defending line + GK command)
    /// and, if an attacker wins, drops the ball on their head to be headed
    /// on goal. False = armed (a corner has been awarded, not yet resolved);
    /// true = nothing to resolve.
    pub corner_contest_resolved: bool,
    /// Realism-bug 2026-07-26 follow-up: the same discrete aerial-contest
    /// mechanism as `corner_contest_resolved`, generalized to open-play
    /// crosses / free-kick crosses / long high passes / goal-kick punts
    /// (anything airborne near a box that isn't a corner, which keeps its
    /// own dedicated resolver/flag). No single per-restart lifecycle
    /// exists for these, so this is a plain tick-cooldown instead of a
    /// bool: `resolve_aerial_contest` only fires once every
    /// `AERIAL_CONTEST_COOLDOWN_TICKS` to avoid re-resolving the same
    /// delivery every tick while the ball is airborne.
    pub last_aerial_contest_tick: u64,
    /// Corner routine picked by `pick_corner_routine` at corner setup.
    /// Lets the corner aerial-contest in `resolve_corner_contest` and
    /// downstream xG accounting know whether the delivery is targeting
    /// the near post, far post, penalty spot, or short. Cleared after
    /// the corner resolves. `None` whenever a corner isn't pending.
    pub pending_corner_routine: Option<CornerRoutine>,
    /// Counter for "ball is owned but nothing is happening" stalls.
    /// The unowned-stall warning can't see these because ownership is
    /// set, but visually the ball sits with a player who isn't moving,
    /// isn't passing, isn't dribbling — same "ball stuck" symptom, no
    /// warning. Reset whenever owner changes or any meaningful motion
    /// resumes; fires a separate warning once it crosses the threshold.
    pub owned_stuck_ticks: u32,
    pub owned_stuck_logged: bool,
    /// Position-based stall detector — catches cases the owned/unowned
    /// counters miss, specifically: rapid ownership flipping keeps
    /// resetting both counters (each "change" looks like progress) but
    /// the ball physically never leaves a small region. We sample the
    /// ball's position every N ticks and if it hasn't moved more than
    /// a threshold distance over a window, it's stuck regardless of
    /// who "owns" it at any given instant.
    pub stall_anchor_pos: Vector3<f32>,
    pub stall_anchor_tick: u32,

    /// Trajectory projection cached at the moment a shot is fired. Lets
    /// the goalkeeper commit to an intercept line instead of re-chasing
    /// the ball's current position every tick (which lost ground vs a
    /// 5.6 u/tick shot). `None` whenever the ball isn't a shot in
    /// flight; cleared on catch, goal, or any ownership event.
    pub cached_shot_target: Option<ShotTarget>,

    /// Per-shot lifecycle marker: when the physics-level `try_save_shot`
    /// resolves a shot mid-flight (catch / parry / dangerous parry), it
    /// stores `(keeper_id, shooter_id)` here so the post-tick stat
    /// credit can fire saves and on-target without relying on the GK
    /// state machine to also re-detect the same shot.
    /// Consumed (cleared to `None`) by the event dispatcher once
    /// stats have been credited. This makes saves-on-target match
    /// physics-resolved saves 1:1 — the previous architecture had two
    /// independent save systems (physics and state-machine) where one
    /// changed ball state without crediting and the other rolled
    /// independent saves that often missed.
    pub pending_save_credit: Option<(u32, u32)>,

    /// Last meaningful touch on the ball. Drives restart resolution
    /// (throw-ins, corners, goal kicks) and pass-origin metadata. Updated
    /// from any path that hands ownership to a player (claim, intercept,
    /// block, save, pass) and from foot-deflections that don't transfer
    /// ownership but still count as a touch for the dead-ball decision.
    pub last_touch_player_id: Option<u32>,
    pub last_touch_team_id: Option<u32>,
    pub last_touch_tick: u64,
    pub last_touch_was_controlled: bool,
    /// Latest tick captured at update entry. Lets per-update helpers
    /// (intercept, block, save, claim, throw-in) record_touch without
    /// having to thread the tick through every signature.
    pub current_tick_cached: u64,

    /// Origin of the most recent live pass — set when a PassTo event
    /// fires from a restart (goal kick, throw-in, corner, free kick).
    /// Read by the delayed-offside resolver. Resets to OpenPlay on any
    /// non-restart pass or once the pass-window expires.
    pub pass_origin_restart: PassOriginRestart,
    /// Same-touch rule (law of the game): the player who takes a restart
    /// (corner / free kick / penalty / throw-in / goal kick) must not be
    /// the next to touch the ball. Set to the taker at the restart
    /// strike, cleared the moment any OTHER player registers a touch
    /// (`record_touch`). While set, every claim / intercept / force-claim
    /// candidate selection skips this player.
    pub restart_taker_lock: Option<u32>,
    /// The player staged to take the current restart (set where the
    /// dead ball is handed to a taker). The same-touch lock engages only
    /// when THIS player strikes — a later open-play pass under a stale
    /// restart origin (e.g. the Corner origin deliberately preserved
    /// through the cross for the aerial-contest resolver) must not lock
    /// whoever happens to make it.
    pub restart_pending_taker: Option<u32>,
    /// §12.2 one-shot latch for the free-kick taker's dedicated decision
    /// (direct shot / box delivery / short). Set once the decision has
    /// been made for the currently-staged free kick so a "play short"
    /// outcome isn't re-rolled every tick until the taker releases the
    /// ball. Reset when a new foul restart is awarded.
    pub free_kick_decided: bool,
    /// §12.4 — the teammate staged as the short-corner option for the
    /// current corner (None when nobody was staged). The state processor
    /// holds this player at his staged spot while the corner origin is
    /// live and un-struck, so the delivery picker's "genuinely standing
    /// there, stationary" gate can actually be met.
    pub corner_short_option: Option<u32>,
    /// realism-bug (2026-07-28): the throw-in's staged close-support
    /// receiver — the specifically prepared, marker-evasion-adjusted
    /// option from `throw_in_shape_targets`. Read by the pass evaluator's
    /// selection multiplier so the thrower actually prefers the receiver
    /// the shape mechanism just spent the dead-ball window positioning,
    /// instead of the fully generic evaluator ignoring that prep and
    /// reaching for whichever teammate scores highest by open-play
    /// criteria (which skewed noticeably longer than real throw-ins).
    pub throw_in_preferred_receiver: Option<u32>,
    /// Set at pass-kick. Lives for the pass window (~220 ticks) and the
    /// offside resolver fires the call only when the receiver becomes
    /// active (touches the ball or claims). Cleared on resolution,
    /// opponent touch, or expiry.
    pub offside_snapshot: Option<OffsideSnapshot>,

    /// Origin of the most-recent live pass (passer's position when the
    /// pass was emitted). Read by the pass-completion classifier to
    /// decide if the pass was progressive / cross / box-entry. None
    /// outside an active pass window.
    pub pending_pass_origin: Option<Vector3<f32>>,
    /// Intended target position of the most-recent live pass. Cleared
    /// alongside `pending_pass_passer`.
    pub pending_pass_target: Option<Vector3<f32>>,
    /// Pass was emitted from the wide channel toward the box — flagged
    /// at emit-time so the completion classifier can credit
    /// `crosses_completed` when the same pass is received.
    pub pending_pass_was_cross: bool,

    /// Snapshot of the most recently *completed* pass — populated by
    /// `credit_completed_pass` AFTER it bumps `passes_completed` and
    /// BEFORE it clears `pending_pass_*`. The shot-handler key-pass
    /// linker reads these (rather than `pending_pass_*` which the
    /// completion path nulls out) so a receive-then-shoot sequence
    /// still credits the assister with a key pass. None outside the
    /// shot-after-pass window.
    pub last_completed_pass_passer_id: Option<u32>,
    /// Where the most recent pass was struck from, and by which team.
    /// Unlike `previous_owner` (cleared mid-flight once the passer is far
    /// from the ball — which erases exactly the long deliveries), these
    /// survive until the next pass or reset, so a receiver can gate
    /// behaviour on how far the ball travelled (lay_off_on_long_ball).
    pub pass_origin_position: Option<Vector3<f32>>,
    /// The last delivery was a goalkeeper kick or a free-kick/goal-kick
    /// strike (2026-10-17). Scopes the open-field header work to goal kicks
    /// and set pieces, so wing switches are untouched.
    pub set_piece_lofted: bool,
    /// Tick the last pass was struck and who it was aimed at (survives the
    /// claim, unlike `pass_target_player_id`): gives a receiver's decisions a
    /// stable per-reception key (long-shot draw, box_attack.rs).
    pub last_pass_tick: u64,
    pub last_pass_to: Option<u32>,
    pub pass_origin_team: Option<u32>,
    pub last_completed_pass_receiver_id: Option<u32>,
    pub last_completed_pass_tick: u64,

    /// Opponents that were within the pressing radius of the passer at
    /// pass-emit time. Read by the interception handler to credit a
    /// successful pressure when their close-range presence forced the
    /// turnover. Capped at 4 entries — the count of "real" pressers in
    /// any single moment is small. Cleared at pass-completion or
    /// pass-window expiry.
    pub pressers_at_pass: [u32; 4],
    pub pressers_at_pass_count: u8,

    /// Most-recent shot's xG and shooter id, used to credit the
    /// conceding goalkeeper with `xg_prevented` when the shot is saved
    /// (positive credit) or scored (negative credit). Cleared on
    /// resolution (save / goal / wide / over) and on any non-shot
    /// ownership change.
    pub last_shot_xg: f32,
    pub last_shot_shooter_id: Option<u32>,
    /// A penalty whose outcome the conversion model already decided:
    /// (taker id, is_goal). The ball is a real strike from the spot; this
    /// only tells the keeper-contact code that a decided GOAL cannot be
    /// caught/parried (it is placed out of his reach) and a decided SAVE
    /// is aimed at him and always saved. Cleared by any new shot.
    pub penalty_decided: Option<(u32, bool)>,
    /// A first-touch roll still owed by a receiver whose pass-target claim
    /// happened while the ball was out of his reach: (receiver, passer,
    /// tick). The roll is made when he actually gets to the ball (ordinary
    /// nearby claim); a failed touch then happens where the ball is.
    pub first_touch_owed: Option<(u32, u32, u64)>,
    /// Qualified assister of the most recent shot — the passer whose
    /// completed pass the shooter received inside the key-pass window
    /// (~3 s) before shooting. None for solo runs / long carries, so a
    /// goal from this shot only records an assist when the reception
    /// genuinely led to the finish. Same lifecycle as
    /// `last_shot_shooter_id`.
    pub last_shot_assister_id: Option<u32>,

    /// Tick of the most recent live rebound — a dangerous GK parry or
    /// a loose shot-block deflection that left the ball contestable in
    /// front of goal. Read by the team shot gate: within the rebound
    /// window (~3 s) the team-level shot SPACING and build-up gates
    /// are suspended so the box scramble / tap-in — one of football's
    /// core goal patterns — can actually fire. The per-possession shot
    /// cap (2) still rules out machine-gun scrambles. 0 = no rebound.
    pub last_rebound_tick: u64,

    /// Last meaningful giveaway: the player who lost possession via a
    /// misplaced pass that was intercepted by an opponent. Read by the
    /// "errors leading to shot/goal" linker — when an opponent shoots
    /// within the response window after this is stamped, the giver is
    /// charged with the error.
    pub last_giveaway_player_id: Option<u32>,
    pub last_giveaway_team_id: Option<u32>,
    pub last_giveaway_tick: u64,
    /// Defensive zone the giveaway happened in (from the giver's
    /// perspective). Lets the goal handler credit
    /// `errors_to_goal_own_box` when an opponent converts a giveaway
    /// from inside the giver's own box.
    pub last_giveaway_was_own_box: bool,
    /// Player charged with `errors_leading_to_shot` for the shot
    /// currently in flight. Held from shoot-time until the shot
    /// resolves; if the shot becomes a goal we also bump
    /// `errors_leading_to_goal` on this player.
    pub pending_error_to_shot_player_id: Option<u32>,

    /// Carry tracking. `carry_owner` is the player currently dribbling /
    /// running with the ball; `carry_start_position` is where the carry
    /// began. Evaluated when the carry ends (owner change / shot / pass)
    /// to credit progressive carries and box entries.
    pub carry_owner: Option<u32>,
    pub carry_start_position: Vector3<f32>,
}

/// Projection of a shot at the moment it's taken. The `PreparingForSave`
/// and `Catching` goalkeeper states read this to know where the ball
/// will actually arrive rather than chasing its current position — a
/// diving keeper commits to a spot on the line, they don't track the
/// ball every frame.
/// A contested interception resolved at first trigger and applied when
/// the first defender's leg actually reaches the ball (so the ball keeps
/// flying until then). See `Ball::try_intercept`.
#[derive(Debug, Clone, Copy)]
pub struct PendingIntercept {
    pub apply_tick: u64,
    pub defender_id: u32,
    pub defender_team: u32,
    pub defender_side: Option<PlayerSide>,
    /// Ball velocity after the touch (xy u/tick, z m/tick).
    pub velocity: Vector3<f32>,
    /// The touch kills the ball and he takes ownership; otherwise it is
    /// a deflection and the ball is left loose.
    pub controlled: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ShotTarget {
    /// y-coordinate at which the shot is projected to cross the goal
    /// line, in field units. Falls outside the posts if the shot is
    /// going wide — the keeper should still attempt the save, the
    /// post-vs-net check happens in `check_goal`.
    pub goal_line_y: f32,
    /// z-coordinate (height) at projected crossing. Above `GOAL_HEIGHT`
    /// (2.44) is an over-the-bar ball the keeper shouldn't commit to.
    pub goal_line_z: f32,
    /// Goal the ball is heading for — left (x=0) or right (x=field_w).
    /// Used so the correct keeper reads the cache.
    pub defending_side: PlayerSide,
    /// Set when the shot took a deflection off a body in the lane.
    /// Catching/Diving states damp the save probability — the keeper
    /// was set for the original trajectory and the redirected ball is
    /// arriving on a new line they haven't committed to.
    pub deflected: bool,
    /// Latch: the physics-layer save (`try_save_shot`) has taken its
    /// one roll for this shot. That check runs every tick the ball is
    /// near the goal line; without the latch it compounded 2-3 rolls
    /// of up to 0.55 per shot ON TOP of the GK state machine's own
    /// per-tick rolls, driving per-shot conversion to ~2% (real ~12%).
    pub physics_save_rolled: bool,
    /// Set when this shot is a direct free kick (`reason == "FK_DIRECT"`).
    /// realism-bug (2026-07-30): `GoalkeeperCatchingState` steers straight
    /// at `goal_line_y` with a 1.6-2.6x speed boost for the ENTIRE shot
    /// flight (the state never exits while `cached_shot_target` is live).
    /// For a close-range shot that's a brief, realistic final adjustment
    /// — but a direct FK's long flight time (130-250u away, 400-800+ms)
    /// gives that boosted, perfectly-informed pursuit enough time to
    /// fully converge onto the exact target line regardless of where the
    /// keeper started, which measured as 0 goals across 120 direct-FK
    /// shots (real ~6.3% conversion). Real keepers commit to a dive
    /// within a short reactive window, not a sustained sprint chasing a
    /// known trajectory. `is_direct_fk` lets Catching's speed_boost be
    /// capped for this one shot type only — every other shot's save
    /// behaviour (including normal long-range open-play shots) is
    /// unchanged.
    pub is_direct_fk: bool,
    /// realism-bug (2026-07-31): the shot's own projected total flight
    /// duration in ticks (computed once at dispatch, same `ticks_to_goal`
    /// used for the `goal_line_y`/`goal_line_z` projection below). Used
    /// as a FIXED `expected_ticks` for the state-machine catch roll
    /// (`is_catch_successful`) on direct free kicks — a first attempt
    /// recomputed "ticks remaining" fresh on every evaluation instead,
    /// which over-counts: each tick re-derives a brand new probability
    /// decomposition assuming THAT tick starts a fresh n-roll sequence,
    /// so the per-tick rate climbs sharply as the remaining-count shrinks
    /// near arrival, stacking on top of everything already rolled in
    /// earlier ticks. A single constant fixed at dispatch time avoids
    /// that entirely — the same total is used for every evaluation of
    /// this shot, so however many times the function actually fires,
    /// the cumulative probability tracks the intended per-shot value
    /// instead of compounding past it. Unused (defaults to 0.0, inert)
    /// for every non-FK shot, which keeps its own fixed 3.0 constant.
    pub total_flight_ticks: f32,
    /// realism-bug (2026-08-02): the simulation tick this shot was
    /// dispatched (`MatchContext::current_tick()` at construction time).
    /// Used by the direct-FK reaction-delay gate in
    /// `GoalkeeperCatchingState::velocity()` to measure genuine elapsed
    /// flight time via tick-count diffing, NOT the ball's own current
    /// (decelerating) velocity — a shot regularly loses speed on
    /// approach, so re-deriving "ticks remaining" from live velocity each
    /// tick made the elapsed-time estimate shrink back down near arrival,
    /// intermittently re-triggering the reaction freeze at exactly the
    /// worst moment regardless of how small the delay constant was.
    /// Unused (defaults to 0, inert) for every non-FK shot.
    pub dispatch_tick: u64,
    /// Set when this "shot" is a deliberate concede-a-restart action
    /// (`reason == "FWD_RUN_SAFE_RESTART"`, 2026-08 dribble/pass/shot
    /// decision plan) rather than a genuine attempt on goal. Read by
    /// `try_block_shot` to nudge its outcome weights toward a real
    /// touch/deflection (the whole point of this action — get rid of
    /// the ball under pressure, not let the defender calmly control it)
    /// rather than the default distribution tuned for a contested shot
    /// at goal. Unused (defaults to `false`, inert) for every other
    /// shot type.
    pub is_safe_restart: bool,
    /// Distance (game units) from where the shot was struck to the centre of
    /// the goal it is heading for. Anchors the keeper's single save
    /// probability on the real save-rate-by-distance curve
    /// (`ball/ball/save.rs`).
    pub shot_distance: f32,
}

#[derive(Default, Clone)]
pub struct BallFlags {
    pub in_flight_state: usize,
    pub running_for_ball: bool,
}

impl BallFlags {
    pub fn reset(&mut self) {
        self.in_flight_state = 0;
        self.running_for_ball = false;
    }
}

impl Ball {
    pub fn with_coord(field_width: f32, field_height: f32) -> Self {
        let x = field_width / 2.0;
        let y = field_height / 2.0;

        Ball {
            position: Vector3::new(x, y, 0.0),
            start_position: Vector3::new(x, y, 0.0),
            field_width,
            field_height,
            velocity: Vector3::zeros(),
            center_field_position: x, // initial ball position = center field
            flags: BallFlags::default(),
            previous_owner: None,
            current_owner: None,
            take_ball_notified_players: Vec::new(),
            notification_cooldown: 0,
            notification_timeout: 0,
            last_boundary_position: None,
            unowned_stopped_ticks: 0,
            ownership_duration: 0,
            claim_cooldown: 0,
            claim_barred: None,
            pass_target_player_id: None,
            pending_pass_passer: None,
            pending_pass_set_tick: 0,
            recent_passers: VecDeque::with_capacity(5),
            contested_claim_count: 0,
            unowned_ticks: 0,
            stall_start_snapshot: None,
            goal_scored: false,
            kickoff_team_side: None,
            cached_landing_position: Vector3::new(x, y, 0.0),
            pending_set_piece_teleport: None,
            pending_player_jumps: Vec::new(),
            pending_corner_teleports: Vec::new(),
            pending_restart_teleports: Vec::new(),
            direct_fk_shot_in_flight: false,
            corner_deflection_guard_until_tick: 0,
            block_rolled: Vec::new(),
            block_roll_shot_tick: 0,
            intercept_attempted: Vec::new(),
            pending_intercept: None,
            corner_contest_resolved: true,
            last_aerial_contest_tick: 0,
            pending_corner_routine: None,
            owned_stuck_ticks: 0,
            owned_stuck_logged: false,
            stall_anchor_pos: Vector3::new(x, y, 0.0),
            stall_anchor_tick: 0,
            cached_shot_target: None,
            pending_save_credit: None,
            restart_taker_lock: None,
            restart_pending_taker: None,
            free_kick_decided: false,
            corner_short_option: None,
            throw_in_preferred_receiver: None,
            last_touch_player_id: None,
            last_touch_team_id: None,
            last_touch_tick: 0,
            last_touch_was_controlled: false,
            current_tick_cached: 0,
            pass_origin_restart: PassOriginRestart::OpenPlay,
            offside_snapshot: None,
            pending_pass_origin: None,
            pending_pass_target: None,
            pending_pass_was_cross: false,
            last_completed_pass_passer_id: None,
            pass_origin_position: None,
            set_piece_lofted: false,
            last_pass_tick: 0,
            last_pass_to: None,
            pass_origin_team: None,
            last_completed_pass_receiver_id: None,
            last_completed_pass_tick: 0,
            pressers_at_pass: [0; 4],
            pressers_at_pass_count: 0,
            last_shot_xg: 0.0,
            last_shot_shooter_id: None,
            penalty_decided: None,
            first_touch_owed: None,
            last_shot_assister_id: None,
            last_rebound_tick: 0,
            last_giveaway_player_id: None,
            last_giveaway_team_id: None,
            last_giveaway_tick: 0,
            last_giveaway_was_own_box: false,
            pending_error_to_shot_player_id: None,
            carry_owner: None,
            carry_start_position: Vector3::new(x, y, 0.0),
        }
    }

    /// Straight-line distance (game units) from the ball to the centre of
    /// the goal defended by `defending_side`.
    pub fn distance_to_goal(&self, defending_side: PlayerSide) -> f32 {
        let gx = match defending_side {
            PlayerSide::Left => 0.0,
            PlayerSide::Right => self.field_width,
        };
        let dx = self.position.x - gx;
        let dy = self.position.y - self.field_height * 0.5;
        (dx * dx + dy * dy).sqrt()
    }

    /// The ONE way a defender's touch (leg tackle, block, push-out)
    /// turns a ball into a loose ball with a given velocity. Shared so
    /// tackles and blocks cannot drift apart.
    ///
    /// * The toucher becomes `previous_owner`/last toucher; nobody owns
    ///   the ball, all pass/shot bookkeeping is cleared.
    /// * If the released ball heads at the toucher's OWN goal mouth it
    ///   becomes a live deflected shot at that goal, so the normal
    ///   keeper/save/goal logic resolves it — there is no own-goal dice.
    /// * If it is leaving the pitch, instant re-claims (the toucher, the
    ///   man he just dispossessed, a covering player) are held off so
    ///   the ball actually crosses the line and produces the restart.
    pub fn release_after_touch(
        &mut self,
        toucher_id: u32,
        toucher_team: u32,
        toucher_side: Option<PlayerSide>,
        velocity: Vector3<f32>,
        tick: u64,
    ) {
        let ball_pos = self.position;
        self.velocity = velocity;
        self.previous_owner = Some(toucher_id);
        self.current_owner = None;
        self.pass_target_player_id = None;
        self.carry_owner = None;
        self.ownership_duration = 0;
        self.clear_pending_pass_metadata();
        self.clear_pass_history();
        self.flags.in_flight_state = 6;
        self.cached_shot_target = None;
        self.record_touch(toucher_id, toucher_team, tick, false);

        if let Some(side) = toucher_side {
            let own_goal_x = match side {
                PlayerSide::Left => 0.0,
                PlayerSide::Right => self.field_width,
            };
            let dx = own_goal_x - ball_pos.x;
            let heading_at_goal = (dx > 0.0 && velocity.x > 0.1) || (dx < 0.0 && velocity.x < -0.1);
            if heading_at_goal {
                let ticks_to_goal = (dx / velocity.x).abs().max(1.0);
                let goal_line_y = ball_pos.y + velocity.y * ticks_to_goal;
                if (goal_line_y - self.field_height * 0.5).abs() < 60.0 {
                    let g = crate::r#match::engine::physics::GRAVITY_PER_TICK;
                    let goal_line_z = (ball_pos.z + velocity.z * ticks_to_goal
                        - 0.5 * g * ticks_to_goal * ticks_to_goal)
                        .max(0.0);
                    self.cached_shot_target = Some(ShotTarget {
                        goal_line_y,
                        goal_line_z,
                        defending_side: side,
                        deflected: true,
                        physics_save_rolled: false,
                        is_direct_fk: false,
                        total_flight_ticks: ticks_to_goal,
                        dispatch_tick: tick,
                        is_safe_restart: false,
                        shot_distance: (ball_pos.x - own_goal_x).abs(),
                    });
                }
            }
        }

        if self.cached_shot_target.is_none() && self.is_exiting_boundary_soon() {
            self.flags.in_flight_state = 30;
            self.corner_deflection_guard_until_tick = tick + 30;
        }
    }

    /// True while this player, having lost a duel, may not reclaim the ball.
    pub fn is_claim_barred(&self, id: u32) -> bool {
        matches!(self.claim_barred, Some((p, until)) if p == id && self.current_tick_cached <= until)
    }

    /// Record a meaningful touch. Drives restart resolution. `controlled`
    /// distinguishes a clean reception from a deflection / failed save.
    pub fn record_touch(&mut self, player_id: u32, team_id: u32, tick: u64, controlled: bool) {
        // A touch by anyone other than the staged taker voids the
        // pending-restart marker (the restart was stolen / re-staged).
        if self.restart_pending_taker.is_some() && self.restart_pending_taker != Some(player_id) {
            self.restart_pending_taker = None;
        }
        // Any touch by a different player releases the restart taker's
        // same-touch lock (§9.4.1).
        if self.restart_taker_lock.is_some() && self.restart_taker_lock != Some(player_id) {
            self.restart_taker_lock = None;
        }
        self.last_touch_player_id = Some(player_id);
        self.last_touch_team_id = Some(team_id);
        self.last_touch_tick = tick;
        self.last_touch_was_controlled = controlled;
    }

    /// How long after a controlled touch the toucher's team is still
    /// treated as "in possession" of a ball nobody owns right now
    /// (~3s at 100 ticks/s). Measured 2026-09-25: of the ticks where no
    /// owner/passer/previous-owner resolved, 96.4% were within 3s of a
    /// controlled touch and only 0.1% were older.
    pub const POSSESSION_TOUCH_WINDOW_TICKS: u64 = 300;

    /// The player whose team should be treated as in possession of a ball
    /// that currently has no owner: the last CONTROLLED touch (claim,
    /// interception, reception — not a block/deflection/failed save), if
    /// recent. A struck pass, a clearance in flight, or a carry that has
    /// outrun its owner is still that team's ball; only an uncontrolled
    /// or long-stale last touch is a genuine loose ball.
    pub fn possessing_touch(&self) -> Option<u32> {
        if self.last_touch_was_controlled
            && self.current_tick_cached.saturating_sub(self.last_touch_tick)
                <= Self::POSSESSION_TOUCH_WINDOW_TICKS
        {
            self.last_touch_player_id
        } else {
            None
        }
    }

    /// Clear the offside snapshot. Called on opponent touch, claim, foul,
    /// or pass expiry.
    pub fn clear_offside_snapshot(&mut self) {
        self.offside_snapshot = None;
    }

    /// Force the ball into a clean dead-ball restart state. Centralises
    /// the flag clearing that every set-piece restart (corner / goal
    /// kick / throw-in / kickoff after goal) used to do by hand,
    /// dropping stale open-play metadata so a shot/pass that was in
    /// flight when the ball went dead cannot leak across the restart.
    ///
    /// This is the canonical "ball just went dead — reset everything
    /// open-play touched" helper. New restart paths should call this
    /// rather than zeroing individual fields, so a future field added
    /// to the open-play set is reset automatically.
    pub fn clear_open_play_metadata(&mut self) {
        self.cached_shot_target = None;
        self.pass_target_player_id = None;
        self.pending_pass_passer = None;
        self.pending_pass_origin = None;
        self.pending_pass_target = None;
        self.pending_pass_was_cross = false;
        self.offside_snapshot = None;
        self.pending_save_credit = None;
        self.pending_error_to_shot_player_id = None;
        self.last_shot_xg = 0.0;
        self.last_shot_shooter_id = None;
        self.last_shot_assister_id = None;
    }

    /// Soft invariant check on the ball's lifecycle flags. Returns the
    /// first violation as `Err(msg)` so debug builds and tests can
    /// assert the ball never enters a contradictory state. Production
    /// callers ignore the result — the cost is a handful of field
    /// reads.
    ///
    /// Invariants checked:
    ///   * Open-play shot metadata implies a previous owner (someone
    ///     fired the shot).
    ///   * Pending save credit references a real shooter id (so the
    ///     stat dispatch can fold the on-target back to a shot taker).
    ///   * A pass target id implies a passer id was set when the pass
    ///     was launched (else the receive-classifier has nothing to
    ///     pair the completion to).
    ///   * Ball/owner position coordinates are finite — non-finite x/y/z
    ///     leak into distance comparisons and trigger
    ///     `partial_cmp().unwrap()` panics in sort paths.
    ///   * On a dead-ball restart (corner / goal kick / throw-in /
    ///     free kick / penalty), open-play metadata (cached shot,
    ///     pending pass envelope, save credit, offside snapshot) must
    ///     be cleared — otherwise a shot that was in flight when the
    ///     ball went dead can leak across the restart and credit
    ///     phantom stats.
    ///   * Pending shot xG implies a shooter id (paired metadata,
    ///     consumed together).
    ///   * Pending pass envelope is coherent: a passer implies an
    ///     origin and target position.
    ///   * Carry tracking is consistent: a carrying owner means the
    ///     current owner matches the carrier.
    pub fn check_invariants(&self) -> Result<(), &'static str> {
        if self.cached_shot_target.is_some() && self.previous_owner.is_none() {
            return Err("cached_shot_target without previous_owner");
        }
        if let Some((_keeper, shooter)) = self.pending_save_credit {
            if shooter == 0 {
                return Err("pending_save_credit shooter id is sentinel zero");
            }
        }
        if self.pass_target_player_id.is_some() && self.pending_pass_passer.is_none() {
            return Err("pass_target without pending_pass_passer");
        }
        // Non-finite coordinates leak into distance comparisons and
        // trigger `partial_cmp().unwrap()` panics in nearby/sort paths.
        if !self.position.x.is_finite()
            || !self.position.y.is_finite()
            || !self.position.z.is_finite()
        {
            return Err("ball position has non-finite coordinate");
        }
        if !self.velocity.x.is_finite()
            || !self.velocity.y.is_finite()
            || !self.velocity.z.is_finite()
        {
            return Err("ball velocity has non-finite coordinate");
        }
        // Dead-ball restart cleanliness — any restart origin must drop
        // open-play metadata.
        if matches!(
            self.pass_origin_restart,
            PassOriginRestart::Corner
                | PassOriginRestart::GoalKick
                | PassOriginRestart::ThrowIn
                | PassOriginRestart::Penalty
        ) {
            if self.cached_shot_target.is_some() {
                return Err("dead-ball restart with leftover cached_shot_target");
            }
            if self.pending_save_credit.is_some() {
                return Err("dead-ball restart with leftover pending_save_credit");
            }
            if self.offside_snapshot.is_some() {
                return Err("dead-ball restart with leftover offside_snapshot");
            }
        }
        // Pending shot xG and shooter id are kept in lock-step.
        if self.last_shot_xg > 0.0 && self.last_shot_shooter_id.is_none() {
            return Err("last_shot_xg without last_shot_shooter_id");
        }
        // Pending pass envelope: any leg must imply the rest.
        if self.pending_pass_passer.is_some()
            && (self.pending_pass_origin.is_none() || self.pending_pass_target.is_none())
        {
            return Err("pending_pass_passer without origin/target metadata");
        }
        // Carry tracking — a current carrier must match the ball owner.
        if let (Some(carrier), Some(owner)) = (self.carry_owner, self.current_owner) {
            if carrier != owner {
                return Err("carry_owner disagrees with current_owner");
            }
        }
        Ok(())
    }
}

#[allow(dead_code, unused_imports)]
mod offside_snapshot_tests {
    use super::*;

    fn snap_left(receiver_x: f32, ball_x: f32, second_last: f32) -> OffsideSnapshot {
        OffsideSnapshot {
            origin: PassOriginRestart::OpenPlay,
            passer_id: 1,
            passer_side: PlayerSide::Left,
            receiver_id: 2,
            ball_x_at_kick: ball_x,
            second_last_defender_x: second_last,
            receiver_x_at_kick: receiver_x,
            receiver_y_at_kick: 200.0,
            set_tick: 0,
        }
    }

    #[test]
    fn left_attacker_beyond_second_last_is_offside() {
        // Receiver ahead of ball AND past the second-last defender.
        let snap = snap_left(700.0, 600.0, 680.0);
        assert!(snap.is_offside());
    }

    #[test]
    fn left_attacker_behind_ball_not_offside() {
        // Receiver is behind the ball — offside cannot occur.
        let snap = snap_left(500.0, 600.0, 680.0);
        assert!(!snap.is_offside());
    }

    #[test]
    fn left_attacker_level_with_defender_not_offside() {
        // Within tolerance — onside.
        let snap = snap_left(681.0, 600.0, 680.0);
        assert!(!snap.is_offside());
    }

    #[test]
    fn restart_origins_offside_exempt() {
        assert!(PassOriginRestart::GoalKick.is_offside_exempt());
        assert!(PassOriginRestart::Corner.is_offside_exempt());
        assert!(PassOriginRestart::ThrowIn.is_offside_exempt());
        assert!(!PassOriginRestart::OpenPlay.is_offside_exempt());
        assert!(!PassOriginRestart::FreeKick.is_offside_exempt());
    }
}

impl Ball {
    /// Update cached landing position. Call after physics changes position/velocity.
    #[inline]
    pub fn update_landing_cache(&mut self) {
        self.cached_landing_position = self.calculate_landing_position();
    }

    pub fn update(
        &mut self,
        context: &mut MatchContext,
        players: &[MatchPlayer],
        tick_context: &GameTickContext,
        events: &mut EventCollection,
    ) {
        self.current_tick_cached = context.current_tick();

        // Decrement claim cooldown
        if self.claim_cooldown > 0 {
            self.claim_cooldown -= 1;
        }

        self.update_velocity();

        self.try_intercept(context, players, events);
        self.try_block_shot(context, players, events);
        self.resolve_shot_at_keeper(context, players, events);
        self.try_notify_standing_ball(players, events);

        // NUCLEAR OPTION: Force claiming if ball unowned and stopped for too long
        self.force_claim_if_deadlock(players, events);

        // Unconditional unowned safety net - forces nearest players to TakeBall
        self.force_takeball_if_unowned_too_long(players, events);
        // `detect_owned_stuck` was too sensitive — it fired on legitimate
        // possession play (defender holding in back line for 6-12s is
        // normal). `detect_position_stall` is the stricter signal: ball
        // hasn't moved ANYWHERE in 1000 ticks, regardless of who owns
        // it. That's a real stall.
        self.detect_position_stall(players);

        let diag_prev = (self.current_owner, self.pass_target_player_id);
        self.process_ownership(context, players, events);
        self.cross_touch_diag(context, players, diag_prev);
        self.tick_carry_tracker(events);

        // Move ball FIRST, then check goal/boundary on new position
        self.move_to(tick_context);
        self.check_goal(context, players, events);
        self.check_over_goal(context, players, events);
        self.check_wide_of_goal(context, players, events);
        self.check_throw_in(context, players, events);
        self.expire_offside_snapshot(context);
        self.update_landing_cache();
    }

    /// TEMP diagnostic (match-logs): a player gains the ball while it is in the air.
    #[allow(unused_variables)]
    fn cross_touch_diag(&self, context: &MatchContext, players: &[MatchPlayer], prev: (Option<u32>, Option<u32>)) {
        #[cfg(feature = "match-logs")]
        {
            if self.current_owner != prev.0 && self.position.z >= 0.7 {
                if let Some(id) = self.current_owner {
                    if let Some(p) = players.iter().find(|p| p.id == id) {
                        let dg = [0.0_f32, self.field_width].iter().map(|gx| {
                            let dx = self.position.x - gx; let dy = self.position.y - self.field_height * 0.5;
                            (dx * dx + dy * dy).sqrt()
                        }).fold(f32::MAX, f32::min);
                        let hd = super::contact::can_header_in(p, self.position, players);
                        let opp = super::contact::opponent_near_ball(players, p.team_id, self.position);
                        crate::match_log_info!(
                            "CROSSTOUCH t={} id={} intended={} z={:.2} dg={:.0} canhead={} opp_near={} restart={:?} gk={} cooldown_ok={}",
                            context.total_match_time, id, prev.1 == Some(id), self.position.z, dg, hd, opp,
                            self.pass_origin_restart, p.tactical_position.current_position.is_goalkeeper(),
                            self.current_tick_cached.saturating_sub(self.last_aerial_contest_tick) >= 60
                        );
                    }
                }
            }
        }
    }

    /// Light update: full ball logic but reads owner position from players slice directly.
    pub fn update_light(
        &mut self,
        context: &mut MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
    ) {
        self.current_tick_cached = context.current_tick();

        if self.claim_cooldown > 0 {
            self.claim_cooldown -= 1;
        }

        self.update_velocity();
        self.try_intercept(context, players, events);
        self.try_block_shot(context, players, events);
        self.resolve_shot_at_keeper(context, players, events);
        let diag_prev = (self.current_owner, self.pass_target_player_id);
        self.process_ownership(context, players, events);
        self.cross_touch_diag(context, players, diag_prev);
        self.tick_carry_tracker(events);

        // Move ball: find owner position from players slice directly
        self.move_to_with_players(players);
        self.check_goal(context, players, events);
        self.check_over_goal(context, players, events);
        self.check_wide_of_goal(context, players, events);
        self.check_throw_in(context, players, events);
        self.expire_offside_snapshot(context);
        self.update_landing_cache();
    }

    /// Calculate where an aerial ball will land (when z reaches 0).
    /// Uses projectile motion: z(t) = h + vz·t − ½g·t² = 0, solving for
    /// the positive root. Ignores air drag — close enough for chase
    /// positioning, and erring long is better than erring short.
    ///
    /// Units are ticks, not seconds: position integration is
    /// `position += velocity` per tick (no dt scaling), while gravity
    /// applies `velocity.z += -GRAVITY * TICK_SECONDS²` per tick (real
    /// meters-per-tick storage — see `update_velocity`'s 2026-08 height-
    /// physics rewrite comment for the full derivation). So the
    /// effective per-tick² gravity is `9.81 * 0.01² ≈ 0.000981`, and the
    /// resulting `time_to_ground` comes out in ticks — which matches
    /// the horizontal integration `x += vx` per tick.
    pub fn calculate_landing_position(&self) -> Vector3<f32> {
        if self.position.z <= 0.1 || self.current_owner.is_some() {
            return self.position;
        }

        let g_per_tick = crate::r#match::engine::physics::GRAVITY_PER_TICK;
        let vz = self.velocity.z;
        let h = self.position.z;

        // Positive root of ½g·t² − vz·t − h = 0
        let discriminant = vz * vz + 2.0 * g_per_tick * h;
        let time_to_ground = (vz + discriminant.sqrt()) / g_per_tick;

        let landing_x = self.position.x + self.velocity.x * time_to_ground;
        let landing_y = self.position.y + self.velocity.y * time_to_ground;

        let clamped_x = landing_x.clamp(0.0, self.field_width);
        let clamped_y = landing_y.clamp(0.0, self.field_height);

        Vector3::new(clamped_x, clamped_y, 0.0)
    }

    /// Where (and in how many ticks) an unowned, lofted ball next passes
    /// through head height (`HEAD_Z`) on its way down. Uses the same drag
    /// and gravity as `update_velocity`. `None` when the ball is not a
    /// real lofted delivery (never rises above 2.6 m, is owned, or does
    /// not come down to head height within `MAX_TICKS`). A ball that is
    /// already inside the heading band and descending returns its current
    /// position at tick 0.
    pub fn predict_head_point(&self) -> Option<(Vector3<f32>, u32)> {
        if self.current_owner.is_some() {
            return None;
        }
        Self::predict_head_point_from(self.position, self.velocity)
    }

    /// Same as [`Ball::predict_head_point`] from a raw position/velocity
    /// (the per-tick ball snapshot players read).
    pub fn predict_head_point_from(
        start_pos: Vector3<f32>,
        start_vel: Vector3<f32>,
    ) -> Option<(Vector3<f32>, u32)> {
        const HEAD_Z: f32 = 2.0;
        const MAX_TICKS: u32 = 220;
        const AIR_DRAG: f32 = 0.04;
        const BALL_MASS: f32 = 0.43;
        if start_pos.z < 1.6 {
            return None;
        }
        let g = crate::r#match::engine::physics::GRAVITY_PER_TICK;
        let mut pos = start_pos;
        let mut vel = start_vel;
        let mut peak = pos.z;
        if pos.z <= HEAD_Z && vel.z < 0.0 {
            return Some((pos, 0));
        }
        for t in 1..=MAX_TICKS {
            let n = vel.norm();
            if n > 0.1 {
                let a = -AIR_DRAG * n / BALL_MASS * 0.016;
                vel.x += vel.x * a;
                vel.y += vel.y * a;
                vel.z += vel.z * a;
            }
            vel.z -= g;
            pos += vel;
            peak = peak.max(pos.z);
            if pos.z < 0.2 {
                return None;
            }
            if vel.z < 0.0 && pos.z <= HEAD_Z && peak >= 2.3 {
                return Some((pos, t));
            }
        }
        None
    }

    /// Check if the ball is aerial (in the air above player reach)
    pub fn is_aerial(&self) -> bool {
        const PLAYER_REACH_HEIGHT: f32 = 2.3;
        self.position.z > PLAYER_REACH_HEIGHT && self.velocity.z.abs() > 0.1
    }

    pub fn is_stands_outside(&self) -> bool {
        self.is_ball_outside()
            && self.velocity.norm_squared() < 0.25 // 0.5^2, allow tiny velocities from physics
            && self.current_owner.is_none()
    }

    pub fn is_ball_stopped_on_field(&self) -> bool {
        !self.is_ball_outside()
            && self.velocity.norm_squared() < 6.25 // 2.5^2, catch slow rolling balls that need claiming
            && self.current_owner.is_none()
    }

    pub fn is_ball_outside(&self) -> bool {
        self.position.x <= 0.0
            || self.position.x >= self.field_width
            || self.position.y <= 0.0
            || self.position.y >= self.field_height
    }

    /// Lightweight movement: just apply velocity to position (no ownership logic)
    pub fn apply_movement(&mut self) {
        self.position.x += self.velocity.x;
        self.position.y += self.velocity.y;
        self.position.z += self.velocity.z;
        if self.position.z < 0.0 {
            self.position.z = 0.0;
        }
    }

    pub fn reset(&mut self) {
        self.position.x = self.start_position.x;
        self.position.y = self.start_position.y;
        self.position.z = 0.0;

        self.velocity = Vector3::zeros();

        self.current_owner = None;
        self.previous_owner = None;
        self.ownership_duration = 0;
        self.claim_cooldown = 0;

        self.flags.reset();
        self.pass_target_player_id = None;
        self.clear_pass_history();
        self.contested_claim_count = 0;
        self.unowned_ticks = 0;
        self.cached_landing_position = self.position;
        self.pending_set_piece_teleport = None;
        self.pending_player_jumps.clear();
        self.pending_corner_teleports.clear();
        self.pending_restart_teleports.clear();
        self.owned_stuck_ticks = 0;
        self.owned_stuck_logged = false;
        self.stall_anchor_pos = self.position;
        self.stall_anchor_tick = 0;
        self.cached_shot_target = None;
        self.pending_save_credit = None;
        self.last_touch_player_id = None;
        self.last_touch_team_id = None;
        self.last_touch_tick = 0;
        self.last_touch_was_controlled = false;
        self.pass_origin_restart = PassOriginRestart::OpenPlay;
        self.restart_taker_lock = None;
        self.restart_pending_taker = None;
        self.offside_snapshot = None;
        self.last_completed_pass_passer_id = None;
        self.last_completed_pass_receiver_id = None;
        self.last_completed_pass_tick = 0;
        self.pass_origin_position = None;
        self.set_piece_lofted = false;
        self.pass_origin_team = None;
    }

    /// Snapshot the most-recent completed pass so the shot-handler
    /// key-pass linker can credit the passer when the receiver
    /// shoots within the key-pass window. Called from
    /// `credit_completed_pass` *before* `clear_pending_pass_metadata`
    /// nulls out the live pass envelope.
    #[inline]
    pub fn record_completed_pass(&mut self, passer_id: u32, receiver_id: u32, tick: u64) {
        self.last_completed_pass_passer_id = Some(passer_id);
        self.last_completed_pass_receiver_id = Some(receiver_id);
        self.last_completed_pass_tick = tick;
    }

    pub fn clear_player_reference(&mut self, player_id: u32) {
        if self.current_owner == Some(player_id) {
            self.current_owner = None;
            self.ownership_duration = 0;
        }
        if self.previous_owner == Some(player_id) {
            self.previous_owner = None;
        }
        if self.pass_target_player_id == Some(player_id) {
            self.pass_target_player_id = None;
        }
        if self.last_completed_pass_passer_id == Some(player_id)
            || self.last_completed_pass_receiver_id == Some(player_id)
        {
            self.last_completed_pass_passer_id = None;
            self.last_completed_pass_receiver_id = None;
        }
        self.take_ball_notified_players
            .retain(|&id| id != player_id);
        self.recent_passers.retain(|&id| id != player_id);
    }

    /// Record a passer in the recent passers ring buffer.
    /// Skips consecutive duplicates and caps at 5 entries.
    pub fn record_passer(&mut self, passer_id: u32) {
        // Skip consecutive duplicates
        if self.recent_passers.back() == Some(&passer_id) {
            return;
        }
        if self.recent_passers.len() >= 5 {
            self.recent_passers.pop_front();
        }
        self.recent_passers.push_back(passer_id);
    }

    /// Clear the recent passers history (e.g. on tackles, interceptions, clearances).
    pub fn clear_pass_history(&mut self) {
        self.recent_passers.clear();
    }

    /// Clear the pass-window metadata used by the pass-completion classifier
    /// and the key-pass linker. Called whenever the live pass is no longer
    /// in flight (claim, interception, expiry, set-piece restart).
    #[inline]
    pub fn clear_pending_pass_metadata(&mut self) {
        self.pending_pass_passer = None;
        self.pending_pass_origin = None;
        self.pending_pass_target = None;
        self.pending_pass_was_cross = false;
    }

    /// Drop any in-flight shot metadata (xG / shooter id). Called once
    /// the shot resolves (save / goal / wide / over / opponent claim).
    #[inline]
    pub fn clear_shot_metadata(&mut self) {
        self.last_shot_xg = 0.0;
        self.last_shot_shooter_id = None;
        self.last_shot_assister_id = None;
    }

    /// Stamp the giveaway tracker for the player who just lost the ball
    /// via a misplaced pass / lost tackle / dispossession. Subsequent
    /// shot / goal events from the opposing team within the response
    /// window will be charged back as an error to this player. The
    /// `was_own_box` flag is read later by the goal handler to layer the
    /// own-box-extra penalty on top of `errors_leading_to_goal`.
    #[inline]
    pub fn stamp_giveaway(&mut self, player_id: u32, team_id: u32, tick: u64, was_own_box: bool) {
        self.last_giveaway_player_id = Some(player_id);
        self.last_giveaway_team_id = Some(team_id);
        self.last_giveaway_tick = tick;
        self.last_giveaway_was_own_box = was_own_box;
    }

    /// Drop the giveaway tracker — the response window has expired or
    /// the giver's team has recovered the ball.
    #[inline]
    pub fn clear_giveaway(&mut self) {
        self.last_giveaway_player_id = None;
        self.last_giveaway_team_id = None;
        self.last_giveaway_was_own_box = false;
    }

    /// Detect and resolve carry transitions. Called once per tick from
    /// `update` / `update_light`, after `process_ownership` has settled
    /// the current owner. When the owner changes (or goes None) we emit
    /// a `BallEvent::CarryEnded` for the previous carrier; the
    /// dispatcher classifies the carry and credits the carrier's stats.
    /// A new carry starts the moment ownership lands on a player.
    pub fn tick_carry_tracker(&mut self, events: &mut EventCollection) {
        match (self.carry_owner, self.current_owner) {
            (Some(prev), Some(curr)) if prev == curr => {
                // Same carrier — nothing to emit.
            }
            (Some(prev), _) => {
                // Carry ended (owner changed or went None).
                events.add_ball_event(BallEvent::CarryEnded(
                    prev,
                    self.carry_start_position,
                    self.position,
                ));
                self.carry_owner = self.current_owner;
                self.carry_start_position = self.position;
            }
            (None, Some(curr)) => {
                // Carry begins.
                self.carry_owner = Some(curr);
                self.carry_start_position = self.position;
            }
            (None, None) => {}
        }
    }
}

#[cfg(test)]
mod completed_pass_tests {
    use super::*;

    #[test]
    fn record_completed_pass_populates_snapshot() {
        let mut ball = Ball::with_coord(840.0, 545.0);
        ball.record_completed_pass(7, 11, 1234);
        assert_eq!(ball.last_completed_pass_passer_id, Some(7));
        assert_eq!(ball.last_completed_pass_receiver_id, Some(11));
        assert_eq!(ball.last_completed_pass_tick, 1234);
    }

    #[test]
    fn clear_pending_pass_metadata_does_not_clear_completed_snapshot() {
        // Regression: the centralized completion path used to clear
        // pending_pass_passer immediately, leaving the shot-handler
        // key-pass linker without a passer to credit. The completed
        // snapshot survives the pending clear.
        let mut ball = Ball::with_coord(840.0, 545.0);
        ball.pending_pass_passer = Some(7);
        ball.pending_pass_set_tick = 100;
        ball.pending_pass_origin = Some(Vector3::new(50.0, 100.0, 0.0));
        ball.pending_pass_target = Some(Vector3::new(150.0, 100.0, 0.0));
        ball.pending_pass_was_cross = true;
        ball.record_completed_pass(7, 11, 200);
        ball.clear_pending_pass_metadata();
        assert!(ball.pending_pass_passer.is_none());
        assert!(ball.pending_pass_origin.is_none());
        assert!(ball.pending_pass_target.is_none());
        assert!(!ball.pending_pass_was_cross);
        // The completed snapshot stays — the key-pass linker reads it.
        assert_eq!(ball.last_completed_pass_passer_id, Some(7));
        assert_eq!(ball.last_completed_pass_receiver_id, Some(11));
        assert_eq!(ball.last_completed_pass_tick, 200);
    }

    #[test]
    fn clear_player_reference_drops_completed_pass_snapshot() {
        // If a player is removed (red card, sub), any completed-pass
        // metadata referencing them must be cleared so the next shot
        // doesn't credit a phantom key pass.
        let mut ball = Ball::with_coord(840.0, 545.0);
        ball.record_completed_pass(7, 11, 200);
        ball.clear_player_reference(7);
        assert!(ball.last_completed_pass_passer_id.is_none());
        assert!(ball.last_completed_pass_receiver_id.is_none());

        // Receiver removal also wipes (consistency).
        ball.record_completed_pass(7, 11, 300);
        ball.clear_player_reference(11);
        assert!(ball.last_completed_pass_passer_id.is_none());
        assert!(ball.last_completed_pass_receiver_id.is_none());
    }
}
