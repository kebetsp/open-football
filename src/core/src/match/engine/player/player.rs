use crate::club::player::events::PositionLoad;
use crate::club::player::traits::{BehavioralDirective, PlayerTrait};
use crate::r#match::PlayerMatchEndStats;
use crate::r#match::defenders::states::DefenderState;
use crate::r#match::engine::result::PlayerMatchPhysicalSnapshot;
use crate::r#match::engine::tactics::TacticalPositions;
use crate::r#match::events::EventCollection;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::goalkeepers::states::state::GoalkeeperState;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::player::memory::PlayerMemory;
use crate::r#match::player::state::{PlayerMatchState, PlayerState};
use crate::r#match::player::statistics::MatchPlayerStatistics;
use crate::r#match::player::waypoints::WaypointManager;
use crate::r#match::{GameTickContext, MatchContext, StateProcessingContext};
use crate::utils::DateUtils;
use crate::{
    PersonAttributes, Player, PlayerAttributes, PlayerFieldPositionGroup, PlayerPositionType,
    PlayerSkills,
};
use chrono::NaiveDate;
use nalgebra::Vector3;
use std::fmt::*;

#[cfg(debug_assertions)]
use log::debug;

/// Trigger half of a whole-team shape rule (plan Phase 5). Ball-lane
/// triggers are from the PLAYER's attacking perspective ("left" =
/// the manager's left wing), side-corrected each tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeTrigger {
    InPossession,
    OutOfPossession,
    BallLeft,
    BallCenter,
    BallRight,
    /// Score-state triggers (Phase 7.3), from this player's team's
    /// perspective: active while winning / level / losing.
    Leading,
    Level,
    Trailing,
    /// Elapsed-time trigger: active from this DISPLAYED minute onward
    /// (0-90 scale, mapped onto the engine's compressed match clock).
    AfterMinute(u32),
    /// Combined score x time (the wishlist #14 case: "when winning
    /// after 70 minutes, drop into a mid-block").
    LeadingAfterMinute(u32),
    TrailingAfterMinute(u32),
}

/// Action half of a shape rule — computed relative to the player's
/// `shape_base` and side, so no absolute-coordinate mirroring bugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeAction {
    /// Fullback inverts: step up ~16% of the pitch and halfway toward
    /// the central lane.
    StepIntoMidfield,
    /// Pin to the nearest touchline, slightly advanced.
    HoldWidth,
    /// Come infield ~55% toward the central lane, same depth.
    TuckInside,
    /// Drop ~12% of the pitch toward own goal, same lane.
    DropDeeper,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeRule {
    pub trigger: ShapeTrigger,
    pub action: ShapeAction,
}

impl ShapeRule {
    /// Target position for this rule's action, relative to `base`.
    pub fn target(
        &self,
        base: Vector3<f32>,
        side: PlayerSide,
        field_width: f32,
        field_height: f32,
    ) -> Vector3<f32> {
        let dir = match side {
            PlayerSide::Left => 1.0,
            PlayerSide::Right => -1.0,
        };
        let centre = field_height / 2.0;
        match self.action {
            ShapeAction::StepIntoMidfield => Vector3::new(
                (base.x + dir * field_width * 0.16).clamp(20.0, field_width - 20.0),
                base.y + (centre - base.y) * 0.5,
                0.0,
            ),
            ShapeAction::HoldWidth => {
                let ty = if base.y < centre { 28.0 } else { field_height - 28.0 };
                Vector3::new(
                    (base.x + dir * field_width * 0.05).clamp(20.0, field_width - 20.0),
                    ty,
                    0.0,
                )
            }
            ShapeAction::TuckInside => {
                Vector3::new(base.x, base.y + (centre - base.y) * 0.55, 0.0)
            }
            ShapeAction::DropDeeper => Vector3::new(
                (base.x - dir * field_width * 0.12).clamp(20.0, field_width - 20.0),
                base.y,
                0.0,
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatchPlayer {
    pub id: u32,
    pub position: Vector3<f32>,
    pub start_position: Vector3<f32>,
    pub attributes: PersonAttributes,
    pub team_id: u32,
    pub player_attributes: PlayerAttributes,
    pub skills: PlayerSkills,
    pub tactical_position: TacticalPositions,
    pub velocity: Vector3<f32>,
    pub side: Option<PlayerSide>,
    pub state: PlayerState,
    pub in_state_time: u64,
    pub statistics: MatchPlayerStatistics,
    pub use_extended_state_logging: bool,

    pub waypoint_manager: WaypointManager,

    pub memory: PlayerMemory,

    /// Accumulates fractional condition changes across ticks
    pub fatigue_accumulator: f32,

    /// Cached waypoint vectors (only changes on substitution/half-time swap)
    cached_waypoints: Vec<Vector3<f32>>,

    /// Signature moves (PPMs) — read by decision helpers to bias behaviour.
    pub traits: Vec<PlayerTrait>,

    /// Yellow cards accumulated in this match. 2 → red.
    pub yellow_cards: u8,
    /// Fouls committed in this match. Feeds end-of-match stats.
    pub fouls_committed: u8,
    /// Player has been sent off — skip state processing, treat as off field.
    pub is_sent_off: bool,
    /// Ticks remaining before this player may attempt another tackle.
    /// Decremented each tick in `update()`. Blocks Tackling-state entry
    /// via `can_attempt_tackle()`. Prevents the Tackling-state machine
    /// from re-firing attempts via Standing/Running/Covering/etc. paths
    /// within the same second, which would otherwise produce 40+ fouls
    /// per team in the first 5 minutes of every match.
    pub tackle_cooldown: u16,
    /// Tagged reason for the next Shoot event. Set by each transition
    /// point that routes into the Shooting state (e.g. "FWD_RUN_PRIO05",
    /// "FWD_POINT_BLANK", "MID_POINT_BLANK_RUN"). The Shooting state
    /// reads this and attaches it to the emitted Shoot event so the
    /// per-match shot-reason log shows which code path fired the shot.
    /// Cleared after consumption.
    pub pending_shot_reason: Option<&'static str>,
    /// Manager-issued per-match movement directive (at most one).
    /// Injected via the match API; checked by strategy states at their
    /// decision points. See `BehavioralDirective` in club/player/ability.
    /// May be swapped at runtime by a teammate trigger (see below);
    /// `directive_base` holds the static value it reverts to.
    pub behavioral_directive: Option<BehavioralDirective>,
    /// The statically-assigned directive (from the match API). The
    /// effective `behavioral_directive` reverts to this when a teammate
    /// trigger deactivates.
    pub directive_base: Option<BehavioralDirective>,
    /// Cross-player defensive assignment: press this opponent whenever
    /// they have the ball (velocity override in the state processor).
    pub press_target: Option<u32>,
    /// Cross-player defensive assignment: man-mark this opponent while
    /// our team is out of possession (goal-side stick, plus the press
    /// chase when they receive).
    pub mark_target: Option<u32>,
    /// Cross-player attacking trigger: when teammate `.0` has the ball,
    /// this player's effective directive becomes `.1` (evaluated per
    /// possession change in the engine loop).
    pub teammate_trigger: Option<(u32, BehavioralDirective)>,
    /// §11.5 kickoff short-pass window, in remaining ticks (0 = inactive).
    /// Set on the kickoff taker by `assign_kickoff`; while active and the
    /// player still owns the ball, the state processor forces the role's
    /// Passing state so the taker taps a short ball instead of setting
    /// off on a solo carry. Cleared in the engine loop the moment the
    /// taker releases the ball or the budget runs out.
    pub kickoff_pass_pending: u16,
    /// realism-bug (2026-07-25): opportunistic byline-run commitment, in
    /// remaining ticks (0 = inactive). Armed probabilistically (see
    /// `run.rs`) when a player gains fresh possession in a wide channel
    /// with room to advance — real football's winger-vs-isolated-back
    /// trigger, not something a single-tick value comparison could
    /// reliably sustain (the value function's own per-tick winner
    /// (`carry_candidates`) proved too fragile — a small geometry shift
    /// could flip it before a run ever developed). While active, the
    /// forward/midfielder Dribbling states bypass the generic exits that
    /// would otherwise cancel a genuine wide duel (dribble timeout,
    /// single-marker pass-out, premature should_cross) — mirroring the
    /// BylineAndCross DIRECTIVE's own already-proven bypass, just
    /// triggered for undirected play instead of an explicit instruction.
    /// Cleared on ball release, genuine two-man pressure, or budget
    /// expiry (engine loop, same pattern as the three fields above).
    pub byline_commitment_ticks: u16,
    /// realism-bug (2026-07-27): a genuine, clean 1v1-vs-GK opportunity had
    /// no code path at all that could choose "carry the ball around the
    /// keeper" — evaluate_forward_shot_decision only ever returns Shoot/
    /// Pass/Hold, and the existing GK-avoidance steering in
    /// ForwardDribblingState::velocity() only sidesteps to open a shooting
    /// ANGLE, never continues past him into the net. Same persistent-ticks
    /// pattern as byline_commitment_ticks: while active, ForwardDribblingState
    /// skips its normal shot-dispatch priorities and velocity() steers past
    /// the keeper's current position instead of at/around him for an angle.
    /// Cleared on ball release or on release-condition (carrier's position
    /// has advanced past the keeper toward goal).
    pub round_keeper_commitment_ticks: u16,
    /// realism-bug follow-up (2026-07-27): frozen at arm time (not
    /// recomputed live from the GK's current position). Raw tick trace
    /// showed the GK retreating toward his own goal in the SAME direction
    /// as the attacking carrier during the whole chase -- continuously
    /// re-targeting his live position meant the effective closing rate was
    /// only the DIFFERENCE of their speeds (~0.235u/tick measured, far
    /// below either player's own ~0.35u/tick absolute speed), so the gap
    /// never closed inside any reasonable ticks budget. A real striker
    /// commits to a route based on the keeper's position at the moment of
    /// decision, not a target that keeps receding as the keeper recovers.
    pub round_keeper_target_x: f32,
    pub round_keeper_target_y: f32,
    /// realism-bug (2026-07-27): one-shot-per-possession lock so the
    /// round-keeper probability roll (in the engine loop) fires at most
    /// once per continuous spell of ball ownership, not every tick the
    /// clean-1v1 condition happens to hold (which would compound a small
    /// per-tick probability into near-certainty over a multi-second
    /// breakaway). Reset to false whenever the player doesn't own the ball.
    pub gk1v1_decision_made: bool,
    /// realism-bug (2026-07-19): throw-in taker's forced-release window,
    /// in remaining ticks (0 = inactive). Same pattern as
    /// `kickoff_pass_pending` — a throw-in is a single discrete motion
    /// in real football (the ball leaves the hands immediately; Law 15
    /// also bars the thrower touching it again until another player
    /// does), but nothing previously stopped the thrower's normal
    /// open-play state machine from dribbling him forward with the ball
    /// for seconds after the restart. Set on the thrower when the
    /// set-piece teleport applies; while active and he still owns the
    /// ball, the state processor forces the role's Passing state.
    /// Cleared the moment he releases the ball or the budget runs out.
    pub throw_in_pass_pending: u16,
    /// realism-bug (2026-07-20): free-kick taker's forced-release window,
    /// in remaining ticks (0 = inactive). Same pattern as
    /// `throw_in_pass_pending` above. Only armed for the outcomes that
    /// leave the taker to his normal open-play state machine —
    /// ShortRoutine/Recycle and the >280u "Deep" fallthrough in
    /// `resolve_free_kick` — never for DirectShot/BoxDelivery, which
    /// already dispatch a discrete Shoot/PassTo event immediately and
    /// bypass his AI entirely. Without this, a deep or short-routine
    /// free kick left the taker owning the ball like any open-play
    /// possession, free to dribble forward solo — unrealistic for a
    /// restart real teams take short and circulate, not a solo carry.
    /// Set in `resolve_free_kick`; while active and he still owns the
    /// ball, the state processor forces the role's Passing state.
    /// Cleared the moment he releases the ball or the budget runs out.
    pub free_kick_pass_pending: u16,
    /// Realism-bug 2026-07-26: set by the general (non-corner) aerial
    /// contest resolver — cross / free-kick cross / long high pass — the
    /// instant it decides this player won the header. Consumed by
    /// `ForwardHeadingState`/`MidfielderHeadingState`/`DefenderState`'s
    /// header carve-out (same clean-contact-only roll the corner-contest
    /// carve-out already uses via `is_team_attacking_corner()`) so the
    /// discretely-decided win isn't re-rolled as a second full duel.
    /// Ticks-remaining counter (not a plain bool) so it self-clears via
    /// the same per-tick decrement loop as `kickoff_pass_pending` et al.
    /// instead of needing bespoke reset logic.
    pub aerial_contest_won: u8,
    /// Realism-bug 2026-07-26 (passing follow-up): the lead-adjusted aim
    /// point of a pass currently in flight toward this player. Set (on
    /// the RECEIVER, not the passer) by `handle_pass_to_event` at pass
    /// dispatch time, using the exact same `ideal_target` the ball
    /// itself is aimed at — closing a coordination gap: previously
    /// `Ball.pending_pass_target` had zero readers anywhere in a
    /// receiving player's own movement code, so a receiver's run during
    /// the ball's flight was entirely independent of where the ball was
    /// actually headed (confirmed by a full-tree grep before this fix).
    /// `None` when no pass is currently incoming. Read by
    /// `incoming_pass_awareness_velocity` (processor.rs) as a BLENDED
    /// nudge — not a hard override — toward the target, weighted by the
    /// receiver's own off_ball_attack composite (off_the_ball/
    /// anticipation/decisions/pace) so a sharp off-ball mover reads and
    /// meets a leading ball far better than a poor one, preserving
    /// exactly the skill differentiation the passer's side already has
    /// (`anticipation`-scaled `lead_fraction` in `handle_pass_to_event`)
    /// instead of flattening every receiver into an equally good one.
    pub incoming_pass_target: Option<Vector3<f32>>,
    /// Ticks remaining on `incoming_pass_target` (0 = inactive). Self-
    /// clears via the same per-tick decrement loop as
    /// `kickoff_pass_pending` et al., sized to the passing lead
    /// formula's own flight-time ceiling (`flight_time_est.clamp(8.0,
    /// 80.0)` in `handle_pass_to_event`) plus margin, so it naturally
    /// expires around when the ball would have arrived even if nothing
    /// else clears it first.
    pub incoming_pass_ticks: u16,
    /// "Link with X" pair preference (wishlist #5): passes toward this
    /// teammate get a success/EV bias in the pass evaluator, and the
    /// pair's chemistry is pinned high at kickoff. Set symmetrically on
    /// both players by the API layer.
    pub link_target: Option<u32>,
    /// "Block passes into X" (wishlist #8): this player is assigned to
    /// cut out passes to the named OPPONENT — opposing passers see a
    /// success/selection penalty on passes to that target while this
    /// interceptor is within range of them.
    pub intercept_target: Option<u32>,
    /// "Feed X" directed supply (wishlist #9): stamped on every
    /// TEAMMATE of the named target — passes to the target are
    /// preferred, long balls especially.
    pub supply_target: Option<u32>,
    /// "Push the keeper up for corners late on" (wishlist #16): set on
    /// the GK only. From this displayed minute, own-team corners
    /// teleport him into the opposition box (no state override) and a
    /// velocity override keeps him there through the delivery, then
    /// sprints him home. The manager's risk is real — the goal is
    /// genuinely open while he's up.
    pub gk_up_after_minute: Option<u32>,
    /// §13.4: genuine restart-retreat steering target (foul wall/box-clear,
    /// corner defensive block/attacking push, goal-kick recovery shape).
    /// Set instead of an instant position snap so the player walks there
    /// tick-by-tick from wherever they actually are — a fast restart can
    /// then genuinely catch them short. Only ever read/written during a
    /// `dead_ball_retreat_active` window by the engine's light retreat
    /// tick (`FootballEngine::apply_restart_retreat_tick`, tick.rs) —
    /// never by the normal state processor, so it can't interact with
    /// open-play movement. Force-cleared on every player the instant a
    /// retreat window ends (early or at the cap), so a stale target can
    /// never leak into open play.
    pub restart_retreat_target: Option<Vector3<f32>>,
    /// Manager-issued start-position anchors (per axis, first-half
    /// coordinates). `setup_player_on_field` overwrites start_position
    /// from the formation table, which silently discarded the API's
    /// startX/startY overrides — these survive setup and are mirrored
    /// at the halftime swap.
    pub start_anchor_x: Option<f32>,
    pub start_anchor_y: Option<f32>,
    /// Whole-team shape rules (plan Phase 5): each tick the engine loop
    /// evaluates the first rule whose trigger holds and moves this
    /// player's `start_position` (the shape anchor every positional
    /// state blends toward — never a hard snap) to the action's target;
    /// with no active rule it reverts to `shape_base`.
    pub shape_rules: Vec<ShapeRule>,
    /// The formation-table start position (anchors included) captured
    /// after setup and re-captured after the halftime swap — the revert
    /// target for shape rules.
    pub shape_base: Option<Vector3<f32>>,
    /// Team-wide directional attack bias: -1 = manager's left, 1 =
    /// right, 0 = centre. Pass-evaluator weight toward that lateral
    /// third ("build down the left").
    pub attack_bias: Option<i8>,

    /// Manager flag protecting this player from fatigue / development subs.
    /// Mirrored from `Player::is_force_match_selection` at squad-build time.
    pub is_force_match_selection: bool,

    /// Player's birth date, mirrored from the source `Player`. Read by the
    /// in-match substitution logic to apply age-appropriate protection
    /// thresholds for under-18 players (lower fatigue ceiling, condition
    /// floor that overrides the manager's force-selection flag).
    pub birth_date: NaiveDate,

    /// Match-time (in ms) at which the player entered the field.
    /// Starters are stamped with 0; substitutes are stamped with
    /// `context.total_match_time` at swap time. The end-of-match rating
    /// helper computes minutes-played as
    /// `(exit_or_now - entry_match_time_ms) / 60_000`, which fixes the
    /// previous behaviour where every player — even an 80th-minute sub —
    /// was credited with full match minutes.
    pub entry_match_time_ms: u64,
    /// Last tick at which a pressure event was credited for this
    /// player. Used as a per-player cooldown so a defender shadowing
    /// the carrier across many ticks racks up one pressure per "press
    /// burst" rather than one per tick.
    pub last_pressure_tick: u64,

    /// Condition the player carried onto the pitch — kickoff for
    /// starters, swap-time for substitutes. Stamped once at
    /// construction (or at the substitution swap) and never mutated
    /// after that. Read by the engine end-of-match path to build the
    /// `PlayerMatchPhysicalSnapshot` that drives the post-match
    /// condition-drop formula on the persisted `Player`.
    pub starting_condition: i16,

    /// Recovery debt the player walked onto the pitch with — copied
    /// from `Player::load::recovery_debt` at construction (or at the
    /// substitution swap) and never mutated during the match. Read by
    /// the in-match condition processor so a player with heavy legs
    /// tires faster than a fresh player with the same NF/stamina, and
    /// so a back-to-back schedule shows up in the tick-by-tick drain
    /// not just the post-match summary. Lives on MatchPlayer (rather
    /// than reaching back through `PlayerLoad`) so the hot per-tick
    /// path doesn't touch the persisted simulator data.
    pub starting_recovery_debt: f32,

    /// Home-crowd arousal multiplier applied inside `effective_skill`
    /// (1.0 = neutral). Stamped once at match start from the match
    /// environment: home players slightly above 1.0, away players
    /// slightly below, both scaled by `crowd_intensity ×
    /// home_advantage`. This is the play-quality half of home
    /// advantage (the documented crowd-arousal / travel-discomfort
    /// effect, worth ~+0.35 home goals at equal strength in real
    /// football); the officiating half lives in
    /// `RefereeProfile::home_bias`. Flows through every skill-mediated
    /// action via `effective_skill`, so it shifts duels, passing,
    /// saves and finishing continuously instead of dialling any single
    /// outcome.
    pub crowd_arousal: f32,
}

impl MatchPlayer {
    /// Age in whole years on `today`. Defined here (not on `Player`) so
    /// match-side code never has to reach back through the simulator
    /// data graph.
    #[inline]
    pub fn age_at(&self, today: NaiveDate) -> u8 {
        DateUtils::age(self.birth_date, today)
    }
}

impl MatchPlayer {
    /// Fast trait lookup used inside hot decision paths.
    #[inline]
    pub fn has_trait(&self, t: PlayerTrait) -> bool {
        self.traits.iter().any(|x| *x == t)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PlayerSide {
    Left,
    Right,
}

impl PlayerSide {
    /// Forward direction along the pitch's x-axis: +1 toward larger x
    /// for Left teams, -1 for Right. The math everywhere else in the
    /// engine uses this — never write `match side { Left => 1.0, ... }`
    /// inline or you'll inevitably miss a sign and produce the
    /// "right-side never reaches attacking third" bug we hit before.
    #[inline]
    pub fn forward_dir_x(self) -> f32 {
        match self {
            PlayerSide::Left => 1.0,
            PlayerSide::Right => -1.0,
        }
    }

    /// Attacking progress for an x-coordinate, normalised to [0.0, 1.0]:
    ///   0.0 = own goal line, 1.0 = opponent goal line.
    /// Use this everywhere a "what fraction of the pitch have we
    /// covered" check is needed (attacking third, defensive third,
    /// halfway thresholds). Replaces the legacy
    ///   `(x * forward_dir) / field_width`
    /// formula, which produced NEGATIVE values for right-side teams —
    /// that broke `> 0.66` (attacking third) tests by construction
    /// while leaving `< 0.33` (defensive third) accidentally correct.
    #[inline]
    pub fn attacking_progress_x(self, x: f32, field_width: f32) -> f32 {
        if field_width <= 0.0 {
            return 0.0;
        }
        let raw = match self {
            PlayerSide::Left => x / field_width,
            PlayerSide::Right => (field_width - x) / field_width,
        };
        raw.clamp(0.0, 1.0)
    }

    /// Forward delta along x: positive = toward opponent goal, negative
    /// = toward own goal. The right shape for "is this pass forward?",
    /// "did this player advance?", and any signed forward-progress
    /// arithmetic. Side-direction is baked in.
    #[inline]
    pub fn forward_delta(self, from_x: f32, to_x: f32) -> f32 {
        (to_x - from_x) * self.forward_dir_x()
    }

    /// Same as `forward_delta` but normalised by field width — gives a
    /// signed [-1.0, 1.0]-ish ratio for the cases that previously used
    /// `((to.x - from.x) * dir) / field_width`.
    #[inline]
    pub fn forward_delta_norm(self, from_x: f32, to_x: f32, field_width: f32) -> f32 {
        if field_width <= 0.0 {
            return 0.0;
        }
        self.forward_delta(from_x, to_x) / field_width
    }
}

impl MatchPlayer {
    pub fn from_player(
        team_id: u32,
        player: &Player,
        position: PlayerPositionType,
        use_extended_state_logging: bool,
    ) -> Self {
        MatchPlayer {
            id: player.id,
            position: Vector3::zeros(),
            start_position: Vector3::zeros(),
            attributes: player.attributes,
            team_id,
            player_attributes: player.player_attributes,
            skills: player.skills,
            velocity: Vector3::zeros(),
            tactical_position: TacticalPositions::new(position, None),
            side: None,
            state: Self::default_state(position),
            in_state_time: 0,
            statistics: MatchPlayerStatistics::new(),
            waypoint_manager: WaypointManager::new(),
            use_extended_state_logging,
            memory: PlayerMemory::new(),
            fatigue_accumulator: 0.0,
            cached_waypoints: Vec::new(),
            traits: player.traits.clone(),
            yellow_cards: 0,
            fouls_committed: 0,
            is_sent_off: false,
            tackle_cooldown: 0,
            pending_shot_reason: None,
            behavioral_directive: None,
            directive_base: None,
            press_target: None,
            mark_target: None,
            teammate_trigger: None,
            kickoff_pass_pending: 0,
            byline_commitment_ticks: 0,
            round_keeper_commitment_ticks: 0,
            round_keeper_target_x: 0.0,
            round_keeper_target_y: 0.0,
            gk1v1_decision_made: false,
            throw_in_pass_pending: 0,
            free_kick_pass_pending: 0,
            aerial_contest_won: 0,
            incoming_pass_target: None,
            incoming_pass_ticks: 0,
            link_target: None,
            intercept_target: None,
            supply_target: None,
            gk_up_after_minute: None,
            restart_retreat_target: None,
            start_anchor_x: None,
            start_anchor_y: None,
            shape_rules: Vec::new(),
            shape_base: None,
            attack_bias: None,
            is_force_match_selection: player.is_force_match_selection,
            birth_date: player.birth_date,
            entry_match_time_ms: 0,
            last_pressure_tick: 0,
            starting_condition: player.player_attributes.condition,
            starting_recovery_debt: player.load.recovery_debt,
            crowd_arousal: 1.0,
        }
    }

    /// Input-style constructor used by the distributed worker wire
    /// layer to rebuild a `MatchPlayer` from the bincode payload. Takes
    /// only the fields that meaningfully cross the network — engine
    /// runtime state (memory, waypoints, in-state timers, statistics,
    /// fatigue accumulator) is initialised to the same defaults
    /// `from_player` uses at kickoff, so the rebuilt player is
    /// indistinguishable from one freshly selected on the worker.
    #[allow(clippy::too_many_arguments)]
    pub fn from_inputs(
        id: u32,
        team_id: u32,
        position: [f32; 3],
        start_position: [f32; 3],
        attributes: PersonAttributes,
        player_attributes: PlayerAttributes,
        skills: PlayerSkills,
        tactical_position: PlayerPositionType,
        side: Option<PlayerSide>,
        traits: Vec<PlayerTrait>,
        birth_date: NaiveDate,
        is_force_match_selection: bool,
        starting_condition: i16,
        starting_recovery_debt: f32,
        use_extended_state_logging: bool,
    ) -> Self {
        MatchPlayer {
            id,
            position: Vector3::new(position[0], position[1], position[2]),
            start_position: Vector3::new(start_position[0], start_position[1], start_position[2]),
            attributes,
            team_id,
            player_attributes,
            skills,
            velocity: Vector3::zeros(),
            tactical_position: TacticalPositions::new(tactical_position, side),
            side,
            state: Self::default_state(tactical_position),
            in_state_time: 0,
            statistics: MatchPlayerStatistics::new(),
            waypoint_manager: WaypointManager::new(),
            use_extended_state_logging,
            memory: PlayerMemory::new(),
            fatigue_accumulator: 0.0,
            cached_waypoints: Vec::new(),
            traits,
            yellow_cards: 0,
            fouls_committed: 0,
            is_sent_off: false,
            tackle_cooldown: 0,
            pending_shot_reason: None,
            behavioral_directive: None,
            directive_base: None,
            press_target: None,
            mark_target: None,
            teammate_trigger: None,
            kickoff_pass_pending: 0,
            byline_commitment_ticks: 0,
            round_keeper_commitment_ticks: 0,
            round_keeper_target_x: 0.0,
            round_keeper_target_y: 0.0,
            gk1v1_decision_made: false,
            throw_in_pass_pending: 0,
            free_kick_pass_pending: 0,
            aerial_contest_won: 0,
            incoming_pass_target: None,
            incoming_pass_ticks: 0,
            link_target: None,
            intercept_target: None,
            supply_target: None,
            gk_up_after_minute: None,
            restart_retreat_target: None,
            start_anchor_x: None,
            start_anchor_y: None,
            shape_rules: Vec::new(),
            shape_base: None,
            attack_bias: None,
            is_force_match_selection,
            birth_date,
            entry_match_time_ms: 0,
            last_pressure_tick: 0,
            starting_condition,
            starting_recovery_debt,
            // Wire payloads predate the arousal field; the worker
            // re-stamps it at match start like the local path does.
            crowd_arousal: 1.0,
        }
    }

    /// Build the per-match end-of-game stats snapshot for this player.
    /// Owns the field-by-field mapping so the engine end-of-match path
    /// and the substitution snapshot path don't drift. `minutes_played`
    /// is computed by the caller from `entry_match_time_ms` and the
    /// exit / now match time — see `engine.rs` and `substitutions.rs`.
    pub fn to_match_end_stats(&self, minutes_played: u16) -> PlayerMatchEndStats {
        PlayerMatchEndStats {
            shots_on_target: self.memory.shots_on_target as u16,
            shots_total: self.memory.shots_taken as u16,
            passes_attempted: self.statistics.passes_attempted,
            passes_completed: self.statistics.passes_completed,
            tackles: self.statistics.tackles,
            interceptions: self.statistics.interceptions,
            saves: self.statistics.saves,
            shots_faced: self.statistics.shots_faced,
            goals: self.statistics.goals_count(),
            assists: self.statistics.assists_count(),
            match_rating: 0.0,
            raw_match_rating: 0.0,
            xg: self.memory.xg_total,
            position_group: self.tactical_position.current_position.position_group(),
            fouls: self.fouls_committed as u16,
            yellow_cards: self.statistics.yellow_cards_count(),
            red_cards: self.statistics.red_cards_count(),
            minutes_played,
            key_passes: self.statistics.key_passes,
            progressive_passes: self.statistics.progressive_passes,
            progressive_carries: self.statistics.progressive_carries,
            successful_dribbles: self.statistics.successful_dribbles,
            attempted_dribbles: self.statistics.attempted_dribbles,
            successful_pressures: self.statistics.successful_pressures,
            pressures: self.statistics.pressures,
            blocks: self.statistics.blocks,
            clearances: self.statistics.clearances,
            passes_into_box: self.statistics.passes_into_box,
            crosses_attempted: self.statistics.crosses_attempted,
            crosses_completed: self.statistics.crosses_completed,
            xg_chain: self.statistics.xg_chain,
            xg_buildup: self.statistics.xg_buildup,
            miscontrols: self.statistics.miscontrols,
            heavy_touches: self.statistics.heavy_touches,
            carry_distance: self.statistics.carry_distance,
            errors_leading_to_shot: self.statistics.errors_leading_to_shot,
            errors_leading_to_goal: self.statistics.errors_leading_to_goal,
            xg_prevented: self.statistics.xg_prevented,
            offsides: self.statistics.offsides,
            own_goals: self.statistics.own_goals_count(),
            zone_stats: self.statistics.zone_stats,
        }
    }

    /// Compute minutes spent on the pitch for this player given the
    /// current absolute match time (ms). Capped at 120 minutes so an
    /// extra-time match doesn't produce silly minute totals.
    pub fn minutes_played_at(&self, now_match_time_ms: u64) -> u16 {
        let elapsed = now_match_time_ms.saturating_sub(self.entry_match_time_ms);
        ((elapsed / 60_000) as u16).min(120)
    }

    /// Build the post-match physical snapshot for this player at the
    /// given absolute match time (substitution-off or full-time).
    /// Captures the starting tank, the current (drained) condition,
    /// and a high-intensity share that blends the position-group
    /// default with the player's actual high-intensity involvement
    /// (pressures, tackles, dribbles, crosses) so the persisted
    /// `Player::on_match_exertion` reflects how the player actually
    /// played, not just the position they nominally occupied. A
    /// fullback who pressed all match should bill more than one who
    /// sat in a low block.
    pub fn to_physical_snapshot(&self, now_match_time_ms: u64) -> PlayerMatchPhysicalSnapshot {
        let elapsed_ms = now_match_time_ms.saturating_sub(self.entry_match_time_ms);
        // Fractional minutes — `minutes_played_at` rounds down to a
        // u16 which loses information for cameo subs. The post-match
        // formula reads `duration = minutes / 90` so the fractional
        // resolution actually matters here.
        let minutes_played = (elapsed_ms as f32 / 60_000.0).min(120.0);
        let group = self.tactical_position.current_position.position_group();
        let position_default = PositionLoad::high_intensity_share(group);
        PlayerMatchPhysicalSnapshot {
            player_id: self.id,
            minutes_played,
            starting_condition: self.starting_condition,
            final_match_energy: self.player_attributes.condition,
            high_intensity_load_hint: Self::derive_high_intensity_hint(
                position_default,
                &self.statistics,
                minutes_played,
            ),
        }
    }

    /// Blend the position-group default high-intensity share with the
    /// observed action density (pressures, tackles, successful
    /// dribbles, crosses) so the post-match condition drop reflects
    /// how the player actually played. Keepers and defenders sitting
    /// deep stay near their position default; an attacking fullback
    /// who pressed every action will read materially higher.
    ///
    /// A "calibration baseline" of 0.50 actions/min maps to the
    /// position default; anything above lifts the hint linearly, up
    /// to a cap of 1.0 (the engine's mathematical ceiling). This is
    /// deliberately conservative — the position default is right for
    /// average involvement; behaviour-driven correction is a tilt,
    /// not a rewrite.
    pub(crate) fn derive_high_intensity_hint(
        position_default: f32,
        stats: &crate::r#match::engine::player::statistics::MatchPlayerStatistics,
        minutes_played: f32,
    ) -> f32 {
        if minutes_played < 1.0 {
            return position_default;
        }
        let hi_actions = stats.pressures as f32
            + stats.tackles as f32
            + stats.successful_dribbles as f32
            + stats.crosses_attempted as f32
            + stats.progressive_carries as f32;
        let per_min = hi_actions / minutes_played.max(1.0);
        // 0.50 actions/min ≈ the position default; deviation tilts the
        // hint by ±0.4 absolute at the extremes (per-min ≈ 0 → -0.4×,
        // per-min ≈ 1.5 → +0.4×). Clamped tightly so a stat-stuffer
        // can't pin the hint to 1.0 and a quiet game can't pin it to 0.
        const CAL_BASELINE: f32 = 0.50;
        const PER_MIN_GAIN: f32 = 0.40;
        let tilt = ((per_min - CAL_BASELINE) * PER_MIN_GAIN).clamp(-0.15, 0.20);
        (position_default + tilt).clamp(0.02, 1.0)
    }

    /// Consumes the tackle cooldown (ticks it down by 1). Called once per
    /// simulation tick from `update()`.
    #[inline]
    pub fn tick_tackle_cooldown(&mut self) {
        self.tackle_cooldown = self.tackle_cooldown.saturating_sub(1);
    }

    /// Can this player currently attempt a sliding tackle? False while the
    /// post-attempt cooldown is still counting down — regardless of which
    /// state routed them into Tackling.
    #[inline]
    pub fn can_attempt_tackle(&self) -> bool {
        self.tackle_cooldown == 0
    }

    /// Start the post-tackle cooldown. Called right after any attempt
    /// resolves (success, miss, or foul).
    #[inline]
    pub fn start_tackle_cooldown(&mut self) {
        // 3000 ticks ≈ 30 seconds. Real football: a player contests 2-4
        // tackles per 90 minutes — one every ~25 minutes. The previous
        // 15-second cooldown still let attempts run at 205/team/match
        // (5x real) because 10 outfield players × 15s allowed up to one
        // attempt per second team-wide. 30s halves the team-wide ceiling
        // and matches the realistic "commit, resolve, regroup,
        // reposition" cadence — a defender who lunges and either wins,
        // misses, or fouls is realistically out of the next play for
        // half a minute, not 15 seconds.
        self.tackle_cooldown = 3000;
    }

    pub fn rebuild_waypoint_cache(&mut self) {
        self.cached_waypoints = self
            .tactical_position
            .tactical_positions
            .iter()
            .filter(|tp| tp.position == self.tactical_position.current_position)
            .flat_map(|tp| &tp.waypoints)
            .map(|(x, y)| Vector3::new(*x, *y, 0.0))
            .collect();
    }

    pub fn update(
        &mut self,
        context: &MatchContext,
        tick_context: &GameTickContext,
        events: &mut EventCollection,
    ) {
        self.tick_tackle_cooldown();

        let player_events = PlayerMatchState::process(self, context, tick_context);

        events.add_from_collection(player_events);

        self.update_waypoint_index(tick_context);

        self.check_boundary_collision(context);
        self.move_to();
    }

    /// Movement-only update: apply existing velocity without re-evaluating AI state.
    #[inline]
    pub fn update_movement_only(&mut self, context: &MatchContext) {
        self.in_state_time += 1;
        self.check_boundary_collision(context);
        self.move_to();
    }

    #[inline]
    pub fn check_boundary_collision(&mut self, context: &MatchContext) {
        let field_width = context.field_size.width as f32 + 1.0;
        let field_height = context.field_size.height as f32 + 1.0;

        // Clamp position to field boundaries
        self.position.x = self.position.x.clamp(0.0, field_width);
        self.position.y = self.position.y.clamp(0.0, field_height);

        // Only stop velocity if player is trying to move OUT of bounds
        // Allow velocity that moves them back into the field
        if self.position.x <= 0.0 && self.velocity.x < 0.0 {
            // At left boundary, trying to move further left
            self.velocity.x = 0.0;
        } else if self.position.x >= field_width && self.velocity.x > 0.0 {
            // At right boundary, trying to move further right
            self.velocity.x = 0.0;
        }

        if self.position.y <= 0.0 && self.velocity.y < 0.0 {
            // At bottom boundary, trying to move further down
            self.velocity.y = 0.0;
        } else if self.position.y >= field_height && self.velocity.y > 0.0 {
            // At top boundary, trying to move further up
            self.velocity.y = 0.0;
        }
    }

    pub fn set_default_state(&mut self) {
        self.state = Self::default_state(self.tactical_position.current_position);
        self.rebuild_waypoint_cache();
    }

    fn default_state(position: PlayerPositionType) -> PlayerState {
        match position.position_group() {
            PlayerFieldPositionGroup::Goalkeeper => {
                PlayerState::Goalkeeper(GoalkeeperState::Standing)
            }
            PlayerFieldPositionGroup::Defender => PlayerState::Defender(DefenderState::Standing),
            PlayerFieldPositionGroup::Midfielder => {
                PlayerState::Midfielder(MidfielderState::Standing)
            }
            PlayerFieldPositionGroup::Forward => PlayerState::Forward(ForwardState::Standing),
        }
    }

    pub fn run_for_ball(&mut self) {
        self.state = match self.tactical_position.current_position.position_group() {
            PlayerFieldPositionGroup::Goalkeeper => {
                PlayerState::Goalkeeper(GoalkeeperState::TakeBall)
            }
            PlayerFieldPositionGroup::Defender => PlayerState::Defender(DefenderState::TakeBall),
            PlayerFieldPositionGroup::Midfielder => {
                PlayerState::Midfielder(MidfielderState::TakeBall)
            }
            PlayerFieldPositionGroup::Forward => PlayerState::Forward(ForwardState::TakeBall),
        }
    }

    #[inline]
    pub fn move_to(&mut self) {
        #[cfg(debug_assertions)]
        let old_position = self.position;

        // Apply velocity only if finite. `is_finite` rules out both NaN
        // and ±Infinity — either poisons the position, and a corrupt
        // position is excluded from the viewer recording so the player
        // literally disappears mid-match.
        if self.velocity.x.is_finite() {
            self.position.x += self.velocity.x;
        }

        if self.velocity.y.is_finite() {
            self.position.y += self.velocity.y;
        }

        // Last-resort salvage: if position is already corrupt from an
        // earlier tick (before the velocity guard was in place, or from
        // external code paths), reset to the player's tactical start
        // position. The player briefly teleports rather than vanishing.
        if !self.position.x.is_finite() || !self.position.y.is_finite() {
            self.position = self.start_position;
            self.velocity = Vector3::zeros();
        }

        #[cfg(debug_assertions)]
        {
            // Check for abnormally large position changes
            let position_delta = self.position - old_position;
            let position_change = position_delta.norm();

            const MAX_REASONABLE_POSITION_CHANGE: f32 = 20.0;

            if position_change > MAX_REASONABLE_POSITION_CHANGE {
                debug!(
                    "Player {:?} position jumped abnormally! {} from: ({:.2}, {:.2}) to: ({:.2}, {:.2}), delta: ({:.2}, {:.2}), distance: {:.2}, velocity: ({:.2}, {:.2})",
                    self.state,
                    self.id,
                    old_position.x,
                    old_position.y,
                    self.position.x,
                    self.position.y,
                    position_delta.x,
                    position_delta.y,
                    position_change,
                    self.velocity.x,
                    self.velocity.y
                );
            }
        }
    }

    pub fn heading(&self) -> f32 {
        self.velocity.y.atan2(self.velocity.x)
    }

    pub fn has_ball(&self, ctx: &StateProcessingContext<'_>) -> bool {
        ctx.ball().owner_id() == Some(self.id)
    }

    pub fn update_waypoint_index(&mut self, tick_context: &GameTickContext) {
        if self.cached_waypoints.is_empty() {
            self.rebuild_waypoint_cache();
        }
        self.waypoint_manager.update(
            &tick_context.positions.players.position(self.id),
            &self.cached_waypoints,
        );
    }

    pub fn get_waypoints_as_vectors(&self) -> &[Vector3<f32>] {
        &self.cached_waypoints
    }

    pub fn should_follow_waypoints(&self, ctx: &StateProcessingContext) -> bool {
        // Ball carrier doesn't follow waypoints — they move freely
        if self.has_ball(ctx) {
            return false;
        }

        // Best chaser pursues the ball, not waypoints
        if !ctx.ball().is_owned() && ctx.team().is_best_player_to_chase_ball() {
            return false;
        }

        // If any teammate is too close (< 12u, the natural "shoulder-
        // to-shoulder" bunching distance), follow waypoints back to
        // formation. This is the anti-grouping reinforcement: the
        // moment two of our players are crammed into one yard, one
        // of them peels off to their assigned tactical position. The
        // shorter of them (by id) peels, to avoid both trying to move
        // simultaneously.
        let me_id = self.id;
        let me_pos = self.position;
        let teammate_crowding = ctx.players().teammates().all().any(|t| {
            if t.id == me_id {
                return false;
            }
            let d_sq = (t.position - me_pos).norm_squared();
            if d_sq >= 144.0 {
                return false;
            } // 12² = 144
            // Only one of the pair peels (the lower id). Keeps the
            // behaviour deterministic per-tick and avoids both
            // leaving their post simultaneously.
            t.id > me_id
        });
        if teammate_crowding {
            return true;
        }

        // Everyone else follows waypoints to maintain tactical shape
        // Waypoints represent position-specific movement patterns that keep
        // formation spread and prevent clustering
        true
    }
}

#[derive(Copy, Clone)]
pub struct MatchPlayerLite {
    pub id: u32,
    pub position: Vector3<f32>,
    pub tactical_positions: PlayerPositionType,
}

impl MatchPlayerLite {
    pub fn has_ball(&self, ctx: &StateProcessingContext<'_>) -> bool {
        ctx.ball().owner_id() == Some(self.id)
    }

    pub fn velocity(&self, ctx: &StateProcessingContext<'_>) -> Vector3<f32> {
        ctx.tick_context.positions.players.velocity(self.id)
    }

    pub fn distance(&self, ctx: &StateProcessingContext<'_>) -> f32 {
        ctx.tick_context.grid.get(self.id, ctx.player.id)
    }
}

impl From<&MatchPlayer> for MatchPlayerLite {
    fn from(player: &MatchPlayer) -> Self {
        MatchPlayerLite {
            id: player.id,
            position: player.position,
            tactical_positions: player.tactical_position.current_position,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::club::player::builder::PlayerBuilder;
    use crate::r#match::MatchPlayer;
    use crate::shared::fullname::FullName;
    use crate::{
        PersonAttributes, PlayerAttributes, PlayerPosition, PlayerPositionType, PlayerPositions,
        PlayerSkills,
    };
    use chrono::NaiveDate;

    fn build_player(pos: PlayerPositionType) -> MatchPlayer {
        let player = PlayerBuilder::new()
            .id(1)
            .full_name(FullName::new("T".to_string(), "P".to_string()))
            .birth_date(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap())
            .country_id(1)
            .attributes(PersonAttributes::default())
            .skills(PlayerSkills::default())
            .positions(PlayerPositions {
                positions: vec![PlayerPosition {
                    position: pos,
                    level: 18,
                }],
            })
            .player_attributes(PlayerAttributes::default())
            .build()
            .unwrap();
        MatchPlayer::from_player(1, &player, pos, false)
    }

    #[test]
    fn starter_minutes_count_full_90() {
        let p = build_player(PlayerPositionType::DefenderCenter);
        // entry_match_time_ms defaults to 0 (kickoff). At full time
        // the helper returns the elapsed minutes.
        let minutes = p.minutes_played_at(90 * 60_000);
        assert_eq!(minutes, 90);
    }

    #[test]
    fn substituted_in_player_minutes_only_count_post_entry() {
        // Substitute came on at the 70th minute; at full time their
        // minutes-played should be 20, not 90 — the rating helper
        // damps cameo bonuses based on this number, so an 80th-minute
        // sub posting a key pass shouldn't be rated like a starter.
        let mut p = build_player(PlayerPositionType::ForwardCenter);
        p.entry_match_time_ms = 70 * 60_000;
        let minutes = p.minutes_played_at(90 * 60_000);
        assert_eq!(minutes, 20);
    }

    #[test]
    fn match_end_stats_carry_substituted_minutes() {
        // The end-of-match snapshot accepts the minutes count from the
        // caller — the substitution path computes it via
        // `minutes_played_at` and writes the right value into the
        // resulting `PlayerMatchEndStats`.
        let mut p = build_player(PlayerPositionType::MidfielderCenter);
        p.entry_match_time_ms = 60 * 60_000;
        let minutes = p.minutes_played_at(90 * 60_000);
        let stats = p.to_match_end_stats(minutes);
        assert_eq!(stats.minutes_played, 30);
    }

    #[test]
    fn minutes_played_at_caps_at_120_for_extra_time() {
        let p = build_player(PlayerPositionType::DefenderCenter);
        // Pathological — 200 minute match should clamp to 120.
        let minutes = p.minutes_played_at(200 * 60_000);
        assert_eq!(minutes, 120);
    }
}
