use crate::PlayerFieldPositionGroup;
use crate::r#match::common_states::CommonInjuredState;
use crate::r#match::defenders::states::{DefenderState, DefenderStrategies};
use crate::r#match::events::{Event, EventCollection};
use crate::r#match::forwarders::states::{ForwardState, ForwardStrategies};
use crate::r#match::goalkeepers::states::state::{GoalkeeperState, GoalkeeperStrategies};
use crate::r#match::midfielders::states::{MidfielderState, MidfielderStrategies};
use crate::r#match::player::memory::PlayerMemory;
use crate::r#match::player::state::PlayerState;
use crate::r#match::player::state::PlayerState::{Defender, Forward, Goalkeeper, Midfielder};
use crate::r#match::player::strategies::common::PlayerOperationsImpl;
use crate::r#match::player::strategies::common::PlayersOperationsImpl;
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::team::TeamOperationsImpl;
use crate::r#match::{
    BallOperationsImpl, GameTickContext, MatchContext, MatchPlayer, SteeringBehavior,
};
use log::debug;
use nalgebra::Vector3;

pub trait StateProcessingHandler {
    /// Decide whether the state should transition or emit an event this tick.
    fn process(&self, _ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        None
    }
    /// Per-tick velocity contribution. Default: no movement from this state.
    fn velocity(&self, _ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        None
    }
    /// Side-effects after the state resolves. Default: no-op.
    fn process_conditions(&self, _ctx: ConditionContext) {}
}

impl PlayerFieldPositionGroup {
    pub fn process(
        &self,
        in_state_time: u64,
        player: &mut MatchPlayer,
        context: &MatchContext,
        tick_context: &GameTickContext,
    ) -> StateProcessingResult {
        // Universal loose-ball override. Applied once at dispatch time so
        // every state benefits without needing its own copy of the guard.
        // Without this, the "designated chaser" selected by distance could
        // be in a state (Shooting, Finishing, Pressing, Dribbling, …) that
        // had no idea to abandon its current job and claim the ball — and
        // the ball would sit untouched while everyone assumed someone else
        // was going for it.
        //
        // The symmetric case also matters: a player already IN TakeBall
        // who's no longer the closest (ball rolled past them, teammate
        // got closer) should yield back to Running. Without the yield,
        // chasers pile up over time because TakeBall only exits on
        // ownership, not on "someone else is a better chaser now".
        let override_state_time = if Self::should_yield_takeball(*self, player, tick_context) {
            player.state = Self::yield_state_for(*self);
            0
        } else if Self::should_force_takeball(*self, player, tick_context) {
            player.state = Self::takeball_state_for(*self);
            0
        } else {
            in_state_time
        };
        let _ = context; // all needed state lives in player + tick_context

        let player_state = player.state;
        let state_processor =
            StateProcessor::new(override_state_time, player, context, tick_context);

        match player_state {
            // Common states
            PlayerState::Injured => state_processor.process(CommonInjuredState::default()),
            // // Specific states
            Goalkeeper(state) => GoalkeeperStrategies::process(state, state_processor),
            Defender(state) => DefenderStrategies::process(state, state_processor),
            Midfielder(state) => MidfielderStrategies::process(state, state_processor),
            Forward(state) => ForwardStrategies::process(state, state_processor),
        }
    }

    /// TakeBall variant for this position group. Outfield players commit
    /// to claiming a loose ball the same way; goalkeepers get their own
    /// TakeBall which handles the "only if near my box" rules internally.
    #[inline]
    fn takeball_state_for(group: PlayerFieldPositionGroup) -> PlayerState {
        match group {
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

    /// Default state to drop into when yielding TakeBall back to the pack.
    /// Outfield players go to Running — their off-ball velocity reshapes
    /// the defensive block with the new chaser designated. GK returns to
    /// Attentive — back to reading the game.
    #[inline]
    fn yield_state_for(group: PlayerFieldPositionGroup) -> PlayerState {
        match group {
            PlayerFieldPositionGroup::Goalkeeper => {
                PlayerState::Goalkeeper(GoalkeeperState::Standing)
            }
            PlayerFieldPositionGroup::Defender => PlayerState::Defender(DefenderState::Running),
            PlayerFieldPositionGroup::Midfielder => {
                PlayerState::Midfielder(MidfielderState::Running)
            }
            PlayerFieldPositionGroup::Forward => PlayerState::Forward(ForwardState::Running),
        }
    }

    /// True when this player is in TakeBall but another teammate is
    /// strictly-closer to the ball. Releases the chase so the pack doesn't
    /// accumulate ex-chasers who overshot or got passed by the ball.
    fn should_yield_takeball(
        _group: PlayerFieldPositionGroup,
        player: &MatchPlayer,
        tick_context: &GameTickContext,
    ) -> bool {
        if !matches!(
            player.state,
            PlayerState::Goalkeeper(GoalkeeperState::TakeBall)
                | PlayerState::Defender(DefenderState::TakeBall)
                | PlayerState::Midfielder(MidfielderState::TakeBall)
                | PlayerState::Forward(ForwardState::TakeBall)
        ) {
            return false;
        }
        // If the ball has been claimed, TakeBall's own `process` will
        // handle the transition to Running. Don't front-run it.
        if tick_context.ball.is_owned {
            return false;
        }
        let Some(my_side) = player.side else {
            return false;
        };
        // Use landing_position here to match `should_force_takeball`.
        // If yield used the current aerial position and force used
        // landing, a designated chaser could get yielded mid-flight
        // because a teammate happens to be closer to the ball's apex
        // — and nobody converges on the bounce.
        let ball_pos = tick_context.positions.ball.landing_position;
        let my_dist_sq = (ball_pos - player.position).norm_squared();
        // Hysteresis: only yield if a teammate is MEANINGFULLY closer
        // (by at least HYSTERESIS units). Otherwise tick-to-tick jitter
        // in movement swaps the "closest" designation between teammates
        // every tick, turning the chase into a ping-pong where each
        // player keeps yielding to the other and nobody commits long
        // enough to cover the final few units into the claim radius.
        const HYSTERESIS: f32 = 8.0;
        let yield_threshold_sq = {
            let my_dist = my_dist_sq.sqrt();
            let threshold = (my_dist - HYSTERESIS).max(0.0);
            threshold * threshold
        };
        for tm in tick_context.positions.players.as_slice() {
            if tm.player_id == player.id || tm.side != my_side {
                continue;
            }
            let d_sq = (ball_pos - tm.position).norm_squared();
            if d_sq < yield_threshold_sq {
                return true;
            }
        }
        false
    }

    /// True when this player should ignore their current-state logic and
    /// sprint to claim a loose ball. Fires when:
    ///   - The ball is not owned (free, not in-flight-with-intent),
    ///   - The ball is within meaningful chase range (saves compute on
    ///     balls that have rolled into the far corner — someone closer
    ///     will handle them),
    ///   - This player is the strictly-closest teammate by raw distance
    ///     (no ability weighting — we want exactly one claimant, not the
    ///     tolerance band of `is_best_player_to_chase_ball`),
    ///   - Not already in TakeBall (don't re-trigger and reset timers).
    fn should_force_takeball(
        group: PlayerFieldPositionGroup,
        player: &MatchPlayer,
        tick_context: &GameTickContext,
    ) -> bool {
        // Already chasing — leave the state alone.
        if matches!(
            player.state,
            PlayerState::Goalkeeper(GoalkeeperState::TakeBall)
                | PlayerState::Defender(DefenderState::TakeBall)
                | PlayerState::Midfielder(MidfielderState::TakeBall)
                | PlayerState::Forward(ForwardState::TakeBall)
        ) {
            return false;
        }

        // Ball must actually be loose.
        if tick_context.ball.is_owned {
            return false;
        }

        // See `should_yield_takeball` for why landing position is
        // preferred: lofted clearances need their chaser to converge on
        // the bounce, not the apex. `landing_position == position` for
        // ground balls, so this doesn't change ground-ball behaviour.
        let ball_pos = tick_context.positions.ball.landing_position;

        // Goalkeepers only claim balls near their box — the outfield
        // claimants handle anything further. Prevents the GK sprinting
        // 80m for a loose ball when a defender is 2m from it. GK will
        // transition to TakeBall via their own Standing/Walking guard
        // when the ball actually threatens their area.
        if group == PlayerFieldPositionGroup::Goalkeeper {
            let gk_dist_sq = (ball_pos - player.position).norm_squared();
            if gk_dist_sq > 60.0 * 60.0 {
                return false;
            }
        }

        let my_dist_sq = (ball_pos - player.position).norm_squared();

        // Am I the strictly-closest teammate? Tie-break by player id so
        // two players at exactly equal distance don't both trigger.
        //
        // CRITICAL: use `tick_context.positions.players` (live, updated
        // every tick) rather than `context.players` (a static snapshot
        // taken at match start, frozen thereafter). With the snapshot,
        // every player compared their *current* position against every
        // teammate's *match-start* position — all of them thought they
        // were closest, all of them flipped to TakeBall at once.
        //
        // Team membership is derived from `side` because the live store
        // doesn't carry team_id. Sent-off players are stashed at
        // (-500, -500), so they naturally fail any distance comparison
        // — no explicit filter needed.
        let my_side = match player.side {
            Some(s) => s,
            None => return false,
        };
        for tm in tick_context.positions.players.as_slice() {
            if tm.player_id == player.id || tm.side != my_side {
                continue;
            }
            let d_sq = (ball_pos - tm.position).norm_squared();
            if d_sq < my_dist_sq {
                return false;
            }
            if d_sq == my_dist_sq && tm.player_id < player.id {
                return false;
            }
        }

        true
    }
}

pub struct StateProcessor<'p> {
    in_state_time: u64,
    player: &'p mut MatchPlayer,
    context: &'p MatchContext,
    tick_context: &'p GameTickContext,
}

impl<'p> StateProcessor<'p> {
    pub fn new(
        in_state_time: u64,
        player: &'p mut MatchPlayer,
        context: &'p MatchContext,
        tick_context: &'p GameTickContext,
    ) -> Self {
        StateProcessor {
            in_state_time,
            player,
            context,
            tick_context,
        }
    }

    pub fn process<H: StateProcessingHandler>(self, handler: H) -> StateProcessingResult {
        // Match progress drives the late-game fatigue curve. Uses the
        // match half-time constant so debug / release builds both give
        // the correct 0..1 progression over their configured match length.
        let half_ms = crate::r#match::engine::engine::MATCH_HALF_TIME_MS as f32;
        let full_ms = half_ms * 2.0;
        let match_progress = (self.context.total_match_time as f32 / full_ms).clamp(0.0, 1.0);
        let condition_ctx = ConditionContext {
            in_state_time: self.in_state_time,
            player: self.player,
            match_progress,
        };

        // Process player conditions
        handler.process_conditions(condition_ctx);

        self.process_inner(handler)
    }

    pub fn process_inner<H: StateProcessingHandler>(self, handler: H) -> StateProcessingResult {
        let player_id = self.player.id;
        let need_extended_state_logging = self.player.use_extended_state_logging;

        let processing_ctx = self.into_ctx();
        let mut result = StateProcessingResult::new();

        if let Some(velocity) = handler.velocity(&processing_ctx) {
            // Apply coach tempo multiplier to all player movement
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            result.velocity = Some(velocity * tempo);
        }

        // §9.2.2 live recovery pull: while OUR goalkeeper holds the ball
        // in open play, teammates drift back toward their formation
        // anchors — a continuous velocity blend, not a scripted
        // pause-and-walk. Applied BEFORE the press/mark overrides so an
        // explicit man-assignment still wins outright.
        if let Some(pull) = Self::gk_possession_recovery_velocity(&processing_ctx) {
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            let base = result.velocity.unwrap_or_else(Vector3::zeros);
            result.velocity = Some(base * 0.35 + pull * tempo * 0.65);
        }

        // §12.5 universal off-ball separation: same-team outfielders
        // inside the §11.9 exclusion radius of each other separate, in
        // EVERY phase — defensive shape included. A weighted blend that
        // progressively replaces the state's own movement at close range
        // (an additive nudge loses to the state's convergence pull).
        // Applied BEFORE the press/mark overrides so explicit
        // man-assignments still win outright.
        if let Some((separation, w)) =
            crate::r#match::player::strategies::common::spacing::separation_nudge(&processing_ctx)
        {
            let base = result.velocity.unwrap_or_else(Vector3::zeros);
            result.velocity = Some(base * (1.0 - w) + separation * w);
        }

        // Realism-bug 2026-07-26 (passing follow-up): blend the
        // receiver's own movement toward an incoming pass's actual aim
        // point. A full-tree grep confirmed nothing previously read
        // `Ball.pending_pass_target` from inside any receiving player's
        // own movement code — a receiver's run during the ball's flight
        // was entirely decoupled from where the ball was actually
        // headed. Applied BEFORE the press/mark override so an explicit
        // man-assignment still wins outright (same priority position as
        // `separation_nudge`/`gk_possession_recovery_velocity` above).
        if let Some((nudge, w)) = Self::incoming_pass_awareness_velocity(&processing_ctx) {
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            let base = result.velocity.unwrap_or_else(Vector3::zeros);
            result.velocity = Some(base * (1.0 - w) + nudge * tempo * w);
        }

        // Cross-player assignment overrides (press / mark). Applied AFTER
        // the state handler so the manager's man-assignment wins over
        // normal state movement, regardless of which state the player is
        // in. States still handle their own transitions (tackling fires
        // naturally once the chase brings the ball into range).
        if let Some(velocity) = Self::assignment_override_velocity(&processing_ctx) {
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            result.velocity = Some(velocity * tempo);
        }

        // Milestone 13 (possession-decision-intelligence PRD) —
        // event-triggered reaction to a loose ball after a block/parry/
        // deflection. Applied AFTER the assignment override so an
        // explicit press/mark target still wins outright (checked again
        // inside the function itself, matching the same priority
        // convention as the block above).
        if let Some(velocity) = Self::rebound_crash_velocity(&processing_ctx) {
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            result.velocity = Some(velocity * tempo);
        }

        // realism-bug (2026-07-28): mandatory free-kick retreat (Law 13).
        // Applied AFTER the assignment override so an encroaching
        // defender retreats even if a manager has explicitly told him to
        // press/mark the taker — a hard legal requirement outranks a
        // tactical instruction. No tempo multiplier: a legally-required
        // retreat is urgent regardless of the team's set tempo, matching
        // §13.4's own full-pace retreat during the dead-ball freeze.
        if let Some(velocity) = Self::free_kick_retreat_velocity(&processing_ctx) {
            result.velocity = Some(velocity);
        }

        // §12.4: the staged short-corner option holds his spot until the
        // corner is struck — without this his state machine runs him back
        // toward his anchor and the delivery picker's "genuinely
        // stationary in the zone" gate can never be met.
        if let Some(velocity) = Self::corner_short_hold_velocity(&processing_ctx) {
            result.velocity = Some(velocity);
        }

        // realism-bug (2026-07-20): the free-kick forced-Passing override
        // (process(), below) makes the taker's STATE correct immediately,
        // but the Passing state's own internal evaluation can still take
        // several seconds to settle on a target — during which he was
        // free to drift at full speed. Raw trace: a Deep free kick
        // (armed at t, window 3s) didn't actually release the pass until
        // t+3.84s — 840ms past the window's own expiry — covering 51u in
        // that single window in one measured example (worse cases up to
        // 320u+ in a 100-match batch). Real players stay close to the
        // restart spot while deciding, not drifting a third of the pitch.
        // Caps speed rather than freezing him (unlike the corner short-
        // hold above) so a genuine small repositioning for a better
        // passing angle still looks natural.
        if let Some(velocity) = Self::free_kick_taker_hold_velocity(&processing_ctx, result.velocity)
        {
            result.velocity = Some(velocity);
        }

        // common logic
        let complete_result = |state_results: StateChangeResult,
                               mut result: StateProcessingResult| {
            // Propagate the tackle-cooldown signal regardless of whether
            // the handler also changed state — a successful tackle
            // returns a state-change + cooldown, but a keep-current-state
            // (None) wouldn't hit the `if let Some(state)` branch below.
            result.start_tackle_cooldown = state_results.start_tackle_cooldown;
            // Propagate the shot reason the same way — tagged at the
            // transition point, consumed by the Shooting state when it
            // composes the Shoot event.
            result.shot_reason = state_results.shot_reason;
            if let Some(state) = state_results.state {
                if need_extended_state_logging {
                    debug!("Player, Id={}, State {:?}", player_id, state);
                }
                result.state = Some(state);
                result.events = state_results.events;
            }
            result
        };

        // §11.5: the kickoff taker's first act is a short pass, not a
        // solo carry — while the window is open and he still owns the
        // ball, force the role's Passing state ahead of normal action
        // selection. The window is cleared in the engine loop as soon
        // as the ball leaves his feet, so this cannot pin a player in
        // Passing during open play.
        if let Some(state_result) = Self::kickoff_pass_override(&processing_ctx) {
            return complete_result(state_result, result);
        }

        // realism-bug (2026-07-19): same pattern as the kickoff override
        // directly above — the throw-in taker's first act must be a
        // release, not a solo carry. Without this the thrower simply
        // owns the ball like any open-play possession and his normal
        // state machine dribbles him forward for seconds (measured 13-
        // 54u of carry across 6/6 raw examples) before eventually
        // passing, or in one case losing it to a challenge outright —
        // both impossible for a real throw-in, which is a single
        // discrete motion. Window cleared in the engine loop the moment
        // the ball leaves his hands.
        if let Some(state_result) = Self::throw_in_pass_override(&processing_ctx) {
            return complete_result(state_result, result);
        }

        // realism-bug (2026-07-20): same pattern as the two overrides
        // directly above — the free-kick taker's ShortRoutine/Recycle/
        // Deep fallthrough must release quickly, not turn into a solo
        // carry from a deep restart.
        if let Some(state_result) = Self::free_kick_pass_override(&processing_ctx) {
            return complete_result(state_result, result);
        }

        if let Some(state_result) = handler.process(&processing_ctx) {
            return complete_result(state_result, result);
        }

        result
    }

    /// §12.4 hold for the staged short-corner option: zero velocity while
    /// the corner origin is live and the taker still holds the ball. Once
    /// the kick is struck (owner goes None / origin decays on the next
    /// touch) this stops matching and normal movement resumes.
    fn corner_short_hold_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        use crate::r#match::PassOriginRestart;
        let ball = &ctx.tick_context.ball;
        if ball.pass_origin_restart != PassOriginRestart::Corner {
            return None;
        }
        if ball.corner_short_option != Some(ctx.player.id) {
            return None;
        }
        // Only while the taker is still preparing the kick.
        ball.current_owner?;
        Some(Vector3::zeros())
    }

    /// Speed cap for the free-kick taker while `free_kick_pass_pending`
    /// is armed (realism-bug 2026-07-20). Scales the state's own computed
    /// velocity down to a slow-walk magnitude rather than replacing it
    /// outright, so a genuine small reposition for a better passing
    /// angle still looks natural — it just can't cover real ground
    /// while the Passing state's evaluation takes its time.
    fn free_kick_taker_hold_velocity(
        ctx: &StateProcessingContext,
        current: Option<Vector3<f32>>,
    ) -> Option<Vector3<f32>> {
        if ctx.player.free_kick_pass_pending == 0 || !ctx.player.has_ball(ctx) {
            return None;
        }
        const MAX_HOLD_SPEED: f32 = 0.12;
        let v = current.unwrap_or_else(Vector3::zeros);
        let mag = v.norm();
        if mag > MAX_HOLD_SPEED {
            Some(v * (MAX_HOLD_SPEED / mag))
        } else {
            Some(v)
        }
    }

    /// Forced Passing-state transition for the kickoff taker (§11.5).
    /// Active only while `kickoff_pass_pending` > 0 AND the player owns
    /// the ball; a no-op when already in a Passing state so the pass
    /// evaluation isn't restarted every tick.
    fn kickoff_pass_override(ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.kickoff_pass_pending == 0 || !ctx.player.has_ball(ctx) {
            return None;
        }
        match ctx.player.state {
            Forward(ForwardState::Passing)
            | Midfielder(MidfielderState::Passing)
            | Defender(DefenderState::Passing) => None,
            Forward(_) => Some(StateChangeResult::with_forward_state(ForwardState::Passing)),
            Midfielder(_) => Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Passing,
            )),
            Defender(_) => Some(StateChangeResult::with_defender_state(
                DefenderState::Passing,
            )),
            // assign_kickoff never picks the goalkeeper; an injured
            // taker has bigger problems than pass selection.
            Goalkeeper(_) | PlayerState::Injured => None,
        }
    }

    /// Forced Passing-state transition for the throw-in taker
    /// (realism-bug 2026-07-19). Active only while `throw_in_pass_pending`
    /// > 0 AND the player owns the ball; a no-op when already in a
    /// Passing state so the pass evaluation isn't restarted every tick.
    /// Identical shape to `kickoff_pass_override` above.
    fn throw_in_pass_override(ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.throw_in_pass_pending == 0 || !ctx.player.has_ball(ctx) {
            return None;
        }
        match ctx.player.state {
            Forward(ForwardState::Passing)
            | Midfielder(MidfielderState::Passing)
            | Defender(DefenderState::Passing) => None,
            Forward(_) => Some(StateChangeResult::with_forward_state(ForwardState::Passing)),
            Midfielder(_) => Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Passing,
            )),
            Defender(_) => Some(StateChangeResult::with_defender_state(
                DefenderState::Passing,
            )),
            // pick_thrower excludes goalkeepers; an injured taker has
            // bigger problems than pass selection.
            Goalkeeper(_) | PlayerState::Injured => None,
        }
    }

    /// Forced Passing-state transition for the free-kick taker
    /// (realism-bug 2026-07-20). Active only while `free_kick_pass_pending`
    /// > 0 AND the player owns the ball; a no-op when already in a
    /// Passing state so the pass evaluation isn't restarted every tick.
    /// Identical shape to `throw_in_pass_override` above.
    fn free_kick_pass_override(ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.free_kick_pass_pending == 0 || !ctx.player.has_ball(ctx) {
            return None;
        }
        match ctx.player.state {
            Forward(ForwardState::Passing)
            | Midfielder(MidfielderState::Passing)
            | Defender(DefenderState::Passing)
            | Goalkeeper(GoalkeeperState::Passing) => None,
            Forward(_) => Some(StateChangeResult::with_forward_state(ForwardState::Passing)),
            Midfielder(_) => Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Passing,
            )),
            Defender(_) => Some(StateChangeResult::with_defender_state(
                DefenderState::Passing,
            )),
            // realism-bug (2026-07-28): deep-FK team-positioning system —
            // band 1 (a restart within ~25m of the taker's own goal) now
            // has `pick_free_kick_taker` select the GK himself. Force him
            // into `GoalkeeperState::Passing`, the same short/medium/long
            // distribution decision a save already routes through
            // (`GoalkeeperPassingState`, reachable from Diving/
            // PreparingForSave) — reused rather than reinvented, and
            // already skill/pressure/space-graduated.
            Goalkeeper(_) => Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Passing,
            )),
            PlayerState::Injured => None,
        }
    }

    pub fn into_ctx(self) -> StateProcessingContext<'p> {
        StateProcessingContext::from(self)
    }

    /// "GK up for corners" velocity override. Active only for the
    /// flagged goalkeeper, from the configured displayed minute, while
    /// the ball's restart origin is a corner at the OPPONENT's end
    /// (own-team corner). Adds a sprint-home leg for whenever the
    /// window has closed but the keeper is still stranded upfield.
    fn gk_up_override_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        use crate::r#match::PassOriginRestart;
        let player = ctx.player;
        let threshold = player.gk_up_after_minute?;
        if !player.tactical_position.current_position.is_goalkeeper() {
            return None;
        }
        let field_w = ctx.context.field_size.width as f32;
        let field_h = ctx.context.field_size.height as f32;
        let (own_goal_x, opp_goal_x) = match player.side? {
            crate::r#match::PlayerSide::Left => (0.0_f32, field_w),
            crate::r#match::PlayerSide::Right => (field_w, 0.0_f32),
        };
        let minute = (ctx.context.total_match_time * 90
            / crate::r#match::engine::engine::MATCH_TIME_MS) as u32;
        let ball_pos = ctx.tick_context.positions.ball.position;
        let corner_live = ctx.tick_context.ball.pass_origin_restart == PassOriginRestart::Corner
            && (ball_pos.x - opp_goal_x).abs() < field_w * 0.30;

        if minute >= threshold && corner_live {
            // Hold a spot around the penalty-spot depth, offset from the
            // centre so he doesn't stack on the pushed-up centre-backs.
            let target_x = if opp_goal_x == 0.0 { 88.0 } else { field_w - 88.0 };
            let target = Vector3::new(target_x, field_h * 0.5 + 34.0, 0.0);
            let to_target = target - player.position;
            if to_target.magnitude() < 4.0 {
                return Some(ctx.player().separation_velocity() * 0.1);
            }
            return Some(to_target.normalize() * player.skills.physical.pace);
        }

        // Sprint-home leg: window closed but keeper is stranded upfield.
        let home_spot = Vector3::new(
            own_goal_x + if own_goal_x == 0.0 { 10.0 } else { -10.0 },
            field_h * 0.5,
            0.0,
        );
        let dist_home = (player.position - home_spot).magnitude();
        if dist_home > field_w * 0.30 {
            return Some(
                (home_spot - player.position).normalize()
                    * (player.skills.physical.pace * 1.1),
            );
        }
        None
    }

    /// Velocity override for manager-issued cross-player assignments.
    ///
    /// §9.2.2 — recovery pull while our own goalkeeper holds the ball in
    /// open play. Teammates drift toward their formation anchor
    /// (`start_position` — the live anchor, same recovery target the
    /// §9.2.1 dead-ball shape uses) at a jog. Returns None for the GK
    /// himself, the rare on-ball teammate, players already home, and
    /// whenever our keeper doesn't hold the ball.
    fn gk_possession_recovery_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let player = ctx.player;
        if player.tactical_position.current_position.is_goalkeeper() {
            return None;
        }
        if !ctx.ball().is_held_by_own_goalkeeper() {
            return None;
        }
        let to_anchor = player.start_position - player.position;
        let dist = to_anchor.magnitude();
        if dist < 25.0 {
            return None;
        }
        // Jog, not sprint: recovery under no direct pressure. Half the
        // press-chase magnitude convention (direction × pace).
        Some(to_anchor.normalize() * player.skills.physical.pace * 0.5)
    }

    /// Realism-bug 2026-07-26 (passing follow-up): a BLENDED nudge (not
    /// a hard override, matching `separation_nudge`'s idiom) toward the
    /// lead-adjusted aim point of a pass currently in flight to this
    /// player (`incoming_pass_target`, set by `handle_pass_to_event` at
    /// dispatch time). Returns `(direction*speed, blend_weight)` so a
    /// low off-ball player still mostly follows their own state's
    /// movement while a sharp one commits harder to meeting the ball —
    /// weighted by the SAME `off_ball_attack` composite
    /// (off_the_ball/anticipation/decisions/pace/teamwork/bravery)
    /// `receiver_positioning` already uses elsewhere in the pass
    /// evaluator, so this doesn't invent a new skill axis, it reuses the
    /// one already governing off-ball quality. Deliberately excludes
    /// goalkeepers — GK positioning is separately, extensively tuned
    /// elsewhere in this engine and touching it here would be
    /// unrelated blast radius for this fix.
    fn incoming_pass_awareness_velocity(
        ctx: &StateProcessingContext,
    ) -> Option<(Vector3<f32>, f32)> {
        let player = ctx.player;
        if player.incoming_pass_ticks == 0 {
            return None;
        }
        let target = player.incoming_pass_target?;
        if player.tactical_position.current_position.is_goalkeeper() {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None; // pass has arrived — this is no longer relevant
        }

        let to_target = target - player.position;
        if to_target.magnitude() < 3.0 {
            return None; // already there; let normal movement settle it
        }

        let minute = sc::minute_from_ms(ctx.context.total_match_time);
        let reactiveness = sc::off_ball_attack(player, minute);
        // Even a poor off-ball mover gets SOME pull (the ball is
        // visibly there) but a sharp one nearly fully commits — the
        // spread that keeps a good player's leading pass genuinely
        // better-received than an average one, rather than flattening
        // every receiver into an equally good one.
        let weight = (0.25 + reactiveness * 0.55).clamp(0.25, 0.80);

        let steer = SteeringBehavior::Arrive {
            target,
            slowing_distance: 8.0,
        }
        .calculate(player)
        .velocity;

        Some((steer, weight))
    }

    /// PRESS (`press_target`): whenever the assigned opponent has the
    /// ball, sprint straight at them (predictive-free chase — the visible
    /// proof is the dot hunting the target on every possession).
    /// MARK (`mark_target`): same chase when the target has the ball;
    /// otherwise, while our team is out of possession, hold a goal-side
    /// position 8u off the target. In possession the marker plays normal.
    ///
    /// Never fires for goalkeepers or for a player currently on the ball.
    fn assignment_override_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let player = ctx.player;
        if player.press_target.is_none() && player.mark_target.is_none() {
            return None;
        }
        if player
            .tactical_position
            .current_position
            .is_goalkeeper()
        {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None;
        }

        let chase = |target_id: u32| -> Vector3<f32> {
            let target_pos = ctx.tick_context.positions.players.position(target_id);
            let to_target = target_pos - player.position;
            if to_target.magnitude() < 2.0 {
                return ctx.player().separation_velocity() * 0.05;
            }
            let direction = to_target.normalize();
            direction * player.skills.physical.pace
                + ctx.player().separation_velocity() * 0.05
        };

        let ball_owner = ctx.ball().owner_id();

        if let Some(target_id) = player.press_target {
            if ball_owner == Some(target_id) && !ctx.ball().is_held_by_opponent_goalkeeper() {
                return Some(chase(target_id));
            }
        }

        if let Some(target_id) = player.mark_target {
            if ball_owner == Some(target_id) {
                return Some(chase(target_id));
            }
            if !ctx.team().is_control_ball() {
                let target_pos = ctx.tick_context.positions.players.position(target_id);
                let own_goal = ctx.ball().direction_to_own_goal();
                let goal_side = (own_goal - target_pos).normalize() * 8.0;
                let anchor = target_pos + goal_side;
                let to_anchor = anchor - player.position;
                if to_anchor.magnitude() > 3.0 {
                    return Some(
                        to_anchor.normalize() * player.skills.physical.pace * 0.9
                            + ctx.player().separation_velocity() * 0.3,
                    );
                }
                return Some(ctx.player().separation_velocity() * 0.3);
            }
        }

        None
    }

    /// Milestone 13 (possession-decision-intelligence PRD) —
    /// event-triggered reaction to a loose ball after a block, parry, or
    /// deflection. Architecturally different from every other milestone
    /// in this PRD: a direct reaction to a discrete engine EVENT
    /// (`Ball.last_rebound_tick`, already armed by three real loose-ball
    /// mechanisms — a blocked-shot central spill, an unlucky deflection
    /// that stays live, and a dangerous keeper parry — see
    /// `ball/interactions.rs`), not an extension of the carrier/off-ball
    /// VALUE functions every other milestone shares. Real forwards and
    /// attacking midfielders crash toward a genuine second ball near the
    /// box; defenders and deeper midfielders hold shape instead — scoped
    /// to attacking positions only.
    ///
    /// `REBOUND_WINDOW_TICKS` mirrors the window `team.rs`'s
    /// `can_shoot_debug`/`shot_clarity_debug` already use for the
    /// identical "is this rebound situation still live" question — reused
    /// rather than re-derived so both consumers of the same underlying
    /// event agree on how long it stays live.
    fn rebound_crash_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let player = ctx.player;
        if player.press_target.is_some() || player.mark_target.is_some() {
            return None; // an explicit assignment always wins
        }
        let pos_type = player.tactical_position.current_position;
        if pos_type.is_goalkeeper() {
            return None;
        }
        if !(pos_type.is_forward() || pos_type.is_attacking_midfielder()) {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None; // already on it — normal state machine drives him
        }
        if ctx.team().is_control_ball() {
            return None; // ball isn't loose, our own team already controls it
        }

        const REBOUND_WINDOW_TICKS: u64 = 300;
        let rebound_tick = ctx.tick_context.ball.last_rebound_tick;
        if rebound_tick == 0 {
            return None;
        }
        let current_tick = ctx.context.current_tick();
        if current_tick.saturating_sub(rebound_tick) >= REBOUND_WINDOW_TICKS {
            return None;
        }

        let ball_pos = ctx.tick_context.positions.ball.position;
        let to_ball = ball_pos - player.position;
        let dist = to_ball.magnitude();
        // Only crash from realistic range — a striker already up near
        // the box, not one dragged in from the halfway line.
        if dist > 220.0 || dist < 2.0 {
            return None;
        }
        Some(to_ball.normalize() * player.skills.physical.pace)
    }

    /// realism-bug (2026-07-28): mandatory free-kick retreat. §13.4's
    /// `apply_restart_retreat_tick` only walks a defender out during the
    /// frozen `dead_ball_retreat_active` window; ground-truth engine
    /// diagnostics (not just position-sample reconstruction) showed that
    /// window closing with the NEAREST defender still inside the legal
    /// 9.15m (73u) minimum on ~50% of free kicks, and by the moment the
    /// taker actually released the ball (~150-200ms later, once normal
    /// AI had resumed with zero dead-ball awareness) that had climbed to
    /// ~93% — the freeze ends, but nothing stops a defender's ordinary
    /// Pressing/Tackling AI from closing straight back in. This is the
    /// second half of the fix (the Tackling states above refuse to roll
    /// a challenge while encroaching; this makes retreat itself an
    /// obligation, not just "can't win the ball yet" — same idiom as
    /// `assignment_override_velocity`, but placed AFTER it so a
    /// manager's explicit press/mark instruction can never override a
    /// hard Law 13 requirement). Steers directly away from the ball at
    /// full retreat pace — the same calibrated speed conversion
    /// `apply_restart_retreat_tick` already uses (raw `skills.physical.pace`
    /// is a 1-20 rating, not a per-tick velocity; `rebound_crash_velocity`
    /// above still has that bug, but it's out of scope here).
    fn free_kick_retreat_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let player = ctx.player;
        if player.tactical_position.current_position.is_goalkeeper() {
            return None; // GK retreat is out of scope — mirrors the wall/box-clear exclusion
        }
        if !ctx.ball().is_free_kick_encroaching() {
            return None;
        }
        let ball_pos = ctx.tick_context.positions.ball.position;
        let away = player.position - ball_pos;
        let dist = away.magnitude();
        let dir = if dist > 0.5 {
            away.normalize()
        } else {
            // Degenerate (standing on the ball) — retreat toward own goal,
            // the only direction guaranteed not to re-encroach the box.
            (ctx.ball().direction_to_own_goal() - player.position).normalize()
        };
        let speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        Some(dir * speed)
    }
}

pub struct ConditionContext<'sp> {
    pub in_state_time: u64,
    pub player: &'sp mut MatchPlayer,
    /// Match progress 0.0..1.0 (0 = kickoff, 1.0 = 90'). Feeds the
    /// second-half fatigue-curve: recovery slows and sprint cost rises
    /// as the match progresses, so late-game players genuinely fade.
    pub match_progress: f32,
}

pub struct StateProcessingContext<'sp> {
    pub in_state_time: u64,
    pub player: &'sp MatchPlayer,
    pub context: &'sp MatchContext,
    pub tick_context: &'sp GameTickContext,
}

impl<'sp> StateProcessingContext<'sp> {
    #[inline]
    pub fn ball(&'sp self) -> BallOperationsImpl<'sp> {
        BallOperationsImpl::new(self)
    }

    #[inline]
    pub fn player(&'sp self) -> PlayerOperationsImpl<'sp> {
        PlayerOperationsImpl::new(self)
    }

    #[inline]
    pub fn players(&'sp self) -> PlayersOperationsImpl<'sp> {
        PlayersOperationsImpl::new(self)
    }

    #[inline]
    pub fn team(&'sp self) -> TeamOperationsImpl<'sp> {
        TeamOperationsImpl::new(self)
    }

    #[inline]
    pub fn memory(&self) -> &PlayerMemory {
        &self.player.memory
    }

    #[inline]
    pub fn current_tick(&self) -> u64 {
        self.context.current_tick()
    }
}

impl<'sp> From<StateProcessor<'sp>> for StateProcessingContext<'sp> {
    fn from(value: StateProcessor<'sp>) -> Self {
        StateProcessingContext {
            in_state_time: value.in_state_time,
            player: value.player,
            context: value.context,
            tick_context: value.tick_context,
        }
    }
}

pub struct StateProcessingResult {
    pub state: Option<PlayerState>,
    pub velocity: Option<Vector3<f32>>,
    pub events: EventCollection,
    /// Propagated up from the per-state `StateChangeResult`. Consumed by
    /// `state.rs` to bump `player.tackle_cooldown`.
    pub start_tackle_cooldown: bool,
    /// Tagged reason to attach to the next Shoot event fired by this
    /// player. Matches the pass-reason pattern. Written to
    /// `player.pending_shot_reason` by `state.rs` so the Shooting state
    /// can read it when composing the event.
    pub shot_reason: Option<&'static str>,
}

impl Default for StateProcessingResult {
    fn default() -> Self {
        Self::new()
    }
}

impl StateProcessingResult {
    pub fn new() -> Self {
        StateProcessingResult {
            state: None,
            velocity: None,
            events: EventCollection::new(),
            start_tackle_cooldown: false,
            shot_reason: None,
        }
    }
}

pub struct StateChangeResult {
    pub state: Option<PlayerState>,
    pub velocity: Option<Vector3<f32>>,

    pub events: EventCollection,

    /// Defender signalled "I just attempted a tackle" — the state.rs
    /// update loop consumes this and bumps `player.tackle_cooldown` so
    /// the next ~100 ticks of Tackling-state entries short-circuit
    /// without rolling an attempt. Must live on the result (not be
    /// applied directly in the state) because `ctx.player` is an
    /// immutable borrow inside the state processor.
    pub start_tackle_cooldown: bool,
    /// Tag the NEXT Shoot event fired by this player with this reason.
    /// Set by transitions to the Shooting state so the resulting
    /// Shoot event carries the decision-path context. Mirrors how
    /// pass events carry `with_reason(...)` — see Shooting state
    /// for the consumer.
    pub shot_reason: Option<&'static str>,
}

impl Default for StateChangeResult {
    fn default() -> Self {
        Self::new()
    }
}

impl StateChangeResult {
    pub fn new() -> Self {
        StateChangeResult {
            state: None,
            velocity: None,
            events: EventCollection::new(),
            start_tackle_cooldown: false,
            shot_reason: None,
        }
    }

    /// Tag the next Shoot event fired by this player with `reason`.
    /// Fluent helper to keep transition sites readable —
    /// `StateChangeResult::with_forward_state(Shooting).with_shot_reason("FWD_PRIO_06")`.
    pub fn with_shot_reason(mut self, reason: &'static str) -> Self {
        self.shot_reason = Some(reason);
        self
    }

    pub fn with(state: PlayerState) -> Self {
        StateChangeResult {
            state: Some(state),
            ..Self::new()
        }
    }

    pub fn with_goalkeeper_state(state: GoalkeeperState) -> Self {
        StateChangeResult {
            state: Some(Goalkeeper(state)),
            ..Self::new()
        }
    }

    pub fn with_goalkeeper_state_and_event(state: GoalkeeperState, event: Event) -> Self {
        StateChangeResult {
            state: Some(Goalkeeper(state)),
            events: EventCollection::with_event(event),
            ..Self::new()
        }
    }

    pub fn with_defender_state(state: DefenderState) -> Self {
        StateChangeResult {
            state: Some(Defender(state)),
            ..Self::new()
        }
    }

    pub fn with_defender_state_and_event(state: DefenderState, event: Event) -> Self {
        StateChangeResult {
            state: Some(Defender(state)),
            events: EventCollection::with_event(event),
            ..Self::new()
        }
    }

    pub fn with_midfielder_state(state: MidfielderState) -> Self {
        StateChangeResult {
            state: Some(Midfielder(state)),
            ..Self::new()
        }
    }

    pub fn with_midfielder_state_and_event(state: MidfielderState, event: Event) -> Self {
        StateChangeResult {
            state: Some(Midfielder(state)),
            events: EventCollection::with_event(event),
            ..Self::new()
        }
    }

    pub fn with_forward_state(state: ForwardState) -> Self {
        StateChangeResult {
            state: Some(Forward(state)),
            ..Self::new()
        }
    }

    pub fn with_forward_state_and_event(state: ForwardState, event: Event) -> Self {
        StateChangeResult {
            state: Some(Forward(state)),
            events: EventCollection::with_event(event),
            ..Self::new()
        }
    }

    pub fn with_event(event: Event) -> Self {
        StateChangeResult {
            events: EventCollection::with_event(event),
            ..Self::new()
        }
    }

    pub fn with_events(events: EventCollection) -> Self {
        StateChangeResult {
            events,
            ..Self::new()
        }
    }
}
