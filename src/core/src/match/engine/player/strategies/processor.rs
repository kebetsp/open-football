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

        // Step 1-hard experiment (docs/ball-distribution-fix-plan.md §9,
        // 2026-08-26): tests whether Step 1's soft `shape_rules` anchor
        // (`start_position`, which states merely "blend toward" at
        // whatever weight their own velocity code happens to give it)
        // is being diluted by competing live pulls. Applied BEFORE the
        // press/mark override so an explicit man-assignment still wins
        // outright, matching the priority convention every override in
        // this pipeline already follows.
        // §10 thread 1 comparison (2026-08-26), corrected same day per
        // Pavel's explicit standing rule: goals/match is informational
        // ONLY, never a selection criterion for this or any other fix —
        // an earlier version of this comment picked a lower weight
        // partly to keep goals "safely in band", which was wrong and has
        // been reverted. `WIDTH_OVERRIDE_BLEND_WEIGHT=1.0` (a genuine
        // full-strength override, matching §10's own "must remain an
        // actual override, not a blended nudge" design intent) gives the
        // best result on the plan's primary DoD metric (near-any-
        // boundary). Left as a named constant rather than removing the
        // blend entirely because a real, still-unexplained tradeoff was
        // found on the SECONDARY DoD metric (corner-relevant reads worse
        // at high weight than at low weight, non-monotonically — see
        // docs/ball-distribution-fix-plan.md §11/§12) — that's an open
        // question about the DoD metrics themselves, not something to
        // resolve by watering down the override for goals' sake.
        // Principles-of-Play plan (2026-08-30): generalized from
        // `width_hold_override_velocity` (which only ever pursued the
        // HoldWidth target) to any currently-active `ShapeRule`, since
        // `player.start_position` already holds whatever that rule
        // computed this tick (`engine/run.rs`'s shape-rule pass) —
        // width and the new back-line-compactness rule both flow
        // through the same generic pursuit, no per-action branching
        // needed here. Subsumes the old function outright rather than
        // running both side by side (double-application risk: both
        // would fire for the same player/tick under the old width-only
        // gating, since a wide defender can carry both rules).
        const SHAPE_OVERRIDE_BLEND_WEIGHT: f32 = 1.0;
        if let Some(velocity) = Self::shape_rule_override_velocity(&processing_ctx) {
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            let base = result.velocity.unwrap_or_else(Vector3::zeros);
            result.velocity = Some(
                (base * (1.0 - SHAPE_OVERRIDE_BLEND_WEIGHT)
                    + velocity * SHAPE_OVERRIDE_BLEND_WEIGHT)
                    * tempo,
            );
        }

        // Kicker-mode baseline (2026-09-12, Pavel's request — see
        // `kicker_line_hold_velocity`'s own doc comment for the full
        // design). A full-strength override, same convention as
        // `shape_rule_override_velocity` immediately above (which this
        // slot deliberately mirrors) — when active it replaces whatever
        // the state machine and every earlier override proposed for an
        // off-ball outfield player, full stop. Every "advanced"
        // positioning override below this point (assignment/give-and-go/
        // drop-and-spin/width-commitment/wide-run-tracking/shot-lane-
        // cover/shot-block-reaction/rebound-crash) is separately gated
        // off at its own definition when kicker mode is active, so none
        // of them can silently re-override this one later in the chain.
        if let Some(velocity) = Self::kicker_line_hold_velocity(&processing_ctx) {
            result.velocity = Some(velocity);
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

        // Chunk 5b (docs/zone-staged-possession-progression.md): the
        // "go" half of a give-and-go. Applied AFTER the assignment
        // override so an explicit press/mark target still wins outright
        // (also re-checked inside the function itself), and BEFORE the
        // defensive wide-run-tracking/shot-block/rebound overrides below
        // so any genuine defensive or reactive duty still takes priority
        // over an attacking off-ball run. Tempo-scaled like ordinary
        // attacking movement, unlike the defensive/legal overrides below
        // which are deliberately full-urgency regardless of tempo.
        if let Some(velocity) = Self::give_and_go_run_velocity(&processing_ctx) {
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            result.velocity = Some(velocity * tempo);
        }

        // docs/positional-attack-phase-design.md, Stage 3: the drop-
        // and-spin run. Same placement/priority reasoning as give-and-
        // go immediately above (attacking off-ball run, tempo-scaled,
        // still below the defensive/legal overrides that follow).
        if let Some(velocity) = Self::drop_and_spin_run_velocity(&processing_ctx) {
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            result.velocity = Some(velocity * tempo);
        }

        // realism-bug (2026-09-12, Follow-up #8): forward width
        // commitment, state-independent. Same placement/priority
        // reasoning as give-and-go/drop-and-spin immediately above
        // (attacking off-ball positioning, tempo-scaled, still below the
        // defensive/legal overrides that follow) — this is what fixes
        // the state-fragmentation bug: previously this only fired from
        // inside `ForwardRunningState::velocity()`, so a committed
        // player in any other forward state ignored it entirely.
        if let Some(velocity) = Self::width_commitment_override_velocity(&processing_ctx) {
            let tempo = processing_ctx.team().coach_instruction().tempo_multiplier();
            result.velocity = Some(velocity * tempo);
        }

        // realism-bug (2026-08-27): defensive wide-run tracking. Applied
        // AFTER the assignment override so an explicit press/mark target
        // still wins outright (also re-checked inside the function
        // itself), and BEFORE the shot-block reaction so an already-live
        // shot takes priority over a broader tracking duty. No tempo
        // multiplier — tracking a genuine wide run is a real defensive
        // urgency, not a tempo-scaled positioning preference.
        if let Some(velocity) = Self::wide_run_track_override_velocity(&processing_ctx) {
            result.velocity = Some(velocity);
        }

        // realism-bug (2026-08-30, deflection-rate investigation):
        // pre-shot corridor cover. Applied AFTER wide-run tracking and
        // BEFORE the reactive shot-block lunge, so an already-live shot
        // (a strictly better, real-trajectory signal) still wins outright
        // — this function also re-checks that itself and hands off
        // cleanly. No tempo multiplier — getting goal-side of a live
        // shooting threat is a defensive urgency, not tempo-scaled.
        if let Some(velocity) = Self::shot_lane_cover_override_velocity(&processing_ctx) {
            result.velocity = Some(velocity);
        }

        // realism-bug (2026-08-26): reactive shot-block lunge. Applied
        // AFTER the assignment override so an explicit press/mark target
        // still wins outright (also re-checked inside the function
        // itself). No tempo multiplier — a reflexive block attempt is at
        // full urgency regardless of the team's set tempo, same
        // reasoning as the free-kick retreat below.
        if let Some(velocity) = Self::shot_block_reaction_velocity(&processing_ctx) {
            result.velocity = Some(velocity);
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

        // engine audit Module 3 (2026-08-29): universal offside-line
        // hold. Applied LAST among the velocity overrides so it clamps
        // whatever the final proposed x-movement is, regardless of which
        // state or earlier override produced it — see the function's own
        // doc comment for the full rationale.
        if let Some(velocity) = Self::offside_line_hold_velocity(&processing_ctx, result.velocity)
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

        // Kicker-mode baseline (2026-09-12) — see `kicker_mode_pass_override`'s
        // own doc comment. Runs after the four restart-specific forced-
        // passing overrides above (kickoff/throw-in/free-kick still win
        // for their own specific scenarios) and before
        // `carrier_contained_pass_override` immediately below, which
        // becomes unreachable while this is active (this override is a
        // strict superset of its trigger condition) — left in place
        // rather than separately gated, since ordering alone makes it
        // inert.
        if let Some(state_result) = Self::kicker_mode_pass_override(&processing_ctx) {
            return complete_result(state_result, result);
        }

        // docs/positional-play-master-design.md, Stage 3.5 follow-up
        // (2026-09-11): found by directly re-measuring the actual
        // outcome after generalizing `carrier_contained_reset_ticks` —
        // 96.4% of wide carries that genuinely reach the byline area end
        // in the carrier being DISPOSSESSED, not a clean pass of any
        // kind. Root cause: the earlier generalization only biased WHICH
        // pass the evaluator picks once a pass is actually attempted —
        // it never forced the attempt itself. The byline-committed case
        // doesn't have this problem because its OWN dribbling-state
        // logic already aborts to Passing the same tick the contained
        // condition is detected (see `run.rs`'s "Armed here, one tick
        // ahead of the state machine's own abort" comment); the newly-
        // generalized population had no equivalent — the carrier kept
        // dribbling/trying to carry into the very defender who was
        // blocking him, giving that defender more ticks to win a tackle
        // before any pass decision ever ran. Same pattern as `free_kick_
        // pass_override` directly above: force the role's Passing state
        // outright while contained and still on the ball, so the
        // already-built reward bias actually gets a chance to apply
        // before the ball is lost, not after.
        if let Some(state_result) = Self::carrier_contained_pass_override(&processing_ctx) {
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

    /// Kicker-mode on-ball decision baseline (2026-09-12, Pavel's
    /// request — docs/kicker-mode-baseline.md). Forces ANY outfield
    /// player straight into their role's `Passing` state the instant
    /// they have the ball, bypassing Running/Standing/Walking/
    /// Dribbling/CreatingSpace's own carry-vs-pass-vs-shoot value
    /// comparisons entirely — those are exactly the "advanced" decision
    /// logic this mode exists to strip out. `PassEvaluator::
    /// find_best_pass_option`'s own kicker-mode branch (evaluator.rs)
    /// then does the actual line/lane target selection once inside
    /// Passing.
    ///
    /// Deliberately does NOT touch shooting: each role's `Passing`
    /// state already has its own shot-decision check near the top of
    /// its `process()` (e.g. `MidfielderPassingState`'s
    /// `evaluate_forward_shot_decision` call) — routing every ball-
    /// holder through Passing still lets a genuine chance near goal
    /// resolve as a shot exactly where it already would have, it just
    /// removes the EARLIER shot/carry/pass comparisons that used to run
    /// from Running/Dribbling before a player ever reached Passing.
    /// Real football's "hold the ball, buy time, wait for the gap" is
    /// what this produces for the 95%+ of the pitch that isn't a
    /// genuine chance.
    ///
    /// Placed after the three restart-specific forced-passing overrides
    /// above (kickoff/throw-in/free-kick) so those still own their own
    /// specific restart scenarios untouched, and before
    /// `carrier_contained_pass_override` below (an "advanced" feature
    /// this override makes structurally unreachable by ordering alone —
    /// see that function's own call site for why it's fine left as-is
    /// rather than separately gated).
    fn kicker_mode_pass_override(ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if !crate::r#match::player::strategies::common::kicker_mode::is_active() {
            return None;
        }
        if !ctx.player.has_ball(ctx) {
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
            // A goalkeeper holding the ball in open play (a catch, a
            // back-pass) gets the same treatment — his existing
            // Passing state's own short/medium/long distribution logic
            // is untouched, this just routes him there directly instead
            // of via his normal Holding/ComingOut decision tree.
            Goalkeeper(_) => Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Passing,
            )),
            PlayerState::Injured => None,
        }
    }

    /// docs/positional-play-master-design.md, Stage 3.5 follow-up
    /// (2026-09-11) — see the call site's own doc comment for the full
    /// diagnosis. Forces the role's Passing state while `carrier_
    /// contained_reset_ticks` is armed and the player still has the
    /// ball, exactly mirroring `free_kick_pass_override` immediately
    /// above (same idiom, same reasoning: a decision that only biases
    /// pass SELECTION is worthless if the pass ATTEMPT itself never
    /// happens in time). Self-limiting by construction — `run.rs`'s own
    /// decay clears the ticks field the instant the ball leaves this
    /// player's feet, whether by a completed pass or a lost duel, so
    /// this override can never hold a player in Passing indefinitely.
    fn carrier_contained_pass_override(ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.carrier_contained_reset_ticks == 0 || !ctx.player.has_ball(ctx) {
            return None;
        }
        // docs/positional-play-master-design.md, Follow-up #6
        // (2026-09-11, target: >=8 open-play byline-reaching carries/
        // match): a match-logs diagnostic found internal
        // byline_commitment_ticks arms firing ~23/match while the
        // matching CROSSDIAG LOSTBALL count sat at almost exactly the
        // same rate -- nearly every committed run was ending via a
        // FORCED PASS (this override, unconditional at the time) long
        // before the carrier got anywhere near the byline. This
        // override existed to make sure a genuinely-contained carrier
        // near goal actually releases the ball rather than dying on the
        // vine (Follow-up #4) -- but forcing it unconditionally,
        // wherever `carrier_contained_reset_ticks` happens to arm
        // (including a byline-committed run's own much looser,
        // instant-fire `single_blocker` abort check, still active deep
        // in a developing run), was cutting runs short well before they
        // could ever register as an external byline-reaching carry.
        // Scoped to only force the immediate release once the carrier
        // is ALREADY near the byline (reusing the same NEAR_BYLINE_DEPTH
        // threshold Follow-up #5's distance-aware dwell fix already
        // established as the genuine "act now" zone) -- far from goal,
        // `carrier_contained_reset_bonus` (evaluator.rs) still biases
        // the natural pass decision toward releasing, it just no longer
        // forcibly interrupts a run that's still developing.
        let goal_x = match ctx.player.side {
            Some(crate::r#match::PlayerSide::Left) => ctx.context.field_size.width as f32,
            _ => 0.0,
        };
        let depth_from_goal = (goal_x - ctx.player.position.x).abs();
        const NEAR_BYLINE_DEPTH: f32 = 150.0;
        if depth_from_goal > NEAR_BYLINE_DEPTH {
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
            // A goalkeeper being "contained" while dribbling is an
            // exotic edge case (GKs rarely carry forward into traffic),
            // but the field carries no role restriction and the same
            // "release now" logic is correct for him too if it ever
            // arms — reuses the existing GoalkeeperState::Passing route,
            // same as free_kick_pass_override above.
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

    /// Originated as a width-only "Step 1-hard experiment"
    /// (docs/ball-distribution-fix-plan.md §9/§10, 2026-08-26), then
    /// generalized to any active `ShapeRule`
    /// (docs/principles-of-play-structural-shape-plan.md, 2026-08-30) —
    /// see this function's call site for the full rationale. A genuine
    /// velocity OVERRIDE (mirrors `assignment_override_velocity`'s
    /// press/mark pattern exactly — full replacement when active, not a
    /// blended nudge; a reward-side nudge is exactly the category of
    /// fix the ball-distribution plan spent nine attempts proving
    /// doesn't work here), not a soft anchor nudge: while a structural
    /// rule is currently active for this player, steer directly at
    /// `player.start_position` — whatever that rule computed this tick
    /// (`ShapeAction::HoldWidth`'s touchline point, `CompactBackLine`'s
    /// back-line target, or any future action) — same target, stronger
    /// enforcement than the soft "blend toward it" every state's own
    /// velocity code otherwise gives it. Exempts the ball carrier —
    /// their own on-ball decision logic (`carry_candidates`/
    /// `pass_value_from`) is a separate, more nuanced system this
    /// shouldn't fight.
    ///
    /// Scoping (§10 thread 1, carried forward unchanged): the original
    /// width-only proof-of-concept fired in literally any state,
    /// including mid-tackle, which is why it needed scoping before
    /// shipping. Three exemptions, all checked before the pursuit is
    /// computed:
    /// (1) any active Tackling state (any role) — a live 1v1 challenge
    /// deserves the player's full attention, not being yanked out to a
    /// structural target mid-duel;
    /// (2) `scan_for_dangerous_runs()` returning `Some` — that function
    /// is already computed relative to *this* player (`opponents().nearby`
    /// scans around `ctx.player.position`) and already excludes runners
    /// a teammate is already engaging, so a hit here specifically means
    /// *this* player is the relevant nearby defender for a still-live
    /// dangerous opponent run. This covers the transition moment right
    /// after regaining the ball (`is_control_ball()` can go true off a
    /// favourable-direction loose ball, not only a firm claim) where a
    /// dangerous opponent may still be live and shouldn't be abandoned
    /// instantly for the touchline;
    /// (3) real bug (2026-08-31, Pavel: "red #11 winger is frozen, the
    /// ball lands a few metres from him and he doesn't even try to get
    /// it") — TakeBall (any role). `shape_rule_active` is derived from
    /// `in_possession`, a phase-level flag that can lag a beat behind the
    /// literal instant the ball goes loose (e.g. a heavy touch, a
    /// deflection near a still-nominally-"in possession" team), while
    /// `should_force_takeball` (`engine/run.rs`'s dispatch-time override)
    /// separately and carefully designates the single strictly-closest
    /// teammate to claim it, with its own hysteresis against thrash. Before
    /// this exemption, that designation was won at the state level (the
    /// player's `PlayerState` really did flip to TakeBall) but then
    /// silently overwritten back to the touchline anchor by this function
    /// one step later in the same tick — the state machine said "go get
    /// it," the velocity pipeline said "no, hold width," and the velocity
    /// pipeline always won (`SHAPE_OVERRIDE_BLEND_WEIGHT=1.0`, full
    /// replacement). Verified with raw data before fixing (temporary
    /// `SHAPEFREEZE_DIAG` trace, since removed): the collision fires
    /// ~7,350 times per match on average — a genuinely common, not rare,
    /// interaction. An external raw-position check (settled ball 20-90u
    /// away, player closes <8u of it over the next 1.2s) found the strict
    /// symptom itself at ~1.3 incidents/match before this fix, dropping to
    /// ~0.3/match after — same-binary A/B, `git stash` isolated to this
    /// file, 6 matches per side.
    fn shape_rule_override_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let player = ctx.player;
        // `shape_rule_active` (set in `engine/run.rs` alongside
        // `start_position` itself) is the load-bearing check: without
        // it this would also force every rule-holding player back onto
        // their bare `shape_base` whenever no rule is currently active
        // (e.g. a wide player who's out of possession, or a defender
        // with the ball deep in the opponent's third) — far broader
        // than intended. This is the single condition that replaces
        // the old function's own `flank_side(...)?` gate: any rule,
        // for any role, not just HoldWidth for wide roles.
        if !player.shape_rule_active {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None;
        }
        match ctx.player.state {
            Defender(DefenderState::Tackling)
            | Midfielder(MidfielderState::Tackling)
            | Forward(ForwardState::Tackling) => return None,
            Goalkeeper(GoalkeeperState::TakeBall)
            | Defender(DefenderState::TakeBall)
            | Midfielder(MidfielderState::TakeBall)
            | Forward(ForwardState::TakeBall) => return None,
            // realism-bug (2026-09-08, "I see no difference" investigation
            // — docs/positional-football-master-plan.md Stage 3): a
            // defender/midfielder actively engaging a ball carrier
            // (jockeying OR the full urgency-chase sub-mode, both live in
            // `Pressing`) was NOT exempt here, unlike `Tackling`/
            // `TakeBall` — so whenever a shape rule was also active
            // (e.g. the default `CompactBackLine`), this full-strength
            // (weight 1.0, no blend) override could silently replace the
            // jockeying velocity computed and logged moments earlier with
            // a "walk back to formation shape" pursuit instead. Measured
            // via a temporary same-tick/same-player correlation on a
            // clean single-match trace: 5.0% of should_jockey==true ticks
            // were overridden this way (83/1647). Real, but modest — not
            // the dominant explanation for low visible effect on its own,
            // just one confirmed contributing gap. Chasing hard (the
            // non-jockeying half of Pressing) is equally legitimate
            // defending, not idle shape-holding, so the whole state is
            // exempted, matching Tackling's own wholesale exemption.
            Defender(DefenderState::Pressing) | Midfielder(MidfielderState::Pressing) => {
                return None;
            }
            _ => {}
        }
        if ctx.player().defensive().scan_for_dangerous_runs().is_some() {
            return None;
        }
        let target = player.start_position;
        let to_target = target - player.position;
        let dist = to_target.magnitude();
        if dist < 3.0 {
            return Some(Vector3::zeros());
        }
        let max_speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        Some(to_target.normalize() * max_speed)
    }

    /// Kicker-mode positioning baseline (2026-09-12, Pavel's request —
    /// docs/kicker-mode-baseline.md). Opt-in via `KICKER_MODE`
    /// (`kicker_mode::is_active()`). The goal, in his own words: "players
    /// move in their lines passing horizontally until they find a gap to
    /// make a pass to the next line... there should be a distance
    /// between forwards and defenders (delay principle)". This is that
    /// line-holding half; `kicker_mode_pass_override`/
    /// `PassEvaluator::kicker_find_pass_option` (evaluator.rs) is the
    /// matching on-ball decision half.
    ///
    /// Three lines (back/mid/front — `PlayerFieldPositionGroup::
    /// {Defender,Midfielder,Forward}`, goalkeepers excluded) hold a real
    /// depth separation (`LINE_GAP`) from each other. The reference
    /// depth the lines are built from differs by possession:
    ///   - IN possession: the ball's own depth (a proxy for the
    ///     carrier's depth), offset by how many line-steps this
    ///     player's own line sits from the carrier's line — teammates in
    ///     the carrier's own line hold roughly alongside him (a
    ///     horizontal outlet), one line ahead pushes forward to open a
    ///     gap to receive into, one line behind drops to recycle.
    ///   - OUT of possession: `DELAY_GAP` behind the ball (toward this
    ///     team's own goal), floored at `MIN_DEPTH_FROM_GOAL` so the
    ///     back line never collapses onto its own goal line — this is
    ///     the explicit, ENFORCED version of "distance between forwards
    ///     and defenders" Pavel asked for, not left to emerge on its
    ///     own. Midfield/front lines (this team's own forwards, now
    ///     defending) sit progressively less deep, `LINE_GAP` steps
    ///     ahead of the back line — the classic compact defensive block.
    ///
    /// Deliberately NOT built: active lane-stepping (a defender
    /// deliberately positioning INSIDE a specific passing lane, not just
    /// holding a line depth). Pavel's own framing downplayed strict
    /// lateral positioning as "not the important part" — lateral
    /// position is left at each player's own formation slot
    /// (`start_position.y`), and "defenders closing passing lanes"
    /// emerges only passively, from `has_clear_pass`'s existing ray-cast
    /// obstruction check against wherever the lines happen to stand. If
    /// that passive effect turns out to be too weak to ever produce a
    /// genuine "no lane, recycle" decision, active lane-stepping is the
    /// natural next feature to add back in, not something this first
    /// cut should reach for.
    ///
    /// Excluded from this positioning (matching the "can't fully
    /// disable" list from Pavel's own request): the ball carrier (his
    /// own movement is driven by `kicker_mode_pass_override` forcing him
    /// into the role's Passing state), a live loose-ball chase
    /// (TakeBall), a live 1v1 duel (Tackling), and any player currently
    /// staged for a restart (`restart_retreat_target` — corner/free-
    /// kick/throw-in/goal-kick teleport-and-hold machinery keeps working
    /// untouched).
    fn kicker_line_hold_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        use crate::r#match::player::strategies::common::kicker_mode::{self, line_index};

        if !kicker_mode::is_active() {
            return None;
        }
        let player = ctx.player;
        if player.tactical_position.current_position.is_goalkeeper() {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None; // the carrier's own movement is handled elsewhere
        }
        match ctx.player.state {
            Defender(DefenderState::TakeBall)
            | Midfielder(MidfielderState::TakeBall)
            | Forward(ForwardState::TakeBall)
            | Defender(DefenderState::Tackling)
            | Midfielder(MidfielderState::Tackling)
            | Forward(ForwardState::Tackling) => return None,
            _ => {}
        }
        if player.restart_retreat_target.is_some() {
            return None; // corner/free-kick/throw-in/goal-kick staging wins
        }

        let group = player.tactical_position.current_position.position_group();
        let field_w = ctx.context.field_size.width as f32;
        let field_h = ctx.context.field_size.height as f32;
        let side = player.side?;
        let (own_goal_x, dir) = match side {
            crate::r#match::PlayerSide::Left => (0.0_f32, 1.0_f32),
            crate::r#match::PlayerSide::Right => (field_w, -1.0_f32),
        };

        const LINE_GAP: f32 = 110.0; // ~14m between adjacent lines
        const MIN_DEPTH_FROM_GOAL: f32 = 60.0;

        // "Progress" = distance from THIS player's own goal toward the
        // opponent's, regardless of side — all the geometry below works
        // in this one scalar space and converts back to absolute X only
        // once, at the very end. Avoids the sign-juggling that a
        // per-side `dir` multiplier invites when both a team-relative
        // AND a cross-team comparison are needed in the same function.
        let progress = |x: f32| (x - own_goal_x) * dir;
        let from_progress = |d: f32| own_goal_x + dir * d;

        // realism-bug (2026-09-12, first cut of this function): the
        // original design re-anchored every tick to the BALL's current
        // depth (in possession) with each line's offset computed
        // relative to whichever player happened to be the CARRIER's own
        // line that specific tick. Verified with raw position data
        // (kicker_verify.mjs, 15 matches): the intended ~110-220u line
        // separation washed out to ~15-23u on average — because the
        // carrier's own line identity changes shot to shot (sometimes a
        // defender carries, sometimes a forward), a role's offset
        // FLIPS SIGN depending on who's on the ball right now, and that
        // flipping cancels out under a time-average. Replaced with
        // `TeamTacticalState.defensive_line_x` — already a STABLE,
        // phase-aware, once-per-tick team-level depth (pushes higher in
        // Attack/HighPress, drops in LowBlock; see its own doc comments
        // in teamplay/tactical.rs) that doesn't reset identity based on
        // who's touching the ball. Every line is now a FIXED real offset
        // from this one stable reference, so the intended separation
        // can't average itself away the same way.
        let back_line_progress = progress(ctx.team().tactical().defensive_line_x)
            .max(MIN_DEPTH_FROM_GOAL);
        let my_line_progress = back_line_progress + LINE_GAP * line_index(group) as f32;
        let target_x = from_progress(my_line_progress).clamp(20.0, field_w - 20.0);
        let target_y = player.start_position.y.clamp(15.0, field_h - 15.0);
        let target = Vector3::new(target_x, target_y, 0.0);

        let to_target = target - player.position;
        let dist = to_target.magnitude();
        if dist < 3.0 {
            return Some(Vector3::zeros());
        }
        let max_speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        Some(to_target.normalize() * max_speed * 0.65)
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
    /// realism-bug (2026-08-27, deep root-cause investigation into why
    /// wide/byline duels barely happen — the founding belief behind the
    /// whole ball-distribution project): no defender was ever assigned
    /// to track an opponent's run into the wide channel toward the
    /// byline. Measured: across 1.9M+ `scan_for_dangerous_runs` calls
    /// (the existing "should I break from shape" detector), ZERO
    /// candidates ever satisfied its own wide+deep+fast gate — its 100u
    /// search radius is shorter than the ~150-173u median distance
    /// defenders actually sit at when a wide run reaches this zone
    /// (direct position measurement: nearest opponent in the genuine
    /// deep/wide byline-approach zone is median 173u/21.6m away, within
    /// 3m only 3.5% of the time), and its goal-CENTER-alignment model
    /// structurally under-recognizes a lateral byline run (a run
    /// parallel to the touchline becomes LESS aligned with "toward goal
    /// center" the closer it gets to the byline — backwards from real
    /// danger). This is a genuine new structural override — the class
    /// of fix this engine has repeatedly shown works (width_hold/
    /// assignment/shot_block overrides above), unlike reward-value
    /// tuning, which failed on this exact family of problem many times
    /// (docs/ball-distribution-fix-plan.md). Assigns the single nearest
    /// own-team defender (not already engaging another opponent, no
    /// explicit press/mark assignment already covering them) to shadow
    /// an opposing player making a genuine wide run into the defensive
    /// third, goal-side of them, all the way to the byline if needed —
    /// the fullback-tracks-the-winger duty real football has and this
    /// engine never modeled.
    fn wide_run_track_override_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        if crate::r#match::player::strategies::common::kicker_mode::is_active() {
            return None; // kicker-mode baseline: gated off, see kicker_line_hold_velocity
        }
        let player = ctx.player;
        if player.press_target.is_some() || player.mark_target.is_some() {
            return None; // an explicit assignment always wins
        }
        if matches!(ctx.player.state, Defender(DefenderState::Tackling)) {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None; // can't track a runner while also carrying the ball
        }
        // The persistent assignment (who to track, and whether the
        // assignment is still live) is decided in `run.rs`'s per-tick
        // loop — see `wide_run_track_target`'s own doc comment for why
        // this needs to be persistent rather than recomputed here every
        // tick. This function only steers.
        let target_id = player.wide_run_track_target?;
        let runner_pos = ctx.tick_context.positions.players.position(target_id);

        #[cfg(feature = "match-logs")]
        {
            use crate::r#match::player::strategies::players::ops::defensive::byline_run_diag as brd;
            use std::sync::atomic::Ordering;
            brd::WRT_FIRED.fetch_add(1, Ordering::Relaxed);
            brd::WRT_SUM_START_DIST_X100.fetch_add(
                ((runner_pos - player.position).magnitude() * 100.0) as u64,
                Ordering::Relaxed,
            );
        }

        // Shadow from the goal side, a few units off the runner's own
        // position — a real tracking defender shepherds from inside/
        // goal-side, not by running straight through them.
        let own_goal = ctx.ball().direction_to_own_goal();
        let to_goal = (own_goal - runner_pos).normalize();
        let shadow_point = runner_pos + to_goal * 6.0;
        let to_target = shadow_point - player.position;
        let dist = to_target.magnitude();
        if dist < 2.0 {
            return Some(Vector3::zeros());
        }
        let max_speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        Some(to_target.normalize() * max_speed)
    }

    /// realism-bug (2026-09-12, `docs/positional-play-master-design.md`
    /// Follow-up #8): `width_commitment_ticks`'s real bug wasn't the arm
    /// gate (opportunity/room-check rate was already healthy) — it was
    /// that the consumption logic lived ONLY inside
    /// `ForwardRunningState::velocity()`. A raw position+state trace of
    /// real arm episodes showed the committed player spends most of a
    /// commitment window in OTHER forward states entirely (Pressing,
    /// Returning, CreatingSpace, Intercepting, RunningInBehind,
    /// TakeBall) — none of which ever consulted the field, so the
    /// commitment was silently ignored whenever `process()` picked any
    /// state but `Running`. Fixed the same way every other
    /// state-spanning commitment in this file is fixed: a single
    /// processor-level override that runs regardless of which state the
    /// player's own `process()` selected, mirroring
    /// `wide_run_track_override_velocity`/`give_and_go_run_velocity`
    /// immediately above. This is a state-independence fix, not a new
    /// behavior — the target geometry is unchanged from the original
    /// `ForwardRunningState`-only version.
    fn width_commitment_override_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        if crate::r#match::player::strategies::common::kicker_mode::is_active() {
            return None; // kicker-mode baseline: gated off, see kicker_line_hold_velocity
        }
        let player = ctx.player;
        if player.width_commitment_ticks == 0 {
            return None;
        }
        if player.press_target.is_some() || player.mark_target.is_some() {
            return None; // an explicit assignment always wins
        }
        if matches!(ctx.player.state, Forward(ForwardState::Tackling)) {
            return None; // don't abort a genuine tackle attempt mid-duel
        }
        if ctx.ball().owner_id() == Some(player.id) {
            // Should already be released to 0 by `run.rs`'s per-tick
            // loop the instant he receives the ball; this is a defensive
            // fallback so on-ball decision logic is never overridden.
            return None;
        }
        let field_width = ctx.context.field_size.width as f32;
        let field_height = ctx.context.field_size.height as f32;
        let ball_pos = ctx.tick_context.positions.ball.position;
        let near_flank_y = if player.position.y < field_height * 0.5 {
            field_height * 0.08
        } else {
            field_height * 0.92
        };
        let attacking_direction = match player.side {
            Some(crate::r#match::PlayerSide::Left) => 1.0,
            Some(crate::r#match::PlayerSide::Right) => -1.0,
            None => 0.0,
        };
        let target_x = (ball_pos.x + attacking_direction * 40.0).clamp(30.0, field_width - 30.0);
        let target = Vector3::new(target_x, near_flank_y, 0.0);
        let dist_to_target = (target - player.position).magnitude();
        if dist_to_target < 8.0 {
            return Some(Vector3::zeros());
        }
        let max_speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        Some((target - player.position).normalize() * max_speed)
    }

    /// Chunk 5b (docs/zone-staged-possession-progression.md): the "go"
    /// half of a give-and-go started under
    /// `AttackPattern::UnderlapOverlapCombination`. `run.rs`'s arm
    /// branch (sibling to Chunk 5's own `WideIsolation` one) pairs
    /// `give_and_go_run_ticks`/`give_and_go_run_target_x/y` with the
    /// pass-preference bias `combo_partner` (`evaluator.rs`) — this
    /// function only steers, using the SAME frozen-at-arm-time target
    /// pattern as `round_keeper_target_x/y` (a real player commits to a
    /// route decided at the moment he releases the ball, not one that
    /// keeps drifting) and the SAME real-pace conversion
    /// `wide_run_track_override_velocity` above uses
    /// (`max_speed_with_condition`, not the raw skill value
    /// `rebound_crash_velocity` below still has as a known, separately
    /// flagged bug).
    ///
    /// Deliberately does NOT force a return pass to this player — see
    /// `give_and_go_run_ticks`'s own doc comment (player.rs) for why
    /// that's an honest scope limit, not an oversight: the run itself
    /// is what makes him genuinely open, and the already-verified
    /// pass-value machinery rewards feeding an open teammate on its
    /// own.
    fn give_and_go_run_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        if crate::r#match::player::strategies::common::kicker_mode::is_active() {
            return None; // kicker-mode baseline: gated off, see kicker_line_hold_velocity
        }
        let player = ctx.player;
        if player.press_target.is_some() || player.mark_target.is_some() {
            return None; // an explicit assignment always wins
        }
        if player.give_and_go_run_ticks == 0 {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None; // he's got the ball again — normal state machine drives him
        }
        if !ctx.team().is_control_ball() {
            return None; // we lost it — no reason to keep sprinting for a return that can't come
        }
        let target = Vector3::new(
            player.give_and_go_run_target_x,
            player.give_and_go_run_target_y,
            0.0,
        );
        let to_target = target - player.position;
        let dist = to_target.magnitude();
        if dist < 3.0 {
            return Some(Vector3::zeros());
        }
        let max_speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        Some(to_target.normalize() * max_speed)
    }

    /// docs/positional-attack-phase-design.md, Stage 3: the "spin"
    /// half of the drop-and-spin primitive — mirrors
    /// `give_and_go_run_velocity` above exactly (same gates, same
    /// shape), consuming `drop_and_spin_run_ticks`/`_target_x/y`
    /// instead. Deliberately does NOT clamp for offside itself — the
    /// universal `offside_line_hold_velocity` override (engine audit
    /// Module 3) runs last in this chain and clamps whatever velocity
    /// any earlier override produces, so a run "into the box" is
    /// automatically held onside without this function needing its
    /// own copy of that logic.
    fn drop_and_spin_run_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        if crate::r#match::player::strategies::common::kicker_mode::is_active() {
            return None; // kicker-mode baseline: gated off, see kicker_line_hold_velocity
        }
        let player = ctx.player;
        if player.press_target.is_some() || player.mark_target.is_some() {
            return None; // an explicit assignment always wins
        }
        if player.drop_and_spin_run_ticks == 0 {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None; // still deciding what to do with the deep touch
        }
        if !ctx.team().is_control_ball() {
            return None; // we lost it — no run to make
        }
        let target = Vector3::new(
            player.drop_and_spin_target_x,
            player.drop_and_spin_target_y,
            0.0,
        );
        let to_target = target - player.position;
        let dist = to_target.magnitude();
        if dist < 3.0 {
            return Some(Vector3::zeros());
        }
        let max_speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        Some(to_target.normalize() * max_speed)
    }

    /// realism-bug (2026-08-30, deflection-rate investigation): a
    /// genuine pre-shot corridor gap, not (only) an in-flight
    /// reaction-time one. A temporary diagnostic (`shot_dispatch_diag`,
    /// `player/events/players.rs`) measured the shot corridor at the
    /// exact instant of dispatch (t=0, before any in-flight reaction
    /// has a chance to matter): ~83% of open-play shots have NO
    /// opposing outfielder anywhere near the eventual shot line at all,
    /// even though a raw "is anyone nearby" control measure shows a
    /// defender is typically only ~4.7m from the shooter — he's usually
    /// the player who was just beaten (behind/beside the shooter), not
    /// one positioned goal-side on the shooting lane.
    ///
    /// `DefensiveOperationsImpl::cover_target_position()` already
    /// computes the right predictive point (goal-side of the carrier,
    /// on the line toward goal centre — the best guess available before
    /// the shot direction itself is known) and `DefenderCoveringState`
    /// already steers toward it — but only softly, from inside a
    /// specific FSM state a real defender doesn't reliably occupy at
    /// the moment that matters (role reassignment, a tether blend
    /// toward the formation anchor, a dampened recovery-run speed).
    /// This promotes the same target to a hard override — the identical
    /// "an override wins the tick outright, a state-internal nudge
    /// doesn't" pattern already proven for width/shape/offside/the
    /// reactive shot-block lunge just below — so the single best-placed
    /// (rank-1 `Cover`) defender actually reaches the shot-blocking line
    /// at full pace, continuously, whenever an opponent is genuinely in
    /// shooting range, rather than merely being nudged toward it.
    ///
    /// Yields cleanly (returns `None`) the instant a shot is actually
    /// struck — `shot_block_reaction_velocity` below then takes over,
    /// since it targets the real, live trajectory, a strictly better
    /// signal than this function's predictive "toward goal centre"
    /// proxy for where the shot will actually go.
    fn shot_lane_cover_override_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        if crate::r#match::player::strategies::common::kicker_mode::is_active() {
            return None; // kicker-mode baseline: gated off (no shots dispatched anyway)
        }
        let player = ctx.player;
        if player.press_target.is_some() || player.mark_target.is_some() {
            return None; // an explicit assignment always wins
        }
        // Matches `defensive_role_for_ball_carrier_unified`'s own pool
        // (every defender plus every reachable midfielder) — restricting
        // this to defenders only would silently starve the override
        // whenever the true rank-1 covering player is a tracked-back
        // midfielder, since role is computed per-caller and only one
        // player can genuinely hold rank 1.
        let position = player.tactical_position.current_position;
        if !(position.is_defender() || position.is_midfielder()) {
            return None;
        }
        if matches!(
            ctx.player.state,
            Defender(DefenderState::Tackling) | Midfielder(MidfielderState::Tackling)
        ) {
            return None;
        }
        if ctx.ball().owner_id() == Some(player.id) {
            return None; // can't cover the shot line while carrying the ball
        }
        if ctx.tick_context.ball.cached_shot_target.is_some() {
            return None; // a shot is already live — shot_block_reaction_velocity owns this
        }
        let carrier = ctx.players().opponents().with_ball().next()?;
        let own_goal = ctx.ball().direction_to_own_goal();
        if (own_goal - carrier.position).magnitude() > 220.0 {
            return None; // matches evaluate_forward_shot_decision's own shooting-range cap
        }
        if ctx.player().defensive().defensive_role_for_ball_carrier_unified()
            != crate::r#match::player::strategies::common::players::ops::defensive::DefensiveRole::Cover
        {
            return None; // only the single best-placed covering defender takes this duty
        }
        let max_speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        // Lead-adjusted target, not `cover_target_position()`'s raw
        // instantaneous point. Measured with a temporary diagnostic
        // (`COVEROVERRIDEDIAG`): the override fired ~2700 ticks/match but
        // the covering defender's average distance to the un-led target
        // stayed ~57-58u the whole time — never converging. Root cause,
        // same class already diagnosed for the 2026-07-27 "round the
        // keeper" investigation: a carrier running toward goal drags the
        // "goal-side of the carrier" target along with him at up to his
        // own full pace, so a defender merely chasing the current point
        // only closes the gap by the (often near-zero) DIFFERENCE in
        // their speeds, not his own full speed. Predicting the carrier's
        // position `lead_ticks` ahead (mirroring `DefenderPressingState`'s
        // own "crude lead time" pattern) lets the defender aim at where
        // the corridor will actually be by the time he can get there,
        // rather than perpetually chasing where it was.
        let base_dist = (carrier.position - player.position).magnitude();
        let lead_ticks = if max_speed > 0.01 {
            (base_dist / max_speed).min(30.0)
        } else {
            0.0
        };
        let carrier_velocity = carrier.velocity(ctx);
        let predicted_carrier_pos = carrier.position + carrier_velocity * lead_ticks;
        let to_goal_dir = (own_goal - predicted_carrier_pos).normalize();
        // 12.0 mirrors `DefensiveOperationsImpl::COVER_GOAL_SIDE_OFFSET`
        // (private to defensive.rs) — same real-world "sit a body's
        // width goal-side of the play" depth, applied to the predicted
        // point instead of the raw current one.
        let target = predicted_carrier_pos + to_goal_dir * 12.0;
        let to_target = target - player.position;
        let dist = to_target.magnitude();
        #[cfg(feature = "match-logs")]
        {
            use crate::r#match::engine::player::events::players::shot_dispatch_diag as sdd;
            use crate::r#match::PlayerSide;
            use std::sync::atomic::Ordering;
            sdd::COVER_OVERRIDE_TICKS.fetch_add(1, Ordering::Relaxed);
            sdd::COVER_OVERRIDE_SUM_DIST_X100.fetch_add((dist * 100.0) as u64, Ordering::Relaxed);
            let (last_id_atomic, run_start_atomic, run_ticks_atomic, my_id) = match player.side {
                Some(PlayerSide::Left) => (
                    &sdd::LAST_COVER_ID_HOME,
                    &sdd::RUN_START_DIST_X100_HOME,
                    &sdd::RUN_TICKS_HOME,
                    player.id as u64,
                ),
                _ => (
                    &sdd::LAST_COVER_ID_AWAY,
                    &sdd::RUN_START_DIST_X100_AWAY,
                    &sdd::RUN_TICKS_AWAY,
                    player.id as u64,
                ),
            };
            let prev = last_id_atomic.swap(my_id, Ordering::Relaxed);
            let run_ticks = if prev == my_id {
                sdd::COVER_SAME_TICKS.fetch_add(1, Ordering::Relaxed);
                run_ticks_atomic.fetch_add(1, Ordering::Relaxed) + 1
            } else {
                // New run (either a genuine switch or the very first tick
                // ever) — reset the start-distance baseline and counter.
                if prev != 0 {
                    sdd::COVER_SWITCH_TICKS.fetch_add(1, Ordering::Relaxed);
                }
                run_start_atomic.store((dist * 100.0) as u64, Ordering::Relaxed);
                run_ticks_atomic.store(0, Ordering::Relaxed);
                0
            };
            let run_start_dist = run_start_atomic.load(Ordering::Relaxed) as f64 / 100.0;
            let progress_x100 = ((run_start_dist - dist as f64) * 100.0) as i64;
            sdd::PROGRESS_SUM_X100.fetch_add(progress_x100, Ordering::Relaxed);
            sdd::PROGRESS_SAMPLES.fetch_add(1, Ordering::Relaxed);
            if run_ticks >= 100 {
                sdd::LATE_PROGRESS_SUM_X100.fetch_add(progress_x100, Ordering::Relaxed);
                sdd::LATE_PROGRESS_SAMPLES.fetch_add(1, Ordering::Relaxed);
            }
        }
        if dist < 2.0 {
            return Some(Vector3::zeros());
        }
        Some(to_target.normalize() * max_speed)
    }

    /// realism-bug (2026-08-26, corner/deflection investigation): a
    /// genuine reactive lunge into a live shot's flight path. Raw
    /// diagnostic data (temporary `block_diag`, match-logs only) showed
    /// `Ball::try_block_shot`'s own corridor gate (BLOCK_LOOKAHEAD=40u/
    /// BLOCK_CORRIDOR=7u) is satisfied on only ~0.6% of live-shot ticks
    /// (49/8156 across 20 matches), even though a broad "opponent
    /// goal-side within 10m" cover metric is close to real (84% engine
    /// vs 88% real, StatsBomb Bundesliga 2023/24) — defenders are
    /// generally in the area but essentially never precisely on the
    /// shot's exact line, because nothing ever moves them there: the
    /// corridor check is purely passive (wherever a defender already
    /// happens to be standing). Real defenders react to a struck shot
    /// by throwing a body/leg into its path; this is that reaction.
    ///
    /// This does NOT grant a block directly — `try_block_shot`'s own
    /// existing skill-gated RNG still decides the outcome once (if) the
    /// defender is actually in the corridor; this override only changes
    /// whether they get the chance to be there. Bounded to a short,
    /// realistic reactive range (60u/7.5m) so it can't warp a defender
    /// in from across the pitch, and excluded for direct free kicks
    /// (§13.3's own established rule: no generic outfield block chance
    /// beyond the wall for an already-in-flight dead-ball strike).
    fn shot_block_reaction_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        if crate::r#match::player::strategies::common::kicker_mode::is_active() {
            return None; // kicker-mode baseline: gated off (no shots dispatched anyway)
        }
        let player = ctx.player;
        if player.press_target.is_some() || player.mark_target.is_some() {
            return None; // an explicit assignment always wins
        }
        if player.tactical_position.current_position.is_goalkeeper() {
            return None; // GK has its own dedicated save mechanics
        }
        let shot = ctx.tick_context.ball.cached_shot_target?;
        if shot.is_direct_fk {
            return None;
        }
        // Only the defending side (opponents of the shooter) reacts.
        if player.side != Some(shot.defending_side) {
            return None;
        }
        match ctx.player.state {
            Defender(DefenderState::Tackling)
            | Midfielder(MidfielderState::Tackling)
            | Forward(ForwardState::Tackling) => return None,
            _ => {}
        }

        let ball = &ctx.tick_context.positions.ball;
        let ball_vel_2d = (ball.velocity.x * ball.velocity.x + ball.velocity.y * ball.velocity.y).sqrt();
        if ball_vel_2d < 0.5 {
            return None; // ball has stopped / nearly — not a live shot
        }
        let shot_dir = Vector3::new(
            ball.velocity.x / ball_vel_2d,
            ball.velocity.y / ball_vel_2d,
            0.0,
        );

        let to_player = player.position - ball.position;
        let projection = to_player.x * shot_dir.x + to_player.y * shot_dir.y;
        const REACTION_LOOKAHEAD: f32 = 40.0; // matches try_block_shot's BLOCK_LOOKAHEAD
        let clamped_proj = projection.clamp(1.0, REACTION_LOOKAHEAD);
        let target = ball.position + shot_dir * clamped_proj;

        let to_target = target - player.position;
        let dist = to_target.magnitude();
        // Only a realistic reactive range — a defender already broadly
        // engaged nearby, not one dragged in from across the pitch.
        if !(1.0..=60.0).contains(&dist) {
            return None;
        }

        let max_speed = player
            .skills
            .max_speed_with_condition(player.player_attributes.condition);
        Some(to_target.normalize() * max_speed)
    }

    fn rebound_crash_velocity(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        if crate::r#match::player::strategies::common::kicker_mode::is_active() {
            return None; // kicker-mode baseline: gated off (no shots dispatched anyway)
        }
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

    /// Universal offside-line hold — engine audit Module 3 (2026-08-29),
    /// the codebase's own proposed alternative for offside's flagged "no
    /// single owner" problem. Applied to EVERY state's velocity, after
    /// the handler (and every earlier override) has already produced
    /// one, regardless of which state a player is in.
    ///
    /// The old design required each state to individually opt in via
    /// `onside_hold_x`/`clamp_chase_target_x` — RunningInBehind,
    /// Assisting, Midfielder Walking/Running, and the 4 TakeBall
    /// variants did; Intercepting, Pressing, Standing, and everything
    /// else never did, which is exactly why fixing TakeBall's own gap
    /// (2026-08-19) immediately made Intercepting grow to ~17% of the
    /// remaining violations — direct proof the per-state pattern doesn't
    /// generalize, it just relocates the gap to whichever state wasn't
    /// checked yet. All 8 of those per-state call sites are removed as
    /// part of this same change — this function is now the only place
    /// offside movement discipline is enforced.
    ///
    /// A downstream clamp on the FINAL proposed velocity, rather than an
    /// upstream clamp on each state's own target, is what makes this a
    /// structural fix instead of one more place to remember: it cannot
    /// be skipped by a state that never heard of offside, and it still
    /// correctly composes with an earlier override that already touched
    /// the same tick's velocity (e.g. width-holding) — only the forward
    /// (x) component is restricted, so a legitimate lateral/width
    /// adjustment from an earlier override passes through untouched.
    /// Placed LAST among the velocity overrides in `process_inner` for
    /// exactly this reason — it must see the truly final proposed
    /// velocity, not an intermediate one a later override might still
    /// replace.
    ///
    /// Reuses `clamp_chase_target_x`'s existing line/margin/ball-crossed
    /// math (translating this tick's proposed x-displacement into an
    /// equivalent "chase target" and clamping that) rather than
    /// re-deriving the geometry a third time.
    ///
    /// Deliberately does NOT exclude goalkeepers — Law 11 applies to
    /// every player on the pitch regardless of role, matching the
    /// existing GK TakeBall clamp this replaces (added per an explicit
    /// "including GK" scope decision). In practice a GK essentially
    /// never proposes a velocity that would trip this, since
    /// `should_force_takeball`'s own 60u claim radius keeps him anchored
    /// near his own box.
    fn offside_line_hold_velocity(
        ctx: &StateProcessingContext,
        proposed_velocity: Option<Vector3<f32>>,
    ) -> Option<Vector3<f32>> {
        let player = ctx.player;
        // Law 11: no offside offense is possible when receiving directly
        // from a corner, throw-in, or goal kick.
        if ctx.ball().is_offside_exempt_restart() {
            return None;
        }
        // Ball carrier is exempt — you can't be offside while you hold it.
        if ctx.ball().owner_id() == Some(player.id) {
            return None;
        }
        // Offside only matters while our own team controls (or is
        // genuinely about to regain) the ball; if the opponent has it,
        // there's nothing to be offside FROM. This also structurally
        // keeps this override from ever competing with
        // `assignment_override_velocity`'s active press/mark branches —
        // both of those only fire when the OPPONENT holds the ball — so
        // no explicit priority re-check against it is needed here the
        // way most other overrides in this chain require.
        if !ctx.team().is_control_ball() {
            return None;
        }
        let v = proposed_velocity?;
        if v.x == 0.0 {
            return None;
        }
        let natural_target_x = player.position.x + v.x;
        let ball_x = ctx.tick_context.positions.ball.position.x;
        let clamped_x = ctx
            .player()
            .defensive()
            .clamp_chase_target_x(natural_target_x, ball_x);
        if (clamped_x - natural_target_x).abs() < 0.01 {
            return None; // no clamp needed, let the existing velocity stand
        }
        Some(Vector3::new(clamped_x - player.position.x, v.y, v.z))
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
