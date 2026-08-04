use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::forwarders::states::common::{ActivityIntensity, ForwardCondition};
use crate::r#match::{
    ConditionContext, MATCH_TIME_MS, StateChangeResult, StateProcessingContext,
    StateProcessingHandler, SteeringBehavior,
};
use nalgebra::Vector3;

#[derive(Default, Clone)]
pub struct ForwardReturningState {}

impl StateProcessingHandler for ForwardReturningState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        // Take ball only if best positioned — prevents swarming
        if ctx.ball().should_take_ball_immediately() && ctx.team().is_best_player_to_chase_ball() {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::TakeBall,
            ));
        }

        if ctx.team().is_control_ball() {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        // realism-bug (2026-08-04): Law 12 — a goalkeeper in secure
        // possession (or a live opponent goal kick) is dead and cannot
        // be legally contested. Without this exclusion, the two checks
        // below immediately route a forward who just correctly started
        // retreating (Pressing/Running → Returning, on the same
        // `is_opponent_restart()` signal) straight back into
        // Intercepting/Tackling — which this state exists specifically
        // to retreat AWAY from — producing a Returning↔Intercepting/
        // Tackling↔Pressing oscillation that keeps him glued near the
        // keeper instead of actually walking back to his anchor.
        let contestable = !ctx.ball().is_opponent_restart();

        if contestable && !ctx.team().is_control_ball() && ctx.ball().distance() < 200.0 {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Intercepting,
            ));
        }

        if contestable && !ctx.team().is_control_ball() && ctx.ball().distance() < 100.0 {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Tackling,
            ));
        }

        // Stay in returning state until very close to start position
        if ctx.player().distance_from_start_position() < 2.0 {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Standing,
            ));
        }

        // Intercept if ball coming towards player and is closer than before
        if contestable && !ctx.team().is_control_ball() && ctx.ball().is_towards_player_with_angle(0.9) {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Intercepting,
            ));
        }

        // Transition to Pressing late in the game only if ball is close as well
        if contestable
            && ctx.team().is_loosing()
            && ctx.context.total_match_time > (MATCH_TIME_MS - 180)
            && ctx.ball().distance() < 30.0
        {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Pressing,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        Some(
            SteeringBehavior::Arrive {
                target: ctx.player.start_position,
                slowing_distance: 10.0,
            }
            .calculate(ctx.player)
            .velocity,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Returning is moderate intensity - getting back to position
        ForwardCondition::with_velocity(ActivityIntensity::Moderate).process(ctx);
    }
}
