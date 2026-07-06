use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::midfielders::states::common::{ActivityIntensity, MidfielderCondition};
use crate::r#match::player::strategies::common::set_pieces::player_corner_zone;
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
    SteeringBehavior,
};
use nalgebra::Vector3;

/// Midfielder making a pre-planned run to their assigned corner zone.
/// Exits back to Running when the corner is over or possession changes.
#[derive(Default, Clone)]
pub struct MidfielderCornerRunState {}

impl StateProcessingHandler for MidfielderCornerRunState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        if !ctx.ball().is_team_attacking_corner() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // Only exit if someone has the ball — ball in flight means current_owner
        // is None, so runners hold their zones until the delivery is claimed.
        if player_corner_zone(ctx).is_none()
            && ctx.tick_context.ball.current_owner.is_some()
        {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let zone_target = match player_corner_zone(ctx) {
            Some(t) => t,
            None => return Some(Vector3::zeros()),
        };

        let dist = (zone_target - ctx.player.position).magnitude();
        if dist < 4.0 {
            return Some(Vector3::zeros());
        }

        Some(
            SteeringBehavior::Seek { target: zone_target }
                .calculate(ctx.player)
                .velocity,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        MidfielderCondition::new(ActivityIntensity::High).process(ctx);
    }
}
