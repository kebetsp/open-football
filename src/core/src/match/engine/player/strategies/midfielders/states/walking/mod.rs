use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::midfielders::states::common::{ActivityIntensity, MidfielderCondition};
use crate::r#match::player::strategies::common::players::MatchPlayerIteratorExt;
use crate::r#match::player::strategies::common::players::ops::slot_coverage::find_vacant_forward_slot;
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
    SteeringBehavior,
};
use nalgebra::Vector3;

#[derive(Default, Clone)]
pub struct MidfielderWalkingState {}

impl StateProcessingHandler for MidfielderWalkingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // Corner set-piece: take assigned zone position.
        if ctx.ball().is_team_attacking_corner() {
            use crate::r#match::player::strategies::common::set_pieces::player_corner_zone;
            if player_corner_zone(ctx).is_some() {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::CornerRun,
                ));
            }
        }

        // Opponent goal kick — don't press the GK, let velocity() pull back
        if ctx.ball().is_opponent_goal_kick() {
            return None;
        }

        // CRITICAL: Check for opponent with ball first (highest priority)
        // Using new chaining syntax: nearby(100.0).with_ball(ctx)
        if let Some(opponent) = ctx
            .players()
            .opponents()
            .nearby(60.0)
            .with_ball(ctx)
            .next()
        {
            let opponent_distance = (opponent.position - ctx.player.position).magnitude();

            // If opponent with ball is close, tackle immediately
            if opponent_distance < 40.0 {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Tackling,
                ));
            }

            // If opponent with ball is nearby, press them (already filtered by nearby(100.0))
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Pressing,
            ));
        }

        // Ball in flight (clearance, long pass) — anticipate the landing
        // so midfielders contest it instead of letting opposing players
        // collect every clearance. Same pattern as Standing: run the
        // intercept state if predicted landing is inside our reach.
        if !ctx.ball().is_owned() && ctx.ball().is_in_flight() {
            let landing = ctx.tick_context.positions.ball.landing_position;
            if (landing - ctx.player.position).norm_squared() < 100.0 * 100.0 {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Intercepting,
                ));
            }
        }

        // Loose-ball claim lives in the dispatcher.

        // Notification: take ball only if best positioned
        if ctx.ball().should_take_ball_immediately() && ctx.team().is_best_player_to_chase_ball() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::TakeBall,
            ));
        }

        if ctx.team().is_control_ball() {
            if ctx.ball().is_towards_player_with_angle(0.8) && ctx.ball().distance() < 250.0 {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Intercepting,
                ));
            }
        } else {
            if ctx.ball().distance() < 200.0 && ctx.ball().stopped() {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Running,
                ));
            }

            if ctx.ball().is_towards_player_with_angle(0.8) {
                if ctx.ball().distance() < 100.0 {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Intercepting,
                    ));
                }

                if ctx.ball().distance() < 150.0 && ctx.ball().is_towards_player_with_angle(0.8) {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Pressing,
                    ));
                }
            }

            // Compute the nearest opponent distance in a single pass — no
            // intermediate Vec.
            let mut any_nearby = false;
            let mut closest_opponent_distance = f32::MAX;
            for opponent in ctx.players().opponents().nearby(150.0) {
                any_nearby = true;
                let distance = ctx.player().distance_to_player(opponent.id);
                if distance < closest_opponent_distance {
                    closest_opponent_distance = distance;
                }
            }

            if any_nearby {
                let ball_distance = ctx.ball().distance();
                if ball_distance < 50.0 && closest_opponent_distance < 50.0 {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Tackling,
                    ));
                }
            }
        }

        // Don't walk when opponent has ball on our side — go press or guard
        if !ctx.team().is_control_ball() && ctx.ball().on_own_side() {
            if ctx.ball().distance() < 150.0 {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Pressing,
                ));
            }
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Guarding,
            ));
        }

        // During possession, hold shape for longer — velocity() uses the
        // dynamic anchor to keep midfielders in position rather than rushing
        // toward the ball. When the opponent has the ball, exit quickly.
        let walk_limit = if ctx.team().is_control_ball() { 120 } else { 20 };
        if ctx.in_state_time > walk_limit {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        if ctx.ball().is_opponent_goal_kick() {
            return Some(
                SteeringBehavior::Arrive {
                    target: ctx.player.start_position,
                    slowing_distance: 30.0,
                }
                .calculate(ctx.player)
                .velocity,
            );
        }

        // Step into the forward zone if no striker is ahead during attack
        if let Some(slot_target) = find_vacant_forward_slot(ctx) {
            return Some(
                SteeringBehavior::Arrive {
                    target: slot_target,
                    slowing_distance: 30.0,
                }
                .calculate(ctx.player)
                .velocity,
            );
        }

        if ctx.player.should_follow_waypoints(ctx) {
            let waypoints = ctx.player.get_waypoints_as_vectors();

            if !waypoints.is_empty() {
                return Some(
                    SteeringBehavior::FollowPath {
                        waypoints,
                        current_waypoint: ctx.player.waypoint_manager.current_index,
                        path_offset: 5.0,
                    }
                    .calculate(ctx.player)
                    .velocity,
                );
            }
        }

        // Dynamic shape anchor: the team block shifts longitudinally with the
        // ball. Factor 0.15 means a ball 200u ahead of centre shifts the
        // anchor ~30u forward — enough to feel connected without emptying the
        // midfield. Works for both attacking directions because the sign of
        // (ball_x - centre_x) naturally flips when the team defends the other end.
        let pitch_center_x = ctx.context.field_size.width as f32 / 2.0;
        let ball_x = ctx.tick_context.positions.ball.position.x;
        let mut shape_anchor = ctx.player.start_position;
        shape_anchor.x += (ball_x - pitch_center_x) * 0.15;

        let dist = (shape_anchor - ctx.player.position).magnitude();
        if dist < 5.0 {
            return Some(Vector3::zeros());
        }

        Some(
            SteeringBehavior::Arrive {
                target: shape_anchor,
                slowing_distance: 30.0,
            }
            .calculate(ctx.player)
            .velocity
                * 0.4, // Walking pace
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Walking is low intensity - minimal fatigue
        MidfielderCondition::with_velocity(ActivityIntensity::Low).process(ctx);
    }
}
