use crate::r#match::events::Event;
use crate::r#match::goalkeepers::states::common::{ActivityIntensity, GoalkeeperCondition};
use crate::r#match::goalkeepers::states::state::GoalkeeperState;
use crate::r#match::player::events::PlayerEvent;
use crate::r#match::player::strategies::players::ops::goalkeeper_skill::{
    GoalkeeperSkillProfile,
};
use crate::r#match::{
    ConditionContext, PlayerDistanceFromStartPosition, StateChangeResult, StateProcessingContext,
    StateProcessingHandler, SteeringBehavior,
};
use nalgebra::Vector3;

#[derive(Default, Clone)]
pub struct GoalkeeperCatchingState {}

impl StateProcessingHandler for GoalkeeperCatchingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if self.is_catch_successful(ctx) {
            let mut holding_result =
                StateChangeResult::with_goalkeeper_state(GoalkeeperState::HoldingBall);

            holding_result
                .events
                .add_player_event(PlayerEvent::CaughtBall(ctx.player.id));

            return Some(holding_result);
        }

        // Shot is live: stay in Catching and keep sprinting toward the
        // intercept line. The old logic exited to Standing / ComingOut
        // the moment the ball was >12u away, which meant a keeper
        // aiming for the far post gave up the instant the shot was
        // fired. With a cached shot target the keeper commits.
        if ctx.tick_context.ball.cached_shot_target.is_some() {
            return None;
        }

        // Ball is moving away from the keeper at speed — only credit
        // a parry when the ball was actually within reach (the keeper
        // got a hand to it). Otherwise the shot just missed past the
        // keeper and "parry" would falsely credit a save for a wide
        // shot the GK never touched.
        let ball_speed = ctx.tick_context.positions.ball.velocity.norm();
        let ball_distance = ctx.ball().distance();
        if ball_speed > 2.0 && !ctx.ball().is_towards_player_with_angle(0.6) {
            if ctx.tick_context.ball.cached_shot_target.is_some() && ball_distance < 25.0 {
                return Some(StateChangeResult::with_goalkeeper_state_and_event(
                    GoalkeeperState::Standing,
                    Event::PlayerEvent(PlayerEvent::ParriedBall(ctx.player.id)),
                ));
            }
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Standing,
            ));
        }

        // If ball is too far, decide based on distance from goal
        if ctx.ball().distance() > 12.0 {
            // If already far from goal, return rather than chasing further
            if ctx.player().distance_from_start_position() > 40.0 {
                return Some(StateChangeResult::with_goalkeeper_state(
                    GoalkeeperState::ReturningToGoal,
                ));
            }
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::ComingOut,
            ));
        }

        if ctx.player().position_to_distance() == PlayerDistanceFromStartPosition::Big {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::ReturningToGoal,
            ));
        }

        if ctx.in_state_time > 30 {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Standing,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let prof = GoalkeeperSkillProfile::from_ctx(ctx);
        // Sprint reaction speed: 1.6..2.6x band, gated by explosive_mult.
        let speed_boost =
            (1.6 + prof.shot_stopping * 0.5 + prof.dive_reach * 0.5) * prof.explosive_mult;

        // Shot in flight → commit to the intercept line, don't chase
        // the current ball position (it's moving at 5.6 u/tick and
        // outrunning the keeper's pursuit steering).
        if let Some(target) = &ctx.tick_context.ball.cached_shot_target {
            let goal_pos = ctx.ball().direction_to_own_goal();
            let intercept = Vector3::new(goal_pos.x, target.goal_line_y, 0.0);
            // realism-bug (2026-07-30, extended 2026-07-31): a direct
            // free kick's flight time (130-250u away, 400-800+ms / 40-90
            // ticks) is long enough that even at NORMAL running speed
            // (the 1.0x this comment originally capped it to) the keeper
            // can walk down almost any reasonable initial gap over the
            // whole flight — measured directly: after fixing shot aim to
            // exploit the keeper's actual starting position, the median
            // lateral error at save-check time barely moved (1.89u →
            // 1.84u across 5000+ evaluations), because the keeper simply
            // walks the rest of the way to the EXACT, perfectly-known
            // `target.goal_line_y` regardless of where he started. A
            // real keeper mostly commits to his stance BEFORE the kick
            // (already modelled in `calculate_optimal_position`'s
            // far-post lean) and only makes a small late reactive
            // adjustment/dive as the ball arrives — not a sustained,
            // omniscient walk across the goal for the better part of a
            // second. Capped to a slow creep (reflecting that small late
            // adjustment) instead of full running speed; the keeper's
            // actual SAVE range is unaffected — `is_catch_successful`'s
            // separate `reach` term already models diving distance on
            // top of wherever he's standing when the ball arrives. Every
            // other shot type (including normal long-range open-play
            // shots) keeps the existing boosted-sprint behaviour.
            let effective_boost = if target.is_direct_fk { 0.2 } else { speed_boost };

            // realism-bug (2026-08-02, Pavel's direction): the reaction-
            // delay freeze (previously here, 150ms then retuned to 80ms)
            // is removed entirely as a direct test of whether it has any
            // measurable effect on the saved-vs-goal split at all, given
            // outcome batches kept reading noisy/inconsistent across
            // several delay values (0/6/8/15 ticks) — see the decisions
            // log 2026-08-01/02 entries for the full measurement history.
            // `ShotTarget.dispatch_tick` (added for this mechanism) is
            // left in place, unused here, in case the delay is revisited.

            return Some(
                SteeringBehavior::Arrive {
                    target: intercept,
                    slowing_distance: 2.0,
                }
                .calculate(ctx.player)
                .velocity
                    * effective_boost,
            );
        }

        let ball_distance = ctx.ball().distance();
        if ball_distance > 3.0 {
            Some(
                SteeringBehavior::Pursuit {
                    target: ctx.tick_context.positions.ball.position,
                    target_velocity: ctx.tick_context.positions.ball.velocity,
                }
                .calculate(ctx.player)
                .velocity
                    * speed_boost,
            )
        } else {
            Some(
                SteeringBehavior::Arrive {
                    target: ctx.tick_context.positions.ball.position,
                    slowing_distance: 1.5,
                }
                .calculate(ctx.player)
                .velocity
                    * (speed_boost * 0.8),
            )
        }
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Catching is a moderate intensity activity requiring focused effort
        GoalkeeperCondition::new(ActivityIntensity::Moderate).process(ctx);
    }
}

impl GoalkeeperCatchingState {
    fn is_catch_successful(&self, ctx: &StateProcessingContext) -> bool {
        let prof = GoalkeeperSkillProfile::from_ctx(ctx);

        // Shot in flight: the save is decided ONCE, at the moment the ball
        // reaches the keeper, by `Ball::resolve_shot_at_keeper`
        // (ball/ball/save.rs; design: docs/goalkeeper-shot-stopping-from-null.md).
        // This state only steers toward the intercept line. It used to roll
        // a save every tick of the flight and snap the ball to the keeper
        // from any distance (up to 85 m) -- removed 2026-10-14.
        if ctx.tick_context.ball.cached_shot_target.is_some() {
            return false;
        }

        let distance_to_ball = ctx.ball().distance();
        let max_catch_distance = prof.effective_catch_distance;
        if distance_to_ball > max_catch_distance {
            return false;
        }

        let ball_speed = ctx.tick_context.positions.ball.velocity.norm();
        if ball_speed > 0.5 && !ctx.ball().is_towards_player_with_angle(0.6) {
            return false;
        }

        let ball_height = ctx.tick_context.positions.ball.position.z;
        let stretch = (distance_to_ball / max_catch_distance).clamp(0.0, 1.0);
        let power = ((ball_speed - 1.5) / 6.0).clamp(0.0, 1.0);

        // Awkward-height penalty: ground or above-head balls are harder.
        let height_pen = if (0.5..=1.8).contains(&ball_height) {
            0.0
        } else if ball_height < 0.2 {
            0.18
        } else if ball_height > 2.5 {
            0.22
        } else {
            0.06
        };

        let direction_factor = if ctx.ball().is_towards_player_with_angle(0.7) {
            0.0
        } else {
            0.18
        };

        let catch_difficulty = (power * 0.28
            + stretch * 0.22
            + height_pen * 0.18
            + direction_factor * 0.12
            + (1.0 - prof.condition_mult) * 0.10
            + prof.poor_skill_penalty * 0.10)
            .clamp(0.0, 1.0);

        let catch_prob = prof.catch_probability(catch_difficulty);
        ctx.context.rng.unit_f32() < catch_prob
    }
}
