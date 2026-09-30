use crate::r#match::events::Event;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::midfielders::states::common::{ActivityIntensity, MidfielderCondition};
use crate::r#match::player::events::PlayerEvent;
use crate::r#match::player::strategies::common::players::ops::defensive::DefensiveOperationsImpl;
use crate::r#match::player::strategies::common::players::ops::midfielder_skill::MidfielderSkillProfile;
use crate::r#match::player::strategies::common::players::ops::tackle::{self, DuelAction, DuelSkills, TackleChoice};
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
    SteeringBehavior,
};
use nalgebra::Vector3;
#[cfg(feature = "match-logs")]
use std::sync::atomic::Ordering;

// 2026-09-29: attempt/contain/concede-out logic replaced by the shared
// unified tackle system (`common/players/ops/tackle.rs`) — see the
// defender state for the full note. Engagement range is the shared
// `TACKLE_ENGAGEMENT_DISTANCE`.

#[derive(Default, Clone)]
pub struct MidfielderTacklingState {}

impl MidfielderTacklingState {
    fn duel_skills(ctx: &StateProcessingContext) -> DuelSkills {
        let p = MidfielderSkillProfile::from_ctx(ctx);
        DuelSkills {
            tackle: p.tackle_profile,
            discipline: p.discipline,
            condition: p.mid_condition_mult,
        }
    }
}

impl StateProcessingHandler for MidfielderTacklingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        #[cfg(feature = "match-logs")]
        crate::tackle_stats::MID_ENTRIES.fetch_add(1, Ordering::Relaxed);

        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // Don't try to claim a ball in protected flight.
        if ctx.ball().is_in_flight() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // Law 13: no challenge inside the free-kick retreat distance.
        if ctx.ball().is_free_kick_encroaching() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Pressing,
            ));
        }

        // Law 12: a goalkeeper holding the ball / opponent goal kick.
        if ctx.ball().is_opponent_restart() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Pressing,
            ));
        }

        let ball_distance = ctx.ball().distance();

        if ball_distance > 150.0 {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Returning,
            ));
        }

        // If ball is moving away but opponent still nearby, keep pressing
        if ball_distance > 80.0 && !ctx.ball().is_towards_player_with_angle(0.8) {
            return if ctx.team().is_control_ball() {
                Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::AttackSupporting,
                ))
            } else {
                Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Pressing,
                ))
            };
        }

        // Per-player tackle cooldown.
        if !ctx.player.can_attempt_tackle() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Pressing,
            ));
        }

        // Only the best-positioned teammate engages; the rest cover.
        if !ctx.team().is_best_player_to_chase_ball() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Pressing,
            ));
        }

        let opponents = ctx.players().opponents();
        let mut opponents_with_ball = opponents.with_ball();

        if let Some(opponent) = opponents_with_ball.next() {
            let opponent_distance = ctx.tick_context.grid.get(ctx.player.id, opponent.id);
            if opponent_distance > DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Pressing,
                ));
            }

            return match tackle::duel(ctx, &opponent, Self::duel_skills(ctx)) {
                DuelAction::Hold => {
                    #[cfg(feature = "match-logs")]
                    crate::tackle_stats::MID_GATE_CONTAIN_OR_CONCEDE
                        .fetch_add(1, Ordering::Relaxed);
                    None
                }
                DuelAction::Act(events) => {
                    #[cfg(feature = "match-logs")]
                    crate::tackle_stats::MID_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
                    let next = if events.is_empty() {
                        MidfielderState::Pressing
                    } else {
                        MidfielderState::Standing
                    };
                    let mut result = StateChangeResult::with_midfielder_state(next);
                    result.events.add_range(events);
                    result.start_tackle_cooldown = true;
                    Some(result)
                }
            };
        } else if self.can_intercept_ball(ctx) {
            // can_intercept_ball already checks is_in_flight
            return Some(StateChangeResult::with_midfielder_state_and_event(
                MidfielderState::Running,
                Event::PlayerEvent(PlayerEvent::ClaimBall(ctx.player.id)),
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Holding: stand off goal-side and mirror the carrier.
        if let Some(opponent) = ctx.players().opponents().with_ball().next() {
            if opponent.distance(ctx) <= DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE
                && tackle::choose(ctx, &opponent, Self::duel_skills(ctx), ctx.in_state_time as f32)
                    == TackleChoice::Hold
            {
                return Some(tackle::jockey_velocity(ctx, &opponent));
            }
        }

        let tackling_skill = ctx.player.skills.technical.tackling / 20.0;
        let pace = ctx.player.skills.physical.acceleration / 20.0;
        // Explosive closing speed — skilled tacklers close gaps faster
        let speed_boost = 1.3 + tackling_skill * 0.3 + pace * 0.3; // 1.3x - 1.9x

        Some(
            SteeringBehavior::Pursuit {
                target: ctx.tick_context.positions.ball.position,
                target_velocity: ctx.tick_context.positions.ball.velocity,
            }
            .calculate(ctx.player)
            .velocity
                * speed_boost
                + ctx.player().separation_velocity() * 0.2,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Tackling is explosive and very demanding physically
        MidfielderCondition::new(ActivityIntensity::VeryHigh).process(ctx);
    }
}

impl MidfielderTacklingState {
    fn can_intercept_ball(&self, ctx: &StateProcessingContext) -> bool {
        if ctx.ball().is_in_flight() {
            return false;
        }

        let ball_position = ctx.tick_context.positions.ball.position;
        let ball_velocity = ctx.tick_context.positions.ball.velocity;
        let player_position = ctx.player.position;
        let player_speed = ctx.player.skills.physical.pace;

        if !ctx.tick_context.ball.is_owned && ball_velocity.magnitude() > 0.1 {
            let time_to_ball = (ball_position - player_position).magnitude() / player_speed;
            let ball_travel_distance = ball_velocity.magnitude() * time_to_ball;
            let ball_intercept_position =
                ball_position + ball_velocity.normalize() * ball_travel_distance;
            let player_intercept_distance = (ball_intercept_position - player_position).magnitude();

            player_intercept_distance <= DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE
        } else {
            false
        }
    }
}
