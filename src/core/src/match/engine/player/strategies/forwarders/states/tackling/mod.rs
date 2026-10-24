use crate::r#match::events::Event;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::forwarders::states::common::{ActivityIntensity, ForwardCondition};
use crate::r#match::player::events::PlayerEvent;
use crate::r#match::player::strategies::common::players::ops::defensive::DefensiveOperationsImpl;
use crate::r#match::player::strategies::common::players::ops::tackle::{self, DuelAction, DuelSkills, TackleChoice};
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::player::strategies::players::skills::SkillCurve;
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
    SteeringBehavior,
};
use nalgebra::Vector3;

// 2026-09-29: attempt logic replaced by the shared unified tackle
// system (`common/players/ops/tackle.rs`) — see the defender state for
// the full note. Forwards keep their skill gate below: most strikers
// don't drill defensive tackles.
const TACKLE_DISTANCE_THRESHOLD: f32 = 8.0; // loose-ball interception reach only
const CHASE_DISTANCE_THRESHOLD: f32 = 100.0; // Maximum distance to chase for tackle
const PRESSURE_DISTANCE: f32 = 20.0; // Distance to apply pressure without tackling

#[derive(Default, Clone)]
pub struct ForwardTacklingState {}

impl ForwardTacklingState {
    fn duel_skills(ctx: &StateProcessingContext) -> DuelSkills {
        let minute = sc::minute_from_ms(ctx.context.total_match_time);
        let s = &ctx.player.skills;
        DuelSkills {
            tackle: sc::defensive_duel(ctx.player, minute).clamp(0.0, 1.0),
            discipline: ((s.mental.composure + s.mental.decisions) / 40.0).clamp(0.0, 1.0),
            condition: (ctx.player.player_attributes.condition as f32 / 10000.0).clamp(0.5, 1.0),
        }
    }
}

impl StateProcessingHandler for ForwardTacklingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        #[cfg(feature = "match-logs")]
        crate::tackle_stats::FWD_ENTRIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // If player has gained possession, transition to running
        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        // Don't try to claim a ball in protected flight.
        if ctx.ball().is_in_flight() {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        // Law 13: no challenge inside the free-kick retreat distance.
        if ctx.ball().is_free_kick_encroaching() {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Pressing,
            ));
        }

        // Law 12: a goalkeeper holding the ball / opponent goal kick.
        if ctx.ball().is_opponent_restart() {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Pressing,
            ));
        }

        // Per-player tackle cooldown.
        if !ctx.player.can_attempt_tackle() {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Pressing,
            ));
        }

        // Only the best-positioned teammate engages.
        if !ctx.team().is_best_player_to_chase_ball() {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Pressing,
            ));
        }

        // Skill gate — most strikers don't drill defensive tackles
        // (sigmoid pivot at 8/20).
        let tackling_p =
            SkillCurve::new(ctx.player.skills.technical.tackling, 8.0, 0.6).probability();
        if ctx.context.rng.unit_f32() >= tackling_p {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Pressing,
            ));
        }

        let opponents = ctx.players().opponents();

        if let Some(opponent) = opponents.with_ball().next() {
            let opponent_distance = ctx.tick_context.grid.get(ctx.player.id, opponent.id);

            if opponent_distance <= DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE {
                return match tackle::duel(ctx, &opponent, Self::duel_skills(ctx)) {
                    DuelAction::Hold => None,
                    DuelAction::Act(events, recovery) => {
                        #[cfg(feature = "match-logs")]
                        crate::tackle_stats::FWD_ATTEMPTS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let next = if events.is_empty() {
                            ForwardState::Pressing
                        } else {
                            ForwardState::Standing
                        };
                        let mut result = StateChangeResult::with_forward_state(next);
                        result.events.add_range(events);
                        result.off_balance_ticks = recovery;
                        Some(result)
                    }
                };
            }

            // If opponent is further but still chaseable, continue pursuit
            if opponent_distance <= CHASE_DISTANCE_THRESHOLD {
                return None; // Continue chasing
            }
        }

        // Check for loose ball interception opportunities
        if !ctx.ball().is_owned() && self.can_intercept_ball(ctx) {
            return Some(StateChangeResult::with_forward_state_and_event(
                ForwardState::Running,
                Event::PlayerEvent(PlayerEvent::ClaimBall(ctx.player.id)),
            ));
        }

        let ball_distance = ctx.ball().distance();

        if ctx.team().is_control_ball() {
            if ball_distance > CHASE_DISTANCE_THRESHOLD {
                return Some(StateChangeResult::with_forward_state(
                    ForwardState::Returning,
                ));
            }

            return Some(StateChangeResult::with_forward_state(
                ForwardState::Assisting,
            ));
        } else if ball_distance <= PRESSURE_DISTANCE {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Pressing,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let opponents = ctx.players().opponents();

        if let Some(opponent) = opponents.with_ball().next() {
            let opponent_distance = ctx.tick_context.grid.get(ctx.player.id, opponent.id);

            if opponent_distance <= DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE {
                // Holding: stand off goal-side and mirror the carrier.
                if tackle::choose(ctx, &opponent, Self::duel_skills(ctx), ctx.in_state_time as f32)
                    == TackleChoice::Hold
                {
                    return Some(tackle::jockey_velocity(ctx, &opponent));
                }
                return Some(
                    SteeringBehavior::Arrive {
                        target: opponent.position,
                        slowing_distance: 1.0,
                    }
                    .calculate(ctx.player)
                    .velocity,
                );
            } else {
                // Chase with urgency that scales with proximity to the
                // opponent's goal (winning it there is a high-value
                // turnover).
                let dist_to_goal =
                    (ctx.player().opponent_goal_position() - opponent.position).magnitude();
                let danger_proximity = (1.0 - (dist_to_goal / 250.0)).clamp(0.0, 1.0);
                let urgency_boost = 1.0 + danger_proximity * 0.6;
                return Some(
                    SteeringBehavior::Pursuit {
                        target: opponent.position,
                        target_velocity: Vector3::zeros(),
                    }
                    .calculate(ctx.player)
                    .velocity
                        * urgency_boost,
                );
            }
        }

        // If no opponent with ball, go for loose ball
        if !ctx.ball().is_owned() {
            return Some(
                SteeringBehavior::Pursuit {
                    target: ctx.tick_context.positions.ball.position,
                    target_velocity: ctx.tick_context.positions.ball.velocity,
                }
                .calculate(ctx.player)
                .velocity,
            );
        }

        // Default movement toward ball position
        Some(
            SteeringBehavior::Arrive {
                target: ctx.tick_context.positions.ball.position,
                slowing_distance: 20.0,
            }
            .calculate(ctx.player)
            .velocity,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Tackling is very high intensity - explosive action
        ForwardCondition::new(ActivityIntensity::VeryHigh).process(ctx);
    }
}

impl ForwardTacklingState {
    /// Check if player can intercept a loose ball
    fn can_intercept_ball(&self, ctx: &StateProcessingContext) -> bool {
        // Don't try to intercept if ball is owned or in flight
        if ctx.ball().is_owned() || ctx.ball().is_in_flight() {
            return false;
        }

        let ball_position = ctx.tick_context.positions.ball.position;
        let ball_velocity = ctx.tick_context.positions.ball.velocity;
        let player_position = ctx.player.position;
        let player_speed = ctx.player.skills.physical.pace / 20.0 * 10.0; // Convert to game units

        // If ball is moving, calculate interception
        if ball_velocity.magnitude() > 0.5 {
            // Calculate if player can reach ball before it goes too far
            let time_to_ball = (ball_position - player_position).magnitude() / player_speed;
            let ball_future_position = ball_position + ball_velocity * time_to_ball;
            let intercept_distance = (ball_future_position - player_position).magnitude();

            // Check if interception is feasible
            if intercept_distance <= TACKLE_DISTANCE_THRESHOLD * 2.0 {
                // Also check if any opponent is closer to the interception point
                let closest_opponent_distance = ctx
                    .players()
                    .opponents()
                    .all()
                    .map(|opp| (ball_future_position - opp.position).magnitude())
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(f32::MAX);

                return intercept_distance < closest_opponent_distance * 0.9; // Need to be clearly closer
            }
        } else {
            // Ball is stationary - simple distance check
            let ball_distance = (ball_position - player_position).magnitude();

            if ball_distance <= TACKLE_DISTANCE_THRESHOLD {
                // Check if any opponent is closer
                let closest_opponent_distance = ctx
                    .players()
                    .opponents()
                    .all()
                    .map(|opp| (ball_position - opp.position).magnitude())
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(f32::MAX);

                return ball_distance < closest_opponent_distance * 0.8;
            }
        }

        false
    }
}
