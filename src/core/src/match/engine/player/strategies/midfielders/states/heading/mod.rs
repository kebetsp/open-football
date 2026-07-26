use crate::r#match::events::Event;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::midfielders::states::common::{ActivityIntensity, MidfielderCondition};
use crate::r#match::player::events::{PlayerEvent, ShootingEventContext};
use crate::r#match::player::strategies::common::passing::resolve_aerial_duel;
use crate::r#match::player::strategies::players::ShotType;
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
    SteeringBehavior,
};
use nalgebra::Vector3;
use std::cmp::Ordering;

const HEADING_HEIGHT_THRESHOLD: f32 = 1.5;
const HEADING_DISTANCE_THRESHOLD: f32 = 4.0;

/// Realism-bug 2026-07-26 follow-up: midfielders had no heading state at
/// all — a midfielder who won an aerial duel (corner, cross, free-kick
/// cross, long ball) had no code path to strike the ball, so it fell
/// through to ordinary grounded possession logic instead ("controls the
/// ball instead of heading it" on a ball that structurally can't be
/// controlled). This mirrors `ForwardHeadingState` — same skills-driven
/// duel + contact model, same corner-contest carve-out.
#[derive(Default, Clone)]
pub struct MidfielderHeadingState {}

impl StateProcessingHandler for MidfielderHeadingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        let ball_position = ctx.tick_context.positions.ball.position;

        if ctx.ball().distance() > HEADING_DISTANCE_THRESHOLD {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        if ball_position.z < HEADING_HEIGHT_THRESHOLD {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // Discrete aerial-contest carve-out (corner or the general
        // cross/FK-cross/long-ball resolver): the contest already
        // decided this player won the jump — a clean-contact-only roll,
        // same formula as the forward/defender equivalents.
        if ctx.ball().is_team_attacking_corner() || ctx.player.aerial_contest_won > 0 {
            let heading = ctx.player.skills.technical.heading / 20.0;
            let jumping = ctx.player.skills.physical.jumping / 20.0;
            let p = (0.62 + (heading + jumping) * 0.5 * 0.30).clamp(0.55, 0.95);
            return if ctx.context.rng.unit_f32() < p {
                Some(StateChangeResult::with_midfielder_state_and_event(
                    MidfielderState::Running,
                    Event::PlayerEvent(PlayerEvent::Shoot(
                        ShootingEventContext::new()
                            .with_player_id(ctx.player.id)
                            .with_target(ctx.player().shooting_direction())
                            .with_reason("MID_HEADING_ON_GOAL")
                            .with_shot_type(ShotType::Header)
                            .build(ctx),
                    )),
                ))
            } else {
                Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Running,
                ))
            };
        }

        let attacker_full = ctx.context.players.by_id(ctx.player.id);
        let defender_full = ctx
            .players()
            .opponents()
            .all()
            .filter(|opp| {
                if let Some(full) = ctx.context.players.by_id(opp.id) {
                    !full.tactical_position.current_position.is_goalkeeper()
                } else {
                    true
                }
            })
            .min_by(|a, b| {
                let da = (a.position - ctx.player.position).magnitude();
                let db = (b.position - ctx.player.position).magnitude();
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })
            .and_then(|m| ctx.context.players.by_id(m.id));

        let minute = (ctx.context.total_match_time / 60_000) as u32;
        let won_duel = match attacker_full {
            Some(att) => resolve_aerial_duel(ctx, att, defender_full, minute),
            None => self.attempt_heading(ctx),
        };

        if !won_duel {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        if self.attempt_heading(ctx) {
            Some(StateChangeResult::with_midfielder_state_and_event(
                MidfielderState::Running,
                Event::PlayerEvent(PlayerEvent::Shoot(
                    ShootingEventContext::new()
                        .with_player_id(ctx.player.id)
                        .with_target(ctx.player().shooting_direction())
                        .with_reason("MID_HEADING_ON_GOAL")
                        .with_shot_type(ShotType::Header)
                        .build(ctx),
                )),
            ))
        } else {
            Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ))
        }
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let ball_position = ctx.tick_context.positions.ball.position;
        Some(
            SteeringBehavior::Arrive {
                target: ball_position,
                slowing_distance: 3.0,
            }
            .calculate(ctx.player)
            .velocity,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        MidfielderCondition::new(ActivityIntensity::VeryHigh).process(ctx);
    }
}

impl MidfielderHeadingState {
    fn attempt_heading(&self, ctx: &StateProcessingContext) -> bool {
        let heading_skill = ctx.player.skills.technical.heading / 20.0;
        let jumping_skill = ctx.player.skills.physical.jumping / 20.0;
        let overall_skill = (heading_skill + jumping_skill) / 2.0;

        let random_value: f32 = ctx.context.rng.unit_f32();
        random_value < overall_skill
    }
}
