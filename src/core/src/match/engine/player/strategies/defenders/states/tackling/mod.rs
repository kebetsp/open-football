use crate::r#match::defenders::states::DefenderState;
use crate::r#match::defenders::states::common::{ActivityIntensity, DefenderCondition};
use crate::r#match::player::strategies::common::players::ops::defender_skill::DefenderSkillProfile;
use crate::r#match::player::strategies::common::players::ops::defensive::DefensiveOperationsImpl;
use crate::r#match::player::strategies::common::players::ops::tackle::{self, DuelAction, DuelSkills, TackleChoice};
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
    SteeringBehavior,
};
use nalgebra::Vector3;
#[cfg(feature = "match-logs")]
use std::sync::atomic::Ordering;

// 2026-09-29: the tackle/contain/concede-out comparison and the
// per-role attempt bodies were replaced by the shared unified tackle
// system (`common/players/ops/tackle.rs`): hold / standing challenge /
// slide chosen from the threat at the carrier's position, and every
// ball outcome decided by one physical contact model. This state now
// only owns the guards that make a challenge legal and the hand-off.
const PRESSING_DISTANCE: f32 = 80.0;
const RETURN_DISTANCE: f32 = 120.0;

#[derive(Default, Clone)]
pub struct DefenderTacklingState {}

impl DefenderTacklingState {
    fn duel_skills(ctx: &StateProcessingContext) -> DuelSkills {
        let p = DefenderSkillProfile::from_ctx(ctx);
        DuelSkills {
            tackle: p.tackle_profile,
            discipline: p.discipline,
            condition: p.def_condition_mult,
        }
    }
}

impl StateProcessingHandler for DefenderTacklingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        #[cfg(feature = "match-logs")]
        crate::tackle_stats::DEF_ENTRIES.fetch_add(1, Ordering::Relaxed);

        // If we have the ball or our team controls it, transition to running
        if ctx.player.has_ball(ctx) || ctx.team().is_control_ball() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Running,
            ));
        }

        // Don't try to claim a ball that is in protected flight.
        if ctx.ball().is_in_flight() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Returning,
            ));
        }

        // Law 13: an opponent inside the free-kick retreat distance may
        // not challenge at all.
        if ctx.ball().is_free_kick_encroaching() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Pressing,
            ));
        }

        // Law 12: a goalkeeper holding the ball (or a live opponent goal
        // kick) cannot be challenged.
        if ctx.ball().is_opponent_restart() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Pressing,
            ));
        }

        if let Some(opponent) = ctx.players().opponents().with_ball().next() {
            let distance_to_opponent = opponent.distance(ctx);

            if distance_to_opponent > PRESSING_DISTANCE {
                #[cfg(feature = "match-logs")]
                crate::tackle_stats::DEF_GATE_TOO_FAR.fetch_add(1, Ordering::Relaxed);
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Pressing,
                ));
            }

            if distance_to_opponent > DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE {
                #[cfg(feature = "match-logs")]
                crate::tackle_stats::DEF_GATE_NOT_IN_RANGE.fetch_add(1, Ordering::Relaxed);
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Pressing,
                ));
            }

            // Per-player tackle cooldown (a just-tackled player can't
            // lunge again for ~1 s).
            if !ctx.player.can_attempt_tackle() {
                #[cfg(feature = "match-logs")]
                crate::tackle_stats::DEF_GATE_COOLDOWN.fetch_add(1, Ordering::Relaxed);
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Pressing,
                ));
            }

            // Only the best-positioned teammate engages; the rest cover.
            if !ctx.team().is_best_player_to_chase_ball() {
                #[cfg(feature = "match-logs")]
                crate::tackle_stats::DEF_GATE_NOT_BEST.fetch_add(1, Ordering::Relaxed);
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Pressing,
                ));
            }

            // The unified decision: hold (stay in THIS state so standoff
            // time accumulates), challenge, or slide.
            return match tackle::duel(ctx, &opponent, Self::duel_skills(ctx)) {
                DuelAction::Hold => {
                    #[cfg(feature = "match-logs")]
                    crate::tackle_stats::DEF_GATE_CONTAIN_OR_CONCEDE.fetch_add(1, Ordering::Relaxed);
                    None
                }
                DuelAction::Act(events) => {
                    #[cfg(feature = "match-logs")]
                    crate::tackle_stats::DEF_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
                    let next = if events.is_empty() {
                        DefenderState::Pressing
                    } else {
                        DefenderState::Standing
                    };
                    let mut result = StateChangeResult::with_defender_state(next);
                    result.events.add_range(events);
                    result.start_tackle_cooldown = true;
                    Some(result)
                }
            };
        } else {
            // Ball is loose - check for interception
            if self.can_intercept_ball(ctx) && !ctx.ball().is_in_flight() {
                return Some(StateChangeResult::with_defender_state_and_event(
                    DefenderState::Running,
                    crate::r#match::events::Event::PlayerEvent(
                        crate::r#match::player::events::PlayerEvent::ClaimBall(ctx.player.id),
                    ),
                ));
            }

            // If ball is too far away and not coming toward us, return to position
            let ball_distance = ctx.ball().distance();
            if ball_distance > RETURN_DISTANCE && !ctx.ball().is_towards_player_with_angle(0.8) {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Returning,
                ));
            }

            // Fallback: if ball is loose and very close, try to claim it
            if !ctx.tick_context.ball.is_owned && ball_distance < 5.0 && !ctx.ball().is_in_flight()
            {
                return Some(StateChangeResult::with_defender_state_and_event(
                    DefenderState::Running,
                    crate::r#match::events::Event::PlayerEvent(
                        crate::r#match::player::events::PlayerEvent::ClaimBall(ctx.player.id),
                    ),
                ));
            }

            // If an opponent is right next to the player but doesn't have the ball, press
            if let Some(close_opponent) = ctx.players().opponents().nearby(15.0).next() {
                if close_opponent.distance(ctx) < 10.0 {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Pressing,
                    ));
                }
            }
        }

        if ctx.in_state_time > 30 {
            let ball_distance = ctx.ball().distance();
            if ball_distance > PRESSING_DISTANCE {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Returning,
                ));
            }
            // Stuck in tackling too long without engaging — drop back to standing
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Standing,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Holding: stand off goal-side and mirror the carrier instead of
        // charging in. Acting or chasing: the normal explosive pursuit.
        if let Some(opponent) = ctx.players().opponents().with_ball().next() {
            if opponent.distance(ctx) <= DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE
                && tackle::choose(ctx, &opponent, Self::duel_skills(ctx), ctx.in_state_time as f32)
                    == TackleChoice::Hold
            {
                return Some(tackle::jockey_velocity(ctx, &opponent));
            }
        }

        let target = self.calculate_intelligent_target(ctx);

        // Closing-speed boost driven by the unified defender profile.
        let def_profile = DefenderSkillProfile::from_ctx(ctx);
        let speed_boost = def_profile.tackle_speed_boost();

        Some(
            SteeringBehavior::Pursuit {
                target,
                target_velocity: Vector3::zeros(),
            }
            .calculate(ctx.player)
            .velocity
                * speed_boost
                + ctx.player().separation_velocity() * 0.15,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Tackling is explosive and very demanding physically
        DefenderCondition::new(ActivityIntensity::VeryHigh).process(ctx);
    }
}

impl DefenderTacklingState {
    fn calculate_intelligent_target(&self, ctx: &StateProcessingContext) -> Vector3<f32> {
        let ball_position = ctx.tick_context.positions.ball.position;
        let player_position = ctx.player.position;
        let own_goal_position = ctx.ball().direction_to_own_goal();

        // Check if ball is dangerously close to own goal
        let ball_distance_to_own_goal = (ball_position - own_goal_position).magnitude();
        let is_ball_near_own_goal =
            ball_distance_to_own_goal < ctx.context.field_size.width as f32 * 0.2;

        // Check if we're between the ball and our goal
        let player_distance_to_own_goal = (player_position - own_goal_position).magnitude();
        let is_player_closer_to_goal = player_distance_to_own_goal < ball_distance_to_own_goal;

        if is_ball_near_own_goal && !is_player_closer_to_goal {
            // If ball is near our goal and we're not between ball and goal,
            // position ourselves between the ball and the goal
            let ball_to_goal_direction = (own_goal_position - ball_position).normalize();
            let intercept_distance = 5.0; // Stand 5 units in front of the ball towards our goal
            ball_position + ball_to_goal_direction * intercept_distance
        } else {
            // Otherwise, pursue the ball directly
            ball_position
        }
    }

    fn exists_nearby(&self, ctx: &StateProcessingContext) -> bool {
        const DISTANCE: f32 = 30.0;

        ctx.players().opponents().exists(DISTANCE) || ctx.players().teammates().exists(DISTANCE)
    }

    fn can_intercept_ball(&self, ctx: &StateProcessingContext) -> bool {
        if self.exists_nearby(ctx) {
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

#[cfg(test)]
mod tests {
    use crate::PlayerSkills;
    use crate::club::player::builder::PlayerBuilder;
    use crate::r#match::MatchPlayer;
    use crate::r#match::player::strategies::players::ops::skill_composites as sc;
    use crate::shared::fullname::FullName;
    use crate::{
        PersonAttributes, PlayerAttributes, PlayerPosition, PlayerPositionType, PlayerPositions,
    };
    use chrono::NaiveDate;

    fn defender(tackling: f32, marking: f32, positioning: f32) -> MatchPlayer {
        let mut attrs = PlayerAttributes::default();
        attrs.condition = 9000;
        let mut skills = PlayerSkills::default();
        skills.technical.tackling = tackling;
        skills.technical.marking = marking;
        skills.mental.positioning = positioning;
        skills.mental.anticipation = 12.0;
        skills.mental.concentration = 12.0;
        skills.mental.bravery = 12.0;
        skills.physical.strength = 13.0;
        skills.physical.balance = 12.0;
        skills.physical.agility = 12.0;
        skills.physical.stamina = 14.0;
        skills.physical.natural_fitness = 14.0;
        let p = PlayerBuilder::new()
            .id(1)
            .full_name(FullName::new("D".into(), "Z".into()))
            .birth_date(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap())
            .country_id(1)
            .attributes(PersonAttributes::default())
            .skills(skills)
            .positions(PlayerPositions {
                positions: vec![PlayerPosition {
                    position: PlayerPositionType::DefenderCenter,
                    level: 18,
                }],
            })
            .player_attributes(attrs)
            .build()
            .unwrap();
        MatchPlayer::from_player(1, &p, PlayerPositionType::DefenderCenter, false)
    }

    fn attacker(dribbling: f32, technique: f32, agility: f32) -> MatchPlayer {
        let mut attrs = PlayerAttributes::default();
        attrs.condition = 9000;
        let mut skills = PlayerSkills::default();
        skills.technical.dribbling = dribbling;
        skills.technical.technique = technique;
        skills.mental.flair = 12.0;
        skills.mental.composure = 12.0;
        skills.mental.decisions = 12.0;
        skills.physical.agility = agility;
        skills.physical.acceleration = 14.0;
        skills.physical.balance = 12.0;
        skills.physical.strength = 11.0;
        skills.physical.stamina = 14.0;
        skills.physical.natural_fitness = 14.0;
        let p = PlayerBuilder::new()
            .id(2)
            .full_name(FullName::new("A".into(), "Z".into()))
            .birth_date(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap())
            .country_id(1)
            .attributes(PersonAttributes::default())
            .skills(skills)
            .positions(PlayerPositions {
                positions: vec![PlayerPosition {
                    position: PlayerPositionType::ForwardCenter,
                    level: 18,
                }],
            })
            .player_attributes(attrs)
            .build()
            .unwrap();
        MatchPlayer::from_player(2, &p, PlayerPositionType::ForwardCenter, false)
    }

    #[test]
    fn strong_tackler_dominates_weak_dribbler() {
        let strong = defender(18.0, 17.0, 16.0);
        let weak_attacker = attacker(7.0, 7.0, 9.0);
        let diff = sc::defensive_duel(&strong, 30) - sc::dribble_attack(&weak_attacker, 30);
        assert!(
            diff > 0.20,
            "expected strong defender advantage, got diff={diff}"
        );
    }

    #[test]
    fn weak_tackler_loses_to_strong_dribbler() {
        let weak = defender(7.0, 7.0, 8.0);
        let elite_attacker = attacker(18.0, 17.0, 17.0);
        let diff = sc::defensive_duel(&weak, 30) - sc::dribble_attack(&elite_attacker, 30);
        assert!(diff < -0.10, "expected attacker advantage, got diff={diff}");
    }

    #[test]
    fn marking_and_positioning_help_defender_in_duel() {
        let positional = defender(12.0, 18.0, 18.0);
        let raw_tackler = defender(15.0, 8.0, 8.0);
        // The positional defender should match or exceed a stronger
        // pure tackler thanks to marking + positioning weight (0.13 +
        // 0.17). This validates the spec's expectation that the duel
        // composite isn't dominated by raw tackling alone.
        assert!(sc::defensive_duel(&positional, 30) >= sc::defensive_duel(&raw_tackler, 30));
    }
}
