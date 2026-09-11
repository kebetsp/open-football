use crate::r#match::defenders::states::DefenderState;
use crate::r#match::defenders::states::common::{ActivityIntensity, DefenderCondition};
use crate::r#match::player::strategies::players::DefensiveRole;
use crate::r#match::{
    ConditionContext, MATCH_TIME_MS, StateChangeResult, StateProcessingContext,
    StateProcessingHandler, SteeringBehavior,
};
use nalgebra::Vector3;

#[derive(Default, Clone)]
pub struct DefenderReturningState {}

impl StateProcessingHandler for DefenderReturningState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Passing,
            ));
        }

        // Crisis override — abandon return journey and engage. Standing
        // will re-evaluate via the role block next tick.
        if ctx.player().defensive().is_defensive_crisis() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Standing,
            ));
        }

        // Take ball only if best positioned — prevents swarming
        if ctx.ball().should_take_ball_immediately() && ctx.team().is_best_player_to_chase_ball() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::TakeBall,
            ));
        }

        if ctx.player().distance_from_start_position() < 10.0 {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Standing,
            ));
        }

        if ctx.team().is_control_ball() {
            if ctx.player().distance_from_start_position() < 5.0 {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Standing,
                ));
            }
        } else {
            if ctx.ball().distance() < 100.0 {
                if let Some(opponent) = ctx.players().opponents().with_ball().next() {
                    // docs/positional-play-master-design.md Stage A: this
                    // used to break EVERY returning defender within 100u
                    // off their walk back to shape, regardless of whether
                    // they were actually the best-placed responder — real
                    // football has one defender close down a carrier
                    // while the rest keep recovering into their own
                    // positions. Gated on the same unified Primary role
                    // every other defensive state already uses, so only
                    // the genuinely closest defender abandons the return
                    // journey; everyone else keeps walking back.
                    let is_primary = matches!(
                        ctx.player().defensive().defensive_role_for_ball_carrier_unified(),
                        DefensiveRole::Primary
                    );
                    if is_primary {
                        // 2026-08-30 (engine-audit Module 6): was an
                        // unconditional Tackling transition off a 100u
                        // ball-distance proxy — the worst mismatch found in
                        // this whole sweep (90u looser than Tackling's real
                        // 10u gate), never checked cooldown, and didn't even
                        // check the actual carrier's distance. Now routes
                        // through the shared gate; falls back to Pressing
                        // (closing the gap) when not genuinely in range yet,
                        // same convention as every other fixed state.
                        if ctx
                            .player()
                            .defensive()
                            .can_engage_tackle(opponent.distance(ctx))
                        {
                            return Some(StateChangeResult::with_defender_state(
                                DefenderState::Tackling,
                            ));
                        }
                        return Some(StateChangeResult::with_defender_state(
                            DefenderState::Pressing,
                        ));
                    }
                } else if ctx.team().is_best_player_to_chase_ball() {
                    // Loose ball, not an opponent's controlled carry —
                    // only the genuinely best-placed teammate breaks off
                    // to collect it, same discipline as every other
                    // "loose ball nearby" check elsewhere in this file.
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::TakeBall,
                    ));
                }
            }

            if ctx.ball().is_towards_player_with_angle(0.8) && ctx.ball().distance() < 200.0 {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Intercepting,
                ));
            }

            if ctx.team().is_loosing()
                && ctx.context.total_match_time > (MATCH_TIME_MS - 180)
                && ctx.ball().distance() < 30.0
            {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Pressing,
                ));
            }
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
        // Returning to position involves jogging back - moderate intensity
        DefenderCondition::with_velocity(ActivityIntensity::Moderate).process(ctx);
    }
}
