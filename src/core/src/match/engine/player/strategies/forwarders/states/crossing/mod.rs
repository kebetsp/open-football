use crate::r#match::events::Event;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::forwarders::states::common::{ActivityIntensity, ForwardCondition};
use crate::r#match::player::events::{PassingEventContext, PlayerEvent};
use crate::r#match::player::strategies::common::set_pieces::{corner_box_loaded, pick_corner_delivery};
use crate::r#match::{
    ConditionContext, MatchPlayerLite, StateChangeResult, StateProcessingContext,
    StateProcessingHandler,
};
use nalgebra::Vector3;

const CROSS_EXECUTION_TIME: u64 = 5;
const CORNER_SETUP_MAX: u64 = 200;

#[derive(Default, Clone)]
pub struct ForwardCrossingState {}

impl StateProcessingHandler for ForwardCrossingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        // Lost possession - transition out
        if !ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        // Not in a wide position - should pass instead
        if !self.is_in_wide_position(ctx) {
            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        // CORNER SET-UP HOLD: wait until ≥2 runners have reached their zones
        // (or the setup window expires) so the delivery has a live target.
        if ctx.ball().is_team_attacking_corner()
            && !corner_box_loaded(ctx)
            && ctx.in_state_time < CORNER_SETUP_MAX
        {
            return None;
        }

        // After windup time, deliver the cross
        if ctx.in_state_time > CROSS_EXECUTION_TIME {
            if ctx.ball().is_team_attacking_corner() {
                // Zone-based corner delivery: ball goes to the zone centre so it
                // arrives at the intended area regardless of where the runner is
                // at the moment of the kick.
                let zone_rotation = ((ctx.context.total_match_time / 2000) % 3) as u8;
                if let Some((receiver_id, zone_target)) = pick_corner_delivery(ctx, zone_rotation) {
                    #[cfg(feature = "match-logs")]
                    {
                        use std::sync::atomic::Ordering;
                        use crate::r#match::player::strategies::common::players::ops::forward_shot_decision::mid_run_diag;
                        mid_run_diag::CORNER_CROSS_SENT.fetch_add(1, Ordering::Relaxed);
                    }
                    return Some(StateChangeResult::with_forward_state_and_event(
                        ForwardState::Running,
                        Event::PlayerEvent(PlayerEvent::PassTo(
                            PassingEventContext::new()
                                .with_from_player_id(ctx.player.id)
                                .with_to_player_id(receiver_id)
                                .with_pass_target(zone_target)
                                .with_reason("FWD_CORNER")
                                .build(ctx),
                        )),
                    ));
                }
                return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
            }

            // Open-play cross: use the original target-selection logic.
            if let Some(target) = self.find_cross_target(ctx) {
                return Some(StateChangeResult::with_forward_state_and_event(
                    ForwardState::Running,
                    Event::PlayerEvent(PlayerEvent::PassTo(
                        PassingEventContext::new()
                            .with_from_player_id(ctx.player.id)
                            .with_to_player_id(target.id)
                            .with_reason("FWD_CROSS")
                            .build(ctx),
                    )),
                ));
            }

            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        None
    }

    fn velocity(&self, _ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Stationary while preparing the cross
        Some(Vector3::new(0.0, 0.0, 0.0))
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        ForwardCondition::new(ActivityIntensity::VeryHigh).process(ctx);
    }
}

impl ForwardCrossingState {
    fn is_in_wide_position(&self, ctx: &StateProcessingContext) -> bool {
        let field_height = ctx.context.field_size.height as f32;
        let y = ctx.player.position.y;
        let wide_margin = field_height * 0.2;

        y < wide_margin || y > field_height - wide_margin
    }

    /// Find the best teammate in or near the penalty area to cross to.
    ///
    /// For corners, delivery zone rotates each kick: 0 = central/penalty-spot,
    /// 1 = near post, 2 = far post. For open-play crosses the zone bonus is off
    /// and the best in-box target wins as before.
    fn find_cross_target<'a>(&self, ctx: &StateProcessingContext<'a>) -> Option<MatchPlayerLite> {
        let goal_pos = ctx.player().opponent_goal_position();
        let field_height = ctx.context.field_size.height as f32;
        let is_corner = ctx.ball().is_team_attacking_corner();

        // Rotate delivery zone using match time so consecutive corners vary:
        //   0 = central / penalty-spot area
        //   1 = near post (same lateral side as the corner taker)
        //   2 = far post (opposite side)
        let corner_zone: u8 = if is_corner {
            ((ctx.context.total_match_time / 2000) % 3) as u8
        } else {
            255
        };
        let taker_near_top = ctx.player.position.y < field_height / 2.0;

        let mut best_target: Option<(MatchPlayerLite, f32)> = None;

        for teammate in ctx.players().teammates().all() {
            if teammate.id == ctx.player.id {
                continue;
            }

            let dist_to_goal = (teammate.position - goal_pos).magnitude();

            // Widened 150 → 165 so players arriving at the penalty-area edge
            // are not excluded.
            if dist_to_goal > 165.0 {
                continue;
            }

            // Ground-lane check only for open-play — corners are lofted.
            if !is_corner && !ctx.player().has_clear_pass(teammate.id) {
                continue;
            }

            let heading_skill = ctx.context.players.by_id(teammate.id)
                .map(|p| p.skills.technical.heading)
                .unwrap_or(10.0);

            let corner_cb_bonus = if is_corner
                && teammate.tactical_positions.is_central_defender()
            {
                8.0
            } else {
                0.0
            };

            // Soft zone preference — does not hard-exclude other targets, so
            // if nobody is in the preferred zone the best available player wins.
            let zone_bonus: f32 = if is_corner {
                let lateral = (teammate.position.y - goal_pos.y).abs();
                let same_side = (teammate.position.y < field_height / 2.0) == taker_near_top;
                match corner_zone {
                    0 => {
                        // Central: 60–140u from goal, within 70u lateral of goal centre
                        if dist_to_goal >= 60.0 && dist_to_goal <= 140.0 && lateral < 70.0 {
                            25.0
                        } else {
                            0.0
                        }
                    }
                    1 => {
                        // Near post: same side as taker, ≤ 65u from goal
                        if dist_to_goal <= 65.0 && same_side { 25.0 } else { 0.0 }
                    }
                    2 => {
                        // Far post: opposite side, ≤ 80u from goal
                        if dist_to_goal <= 80.0 && !same_side { 25.0 } else { 0.0 }
                    }
                    _ => 0.0,
                }
            } else {
                0.0
            };

            let score = heading_skill + corner_cb_bonus + zone_bonus
                + (165.0 - dist_to_goal) / 10.0;

            if let Some((_, best_score)) = &best_target {
                if score > *best_score {
                    best_target = Some((teammate, score));
                }
            } else {
                best_target = Some((teammate, score));
            }
        }

        best_target.map(|(t, _)| t)
    }
}
