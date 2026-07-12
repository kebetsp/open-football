use crate::club::player::traits::BehavioralDirective;
use crate::r#match::events::Event;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::midfielders::states::common::{ActivityIntensity, MidfielderCondition};
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
pub struct MidfielderCrossingState {}

impl StateProcessingHandler for MidfielderCrossingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if !ctx.player.has_ball(ctx) {
            // Lost possession, transition to Running
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // CORNER SET-UP HOLD: wait until ≥2 runners have reached their zones
        // (or the setup window expires) so the delivery has a live target.
        if ctx.ball().is_team_attacking_corner()
            && !corner_box_loaded(ctx)
            && ctx.in_state_time < CORNER_SETUP_MAX
        {
            return None;
        }

        // BYLINE-DIRECTIVE HOLD: a byline_and_cross winger often beats
        // their own runners to the byline. Shield the ball briefly until
        // a teammate arrives within crossing range, instead of falling
        // through to a target-less delivery that degrades into a plain
        // pass-out (which erases the instructed pattern).
        if ctx.player.behavioral_directive == Some(BehavioralDirective::BylineAndCross)
            && !ctx.ball().is_team_attacking_corner()
            && ctx.in_state_time < 60
        {
            let goal_pos = ctx.player().opponent_goal_position();
            let target_ready = ctx
                .players()
                .teammates()
                .all()
                .any(|t| t.id != ctx.player.id && (t.position - goal_pos).magnitude() < 150.0);
            if !target_ready {
                return None;
            }
        }

        // After windup time, deliver the cross
        if ctx.in_state_time > CROSS_EXECUTION_TIME {
            if ctx.ball().is_team_attacking_corner() {
                // Zone-based corner delivery: ball goes to the zone centre.
                let zone_rotation = ((ctx.context.total_match_time / 2000) % 3) as u8;
                if let Some((receiver_id, zone_target, zone_kind)) =
                    pick_corner_delivery(ctx, zone_rotation)
                {
                    #[cfg(feature = "match-logs")]
                    {
                        use std::sync::atomic::Ordering;
                        use crate::r#match::player::strategies::common::players::ops::forward_shot_decision::mid_run_diag;
                        mid_run_diag::CORNER_CROSS_SENT.fetch_add(1, Ordering::Relaxed);
                    }
                    let mut result = StateChangeResult::with_midfielder_state_and_event(
                        MidfielderState::Running,
                        Event::PlayerEvent(PlayerEvent::PassTo(
                            PassingEventContext::new()
                                .with_from_player_id(ctx.player.id)
                                .with_to_player_id(receiver_id)
                                .with_pass_target(zone_target)
                                .with_reason("MID_CORNER")
                                .build(ctx),
                        )),
                    );
                    result.events.add_player_event(PlayerEvent::CornerDelivery(
                        ctx.player.id,
                        zone_kind,
                    ));
                    return Some(result);
                }
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Passing,
                ));
            }

            // Open-play cross: use the original target-selection logic.
            if let Some(target) = self.find_cross_target(ctx) {
                return Some(StateChangeResult::with_midfielder_state_and_event(
                    MidfielderState::Running,
                    Event::PlayerEvent(PlayerEvent::PassTo(
                        PassingEventContext::new()
                            .with_from_player_id(ctx.player.id)
                            .with_to_player_id(target.id)
                            .with_reason("MID_CROSS")
                            .build(ctx),
                    )),
                ));
            }

            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Passing,
            ));
        }

        None
    }

    fn velocity(&self, _ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Stationary while preparing the cross
        Some(Vector3::new(0.0, 0.0, 0.0))
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Crossing is very high intensity - explosive action
        MidfielderCondition::new(ActivityIntensity::VeryHigh).process(ctx);
    }
}

impl MidfielderCrossingState {
    /// Find the best teammate in or near the penalty area to cross to.
    fn find_cross_target<'a>(&self, ctx: &StateProcessingContext<'a>) -> Option<MatchPlayerLite> {
        let goal_pos = ctx.player().opponent_goal_position();

        let mut best_target: Option<(MatchPlayerLite, f32)> = None;

        for teammate in ctx.players().teammates().all() {
            // Skip self
            if teammate.id == ctx.player.id {
                continue;
            }

            let dist_to_goal = (teammate.position - goal_pos).magnitude();

            // Must be within 150 units of opponent goal (in/near the box)
            if dist_to_goal > 150.0 {
                continue;
            }

            // Must have a clear passing lane — EXCEPT on a corner or a
            // byline-directive delivery, where the ball is lofted over the
            // packed defenders, so a blocked ground lane doesn't
            // disqualify a central target.
            let lofted_delivery = ctx.ball().is_team_attacking_corner()
                || ctx.player.behavioral_directive == Some(BehavioralDirective::BylineAndCross);
            if !lofted_delivery && !ctx.player().has_clear_pass(teammate.id) {
                continue;
            }

            // Check how many opponents are near the cross path (interception risk)
            let pass_vector = teammate.position - ctx.player.position;
            let pass_distance = pass_vector.magnitude();
            let pass_direction = pass_vector.normalize();

            let opponents_in_path = ctx
                .players()
                .opponents()
                .all()
                .filter(|opponent| {
                    let to_opponent = opponent.position - ctx.player.position;
                    let projection = to_opponent.dot(&pass_direction);
                    if projection <= 0.0 || projection >= pass_distance {
                        return false;
                    }
                    let projected_point = ctx.player.position + pass_direction * projection;
                    let perp_distance = (opponent.position - projected_point).magnitude();
                    perp_distance < 6.0
                })
                .count();

            // Skip crosses with 2+ opponents directly in the path.
            // Exception: on a corner or byline-directive delivery the ball is
            // lofted over them — the aerial ball rises above the 2.5u
            // interception gate in interactions.rs, so ground-lane crowding
            // is irrelevant for those deliveries.
            if opponents_in_path >= 2 && !lofted_delivery {
                continue;
            }

            // Score: prefer players with good heading skill and proximity to goal center
            let heading_skill = if let Some(player) = ctx.context.players.by_id(teammate.id) {
                player.skills.technical.heading
            } else {
                10.0
            };

            // Penalize targets with tight marking
            let close_opponents = ctx.tick_context.grid.opponents(teammate.id, 8.0).count();
            let marking_penalty = match close_opponents {
                0 => 1.0,
                1 => 0.6,
                _ => 0.25,
            };

            // Reduce score if 1 opponent in cross path
            let path_penalty = if opponents_in_path == 1 { 0.6 } else { 1.0 };

            // On a corner, prefer the pushed-up centre-back — the designated
            // aerial target ("find the big man"). Only biases corner
            // deliveries; in open play CBs aren't in the box so it's inert.
            let corner_cb_bonus = if ctx.ball().is_team_attacking_corner()
                && teammate.tactical_positions.is_central_defender()
            {
                12.0
            } else {
                0.0
            };

            let score = (heading_skill + corner_cb_bonus + (150.0 - dist_to_goal) / 10.0)
                * marking_penalty
                * path_penalty;

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
