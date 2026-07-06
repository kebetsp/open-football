use crate::club::player::traits::BehavioralDirective;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::forwarders::states::common::{ActivityIntensity, ForwardCondition};
use crate::r#match::player::strategies::common::players::ops::forward_shot_decision::{
    ShotDecision, evaluate_forward_shot_decision,
};
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
    SteeringBehavior,
};
use nalgebra::Vector3;

#[derive(Default, Clone)]
pub struct ForwardDribblingState {}

impl StateProcessingHandler for ForwardDribblingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if !ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        // Behavioural directive: lay_it_off_first_touch — a reception can
        // land straight in Dribbling; release immediately here too.
        if ctx.player.behavioral_directive == Some(BehavioralDirective::LayItOffFirstTouch)
            && ctx.tick_context.ball.ownership_duration < 30
        {
            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        // Behavioural directive: byline_and_cross. While wide, commit to
        // the touchline carry: cross on reaching the byline, bail out only
        // under a genuine two-man press. The generic exits below (no-
        // opponent bail, dribble timeout, single-chaser pass-out) would
        // otherwise cancel the run on almost every touchline duel.
        if ctx.player.behavioral_directive == Some(BehavioralDirective::BylineAndCross) {
            let field_h = ctx.context.field_size.height as f32;
            let y = ctx.player.position.y;
            if y < field_h * 0.26 || y > field_h * 0.74 {
                let goal_x = ctx.player().opponent_goal_position().x;
                if (goal_x - ctx.player.position.x).abs() < 40.0 {
                    return Some(StateChangeResult::with_forward_state(
                        ForwardState::Crossing,
                    ));
                }
                if ctx.players().opponents().nearby(8.0).count() >= 2 {
                    return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
                }
                return None; // keep carrying; velocity() steers to the byline
            }
        }

        // No opponents nearby — just run, dribbling is for beating defenders
        if !ctx.players().opponents().exists(25.0) {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        let distance_to_goal = ctx.ball().distance_to_opponent_goal();
        let can_shoot = ctx.team().can_shoot() && ctx.player().can_shoot();

        // PRIORITY 0: Near opponent goalkeeper.
        if let Some(gk) = ctx.players().opponents().goalkeeper().next() {
            let distance_to_gk = (ctx.player.position - gk.position).magnitude();
            if distance_to_gk < 25.0 && distance_to_goal < 120.0 && can_shoot {
                if let Some(result) = dispatch_shot(ctx, "FWD_DRIB_NEAR_GK") {
                    return Some(result);
                }
                // dispatch_shot returned None (Hold). In the penalty zone
                // (<12u) the keeper can physically grab or block — pass to
                // a teammate rather than keep running into them.
                if distance_to_gk < 12.0 {
                    return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
                }
            }
        }

        // PRIORITY 1: In shooting range with a clear lane.
        if can_shoot && ctx.player().shooting().in_shooting_range() && ctx.player().has_clear_shot()
        {
            if let Some(result) = dispatch_shot(ctx, "FWD_DRIB_CLEAR") {
                return Some(result);
            }
        }

        // PRIORITY 1b: Range-based fallback with lane check.
        if can_shoot && ctx.player().should_attempt_shot() && ctx.player().has_clear_shot() {
            if let Some(result) = dispatch_shot(ctx, "FWD_DRIB_RANGE") {
                return Some(result);
            }
        }

        // Prevent infinite dribbling - timeout after 40 ticks to reassess.
        if ctx.in_state_time > 40 {
            if can_shoot && distance_to_goal < 60.0 && ctx.player().has_clear_shot() {
                if let Some(result) = dispatch_shot(ctx, "FWD_DRIB_TIMEOUT") {
                    return Some(result);
                }
            }
            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        // Under REAL pressure from multiple defenders — pass.
        // The flicker bug: the old "no space to dribble" check below
        // (opponents within 15u) fired against Passing's "opponents
        // within 20u → back to Dribbling" rule, so a lone chaser at
        // 17u produced Dribbling → Passing → Dribbling every few
        // ticks. Now we require two real pressers OR a long commit
        // window before abandoning the dribble.
        let close_defenders = ctx.players().opponents().nearby(8.0).count();
        if close_defenders >= 2 {
            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        // Only abandon dribbling for a pass when genuinely boxed in —
        // opponent within 10u AND we've been dribbling long enough to
        // commit to the decision (≥15 ticks). The old `has_space_to_dribble`
        // (15u threshold) fired too eagerly against a single chaser.
        if ctx.in_state_time >= 15 && ctx.players().opponents().nearby(10.0).next().is_some() {
            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        // Cross from wide position in attacking third
        if self.should_cross(ctx) {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Crossing,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let goal = ctx.player().opponent_goal_position();

        // Behavioural directives: byline_and_cross / stay_wide_no_cut_inside
        // — steer along the touchline at the player's own channel instead
        // of angling in toward goal centre. Uses start_position.y as the
        // channel when the player's slot is a wide one; otherwise holds
        // the current wide lane. (StayWide shapes only the carry route;
        // BylineAndCross additionally bypasses the pass tree in process().)
        if matches!(
            ctx.player.behavioral_directive,
            Some(BehavioralDirective::BylineAndCross | BehavioralDirective::StayWideNoCutInside)
        ) {
            let field_h = ctx.context.field_size.height as f32;
            let y = ctx.player.position.y;
            if y < field_h * 0.26 || y > field_h * 0.74 {
                let start_y = ctx.player.start_position.y;
                let channel_y = if start_y < field_h * 0.30 || start_y > field_h * 0.70 {
                    start_y
                } else {
                    y
                };
                return Some(
                    SteeringBehavior::Arrive {
                        target: Vector3::new(goal.x, channel_y, 0.0),
                        slowing_distance: 30.0,
                    }
                    .calculate(ctx.player)
                    .velocity,
                );
            }
        }

        // GK-avoidance: when the keeper is in close range, sidestep to
        // create a shooting angle rather than running straight into them.
        // "Round the keeper" — the forward angles laterally away from
        // the pitch centre to open up a gap.
        if let Some(gk) = ctx.players().opponents().goalkeeper().next() {
            let gk_dist = (gk.position - ctx.player.position).magnitude();
            let goal_dist = ctx.ball().distance_to_opponent_goal();
            if gk_dist < 20.0 && goal_dist < 80.0 {
                let to_goal = (goal - ctx.player.position).normalize();
                let lateral = Vector3::new(-to_goal.y, to_goal.x, 0.0);
                let center_y = ctx.context.field_size.height as f32 / 2.0;
                let side = if ctx.player.position.y > center_y { -1.0 } else { 1.0 };
                let target = ctx.player.position + lateral * side * 18.0;
                let clamped = Vector3::new(
                    target.x.clamp(15.0, ctx.context.field_size.width as f32 - 15.0),
                    target.y.clamp(15.0, ctx.context.field_size.height as f32 - 15.0),
                    0.0,
                );
                return Some(
                    SteeringBehavior::Arrive {
                        target: clamped,
                        slowing_distance: 10.0,
                    }
                    .calculate(ctx.player)
                    .velocity,
                );
            }
        }

        Some(
            SteeringBehavior::Arrive {
                target: goal,
                slowing_distance: 150.0,
            }
            .calculate(ctx.player)
            .velocity,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Dribbling is high intensity - sustained movement with ball
        ForwardCondition::with_velocity(ActivityIntensity::High).process(ctx);
    }
}

impl ForwardDribblingState {
    fn should_cross(&self, ctx: &StateProcessingContext) -> bool {
        let field_height = ctx.context.field_size.height as f32;
        let y = ctx.player.position.y;
        let wide_margin = field_height * 0.2;

        let is_wide = y < wide_margin || y > field_height - wide_margin;
        if !is_wide {
            return false;
        }

        let distance_to_goal = ctx.ball().distance_to_opponent_goal();
        // In attacking area but not too close to goal
        if distance_to_goal > 300.0 || distance_to_goal < 60.0 {
            return false;
        }

        // Has teammates in the box
        let goal_pos = ctx.player().opponent_goal_position();
        let teammates_in_box = ctx.players().teammates().nearby_at(goal_pos, 120.0).count();

        let crossing = ctx.player.skills.technical.crossing / 20.0;
        teammates_in_box >= 1 && crossing > 0.4
    }
}

/// Funnel a candidate shot through the centralised gate stack. Returns
/// `Some(Shooting)` when the helper greenlights the strike, `Some(Passing)`
/// when a teammate has the better look, and `None` to let the caller fall
/// through to its next priority. Without this, the legacy direct
/// `with_shot_reason` paths bypassed xG / sprint-balance / 1v1 / pass-EV
/// gating because `ForwardShootingState` skips its own helper roll once a
/// `pending_shot_reason` is set.
fn dispatch_shot(ctx: &StateProcessingContext, tag: &'static str) -> Option<StateChangeResult> {
    match evaluate_forward_shot_decision(ctx, tag) {
        ShotDecision::Shoot { reason } => Some(
            StateChangeResult::with_forward_state(ForwardState::Shooting).with_shot_reason(reason),
        ),
        ShotDecision::Pass => Some(StateChangeResult::with_forward_state(ForwardState::Passing)),
        ShotDecision::Hold => None,
    }
}
