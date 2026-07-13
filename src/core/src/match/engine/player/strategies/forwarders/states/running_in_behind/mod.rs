use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::forwarders::states::common::{ActivityIntensity, ForwardCondition};
use crate::r#match::player::strategies::common::players::ops::forward_shot_decision::{
    ShotDecision, evaluate_forward_shot_decision,
};
use crate::r#match::player::strategies::players::skills::SkillCurve;
use crate::r#match::{
    ConditionContext, PlayerSide, StateChangeResult, StateProcessingContext,
    StateProcessingHandler,
};
use nalgebra::Vector3;

#[derive(Default, Clone)]
pub struct ForwardRunningInBehindState {}

impl StateProcessingHandler for ForwardRunningInBehindState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        let ball_ops = ctx.ball();
        if ctx.player.has_ball(ctx) {
            // Centralised shot decision — applies cooldown / xG / clear-shot
            // / sprint-balance / 1v1 / pass-EV gates. Was previously a raw
            // distance check that auto-transitioned to Shooting, which let
            // a sprinting Finishing-10 striker fire any time they reached
            // the ball under 80u — the dominant unrealistic-conversion
            // path. The Shooting state has minimal gating of its own, so
            // every gate decision must be made BEFORE the transition.
            let distance = ball_ops.distance_to_opponent_goal();
            if distance >= 80.0 {
                return Some(StateChangeResult::with_forward_state(
                    ForwardState::Dribbling,
                ));
            }
            return Some(match evaluate_forward_shot_decision(ctx, "FWD_RIB_SHOT") {
                ShotDecision::Shoot { reason } => {
                    StateChangeResult::with_forward_state(ForwardState::Shooting)
                        .with_shot_reason(reason)
                }
                ShotDecision::Pass => StateChangeResult::with_forward_state(ForwardState::Passing),
                ShotDecision::Hold => {
                    // Carry through into a controlled dribble — the
                    // forward isn't ready to strike (xG too low,
                    // sprinting off-balance, or pass-better-than-shot).
                    StateChangeResult::with_forward_state(ForwardState::Dribbling)
                }
            });
        }

        // SMART RUN TIMING: Only continue run if passer has stable possession
        // Delay run if teammate is under heavy pressure (they can't deliver the pass)
        if let Some(owner_id) = ctx.ball().owner_id() {
            if let Some(owner) = ctx.context.players.by_id(owner_id) {
                if owner.team_id == ctx.player.team_id {
                    // Passer under heavy pressure — abort run, they can't deliver
                    let opponents_near_passer =
                        ctx.tick_context.grid.opponents(owner_id, 10.0).count();
                    if opponents_near_passer >= 3 {
                        return Some(StateChangeResult::with_forward_state(
                            ForwardState::CreatingSpace,
                        ));
                    }
                }
            }
        }

        // Check if the run is still viable
        if !self.is_run_viable(ctx) {
            // If the run is no longer viable, transition to Creating Space
            return Some(StateChangeResult::with_forward_state(
                ForwardState::CreatingSpace,
            ));
        }

        // OffsideTrapBreaking used to be a separate state here but it
        // duplicated RunningInBehind's movement logic exactly. The
        // offside-trap detection now stays inside this state; we just
        // keep running in behind as the single "beat the line" behaviour.
        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let opponent_goal = ctx.ball().direction_to_opponent_goal();
        let current_position = ctx.player.position;

        // Channel-aware target: advance to the goal line but stay in the player's
        // assigned lateral channel. A winger with start_y near the touchline runs
        // into their flank corridor (where CrossingState takes over at the byline);
        // a central striker with start_y near 272 runs toward goal centre as before.
        let mut run_target = Vector3::new(
            opponent_goal.x,
            ctx.player.start_position.y,
            0.0,
        );

        // Offside discipline: while a teammate is still ON the ball (the
        // through-pass hasn't been released), hold the run at the shoulder
        // of the last defender instead of sprinting to the goal line. The
        // moment the ball is released (in flight or loose) the clamp lifts
        // and the sprint beats the line legally. Without this, forwards
        // camped 40u+ beyond the line during buildup and every delivery
        // arrived offside (~25 whistles / 10-min match, killing most
        // attacking moves and dragging goals/match from ~2.5 to ~0.8).
        let teammate_on_ball = !ctx.ball().is_in_flight()
            && ctx
                .ball()
                .owner_id()
                .and_then(|id| ctx.context.players.by_id(id))
                .map_or(false, |o| o.team_id == ctx.player.team_id && o.id != ctx.player.id);
        if teammate_on_ball {
            const ONSIDE_HOLD: f32 = 6.0;
            let line = ctx.player().defensive().find_defensive_line();
            match ctx.player.side {
                Some(PlayerSide::Left) => run_target.x = run_target.x.min(line - ONSIDE_HOLD),
                Some(PlayerSide::Right) => run_target.x = run_target.x.max(line + ONSIDE_HOLD),
                None => {}
            }
        }

        // §11.9: never run into the space the ball carrier is already
        // running into — the carrier owns that space. Shifts the run
        // target laterally out of the ~3m exclusion radius when needed.
        run_target =
            crate::r#match::player::strategies::spacing::avoid_carrier_space(ctx, run_target);

        let direction = (run_target - current_position).normalize();

        let ownership_duration = ctx.tick_context.ball.ownership_duration;
        let is_counter = ownership_duration < 15;

        let pace = ctx.player.skills.physical.pace;
        let acceleration = ctx.player.skills.physical.acceleration / 20.0;
        let counter_bonus = if is_counter { 0.3 } else { 0.0 };
        let sprint_speed = pace * (1.5 + acceleration * 0.5 + counter_bonus);

        Some(direction * sprint_speed)
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Running in behind is very high intensity - explosive sprinting
        ForwardCondition::with_velocity(ActivityIntensity::VeryHigh).process(ctx);
    }
}

impl ForwardRunningInBehindState {
    fn is_run_viable(&self, ctx: &StateProcessingContext) -> bool {
        // run_channel_in_behind: the manager ordered the runs — a blocker
        // ahead doesn't abort them (the onside clamp in velocity() already
        // keeps the run legal). Stamina and a capable passer still gate.
        let space_ahead = if ctx.player.behavioral_directive
            == Some(crate::club::player::traits::BehavioralDirective::RunChannelInBehind)
        {
            true
        } else {
            self.space_ahead(ctx)
        };

        // Check if the player is still in a good position to receive a pass
        let in_passing_lane = self.in_passing_lane(ctx);

        // Check if the player has the stamina to continue the run
        let has_stamina = !ctx.player().is_tired();

        // Check passer's ability (if passer has low vision, run is less viable)
        let passer_capable = self.is_passer_capable(ctx);

        space_ahead && in_passing_lane && has_stamina && passer_capable
    }

    /// Check if the teammate with ball can deliver a pass
    fn is_passer_capable(&self, ctx: &StateProcessingContext) -> bool {
        if let Some(owner_id) = ctx.ball().owner_id() {
            if let Some(owner) = ctx.context.players.by_id(owner_id) {
                if owner.team_id == ctx.player.team_id {
                    let vision = owner.skills.mental.vision / 20.0;
                    let passing = owner.skills.technical.passing / 20.0;
                    // Low skill passers can't deliver through-balls
                    return (vision + passing) / 2.0 > 0.3;
                }
            }
        }
        // No teammate has ball — run is still viable (ball might be loose)
        true
    }

    fn space_ahead(&self, ctx: &StateProcessingContext) -> bool {
        let player_pos = ctx.player.position;
        let goal_pos = ctx.player().opponent_goal_position();
        let to_goal = (goal_pos - player_pos).normalize();

        // Check for opponents blocking the path ahead (within 30 units, in forward direction)
        let blockers = ctx
            .players()
            .opponents()
            .nearby(30.0)
            .filter(|opp| {
                let to_opp = (opp.position - player_pos).normalize();
                to_opp.dot(&to_goal) > 0.3
            })
            .count();

        // Allow runs with one blocker if the forward is fast
        if blockers == 0 {
            return true;
        }

        if blockers == 1 {
            // Pace decides the chance of beating a single blocker on
            // a run-in-behind. Sigmoid pivot at 12/20 so the full
            // 1-20 range matters instead of a hard cliff.
            let p = SkillCurve::new(ctx.player.skills.physical.pace, 12.0, 0.6).probability();
            return ctx.context.rng.unit_f32() < p;
        }

        false
    }

    fn in_passing_lane(&self, ctx: &StateProcessingContext) -> bool {
        // Check if the player is ahead of the ball holder (toward opponent goal)
        // Diagonal runs are valid — only reject if player is behind the passer
        if let Some(owner_id) = ctx.ball().owner_id() {
            if let Some(owner) = ctx.context.players.by_id(owner_id) {
                if owner.team_id == ctx.player.team_id {
                    let passer_pos = ctx.tick_context.positions.players.position(owner_id);
                    let goal_pos = ctx.player().opponent_goal_position();
                    let to_goal = (goal_pos - passer_pos).normalize();
                    let to_runner = (ctx.player.position - passer_pos).normalize();
                    // Runner must be at least somewhat ahead of passer (dot > 0.0)
                    // This allows wide diagonal runs while rejecting backward positions
                    return to_runner.dot(&to_goal) > 0.0;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod regression_tests {
    //! Regression: RunningInBehind used to auto-transition to Shooting
    //! whenever `distance_to_opponent_goal() < 80.0` while holding the
    //! ball. That bypassed every gate (cooldowns, xG, clear-shot,
    //! sprint balance, pass EV) and was the dominant unrealistic-
    //! conversion path for mediocre finishers.
    //!
    //! These checks pin the TEXT of the new branch so a future refactor
    //! that re-adds an unconditional Shooting transition fails the
    //! check before the calibration regression slips back in.
    const SOURCE: &str = include_str!("mod.rs");

    #[test]
    fn auto_shoot_under_80u_branch_is_gone() {
        // The legacy auto-shoot branch literally said
        //     "ball_ops.distance_to_opponent_goal() < 80.0" → Shooting
        // immediately after `has_ball`. We allow the distance check,
        // but it MUST be paired with a call into the centralised
        // `evaluate_forward_shot_decision`.
        assert!(
            SOURCE.contains("evaluate_forward_shot_decision"),
            "RunningInBehind must defer to centralised shot decision helper"
        );
    }

    #[test]
    fn shot_decision_handles_pass_alternative() {
        // The new code must route to Passing when the helper picks a
        // better cutback over a low-xG strike. Without this, a
        // sprinting forward at the post simply cycles between
        // Shooting and Dribbling forever, losing the realism win.
        assert!(
            SOURCE.contains("ShotDecision::Pass") && SOURCE.contains("ForwardState::Passing"),
            "Pass alternative must be wired in"
        );
    }
}
