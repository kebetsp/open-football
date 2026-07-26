use crate::club::player::traits::BehavioralDirective;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::forwarders::states::common::{ActivityIntensity, ForwardCondition};
use crate::r#match::player::strategies::common::players::ops::forward_shot_decision::{
    ShotDecision, evaluate_forward_shot_decision,
};
use crate::r#match::player::strategies::common::players::ops::on_ball_value;
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

        // Conditional (Level 2): lay off ONLY long-ball receptions. Wider
        // 60-tick window: a long delivery takes longer to control and the
        // receiving state routes through here late.
        if ctx.player.behavioral_directive == Some(BehavioralDirective::LayOffOnLongBall)
            && ctx.tick_context.ball.ownership_duration < 60
            && ctx.ball().is_long_reception(140.0)
        {
            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        // Behavioural directive: byline_and_cross. While wide, commit to
        // the touchline carry: cross on reaching the byline, bail out only
        // under a genuine two-man press. The generic exits below (no-
        // opponent bail, dribble timeout, single-chaser pass-out) would
        // otherwise cancel the run on almost every touchline duel.
        if ctx.player.behavioral_directive == Some(BehavioralDirective::BylineAndCross) {
            // Channel-gated (start_position.y), not current-y — see the
            // Running-state twin for why.
            let field_h = ctx.context.field_size.height as f32;
            let start_y = ctx.player.start_position.y;
            if start_y < field_h * 0.30 || start_y > field_h * 0.70 {
                let goal_x = ctx.player().opponent_goal_position().x;
                let y = ctx.player.position.y;
                let currently_wide = y < field_h * 0.30 || y > field_h * 0.70;
                if currently_wide && (goal_x - ctx.player.position.x).abs() < 40.0 {
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

        // realism-bug (2026-07-25): opportunistic byline-run commitment
        // (`/goal`: >=3 genuine byline-style runs/match — a run toward
        // the byline ending in a cross/pass from past the penalty spot,
        // not necessarily reaching the actual line). Superseded a first
        // attempt that recomputed "is this the best candidate right now"
        // every tick via `carry_candidates` — that was too fragile to
        // sustain a real run (a small geometry shift flips the winner)
        // and barely moved the measured external rate. Now driven by
        // `byline_commitment_ticks`, a PERSISTENT per-player field armed
        // probabilistically on fresh wide possession (`run.rs`) — the
        // same "temporary override that survives across ticks/states"
        // pattern already proven for kickoff/throw-in/free-kick takers.
        let committing_to_byline = ctx.player.byline_commitment_ticks > 0;

        // Release once past the penalty spot (88u = 11m*8u/m; 100u used
        // for a small margin) while still wide-ish — matches the /goal
        // spec exactly ("a cross attempt or a pass somewhere from a
        // point after the penalty spot," not literal byline arrival).
        // Checked BEFORE the generic exits below so a committed run
        // releases cleanly on arrival instead of running past its own
        // target and hitting the dribble timeout instead.
        if committing_to_byline {
            let field_h = ctx.context.field_size.height as f32;
            let y = ctx.player.position.y;
            let currently_wide = y < field_h * 0.38 || y > field_h * 0.62;
            // realism-bug (2026-07-26): `distance_to_goal` is Euclidean
            // distance to the goal CENTRE, not depth to the goal LINE —
            // for a genuinely wide player that distance stays large from
            // the y-offset alone (e.g. y=50 vs a ~272 goal-centre y is
            // already 222u away, before x is even considered), so this
            // check could never fire once the velocity() steering fix
            // (above/below) actually kept him wide instead of drifting
            // central. Fixed to real depth — x-distance to the goal
            // line — the same measure `run.rs`'s arm logic already uses.
            let goal_x = ctx.player().opponent_goal_position().x;
            let depth = (goal_x - ctx.player.position.x).abs();
            if currently_wide && depth <= 100.0 {
                return Some(StateChangeResult::with_forward_state(
                    ForwardState::Crossing,
                ));
            }
        }

        // Prevent infinite dribbling - timeout after 40 ticks to reassess.
        // SKIPPED entirely while committing to a byline run — realistic
        // dribble speed is ~0.4-0.6u/tick (CLAUDE.md), so even the first
        // attempt's 100-tick extension only covered ~40-60u, nowhere near
        // the 150-420u a genuine run needs to reach the release point.
        // The outer `byline_commitment_ticks` budget (`run.rs`, sized for
        // the full trigger distance at realistic speed) is the real
        // bound here — this per-entry state timeout would just cut the
        // run short well before that budget, or before the explicit
        // release check above ever gets a chance to fire.
        if !committing_to_byline && ctx.in_state_time > 40 {
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
        // window before abandoning the dribble. Always active, even
        // when committing to a byline run — a genuine two-man press is
        // exactly the condition the BylineAndCross directive itself
        // bails out under too.
        let close_defenders = ctx.players().opponents().nearby(8.0).count();
        if close_defenders >= 2 {
            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        // Only abandon dribbling for a pass when genuinely boxed in —
        // opponent within 10u AND we've been dribbling long enough to
        // commit to the decision (≥15 ticks). The old `has_space_to_dribble`
        // (15u threshold) fired too eagerly against a single chaser.
        // Skipped when undirected byline commitment is active — a single
        // marker tracking a winger's touchline run is normal defending,
        // not being boxed in; real wingers keep going against exactly
        // this until a second defender arrives (see close_defenders
        // above), matching the existing directive's own rule.
        if !committing_to_byline
            && ctx.in_state_time >= 15
            && ctx.players().opponents().nearby(10.0).next().is_some()
        {
            return Some(StateChangeResult::with_forward_state(ForwardState::Passing));
        }

        // Cross from wide position in attacking third. Skipped when
        // genuinely committing to a byline run — the explicit release
        // check above already owns that decision for the committed
        // case; `should_cross` allows a cross from up to 300u out (most
        // of the attacking half), which would otherwise second-guess
        // and cut short a run before it reaches the intended depth.
        if !committing_to_byline && self.should_cross(ctx) {
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
        //
        // realism-bug (2026-07-26): the undirected `byline_commitment_
        // ticks` mechanism (`process()`, above) stopped the state machine
        // from INTERRUPTING a committed run, but this function — the
        // thing that actually decides which way the player moves — never
        // checked it, only the explicit directive. So a "committed"
        // player's decision-making correctly waited for the byline, but
        // his real movement was still picked fresh every tick by
        // `carry_candidates` below, which can legitimately favour cutting
        // inside. Measured directly (a temporary outcome-tagged
        // diagnostic across 30 matches): of runs that reached a genuine
        // decision, only 39.6% ended in a cross, 35.7% in a shot, and
        // 24.7% just expired (the run having drifted off the byline
        // entirely) — exactly the "I see shots and cuts inside" pattern
        // reported. Fixed by folding `byline_commitment_ticks > 0` into
        // the same channel-steering branch the directive already uses —
        // current position as the channel (there's no fixed start-
        // position channel concept for undirected play, unlike the
        // directive's own wide-slot assumption).
        if matches!(
            ctx.player.behavioral_directive,
            Some(BehavioralDirective::BylineAndCross | BehavioralDirective::StayWideNoCutInside)
        ) || ctx.player.byline_commitment_ticks > 0
        {
            let field_h = ctx.context.field_size.height as f32;
            let start_y = ctx.player.start_position.y;
            let y = ctx.player.position.y;
            let channel_wide = start_y < field_h * 0.30 || start_y > field_h * 0.70;
            let currently_wide = y < field_h * 0.30 || y > field_h * 0.70;
            if channel_wide || currently_wide {
                let channel_y = if channel_wide { start_y } else { y };
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

        // Option B / B1: carry-target selection via the shared on-ball
        // value function instead of a fixed goal-centre target. This is
        // the fix for the reported 2v1 (the carrier no longer runs
        // straight at the keeper) — see docs/on-ball-decision-logic-spec-
        // optionB.md. `goal` above is still used by the directive/GK
        // branches; only the generic fallback target changes.
        let (carry_target, _value) = on_ball_value::carry_candidates(ctx);
        let dist = (carry_target - ctx.player.position).magnitude();
        if dist < 6.0 {
            // Best candidate is roughly where we already are — the
            // spec's "hold falls out naturally" case. Still drift gently
            // toward goal so the player doesn't freeze mid-pitch.
            return Some(
                SteeringBehavior::Arrive {
                    target: goal,
                    slowing_distance: 150.0,
                }
                .calculate(ctx.player)
                .velocity,
            );
        }
        Some(
            SteeringBehavior::Arrive {
                target: carry_target,
                slowing_distance: 20.0,
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
