use crate::r#match::goalkeepers::states::common::{ActivityIntensity, GoalkeeperCondition};
use crate::r#match::goalkeepers::states::state::GoalkeeperState;
use crate::r#match::{
    ConditionContext, MatchPlayerLite, PassOriginRestart, PlayerDistanceFromStartPosition,
    PlayerSide, StateChangeResult, StateProcessingContext, StateProcessingHandler,
    SteeringBehavior, VectorExtensions,
};
use nalgebra::Vector3;

const DANGER_ZONE_RADIUS: f32 = 35.0;
const CLOSE_DANGER_DISTANCE: f32 = 100.0;
const FAR_THREAT_DISTANCE: f32 = 300.0;

#[derive(Default, Clone)]
pub struct GoalkeeperStandingState {}

impl StateProcessingHandler for GoalkeeperStandingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        // Shot in flight at our goal — react immediately. The
        // `is_towards_player_with_angle` check below would miss shots
        // aimed at the corners (angle between ball velocity and
        // ball→keeper can be ~30°, cosine < 0.6). The cached target is
        // set precisely so the keeper has a deterministic intercept
        // line; honour it.
        if let Some(target) = &ctx.tick_context.ball.cached_shot_target {
            if Some(target.defending_side) == ctx.player.side {
                return Some(StateChangeResult::with_goalkeeper_state(
                    GoalkeeperState::PreparingForSave,
                ));
            }
        }

        // Direct catch for close slow balls
        let ball_distance = ctx.ball().distance();
        if ball_distance < 10.0
            && !ctx.ball().is_owned()
            && ctx.ball().on_own_side()
            && ctx.tick_context.positions.ball.velocity.norm() < 10.0
        {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Catching,
            ));
        }

        // If goalkeeper has the ball, distribute it (never run with ball)
        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Distributing,
            ));
        }

        let ball_on_own_side = ctx.ball().on_own_side();

        // Skill-based threat assessment
        let anticipation = ctx.player.skills.mental.anticipation / 20.0;
        let command_of_area = ctx.player.skills.goalkeeping.command_of_area / 20.0;

        // Check for immediate threats requiring urgent action
        if ctx.ball().is_opponent_free_kick_setup() {
            return None; // hold the free-kick stance until the kick is struck
        }

        if let Some(opponent) = ctx.players().opponents().with_ball().next() {
            let opponent_distance = opponent.distance(ctx);

            // Opponent very close with ball - prepare for save or come out
            if opponent_distance < CLOSE_DANGER_DISTANCE {
                // Check if should come out or prepare for shot
                if self.should_rush_out_for_ball(ctx, &opponent) {
                    return Some(StateChangeResult::with_goalkeeper_state(
                        GoalkeeperState::ComingOut,
                    ));
                } else {
                    return Some(StateChangeResult::with_goalkeeper_state(
                        GoalkeeperState::PreparingForSave,
                    ));
                }
            }

            // Opponent approaching at medium range — the dedicated
            // Attentive state added no behaviour over Standing (both
            // repositioned and rechecked threats every tick), so we stay
            // put and let the next-tick threat scan drive the response.
        }

        // Check if ball is coming toward goal — react faster to shots
        // Shot velocities are ~1.0-2.0/tick, thresholds must match
        if ctx.ball().is_towards_player_with_angle(0.6) && ball_on_own_side {
            let ball_speed = ctx.tick_context.positions.ball.velocity.norm();

            if ball_speed > 0.5 {
                // Ball moving toward goal — prepare immediately
                if ball_distance < FAR_THREAT_DISTANCE * (1.0 + anticipation * 0.5) {
                    return Some(StateChangeResult::with_goalkeeper_state(
                        GoalkeeperState::PreparingForSave,
                    ));
                }
            }

            // Ball coming slowly — stay in Standing; PreparingForSave
            // will fire above once the ball is close enough and fast
            // enough to matter.
        }

        // Check for loose ball in dangerous area
        if !ctx.ball().is_owned()
            && ball_on_own_side
            && ball_distance < 40.0 * (1.0 + command_of_area * 0.3)
        {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::ComingOut,
            ));
        }

        // Ball on own side but no immediate threat — Standing is the
        // right rest state; Attentive was an identical idle state with
        // slightly different thresholds.

        // Check positioning
        match ctx.player().position_to_distance() {
            PlayerDistanceFromStartPosition::Small => {
                // Good positioning + opponent in danger zone → prepare for save.
                // UnderPressure was a pass-through to Catching/Distributing.
                if self.is_opponent_in_danger_zone(ctx) {
                    return Some(StateChangeResult::with_goalkeeper_state(
                        GoalkeeperState::PreparingForSave,
                    ));
                }
            }
            PlayerDistanceFromStartPosition::Medium => {
                // Need to adjust position - walk to better spot
                return Some(StateChangeResult::with_goalkeeper_state(
                    GoalkeeperState::Walking,
                ));
            }
            PlayerDistanceFromStartPosition::Big => {
                // Far from position - walk back
                return Some(StateChangeResult::with_goalkeeper_state(
                    GoalkeeperState::Walking,
                ));
            }
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Calculate optimal position based on ball and goal
        let optimal_position = self.calculate_optimal_position(ctx);
        let distance_to_optimal = ctx.player.position.distance_to(&optimal_position);

        // realism-bug (2026-09-13, "still bursts, not smooth" follow-up):
        // this used to be three hard distance-threshold tiers (5.0/15.0
        // units), each re-scaling the already-max_speed-bounded Arrive
        // output by a DIFFERENT flat multiplier (0.7/1.0/1.3) with no
        // smoothing between them. `GoalkeeperStandingState` is the
        // keeper's default, most-common state (idle/tracking, not a
        // dramatic save reaction — unlike GoalkeeperCatchingState's
        // similar-looking but deliberately-tuned, sourced "explosive
        // reaction" speed_boost, which this does NOT touch), and his
        // `optimal_position` shifts continuously as the ball moves
        // side-to-side along a fixed-X goal line -- so `distance_to_
        // optimal` oscillates across these small boundaries constantly
        // during ordinary play, snapping the multiplier on every
        // crossing. Ground-truth confirmed (duel_debug): this single
        // state was 28.2% of all vertical-classified jerk samples
        // across the whole match, the largest single contributor.
        // Replaced with one continuous Arrive call: `slowing_distance`
        // and the post-Arrive scale both vary smoothly with distance
        // instead of stepping, and the scale never exceeds 1.0 (never
        // boosts past the keeper's real max_speed — there was no
        // sourced rationale for the old 1.3x tier the way there is for
        // Catching's).
        let urgency = (distance_to_optimal / 15.0).clamp(0.0, 1.0); // 0 = calm, 1 = urgent
        let slowing_distance = 3.0 + (10.0 - 3.0) * urgency; // 3..10, continuous
        let scale = 0.7 + (1.0 - 0.7) * urgency; // 0.7..1.0, continuous, never boosts

        Some(
            SteeringBehavior::Arrive {
                target: optimal_position,
                slowing_distance,
            }
            .calculate(ctx.player)
            .velocity
                * scale,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Standing goalkeepers recover condition well due to low activity
        GoalkeeperCondition::new(ActivityIntensity::Recovery).process(ctx);
    }
}

impl GoalkeeperStandingState {
    fn is_opponent_in_danger_zone(&self, ctx: &StateProcessingContext) -> bool {
        if let Some(opponent_with_ball) = ctx.players().opponents().with_ball().next() {
            let opponent_distance = ctx
                .tick_context
                .grid
                .get(ctx.player.id, opponent_with_ball.id);

            return opponent_distance < DANGER_ZONE_RADIUS;
        }

        false
    }

    /// Determine if goalkeeper should rush out for the ball
    fn should_rush_out_for_ball(
        &self,
        ctx: &StateProcessingContext,
        opponent: &MatchPlayerLite,
    ) -> bool {
        let ball_position = ctx.tick_context.positions.ball.position;
        let keeper_position = ctx.player.position;
        let opponent_position = opponent.position;

        // Distance calculations
        let keeper_to_ball = (ball_position - keeper_position).magnitude();
        let opponent_to_ball = (ball_position - opponent_position).magnitude();

        // Goalkeeper skills
        let anticipation = ctx.player.skills.mental.anticipation / 20.0;
        let decisions = ctx.player.skills.mental.decisions / 20.0;
        let rushing_out = (anticipation + decisions) / 2.0;

        // Opponent skills
        let opponent_control = ctx.player().skills(opponent.id).technical.first_touch / 20.0;
        let opponent_pace = ctx.player().skills(opponent.id).physical.pace / 20.0;

        // Calculate time to reach ball (rough estimate)
        let keeper_speed = ctx.player.skills.physical.acceleration * (1.0 + rushing_out * 0.3);
        let opponent_speed = opponent_pace * 20.0;

        let keeper_time = keeper_to_ball / keeper_speed;
        let opponent_time = opponent_to_ball / opponent_speed;

        // Factors favoring rushing out:
        // 1. Keeper can reach ball first (with skill advantage)
        // 2. Ball is loose or opponent has poor control
        // 3. Ball is within reasonable distance

        let can_reach_first = keeper_time < opponent_time * (1.0 + rushing_out * 0.2);
        let ball_loose_or_poor_control = !ctx.ball().is_owned() || opponent_control < 0.5;
        let reasonable_distance = keeper_to_ball < CLOSE_DANGER_DISTANCE * 1.5;

        can_reach_first && ball_loose_or_poor_control && reasonable_distance
    }

    /// Calculate optimal goalkeeper position based on ball and goal
    fn calculate_optimal_position(&self, ctx: &StateProcessingContext) -> Vector3<f32> {
        let goal_center = ctx.ball().direction_to_own_goal();
        let ball_position = ctx.tick_context.positions.ball.position;

        // Goalkeeper skills affecting positioning
        let positioning_skill = ctx.player.skills.mental.positioning / 20.0;
        let command_of_area = ctx.player.skills.goalkeeping.command_of_area / 20.0;

        // Calculate distance from goal to ball
        let goal_to_ball = ball_position - goal_center;
        let distance_to_ball = goal_to_ball.magnitude();

        // realism-bug (2026-09-13, "still bursts, not smooth" follow-up
        // — the real dominant cause, found only after the speed-tier
        // fix above barely moved the numbers): `on_own_side()` is a
        // hard boolean, true/false purely on which side of the exact
        // halfway line (`field_size.half_width`) the ball currently
        // sits. The two branches below compute COMPLETELY different
        // target Y positions — up to ~100 units of lateral offset
        // (defensive sweeping, angle-bisection) vs. dead-centre on the
        // goal (conservative, opponent's half) — with nothing between
        // them. Every time the ball crosses halfway (routinely, many
        // times a match, as possession swings back and forth) the
        // keeper's TARGET itself snapped instantly by up to 100 units,
        // regardless of how smoothly he steers toward it — the earlier
        // speed-tier fix only smoothed how fast he chases a target that
        // was still teleporting. The discontinuity is purely in Y
        // (screen-vertical), matching the reported symptom exactly. Both
        // formulas are real, sourced, already-tuned behaviour (see their
        // own comments) — kept as-is, just computed unconditionally and
        // BLENDED across a real transition band around halfway instead
        // of hard-switched.
        let half_width = ctx.context.field_size.half_width as f32;
        let signed_own_side_depth = match ctx.player.side {
            Some(PlayerSide::Left) => half_width - ball_position.x,
            Some(PlayerSide::Right) => ball_position.x - half_width,
            None => 0.0,
        };
        const HALFWAY_TRANSITION_BAND: f32 = 40.0; // ~5m either side -- reasoned, not sourced
        let own_side_weight =
            ((signed_own_side_depth / HALFWAY_TRANSITION_BAND) * 0.5 + 0.5).clamp(0.0, 1.0);

        // Base distance from goal line (in meters/units)
        let mut optimal_distance_from_goal = 12.0; // Start about 12 units from goal line

        let own_side_position = {
            // Ball on defensive half - position based on threat level
            let threat_distance = distance_to_ball.min(300.0) / 300.0; // Normalize to 0-1

            // realism-bug (2026-07-26): sourced GK sweeping doctrine
            // (FBref "Sweeper" stats — themastermindsite.com's 2022-23
            // ranking) puts elite sweeper-keepers' defensive actions at
            // 15-20 yards (110-146u at this engine's 8u/m scale) from
            // goal, shot-stoppers below ~15.5yd (113u); this coefficient
            // was 20.0, capping even a max-skill keeper at ~30-43u total
            // (measured: 0.00% of GK position samples across 30 fresh
            // matches ever exceeded 100u, corner/FK/long-delivery
            // response-window peaks averaged only 16-39u — see CLAUDE.md
            // decisions log 2026-07-26). Raised to 100.0 so a genuinely
            // close threat (a corner/cross/FK arriving in the box) lets
            // a good keeper's rest position climb into the real 75-130u
            // range instead of being structurally capped regardless of
            // skill or threat. `clamp_to_penalty_area` below still
            // hard-bounds this to the real box, so a weak keeper (low
            // command_of_area/positioning) still stays correctly
            // shallow — only the achievable CEILING moved.
            optimal_distance_from_goal += (1.0 - threat_distance) * 100.0 * command_of_area;

            // Better positioning = more accurate placement
            optimal_distance_from_goal *= 0.8 + positioning_skill * 0.4;

            // Narrow the angle - position on line between goal and ball
            let direction_to_ball = if distance_to_ball > 1.0 {
                goal_to_ball.normalize()
            } else {
                Vector3::new(1.0, 0.0, 0.0) // Fallback if ball too close to goal
            };

            let mut new_position = goal_center + direction_to_ball * optimal_distance_from_goal;

            // Lateral adjustment for angle coverage
            let ball_y_offset = ball_position.y - goal_center.y;

            // realism-bug (2026-07-20): during a live direct free kick
            // with a formed wall (award_restart_for_foul, ≤280u), the
            // generic "narrow the angle toward the ball" pull below is
            // the wrong model — it leans toward the ball's own side,
            // which is the NEAR post, exactly the side the wall (now
            // asymmetric — see engine/tick.rs::resolve_free_kick) is
            // shifted to cover most heavily. A real keeper instead
            // favours the FAR post, trusting the wall for the direct
            // near-post line. Same near_bias formula as the wall
            // placement / aim-selection code — must stay in sync, or
            // the keeper drifts toward a side that isn't actually open.
            let is_direct_fk = ctx.tick_context.ball.pass_origin_restart
                == PassOriginRestart::DirectFreeKick;
            if is_direct_fk && distance_to_ball <= 280.0 {
                const GK_FAR_POST_BIAS: f32 = 11.0;
                let near_bias =
                    (ball_y_offset / (distance_to_ball.max(1.0) * 0.6)).clamp(-1.0, 1.0);
                new_position.y += -near_bias * GK_FAR_POST_BIAS;
            } else {
                // realism-bug (2026-07-25): sourced coaching doctrine
                // (angle-bisection — "a line from the ball, through the
                // keeper, to the centre of the goal") says the keeper's
                // lateral offset should track the ANGLE from goal-centre
                // to the ball, not a flat fraction of the ball's raw
                // y-offset. `direction_to_ball.y` already IS that
                // bisector angle (normalized) — the bug was applying a
                // second, redundant, badly-scaled flat-offset term on
                // top of it instead of using it directly. Measured
                // baseline (25-match external position check): mean
                // lateral deviation from centre only ~8-9 units (~1.1m)
                // even with the ball well into the defensive third,
                // because `optimal_distance_from_goal` (the OLD flat
                // term's only real lever, via `positioning_skill`)
                // collapses to a small base value except when the ball
                // is genuinely close — so the flat term dominated and
                // stayed tiny for most of a match. Fix: derive the
                // lateral offset from `direction_to_ball.y` (the real
                // bisector angle) at a fixed ANGLE_COVERAGE_DEPTH
                // standing depth (100u, recalibrated from an initial 44u
                // six-yard-box-equivalent value that externally measured
                // as barely moving the realized lateral slope — see the
                // decisions log entry for both calibration attempts),
                // decoupled from the small, separately-threat-gated
                // `optimal_distance_from_goal` used for forward
                // advancement. Skill still modulates it (a poorly
                // positioned keeper narrows the angle less precisely).
                const ANGLE_COVERAGE_DEPTH: f32 = 100.0;
                let skill_factor = 0.6 + positioning_skill * 0.4;
                new_position.y = goal_center.y + direction_to_ball.y * ANGLE_COVERAGE_DEPTH * skill_factor;
            }

            new_position
        };

        let opponent_side_position = {
            // Ball on opponent's half - stay closer to goal but ready
            let opponent_distance_from_goal = 12.0 + command_of_area * 8.0;

            let mut new_position = goal_center;
            new_position.x += opponent_distance_from_goal
                * (if ctx.player.side == Some(PlayerSide::Left) {
                    1.0
                } else {
                    -1.0
                });

            new_position
        };

        let blended = own_side_position * own_side_weight
            + opponent_side_position * (1.0 - own_side_weight);

        // Keep within penalty area
        self.clamp_to_penalty_area(ctx, blended)
    }

    fn clamp_to_penalty_area(
        &self,
        ctx: &StateProcessingContext,
        position: Vector3<f32>,
    ) -> Vector3<f32> {
        let penalty_area = ctx
            .context
            .penalty_area(ctx.player.side == Some(PlayerSide::Left));
        Vector3::new(
            position.x.clamp(penalty_area.min.x, penalty_area.max.x),
            position.y.clamp(penalty_area.min.y, penalty_area.max.y),
            0.0,
        )
    }
}
