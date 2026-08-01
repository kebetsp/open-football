use crate::r#match::events::Event;
use crate::r#match::goalkeepers::states::common::{ActivityIntensity, GoalkeeperCondition};
use crate::r#match::goalkeepers::states::state::GoalkeeperState;
use crate::r#match::player::events::PlayerEvent;
use crate::r#match::player::strategies::players::ops::goalkeeper_skill::GoalkeeperSkillProfile;
use crate::r#match::{
    ConditionContext, PlayerDistanceFromStartPosition, StateChangeResult, StateProcessingContext,
    StateProcessingHandler, SteeringBehavior,
};
use nalgebra::Vector3;

#[derive(Default, Clone)]
pub struct GoalkeeperCatchingState {}

impl StateProcessingHandler for GoalkeeperCatchingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if self.is_catch_successful(ctx) {
            let mut holding_result =
                StateChangeResult::with_goalkeeper_state(GoalkeeperState::HoldingBall);

            holding_result
                .events
                .add_player_event(PlayerEvent::CaughtBall(ctx.player.id));

            return Some(holding_result);
        }

        // Shot is live: stay in Catching and keep sprinting toward the
        // intercept line. The old logic exited to Standing / ComingOut
        // the moment the ball was >12u away, which meant a keeper
        // aiming for the far post gave up the instant the shot was
        // fired. With a cached shot target the keeper commits.
        if ctx.tick_context.ball.cached_shot_target.is_some() {
            return None;
        }

        // Ball is moving away from the keeper at speed — only credit
        // a parry when the ball was actually within reach (the keeper
        // got a hand to it). Otherwise the shot just missed past the
        // keeper and "parry" would falsely credit a save for a wide
        // shot the GK never touched.
        let ball_speed = ctx.tick_context.positions.ball.velocity.norm();
        let ball_distance = ctx.ball().distance();
        if ball_speed > 2.0 && !ctx.ball().is_towards_player_with_angle(0.6) {
            if ctx.tick_context.ball.cached_shot_target.is_some() && ball_distance < 25.0 {
                return Some(StateChangeResult::with_goalkeeper_state_and_event(
                    GoalkeeperState::Standing,
                    Event::PlayerEvent(PlayerEvent::ParriedBall(ctx.player.id)),
                ));
            }
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Standing,
            ));
        }

        // If ball is too far, decide based on distance from goal
        if ctx.ball().distance() > 12.0 {
            // If already far from goal, return rather than chasing further
            if ctx.player().distance_from_start_position() > 40.0 {
                return Some(StateChangeResult::with_goalkeeper_state(
                    GoalkeeperState::ReturningToGoal,
                ));
            }
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::ComingOut,
            ));
        }

        if ctx.player().position_to_distance() == PlayerDistanceFromStartPosition::Big {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::ReturningToGoal,
            ));
        }

        if ctx.in_state_time > 30 {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Standing,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let prof = GoalkeeperSkillProfile::from_ctx(ctx);
        // Sprint reaction speed: 1.6..2.6x band, gated by explosive_mult.
        let speed_boost =
            (1.6 + prof.shot_stopping * 0.5 + prof.dive_reach * 0.5) * prof.explosive_mult;

        // Shot in flight → commit to the intercept line, don't chase
        // the current ball position (it's moving at 5.6 u/tick and
        // outrunning the keeper's pursuit steering).
        if let Some(target) = &ctx.tick_context.ball.cached_shot_target {
            let goal_pos = ctx.ball().direction_to_own_goal();
            let intercept = Vector3::new(goal_pos.x, target.goal_line_y, 0.0);
            // realism-bug (2026-07-30, extended 2026-07-31): a direct
            // free kick's flight time (130-250u away, 400-800+ms / 40-90
            // ticks) is long enough that even at NORMAL running speed
            // (the 1.0x this comment originally capped it to) the keeper
            // can walk down almost any reasonable initial gap over the
            // whole flight — measured directly: after fixing shot aim to
            // exploit the keeper's actual starting position, the median
            // lateral error at save-check time barely moved (1.89u →
            // 1.84u across 5000+ evaluations), because the keeper simply
            // walks the rest of the way to the EXACT, perfectly-known
            // `target.goal_line_y` regardless of where he started. A
            // real keeper mostly commits to his stance BEFORE the kick
            // (already modelled in `calculate_optimal_position`'s
            // far-post lean) and only makes a small late reactive
            // adjustment/dive as the ball arrives — not a sustained,
            // omniscient walk across the goal for the better part of a
            // second. Capped to a slow creep (reflecting that small late
            // adjustment) instead of full running speed; the keeper's
            // actual SAVE range is unaffected — `is_catch_successful`'s
            // separate `reach` term already models diving distance on
            // top of wherever he's standing when the ball arrives. Every
            // other shot type (including normal long-range open-play
            // shots) keeps the existing boosted-sprint behaviour.
            let effective_boost = if target.is_direct_fk { 0.2 } else { speed_boost };
            return Some(
                SteeringBehavior::Arrive {
                    target: intercept,
                    slowing_distance: 2.0,
                }
                .calculate(ctx.player)
                .velocity
                    * effective_boost,
            );
        }

        let ball_distance = ctx.ball().distance();
        if ball_distance > 3.0 {
            Some(
                SteeringBehavior::Pursuit {
                    target: ctx.tick_context.positions.ball.position,
                    target_velocity: ctx.tick_context.positions.ball.velocity,
                }
                .calculate(ctx.player)
                .velocity
                    * speed_boost,
            )
        } else {
            Some(
                SteeringBehavior::Arrive {
                    target: ctx.tick_context.positions.ball.position,
                    slowing_distance: 1.5,
                }
                .calculate(ctx.player)
                .velocity
                    * (speed_boost * 0.8),
            )
        }
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Catching is a moderate intensity activity requiring focused effort
        GoalkeeperCondition::new(ActivityIntensity::Moderate).process(ctx);
    }
}

impl GoalkeeperCatchingState {
    fn is_catch_successful(&self, ctx: &StateProcessingContext) -> bool {
        let prof = GoalkeeperSkillProfile::from_ctx(ctx);

        // Shot-in-flight: judge the save from the *intercept line*, not
        // from current ball distance. A ball aimed into the corner
        // passes the GK 8-15 units wide of their current position —
        // real keepers reach 3-4 m (6-8 u) diving, so the relevant
        // metric is "how far off the line am I?", not "am I touching
        // the ball right now?".
        if let Some(target) = &ctx.tick_context.ball.cached_shot_target {
            // Ball over the bar — no save attempt worth making.
            if target.goal_line_z > 2.44 {
                return false;
            }
            // Effective reach in game units: weak ~14u, elite ~30u.
            let reach = 10.0 + prof.dive_reach * 12.0 + prof.shot_stopping * 4.0;
            let lateral_error = (ctx.player.position.y - target.goal_line_y).abs();
            if lateral_error > reach {
                return false;
            }

            // Build shot difficulty in 0..1 from placement, power,
            // reaction-window, and keeper-offline factors.
            let placement = (lateral_error / reach).clamp(0.0, 1.0);
            let ball_speed = ctx.tick_context.positions.ball.velocity.norm();
            let power = ((ball_speed - 2.0) / 6.0).clamp(0.0, 1.0);
            let lateral_factor = placement; // already a 0..1 lateral error.
            let height_factor = (target.goal_line_z / 2.44).clamp(0.0, 1.0);
            let reaction = (1.0 - prof.shot_stopping).clamp(0.0, 1.0) * 0.4;

            let shot_difficulty = (power * 0.28
                + placement * 0.24
                + lateral_factor * 0.18
                + height_factor * 0.10
                + reaction * 0.10
                + (1.0 - prof.condition_mult) * 0.10)
                .clamp(0.0, 1.0);

            // Per-shot save probability, then converted to per-tick.
            // Calibrated for ~3 ticks of approach during a save.
            let mut save_prob = prof.save_probability(shot_difficulty);
            // Deflection damping: the GK was set for the original
            // trajectory. A redirected shot arrives on a line they
            // haven't committed to, so reaction window is shorter.
            // Real PL data: deflected on-target shots produce ~30% goals
            // vs ~10% for clean on-target shots — a ~3× boost to
            // goal-per-shot, which we model as a ~0.50 multiplier to
            // save_prob (keepers save the rest by reflex or blocked
            // shot recovery).
            if target.deflected {
                save_prob *= 0.50;
            }
            // realism-bug (2026-07-30, revised 2026-07-31 twice): `per_tick_save(p, n)`
            // derives a per-tick rate such that `n` REPEATED rolls compound
            // back to the intended per-shot probability `p` — but this
            // state never exits while `cached_shot_target` is live, so the
            // ACTUAL number of ticks this roll fires is however long the
            // keeper stays within `reach`, not a fixed constant. A hardcoded
            // `expected_ticks=40.0` for direct FKs was wrong: measured
            // actual evaluation counts per shot ranged 24-255 (median ~90),
            // so the same per-tick rate still compounded well past
            // `save_prob` for shots that ran longer than 40 ticks (measured:
            // state-machine saves fired on ~90% of direct FK shots even
            // after separately fixing the physics-layer height gate). A
            // second attempt recomputed "ticks remaining" fresh on every
            // evaluation instead — this was ALSO wrong, in the opposite
            // direction: recomputing a brand-new n-roll decomposition every
            // tick means the per-tick rate climbs sharply as the remaining
            // count shrinks near arrival, so the last few ticks each carry
            // a large chunk of `save_prob` on their own, stacked on top of
            // everything already rolled earlier — net over-counting again.
            // The fix that actually works: `total_flight_ticks`, a FIXED
            // per-shot constant computed once at dispatch (the same
            // `ticks_to_goal` already used for the goal_line_y/z
            // projection) and stored on `ShotTarget`. Using the same
            // constant for every evaluation of this shot means however
            // many times the function actually fires, the cumulative
            // probability tracks the intended per-shot value instead of
            // compounding past it in either direction. Every other shot
            // type keeps the untouched fixed 3.0 calibration.
            let expected_ticks = if target.is_direct_fk {
                target.total_flight_ticks.max(1.0)
            } else {
                3.0
            };
            let per_tick = prof.per_tick_save(save_prob, expected_ticks);
            let roll = ctx.context.rng.unit_f32();
            // realism-bug (2026-07-31) diagnostic, scratch-only: every
            // per-tick evaluation of the state-machine save for a direct
            // FK — lateral error, shot_difficulty, save_prob, per_tick
            // rate, and the roll outcome. Grouping these by shot (the
            // count of consecutive lines) reveals the REAL in-reach
            // window length vs. the assumed expected_ticks=40. Remove
            // once the investigation concludes.
            #[cfg(feature = "match-logs")]
            if target.is_direct_fk {
                log::info!(
                    "FKCONV_CATCH lat_err={:.2} reach={:.2} shot_diff={:.3} save_prob={:.3} per_tick={:.4} roll={:.4} saved={}",
                    lateral_error,
                    reach,
                    shot_difficulty,
                    save_prob,
                    per_tick,
                    roll,
                    roll < per_tick
                );
            }
            return roll < per_tick;
        }

        let distance_to_ball = ctx.ball().distance();
        let max_catch_distance = prof.effective_catch_distance;
        if distance_to_ball > max_catch_distance {
            return false;
        }

        let ball_speed = ctx.tick_context.positions.ball.velocity.norm();
        if ball_speed > 0.5 && !ctx.ball().is_towards_player_with_angle(0.6) {
            return false;
        }

        let ball_height = ctx.tick_context.positions.ball.position.z;
        let stretch = (distance_to_ball / max_catch_distance).clamp(0.0, 1.0);
        let power = ((ball_speed - 1.5) / 6.0).clamp(0.0, 1.0);

        // Awkward-height penalty: ground or above-head balls are harder.
        let height_pen = if (0.5..=1.8).contains(&ball_height) {
            0.0
        } else if ball_height < 0.2 {
            0.18
        } else if ball_height > 2.5 {
            0.22
        } else {
            0.06
        };

        let direction_factor = if ctx.ball().is_towards_player_with_angle(0.7) {
            0.0
        } else {
            0.18
        };

        let catch_difficulty = (power * 0.28
            + stretch * 0.22
            + height_pen * 0.18
            + direction_factor * 0.12
            + (1.0 - prof.condition_mult) * 0.10
            + prof.poor_skill_penalty * 0.10)
            .clamp(0.0, 1.0);

        let catch_prob = prof.catch_probability(catch_difficulty);
        ctx.context.rng.unit_f32() < catch_prob
    }
}
