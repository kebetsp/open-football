use crate::r#match::defenders::states::DefenderState;
use crate::r#match::defenders::states::common::{ActivityIntensity, DefenderCondition};
use crate::r#match::player::strategies::common::players::ops::defender_skill::DefenderSkillProfile;
use crate::r#match::player::strategies::players::DefensiveRole;
use crate::r#match::{
    ConditionContext, MatchPlayerLite, StateChangeResult, StateProcessingContext,
    StateProcessingHandler,
};
use nalgebra::Vector3;

// 2026-08-30 (engine-audit Module 6): the local
// `TACKLING_DISTANCE_THRESHOLD = 10.0` this file used to define happened
// to already match Tackling's own gate by coincidence — it still never
// checked the tackle cooldown, and the box-emergency branch below used
// an entirely different, unfixed 25.0. Both now route through the
// shared `DefensiveOperationsImpl::can_engage_tackle`.
const STAMINA_THRESHOLD: f32 = 20.0; // Minimum stamina to continue marking
const BALL_PROXIMITY_THRESHOLD: f32 = 15.0; // Increased from 10.0 - react earlier to ball
const HEADING_HEIGHT: f32 = 1.5;
const HEADING_DISTANCE: f32 = 5.0;
// 2026-09-16 (Pavel's direct report: "Red #4 spent minutes 1 to 5
// chasing Blue #7 in the middle of the pitch while the ball hasn't even
// been there"). Root cause, confirmed with raw data across three failed
// scoring-formula fixes before finding this: `find_help_target`/
// `find_unmarked_opponent`'s scan radii are so tight (28-100u) that
// 76-93% of selections have <=1 real candidate -- there is usually no
// competition for the scoring formula to decide at all, so tuning it
// (tried three times) could never matter. The actual loop is
// positional, not a scoring bug: this state's ONLY distance check is
// "am I still close to the opponent I'm marking" (below) -- there was
// NEVER a check for "have I wandered far from my own defensive zone,"
// unlike guarding/mod.rs's own equivalent state, which has exactly
// this exit (its `MAX_DISTANCE_FROM_START`/`Returning` transition).
// Since chasing keeps the target close, and staying close is the only
// thing that keeps him selected, the loop had no exit at all -- a
// defender could mark the same opponent anywhere on the pitch,
// indefinitely, regardless of the ball. Mirrors Guarding's own constant
// value (120u/15m) for the same real-world concept rather than
// inventing a second slightly-different number.
const MAX_DISTANCE_FROM_START: f32 = 120.0;

#[derive(Default, Clone)]
pub struct DefenderMarkingState {}

impl StateProcessingHandler for DefenderMarkingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        // BOX EMERGENCY — stop marking an off-ball runner if the
        // carrier is INSIDE our penalty area and we're one of the two
        // closest defenders. A shot is imminent; engage the carrier
        // regardless of marking duties.
        if ctx.player().defensive().is_box_emergency_for_me() {
            if let Some(carrier) = ctx.players().opponents().with_ball().next() {
                let d = carrier.distance(ctx);
                if ctx.player().defensive().can_engage_tackle(d) {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Tackling,
                    ));
                }
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Pressing,
                ));
            }
        }

        // Take ball only if best positioned — prevents swarming
        if ctx.ball().should_take_ball_immediately() && ctx.team().is_best_player_to_chase_ball() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::TakeBall,
            ));
        }

        // 1. Check if the defender has enough stamina to continue marking
        let stamina = ctx.player.player_attributes.condition_percentage() as f32;
        if stamina < STAMINA_THRESHOLD {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Resting,
            ));
        }

        // Check if ball is aerial and at heading height
        let ball_position = ctx.tick_context.positions.ball.position;
        let ball_distance = ctx.ball().distance();

        if ball_position.z > HEADING_HEIGHT
            && ball_distance < HEADING_DISTANCE
            && ctx.ball().is_towards_player_with_angle(0.6)
        {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Heading,
            ));
        }

        // 2026-09-14 (Pavel's direct report, same mechanism as the
        // Guarding fix above it in this same investigation): unlike
        // Guarding, Marking had NO direct "a ball carrier is nearby,
        // react to him" check at all. `find_best_marking_target` below,
        // when a carrier exists, first tries `find_help_target()` --
        // which explicitly EXCLUDES ball carriers from its own scan
        // ("Help role: pick the most dangerous NON-CARRIER opponent").
        // If that finds any qualifying alternate target within its 28u
        // scan radius, this state tracked THAT player instead, silently
        // ignoring a live, ball-carrying opponent who might be right in
        // front of it. Checked first, unconditionally: any carrier
        // within realistic marking/engagement range is this defender's
        // job before any off-ball target selection runs.
        if let Some(ball_carrier) = ctx.players().opponents().with_ball().next() {
            let dist_to_carrier = ball_carrier.distance(ctx);
            if ctx.player().defensive().can_engage_tackle(dist_to_carrier) {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Tackling,
                ));
            }
            if dist_to_carrier < 60.0 {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Pressing,
                ));
            }
        }

        // 2. Find best opponent to mark using coordination system
        // First try to find an unmarked opponent if current target is being engaged by another defender
        let opponent_to_mark = self.find_best_marking_target(ctx);

        if let Some(opponent) = opponent_to_mark {
            let distance_to_opponent = opponent.distance(ctx);

            // Priority: If opponent with ball is close, press/tackle immediately
            if opponent.has_ball(ctx) {
                if ctx
                    .player()
                    .defensive()
                    .can_engage_tackle(distance_to_opponent)
                {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Tackling,
                    ));
                }
                // Press the ball carrier aggressively — any marking defender should engage
                if distance_to_opponent < 30.0 {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Pressing,
                    ));
                }
            }

            // If opponent is too far, switch to running to catch up.
            // The "too far" threshold is the marking_profile-driven
            // ideal_marking_distance scaled 2x, so a poor marker has a
            // wider tolerance (he stands further off anyway) and an
            // elite marker breaks contact earlier.
            let def_profile = DefenderSkillProfile::from_ctx(ctx);
            if distance_to_opponent > def_profile.ideal_marking_distance * 2.0 {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Running,
                ));
            }

            // 2026-09-16: don't follow an opponent too far from our own
            // defensive zone -- see the const doc comment above for the
            // full root-cause trace. This is the actual fix: the check
            // above only ever compares against the OPPONENT'S own
            // position, which chasing keeps satisfied by construction.
            //
            // Uses `shape_base`, NOT `start_position` -- caught via a
            // raw diagnostic showing `start_position` taking on 75-200+
            // distinct values within a single half for most defenders,
            // when it should be stable. Root cause: EVERY defender gets
            // an unconditional default `ShapeRule{DefendingOwnThird,
            // CompactBackLine}` (flow/field.rs, pre-existing, 2026-08-30
            // Principles-of-Play work, nothing to do with this fix) --
            // `CompactBackLine`'s own target legitimately shifts with
            // live ball position and team phase (a real, correct team-
            // shape mechanism, not a bug), which continuously overwrites
            // `start_position` while a defender is in his own defensive
            // third. Comparing against a target that itself tracks the
            // ball defeats the whole point of this check. `shape_base`
            // is the genuinely static formation-table anchor (confirmed
            // this session, `flow/field.rs`'s own setup/halftime/sub
            // writers only -- the one stable reference every other
            // zone-anchored mechanism in this codebase already uses for
            // exactly this reason, e.g. Chunk C's own rotation target).
            if let Some(shape_base) = ctx.player.shape_base {
                let dist_from_start = (ctx.player.position - shape_base).magnitude();
                if dist_from_start > MAX_DISTANCE_FROM_START {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Returning,
                    ));
                }
            }

            // If ball is close and unmarked, consider intercepting
            if ctx.ball().distance() < BALL_PROXIMITY_THRESHOLD && !opponent.has_ball(ctx) {
                if ctx.ball().is_towards_player_with_angle(0.7) {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Intercepting,
                    ));
                }
            }

            // Role check: if a ball carrier exists and our role has
            // flipped away from Help, route back through Standing so the
            // role block can reassign us (Primary if we're now closest,
            // Cover if we're goal-side second, Hold otherwise).
            if ctx.players().opponents().with_ball().next().is_some() {
                let role = ctx.player().defensive().defensive_role_for_ball_carrier_unified();
                if role != DefensiveRole::Help {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Standing,
                    ));
                }
            }

            // Continue marking
            None
        } else {
            // No opponent to mark — drop to Standing so the role block
            // there can re-evaluate (HoldingLine would route Help straight
            // back to Marking, causing a state ping-pong).
            Some(StateChangeResult::with_defender_state(
                DefenderState::Standing,
            ))
        }
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Profile-driven marking: ideal distance and goal-side weight
        // both scale with marking_profile / positioning curve, so an
        // elite defender stands closer and more goal-side, while a
        // poor marker stays at arm's length and gets shrugged off runs.
        if let Some(opponent_to_mark) = self.find_best_marking_target(ctx) {
            let def_profile = DefenderSkillProfile::from_ctx(ctx);
            let own_goal = ctx.ball().direction_to_own_goal();
            let opponent_velocity = opponent_to_mark.velocity(ctx);

            let prediction_time = 0.3;
            let opponent_future_position =
                opponent_to_mark.position + opponent_velocity * prediction_time;

            let mark_dist = def_profile.ideal_marking_distance;

            // realism-bug (2026-08-30, deflection-rate investigation): a
            // temporary diagnostic (`shot_dispatch_diag`, players.rs)
            // found that at the exact moment a shot is struck, the
            // globally-nearest opposing outfielder is BEHIND the ball
            // along the shot direction (structurally incapable of
            // blocking, not merely mis-angled) in 96% of cases with no
            // corridor candidate at all — and 88% of those have had at
            // least ~2 seconds of settled possession beforehand (49.7%
            // had 5-10s), ruling out "no time to recover" as the
            // dominant explanation. The skill-based `goal_side_weight`
            // (0.45-0.85) blends toward the BALL's own position almost
            // as much as toward goal for an average marker — real
            // defensive doctrine is unconditional once a mark is inside
            // realistic shooting range: stay strictly goal-side, jockey
            // laterally, never let him in behind you. Ramp the blend
            // toward fully goal-side (no ball-side pull at all) the
            // closer the mark gets to goal, using the same `80.0`u
            // shooting-danger threshold `DefenderPressingState`'s own
            // `shot_zone_bias` already established for an analogous
            // purpose — reused, not reinvented.
            let dist_to_goal = (own_goal - opponent_future_position).magnitude();
            let danger_ramp = (1.0 - (dist_to_goal / 80.0)).clamp(0.0, 1.0);
            let goal_side_w = def_profile.goal_side_weight + (1.0 - def_profile.goal_side_weight) * danger_ramp;

            let to_goal = (own_goal - opponent_future_position).normalize();
            let goal_side_offset = to_goal * mark_dist * goal_side_w;

            let ball_position = ctx.tick_context.positions.ball.position;
            let to_ball = (ball_position - opponent_future_position).normalize();
            let ball_side_offset = to_ball * mark_dist * (1.0 - goal_side_w);

            let desired_position = opponent_future_position + goal_side_offset + ball_side_offset;

            let to_desired = desired_position - ctx.player.position;
            let distance = to_desired.magnitude();

            if distance < 1.0 {
                return Some(to_desired * 0.5);
            }

            let direction = to_desired.normalize();
            // Urgency relative to the profile-driven mark distance.
            let urgency = (distance / mark_dist.max(4.0)).clamp(0.6, 2.0);
            // Recovery-run mult bakes pace/stamina/condition into a
            // 0.58..1.05 multiplier so tired markers can't keep up
            // with fast attackers.
            //
            // realism-bug (2026-09-13, "ball carrier moves directly
            // vertically, in bursts" investigation): `skills.physical.
            // pace` is a raw 1-20 SKILL RATING, not a per-tick speed --
            // used directly here it could reach `20 * 2.0(urgency) *
            // 1.05(recovery) * 1.3(threat_boost) ~= 54.6` units/tick,
            // ~90x a real player's actual top speed
            // (max_speed_with_condition is ~0.36-0.6 units/tick, which
            // already bakes pace/acceleration into a real scale --
            // same pattern already established and used everywhere
            // else in this codebase). Surfaced by the same ground-truth
            // trace (duel_debug's own recorded `has_ball`/velocity):
            // when a marker wins the ball mid-duel, his state doesn't
            // transition off Marking until the NEXT tick's `process()`
            // runs, so this exact velocity computation fires for one
            // real tick while he already, genuinely has the ball --
            // and was, after fixing the larger sibling bug in the
            // Passing states, the single largest remaining source of
            // carrier bursting in the same measurement (87.5% of
            // vertical-classified carrier jerk samples). `urgency`/
            // `recovery_run_mult`/`threat_boost` are still legitimate
            // multipliers on top of a REAL base speed -- only the base
            // itself was wrong.
            let speed = ctx
                .player
                .skills
                .max_speed_with_condition(ctx.player.player_attributes.condition)
                * urgency
                * def_profile.recovery_run_mult;

            let threat_boost = if opponent_to_mark.has_ball(ctx) && distance < 20.0 {
                1.3
            } else {
                1.0
            };

            Some(direction * speed * threat_boost + ctx.player().separation_velocity() * 0.2)
        } else {
            Some(Vector3::new(0.0, 0.0, 0.0))
        }
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Marking involves constant movement following opponent - moderate intensity
        DefenderCondition::with_velocity(ActivityIntensity::Moderate).process(ctx);
    }
}

impl DefenderMarkingState {
    /// Find the best marking target using the role system.
    /// In the Help role (ball carrier active), pick the most dangerous
    /// non-carrier unmarked opponent — this cuts pass lanes around the
    /// primary presser. Otherwise (no live ball-carrier scenario) fall
    /// back to the generic "find most dangerous opponent" scan.
    fn find_best_marking_target(&self, ctx: &StateProcessingContext) -> Option<MatchPlayerLite> {
        if ctx.players().opponents().with_ball().next().is_some() {
            if let Some(help_target) = ctx.player().defensive().find_help_target() {
                return Some(help_target);
            }
        }

        // No live carrier: mark the most dangerous unmarked opponent.
        ctx.player()
            .defensive()
            .find_unmarked_opponent(100.0)
            .or_else(|| self.find_most_dangerous_opponent(ctx))
    }

    /// Find the most dangerous opponent to mark based on multiple factors
    fn find_most_dangerous_opponent(
        &self,
        ctx: &StateProcessingContext,
    ) -> Option<MatchPlayerLite> {
        let player_ops = ctx.player();
        let own_goal_position = ctx.ball().direction_to_own_goal();
        let ball_position = ctx.tick_context.positions.ball.position;

        let mut best_opponent = None;
        let mut best_score = f32::MIN;

        // Direct iteration — no .collect() needed
        for opponent in ctx.players().opponents().nearby(150.0) {
            let mut danger_score = 0.0;

            if opponent.has_ball(ctx) {
                danger_score += 100.0;
            }

            let distance_to_own_goal = (opponent.position - own_goal_position).magnitude();
            danger_score += (500.0 - distance_to_own_goal) / 10.0;

            // Removed 2026-09-16 (same fix as guarding/mod.rs's own
            // Factor 5, same day, same root cause -- see that file's
            // doc comment for the full trace/verification): an
            // additive "closer to THIS defender scores higher" bonus,
            // on top of the already-hard `nearby(150.0)` eligibility
            // scan below, closed a positive-feedback loop with this
            // function's own consumer (the defender's velocity() chases
            // whoever this picks) -- verified via A/B to be the actual
            // driver of Pavel's "Red #4 chasing Blue #7 for 5 minutes,
            // ball nowhere near" report, independent of Chunk B.

            let opponent_velocity = opponent.velocity(ctx);
            let speed_sq = opponent_velocity.norm_squared();

            if speed_sq > 0.01 {
                let speed = speed_sq.sqrt();
                let to_our_goal = (own_goal_position - opponent.position).normalize();
                let velocity_dir = opponent_velocity * (1.0 / speed);
                let alignment = velocity_dir.dot(&to_our_goal);

                if alignment > 0.0 {
                    danger_score += alignment * 30.0;
                    if speed > 3.0 && alignment > 0.7 {
                        danger_score += 25.0;
                    }
                }
            }

            if !opponent.has_ball(ctx) {
                let distance_to_ball = (opponent.position - ball_position).magnitude();
                danger_score += (50.0 - distance_to_ball.min(50.0)) / 5.0;
            }

            // Receiver threat — weights off_the_ball, finishing,
            // anticipation, pace + composure heavily, replacing the
            // simple (pace+dribbling+finishing)/3 average. A pacy poacher
            // (high finishing + off_ball) is now scored materially above
            // a midfield carrier with the same average skill total.
            let opponent_skills = player_ops.skills(opponent.id);
            let receiver_threat = ((opponent_skills.mental.off_the_ball / 20.0).powf(1.45) * 0.22
                + (opponent_skills.physical.pace / 20.0).powf(1.25) * 0.14
                + (opponent_skills.physical.acceleration / 20.0).powf(1.25) * 0.12
                + (opponent_skills.technical.finishing / 20.0).powf(1.45) * 0.16
                + (opponent_skills.mental.anticipation / 20.0).powf(1.30) * 0.14
                + (opponent_skills.mental.composure / 20.0).powf(1.20) * 0.08)
                .clamp(0.0, 1.0);
            danger_score += receiver_threat * 30.0;

            // Chunk B (2026-09-15,
            // docs/zonal-coverage-rotation-and-tackle-risk-design.md):
            // same zone preference as guarding.rs's own `find_guard_
            // target` and `calculate_opponent_danger_score` (defensive.rs)
            // -- one shared multiplier, three call sites, so the whole
            // Guarding/Marking off-ball scan family agrees on what "my
            // zone" means.
            danger_score *= ctx.player().defensive().zone_preference_multiplier(opponent.position);

            // 2026-09-16: real ball-relevance gate -- see
            // `ball_relevance_multiplier`'s own doc comment (defensive.rs)
            // for the full trace/verification. Removing the old
            // "closeness to defender" term alone did not fix Pavel's
            // report; this is the term that actually did.
            danger_score *= ctx.player().defensive().ball_relevance_multiplier(opponent.position);

            if danger_score > best_score {
                best_score = danger_score;
                best_opponent = Some(opponent);
            }
        }

        best_opponent
    }
}
