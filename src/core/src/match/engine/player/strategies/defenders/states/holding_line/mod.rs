use nalgebra::Vector3;

use crate::r#match::defenders::states::DefenderState;
use crate::r#match::defenders::states::common::{ActivityIntensity, DefenderCondition};
use crate::r#match::player::strategies::common::players::ops::defender_skill::DefenderSkillProfile;
use crate::r#match::player::strategies::players::DefensiveRole;
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
};

const MAX_DEFENSIVE_LINE_DEVIATION: f32 = 35.0; // Tighter line — less room for attackers
const BALL_PROXIMITY_THRESHOLD: f32 = 150.0; // React to ball from further out
const MARKING_DISTANCE_THRESHOLD: f32 = 50.0; // Pick up attackers from further away
const PRESSING_DISTANCE_THRESHOLD: f32 = 60.0; // Step out to press ball carrier earlier

#[derive(Default, Clone)]
pub struct DefenderHoldingLineState {}

impl StateProcessingHandler for DefenderHoldingLineState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        // TEMPORARY diagnostic (docs/positional-play-master-design.md
        // Stage A). Removed once the investigation concludes.
        crate::match_log_info!("HLDIAG_ENTRY");
        // Attacking corner: centre-backs push up to attack the delivery
        // (self-terminates the instant the corner is over).
        if !ctx.player.has_ball(ctx)
            && ctx
                .player
                .tactical_position
                .current_position
                .is_central_defender()
            && ctx.ball().is_team_attacking_corner()
        {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::AttackingCorner,
            ));
        }

        // BOX EMERGENCY — ball is in our penalty area with an opposing
        // carrier. Break shape and engage. The two closest defenders
        // attack; the rest hold line so the far side isn't exposed.
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

        // STEP UP — attacker is approaching the penalty area and I'm
        // the closest defender. Meet them outside the box instead of
        // collapsing deep. Real football: defenders engage at the 18-yard
        // line, not at the 6-yard line.
        if ctx.player().defensive().should_step_up_to_meet_attacker() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Pressing,
            ));
        }

        // 1. My own actual zonal target — the EXACT point velocity()
        // steers toward (own tactical slot, ball-proportional offset,
        // compactness shift). docs/positional-play-master-design.md
        // Stage A found this gate used to compare against a SEPARATE,
        // more volatile reference (a blend of the live back-line
        // average with the team's phase-driven `defensive_line_x`) —
        // a real, structural mismatch: raw diagnostic traces showed
        // defenders sitting within a fraction of a unit of their own
        // start position still measuring 250+ units of "deviation",
        // because `defensive_line_x` itself can jump by hundreds of
        // units the instant GamePhase flips (this engine spends the
        // majority of match time oscillating between
        // DefensiveTransition/AttackingTransition — measured this same
        // investigation), long before the physical line has had any
        // chance to actually walk anywhere. A defender who hadn't moved
        // at all was being flagged as badly out of position and kicked
        // into Running (an explicitly chase-labelled state) purely
        // because the abstract target moved, not because they had.
        // Measuring against the same target `velocity()` already
        // computes removes the mismatch by construction — the gate and
        // the actual movement now agree on what "in position" means.
        let ball_position = ctx.tick_context.positions.ball.position;
        let zonal_target = self.calculate_zonal_position(ctx, ball_position);
        let distance_from_line = (ctx.player.position.x - zonal_target.x).abs();

        // 3. If the defender is too far from the defensive line, switch to Running state
        if distance_from_line > MAX_DEFENSIVE_LINE_DEVIATION {
            // TEMPORARY diagnostic (docs/positional-play-master-design.md
            // Stage A): how often does the deviation check itself fire,
            // and by how much. Removed once the investigation concludes.
            crate::match_log_info!(
                "HLDIAG_DEVIATION dist_from_line={:.1} zonal_target_x={:.1} my_x={:.1} start_x={:.1}",
                distance_from_line,
                zonal_target.x,
                ctx.player.position.x,
                ctx.player.start_position.x
            );
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Running,
            ));
        }
        crate::match_log_info!("HLDIAG_PASS_DEVIATION");

        // Loose ball nearby — go claim it directly
        if !ctx.ball().is_owned() && ctx.ball().distance() < 40.0 && ctx.ball().speed() < 3.0 {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::TakeBall,
            ));
        }

        // Role-driven engagement: if the opponent has the ball, break
        // from the line according to our defensive role. A counter-press
        // window widens the trigger so even distant defenders commit to
        // chasing just after we lose possession.
        if let Some(opponent_with_ball) = ctx.players().opponents().with_ball().next() {
            let distance = opponent_with_ball.distance(ctx);

            // Tackle range — but only if I'm the Primary closer than
            // any teammate. Otherwise I hold the line while the closer
            // defender engages. Stops the whole back four lunging at
            // the same carrier.
            let is_primary = matches!(
                ctx.player().defensive().defensive_role_for_ball_carrier_unified(),
                DefensiveRole::Primary
            );
            if ctx.player().defensive().can_engage_tackle(distance) && is_primary {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Tackling,
                ));
            }

            let counter_press_active = ctx.team().counterpress_window();
            let counter_press_range = if counter_press_active {
                35.0 + ctx.team().press_intensity() * 55.0
            } else {
                0.0
            };

            match ctx.player().defensive().defensive_role_for_ball_carrier_unified() {
                DefensiveRole::Primary => {
                    if distance < PRESSING_DISTANCE_THRESHOLD
                        || (counter_press_active && ctx.ball().distance() < counter_press_range)
                    {
                        return Some(StateChangeResult::with_defender_state(
                            DefenderState::Pressing,
                        ));
                    }
                    // Primary but out of range — chase via Running so we
                    // can close the gap instead of holding the line.
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Running,
                    ));
                }
                DefensiveRole::Cover => {
                    if distance < 100.0 && ctx.ball().on_own_side() {
                        return Some(StateChangeResult::with_defender_state(
                            DefenderState::Covering,
                        ));
                    }
                }
                DefensiveRole::Help => {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Marking,
                    ));
                }
                DefensiveRole::Hold => {
                    // Stay on the line — fall through to run/guard checks
                    // for secondary threats below.
                }
            }
        }

        // Break line to track dangerous runners if we're the best
        // positioned defender for them (no ball carrier scenario, or
        // our role was Hold).
        //
        // docs/positional-play-master-design.md Stage A: this used to
        // call this file's own PRIVATE, orphaned duplicate of
        // `scan_for_dangerous_runs` (below), which still carried the
        // stale `DANGEROUS_RUN_SPEED = 2.0` — a threshold the shared
        // copy in `defensive.rs` was already found and fixed (2026-08-27)
        // to be unreachable at this engine's real velocity scale (a
        // fully-fit elite player tops out ~0.36-0.6 u/tick). That fix
        // never propagated to this shadow copy, so this branch has been
        // silently dead for any real running pace since it was written —
        // a Hold-role defender could never actually break to track a
        // genuine dangerous run here. Switched to the shared, already-
        // corrected implementation (also gains its `is_opponent_being_
        // engaged` de-duplication, which this file's own copy lacked).
        if let Some(dangerous_runner) = ctx.player().defensive().scan_for_dangerous_runs() {
            let distance_to_runner = dangerous_runner.distance(ctx);
            if distance_to_runner < 25.0 {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Marking,
                ));
            }
        }

        // Guard unmarked attackers in our zone who are trying to get open
        if ctx.ball().on_own_side() {
            if let Some(unmarked) = ctx
                .player()
                .defensive()
                .find_unmarked_opponent(MARKING_DISTANCE_THRESHOLD * 2.0)
            {
                let dist = unmarked.distance(ctx);
                if dist < 60.0 {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Guarding,
                    ));
                }
            }
        }

        // React to balls played behind the defensive line (through balls)
        if !ctx.ball().is_owned() && ctx.ball().speed() > 1.0 {
            let ball_pos = ctx.tick_context.positions.ball.position;
            let ball_vel = ctx.tick_context.positions.ball.velocity;
            let own_goal = ctx.ball().direction_to_own_goal();
            let to_goal = (own_goal - ball_pos).normalize();
            let ball_dir = ball_vel.normalize();
            let heading_toward_goal = ball_dir.dot(&to_goal);

            // Ball is moving toward our goal and is close enough to react
            if heading_toward_goal > 0.4 && ctx.ball().distance() < 200.0 {
                // Check if ball is behind us or at our level (through ball)
                let is_behind_or_level = if own_goal.x < ctx.context.field_size.width as f32 / 2.0 {
                    ball_pos.x < ctx.player.position.x + 15.0
                } else {
                    ball_pos.x > ctx.player.position.x - 15.0
                };

                // docs/defensive-delay-and-engagement-timing.md,
                // "single-defender-exclusivity fix for Intercepting":
                // this check used to be purely local (is the ball
                // heading toward MY goal, am I behind-or-level with
                // it) with zero teammate coordination -- every
                // defender watching the same through ball independently
                // satisfied it on the same tick, producing "all four
                // defenders placed very high as if all of them were
                // trying to intercept the pass". Only the genuinely
                // fastest defender-group teammate to the ball's own
                // point now breaks; the rest fall through and hold the
                // line via this state's own normal zonal behavior.
                if is_behind_or_level
                    && ctx.player().defensive().is_best_positioned_to_intercept(ball_pos)
                {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Intercepting,
                    ));
                }
            }
        }

        // docs/positional-play-master-design.md Part 7/Stage A: both of
        // these fired on raw ball distance/heading regardless of
        // possession — a Cover/Hold-role defender got yanked into
        // Marking/Intercepting the instant the ball was merely near
        // THEM, even while a teammate held the ball (no threat at all)
        // or while the role block above had already judged this
        // defender shouldn't engage a live opponent carrier. Real
        // defenders don't scramble at their own team's circulation, and
        // a role verdict that already said "hold" shouldn't be
        // re-litigated by a blind proximity check two lines later.
        // Gated to a genuinely LOOSE ball (no owner at all) — the one
        // case where "the ball entered my zone" is a real, role-
        // independent trigger (a rebound, a loose ball nobody has
        // claimed yet), matching Pavel's own "the ball entering their
        // zone" as a legitimate break-shape trigger, distinct from
        // "the ball merely exists somewhere on the pitch".
        if !ctx.ball().is_owned() {
            // docs/defensive-delay-and-engagement-timing.md,
            // "single-defender-exclusivity fix for Intercepting": both
            // Intercepting outcomes below used to fire for every
            // defender who individually satisfied their own local
            // distance/angle check against the same loose ball, with no
            // teammate coordination -- the same "all four defenders
            // step up together" bug as the through-ball branch above.
            // Gated on the same exclusivity check; the Marking outcome
            // (guarding a DIFFERENT, specific opponent) is untouched --
            // that's a distinct decision, not a race to the loose ball.
            let loose_ball_pos = ctx.tick_context.positions.ball.position;
            let i_am_best_positioned =
                ctx.player().defensive().is_best_positioned_to_intercept(loose_ball_pos);

            if ctx.ball().distance() < 250.0
                && ctx.ball().is_towards_player_with_angle(0.8)
                && i_am_best_positioned
            {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Intercepting,
                ));
            }

            if ctx.ball().distance() < BALL_PROXIMITY_THRESHOLD {
                let opponent_nearby = self.is_opponent_nearby(ctx);
                if opponent_nearby {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Marking,
                    ));
                }
                if i_am_best_positioned {
                    return Some(StateChangeResult::with_defender_state(
                        DefenderState::Intercepting,
                    ));
                }
                // Not the best-positioned defender and no opponent to
                // guard here -- fall through and hold the line.
            }
        }

        // Offside-trap detection used to transition into a dedicated
        // DefenderState::OffsideTrap that was a pass-through — it just
        // bounced back to HoldingLine. Staying in HoldingLine with the
        // same zonal-line logic is the simpler model; if we want trap
        // pressing later, reintroduce as a team-level flag (Phase 2).
        crate::match_log_info!("HLDIAG_HOLD");
        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let current_position = ctx.player.position;
        let ball_position = ctx.tick_context.positions.ball.position;

        // Calculate target position based on zonal coverage
        let target_position = self.calculate_zonal_position(ctx, ball_position);

        let to_target = target_position - current_position;
        let distance = to_target.magnitude();

        // Define thresholds for movement
        const MIN_DISTANCE_THRESHOLD: f32 = 1.0;
        const SLOWING_DISTANCE: f32 = 5.0;

        // Base movement speed — line_holding_mult bakes condition,
        // concentration curve, and fatigue into a 0.75..1.03 multiplier.
        // Replaces the raw pace/20 clamp so a tired CB shuffles slower
        // and a fresh, concentrated CB holds the line crisply.
        let def_profile = DefenderSkillProfile::from_ctx(ctx);
        let pace_influence = (ctx.player.skills.physical.pace / 20.0).clamp(0.6, 1.2);
        let base_speed = 3.0 * pace_influence * def_profile.line_holding_mult;

        if distance > MIN_DISTANCE_THRESHOLD {
            let direction = to_target.normalize();

            // Speed factor - slow down as approaching target
            let speed_factor = if distance > SLOWING_DISTANCE {
                1.0
            } else {
                (distance / SLOWING_DISTANCE).clamp(0.25, 1.0)
            };

            Some(direction * base_speed * speed_factor)
        } else {
            // In position - stay still (no artificial jitter)
            Some(Vector3::new(0.0, 0.0, 0.0))
        }
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Holding line involves minimal movement - allows for recovery
        DefenderCondition::with_velocity(ActivityIntensity::Recovery).process(ctx);
    }
}

impl DefenderHoldingLineState {
    /// Calculate zonal defensive position - creates natural staggered formation
    /// Each defender positions based on their assigned opponent or zone
    fn calculate_zonal_position(
        &self,
        ctx: &StateProcessingContext,
        ball_position: Vector3<f32>,
    ) -> Vector3<f32> {
        let tactical_position = ctx.player.start_position;
        let current_position = ctx.player.position;
        let own_goal = ctx.ball().direction_to_own_goal();
        let field_center_y = ctx.context.field_size.height as f32 / 2.0;

        // Determine if this defender is a wide defender (fullback) or central
        let distance_from_center = (tactical_position.y - field_center_y).abs();
        let is_wide_defender = distance_from_center > 40.0;

        // Find the nearest opponent in this defender's zone
        let zone_half_width = 50.0;
        let nearest_opponent_in_zone = ctx
            .players()
            .opponents()
            .nearby(100.0)
            .filter(|opp| {
                let lateral_dist = (opp.position.y - tactical_position.y).abs();
                lateral_dist < zone_half_width
            })
            .min_by(|a, b| {
                let dist_a = (a.position - current_position).magnitude();
                let dist_b = (b.position - current_position).magnitude();
                dist_a.total_cmp(&dist_b)
            });

        // DEPTH (X) CALCULATION
        let target_x = if let Some(opponent) = nearest_opponent_in_zone {
            // Track the opponent - position between them and goal
            let opponent_x = opponent.position.x;
            let goal_x = own_goal.x;

            // Position goal-side of opponent (between opponent and goal)
            let goal_direction = (goal_x - opponent_x).signum();
            let marking_offset = 8.0 * goal_direction; // Stay 8 units goal-side

            opponent_x + marking_offset
        } else {
            // No opponent in zone - use base tactical position with ball influence
            let ball_influence = (ball_position.x - tactical_position.x) * 0.2;

            // Wide defenders push up more when ball is on their side
            let wide_push = if is_wide_defender {
                let ball_on_my_side = (ball_position.y - field_center_y).signum()
                    == (tactical_position.y - field_center_y).signum();
                if ball_on_my_side { 15.0 } else { -5.0 }
            } else {
                0.0
            };

            tactical_position.x + ball_influence + wide_push
        };

        // LATERAL (Y) CALCULATION
        // compactness_target is the team-shared signal (tactic + phase +
        // game-management). Polish spec: shift = ball_offset * (0.08 +
        // compactness_target * 0.18) so a fully-compact low-block
        // squeezes harder toward the ball side than the legacy 0.12 cap.
        let compactness = ctx.team().compactness_target();

        let target_y = if let Some(opponent) = nearest_opponent_in_zone {
            // Track opponent laterally but don't go too far from zone
            let opponent_y = opponent.position.y;
            let max_drift = 25.0;
            let drift = (opponent_y - tactical_position.y).clamp(-max_drift, max_drift);
            tactical_position.y + drift
        } else {
            let ball_offset = ball_position.y - field_center_y;
            let shift = ball_offset * (0.08 + compactness * 0.18);
            tactical_position.y + shift
        };

        Vector3::new(target_x, target_y, 0.0)
    }

    /// Checks if an opponent player is nearby within the MARKING_DISTANCE_THRESHOLD.
    fn is_opponent_nearby(&self, ctx: &StateProcessingContext) -> bool {
        ctx.players().opponents().exists(MARKING_DISTANCE_THRESHOLD)
    }
}
