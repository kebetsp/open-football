use crate::TacticalStyle;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::player::strategies::common::players::ops::on_ball_value;
use crate::r#match::player::strategies::spacing;
use crate::r#match::forwarders::states::common::{ActivityIntensity, ForwardCondition};
use crate::r#match::player::strategies::players::skills::SkillCurve;
use crate::r#match::{
    ConditionContext, MatchPlayerLite, PlayerSide, StateChangeResult, StateProcessingContext,
    StateProcessingHandler, SteeringBehavior,
};

use nalgebra::Vector3;

const MAX_DISTANCE_FROM_BALL: f32 = 80.0;
const MIN_DISTANCE_FROM_BALL: f32 = 30.0;
const OPTIMAL_PASSING_DISTANCE_MIN: f32 = 20.0;
const OPTIMAL_PASSING_DISTANCE_MAX: f32 = 70.0;
const SPACE_SCAN_RADIUS: f32 = 250.0;
const PASSING_LANE_IMPORTANCE: f32 = 15.0; // High weight for clear passing lanes

/// Pressure-relief support ("show for the ball" — wishlist item
/// "pressure-sensitive spread distance"). Same mechanism and same tens-
/// scale reasoning as `spacing.rs`'s equivalent constants (the shared
/// off-ball scorer used by midfielders/defenders) — forwards keep their
/// own independently-tuned zone scorer, but the pressure logic itself
/// should behave the same way everywhere it applies.
const PRESSURE_RELIEF_WEIGHT: f32 = 30.0;
const PRESSURE_RELIEF_RADIUS: f32 = 45.0;

#[derive(Default, Clone)]
pub struct ForwardCreatingSpaceState {}

impl StateProcessingHandler for ForwardCreatingSpaceState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        // Check if player has the ball
        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        // Take ball only if best positioned — prevents swarming
        if ctx.ball().should_take_ball_immediately() && ctx.team().is_best_player_to_chase_ball() {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::TakeBall,
            ));
        }

        // Check if team lost possession
        if !ctx.team().is_control_ball() {
            return Some(StateChangeResult::with_forward_state(ForwardState::Running));
        }

        // If ball is close and moving toward player
        if ctx.ball().distance() < 100.0 && ctx.ball().is_towards_player_with_angle(0.8) {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Intercepting,
            ));
        }

        // Check if created good space
        if self.has_created_good_space(ctx) {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::Assisting,
            ));
        }

        // Check for forward run opportunity
        if self.should_make_forward_run(ctx) {
            return Some(StateChangeResult::with_forward_state(
                ForwardState::RunningInBehind,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Scoring-based zone finder: evaluates passing lane quality, goal angle,
        // offside avoidance, and ball-holder distance across ~40 candidates.
        // Works during build-up and final third alike — the scoring naturally
        // trades off goal threat vs. availability to receive. Tactical style
        // adjustment (Possession → come shorter, Attacking → push higher,
        // WidePlay → wider) is applied inside find_optimal_free_zone.
        let target = self.find_optimal_free_zone(ctx);
        let dist = (target - ctx.player.position).magnitude();

        if dist < 8.0 {
            return Some(Vector3::zeros());
        }

        Some(
            SteeringBehavior::Arrive {
                target,
                slowing_distance: 20.0,
            }
            .calculate(ctx.player)
            .velocity,
        )
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Creating space is moderate intensity - tactical movement
        ForwardCondition::with_velocity(ActivityIntensity::Moderate).process(ctx);
    }
}

impl ForwardCreatingSpaceState {
    /// Find optimal free zone for a forward
    /// Find optimal free zone for a forward - optimized to search gaps between opponents
    fn find_optimal_free_zone(&self, ctx: &StateProcessingContext) -> Vector3<f32> {
        let field_width = ctx.context.field_size.width as f32;
        let field_height = ctx.context.field_size.height as f32;
        let player_pos = ctx.player.position;
        let goal_pos = ctx.player().opponent_goal_position();

        // Pre-collect ALL opponent positions once for reuse in scoring
        let all_opponent_positions: Vec<Vector3<f32>> = ctx
            .players()
            .opponents()
            .all()
            .map(|opp| opp.position)
            .collect();

        // Collect relevant nearby opponents for gap-finding
        const SPACE_SCAN_RADIUS_SQ: f32 = SPACE_SCAN_RADIUS * SPACE_SCAN_RADIUS;
        let opponents: Vec<Vector3<f32>> = all_opponent_positions
            .iter()
            .filter(|&&pos| (pos - player_pos).norm_squared() < SPACE_SCAN_RADIUS_SQ)
            .copied()
            .collect();

        // If no nearby opponents, move toward goal
        if opponents.is_empty() {
            let forward_direction = (goal_pos - player_pos).normalize();
            return self
                .apply_forward_tactical_adjustment(ctx, player_pos + forward_direction * 30.0);
        }

        // Pre-compute values used in scoring
        let ball_holder = self.get_ball_holder(ctx);
        let attacking_direction = self.get_attacking_direction(ctx);
        let is_attacking_left = attacking_direction.x > 0.0;

        // Pre-compute defensive line for offside checks. realism-bug
        // (offside investigation): this used to fold over
        // `all_opponent_positions` unfiltered, which includes the GK —
        // since the GK is normally the single deepest opponent, that
        // made this "line" almost always equal to the GK's own depth
        // (far too permissive; a candidate standing well beyond the
        // real last outfield defender still scored as onside). Uses
        // the shared, GK-excluding `find_defensive_line` instead.
        let last_defender_x = ctx.player().defensive().find_defensive_line();

        // Find gaps between opponents using improved multi-strategy approach
        let mut candidate_positions = Vec::with_capacity(40);

        // Strategy 1: Midpoints between adjacent opponents
        for i in 0..opponents.len() {
            for j in (i + 1)..opponents.len() {
                let midpoint = (opponents[i] + opponents[j]) * 0.5;
                let gap_width = (opponents[i] - opponents[j]).magnitude();

                if gap_width > 12.0 && gap_width < 80.0 {
                    candidate_positions.push(midpoint);
                    let to_goal = (goal_pos - midpoint).normalize();
                    candidate_positions.push(midpoint + to_goal * 10.0);
                }
            }
        }

        // Strategy 2: Positions offset from opponents
        for &opp_pos in &opponents {
            let to_goal = (goal_pos - opp_pos).normalize();
            let perpendicular = Vector3::new(-to_goal.y, to_goal.x, 0.0);
            candidate_positions.push(opp_pos + perpendicular * 25.0 + to_goal * 20.0);
            candidate_positions.push(opp_pos - perpendicular * 25.0 + to_goal * 20.0);
            candidate_positions.push(opp_pos + to_goal * 15.0);
        }

        // Strategy 3: Grid-based open space detection — wide field scan
        let forward_direction = (goal_pos - player_pos).normalize();
        for x_offset in [25.0, 50.0, 80.0, 120.0] {
            for y_offset in [-100.0, -60.0, -30.0, 0.0, 30.0, 60.0, 100.0] {
                let lateral = Vector3::new(-forward_direction.y, forward_direction.x, 0.0);
                let candidate = player_pos + forward_direction * x_offset + lateral * y_offset;
                candidate_positions.push(candidate);
            }
        }

        // Strategy 4: Wide channel positions (flanks and half-spaces)
        let ball_pos = ctx.tick_context.positions.ball.position;
        let atk_dir = self.get_attacking_direction(ctx);
        for &wing_y in &[
            field_height * 0.10,
            field_height * 0.25,
            field_height * 0.75,
            field_height * 0.90,
        ] {
            for &fwd in &[40.0, 80.0, 120.0] {
                let x = (ball_pos.x + atk_dir.x * fwd).clamp(30.0, field_width - 30.0);
                candidate_positions.push(Vector3::new(x, wing_y, 0.0));
            }
        }

        candidate_positions.push(player_pos);

        // §11.9 hard exclusion: a zone the carrier is running into, or
        // one another teammate already occupies/targets, is not a free
        // zone — drop it before scoring instead of merely down-scoring
        // (the separation penalty alone did not stop actual overlap).
        let claimed = spacing::claimed_points(ctx);

        // Evaluate candidates using pre-collected data
        let mut best_position = player_pos;
        let mut best_score = f32::MIN;
        let mut best_any = player_pos;
        let mut best_any_score = f32::MIN;
        let mut found_valid = false;

        for candidate in candidate_positions {
            let clamped = Vector3::new(
                candidate.x.clamp(20.0, field_width - 20.0),
                candidate.y.clamp(20.0, field_height - 20.0),
                0.0,
            );

            let score = self.evaluate_forward_position_fast(
                ctx,
                clamped,
                &all_opponent_positions,
                &ball_holder,
                last_defender_x,
                is_attacking_left,
            );

            if score > best_any_score {
                best_any_score = score;
                best_any = clamped;
            }
            if spacing::violates_exclusion(clamped, &claimed) {
                continue;
            }
            found_valid = true;
            if score > best_score {
                best_score = score;
                best_position = clamped;
            }
        }

        // All ~40 candidates claimed (packed box) — fall back to the best
        // scorer rather than standing still.
        if !found_valid {
            best_position = best_any;
        }

        self.apply_forward_tactical_adjustment(ctx, best_position)
    }

    /// Fast position evaluation using pre-collected opponent data
    fn evaluate_forward_position_fast(
        &self,
        ctx: &StateProcessingContext,
        position: Vector3<f32>,
        all_opponents: &[Vector3<f32>],
        ball_holder: &Option<MatchPlayerLite>,
        last_defender_x: f32,
        is_attacking_left: bool,
    ) -> f32 {
        let mut score = 0.0;
        let goal_pos = ctx.player().opponent_goal_position();
        let distance_to_goal = (position - goal_pos).magnitude();

        // Space score using pre-collected opponents
        let mut congestion = 0.0f32;
        for &opp_pos in all_opponents {
            let distance = (opp_pos - position).magnitude();
            if distance < 30.0 {
                congestion += (30.0 - distance) / 30.0;
            }
        }
        score += (10.0 - congestion.min(10.0)) * 3.0;

        // Goal threat score
        let goal_threat = if distance_to_goal < 15.0 {
            8.0
        } else if distance_to_goal < 25.0 {
            10.0
        } else if distance_to_goal < 35.0 {
            6.0
        } else {
            (100.0 - distance_to_goal).max(0.0) / 20.0
        };
        score += goal_threat * 6.0;

        // PRD: attacker-angle-seeking-and-gk-drag (Option B completion,
        // Milestone 2). Real occluded-angle geometry — same
        // `effective_open_angle` the carrier's own shot decision and
        // carry-value already use — instead of the goal-threat/channel
        // terms above, which are pure distance/central-corridor proxies
        // with no angle awareness at all. Additive, not a replacement:
        // this project's calibration history (§10.3/§11.9/GK-proximity
        // tuning already baked into this same function) makes deleting
        // or down-weighting the existing terms a real regression risk,
        // so a genuinely wide, angle-exploiting position now competes
        // for the argmax rather than being silently outscored by the
        // central-channel bonus in every case. Scaled to ~0-25, the same
        // order of magnitude as the channel/width bonuses it competes
        // against. Uses the keeper's actual (not projected) position —
        // this is "how good is standing here right now", not the
        // carrier's own forward-looking carry decision.
        if distance_to_goal < 250.0 {
            let gk_pos = ctx
                .players()
                .opponents()
                .goalkeeper()
                .next()
                .map(|g| g.position);
            let angle_score =
                (on_ball_value::effective_open_angle(ctx, position, gk_pos) / 1.31).clamp(0.0, 1.0);
            score += angle_score * 25.0;
        }

        // Box area bonus
        if distance_to_goal < 180.0 {
            score += 30.0;
        } else if distance_to_goal < 250.0 {
            score += 20.0;
        }

        // Offside check using pre-computed defensive line
        let margin = crate::r#match::player::strategies::common::players::ops::defensive::DefensiveOperationsImpl::OFFSIDE_HOLD_MARGIN;
        let is_offside = if is_attacking_left {
            position.x > last_defender_x + margin
        } else {
            position.x < last_defender_x - margin
        };
        if !is_offside {
            score += 15.0;
        } else {
            score -= 50.0;
        }

        // Channel positioning
        let field_height = ctx.context.field_size.height as f32;
        let channel_width = field_height / 5.0;
        let center = field_height / 2.0;
        if (position.y - center).abs() < channel_width * 1.5 {
            score += 20.0;
            if distance_to_goal < 300.0 {
                score += 15.0;
            }
        }

        // Behind defensive line using pre-computed data
        let avg_defender_x =
            all_opponents.iter().map(|p| p.x).sum::<f32>() / all_opponents.len().max(1) as f32;
        let is_behind = if is_attacking_left {
            position.x > avg_defender_x
        } else {
            position.x < avg_defender_x
        };
        if is_behind {
            score += 30.0;
        }

        // Ball holder awareness
        if let Some(holder) = ball_holder {
            let holder_distance = (position - holder.position).magnitude();

            // Pressure-relief support ("show for the ball" — wishlist
            // "pressure-sensitive spread distance"): how surrounded the
            // HOLDER currently is. Uses the bucketed-distance
            // `pressure_intensity_for` (needs ≥3 opponents within 10u to
            // saturate) rather than `on_ball_value::congestion_risk`
            // (designed as a mild risk-COST term, saturates from a
            // single opponent at ~24u — too coarse a trigger for
            // "genuinely surrounded, needs help"; measured via a
            // match-logs trace against `spacing.rs`'s identical block).
            // 0.0 when the holder has time and space.
            let holder_pressure = ctx.player().pressure().pressure_intensity_for(holder.id);

            // Engagement, not raw pressure: mild pressure is the most
            // common reading in ordinary play, and nudging scores at
            // that level measurably suppressed goals in `spacing.rs`'s
            // identical mechanism (regression-gate isolation, ~3.67 vs
            // ~2.89 goals/match across two batch pairs) — below the
            // floor this whole block is byte-for-byte inert.
            const PRESSURE_ENGAGEMENT_FLOOR: f32 = 0.5;
            let relief_engagement = ((holder_pressure - PRESSURE_ENGAGEMENT_FLOOR)
                / (1.0 - PRESSURE_ENGAGEMENT_FLOOR))
                .clamp(0.0, 1.0);

            if holder_distance >= OPTIMAL_PASSING_DISTANCE_MIN
                && holder_distance <= OPTIMAL_PASSING_DISTANCE_MAX
            {
                score += 25.0;
            } else if holder_distance < OPTIMAL_PASSING_DISTANCE_MIN {
                // Strong penalty for being too close — prevents
                // clustering. Relaxed under genuine pressure: a close
                // option stops being a redundant trivial pass and
                // becomes a genuine escape outlet once the holder is
                // actually surrounded.
                score -= (OPTIMAL_PASSING_DISTANCE_MIN - holder_distance)
                    * 1.5
                    * (1.0 - relief_engagement);
            } else if holder_distance > OPTIMAL_PASSING_DISTANCE_MAX {
                score -= (holder_distance - OPTIMAL_PASSING_DISTANCE_MAX) * 0.5;
            }

            // Pressure-relief bonus: reward a close, genuinely CLEAN
            // outlet specifically when the holder needs one — gated on
            // both the holder's pressure and the candidate spot itself
            // being clear to receive at (`congestion`, already computed
            // above for the space-score term). Additive, doesn't gate
            // the angle/box/width terms below. Zero below the engagement
            // floor — ordinary unpressed forward play is untouched.
            if relief_engagement > 0.0 && holder_distance < PRESSURE_RELIEF_RADIUS {
                let closeness = ((PRESSURE_RELIEF_RADIUS - holder_distance)
                    / PRESSURE_RELIEF_RADIUS)
                    .clamp(0.0, 1.0);
                let candidate_openness = (1.0 - congestion.min(10.0) / 10.0).clamp(0.0, 1.0);
                score +=
                    relief_engagement * closeness * candidate_openness * PRESSURE_RELIEF_WEIGHT;
            }

            // WIDTH BONUS: reward lateral distance from ball holder
            // Forwards that provide width are much more useful
            let lateral_distance = (position.y - holder.position.y).abs();
            if lateral_distance > 80.0 {
                score += 30.0; // Excellent width
            } else if lateral_distance > 50.0 {
                score += 20.0; // Good width
            } else if lateral_distance > 30.0 {
                score += 10.0; // Moderate width
            } else if lateral_distance < 15.0 {
                score -= 15.0; // Too narrow — on same channel as holder
            }

            // Clear passing lane check using pre-collected opponents
            let direction = (position - holder.position).normalize();
            let distance = (position - holder.position).magnitude();
            let lane_blocked = all_opponents.iter().any(|&opp_pos| {
                let to_opp = opp_pos - holder.position;
                let projection = to_opp.dot(&direction);
                if projection <= 0.0 || projection >= distance {
                    return false;
                }
                let projected_point = holder.position + direction * projection;
                (opp_pos - projected_point).norm_squared() < 4.0 * 4.0
            });

            if !lane_blocked {
                score += PASSING_LANE_IMPORTANCE;
            } else {
                score -= 10.0;
            }
        }

        // Universal teammate repulsion — crowding any teammate is wasteful
        // regardless of role. At 90u radius with a linear penalty, a player
        // 45u away costs ~25 pts; within 10u it costs ~44 pts.
        let teammate_crowd_penalty: f32 = ctx
            .players()
            .teammates()
            .all()
            .map(|t| (t.position - position).magnitude())
            .filter(|&d| d < 90.0)
            .map(|d| (90.0 - d) / 90.0)
            .sum::<f32>();
        score -= teammate_crowd_penalty * 50.0;

        // GK proximity penalty — the keeper physically dominates the
        // 6-yard area. Standing next to them without the ball is not
        // a useful attacking position. Strong enough to overcome the
        // goal-threat and box-area bonuses that otherwise pull forwards
        // toward the goal mouth (and the keeper standing in it).
        if let Some(gk) = ctx.players().opponents().goalkeeper().next() {
            let gk_dist = (gk.position - position).magnitude();
            if gk_dist < 12.0 {
                score -= 100.0 + (12.0 - gk_dist) * 6.0; // −100 to −172
            } else if gk_dist < 25.0 {
                score -= (25.0 - gk_dist) * 3.0; // 0 to −39
            }
        }

        score
    }

    // Helper methods

    fn get_attacking_direction(&self, ctx: &StateProcessingContext) -> Vector3<f32> {
        match ctx.player.side {
            Some(PlayerSide::Left) => Vector3::new(1.0, 0.0, 0.0),
            Some(PlayerSide::Right) => Vector3::new(-1.0, 0.0, 0.0),
            None => Vector3::new(1.0, 0.0, 0.0),
        }
    }

    /// realism-bug (offside investigation): delegates to the shared
    /// `find_defensive_line`/`OFFSIDE_HOLD_MARGIN` instead of its own
    /// `is_defender()`-only line (which missed a deep-sitting
    /// midfielder or wing-back) and its own separate 2.0 tolerance.
    fn would_be_offside(&self, ctx: &StateProcessingContext, position: Vector3<f32>) -> bool {
        let attacking_direction = self.get_attacking_direction(ctx);
        let is_attacking_left = attacking_direction.x > 0.0;
        let defensive = ctx.player().defensive();
        let line = defensive.find_defensive_line();
        let margin = crate::r#match::player::strategies::common::players::ops::defensive::DefensiveOperationsImpl::OFFSIDE_HOLD_MARGIN;

        if is_attacking_left {
            position.x > line + margin
        } else {
            position.x < line - margin
        }
    }

    fn would_be_offside_now(&self, ctx: &StateProcessingContext) -> bool {
        self.would_be_offside(ctx, ctx.player.position)
    }

    fn get_defensive_line_height(&self, ctx: &StateProcessingContext) -> f32 {
        let field_width = ctx.context.field_size.width as f32;
        let (sum_x, count) = ctx
            .players()
            .opponents()
            .all()
            .filter(|p| p.tactical_positions.is_defender())
            .fold((0.0_f32, 0u32), |(s, c), p| (s + p.position.x, c + 1));

        if count == 0 {
            return 0.5;
        }

        let avg_x = sum_x / count as f32;
        avg_x / field_width
    }

    fn has_space_behind_defense(&self, ctx: &StateProcessingContext) -> bool {
        let defensive_line = self.get_defensive_line_height(ctx);
        let field_width = ctx.context.field_size.width as f32;
        let attacking_direction = self.get_attacking_direction(ctx);

        if attacking_direction.x > 0.0 {
            defensive_line < 0.7 && (field_width - defensive_line * field_width) > 30.0
        } else {
            defensive_line > 0.3 && (defensive_line * field_width) > 30.0
        }
    }

    fn has_clear_passing_lane(
        &self,
        from: Vector3<f32>,
        to: Vector3<f32>,
        ctx: &StateProcessingContext,
    ) -> bool {
        let direction = (to - from).normalize();
        let distance = (to - from).magnitude();

        // Pre-filter: only check opponents near the player (within pass distance + margin)
        !ctx.players()
            .opponents()
            .nearby(distance + 10.0)
            .any(|opp| {
                let to_opp = opp.position - from;
                let projection = to_opp.dot(&direction);

                if projection <= 0.0 || projection >= distance {
                    return false;
                }

                let projected_point = from + direction * projection;
                let perp_distance = (opp.position - projected_point).magnitude();

                perp_distance < 4.0
            })
    }

    fn apply_forward_tactical_adjustment(
        &self,
        ctx: &StateProcessingContext,
        mut position: Vector3<f32>,
    ) -> Vector3<f32> {
        // Get player's team tactics
        let player_tactics = match ctx.player.side {
            Some(PlayerSide::Left) => &ctx.context.tactics.left,
            Some(PlayerSide::Right) => &ctx.context.tactics.right,
            None => return position,
        };

        // Adjust based on tactical style
        match player_tactics.tactical_style() {
            TacticalStyle::Attacking => {
                // Push higher up the pitch
                let attacking_direction = self.get_attacking_direction(ctx);
                position += attacking_direction * 10.0;
            }
            TacticalStyle::Counterattack => {
                // Stay ready to exploit space
                if self.has_space_behind_defense(ctx) {
                    let attacking_direction = self.get_attacking_direction(ctx);
                    position += attacking_direction * 15.0;
                }
            }
            TacticalStyle::WidePlay | TacticalStyle::WingPlay => {
                // Push wider
                let field_height = ctx.context.field_size.height as f32;
                if position.y < field_height / 2.0 {
                    position.y = (position.y - 10.0).max(10.0);
                } else {
                    position.y = (position.y + 10.0).min(field_height - 10.0);
                }
            }
            TacticalStyle::Possession => {
                // Come shorter to help build play
                let ball_pos = ctx.tick_context.positions.ball.position;
                let to_ball = (ball_pos - position).normalize();
                position += to_ball * 5.0;
            }
            _ => {}
        }

        // Ensure within bounds
        let field_width = ctx.context.field_size.width as f32;
        let field_height = ctx.context.field_size.height as f32;
        position.x = position.x.clamp(10.0, field_width - 10.0);
        position.y = position.y.clamp(10.0, field_height - 10.0);

        position
    }

    // Keep existing helper methods for compatibility
    fn has_created_good_space(&self, ctx: &StateProcessingContext) -> bool {
        // Reduced strictness - forwards should transition to assisting more often
        let space_created = !ctx.players().opponents().exists(12.0); // Reduced from 20.0
        let in_support_position = self.is_in_good_support_position(ctx);
        let has_clear_lane = self.has_clear_passing_lane_from_ball_holder(ctx);
        let minimum_time_in_state = 15; // Reduced from 30
        let reasonable_distance = ctx.ball().distance() < MAX_DISTANCE_FROM_BALL;

        // More lenient check - any 3 of the 4 conditions is enough
        let conditions_met = [
            space_created,
            in_support_position,
            has_clear_lane,
            reasonable_distance,
        ]
        .iter()
        .filter(|&&c| c)
        .count();

        conditions_met >= 3 && ctx.in_state_time > minimum_time_in_state
    }

    fn should_make_forward_run(&self, ctx: &StateProcessingContext) -> bool {
        if !ctx.team().is_control_ball() {
            return false;
        }

        let ball_holder_can_pass = self.ball_holder_can_make_forward_pass(ctx);
        let not_offside = !self.would_be_offside_now(ctx);
        let in_good_phase = self.is_in_good_attacking_phase(ctx);
        let not_too_far = ctx.ball().distance() < MAX_DISTANCE_FROM_BALL;

        if !ball_holder_can_pass || !not_offside || !in_good_phase || !not_too_far {
            return false;
        }

        // Use the SAME space check that RunningInBehind uses to decide viability.
        // This prevents oscillation: if RunningInBehind would immediately fail,
        // don't start the run.
        let player_pos = ctx.player.position;
        let goal_pos = ctx.player().opponent_goal_position();
        let to_goal = (goal_pos - player_pos).normalize();

        let blockers = ctx
            .players()
            .opponents()
            .nearby(30.0)
            .filter(|opp| {
                let to_opp = (opp.position - player_pos).normalize();
                to_opp.dot(&to_goal) > 0.3
            })
            .count();

        if blockers >= 2 {
            return false;
        }
        if blockers == 1 {
            // Pace decides whether the runner can beat a single blocker.
            // Smooth sigmoid (pivot 12/20) — a slow striker (pace=6) very
            // occasionally still tries; a quick one (pace=17) almost
            // always commits.
            let p = SkillCurve::new(ctx.player.skills.physical.pace, 12.0, 0.6).probability();
            if ctx.context.rng.unit_f32() >= p {
                return false;
            }
        }

        // Also check passing lane: runner must be ahead of the passer
        if let Some(owner_id) = ctx.ball().owner_id() {
            if let Some(owner) = ctx.context.players.by_id(owner_id) {
                if owner.team_id == ctx.player.team_id {
                    let passer_pos = ctx.tick_context.positions.players.position(owner_id);
                    let to_goal_from_passer = (goal_pos - passer_pos).normalize();
                    let to_runner = (player_pos - passer_pos).normalize();
                    if to_runner.dot(&to_goal_from_passer) <= 0.0 {
                        return false; // Behind the passer — run wouldn't be viable
                    }
                }
            }
        }

        true
    }

    fn is_in_good_support_position(&self, ctx: &StateProcessingContext) -> bool {
        let ball_distance = ctx.ball().distance();
        let goal_distance = ctx.ball().distance_to_opponent_goal();

        // Standard support position check
        let in_normal_range =
            ball_distance >= MIN_DISTANCE_FROM_BALL && ball_distance <= MAX_DISTANCE_FROM_BALL;

        // Also good if in dangerous position close to goal
        let in_dangerous_area =
            goal_distance < 300.0 && ball_distance < MAX_DISTANCE_FROM_BALL + 20.0;

        // Good if in passing range from ball holder
        let in_passing_range = if let Some(holder) = self.get_ball_holder(ctx) {
            let holder_distance = (ctx.player.position - holder.position).magnitude();
            holder_distance >= OPTIMAL_PASSING_DISTANCE_MIN
                && holder_distance <= OPTIMAL_PASSING_DISTANCE_MAX
        } else {
            false
        };

        in_normal_range || in_dangerous_area || in_passing_range
    }

    fn has_clear_passing_lane_from_ball_holder(&self, ctx: &StateProcessingContext) -> bool {
        if let Some(holder) = self.get_ball_holder(ctx) {
            self.has_clear_passing_lane(holder.position, ctx.player.position, ctx)
        } else {
            true
        }
    }

    fn get_ball_holder(&self, ctx: &StateProcessingContext) -> Option<MatchPlayerLite> {
        ctx.players()
            .teammates()
            .all()
            .find(|t| ctx.ball().owner_id() == Some(t.id))
    }

    fn ball_holder_can_make_forward_pass(&self, ctx: &StateProcessingContext) -> bool {
        if let Some(holder) = self.get_ball_holder(ctx) {
            // Check if holder is under pressure
            let holder_under_pressure = ctx.tick_context.grid.opponents(holder.id, 8.0).count() > 0;

            !holder_under_pressure
        } else {
            false
        }
    }

    fn is_in_good_attacking_phase(&self, ctx: &StateProcessingContext) -> bool {
        let ball_distance_to_goal = ctx.ball().distance_to_opponent_goal();
        let field_width = ctx.context.field_size.width as f32;

        ball_distance_to_goal < field_width * 0.7
    }

}
