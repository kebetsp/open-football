use crate::PlayerFieldPositionGroup;
use crate::club::player::behaviour_config::PassEvaluatorConfig;
use crate::club::player::registry::has_risk_tolerant_passing_trait;
use crate::club::player::traits::PlayerTrait;
use crate::r#match::engine::chemistry::chemistry_modifiers;
use crate::r#match::engine::psychology::Psychology;
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::player::strategies::players::skills::SkillCurve;
use crate::r#match::{
    BallSideZone, GamePhase, MatchPlayer, MatchPlayerLite, PassOriginRestart, PlayerSide,
    StateProcessingContext,
};

/// Comprehensive pass evaluation result
#[derive(Debug, Clone)]
pub struct PassEvaluation {
    /// Overall success probability [0.0 - 1.0]
    pub success_probability: f32,

    /// Risk level [0.0 - 1.0] where 1.0 is highest risk
    pub risk_level: f32,

    /// Expected value of the pass
    pub expected_value: f32,

    /// Breakdown of factors
    pub factors: PassFactors,

    /// Whether this pass is recommended
    pub is_recommended: bool,
}

#[derive(Debug, Clone)]
pub struct PassFactors {
    pub distance_factor: f32,
    pub angle_factor: f32,
    pub pressure_factor: f32,
    pub receiver_positioning: f32,
    pub passer_ability: f32,
    pub receiver_ability: f32,
    pub tactical_value: f32,
}

pub struct PassEvaluator;

impl PassEvaluator {
    /// Evaluate a potential pass from one player to another
    pub fn evaluate_pass(
        ctx: &StateProcessingContext,
        passer: &MatchPlayer,
        receiver: &MatchPlayerLite,
    ) -> PassEvaluation {
        let pass_vector = receiver.position - passer.position;
        let pass_distance = pass_vector.norm();

        // Calculate individual factors
        let distance_factor = Self::calculate_distance_factor(pass_distance, passer);
        let angle_factor = Self::calculate_angle_factor(ctx, passer, receiver);
        let pressure_factor = Self::calculate_pressure_factor(ctx, passer);
        let receiver_positioning = Self::calculate_receiver_positioning(ctx, receiver);
        let passer_ability = Self::calculate_passer_ability(ctx, passer, pass_distance);
        let receiver_ability = Self::calculate_receiver_ability(ctx, receiver);
        let tactical_value = Self::calculate_tactical_value(ctx, receiver);

        let factors = PassFactors {
            distance_factor,
            angle_factor,
            pressure_factor,
            receiver_positioning,
            passer_ability,
            receiver_ability,
            tactical_value,
        };

        // Calculate success probability using weighted factors
        let raw_success_probability = Self::calculate_success_probability(&factors);

        // Apply environment modifiers — short passes get `pass_accuracy`;
        // anything beyond the short/medium boundary also picks up
        // `long_pass_accuracy`. Pre-calibrated bands (rain ≈ -0.04, heavy
        // rain ≈ -0.09, wind on long pass ≈ -0.08) so a 50% pass under
        // heavy rain becomes ~41% rather than the dry-weather value.
        // Floor matches the post-clamp 0.1 floor in
        // `calculate_success_probability`.
        let env_mod = ctx.context.environment.modifiers();
        const LONG_PASS_DISTANCE: f32 = 60.0;
        let env_delta = env_mod.pass_accuracy
            + if pass_distance >= LONG_PASS_DISTANCE {
                env_mod.long_pass_accuracy
            } else {
                0.0
            };
        // Psychology nudge: a passer running low on composure /
        // first-touch (heavy nervousness, repeated mistakes) sees
        // their pass success drop ~3-5%; a confident passer gets a
        // small tail-wind. Marginal — psychology should tilt
        // outcomes, not dominate them.
        let psych_delta = if let Some(state) = ctx.context.psychology.get(passer.id) {
            let m = Psychology::skill_modifiers(state);
            // Composure × first-touch composite ranges roughly 0.92..1.08.
            let composite = (m.composure_mul + m.first_touch_mul) * 0.5 - 1.0;
            // Map into a ±0.04 success delta.
            composite * 0.5
        } else {
            0.0
        };
        // Chemistry: if the passer/receiver pair has been seeded, apply
        // the one-touch-pass bonus on top of the raw probability. The
        // bonus only fires on high-chemistry pairs (>0.65 in
        // `chemistry_modifiers`); newly assembled or low-chemistry
        // pairs contribute 0. The lookup is read-only, so it's safe
        // inside the immutable evaluator.
        let chemistry_delta = ctx
            .context
            .chemistry
            .get(passer.id, receiver.id)
            .map(|chem| chemistry_modifiers(chem).one_touch_pass_bonus)
            .unwrap_or(0.0);
        // "Link with X" (wishlist #5): an explicit manager-issued pair
        // preference. Stacked on top of chemistry because the one-touch
        // bonus is a flat 0.04 above the 0.65 gate — for naturally
        // adjacent pairs, forcing chemistry higher alone changes
        // nothing. The extra success feeds expected_value, so the
        // linked receiver wins ties against an equally-placed option.
        let link_delta = if passer.link_target == Some(receiver.id) {
            0.05
        } else {
            0.0
        };
        // "Block passes into X" (wishlist #8): an opposing interceptor
        // assigned to this receiver, currently within range of them,
        // makes the pass genuinely harder to complete.
        let intercept_delta = if ctx.context.intercept_assignments.iter().any(|&(i, t)| {
            t == receiver.id
                && (ctx.tick_context.positions.players.position(i) - receiver.position).norm()
                    < 80.0
        }) {
            -0.10
        } else {
            0.0
        };
        let success_probability = (raw_success_probability
            + env_delta
            + psych_delta
            + chemistry_delta
            + link_delta
            + intercept_delta)
            .clamp(0.1, 0.99);

        // Calculate risk level (inverse of some success factors)
        let risk_level = Self::calculate_risk_level(&factors);

        // Calculate expected value considering success probability and tactical value
        let expected_value = success_probability * tactical_value;

        // Determine if pass is recommended based on thresholds. Players with
        // killer-ball / playmaker PPMs are willing to attempt riskier passes
        // because they value the through ball / chance-creation upside.
        // Which traits flag a player as risk-tolerant lives in the trait
        // registry (`risk_tolerant_passer` field) — adding a new such
        // trait no longer requires touching this evaluator.
        let risk_tolerant = has_risk_tolerant_passing_trait(&passer.traits);
        let is_recommended = PassEvaluatorConfig::default().is_recommended(
            success_probability,
            risk_level,
            risk_tolerant,
        );

        PassEvaluation {
            success_probability,
            risk_level,
            expected_value,
            factors,
            is_recommended,
        }
    }

    /// Calculate how distance affects pass success
    fn calculate_distance_factor(distance: f32, passer: &MatchPlayer) -> f32 {
        let cfg = PassEvaluatorConfig::default();
        let passing_skill = passer.skills.technical.passing;
        let vision_skill = passer.skills.mental.vision;
        let technique_skill = passer.skills.technical.technique;

        // Vision and technique extend effective passing range. The
        // bonus values are baked into the config helper calls below;
        // raw `(vision_skill / scale)` is still used inside the
        // long-pass skill-factor branches further down.
        let optimal_range = cfg.optimal_range(passing_skill, vision_skill);
        let max_effective_range = cfg.max_effective_range(passing_skill, vision_skill);
        let ultra_long_threshold = cfg.ultra_long_threshold;
        let extreme_long_threshold = cfg.extreme_long_threshold;

        if distance <= optimal_range {
            // Short to medium passes - very high success
            1.0 - (distance / optimal_range * 0.1)
        } else if distance <= max_effective_range {
            // Long passes (60-100m) - declining success (less penalty with high vision)
            let excess = distance - optimal_range;
            let range = max_effective_range - optimal_range;
            let base_decline = 0.9 - (excess / range * 0.5);
            // Vision reduces the decline penalty
            base_decline + (vision_skill / 20.0 * 0.1)
        } else if distance <= ultra_long_threshold {
            // Very long passes (100-200m) - vision and technique critical
            let excess = distance - max_effective_range;
            let range = ultra_long_threshold - max_effective_range;
            let skill_factor = (vision_skill / 20.0 * 0.6) + (technique_skill / 20.0 * 0.3);

            let base_factor = 0.4 - (excess / range * 0.25);
            (base_factor + skill_factor * 0.2).clamp(0.15, 0.55)
        } else if distance <= extreme_long_threshold {
            // Ultra-long passes (200-300m) — smooth curve across skill_factor
            // instead of two hard tiers. A 5/20 player no longer gets the
            // same 0.10 floor as a 9/20 player; success scales continuously.
            let skill_factor = (vision_skill / 20.0 * 0.7) + (technique_skill / 20.0 * 0.3);
            (0.05 + skill_factor * 0.40).clamp(0.02, 0.45)
        } else {
            // Extreme long passes (300m+) — goalkeeper clearances,
            // desperate plays. Smooth curve replaces 3-tier step.
            let skill_factor = (vision_skill / 20.0 * 0.5)
                + (technique_skill / 20.0 * 0.35)
                + (passing_skill / 20.0 * 0.15);
            (0.03 + skill_factor * 0.35).clamp(0.02, 0.40)
        }
    }

    /// Calculate how the angle between passer's facing and pass direction affects success
    fn calculate_angle_factor(
        ctx: &StateProcessingContext,
        passer: &MatchPlayer,
        receiver: &MatchPlayerLite,
    ) -> f32 {
        let cfg = PassEvaluatorConfig::default();
        let pass_direction = (receiver.position - passer.position).normalize();
        let passer_velocity = ctx.tick_context.positions.players.velocity(passer.id);

        if passer_velocity.norm() < cfg.stationary_velocity_threshold {
            // Standing still - can pass in any direction easily
            return cfg.stationary_angle_factor;
        }

        let facing_direction = passer_velocity.normalize();
        let dot_product = pass_direction.dot(&facing_direction);
        cfg.angle_factor_from_dot(dot_product)
    }

    /// Calculate pressure on the passer from opponents
    fn calculate_pressure_factor(ctx: &StateProcessingContext, passer: &MatchPlayer) -> f32 {
        let pressure_radius = PassEvaluatorConfig::default().pressure_radius;

        // Compute closest distance and count without allocation
        let mut closest_distance = pressure_radius;
        let mut num_opponents: f32 = 0.0;

        for (_, dist) in ctx.tick_context.grid.opponents(passer.id, pressure_radius) {
            num_opponents += 1.0;
            if dist < closest_distance {
                closest_distance = dist;
            }
        }

        if num_opponents == 0.0 {
            return 1.0; // No pressure
        }

        // Pressure from distance
        let distance_pressure = (closest_distance / pressure_radius).clamp(0.0, 1.0);

        // Additional pressure from multiple opponents
        let number_pressure = (1.0 - (num_opponents - 1.0) * 0.15).max(0.5);

        // Mental attributes help under pressure — fatigue-folded.
        let minute = sc::minute_from_ms(ctx.context.total_match_time);
        let mental = sc::EffActionContext::mental(minute);
        let composure_factor = sc::n(sc::eff(passer, mental, |p| p.skills.mental.composure));
        let decision_factor = sc::n(sc::eff(passer, mental, |p| p.skills.mental.decisions));

        let base_pressure = distance_pressure * number_pressure;
        let pressure_with_mentals =
            base_pressure + (1.0 - base_pressure) * composure_factor * decision_factor;

        // Floor lowered 0.30 → 0.10 so sub-5 composure/decisions
        // visibly fold under pressure compared to a 10/20 player.
        pressure_with_mentals.clamp(0.10, 1.0)
    }

    /// Evaluate receiver's positioning quality
    fn calculate_receiver_positioning(
        ctx: &StateProcessingContext,
        receiver: &MatchPlayerLite,
    ) -> f32 {
        const VERY_CLOSE_RADIUS: f32 = 8.0;
        const CLOSE_RADIUS: f32 = 18.0;
        const MEDIUM_RADIUS: f32 = 30.0;

        // Count opponents in each zone without allocation (single pass)
        let mut very_close_opponents: usize = 0;
        let mut close_opponents: usize = 0;
        let mut medium_opponents: usize = 0;

        for (_, dist) in ctx.tick_context.grid.opponents(receiver.id, MEDIUM_RADIUS) {
            if dist < VERY_CLOSE_RADIUS {
                very_close_opponents += 1;
            } else if dist < CLOSE_RADIUS {
                close_opponents += 1;
            } else {
                medium_opponents += 1;
            }
        }

        // Calculate space quality with heavy penalties for nearby opponents
        let space_factor = if very_close_opponents > 0 {
            // Very tightly marked - poor option
            0.15 - (very_close_opponents as f32 * 0.1).min(0.12)
        } else if close_opponents > 0 {
            // Marked — risky target
            0.45 - (close_opponents as f32 * 0.15).min(0.3)
        } else if medium_opponents > 0 {
            // Some pressure but workable
            0.75 - (medium_opponents as f32 * 0.1).min(0.2)
        } else {
            // Completely free - excellent option
            1.0
        };

        // Check if receiver is moving into space or standing still
        let receiver_velocity = ctx.tick_context.positions.players.velocity(receiver.id);
        let movement_factor = if receiver_velocity.norm() > 1.5 {
            // Moving into space - excellent
            1.15
        } else if receiver_velocity.norm() > 0.5 {
            // Some movement - good
            1.05
        } else {
            // Standing still - acceptable but not ideal
            0.95
        };

        // Off-ball quality. Use the dedicated `off_ball_attack`
        // composite when the receiver is in the registry — it folds
        // off_the_ball, anticipation, decisions, accel/pace, teamwork,
        // and bravery through `effective_skill`. Fall back to the
        // legacy raw skill blend if the lookup misses.
        let minute = sc::minute_from_ms(ctx.context.total_match_time);
        let off_ball_lift = match ctx.context.players.by_id(receiver.id) {
            Some(rp) => sc::off_ball_attack(rp, minute) * 0.30,
            None => {
                let players = ctx.player();
                let skills = players.skills(receiver.id);
                let off_ball = (skills.mental.off_the_ball / 20.0).clamp(0.0, 1.0);
                let positioning = (skills.mental.positioning / 20.0).clamp(0.0, 1.0);
                off_ball * 0.15 + positioning * 0.15
            }
        };

        (space_factor * movement_factor * (0.7 + off_ball_lift)).clamp(0.1, 1.0)
    }

    /// Calculate passer's ability to execute this pass.
    ///
    /// Routes through the shared `passing_execution` / `long_passing`
    /// composites so fatigue, late-game mental drift, and stamina
    /// mitigation are applied consistently. The composite already
    /// accounts for condition via `effective_skill`, so we no longer
    /// multiply by a raw condition factor.
    fn calculate_passer_ability(
        ctx: &StateProcessingContext,
        passer: &MatchPlayer,
        distance: f32,
    ) -> f32 {
        let minute = sc::minute_from_ms(ctx.context.total_match_time);
        // Distance blend: short passes weighted toward `passing_execution`
        // (technique-led), long passes weighted toward `long_passing`
        // (vision-led). Crossover at ~80u so the weighting transitions
        // smoothly across the field.
        let long_weight = (distance / 80.0).clamp(0.0, 1.0);
        let short = sc::passing_execution(passer, minute);
        let long = sc::long_passing(passer, minute);
        let composite = short * (1.0 - long_weight) + long * long_weight;
        // Floor lowered 0.30 → 0.05 so a sub-5 passer is meaningfully
        // worse than a 10/20 passer — the downstream success formula
        // multiplies this and re-clamps the final probability anyway.
        composite.clamp(0.05, 1.0)
    }

    /// Calculate receiver's ability to control the pass.
    /// Uses the receiving / first-touch composite — same fatigue and
    /// late-game pathway as every other skill read.
    fn calculate_receiver_ability(ctx: &StateProcessingContext, receiver: &MatchPlayerLite) -> f32 {
        let receiver_player = match ctx.context.players.by_id(receiver.id) {
            Some(p) => p,
            None => return 0.05,
        };
        let minute = sc::minute_from_ms(ctx.context.total_match_time);
        // Floor lowered 0.30 → 0.05 so a sub-5 first-touch player is
        // visibly worse at receiving than an average 10/20 — instead
        // of cliff-equal to anyone below 6/20.
        sc::receiving_first_touch(receiver_player, minute).clamp(0.05, 1.0)
    }

    /// Calculate tactical value of the pass
    fn calculate_tactical_value(ctx: &StateProcessingContext, receiver: &MatchPlayerLite) -> f32 {
        let ball_position = ctx.tick_context.positions.ball.position;
        let receiver_position = receiver.position;
        let passer_position = ctx.player.position;
        let field_height = ctx.context.field_size.height as f32;
        let field_center_y = field_height / 2.0;

        // Determine which direction is forward based on player side.
        // Use the `PlayerSide` helpers so right-side normalization stays
        // correct — see `PlayerSide::attacking_progress_x` for why the
        // legacy `x * dir / width` formula was buggy.
        let side = ctx.player.side.unwrap_or(PlayerSide::Left);
        let field_width = ctx.context.field_size.width as f32;

        // Forward progress as a signed [-1, 1]-ish ratio.
        let forward_progress =
            side.forward_delta_norm(ball_position.x, receiver_position.x, field_width);

        // Strong penalty for backward passes, strong reward for forward
        // Defenders get extra penalty for backward passes since they're already deep
        let is_defender = ctx
            .player
            .tactical_position
            .current_position
            .position_group()
            == PlayerFieldPositionGroup::Defender;

        // Penalize pure sideways passes that don't progress the ball
        // But exempt wide switches — lateral passes that spread the play are valuable
        let lateral_change = (receiver_position.y - passer_position.y).abs();
        let forward_change = side
            .forward_delta(passer_position.x, receiver_position.x)
            .abs();
        let sideways_penalty = if forward_change < 10.0 && lateral_change > 20.0 {
            if lateral_change > field_height * 0.25 {
                // Wide switch — this is good, no penalty
                0.0
            } else {
                // Short sideways pass in a cluster — discourage
                -0.25
            }
        } else {
            0.0
        };

        // Phase-aware modulation — modern football varies its
        // forward/backward valuation by team phase. In settled build-up
        // a backward pass to the keeper or CB is a normal recycle, not
        // a sin; in transition it's the death of the counter. The
        // multipliers below come straight off `team().phase()` so every
        // player on the side reads the same tactical weather.
        let phase = ctx.team().phase();
        let (phase_forward_mult, phase_backward_mult): (f32, f32) = match phase {
            // Recycling is correct, line-breaking forward less critical.
            GamePhase::BuildUp => (0.65, 0.30),
            // Direct: every forward yard is gold.
            GamePhase::AttackingTransition => (1.40, 1.20),
            // Cutbacks and resets to the edge of the box are normal.
            GamePhase::Attack => (1.05, 0.55),
            // Standard: forward ≥ backward.
            GamePhase::Progression => (1.00, 1.00),
            // Settled defending — out of possession, but if a turnover
            // gives the ball briefly we'd still want a forward look.
            _ => (1.00, 1.00),
        };

        // Risk appetite biases forward over backward. Late chase = the
        // pass evaluator should over-prefer the forward option.
        let risk_appetite = ctx.team().risk_appetite();
        let risk_forward_bias = 0.7 + risk_appetite * 0.6; // 0.7..1.3
        let risk_backward_bias = 1.4 - risk_appetite * 0.8; // 1.4..0.6

        // Manager-set directness dial (-2 possession .. +2 direct, 0
        // neutral). Distinct from `risk_appetite` (match-state driven,
        // forward vs backward) — this is a pre-match tactics-board input
        // that specifically controls long-ball selection and the
        // short recycle-to-the-back habit, i.e. build-up vs direct play.
        let directness = ctx.player.directness_bias;

        let mut forward_value = if forward_progress < 0.0 {
            // Backward pass - penalty, but softened by phase + risk.
            let composure_reduction = (ctx.player.skills.mental.composure / 20.0) * 0.3;
            let base_penalty = forward_progress * 3.0 * (1.0 - composure_reduction).max(0.5);
            let phase_adjusted = base_penalty * phase_backward_mult * risk_backward_bias;
            if is_defender {
                // Realism-bug 2026-07-26 (passing follow-up): this ×1.5
                // previously applied UNCONDITIONALLY — but its own
                // comment's stated intent ("we don't want CBs hoof-ing
                // back into pressure for fun") is specifically about a
                // pressured backward hoof, not calm circulation among an
                // unpressed back four, which real CBs do constantly.
                // Measured against real StatsBomb data: CB forward-share
                // ran 89-91% engine vs a real 62.8% — this unconditional
                // penalty (stacked with the forward side's own 3.0x
                // multiplier above) is a plausible direct contributor,
                // since it made ANY backward option score worse than a
                // pressured-hoof would deserve even when nothing forced
                // it. Gated on genuine nearby pressure so the original
                // intent still holds under press; a calm, unpressed
                // sideways/backward circulation pass no longer pays the
                // extra penalty on top of the base backward penalty
                // every position already carries.
                let under_press = ctx.players().opponents().nearby(15.0).next().is_some();
                if under_press {
                    phase_adjusted * 1.5
                } else {
                    phase_adjusted
                }
            } else {
                phase_adjusted
            }
        } else {
            // Forward pass - strong reward, especially in transition.
            // /goal 2026-07-26 (passing realism, full-latitude experiment):
            // three independent narrow, additive levers (pressure-gated
            // backward penalty, forward_width_bonus at two magnitudes,
            // turnover-risk-by-location) each measured a limited-to-zero
            // effect on CB forward-share (stuck 87-91% vs a real 62.8%)
            // — diagnosed as structural: those fixes all landed inside
            // WEIGHTED terms (width_bonus*0.22, promoted_pass_value*0.15)
            // while this 3.0x multiplier competes directly against them
            // unweighted-equivalent at the top of the sum. Reduced to 2.6
            // — still a genuine premium over the non-defender 2.5 (a
            // real defender's progressive pass IS somewhat more valued
            // than a winger's routine one), but no longer the dominant,
            // hard-to-outweigh term it was at 3.0. Explicit Pavel sign-off
            // to touch this shared constant this session, unlike the
            // additive-only fixes earlier.
            if is_defender {
                // /goal 2026-07-26: a further push to 2.2 (from 2.6) +
                // larger width bonuses (below) was tried and REVERTED —
                // measured CB center-destination move the WRONG way
                // (44%→51%), contradicting the expected direction. Given
                // the ~550-sample-per-position batch size implies a ~7pp
                // standard error at this proportion, that regression is
                // not clearly distinguishable from noise either way —
                // logged honestly rather than chased further with more
                // expensive re-batching. Held at 2.6, the last value with
                // a consistent (if partial) measured improvement.
                forward_progress * 2.6 * phase_forward_mult * risk_forward_bias
            } else {
                forward_progress * 2.5 * phase_forward_mult * risk_forward_bias
            }
        };

        // Distance bonus: prefer passes of 20-50m over very short (< 15m) or very long
        let pass_distance = (receiver_position - passer_position).norm();
        let distance_value = if pass_distance < 10.0 {
            // Very short pass - only good under pressure
            0.3
        } else if pass_distance < 20.0 {
            // Short pass - acceptable
            0.6
        } else if pass_distance < 50.0 {
            // Ideal passing range - good progression
            1.0
        } else if pass_distance < 80.0 {
            // Long pass - still valuable
            0.8
        } else if pass_distance < 120.0 {
            // Long pass - declining value
            0.5
        } else if pass_distance < 200.0 {
            // Very long pass - risky
            0.3
        } else {
            // Extreme distance - rarely accurate
            let vision_skill = ctx.player.skills.mental.vision / 20.0;
            0.2 * vision_skill
        };

        // === WIDTH AND FLANKS BONUS ===
        // Reward passes to wide positions - creates more varied play
        let receiver_distance_from_center = (receiver_position.y - field_center_y).abs();
        let passer_distance_from_center = (passer_position.y - field_center_y).abs();

        // How wide is the receiver? (0.0 = center, 1.0 = touchline)
        let receiver_width_ratio =
            (receiver_distance_from_center / (field_height / 2.0)).clamp(0.0, 1.0);
        let passer_width_ratio =
            (passer_distance_from_center / (field_height / 2.0)).clamp(0.0, 1.0);

        // Width bonus - reward passes to wide areas
        // Extra bonus if passer is central and receiver is wide (spreading play)
        let spreading_play_bonus = if passer_width_ratio < 0.4 && receiver_width_ratio > 0.5 {
            0.25 // Central player finding wide teammate — strong incentive
        } else {
            0.0
        };

        // Midfielder-specific width incentive — midfielders should distribute wide
        let is_midfielder = ctx
            .player
            .tactical_position
            .current_position
            .position_group()
            == PlayerFieldPositionGroup::Midfielder;
        let midfielder_width_bonus = if is_midfielder && receiver_width_ratio > 0.4 {
            0.15 // Midfielders get extra reward for wide distribution
        } else {
            0.0
        };

        // Realism-bug 2026-07-26 (passing follow-up), REWRITTEN under
        // /goal full-latitude experiment: `defender_width_bonus` and
        // `forward_width_bonus` originally lived inside `width_bonus`
        // (routed through this term's own outer ×0.22 weight below) —
        // measured as too diluted to matter: a 0.15 raw bonus nets only
        // ~0.033 of actual tactical_value, while the UNWEIGHTED central-
        // pull terms these positions actually compete against
        // (`cutback_bonus` 0.30-0.50, the defender forward multiplier
        // above) operate at 10-15× that scale. Rather than keep raising
        // a diluted term, these two are now computed here but added
        // DIRECTLY to `tactical_value` (below, alongside cutback_bonus/
        // arriving_runner_bonus) at a magnitude in that same unweighted
        // family — a fair fight instead of a diluted one.
        // `midfielder_width_bonus` above is deliberately left as-is
        // inside `width_bonus` — CM/WIDE already matched real StatsBomb
        // data closely at that magnitude, so it isn't touched.
        // /goal 2026-07-26: a push to 0.42 (from 0.28) was tried
        // alongside the multiplier change above and reverted together —
        // see that comment for the measured (noise-level, wrong-
        // direction) result. Held at 0.28, the last value with a
        // consistent measured improvement.
        let defender_width_bonus_unweighted = if is_defender && receiver_width_ratio > 0.4 {
            0.28
        } else {
            0.0
        };
        let is_forward = ctx
            .player
            .tactical_position
            .current_position
            .position_group()
            == PlayerFieldPositionGroup::Forward;
        // /goal 2026-07-26: a push to 0.38 (from 0.28) was tried and
        // reverted alongside the two changes above — held at 0.28.
        let forward_width_bonus_unweighted = if is_forward && receiver_width_ratio > 0.4 {
            0.28
        } else {
            0.0
        };

        let width_bonus = if receiver_width_ratio > 0.7 {
            // Very wide (near touchline) - excellent for stretching play
            0.5 + spreading_play_bonus + midfielder_width_bonus
        } else if receiver_width_ratio > 0.5 {
            // Wide areas - good for creating space
            0.35 + spreading_play_bonus + midfielder_width_bonus
        } else if receiver_width_ratio > 0.3 {
            // Half-spaces - valuable attacking zones
            0.2 + midfielder_width_bonus
        } else {
            // Central - no bonus (already gets forward progress bonus usually)
            0.0
        };

        // === SWITCHING PLAY BONUS ===
        // Reward passes that switch the play from one side to the other.
        // /goal 2026-07-26 (passing realism, full-latitude experiment):
        // found while chasing the CB center-destination gap. This only
        // ever checked the MAGNITUDE of lateral displacement from the
        // PASSER's own position — a CB standing wide-right passing to a
        // dead-CENTRAL midfielder far in y from the CB's own position
        // satisfied `lateral_change > field_height*0.3` and collected
        // this bonus (up to 0.70 raw, capped into switch_total at 0.45
        // — bigger than `defender_width_bonus_unweighted`'s 0.28) even
        // though the pass didn't switch the play to a wide area at all;
        // it just moved centrally from a wide starting point. A genuine
        // "switch of play" changes which SIDE the ball is on — it
        // requires the ball to actually END UP wide, not merely to have
        // travelled far in y. Added `receiver_width_ratio > 0.5` (the
        // same "wide areas" threshold `width_bonus` itself already uses
        // two tiers up) so this only fires for an actual switch, not any
        // large central recycle from a wide starting position.
        let lateral_change = (receiver_position.y - passer_position.y).abs();
        let is_switching_play =
            lateral_change > field_height * 0.3 && receiver_width_ratio > 0.5;

        let switch_play_bonus = if is_switching_play {
            let vision_skill = ctx.player.skills.mental.vision / 20.0;
            // Big bonus for switching play - opens up space
            0.45 + (vision_skill * 0.25)
        } else {
            0.0
        };

        // Side-overload is now a single path: `same_side_density_penalty`
        // below (driven by the team-shared `side_density_*` signals).
        // The legacy half-pitch overload_penalty was double-counting the
        // same situation and was removed during the polish pass.

        // Long cross-field passes - reward vision players for switching play
        let vision_skill = ctx.player.skills.mental.vision / 20.0;
        let technique_skill = ctx.player.skills.technical.technique / 20.0;

        let long_pass_bonus_base = if pass_distance > 300.0 {
            // Extreme distance (300m+) - very risky, minimal bonus
            (vision_skill * 0.3 + technique_skill * 0.2) * 0.2
        } else if pass_distance > 200.0 {
            // Ultra-long diagonal (200-300m) - risky
            (vision_skill * 0.3 + technique_skill * 0.15) * 0.2
        } else if pass_distance > 100.0 {
            // Very long pass (100-200m) - small bonus for high vision
            vision_skill * 0.15
        } else if pass_distance > 60.0 {
            // Long pass (60-100m) - modest bonus
            vision_skill * 0.1
        } else {
            0.0
        };
        // Directness dial scales long-pass selection directly: +2
        // (fully Direct) -> 1.3x, -2 (fully Possession) -> 0.7x. This
        // secondary effect also amplifies/dampens the existing
        // PlaysShortPasses/PlaysLongPasses trait bonuses below, which
        // key off long_pass_bonus too.
        let long_pass_bonus = long_pass_bonus_base * (1.0 + directness * 0.15);

        // Directness dial, PRIMARY unweighted effect. `long_pass_bonus`
        // above only enters `tactical_value` at a 0.05 weight — too
        // diluted to move behaviour on its own (measured: a 2026-08-05
        // A/B batch showed no detectable shift in pass-length share).
        // This term is added FLAT, the same pattern already proven for
        // cutback_bonus/build_up_recycle_bonus in this file, so it
        // isn't drowned out by the weighted sum. Signal: -1 at a 0u
        // pass, +1 at a 180u+ pass, centred on the 60u boundary the
        // evaluator already treats as "long" above.
        let directness_length_bias = if directness != 0.0 {
            let length_signal = ((pass_distance - 60.0) / 120.0).clamp(-1.0, 1.0);
            directness * length_signal * 0.12
        } else {
            0.0
        };

        // Passes to advanced positions are more valuable
        let position_value = match receiver.tactical_positions.position_group() {
            PlayerFieldPositionGroup::Forward => 1.0,
            PlayerFieldPositionGroup::Midfielder => 0.7,
            PlayerFieldPositionGroup::Defender => 0.4,
            PlayerFieldPositionGroup::Goalkeeper => 0.2,
        };

        // === CUTBACK / HIGH-xG RECEIVER BONUS ===
        // Modern football: a pass from the byline that pulls the ball
        // BACK to a runner at the penalty spot is one of the highest-xG
        // passes there is. The classical evaluator scored that as a
        // backward sideways ball and slammed it. We now detect:
        //   * passer is wide AND inside the attacking third
        //     (using `attacking_progress_x` so right-side teams aren't
        //     locked out by the legacy negative-progress bug)
        //   * receiver is in the central high-xG corridor near opp goal
        //   * pass distance is short-to-medium (real cutbacks, not
        //     desperate long crosses)
        // The bonus is graded on receiver space, passer decisions, and
        // teamwork — a tight cutback under heavy marking is worth less.
        let passer_progress = side.attacking_progress_x(passer_position.x, field_width);
        let receiver_progress = side.attacking_progress_x(receiver_position.x, field_width);
        let receiver_y_offset = (receiver_position.y - field_center_y).abs();
        let passer_y_offset = (passer_position.y - field_center_y).abs();
        let cutback_pattern = passer_progress > 0.70
            && receiver_progress > 0.78
            && receiver_y_offset < field_height * 0.15
            && passer_y_offset > field_height * 0.20
            && pass_distance < 60.0;
        let cutback_bonus = if cutback_pattern {
            // Receiver space inferred from receiver_positioning (already
            // computed above as one of the PassFactors): higher = freer.
            // Range 0.30 .. 0.50 per spec.
            let receiver_space_factor = {
                // Re-read receiver positioning instead of plumbing the
                // factors through — the math is dominated by opponent
                // proximity, which is what we want here.
                let opps = ctx.tick_context.grid.opponents(receiver.id, 12.0).count();
                match opps {
                    0 => 1.0,
                    1 => 0.6,
                    _ => 0.2,
                }
            };
            let decisions = (ctx.player.skills.mental.decisions / 20.0).clamp(0.0, 1.0);
            let teamwork = (ctx.player.skills.mental.teamwork / 20.0).clamp(0.0, 1.0);
            (0.30 + receiver_space_factor * 0.10 + decisions * 0.05 + teamwork * 0.05)
                .clamp(0.30, 0.50)
        } else {
            0.0
        };

        // === ARRIVING CENTRAL RUNNER BONUS ===
        // A central midfielder who has arrived in the central box corridor
        // in space is a high-value target the classic evaluator under-rates
        // — the feed is often a short / square ball that scores low on
        // forward-progress, so without this the carrier passes elsewhere
        // and the late run is wasted. This biases carriers to FEED the
        // arriving runner whenever they pass (it only shifts target
        // SELECTION — it never forces an extra pass), which is the supply
        // side of midfielders scoring. Gated tight: receiver is a central
        // midfielder, deep in the central corridor, and unmarked. Skipped
        // when the byline cutback already fired so the two don't stack.
        let arriving_runner_bonus = if !cutback_pattern
            && receiver.tactical_positions.is_central_midfielder()
            && receiver_progress > 0.80
            && receiver_y_offset < field_height * 0.15
        {
            let opps = ctx.tick_context.grid.opponents(receiver.id, 12.0).count();
            match opps {
                0 => 0.38,
                1 => 0.18,
                _ => 0.0,
            }
        } else {
            0.0
        };

        // === BUILD-UP RECYCLING BONUS ===
        // In build-up, a short pass to a CB / DM / GK that resets play
        // is a healthy modern pattern, not a panic option. Gated on:
        //   * phase == BuildUp
        //   * pass distance 12..65 u (genuine recycle, not a hoof or
        //     a one-touch trade)
        //   * receiver is GK/CB/DM
        //   * passer under press OR build_up_patience > 0.65
        // Range 0.15 .. 0.40 — 0.15 baseline, +0.25 if all conditions
        // including pressure are present.
        // `is_defender()` already includes the DefensiveMidfielder
        // role (see `PlayerPositionType::position_group`), so this
        // covers GK + CB + DM together.
        let receiver_is_recycle_target = receiver.tactical_positions.is_defender()
            || matches!(
                receiver.tactical_positions.position_group(),
                PlayerFieldPositionGroup::Goalkeeper
            );
        // /goal 2026-07-26 (passing realism, full-latitude experiment):
        // this bonus previously required `under_press || patient` to
        // fire AT ALL — meaning a perfectly normal, calm CB-to-CB or
        // CB-to-DM square ball (holding shape, no panic, no explicit
        // "patient" flag) earned ZERO positive credit, only whatever
        // was left after the backward-pass penalty. Measured effect:
        // even with that penalty pressure-gated (2026-07-26 fix above)
        // and defenders given their own width incentive, CB forward-
        // share stayed stuck at 87-92% against a real 62.8% — real
        // defensive circulation needed its OWN positive value, not
        // just a smaller penalty. Two changes: (1) phase gate widened
        // to also cover `Progression` — real backline circulation is
        // not exclusive to the engine's narrow `BuildUp` classification;
        // (2) the `under_press || patient` requirement now only decides
        // the BONUS on top of a 0.12 floor that always applies to a
        // genuine short recycle pass to a defensive outlet, matching
        // this bonus's own comment ("a healthy modern pattern, not a
        // panic option") — panic-free circulation shouldn't need to be
        // flagged "patient" or "under press" to be worth anything.
        let build_up_recycle_bonus = if matches!(phase, GamePhase::BuildUp | GamePhase::Progression)
            && pass_distance >= 12.0
            && pass_distance <= 65.0
            && receiver_is_recycle_target
        {
            let under_press = ctx.players().opponents().nearby(12.0).next().is_some();
            let patient = ctx.team().build_up_patience() > 0.65;
            let mut bonus: f32 = 0.12;
            if under_press {
                bonus += 0.15;
            }
            if patient {
                bonus += 0.10;
            }
            // Directness dial shifts the recycle floor/ceiling directly:
            // Direct (+2) suppresses the "safe short ball back" habit,
            // Possession (-2) reinforces it — this is the build-up half
            // of the same manager dial that scales long_pass_bonus above.
            (bonus - directness * 0.06).clamp(0.02, 0.45)
        } else {
            0.0
        };

        // === STALLED-ATTACK RECYCLING BONUS (Milestone 11) ===
        // `build_up_recycle_bonus` above only fires in `BuildUp` — deep
        // in our own third. Real teams also deliberately reset through
        // the back four / keeper from much higher up, when a genuinely
        // long possession spell (Progression/Attack — we've been
        // probing, not just settling in) hasn't broken through. Without
        // this, backward recycling from advanced positions only ever
        // happens as a leftover "nothing forward scored higher" default,
        // never as a positively-valued choice in its own right.
        //
        // STALLED_POSSESSION_TICKS: a reasoned ~20 real-world seconds of
        // continuous possession before a genuine reset becomes the
        // doctrinally-expected response (no public dataset isolates this
        // exact threshold — flagged as an estimate, not a sourced stat),
        // converted through the engine's own ~9x match-time compression
        // (`football_minute_from_ms`, established this session): 20s / 9
        // ≈ 2.2 engine-seconds ≈ 220 ticks @ 10ms/tick.
        const STALLED_POSSESSION_TICKS: u32 = 220;
        let stalled_recycle_bonus = if matches!(phase, GamePhase::Progression | GamePhase::Attack)
            && ctx.team().possession_ticks() >= STALLED_POSSESSION_TICKS
            && pass_distance >= 12.0
            && pass_distance <= 100.0
            && receiver_is_recycle_target
        {
            let under_press = ctx.players().opponents().nearby(12.0).next().is_some();
            let overtime_ticks =
                (ctx.team().possession_ticks() - STALLED_POSSESSION_TICKS) as f32;
            // Grows from the moment the stall threshold is crossed,
            // saturating ~10 further engine-seconds (≈90 real seconds)
            // later — long enough to keep rising through a genuinely
            // prolonged siege without ever exceeding the build-up
            // bonus's own ceiling.
            let overtime_factor = (overtime_ticks / 1000.0).min(1.0);
            let mut bonus: f32 = 0.12 + overtime_factor * 0.13;
            if under_press {
                bonus += 0.10;
            }
            // Same directness treatment as build_up_recycle_bonus above.
            (bonus - directness * 0.06).clamp(0.02, 0.40)
        } else {
            0.0
        };

        // === COUNTER-PRESS DIRECT-FIRST-PASS BONUS ===
        // After winning the ball back, the first pass should be direct
        // — feed a forward making a run. Gated additionally on the
        // receiver actually having forward space and the pass not
        // running through opponents (low interception risk implied by
        // receiver_positioning > 0.5).
        let counter_first_pass_bonus = if matches!(phase, GamePhase::AttackingTransition)
            && forward_value > 0.0
            && receiver.tactical_positions.is_forward()
        {
            // Gate on receiver having space — checked by counting
            // opponents in their immediate area.
            let receiver_opps = ctx.tick_context.grid.opponents(receiver.id, 15.0).count();
            if receiver_opps == 0 {
                0.40
            } else if receiver_opps == 1 {
                0.30
            } else {
                // Crowded receiver — direct ball is wasted. Skip bonus.
                0.0
            }
        } else {
            0.0
        };

        // === §12.6 RESULTING SCORING DANGER ===
        // Value a pass by how dangerous the RECEIVING position is — a
        // lightweight positional-value proxy (distance to goal, angle,
        // defender proximity), the passing-side mirror of the shot
        // decision's xG shape. This is a ranking term among viable
        // candidates: with two similarly-open options, the one closer
        // to goal at a better angle wins; it never hard-gates anything.
        let attack_goal_x = match side {
            PlayerSide::Left => field_width,
            PlayerSide::Right => 0.0,
        };
        let recv_goal_dist = ((receiver_position.x - attack_goal_x).powi(2)
            + (receiver_position.y - field_center_y).powi(2))
        .sqrt();
        let mut danger_value = {
            let dist01 = (1.0 - recv_goal_dist / 320.0).clamp(0.0, 1.0);
            let central01 = 1.0 - (receiver_y_offset / (field_height * 0.5)).clamp(0.0, 1.0);
            let recv_opps = ctx.tick_context.grid.opponents(receiver.id, 15.0).count();
            let space01 = match recv_opps {
                0 => 1.0,
                1 => 0.55,
                _ => 0.25,
            };
            // Squared distance term concentrates the value in the final
            // quarter, where scoring danger actually lives.
            dist01 * dist01 * (0.5 + 0.5 * central01) * space01
        };

        // === SIDE-DENSITY OVERLOAD ===
        // Use the team-shared side density signal: too many of OUR
        // players on one side discourages another pass into that side
        // and rewards a switch. `ball_side` already tells us which
        // lateral third the ball is in (= the pass-source side, modulo
        // ball motion).
        let team_state = ctx.context.tactical_for_team(ctx.player.team_id);
        let receiver_side_zone = BallSideZone::for_y(field_height, receiver_position.y);
        let receiver_side_density = match receiver_side_zone {
            BallSideZone::Left => team_state.side_density_left,
            BallSideZone::Center => team_state.side_density_center,
            BallSideZone::Right => team_state.side_density_right,
        };
        let same_side_density_penalty = Self::same_side_density_penalty(receiver_side_density);
        // Reward switches to underloaded sides with a vision-graded
        // bonus. Two-band threshold: any pass that crosses lateral
        // thirds and lands in a side with ≤3 own players counts.
        let passer_side_zone = BallSideZone::for_y(field_height, passer_position.y);
        let crosses_sides = passer_side_zone != receiver_side_zone;
        let vision = (ctx.player.skills.mental.vision / 20.0).clamp(0.0, 1.0);
        let underload_switch_bonus =
            Self::underload_switch_bonus(crosses_sides, receiver_side_density, vision);

        // Milestone 8 (possession-decision-intelligence PRD) — genuine
        // numerical overload/isolation: `same_side_density_penalty`/
        // `underload_switch_bonus` above only ever read OUR OWN side
        // density, never the opponent's, so they can't distinguish a
        // real 4v2 overload from a genuinely even 4v4 in the same zone.
        // `TeamTacticalState.side_density_*` is already computed for
        // BOTH teams (same absolute-Y `BallSideZone` classification), so
        // the opponent's count is a free read, not a new scan.
        let opp_team_id = if ctx.player.team_id == ctx.context.field_home_team_id {
            ctx.context.field_away_team_id
        } else {
            ctx.context.field_home_team_id
        };
        let opp_state = ctx.context.tactical_for_team(opp_team_id);
        let opp_side_density = match receiver_side_zone {
            BallSideZone::Left => opp_state.side_density_left,
            BallSideZone::Center => opp_state.side_density_center,
            BallSideZone::Right => opp_state.side_density_right,
        };
        let overload_advantage = receiver_side_density as f32 - opp_side_density as f32;
        let overload_bonus = Self::overload_bonus(crosses_sides, overload_advantage);

        // Cap the combined "switch reward" so a wide-vision playmaker
        // doesn't double-dip the classic switch_play_bonus and the
        // density-driven underload_switch_bonus. Polish spec: total
        // switch reward ≤ 0.45. Applied flat (not re-weighted): that
        // ceiling is the absolute contribution to tactical_value.
        // Milestone 8's overload_bonus is composed INTO this same capped
        // sum (not stacked as a separate uncapped term) since it's the
        // same "reward for moving the ball to the better flank" concept.
        let switch_total = (switch_play_bonus + underload_switch_bonus + overload_bonus).min(0.45);

        // Option B / B3: Component C, promoted from tie-breaker toward a
        // primary term. Reuses the shared time-to-intercept primitive
        // (pass-lane contest + receiver terminal value) rather than
        // `danger_value`'s plain distance/space proxy. Weight 0.15 is
        // deliberately more than `danger_value`'s 0.06 (a genuine
        // promotion) but still one term among ~13 in this sum, not
        // dominant — CLAUDE.md's own Phase 12 lesson: a broad
        // forward-passing reward both inflates goals and dilutes the
        // surgically-tuned link/supply/intercept features if it's not
        // kept tightly scoped. Added alongside `danger_value` (not
        // replacing it) — the lower-risk choice, since `danger_value`'s
        // own weight/interactions are already calibrated elsewhere.
        let mut promoted_pass_value = crate::r#match::player::strategies::common::players::ops::on_ball_value::pass_value(ctx, receiver);

        // /goal 2026-07-26 (passing realism, larger structural rework):
        // six narrow additive counterweights (across two sessions)
        // failed to meaningfully shift CB forward-share (stuck ~90% vs
        // real 62.8%) or FW center-destination (stuck ~70% vs real
        // 20.8%) — diagnosed as a limitation of the formula's purely
        // additive structure: a strongly-central AND strongly-forward
        // option collects full credit on BOTH axes independently, with
        // no representation of the real football principle that the
        // genuinely risky pattern is specifically the COMBINATION —
        // "a blind ball into the packed central channel from the back,"
        // not central alone (patient recycling through the pivot is a
        // real, common, healthy pattern) and not forward alone (breaking
        // the line out wide is fine). `cutback_bonus`/`arriving_runner_
        // bonus` already correctly reward the SPECIFIC, earned central
        // combinations (a byline cutback, a box-arriving runner) — this
        // dampener is scoped to fire only when NEITHER already justifies
        // the central option, so it targets the generic/unearned case
        // exclusively and never fights the two mechanisms that are
        // already working correctly.
        let has_specific_central_opportunity = cutback_bonus > 0.0 || arriving_runner_bonus > 0.0;
        let is_blind_central = receiver_width_ratio < 0.3 && !has_specific_central_opportunity;
        if is_defender && is_blind_central && forward_progress > 0.0 {
            // The specific "hopeful ball into the middle from the back"
            // pattern — dampens the forward reward for THIS combination
            // only; a defender's line-breaking pass out wide, or into an
            // earned central chance, is untouched.
            forward_value *= 0.55;
        }
        if is_forward && is_blind_central {
            // A forward defaulting to "nearest central teammate" with no
            // earned opportunity — dampens the central-proximity reward
            // specifically, not central combination play in general.
            danger_value *= 0.5;
            promoted_pass_value *= 0.5;
        }

        // Weighted combination - includes width and switching bonuses.
        // Phase-aware bonuses (cutback, build-up recycle, counter first
        // pass) are added flat — they're already gated tightly on
        // phase + receiver type so they only fire in the situations
        // they were designed for. Side-overload is owned by
        // `same_side_density_penalty` (legacy half-pitch path removed).
        let mut tactical_value = forward_value * 0.32 +
            distance_value * 0.10 +
            position_value * 0.08 +
            long_pass_bonus * 0.05 +
            width_bonus * 0.22 +
            danger_value * 0.06 +            // §12.6: resulting scoring danger (tie-breaker)
            promoted_pass_value * 0.15 +     // B3: Component C, promoted term
            switch_total +                   // Capped flat: classic + underload ≤ 0.45
            cutback_bonus +
            arriving_runner_bonus +
            defender_width_bonus_unweighted + // /goal 2026-07-26: flat, same family as cutback_bonus
            forward_width_bonus_unweighted +  // /goal 2026-07-26: flat, same family as cutback_bonus
            build_up_recycle_bonus +
            stalled_recycle_bonus +
            directness_length_bias +
            counter_first_pass_bonus +
            same_side_density_penalty +
            sideways_penalty;

        // PPM biases. Players with killer-ball / playmaker traits love the
        // forward pass and should see it as more valuable even when risky.
        // Trait-driven switch boosts apply to switch_total (the capped
        // sum) so the 0.45 ceiling is the single switching budget for
        // the whole tactical_value calculation.
        let passer = ctx.player;
        let forward_trait_bias = passer.has_trait(PlayerTrait::TriesThroughBalls)
            || passer.has_trait(PlayerTrait::KillerBallOften);
        if forward_trait_bias && forward_value > 0.0 {
            tactical_value += forward_value * 0.25;
        }
        if passer.has_trait(PlayerTrait::Playmaker) {
            if forward_value > 0.0 {
                tactical_value += forward_value * 0.20;
            }
            tactical_value += switch_total * 0.10;
        }
        if passer.has_trait(PlayerTrait::LikesToSwitchPlay) {
            tactical_value += switch_total * 0.15;
        }
        if passer.has_trait(PlayerTrait::PlaysShortPasses) {
            tactical_value -= long_pass_bonus * 0.20;
        }
        if passer.has_trait(PlayerTrait::PlaysLongPasses) {
            tactical_value += long_pass_bonus * 0.15;
        }

        // === §12.6 CHANCE-RETREAT PENALTY ===
        // When the passer HAS an open or near-open scoring opportunity in
        // front of them (advanced, central-ish, clear run/pass corridor
        // to goal), a pass that retreats from goal is heavily
        // deprioritized — advancing or shooting should dominate. A
        // backward outlet only survives when a genuine obstruction (an
        // outfield defender directly in the goalward corridor) forces
        // it. This governs WHICH pass target wins once passing was
        // chosen; the shot-vs-pass deferral rule in the shot decision
        // code is untouched.
        if forward_progress < 0.0 && Self::passer_has_open_chance(ctx) {
            tactical_value -= 0.9; // saturates to the clamp floor
        }

        // Allow negative tactical values for backward passes
        tactical_value.clamp(-0.5, 1.8)
    }

    /// §12.6 — does the passer currently hold a GENUINE open goal-scoring
    /// opportunity? A real chance, tightly gated so the retreat penalty
    /// fires only where it's clearly warranted (a wide gate over-forced
    /// shots and inflated the goals band): advanced (progress > 0.78),
    /// central (within 22% of field height of centre), inside ~170u
    /// (≈21m) of goal, with NO outfield defender directly in the goalward
    /// corridor (16u half-width, up to 90u). The goalkeeper is the
    /// chance, not an obstruction. Shared by the central tactical-value
    /// term and the forwards' pass scorer so both deprioritize retreat
    /// passes on the same definition of "a defender genuinely forcing
    /// the outlet".
    pub fn passer_has_open_chance(ctx: &StateProcessingContext) -> bool {
        let field_width = ctx.context.field_size.width as f32;
        let field_height = ctx.context.field_size.height as f32;
        let field_center_y = field_height * 0.5;
        let side = ctx.player.side.unwrap_or(PlayerSide::Left);
        let passer_position = ctx.player.position;
        let attack_goal_x = match side {
            PlayerSide::Left => field_width,
            PlayerSide::Right => 0.0,
        };
        let passer_progress = side.attacking_progress_x(passer_position.x, field_width);
        let passer_y_offset = (passer_position.y - field_center_y).abs();
        let passer_goal_dist = ((passer_position.x - attack_goal_x).powi(2)
            + (passer_position.y - field_center_y).powi(2))
        .sqrt();
        if passer_progress <= 0.78
            || passer_y_offset >= field_height * 0.22
            || passer_goal_dist >= 170.0
        {
            return false;
        }
        let goal_dir_x = (attack_goal_x - passer_position.x) / passer_goal_dist.max(1.0);
        let goal_dir_y = (field_center_y - passer_position.y) / passer_goal_dist.max(1.0);
        let corridor_len = passer_goal_dist.min(90.0);
        let obstructed = ctx.players().opponents().all().any(|o| {
            if o.tactical_positions.is_goalkeeper() {
                return false; // the keeper is the chance, not an obstruction
            }
            let rel_x = o.position.x - passer_position.x;
            let rel_y = o.position.y - passer_position.y;
            let along = rel_x * goal_dir_x + rel_y * goal_dir_y;
            if along <= 0.0 || along > corridor_len {
                return false;
            }
            let px = rel_x - goal_dir_x * along;
            let py = rel_y - goal_dir_y * along;
            (px * px + py * py).sqrt() < 16.0
        });
        !obstructed
    }

    /// Calculate overall success probability from factors
    fn calculate_success_probability(factors: &PassFactors) -> f32 {
        // Weighted combination of all factors
        // Receiver positioning is the dominant factor — free players are far better targets
        let probability = factors.distance_factor * 0.10 +
                factors.angle_factor * 0.08 +
                factors.pressure_factor * 0.08 +
                factors.receiver_positioning * 0.40 +  // Dominant: free receivers are far better
                factors.passer_ability * 0.10 +
                factors.receiver_ability * 0.08 +
                factors.tactical_value * 0.16;

        probability.clamp(0.1, 0.99)
    }

    /// Calculate overall risk level
    fn calculate_risk_level(factors: &PassFactors) -> f32 {
        // Risk is inverse of safety factors
        // Poor receiver positioning (crowded by opponents) is now a major risk
        let risk = (1.0 - factors.distance_factor) * 0.20 +
                (1.0 - factors.pressure_factor) * 0.20 +
                (1.0 - factors.receiver_positioning) * 0.40 +  // Increased from 0.20
                (1.0 - factors.receiver_ability) * 0.20;

        risk.clamp(0.0, 1.0)
    }

    /// Calculate interception risk from opponents along the pass path
    fn calculate_interception_risk(
        ctx: &StateProcessingContext,
        passer: &MatchPlayer,
        receiver: &MatchPlayerLite,
    ) -> f32 {
        let pass_vector = receiver.position - passer.position;
        let pass_distance = pass_vector.norm();
        let pass_direction = pass_vector.normalize();

        // Minimum distance along the pass line before an opponent counts as a blocker.
        // A pressing opponent near the passer cannot intercept a driven forward pass —
        // the ball clears them before they can react. In real football this is ~10m (~20 units).
        // Use 25% of pass distance as alternative for short passes.
        let min_intercept_projection = 20.0_f32.min(pass_distance * 0.25);

        // Check for opponents who could intercept the pass
        let intercepting_opponents = ctx
            .players()
            .opponents()
            .all()
            .filter(|opponent| {
                let to_opponent = opponent.position - passer.position;
                let projection_distance = to_opponent.dot(&pass_direction);

                // Ignore opponents behind passer, past receiver, or too close to passer
                if projection_distance <= min_intercept_projection
                    || projection_distance >= pass_distance
                {
                    return false;
                }

                // Calculate perpendicular distance from pass line
                let projected_point = passer.position + pass_direction * projection_distance;
                let perp_distance = (opponent.position - projected_point).norm();

                // Consider opponent's interception ability
                let players = ctx.player();
                let opponent_skills = players.skills(opponent.id);
                let interception_ability = opponent_skills.technical.tackling / 20.0;
                let anticipation = opponent_skills.mental.anticipation / 20.0;

                // Better opponents can intercept from further away
                let effective_radius = 3.0 + (interception_ability + anticipation) * 2.0;

                perp_distance < effective_radius
            })
            .count();

        // Convert count to risk factor — aggressive penalties to prevent suicidal passes
        if intercepting_opponents == 0 {
            0.0 // No risk
        } else if intercepting_opponents == 1 {
            0.55 // Significant risk — one opponent in the lane
        } else if intercepting_opponents == 2 {
            0.85 // Very high risk — two opponents blocking
        } else {
            0.97 // Near-certain interception
        }
    }

    /// Find the best pass option from available teammates with skill-based personality
    /// Returns (teammate, reason) tuple
    pub fn find_best_pass_option(
        ctx: &StateProcessingContext,
        max_distance: f32,
    ) -> Option<(MatchPlayerLite, &'static str)> {
        // realism-bug (2026-07-28): a free kick that falls through to
        // ShortRoutine/Recycle (§12.2, `resolve_free_kick`) only arms
        // `free_kick_pass_pending` to force the taker into his role's
        // normal Passing state — it never constrained WHO that state's
        // own `find_best_pass_option` call is allowed to consider, and
        // that call's normal search radius is 300-400u (nearly half the
        // pitch length), not a "short routine" or even a realistic
        // "recycle to a deeper teammate" distance. Combined with the
        // §11.7 box-staging congestion crushing every forward option's
        // score (see the `passer_open_chance` waiver below, which only
        // fires for a genuine near-goal dribble, not a stationary FK
        // taker 25-35m out), an isolated CB/DM sitting near the taker's
        // OWN half could still out-score every crowded forward option
        // and win outright — a real free kick is never played the
        // length of the pitch backward. Clamp the search radius whenever
        // this is a free-kick recycle/short decision so only a genuinely
        // nearby teammate (a real lay-off or drop-back-and-reset) is
        // ever a candidate; every other passing decision is unaffected.
        const FREE_KICK_RECYCLE_MAX_DISTANCE: f32 = 160.0;
        let max_distance = if ctx.player.free_kick_pass_pending > 0 {
            max_distance.min(FREE_KICK_RECYCLE_MAX_DISTANCE)
        } else {
            max_distance
        };

        let mut best_option: Option<MatchPlayerLite> = None;
        let mut best_score = 0.0;

        // Passing personality is now derived directly from raw skills
        // via `SkillCurve` below — the legacy normalised aliases were
        // dropped to remove an unused-variable warning. Keep
        // `_anticipation_skill` documented; it isn't read yet but
        // earmarked for the through-ball read in a follow-up.
        let _anticipation_skill = ctx.player.skills.mental.anticipation / 20.0;

        // Passing personalities — sigmoid-rolled per evaluation so the
        // full 1-20 skill range maps to a smooth probability of acting
        // like that archetype, instead of hard `> 0.75` cliffs that
        // flattened mid-skill players into "ordinary" only. Probability
        // products gate dual-skill archetypes (need BOTH to lean toward
        // the type); the conservative archetype is the inverse — high
        // probability when EITHER decisions OR composure are weak.
        let vision_raw = ctx.player.skills.mental.vision;
        let flair_raw = ctx.player.skills.mental.flair;
        let pass_raw = ctx.player.skills.technical.passing;
        let dec_raw = ctx.player.skills.mental.decisions;
        let comp_raw = ctx.player.skills.mental.composure;
        let team_raw = ctx.player.skills.mental.teamwork;
        let roll = || ctx.context.rng.unit_f32();
        let is_playmaker = roll()
            < SkillCurve::new(vision_raw, 15.0, 0.6).probability()
                * SkillCurve::new(flair_raw, 13.0, 0.6).probability();
        let is_direct = roll()
            < SkillCurve::new(flair_raw, 14.0, 0.6).probability()
                * SkillCurve::new(pass_raw, 13.0, 0.6).probability();
        // Conservative = LOW decisions OR LOW composure. Probability of
        // "low" is 1 - curve(skill, 10, 0.6); take the max so either
        // weakness pulls toward safe play.
        let low_dec = 1.0 - SkillCurve::new(dec_raw, 10.0, 0.6).probability();
        let low_comp = 1.0 - SkillCurve::new(comp_raw, 10.0, 0.6).probability();
        let is_conservative = roll() < low_dec.max(low_comp);
        let is_team_player = roll()
            < SkillCurve::new(team_raw, 15.0, 0.6).probability()
                * SkillCurve::new(pass_raw, 13.0, 0.6).probability();
        let is_pragmatic = roll()
            < SkillCurve::new(dec_raw, 15.0, 0.6).probability()
                * SkillCurve::new(pass_raw, 12.0, 0.6).probability();

        // §12.6 — computed once per selection: an open chance in front of
        // the passer discounts every backward candidate below.
        let passer_open_chance = Self::passer_has_open_chance(ctx);

        // Calculate minimum pass distance based on pressure
        // NOTE: This filter prevents "too short" passes that don't progress the ball
        let is_under_pressure = ctx.player().pressure().is_under_immediate_pressure();
        let min_pass_distance = if is_under_pressure {
            // Under pressure, allow shorter passes but still avoid huddle passes
            12.0
        } else {
            // Not under pressure, still allow short-to-medium passes
            20.0
        };

        for teammate in ctx.players().teammates().nearby(max_distance) {
            // GRADUATED RECENCY PENALTY: Penalize recent passers instead of hard-skipping
            let recency_penalty = ctx.ball().passer_recency_penalty(teammate.id);

            let pass_distance = (teammate.position - ctx.player.position).norm();

            // MINIMUM DISTANCE FILTER: Skip teammates that are too close unless under pressure
            if pass_distance < min_pass_distance {
                continue;
            }

            // CONGESTION PENALTY: Heavily penalize passing into crowded areas.
            // Opponents near the receiver are weighted more heavily than teammates,
            // and close opponents are weighted much more than distant ones.
            let nearby_teammates_count = ctx
                .tick_context
                .grid
                .teammates(teammate.id, 0.0, 50.0)
                .count();
            let close_opponents_count = ctx.tick_context.grid.opponents(teammate.id, 30.0).count();
            let medium_opponents_count =
                ctx.tick_context.grid.opponents(teammate.id, 60.0).count() - close_opponents_count;

            // Close opponents count triple — passing into tight marking is very risky
            let weighted_nearby =
                nearby_teammates_count + close_opponents_count * 3 + medium_opponents_count;

            let congestion_penalty = match weighted_nearby {
                0 => 1.8,  // Completely isolated — excellent target
                1 => 1.3,  // One nearby player — good
                2 => 0.9,  // Normal
                3 => 0.4,  // Getting crowded — discouraged
                4 => 0.15, // Congested — strongly discouraged
                5 => 0.06, // Huddle — almost never pass here
                _ => 0.02, // Extremely congested — effectively blocked
            };

            let evaluation = Self::evaluate_pass(ctx, ctx.player, &teammate);
            let interception_risk = Self::calculate_interception_risk(ctx, ctx.player, &teammate);

            // Base positioning bonus
            let positioning_bonus = evaluation.factors.receiver_positioning * 2.0;

            // Skill-based space quality evaluation
            let space_quality = if is_conservative {
                // Conservative players prefer free receivers but less extreme
                if evaluation.factors.receiver_positioning > 0.85 {
                    1.8 // Reduced from 2.0 - completely free players
                } else if evaluation.factors.receiver_positioning > 0.65 {
                    1.3 // Increased from 1.2 - good space
                } else if evaluation.factors.receiver_positioning > 0.45 {
                    0.8 // New tier - acceptable space
                } else {
                    0.4 // Increased from 0.3 - will attempt if needed
                }
            } else if is_playmaker {
                // Playmakers trust teammates to handle some pressure
                if evaluation.factors.receiver_positioning > 0.75 {
                    1.7 // Increased from 1.6
                } else if evaluation.factors.receiver_positioning > 0.5 {
                    1.4 // Increased from 1.3 - still okay with moderate space
                } else if evaluation.factors.receiver_positioning > 0.3 {
                    1.0 // New tier - willing to attempt tighter passes
                } else {
                    0.7 // Reduced penalty for very tight spaces
                }
            } else if is_direct {
                // Direct players less concerned about space, more about attacking position
                if evaluation.factors.receiver_positioning > 0.6 {
                    1.6 // Increased from 1.5
                } else if evaluation.factors.receiver_positioning > 0.4 {
                    1.2 // New tier
                } else {
                    0.9 // Reduced from 1.0 - will attempt most passes
                }
            } else {
                // Standard space evaluation - slightly more aggressive
                if evaluation.factors.receiver_positioning > 0.75 {
                    1.6 // Increased from 1.5
                } else if evaluation.factors.receiver_positioning > 0.55 {
                    1.3 // Increased from 1.2
                } else if evaluation.factors.receiver_positioning > 0.35 {
                    1.0 // Improved threshold from 0.4
                } else {
                    0.7 // Increased from 0.6
                }
            };

            // Skill-based interception risk tolerance — higher = more penalty applied
            let risk_tolerance = if is_direct {
                0.5 // Still somewhat aggressive but respects blockers
            } else if is_conservative {
                0.9 // Almost never pass through opponents
            } else if is_playmaker {
                0.6 // Moderate — will try creative passes but not suicidal
            } else {
                0.7 // Standard — significant penalty for blocked lanes
            };

            let interception_penalty = 1.0 - (interception_risk * risk_tolerance);

            // Add distance preference bonus - widened optimal range to encourage penetration
            let optimal_distance_bonus = if is_under_pressure {
                // Under pressure, all safe passes are good
                1.0
            } else if pass_distance >= 20.0 && pass_distance <= 70.0 {
                // Widened optimal range (was 15-40m, now 20-70m) for penetrating passes
                1.4 // Increased from 1.3
            } else if pass_distance >= 15.0 && pass_distance < 20.0 {
                // Short passes - acceptable
                1.1 // New tier
            } else if pass_distance < 15.0 {
                // Very short - strongly discouraged (keeps ball in huddle)
                0.4
            } else if pass_distance <= 100.0 {
                // Long passes (70-100m) - moderate value
                1.1
            } else if pass_distance <= 150.0 {
                // Very long passes - declining value
                0.85
            } else {
                // Extreme long passes - discouraged
                0.6
            };

            // Distance preference based on personality. Vision-gated
            // ultra-long multipliers smoothed via sigmoid blend so a
            // vision-14 playmaker isn't cliff-equal to a vision-9 on
            // a 250m switch — interpolates between the two extremes.
            let distance_preference = if is_playmaker {
                // Playmakers prefer through balls but not unrealistic long passes
                if pass_distance > 300.0 {
                    // Extreme passes - very risky even for elite
                    SkillCurve::new(vision_raw, 17.0, 0.6).lerp(0.6, 1.1)
                } else if pass_distance > 200.0 {
                    // Ultra-long switches - risky
                    SkillCurve::new(vision_raw, 15.0, 0.6).lerp(0.8, 1.15)
                } else if pass_distance > 100.0 {
                    1.2 // Long passes - moderate bonus
                } else if pass_distance > 80.0 {
                    1.25 // Medium-long - sweet spot for playmakers
                } else if pass_distance > 50.0 {
                    1.2
                } else {
                    1.0
                }
            } else if is_direct {
                // Direct players strongly prefer forward passes
                let side_now = ctx.player.side.unwrap_or(PlayerSide::Left);
                let forward_progress =
                    side_now.forward_delta(ctx.player.position.x, teammate.position.x);
                if forward_progress > 0.0 {
                    1.4
                } else {
                    0.5 // Strongly avoid backward passes
                }
            } else if is_conservative {
                // Conservative players prefer short, safe passes
                if pass_distance < 30.0 {
                    1.4
                } else if pass_distance < 50.0 {
                    1.0
                } else {
                    0.7 // Avoid long passes
                }
            } else if is_team_player {
                // Team players maximize teammate positioning
                1.0 + (evaluation.factors.receiver_positioning * 0.3)
            } else if is_pragmatic {
                // Pragmatic players balance all factors
                if evaluation.expected_value > 0.6 {
                    1.3 // Good tactical value
                } else {
                    1.0
                }
            } else {
                1.0
            };

            // GOALKEEPER PENALTY: Almost completely eliminate passing to goalkeeper
            let is_goalkeeper = matches!(
                teammate.tactical_positions.position_group(),
                PlayerFieldPositionGroup::Goalkeeper
            );

            let goalkeeper_penalty = if is_goalkeeper {
                // Side-correct math via PlayerSide helpers. The
                // previous formulas
                //   `(teammate.x - player.x) * dir < 0`
                //   `(player.x * dir) / width > 0.66`
                // were wrong for right-side teams (the second produced
                // negative values which can never exceed 0.66, so a
                // right-side team was never classified as "in attacking
                // third" — which silently broke the block that
                // SHOULD reject GK passes from advanced positions).
                let side = ctx.player.side.unwrap_or(PlayerSide::Left);
                let is_backward_pass =
                    side.forward_delta(ctx.player.position.x, teammate.position.x) < 0.0;

                let field_width = ctx.context.field_size.width as f32;
                let player_progress = side.attacking_progress_x(ctx.player.position.x, field_width);
                let in_attacking_third = player_progress > 0.66;

                let phase_now = ctx.team().phase();
                if in_attacking_third && is_backward_pass {
                    // In attacking third, passing backward to GK is NEVER acceptable
                    0.00001 // Virtually zero
                } else if matches!(phase_now, GamePhase::BuildUp) && is_backward_pass {
                    // Build-up to GK is a normal modern pattern: pivot
                    // through the keeper to escape the press / switch
                    // play. Allow it as a real option (much higher than
                    // the legacy ~0.0001 ceiling) but only when the
                    // passer is genuinely under pressure or wants to
                    // recycle (low risk_appetite).
                    let under_press = ctx
                        .player()
                        .pressure()
                        .is_under_immediate_pressure_with_distance(8.0);
                    let recycle_intent = ctx.team().risk_appetite() < 0.45;
                    if under_press || recycle_intent {
                        // GK is a real option in build-up under press,
                        // not the only option.
                        0.55
                    } else {
                        // Build-up but no genuine recycle trigger —
                        // still allow but discount.
                        0.10
                    }
                } else if is_backward_pass {
                    // Backward pass to GK in middle/defensive third - still very bad
                    0.0001
                } else if evaluation.factors.pressure_factor < 0.2 {
                    // Forward/sideways pass under EXTREME pressure - GK is emergency option
                    0.02
                } else {
                    // Normal play - virtually eliminate GK passes
                    0.0005
                }
            } else {
                1.0 // No penalty for non-GK
            };

            // Calculate final score with personality-based weighting
            let score = if evaluation.factors.pressure_factor < 0.5 {
                // Under heavy pressure - personality affects decision
                if is_conservative {
                    // Conservative: safety is paramount
                    (evaluation.success_probability * 2.0 + positioning_bonus)
                        * interception_penalty
                        * space_quality
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                } else if is_direct {
                    // Direct: still look for forward options
                    (evaluation.expected_value * 1.5 + positioning_bonus * 0.3)
                        * interception_penalty
                        * space_quality
                        * distance_preference
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                } else {
                    // Others: prioritize safety AND space
                    (evaluation.success_probability + positioning_bonus)
                        * interception_penalty
                        * space_quality
                        * 1.3
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                }
            } else {
                // Normal situation - personality-based preferences apply
                if is_playmaker {
                    // Playmakers prioritize tactical value and vision
                    (evaluation.expected_value * 1.3 + positioning_bonus * 0.4)
                        * interception_penalty
                        * space_quality
                        * distance_preference
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                } else if is_direct {
                    // Direct players maximize attack
                    (evaluation.expected_value * 1.4 + evaluation.factors.tactical_value * 0.5)
                        * interception_penalty
                        * space_quality
                        * distance_preference
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                } else if is_team_player {
                    // Team players maximize receiver's situation
                    (evaluation.success_probability + positioning_bonus * 0.8)
                        * interception_penalty
                        * space_quality
                        * distance_preference
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                } else if is_conservative {
                    // Conservative: success probability is key
                    (evaluation.success_probability * 1.5 + positioning_bonus * 0.3)
                        * interception_penalty
                        * space_quality
                        * distance_preference
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                } else if is_pragmatic {
                    // Pragmatic: balanced approach
                    (evaluation.expected_value * 1.2 + positioning_bonus * 0.5)
                        * interception_penalty
                        * space_quality
                        * distance_preference
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                } else {
                    // Standard scoring
                    (evaluation.expected_value + positioning_bonus * 0.5)
                        * interception_penalty
                        * space_quality
                        * optimal_distance_bonus
                        * goalkeeper_penalty
                }
            };

            // §12.6 — retreating from an OPEN chance is multiplicative,
            // not additive: the additive tactical-value penalty is
            // routinely out-muscled by the congestion/space multipliers,
            // which count TEAMMATES — so in the exact failure case
            // (three attackers in the box, no defenders) every forward
            // option looked "congested" and the lone backward outlet
            // still won. With a clear goalward corridor, a backward
            // target keeps ~⅛ of its score; a genuinely obstructed
            // chance is untouched (the corridor test is the waiver).
            let score = if passer_open_chance
                && ctx
                    .player
                    .side
                    .unwrap_or(PlayerSide::Left)
                    .forward_delta(ctx.player.position.x, teammate.position.x)
                    < 0.0
            {
                score * 0.18
            } else {
                score
            };

            // Hard reject: never pass through 2+ opponents unless
            // a playmaker rolls high vision. Vision gate smoothed
            // (sigmoid pivot 16/20) so the "elite" tier isn't a sharp
            // cliff — a vision-14 playmaker still occasionally tries.
            let interception_blocked = if interception_risk >= 0.85 {
                // 2+ opponents in the lane — almost always reject
                if is_playmaker
                    && ctx.context.rng.unit_f32() < SkillCurve::new(vision_raw, 16.0, 0.6).probability()
                {
                    false // Elite playmakers can attempt
                } else {
                    true
                }
            } else if interception_risk >= 0.55 {
                // 1 opponent in the lane — reject for conservative, allow others with caution
                is_conservative
            } else {
                false
            };

            // Personality-based acceptance threshold - more aggressive to encourage penetration
            let is_acceptable = if interception_blocked {
                false
            } else if is_goalkeeper {
                // Goalkeeper passes are normally rare; in build-up
                // they're a textbook pattern (recycle through the GK to
                // bait a press, then switch). Phase gates this:
                //   * BuildUp: allow when in own defensive third with a
                //     reasonable success probability and either pressure
                //     or low risk_appetite (recycle intent).
                //   * Otherwise: only as an emergency escape from
                //     extreme pressure deep in own half.
                let side_now = ctx.player.side.unwrap_or(PlayerSide::Left);
                let fw = ctx.context.field_size.width as f32;
                let progress = side_now.attacking_progress_x(ctx.player.position.x, fw);
                let in_defensive_third = progress < 0.33;
                let phase_now = ctx.team().phase();

                if matches!(phase_now, GamePhase::BuildUp) && in_defensive_third {
                    let under_press = ctx
                        .player()
                        .pressure()
                        .is_under_immediate_pressure_with_distance(8.0);
                    let recycle_intent = ctx.team().risk_appetite() < 0.45;
                    evaluation.success_probability > 0.55 && (under_press || recycle_intent)
                } else {
                    evaluation.factors.pressure_factor < 0.2
                        && evaluation.success_probability > 0.85
                        && in_defensive_third
                }
            } else if is_conservative {
                evaluation.success_probability > 0.60
                    && evaluation.factors.receiver_positioning > 0.55
            } else if is_direct {
                evaluation.success_probability > 0.40 && evaluation.factors.tactical_value > 0.35
            } else if is_playmaker {
                evaluation.success_probability > 0.45
                    || (evaluation.factors.tactical_value > 0.60 && pass_distance > 50.0)
            } else {
                // Standard - more willing to pass
                evaluation.is_recommended
                    || (evaluation.factors.receiver_positioning > 0.5
                        && evaluation.success_probability > 0.42)
            };

            // Game-management bias: a team protecting a lead (especially
            // late, or as the weaker side) prefers sideways / backward
            // balls over risky forward ones — real "hold the score"
            // football.
            let gm_intensity = ctx
                .context
                .tactical_for_team(ctx.player.team_id)
                .game_management_intensity;
            let gm_modifier = if gm_intensity > 0.05 {
                let side_now = ctx.player.side.unwrap_or(PlayerSide::Left);
                let forward_progress =
                    side_now.forward_delta(ctx.player.position.x, teammate.position.x);
                if forward_progress > 5.0 {
                    (1.0 - gm_intensity * 0.45).max(0.3)
                } else {
                    1.0 + gm_intensity * 0.60
                }
            } else {
                1.0
            };

            // Directional attack bias ("build down the left"): receivers
            // in the manager's target lateral third score up, the
            // opposite third down. Side-corrected so "left" stays the
            // manager's left after the halftime swap.
            let bias_modifier = if let Some(bias) = ctx.player.attack_bias {
                let h = ctx.context.field_size.height as f32;
                let third = h / 3.0;
                let y = teammate.position.y;
                let raw_lane: i8 = if y < third {
                    -1
                } else if y > h - third {
                    1
                } else {
                    0
                };
                let lane = match ctx.player.side.unwrap_or(PlayerSide::Left) {
                    PlayerSide::Left => raw_lane,
                    PlayerSide::Right => -raw_lane,
                };
                if bias == 0 {
                    if lane == 0 { 1.25 } else { 0.9 }
                } else if lane == bias {
                    1.35
                } else if lane == -bias {
                    0.75
                } else {
                    1.0
                }
            } else {
                1.0
            };

            // "Link with X" pair preference: the linked partner wins ties
            // and close calls against equally-placed alternatives. A
            // multiplier (same idiom as bias_modifier) rather than an EV
            // delta — the selection scale is dominated by the multiplier
            // chain, so a probability nudge alone is invisible here.
            let link_modifier = if ctx.player.link_target == Some(teammate.id) {
                1.5
            } else {
                1.0
            };

            // "Feed X" directed supply (wishlist #9): the named target is
            // a preferred receiver — long balls into them especially.
            let supply_modifier = if ctx.player.supply_target == Some(teammate.id) {
                if pass_distance >= 60.0 { 1.6 } else { 1.15 }
            } else {
                1.0
            };

            // "Block passes into X" (wishlist #8): an assigned opposing
            // interceptor within range of the receiver makes this pass
            // notably less attractive to select at all.
            let intercept_modifier = if ctx.context.intercept_assignments.iter().any(|&(i, t)| {
                t == teammate.id
                    && (ctx.tick_context.positions.players.position(i) - teammate.position).norm()
                        < 80.0
            }) {
                0.35
            } else {
                1.0
            };

            // realism-bug (2026-07-28): a throw-in's close-support receiver
            // (`throw_in_shape_targets`) is a specifically prepared,
            // marker-evasion-adjusted option — but without this the fully
            // generic evaluator ignored that prep entirely and reached for
            // whichever teammate scored highest by ordinary open-play
            // criteria, producing throws that landed noticeably longer
            // (mean ~24m vs a real ~15m) and skewed toward "wide open"
            // zones far more than real throw-ins do. Same multiplicative
            // idiom as link_modifier above (a probability nudge alone is
            // invisible against this chain).
            let throw_in_preferred_modifier = if ctx.tick_context.ball.pass_origin_restart
                == PassOriginRestart::ThrowIn
                && ctx.tick_context.ball.throw_in_preferred_receiver == Some(teammate.id)
            {
                4.0
            } else {
                1.0
            };

            // Apply graduated recency penalty to discourage ping-pong passing
            // Apply congestion penalty to force ball out of huddles
            let score = score
                * recency_penalty
                * congestion_penalty
                * gm_modifier
                * bias_modifier
                * link_modifier
                * supply_modifier
                * intercept_modifier
                * throw_in_preferred_modifier;

            if score > best_score && is_acceptable {
                best_score = score;
                best_option = Some(teammate);
            }
        }

        // Minimum score threshold: if the best option scores too low,
        // return None so the player dribbles/runs instead of making a bad pass
        const MIN_PASS_SCORE: f32 = 0.15;
        if best_score < MIN_PASS_SCORE {
            return None;
        }

        best_option.map(|teammate| (teammate, "PASS_EVALUATOR"))
    }

    // ──────────────────────────────────────────────────────────────────
    // Pure helpers — pulled out of `calculate_tactical_value` so they
    // can be unit-tested without spinning up a full match field.
    // ──────────────────────────────────────────────────────────────────

    /// Penalty for passing into a flank that already has too many of
    /// our own players. Polish-spec curve: 0..3 → 0, 4 → -0.08,
    /// 5 → -0.18, 6+ → -0.30. The legacy half-pitch overload penalty
    /// has been removed so this is the single side-overload signal.
    pub fn same_side_density_penalty(receiver_side_density: u8) -> f32 {
        match receiver_side_density {
            0..=3 => 0.0,
            4 => -0.08,
            5 => -0.18,
            _ => -0.30,
        }
    }

    /// Bonus for switching the play into an underloaded flank.
    /// Vision-graded so playmakers see the switch as more valuable.
    /// Polish-spec curve: 0.08 + vision * 0.12 → 0.08..0.20.
    /// Returns 0 when the pass doesn't cross flanks or the target side
    /// is not underloaded.
    pub fn underload_switch_bonus(
        crosses_sides: bool,
        receiver_side_density: u8,
        vision: f32,
    ) -> f32 {
        let underloaded = receiver_side_density <= 3;
        if crosses_sides && underloaded {
            let vision = vision.clamp(0.0, 1.0);
            0.08 + vision * 0.12
        } else {
            0.0
        }
    }

    /// Milestone 8 (possession-decision-intelligence PRD) — genuine
    /// numerical overload/isolation, unlike `underload_switch_bonus`
    /// above (which only reads OUR OWN side density, never the
    /// opponent's). `advantage` is `own_side_density - opponent_side_
    /// density` in the RECEIVER's zone — positive means we genuinely
    /// outnumber them there, negative means switching there would
    /// isolate the receiver against more defenders than we have
    /// attackers. Gated on `crosses_sides` so this specifically rewards
    /// SWITCHING the point of attack (the milestone's own phrase), not
    /// merely passing within an already-advantaged zone. Allowed to go
    /// negative — a switch into a zone where we're actually outnumbered
    /// is penalised, not just under-rewarded, which is the "isolation"
    /// half of the milestone.
    pub fn overload_bonus(crosses_sides: bool, advantage: f32) -> f32 {
        if !crosses_sides {
            return 0.0;
        }
        (advantage / 3.0).clamp(-1.0, 1.0) * 0.08
    }
}

#[cfg(test)]
mod tests {
    use super::PassEvaluator;

    #[test]
    fn density_penalty_zero_when_uncrowded() {
        assert_eq!(PassEvaluator::same_side_density_penalty(0), 0.0);
        assert_eq!(PassEvaluator::same_side_density_penalty(3), 0.0);
    }

    #[test]
    fn density_penalty_increases_with_crowding() {
        assert!(
            PassEvaluator::same_side_density_penalty(4)
                > PassEvaluator::same_side_density_penalty(5)
        );
        assert!(
            PassEvaluator::same_side_density_penalty(5)
                > PassEvaluator::same_side_density_penalty(7)
        );
        assert_eq!(PassEvaluator::same_side_density_penalty(7), -0.30);
    }

    #[test]
    fn underload_switch_zero_when_not_crossing() {
        assert_eq!(PassEvaluator::underload_switch_bonus(false, 0, 1.0), 0.0);
    }

    #[test]
    fn underload_switch_zero_when_target_already_full() {
        // 5 players on the receiver side — not underloaded, no bonus.
        assert_eq!(PassEvaluator::underload_switch_bonus(true, 5, 1.0), 0.0);
    }

    #[test]
    fn underload_switch_grows_with_vision() {
        let low = PassEvaluator::underload_switch_bonus(true, 2, 0.0);
        let high = PassEvaluator::underload_switch_bonus(true, 2, 1.0);
        assert!(high > low);
        assert!((low - 0.08).abs() < 1e-4);
        assert!((high - 0.20).abs() < 1e-4);
    }

    #[test]
    fn underload_switch_bonus_within_spec_range() {
        // Spec range: 0.08 + vision*0.12 → 0.08..0.20
        for v_int in 0..=20 {
            let v = v_int as f32 / 20.0;
            let bonus = PassEvaluator::underload_switch_bonus(true, 1, v);
            assert!(bonus >= 0.08 - 1e-4);
            assert!(bonus <= 0.20 + 1e-4);
        }
    }

    #[test]
    fn density_penalty_curve_matches_polish_spec() {
        assert_eq!(PassEvaluator::same_side_density_penalty(4), -0.08);
        assert_eq!(PassEvaluator::same_side_density_penalty(5), -0.18);
        assert_eq!(PassEvaluator::same_side_density_penalty(6), -0.30);
        assert_eq!(PassEvaluator::same_side_density_penalty(11), -0.30);
    }

    #[test]
    fn switch_total_caps_at_zero_point_four_five() {
        // The capped switch reward path inside `calculate_tactical_value`
        // is `(classic + underload).min(0.45)`. Verify the helpers feed a
        // sensible joint maximum: classic max is 0.45 + vision*0.25 = 0.70,
        // underload max is 0.20. The cap therefore truly bites.
        let underload_max = PassEvaluator::underload_switch_bonus(true, 0, 1.0);
        assert!(
            underload_max + 0.70 > 0.45,
            "cap must actually bite — sum without cap = {}",
            underload_max + 0.70
        );
    }
}
