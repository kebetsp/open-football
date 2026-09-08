use crate::r#match::events::Event;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::midfielders::states::common::{ActivityIntensity, MidfielderCondition};
use crate::r#match::player::events::{FoulSeverity, PlayerEvent, ShootingEventContext};
use crate::r#match::player::strategies::common::players::MatchPlayerIteratorExt;
use crate::r#match::player::strategies::common::players::ops::defensive_value as dv;
use crate::r#match::player::strategies::common::players::ops::midfielder_skill::MidfielderSkillProfile;
use crate::r#match::player::strategies::players::DefensiveRole;
use crate::r#match::{
    ConditionContext, MatchPlayerLite, StateChangeResult, StateProcessingContext,
    StateProcessingHandler,
};
use nalgebra::Vector3;
#[cfg(feature = "match-logs")]
use std::sync::atomic::Ordering;

#[derive(Default, Clone)]
pub struct MidfielderPressingState {}

impl StateProcessingHandler for MidfielderPressingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        if ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // Ball coming toward this player (pass to us) — intercept it
        if ctx.ball().is_towards_player_with_angle(0.8) && ctx.ball().distance() < 150.0 {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Intercepting,
            ));
        }

        // Loose ball very close — take it regardless of speed
        if !ctx.ball().is_owned() && ctx.ball().distance() < 30.0 {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::TakeBall,
            ));
        }

        // If team has possession, stop pressing and contribute to attack
        if ctx.team().is_control_ball() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::AttackSupporting,
            ));
        }

        // Opponent restart (GK holding or goal kick) — retreat to start position.
        if ctx.ball().is_opponent_restart() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Returning,
            ));
        }

        // Press time budget: combines the team-shared press intensity
        // with the midfielder's own pressing_profile (work_rate/stamina/
        // anticipation/positioning blend, dampened by condition). Poor
        // pressers and tired midfielders give up earlier; elite pressers
        // sustain longer hunts.
        let intensity = ctx.team().press_intensity();
        let mid_profile = MidfielderSkillProfile::from_ctx(ctx);
        let max_press_time = mid_profile.press_time_ticks(intensity);
        if ctx.in_state_time > max_press_time {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // Back off during foul protection — don't crowd the free kick
        if ctx.ball().is_in_flight() && ctx.ball().is_owned() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        // Early exit if a teammate is significantly closer to avoid circular running
        let ball_distance = ctx.ball().distance();
        let ball_position = ctx.tick_context.positions.ball.position;

        if let Some(closest_teammate) = ctx
            .players()
            .teammates()
            .all()
            .filter(|t| t.id != ctx.player.id)
            .min_by(|a, b| {
                let dist_a = (a.position - ball_position).magnitude();
                let dist_b = (b.position - ball_position).magnitude();
                dist_a.total_cmp(&dist_b)
            })
        {
            let teammate_distance = (closest_teammate.position - ball_position).magnitude();

            // If teammate is closer by 10+ units, give up pressing
            if teammate_distance < ball_distance - 10.0 {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Running,
                ));
            }
        }

        // CRITICAL: Tackle if an opponent has the ball nearby
        if let Some(opponent) = ctx.players().opponents().nearby(60.0).with_ball(ctx).next() {
            let opponent_distance = (opponent.position - ctx.player.position).magnitude();

            // Engage tackle aggressively — midfielders must win the ball.
            // Realism-bug 2026-07-27: aligned with MidfielderTacklingState's
            // own TACKLE_DISTANCE_THRESHOLD (8u). At 50u this was a 42u dead
            // zone -- Tackling::process() immediately found opponent_distance
            // > 8u and bounced back to Pressing every tick, using Tackling's
            // weaker steering instead of Pressing's own chase logic for the
            // whole close-in. Measured entries:attempts was ~1000:1 (worse
            // than defenders' ~300:1) before this fix.
            //
            // realism-bug (2026-07-30): that fix alone reintroduced the
            // exact 1-tick thrash it was meant to remove, just gated on
            // cooldown instead of distance — a midfielder on
            // `tackle_cooldown` still satisfies `opponent_distance < 8.0`
            // every tick, gets bounced back out of Tackling by its own
            // cooldown check, and immediately re-enters here next tick.
            // Raw traces showed this running every single 30ms sample for
            // 18+ seconds in lockstep with an opposing defender doing the
            // same. Requiring `can_attempt_tackle()` here lets a cooldown
            // midfielder fall through to normal Pressing chase/cover
            // behaviour instead of flickering in place.
            // D8.2 of docs/defensive-decision-logic-spec.md: gate the
            // immediate engage on the unified role, mirroring
            // DefenderRunningState's own already-shipped "only the
            // designated closer tackles" fix (D2) — the same pileup
            // risk applies here now that a genuinely closer defender or
            // another midfielder can outrank this one in the shared
            // pool. Non-Primary falls through to the normal chase below
            // (unchanged) rather than lunging.
            let role = ctx
                .player()
                .defensive()
                .defensive_role_for_ball_carrier_unified();
            if opponent_distance < 8.0
                && role == DefensiveRole::Primary
                && ctx.player.can_attempt_tackle()
            {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Tackling,
                ));
            }

            // D5 of docs/defensive-decision-logic-spec.md: generalizes
            // D4.6's last-ditch slide from DefenderPressingState — a
            // midfielder tracking back on a counter, still trailing at
            // 8-34u out, is exactly the same real scenario (a normal
            // running stride probably loses the race; a committed slide
            // covers ground a stride can't). Mirrors the defender wiring
            // exactly: checked here in the chase itself, before the
            // 8.0u tackle-transition above (which assumes arrival), with
            // the same `can_attempt_tackle()`/`is_best_player_to_chase_
            // ball()` gates so it can't fire on repeat or pile up
            // multiple midfielders on the same ball.
            if opponent_distance >= 8.0
                && opponent_distance <= dv::SLIDE_REACH_DISTANCE
                && ctx.player.can_attempt_tackle()
                && ctx.team().is_best_player_to_chase_ball()
            {
                let mid_profile = MidfielderSkillProfile::from_ctx(ctx);
                if dv::desperate_slide_value(ctx, &opponent, mid_profile.tackle_profile) > 0.0 {
                    return Some(self.attempt_desperate_slide(ctx, &opponent));
                }
            }
        }

        // Loose ball nearby — go claim it directly instead of pressing thin air
        // But only if no teammate is closer (avoid chasing ball during teammate's pass)
        if !ctx.ball().is_owned() && ball_distance < 50.0 && ctx.ball().speed() < 3.0 {
            let ball_pos = ctx.tick_context.positions.ball.position;
            let cutoff = ball_distance - 5.0;
            let closer_teammate = cutoff > 0.0
                && ctx
                    .players()
                    .teammates()
                    .nearby_at(ball_pos, cutoff)
                    .next()
                    .is_some();

            if !closer_teammate {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::TakeBall,
                ));
            }
        }

        // Only give up pressing if ball is truly far away
        if ctx.ball().distance() > 300.0 {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Returning,
            ));
        }

        // Check if the pressing is ineffective (opponent still has ball after some time)
        if ctx.in_state_time > 30 && !self.is_making_progress(ctx) {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Running,
            ));
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Only pursue if opponent has the ball
        if let Some(opponent) = ctx
            .players()
            .opponents()
            .nearby(500.0)
            .with_ball(ctx)
            .next()
        {
            let distance_to_opponent = (opponent.position - ctx.player.position).magnitude();

            // Predictive pursuit — lead the carrier based on their
            // current velocity. Aiming at their present position means
            // arriving several ticks behind a running attacker. Same
            // fix applied to defender pressing: estimate how long we'd
            // take to close the gap at our own pace, project the
            // carrier along their velocity by that many ticks, then bias
            // the aim point slightly toward our own goal to close the
            // shooting lane rather than just touch their back.
            let opp_velocity = ctx.tick_context.positions.players.velocity(opponent.id);
            let opp_speed = opp_velocity.magnitude();
            let pace = ctx.player.skills.physical.pace;
            let lead_ticks = if pace > 0.01 {
                (distance_to_opponent / pace).min(30.0)
            } else {
                0.0
            };
            let predicted = opponent.position + opp_velocity * lead_ticks;
            let own_goal = ctx.ball().direction_to_own_goal();
            let to_own_goal = (own_goal - predicted).normalize();
            let goalside_bias = if opp_speed > 0.1 { 2.0 } else { 0.0 };
            // D1 (docs/attacking-defensive-decision-rebuild-plan.md §2.3)
            // — jockey/"show outside": see the identical comment in
            // defenders/states/pressing/mod.rs for the doctrine.
            let field_height = ctx.context.field_size.height as f32;
            let jockey_sign = dv::jockey_show_outside_sign(
                predicted.y,
                field_height,
                ctx.player.defensive_show_outside,
            );
            let jockey_offset = Vector3::new(0.0, jockey_sign * dv::JOCKEY_SHOW_OUTSIDE_BIAS, 0.0);

            // docs/defensive-delay-and-engagement-timing.md, Stage 3
            // Chunk 2 (docs/positional-football-master-plan.md,
            // 2026-09-07): mirrors DefenderPressingState's own Chunk 1
            // delay/jockey phase (same doc, same sourced doctrine) —
            // extends the fix to midfielders specifically because
            // Pavel's own complaint named the midfield collapse as a
            // direct consequence of the defensive line losing duels too
            // early ("if defenders are not even defending, where do you
            // think the midfield is? Also somewhere behind the
            // attacking midfield because they lost their duel"). Same
            // 36u outer / 8u inner zone (8u matches this state's own
            // MidfielderTacklingState engagement threshold, used inline
            // a few lines above in process() rather than a shared
            // constant), same 20u hold distance, same three triggers.
            const JOCKEY_ZONE_OUTER: f32 = 36.0;
            const JOCKEY_ZONE_INNER: f32 = 8.0;
            const JOCKEY_HOLD_DISTANCE: f32 = 20.0;
            const JOCKEY_DANGER_OVERRIDE: f32 = 0.15;
            const JOCKEY_HEAVY_TOUCH_GAP: f32 = 12.0;
            let jockey_field_height = ctx.context.field_size.height as f32;
            let touchline_margin = jockey_field_height * 0.12;
            let carrier_near_touchline = opponent.position.y < touchline_margin
                || opponent.position.y > jockey_field_height - touchline_margin;
            let ball_carrier_gap =
                (ctx.tick_context.positions.ball.position - opponent.position).magnitude();
            let danger = dv::value_conceded_if_beaten(ctx, &opponent);
            // 4th delay-exit trigger, added Stage 4 (docs/defensive-
            // delay-and-engagement-timing.md's own open "cover has
            // arrived" question, docs/positional-football-master-plan.md
            // 2026-09-07): mirrors DefenderPressingState's own Chunk-1
            // wiring exactly — see that file's comment for the full
            // doctrine.
            let cover_ready = ctx.player().defensive().cover_is_ready();
            let should_jockey = distance_to_opponent > JOCKEY_ZONE_INNER
                && distance_to_opponent <= JOCKEY_ZONE_OUTER
                && danger < JOCKEY_DANGER_OVERRIDE
                && !carrier_near_touchline
                && ball_carrier_gap < JOCKEY_HEAVY_TOUCH_GAP
                && !cover_ready;

            if should_jockey {
                let jockey_target = opponent.position + to_own_goal * JOCKEY_HOLD_DISTANCE;
                let jockey_velocity = crate::r#match::SteeringBehavior::Arrive {
                    target: jockey_target,
                    slowing_distance: 18.0,
                }
                .calculate(ctx.player)
                .velocity;
                let jockey_separation = ctx.player().separation_velocity() * 0.15;
                crate::match_log_info!(
                    "DELAYJOCKEY hold id={} tick={} dist={:.1} danger={:.2} gap={:.1} cover_ready={}",
                    ctx.player.id,
                    ctx.context.current_tick(),
                    distance_to_opponent,
                    danger,
                    ball_carrier_gap,
                    cover_ready
                );
                return Some(jockey_velocity + jockey_separation);
            } else if cover_ready
                && distance_to_opponent > JOCKEY_ZONE_INNER
                && distance_to_opponent <= JOCKEY_ZONE_OUTER
            {
                crate::match_log_info!(
                    "DELAYJOCKEY cover_arrived_commit id={} dist={:.1} danger={:.2}",
                    ctx.player.id,
                    distance_to_opponent,
                    danger
                );
            }

            let intercept_target = predicted + to_own_goal * goalside_bias + jockey_offset;

            let to_target = intercept_target - ctx.player.position;
            let direction = if to_target.magnitude() > 0.01 {
                to_target.normalize()
            } else {
                Vector3::zeros()
            };
            let speed = pace;

            let pressing_velocity = direction * speed;

            // Reduce separation when actively pressing to allow close approach
            let separation = if distance_to_opponent < 25.0 {
                ctx.player().separation_velocity() * 0.05
            } else {
                ctx.player().separation_velocity() * 0.15
            };

            Some(pressing_velocity + separation)
        } else if !ctx.ball().is_owned() && ctx.ball().distance() < 100.0 {
            // Loose ball — pursue it directly
            let direction =
                (ctx.tick_context.positions.ball.position - ctx.player.position).normalize();
            let speed = ctx.player.skills.physical.pace;
            Some(direction * speed)
        } else {
            // Teammate has ball — maintain position
            Some(Vector3::zeros())
        }
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Pressing is high intensity - sustained running and pressure
        MidfielderCondition::with_velocity(ActivityIntensity::High).process(ctx);
    }
}

impl MidfielderPressingState {
    // New helper function to determine if pressing is making progress
    fn is_making_progress(&self, ctx: &StateProcessingContext) -> bool {
        let player_velocity = ctx.player.velocity;

        // Calculate dot product between player velocity and direction to ball
        let to_ball = ctx.tick_context.positions.ball.position - ctx.player.position;
        let to_ball_normalized = if to_ball.magnitude() > 0.0 {
            to_ball / to_ball.magnitude()
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        };

        let moving_toward_ball = player_velocity.dot(&to_ball_normalized) > 0.0;

        // Check if other teammates are better positioned to press
        let other_pressing_teammates = ctx
            .players()
            .teammates()
            .all()
            .filter(|t| {
                let dist = (t.position - ctx.tick_context.positions.ball.position).magnitude();
                dist < ctx.ball().distance() * 0.8 // 20% closer than current player
            })
            .count();

        // Continue pressing if moving toward ball and not many better-positioned teammates
        moving_toward_ball && other_pressing_teammates < 2
    }

    /// D5 of docs/defensive-decision-logic-spec.md: midfielder mirror of
    /// `DefenderPressingState::attempt_desperate_slide` — rolls the same
    /// shared `dv::slide_outcome_probs` table (Success/Partial/Foul/
    /// OwnRisk/Miss) `desperate_slide_value` used to decide this was
    /// worth trying, so the decision and the roll can't drift apart. The
    /// dispatched events (`"DEF_SLIDE_SUCCESS"` etc.) are handled by the
    /// same generic `handle_shoot_event`/`CommitFoul` branches the
    /// defender wiring already uses (players.rs) — those branches key
    /// off `shoot_event_model.from_player_id`/`ctx.player.id`, not the
    /// player's role, so no execution-side changes were needed for this
    /// generalization.
    fn attempt_desperate_slide(
        &self,
        ctx: &StateProcessingContext,
        opponent: &MatchPlayerLite,
    ) -> StateChangeResult {
        #[cfg(feature = "match-logs")]
        crate::tackle_stats::MID_SLIDE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);

        let tackle_profile = MidfielderSkillProfile::from_ctx(ctx).tackle_profile;
        let probs = dv::slide_outcome_probs(ctx, opponent, tackle_profile);
        let rng = &ctx.context.rng;
        let roll = rng.random_range(0.0f32..1.0);

        let p_success = probs.p_success;
        let p_partial = p_success + probs.p_partial;
        let p_foul = p_partial + probs.p_foul;
        let p_own_risk = p_foul + probs.p_own_risk;
        // remainder (roll >= p_own_risk) is Miss.

        let mut result = if roll < p_success {
            // D6 (2026-08-22): same distinction as the defender wiring —
            // "tackle to get the ball" and "tackle to push the ball
            // away" are two different decisions. Push toward a boundary
            // only if the reachability race actually favours it,
            // otherwise a safe lay-off to the nearest teammate.
            let push_to_boundary = dv::slide_should_push_to_boundary(ctx);
            #[cfg(feature = "match-logs")]
            if push_to_boundary {
                crate::tackle_stats::MID_SLIDE_SUCCESS.fetch_add(1, Ordering::Relaxed);
            } else {
                crate::tackle_stats::MID_SLIDE_SUCCESS_LAYOFF.fetch_add(1, Ordering::Relaxed);
            }
            let reason = if push_to_boundary {
                "DEF_SLIDE_SUCCESS"
            } else {
                "DEF_SLIDE_SUCCESS_LAYOFF"
            };
            StateChangeResult::with_midfielder_state_and_event(
                MidfielderState::Standing,
                Event::PlayerEvent(PlayerEvent::Shoot(
                    ShootingEventContext::new()
                        .with_player_id(ctx.player.id)
                        .with_target(ctx.tick_context.positions.ball.position)
                        .with_reason(reason)
                        .build(ctx),
                )),
            )
        } else if roll < p_partial {
            #[cfg(feature = "match-logs")]
            crate::tackle_stats::MID_SLIDE_PARTIAL.fetch_add(1, Ordering::Relaxed);
            StateChangeResult::with_midfielder_state_and_event(
                MidfielderState::Standing,
                Event::PlayerEvent(PlayerEvent::Shoot(
                    ShootingEventContext::new()
                        .with_player_id(ctx.player.id)
                        .with_target(ctx.tick_context.positions.ball.position)
                        .with_reason("DEF_SLIDE_PARTIAL")
                        .build(ctx),
                )),
            )
        } else if roll < p_foul {
            #[cfg(feature = "match-logs")]
            crate::tackle_stats::MID_SLIDE_FOUL.fetch_add(1, Ordering::Relaxed);
            let severity = if rng.random_range(0.0f32..1.0) < 0.03 {
                FoulSeverity::Violent
            } else if rng.random_range(0.0f32..1.0) < 0.35 {
                FoulSeverity::Reckless
            } else {
                FoulSeverity::Normal
            };
            StateChangeResult::with_midfielder_state_and_event(
                MidfielderState::Standing,
                Event::PlayerEvent(PlayerEvent::CommitFoul(ctx.player.id, severity)),
            )
        } else if roll < p_own_risk {
            #[cfg(feature = "match-logs")]
            crate::tackle_stats::MID_SLIDE_OWN_RISK.fetch_add(1, Ordering::Relaxed);
            StateChangeResult::with_midfielder_state_and_event(
                MidfielderState::Standing,
                Event::PlayerEvent(PlayerEvent::Shoot(
                    ShootingEventContext::new()
                        .with_player_id(ctx.player.id)
                        .with_target(ctx.tick_context.positions.ball.position)
                        .with_reason("DEF_SLIDE_OWN_RISK")
                        .build(ctx),
                )),
            )
        } else {
            #[cfg(feature = "match-logs")]
            crate::tackle_stats::MID_SLIDE_MISS.fetch_add(1, Ordering::Relaxed);
            StateChangeResult::with_midfielder_state(MidfielderState::Standing)
        };
        result.start_tackle_cooldown = true;
        result
    }
}
