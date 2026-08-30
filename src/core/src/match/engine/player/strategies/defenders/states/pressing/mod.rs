use crate::r#match::defenders::states::DefenderState;
use crate::r#match::defenders::states::common::{ActivityIntensity, DefenderCondition};
use crate::r#match::events::Event;
use crate::r#match::player::events::{FoulSeverity, PlayerEvent, ShootingEventContext};
use crate::r#match::player::strategies::common::players::ops::defender_skill::DefenderSkillProfile;
use crate::r#match::player::strategies::common::players::ops::defensive::DefensiveOperationsImpl;
use crate::r#match::player::strategies::common::players::ops::defensive_value as dv;
use crate::r#match::player::strategies::players::DefensiveRole;
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::{
    ConditionContext, MatchPlayerLite, StateChangeResult, StateProcessingContext,
    StateProcessingHandler,
};
use nalgebra::Vector3;
#[cfg(feature = "match-logs")]
use std::sync::atomic::Ordering;

// Realism-bug 2026-07-27: aligned with DefenderTacklingState's own
// TACKLE_DISTANCE_THRESHOLD (10u). At 25u, a defender entering Tackling
// found distance_to_opponent > 10u on the very next check and bounced
// straight back to Pressing -- a 15u dead zone where the state machine
// thrashed every tick and used Tackling's weaker (non-predictive) chase
// steering instead of Pressing's lead-time+goalside-bias closing
// algorithm. Measured: DEF_GATE_NOT_IN_RANGE fired 1800-4700 times/match
// (vs 5-18 real attempts) because of this exact mismatch.
//
// 2026-08-30 (engine-audit Module 6): the local constant this fix
// introduced is now `DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE`
// — the same mismatch, independently reintroduced, was found live in 6
// more defender states that never picked up this fix. One shared source
// now backs every transition site instead of 8 separate copies.
const BASE_PRESSING_DISTANCE: f32 = 45.0;
const MAX_PRESSING_BONUS: f32 = 35.0; // effective range: 45-80
const BASE_PRESSING_DISTANCE_DEFENSIVE_THIRD: f32 = 40.0;
const MAX_PRESSING_BONUS_DEFENSIVE_THIRD: f32 = 30.0; // effective range: 40-70
const CLOSE_PRESSING_DISTANCE: f32 = 25.0; // Wider close pressing zone for tight approach
const STAMINA_THRESHOLD: f32 = 25.0; // Press until truly exhausted. Lowered
// from 30% to match hysteresis with Resting's 45% crisis re-entry
// gate (see `defenders/resting/mod.rs`) — the 25%–45% band is a
// "stay put, slow walk" zone that prevents Pressing↔Resting flicker
// when a crisis stays active while the defender is exhausted.
const FIELD_THIRD_THRESHOLD: f32 = 0.33;

#[derive(Default, Clone)]
pub struct DefenderPressingState {}

impl StateProcessingHandler for DefenderPressingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        // 1. Skill-aware fatigue gate. Stop pressing when the defender's
        // condition multiplier (stamina/natural_fitness/match_readiness/
        // determination blend with the fatigue curve) drops below the
        // sustainable-press floor. Floors map roughly to 25% raw
        // condition for a baseline-fitness CB and lower for elite-fit
        // ones — preserving the previous hysteresis with Resting's
        // crisis re-entry but skill-graded.
        let def_profile = DefenderSkillProfile::from_ctx(ctx);
        let stamina = ctx.player.player_attributes.condition_percentage() as f32;
        if stamina < STAMINA_THRESHOLD || def_profile.def_condition_mult < 0.55 {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::Resting,
            ));
        }

        // 2. Back off during foul protection — don't crowd the free kick
        if ctx.ball().is_in_flight() && ctx.ball().is_owned() {
            return Some(StateChangeResult::with_defender_state(
                DefenderState::HoldingLine,
            ));
        }

        // 3. Identify the opponent player with the ball
        if let Some(opponent) = ctx.players().opponents().with_ball().next() {
            let distance_to_opponent = opponent.distance(ctx);

            // If close enough to tackle, transition to Tackling state.
            // Repeat-tackle prevention lives on the player via
            // `tackle_cooldown` — a single-state cooldown here wouldn't
            // cover the Standing/Running/Covering re-entry paths.
            //
            // realism-bug (2026-07-30): the cooldown check alone isn't
            // enough — it only lives INSIDE Tackling::process(), which
            // bounces a cooldown player straight back to Pressing. With
            // no cooldown awareness here, Pressing immediately routed
            // them right back into Tackling on the very next tick (still
            // within TACKLING_DISTANCE_THRESHOLD), producing a 1-tick
            // Tackling<->Pressing flicker that lasted the full ~30s
            // cooldown window. Measured via raw position/state traces:
            // multi-player clusters stuck near-motionless (88-96% of
            // samples <0.6u/s) for up to 18-19 real seconds, both the
            // engaged defender AND an opposing presser oscillating in
            // lockstep the entire time. Gating entry here on
            // `can_attempt_tackle()` lets a cooldown player fall through
            // to the role-based coordination below (Cover/Marking/
            // HoldingLine) instead — a real defender who just missed a
            // tackle jockeys/covers, he doesn't keep re-lunging.
            if ctx.player().defensive().can_engage_tackle(distance_to_opponent) {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Tackling,
                ));
            }

            // D4.6 of docs/defensive-decision-logic-spec.md: last-ditch
            // sliding block. Deliberately checked HERE, in the chase
            // itself, not in DefenderTacklingState — that state's own
            // tackle/contain/concede-out comparison (D1-D4) assumes the
            // defender has already arrived (goal-side, in control). A
            // defender who is still trailing at 10-34u out may never
            // legally reach that state before the attacker's gone: this
            // is the earlier decision point the spec calls for, gated
            // on the same `can_attempt_tackle()`/`is_best_player_to_
            // chase_ball()` conventions the normal engage above already
            // uses, so it can't fire on repeat every tick or pile up
            // multiple defenders on the same ball.
            if distance_to_opponent >= DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE
                && ctx.player.can_attempt_tackle()
                && ctx.team().is_best_player_to_chase_ball()
                && dv::desperate_slide_value(ctx, &opponent, def_profile.tackle_profile) > 0.0
            {
                return Some(self.attempt_desperate_slide(ctx, &opponent));
            }

            // Pressing distance scales with team intensity AND the
            // defender's own press_profile — work-rate/anticipation/
            // stamina/positioning blend with condition fatigue baked in.
            // Poor pressers run a smaller bubble; elite pressers commit
            // a bit further. Replaces the static base + intensity formula.
            let intensity = ctx.team().press_intensity();
            let def_profile = DefenderSkillProfile::from_ctx(ctx);
            let profile_bonus = def_profile.press_profile * 18.0;
            let pressing_threshold = if ctx.ball().on_own_side()
                && ctx.ball().distance_to_own_goal()
                    < ctx.context.field_size.width as f32 * FIELD_THIRD_THRESHOLD
            {
                BASE_PRESSING_DISTANCE_DEFENSIVE_THIRD
                    + MAX_PRESSING_BONUS_DEFENSIVE_THIRD * intensity
                    + profile_bonus
            } else {
                BASE_PRESSING_DISTANCE + MAX_PRESSING_BONUS * intensity + profile_bonus
            };

            // If the opponent is too far away, stop pressing
            if distance_to_opponent > pressing_threshold {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::HoldingLine,
                ));
            }

            // Role-based coordination: a defender only stays in Pressing
            // while they're still the Primary for the current carrier.
            // When the carrier dribbles past (or another defender gets
            // closer), role flips to Cover/Help/Hold and we drop back —
            // Standing's role block will reassign us to the right state
            // on the next tick.
            match ctx.player().defensive().defensive_role_for_ball_carrier_unified() {
                DefensiveRole::Primary => {
                    // Stay on the carrier — aggressive press continues.
                    None
                }
                DefensiveRole::Cover => Some(StateChangeResult::with_defender_state(
                    DefenderState::Covering,
                )),
                DefensiveRole::Help => Some(StateChangeResult::with_defender_state(
                    DefenderState::Marking,
                )),
                DefensiveRole::Hold => Some(StateChangeResult::with_defender_state(
                    DefenderState::HoldingLine,
                )),
            }
        } else {
            // No opponent with the ball - ball might be loose
            // Check if we should intercept
            if !ctx.ball().is_owned() && ctx.ball().distance() < 50.0 && ctx.ball().speed() < 3.0 {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::TakeBall,
                ));
            }
            if ctx.ball().distance() < 60.0 && !ctx.ball().is_owned() {
                return Some(StateChangeResult::with_defender_state(
                    DefenderState::Intercepting,
                ));
            }
            Some(StateChangeResult::with_defender_state(
                DefenderState::HoldingLine,
            ))
        }
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // Move towards the opponent with the ball

        let opponents = ctx.players().opponents();
        let mut opponent_with_ball = opponents.with_ball();

        if let Some(opponent) = opponent_with_ball.next() {
            let distance_to_opponent = opponent.distance(ctx);

            // Intercept the carrier's future position based on their
            // velocity — stops them slipping past because the defender
            // was aiming at where they WERE, not where they're going.
            // Previously this was a 5u goal-side offset which meant a
            // running defender arrived ~5u behind the carrier and kept
            // chasing without ever making contact.
            let opp_velocity = ctx.tick_context.positions.players.velocity(opponent.id);
            let opp_speed = opp_velocity.magnitude();
            let pace = ctx.player.skills.physical.pace;
            // Press boost from the unified profile (press_profile +
            // mobility composite). Tired / low-stamina chasers drop
            // into a softer press; elite pressers genuinely outsprint
            // journeyman carriers. Stays bounded by the original
            // calibrated envelope.
            let minute = sc::minute_from_ms(ctx.context.total_match_time);
            let def_profile = DefenderSkillProfile::from_ctx(ctx);
            let mobility = sc::mobility(ctx.player, minute);
            let press_composite = 0.65 * def_profile.press_profile + 0.35 * mobility;
            let press_boost = (1.40 + (press_composite - 0.50) * 0.50).clamp(1.15, 1.65);

            // realism-bug (2026-08-30, shot/cross-block investigation,
            // defenders first — see docs/defensive-decision-logic-spec.md):
            // "block shots and crosses" needs someone to actually get
            // there before the strike, not just react afterward
            // (`shot_block_reaction_velocity`, processor.rs) or once
            // already glued into Tackling range (the D1-D6
            // tackle/contain/concede-out comparison below). Reuses the
            // SAME danger primitive `contain_value`/`concede_out_value`
            // already trust — `value_conceded_if_beaten` (self shot
            // danger OR best-teammate delivery danger, discounted by
            // cover) — rather than inventing a second "is this
            // dangerous" reading: real doctrine is a defender doesn't
            // close down a harmless sideways ball at the same urgency
            // as a striker with a live sight of goal. Multiplicative,
            // bounded to a modest ceiling (comparable to `press_boost`'s
            // own ±25% spread around its 1.40 base) so this nudges
            // closing speed rather than producing an implausible sprint.
            // Reasoned magnitude, category (d) — no dataset publishes
            // "closing speed vs. shot danger," flagged as an estimate
            // like every other new constant in this investigation.
            let danger = dv::value_conceded_if_beaten(ctx, &opponent);
            const URGENCY_SPEED_BOOST: f32 = 0.35;
            let urgency_boost = 1.0 + danger * URGENCY_SPEED_BOOST;

            let speed = pace * press_boost * urgency_boost;

            // Crude lead time: distance / (our speed). If the carrier is
            // moving quickly, aim ahead along their velocity so we meet
            // them. If they're stationary, aim right at the ball.
            let lead_ticks = if speed > 0.01 {
                (distance_to_opponent / speed).min(30.0)
            } else {
                0.0
            };
            let predicted = opponent.position + opp_velocity * lead_ticks;

            // Bias predicted point toward the goal-side so we close the
            // shooting lane even on chase — the defender wants to be
            // BETWEEN the carrier and our goal, not just on top of them.
            //
            // When the carrier is inside shooting range (<80u from our
            // goal), ramp the goalside bias HARD. This puts the defender
            // squarely in the shot-line, which gives them a real chance
            // to block the strike via `try_block_shot`. Real football:
            // a defender closing down in the box shows the shooter
            // his body and steps along the shot line — he doesn't just
            // run at the ball.
            let own_goal = ctx.ball().direction_to_own_goal();
            let to_own_goal = (own_goal - predicted).normalize();
            let carrier_to_goal = (own_goal - predicted).magnitude();
            let shot_zone_bias = if carrier_to_goal < 80.0 {
                // In shot zone: step 8-12u goal-side so we're actually
                // in the shot corridor. Heavier bias closer to goal.
                let zone_factor = 1.0 - (carrier_to_goal / 80.0).clamp(0.0, 1.0);
                8.0 + zone_factor * 4.0
            } else if opp_speed > 0.1 {
                2.0
            } else {
                0.0
            };
            // D1 (docs/attacking-defensive-decision-rebuild-plan.md §2.3)
            // — jockey/"show outside": lean the intercept target toward
            // the carrier's inside shoulder (away from whichever
            // touchline he's already nearer to), so the defender's body
            // covers the central lane on arrival and the carrier is
            // steered wide, not straight at.
            let field_height = ctx.context.field_size.height as f32;
            let jockey_sign = dv::jockey_show_outside_sign(
                predicted.y,
                field_height,
                ctx.player.defensive_show_outside,
            );
            let jockey_offset = Vector3::new(0.0, jockey_sign * dv::JOCKEY_SHOW_OUTSIDE_BIAS, 0.0);

            let intercept_target = predicted + to_own_goal * shot_zone_bias + jockey_offset;

            let to_target = intercept_target - ctx.player.position;
            let direction = if to_target.magnitude() > 0.01 {
                to_target.normalize()
            } else {
                Vector3::zeros()
            };

            let pressing_velocity = direction * speed;

            // Reduce separation velocity when actively pressing to allow close approach
            // When very close, disable separation entirely to enable tackling
            let separation = if distance_to_opponent < CLOSE_PRESSING_DISTANCE {
                ctx.player().separation_velocity() * 0.05 // Almost no separation when actively pressing
            } else {
                ctx.player().separation_velocity() * 0.15 // Minimal separation when pressing
            };

            return Some(pressing_velocity + separation);
        }

        // Loose ball nearby — pursue it
        if !ctx.ball().is_owned() && ctx.ball().distance() < 80.0 {
            let direction =
                (ctx.tick_context.positions.ball.position - ctx.player.position).normalize();
            let speed = ctx.player.skills.physical.pace;
            return Some(direction * speed);
        }

        None
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Pressing is very demanding - high intensity chasing and pressure
        DefenderCondition::with_velocity(ActivityIntensity::High).process(ctx);
    }
}

impl DefenderPressingState {
    /// D4.6 of docs/defensive-decision-logic-spec.md: rolls the RNG
    /// against `dv::slide_outcome_probs` (the SAME probability table
    /// `desperate_slide_value` used to decide this was worth trying, so
    /// the decision and the roll can't drift apart) and dispatches the
    /// matching outcome. Every outcome sets `start_tackle_cooldown` —
    /// real doctrine: a slide leaves the defender briefly grounded and
    /// out of the game, whether it connects or not, so he can't spam
    /// another attempt next tick regardless of what happened.
    fn attempt_desperate_slide(
        &self,
        ctx: &StateProcessingContext,
        opponent: &MatchPlayerLite,
    ) -> StateChangeResult {
        #[cfg(feature = "match-logs")]
        crate::tackle_stats::DEF_SLIDE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);

        let tackle_profile = DefenderSkillProfile::from_ctx(ctx).tackle_profile;
        let probs = dv::slide_outcome_probs(ctx, opponent, tackle_profile);
        let rng = &ctx.context.rng;
        let roll = rng.random_range(0.0f32..1.0);

        let p_success = probs.p_success;
        let p_partial = p_success + probs.p_partial;
        let p_foul = p_partial + probs.p_foul;
        let p_own_risk = p_foul + probs.p_own_risk;
        // remainder (roll >= p_own_risk) is Miss.

        let mut result = if roll < p_success {
            // D6 (2026-08-22): "tackle to get the ball" and "tackle to
            // push the ball away" are two different decisions. Winning
            // the slide already denied the attacker either way — this
            // only decides where the ball goes next: toward a boundary
            // ONLY if that's genuinely reachable before an opponent gets
            // there (dv::slide_should_push_to_boundary, the same race
            // check concede_out_value now uses), otherwise a safe
            // lay-off to the nearest teammate (handle_shoot_event's
            // "DEF_SLIDE_SUCCESS_LAYOFF" branch).
            let push_to_boundary = dv::slide_should_push_to_boundary(ctx);
            #[cfg(feature = "match-logs")]
            if push_to_boundary {
                crate::tackle_stats::DEF_SLIDE_SUCCESS.fetch_add(1, Ordering::Relaxed);
            } else {
                crate::tackle_stats::DEF_SLIDE_SUCCESS_LAYOFF.fetch_add(1, Ordering::Relaxed);
            }
            let reason = if push_to_boundary {
                "DEF_SLIDE_SUCCESS"
            } else {
                "DEF_SLIDE_SUCCESS_LAYOFF"
            };
            StateChangeResult::with_defender_state_and_event(
                DefenderState::Standing,
                Event::PlayerEvent(PlayerEvent::Shoot(
                    ShootingEventContext::new()
                        .with_player_id(ctx.player.id)
                        // Discarded and recomputed fresh at dispatch
                        // time by handle_shoot_event's "DEF_SLIDE_
                        // SUCCESS"/"DEF_SLIDE_SUCCESS_LAYOFF" branches —
                        // same convention DEF_CONCEDE_OUT already
                        // established for the identical reason (a tick
                        // or two of drift makes a stale target pointless
                        // to thread through).
                        .with_target(ctx.tick_context.positions.ball.position)
                        .with_reason(reason)
                        .build(ctx),
                )),
            )
        } else if roll < p_partial {
            #[cfg(feature = "match-logs")]
            crate::tackle_stats::DEF_SLIDE_PARTIAL.fetch_add(1, Ordering::Relaxed);
            StateChangeResult::with_defender_state_and_event(
                DefenderState::Standing,
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
            crate::tackle_stats::DEF_SLIDE_FOUL.fetch_add(1, Ordering::Relaxed);
            // Mirrors DefenderTacklingState::attempt_sliding_tackle's
            // own severity classification, simplified: a genuinely
            // desperate/mistimed challenge skews more toward a
            // card-worthy tackle than an ordinary contact foul.
            let severity = if rng.random_range(0.0f32..1.0) < 0.03 {
                FoulSeverity::Violent
            } else if rng.random_range(0.0f32..1.0) < 0.35 {
                FoulSeverity::Reckless
            } else {
                FoulSeverity::Normal
            };
            StateChangeResult::with_defender_state_and_event(
                DefenderState::Standing,
                Event::PlayerEvent(PlayerEvent::CommitFoul(ctx.player.id, severity)),
            )
        } else if roll < p_own_risk {
            #[cfg(feature = "match-logs")]
            crate::tackle_stats::DEF_SLIDE_OWN_RISK.fetch_add(1, Ordering::Relaxed);
            StateChangeResult::with_defender_state_and_event(
                DefenderState::Standing,
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
            crate::tackle_stats::DEF_SLIDE_MISS.fetch_add(1, Ordering::Relaxed);
            // Miss — the same outcome as not sliding at all, per the
            // spec's own framing. The opponent keeps the ball; no ball
            // event to dispatch, just the recovery cost below.
            StateChangeResult::with_defender_state(DefenderState::Standing)
        };
        result.start_tackle_cooldown = true;
        result
    }
}

impl DefenderPressingState {}
