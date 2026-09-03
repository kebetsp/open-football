use crate::PlayerFieldPositionGroup;
use crate::club::player::traits::BehavioralDirective;
use crate::r#match::events::Event;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::midfielders::states::common::{ActivityIntensity, MidfielderCondition};
use crate::r#match::player::events::{PassingEventContext, PlayerEvent};
use crate::r#match::player::strategies::common::players::MatchPlayerIteratorExt;
use crate::r#match::player::strategies::common::players::ops::forward_shot_decision::{
    ShotDecision, evaluate_forward_shot_decision,
};
use crate::r#match::player::strategies::common::players::ops::midfielder_skill::MidfielderSkillProfile;
use crate::r#match::player::strategies::players::DefensiveRole;
use crate::r#match::player::strategies::players::skills::SkillCurve;
use crate::r#match::{
    BallZone, ConditionContext, GamePhase, MatchPlayerLite, PassEvaluator, PlayerSide,
    StateChangeResult, StateProcessingContext, StateProcessingHandler, SteeringBehavior,
};
use nalgebra::Vector3;
use std::cmp::Ordering;

// Shooting distance constants for midfielders — more conservative than forwards
const MAX_SHOOTING_DISTANCE: f32 = 88.0; // Edge-of-box / arriving-midfielder strikes
const STANDARD_SHOOTING_DISTANCE: f32 = 52.0; // Standard shooting range for midfielders
const POINT_BLANK_DISTANCE: f32 = 20.0; // ~10m - must shoot, goalkeeper is right there

// Zone 1 (docs/zone-staged-possession-progression.md) open-space scan
// constants — hoisted here so `zone1_open_space_heading` (carrier's own
// scan, needs a winning HEADING) and `zone1_max_open_distance` (Chunk 3
// teammate-openness scan, needs only the best VALUE) share one
// definition instead of two independently-drifting copies.
const ZONE1_HEADING_ANGLES_DEG: [f32; 7] = [-70.0, -45.0, -20.0, 0.0, 20.0, 45.0, 70.0];
const ZONE1_MIN_OPEN_SPACE: f32 = 45.0; // reasoned estimate, not sourced — see zone1_open_space_heading's doc comment
const ZONE1_SCAN_MAX: f32 = 140.0;
// Chunk 3: "if there is a teammate who has roughly the same amount of
// free area — go there." Reasoned tolerance (no sourced data for this
// specific comparison), ~14% of the scan range — a teammate doesn't
// need to be STRICTLY as open, just close enough to count as "more or
// less" the same.
const ZONE1_OPENNESS_TOLERANCE: f32 = 20.0;
// Same "real, actionable nearby pass option" radius this file already
// uses elsewhere (`find_draw_and_release_pass`, `find_one_two_return`).
const ZONE1_TEAMMATE_SEARCH_RADIUS: f32 = 200.0;

// Zone 2 (final third, docs/zone-staged-possession-progression.md
// Chunk 6) constants. Reasoned estimates, not sourced — same
// discipline as Zone 1's own constants above.
//
// Soft goal-alignment nudge added to a heading's raw open-distance
// score in `zone2_open_space_heading`, only when `zone2_local_numbers`
// reads a genuine 1v1-or-worse. Small relative to `ZONE1_SCAN_MAX`
// (140.0) so it only tips a genuinely close decision between two
// similarly-open headings, never overrides one offering dramatically
// more real space — a nudge, not a funnel.
const ZONE2_GOAL_AWARENESS_BONUS: f32 = 20.0;
// Radius for the "1v1 vs 2v1+" local numbers count — a reasoned
// estimate of a real local duel zone near the box; no sourced dataset
// isolates this specific distance.
const ZONE2_LOCAL_NUMBERS_RADIUS: f32 = 60.0;
// Same forward-cone dot-product threshold `should_dribble`'s own
// `opponents_ahead` count already uses a few lines below, reused for
// consistency rather than a third independently-tuned cone.
const ZONE2_LOCAL_NUMBERS_FORWARD_COS: f32 = 0.3;

#[derive(Default, Clone)]
pub struct MidfielderRunningState {}

impl StateProcessingHandler for MidfielderRunningState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        // Offside discipline — if we don't have the ball and we've run
        // past the opposing defensive line, drop back before a teammate
        // plays a pass that finds us offside.
        if !ctx.player.has_ball(ctx) && ctx.player().defensive().is_stranded_offside() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Returning,
            ));
        }

        // COUNTER-PRESS: we just lost the ball. The closest midfielder
        // to the new carrier committing to an immediate press is the
        // single biggest recovery mechanism in real football — it's
        // why modern high-tempo sides look so relentless. Mirrors the
        // defender counter-press in `defenders/running` line ~108. Only
        // fires for the midfielder best positioned to chase (avoids
        // whole midfield collapsing on one runner).
        if !ctx.player.has_ball(ctx)
            && ctx.team().counterpress_window()
            && !ctx.team().is_control_ball()
        {
            let ball_dist = ctx.ball().distance();
            // Use the per-player eligibility helper so only the
            // best-positioned 2-3 midfielders engage. The ball-best
            // chaser check (already enforced at squad level) layers on
            // top so we don't double-up with a defender or forward who
            // also won the eligibility roll.
            let elected = ctx.player().pressure().should_counterpress()
                && ctx.team().is_best_player_to_chase_ball();
            let immediate = ball_dist < 25.0;
            if immediate {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Tackling,
                ));
            }
            if elected && ball_dist < 80.0 {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Pressing,
                ));
            }
        }

        // Opponent restart (GK holding or goal kick) — retreat immediately.
        if !ctx.player.has_ball(ctx) && ctx.ball().is_opponent_restart() {
            return Some(StateChangeResult::with_midfielder_state(
                MidfielderState::Returning,
            ));
        }

        // Phase-first dispatch — midfielders are the engine's pivot
        // between defence and attack, so the phase signal matters most
        // for them. See `phase_dispatch` for behaviour per phase.
        if let Some(phase_action) = self.phase_dispatch(ctx) {
            return Some(phase_action);
        }

        if ctx.player.has_ball(ctx) {
            // Corner taker: set the corner up via Crossing (which holds the
            // delivery until centre-backs have pushed up to attack it).
            if ctx.ball().is_team_attacking_corner() {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Crossing,
                ));
            }

            // Behavioural directive: lay_it_off_first_touch — release the
            // ball immediately on reception instead of turning/carrying.
            // 30-tick window — see the forward-state twin for rationale.
            if ctx.player.behavioral_directive == Some(BehavioralDirective::LayItOffFirstTouch)
                && ctx.tick_context.ball.ownership_duration < 30
            {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Passing,
                ));
            }

            // Conditional (Level 2): lay off ONLY long-ball receptions —
            // see the forward-state twin.
            if ctx.player.behavioral_directive == Some(BehavioralDirective::LayOffOnLongBall)
                && ctx.tick_context.ball.ownership_duration < 60
                && ctx.ball().is_long_reception(140.0)
            {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Passing,
                ));
            }

            // Conditional (Level 2): carry into a clear forward cone —
            // see the forward-state twin.
            if ctx.player.behavioral_directive == Some(BehavioralDirective::CarryWhenSpaceAhead) {
                let goal_pos = ctx.player().opponent_goal_position();
                let to_goal = (goal_pos - ctx.player.position).normalize();
                let blocker_ahead = ctx.players().opponents().nearby(35.0).any(|o| {
                    (o.position - ctx.player.position).normalize().dot(&to_goal) > 0.3
                });
                if !blocker_ahead {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Dribbling,
                    ));
                }
            }

            // Behavioural directive: byline_and_cross. A wide midfielder
            // (the 4-4-2 winger) with this directive bypasses the entire
            // pass/shot decision tree while in the wide band of the
            // attacking half: carry to the byline (Dribbling velocity
            // steers the route), cross on arrival. Mirrors the forward-
            // state implementation; wingers in this formation are
            // midfielders, so the directive must live here too.
            if ctx.player.behavioral_directive == Some(BehavioralDirective::BylineAndCross) {
                let field_h = ctx.context.field_size.height as f32;
                let field_w = ctx.context.field_size.width as f32;
                // Channel-gated (start_position.y), not current-y, so the
                // directive engages regardless of where team shape has
                // pulled the player — see the forward-state twin.
                let start_y = ctx.player.start_position.y;
                let channel_wide = start_y < field_h * 0.30 || start_y > field_h * 0.70;
                let in_attacking_zone =
                    ctx.ball().distance_to_opponent_goal() < field_w * 0.70;
                if channel_wide && in_attacking_zone {
                    let y = ctx.player.position.y;
                    let currently_wide = y < field_h * 0.30 || y > field_h * 0.70;
                    let goal_x = ctx.player().opponent_goal_position().x;
                    if currently_wide && (goal_x - ctx.player.position.x).abs() < 40.0 {
                        return Some(StateChangeResult::with_midfielder_state(
                            MidfielderState::Crossing,
                        ));
                    }
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Dribbling,
                    ));
                }
            }

            let distance_to_goal = ctx.ball().distance_to_opponent_goal();
            let coach = ctx.team().coach_instruction();
            let can_shoot = ctx.team().can_shoot();

            // ── Midfielder snapshot under pressure ─────────────────────
            //
            // Same asymmetric pattern as the forward snapshot in
            // `forwarders/states/running/mod.rs`: a midfielder who just
            // received the ball (in_state_time < 8) inside shooting
            // range (< 60u) with a defender right on them (within 10u),
            // AND whose first_touch is below the defender's tackling
            // (by ≥0.5), fires immediately instead of going through
            // the normal control + decision tree. Without this,
            // arriving box-runners who would-be cut-back recipients
            // get tackled before they can shoot — they're MIDfielders
            // and have lower first-touch than dedicated forwards, so
            // strong defenders out-touch them on virtually every
            // reception. Adding the midfielder path lifts the
            // weak-team scoring contribution from runners-into-the-
            // box, which the forward-only path missed entirely.
            //
            // Calibration-neutral at equal skill: at first_touch =
            // tackling, 11 < 10.5 is false, snapshot doesn't fire.
            if can_shoot && ctx.in_state_time < 8 && distance_to_goal < 60.0 {
                let nearest_threat = ctx
                    .players()
                    .opponents()
                    .nearby_raw(10.0)
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                    .map(|(id, _)| id);
                if let Some(threat_id) = nearest_threat {
                    let defender_tackling = ctx.player().skills(threat_id).technical.tackling;
                    let attacker_first_touch = ctx.player.skills.technical.first_touch;
                    if attacker_first_touch < defender_tackling - 0.5 {
                        return Some(
                            StateChangeResult::with_midfielder_state(MidfielderState::Shooting)
                                .with_shot_reason("MID_SNAPSHOT_PRESSED"),
                        );
                    }
                }
            }

            // Emergency clearance: under heavy pressure in our own box.
            // Route to Passing so its emergency-clearance code path fires
            // (Passing already has `emit_emergency_clearance` gated on
            // `in_box_danger_zone` + `is_under_heavy_pressure`). Running
            // previously had no such escape hatch — a midfielder under
            // two-defender press in their own area kept trying to play
            // out, lost the ball, and conceded via the ensuing turnover.
            if ctx.player().pressure().is_under_heavy_pressure()
                && ctx.ball().distance_to_own_goal() < ctx.context.field_size.width as f32 * 0.18
            {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Passing,
                ));
            }

            // ── MIDFIELDER SHOOTING (unified, skill-driven) ──────────────
            // Every midfielder in shooting range consults the SAME shot
            // helper the forwards use, so the shoot/pass/hold decision
            // scales continuously with the player's actual shooting
            // attributes (selection / execution / composure) rather than
            // the old binary `mid_shot_selection >= 0.32/0.42` cliffs —
            // which scored a default central mid ~0 yet let an elite one
            // fire unlimited, additive shots. The helper applies its xG
            // floor, inside-six floor, GK / 1v1 read, pass-EV deferral
            // (a playmaker lays it off when a teammate is better placed),
            // and the anti-monopoly volume cap. Net effect: a deep regista
            // rarely shoots, a box-to-box #8 arriving centrally shoots like
            // a forward, and no single player monopolises the attempts.
            // Hoisted above the possession / recycle defaults so a real
            // opening isn't recycled back to a forward.
            if can_shoot && distance_to_goal <= MAX_SHOOTING_DISTANCE {
                #[cfg(feature = "match-logs")]
                {
                    use std::sync::atomic::Ordering;
                    crate::r#match::player::strategies::common::players::ops::forward_shot_decision::mid_run_diag::MID_INRANGE_TICKS.fetch_add(1, Ordering::Relaxed);
                }
                // Tier 1 — a CLEAR, good-angle chance in range is taken by
                // ANY midfielder. This is chance-quality, NOT a skill gate:
                // whether you SHOOT a clear central look doesn't depend on
                // how good a finisher you are (skill decides whether it goes
                // IN — see the conversion gradient). Without this the
                // willingness roll declines half of even the clean chances
                // and they're recycled away, dropping mid goals AND team
                // totals. The anti-monopoly cap still applies: once a player
                // has hogged (>6 attempts) this falls through to the
                // willingness roll like everyone else, so it can't be abused.
                let sp = ctx.player().shooting().shot_profile();
                // xG bar 0.085 → 0.110 → 0.180: a tighter Tier-1 keeps the
                // box-to-box mid path open for genuine sitters only while
                // the helper's willingness roll handles the grey zone.
                // 0.110 was tuned while the fatigue cliff silently
                // flattened midfielders from ~minute 20 (they sprint the
                // most box-to-box, so they hit the old 15%-condition floor
                // first and stopped arriving in range at all). Fatigue
                // normalization revived them and this deterministic tier —
                // which bypasses the willingness roll entirely — ballooned
                // MID share to 45% of goals (target 32%) at 0.110 and was
                // still 55% of MID shot volume at 0.140. At 0.180 only a
                // true big chance (penalty-spot central, keeper at mercy)
                // fires deterministically — which is the realistic shape:
                // nobody declines those. The anti-monopoly cap also
                // tightens 6 → 4 to mirror the FWD path.
                let clear_good = distance_to_goal <= STANDARD_SHOOTING_DISTANCE
                    && coach.shooting_reluctance() < 0.5
                    && ctx.player().has_clear_shot()
                    && ctx.player().shooting().has_good_angle()
                    && sp.expected_xg(distance_to_goal, true) >= 0.180;
                if clear_good && ctx.memory().shots_taken <= 4 {
                    #[cfg(feature = "match-logs")]
                    {
                        use std::sync::atomic::Ordering;
                        crate::r#match::player::strategies::common::players::ops::forward_shot_decision::mid_run_diag::MID_SHOOT_FIRED.fetch_add(1, Ordering::Relaxed);
                    }
                    return Some(
                        StateChangeResult::with_midfielder_state(MidfielderState::Shooting)
                            .with_shot_reason("MID_CLEAR_CHANCE"),
                    );
                }
                // Tier 2 — speculative / long-range / hogger: skill-driven
                // willingness via the shared helper.
                match evaluate_forward_shot_decision(ctx, "MID_SHOOT") {
                    ShotDecision::Shoot { reason } => {
                        #[cfg(feature = "match-logs")]
                        {
                            use std::sync::atomic::Ordering;
                            crate::r#match::player::strategies::common::players::ops::forward_shot_decision::mid_run_diag::MID_SHOOT_FIRED.fetch_add(1, Ordering::Relaxed);
                        }
                        // Beyond standard range, route to the dedicated
                        // long-range strike; closer is a normal finish.
                        let state = if distance_to_goal > STANDARD_SHOOTING_DISTANCE {
                            MidfielderState::DistanceShooting
                        } else {
                            MidfielderState::Shooting
                        };
                        return Some(
                            StateChangeResult::with_midfielder_state(state)
                                .with_shot_reason(reason),
                        );
                    }
                    ShotDecision::Pass => {
                        // Helper judged a teammate the better option — lay
                        // it off (the playmaker's creative choice).
                        if let Some((target, _)) = self.find_best_pass_option(ctx) {
                            return Some(StateChangeResult::with_midfielder_state_and_event(
                                MidfielderState::Standing,
                                Event::PlayerEvent(PlayerEvent::PassTo(
                                    PassingEventContext::new()
                                        .with_from_player_id(ctx.player.id)
                                        .with_to_player_id(target.id)
                                        .with_reason("MID_SHOOT_LAYOFF")
                                        .build(ctx),
                                )),
                            ));
                        }
                    }
                    ShotDecision::Hold => {}
                }
            }

            // Coach tempo: if wasting time or slowing down, prefer
            // possession — UNLESS a counter window is open: a team
            // protecting a lead that wins the ball against an
            // overcommitted opponent breaks at full speed, instruction
            // or not (see TeamOps::counter_window).
            if coach.prefer_possession()
                && distance_to_goal > POINT_BLANK_DISTANCE
                && !ctx.team().counter_window()
            {
                let ownership_ticks = ctx.tick_context.ball.ownership_duration;
                if ownership_ticks < coach.min_possession_ticks() {
                    return None;
                }
            }

            // PATIENT POSSESSION: use the team-level
            // `should_play_possession` check so all real-football
            // triggers apply (just won ball, tired, leading, late
            // game, no attack ready). See team/team.rs for the rules.
            let under_pressure = ctx.players().opponents().exists(15.0);
            if !under_pressure
                && distance_to_goal > 70.0
                && ctx.tick_context.ball.ownership_duration > 8
                && ctx.team().should_play_possession()
            {
                if let Some(target) = self.find_best_pass_option(ctx).map(|(t, _)| t) {
                    // Score the candidate: only pass if it's a safe,
                    // sideways/backward option (we don't want to fire
                    // the ball forward into a covered attacker).
                    let player_pos = ctx.player.position;
                    let goal_pos = ctx.player().opponent_goal_position();
                    let to_goal = (goal_pos - player_pos).normalize();
                    let to_t = (target.position - player_pos).normalize();
                    let forward_component = to_t.dot(&to_goal);
                    let target_in_space =
                        ctx.tick_context.grid.opponents(target.id, 10.0).count() < 2;
                    // Accept lateral, backward, or mildly-forward only
                    if forward_component < 0.4 && target_in_space {
                        return Some(StateChangeResult::with_midfielder_state_and_event(
                            MidfielderState::Standing,
                            Event::PlayerEvent(PlayerEvent::PassTo(
                                PassingEventContext::new()
                                    .with_from_player_id(ctx.player.id)
                                    .with_to_player_id(target.id)
                                    .with_reason("MID_PATIENT_POSSESSION")
                                    .build(ctx),
                            )),
                        ));
                    }
                }
                // No safe outlet yet — keep the ball, re-evaluate next tick
                return None;
            }

            // (Shooting — including the box arrival / cutback finish and
            // point-blank chances — is handled by the unified skill-driven
            // helper block hoisted above; no separate carve-outs needed.)

            // Priority: Clear ball if congested anywhere (not just boundaries)
            // Only attempt after carrying ball for a while to prevent instant pass-after-receive
            if (self.is_congested_near_boundary(ctx) || ctx.player().movement().is_congested())
                && ctx.in_state_time > 20
                && ctx.tick_context.ball.ownership_duration > 15
            {
                // Try to find a good pass option first using the standard evaluator
                if let Some((target_teammate, _reason)) = self.find_best_pass_option(ctx) {
                    let dist = (target_teammate.position - ctx.player.position).magnitude();
                    // Only pass if target is far enough away to escape congestion
                    if dist > 40.0 {
                        return Some(StateChangeResult::with_midfielder_state_and_event(
                            MidfielderState::Standing,
                            Event::PlayerEvent(PlayerEvent::PassTo(
                                PassingEventContext::new()
                                    .with_from_player_id(ctx.player.id)
                                    .with_to_player_id(target_teammate.id)
                                    .with_reason("MID_RUNNING_EMERGENCY_CLEARANCE_BEST")
                                    .build(ctx),
                            )),
                        ));
                    }
                }

                // Fallback: find teammate at least 40 units away, not a recent passer,
                // and in open space (outside congestion zone)
                if let Some(target_teammate) = ctx
                    .players()
                    .teammates()
                    .nearby(200.0)
                    .filter(|t| {
                        let dist = (t.position - ctx.player.position).magnitude();
                        dist > 40.0
                            && ctx.ball().passer_recency_penalty(t.id) > 0.3
                            && ctx.tick_context.grid.opponents(t.id, 15.0).count() < 2
                    })
                    .max_by(|a, b| {
                        // Prefer the farthest teammate in open space
                        let da = (a.position - ctx.player.position).magnitude();
                        let db = (b.position - ctx.player.position).magnitude();
                        da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                    })
                {
                    return Some(StateChangeResult::with_midfielder_state_and_event(
                        MidfielderState::Standing,
                        Event::PlayerEvent(PlayerEvent::PassTo(
                            PassingEventContext::new()
                                .with_from_player_id(ctx.player.id)
                                .with_to_player_id(target_teammate.id)
                                .with_reason("MID_RUNNING_EMERGENCY_CLEARANCE_NEARBY")
                                .build(ctx),
                        )),
                    ));
                }
            }

            // Shooting is evaluated earlier (the SHOOT-FIRST block above,
            // hoisted ahead of the possession / pass-recycling defaults).
            // `mid_profile` / `goal_dist` are still needed by the
            // carry-forward and dribble gates below.
            let goal_dist = ctx.ball().distance_to_opponent_goal();
            let mid_profile = MidfielderSkillProfile::from_ctx(ctx);

            // CARRY FORWARD: Open path to goal — gate on carry_selection
            // (dribbling / decisions / composure / acceleration / agility
            // composite). Replaces the ad-hoc dribbling+composure+pace
            // blend with the unified midfielder profile.
            let field_width = ctx.context.field_size.width as f32;
            // Zone 1: no extra distance cap — `has_open_space_ahead_zoned`
            // already gates correctly on the real `BallZone` boundary, so
            // this reflex should cover the WHOLE zone, not a narrow
            // near-goal quarter. Chunk 2 of docs/zone-staged-possession-
            // progression.md found the old flat `goal_dist <
            // field_width*0.45` (378u) bound meant Pavel's own quoted
            // scenario — "an attacker gets the ball next to the middle
            // line" — never even reached this gate (goal_dist there is
            // ~420u, already past the old cap). Zone 2's own cap (378u)
            // is confirmed dead weight by Chunk 6's own conflict check
            // (item 4): `BallZone::AttackingThird` itself only ever
            // spans ~280u, always inside the 378u cap, so the cap can
            // never bind for a genuine Zone-2 ticket — left in place
            // as a harmless no-op rather than removed.
            let upper_bound_ok = if self.in_zone1(ctx) {
                true
            } else {
                goal_dist < field_width * 0.45
            };
            if goal_dist > POINT_BLANK_DISTANCE && upper_bound_ok && mid_profile.allows_carry_into_space() {
                if self.in_zone1(ctx) {
                    // Chunk 3 (docs/zone-staged-possession-progression.md):
                    // the gate becomes a real decision point here —
                    // "if there is a teammate who has roughly the same
                    // amount of free area, go there. But if not, carry
                    // the ball all the way you can." — not just a
                    // boolean feeding a blind `return None` (Chunk 1
                    // conflict check item 6). `my_open >=
                    // ZONE1_MIN_OPEN_SPACE` is the same condition
                    // `has_open_space_ahead_zoned` checked before, just
                    // computed once here so its value is available for
                    // the teammate comparison too.
                    let my_open = self.zone1_max_open_distance(ctx, ctx.player.position);
                    if my_open >= ZONE1_MIN_OPEN_SPACE {
                        // Same-binary ablation toggle for Chunk 3's
                        // teammate-open-space comparison specifically
                        // (mirrors `spacing.rs::mobility_ablated`'s own
                        // `OF_SCORE_BLIND` idiom) — Pavel asked for a
                        // way to isolate this one mechanism ("is it
                        // skewing everything too much to the centre")
                        // for later chunks to be A/B'd against, without
                        // a git-stash/rebuild cycle each time. Inert
                        // (Chunk 3 fully active) unless
                        // `ZONE1_CHUNK3_ABLATE` is set; when set, this
                        // reverts to exactly Chunk 1/2's behaviour —
                        // carry as far as space allows, no teammate
                        // comparison. Kept permanently, zero cost in
                        // normal play.
                        if !Self::zone1_chunk3_ablated() {
                            if let Some(target) = self.zone1_similarly_open_teammate(ctx, my_open) {
                                return Some(StateChangeResult::with_midfielder_state_and_event(
                                    MidfielderState::Standing,
                                    Event::PlayerEvent(PlayerEvent::PassTo(
                                        PassingEventContext::new()
                                            .with_from_player_id(ctx.player.id)
                                            .with_to_player_id(target.id)
                                            .with_reason("MID_ZONE1_SIMILARLY_OPEN_TEAMMATE")
                                            .build(ctx),
                                    )),
                                ));
                            }
                        }
                        return None; // no similarly-open teammate (or ablated) — carry as far as space allows
                    }
                    // No real space in any direction — fall through to
                    // should_dribble/pass logic below, same as before.
                } else if self.zone2_open_space_heading(ctx).is_some() {
                    // Chunk 6: goal-aware-but-not-funneled — see
                    // `zone2_open_space_heading`'s own doc comment.
                    return None;
                }
            }

            // Minimum carry time before considering passes — let midfielders run with the ball
            let ownership_ticks = ctx.tick_context.ball.ownership_duration;

            // DRIBBLE: If there's space ahead and player has carry skill,
            // beat opponents. Gated by carry_selection thresholds:
            //   * `allows_take_on_one`  → at most 1 opponent ahead.
            //   * `allows_take_on_two`  → up to 2.
            //   * lower → never dribble into opponents.
            if ownership_ticks > 5 && ownership_ticks < 60 {
                let goal_pos = ctx.player().opponent_goal_position();
                let player_pos = ctx.player.position;
                let to_goal = (goal_pos - player_pos).normalize();

                let opponents_ahead = ctx
                    .players()
                    .opponents()
                    .nearby(40.0)
                    .filter(|opp| {
                        let to_opp = (opp.position - player_pos).normalize();
                        to_opp.dot(&to_goal) > 0.3
                            && (opp.position - player_pos).norm_squared() < 35.0 * 35.0
                    })
                    .count();

                let should_dribble = if opponents_ahead == 0 {
                    false
                } else if mid_profile.allows_take_on_two() {
                    opponents_ahead <= 2
                } else if mid_profile.allows_take_on_one() {
                    opponents_ahead == 1
                } else {
                    false
                };

                let goal_dist_from_opp = ctx.ball().distance_to_opponent_goal();
                let field_width = ctx.context.field_size.width as f32;

                if should_dribble && goal_dist_from_opp < field_width * 0.75 {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Dribbling,
                    ));
                }
            }

            // COUNTER-ATTACK: Quick transition but not instant — need a few ticks to assess
            if ownership_ticks > 8
                && ctx.ball().has_stable_possession()
                && self.is_counter_attack_opportunity(ctx)
            {
                if let Some(forward_target) = self.find_counter_attack_pass(ctx) {
                    return Some(StateChangeResult::with_midfielder_state_and_event(
                        MidfielderState::Running,
                        Event::PlayerEvent(PlayerEvent::PassTo(
                            PassingEventContext::new()
                                .with_from_player_id(ctx.player.id)
                                .with_to_player_id(forward_target.id)
                                .with_reason("MID_RUNNING_COUNTER_ATTACK")
                                .build(ctx),
                        )),
                    ));
                }
            }

            // ONE-TWO COMBINATION: After carrying briefly, check if passer has run ahead
            // into space — return the ball for a wall-pass / give-and-go
            if ownership_ticks >= 10 && ownership_ticks <= 30 && ctx.ball().has_stable_possession()
            {
                if let Some(return_target) = self.find_one_two_return(ctx) {
                    return Some(StateChangeResult::with_midfielder_state_and_event(
                        MidfielderState::Running,
                        Event::PlayerEvent(PlayerEvent::PassTo(
                            PassingEventContext::new()
                                .with_from_player_id(ctx.player.id)
                                .with_to_player_id(return_target.id)
                                .with_reason("MID_RUNNING_ONE_TWO_RETURN")
                                .build(ctx),
                        )),
                    ));
                }
            }

            // DRAW AND RELEASE: If opponent is committing to tackle, draw them in
            // then pass to space they vacated — requires carrying to draw them
            if ownership_ticks > 30 && ctx.ball().has_stable_possession() {
                if let Some(release_target) = self.find_draw_and_release_pass(ctx) {
                    return Some(StateChangeResult::with_midfielder_state_and_event(
                        MidfielderState::Running,
                        Event::PlayerEvent(PlayerEvent::PassTo(
                            PassingEventContext::new()
                                .with_from_player_id(ctx.player.id)
                                .with_to_player_id(release_target.id)
                                .with_reason("MID_RUNNING_DRAW_AND_RELEASE")
                                .build(ctx),
                        )),
                    ));
                }
            }

            // CUTBACK FROM WIDE: a wide carrier near the byline plays a low
            // cutback to a central midfielder arriving unmarked in the box
            // (a first-time shot for the runner) in preference to always
            // launching an aerial cross at the forwards. In a 442 the byline
            // carrier is usually a wide mid, so this is the main engine of
            // midfielder goals. Restricted to a true cutback origin (wide +
            // deep); the shared finder enforces the rest (central runner,
            // in range, unmarked, clear lane). Checked just before CROSSING
            // so a genuine cutback chance is taken over the speculative cross.
            if ownership_ticks > 12 && ctx.ball().has_stable_possession() {
                let field_h = ctx.context.field_size.height as f32;
                let mid_goal = ctx.player().opponent_goal_position();
                // "Deep" = near the byline in X; "off-centre" = poor own
                // angle. Using goal-CENTRE distance here is wrong (a wide
                // carrier is always far from centre), so key off byline X.
                let carrier_byline = (mid_goal.x - ctx.player.position.x).abs() < 90.0;
                let carrier_offcenter =
                    (ctx.player.position.y - field_h / 2.0).abs() > field_h * 0.15;
                if carrier_byline && carrier_offcenter {
                    if let Some(runner) =
                        crate::r#match::player::strategies::common::players::ops::forward_shot_decision::find_cutback_to_arriving_runner(ctx)
                    {
                        #[cfg(feature = "match-logs")]
                        {
                            use std::sync::atomic::Ordering;
                            crate::r#match::player::strategies::common::players::ops::forward_shot_decision::mid_run_diag::MID_CUTBACK.fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(StateChangeResult::with_midfielder_state_and_event(
                            MidfielderState::Standing,
                            Event::PlayerEvent(PlayerEvent::PassTo(
                                PassingEventContext::new()
                                    .with_from_player_id(ctx.player.id)
                                    .with_to_player_id(runner.id)
                                    .with_reason("MID_CUTBACK_TO_RUNNER")
                                    .build(ctx),
                            )),
                        ));
                    }
                }
            }

            // CROSSING: Wide midfielder in attacking third with teammates in the box
            if ownership_ticks > 20 && ctx.ball().has_stable_possession() && self.should_cross(ctx)
            {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Crossing,
                ));
            }

            // SWITCH PLAY: When teammates are overloaded on one side, switch to the other flank
            if ownership_ticks > 20 && ctx.ball().has_stable_possession() {
                let field_height = ctx.context.field_size.height as f32;
                let field_center_y = field_height / 2.0;
                let ball_side = if ctx.player.position.y > field_center_y {
                    1.0
                } else {
                    -1.0
                };

                // Count teammates on ball's side
                let teammates_on_side = ctx
                    .players()
                    .teammates()
                    .all()
                    .filter(|t| {
                        let t_side = if t.position.y > field_center_y {
                            1.0
                        } else {
                            -1.0
                        };
                        t_side == ball_side
                    })
                    .count();

                // Switch if overloaded (3+ teammates on same side) or congested
                let should_switch =
                    teammates_on_side >= 3 || ctx.player().movement().is_congested();

                if should_switch && mid_profile.allows_switch_play() {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::SwitchingPlay,
                    ));
                }
            }

            // COACH TEMPO: When instructed to slow down, prefer passing back to defenders
            if coach.prefer_possession()
                && ownership_ticks > coach.min_possession_ticks()
                && ctx.ball().has_stable_possession()
            {
                if let Some(safe_target) = self.find_safe_backward_pass(ctx) {
                    return Some(StateChangeResult::with_midfielder_state_and_event(
                        MidfielderState::Standing,
                        Event::PlayerEvent(PlayerEvent::PassTo(
                            PassingEventContext::new()
                                .with_from_player_id(ctx.player.id)
                                .with_to_player_id(safe_target.id)
                                .with_reason("MID_COACH_TEMPO_PASS_BACK")
                                .build(ctx),
                        )),
                    ));
                }
            }

            // Enhanced passing decision — look for a good pass
            if ownership_ticks > 15 && ctx.ball().has_stable_possession() && self.should_pass(ctx) {
                if let Some((target_teammate, _reason)) = self.find_best_pass_option(ctx) {
                    return Some(StateChangeResult::with_midfielder_state_and_event(
                        MidfielderState::Running,
                        Event::PlayerEvent(PlayerEvent::PassTo(
                            PassingEventContext::new()
                                .with_from_player_id(ctx.player.id)
                                .with_to_player_id(target_teammate.id)
                                .with_reason("MID_RUNNING_SHOULD_PASS")
                                .build(ctx),
                        )),
                    ));
                }
            }
        } else {
            // Without ball - check for opponent with ball first
            // Only the closest player should chase — others hold tactical shape
            if let Some(opponent) = ctx
                .players()
                .opponents()
                .nearby(150.0)
                .with_ball(ctx)
                .next()
            {
                let opponent_distance = (opponent.position - ctx.player.position).magnitude();

                // D8.2 of docs/defensive-decision-logic-spec.md: role-
                // based shape, mirroring DefenderRunningState's own
                // already-shipped pattern (line ~180 of that file) —
                // this is the exact block D0/D7 diagnosed as the source
                // of the coverage-assignment gap (a distant defender
                // self-reporting "Primary" via a defenders-only pool
                // while a genuinely close midfielder had no coordinated
                // role at all). `Primary` under the unified pool now
                // correctly identifies whichever position group is
                // actually closest; `Help` routes to `Guarding` (an
                // existing, already-tuned state whose own scoring/
                // movement logic is a close semantic match — pick up an
                // unmarked dangerous opponent); `Cover`/non-primary
                // `Hold` fall through unchanged to "stay in running."
                let role = ctx
                    .player()
                    .defensive()
                    .defensive_role_for_ball_carrier_unified();

                // Close — only the designated closer tackles (removed
                // the old unconditional "tackle regardless" — same
                // pileup fix defenders already validated at D2).
                if opponent_distance < 30.0 && role == DefensiveRole::Primary {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Tackling,
                    ));
                }

                if role == DefensiveRole::Help {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Guarding,
                    ));
                }

                // Only the best-positioned Primary presses — prevents team swarming
                if role == DefensiveRole::Primary && ctx.team().is_best_player_to_chase_ball() {
                    if opponent_distance < 50.0 {
                        return Some(StateChangeResult::with_midfielder_state(
                            MidfielderState::Tackling,
                        ));
                    }
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Pressing,
                    ));
                }
                // Cover / non-primary Hold: stay in running (will follow waypoints via velocity)
            }

            // Teammate has the ball — actively support the attack
            if ctx.team().is_control_ball() {
                let ball_distance = ctx.ball().distance();
                let goal_dist = ctx.ball().distance_to_opponent_goal();
                let field_width = ctx.context.field_size.width as f32;

                // ANTI-CLUSTERING: If too many teammates nearby, go find space
                let nearby_teammates = ctx.players().teammates().nearby(25.0).count();
                if nearby_teammates >= 2 && ball_distance > 30.0 {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::CreatingSpace,
                    ));
                }

                // Zone-band gate: each midfielder is responsible for a lateral
                // band centred on their start_position.y (≈136u apart in 4-4-2).
                // Only the 1–2 midfielders whose band contains the ball should
                // push into an attacking run; the others hold shape. 100u half-
                // width gives 64u overlap so there's always at least one
                // midfielder in zone for any ball position.
                let ball_y = ctx.tick_context.positions.ball.position.y;
                let base_in_lateral_zone =
                    (ball_y - ctx.player.start_position.y).abs() < 100.0;

                // Chunk 4 (docs/zone-staged-possession-progression.md):
                // "let your teammates run into their positions in the
                // center and on the other flank" while a real carry is
                // progressing. Measured first, per the doc's own chunk-
                // order item 4, before writing this: of every off-ball
                // tick where `base_in_lateral_zone` is false (a player
                // structurally held out of AttackSupporting/CreatingSpace
                // by the zone-band gate above), 6.2% correspond to a
                // teammate genuinely carrying — real open space ahead
                // (Chunk 1/3's own `zone1_max_open_distance` bar),
                // sustained (ownership_duration > 60 ticks, past every
                // "quick touch" window already used elsewhere in this
                // file — `should_dribble`'s 5..60, `find_one_two_return`'s
                // 10..=30), and still genuinely moving with it (the
                // HOLDER's own velocity, not the ball's — `ball.velocity`
                // is zeroed on every claim in `ownership.rs` and only
                // means anything mid-flight). A real, non-trivial (not
                // near-universal) minority of situations — exactly the
                // gap Pavel described, not a hypothetical one.
                //
                // Widening `in_lateral_zone` for this specific case is
                // enough: `AttackSupportingState`/`CreatingSpaceState`
                // already compute central-support and opposite-flank-
                // free-zone targets respectively, purely from live ball/
                // holder position — they were simply never being entered
                // by an out-of-zone player. No new positioning logic
                // needed, only this eligibility widening.
                let in_lateral_zone = base_in_lateral_zone
                    || (!Self::zone4_ablated() && self.teammate_open_carry_active(ctx).is_some());

                // Only attack when ball is in our lateral zone
                if goal_dist < field_width * 0.4 && ball_distance < 300.0 && in_lateral_zone {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::AttackSupporting,
                    ));
                }

                // Ball is far AND in our zone — offer a passing option
                if ball_distance > 200.0 && in_lateral_zone {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::CreatingSpace,
                    ));
                }

                // Medium range in-zone: actively support after a brief settle
                if ctx.in_state_time > 80 && in_lateral_zone {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::AttackSupporting,
                    ));
                }

                // Ball outside our zone — hold shape; velocity() enforces positioning
                return None;
            }

            // Loose ball nearby — only chase if we're the best positioned teammate
            if ctx.ball().distance() < 50.0 && !ctx.ball().is_owned() {
                let ball_velocity = ctx.tick_context.positions.ball.velocity.norm();
                if ball_velocity < 3.0 && ctx.team().is_best_player_to_chase_ball() {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::TakeBall,
                    ));
                }
            }

            // Loose-ball claim is handled universally at the dispatcher
            // (`PlayerFieldPositionGroup::process`). Duplicating the check
            // here with the tolerance-banded `is_best_player_to_chase_ball`
            // let multiple players enter TakeBall simultaneously.

            // Also respond to ball system notifications
            if ctx.ball().should_take_ball_immediately()
                && ctx.team().is_best_player_to_chase_ball()
            {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::TakeBall,
                ));
            }

            // Corner set-piece: take assigned zone position.
            if ctx.ball().is_team_attacking_corner() {
                use crate::r#match::player::strategies::common::set_pieces::player_corner_zone;
                if player_corner_zone(ctx).is_some() {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::CornerRun,
                    ));
                }
            }

            // Track dangerous runners — opponent forwards sprinting toward our goal
            if ctx.ball().on_own_side() {
                let own_goal = ctx.ball().direction_to_own_goal();
                let has_dangerous_runner = ctx.players().opponents().forwards().any(|opp| {
                    let dist = (opp.position - ctx.player.position).magnitude();
                    if dist > 60.0 {
                        return false;
                    }
                    let vel = opp.velocity(ctx);
                    let speed = vel.norm();
                    if speed < 2.0 {
                        return false;
                    }
                    let to_goal = (own_goal - opp.position).normalize();
                    let alignment = vel.normalize().dot(&to_goal);
                    alignment > 0.5
                });

                // Dangerous runner detected — close them down via Guarding.
                // TrackingRunner was a single-entry ghost state that did the
                // same "stay goal-side of the runner" thing Guarding already
                // does; keeping Guarding removes the duplicate.
                if has_dangerous_runner {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Guarding,
                    ));
                }
            }

            // Guard unmarked attackers on our side when we can't press/intercept
            if ctx.ball().on_own_side() && ctx.ball().distance() > 100.0 {
                return Some(StateChangeResult::with_midfielder_state(
                    MidfielderState::Guarding,
                ));
            }
        }

        // ANTI-OSCILLATION: If carrying ball too long without acting, force a decision
        // POSSESSION RETENTION: Allow longer holding when team is comfortable
        //
        // Idea-to-code conflict check follow-up (2026-09-01): this
        // threshold is keyed on `in_state_time` (ticks since entering
        // Running at all), not "ticks since the carry stopped making
        // progress" — so a genuinely still-open Zone 1 carry
        // (docs/zone-staged-possession-progression.md) was getting cut
        // off at a hard 1.5-2.5s regardless of how much real space
        // remained, the exact "carry the ball all the way you can"
        // idea this mechanism exists for. A player still advancing
        // into real open space isn't oscillating — the anti-
        // oscillation net is meant for a player who's genuinely stuck,
        // not one who's making progress. Raised, not disabled: still a
        // real, finite safety valve (600 ticks/6s) against a
        // pathological case where `has_open_space_ahead_zoned` never
        // clears — self-terminating in normal play anyway, since
        // advancing into more contested territory or crossing into
        // Zone 2 (`in_zone1` false) both end the exemption on their
        // own.
        let in_open_zone1_carry = self.in_zone1(ctx) && self.has_open_space_ahead_zoned(ctx);
        let anti_oscillation_threshold = if in_open_zone1_carry {
            600
        } else if self.should_retain_possession(ctx) {
            250
        } else {
            150
        };
        if ctx.player.has_ball(ctx) && ctx.in_state_time > anti_oscillation_threshold {
            // Prefer passing first
            if let Some((target_teammate, _reason)) = self.find_best_pass_option(ctx) {
                return Some(StateChangeResult::with_midfielder_state_and_event(
                    MidfielderState::Running,
                    Event::PlayerEvent(PlayerEvent::PassTo(
                        PassingEventContext::new()
                            .with_from_player_id(ctx.player.id)
                            .with_to_player_id(target_teammate.id)
                            .with_reason("MID_RUNNING_ANTI_OSCILLATION")
                            .build(ctx),
                    )),
                ));
            }
            // AM carve-out: forward helper picks the anti-oscillation
            // trigger so a low-skill #10 still has a path to a shot
            // when they've been carrying too long without acting.
            if ctx
                .player
                .tactical_position
                .current_position
                .is_attacking_midfielder()
            {
                if let ShotDecision::Shoot { reason } =
                    evaluate_forward_shot_decision(ctx, "AM_RUN_ANTI_OSC_FWD")
                {
                    return Some(
                        StateChangeResult::with_midfielder_state(MidfielderState::Shooting)
                            .with_shot_reason(reason),
                    );
                }
            }
            // Only shoot as fallback at point-blank range with clear shot
            // AND a midfielder with adequate shot selection.
            let distance_to_goal = ctx.ball().distance_to_opponent_goal();
            if distance_to_goal < 25.0 && ctx.player().has_clear_shot() {
                let mid_profile = MidfielderSkillProfile::from_ctx(ctx);
                if mid_profile.mid_shot_selection >= 0.40 {
                    return Some(
                        StateChangeResult::with_midfielder_state(MidfielderState::Shooting)
                            .with_shot_reason("MID_RUN_ANTI_OSCILLATION"),
                    );
                }
            }
            // Last resort: pass to any nearby teammate ahead of the ball (toward opponent goal)
            let player_pos = ctx.player.position;
            let goal_pos = ctx.player().opponent_goal_position();
            let to_goal = (goal_pos - player_pos).normalize();
            if let Some(target_teammate) = ctx
                .players()
                .teammates()
                .nearby(200.0)
                .filter(|t| {
                    let to_teammate = (t.position - player_pos).normalize();
                    to_teammate.dot(&to_goal) > 0.0 // Teammate is ahead (toward opponent goal)
                })
                .next()
            {
                return Some(StateChangeResult::with_midfielder_state_and_event(
                    MidfielderState::Running,
                    Event::PlayerEvent(PlayerEvent::PassTo(
                        PassingEventContext::new()
                            .with_from_player_id(ctx.player.id)
                            .with_to_player_id(target_teammate.id)
                            .with_reason("MID_RUNNING_ANTI_OSCILLATION_FALLBACK")
                            .build(ctx),
                    )),
                ));
            }
            // Absolute last resort: pass to any nearby teammate (even backward)
            if let Some(target_teammate) = ctx.players().teammates().nearby(200.0).next() {
                return Some(StateChangeResult::with_midfielder_state_and_event(
                    MidfielderState::Running,
                    Event::PlayerEvent(PlayerEvent::PassTo(
                        PassingEventContext::new()
                            .with_from_player_id(ctx.player.id)
                            .with_to_player_id(target_teammate.id)
                            .with_reason("MID_RUNNING_ANTI_OSCILLATION_FALLBACK_ANY")
                            .build(ctx),
                    )),
                ));
            }
        }

        None
    }

    fn velocity(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        // run_channel_in_behind (midfielder version): an ordered off-ball
        // run — sprint toward the opponent goal line at the player's own
        // channel while the team holds the ball. Mirrors the forwards'
        // RunningInBehind channel run, including the onside hold: while a
        // teammate is still ON the ball, the run stops at the shoulder of
        // the last defender and releases the moment the pass is struck.
        if !ctx.player.has_ball(ctx)
            && ctx.player.behavioral_directive == Some(BehavioralDirective::RunChannelInBehind)
            && ctx.team().is_control_ball()
        {
            // engine audit Module 3 (2026-08-29): offside-line discipline
            // for this run target used to be clamped here directly; it's
            // now applied once, universally, by
            // `processor.rs::offside_line_hold_velocity` downstream — see
            // that function's doc comment for the full rationale.
            let goal = ctx.player().opponent_goal_position();
            let target_x = goal.x;
            let target = Vector3::new(target_x, ctx.player.start_position.y, 0.0);
            let to_target = target - ctx.player.position;
            if to_target.magnitude() > 4.0 {
                return Some(
                    to_target.normalize() * ctx.player.skills.physical.pace * 1.1
                        + ctx.player().separation_velocity() * 0.3,
                );
            }
        }

        // Simplified waypoint following
        if ctx.player.should_follow_waypoints(ctx) {
            let waypoints = ctx.player.get_waypoints_as_vectors();
            if !waypoints.is_empty() {
                return Some(
                    SteeringBehavior::FollowPath {
                        waypoints,
                        current_waypoint: ctx.player.waypoint_manager.current_index,
                        path_offset: 5.0,
                    }
                    .calculate(ctx.player)
                    .velocity,
                );
            }
        }

        if ctx.player.has_ball(ctx) {
            // POSSESSION RETENTION: When in control mode, move slower with more lateral sway
            // to keep ball and tire opponents instead of always charging forward
            if self.should_retain_possession(ctx) {
                Some(self.calculate_possession_retention_movement(ctx))
            } else if self.in_zone1(ctx) && MidfielderSkillProfile::from_ctx(ctx).allows_carry_into_space() {
                // Zone 1 (build-up/progression): goal-blind carrying —
                // move along the best open heading, never toward a
                // fixed goal point (docs/zone-staged-possession-
                // progression.md, Chunk 1). Zone 2 (attacking third)
                // keeps the old goal-relative movement below, unchanged
                // until that chunk of the plan is implemented.
                //
                // Idea-to-code conflict check follow-up (2026-09-01,
                // per Pavel's direct request): `process()`'s own
                // carry-forward branch already gates on
                // `allows_carry_into_space()` — but `velocity()` never
                // consulted it, so a low-skill/fatigued player who
                // reached this branch via a DIFFERENT `return None`
                // (coach tempo / patient possession holding, which
                // return before ever reaching the carry-forward gate)
                // still got the new open-space movement regardless.
                // The gate only ever blocked one of several
                // equal-outcome paths, never the movement itself — a
                // pre-existing gap (the old, goal-relative
                // `calculate_simple_ball_movement` had the identical
                // hole), now closed by checking it here too, at the
                // one place that actually decides the movement.
                Some(self.calculate_open_space_movement(ctx))
            } else {
                // Zone 2 (final third): goal-aware-but-not-funneled
                // movement (Chunk 6, docs/zone-staged-possession-
                // progression.md) — replaces the old, fully
                // goal-relative `calculate_simple_ball_movement`.
                Some(self.calculate_zone2_movement(ctx))
            }
        } else {
            let start_pos = ctx.player.start_position;
            let field_width = ctx.context.field_size.width as f32;
            let field_height = ctx.context.field_size.height as f32;

            // Off-ball movement when our team isn't in control. Compact
            // defensive block rather than a frozen formation line:
            //   - Ball-side compaction: the whole midfield shifts toward
            //     the ball's lateral (y) position.
            //   - Depth compaction: the line pushes up when the ball
            //     advances, drops back when the ball retreats.
            //   - Ball-side / far-side stagger: the midfielder on the
            //     ball's half steps slightly forward to engage, the
            //     far-side one drops slightly back to cover. This is
            //     what breaks the straight-line "robot" look.
            //   - Separation from nearby teammates naturally spreads
            //     players when formation positions overlap.
            // Loose ball pulls harder (45%) because the designated chaser
            // has already transitioned to TakeBall via `process()`; the
            // rest gently close the gap in support.
            if !ctx.team().is_control_ball() {
                let ball_pos = ctx.tick_context.positions.ball.position;
                let ball_loose = !ctx.ball().is_owned();
                let field_half_x = field_width * 0.5;
                let field_half_y = field_height * 0.5;
                let attacking_left = ctx.player.side == Some(PlayerSide::Left);

                // Lateral (y) shift: always track ball laterally so the
                // midfield slides as a block. 0.3 = a normal compact block,
                // more during a loose-ball scramble (urgency).
                let lateral_coef = if ball_loose { 0.45 } else { 0.30 };
                let lateral_shift = (ball_pos.y - field_half_y) * lateral_coef;

                // Depth (x) shift: push with ball depth. Low coefficient so
                // we don't abandon the defensive line when ball is deep.
                let depth_shift = (ball_pos.x - field_half_x) * 0.15;

                // Ball-side stagger: player on the same lateral half as the
                // ball steps ~10 units toward the opponent's goal to
                // engage; the far-side player drops ~10 units back. A
                // diagonal stagger instead of a flat line.
                let ball_top = ball_pos.y < field_half_y;
                let player_top = start_pos.y < field_half_y;
                let on_ball_side = ball_top == player_top;
                let forward_sign = if attacking_left { 1.0 } else { -1.0 };
                let stagger_x = if on_ball_side { 10.0 } else { -10.0 } * forward_sign;

                let target_x =
                    (start_pos.x + depth_shift + stagger_x).clamp(30.0, field_width - 30.0);
                let target_y = (start_pos.y + lateral_shift).clamp(30.0, field_height - 30.0);

                let target = Vector3::new(target_x, target_y, 0.0);

                // No outer deadzone / zero-velocity early return: the hard
                // stop produced the "arrive-and-jitter" look. Arrive's own
                // quadratic slowing + 3-unit brake zone handles settling.
                // Add separation so players shuffle apart when formation
                // shifts stack them together.
                let arrive = SteeringBehavior::Arrive {
                    target,
                    slowing_distance: 25.0,
                }
                .calculate(ctx.player)
                .velocity;

                return Some(arrive + ctx.player().separation_velocity() * 0.4);
            }

            // Team has ball — off-ball movement: spread across the pitch using unique player slots
            let ball_pos = ctx.tick_context.positions.ball.position;
            let ball_distance = ctx.ball().distance();

            // ANTI-FOLLOWING: If very close to ball carrier, move toward start position
            // to create space instead of calculating a volatile escape direction
            if ball_distance < 40.0 {
                let spread_target = Vector3::new(
                    start_pos.x.clamp(30.0, field_width - 30.0),
                    start_pos.y.clamp(30.0, field_height - 30.0),
                    0.0,
                );
                let dist = (spread_target - ctx.player.position).magnitude();
                if dist < 8.0 {
                    return Some(Vector3::zeros());
                }
                return Some(
                    SteeringBehavior::Arrive {
                        target: spread_target,
                        slowing_distance: 20.0,
                    }
                    .calculate(ctx.player)
                    .velocity,
                );
            }

            let attacking_direction = match ctx.player.side {
                Some(PlayerSide::Left) => 1.0,
                Some(PlayerSide::Right) => -1.0,
                None => 0.0,
            };

            // Quantize ball position to 20-unit grid to prevent wobble
            let qball_x = (ball_pos.x / 20.0).round() * 20.0;
            let qball_y = (ball_pos.y / 20.0).round() * 20.0;

            // Formation-based positioning: stay near start_pos, shift slightly toward ball.
            // Each player keeps their unique formation position — no slot convergence.
            let ball_pull = 0.15; // How much the ball pulls the player (low = keep formation)

            // X: mostly start_pos, pulled slightly toward ball + forward offset
            let forward_offset = attacking_direction * 40.0;
            let target_x = start_pos.x * (1.0 - ball_pull) + (qball_x + forward_offset) * ball_pull;

            // Y: zone-band discipline. Each midfielder orbits their start_position.y
            // within ±80u. When ball is outside that band, lateral pull is zero —
            // the player holds their y and only shifts longitudinally with the ball.
            // When ball is inside the band, normal pull (0.15) applies.
            const ZONE_HALF_WIDTH: f32 = 80.0;
            let in_zone_y = (qball_y - start_pos.y).abs() < ZONE_HALF_WIDTH;
            let lateral_pull = if in_zone_y { ball_pull } else { 0.0 };
            let target_y = (start_pos.y + (qball_y - start_pos.y) * lateral_pull)
                .clamp(start_pos.y - ZONE_HALF_WIDTH, start_pos.y + ZONE_HALF_WIDTH);

            let target = Vector3::new(
                target_x.clamp(30.0, field_width - 30.0),
                target_y.clamp(30.0, field_height - 30.0),
                0.0,
            );

            let dist_to_target = (target - ctx.player.position).magnitude();

            if dist_to_target < 8.0 {
                return Some(Vector3::zeros());
            }

            let arrive_velocity = SteeringBehavior::Arrive {
                target,
                slowing_distance: 20.0,
            }
            .calculate(ctx.player)
            .velocity;

            Some(arrive_velocity)
        }
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Midfielders cover the most ground during a match - box to box running
        // High intensity with velocity-based adjustment
        MidfielderCondition::with_velocity(ActivityIntensity::High).process(ctx);
    }
}

impl MidfielderRunningState {
    /// Phase-first dispatch. Midfielders sit in the spine of the team,
    /// so the phase cue drives more of their behaviour than any other
    /// role. Settled phases fall through to the existing decision tree.
    fn phase_dispatch(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        let phase = ctx.team().phase();
        let has_ball = ctx.player.has_ball(ctx);
        let ball_dist = ctx.ball().distance();
        match phase {
            // Counter-press window after losing the ball. The closest
            // midfielder to the ball engages; others drop into
            // Returning to rebuild shape. Without this window the
            // engine had no concept of "hunt the ball back now" —
            // every press was reactive to distance alone.
            GamePhase::DefensiveTransition if !has_ball => {
                if ball_dist < 45.0 && ctx.team().is_best_player_to_chase_ball() {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Pressing,
                    ));
                }
                // Further-away midfielders reset shape rather than
                // ball-chase into space.
                if ball_dist > 80.0 {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Returning,
                    ));
                }
            }
            // Fast-break window after winning the ball. Midfielder not
            // on the ball makes a forward run to support. The carrier
            // falls through to passing/dribbling below.
            GamePhase::AttackingTransition if !has_ball => {
                if ctx.ball().distance_to_opponent_goal() > 40.0 {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::AttackSupporting,
                    ));
                }
            }
            // Coach-triggered high press — midfielders hunt the ball in
            // the opposition half alongside the forwards.
            GamePhase::HighPress if !has_ball => {
                if ball_dist < 70.0 && ctx.team().is_best_player_to_chase_ball() {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Pressing,
                    ));
                }
            }
            // Low-block: cut passing lanes by dropping into the gap
            // between defenders and the ball. Midfielders shouldn't
            // continue chasing upfield in this phase.
            GamePhase::LowBlock if !has_ball => {
                if ball_dist > 50.0 {
                    return Some(StateChangeResult::with_midfielder_state(
                        MidfielderState::Returning,
                    ));
                }
            }
            _ => {}
        }
        None
    }

    fn find_best_pass_option<'a>(
        &self,
        ctx: &StateProcessingContext<'a>,
    ) -> Option<(MatchPlayerLite, &'static str)> {
        PassEvaluator::find_best_pass_option(ctx, 300.0)
    }

    /// Enhanced passing decision driven by the unified midfielder
    /// skill profile (pass execution, progressive selection, press
    /// resistance) instead of raw vision/passing/decisions thresholds.
    fn should_pass(&self, ctx: &StateProcessingContext) -> bool {
        let profile = MidfielderSkillProfile::from_ctx(ctx);
        let pressing_intensity = self.calculate_pressing_intensity(ctx);
        let distance_to_goal = ctx.ball().distance_to_opponent_goal();

        // 1. MUST PASS: Heavy pressing — low press_resistance forces a
        // release; high resistance lets us carry/shield briefly.
        if pressing_intensity > 0.7 {
            return profile.press_resistance < 0.55 || profile.pass_execution > 0.30;
        }

        // 2. FORCED PASS: Moderate pressure + middling press resistance.
        if pressing_intensity > 0.5 && profile.press_resistance < 0.60 {
            return true;
        }

        // 3. TACTICAL PASS: Elite progressive playmakers look for the
        // line-breaking option even without pressure.
        if profile.allows_killer_ball()
            && self.has_better_positioned_teammate(ctx, distance_to_goal)
        {
            return true;
        }

        // 4. TEAM PLAY: Good distributors (decent execution + high
        // teamwork curve via pass_execution) pass to maintain tempo.
        if pressing_intensity > 0.3 && profile.pass_execution > 0.55 {
            return self.find_best_pass_option(ctx).is_some();
        }

        // 5. LIGHT PRESSURE: continuous pass likelihood from execution.
        if pressing_intensity > 0.2 {
            return profile.pass_execution > 0.45;
        }

        // 6. NO PRESSURE: midfielders distribute — gate on execution.
        if distance_to_goal > 200.0 && profile.pass_execution > 0.45 {
            return self.has_better_positioned_teammate(ctx, distance_to_goal);
        }

        let ownership_ticks = ctx.tick_context.ball.ownership_duration;
        if ownership_ticks > 60 {
            return true;
        }

        false
    }

    /// Calculate pressing intensity based on number and proximity of opponents
    fn calculate_pressing_intensity(&self, ctx: &StateProcessingContext) -> f32 {
        // Use pre-computed distance closure instead of scanning all players
        let mut weighted_pressure = 0.0f32;
        for (_opp_id, dist) in ctx.tick_context.grid.opponents(ctx.player.id, 50.0) {
            if dist < 15.0 {
                weighted_pressure += 0.5; // very close
            } else if dist < 30.0 {
                weighted_pressure += 0.3; // close
            } else {
                weighted_pressure += 0.1; // medium
            }
        }

        (weighted_pressure / 2.0).min(1.0)
    }

    /// Check if there's a teammate in a better position
    fn has_better_positioned_teammate(
        &self,
        ctx: &StateProcessingContext,
        current_distance: f32,
    ) -> bool {
        ctx.players().teammates().nearby(300.0).any(|teammate| {
            let teammate_distance =
                (teammate.position - ctx.player().opponent_goal_position()).magnitude();
            let is_closer = teammate_distance < current_distance * 0.8;
            if !is_closer {
                return false;
            }
            let has_space = ctx.tick_context.grid.opponents(teammate.id, 30.0).count() < 2;
            if !has_space {
                return false;
            }
            ctx.player().has_clear_pass(teammate.id)
        })
    }

    /// ONE-TWO COMBINATION: Check if the player who just passed to us has run into
    /// a better forward position with space. If so, return the ball for a wall-pass.
    fn find_one_two_return<'a>(&self, ctx: &StateProcessingContext<'a>) -> Option<MatchPlayerLite> {
        let recent_passers = ctx.tick_context.ball.recent_passers();
        // Get the most recent passer (last element in the ring buffer vec)
        let passer_id = *recent_passers.last()?;

        // Passer must be a teammate
        let passer = ctx.context.players.by_id(passer_id)?;
        if passer.team_id != ctx.player.team_id {
            return None;
        }

        // Find passer in nearby players
        let passer_lite = ctx
            .players()
            .teammates()
            .all()
            .find(|t| t.id == passer_id)?;

        let player_pos = ctx.player.position;
        let goal_pos = ctx.player().opponent_goal_position();
        let passer_pos = passer_lite.position;

        // Passer must now be closer to opponent goal than us (they continued their run)
        let our_goal_dist = (goal_pos - player_pos).magnitude();
        let passer_goal_dist = (goal_pos - passer_pos).magnitude();
        if passer_goal_dist >= our_goal_dist * 0.9 {
            return None; // Passer didn't run ahead enough
        }

        // Passer must be in open space (no opponents within 50 units)
        let opponents_near_passer = ctx.tick_context.grid.opponents(passer_id, 50.0).count();
        if opponents_near_passer >= 1 {
            return None;
        }

        // Must have clear passing lane back to passer
        if !ctx.player().has_clear_pass(passer_id) {
            return None;
        }

        // Passer must be within reasonable passing distance
        let pass_distance = (passer_pos - player_pos).magnitude();
        if pass_distance > 200.0 || pass_distance < 10.0 {
            return None;
        }

        Some(passer_lite)
    }

    /// DRAW AND RELEASE: Detect an opponent committing to a tackle (approaching fast
    /// within 15-35 units). Find a teammate in the space the opponent is vacating.
    fn find_draw_and_release_pass<'a>(
        &self,
        ctx: &StateProcessingContext<'a>,
    ) -> Option<MatchPlayerLite> {
        let player_pos = ctx.player.position;

        // Find the closest approaching opponent (within 15-35 units, closing in)
        let approaching_opponent = ctx
            .players()
            .opponents()
            .nearby(35.0)
            .filter(|opp| {
                let dist = (opp.position - player_pos).magnitude();
                if dist < 15.0 || dist > 35.0 {
                    return false;
                }

                // Check if opponent is moving toward us
                let opp_velocity = ctx.tick_context.positions.players.velocity(opp.id);
                if opp_velocity.magnitude() < 1.0 {
                    return false;
                }

                let to_us = (player_pos - opp.position).normalize();
                let opp_dir = opp_velocity.normalize();
                opp_dir.dot(&to_us) > 0.6 // Moving toward us
            })
            .min_by(|a, b| {
                let da = (a.position - player_pos).magnitude();
                let db = (b.position - player_pos).magnitude();
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })?;

        // The space the opponent is vacating is roughly behind them (opposite of their movement)
        let opp_velocity = ctx
            .tick_context
            .positions
            .players
            .velocity(approaching_opponent.id);
        let vacated_zone = approaching_opponent.position - opp_velocity.normalize() * 30.0;

        // Find a teammate near the vacated space (or in the channel the opponent left)
        let best_teammate = ctx
            .players()
            .teammates()
            .nearby(200.0)
            .filter(|t| {
                let t_dist_to_vacated = (t.position - vacated_zone).magnitude();
                // Teammate should be near the vacated space or generally in that direction
                t_dist_to_vacated < 60.0
                    && ctx.player().has_clear_pass(t.id)
                    && ctx.tick_context.grid.opponents(t.id, 10.0).count() < 2
            })
            .min_by(|a, b| {
                let da = (a.position - vacated_zone).magnitude();
                let db = (b.position - vacated_zone).magnitude();
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })?;

        Some(best_teammate)
    }

    /// POSSESSION RETENTION: Determine if team should retain possession rather than
    /// attack directly. True when team is comfortable (not losing), in own/mid third,
    /// and not under heavy pressure.
    fn should_retain_possession(&self, ctx: &StateProcessingContext) -> bool {
        // Never retain if losing
        if ctx.team().is_loosing() {
            return false;
        }

        // Don't retain in the attacking third — keep pressing forward.
        // Uses the same canonical `BallZone::AttackingThird` boundary
        // every other zone-aware signal in the engine already agrees
        // on (`TeamTacticalState.ball_zone`), replacing a private,
        // slightly different 0.35-of-field-width threshold this
        // function used to compute on its own
        // (docs/zone-staged-possession-progression.md, Chunk 1
        // conflict check item 3 — a second, uncoordinated zone
        // boundary in the same file).
        if self.in_zone2(ctx) {
            return false;
        }

        // Don't retain under heavy pressure
        let pressing = self.calculate_pressing_intensity(ctx);
        if pressing > 0.5 {
            return false;
        }

        // Don't retain if open space ahead — advance to create tension
        if self.has_open_space_ahead_zoned(ctx) {
            return false;
        }

        // Retain possession when team is in control
        ctx.team().is_control_ball()
    }

    /// Movement for possession retention mode: slower, more lateral,
    /// controlled tempo. Only ever reached in Zone 1 —
    /// `should_retain_possession` now bails out unconditionally in
    /// `BallZone::AttackingThird` — so this is goal-blind by
    /// construction: no `opponent_goal_position()`/`mid_y` pull toward
    /// a fixed central channel anywhere in this function
    /// (docs/zone-staged-possession-progression.md, Chunk 1 conflict
    /// check item 2 — "it doesn't let me dribble in any direction,
    /// only to the center").
    fn calculate_possession_retention_movement(
        &self,
        ctx: &StateProcessingContext,
    ) -> Vector3<f32> {
        let player_pos = ctx.player.position;
        let attack_dir: f32 = match ctx.player.side {
            Some(PlayerSide::Left) => 1.0,
            Some(PlayerSide::Right) => -1.0,
            None => 0.0,
        };

        // Slow forward drift along the generic attacking direction +
        // a wide lateral sway — same controlled-tempo character as
        // before, just with no fixed goal point or mid_y anchor to
        // drift/re-centre toward.
        let phase = (ctx.in_state_time as f32) * std::f32::consts::TAU / 100.0; // Slower period
        let sway = phase.sin() * 0.5; // Wider lateral sway
        let target = player_pos + Vector3::new(attack_dir * 15.0, sway * 20.0, 0.0);

        SteeringBehavior::Arrive {
            target,
            slowing_distance: 30.0,
        }
        .calculate(ctx.player)
        .velocity
            * 0.6 // Slower overall speed in retention mode
    }

    /// COUNTER-ATTACK: Detect if a counter-attack opportunity exists.
    /// True when team just won possession, opponents are high, and space ahead is open.
    fn is_counter_attack_opportunity(&self, ctx: &StateProcessingContext) -> bool {
        let ownership_duration = ctx.tick_context.ball.ownership_duration;

        // Must have just won possession (< 15 ticks)
        if ownership_duration >= 15 {
            return false;
        }

        // Ball must be on own side or midfield (counter goes forward)
        if !ctx.ball().on_own_side() {
            // Allow early midfield counters too
            let goal_dist = ctx.ball().distance_to_opponent_goal();
            let field_width = ctx.context.field_size.width as f32;
            if goal_dist < field_width * 0.4 {
                return false; // Already in attacking third, no need for counter
            }
        }

        // Count opponents ahead of ball (between ball and opponent goal)
        let ball_pos = ctx.tick_context.positions.ball.position;
        let goal_pos = ctx.player().opponent_goal_position();
        let to_goal = (goal_pos - ball_pos).normalize();

        let opponents_ahead = ctx
            .players()
            .opponents()
            .all()
            .filter(|opp| {
                let to_opp = opp.position - ball_pos;
                to_opp.normalize().dot(&to_goal) > 0.3 // Opponent is ahead of ball
            })
            .count();

        // Counter-attack opportunity if few opponents ahead
        opponents_ahead < 3
    }

    /// COUNTER-ATTACK: Find a forward pass target for quick transition.
    /// Prefers forwards making runs toward goal with space around them.
    fn find_counter_attack_pass<'a>(
        &self,
        ctx: &StateProcessingContext<'a>,
    ) -> Option<MatchPlayerLite> {
        let player_pos = ctx.player.position;
        let goal_pos = ctx.player().opponent_goal_position();
        let to_goal = (goal_pos - player_pos).normalize();

        let mut best_target: Option<(MatchPlayerLite, f32)> = None;

        for teammate in ctx.players().teammates().nearby(300.0) {
            let to_teammate = teammate.position - player_pos;

            // Must be ahead of us (toward opponent goal)
            if to_teammate.normalize().dot(&to_goal) < 0.3 {
                continue;
            }

            // Must have space (no opponent within 10 units)
            let opponents_near = ctx.tick_context.grid.opponents(teammate.id, 10.0).count();
            if opponents_near >= 2 {
                continue;
            }

            // Must have clear passing lane
            if !ctx.player().has_clear_pass(teammate.id) {
                continue;
            }

            // Score: prefer forwards, closer to goal, making runs
            let is_forward = teammate.tactical_positions.is_forward();
            let goal_dist = (goal_pos - teammate.position).magnitude();
            let teammate_velocity = ctx.tick_context.positions.players.velocity(teammate.id);
            let making_run = teammate_velocity.magnitude() > 1.0
                && teammate_velocity.normalize().dot(&to_goal) > 0.3;

            let mut score = 1000.0 - goal_dist; // Closer to goal = better
            if is_forward {
                score += 200.0;
            }
            if making_run {
                score += 150.0;
            }
            if opponents_near == 0 {
                score += 100.0;
            }

            if let Some((_, best_score)) = &best_target {
                if score > *best_score {
                    best_target = Some((teammate, score));
                }
            } else {
                best_target = Some((teammate, score));
            }
        }

        best_target.map(|(t, _)| t)
    }

    /// Which zone (`docs/zone-staged-possession-progression.md`'s
    /// stage boundary) this player's team is currently in, on-ball
    /// decision-relevant. Zone 2 = `BallZone::AttackingThird`; Zone 1
    /// = everything else (`DefensiveThird`/`MiddleThird`).
    fn in_zone2(&self, ctx: &StateProcessingContext) -> bool {
        matches!(
            ctx.team().tactical().ball_zone,
            BallZone::AttackingThird
        )
    }

    fn in_zone1(&self, ctx: &StateProcessingContext) -> bool {
        !self.in_zone2(ctx)
    }

    /// Dispatches the open-space check by zone. Both call sites of the
    /// old, single `has_open_space_ahead` (the carry-forward gate in
    /// `process()` and `should_retain_possession`) now route through
    /// this, so Zone 1 becomes goal-blind everywhere at once rather
    /// than in only one of the two places
    /// (docs/zone-staged-possession-progression.md, Chunk 1 conflict
    /// check item 1). Zone 2's branch was the old, single-vector
    /// `has_open_space_ahead_goalcone` through Chunk 5; Chunk 6
    /// replaced it with `zone2_open_space_heading` (a real fan,
    /// goal-aware but not goal-funneled) — that old function is now
    /// deleted, not just unused.
    fn has_open_space_ahead_zoned(&self, ctx: &StateProcessingContext) -> bool {
        if self.in_zone1(ctx) {
            self.zone1_open_space_heading(ctx).is_some()
        } else {
            self.zone2_open_space_heading(ctx).is_some()
        }
    }

    /// Zone 1 (build-up/progression) open-space scan — goal-blind by
    /// design. Samples a fan of headings relative to the generic
    /// attacking direction (never a specific goal point — "no
    /// opponent_goal_position() anywhere in this zone's on-ball
    /// movement/carry decision"), covering both a genuine forward
    /// cone AND wide options, and measures how far each heading is
    /// clear of opponents. Returns the best heading if it clears a
    /// real minimum worth carrying into; `None` when nothing ahead
    /// offers genuine room, so callers fall through to the existing
    /// pass/dribble logic below instead of forcing a carry.
    ///
    /// `MIN_OPEN_SPACE`/the scan radius are reasoned estimates, not
    /// sourced against real tracking data — flagged per this project's
    /// own calibration discipline (docs/zone-staged-possession-
    /// progression.md's own guardrail: "source real numbers... flag as
    /// a reasoned estimate if no dataset exists, don't invent
    /// silently").
    fn zone1_open_space_heading(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let player_pos = ctx.player.position;
        let attack_dir: f32 = match ctx.player.side {
            Some(PlayerSide::Left) => 1.0,
            Some(PlayerSide::Right) => -1.0,
            None => return None,
        };

        // Tie-break: among headings tied for the best open distance —
        // very often several/all 7, since Zone 1 build-up play is
        // rarely fully congested — prefer whichever is most aligned
        // with the carrier's OWN current momentum (`ctx.player.velocity`,
        // already tracked every tick, no new plumbing needed), falling
        // back to a fair reservoir sample only when there's no
        // established direction yet (just received the ball, near-
        // stationary).
        //
        // A flat "50% chance to swap on tie" was tried first and
        // rejected on two grounds, both measured, not assumed: (1) it's
        // not even a fair N-way tie-break (the last-processed candidate
        // won 50% of the time vs ~1.6% for the first — a real +16.6°
        // mean bias toward the array's later entries); (2) even fixed
        // with a proper 1/N reservoir sample, re-rolling a fresh RNG
        // winner independently every tick zig-zagged the chosen heading
        // wildly call to call (raw trace, real match, player 107:
        // -45°,-70°,-20°,-45°,-45°,-45°,-70°,-70°,-20°,-45°,+70°,-70°...
        // — a jittery random walk, not a purposeful run, and the exact
        // oscillation risk docs/zone-staged-possession-progression.md's
        // own guardrail calls out). Anchoring to momentum instead is a
        // real football behaviour (a carrier holds their line unless
        // the picture genuinely changes) and needs no persistent
        // per-player state — it's a deterministic secondary sort key
        // over the SAME tied candidates, not a memory of past ticks.
        let momentum_dir = (ctx.player.velocity.magnitude() > 0.05)
            .then(|| ctx.player.velocity.normalize());

        let mut best: Option<(Vector3<f32>, f32)> = None;
        let mut tie_count: u32 = 0;
        for &angle_deg in ZONE1_HEADING_ANGLES_DEG.iter() {
            let angle = angle_deg.to_radians();
            let heading = Vector3::new(
                attack_dir * angle.cos(),
                attack_dir * angle.sin(),
                0.0,
            )
            .normalize();

            let open_dist = Self::open_dist_for_heading(ctx, player_pos, heading);

            match best {
                None => {
                    best = Some((heading, open_dist));
                    tie_count = 1;
                }
                Some((_, best_dist)) if open_dist > best_dist + 0.5 => {
                    best = Some((heading, open_dist));
                    tie_count = 1;
                }
                Some((best_heading, best_dist)) if (open_dist - best_dist).abs() <= 0.5 => {
                    tie_count += 1;
                    let replace = match momentum_dir {
                        // Deterministic argmax-by-momentum-alignment
                        // over the tied set — correctly converges to
                        // the single best-aligned tied heading by the
                        // end of the loop regardless of scan order.
                        Some(m) => heading.dot(&m) > best_heading.dot(&m),
                        // No established direction yet — fall back to
                        // the fair 1/tie_count reservoir sample.
                        None => ctx.context.rng.unit_f32() < 1.0 / tie_count as f32,
                    };
                    if replace {
                        best = Some((heading, open_dist));
                    }
                }
                _ => {}
            }
        }

        best.and_then(|(heading, dist)| (dist >= ZONE1_MIN_OPEN_SPACE).then_some(heading))
    }

    /// Nearest opponent roughly along `heading` from `origin` (within a
    /// generous cone, dot > 0.75) caps how much room that heading
    /// offers; an empty heading scores the full scan radius. The one
    /// per-heading scan shared by `zone1_open_space_heading` (origin =
    /// the carrier's own position, needs the winning HEADING for
    /// movement) and `zone1_max_open_distance` (Chunk 3, origin = a
    /// candidate teammate's position, needs only the best VALUE) — a
    /// single definition so the two can't silently drift apart.
    /// `origin` may be off the carrier's own position, so this uses
    /// `nearby_at`, not `nearby`.
    fn open_dist_for_heading(
        ctx: &StateProcessingContext,
        origin: Vector3<f32>,
        heading: Vector3<f32>,
    ) -> f32 {
        ctx.players()
            .opponents()
            .nearby_at(origin, ZONE1_SCAN_MAX)
            .filter(|opp| {
                let to_opp = opp.position - origin;
                let dist = to_opp.magnitude();
                dist > 1.0 && to_opp.normalize().dot(&heading) > 0.75
            })
            .map(|opp| (opp.position - origin).magnitude())
            .fold(ZONE1_SCAN_MAX, f32::min)
    }

    /// Chunk 3 (docs/zone-staged-possession-progression.md): the best
    /// open distance available to a HYPOTHETICAL carrier standing at
    /// `origin` — same fan, same scan, as `zone1_open_space_heading`,
    /// just evaluated from someone else's position and returning the
    /// VALUE, not a chosen direction (no tie-break needed — every tied
    /// heading has the identical value by definition). Used to ask "how
    /// open is this teammate," genuinely goal-blind, without touching
    /// `pass_value_from`/`evaluator.rs` (see the Chunk 3 conflict check
    /// in the design doc for why those were deliberately left alone —
    /// shared, cross-cutting, dominated by their own goal-relative
    /// terms).
    fn zone1_max_open_distance(&self, ctx: &StateProcessingContext, origin: Vector3<f32>) -> f32 {
        let attack_dir: f32 = match ctx.player.side {
            Some(PlayerSide::Left) => 1.0,
            Some(PlayerSide::Right) => -1.0,
            None => return 0.0,
        };
        ZONE1_HEADING_ANGLES_DEG
            .iter()
            .map(|&angle_deg| {
                let angle = angle_deg.to_radians();
                let heading = Vector3::new(attack_dir * angle.cos(), attack_dir * angle.sin(), 0.0)
                    .normalize();
                Self::open_dist_for_heading(ctx, origin, heading)
            })
            .fold(0.0_f32, f32::max)
    }

    /// Same-binary ablation toggle for Chunk 3's teammate-open-space
    /// comparison ONLY — `zone1_open_space_heading`/
    /// `calculate_open_space_movement`/the widened Zone 1 bound (Chunks
    /// 1/2) are untouched by this flag; only whether a similarly-open
    /// teammate can redirect the ball away from carrying. Mirrors
    /// `spacing.rs::mobility_ablated`'s own `OF_SCORE_BLIND` idiom —
    /// diagnostic/comparison only, inert unless `ZONE1_CHUNK3_ABLATE`
    /// is set, checked once and cached. Kept permanently (not a
    /// temporary diagnostic) so any later chunk can be A/B'd with
    /// Chunk 3 on vs. off without a rebuild/git-stash cycle.
    fn zone1_chunk3_ablated() -> bool {
        use std::sync::OnceLock;
        static ABLATED: OnceLock<bool> = OnceLock::new();
        *ABLATED.get_or_init(|| std::env::var("ZONE1_CHUNK3_ABLATE").is_ok())
    }

    /// Same-binary ablation toggle for Chunk 4's off-ball reaction
    /// widening ONLY — Chunks 1-3's own mechanisms are untouched by
    /// this flag. Mirrors `zone1_chunk3_ablated`'s exact idiom. When
    /// set, `in_lateral_zone` reverts to exactly its pre-Chunk-4 value
    /// (`base_in_lateral_zone`), i.e. only players whose own start-
    /// position band already overlaps the ball's current lateral
    /// position can react.
    fn zone4_ablated() -> bool {
        use std::sync::OnceLock;
        static ABLATED: OnceLock<bool> = OnceLock::new();
        *ABLATED.get_or_init(|| std::env::var("ZONE4_ABLATE").is_ok())
    }

    /// Chunk 4: is a TEAMMATE (never `ctx.player` themselves — this is
    /// only ever called from the off-ball branch, where that's already
    /// guaranteed) currently carrying the ball through a genuine,
    /// sustained, still-open Zone-1 advance? See the call site's own
    /// comment for the measured rationale behind each of the three
    /// conditions below (real space, sustained hold, still moving) —
    /// none of the three constants here are new inventions: the space
    /// bar is Chunk 1's own `ZONE1_MIN_OPEN_SPACE`, and the duration/
    /// speed bars are reasoned estimates in the same spirit as this
    /// file's other unsourced Zone-1 constants (flagged as such, not
    /// silently invented — no dataset exists for "how long must a
    /// carry be sustained before teammates should react to it").
    fn teammate_open_carry_active(
        &self,
        ctx: &StateProcessingContext,
    ) -> Option<MatchPlayerLite> {
        if !self.in_zone1(ctx) {
            return None;
        }
        let owner_id = ctx.ball().owner_id()?;
        let owner = ctx.context.players.by_id(owner_id)?;
        if owner.team_id != ctx.player.team_id {
            return None;
        }
        let holder = ctx.player().get(owner_id);
        if self.zone1_max_open_distance(ctx, holder.position) < ZONE1_MIN_OPEN_SPACE {
            return None;
        }
        const OPEN_CARRY_MIN_DURATION_TICKS: u32 = 60; // ~600ms — past every "quick touch" window elsewhere in this file
        if ctx.tick_context.ball.ownership_duration <= OPEN_CARRY_MIN_DURATION_TICKS {
            return None;
        }
        const OPEN_CARRY_MIN_HOLDER_SPEED: f32 = 0.10; // u/tick — genuinely still moving, not standing on it
        if holder.velocity(ctx).magnitude() <= OPEN_CARRY_MIN_HOLDER_SPEED {
            return None;
        }
        Some(holder)
    }

    /// Chunk 3: "if there is a teammate who has roughly the same amount
    /// of free area — go there. But if not, carry the ball all the way
    /// you can." `my_open` is the carrier's own already-computed open
    /// distance (`zone1_max_open_distance` at their own position).
    /// Considers any real, nearby, passable teammate — not
    /// position-restricted to "central," per the design doc's own
    /// generalisation of Pavel's illustrative example — and returns
    /// whichever qualifying teammate is most open, or `None` if nobody
    /// clears the "more or less as open" bar.
    fn zone1_similarly_open_teammate<'a>(
        &self,
        ctx: &StateProcessingContext<'a>,
        my_open: f32,
    ) -> Option<MatchPlayerLite> {
        ctx.players()
            .teammates()
            .nearby(ZONE1_TEAMMATE_SEARCH_RADIUS)
            .filter(|t| ctx.player().has_clear_pass(t.id))
            .filter_map(|t| {
                let their_open = self.zone1_max_open_distance(ctx, t.position);
                (their_open + ZONE1_OPENNESS_TOLERANCE >= my_open).then_some((t, their_open))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(t, _)| t)
    }

    /// Zone 1 goal-blind carrying movement. Moves along the best open
    /// heading found by `zone1_open_space_heading`; falls back to a
    /// plain attack-direction advance (never a goal point) when no
    /// heading clears the open-space bar, so the carrier keeps moving
    /// forward rather than freezing.
    fn calculate_open_space_movement(&self, ctx: &StateProcessingContext) -> Vector3<f32> {
        let player_pos = ctx.player.position;
        let heading = self.zone1_open_space_heading(ctx).unwrap_or_else(|| {
            let attack_dir: f32 = match ctx.player.side {
                Some(PlayerSide::Left) => 1.0,
                Some(PlayerSide::Right) => -1.0,
                None => 0.0,
            };
            Vector3::new(attack_dir, 0.0, 0.0)
        });

        // Same smooth sinusoidal lateral sway as the old goal-relative
        // version, just relative to the chosen heading instead of a
        // fixed goal point.
        let phase = (ctx.in_state_time as f32) * std::f32::consts::TAU / 60.0;
        let sway = phase.sin() * 0.2;
        let lateral = Vector3::new(-heading.y * sway, heading.x * sway, 0.0);

        let target = player_pos + (heading + lateral).normalize() * 40.0;

        SteeringBehavior::Arrive {
            target,
            slowing_distance: 20.0,
        }
        .calculate(ctx.player)
        .velocity
    }

    /// Chunk 6 (docs/zone-staged-possession-progression.md): "1v1 vs
    /// 2v1+ branching near the box" — genuinely new, no existing
    /// primitive answered this (Chunk 6 conflict check item 6:
    /// `TeamTacticalState::side_density_left/center/right` is
    /// own-team-only and pitch-thirds-coarse, `carrier_meets_defender`
    /// counts opponents only). Forward-cone counting (not
    /// omnidirectional) — the players that matter for "am I isolated
    /// or backed up" are the ones between the carrier and goal, not
    /// behind them; reuses `should_dribble`'s own `opponents_ahead`
    /// cone shape (`0.3` dot-product threshold) a few lines above in
    /// `process()`, for consistency rather than a third
    /// independently-tuned cone. The carrier counts as the 1st
    /// attacker. `ZONE2_LOCAL_NUMBERS_RADIUS` is a reasoned estimate —
    /// no sourced dataset exists for "how wide is a real local
    /// 1v1/2v1 duel zone near the box."
    fn zone2_local_numbers(&self, ctx: &StateProcessingContext) -> (usize, usize) {
        let player_pos = ctx.player.position;
        let goal_pos = ctx.player().opponent_goal_position();
        let to_goal = (goal_pos - player_pos).normalize();

        let defenders = ctx
            .players()
            .opponents()
            .nearby(ZONE2_LOCAL_NUMBERS_RADIUS)
            .filter(|opp| {
                (opp.position - player_pos).normalize().dot(&to_goal)
                    > ZONE2_LOCAL_NUMBERS_FORWARD_COS
            })
            .count();
        let attackers = 1 + ctx
            .players()
            .teammates()
            .nearby(ZONE2_LOCAL_NUMBERS_RADIUS)
            .filter(|t| {
                (t.position - player_pos).normalize().dot(&to_goal)
                    > ZONE2_LOCAL_NUMBERS_FORWARD_COS
            })
            .count();
        (attackers, defenders)
    }

    /// Zone 2 (final third) open-space scan — goal-AWARE but not
    /// goal-FUNNELED (docs/zone-staged-possession-progression.md's own
    /// framing: "keep SOME goal-awareness... just not have every
    /// candidate collapse toward goal-center"). Reuses Zone 1's exact
    /// attack-direction-relative heading fan and the shared
    /// `open_dist_for_heading` primitive — this is what makes
    /// wide-angle-shot positions and off-axis decoy-run spacing
    /// reachable at all (the design doc's "omnidirectional dribbling"
    /// item, applied to the `has_open_space_ahead_goalcone`/
    /// `calculate_simple_ball_movement` mechanism specifically — see
    /// the Chunk 6 conflict check item 2 for why `carry_candidates`'
    /// own separate, still fully goal-relative `forward`/`lateral` fan
    /// is deliberately NOT touched here).
    ///
    /// Each heading's score gets a soft goal-alignment bonus
    /// (`heading.dot(&to_goal).max(0.0) * ZONE2_GOAL_AWARENESS_BONUS`)
    /// only when `zone2_local_numbers` reads a genuine 1v1-or-worse
    /// (`defenders >= attackers`) — Pavel's own explicit branch: "if
    /// there is a teammate who is in the centre and has the same
    /// amount of free area... go there... if you meet some defenders,
    /// start the positional attack... [or] if not — 2v1 — spreading is
    /// better [vs] moving to the centre for a 1v1." Under a genuine
    /// overload the bonus is zero and the raw open-space scan alone
    /// decides, naturally favouring width without a hard-coded "always
    /// go wide" rule.
    ///
    /// A fresh copy of Zone 1's own already-verified momentum-anchored
    /// tie-break loop, not a shared refactor of
    /// `zone1_open_space_heading` itself — that function is already
    /// proven and shipped (Chunk 1/2); touching it to parameterize a
    /// bias term risks the exact "verified mechanism gets silently
    /// altered" failure this project's own discipline warns against.
    /// `open_dist_for_heading` (the actual per-heading opponent scan)
    /// IS shared, so the two functions can't silently disagree about
    /// what "open" measures.
    fn zone2_open_space_heading(&self, ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        let player_pos = ctx.player.position;
        let attack_dir: f32 = match ctx.player.side {
            Some(PlayerSide::Left) => 1.0,
            Some(PlayerSide::Right) => -1.0,
            None => return None,
        };

        let goal_pos = ctx.player().opponent_goal_position();
        let to_goal = (goal_pos - player_pos).normalize();
        let (attackers, defenders) = self.zone2_local_numbers(ctx);
        let commit_centrally = defenders >= attackers;

        let momentum_dir =
            (ctx.player.velocity.magnitude() > 0.05).then(|| ctx.player.velocity.normalize());

        let mut best: Option<(Vector3<f32>, f32)> = None;
        let mut tie_count: u32 = 0;
        for &angle_deg in ZONE1_HEADING_ANGLES_DEG.iter() {
            let angle = angle_deg.to_radians();
            let heading =
                Vector3::new(attack_dir * angle.cos(), attack_dir * angle.sin(), 0.0).normalize();

            let open_dist = Self::open_dist_for_heading(ctx, player_pos, heading);
            let goal_bias = if commit_centrally {
                heading.dot(&to_goal).max(0.0) * ZONE2_GOAL_AWARENESS_BONUS
            } else {
                0.0
            };
            let score = open_dist + goal_bias;

            match best {
                None => {
                    best = Some((heading, score));
                    tie_count = 1;
                }
                Some((_, best_score)) if score > best_score + 0.5 => {
                    best = Some((heading, score));
                    tie_count = 1;
                }
                Some((best_heading, best_score)) if (score - best_score).abs() <= 0.5 => {
                    tie_count += 1;
                    let replace = match momentum_dir {
                        Some(m) => heading.dot(&m) > best_heading.dot(&m),
                        None => ctx.context.rng.unit_f32() < 1.0 / tie_count as f32,
                    };
                    if replace {
                        best = Some((heading, score));
                    }
                }
                _ => {}
            }
        }

        // The stored `score` includes the goal-alignment bias, which
        // isn't a distance — re-derive the winner's raw open distance
        // to check the actual space-worth-carrying-into bar.
        best.and_then(|(heading, _)| {
            let open_dist = Self::open_dist_for_heading(ctx, player_pos, heading);
            (open_dist >= ZONE1_MIN_OPEN_SPACE).then_some(heading)
        })
    }

    /// Zone 2 carrying movement — mirrors `calculate_open_space_
    /// movement`'s exact shape (sinusoidal sway around the chosen
    /// heading), sourcing the heading from `zone2_open_space_heading`
    /// instead of Zone 1's fully goal-blind scan. Falls back to a
    /// plain toward-goal advance (not Zone 1's generic-attack-
    /// direction fallback) when nothing clears the open-space bar —
    /// Zone 2 is inherently goal-directed once no real alternative
    /// exists, unlike Zone 1's "just keep advancing" fallback.
    fn calculate_zone2_movement(&self, ctx: &StateProcessingContext) -> Vector3<f32> {
        let player_pos = ctx.player.position;
        let heading = self.zone2_open_space_heading(ctx).unwrap_or_else(|| {
            let goal_pos = ctx.player().opponent_goal_position();
            (goal_pos - player_pos).normalize()
        });

        let phase = (ctx.in_state_time as f32) * std::f32::consts::TAU / 60.0;
        let sway = phase.sin() * 0.2;
        let lateral = Vector3::new(-heading.y * sway, heading.x * sway, 0.0);

        let target = player_pos + (heading + lateral).normalize() * 40.0;

        SteeringBehavior::Arrive {
            target,
            slowing_distance: 20.0,
        }
        .calculate(ctx.player)
        .velocity
    }

    fn is_congested_near_boundary(&self, ctx: &StateProcessingContext) -> bool {
        // Check if near any boundary (within 20 units)
        let field_width = ctx.context.field_size.width as f32;
        let field_height = ctx.context.field_size.height as f32;
        let pos = ctx.player.position;

        let near_boundary = pos.x < 20.0
            || pos.x > field_width - 20.0
            || pos.y < 20.0
            || pos.y > field_height - 20.0;

        if !near_boundary {
            return false;
        }

        // Count all nearby players (teammates + opponents) within 15 units
        let nearby_teammates = ctx
            .tick_context
            .grid
            .teammates(ctx.player.id, 0.0, 15.0)
            .count();
        let nearby_opponents = ctx.tick_context.grid.opponents(ctx.player.id, 15.0).count();
        let total_nearby = nearby_teammates + nearby_opponents;

        // If 3 or more players nearby (congestion), need to clear
        total_nearby >= 3
    }

    /// Check if wide midfielder should deliver a cross into the box
    fn should_cross(&self, ctx: &StateProcessingContext) -> bool {
        let field_height = ctx.context.field_size.height as f32;
        let y = ctx.player.position.y;
        let wide_margin = field_height * 0.2;

        // Must be in a wide channel (top or bottom 20%)
        let is_wide = y < wide_margin || y > field_height - wide_margin;
        if !is_wide {
            return false;
        }

        // Must be in attacking third
        let goal_dist = ctx.ball().distance_to_opponent_goal();
        let field_width = ctx.context.field_size.width as f32;
        if goal_dist > field_width * 0.35 {
            return false;
        }

        // Must have at least 1 teammate within 150 units of opponent goal
        let goal_pos = ctx.player().opponent_goal_position();
        let teammates_in_box = ctx.players().teammates().nearby_at(goal_pos, 150.0).count();
        if teammates_in_box < 1 {
            return false;
        }

        // Crossing skill scales the willingness smoothly (sigmoid pivot
        // at 8/20). A bad crosser still attempts a deep cross
        // occasionally; an elite one almost always does.
        let p = SkillCurve::new(ctx.player.skills.technical.crossing, 8.0, 0.6).probability();
        ctx.context.rng.unit_f32() < p
    }

    /// Find a safe backward/lateral pass target for tempo control.
    /// Prefers defenders and GK when coach says to slow down.
    fn find_safe_backward_pass(&self, ctx: &StateProcessingContext) -> Option<MatchPlayerLite> {
        let player_pos = ctx.player.position;
        let own_goal = ctx.ball().direction_to_own_goal();

        let mut best: Option<(MatchPlayerLite, f32)> = None;

        for teammate in ctx.players().teammates().nearby(250.0) {
            let dist = (teammate.position - player_pos).magnitude();
            if dist < 15.0 {
                continue;
            }

            if !ctx.player().has_clear_pass(teammate.id) {
                continue;
            }

            let opp_near = ctx.tick_context.grid.opponents(teammate.id, 12.0).count();
            if opp_near >= 2 {
                continue;
            }

            let group = teammate.tactical_positions.position_group();
            let mut score = 0.0f32;

            match group {
                PlayerFieldPositionGroup::Goalkeeper => score += 45.0,
                PlayerFieldPositionGroup::Defender => score += 35.0,
                PlayerFieldPositionGroup::Midfielder => score += 10.0,
                _ => {}
            }

            // Prefer players closer to own goal (backward direction)
            let teammate_to_own = (own_goal - teammate.position).magnitude();
            let self_to_own = (own_goal - player_pos).magnitude();
            if teammate_to_own < self_to_own {
                score += 20.0;
            }

            if opp_near == 0 {
                score += 15.0;
            }

            if let Some((_, best_score)) = &best {
                if score > *best_score {
                    best = Some((teammate, score));
                }
            } else {
                best = Some((teammate, score));
            }
        }

        best.map(|(t, _)| t)
    }
}
