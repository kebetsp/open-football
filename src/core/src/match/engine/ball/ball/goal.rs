//! Out-of-play resolution: actual goals, and every other endline
//! crossing (over the bar, wide of the posts, or genuinely IN the goal
//! frame but not credited as a goal — e.g. a carried ball with no
//! recent shot attribution) — all resolved by the same real Law-12 rule
//! (corner if the defending team touched it last, otherwise goal kick),
//! via the shared `resolve_corner_or_goal_kick`. Both `check_over_goal`
//! and `check_wide_of_goal` do only their own geometric trigger test and
//! then hand off to that one resolver — previously `check_over_goal` had
//! its own, simpler, WRONG logic that always awarded a goal kick
//! regardless of who touched the ball last, missing the common real case
//! of a defensive header/clearance going out for a corner over the bar
//! (engine audit Module 2 rewrite, 2026-08-28).
//!
//! The wide-of-goal/over-goal flow stages the set-piece teleport via
//! `pending_set_piece_teleport` since the ball can't move other players'
//! positions itself.
//!
//! Every one of these functions is now exhaustive by construction: taker
//! and goalkeeper lookups fall back progressively rather than silently
//! declining to resolve anything (see `resolve_corner_or_goal_kick`'s
//! `.or_else` fallbacks) — a genuine crossing must always produce a real
//! restart, matching the Laws of the Game (a stoppage always happens and
//! someone always takes it), never a silent freeze.

use super::Ball;
use crate::r#match::PassOriginRestart;
use crate::r#match::ball::events::{BallEvent, BallGoalEventMetadata, GoalSide};
use crate::r#match::engine::set_pieces::{CornerScores, pick_corner_routine};
use crate::r#match::events::EventCollection;
use crate::r#match::{MatchContext, MatchPlayer, PlayerSide};
use nalgebra::Vector3;
use std::cmp::Ordering;

impl Ball {
    pub(super) fn check_goal(&mut self, context: &MatchContext, result: &mut EventCollection) {
        // Guard: don't detect another goal if one was already scored this tick
        if self.goal_scored {
            return;
        }

        // realism-bug (2026-08-28, engine audit Module 2): this used to
        // return unconditionally whenever `current_owner.is_some()`, to
        // stop a defender "carrying" the ball into his own goal via an
        // artifact of the OLD `check_boundary_collision` blind boundary
        // snap (which could shove an owned ball across the line by
        // accident, not a genuine deliberate crossing). That snap-back
        // mechanism is gone — every boundary resolution is now exhaustive
        // and geometry-driven, so the original false-positive source no
        // longer exists. Per the real Laws of the Game, the whole ball
        // crossing the whole line is a goal (or own goal) regardless of
        // HOW it got there — kicked, deflected, or genuinely carried
        // (this engine's ball-follows-owner "at the feet" abstraction, per
        // Pavel's 2026-08-28 sign-off, IS a real physical object under a
        // player's control, so a player+ball crossing the line together
        // is exactly what real carrying-it-in looks like). The existing
        // goalscorer-attribution logic below already treats `current_owner`
        // as its own top-priority scorer candidate, and an own goal
        // (`is_auto_goal`) is already exempt from the "requires a recent
        // shot" gate a few lines down — so removing this early return
        // lets a genuine carried own goal score correctly, and lets a
        // carried ATTACKING goal fall through to the same "not a real
        // goal attempt, resolve as an out-of-play crossing instead" path
        // every other undeserving crossing already takes (see
        // `check_wide_of_goal`, which now runs unconditionally on any
        // unscored endline crossing including this one).

        if let Some(goal_side) = context.goal_positions.is_goal(self.position) {
            // Prefer current_owner (e.g. player carrying ball into goal)
            // Fall back to previous_owner (e.g. shooter or passer whose ball went in)
            // realism-bug (2026-08-01): a THIRD fallback to
            // `last_shot_shooter_id`, gated on a live tracked shot
            // (`cached_shot_target.is_some()`) so it can never misattribute
            // an unrelated later crossing to a stale shooter id. Direct
            // free kicks travel 150-250+ units in one uninterrupted
            // flight — long enough to trigger an existing, separate
            // safety-net mechanism elsewhere in the engine that nulls
            // `previous_owner` once the ball drifts far from its last
            // known owner mid-flight (built for pass-tracking, not shots).
            // Once that fires, `current_owner.or(previous_owner)` came up
            // empty and this whole branch was skipped — traced directly:
            // a real on-target, unsaved shot flew straight through the
            // goal frame to x=865 (25 units past the line) over a full
            // second, never credited, before falling through to an
            // out-of-bounds/goal-kick resolution instead. `last_shot_shooter_id`
            // is set at the moment of the shot and isn't touched by that
            // same clearing mechanism, so it survives exactly the case
            // this fallback needs.
            //
            // 2026-08-30 (Pavel's explicit instruction, "for this
            // iteration"): redefined so whoever genuinely touched the
            // ball LAST is the scorer — `last_touch_player_id` (the same
            // field `resolve_corner_or_goal_kick` already trusts as "the
            // true last contact, maintained by `record_touch` on every
            // control change / block / save") now takes priority over
            // the old owner-based chain. This is deliberate instrumentation
            // for the shot/cross-block investigation: a shot or cross that
            // deflects off a defender into the net should now show up as
            // that defender's own goal, not silently re-attributed to the
            // original attacker, so deflection-originated goals become
            // trivially countable via the `auto_goal` flag. Verified
            // before making this change that a clean, undeflected shot is
            // unaffected: `handle_shoot_event` never calls `record_touch`
            // itself, so `last_touch_player_id` stays on the shooter (set
            // when he received/controlled the ball) all the way through
            // an unblocked strike — it only changes when someone else
            // (a block, parry, interception, keeper touch) genuinely
            // makes contact, which is exactly the real-world "last touch"
            // rule. `current_owner`/`previous_owner`/`last_shot_shooter_id`
            // stay as fallbacks for the rare case `last_touch_player_id`
            // is `None` (e.g. very start of a passage of play).
            let goalscorer_fallback = self
                .last_touch_player_id
                .or(self.current_owner)
                .or(self.previous_owner)
                .or_else(|| {
                    if self.cached_shot_target.is_some() {
                        self.last_shot_shooter_id
                    } else {
                        None
                    }
                });
            if let Some(goalscorer) = goalscorer_fallback {
                let Some(player) = context.players.by_id(goalscorer) else {
                    return;
                };
                let is_auto_goal = match player.side {
                    Some(PlayerSide::Left) => goal_side == GoalSide::Home,
                    Some(PlayerSide::Right) => goal_side == GoalSide::Away,
                    _ => false,
                };

                // Require a recent shot or a live shot-target. Without
                // this, passes that happen to roll across the goal line
                // (receiver missed, ball trajectory drifted) credit the
                // passer with a goal — which was producing 10-15 "goals"
                // per match per team that never involved a Shoot event.
                // Real football treats those as out-of-bounds → goal
                // kick, not a goal. Exception: auto-goal path skips this
                // check, because an own goal happens via touch, not a
                // shot by the credited player.
                if !is_auto_goal {
                    let current_tick = context.current_tick();
                    let recent_shot = context
                        .players
                        .by_id(goalscorer)
                        .map(|p| {
                            p.memory.shots_taken > 0
                                && current_tick.saturating_sub(p.memory.last_shot_tick) < 300
                        })
                        .unwrap_or(false);
                    let shot_in_flight = self.cached_shot_target.is_some();
                    if !recent_shot && !shot_in_flight {
                        // Not a shot — treat as ball out of play, not a goal.
                        return;
                    }

                    // Indirect free-kick rule: the kick itself can't
                    // produce a goal. If the ball came from an
                    // IndirectFreeKick origin and only the taker has
                    // touched it since (no second player), the goal
                    // must not stand. We approximate "no second touch"
                    // by checking that the taker is the SOLE recent
                    // passer. If anyone else is in `recent_passers`,
                    // somebody has taken a touch and a goal is legal.
                    if self.pass_origin_restart == PassOriginRestart::IndirectFreeKick {
                        let any_second_touch = self
                            .recent_passers
                            .iter()
                            .any(|&id| id != goalscorer);
                        if !any_second_touch {
                            // Reject: ball stays live, but no goal.
                            return;
                        }
                    }
                }

                // 2026-08-30 (Pavel's explicit instruction, "for this
                // iteration"): this used to re-credit a short-duration
                // touch (<300ms) back to the original attacker, on the
                // theory that a genuine own goal requires the defender to
                // have had meaningful possession, not just a graze. That's
                // exactly backwards for what this iteration needs — a
                // shot/cross that deflects off a defender and loops in is
                // supposed to be recorded as HIS own goal (see the
                // `goalscorer_fallback` comment above), so this
                // re-crediting is disabled: the scorer/own-goal
                // determination above (driven by `last_touch_player_id`)
                // now stands as final, unconditionally.
                let (final_scorer, final_is_auto_goal) = (goalscorer, is_auto_goal);

                // Assist provider: the qualified assister captured at
                // shot time (§9.1.1). `last_shot_assister_id` is only
                // Some when the shooter received a completed teammate
                // pass inside the key-pass window (~3 s) before
                // shooting — a long dribble/solo run yields None, so no
                // assist is credited, matching how real assists are
                // judged. Gated on the scorer being that shot's shooter
                // so a deflection credited to another attacker can't
                // inherit an unrelated pass link.
                let assist_player_id = if !final_is_auto_goal
                    && self.last_shot_shooter_id == Some(final_scorer)
                {
                    self.last_shot_assister_id.filter(|&id| id != final_scorer)
                } else {
                    None
                };

                let goal_event_metadata = BallGoalEventMetadata {
                    side: goal_side,
                    goalscorer_player_id: final_scorer,
                    assist_player_id,
                    auto_goal: final_is_auto_goal,
                };

                result.add_ball_event(BallEvent::Goal(goal_event_metadata));

                // Kickoff setup and reset only when a valid scoreable goal has
                // been identified. Previously these ran unconditionally in the
                // outer `is_goal` branch — so a ball that crossed the line with
                // both `current_owner` and `previous_owner` None triggered a
                // full centre-circle restart with no score change (phantom kickoff).
                self.kickoff_team_side = match goal_side {
                    GoalSide::Home => Some(PlayerSide::Left),
                    GoalSide::Away => Some(PlayerSide::Right),
                };
                self.goal_scored = true;
                self.reset();
            }
            // No scorer found, or the goal-crediting rules above declined
            // (no recent shot, indirect-FK second-touch rule, etc.): skip
            // the reset. Ball stays live, `self.goal_scored` stays false,
            // and `check_wide_of_goal` (which now runs unconditionally on
            // any unscored endline crossing, not just ones outside the
            // posts) resolves it as a corner or goal kick on this same
            // tick, per the real last-toucher rule.
        }
    }

    /// Ball crossed the goal line within goal width but above crossbar.
    /// Real Law 12: still decided by who touched last, exactly like any
    /// other endline crossing — a defensive header/clearance over the
    /// bar is a CORNER, not automatically a goal kick. Delegates to the
    /// shared resolver instead of (as before) always handing it straight
    /// to the defending goalkeeper.
    pub(super) fn check_over_goal(
        &mut self,
        context: &mut MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
    ) {
        let over_side = match context.goal_positions.is_over_goal(self.position) {
            Some(side) => side,
            None => return,
        };
        self.resolve_corner_or_goal_kick(context, players, events, over_side);
    }

    /// Ball crossed an endline (x <= 0 or x >= field_width) and was NOT
    /// credited as a goal this tick (either genuinely wide of the posts,
    /// or within the frame but `check_goal` declined — e.g. a carried
    /// ball with no recent-shot attribution). In real football this is
    /// always a goal kick OR a corner kick, decided by which team last
    /// touched the ball — never silence. `check_over_goal` (run earlier
    /// this tick) already claims the "over the bar" sub-region, so this
    /// function only needs to skip that to avoid double-resolving the
    /// same physical crossing twice in one tick.
    pub(super) fn check_wide_of_goal(
        &mut self,
        context: &mut MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
    ) {
        if self.goal_scored {
            return;
        }
        let field_width = context.field_size.width as f32;

        let crossed_side = if self.position.x <= 0.0 {
            Some(GoalSide::Home)
        } else if self.position.x >= field_width {
            Some(GoalSide::Away)
        } else {
            None
        };
        let Some(side) = crossed_side else {
            return;
        };

        // Already claimed by check_over_goal this tick (which runs
        // earlier in the dispatch order — see `mod.rs`): if it fired, it
        // already relocated the ball off the endline entirely, so this
        // geometric test would be moot in practice, but guard explicitly
        // for clarity and defence-in-depth rather than relying on call
        // order alone.
        if context.goal_positions.is_over_goal(self.position).is_some() {
            return;
        }

        self.resolve_corner_or_goal_kick(context, players, events, side);
    }

    /// Shared endline-crossing resolver: decide corner vs. goal kick by
    /// who touched the ball last (real Law 12), then run the full
    /// calibrated staging for whichever one applies. Called by both
    /// `check_over_goal` and `check_wide_of_goal` — the single place
    /// that owns "what happens when the ball goes out over this line."
    ///
    /// Every player lookup here (corner taker, goalkeeper) falls back
    /// progressively rather than silently doing nothing if the ideal
    /// candidate doesn't exist (e.g. every outfielder sent off) — a real
    /// restart always gets taken by someone, per Pavel's 2026-08-28
    /// sign-off on exhaustive boundary resolution.
    fn resolve_corner_or_goal_kick(
        &mut self,
        context: &mut MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
        side: GoalSide,
    ) {
        let field_width = context.field_size.width as f32;

        let defending_side = match side {
            GoalSide::Home => PlayerSide::Left,
            GoalSide::Away => PlayerSide::Right,
        };
        let attacking_side = match defending_side {
            PlayerSide::Left => PlayerSide::Right,
            PlayerSide::Right => PlayerSide::Left,
        };

        // Decide corner vs goal kick from the last player who TOUCHED the
        // ball. If the defending team put it out, it's a corner for the
        // attacking team. Use `last_touch_player_id` (the true last contact,
        // maintained by `record_touch` on every control change / block /
        // save) rather than `previous_owner` (the last OWNER). They differ
        // exactly on a DEFLECTION: when a defender blocks/parries/clears a
        // shot out, `previous_owner` is still the attacking SHOOTER, so the
        // ball was wrongly given as a goal kick — which is the dominant
        // reason the engine ran ~0.5 corners/match vs ~10 real. Falls back
        // to the owner when no touch is recorded.
        let last_toucher_side: Option<PlayerSide> = self
            .last_touch_player_id
            .or(self.previous_owner)
            .or(self.current_owner)
            .and_then(|pid| players.iter().find(|p| p.id == pid))
            .and_then(|p| p.side);

        let is_corner = last_toucher_side == Some(defending_side);

        if is_corner {
            // Attacking team gets a corner. Place ball at the nearest corner
            // flag and hand it to the attacking team's best corner taker.
            let corner_x = match side {
                GoalSide::Home => 2.0,
                GoalSide::Away => field_width - 2.0,
            };
            let field_height = context.field_size.height as f32;
            // Pick the near corner based on where the ball went out
            let near_top = self.position.y < field_height * 0.5;
            let corner_y = if near_top { 2.0 } else { field_height - 2.0 };

            // Find the attacking team's designated corner taker — score by
            // (crossing, technique, corners) like SetPieceSetup::choose.
            //
            // realism-bug (2026-09-04, "set-piece dribble" report): this
            // used to be restricted only to non-goalkeeper outfielders —
            // no role check at all — so a central defender with decent
            // crossing/technique could legitimately win the max_by score
            // and become "the taker." Real football never does this: a
            // corner is taken by a genuine attacking-minded player (a
            // winger, an attacking full-back), never a centre-back or
            // holding defender — a real defensive/tactical-doctrine
            // violation, not just an unlikely edge case. It also turned
            // out to be architecturally fatal: Forward/Midfielder states
            // have a dedicated `has_ball && is_team_attacking_corner →
            // Crossing` transition that actually delivers the ball
            // (`pick_corner_delivery`); Defender states have NO
            // equivalent — the closest thing, `DefenderWalkingState`'s
            // `AttackingCorner` transition, is explicitly gated `!has_ball`
            // (it's for a centre-back attacking SOMEONE ELSE's delivery,
            // not taking his own). A defender who wins this selection
            // therefore has no path to ever cross the ball — he just
            // falls through to ordinary defensive AI (a generic short
            // pass, an aimless hold, or a clearance) while holding the
            // ball at the corner flag, and the attack dissolves without
            // a delivery. Measured (25+20-match samples,
            // `SETPIECEDIAG_CORNER_TAKER`): ~7% of corners were awarded
            // to a Defender-role taker, and 0/84 real `CornerDelivery`
            // events across a 20-match sample were EVER attributed to
            // one of those ids — a 100% failure rate for exactly the
            // population this fix removes from the primary pool.
            //
            // Fix: three-tier fallback. Primary pool is Forward/
            // Midfielder outfielders only (the roles that can actually
            // deliver). Falls back to any non-GK outfielder (the old
            // behaviour, including defenders) only if the primary pool
            // is empty — e.g. every forward/midfielder has been sent
            // off. Genuine last resort (should be unreachable outside a
            // fully depleted side): if no eligible outfielder exists at
            // all, even the goalkeeper takes it rather than the restart
            // silently vanishing — real football always has SOMEONE
            // take a corner, even a makeshift choice.
            let taker_score = |p: &MatchPlayer| {
                p.skills.technical.crossing * 0.6
                    + p.skills.technical.technique * 0.3
                    + p.skills.technical.corners * 0.1
            };
            fn best_scored<'a>(
                iter: impl Iterator<Item = &'a MatchPlayer>,
                score: impl Fn(&MatchPlayer) -> f32,
            ) -> Option<&'a MatchPlayer> {
                iter.max_by(|a, b| {
                    score(a).partial_cmp(&score(b)).unwrap_or(Ordering::Equal)
                })
            }
            let taker = best_scored(
                players.iter().filter(|p| {
                    p.side == Some(attacking_side)
                        && (p.tactical_position.current_position.is_forward()
                            || p.tactical_position.current_position.is_midfielder())
                }),
                taker_score,
            )
            .or_else(|| {
                best_scored(
                    players.iter().filter(|p| {
                        p.side == Some(attacking_side)
                            && !p.tactical_position.current_position.is_goalkeeper()
                    }),
                    taker_score,
                )
            })
            .or_else(|| {
                best_scored(
                    players.iter().filter(|p| p.side == Some(attacking_side)),
                    taker_score,
                )
            });

            if let Some(taker) = taker {
                let taker_id = taker.id;
                let taker_team = taker.team_id;
                self.position.x = corner_x;
                self.position.y = corner_y;
                self.position.z = 0.0;
                self.velocity = Vector3::zeros();

                self.current_owner = Some(taker_id);
                self.previous_owner = None;
                self.ownership_duration = 0;
                self.claim_cooldown = 30;
                self.flags.in_flight_state = 30;
                self.pass_target_player_id = None;
                self.recent_passers.clear();
                // Same as goal-kick restart: clear stale shot target so
                // the eventual clearance/distribution doesn't false-credit
                // a phantom save (see the goal-kick branch below for the
                // full bug explanation).
                self.cached_shot_target = None;
                self.pass_origin_restart = PassOriginRestart::Corner;
                self.restart_taker_lock = None; // new restart voids the old same-touch chain (§9.4.1)
                self.restart_pending_taker = Some(taker_id);
                self.corner_short_option = None; // §12.4 — set below iff staged
                // Pick the corner routine via the SetPieceHistory-aware
                // helper so repeated identical routines (with no chance
                // produced) get blocked, varying the delivery flavour
                // across the match. The choice is stamped on the ball
                // so the aerial-contest resolver / xG accounting can
                // bias toward the targeted area.
                let scores = CornerScores {
                    near_post: 0.42,
                    penalty_spot: 0.48,
                    far_post: 0.46,
                    short: 0.20,
                    edge_cutback: 0.22,
                };
                let is_home_attacking = taker_team == context.field_home_team_id;
                let chosen_routine = pick_corner_routine(
                    &scores,
                    &context.set_piece_history,
                    is_home_attacking,
                );
                self.pending_corner_routine = Some(chosen_routine);
                #[cfg(feature = "match-logs")]
                {
                    use std::sync::atomic::Ordering;
                    crate::mid_run_diag::CORNERS_AWARDED.fetch_add(1, Ordering::Relaxed);
                }
                self.offside_snapshot = None;
                self.record_touch(taker_id, taker_team, self.current_tick_cached, true);

                #[cfg(feature = "match-logs")]
                log::info!("CLAIMSITE fn=goal.rs:corner id={}", taker_id);
                #[cfg(feature = "match-logs")]
                if let Some(taker_player) = players.iter().find(|p| p.id == taker_id) {
                    log::info!(
                        "SETPIECEDIAG_CORNER_TAKER id={} role={:?} score={:.2}",
                        taker_id,
                        taker_player.tactical_position.current_position.position_group(),
                        taker_score(taker_player)
                    );
                }
                events.add_ball_event(BallEvent::Claimed(taker_id));
                // Teleport the taker onto the ball so `move_to`'s
                // distance check doesn't immediately null ownership
                // on the next tick. The ball struct only has a &[MatchPlayer]
                // here — record the teleport and let the engine apply
                // it when it has &mut field.players.
                self.pending_set_piece_teleport = Some((taker_id, self.position));

                // Dead-ball set-up: send the two best-heading centre-backs
                // up into the box to attack the delivery. In real football
                // the big men walk up during the corner stoppage; the sim
                // has no stoppage, and a CB can't cover the length of the
                // pitch inside the cross window, so position them directly.
                // AttackingCorner keeps them there until the corner
                // resolves, then they sprint back into shape.
                let box_x = match side {
                    GoalSide::Home => 26.0,
                    GoalSide::Away => field_width - 26.0,
                };
                let center_y = field_height / 2.0;
                let mut cbs: Vec<(u32, f32)> = players
                    .iter()
                    .filter(|p| {
                        p.side == Some(attacking_side)
                            && p.id != taker_id
                            && p.tactical_position.current_position.is_central_defender()
                    })
                    .map(|p| (p.id, p.skills.technical.heading))
                    .collect();
                cbs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                // Arm the discrete aerial contest for this corner: it fires
                // once, the instant the cross is struck (see engine.rs
                // resolve_corner_contest).
                self.corner_contest_resolved = false;
                self.pending_corner_teleports.clear();
                for (i, (cb_id, _)) in cbs.iter().take(2).enumerate() {
                    // Near / far post split — wide enough that the far CB
                    // sits beyond the keeper's central cross-claim zone.
                    let y = if i == 0 {
                        center_y - field_height * 0.085
                    } else {
                        center_y + field_height * 0.085
                    };
                    self.pending_corner_teleports
                        .push((*cb_id, Vector3::new(box_x, y, 0.0)));
                }

                // Second attacking wave (wishlist #20 issue 3): two
                // midfielders hold the edge of the box rather than everyone
                // crashing the six-yard area — they pick up knock-downs and
                // second balls and give counter-cover if it's cleared.
                // State-preserving so they resume normal states from the edge.
                {
                    let box_edge = field_width * (165.0 / 840.0);
                    let edge_x = match side {
                        GoalSide::Home => box_edge,
                        GoalSide::Away => field_width - box_edge,
                    };
                    let mut second_wave: Vec<u32> = players
                        .iter()
                        .filter(|p| {
                            p.side == Some(attacking_side)
                                && p.id != taker_id
                                && p.tactical_position.current_position.is_midfielder()
                        })
                        .map(|p| p.id)
                        .collect();
                    second_wave.truncate(2);
                    for (i, mid_id) in second_wave.iter().enumerate() {
                        let ly = if i == 0 {
                            center_y - field_height * 0.08
                        } else {
                            center_y + field_height * 0.08
                        };
                        self.pending_restart_teleports
                            .push((*mid_id, Vector3::new(edge_x, ly, 0.0)));
                    }
                }

                // §9.4.2 short-corner option: sometimes a teammate walks
                // over to the near zone before the kick (real teams stage
                // the short option during the corner stoppage; the sim has
                // no stoppage, so position him directly). The delivery
                // picker only plays short when this zone is ACTUALLY
                // occupied at kick time — with nobody staged, the short
                // branch simply stays unavailable.
                if context.rng.bernoulli(0.25) {
                    let toward_centre_y: f32 =
                        if corner_y < center_y { 1.0 } else { -1.0 };
                    let inward_x: f32 = match side {
                        GoalSide::Home => 1.0,
                        GoalSide::Away => -1.0,
                    };
                    let short_spot = Vector3::new(
                        corner_x + inward_x * 18.0,
                        corner_y + toward_centre_y * 42.0,
                        0.0,
                    );
                    let cb_ids: Vec<u32> =
                        cbs.iter().take(2).map(|(id, _)| *id).collect();
                    if let Some(opt) = players
                        .iter()
                        .filter(|p| {
                            p.side == Some(attacking_side)
                                && p.id != taker_id
                                && !p.tactical_position.current_position.is_goalkeeper()
                                && !cb_ids.contains(&p.id)
                        })
                        .min_by(|a, b| {
                            let da = (a.position - self.position).norm_squared();
                            let db = (b.position - self.position).norm_squared();
                            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                        })
                    {
                        self.pending_restart_teleports.push((opt.id, short_spot));
                        // §12.4 — the processor holds him at this spot so
                        // he's genuinely stationary in the zone at kick
                        // time (previously his state machine ran him back
                        // toward his anchor and 0 short corners fired).
                        self.corner_short_option = Some(opt.id);
                    }
                }

                // Defensive corner shape (wishlist #20 issue 4): the
                // defending team previously ignored corners — measured
                // 2/4 of the back line stranded 300-630u upfield while
                // the ball was delivered into their box. Pull the six
                // nearest defending outfielders into a compact goal-side
                // block: two on the posts, two covering the spot, two on
                // the edge of the area for second balls. State-preserving
                // teleports (pending_restart_teleports) so they resume
                // normal marking/clearing states from a legal starting
                // shape rather than being frozen in a set-piece state.
                let def_goal_x = match side {
                    GoalSide::Home => 0.0,
                    GoalSide::Away => field_width,
                };
                let into = match side {
                    GoalSide::Home => 1.0_f32,
                    GoalSide::Away => -1.0_f32,
                };
                // (depth from goal line, lateral offset from centre).
                let block_slots: [(f32, f32); 6] = [
                    (10.0, -field_height * 0.10), // near post
                    (10.0, field_height * 0.10),  // far post
                    (30.0, -field_height * 0.05), // spot cover L
                    (30.0, field_height * 0.05),  // spot cover R
                    (58.0, -field_height * 0.09), // edge-of-box L (second ball)
                    (58.0, field_height * 0.09),  // edge-of-box R
                ];
                // Selecting by ROLE, not distance, is the whole point: the
                // stranded players are the FULLBACKS pushed highest up, i.e.
                // the FARTHEST defending outfielders — a nearest-N scan
                // excluded exactly the ones that needed pulling back. Put
                // the back line (regardless of how far it strayed) on the
                // four goal-block slots, then the two nearest midfielders on
                // the edge-of-box slots.
                let dist_to_goal = |p: &MatchPlayer| {
                    let dx = p.position.x - def_goal_x;
                    let dy = p.position.y - center_y;
                    (dx * dx + dy * dy).sqrt()
                };
                let mut block_ids: Vec<u32> = Vec::with_capacity(6);
                let mut back_line: Vec<(u32, f32)> = players
                    .iter()
                    .filter(|p| {
                        p.side == Some(defending_side)
                            && p.tactical_position.current_position.is_defender()
                    })
                    .map(|p| (p.id, dist_to_goal(p)))
                    .collect();
                back_line.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                let mut mids: Vec<(u32, f32)> = players
                    .iter()
                    .filter(|p| {
                        p.side == Some(defending_side)
                            && p.tactical_position.current_position.is_midfielder()
                    })
                    .map(|p| (p.id, dist_to_goal(p)))
                    .collect();
                mids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                // Back line fills the goal-block slots (up to 4); midfielders
                // top up any remaining goal slots plus the two edge slots.
                block_ids.extend(back_line.iter().take(4).map(|&(id, _)| id));
                block_ids.extend(
                    mids.iter()
                        .take(6 - block_ids.len())
                        .map(|&(id, _)| id),
                );
                for (slot, def_id) in block_ids.iter().enumerate() {
                    let (depth, lateral) = block_slots[slot];
                    self.pending_restart_teleports.push((
                        *def_id,
                        Vector3::new(def_goal_x + into * depth, center_y + lateral, 0.0),
                    ));
                }

                // "GK up for late corners" (wishlist #16): teleport the
                // flagged attacking keeper up with the CBs — he can't
                // run the pitch length inside the cross window either.
                // Uses the state-preserving restart-teleport list, NOT
                // pending_corner_teleports (which forces the defender
                // AttackingCorner state and would strand a GK in the
                // outfield state machine). The state processor's
                // override holds him there and sprints him home when
                // the corner resolves.
                let minute_now = (context.total_match_time * 90
                    / crate::r#match::engine::engine::MATCH_TIME_MS)
                    as u32;
                if let Some(gk) = players.iter().find(|p| {
                    p.side == Some(attacking_side)
                        && p.tactical_position.current_position.is_goalkeeper()
                }) {
                    if gk.gk_up_after_minute.is_some_and(|m| minute_now >= m) {
                        let gk_x = match side {
                            GoalSide::Home => 88.0,
                            GoalSide::Away => field_width - 88.0,
                        };
                        self.pending_restart_teleports
                            .push((gk.id, Vector3::new(gk_x, center_y + 34.0, 0.0)));
                    }
                }

                // §13.4: real retreat + a real wait-or-exploit window —
                // see docs/tactical-system-phase13-13.4-design.md.
                context.dead_ball_min_ms = context.total_match_time + 400;
                context.dead_ball_until_ms =
                    context.total_match_time + context.rng.range_u64(1, 3) * 1000;
                context.dead_ball_retreat_active = true;
                return;
            }
            // Structurally unreachable now (the fallback above always
            // finds a taker if ANY player of the attacking side exists on
            // the pitch), but keep the fallthrough rather than trust that
            // completely — an empty attacking side (impossible under this
            // engine's fixed 11-a-side squads) would fall through to the
            // goal-kick branch below, which itself falls back to the
            // nearest defending player if even the goalkeeper is missing.
        }

        // Goal kick: give ball to defending goalkeeper. Genuine last
        // resort (should be unreachable — every squad always has exactly
        // one goalkeeper, never removed from the roster even if sent
        // off): if no goalkeeper is found, the nearest defending outfield
        // player takes it instead of the restart silently vanishing.
        let field_height = context.field_size.height as f32;
        let center_y = field_height / 2.0;
        let gk_dist = |p: &MatchPlayer| {
            let dx = p.position.x - context.goal_positions.left.x;
            let dy = p.position.y - center_y;
            (dx * dx + dy * dy).sqrt()
        };
        let gk = players
            .iter()
            .find(|p| {
                p.side == Some(defending_side)
                    && p.tactical_position.current_position.is_goalkeeper()
            })
            .or_else(|| {
                players
                    .iter()
                    .filter(|p| p.side == Some(defending_side))
                    .min_by(|a, b| gk_dist(a).partial_cmp(&gk_dist(b)).unwrap_or(Ordering::Equal))
            });
        if let Some(gk) = gk {
            let gk_id = gk.id;
            let gk_team = gk.team_id;
            let goal_kick_x = match side {
                GoalSide::Home => 50.0,
                GoalSide::Away => field_width - 50.0,
            };

            self.position.x = goal_kick_x;
            self.position.y = context.goal_positions.left.y;
            self.position.z = 0.0;
            self.velocity = Vector3::zeros();

            self.current_owner = Some(gk_id);
            self.previous_owner = None;
            self.ownership_duration = 0;
            self.claim_cooldown = 30;
            self.flags.in_flight_state = 30;
            self.pass_target_player_id = None;
            self.recent_passers.clear();
            // Clear the shot target — the shot ended out of play and is
            // now resolved as a goal kick. Without this clear, the GK's
            // eventual ClearBall event hits gk_clearing_shot with a
            // stale `cached_shot_target=Some`, false-crediting a save for
            // a shot that never reached the keeper.
            self.cached_shot_target = None;
            self.pass_origin_restart = PassOriginRestart::GoalKick;
            self.restart_taker_lock = None; // new restart voids the old same-touch chain (§9.4.1)
            self.restart_pending_taker = Some(gk_id);
            self.offside_snapshot = None;
            self.record_touch(gk_id, gk_team, self.current_tick_cached, true);

            #[cfg(feature = "match-logs")]
            log::info!("CLAIMSITE fn=goal.rs:goal_kick id={}", gk_id);
            events.add_ball_event(BallEvent::Claimed(gk_id));
            // Put the GK onto the ball so the distance check in `move_to`
            // doesn't immediately null ownership because the GK was
            // ~35 units away at the goal line when the ball crossed.
            self.pending_set_piece_teleport = Some((gk_id, self.position));

            // §9.2.1 full-team recovery: every outfield player on both
            // teams drifts toward his formation anchor during the
            // goal-kick window; the non-kicking team's forwards take a
            // pressing line against the spread back four instead of
            // retreating. Queued FIRST so the specific placements below
            // (back-four spread, presser box-clear) override where they
            // apply — the drain applies entries in order, last wins.
            let recovery_ids: Vec<u32> = {
                let rec = crate::r#match::engine::set_pieces::recovery_shape_targets(
                    players,
                    defending_side,
                    self.position,
                    field_width,
                    true,
                    &[gk_id],
                );
                let ids = rec.iter().map(|(id, _)| *id).collect();
                self.pending_restart_teleports.extend(rec);
                ids
            };

            // Goal-kick shape (§8.3): previously nobody was positioned —
            // the kicking team crowded their own box and pressers stood
            // inside it. Spread the kicking team's back four wide across
            // their own third to offer real receiving options, and push
            // any pressing attacker OUT of the penalty area to its edge
            // (a legal, onside retreat). State-preserving teleports so
            // both sides resume normal play from a sane starting shape.
            {
                let gk_goal_x = match side {
                    GoalSide::Home => 0.0,
                    GoalSide::Away => field_width,
                };
                let into = match side {
                    GoalSide::Home => 1.0_f32,
                    GoalSide::Away => -1.0_f32,
                };
                let fh = context.field_size.height as f32;
                let cy = fh * 0.5;
                let box_depth = field_width * (165.0 / 840.0);
                // Kicking team's back four: wide + half-space receiving spots
                // just outside the box, so the GK has a short option each side.
                let recv_slots: [(f32, f32); 4] = [
                    (box_depth + 15.0, -fh * 0.34), // wide left
                    (box_depth + 15.0, fh * 0.34),  // wide right
                    (box_depth + 55.0, -fh * 0.14), // half-space left
                    (box_depth + 55.0, fh * 0.14),  // half-space right
                ];
                let mut kicking_backs: Vec<u32> = players
                    .iter()
                    .filter(|p| {
                        p.side == Some(defending_side)
                            && p.tactical_position.current_position.is_defender()
                    })
                    .map(|p| p.id)
                    .collect();
                kicking_backs.truncate(4);
                for (i, id) in kicking_backs.iter().enumerate() {
                    let (depth, lateral) = recv_slots[i];
                    self.pending_restart_teleports
                        .push((*id, Vector3::new(gk_goal_x + into * depth, cy + lateral, 0.0)));
                }
                // Pressing team: anyone inside the penalty area retreats to
                // its edge (goal-side of the ball), keeping their lateral spot.
                // §9.2.1: skip players the recovery shape already moves —
                // their queued anchor/press-line target is outside the box,
                // and this later entry would override it back to the edge
                // (positions checked here are the PRE-recovery snapshot).
                for p in players.iter() {
                    if p.side != Some(attacking_side)
                        || p.tactical_position.current_position.is_goalkeeper()
                        || recovery_ids.contains(&p.id)
                    {
                        continue;
                    }
                    let depth_from_goal = (p.position.x - gk_goal_x) * into;
                    let in_box = depth_from_goal >= 0.0
                        && depth_from_goal < box_depth + 6.0
                        && (p.position.y - cy).abs() < fh * 0.37;
                    if in_box {
                        let y = p.position.y.clamp(cy - fh * 0.45, cy + fh * 0.45);
                        self.pending_restart_teleports.push((
                            p.id,
                            Vector3::new(gk_goal_x + into * (box_depth + 12.0), y, 0.0),
                        ));
                    }
                }
            }

            // §13.4: real retreat + a real wait-or-exploit window — see
            // docs/tactical-system-phase13-13.4-design.md.
            context.dead_ball_min_ms = context.total_match_time + 400;
            context.dead_ball_until_ms =
                context.total_match_time + context.rng.range_u64(1, 3) * 1000;
            context.dead_ball_retreat_active = true;
        }
    }
}
