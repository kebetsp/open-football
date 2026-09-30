//! Ball-vs-defender interactions during in-flight passes and shots:
//! interception, shot-block, and goalkeeper save. Each runs only on
//! unowned balls with `in_flight_state > 0` so routine possession
//! play isn't disturbed.

use super::Ball;
use crate::PlayerFieldPositionGroup;
use crate::r#match::ball::events::BallEvent;
use crate::r#match::engine::goal::GOAL_WIDTH;
#[cfg(feature = "match-logs")]
use crate::r#match::engine::player::events::players::save_accounting_stats;
use crate::r#match::events::EventCollection;
use crate::r#match::player::strategies::players::ops::effective_skill::{
    ActionContext as EffSkillCtx, effective_skill,
};
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::{MatchContext, MatchPlayer, PassOriginRestart, PlayerSide};
use nalgebra::Vector3;
#[cfg(feature = "match-logs")]
use std::sync::atomic::Ordering;

/// TEMPORARY realism-bug diagnostic (2026-08-26, corner/deflection
/// investigation) — measures whether the block corridor's geometric
/// gate (BLOCK_LOOKAHEAD/BLOCK_CORRIDOR) or the post-gate RNG chance is
/// the binding constraint on the severely-low blocked-shot rate. Removed
/// once the investigation concludes.
#[cfg(feature = "match-logs")]
pub mod block_diag {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static SHOT_LIVE_TICKS: AtomicU64 = AtomicU64::new(0);
    pub static CANDIDATE_TICKS: AtomicU64 = AtomicU64::new(0);
    pub static NEAR_MISS_TICKS: AtomicU64 = AtomicU64::new(0);
    pub static ROLL_PASSED: AtomicU64 = AtomicU64::new(0);
    pub static SUM_MIN_PERP_X100: AtomicU64 = AtomicU64::new(0);
    pub static MIN_PERP_SAMPLES: AtomicU64 = AtomicU64::new(0);
    /// Times `try_intercept` claimed the ball for the DEFENDING side
    /// while a corner-branch deflection's own guard window was still
    /// active — i.e., the block's own teammate re-swallowed a deflection
    /// that was meant to reach the byline for a corner.
    pub static CORNER_DEFLECTION_RECLAIMED: AtomicU64 = AtomicU64::new(0);
    pub fn reset() {
        for c in [
            &SHOT_LIVE_TICKS,
            &CANDIDATE_TICKS,
            &NEAR_MISS_TICKS,
            &ROLL_PASSED,
            &SUM_MIN_PERP_X100,
            &MIN_PERP_SAMPLES,
            &CORNER_DEFLECTION_RECLAIMED,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
}

impl Ball {
    /// Opposing players near the ball's flight path can intercept passes.
    /// Interception chance depends on tackling, anticipation, positioning skills
    /// and proximity to the ball's trajectory.
    pub fn try_intercept(
        &mut self,
        context: &MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
    ) {
        // `context` is held even when this site does not currently
        // draw from `context.rng` so future calibration / env-modifier
        // wiring (slide tackle range, sliding_tackle_success) lands
        // without changing the signature again.
        let _ = context;
        // Only intercept unowned balls that are in flight (active pass)
        if self.current_owner.is_some() || self.flags.in_flight_state == 0 {
            return;
        }

        // §13.3: a struck direct free-kick shot has already had its one
        // legitimate contest — the wall (`wall_blocks_direct_fk`, rolled
        // before the shot is even struck). Keyed on `direct_fk_shot_in_flight`
        // rather than `pass_origin_restart`, which `resolve_free_kick`
        // resets to OpenPlay on the same tick the shot is dispatched (to
        // stop a rebound from re-triggering a phantom wall check) — that
        // happens before this interaction check ever runs on the
        // now-flying ball, so `pass_origin_restart` alone can't gate it.
        // Once live, a covering outfield defender elsewhere on the pitch
        // has no realistic chance to step into an already-in-flight
        // dead-ball strike the way they might read an open-play pass.
        if self.direct_fk_shot_in_flight {
            return;
        }

        // realism-bug (2026-08-26): a block's own corner-branch
        // deflection is still travelling toward the byline — don't let
        // `try_intercept` treat it as an interceptable pass from the
        // shooter's team (see the corner branch's own comment for the
        // full mechanism this closes).
        if self.current_tick_cached <= self.corner_deflection_guard_until_tick {
            return;
        }

        // Don't intercept aerial balls above player reach
        if self.position.z > 2.5 {
            return;
        }

        // Need to know who passed to determine the opposing team
        let passer_team = match self.previous_owner {
            Some(prev_id) => players.iter().find(|p| p.id == prev_id).map(|p| p.team_id),
            None => return,
        };
        let passer_team = match passer_team {
            Some(t) => t,
            None => return,
        };

        // Ball velocity determines the interception corridor width
        let ball_speed_sq = self.velocity.x * self.velocity.x + self.velocity.y * self.velocity.y;
        if ball_speed_sq < 1.0 {
            return; // Ball too slow, normal claiming handles it
        }

        // Interception reach in game units. Field is 840u = 105m, so 1u =
        // 0.125m. Old 2.5u left average defenders mathematically
        // unable to intercept (max score 0.039 vs 0.04 threshold). 5u
        // produced ~0.1 interceptions/team/match — defenders within
        // the radius hit ~0.025 chance, below the 0.035 threshold for
        // anyone but the closest, fastest, best-positioned. 6.5u
        // (~0.8m — a stretch-extension radius for the planted leg) and
        // a slightly higher base coefficient produces ~10
        // interceptions/team/match (real-football band) without the
        // intercept→snap→re-pass loops the previous 8u radius caused.
        const INTERCEPT_RADIUS: f32 = 5.5;
        const INTERCEPT_RADIUS_SQ: f32 = INTERCEPT_RADIUS * INTERCEPT_RADIUS;

        let mut best_interceptor: Option<u32> = None;
        let mut best_chance: f32 = 0.0;

        for player in players {
            // Only opposing team players can intercept
            if player.team_id == passer_team {
                continue;
            }

            // Don't let the pass target's team intercept their own pass target
            if Some(player.id) == self.pass_target_player_id {
                continue;
            }

            // Distance from player to ball
            let dx = player.position.x - self.position.x;
            let dy = player.position.y - self.position.y;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq > INTERCEPT_RADIUS_SQ {
                continue;
            }

            // Base chance: dedicated `interception` composite — anticipation,
            // positioning, concentration, marking, etc. routed through
            // `effective_skill` so fatigue applies. Drop-in replacement for
            // the legacy 4-skill average; magnitude lands in the same band
            // (0..1). Minute derived from the cached tick (10ms ticks).
            let minute = sc::minute_from_ticks(self.current_tick_cached);
            let skill_factor = sc::interception(player, minute);

            // Proximity factor: closer = higher chance (1.0 at 0m, 0.3 at max radius)
            let dist = dist_sq.sqrt();
            let proximity_factor = 1.0 - (dist / INTERCEPT_RADIUS) * 0.7;

            // Fast passes are harder to intercept — penalty coefficient
            // moderated from 0.10 (which made 7 u/tick passes 41% harder
            // than slow ones) back toward a lighter slope.
            let speed_penalty = 1.0 / (1.0 + ball_speed_sq.sqrt() * 0.06);

            // Per-tick interception chance. The 0.13 coefficient with
            // the 0.035 threshold mathematically excluded average
            // defenders (skill 0.5 × proximity 0.65 × speed 0.6 ≈
            // 0.025 per the old radius), so observed interceptions
            // were ~0.1/team/match vs real ~10/team. 0.16 (with the
            // bumped 5.5u radius and lowered 0.030 threshold) brings
            // an average-positioned defender to ~0.038 (above
            // threshold), and an elite defender at point-blank to
            // ~0.07, while still leaving peripheral or off-the-pace
            // defenders below the bar. Population per-team
            // interceptions land near 12–13/match.
            let chance = skill_factor * proximity_factor * speed_penalty * 0.16;

            if chance > best_chance {
                best_chance = chance;
                best_interceptor = Some(player.id);
            }
        }

        // Deterministic threshold. Avg defender (skill 0.5) at 60% of
        // reach with a typical pass score ~0.040 — just above the bar —
        // so most in-path defenders qualify, but peripheral ones don't.
        //
        // Stat-line note: population interceptions/team/match measures
        // ~120 vs the ~10 real-football target. ~3× of that comes from
        // the flight-protection extension (40→120 ticks) tripling the
        // per-pass intercept window; the rest from pass-volume inflation
        // (~1000 attempts/team vs real ~500). Raising the threshold to
        // suppress this inflated goals/match dramatically (it acts as a
        // population-wide pass-success governor), so the cosmetic stat
        // inflation is accepted in favour of in-band goals + draws.
        if best_chance > 0.030 {
            if let Some(interceptor_id) = best_interceptor {
                // Snap the ball to the interceptor and zero the
                // velocity. Before this, velocity was just scaled to
                // Zeroing velocity + handing ownership to the defender
                // prevents the old "own-goal after intercept" bug without
                // needing to teleport the ball. `move_to` will track the
                // ball toward its new owner at 1.5 u/tick over the next
                // 2-3 ticks, so visually the ball decelerates into the
                // defender's feet instead of jumping instantly from its
                // flight path onto the defender — which was visible to
                // the user as "ball appearing on another player without
                // moving".
                //
                // OG risk is fully handled by `self.velocity = zeros()`:
                // a stationary ball can't roll past the 15u owner-drop
                // threshold, so it can't cross the goal line unowned.
                let _ = interceptor_id; // no teleport, keep position as-is
                #[cfg(feature = "match-logs")]
                {
                    if self.current_tick_cached <= self.corner_deflection_guard_until_tick {
                        block_diag::CORNER_DEFLECTION_RECLAIMED.fetch_add(1, Ordering::Relaxed);
                    }
                }
                self.current_owner = Some(interceptor_id);
                self.pass_target_player_id = None;
                self.flags.in_flight_state = 0;
                self.claim_cooldown = 15;
                self.velocity = Vector3::zeros();
                self.position.z = 0.0;
                // Interception ends any in-flight shot — a defender taking
                // control downfield extinguishes the shot. Without this,
                // the next time the keeper grabs a moving ball from an
                // opponent (a long pass that loops to them), the stale
                // shot flag credits a phantom save and inflates the
                // saves/on-target ratio above 100%.
                self.cached_shot_target = None;
                let interceptor_team = players
                    .iter()
                    .find(|p| p.id == interceptor_id)
                    .map(|p| p.team_id)
                    .unwrap_or(0);
                let tick = self.current_tick_cached;
                self.record_touch(interceptor_id, interceptor_team, tick, true);
                self.offside_snapshot = None;
                self.pass_origin_restart = PassOriginRestart::OpenPlay;
                events.add_ball_event(BallEvent::Intercepted(interceptor_id, self.previous_owner));
            }
        }
    }

    /// Shot-block check. Runs only when the ball is a shot in flight
    /// (has a cached goal-line target). A defender whose body is in
    /// the shot's corridor between the current ball position and the
    /// goal line has a skill-weighted chance to block — the ball
    /// deflects to a loose state rather than reaching the keeper.
    /// Real football blocks ~6-10% of shots; we aim for that band.
    ///
    /// Distinct from `try_intercept`:
    /// - Intercept: ≤ 2.5u radius, pass-targeted; tiny per-tick chance
    /// - Block:     ≤ 4u radius, shot-targeted; higher per-event chance
    /// Both are scoped to unowned balls with `in_flight_state > 0`.
    pub fn try_block_shot(
        &mut self,
        context: &MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
    ) {
        // Only live shots — no cache means no shot in flight, no block.
        if self.cached_shot_target.is_none() {
            return;
        }
        if self.current_owner.is_some() || self.flags.in_flight_state == 0 {
            return;
        }

        // §13.3: same reasoning as `try_intercept` above — a direct
        // free-kick shot already had its one legitimate contest (the
        // wall, resolved before the shot was struck). No outfield
        // defender outside the wall gets a second, generic shot-block
        // chance against an already-in-flight dead-ball strike. Keyed on
        // `direct_fk_shot_in_flight`, not `pass_origin_restart` — see the
        // comment on `try_intercept`'s identical guard for why.
        if self.direct_fk_shot_in_flight {
            return;
        }
        // Ball above defender reach — aerial shots aren't blocked at
        // chest height, only grounders and waist-high strikes.
        if self.position.z > 2.0 {
            return;
        }

        let shooter_team = match self.previous_owner {
            Some(prev_id) => players.iter().find(|p| p.id == prev_id).map(|p| p.team_id),
            None => return,
        };
        let shooter_team = match shooter_team {
            Some(t) => t,
            None => return,
        };

        // Defender must be in the shot's path: between the ball and
        // the goal line, in the corridor defined by the shot direction.
        let ball_velocity_2d =
            (self.velocity.x * self.velocity.x + self.velocity.y * self.velocity.y).sqrt();
        if ball_velocity_2d < 0.5 {
            return; // Ball has stopped / nearly — not a live shot.
        }
        let shot_dir_x = self.velocity.x / ball_velocity_2d;
        let shot_dir_y = self.velocity.y / ball_velocity_2d;

        // Block window. Widened from 30u lookahead + 4u corridor so
        // defenders near the shot line have a real chance to get a
        // leg/body in — previously many "close-but-not-perfect"
        // positions fell just outside the corridor and the shot flew
        // through unopposed. Real football blocks ~18-22% of shots
        // (2-3 blocks per team per match from ~13 shots); we were
        // below that with the tight window.
        //
        // 2026-08-30 (shot/cross-block investigation, Pavel's own
        // framing): "defenders who are in the ball line can easier
        // block a shot, but those further from the line still attempt
        // to block — they just have more chance to miss it, or end up
        // with a deflection/corner rather than a perfect block." The
        // corridor was previously a hard binary cliff (in = a real
        // attempt with a graduated `perp_factor` chance; out = no
        // attempt at all, regardless of how close). Widened 7u→14u
        // (~0.9m→1.75m at this project's 8u/m scale — roughly a
        // standing body width vs. a genuine stretched-leg reach, the
        // real physiological ceiling for this) so a defender who's
        // close but not perfectly on the line still gets a real,
        // lower-odds chance instead of none. The graduation itself
        // (worse odds AND a worse outcome mix the further out the
        // contact was) is `perp_factor`/`contact_quality` below — no
        // separate mechanism needed, this widening just lets that
        // existing continuous term actually apply to more real cases
        // instead of being gated out before it ever runs.
        const BLOCK_LOOKAHEAD: f32 = 40.0; // was 30u
        const BLOCK_CORRIDOR: f32 = 14.0; // was 7u, before that 4u

        #[cfg(feature = "match-logs")]
        block_diag::SHOT_LIVE_TICKS.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "match-logs")]
        let mut min_perp_in_lookahead: Option<f32> = None;

        let mut best_blocker: Option<u32> = None;
        let mut best_chance: f32 = 0.0;
        // Carried out of the loop alongside best_blocker/best_chance so
        // the outcome section below can discount clean-recovery
        // probability by how marginal the winning contact actually was
        // (2026-08-30, see the BLOCK_CORRIDOR comment above).
        let mut best_lat: f32 = 0.0;

        for player in players {
            // Only opposing outfielders block (GK save pipeline handles
            // shots that reach the line; a GK blocking a shot at 5u
            // out is already Catching/Diving).
            if player.team_id == shooter_team {
                continue;
            }
            if player.tactical_position.current_position.position_group()
                == PlayerFieldPositionGroup::Goalkeeper
            {
                continue;
            }

            // Project defender position onto the shot line.
            let dx = player.position.x - self.position.x;
            let dy = player.position.y - self.position.y;
            let projection = dx * shot_dir_x + dy * shot_dir_y;
            // Must be ahead of the ball along the shot line, within
            // the lookahead window. 1u minimum so a defender level
            // with the ball (who's already been passed) doesn't count.
            if projection < 1.0 || projection > BLOCK_LOOKAHEAD {
                continue;
            }
            // Perpendicular distance to the line.
            let perp =
                (dx - projection * shot_dir_x).powi(2) + (dy - projection * shot_dir_y).powi(2);
            let perp_dist = perp.sqrt();
            #[cfg(feature = "match-logs")]
            {
                // Only count as a genuine "near miss" candidate if the
                // opponent is also within a plausible real straight-line
                // closing distance of the ball (60u/7.5m) — otherwise a
                // defender on the far side of the pitch who happens to
                // have a nonzero projection along the shot direction
                // pollutes the average with meaningless outliers.
                let straight_dist = (dx * dx + dy * dy).sqrt();
                if straight_dist <= 60.0 {
                    min_perp_in_lookahead = Some(match min_perp_in_lookahead {
                        Some(m) if m <= perp_dist => m,
                        _ => perp_dist,
                    });
                }
            }
            if perp_dist > BLOCK_CORRIDOR {
                continue;
            }

            // Skill mix: bravery (willingness to step into shot),
            // positioning (read the angle), anticipation (read the
            // cue), jumping/agility (get the body in the way), plus
            // tackling (stretching / last-ditch leg out). Weighted
            // toward mental attributes since shot-blocking is 70%
            // reading the shooter's body shape. Routed through
            // `effective_skill` so a tired defender blocks worse.
            let block_minute = sc::minute_from_ticks(self.current_tick_cached);
            let block_tech = EffSkillCtx::technical(block_minute);
            let block_mental = EffSkillCtx::mental(block_minute);
            let block_expl = EffSkillCtx::explosive(block_minute);
            let bravery = effective_skill(player, player.skills.mental.bravery, block_mental);
            let positioning =
                effective_skill(player, player.skills.mental.positioning, block_mental);
            let anticipation =
                effective_skill(player, player.skills.mental.anticipation, block_mental);
            let agility = effective_skill(player, player.skills.physical.agility, block_expl);
            let tackling = effective_skill(player, player.skills.technical.tackling, block_tech);
            let skill_factor = (bravery * 0.25
                + positioning * 0.25
                + anticipation * 0.25
                + agility * 0.15
                + tackling * 0.10)
                / 20.0;

            // Line factor — closer to the ball is better because the
            // defender's body is actually in the way. Farther along the
            // line means the shot has had time to rise / dip / move.
            let line_factor = 1.0 - (projection / BLOCK_LOOKAHEAD) * 0.4;
            // Perp factor — right on the line is best. Steeper fall-off
            // than before (0.5 from center → basically full chance;
            // 1.0 from edge → 60% chance) so wings-of-corridor still
            // produce blocks at meaningful rates.
            let perp_factor = 1.0 - (perp_dist / BLOCK_CORRIDOR) * 0.5;
            // Fast shots are harder to get in front of — but reaction
            // reflexes matter too. Elite defender reads the shape and
            // steps a tick earlier.
            let speed_penalty = 1.0 / (1.0 + ball_velocity_2d * 0.10);

            // Base multiplier 0.55 (was 0.35) — elite defenders
            // (skill_factor ≈ 0.85) at a good angle now block at
            // 30-40% chance, matching the real "closed-down striker
            // gets the ball blocked" rate.
            let chance = skill_factor * line_factor * perp_factor * speed_penalty * 0.55;

            if chance > best_chance {
                best_chance = chance;
                best_blocker = Some(player.id);
                // Signed offset of his centre from the shot line, along
                // (-dir.y, dir.x): decides which way the surface turns
                // the ball (see `tackle::block_contact`).
                best_lat = dx * (-shot_dir_y) + dy * shot_dir_x;
            }
        }

        #[cfg(feature = "match-logs")]
        {
            if best_blocker.is_some() {
                block_diag::CANDIDATE_TICKS.fetch_add(1, Ordering::Relaxed);
            } else if let Some(mp) = min_perp_in_lookahead {
                // A defender was ahead of the ball within BLOCK_LOOKAHEAD
                // but outside BLOCK_CORRIDOR — a genuine geometric near
                // miss, distinct from "nobody was even ahead of the ball."
                block_diag::NEAR_MISS_TICKS.fetch_add(1, Ordering::Relaxed);
                block_diag::SUM_MIN_PERP_X100.fetch_add((mp * 100.0) as u64, Ordering::Relaxed);
                block_diag::MIN_PERP_SAMPLES.fetch_add(1, Ordering::Relaxed);
            }
        }

        // RNG threshold instead of deterministic cutoff: a 30% block
        // chance still allows the shot through 70% of the time, which
        // is what we want — defenders block but don't always block.
        let blocker_id = match best_blocker {
            Some(id) if context.rng.unit_f32() < best_chance.clamp(0.03, 0.38) => id,
            _ => return,
        };
        #[cfg(feature = "match-logs")]
        block_diag::ROLL_PASSED.fetch_add(1, Ordering::Relaxed);

        // WHAT the touch does comes from contact geometry, not a table
        // (block frequency above is unchanged — that roll is the
        // skill/corridor model). The defender's body/leg is a blocking
        // surface of radius `BLOCK_CORRIDOR`; the ball meets it at the
        // offset `best_lat`. A square touch kills the ball and drops it
        // back at his feet; an edge touch barely turns it, so it keeps
        // going at goal, wide, or behind. A composed defender turns the
        // surface a little toward "away from my goal, toward the
        // touchline". Whether it ends up as a corner, a throw-in, an own
        // goal or a rebound is decided by where the ball then goes and by
        // last-touch — never by a dice table.
        let blocker = match players.iter().find(|p| p.id == blocker_id) {
            Some(p) => p,
            None => return,
        };
        let blocker_pos = blocker.position;
        let blocker_team = blocker.team_id;
        let blocker_side = blocker.side;
        let composure = (blocker.skills.mental.composure / 20.0).clamp(0.0, 1.0);
        let technique = (blocker.skills.technical.technique / 20.0).clamp(0.0, 1.0);
        let tick = self.current_tick_cached;

        let centred = (1.0 - best_lat.abs() / BLOCK_CORRIDOR).clamp(0.0, 1.0);
        let steer = centred * (0.15 + 0.35 * (0.5 * composure + 0.5 * technique));
        let restitution = 0.25 - 0.10 * composure;
        // Intent: away from his own goal, leaning to the nearer touchline.
        let away_x = match blocker_side {
            Some(PlayerSide::Left) => 1.0_f32,
            Some(PlayerSide::Right) => -1.0,
            None => {
                if blocker_pos.x < self.field_width * 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
        };
        let lean_y = if blocker_pos.y < self.field_height * 0.5 { -0.5 } else { 0.5 };
        let intent = Vector3::new(away_x, lean_y, 0.0);
        let jitter = (context.rng.unit_f32() - 0.5) * 0.3;
        let v_out = crate::r#match::player::strategies::players::ops::tackle::block_contact(
            self.velocity,
            best_lat,
            BLOCK_CORRIDOR,
            restitution,
            intent,
            steer,
            jitter,
            context.rng.unit_f32(),
        );

        self.position = blocker_pos;
        self.position.z = 0.0;
        self.offside_snapshot = None;
        self.pass_origin_restart = PassOriginRestart::OpenPlay;
        // Dedicated Blocked event so the block credit can't leak into a
        // separate Intercepted that happens to share the same tick.
        events.add_ball_event(BallEvent::Blocked(blocker_id, self.position));
        // A slow ball dropping in front of the box is a live second ball:
        // arm the rebound window (team shot-spacing exemption).
        if (v_out.x * v_out.x + v_out.y * v_out.y).sqrt() < 3.0 {
            self.last_rebound_tick = tick;
        }
        self.release_after_touch(blocker_id, blocker_team, blocker_side, v_out, tick);
    }

    /// Goalkeeper save check. Runs during shot flight: when the ball
    /// approaches the goal line and the defending keeper's body is
    /// within reach of the shot's trajectory, roll a skill-weighted
    /// save. The keeper state machine's `is_catch_successful` path
    /// timed saves to player-state ticks that didn't line up with the
    /// ball's physics step — saves fired too early or too late, and
    /// shots past the keeper cleared into the net. A physics-level
    /// save runs every ball tick with fresh ball position and commits
    /// the ball to the keeper at the moment of contact.
    pub fn try_save_shot(
        &mut self,
        context: &MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
    ) {
        let shot_target = match self.cached_shot_target {
            Some(t) => t,
            None => return,
        };
        if self.current_owner.is_some() || self.flags.in_flight_state == 0 {
            // realism-bug (2026-07-31) diagnostic, scratch-only: check
            // whether the ball gets CLAIMED (current_owner set, almost
            // certainly by the separate state-machine catch mechanic)
            // before this function's own reach window ever opens.
            // Remove once the investigation concludes.
            #[cfg(feature = "match-logs")]
            if shot_target.is_direct_fk {
                save_accounting_stats::FK_GATE_ALREADY_OWNED.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }
        // One roll per shot. This check runs every tick the ball is near
        // the goal line; unlatched it compounded with itself AND the GK
        // state machine's per-tick rolls (measured: disabling this layer
        // entirely moved goals/match 0.83 → 4.0, i.e. it was converting
        // ~80% of would-be goals into saves).
        if shot_target.physics_save_rolled {
            #[cfg(feature = "match-logs")]
            if shot_target.is_direct_fk {
                save_accounting_stats::FK_GATE_ALREADY_ROLLED.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }

        // realism-bug (2026-07-31): this used to check the ball's LIVE,
        // current-tick height (`self.position.z > 2.8`) — but that's a
        // snapshot of an in-flight arc, not where the shot will actually
        // cross the line. A real lofted shot is legitimately several
        // metres up mid-flight and only descends under the bar near the
        // very end; since this function only becomes horizontally
        // "in reach" for the last ~2-3 ticks of a long flight, the live
        // height and the in-reach window essentially never coincided for
        // a genuinely lofted shot — measured as the physics-layer save
        // NEVER once reaching its lateral_error check across 100+ direct
        // free kicks (gate counters: z_too_high + out_of_reach summed to
        // the shot's entire tracked flight, reached_lateral_check=0
        // every time). `goal_line_z` is the already-computed PROJECTED
        // height at arrival (the same value `is_catch_successful`
        // correctly uses for its own "over the bar" check) — using it
        // here instead judges save-ability by where the shot actually
        // ends up, not a mid-arc altitude reading. Shots deliberately
        // sent over the bar (`shot_goes_over_bar`) still correctly
        // project a large `goal_line_z` and skip the save as before.
        if shot_target.goal_line_z > 2.8 {
            #[cfg(feature = "match-logs")]
            if shot_target.is_direct_fk {
                save_accounting_stats::FK_GATE_Z_TOO_HIGH.fetch_add(1, Ordering::Relaxed);
                log::info!(
                    "FKZ tick={} live_z={:.2} proj_z={:.2} x={:.1} vx={:.2} vz={:.2}",
                    context.current_tick(),
                    self.position.z,
                    shot_target.goal_line_z,
                    self.position.x,
                    self.velocity.x,
                    self.velocity.z
                );
            }
            return;
        }

        // Only consider the shot once it's close to the goal line —
        // the save resolves at the moment of contact. Distance in
        // x-units the ball will cover in a single tick determines the
        // window: we check within ~2 ticks of arrival.
        let (goal_x, goal_y) = match shot_target.defending_side {
            PlayerSide::Left => (context.goal_positions.left.x, context.goal_positions.left.y),
            PlayerSide::Right => (
                context.goal_positions.right.x,
                context.goal_positions.right.y,
            ),
        };

        // Reject balls that have already crossed the goal line. Using
        // `.abs()` below meant a shot 2u behind the goal at goal_y+15
        // still satisfied "close to goal line" and "moving toward goal"
        // and got saved out of thin air — the visible bug: ball flies
        // past the goal, then teleports into the keeper's hands. Once
        // the ball is past the line (goal or goal kick, depending on Y),
        // the shot is over.
        let past_goal_line = match shot_target.defending_side {
            PlayerSide::Left => self.position.x < goal_x,
            PlayerSide::Right => self.position.x > goal_x,
        };
        if past_goal_line {
            #[cfg(feature = "match-logs")]
            {
                save_accounting_stats::SAVE_TICKS_PAST_GOAL_LINE.fetch_add(1, Ordering::Relaxed);
                if shot_target.is_direct_fk {
                    save_accounting_stats::FK_GATE_PAST_GOAL_LINE.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.cached_shot_target = None;
            return;
        }

        let dist_to_goal_x = (self.position.x - goal_x).abs();
        let ball_vx = self.velocity.x.abs().max(0.5);
        if dist_to_goal_x > ball_vx * 2.5 {
            #[cfg(feature = "match-logs")]
            {
                save_accounting_stats::SAVE_TICKS_OUT_OF_REACH.fetch_add(1, Ordering::Relaxed);
                if shot_target.is_direct_fk {
                    save_accounting_stats::FK_GATE_OUT_OF_REACH.fetch_add(1, Ordering::Relaxed);
                    log::info!(
                        "FKZ_OOR tick={} z={:.2} dist_x={:.1} vx={:.2}",
                        context.current_tick(),
                        self.position.z,
                        dist_to_goal_x,
                        ball_vx
                    );
                }
            }
            return;
        }
        #[cfg(feature = "match-logs")]
        save_accounting_stats::SAVE_TICKS_REACHED.fetch_add(1, Ordering::Relaxed);

        // Ball must still be traveling toward that goal line.
        let moving_toward_goal = match shot_target.defending_side {
            PlayerSide::Left => self.velocity.x < -0.2,
            PlayerSide::Right => self.velocity.x > 0.2,
        };
        if !moving_toward_goal {
            #[cfg(feature = "match-logs")]
            if shot_target.is_direct_fk {
                save_accounting_stats::FK_GATE_NOT_MOVING_TOWARD_GOAL
                    .fetch_add(1, Ordering::Relaxed);
            }
            return;
        }

        // Ball must be within goal width (else it's wide and the
        // post / out-of-play handler catches it).
        if (self.position.y - goal_y).abs() > GOAL_WIDTH + 1.0 {
            #[cfg(feature = "match-logs")]
            if shot_target.is_direct_fk {
                save_accounting_stats::FK_GATE_OUTSIDE_GOAL_WIDTH.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }

        // Find the defending keeper.
        let keeper = players.iter().find(|p| {
            p.side == Some(shot_target.defending_side)
                && p.tactical_position.current_position.position_group()
                    == PlayerFieldPositionGroup::Goalkeeper
                && !p.is_sent_off
        });
        let keeper = match keeper {
            Some(k) => k,
            None => {
                #[cfg(feature = "match-logs")]
                if shot_target.is_direct_fk {
                    save_accounting_stats::FK_GATE_NO_KEEPER.fetch_add(1, Ordering::Relaxed);
                }
                return;
            }
        };
        #[cfg(feature = "match-logs")]
        if shot_target.is_direct_fk {
            save_accounting_stats::FK_GATE_REACHED_LATERAL_CHECK.fetch_add(1, Ordering::Relaxed);
        }

        // Route through `effective_skill` so a tired keeper has worse
        // reach / handling / reflexes than a fresh one. Routing minute
        // is taken from `MatchContext::total_match_time`.
        let minute_for_effective = sc::minute_from_ms(context.total_match_time);
        let tech_ctx = EffSkillCtx::technical(minute_for_effective);
        let mental_ctx = EffSkillCtx::mental(minute_for_effective);
        let expl_ctx = EffSkillCtx::explosive(minute_for_effective);
        let handling = effective_skill(keeper, keeper.skills.goalkeeping.handling, tech_ctx);
        let reflexes = effective_skill(keeper, keeper.skills.goalkeeping.reflexes, tech_ctx);
        let agility = effective_skill(keeper, keeper.skills.physical.agility, expl_ctx);
        // Concentration acts on the catch / parry split — focused
        // keepers catch cleaner, distracted ones parry into danger.
        let concentration = effective_skill(keeper, keeper.skills.mental.concentration, mental_ctx);
        let scaled_handling = ((handling - 1.0) / 19.0).max(0.0);
        let scaled_reflexes = ((reflexes - 1.0) / 19.0).max(0.0);
        let scaled_agility = ((agility - 1.0) / 19.0).max(0.0);
        let scaled_concentration = ((concentration - 1.0) / 19.0).max(0.0);

        // Diving reach in game units. Field is 840u = 105m, so 1u = 0.126m
        // (half-goal 29u = 3.66m matches real 3.66m). Every keeper, even a
        // youth-level one, can physically dive across most of the goal
        // — skill determines whether they *catch* the ball, not whether
        // they can reach it. The previous 10u floor made corner shots
        // literally unreachable for weak keepers, so blowouts in youth
        // leagues (hnd=1, ref=1) pushed matches to 10+ goals. New reach:
        //   skills 1   → 20u (2.5m, standing dive — can touch the post)
        //   skills 10  → 26u (3.25m, covers most of the goal)
        //   skills 20  → 32u (4.0m, elite full-stretch — beyond the post)
        let reach = 20.0 + scaled_agility * 8.0 + scaled_reflexes * 4.0;
        let lateral_error = (keeper.position.y - shot_target.goal_line_y).abs();
        if lateral_error > reach {
            // realism-bug (2026-07-31) diagnostic, scratch-only: confirm
            // whether the physics-layer save ever gets a legitimate roll
            // for a direct FK, or whether it's structurally out of reach
            // every time (in which case the state-machine layer is the
            // only thing that could be stopping these shots). Remove
            // once the investigation concludes.
            #[cfg(feature = "match-logs")]
            if shot_target.is_direct_fk {
                log::info!(
                    "FKCONV_PHYS lat_err={:.2} reach={:.2} in_reach=false",
                    lateral_error,
                    reach
                );
            }
            return;
        }

        // Base save chance. Centered shot ~0.88; full-stretch ~0.30.
        // Skill handles the rest; this curve is purely geometry.
        let reach_ratio = (lateral_error / reach).clamp(0.0, 1.0);
        let base = 0.88 - reach_ratio * reach_ratio * 0.58;

        // Shot-speed penalty — elite shots beat keepers more often.
        let ball_speed = self.velocity.norm();
        let speed_excess = (ball_speed - 3.0).max(0.0);
        let speed_penalty = (speed_excess * 0.08 * (1.0 - scaled_reflexes * 0.5)).min(0.40);

        // Skill multiplier. Floor 0.72 so a 1.0-skill keeper still saves
        // ~60% of centred shots (real weak keepers save 55-65% overall).
        // Old `0.6 + skill*0.5` gave a 40% floor, pushing youth matches
        // with hnd=1.0 / ref=1.0 GKs into 10-goal blowouts. At 0.72 →
        // 1.07, skill matters (10-pt skill gap = ~30% save-rate gap)
        // but weak keepers can't single-handedly lose 13-4.
        //
        // The composite blend (`gk_shot_stopping`) feeds reflexes,
        // handling, agility, positioning, concentration, anticipation
        // and one_on_ones through `effective_skill` so a tired keeper
        // late in the match plays worse — drop-in replacement for the
        // legacy 3-skill blend, magnitude tuned to the same band.
        let skill = sc::gk_shot_stopping(keeper, minute_for_effective);
        // Per-tick save rate. This save check runs every tick the ball
        // is within reach of the goal line, AND the GK state-machine
        // (Catching, Diving) runs its OWN per-tick save roll. Both
        // compound across the 5-15 ticks of shot flight.
        //
        // Calibration target: per-shot conversion ~12-15% (real Opta
        // ~12% of shots are goals, ~33% on target with 30-35% of those
        // being saves). Earlier `0.45 + 0.35*skill` clamped at 0.55
        // produced top-scorer rates of 1.5+ goals/match — too generous.
        // New `0.55 + 0.40*skill` clamped at 0.68 lifts the per-tick
        // floor for any GK on the pitch (a Reflexes-5 keeper still
        // makes routine saves) and raises the ceiling so elite GKs
        // stop the centred power shots they're paid to stop. Skill
        // gap stays 10pt → ~30% save-rate gap.
        // Trimmed `0.55 + skill*0.40` → `0.50 + skill*0.30` after
        // dev_match audits showed population save% at 79% vs real ~67%
        // AND elite-keeper save% at extreme skill gaps inflated to ~91%
        // (real ~75-80%), driving the gap-9+ upset rate to 0% vs real
        // ~9%. The previous 0.36 coefficient on the skill term was the
        // dominant lever crushing weak-team conversion. Pulling it to
        // 0.30 narrows the strong-vs-weak save spread while leaving the
        // equal-skill baseline (skill_mult ≈ 0.65 at skill 0.5) in the
        // same band as before.
        let skill_mult = 0.52 + skill * 0.32;
        // Environment shifts keeper handling — heavy rain spills more,
        // wind on cross-claims has a subtler effect (the keeper still
        // sets feet under a regular shot).
        let env_mod = context.environment.modifiers();
        let env_handling_delta = env_mod.goalkeeper_handling;
        // Cap trimmed 0.55 → 0.45 after the single-roll latch landed:
        // batch mean sat at ~1.7 goals/match (band ~2-3) with the roll
        // now firing once per shot instead of compounding.
        let save_prob =
            ((base - speed_penalty) * skill_mult + env_handling_delta).clamp(0.05, 0.45);

        #[cfg(feature = "match-logs")]
        save_accounting_stats::SAVE_PHYSICS_FIRED.fetch_add(1, Ordering::Relaxed);

        // Latch before rolling — win or lose, the physics layer gets
        // exactly one attempt at this shot.
        if let Some(t) = self.cached_shot_target.as_mut() {
            t.physics_save_rolled = true;
        }
        let physics_roll = context.rng.unit_f32();
        // realism-bug (2026-07-31) diagnostic, scratch-only: the one
        // physics-layer roll a direct FK ever gets — lateral error,
        // save_prob, and whether it actually saved. Remove once the
        // investigation concludes.
        #[cfg(feature = "match-logs")]
        if shot_target.is_direct_fk {
            log::info!(
                "FKCONV_PHYS lat_err={:.2} reach={:.2} in_reach=true save_prob={:.3} roll={:.3} saved={}",
                lateral_error,
                reach,
                save_prob,
                physics_roll,
                physics_roll < save_prob
            );
        }
        if physics_roll >= save_prob {
            return; // Keeper beaten — shot goes on.
        }
        #[cfg(feature = "match-logs")]
        save_accounting_stats::SAVE_PHYSICS_PASSED.fetch_add(1, Ordering::Relaxed);

        // Save outcome distribution. Catch / safe parry / dangerous
        // parry / corner — the previous code always caught.
        //   catch_prob   = 0.12 + handling*0.26 + positioning*0.10
        //                  + concentration*0.06
        //                  - shot_power*0.18 - reach_stretch*0.18
        //   safe_parry   = 0.20 + reflexes*0.10 + handling*0.07 + agility*0.05
        //                  + concentration*0.04
        //   dangerous    = remainder
        // Concentration shifts the split toward catch/safe parry: a
        // focused keeper does NOT spill the ball back into danger.
        let positioning = (effective_skill(keeper, keeper.skills.mental.positioning, mental_ctx)
            / 20.0)
            .clamp(0.0, 1.0);
        let shot_power_norm = (ball_speed / 8.0).clamp(0.0, 1.0);
        let reach_stretch = reach_ratio;
        let catch_prob =
            (0.12 + scaled_handling * 0.26 + positioning * 0.10 + scaled_concentration * 0.06
                - shot_power_norm * 0.18
                - reach_stretch * 0.18)
                .clamp(0.04, 0.62);
        let safe_parry_prob = (0.20
            + scaled_reflexes * 0.10
            + scaled_handling * 0.07
            + scaled_agility * 0.05
            + scaled_concentration * 0.04)
            .clamp(0.12, 0.52);

        let keeper_id = keeper.id;
        let keeper_pos = keeper.position;
        let keeper_team = keeper.team_id;
        let keeper_side = keeper.side;

        let outcome_roll = context.rng.unit_f32();
        let p_catch = catch_prob;
        let p_safe = (catch_prob + safe_parry_prob).min(0.92);

        self.position.z = 0.0;
        self.previous_owner = self.current_owner.or(self.previous_owner);
        self.pass_target_player_id = None;
        // Stage the save credit before clearing the shot target. This
        // marker is consumed by the event-dispatch step so the GK earns
        // a save in the stats sheet and the shooter's on-target count
        // increments. Without this, the physics save changes ball state
        // (catch/parry) but bypasses the state-machine save events that
        // were the only path crediting saves — leaving ~90% of resolved
        // shots stat-less.
        if let Some(shooter_id) = self.previous_owner {
            self.pending_save_credit = Some((keeper_id, shooter_id));
        }
        self.cached_shot_target = None;
        let tick = self.current_tick_cached;
        self.offside_snapshot = None;
        self.pass_origin_restart = PassOriginRestart::OpenPlay;

        if outcome_roll < p_catch {
            // Clean catch — keeper holds.
            self.position = keeper_pos;
            self.position.z = 0.0;
            self.velocity = Vector3::zeros();
            self.current_owner = Some(keeper_id);
            self.flags.in_flight_state = 0;
            self.claim_cooldown = 200;
            self.record_touch(keeper_id, keeper_team, tick, true);
            #[cfg(feature = "match-logs")]
            log::info!("CLAIMSITE fn=try_save_shot:physics_catch id={}", keeper_id);
            events.add_ball_event(BallEvent::Claimed(keeper_id));
            return;
        }

        if outcome_roll < p_safe {
            #[cfg(feature = "match-logs")]
            crate::mid_run_diag::SAVE_PARRY_FIRED.fetch_add(1, Ordering::Relaxed);
            // Parried OUT for a corner. The outcome is already decided, so
            // resolve it POSITIONALLY — place the ball just past the byline,
            // wide of the post — rather than driving it there by velocity.
            // The velocity approach half-failed: the keeper sits on the goal
            // line, so the ball only reached the post (y±GOAL_WIDTH) by the
            // time it crossed x=0, landing borderline → ~half fell inside
            // for a goal kick. Placing it out (outside `goal_y ± GOAL_WIDTH`,
            // a few units past x=0) makes the endline resolver award the
            // corner reliably next tick (keeper = last toucher → corner for
            // the attackers; save already booked via `pending_save_credit`).
            let goal_y_for_side = match keeper_side {
                Some(PlayerSide::Left) => context.goal_positions.left.y,
                Some(PlayerSide::Right) => context.goal_positions.right.y,
                None => self.position.y,
            };
            let to_top = self.position.y < goal_y_for_side;
            self.position.x = match keeper_side {
                Some(PlayerSide::Left) => -3.0,
                Some(PlayerSide::Right) => self.field_width + 3.0,
                None => self.position.x,
            };
            self.position.y = if to_top {
                (goal_y_for_side - GOAL_WIDTH - 10.0).max(3.0)
            } else {
                (goal_y_for_side + GOAL_WIDTH + 10.0).min(self.field_height - 3.0)
            };
            self.position.z = 0.0;
            self.velocity = Vector3::zeros();
            self.current_owner = None;
            self.flags.in_flight_state = 0;
            self.claim_cooldown = 30;
            self.record_touch(keeper_id, keeper_team, tick, false);
            // NB: do NOT emit Intercepted here — its ClaimBall follow-up
            // forces ownership onto the keeper, which CANCELS the corner
            // (the ball must stay loose and cross out). The save is already
            // booked via `pending_save_credit`, and `record_touch` marks the
            // keeper as last toucher so the endline resolver awards the
            // corner to the attackers.
            return;
        }

        // Dangerous parry — ball spills off the keeper's hands. Arms the
        // rebound window so the attacking team's follow-up shot isn't
        // killed by the team shot-spacing gate.
        self.last_rebound_tick = tick;
        // Real goalkeepers under pressure push the ball toward the side
        // they're already diving, not back into the central goalmouth
        // where the attacking team gets a free tap-in. The previous
        // ±15u y-spread around the ball position landed ~50% of parries
        // in the six-yard tap-in lane.
        let drop_distance = 12.0 + context.rng.unit_f32() * 18.0;
        let drop_x = match keeper_side {
            Some(PlayerSide::Left) => keeper_pos.x + drop_distance,
            Some(PlayerSide::Right) => keeper_pos.x - drop_distance,
            None => keeper_pos.x,
        };
        // Outward y-bias: push the ball *away* from the goal centre. If
        // the ball was already lateral, push further laterally; for
        // central shots, pick a random side and push 14-30u outward.
        let goal_center_y = match keeper_side {
            Some(PlayerSide::Left) => context.goal_positions.left.y,
            Some(PlayerSide::Right) => context.goal_positions.right.y,
            None => self.field_height * 0.5,
        };
        let outward_sign = if (self.position.y - goal_center_y).abs() < 1.0 {
            if context.rng.unit_f32() < 0.5 {
                -1.0
            } else {
                1.0
            }
        } else {
            (self.position.y - goal_center_y).signum()
        };
        let outward_offset = (14.0 + context.rng.unit_f32() * 16.0) * outward_sign;
        let drop_y = self.position.y + outward_offset + (context.rng.unit_f32() - 0.5) * 10.0;
        let drop_y = drop_y.clamp(0.0, self.field_height);
        let drop_x = drop_x.clamp(0.0, self.field_width);
        let dx = drop_x - self.position.x;
        let dy = drop_y - self.position.y;
        let dist = (dx * dx + dy * dy).sqrt().max(1.0);
        // Spill speed: energy shed off the hands, NOT a clearance. The
        // previous constant 3.5 u/tick (43.75 m/s — harder than the
        // engine's hardest shot, capped at 3.2) carried the ball ~10m
        // through the box during the protected flight window, so every
        // "dangerous" parry physically exited the danger zone before
        // anyone could touch it. A real spill comes off the gloves at a
        // fraction of shot speed, worse for keepers with poor handling:
        // ~0.7-1.2 u/tick lands the ball in the 1.5-3.75m drop zone the
        // direction model already aims for, where the box contest can
        // actually happen.
        let parry_speed =
            (ball_speed * (0.22 + 0.18 * (1.0 - scaled_handling))).clamp(0.6, 1.3);
        self.velocity.x = (dx / dist) * parry_speed;
        self.velocity.y = (dy / dist) * parry_speed;
        self.velocity.z = 0.0;
        self.current_owner = None;
        // Flight window 30 → 10 ticks: the genuine time a spilled ball
        // is ungatherable. At 30 the entire rebound lived inside the
        // claims-locked window — and because `previous_owner` stayed the
        // SHOOTER, try_intercept treated the spill as an attacker pass,
        // making DEFENDERS the only players able to win it. Setting the
        // keeper as previous owner (he is physically the last player the
        // ball came off) flips the intercept population to its realistic
        // one — attackers pouncing on the spill — through the untouched
        // existing gate, and the keeper's own bounce-back reclaim still
        // lets him smother a ball that dies at his feet.
        self.previous_owner = Some(keeper_id);
        self.flags.in_flight_state = 10;
        self.claim_cooldown = 0;
        self.record_touch(keeper_id, keeper_team, tick, false);
        events.add_ball_event(BallEvent::Intercepted(keeper_id, self.previous_owner));
    }
}
