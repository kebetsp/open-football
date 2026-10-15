//! Ball-vs-defender interactions during in-flight passes and shots:
//! interception, shot-block, and goalkeeper save. Each runs only on
//! unowned balls with `in_flight_state > 0` so routine possession
//! play isn't disturbed.

use super::{Ball, PendingIntercept};
use crate::PlayerFieldPositionGroup;
use crate::r#match::ball::events::BallEvent;
#[cfg(feature = "match-logs")]
use crate::r#match::engine::player::events::players::save_accounting_stats;
use crate::r#match::events::EventCollection;
use crate::r#match::player::strategies::players::ops::effective_skill::{
    ActionContext as EffSkillCtx, effective_skill,
};
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::player::strategies::players::tackle::{ContactInput, resolve_contact};
use crate::r#match::{MatchContext, MatchPlayer, PassOriginRestart, PlayerSide};
use nalgebra::Vector3;
#[cfg(feature = "match-logs")]
use std::sync::atomic::Ordering;

/// Multiplier on the per-defender block chance now that each defender gets
/// ONE roll per shot (see `Ball::block_rolled`). The old per-tick re-rolling
/// stacked the chance to a 47-61% block rate (real 26%); one roll per
/// defender gives ~15%, which would collapse deflections (6 `Blocked`
/// events/match, close to real) and own goals, so the chance is scaled up
/// to ease the block rate in steps instead. Tunable on one binary with
/// `BLOCK_ONCE_SCALE` while calibrating; the baked default is the chosen
/// value.
fn block_once_scale() -> f32 {
    use std::sync::OnceLock;
    static SCALE: OnceLock<f32> = OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("BLOCK_ONCE_SCALE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(2.5)
    })
}

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

/// Leg reach of an interception swing (~2.25m: a stretched leg).
const INTERCEPT_REACH: f32 = 18.0;

const LEG_REACH_HEIGHT_M: f32 = 1.2;

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
            if !self.intercept_attempted.is_empty() {
                self.intercept_attempted.clear();
            }
            self.pending_intercept = None;
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

        // A leg swing cannot reach a ball above ~1.2m (knee-to-chest height
        // of a standing player stretching); higher balls are headers and
        // go through the aerial contest. Was 2.5m, which let an adjacent
        // marker "intercept" a cross 0.2s after the kick at 1.3m height
        // (22-36% of crosses; real blocked+intercepted ~5%).
        if self.position.z > LEG_REACH_HEIGHT_M {
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
        // Contested interception (2026-10-01, Pavel's redesign). Every
        // opponent near the ball's path gets ONE attempt per pass: each
        // swings a leg (same contact geometry as a tackle), and the FIRST
        // leg to meet the ball decides its fate. A well-timed, centred
        // touch kills the ball and he controls it; a late or edge touch
        // only deflects it (it goes where the contact sends it — a corner,
        // a throw-in, an own goal, a loose ball); a miss lets it go
        // through. A second defender arriving within a few ticks can
        // spoil the first one's control. Nobody is "the best" by
        // decree: a defender only stands down when a teammate is clearly
        // better placed.
        const TRIGGER_RADIUS: f32 = 10.0;
        const TRIGGER_RADIUS_SQ: f32 = TRIGGER_RADIUS * TRIGGER_RADIUS;
        // A candidate stands down if a teammate would reach the ball this
        // many ticks sooner.
        const STAND_DOWN_TICKS: f32 = 6.0;
        // A second contact within this many ticks can spoil the first.
        const SPOIL_WINDOW_TICKS: u32 = 4;
        const SPOIL_PROB: f32 = 0.35;
        // A touch that leaves the ball slower than this (shin, thigh or
        // foot — whatever part met it) has killed it: a controlled
        // interception (u/tick).
        const CONTROL_MAX_SPEED: f32 = 1.5;

        // Apply a contest resolved earlier once his leg reaches the ball.
        if let Some(p) = self.pending_intercept {
            if self.current_tick_cached >= p.apply_tick {
                self.pending_intercept = None;
                let tick = self.current_tick_cached;
                self.pass_target_player_id = None;
                self.offside_snapshot = None;
                self.pass_origin_restart = PassOriginRestart::OpenPlay;
                if p.controlled {
                    self.current_owner = Some(p.defender_id);
                    self.flags.in_flight_state = 0;
                    self.claim_cooldown = 15;
                    self.velocity = Vector3::zeros();
                    self.position.z = 0.0;
                    self.cached_shot_target = None;
                    self.record_touch(p.defender_id, p.defender_team, tick, true);
                    events.add_ball_event(BallEvent::Intercepted(p.defender_id, self.previous_owner));
                } else {
                    // A deflection is not an interception: no
                    // `Intercepted` event (its ClaimBall follow-up would
                    // hand him the ball he only touched).
                    self.release_after_touch(p.defender_id, p.defender_team, p.defender_side, p.velocity, tick);
                }
            }
            return;
        }

        let minute = sc::minute_from_ticks(self.current_tick_cached);
        let mut cands: Vec<usize> = Vec::new();
        for (i, player) in players.iter().enumerate() {
            if self.is_claim_barred(player.id) {
                continue;
            }
            if player.team_id == passer_team
                || Some(player.id) == self.pass_target_player_id
                || self.intercept_attempted.contains(&player.id)
            {
                continue;
            }
            let dx = player.position.x - self.position.x;
            let dy = player.position.y - self.position.y;
            if dx * dx + dy * dy <= TRIGGER_RADIUS_SQ {
                cands.push(i);
            }
        }
        if cands.is_empty() {
            return;
        }

        // Each candidate swings once; record the contact each would make.
        struct Swing {
            idx: usize,
            tick: u32,
            control: f32,
            velocity: Vector3<f32>,
        }
        let mut swings: Vec<Swing> = Vec::new();
        let ball_pos = Vector3::new(self.position.x, self.position.y, 0.0);
        let ball_vel = Vector3::new(self.velocity.x, self.velocity.y, 0.0);
        let rng = &context.rng;
        for &i in &cands {
            let p = &players[i];
            self.intercept_attempted.push(p.id);
            let skill = sc::interception(p, minute).clamp(0.0, 1.0);
            let speed = p.skills.max_speed_with_condition(p.player_attributes.condition);
            let leg_speed = 0.9 + speed;
            let reach = INTERCEPT_REACH;
            let def = Vector3::new(p.position.x, p.position.y, 0.0);
            // Aim where the ball will be when the tip gets there.
            let mut t = 5.0f32;
            for _ in 0..30 {
                let aim = ball_pos + ball_vel * t;
                t = ((aim - def).magnitude() / leg_speed).min(60.0);
            }
            let aim = ball_pos + ball_vel * t;
            let to = aim - def;
            let axis = if to.magnitude() > 1e-3 { to / to.magnitude() } else { Vector3::new(1.0, 0.0, 0.0) };
            // Reaction: sharper players start the leg sooner.
            let tri = rng.unit_f32() - rng.unit_f32();
            let windup = (0.3 + 2.5 * (1.0 - skill) + tri * 1.0).max(0.0);
            // He wants to kill the ball; a clearance goes away from his goal.
            let away_x = match p.side {
                Some(PlayerSide::Left) => 1.0,
                Some(PlayerSide::Right) => -1.0,
                None => if p.position.x < self.field_width * 0.5 { 1.0 } else { -1.0 },
            };
            let lean_y = if p.position.y < self.field_height * 0.5 { -0.5 } else { 0.5 };
            let intent = Vector3::new(away_x, lean_y, 0.0);
            let input = ContactInput {
                defender: def,
                defender_vel: Vector3::new(p.velocity.x, p.velocity.y, 0.0),
                ball: ball_pos,
                ball_vel,
                // No carrier: this is a pass in flight, nobody to foul.
                carrier: ball_pos - ball_vel * 1000.0,
                axis,
                leg_speed,
                reach,
                windup_ticks: windup,
                intent_dir: intent,
                intent_speed: 0.4,
                restitution: 0.10,
                control_skill: (0.3 + 0.7 * skill).clamp(0.0, 1.0),
                jitter: rng.unit_f32() - rng.unit_f32(),
                jitter_sigma: 0.15 * (1.0 - 0.7 * skill),
                loft_roll: rng.unit_f32(),
            };
            let o = resolve_contact(&input);
            if o.touched {
                swings.push(Swing { idx: i, tick: o.ball_tick.unwrap_or(0), control: o.control, velocity: o.velocity });
            }
        }
        if swings.is_empty() {
            return;
        }
        swings.sort_by_key(|s| s.tick);
        let first = &swings[0];
        // A defender clearly later than the first stands down: only a
        // near-simultaneous contender can spoil.
        let contender = swings
            .iter()
            .skip(1)
            .any(|s| (s.tick as f32) <= first.tick as f32 + STAND_DOWN_TICKS.min(SPOIL_WINDOW_TICKS as f32));
        let mut velocity = first.velocity;
        let sp = (velocity.x * velocity.x + velocity.y * velocity.y).sqrt();
        let mut controlled = sp <= CONTROL_MAX_SPEED;
        if controlled && contender && rng.unit_f32() < SPOIL_PROB {
            // Two legs at the ball: it squirts loose between them.
            controlled = false;
            let a = rng.unit_f32() * std::f32::consts::TAU;
            velocity.x += a.cos() * 1.2;
            velocity.y += a.sin() * 1.2;
        }
        let p = &players[first.idx];
        self.pending_intercept = Some(PendingIntercept {
            apply_tick: self.current_tick_cached + first.tick as u64,
            defender_id: p.id,
            defender_team: p.team_id,
            defender_side: p.side,
            velocity,
            controlled,
        });
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

        // One block attempt per defender per shot: a new shot (or a
        // deflection, which gets a new dispatch tick) starts a fresh list.
        if let Some(t) = self.cached_shot_target.as_ref() {
            if self.block_roll_shot_tick != t.dispatch_tick {
                self.block_rolled.clear();
                self.block_roll_shot_tick = t.dispatch_tick;
            }
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

            if self.block_rolled.contains(&player.id) {
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
        // The chance is scaled (see `block_once_scale`) and each defender
        // gets exactly one roll for this shot, win or lose.
        let chance = (best_chance * block_once_scale()).clamp(0.03, 0.5);
        let blocker_id = match best_blocker {
            Some(id) => {
                self.block_rolled.push(id);
                if context.rng.unit_f32() < chance {
                    id
                } else {
                    return;
                }
            }
            None => return,
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

        crate::snap_tag!("block_to_blocker");
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
}
