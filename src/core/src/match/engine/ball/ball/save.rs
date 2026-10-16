//! Goalkeeper shot-stopping: ONE decision per shot, made where the ball
//! reaches the keeper. Design + conflict check:
//! `docs/goalkeeper-shot-stopping-from-null.md` (The Gaffer repo).
//!
//! Replaces three compounding mechanisms (the ball-layer roll capped at
//! 0.45, the Catching state's per-tick roll and the Diving state's catch
//! roll) that together produced a save rate no one could reason about,
//! snapped the ball to the keeper from up to 85 m away, and made the
//! keeper hold ~92% of what he saved (real: ~34%).
//!
//! * `P(save | in reach)` is high (~0.9) and moves in logit space with
//!   how far the keeper has to stretch, his skill, shot distance,
//!   deflection and the direct-free-kick calibration. Whether he can
//!   reach the shot at all is the existing physical reach gate. The
//!   OVERALL result is checked against real data (StatsBomb Bundesliga
//!   2023/24, n=331 on-target open-play shots: saved 55/71/87% at
//!   <11/11-18/>18 m); see the design doc.
//! * A save is then one of four REAL ball outcomes (real shares of saves,
//!   saved-twice excluded: held 38.6 / parried into danger 27.5 / touched
//!   out 22.2 / parried to a safe area 11.6). Parries are real
//!   trajectories launched through `release_after_touch`; nothing is
//!   placed or teleported except a clean hold, which happens at the
//!   keeper's hands.

use super::Ball;
use crate::PlayerFieldPositionGroup;
use crate::r#match::ball::events::BallEvent;
use crate::r#match::engine::goal::GOAL_WIDTH;
use crate::r#match::events::EventCollection;
use crate::r#match::player::strategies::players::ops::effective_skill::{
    ActionContext as EffSkillCtx, effective_skill,
};
use crate::r#match::player::strategies::players::ops::goalkeeper_skill::shot_limited_reach;
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::{MatchContext, MatchPlayer, PassOriginRestart, PlayerSide};
use nalgebra::Vector3;

/// P(save | the keeper can physically reach the shot). The real
/// save-by-distance curve (55/71/87% at <11/11-18/>18 m) already folds in
/// "the keeper could not get there", and the engine models reach
/// separately (a hard physical cut, ~35% of shots faced), so using the
/// real curve here would count that twice. A keeper who reaches a shot
/// saves it most of the time; the distance pattern then comes out of
/// reach. Calibrated so the OVERALL save share lands near the real
/// bands (see docs/goalkeeper-shot-stopping-from-null.md, verification).
const LOGIT_BASE: f32 = 3.0;
/// Weak long shots are slightly easier, close ones slightly harder.
const LOGIT_LN_DISTANCE: f32 = 0.6;
const DISTANCE_REF_M: f32 = 11.0;
/// Logit shift for how far the keeper has to stretch: rho 0 (shot at his
/// hands) +0.5*K, rho 1 (full stretch) -0.5*K.
const K_PLACEMENT: f32 = 1.6;
/// Logit shift per unit of keeper shot-stopping composite around 0.5.
const K_SKILL: f32 = 1.2;
/// A shot that took a deflection: the keeper was set for the old line.
const DEFLECTED_LOGIT: f32 = -0.8;
/// Calibration scalars (measured against the real save share by distance
/// band; see the design doc's verification section).
const SAVE_LOGIT_OFFSET: f32 = 0.0;
const DIRECT_FK_LOGIT_OFFSET: f32 = 0.0;
/// Calibration of the held share of saves (real 38.6%).
const HOLD_SHIFT: f32 = 0.20;

/// Sourced split of a NON-held save (real: danger 27.5 / out 22.2 /
/// safe 11.6 of saves => 44.8 / 36.2 / 18.9 of what is not held).
const SPLIT_DANGER: f32 = 0.448;
const SPLIT_OUT: f32 = 0.362;
const SPLIT_SAFE: f32 = 0.189;

#[derive(Clone, Copy, Debug, PartialEq)]
enum SaveOutcome {
    /// Clean catch: the keeper holds the ball.
    Held,
    /// Spilled in front of goal: a live second ball (rebound window).
    ParryDanger,
    /// Tipped round the post / over the byline: corner by the
    /// last-toucher rule.
    TippedOut,
    /// Pushed wide and away, still in play: nobody owns it, last touch
    /// is the keeper.
    ParrySafe,
}

impl Ball {
    /// The keeper's one chance at a shot. Gates are the same as the old
    /// ball-layer check (live shot, one roll per shot via the latch, on
    /// target, within ~2.5 ticks of the line); the difference is that
    /// this is now the ONLY place a shot can be saved.
    pub fn resolve_shot_at_keeper(
        &mut self,
        context: &MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
    ) {
        let shot_target = match self.cached_shot_target {
            Some(t) => t,
            None => return,
        };
        // Not gated on `in_flight_state`: that counter is a 40-tick claim
        // protection which a long shot outlives, so gating on it left every
        // shot beyond ~18 m without any save decision (found 2026-10-15).
        // Claims during a live shot are blocked by the `cached_shot_target`
        // guards in ownership.rs, independent of that counter.
        if self.current_owner.is_some() {
            return;
        }
        // One decision per shot (also pre-consumed by penalties and the
        // conversion model, which decide their own outcome).
        if shot_target.physics_save_rolled {
            return;
        }
        // Judge save-ability by where the shot ends up, not mid-arc.
        if shot_target.goal_line_z > 2.8 {
            return;
        }

        let (goal_x, goal_y) = match shot_target.defending_side {
            PlayerSide::Left => (context.goal_positions.left.x, context.goal_positions.left.y),
            PlayerSide::Right => (
                context.goal_positions.right.x,
                context.goal_positions.right.y,
            ),
        };
        let past_goal_line = match shot_target.defending_side {
            PlayerSide::Left => self.position.x < goal_x,
            PlayerSide::Right => self.position.x > goal_x,
        };
        if past_goal_line {
            self.cached_shot_target = None;
            return;
        }
        let dist_to_goal_x = (self.position.x - goal_x).abs();
        let ball_vx = self.velocity.x.abs().max(0.5);
        if dist_to_goal_x > ball_vx * 2.5 {
            return;
        }
        let moving_toward_goal = match shot_target.defending_side {
            PlayerSide::Left => self.velocity.x < -0.2,
            PlayerSide::Right => self.velocity.x > 0.2,
        };
        if !moving_toward_goal {
            return;
        }
        if (self.position.y - goal_y).abs() > GOAL_WIDTH + 1.0 {
            return;
        }

        let keeper = players.iter().find(|p| {
            p.side == Some(shot_target.defending_side)
                && p.tactical_position.current_position.position_group()
                    == PlayerFieldPositionGroup::Goalkeeper
                && !p.is_sent_off
        });
        let keeper = match keeper {
            Some(k) => k,
            None => return,
        };

        let minute = sc::minute_from_ms(context.total_match_time);
        let tech = EffSkillCtx::technical(minute);
        let mental = EffSkillCtx::mental(minute);
        let expl = EffSkillCtx::explosive(minute);
        let scaled = |v: f32| ((v - 1.0) / 19.0).max(0.0);
        let handling = scaled(effective_skill(keeper, keeper.skills.goalkeeping.handling, tech));
        let reflexes = scaled(effective_skill(keeper, keeper.skills.goalkeeping.reflexes, tech));
        let agility = scaled(effective_skill(keeper, keeper.skills.physical.agility, expl));
        let concentration = scaled(effective_skill(
            keeper,
            keeper.skills.mental.concentration,
            mental,
        ));
        let positioning = (effective_skill(keeper, keeper.skills.mental.positioning, mental)
            / 20.0)
            .clamp(0.0, 1.0);

        // Reach: what he can physically cover in the time the shot gives
        // him (shared geometry, 2026-10-13).
        let max_reach = 20.0 + agility * 8.0 + reflexes * 4.0;
        let reach = shot_limited_reach(max_reach, &shot_target);
        let lateral_error = (keeper.position.y - shot_target.goal_line_y).abs();
        if lateral_error > reach {
            #[cfg(feature = "match-logs")]
            log::info!(
                "SAVEOOR d_m={:.1} lat={:.1} reach={:.1} fk={}",
                shot_target.shot_distance / 8.0,
                lateral_error,
                reach,
                shot_target.is_direct_fk
            );
            return; // Out of reach: nothing to decide, the shot goes on.
        }
        // No-teleport rule (2026-10-16): the save is only a contact if the
        // BALL is physically within the keeper's reach (dive + step,
        // skill-scaled, shot-time-limited). Otherwise the shot goes on and
        // the check repeats next tick as the ball closes.
        {
            let gx = keeper.position.x - self.position.x;
            let gy = keeper.position.y - self.position.y;
            if (gx * gx + gy * gy).sqrt() > reach {
                return;
            }
        }
        let rho = (lateral_error / reach.max(1.0)).clamp(0.0, 1.0);

        let skill = sc::gk_shot_stopping(keeper, minute);
        let d_m = (shot_target.shot_distance / 8.0).max(2.0);
        let env = context.environment.modifiers().goalkeeper_handling;
        let mut logit = LOGIT_BASE
            + LOGIT_LN_DISTANCE * (d_m / DISTANCE_REF_M).ln()
            + K_PLACEMENT * (0.5 - rho)
            + K_SKILL * (skill - 0.5)
            + env * 2.0
            + SAVE_LOGIT_OFFSET;
        if shot_target.deflected {
            logit += DEFLECTED_LOGIT;
        }
        if shot_target.is_direct_fk {
            logit += DIRECT_FK_LOGIT_OFFSET;
        }
        let p_save = (1.0 / (1.0 + (-logit).exp())).clamp(0.03, 0.97);
        // A penalty the conversion model decided as SAVED is aimed at the
        // keeper and is always saved (the outcome mix is still picked
        // below). A decided GOAL never reaches this code (latch consumed).
        let p_save = match self.penalty_decided {
            Some((taker, false)) if self.previous_owner == Some(taker) => 1.0,
            _ => p_save,
        };

        // The one decision.
        if let Some(t) = self.cached_shot_target.as_mut() {
            t.physics_save_rolled = true;
        }
        #[cfg(feature = "match-logs")]
        log::info!(
            "SAVEDIAG d_m={:.1} rho={:.2} skill={:.2} p_save={:.3} fk={} defl={}",
            d_m,
            rho,
            skill,
            p_save,
            shot_target.is_direct_fk,
            shot_target.deflected
        );
        if context.rng.unit_f32() >= p_save {
            return; // Keeper beaten — the shot goes on.
        }

        let ball_speed = self.velocity.norm();
        let outcome = Self::pick_save_outcome(
            context,
            handling,
            positioning,
            concentration,
            ball_speed,
            rho,
        );
        #[cfg(feature = "match-logs")]
        log::info!(
            "SAVEOUT outcome={:?} d_m={:.1} fk={}",
            outcome,
            d_m,
            shot_target.is_direct_fk
        );

        let keeper_id = keeper.id;
        let keeper_pos = keeper.position;
        let keeper_team = keeper.team_id;
        let keeper_side = keeper.side;

        self.position.z = 0.0;
        self.previous_owner = self.current_owner.or(self.previous_owner);
        self.pass_target_player_id = None;
        // Stage the save credit before clearing the shot target (consumed
        // by the event-dispatch step: save for the keeper, on-target for
        // the shooter).
        if let Some(shooter_id) = self.previous_owner {
            self.pending_save_credit = Some((keeper_id, shooter_id));
        }
        self.cached_shot_target = None;
        let tick = self.current_tick_cached;
        self.offside_snapshot = None;
        self.pass_origin_restart = PassOriginRestart::OpenPlay;

        match outcome {
            SaveOutcome::Held => {
                // The keeper dives/steps onto the ball; the ball stays.
                self.pending_player_jumps.push((
                    keeper_id,
                    Vector3::new(self.position.x, self.position.y, 0.0),
                ));
                self.position.z = 0.0;
                self.velocity = Vector3::zeros();
                self.current_owner = Some(keeper_id);
                self.flags.in_flight_state = 0;
                self.claim_cooldown = 200;
                self.record_touch(keeper_id, keeper_team, tick, true);
                events.add_ball_event(BallEvent::Claimed(keeper_id));
            }
            SaveOutcome::TippedOut => {
                // A real trajectory: the ball leaves the keeper's hand
                // toward the byline, wide of the post. It crosses the
                // line outside the posts after T ticks, so the existing
                // last-toucher rule (keeper) awards the corner. Nothing
                // is placed.
                let v = tipped_out_velocity(
                    self.position.x,
                    self.position.y,
                    goal_y,
                    keeper_side == Some(PlayerSide::Left),
                    self.field_width,
                );
                self.release_after_touch(keeper_id, keeper_team, keeper_side, v, tick);
                // It is leaving play by design: no re-claims, and the
                // own-goal "live deflected shot" release_after_touch may
                // have created must not exist (it crosses wide).
                self.cached_shot_target = None;
                self.flags.in_flight_state = 30;
                self.corner_deflection_guard_until_tick = tick + 30;
            }
            SaveOutcome::ParrySafe => {
                // Pushed wide and away from goal, still in play.
                let outward = if self.position.y < goal_y { -1.0 } else { 1.0 };
                let inward = match keeper_side {
                    Some(PlayerSide::Left) => 1.0,
                    _ => -1.0,
                };
                let angle = 0.9 + context.rng.unit_f32() * 0.5; // 52-80° from the goal-line normal
                let speed = (1.6
                    + (ball_speed / 3.2).clamp(0.0, 1.0) * 0.9
                    + context.rng.unit_f32() * 0.4)
                    .clamp(1.4, 3.0);
                let v = Vector3::new(
                    inward * angle.cos() * speed,
                    outward * angle.sin() * speed,
                    0.0,
                );
                self.release_after_touch(keeper_id, keeper_team, keeper_side, v, tick);
                self.flags.in_flight_state = 10;
            }
            SaveOutcome::ParryDanger => {
                // Spilled off the gloves: a live second ball in front of
                // goal. Arms the rebound window.
                self.last_rebound_tick = tick;
                let drop_distance = 12.0 + context.rng.unit_f32() * 18.0;
                let drop_x = match keeper_side {
                    Some(PlayerSide::Left) => keeper_pos.x + drop_distance,
                    Some(PlayerSide::Right) => keeper_pos.x - drop_distance,
                    None => keeper_pos.x,
                };
                let outward_sign = if (self.position.y - goal_y).abs() < 1.0 {
                    if context.rng.unit_f32() < 0.5 { -1.0 } else { 1.0 }
                } else {
                    (self.position.y - goal_y).signum()
                };
                let outward_offset = (14.0 + context.rng.unit_f32() * 16.0) * outward_sign;
                let drop_y = (self.position.y
                    + outward_offset
                    + (context.rng.unit_f32() - 0.5) * 10.0)
                    .clamp(0.0, self.field_height);
                let drop_x = drop_x.clamp(0.0, self.field_width);
                let dx = drop_x - self.position.x;
                let dy = drop_y - self.position.y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                // Energy shed off the hands, not a clearance.
                let parry_speed =
                    (ball_speed * (0.22 + 0.18 * (1.0 - handling))).clamp(0.6, 1.3);
                self.velocity.x = (dx / dist) * parry_speed;
                self.velocity.y = (dy / dist) * parry_speed;
                self.velocity.z = 0.0;
                self.current_owner = None;
                // The keeper is the last to touch it, so the intercept
                // population is the realistic one (attackers pouncing on
                // the spill).
                self.previous_owner = Some(keeper_id);
                self.flags.in_flight_state = 10;
                self.claim_cooldown = 0;
                self.record_touch(keeper_id, keeper_team, tick, false);
                events.add_ball_event(BallEvent::Intercepted(keeper_id, self.previous_owner));
            }
        }
    }

    /// Which of the four real outcomes a save is. Sourced shares:
    /// held 38.6 / danger 27.5 / out 22.2 / safe 11.6 (see module doc).
    fn pick_save_outcome(
        context: &MatchContext,
        handling: f32,
        positioning: f32,
        concentration: f32,
        ball_speed: f32,
        rho: f32,
    ) -> SaveOutcome {
        let power = (ball_speed / 8.0).clamp(0.0, 1.0);
        let p_hold = (0.12 + handling * 0.26 + positioning * 0.10 + concentration * 0.06
            - power * 0.18
            - rho * 0.18
            + HOLD_SHIFT)
            .clamp(0.05, 0.75);

        // A stretched save is tipped away more; a focused keeper spills
        // less into danger and pushes it safe more.
        let w_danger = SPLIT_DANGER * (1.3 - 0.6 * rho) * (1.15 - 0.3 * concentration);
        let w_out = SPLIT_OUT * (0.6 + 0.8 * rho);
        let w_safe = SPLIT_SAFE * (0.8 + 0.4 * concentration);
        let total = w_danger + w_out + w_safe;

        let r = context.rng.unit_f32();
        if r < p_hold {
            return SaveOutcome::Held;
        }
        let rest = (r - p_hold) / (1.0 - p_hold).max(1e-3) * total;
        if rest < w_danger {
            SaveOutcome::ParryDanger
        } else if rest < w_danger + w_out {
            SaveOutcome::TippedOut
        } else {
            SaveOutcome::ParrySafe
        }
    }
}

/// Ground friction per tick on a rolling ball (`motion.rs`,
/// `GROUND_FRICTION_COEFFICIENT`).
const GROUND_FRICTION_PER_TICK: f32 = 0.015;

/// Launch velocity for a keeper's tip round the post: the ball must cross
/// the byline OUTSIDE the posts (so the last-toucher rule awards the
/// corner). The launch is friction-exact: a ball launched with speed `v`
/// covers `v * (1 - (1-f)^T) / f` in `T` ticks, so the velocity is scaled
/// to cover the whole planned displacement in exactly `T` ticks.
fn tipped_out_velocity(
    ball_x: f32,
    ball_y: f32,
    goal_y: f32,
    left_goal: bool,
    field_width: f32,
) -> Vector3<f32> {
    let to_top = ball_y < goal_y;
    let wide_y = if to_top {
        goal_y - GOAL_WIDTH - 14.0
    } else {
        goal_y + GOAL_WIDTH + 14.0
    };
    // Aim the x-travel to reach the line only at the END of the planned
    // sideways travel (a hair beyond it): a bigger margin makes the ball
    // cross the byline early, while it is still inside the posts.
    let line_beyond = if left_goal { -0.1 } else { field_width + 0.1 };
    let dy = wide_y - ball_y;
    let dx = line_beyond - ball_x;
    let t = (dy.abs() / 1.5).max(2.0);
    let f = GROUND_FRICTION_PER_TICK;
    // Total distance factor over t ticks of friction decay.
    let reach = (1.0 - (1.0 - f).powf(t)) / f;
    Vector3::new(dx / reach, dy / reach, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Roll the ball under ground friction until it crosses the byline;
    /// it must cross OUTSIDE the posts, for every start position inside
    /// the goal mouth and both goals.
    #[test]
    fn tipped_out_ball_crosses_the_byline_outside_the_posts() {
        let field_w = 840.0_f32;
        let goal_y = 272.5_f32;
        for &left in &[true, false] {
            for start_y in [goal_y - 28.0, goal_y - 10.0, goal_y, goal_y + 12.0, goal_y + 28.0] {
                for start_depth in [0.5_f32, 3.0, 7.0] {
                    let x0 = if left { start_depth } else { field_w - start_depth };
                    let v = tipped_out_velocity(x0, start_y, goal_y, left, field_w);
                    let (mut x, mut y) = (x0, start_y);
                    let (mut vx, mut vy) = (v.x, v.y);
                    let mut crossed = None;
                    for _ in 0..200 {
                        x += vx;
                        y += vy;
                        vx *= 1.0 - GROUND_FRICTION_PER_TICK;
                        vy *= 1.0 - GROUND_FRICTION_PER_TICK;
                        let beyond = if left { x < 0.0 } else { x > field_w };
                        if beyond {
                            crossed = Some(y);
                            break;
                        }
                    }
                    let y_cross = crossed.expect("tipped ball must reach the byline");
                    assert!(
                        (y_cross - goal_y).abs() > GOAL_WIDTH + 2.0,
                        "crossed at y={y_cross} inside/at the post (left={left}, start_y={start_y}, depth={start_depth})"
                    );
                }
            }
        }
    }
}
