use crate::r#match::player::strategies::common::players::ops::on_ball_value::time_to_intercept;
use crate::r#match::{MatchPlayerLite, PlayerSide, StateProcessingContext};
use nalgebra::Vector3;
#[cfg(feature = "match-logs")]
use std::sync::atomic::Ordering;

/// Operations for defensive positioning and tactics
pub struct DefensiveOperationsImpl<'p> {
    ctx: &'p StateProcessingContext<'p>,
}

/// Role a defender plays relative to the current ball carrier.
///
/// Computed per-tick from geometry (no flag storage), so when the
/// ball carrier moves or the primary presser is dribbled past, roles
/// reassign naturally — the old cover becomes the new primary, the
/// beaten primary drops into help or hold. Every defender computes
/// its own role consistently because the ranking is deterministic
/// (distance to carrier + player-id tiebreak).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefensiveRole {
    /// Closest defender to the ball carrier — engage (press/tackle).
    Primary,
    /// Second-closest — hold a safety position on the line between
    /// ball carrier and own goal, ready to step up if Primary is beaten.
    Cover,
    /// There's a dangerous unmarked non-carrier opponent within reach —
    /// pick them up (close pass lane / mark shadow runner).
    Help,
    /// No immediate individual responsibility — maintain shape.
    Hold,
}

const THREAT_SCAN_DISTANCE: f32 = 100.0;
// realism-bug (2026-08-27): this was 2.0 — unreachable at this engine's
// real velocity scale (measured average opponent speed in a genuine
// wide/deep candidate zone is 0.27-0.43 u/tick; max_speed_with_condition
// tops out ~0.36-0.6 for even an elite, fully-fit player). This
// threshold has been silently impossible to clear for any normal
// running pace since it was introduced — `scan_for_dangerous_runs`
// could only ever fire for corner-case velocity spikes, not real
// attacking runs. 0.30 is a genuine "running, not standing/walking" bar
// at the real scale (same value chosen for the new
// `wide_run_track_override_velocity`, so both mechanisms agree on what
// "moving with real intent" means).
const DANGEROUS_RUN_SPEED: f32 = 0.30;
const DANGEROUS_RUN_ANGLE: f32 = 0.5;

/// TEMPORARY realism-bug diagnostic (2026-08-27, byline-duel root-cause
/// investigation): does `scan_for_dangerous_runs`'s "aligned with the
/// direction to goal CENTER" model actually detect a genuine wide/deep
/// byline run as dangerous? Removed once the investigation concludes.
#[cfg(feature = "match-logs")]
pub mod byline_run_diag {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static WIDE_DEEP_FAST_CANDIDATES: AtomicU64 = AtomicU64::new(0);
    pub static WIDE_DEEP_FAST_PASSED_ALIGNMENT: AtomicU64 = AtomicU64::new(0);
    pub static SUM_ALIGNMENT_X1000: AtomicU64 = AtomicU64::new(0);
    pub static SCAN_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static SCAN_HAD_ANY_NEARBY: AtomicU64 = AtomicU64::new(0);
    pub static WRT_FIRED: AtomicU64 = AtomicU64::new(0);
    pub static WRT_SUM_START_DIST_X100: AtomicU64 = AtomicU64::new(0);
    pub static WIDE_DEEP_SPEED_SAMPLES: AtomicU64 = AtomicU64::new(0);
    pub static WIDE_DEEP_SUM_SPEED_X1000: AtomicU64 = AtomicU64::new(0);
    pub fn reset() {
        for c in [
            &WIDE_DEEP_FAST_CANDIDATES,
            &WIDE_DEEP_FAST_PASSED_ALIGNMENT,
            &SUM_ALIGNMENT_X1000,
            &SCAN_CALLS,
            &SCAN_HAD_ANY_NEARBY,
            &WRT_FIRED,
            &WRT_SUM_START_DIST_X100,
            &WIDE_DEEP_SPEED_SAMPLES,
            &WIDE_DEEP_SUM_SPEED_X1000,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
}

/// TEMPORARY realism-bug diagnostic (2026-09-14, single-defender-
/// exclusivity fix for Intercepting): does `is_best_positioned_to_
/// intercept` actually block a meaningful share of would-be-Intercepting
/// decisions, or is CALLS≈WINS (meaning the old bug was never really
/// firing for multiple defenders at once)? Removed once verified.
#[cfg(feature = "match-logs")]
pub mod intercept_exclusivity_diag {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    pub static WINS: AtomicU64 = AtomicU64::new(0);
    pub fn reset() {
        CALLS.store(0, Ordering::Relaxed);
        WINS.store(0, Ordering::Relaxed);
    }
}

// Coordination constants
const ENGAGEMENT_DISTANCE: f32 = 20.0; // Distance at which a defender is considered "engaging" an opponent
#[allow(dead_code)]
const MIN_MARKING_SEPARATION: f32 = 15.0; // Minimum distance between defenders marking different opponents

// Role assignment thresholds
//
// 2026-08-30 (defensive-positioning gap #2, flagged 2026-08-30 alongside
// the marking-priority fix): measured via a real-match diagnostic
// (COVERDIAG, temporary, removed after) that the second-closest
// defender to a ball carrier (rank==1) was BEYOND the old 45u cutoff
// 85% of the time -- median real distance 87.1u, almost double the
// threshold, across 33,205 sampled rank-1 instances. That meant the
// Cover role (a genuine second body between the carrier and goal,
// exactly what Pavel's "nobody between him and GK" report was missing)
// was almost never actually assigned -- a recovering defender fell
// through to Help (chasing an unrelated mark) or Hold instead, every
// time he was realistically more than ~5.6m from the carrier, which is
// most of the time in open play. Raised to 180u (~22.5m) -- a real
// recovery-run range a defender can plausibly still close down in time
// to matter, per real defensive doctrine (delay/cover duty starts the
// instant you're not the nearest body, not only once you're already
// close) -- covering ~66% of the measured real distribution while
// still excluding the most extreme, clearly-unreachable cases (the
// measured distribution had no sharp natural cutoff; this is a
// reasoned, doctrine-bounded choice, not a pure percentile fit).
const COVER_MAX_DISTANCE: f32 = 180.0;
const HELP_SCAN_RADIUS: f32 = 28.0; // Range in which Help defender looks for a pass option
const COVER_GOAL_SIDE_OFFSET: f32 = 12.0; // Cover sits this far goal-side of carrier

// docs/defensive-delay-and-engagement-timing.md's 4th delay-exit trigger
// (docs/positional-football-master-plan.md, Stage 4, 2026-09-07): "the
// defender's own cover has arrived (a teammate is now goal-side and
// ready, so committing to a challenge no longer risks leaving nobody
// behind)". A covering teammate doesn't need to stand on the exact
// `cover_target_position()` point to count as "ready" -- real doctrine
// treats it as a zone, not a pixel, the same way `JOCKEY_HOLD_DISTANCE`/
// `COVER_GOAL_SIDE_OFFSET` are themselves target zones elsewhere in this
// file. No sourced real-world "how close counts as arrived" figure
// exists (category (d), flagged estimate) -- chosen close to
// `JOCKEY_ZONE_OUTER`'s own 36u so "cover has arrived" and "I'm still
// within jockeying range of the carrier" describe a comparable real
// scale of ground.
const COVER_ARRIVED_RADIUS: f32 = 30.0;

/// D8.1 (2026-08-22): reachability window for including a midfielder in
/// the unified Primary/Cover/Help/Hold pool for a given ball-carrier
/// threat. Reuses `on_ball_value::time_to_intercept`'s own
/// reaction-plus-speed model rather than a flat radius — the same
/// "genuine race/reachability, not an invented distance" discipline
/// `defensive_value.rs::boundary_push_viability` (D6) already
/// established for a different decision. The value is an honest unit
/// conversion of D0's own already-established "byline 1v1" proximity
/// figure (130u — the same geometry `SAFE_RESTART_LINE_MARGIN`/
/// `CONCEDE_GOAL_SIDE_SATURATION` already share) through
/// `time_to_intercept` at a typical midfielder max speed (~0.45u/tick):
/// (130.0 / 0.45) + 15.0 (REACTION_TICKS) ≈ 304 ticks (~3s), rounded —
/// not a fresh invented number. No sourced real-world dataset exists for
/// "how close must a midfielder be to be a genuine defensive option"
/// (category (d)) — D8.1's own raw-data verification is what actually
/// validates or revises this, not the reasoning alone.
const MIDFIELDER_REACHABILITY_TICKS: f32 = 300.0;

impl<'p> DefensiveOperationsImpl<'p> {
    pub fn new(ctx: &'p StateProcessingContext<'p>) -> Self {
        DefensiveOperationsImpl { ctx }
    }

    /// Scan for opponents making dangerous runs toward goal
    pub fn scan_for_dangerous_runs(&self) -> Option<MatchPlayerLite> {
        self.scan_for_dangerous_runs_with_distance(THREAT_SCAN_DISTANCE)
    }

    /// Scan for dangerous runs with custom scan distance
    pub fn scan_for_dangerous_runs_with_distance(
        &self,
        scan_distance: f32,
    ) -> Option<MatchPlayerLite> {
        let own_goal_position = self.ctx.ball().direction_to_own_goal();

        #[cfg(feature = "match-logs")]
        {
            byline_run_diag::SCAN_CALLS.fetch_add(1, Ordering::Relaxed);
            if self.ctx.players().opponents().nearby(scan_distance).next().is_some() {
                byline_run_diag::SCAN_HAD_ANY_NEARBY.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Find the closest opponent making a dangerous run — no intermediate Vec
        self.ctx
            .players()
            .opponents()
            .nearby(scan_distance)
            .filter(|opp| {
                let velocity = opp.velocity(self.ctx);
                let speed = velocity.norm();

                // Must be moving at significant speed
                if speed < DANGEROUS_RUN_SPEED {
                    return false;
                }

                // Check if running toward our goal
                let to_goal = (own_goal_position - opp.position).normalize();
                let velocity_dir = velocity.normalize();
                let alignment = velocity_dir.dot(&to_goal);

                #[cfg(feature = "match-logs")]
                {
                    // Same wide/deep zone `byline_commitment_ticks` (run.rs)
                    // targets: y outside the central 38-62% band, x-depth
                    // from goal 130-380u, moving at real pace.
                    let h = self.ctx.context.field_size.height as f32;
                    let wide = opp.position.y < h * 0.38 || opp.position.y > h * 0.62;
                    let goal_x = own_goal_position.x;
                    let depth = (goal_x - opp.position.x).abs();
                    if wide && (130.0..380.0).contains(&depth) && speed >= DANGEROUS_RUN_SPEED {
                        byline_run_diag::WIDE_DEEP_FAST_CANDIDATES.fetch_add(1, Ordering::Relaxed);
                        byline_run_diag::SUM_ALIGNMENT_X1000
                            .fetch_add((alignment.max(-1.0) * 1000.0 + 1000.0) as u64, Ordering::Relaxed);
                        if alignment >= DANGEROUS_RUN_ANGLE {
                            byline_run_diag::WIDE_DEEP_FAST_PASSED_ALIGNMENT
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                if alignment < DANGEROUS_RUN_ANGLE {
                    return false;
                }

                // Check if in attacking position (closer to our goal than most defenders)
                let defender_x = self.ctx.player.position.x;
                let is_in_dangerous_position =
                    if own_goal_position.x < self.ctx.context.field_size.width as f32 / 2.0 {
                        opp.position.x < defender_x + 20.0 // Attacker is ahead or close
                    } else {
                        opp.position.x > defender_x - 20.0
                    };

                if !is_in_dangerous_position {
                    return false;
                }

                // Skip runners already being engaged by a teammate
                // defender. Without this filter, two defenders both
                // identify the same runner and both mark them,
                // double-teaming while secondary runners go free.
                if self.is_opponent_being_engaged(opp) {
                    return false;
                }

                true
            })
            .min_by(|a, b| {
                let dist_a = a.distance(self.ctx);
                let dist_b = b.distance(self.ctx);
                dist_a.total_cmp(&dist_b)
            })
    }

    /// Find the opponent defensive line position — single-pass, zero allocation
    pub fn find_defensive_line(&self) -> f32 {
        // realism-bug (offside investigation): this used to filter to
        // `is_defender()`-tagged opponents only, which misses a deep-
        // sitting midfielder or wing-back who's actually the deepest
        // outfield player — the real offside rule (`build_offside_snapshot`
        // in players.rs) sorts ALL opponents by depth (GK included) and
        // takes the second-to-last. GK is excluded here deliberately
        // (not to approximate "second-to-last" by omission — a GK who
        // isn't the deepest player is the rare edge case the real rule
        // already handles by including him in its own sort; this
        // self-assessment helper only needs the deepest OUTFIELD line,
        // which is what "last defender" means in the ordinary case).
        let (sum, count, min_x, max_x) = self
            .ctx
            .players()
            .opponents()
            .all()
            .filter(|p| !p.tactical_positions.is_goalkeeper())
            .map(|p| p.position.x)
            .fold((0.0f32, 0u32, f32::MAX, f32::MIN), |(s, c, mn, mx), x| {
                (s + x, c + 1, mn.min(x), mx.max(x))
            });

        if count == 0 {
            return self.ctx.context.field_size.width as f32 / 2.0;
        }

        match self.ctx.player.side {
            Some(PlayerSide::Left) => max_x,
            Some(PlayerSide::Right) => min_x,
            None => sum / count as f32,
        }
    }

    /// Shared "shoulder of the last defender" tolerance. Previously
    /// four different constants (1.5 / 2.0 / 5.0 / 8.0) existed across
    /// the various offside-adjacent checks in this codebase for
    /// conceptually the same margin — unified here so every caller
    /// agrees on what "held onside" means.
    pub const OFFSIDE_HOLD_MARGIN: f32 = 6.0;

    /// Clamp a movement target's x-coordinate so a player chasing/
    /// running toward `natural_target_x` doesn't get ahead of the
    /// opponent's defensive line while `teammate_on_ball` is true (a
    /// teammate still holds the ball, or — for a chase/pursuit target —
    /// the ball itself hasn't yet travelled past the line). The clamp
    /// lifts once the caller passes `teammate_on_ball = false`. Shared
    /// by every "hold onside until the ball is genuinely released"
    /// site (RunningInBehind, Midfielder Walking/Running, TakeBall)
    /// instead of each reimplementing the same min/max-by-side logic.
    pub fn onside_hold_x(&self, natural_target_x: f32, teammate_on_ball: bool) -> f32 {
        if !teammate_on_ball {
            return natural_target_x;
        }
        let line = self.find_defensive_line();
        match self.ctx.player.side {
            Some(PlayerSide::Left) => natural_target_x.min(line - Self::OFFSIDE_HOLD_MARGIN),
            Some(PlayerSide::Right) => natural_target_x.max(line + Self::OFFSIDE_HOLD_MARGIN),
            None => natural_target_x,
        }
    }

    /// realism-bug (offside investigation): `TakeBall` (the universal
    /// "go collect the ball" state every position group enters the
    /// instant they're closest to an unowned ball — including a
    /// teammate's pass still in flight, since `current_owner` is None
    /// while airborne) had zero offside awareness. Measured: 84% of
    /// real offside calls happened with the receiver already in
    /// TakeBall at the moment of the kick, sprinting well ahead of the
    /// defensive line before a teammate's pass even targeted them
    /// (mean overshoot ~7.6m). `onside_hold_x` doesn't fit here — it
    /// keys on "does a teammate still hold the ball," which is never
    /// true during TakeBall by construction (the ball must be unowned
    /// to trigger it). The right condition is instead "has the BALL
    /// ITSELF already travelled past the line" — a real striker times
    /// his run to arrive with or after a through ball, not ahead of it.
    /// If the ball is still short of the line, hold the chase target at
    /// the line; once the ball has genuinely crossed it, the clamp
    /// lifts and the chase proceeds normally (a legitimate onside run
    /// to meet a ball already released into the space behind the
    /// defence).
    pub fn clamp_chase_target_x(&self, chase_target_x: f32, ball_x: f32) -> f32 {
        let line = self.find_defensive_line();
        match self.ctx.player.side {
            Some(PlayerSide::Left) => {
                if ball_x > line {
                    chase_target_x
                } else {
                    chase_target_x.min(line - Self::OFFSIDE_HOLD_MARGIN)
                }
            }
            Some(PlayerSide::Right) => {
                if ball_x < line {
                    chase_target_x
                } else {
                    chase_target_x.max(line + Self::OFFSIDE_HOLD_MARGIN)
                }
            }
            None => chase_target_x,
        }
    }

    /// Distance at which ANY defender-role state may legally transition
    /// into `DefenderState::Tackling`. Must exactly match
    /// `DefenderTacklingState`'s own attempt-gate distance
    /// (`TACKLE_DISTANCE_THRESHOLD`) — a looser outer-state threshold
    /// means the transition fires, then bounces straight back out the
    /// instant `Tackling::process()` runs its own check, repeating every
    /// tick for as long as the mismatch condition holds (see
    /// engine-audit-results.md, Module 6).
    ///
    /// Centralized 2026-08-30 after the 2026-07-27 Pressing fix (which
    /// found and corrected exactly this mismatch, 25u vs the real 10u
    /// gate) turned out to have fixed only 1 of 8 transition sites — the
    /// identical bug, independently reintroduced, was still live in
    /// covering/intercepting/marking/guarding/standing/holding_line/
    /// walking, none of them sharing a constant with Tackling or with
    /// each other. One source of truth now closes off the whole class,
    /// not just the instance that happened to get noticed.
    pub const TACKLE_ENGAGEMENT_DISTANCE: f32 = 10.0;

    /// Whether a defender-role state is close enough, and off cooldown,
    /// to legally transition into `Tackling` right now. Centralizes
    /// both halves of the gate `Tackling` itself enforces on entry, so a
    /// transition only ever fires when it will actually be accepted —
    /// the previous per-site pattern duplicated the distance check with
    /// its own local constant (values ranging 15u-40u across the 7
    /// broken sites) and never checked the cooldown at all, so even a
    /// distance-correct site could still fire a doomed transition while
    /// the player was on tackle cooldown.
    pub fn can_engage_tackle(&self, distance_to_opponent: f32) -> bool {
        distance_to_opponent < Self::TACKLE_ENGAGEMENT_DISTANCE && self.ctx.player.can_attempt_tackle()
    }

    /// Find the own team's defensive line position — single-pass, zero allocation
    pub fn find_own_defensive_line(&self) -> f32 {
        let (sum, count) = self
            .ctx
            .players()
            .teammates()
            .defenders()
            .map(|p| p.position.x)
            .fold((0.0f32, 0u32), |(s, c), x| (s + x, c + 1));

        if count == 0 {
            self.ctx.context.field_size.width as f32 / 2.0
        } else {
            sum / count as f32
        }
    }

    /// Check if there's exploitable space behind opponent defense
    pub fn check_space_behind_defense(&self, defensive_line: f32) -> bool {
        let player_x = self.ctx.player.position.x;

        match self.ctx.player.side {
            Some(PlayerSide::Left) => {
                // Space exists if defensive line is high and there's room behind
                defensive_line < self.ctx.context.field_size.width as f32 * 0.7
                    && player_x < defensive_line + 20.0
            }
            Some(PlayerSide::Right) => {
                defensive_line > self.ctx.context.field_size.width as f32 * 0.3
                    && player_x > defensive_line - 20.0
            }
            None => false,
        }
    }

    /// Check if player is the last defender
    pub fn is_last_defender(&self) -> bool {
        self.ctx
            .players()
            .teammates()
            .defenders()
            .all(|d| match self.ctx.player.side {
                Some(PlayerSide::Left) => d.position.x >= self.ctx.player.position.x,
                Some(PlayerSide::Right) => d.position.x <= self.ctx.player.position.x,
                None => false,
            })
    }

    /// Check if should hold defensive line — single-pass, zero allocation
    pub fn should_hold_defensive_line(&self) -> bool {
        let (sum, count) = self
            .ctx
            .players()
            .teammates()
            .defenders()
            .map(|d| d.position.x)
            .fold((0.0f32, 0u32), |(s, c), x| (s + x, c + 1));

        if count == 0 {
            return false;
        }

        let avg_defender_x = sum / count as f32;

        (self.ctx.player.position.x - avg_defender_x).abs() < 5.0
            && self.ctx.ball().distance() > 200.0
            && !self.ctx.team().is_control_ball()
    }

    /// True when this attacker is sitting beyond the opposing defensive
    /// line AND their team doesn't have possession — i.e. stranded in an
    /// offside position with nothing to do. Forwards use this to trigger
    /// a drop back; otherwise they camp near the opponent goal, the ball
    /// gets cleared upfield, and every subsequent pass reaches them
    /// offside. Needs an `OFFSIDE_MARGIN` so a forward holding the line
    /// on the shoulder of the last defender isn't forced to retreat
    /// unnecessarily — only steps back when clearly stranded.
    pub fn is_stranded_offside(&self) -> bool {
        if self.ctx.team().is_control_ball() {
            return false;
        }
        const OFFSIDE_MARGIN: f32 = 8.0; // ~1m, the shoulder-of-last-defender tolerance
        let line = self.find_defensive_line();
        let my_x = self.ctx.player.position.x;
        match self.ctx.player.side {
            Some(PlayerSide::Left) => my_x > line + OFFSIDE_MARGIN,
            Some(PlayerSide::Right) => my_x < line - OFFSIDE_MARGIN,
            None => false,
        }
    }

    /// The opposing team has the ball in our defensive third — every
    /// defender, regardless of fatigue/position/current state, must drop
    /// every passive duty (Resting, Returning, Guarding) and engage.
    /// Without this, the passive-state transition logic kept defenders
    /// stuck "returning to position" or "resting until 90% stamina"
    /// while red attackers were literally inside the penalty box.
    pub fn is_defensive_crisis(&self) -> bool {
        if !self.ctx.ball().on_own_side() {
            return false;
        }
        if self.ctx.players().opponents().with_ball().next().is_none() {
            return false;
        }
        let own_goal = self.ctx.ball().direction_to_own_goal();
        let ball_to_goal = (self.ctx.tick_context.positions.ball.position - own_goal).magnitude();
        // Defensive third distance from own goal — third of the field length.
        let third = self.ctx.context.field_size.width as f32 / 3.0;
        ball_to_goal < third
    }

    /// Box emergency — the ball is being carried by an opponent
    /// INSIDE our penalty area. Every defender close enough to make a
    /// difference should abandon their shape-holding duties and engage
    /// immediately. Previously defenders stuck to Cover/Hold roles and
    /// let attackers dribble through the 18-yard box unopposed; real
    /// football: once the ball is in your box, position and coordination
    /// stop mattering — stop the shot NOW.
    ///
    /// Returns `true` if THIS defender should break shape and press the
    /// carrier. Two nearest defenders engage; the rest stay put so we
    /// don't leave the other side of the box open.
    pub fn is_box_emergency_for_me(&self) -> bool {
        if !self.ctx.ball().in_own_penalty_area() {
            return false;
        }
        let Some(carrier) = self.ctx.players().opponents().with_ball().next() else {
            return false;
        };
        let my_id = self.ctx.player.id;
        let my_dist = (self.ctx.player.position - carrier.position).magnitude();
        // Only the two closest defenders engage. The rest hold shape
        // so the box isn't emptied. Rank by distance, tiebreak on id.
        let closer_defenders = self
            .ctx
            .players()
            .teammates()
            .defenders()
            .filter(|d| d.id != my_id)
            .filter(|d| {
                let dist = (d.position - carrier.position).magnitude();
                dist < my_dist || (dist == my_dist && d.id < my_id)
            })
            .count();
        closer_defenders < 2
    }

    /// Attacker is approaching our penalty area and is in our zone —
    /// step out to meet them BEFORE they reach the box. Real football:
    /// a deep block doesn't mean waiting motionless at the edge of the
    /// 6-yard line; defenders hold a line at the 18-yard box and step
    /// to the carrier as they cross some trigger distance.
    pub fn should_step_up_to_meet_attacker(&self) -> bool {
        let Some(carrier) = self.ctx.players().opponents().with_ball().next() else {
            return false;
        };
        let own_goal = self.ctx.ball().direction_to_own_goal();
        let carrier_to_goal_sq = (carrier.position - own_goal).norm_squared();
        let field_width = self.ctx.context.field_size.width as f32;
        // Trigger zone: carrier within ~20% of field length from our goal
        // (the edge of our defensive third, approaching the box).
        let trigger = field_width * 0.22;
        if carrier_to_goal_sq > trigger * trigger {
            return false;
        }
        // Only the closest defender to the carrier steps up — others hold
        // shape and track secondary runners.
        let my_id = self.ctx.player.id;
        let my_dist_sq = (self.ctx.player.position - carrier.position).norm_squared();
        let closer = self
            .ctx
            .players()
            .teammates()
            .defenders()
            .filter(|d| d.id != my_id)
            .any(|d| (d.position - carrier.position).norm_squared() < my_dist_sq);
        !closer
    }

    /// Calculate optimal covering position
    pub fn calculate_covering_position(&self) -> Vector3<f32> {
        let goal_position = self.ctx.ball().direction_to_own_goal();
        let ball_position = self.ctx.tick_context.positions.ball.position;

        // Position between ball and goal
        let to_ball = ball_position - goal_position;
        let covering_distance = to_ball.magnitude() * 0.3; // 30% of the way from goal to ball

        goal_position + to_ball.normalize() * covering_distance
    }

    /// Check if in dangerous defensive position (near own goal)
    pub fn in_dangerous_position(&self) -> bool {
        let distance_to_goal =
            (self.ctx.player.position - self.ctx.ball().direction_to_own_goal()).magnitude();
        let danger_threshold = self.ctx.context.field_size.width as f32 * 0.15; // 15% of field width

        distance_to_goal < danger_threshold
    }

    // ==================== DEFENDER COORDINATION ====================

    /// Check if an opponent is already being engaged by a teammate
    /// defender OR midfielder. Widened from defenders-only (2026-08-22,
    /// D8.2) — now that midfielders can also hold Primary/Cover/Help
    /// roles (`defensive_role_for_ball_carrier_unified`), leaving this
    /// defenders-only would let a defender double-mark an opponent a
    /// midfielder is already engaging. No reachability pre-filter is
    /// needed here (unlike the unified pool) — the existing
    /// `ENGAGEMENT_DISTANCE`/velocity-alignment checks below already do
    /// the real gating for "is someone genuinely engaging them right
    /// now," a much tighter question than "could someone reach them."
    /// Returns true if another teammate is closer AND actively moving
    /// toward them.
    pub fn is_opponent_being_engaged(&self, opponent: &MatchPlayerLite) -> bool {
        let my_distance = (self.ctx.player.position - opponent.position).magnitude();

        let teammates = self.ctx.players().teammates();
        teammates
            .defenders()
            .chain(teammates.midfielders())
            .filter(|d| d.id != self.ctx.player.id)
            .any(|teammate| {
                let teammate_distance = (teammate.position - opponent.position).magnitude();

                // Teammate must be within engagement distance
                if teammate_distance > ENGAGEMENT_DISTANCE {
                    return false;
                }

                // Teammate must be closer than us
                if teammate_distance >= my_distance {
                    return false;
                }

                // Teammate must be actively moving toward this opponent (not just standing nearby)
                let teammate_velocity = teammate.velocity(self.ctx);
                let speed = teammate_velocity.norm();

                if speed > 0.5 {
                    let to_opponent = (opponent.position - teammate.position).normalize();
                    let velocity_dir = teammate_velocity.normalize();
                    let alignment = velocity_dir.dot(&to_opponent);

                    // Actively pursuing: moving toward opponent with reasonable alignment
                    alignment > 0.3
                } else {
                    // Stationary but very close — counts as engaged (physically blocking)
                    teammate_distance < 8.0
                }
            })
    }

    /// Assign a defensive role to this defender relative to the current
    /// ball carrier. Recomputed each tick, so role changes with geometry
    /// (e.g. primary gets dribbled past → old cover promotes to primary
    /// next tick).
    ///
    /// D8.2 (2026-08-22): superseded — all 9 real call sites across the
    /// six defender states were switched to `defensive_role_for_ball_
    /// carrier_unified` (defenders + reachability-gated midfielders in
    /// one shared pool, so a defender and a midfielder agree on who's
    /// actually Primary instead of each independently ranking only their
    /// own position group). Kept, not deleted, so the pre-D8.2 behavior
    /// stays a one-line diff away if a revert is ever needed; genuinely
    /// unused right now — safe to remove in a later cleanup pass.
    #[allow(dead_code)]
    pub fn defensive_role_for_ball_carrier(&self) -> DefensiveRole {
        let Some(ball_carrier) = self.ctx.players().opponents().with_ball().next() else {
            return DefensiveRole::Hold;
        };

        let my_id = self.ctx.player.id;
        let my_dist = (self.ctx.player.position - ball_carrier.position).magnitude();

        // Rank among defender teammates by distance to ball carrier.
        // Tiebreak on id so every defender agrees on the ranking.
        let rank = self
            .ctx
            .players()
            .teammates()
            .defenders()
            .filter(|d| d.id != my_id)
            .filter(|d| {
                let dist = (d.position - ball_carrier.position).magnitude();
                dist < my_dist || (dist == my_dist && d.id < my_id)
            })
            .count();

        match rank {
            0 => DefensiveRole::Primary,
            1 if my_dist < COVER_MAX_DISTANCE => DefensiveRole::Cover,
            _ => {
                let carrier_id = ball_carrier.id;
                let has_help_target = self
                    .ctx
                    .players()
                    .opponents()
                    .nearby(HELP_SCAN_RADIUS)
                    .any(|opp| opp.id != carrier_id && !self.is_opponent_being_engaged(&opp));
                if has_help_target {
                    DefensiveRole::Help
                } else {
                    DefensiveRole::Hold
                }
            }
        }
    }

    /// D8.1 (2026-08-22, isolated — not called from anywhere except this
    /// file's own unified role function and, eventually, a diagnostic).
    /// Ranking pool for `defensive_role_for_ball_carrier_unified`: every
    /// defender teammate, unconditional (identical to the pre-D8 pool
    /// `defensive_role_for_ball_carrier` above already uses), plus every
    /// midfielder teammate whose `time_to_intercept` to the ball
    /// carrier's own position is within `MIDFIELDER_REACHABILITY_TICKS`.
    /// Defenders are never reachability-gated here — D8 doesn't change
    /// anything about how a defender is judged eligible, only widens WHO
    /// ELSE competes for the same role.
    fn unified_defensive_pool(&self, ball_carrier_pos: Vector3<f32>) -> Vec<MatchPlayerLite> {
        let teammates = self.ctx.players().teammates();
        let mut pool: Vec<MatchPlayerLite> = teammates.defenders().collect();
        pool.extend(teammates.midfielders().filter(|m| {
            time_to_intercept(self.ctx, m, ball_carrier_pos) <= MIDFIELDER_REACHABILITY_TICKS
        }));
        pool
    }

    /// D8.1 (2026-08-22, isolated — new function, not called from any
    /// state's `process()` yet; the pre-D8 `defensive_role_for_ball_
    /// carrier` above is untouched and still the only one every existing
    /// defender state consumes, so this has zero effect on live match
    /// behavior until D8.2 wires it in). Same Primary/Cover/Help/Hold
    /// ranking logic as the pre-D8 function, computed over `unified_
    /// defensive_pool` instead of `.teammates().defenders()` alone — a
    /// genuinely closer, reachable midfielder can out-rank a defender
    /// for Primary/Cover, and vice versa. Callable from any position
    /// group's context: the pool always includes every defender plus
    /// every reachable midfielder regardless of the caller's own role,
    /// which is what lets D8.2 wire this into midfielder states without
    /// a parallel implementation.
    ///
    /// Self-eligibility: a midfielder calling this while too far away to
    /// plausibly be a candidate at all (fails its own reachability
    /// check) is NOT in the pool by construction — short-circuits to
    /// `Hold` rather than computing a rank against a pool it was never
    /// actually part of, which could otherwise produce a spurious
    /// Primary/Cover for a midfielder nowhere near the play.
    pub fn defensive_role_for_ball_carrier_unified(&self) -> DefensiveRole {
        let Some(ball_carrier) = self.ctx.players().opponents().with_ball().next() else {
            return DefensiveRole::Hold;
        };

        if self.ctx.player.tactical_position.current_position.is_midfielder() {
            let my_lite = MatchPlayerLite {
                id: self.ctx.player.id,
                position: self.ctx.player.position,
                tactical_positions: self.ctx.player.tactical_position.current_position,
            };
            if time_to_intercept(self.ctx, &my_lite, ball_carrier.position)
                > MIDFIELDER_REACHABILITY_TICKS
            {
                return DefensiveRole::Hold;
            }
        }

        let my_id = self.ctx.player.id;
        let my_dist = (self.ctx.player.position - ball_carrier.position).magnitude();

        let pool = self.unified_defensive_pool(ball_carrier.position);
        let rank = pool
            .iter()
            .filter(|d| d.id != my_id)
            .filter(|d| {
                let dist = (d.position - ball_carrier.position).magnitude();
                dist < my_dist || (dist == my_dist && d.id < my_id)
            })
            .count();

        match rank {
            0 => DefensiveRole::Primary,
            1 if my_dist < COVER_MAX_DISTANCE => DefensiveRole::Cover,
            _ => {
                let carrier_id = ball_carrier.id;
                let has_help_target = self
                    .ctx
                    .players()
                    .opponents()
                    .nearby(HELP_SCAN_RADIUS)
                    .any(|opp| opp.id != carrier_id && !self.is_opponent_being_engaged(&opp));
                if has_help_target {
                    DefensiveRole::Help
                } else {
                    DefensiveRole::Hold
                }
            }
        }
    }

    /// Target position for a Cover-role defender: on the line between
    /// ball carrier and own goal, `COVER_GOAL_SIDE_OFFSET` units behind
    /// the carrier. If Primary is beaten, this defender is the next
    /// body between attacker and goal.
    pub fn cover_target_position(&self) -> Option<Vector3<f32>> {
        let ball_carrier = self.ctx.players().opponents().with_ball().next()?;
        let own_goal = self.ctx.ball().direction_to_own_goal();
        let to_goal = own_goal - ball_carrier.position;
        let to_goal_dist = to_goal.magnitude();
        if to_goal_dist < 0.1 {
            return Some(own_goal);
        }
        let dir = to_goal / to_goal_dist;
        Some(ball_carrier.position + dir * COVER_GOAL_SIDE_OFFSET)
    }

    /// Has a teammate arrived at (or near) the Cover position for the
    /// current ball carrier? The 4th delay-exit trigger from
    /// docs/defensive-delay-and-engagement-timing.md -- lets a delaying
    /// Primary defender/midfielder commit to a challenge once a genuine
    /// second line is in place, rather than jockeying forever even after
    /// cover has shown up. Deliberately does NOT recompute
    /// `defensive_role_for_ball_carrier_unified` for every teammate
    /// (expensive, and the wrong question anyway -- a teammate standing
    /// near the ideal cover spot for some OTHER reason still provides
    /// the same real safety net) -- instead reuses the grid-backed
    /// `nearby_at` lookup already used elsewhere in this codebase for a
    /// cheap spatial check, per this design doc's own open question
    /// about avoiding a full-team scan. `self.ctx.player.id` is excluded
    /// so a Primary defender can't read his own proximity to a point
    /// well behind himself as "cover has arrived".
    pub fn cover_is_ready(&self) -> bool {
        let Some(cover_point) = self.cover_target_position() else {
            return false;
        };
        let my_id = self.ctx.player.id;
        self.ctx
            .players()
            .teammates()
            .nearby_at(cover_point, COVER_ARRIVED_RADIUS)
            .any(|t| t.id != my_id)
    }

    /// Is THIS defender the single fastest defender-group teammate to
    /// reach `point` (typically the ball's own current position, for a
    /// loose/through ball with no carrier)?
    ///
    /// docs/defensive-delay-and-engagement-timing.md, "single-defender-
    /// exclusivity fix for Intercepting" (2026-09-14): `DefenderHoldingLineState`'s
    /// three `Intercepting` transitions (the through-ball branch and the
    /// two loose-ball branches) were each a pure per-player check --
    /// "is the ball heading toward MY goal, within MY distance/angle
    /// threshold" -- with zero teammate coordination. Since all four
    /// defenders watch the same shared goal and the same loose ball,
    /// this condition is very likely to be satisfied for several or all
    /// of them on the same tick, which is the confirmed mechanism behind
    /// Pavel's report ("all four defenders placed very high as if all of
    /// them were trying to intercept the pass").
    ///
    /// Reuses `time_to_intercept` (`on_ball_value.rs`) -- the same real
    /// race-in-time model (condition-adjusted max speed + reaction floor)
    /// already trusted for the passing-lane and D8 cover-role work --
    /// rather than a flat distance comparison, so a genuinely faster
    /// covering teammate correctly wins even from further away. Same
    /// rank-0 tie-break convention as `defensive_role_for_ball_carrier_
    /// unified` (lower id wins an exact tie) for determinism.
    ///
    /// Scoped to defender-group teammates only, matching the reported
    /// symptom ("all four defenders") -- not the wider unified Primary/
    /// Cover/Help/Hold pool, which answers a different question (who
    /// should engage an opponent WHO HAS THE BALL) and doesn't apply
    /// here (these three branches are all gated on a ball with no
    /// owner). Whether midfielders should also compete for this is an
    /// open question, not resolved by this pass -- see the doc's own
    /// "not yet investigated" note before extending scope.
    pub fn is_best_positioned_to_intercept(&self, point: Vector3<f32>) -> bool {
        #[cfg(feature = "match-logs")]
        intercept_exclusivity_diag::CALLS.fetch_add(1, Ordering::Relaxed);

        let my_id = self.ctx.player.id;
        let my_lite = MatchPlayerLite {
            id: my_id,
            position: self.ctx.player.position,
            tactical_positions: self.ctx.player.tactical_position.current_position,
        };
        let my_time = time_to_intercept(self.ctx, &my_lite, point);

        let result = !self
            .ctx
            .players()
            .teammates()
            .defenders()
            .any(|d| {
                if d.id == my_id {
                    return false;
                }
                let t = time_to_intercept(self.ctx, &d, point);
                t < my_time || (t == my_time && d.id < my_id)
            });

        #[cfg(feature = "match-logs")]
        if result {
            intercept_exclusivity_diag::WINS.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    /// Most dangerous non-ball-carrier opponent within Help scan radius
    /// that isn't already being engaged. Target for a Help-role defender.
    pub fn find_help_target(&self) -> Option<MatchPlayerLite> {
        let ball_carrier_id = self.ctx.players().opponents().with_ball().next()?.id;
        let own_goal = self.ctx.ball().direction_to_own_goal();
        self.ctx
            .players()
            .opponents()
            .nearby(HELP_SCAN_RADIUS)
            .filter(|opp| opp.id != ball_carrier_id && !self.is_opponent_being_engaged(opp))
            .max_by(|a, b| {
                let score_a = self.calculate_opponent_danger_score(a, own_goal);
                let score_b = self.calculate_opponent_danger_score(b, own_goal);
                score_a.total_cmp(&score_b)
            })
    }

    /// Find an unmarked dangerous opponent that this defender should cover
    /// Returns the most dangerous opponent not already engaged by a teammate — zero allocation
    pub fn find_unmarked_opponent(&self, max_distance: f32) -> Option<MatchPlayerLite> {
        let own_goal = self.ctx.ball().direction_to_own_goal();

        self.ctx
            .players()
            .opponents()
            .nearby(max_distance)
            .filter(|opp| !self.is_opponent_being_engaged(opp))
            .max_by(|a, b| {
                let score_a = self.calculate_opponent_danger_score(a, own_goal);
                let score_b = self.calculate_opponent_danger_score(b, own_goal);
                score_a.total_cmp(&score_b)
            })
    }

    /// Calculate danger score for an opponent.
    ///
    /// 2026-08-30 (Pavel's direct report, unhedged): defenders were
    /// repeatedly abandoning a genuinely dangerous mark (a stationary,
    /// unmarked attacker just outside the box) to go cover a much less
    /// dangerous one (further out, but currently moving and near the
    /// ball) -- e.g. "red #5 next to blue #10... suddenly moves up to
    /// cover blue #6 closer to the central circle, leaving blue #10 all
    /// alone in front of the box." Verified this formula genuinely
    /// produces exactly that swap before touching it (not just a
    /// plausible story): a stationary attacker 140u from goal, ball 80u
    /// away, scored 82.0; a running, ball-side attacker 300u from goal
    /// scored 120.0 -- the distant, active one wins under the old
    /// weights every time this pattern occurs, which is backwards from
    /// real defensive doctrine (a marker's first priority is who is
    /// closest to goal and unmarked; movement/reception-likelihood are
    /// tie-breakers among similarly-positioned opponents, not able to
    /// override a large goal-proximity gap). Rebalanced: distance-to-
    /// goal's max contribution raised 100->200 (divisor 5.0->2.5),
    /// movement and ball-proximity's caps roughly halved (50->20 each)
    /// so distance-to-goal decisively dominates while the other two
    /// still break ties between similarly-dangerous opponents. Re-ran
    /// the same two representative cases with the new weights: 148.0 vs
    /// 112.0 -- the stationary, closer-to-goal attacker now correctly
    /// wins. Reasoned rebalance, category (d) -- no dataset publishes
    /// exact marking-priority weights; the magnitude was chosen to flip
    /// the specific reported failure case, not tuned against a broader
    /// sourced target.
    fn calculate_opponent_danger_score(
        &self,
        opponent: &MatchPlayerLite,
        own_goal: Vector3<f32>,
    ) -> f32 {
        let mut score = 0.0;

        // Distance to our goal (closer = more dangerous) -- the
        // dominant term, see the doc comment above.
        let distance_to_goal = (opponent.position - own_goal).magnitude();
        score += (500.0 - distance_to_goal.min(500.0)) / 2.5;

        // Has the ball
        if opponent.has_ball(self.ctx) {
            score += 150.0;
        }

        // Running toward goal -- a tie-breaker, not able to override a
        // large goal-proximity gap on its own (see doc comment above).
        let velocity = opponent.velocity(self.ctx);
        if velocity.norm() > 2.0 {
            let to_goal = (own_goal - opponent.position).normalize();
            let alignment = velocity.normalize().dot(&to_goal);
            if alignment > 0.5 {
                score += alignment * 20.0;
            }
        }

        // Close to ball (potential receiver) -- same tie-breaker role.
        let ball_distance =
            (opponent.position - self.ctx.tick_context.positions.ball.position).magnitude();
        score += (100.0 - ball_distance.min(100.0)) / 5.0;

        score
    }

    /// Get the number of defenders currently engaging opponents within a distance
    pub fn count_engaging_defenders(&self, radius: f32) -> usize {
        let radius_sq = radius * radius;
        self.ctx
            .players()
            .teammates()
            .defenders()
            .filter(|d| {
                d.id != self.ctx.player.id && {
                    // Check if this defender is close to any opponent
                    self.ctx
                        .players()
                        .opponents()
                        .nearby(ENGAGEMENT_DISTANCE)
                        .any(|opp| (d.position - opp.position).norm_squared() < radius_sq)
                }
            })
            .count()
    }
}
