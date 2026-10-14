//! Ball physics and movement: velocity integration with friction /
//! drag / gravity / bounce, and owner tracking that lets the ball follow
//! a possessing player smoothly. Boundary/goal-line resolution lives in
//! `goal.rs`/`restart.rs` — every crossing is now handled exhaustively
//! there (see their module docs), so this file no longer has a fallback
//! "just snap it back in-field" mechanism of its own.

use super::Ball;
use crate::r#match::{GameTickContext, MatchPlayer};
use nalgebra::Vector3;

impl Ball {
    pub fn update_velocity(&mut self) {
        const BALL_MASS: f32 = 0.43;
        // Football bounce retention on grass is ~25-35%. The previous
        // 0.6 produced trampoline bounces where a lofted clearance
        // bounced to 30m+ and stayed airborne (above PLAYER_JUMP_REACH)
        // for 3-5 cycles before a defender could claim. 0.3 keeps the
        // second bounce low enough to reach on the return trip.
        const BOUNCE_COEFFICIENT: f32 = 0.3;
        // Global ball velocity safety cap. Sits above every action-specific
        // cap (shot 3.2, pass 3.2, clearance 7.0) so it never clamps real
        // physics but still catches runaway bug velocities. Clearances are
        // the highest-magnitude legitimate action because they stack
        // meaningful horizontal AND vertical velocity for lofted hoofs.
        const MAX_VELOCITY: f32 = 8.0;

        // Physics constants for realistic ball behavior
        // Air drag: affects aerial balls (proportional to v²)
        const AIR_DRAG_COEFFICIENT: f32 = 0.04; // Reduced for more realistic air resistance

        // Ground friction: affects rolling balls (proportional to v for smooth deceleration)
        // A real football on grass decelerates at about 0.5-1.5 m/s² depending on grass conditions.
        // Engine audit #1 (2026-08-28) corroboration: a peer-reviewed
        // rolling-friction coefficient of μr≈0.07 for a soccer ball on
        // grass ("The Coefficient of Friction of Soccer Balls," ResearchGate,
        // publication 342287745) converts to a≈μr·g≈0.07·9.81≈0.69 m/s² —
        // squarely inside this already-cited 0.5-1.5 m/s² band.
        const GROUND_FRICTION_COEFFICIENT: f32 = 0.015; // Smooth velocity-proportional friction

        // 2026-08 realism-bug fix: a genuine struck shot (`cached_shot_target`
        // is Some for the whole tracked flight, from dispatch to resolution)
        // gets much lighter friction once it lands (z<=0.1) before reaching
        // the goal line. Real driven shots are hit far too hard, and the
        // remaining distance is covered far too quickly, for rolling
        // friction to meaningfully matter the way it legitimately does for
        // a slow, deliberately-placed pass — a shot that skims/lands short
        // of goal should keep arriving with real pace, not visibly crawl
        // in. At the same per-tick decay rate as a normal pass (0.015),
        // a long-range shot that lands ~60-70 ticks from goal loses over
        // half its speed before arrival — this drops that to ~10-15%,
        // leaving the "did it stay elevated the whole way" question (a
        // separate, deliberately NOT-fixed-here issue — see the shot
        // z-velocity comment above) untouched, and only fixes how the
        // ball behaves for whatever portion of the trip it does spend
        // rolling. Ordinary passes/crosses/corners/goal-kicks/dribbles
        // are unaffected — this only engages while a shot is actively
        // being tracked toward goal.
        // No direct source found for "struck shot skid friction" specifically
        // (flagged estimate, category d) — general rolling-football-on-grass
        // deceleration is sourced at 0.5-3 m/s² (FizziQ/StickMan Physics
        // video-analysis writeups), which is what the existing 0.015
        // coefficient above is already meant to represent for a normal,
        // already-slow rolling pass. A ball that's just been struck hard
        // and is skidding (not yet in pure rolling contact) sheds speed
        // more slowly than that — reasoned, not directly sourced.
        // Engine audit #1 (2026-08-28): searched specifically for a
        // sourced sliding-vs-rolling-friction distinction for a struck
        // soccer ball (sports-science/biomechanics literature, coefficient-
        // of-friction studies for soccer balls on turf). Found real,
        // credible sources for GENERAL rolling friction (the constant
        // above, now additionally corroborated by a peer-reviewed
        // μr≈0.07 figure) but none that isolate a skidding-shot-specific
        // coefficient distinct from ordinary rolling — this remains the
        // best available estimate, not a confirmed source, and should
        // stay flagged as such rather than presented as sourced.
        const GROUND_FRICTION_COEFFICIENT_SHOT: f32 = 0.003;

        // CRITICAL: Global velocity sanity check - prevent cosmic-speed balls
        // Check for NaN or infinity and reset to zero
        if self.velocity.x.is_nan()
            || self.velocity.y.is_nan()
            || self.velocity.z.is_nan()
            || self.velocity.x.is_infinite()
            || self.velocity.y.is_infinite()
            || self.velocity.z.is_infinite()
        {
            self.velocity = Vector3::zeros();
            return;
        }

        let mut velocity_norm_sq = self.velocity.norm_squared();

        // Clamp velocity if it exceeds maximum
        if velocity_norm_sq > MAX_VELOCITY * MAX_VELOCITY {
            let velocity_norm = velocity_norm_sq.sqrt();
            self.velocity = self.velocity * (MAX_VELOCITY / velocity_norm);
            velocity_norm_sq = MAX_VELOCITY * MAX_VELOCITY;
        }

        // engine audit #1 (2026-08-28): this used to branch into a
        // COMPLETELY DIFFERENT, made-up deceleration model ("multiply by
        // 0.8 every tick") once speed dropped below a STOPPING_THRESHOLD
        // gate, instead of just continuing the real friction/drag physics
        // below — a ball's final approach to rest doesn't actually change
        // physical laws partway through, it just keeps decelerating under
        // the same forces, asymptotically. The real ground/aerial physics
        // now always runs, unconditionally, regardless of current speed —
        // friction/drag are numerically safe at any magnitude (they only
        // ever multiply velocity down, never divide) — and a separate,
        // purely NUMERICAL cleanup after both branches snaps velocity to
        // exact zero once truly negligible — floating-point hygiene (a
        // real ball under constant proportional friction mathematically
        // never reaches exactly zero), not a second physics model.
        let velocity_norm = velocity_norm_sq.sqrt();
        let is_on_ground = self.position.z <= 0.1;

        if is_on_ground {
            // GROUND PHYSICS: rolling friction proportional to velocity
            // (smooth exponential decay), applied unconditionally at any
            // nonzero speed — a real ball doesn't stop decelerating once
            // it's slow, and the multiply is numerically safe at any
            // magnitude (0 * factor = 0). This USED to be gated behind
            // `horizontal_speed_sq > STOPPING_THRESHOLD²`, which was safe
            // only because the old "nearly stopped" branch above (now
            // removed, see the comment above this function's body) caught
            // everything below that gate with its own separate decay.
            // Removing that branch without also removing this gate left a
            // real gap — a ball with speed between the final numerical
            // snap (~0.01) and the old gate (~0.05) would never decelerate
            // at all, holding constant velocity forever (caught by the
            // `temp_decay_verification` test below, which measured a
            // string of exact 1.0 ratios instead of the constant friction
            // factor).
            let friction_coeff = if self.cached_shot_target.is_some() {
                GROUND_FRICTION_COEFFICIENT_SHOT
            } else {
                GROUND_FRICTION_COEFFICIENT
            };
            let friction_factor = 1.0 - friction_coeff;
            self.velocity.x *= friction_factor;
            self.velocity.y *= friction_factor;

            // Keep ball on ground, but allow upward kicks to take effect
            // (positive z velocity means ball is being kicked into the air)
            if self.velocity.z <= 0.0 {
                self.velocity.z = 0.0;
                self.position.z = 0.0;
            }
        } else {
            // AERIAL PHYSICS: Air drag (proportional to v²) + gravity
            // Air drag is gentler than ground friction for realistic flight

            // Air drag force: F = -0.5 * C * v² * direction
            let air_drag_force = if velocity_norm > 0.1 {
                -AIR_DRAG_COEFFICIENT * velocity_norm * self.velocity
            } else {
                Vector3::zeros()
            };

            // 2026-08 realism-bug fix (full height-physics rewrite):
            // gravity is split from air drag and given its own,
            // dimensionally-correct per-tick scaling.
            //
            // `position.z` is REAL METERS, not the 8-units/meter scale
            // used for x/y — confirmed definitively by `flow/goal.rs`'s
            // LIVE, actively-used goal detection: `GOAL_HEIGHT: f32 =
            // 2.44; // Crossbar height in meters`, compared directly
            // against `position.z` with no scaling, and independently
            // corroborated by `ownership.rs`'s PLAYER_HEIGHT=1.8m /
            // PLAYER_JUMP_REACH=3.5m / MAX_BALL_HEIGHT=4.0m (also
            // direct, unscaled comparisons) and every defender/GK
            // reach threshold in the 1.5-2.8m range across the engine.
            //
            // Since `move_to()` does `position.z += velocity.z` with
            // NO separate dt multiply, `velocity.z` must be stored as
            // real METERS PER TICK (v_z_real_mps * TICK_SECONDS) for
            // that accumulation to produce correct real heights. A
            // real acceleration `a` [m/s²] changes real velocity by
            // `a*dt` each tick; the STORED (already dt-premultiplied)
            // velocity therefore changes by `a*dt²` per tick — NOT the
            // flat `a*0.016` this line previously used, which was off
            // by ~160x (0.016 vs the correct TICK_SECONDS²=0.0001 at
            // this engine's real, already-established 100-tick/s rate
            // — see MAX_SHOT_VELOCITY's comment in players.rs for that
            // rate) and was also missing the meters-vs-8-units-per-
            // meter distinction entirely. Combined, this made a
            // corner's launch (LOFTED_DELIVERY_MAX_Z_VELOCITY=2.9)
            // produce a measured, verified real apex of ~26 metres —
            // taller than most stadiums — invisible only because the
            // 2D top-down renderer never draws height at all.
            //
            // Air drag's existing per-tick scaling (0.016) is
            // deliberately left UNCHANGED here — it was never part of
            // this bug (it acts on x/y, which are already correctly
            // on the tick-native game-unit scale) and touching it is
            // out of scope for the height-physics fix; its effect on
            // z is negligible now that z's own magnitude is correctly
            // tiny (real vertical launch speeds of a few m/s, stored
            // as fractions of a metre per 10ms tick).
            let air_drag_accel = air_drag_force / BALL_MASS;
            self.velocity.x += air_drag_accel.x * 0.016;
            self.velocity.y += air_drag_accel.y * 0.016;
            self.velocity.z +=
                air_drag_accel.z * 0.016 - crate::r#match::engine::physics::GRAVITY_PER_TICK;
        }

        // Numerical cleanup only, not a second physics model (see comment
        // above): a real ball under constant proportional friction never
        // mathematically reaches exact zero, so once negligible, snap it.
        if self.velocity.norm_squared() < 0.0001 {
            // 0.01^2
            self.velocity = Vector3::zeros();
            self.position.z = 0.0;
        }

        // Check ground collision and bounce
        if self.position.z <= 0.0 && self.velocity.z < 0.0 {
            // Ball hit the ground
            self.position.z = 0.0;
            self.velocity.z = -self.velocity.z * BOUNCE_COEFFICIENT;

            // Apply some horizontal speed loss on bounce (realistic)
            self.velocity.x *= 0.95;
            self.velocity.y *= 0.95;

            // If bounce is too small, stop vertical movement. 2026-08
            // height-physics rewrite: 0.3 was calibrated for the OLD,
            // ~20-160x-too-large z-velocity scale — under real metres-
            // per-tick storage every legitimate bounce (even off a high
            // corner/clearance) is well under 0.3, so this threshold
            // used to kill every single bounce outright. 0.01 (real
            // ~1 m/s) only cuts off truly negligible residual bounces.
            if self.velocity.z.abs() < 0.01 {
                self.velocity.z = 0.0;
            }
        }
    }

    pub(super) fn move_to(&mut self, tick_context: &GameTickContext) {
        // Clear notified players only when ball state changes significantly:
        // 1. Ball starts moving (not stopped anymore)
        // 2. Ball has an owner (claimed)
        // Maximum distance owner can be from ball - must match deadlock claim distances
        // This allows deadlock resolution while preventing truly absurd teleports
        const MAX_OWNER_TELEPORT_DISTANCE: f32 = 15.0;
        const MAX_OWNER_TELEPORT_DISTANCE_SQUARED: f32 =
            MAX_OWNER_TELEPORT_DISTANCE * MAX_OWNER_TELEPORT_DISTANCE;

        // Ball moves toward owner at this speed (units/tick) instead of teleporting
        const BALL_TRACK_SPEED: f32 = 1.5;
        // Snap to owner if within this distance (avoids jitter)
        const SNAP_DISTANCE: f32 = 2.0;
        const SNAP_DISTANCE_SQUARED: f32 = SNAP_DISTANCE * SNAP_DISTANCE;

        let has_owner = self.current_owner.is_some();

        // Clear notifications when ball is no longer in a "take ball" scenario
        // Use a higher threshold to avoid clearing notifications set by try_notify_standing_ball
        // which uses is_ball_stopped_on_field (velocity < 2.5)
        const CLEAR_NOTIFICATION_VELOCITY: f32 = 3.0;
        let is_clearly_moving = self.velocity.norm() > CLEAR_NOTIFICATION_VELOCITY;
        if (is_clearly_moving || has_owner) && !self.take_ball_notified_players.is_empty() {
            self.take_ball_notified_players.clear();
        }

        if let Some(owner_player_id) = self.current_owner {
            let owner_position = tick_context.positions.players.position(owner_player_id);

            let dx = owner_position.x - self.position.x;
            let dy = owner_position.y - self.position.y;
            let distance_squared = dx * dx + dy * dy;

            if distance_squared <= MAX_OWNER_TELEPORT_DISTANCE_SQUARED {
                if distance_squared <= SNAP_DISTANCE_SQUARED {
                    // Close enough - snap to owner
                    crate::snap_tag!("owner_follow_a");
                    self.position = owner_position;
                    self.position.z = 0.0;
                    self.velocity = Vector3::zeros();
                } else {
                    // Move ball toward owner smoothly instead of teleporting
                    let distance = distance_squared.sqrt();
                    let dir_x = dx / distance;
                    let dir_y = dy / distance;
                    self.position.x += dir_x * BALL_TRACK_SPEED;
                    self.position.y += dir_y * BALL_TRACK_SPEED;
                    self.position.z = 0.0;
                    self.velocity = Vector3::zeros();
                }
            } else {
                // Owner is too far - this shouldn't happen but is a safety net
                // Clear ownership and let ball move naturally
                self.previous_owner = self.current_owner;
                self.current_owner = None;
                self.ownership_duration = 0;

                // Move ball normally
                self.position.x += self.velocity.x;
                self.position.y += self.velocity.y;
                self.position.z += self.velocity.z;

                if self.position.z < 0.0 {
                    self.position.z = 0.0;
                }
            }
        } else {
            self.position.x += self.velocity.x;
            self.position.y += self.velocity.y;
            self.position.z += self.velocity.z;

            // Ensure ball doesn't go below ground
            if self.position.z < 0.0 {
                self.position.z = 0.0;
            }
        }
    }

    pub(super) fn move_to_with_players(&mut self, players: &[MatchPlayer]) {
        const MAX_OWNER_TELEPORT_DISTANCE_SQUARED: f32 = 15.0 * 15.0;
        const BALL_TRACK_SPEED: f32 = 1.5;
        const SNAP_DISTANCE_SQUARED: f32 = 2.0 * 2.0;

        if let Some(owner_id) = self.current_owner {
            if let Some(owner) = players.iter().find(|p| p.id == owner_id) {
                let dx = owner.position.x - self.position.x;
                let dy = owner.position.y - self.position.y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= MAX_OWNER_TELEPORT_DISTANCE_SQUARED {
                    if dist_sq <= SNAP_DISTANCE_SQUARED {
                        crate::snap_tag!("owner_follow_b");
                        self.position = owner.position;
                        self.position.z = 0.0;
                        self.velocity = Vector3::zeros();
                    } else {
                        let dist = dist_sq.sqrt();
                        self.position.x += (dx / dist) * BALL_TRACK_SPEED;
                        self.position.y += (dy / dist) * BALL_TRACK_SPEED;
                        self.position.z = 0.0;
                        self.velocity = Vector3::zeros();
                    }
                } else {
                    self.previous_owner = self.current_owner;
                    self.current_owner = None;
                    self.ownership_duration = 0;
                    self.apply_movement();
                }
            } else {
                self.apply_movement();
            }
        } else {
            self.apply_movement();
        }
    }

    // `check_boundary_collision` (the blind "snap 10 units back in-field,
    // zero velocity" fallback) was removed 2026-08-28 (engine audit
    // Module 2 rewrite) — every boundary crossing now resolves
    // exhaustively and geometrically via `check_goal`/`check_over_goal`/
    // `check_wide_of_goal`/`check_throw_in`, each with progressive
    // fallbacks so a real restart is always awarded (per Pavel's sign-
    // off: "always resolve a real outcome," not a silent teleport-and-
    // freeze). See `goal.rs`'s module doc for the full design.
}

#[cfg(test)]
mod decay_regression_tests {
    use super::*;

    /// Engine audit #1 (2026-08-28): the ball's final approach to rest
    /// used to switch to a completely different, made-up deceleration
    /// model ("*0.8 every tick") once speed dropped below a threshold,
    /// instead of continuing the same real friction physics all the way
    /// down. Regression guard: the tick-over-tick deceleration RATIO
    /// should stay constant (the friction factor) for the whole approach
    /// to rest — a discontinuity here means a second physics model (or a
    /// dead zone with no deceleration at all — the actual bug this test
    /// caught during the fix, when the friction application was still
    /// gated by a since-removed low-speed threshold) has crept back in.
    #[test]
    fn ground_deceleration_ratio_stays_constant_all_the_way_to_rest() {
        let mut ball = Ball::with_coord(840.0, 545.0);
        ball.velocity = Vector3::new(0.3, 0.0, 0.0);
        ball.position.z = 0.0;

        let mut prev_norm = ball.velocity.norm();
        let mut ratios = Vec::new();
        for _ in 0..300 {
            ball.update_velocity();
            let norm = ball.velocity.norm();
            if prev_norm > 0.0001 && norm > 0.0 {
                ratios.push(norm / prev_norm);
            }
            prev_norm = norm;
            if norm == 0.0 {
                break;
            }
        }
        for r in &ratios {
            assert!(
                (*r - 0.985).abs() < 0.001,
                "deceleration ratio {} deviated from the constant friction factor 0.985 — a second physics model or a no-friction dead zone is firing",
                r
            );
        }
        assert!(ratios.len() > 10, "expected many ticks before the ball stops");
    }
}
