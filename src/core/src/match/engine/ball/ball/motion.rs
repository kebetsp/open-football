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
        // Safety rail, not a mechanism (2026-10-20): every kick's speed is
        // bounded by the strike model (`engine::kick`, derived from foot
        // speed and the collision, ~37 m/s), and leg contacts are bounded by
        // the same limit. This only catches a future source that forgets to
        // be; the log below fires if it ever does. Measured in true m/s
        // (the old check mixed u/tick with metres/tick).
        const RAIL_MPS: f32 = crate::r#match::engine::kick::KICK_SPEED_LIMIT_MPS * 1.02;

        // Physics constants for realistic ball behavior
        // Air drag: affects aerial balls (proportional to v²)
        const AIR_DRAG_COEFFICIENT: f32 = 0.04; // Reduced for more realistic air resistance

        // Ground friction: see `engine::physics` (roll_decel) — constant
        // rolling deceleration plus v^2 drag. The old 0.985-per-tick
        // multiplier (decel proportional to speed, ~30 m/s^2 on a 20 m/s
        // pass) and the lighter shot-only multiplier (0.997) are gone: the
        // one model below serves passes, clearances, deflections and shots.

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

        let true_mps = crate::r#match::engine::kick::speed_mps(&self.velocity);
        if true_mps > RAIL_MPS {
            #[cfg(feature = "match-logs")]
            crate::match_log_info!("BALLRAIL fired speed_mps={:.1}", true_mps);
            self.velocity = self.velocity * (RAIL_MPS / true_mps);
        }
        let velocity_norm_sq = self.velocity.norm_squared();

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
            // GROUND PHYSICS: constant rolling deceleration plus air drag
            // (v^2) along the direction of travel, applied at any nonzero
            // speed so a slow ball keeps slowing until it stops (the
            // numerical snap below only removes sub-0.01 residue).
            let h_speed = (self.velocity.x * self.velocity.x
                + self.velocity.y * self.velocity.y)
                .sqrt();
            if h_speed > 0.0 {
                let ratio = crate::r#match::engine::physics::roll_step(h_speed) / h_speed;
                self.velocity.x *= ratio;
                self.velocity.y *= ratio;
            }

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

    /// Regression guard: the ball's whole approach to rest follows ONE
    /// model (`physics::roll_decel`): every tick sheds exactly that
    /// deceleration, with no dead zone (a ball that stops slowing) and no
    /// second made-up model near rest (the 2026-08-28 audit bug).
    #[test]
    fn ground_deceleration_follows_one_model_all_the_way_to_rest() {
        use crate::r#match::engine::physics::roll_decel;
        let mut ball = Ball::with_coord(840.0, 545.0);
        ball.velocity = Vector3::new(1.0, 0.0, 0.0);
        ball.position.z = 0.0;

        let mut ticks = 0;
        loop {
            let before = ball.velocity.norm();
            ball.update_velocity();
            let after = ball.velocity.norm();
            ticks += 1;
            if after == 0.0 {
                break;
            }
            assert!(
                ((before - after) - roll_decel(before)).abs() < 1e-5,
                "tick {ticks}: shed {} but model says {}",
                before - after,
                roll_decel(before)
            );
            assert!(ticks < 5000, "ball never came to rest");
        }
        assert!(ticks > 100, "expected many ticks before the ball stops");
    }
}
