//! Centralized ball physics constants — real, sourced, unit-converted once.
//!
//! Every other place in the engine that needs gravity or a height→launch-
//! velocity conversion should import from here rather than redefining its
//! own copy. This file exists specifically because 9 independent copies
//! of the same constant already let a real bug hide in exactly one of
//! them while the others stayed correct: gravity was ~160x too weak for
//! roughly six weeks (invisible because the 2D top-down renderer never
//! draws height) until the 2026-08-02 height-physics rewrite found and
//! fixed it — see CLAUDE.md's decisions log for that entry, and the
//! engine audit's Module 1 finding for why the duplication itself, not
//! just that one bad value, was the real problem.
//!
//! Units: `position.z` is real METRES (confirmed by `flow/goal.rs`'s
//! `GOAL_HEIGHT = 2.44` — a direct, unscaled comparison against
//! `position.z`, corroborated by `ownership.rs`'s PLAYER_HEIGHT=1.8m /
//! PLAYER_JUMP_REACH=3.5m). The engine steps at a fixed 100 ticks/second
//! (10ms per tick — see `MAX_SHOT_VELOCITY`'s own comment in players.rs
//! for the established rate), and position integration is
//! `position += velocity` per tick with no separate dt multiply — so
//! `velocity.z` must be stored as real METRES PER TICK, and a real
//! acceleration `a` [m/s²] changes that already-dt-premultiplied
//! quantity by `a * TICK_SECONDS²` each tick, not `a * TICK_SECONDS`.

/// Real-world gravitational acceleration, m/s².
pub const GRAVITY: f32 = 9.81;

/// Engine tick duration, seconds. 100 ticks/second.
pub const TICK_SECONDS: f32 = 0.01;

/// Per-tick gravity delta applied to `velocity.z` each tick (real
/// metres-per-tick², since `velocity.z` is metres-per-tick — see the
/// module doc above). `9.81 * 0.01² = 0.000981`.
pub const GRAVITY_PER_TICK: f32 = GRAVITY * TICK_SECONDS * TICK_SECONDS;

/// Aerial drag as the integrator applies it: dv/dt = -AIR_DRAG_C * v^2
/// (u/tick^2 per (u/tick)^2), i.e. speed decays as v0 * exp(-c * distance).
/// = 0.04 / 0.43 kg * 0.016 (see `Ball::update_velocity`).
pub const AIR_DRAG_C: f32 = 0.04 / 0.43 * 0.016;

/// Horizontal launch speed (u/tick) that covers `distance` units in
/// `ticks` ticks of flight under `AIR_DRAG_C`: x(T) = ln(1 + c*v0*T)/c.
pub fn air_launch_speed(distance: f32, ticks: f32) -> f32 {
    ((AIR_DRAG_C * distance.max(0.0)).exp() - 1.0) / (AIR_DRAG_C * ticks.max(1.0))
}

/// The real vertical launch speed (metres/tick, i.e. already in the
/// engine's own velocity storage units) needed for a ball launched from
/// z=0 to reach `height_m` at the peak of its arc, under real ballistics
/// (`v_z = sqrt(2 * g * h)`).
pub fn z_launch_velocity_for_height(height_m: f32) -> f32 {
    let v_z_real_mps = (2.0 * GRAVITY * height_m.max(0.0)).sqrt();
    v_z_real_mps * TICK_SECONDS
}

// ---------------------------------------------------------------------
// Ground roll (2026-10-20). One shared model for how a ball on the grass
// slows down, so the motion integrator and everything that PREDICTS
// where a rolling ball ends up (pass weighting, lead estimate, keeper tip
// launch, push-out reach) read the same numbers.
//
// The old model multiplied speed by 0.985 every tick (decel = 1.5 * v per
// second). That is a 30 m/s^2 brake on a 20 m/s pass and almost nothing
// at walking pace: backwards from real rolling friction, which is a
// roughly CONSTANT deceleration plus air drag that grows with v^2.
//
// Real ball on grass: rolling resistance mu_r ~ 0.07..0.15 -> a_roll ~
// 0.7..1.5 m/s^2 (FIFA roll-test: ~4 m/s ball rolls 5-8 m -> ~1.3 m/s^2),
// air drag 0.5*rho*Cd*A/m = 0.5*1.2*0.25*0.038/0.43 ~ 0.0133 1/m.
// Chosen: a_roll = 1.5 m/s^2 (upper end of the band: includes grass
// drag), drag 0.0133 1/m. A 20 m/s ground ball then rolls ~55 m before
// stopping, a 5 m/s ball ~8 m.
//
// Units: speed is u/tick (12.5 m/s per u/tick: 100 ticks/s, 8 u/m),
// decel per tick is u/tick^2 = m/s^2 * 8 / 100^2.
// ---------------------------------------------------------------------

/// Constant rolling deceleration, u/tick^2 (1.5 m/s^2).
pub const ROLL_DECEL: f32 = 1.5 * 8.0 / 10_000.0;
/// Quadratic drag, u/tick^2 per (u/tick)^2 (0.0133 1/m converted).
pub const ROLL_DRAG: f32 = 0.0133 * 12.5 * 12.5 * 8.0 / 10_000.0;

/// Deceleration (u/tick^2) of a ball rolling at `speed` u/tick.
pub fn roll_decel(speed: f32) -> f32 {
    ROLL_DECEL + ROLL_DRAG * speed * speed
}

/// Speed after one tick of rolling (never negative).
pub fn roll_step(speed: f32) -> f32 {
    (speed - roll_decel(speed)).max(0.0)
}

/// Distance (units) covered slowing from `v0` to `v1` u/tick (v1 <= v0).
pub fn roll_distance(v0: f32, v1: f32) -> f32 {
    let (v0, v1) = (v0.max(0.0), v1.clamp(0.0, v0.max(0.0)));
    ((ROLL_DECEL + ROLL_DRAG * v0 * v0) / (ROLL_DECEL + ROLL_DRAG * v1 * v1)).ln()
        / (2.0 * ROLL_DRAG)
}

/// Distance a ball launched at `v0` u/tick rolls before stopping.
pub fn roll_stop_distance(v0: f32) -> f32 {
    roll_distance(v0, 0.0)
}

/// Launch speed (u/tick) that covers `distance` units and still has
/// `arrive_speed` u/tick left at that point.
pub fn roll_launch_speed(distance: f32, arrive_speed: f32) -> f32 {
    let a = ROLL_DECEL + ROLL_DRAG * arrive_speed * arrive_speed;
    ((a * (2.0 * ROLL_DRAG * distance.max(0.0)).exp() - ROLL_DECEL) / ROLL_DRAG)
        .max(0.0)
        .sqrt()
}

/// Ticks to slow from `v0` to `v1` u/tick.
pub fn roll_ticks(v0: f32, v1: f32) -> f32 {
    let k = (ROLL_DRAG / ROLL_DECEL).sqrt();
    let (v0, v1) = (v0.max(0.0), v1.clamp(0.0, v0.max(0.0)));
    ((v0 * k).atan() - (v1 * k).atan()) / (ROLL_DECEL * ROLL_DRAG).sqrt()
}

#[cfg(test)]
mod roll_tests {
    use super::*;

    #[test]
    fn closed_forms_match_tick_stepping() {
        for &v0 in &[0.3_f32, 1.0, 2.0, 3.2] {
            let (mut v, mut x, mut t) = (v0, 0.0_f32, 0u32);
            while v > 0.0 {
                x += v;
                v = roll_step(v);
                t += 1;
            }
            let d = roll_stop_distance(v0);
            assert!((x - d).abs() / d < 0.03, "v0={v0}: stepped {x} vs closed {d}");
            let tt = roll_ticks(v0, 0.0);
            assert!((t as f32 - tt).abs() / tt < 0.03, "v0={v0}: ticks {t} vs {tt}");
        }
    }

    #[test]
    fn launch_speed_arrives_with_requested_speed() {
        for &(d, va) in &[(40.0_f32, 0.1_f32), (100.0, 0.375), (200.0, 0.5)] {
            let v0 = roll_launch_speed(d, va);
            assert!((roll_distance(v0, va) - d).abs() < 0.01 * d);
        }
    }

    #[test]
    fn air_launch_speed_lands_at_target() {
        // step the integrator's own drag law and check the distance covered
        for &(d, t) in &[(100.0_f32, 60.0_f32), (300.0, 150.0), (400.0, 220.0)] {
            let mut v = air_launch_speed(d, t);
            let mut x = 0.0;
            for _ in 0..t as u32 {
                x += v;
                v -= AIR_DRAG_C * v * v;
            }
            assert!((x - d).abs() / d < 0.03, "d={d} t={t}: covered {x}");
        }
    }

    #[test]
    fn real_world_anchors() {
        // 20 m/s (1.6 u/tick) ground ball rolls 45-65 m; 5 m/s rolls 5-9 m.
        let m = |v: f32| roll_stop_distance(v) / 8.0;
        assert!((45.0..65.0).contains(&m(1.6)), "{}", m(1.6));
        assert!((5.0..9.0).contains(&m(0.4)), "{}", m(0.4));
    }
}
