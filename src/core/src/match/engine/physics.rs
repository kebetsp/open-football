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

/// The real vertical launch speed (metres/tick, i.e. already in the
/// engine's own velocity storage units) needed for a ball launched from
/// z=0 to reach `height_m` at the peak of its arc, under real ballistics
/// (`v_z = sqrt(2 * g * h)`).
pub fn z_launch_velocity_for_height(height_m: f32) -> f32 {
    let v_z_real_mps = (2.0 * GRAVITY * height_m.max(0.0)).sqrt();
    v_z_real_mps * TICK_SECONDS
}
