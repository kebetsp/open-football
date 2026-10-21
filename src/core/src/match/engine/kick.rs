//! Kick physics (2026-10-20): how fast a struck ball actually leaves the boot.
//!
//! Replaces the old magic caps (`MAX_PASS_VELOCITY`, `MAX_SHOT_VELOCITY`,
//! `MAX_CLEAR_VELOCITY` = 3.2 u/tick, plus uniform "power variation"
//! rolls). The limit is not clamped on afterwards; it follows from the
//! collision itself.
//!
//! Foot-ball collision (Asai/Plagenhoef): a foot of effective mass `M` at
//! speed `v_foot` hits a ball of mass `m` with restitution `e`:
//!
//! ```text
//! v_ball = v_foot * (1 + e) / (1 + m / M) * c
//! ```
//!
//! `c` (<= 1) is contact quality: 1.0 only for a clean hit through the ball's
//! centre, less for an off-centre or mistimed strike. The chain that bounds
//! the ball is therefore
//!
//! * a human cannot swing a foot faster than `FOOT_SPEED_ABS_MAX` (instep
//!   kicks of the hardest strikers, ~27 m/s),
//! * the collision gain is `Q_KICK` (~1.35 for e = 0.56, M = 2.8 kg,
//!   m = 0.43 kg),
//! * contact quality cannot exceed 1,
//!
//! so no kick can leave faster than `Q_KICK * FOOT_SPEED_ABS_MAX` (~37 m/s,
//! about the fastest recorded: Roberto Carlos 137 km/h = 38 m/s). Weaker or
//! tired players have a lower personal foot-speed ceiling; every kick draws
//! its foot speed and contact from truncated normal distributions, so the
//! spread is realistic and the bound holds by construction.
//!
//! Units: callers hold velocities in engine units (x/y in u/tick, z in
//! metres/tick). `MPS_PER_UTK` converts u/tick to m/s (100 ticks/s, 8 u/m).

use crate::r#match::engine::flow::rng::MatchRng;
use crate::r#match::MatchPlayer;
use nalgebra::Vector3;

/// m/s per (u/tick): 100 ticks per second / 8 units per metre.
pub const MPS_PER_UTK: f32 = 12.5;
/// m/s per (m/tick) for the z component.
pub const MPS_PER_MTK: f32 = 100.0;

/// Coefficient of restitution of a boot against a football (0.5..0.6).
const E_FOOT: f32 = 0.56;
/// Effective mass of the swinging leg/foot at impact, kg (literature 2..4).
const M_FOOT: f32 = 2.8;
/// Ball mass, kg.
const M_BALL: f32 = 0.43;
/// Collision gain, ball speed per unit foot speed (~1.35).
pub const Q_KICK: f32 = (1.0 + E_FOOT) / (1.0 + M_BALL / M_FOOT);
/// Fastest instep swing a human produces, m/s.
pub const FOOT_SPEED_ABS_MAX: f32 = 27.5;
/// Theoretical top ball speed of any kick, m/s (~37).
pub const KICK_SPEED_LIMIT_MPS: f32 = Q_KICK * FOOT_SPEED_ABS_MAX;
/// The same limit in engine units (u/tick, ~2.97).
pub const KICK_SPEED_LIMIT_UTK: f32 = KICK_SPEED_LIMIT_MPS / MPS_PER_UTK;

/// A throw (hand) has no boot collision: gain 1, slower arm.
const THROW_ARM_ABS_MAX: f32 = 25.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrikeKind {
    Kick,
    Throw,
}

/// One player's striking ability.
#[derive(Clone, Copy, Debug)]
pub struct Striker {
    /// 0..1: how hard he can swing (strength, technique, long-range power).
    pub power: f32,
    /// 0..1: how cleanly he makes contact and how repeatable the swing is.
    pub quality: f32,
    pub kind: StrikeKind,
}

/// One standard-normal draw (Box-Muller), clamped to +-2.5 sigma.
fn gauss(rng: &MatchRng) -> f32 {
    let u1 = rng.unit_f32().max(1e-7);
    let u2 = rng.unit_f32();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()).clamp(-2.5, 2.5)
}

impl Striker {
    pub fn new(power: f32, quality: f32, kind: StrikeKind) -> Self {
        Striker { power: power.clamp(0.0, 1.0), quality: quality.clamp(0.0, 1.0), kind }
    }

    /// From a player's (fatigue-adjusted) skills. Goalkeepers kick with
    /// their kicking skill; outfielders with strength, technique and
    /// long-range power; accuracy from technique and passing.
    pub fn from_player(player: &MatchPlayer, minute: u32, kind: StrikeKind) -> Self {
        use crate::r#match::player::strategies::players::ops::effective_skill::{
            ActionContext, effective_skill,
        };
        let s = &player.skills;
        let expl = ActionContext::explosive(minute);
        let tech = ActionContext::technical(minute);
        let e = |base: f32, c: ActionContext| (effective_skill(player, base, c) / 20.0).clamp(0.0, 1.0);
        if player.tactical_position.current_position.is_goalkeeper() {
            let kicking = e(s.goalkeeping.kicking, tech);
            let strength = e(s.physical.strength, expl);
            return Striker::new(kicking * 0.6 + strength * 0.4, kicking, kind);
        }
        let strength = e(s.physical.strength, expl);
        let technique = e(s.technical.technique, tech);
        let long_shots = e(s.technical.long_shots, tech);
        let passing = e(s.technical.passing, tech);
        Striker::new(
            strength * 0.45 + technique * 0.30 + long_shots * 0.25,
            technique * 0.6 + passing * 0.4,
            kind,
        )
    }

    fn gain(&self) -> f32 {
        match self.kind {
            StrikeKind::Kick => Q_KICK,
            StrikeKind::Throw => 1.0,
        }
    }

    fn swing_abs_max(&self) -> f32 {
        match self.kind {
            StrikeKind::Kick => FOOT_SPEED_ABS_MAX,
            StrikeKind::Throw => THROW_ARM_ABS_MAX,
        }
    }

    /// His own top swing speed on an average day, m/s.
    fn swing_cap(&self) -> f32 {
        match self.kind {
            StrikeKind::Kick => 18.0 + 8.5 * self.power,
            StrikeKind::Throw => 14.0 + 8.0 * self.power,
        }
    }

    /// The fastest ball this player can strike on an average day, m/s.
    pub fn top_ball_speed_mps(&self) -> f32 {
        self.gain() * self.swing_cap()
    }

    /// Ball speed (m/s) from a strike meant to produce `intended_mps`.
    /// Returns `(speed, saturated)`; `saturated` means the player could not
    /// swing hard enough, so the ball is slower than meant.
    pub fn strike_speed(&self, intended_mps: f32, rng: &MatchRng) -> (f32, bool) {
        let q = self.quality;
        let sigma_c = 0.015 + 0.07 * (1.0 - q);
        // He swings expecting an average-quality contact.
        let c_mean = 1.0 - 0.8 * sigma_c;
        let foot_wanted = intended_mps / (self.gain() * c_mean);
        // Best swing today varies a little around his ceiling, never above
        // what a human can do.
        let cap_today = (self.swing_cap() * (1.0 + 0.03 * gauss(rng)))
            .clamp(self.swing_cap() * 0.9, self.swing_abs_max());
        let saturated = foot_wanted > cap_today * 1.005;
        let foot_target = foot_wanted.min(cap_today);
        // Harder swings are less repeatable.
        let effort = foot_target / cap_today;
        let sigma_f = (0.01 + 0.10 * (1.0 - q)) * (0.6 + 0.4 * effort);
        let foot = (foot_target * (1.0 + sigma_f * gauss(rng))).min(self.swing_abs_max());
        // Contact: sweet spot is the ceiling, mishits only lose speed.
        let c = (1.0 - sigma_c * gauss(rng).abs()).clamp(0.6, 1.0);
        (self.gain() * foot * c, saturated)
    }

    /// Strike a ball meant to leave with engine-unit velocity `v` (x/y in
    /// u/tick, z in m/tick). Direction is kept, speed follows the model.
    /// Returns the achieved velocity and whether the player was maxed out.
    pub fn strike_vector(&self, v: Vector3<f32>, rng: &MatchRng) -> (Vector3<f32>, bool) {
        let intended = speed_mps(&v);
        // A soft touch is not a collision-limited strike.
        if intended < 2.0 {
            return (v, false);
        }
        let (achieved, saturated) = self.strike_speed(intended, rng);
        (v * (achieved / intended), saturated)
    }
}

/// Total speed of an engine-unit velocity, m/s.
pub fn speed_mps(v: &Vector3<f32>) -> f32 {
    let hx = v.x * MPS_PER_UTK;
    let hy = v.y * MPS_PER_UTK;
    let z = v.z * MPS_PER_MTK;
    (hx * hx + hy * hy + z * z).sqrt()
}

/// A lofted ball that cannot be hit as hard as the wanted trajectory
/// needs is not simply slowed (it would drop short): the striker uses the
/// launch angle that lands it at `distance_units` with the speed he has
/// (vacuum range `R = v^2 sin 2θ / g`), taking the root nearest the
/// intended angle. If even 45° falls short, it falls short.
pub fn relaunch_lofted(
    intended: &Vector3<f32>,
    achieved_mps: f32,
    distance_units: f32,
) -> Vector3<f32> {
    let g = crate::r#match::engine::physics::GRAVITY;
    let h_dir = Vector3::new(intended.x, intended.y, 0.0);
    let h_len = h_dir.norm();
    if h_len < 1e-6 || achieved_mps < 1.0 {
        return *intended;
    }
    let h_mps = h_len * MPS_PER_UTK;
    let z_mps = intended.z * MPS_PER_MTK;
    let theta0 = z_mps.atan2(h_mps);
    let x = (g * (distance_units / 8.0) / (achieved_mps * achieved_mps)).min(1.0);
    let low = 0.5 * x.asin();
    let high = std::f32::consts::FRAC_PI_2 - low;
    let theta = if (theta0 - low).abs() <= (theta0 - high).abs() { low } else { high };
    let dir = h_dir / h_len;
    let h = achieved_mps * theta.cos() / MPS_PER_UTK;
    Vector3::new(dir.x * h, dir.y * h, achieved_mps * theta.sin() / MPS_PER_MTK)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn many(s: Striker, intended: f32, n: usize) -> Vec<f32> {
        let rng = MatchRng::from_seed(7);
        (0..n).map(|_| s.strike_speed(intended, &rng).0).collect()
    }

    #[test]
    fn the_limit_follows_from_the_formula() {
        assert!((KICK_SPEED_LIMIT_MPS - 37.1).abs() < 0.3, "{KICK_SPEED_LIMIT_MPS}");
        // Nothing, however badly it is wanted, beats the limit.
        for &(p, q) in &[(1.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0)] {
            let top = many(Striker::new(p, q, StrikeKind::Kick), 500.0, 20_000)
                .into_iter()
                .fold(0.0_f32, f32::max);
            assert!(top <= KICK_SPEED_LIMIT_MPS + 1e-3, "p={p} q={q}: {top}");
        }
    }

    #[test]
    fn weaker_players_top_out_lower_and_sit_in_a_realistic_band() {
        let med = |s: Striker| {
            let mut v = many(s, 500.0, 5_000);
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let weak = med(Striker::new(0.2, 0.4, StrikeKind::Kick));
        let avg = med(Striker::new(0.55, 0.6, StrikeKind::Kick));
        let elite = med(Striker::new(0.95, 0.9, StrikeKind::Kick));
        assert!(weak < avg && avg < elite, "{weak} {avg} {elite}");
        assert!((24.0..30.0).contains(&weak), "{weak}");
        assert!((30.0..36.0).contains(&elite), "{elite}");
    }

    #[test]
    fn ordinary_passes_are_unbiased_and_normally_spread() {
        let s = Striker::new(0.55, 0.7, StrikeKind::Kick);
        let v = many(s, 15.0, 40_000);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        assert!((mean - 15.0).abs() / 15.0 < 0.03, "mean {mean}");
        assert!((0.02..0.08).contains(&(sd / mean)), "relative sd {}", sd / mean);
        // Contact only loses speed, so skew is mildly negative at worst.
        let skew = v.iter().map(|x| ((x - mean) / sd).powi(3)).sum::<f32>() / v.len() as f32;
        assert!(skew.abs() < 0.6, "skew {skew}");
    }

    #[test]
    fn better_technique_is_more_repeatable() {
        let rel = |q: f32| {
            let v = many(Striker::new(0.6, q, StrikeKind::Kick), 18.0, 20_000);
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt() / m
        };
        assert!(rel(0.9) < rel(0.3));
    }

    #[test]
    fn throws_stay_below_a_throw() {
        let top = many(Striker::new(1.0, 1.0, StrikeKind::Throw), 200.0, 10_000)
            .into_iter()
            .fold(0.0_f32, f32::max);
        assert!(top <= THROW_ARM_ABS_MAX + 1e-3, "{top}");
    }

    #[test]
    fn a_maxed_out_lob_relaunches_to_land_at_the_target() {
        // 60 m (480 u) at a 20 degree flat trajectory wants more than 37 m/s;
        // with 30 m/s the relaunch must still reach 60 m in vacuum.
        let want = Vector3::new(3.5, 0.0, 0.45); // ~45 m/s horizontal-ish
        let out = relaunch_lofted(&want, 30.0, 480.0);
        let h = out.x * MPS_PER_UTK;
        let vz = out.z * MPS_PER_MTK;
        let range = 2.0 * h * vz / crate::r#match::engine::physics::GRAVITY;
        assert!((range - 60.0).abs() < 1.0, "range {range}");
        assert!((speed_mps(&out) - 30.0).abs() < 0.1);
    }
}
