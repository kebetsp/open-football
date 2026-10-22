//! Manager directives as INPUTS to the new decision domains.
//!
//! Until 2026-10-22 a player carrying a behavioural directive, a link
//! target or a supply target was taken OUT of the new on-ball (domain 2/4)
//! and off-ball (domain 3) systems and handed back to the July state
//! machine code, which was never retuned. The instruction-system audit
//! (`docs/instruction-system-audit-2026-10.md`) measured the result: shoot
//! on sight took 0.03 shots per match (default 1.5), lay-off held the ball
//! longer, link-with made the pair pass each other less. A supply target is
//! set on EVERY teammate of the fed player, so that bypass demoted the whole
//! team.
//!
//! Now a directive only tilts the new systems' own values. These helpers
//! are pure reads of the player's directive fields and a few ball facts;
//! the domains call them at the points where they compare options.
//!
//! `BylineAndCross` is the one directive that still bypasses: its July
//! state-machine implementation (commitment window, channel-locked carry,
//! release past the penalty spot) measured 1.6 -> 10 crosses per match and
//! has no equivalent in the new domains.

use crate::club::player::traits::BehavioralDirective as D;
use crate::r#match::player::PlayerSide;
use crate::r#match::{MatchPlayer, StateProcessingContext};
use nalgebra::Vector3;

/// A reception is "first touch" for this many ticks (0.45 s).
const FIRST_TOUCH_TICKS: u32 = 45;
/// A pass longer than this counts as a long ball for lay_off_on_long_ball
/// and for the supply bias (matches `is_long_reception`'s directive gate).
const LONG_BALL_UNITS: f32 = 140.0;

/// True when the player's behaviour is governed by the July state-machine
/// implementation of a directive instead of the new domains.
pub fn bypasses_new_systems(p: &MatchPlayer) -> bool {
    matches!(p.behavioral_directive, Some(D::BylineAndCross))
}

/// Multipliers/offsets the on-ball domains apply at their comparison
/// points. `NEUTRAL` changes nothing.
#[derive(Clone, Copy)]
pub struct OnBallBias {
    pub shoot_on_sight: bool,
    /// Multiplier on carry/dribble values.
    pub carry_mul: f32,
    /// Hold is not an option while a first-touch lay-off is active.
    pub no_hold: bool,
    /// Pass value is `v * pass_mul + pass_add` while a lay-off is active.
    pub pass_mul: f32,
    pub pass_add: f32,
    /// StayWideNoCutInside: carries/dribbles may not turn inward.
    pub stay_wide: bool,
}

impl OnBallBias {
    pub const NEUTRAL: OnBallBias = OnBallBias {
        shoot_on_sight: false,
        carry_mul: 1.0,
        no_hold: false,
        pass_mul: 1.0,
        pass_add: 0.0,
        stay_wide: false,
    };

    /// Shoot-on-sight: a shot with a clear sight of goal is taken from
    /// range. `open` = no opponent within a body length.
    pub fn shot(&self, v: f32, open: bool) -> f32 {
        if self.shoot_on_sight && open {
            v.max(0.0) * 3.0 + 0.45
        } else {
            v
        }
    }

    /// Shoot-on-sight keeps a shot a live option a little beyond the
    /// domain's normal shooting radius.
    pub fn range_mul(&self) -> f32 {
        if self.shoot_on_sight {
            1.25
        } else {
            1.0
        }
    }

    pub fn pass(&self, v: f32) -> f32 {
        v * self.pass_mul + if self.pass_mul != 1.0 { self.pass_add } else { 0.0 }
    }
}

pub fn on_ball(ctx: &StateProcessingContext) -> OnBallBias {
    let p = ctx.player;
    let mut b = OnBallBias::NEUTRAL;
    let first_touch = ctx.tick_context.ball.ownership_duration < FIRST_TOUCH_TICKS;
    match p.behavioral_directive {
        Some(D::ShootOnSight) => b.shoot_on_sight = true,
        Some(D::LayItOffFirstTouch) if first_touch => {
            b.no_hold = true;
            b.carry_mul = 0.2;
            b.pass_mul = 1.5;
            b.pass_add = 0.15;
        }
        Some(D::LayOffOnLongBall)
            if first_touch && ctx.ball().is_long_reception(LONG_BALL_UNITS) =>
        {
            b.no_hold = true;
            b.carry_mul = 0.2;
            b.pass_mul = 1.5;
            b.pass_add = 0.15;
        }
        Some(D::CarryWhenSpaceAhead) => b.carry_mul = 1.6,
        Some(D::StayWideNoCutInside) => b.stay_wide = true,
        _ => {}
    }
    b
}

/// Receiver-specific pass multiplier: link partner, supply target, and the
/// team-wide attack-side preference. `pass_dist` in engine units.
pub fn receiver_mult(
    ctx: &StateProcessingContext,
    receiver_id: u32,
    receiver_y: f32,
    pass_dist: f32,
) -> f32 {
    let p = ctx.player;
    let mut m = 1.0;
    if p.link_target == Some(receiver_id) {
        m *= 1.2;
    }
    if p.supply_target == Some(receiver_id) {
        m *= if pass_dist > LONG_BALL_UNITS { 1.25 } else { 1.05 };
    }
    if let Some(bias) = p.attack_bias {
        let h = ctx.context.field_size.height as f32;
        let third = h / 3.0;
        let raw_lane: i8 = if receiver_y < third {
            -1
        } else if receiver_y > h - third {
            1
        } else {
            0
        };
        let lane = match p.side.unwrap_or(PlayerSide::Left) {
            PlayerSide::Left => raw_lane,
            PlayerSide::Right => -raw_lane,
        };
        m *= if bias == 0 {
            if lane == 0 { 1.25 } else { 0.9 }
        } else if lane == bias {
            1.35
        } else if lane == -bias {
            0.75
        } else {
            1.0
        };
    }
    m
}

/// Opposing "block passes into X" assignments: when a defender whose
/// `intercept_target` is the receiver stands near the passing lane, the
/// pass is far less attractive for the carrier.
pub fn intercept_mult(
    ctx: &StateProcessingContext,
    from: Vector3<f32>,
    receiver_id: u32,
    receiver_pos: Vector3<f32>,
) -> f32 {
    let lane = receiver_pos - from;
    let len2 = lane.magnitude_squared();
    if len2 < 1.0 {
        return 1.0;
    }
    for o in ctx.players().opponents().nearby(400.0) {
        let Some(full) = ctx.context.players.by_id(o.id) else {
            continue;
        };
        if full.intercept_target != Some(receiver_id) {
            continue;
        }
        let t = ((o.position - from).dot(&lane) / len2).clamp(0.0, 1.0);
        let closest = from + lane * t;
        // Near the lane, or shadowing the receiver.
        if (o.position - closest).magnitude() < 56.0
            || (o.position - receiver_pos).magnitude() < 56.0
        {
            return 0.35;
        }
    }
    1.0
}

/// StayWideNoCutInside: keep a carry/dribble target on the carrier's own
/// lateral line — advance along the touchline instead of angling inward.
pub fn keep_wide(from: Vector3<f32>, target: Vector3<f32>, field_height: f32) -> Vector3<f32> {
    let mid = field_height * 0.5;
    let inward = -(from.y - mid).signum() * (target.y - from.y);
    if inward > 0.0 {
        Vector3::new(target.x, from.y, target.z)
    } else {
        target
    }
}
