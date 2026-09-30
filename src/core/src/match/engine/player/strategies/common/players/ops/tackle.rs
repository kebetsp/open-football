//! Unified tackle system (2026-09-29, Pavel's from-scratch redesign).
//!
//! One shared implementation for defenders, midfielders and forwards
//! that replaces the three per-role tackling bodies, the old
//! tackle/contain/concede-out value comparison and the two duplicated
//! "desperate slide" copies.
//!
//! Layers:
//!   1. Standing challenge — a short lunge (small reach, soft strike).
//!   2. Slide — a committed, long-reach, hard-strike lunge.
//!   3. Box risk — not a rule: a foul costs a penalty's worth of value
//!      inside the box, so slides there only win the comparison when
//!      the alternative is close to a certain goal.
//!
//! The ball's fate is decided by ONE physical contact model
//! ([`resolve_contact`]): the leg is a capsule that extends from the
//! defender along the slide axis; the ball is a moving point. The
//! FIRST tick at which the ball touches the capsule fixes the contact
//! normal (from the closest point on the leg), the ball then reflects
//! off the leg's surface (with energy loss and the leg's own velocity)
//! and a foot-strike term steers it toward the defender's intent in
//! proportion to how centred the contact was. Timing therefore decides
//! everything and nothing is a dice table:
//!   * barely touched (graze)       -> tiny turn, ball keeps its line
//!   * a little late (edge of tip)  -> large turn, can go back / wide
//!   * well late (side of the leg)  -> ball thrown sideways across it
//!   * on time (centred tip)        -> goes where he aimed
//!   * leg reaches the man first    -> foul (no separate foul roll for
//!                                     slides)
//!
//! Units: positions/velocities in engine units (8u = 1m, u/tick, 100
//! ticks/s); `velocity.z` in metres/tick (see `engine::physics`).

use crate::r#match::engine::physics::z_launch_velocity_for_height;
use crate::r#match::events::Event;
use crate::r#match::player::events::{FoulSeverity, PlayerEvent};
use crate::r#match::player::strategies::common::players::ops::defensive_value as dv;
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::{MatchPlayerLite, PlayerSide, StateProcessingContext, SteeringBehavior};
use nalgebra::Vector3;

/// Half-thickness of the "leg + ball" capsule (leg ~0.10m + ball ~0.11m
/// radius at 8u/m). Reasoned estimate, category (d).
pub const LEG_CONTACT_RADIUS: f32 = 1.9;
/// Body (torso) contact radius for the foul check (~0.35m).
pub const BODY_CONTACT_RADIUS: f32 = 2.8;
/// Search horizon: 0.6 s.
pub const CONTACT_HORIZON_TICKS: u32 = 60;
/// Hard cap on any deflected ball's speed (MAX_SHOT_VELOCITY).
pub const MAX_DEFLECT_SPEED: f32 = 3.2;
/// Max deflection apex height (m) — a slide lifts, it doesn't lob.
pub const MAX_DEFLECT_HEIGHT_M: f32 = 2.5;
/// Ball must be this many ticks ahead of the body contact for a clean
/// "ball first" ruling (otherwise it is a simultaneous/man-first foul).
pub const BALL_FIRST_SLACK_TICKS: f32 = 2.0;

#[derive(Debug, Clone, Copy)]
pub struct ContactInput {
    /// Defender position (leg root).
    pub defender: Vector3<f32>,
    /// Defender's own velocity (u/tick): the leg root keeps moving
    /// while the leg extends — a lunge builds on the run.
    pub defender_vel: Vector3<f32>,
    /// Ball position at decision time.
    pub ball: Vector3<f32>,
    /// Incoming ball velocity (u/tick, xy used).
    pub ball_vel: Vector3<f32>,
    /// Carrier body position (for man-vs-ball ordering).
    pub carrier: Vector3<f32>,
    /// Unit slide axis (xy).
    pub axis: Vector3<f32>,
    /// Leg-tip speed while extending (u/tick).
    pub leg_speed: f32,
    /// Maximum leg length (u).
    pub reach: f32,
    /// Ticks before the leg starts extending (wind-up / commit delay).
    pub windup_ticks: f32,
    /// Unit direction the defender wants the ball to go (xy).
    pub intent_dir: Vector3<f32>,
    /// Speed he wants to give it (u/tick).
    pub intent_speed: f32,
    /// Surface restitution along the normal (0 = dead, 1 = elastic).
    pub restitution: f32,
    /// 0..1 — how well he can steer the ball on a centred touch.
    pub control_skill: f32,
    /// Standard-normal-ish sample in [-1, 1] for surface jitter.
    pub jitter: f32,
    /// Jitter scale in radians (shrinks with skill).
    pub jitter_sigma: f32,
    /// Uniform [0, 1) sample for the loft angle.
    pub loft_roll: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ContactOutcome {
    /// Leg touched the ball.
    pub touched: bool,
    /// Tick of ball contact.
    pub ball_tick: Option<u32>,
    /// Leg reached the carrier's body.
    pub body_contact: bool,
    /// Foul: the man was hit first / without a clean ball-first touch.
    pub foul: bool,
    /// Ball velocity after contact (xy u/tick, z metres/tick). Equal to
    /// the incoming velocity when `touched` is false.
    pub velocity: Vector3<f32>,
    /// 0..1: how centred a tip contact was (1 = dead centre).
    pub control: f32,
    /// Fraction along the leg where contact happened (1 = tip).
    pub leg_fraction: f32,
}

fn xy(v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v.x, v.y, 0.0)
}

fn norm_or(v: Vector3<f32>, fallback: Vector3<f32>) -> Vector3<f32> {
    let m = v.magnitude();
    if m > 1e-5 {
        v / m
    } else {
        fallback
    }
}

/// Closest point on segment [a, b] to p (xy).
fn closest_on_segment(a: Vector3<f32>, b: Vector3<f32>, p: Vector3<f32>) -> Vector3<f32> {
    let ab = b - a;
    let len2 = ab.x * ab.x + ab.y * ab.y;
    if len2 < 1e-6 {
        return a;
    }
    let t = (((p - a).x * ab.x + (p - a).y * ab.y) / len2).clamp(0.0, 1.0);
    a + ab * t
}

fn rotate(v: Vector3<f32>, angle: f32) -> Vector3<f32> {
    let (s, c) = angle.sin_cos();
    Vector3::new(v.x * c - v.y * s, v.x * s + v.y * c, 0.0)
}

fn launch_vz(height_m: f32) -> f32 {
    z_launch_velocity_for_height(height_m)
}

/// Length of the leg (u) at tick `t`.
fn leg_length_at(input: &ContactInput, t: f32) -> f32 {
    ((t - input.windup_ticks).max(0.0) * input.leg_speed).min(input.reach)
}

pub fn resolve_contact(input: &ContactInput) -> ContactOutcome {
    let s = norm_or(xy(input.axis), Vector3::new(1.0, 0.0, 0.0));
    let s_perp = Vector3::new(-s.y, s.x, 0.0);
    let v_in = xy(input.ball_vel);
    let d_start = xy(input.defender);

    let mut ball_tick: Option<u32> = None;
    let mut body_tick: Option<u32> = None;

    let mut outcome = ContactOutcome {
        touched: false,
        ball_tick: None,
        body_contact: false,
        foul: false,
        velocity: input.ball_vel,
        control: 0.0,
        leg_fraction: 0.0,
    };

    for t in 0..=CONTACT_HORIZON_TICKS {
        let tf = t as f32;
        let leg_len = leg_length_at(input, tf);
        let d0 = d_start + xy(input.defender_vel) * tf.min(20.0);
        let tip = d0 + s * leg_len;
        let ball_t = xy(input.ball) + v_in * tf;
        let body_t = xy(input.carrier) + v_in * tf;

        if body_tick.is_none() && leg_len > 0.0 {
            let cb = closest_on_segment(d0, tip, body_t);
            if (body_t - cb).magnitude() <= BODY_CONTACT_RADIUS {
                body_tick = Some(t);
            }
        }

        let c = closest_on_segment(d0, tip, ball_t);
        let off = ball_t - c;
        let dist = off.magnitude();
        if dist > LEG_CONTACT_RADIUS {
            continue;
        }

        // Leg-surface velocity at the contact point: only while the
        // leg is still extending, and proportional to how far out
        // along it the contact is.
        let extending = tf > input.windup_ticks && leg_len < input.reach - 1e-3;
        let frac = if leg_len > 1e-3 {
            ((c - d0).magnitude() / leg_len).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let v_leg = if extending {
            s * (input.leg_speed * frac)
        } else {
            Vector3::zeros()
        };

        // Surface normal: from the closest point on the leg to the
        // ball; head-on (dist ~ 0) falls back to the approach line.
        let v_rel = v_in - v_leg;
        let n = norm_or(off, norm_or(-v_rel, -s));
        // Only an approaching ball registers contact.
        if v_rel.x * n.x + v_rel.y * n.y >= 0.0 {
            continue;
        }

        // Tip contact = ball beyond (or at) the end of the leg.
        let along = (ball_t - d0).x * s.x + (ball_t - d0).y * s.y;
        let perp = ((ball_t - d0).x * s_perp.x + (ball_t - d0).y * s_perp.y).abs();
        let tip_contact = leg_len > 1e-3 && along >= leg_len - LEG_CONTACT_RADIUS * 0.5;
        let control = if tip_contact {
            (1.0 - perp / LEG_CONTACT_RADIUS).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Foot steering: a centred touch lets him turn the surface
        // toward his intent.
        let steer = (control * input.control_skill).clamp(0.0, 1.0);
        let intent = norm_or(xy(input.intent_dir), -s);
        let mut n_eff = norm_or(n * (1.0 - steer) + intent * steer, n);
        n_eff = rotate(n_eff, input.jitter * input.jitter_sigma);
        // The steered/jittered normal must still oppose the ball;
        // otherwise fall back to the physical one.
        if v_rel.x * n_eff.x + v_rel.y * n_eff.y >= 0.0 {
            n_eff = n;
        }

        let vn = v_rel.x * n_eff.x + v_rel.y * n_eff.y; // < 0
        let mut v_out = v_leg + v_rel - n_eff * ((1.0 + input.restitution) * vn);
        // Foot-strike contribution (aimed, scaled by control).
        v_out += intent * (input.intent_speed * steer);

        let speed = v_out.magnitude();
        if speed > MAX_DEFLECT_SPEED {
            v_out = v_out * (MAX_DEFLECT_SPEED / speed);
        }

        // Height: a leg at ground level always catches the lower half
        // of the ball. vz (m/tick) from the normal component
        // (u/tick -> m/tick via /8) times a lift angle (10°..~35°),
        // plus a small strike lift; capped at the max deflect height.
        let lift_angle = 0.17 + 0.45 * input.loft_roll;
        let vz = ((-vn / 8.0) * lift_angle.tan() + steer * input.intent_speed / 8.0 * 0.15)
            .clamp(0.0, launch_vz(MAX_DEFLECT_HEIGHT_M));

        ball_tick = Some(t);
        outcome.touched = true;
        outcome.ball_tick = ball_tick;
        outcome.velocity = Vector3::new(v_out.x, v_out.y, vz);
        outcome.control = control;
        outcome.leg_fraction = if tip_contact { 1.0 } else { frac };
        break;
    }

    // Man-vs-ball ordering. Once the ball is struck the leg can still
    // follow through into the man; only a body hit that comes BEFORE
    // (or without) a clean ball touch is a foul.
    if let Some(bt) = body_tick {
        outcome.body_contact = true;
        outcome.foul = match ball_tick {
            None => true,
            Some(ct) => (bt as f32) + BALL_FIRST_SLACK_TICKS <= ct as f32,
        };
    }

    outcome
}


// ---------------------------------------------------------------------
// Decision + execution layer
// ---------------------------------------------------------------------

/// Distance inside which a defender is "engaged" with the carrier and
/// the hold / challenge / slide decision runs. One shared constant: it
/// exceeds the jockey standoff so a jockeying defender can lunge, and
/// every state that transitions into Tackling uses it (see
/// `DefensiveOperationsImpl::TACKLE_ENGAGEMENT_DISTANCE`).
pub const ENGAGE_DISTANCE: f32 = 24.0;
/// Standing challenge: a lunge from ~2.75m.
pub const LUNGE_REACH: f32 = 22.0;
/// Committed slide: ~4.25m.
pub const SLIDE_REACH: f32 = 34.0;
/// Below this a slide is pointless — a standing challenge covers it.
pub const SLIDE_MIN_DISTANCE: f32 = 8.0;
/// Where a holding defender stands off the carrier (~2m).
pub const JOCKEY_STANDOFF: f32 = 16.0;

/// Role-agnostic skill inputs, built by each tackling state from its
/// own skill profile so the three roles share one implementation.
#[derive(Debug, Clone, Copy)]
pub struct DuelSkills {
    /// 0..1 role-appropriate tackle composite.
    pub tackle: f32,
    /// 0..1 discipline (suppresses fouls).
    pub discipline: f32,
    /// 0..1 fatigue multiplier (1 = fresh).
    pub condition: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TackleKind {
    Challenge,
    Slide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TackleChoice {
    Hold,
    Act(TackleKind),
}

/// What the state should do this tick.
pub enum DuelAction {
    /// Keep jockeying (stay in the Tackling state so standoff time
    /// accumulates).
    Hold,
    /// A tackle was attempted; `events` may be empty (clean miss).
    Act(Vec<Event>),
}

fn tri(rng_a: f32, rng_b: f32) -> f32 {
    // Triangular sample in [-1, 1] from two uniforms.
    rng_a + rng_b - 1.0
}

fn own_box_contains_ball(ctx: &StateProcessingContext) -> bool {
    ctx.context
        .penalty_area(ctx.player.side == Some(PlayerSide::Left))
        .contains(&ctx.tick_context.positions.ball.position)
}

/// Build the contact input for a tackle of `kind` against `opponent`
/// at the current instant. `windup`, `aim_err`, `carrier_turn`,
/// `jitter`, `loft` are the stochastic parts (0.0 for a deterministic
/// prediction).
fn build_input(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    skills: DuelSkills,
    kind: TackleKind,
    windup: f32,
    aim_err: f32,
    dodge: Vector3<f32>,
    jitter: f32,
    loft: f32,
) -> ContactInput {
    let ball = ctx.tick_context.positions.ball.position;
    let defender = ctx.player.position;
    // The carrier keeps moving — and may sidestep the lunge (`dodge`, a
    // lateral velocity added to his own; zero in predictions).
    let v_base = xy(opponent.velocity(ctx));
    let v_in = v_base + xy(dodge);
    let max_speed = ctx
        .player
        .skills
        .max_speed_with_condition(ctx.player.player_attributes.condition)
        .max(0.2);
    let (leg_speed, reach) = match kind {
        // Leg-tip speed RELATIVE to the body: a lunge/slide is far
        // quicker than a running stride (reasoned estimate, category d).
        TackleKind::Slide => (1.0 + max_speed, SLIDE_REACH),
        TackleKind::Challenge => (0.7 + max_speed, LUNGE_REACH),
    };

    // Aim at the point where the leg tip can meet the ball (constant
    // velocity assumption); if it can't within reach, aim along the
    // ball's path at full reach.
    let mut t = 5.0f32;
    for _ in 0..30 {
        let aim = xy(ball) + v_base * t;
        t = ((aim - xy(defender)).magnitude() / leg_speed).min(60.0);
    }
    // He aims where the carrier is HEADING, not where he'll sidestep to.
    let aim_point = xy(ball) + v_base * t;
    let mut axis = norm_or(aim_point - xy(defender), norm_or(xy(opponent.position - defender), Vector3::new(1.0, 0.0, 0.0)));
    if aim_err != 0.0 {
        axis = rotate(axis, aim_err);
    }

    // Intent: away from his own goal, leaning to the nearest touchline
    // (wide is safe). Challenges are soft (win it / knock it clear of
    // the man), slides hit harder.
    let own_goal = ctx.ball().direction_to_own_goal();
    let away = norm_or(xy(ball - own_goal), Vector3::new(1.0, 0.0, 0.0));
    let field_h = ctx.context.field_size.height as f32;
    let lateral = if ball.y < field_h * 0.5 {
        Vector3::new(0.0, -1.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let (intent, intent_speed, restitution) = match kind {
        TackleKind::Slide => (norm_or(away * 0.5 + lateral * 0.9, away), 1.3, 0.35),
        TackleKind::Challenge => (norm_or(away * 0.7 + lateral * 0.4, away), 0.55, 0.2),
    };

    ContactInput {
        defender,
        defender_vel: xy(ctx.player.velocity),
        ball,
        ball_vel: Vector3::new(v_in.x, v_in.y, 0.0),
        // The engine keeps an owned ball on the carrier's feet, but in
        // a real dribble the ball is ~0.6m ahead of his body. Model
        // that so "man before ball" depends on the approach angle: a
        // front-on lunge meets the ball first, a lunge from behind or
        // the side meets the man first. (Standing carrier: body sits
        // on the far side of the ball from the tackler.)
        carrier: {
            let moving = v_in.magnitude() > 0.1;
            let dir = if moving {
                norm_or(v_in, Vector3::new(1.0, 0.0, 0.0))
            } else {
                norm_or(xy(ball - defender), Vector3::new(1.0, 0.0, 0.0))
            };
            if moving {
                xy(ball) - dir * 5.0
            } else {
                xy(ball) + dir * 3.0
            }
        },
        axis,
        leg_speed,
        reach,
        windup_ticks: windup,
        intent_dir: intent,
        intent_speed,
        restitution,
        control_skill: (0.3 + 0.7 * skills.tackle).clamp(0.0, 1.0),
        jitter,
        jitter_sigma: 0.15 * (1.0 - 0.7 * skills.tackle),
        loft_roll: loft,
    }
}

fn base_windup(kind: TackleKind, skills: DuelSkills) -> f32 {
    let slack = (1.0 - skills.tackle).clamp(0.0, 1.0);
    match kind {
        TackleKind::Slide => 2.0 + 6.0 * slack,
        TackleKind::Challenge => 0.5 + 2.5 * slack,
    }
}

/// Deterministic prediction of a slide: value of committing now,
/// averaged over an early / expected / late wind-up.
fn slide_value(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    skills: DuelSkills,
    threat: f32,
) -> f32 {
    let own_goal = ctx.ball().direction_to_own_goal();
    let to_own_goal = norm_or(xy(own_goal - ctx.tick_context.positions.ball.position), Vector3::new(-1.0, 0.0, 0.0));
    let in_box = own_box_contains_ball(ctx);
    let w0 = base_windup(TackleKind::Slide, skills);
    let mut total = 0.0;
    for w in [w0 - 2.0, w0, w0 + 3.0] {
        let input = build_input(ctx, opponent, skills, TackleKind::Slide, w.max(0.0), 0.0, Vector3::zeros(), 0.0, 0.5);
        let o = resolve_contact(&input);
        total += if o.foul {
            if in_box {
                // A spot kick is worth ~0.76 xG to them — dwarfs
                // everything except a certain goal.
                -0.8
            } else {
                // Tactical stop: the attack ends, at the price of a
                // free kick and card risk.
                0.7 * threat - 0.25
            }
        } else if o.touched {
            let v = xy(o.velocity);
            let speed = v.magnitude();
            let toward_own_goal = speed > 0.5 && v.dot(&to_own_goal) > 0.25 * speed;
            let after = if toward_own_goal { 1.0 } else { 0.25 };
            threat * (1.0 - after)
        } else {
            // Committed and got nothing: beaten, and out of position.
            -threat - 0.1
        };
    }
    total / 3.0
}

/// Pure (RNG-free) decision: hold, standing challenge or slide.
/// `standoff_ticks` is how long this duel has been going on
/// (`ctx.in_state_time`).
pub fn choose(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    skills: DuelSkills,
    standoff_ticks: f32,
) -> TackleChoice {
    let own_goal = ctx.ball().direction_to_own_goal();
    let margin = dv::is_goal_side(ctx.player.position, opponent.position, own_goal);
    let d = opponent.distance(ctx);
    let threat = dv::value_conceded_if_beaten(ctx, opponent);
    let p_win = dv::tackle_win_probability(ctx, opponent, skills.tackle);
    let covered = if dv::has_covering_teammate(ctx, opponent.position) {
        1.0
    } else {
        0.0
    };

    // Holding off is NOT free: it works while goal-side (J) but the
    // longer the standoff, the likelier the attacker gets a shot or
    // gets by — and that costs the threat at his position (T).
    // Near goal the clock runs faster.
    let tau = 150.0 * (1.0 - 0.7 * threat);
    let g = 1.0 - (-standoff_ticks / tau.max(30.0)).exp();
    let goal_side_factor = (margin / 20.0).clamp(0.0, 1.0);
    let jockey_benefit = 0.2 * goal_side_factor * (0.6 + 0.4 * covered);
    let hold = jockey_benefit * (1.0 - g) - threat * g;

    let mut best = (hold, TackleChoice::Hold);

    if d <= LUNGE_REACH {
        let reach_quality = if d <= 12.0 {
            1.0
        } else {
            (1.0 - (d - 12.0) / (LUNGE_REACH - 12.0) * 0.6).clamp(0.3, 1.0)
        };
        let p_hit = p_win * reach_quality;
        let p_foul = (0.12 - 0.06 * skills.discipline).max(0.03);
        let p_beaten = (1.0 - p_hit - p_foul).max(0.0);
        let v_win = 0.35 + 0.65 * threat;
        // Standing challenges carry NO penalty-box surcharge: a normal
        // tackle in the box is welcome (Pavel, 2026-09-29).
        let challenge = p_hit * v_win - p_foul * 0.15 - p_beaten * threat;
        if challenge > best.0 {
            best = (challenge, TackleChoice::Act(TackleKind::Challenge));
        }
    }

    let slide_candidate = (SLIDE_MIN_DISTANCE..=SLIDE_REACH).contains(&d)
        && (margin <= 0.0 || threat >= 0.6)
        && threat > 0.25;
    if slide_candidate {
        let sv = slide_value(ctx, opponent, skills, threat);
        if sv > best.0 {
            best = (sv, TackleChoice::Act(TackleKind::Slide));
        }
    }
    best.1
}

/// Velocity for a defender who is holding: stand off goal-side of the
/// carrier and mirror his movement.
pub fn jockey_velocity(ctx: &StateProcessingContext, opponent: &MatchPlayerLite) -> Vector3<f32> {
    let own_goal = ctx.ball().direction_to_own_goal();
    let to_goal = norm_or(xy(own_goal - opponent.position), Vector3::new(-1.0, 0.0, 0.0));
    let target = opponent.position + to_goal * JOCKEY_STANDOFF;
    SteeringBehavior::Pursuit {
        target,
        target_velocity: opponent.velocity(ctx),
    }
    .calculate(ctx.player)
    .velocity
        + ctx.player().separation_velocity() * 0.15
}

/// Run the whole duel for one tick: decide, and if a tackle is chosen,
/// resolve it through the contact model and return the events.
pub fn duel(
    ctx: &StateProcessingContext,
    opponent: &MatchPlayerLite,
    skills: DuelSkills,
) -> DuelAction {
    let kind = match choose(ctx, opponent, skills, ctx.in_state_time as f32) {
        TackleChoice::Hold => return DuelAction::Hold,
        TackleChoice::Act(k) => k,
    };
    let rng = &ctx.context.rng;
    let p_win = dv::tackle_win_probability(ctx, opponent, skills.tackle);
    let evasion = (1.0 - p_win / 0.55).clamp(0.0, 1.0);
    let jitter = tri(rng.unit_f32(), rng.unit_f32());
    let loft = rng.unit_f32();
    let aim_sigma = 0.03 + 0.12 * (1.0 - skills.tackle);
    let aim_err = tri(rng.unit_f32(), rng.unit_f32()) * aim_sigma;
    // The duel itself: the carrier sidesteps the lunge. How hard he can
    // is set by how much better he is than the tackler (1 - p_win),
    // scaled by his own speed; direction is a coin flip.
    let _ = evasion;
    let to_ball = norm_or(
        xy(ctx.tick_context.positions.ball.position - ctx.player.position),
        Vector3::new(1.0, 0.0, 0.0),
    );
    let perp = Vector3::new(-to_ball.y, to_ball.x, 0.0);
    let side = if rng.unit_f32() < 0.5 { 1.0 } else { -1.0 };
    let carrier_speed = xy(opponent.velocity(ctx)).magnitude().max(0.3);
    let dodge = perp * (side * (1.0 - p_win) * 1.2 * carrier_speed * rng.unit_f32());
    let windup_extra = tri(rng.unit_f32(), rng.unit_f32())
        * match kind {
            TackleKind::Slide => 4.0,
            TackleKind::Challenge => 1.5,
        }
        * (1.0 - 0.5 * skills.tackle);
    let windup = (base_windup(kind, skills) + windup_extra + (1.0 - skills.condition) * 3.0).max(0.0);

    let input = build_input(ctx, opponent, skills, kind, windup, aim_err, dodge, jitter, loft);
    let outcome = resolve_contact(&input);

    let aggression01 = (ctx.player.skills.mental.aggression / 20.0).clamp(0.0, 1.0);
    let defender_id = ctx.player.id;
    let mut events: Vec<Event> = Vec::new();

    #[cfg(feature = "match-logs")]
    {
        let vin = xy(input.ball_vel);
        let vout = xy(outcome.velocity);
        let turn = if outcome.touched && vin.magnitude() > 1e-3 {
            (vout.y.atan2(vout.x) - vin.y.atan2(vin.x)).to_degrees()
        } else {
            0.0
        };
        crate::match_log_info!(
            "TACKLE kind={:?} tick={} id={} d={:.1} touched={} bodyc={} foul={} ctrl={:.2} legfrac={:.2} turn={:.0} vin={:.2} vout={:.2} vz={:.3} box={}",
            kind,
            ctx.context.current_tick(),
            defender_id,
            opponent.distance(ctx),
            outcome.touched,
            outcome.body_contact,
            outcome.foul,
            outcome.control,
            outcome.leg_fraction,
            turn,
            vin.magnitude(),
            vout.magnitude(),
            outcome.velocity.z,
            own_box_contains_ball(ctx)
        );
    }

    // Incidental contact foul on a standing challenge (shirt/shoulder/
    // hand): the calibrated discipline model, only when the geometry
    // itself was clean. Slides get their fouls purely from geometry.
    let mut foul = outcome.foul;
    if !foul && kind == TackleKind::Challenge && (outcome.touched || outcome.body_contact) {
        let mut base = 0.075 + aggression01 * 0.12 - skills.discipline * 0.075;
        base *= if outcome.touched { 0.30 } else { 1.80 };
        base += (1.0 - skills.condition).max(0.0) * 0.08;
        if own_box_contains_ball(ctx) {
            base *= 0.30;
        }
        if ctx.player.yellow_cards > 0 {
            base *= 0.70;
        }
        let p = (base * sc::FOUL_TIME_COMPRESSION).clamp(0.006, 0.85);
        foul = rng.unit_f32() < p;
    }

    if foul {
        let severity = if aggression01 > 0.75 && !outcome.touched && rng.unit_f32() < 0.008 {
            FoulSeverity::Violent
        } else if !outcome.touched && aggression01 > 0.55 && rng.unit_f32() < 0.35 {
            FoulSeverity::Reckless
        } else {
            FoulSeverity::Normal
        };
        events.push(Event::PlayerEvent(PlayerEvent::CommitFoul(defender_id, severity)));
    } else if outcome.touched {
        events.push(Event::PlayerEvent(PlayerEvent::TackleContact(
            defender_id,
            outcome.velocity,
        )));
    }
    DuelAction::Act(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn angle_deg(from: Vector3<f32>, to: Vector3<f32>) -> f32 {
        let a = from.y.atan2(from.x);
        let b = to.y.atan2(to.x);
        let mut d = (b - a).to_degrees();
        while d > 180.0 {
            d -= 360.0;
        }
        while d < -180.0 {
            d += 360.0;
        }
        d
    }

    /// A ball heading down +x at `ball_speed`; the defender stands 14u to
    /// the side and 20u ahead and slides toward the point where his leg
    /// tip can meet the ball (solved for the leg speed). `windup`
    /// (ticks) is the commit delay: 0 = on time, larger = late.
    fn slide_at(ball_speed: f32, windup: f32, jitter: f32) -> ContactOutcome {
        let ball_vel = Vector3::new(ball_speed, 0.0, 0.0);
        let ball = Vector3::new(0.0, 0.0, 0.0);
        let defender = Vector3::new(20.0, -14.0, 0.0);
        let leg_speed = 1.4f32;
        // Intercept solve (ignoring wind-up: he AIMS on time, then is late).
        let mut t = 5.0f32;
        for _ in 0..40 {
            let aim = ball + ball_vel * t;
            t = (aim - defender).magnitude() / leg_speed;
        }
        let aim_point = ball + ball_vel * t;
        let axis = (aim_point - defender).normalize();
        resolve_contact(&ContactInput {
            defender,
            defender_vel: Vector3::zeros(),
            ball,
            ball_vel,
            carrier: Vector3::new(-3.0, 0.0, 0.0),
            axis,
            leg_speed,
            reach: 34.0,
            windup_ticks: windup,
            intent_dir: Vector3::new(-0.6, -0.8, 0.0).normalize(),
            intent_speed: 1.4,
            restitution: 0.35,
            control_skill: 0.8,
            jitter,
            jitter_sigma: 0.08,
            loft_roll: 0.5,
        })
    }

    fn slide(windup: f32, jitter: f32) -> ContactOutcome {
        slide_at(0.5, windup, jitter)
    }

    #[test]
    fn miss_when_far_too_late() {
        let o = slide(60.0, 0.0);
        assert!(!o.touched);
        assert_eq!(o.velocity.x, 0.5);
    }

    #[test]
    fn sweep_lateness_prints_distribution() {
        // Offline sanity sweep: prints the outgoing angle/speed as
        // lateness grows so the three anchors (graze / a little late /
        // well late) can be eyeballed. Run with --nocapture.
        for speed in [0.5f32, 1.2] {
            println!("--- incoming ball speed {speed} u/tick");
            println!("windup | touched | foul | turn(deg) | speed | vz(m/t) | ctrl | legfrac");
            for w in [
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0,
            ] {
                let o = slide_at(speed, w, 0.0);
                let turn = angle_deg(Vector3::new(speed, 0.0, 0.0), o.velocity);
                println!(
                    "{:6.1} | {:7} | {:4} | {:9.1} | {:5.2} | {:7.4} | {:4.2} | {:4.2}",
                    w,
                    o.touched,
                    o.foul,
                    turn,
                    Vector3::new(o.velocity.x, o.velocity.y, 0.0).magnitude(),
                    o.velocity.z,
                    o.control,
                    o.leg_fraction
                );
            }
        }
    }

    #[test]
    fn velocity_never_exceeds_cap() {
        for w in 0..30 {
            let o = slide(w as f32, 0.7);
            let v = Vector3::new(o.velocity.x, o.velocity.y, 0.0).magnitude();
            assert!(v <= MAX_DEFLECT_SPEED + 1e-3);
            assert!(o.velocity.z >= 0.0);
        }
    }
}
