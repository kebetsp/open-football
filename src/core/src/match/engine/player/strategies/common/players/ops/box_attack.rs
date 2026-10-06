//! Domain 4 — attacking the penalty area and delivering into it, designed
//! from null. Read `docs/box-attack-from-null.md` first (football answers,
//! sourcing, guards) and `docs/off-ball-attack-gap-analysis.md` (why this
//! domain exists).
//!
//! **Ownership.** While the ball is in the final third, this module owns
//! the carrier's decision (`decide_final_third`, called from
//! `decide_on_ball_action` in place of domain 2's evaluation) and, once
//! the carrier is within crossing/shooting range of the goal line, the
//! off-ball runners' targets (`plan_box_runs`, called from
//! `assign_attacking_support` in place of domain 3's Final-phase
//! Width/Advance/Penetrate targets for the players it assigns).
//!
//! **Currency.** Every option is valued as expected goals, and 0 on losing
//! the ball, so a shot, a cross, a pass, a carry and a take-on compare on
//! one scale. The numbers come from StatsBomb open data (Bundesliga
//! 2023/24, 34 matches, priority (a)): the shot model is a fit of
//! StatsBomb's own xG to distance, goal angle, blockers and keeper cover;
//! the cross completion (0.35), the cross-to-shot rate and the value of
//! keeping the ball in the final third (0.022 xG per possession) are
//! measured; the take-on success rate (52.7%) is measured. Anything else
//! is flagged "estimate" at its constant.
//!
//! **Call-graph check (`docs/decision-from-null.md`).** Depends on
//! measurements only (positions, goal geometry, `GOAL_WIDTH`, skills) plus
//! domain 2's own from-null geometry helpers (`intercept_risk`,
//! `openness`, `open_distance_in_heading`, `nearest_blocker`,
//! `nearest_opponent_dist`, `rotate_xy`). No pre-rewrite decision function
//! is called; the old `xg` module and `on_ball_value` are not touched.

use crate::PlayerFieldPositionGroup;
use crate::r#match::engine::goal::GOAL_WIDTH;
use crate::r#match::player::strategies::common::players::ops::on_ball_decision::{
    Action, FINAL_THIRD_BOUNDARY, intercept_risk, nearest_blocker, nearest_opponent_dist,
    open_distance_in_heading, open_distance_capped, carry_fair_scoring, carry_sticky_mult, openness, rotate_xy,
};
use crate::r#match::{
    BoxPlan, BoxRunner, BoxZone, MatchContext, MatchPlayerLite, StateProcessingContext,
};
use nalgebra::Vector3;

const UNITS_PER_METRE: f32 = 8.0;

// ---- Shot model: fit of StatsBomb xG on foot open-play shots (n=738).
// logit(xG) = A + B*dist_m + C*angle_rad + D*blockers + E*keeper_cover
const XG_A: f32 = -1.991;
const XG_B_DIST_M: f32 = -0.081;
const XG_C_ANGLE: f32 = 2.381;
const XG_D_BLOCKER: f32 = -0.255;
const XG_E_KEEPER: f32 = -0.519;
/// A keeper's body is ~1m wide for the cover measure (matches the fit).
const KEEPER_BODY_M: f32 = 1.0;

/// Expected goals of a foot shot from `from` at the goal centred on
/// `goal`, given the opposing players (goalkeeper included) around.
pub fn shot_xg(from: Vector3<f32>, goal: Vector3<f32>, opponents: &[MatchPlayerLite]) -> f32 {
    let dx = (goal.x - from.x).abs();
    if dx < 1.0 {
        return 0.0;
    }
    let dy = from.y - goal.y;
    let gw = GOAL_WIDTH;
    let angle = ((dy - gw).atan2(dx) - (dy + gw).atan2(dx)).abs();
    let dist_m = (dx * dx + dy * dy).sqrt() / UNITS_PER_METRE;
    // Triangle shooter -> left post -> right post, in (x, y) space with the
    // goal line at goal.x.
    let a = (from.x, from.y);
    let b = (goal.x, goal.y - gw);
    let c = (goal.x, goal.y + gw);
    let sign = |p: (f32, f32), q: (f32, f32), r: (f32, f32)| {
        (p.0 - r.0) * (q.1 - r.1) - (q.0 - r.0) * (p.1 - r.1)
    };
    let inside = |p: (f32, f32)| {
        let d1 = sign(p, a, b);
        let d2 = sign(p, b, c);
        let d3 = sign(p, c, a);
        !((d1 < 0.0 || d2 < 0.0 || d3 < 0.0) && (d1 > 0.0 || d2 > 0.0 || d3 > 0.0))
    };
    let mut blockers = 0.0f32;
    let mut keeper_cover = 0.0f32;
    for o in opponents {
        let p = (o.position.x, o.position.y);
        if !inside(p) {
            continue;
        }
        if o.tactical_positions.position_group() == PlayerFieldPositionGroup::Goalkeeper {
            let d_m = ((o.position.x - from.x).powi(2) + (o.position.y - from.y).powi(2)).sqrt()
                / UNITS_PER_METRE;
            keeper_cover = (KEEPER_BODY_M / d_m.max(0.5) / angle.max(1e-3)).min(1.0);
        } else {
            blockers += 1.0;
        }
    }
    let logit = XG_A
        + XG_B_DIST_M * dist_m
        + XG_C_ANGLE * angle
        + XG_D_BLOCKER * blockers.min(4.0)
        + XG_E_KEEPER * keeper_cover;
    (1.0 / (1.0 + (-logit).exp())).clamp(0.0, 0.6)
}

// ---- Value constants (sourced unless marked estimate).
/// Value of having the ball at a location when the carrier does NOT shoot
/// at once: the xG of the shots the possession goes on to produce, measured
/// on every non-shot pass/carry/dribble by the possession team in the
/// attacking third (StatsBomb, ~8,000 events). Central: 8-12m 0.102,
/// 12-16.5m 0.062, 16.5-22m 0.046, 22-30m 0.031, 30-40m 0.020 -> fitted as
/// 0.10*exp(-(d-10)/14). Wide (>= 10m off the centre line): flat 0.022-0.04,
/// taken as 0.028. This is the number a shot has to beat.
const CONT_CENTRAL_PEAK: f32 = 0.10;
const CONT_CENTRAL_DECAY_M: f32 = 14.0;
const CONT_WIDE: f32 = 0.028;
/// A receiver shoots with some time pressure taken off the model
/// (estimate; multiplier on his shot xG when valuing a pass to him).
const RECV_SHOT_FACTOR: f32 = 0.75;

fn continuation(point: Vector3<f32>, goal: Vector3<f32>) -> f32 {
    let dx = (goal.x - point.x).abs();
    let dy = (point.y - goal.y).abs();
    let d_m = (dx * dx + dy * dy).sqrt() / UNITS_PER_METRE;
    let central = if d_m <= 10.0 {
        CONT_CENTRAL_PEAK
    } else {
        CONT_CENTRAL_PEAK * (-(d_m - 10.0) / CONT_CENTRAL_DECAY_M).exp()
    };
    let w = ((dy / UNITS_PER_METRE - 10.0) / 10.0).clamp(0.0, 1.0);
    (1.0 - w) * central + w * CONT_WIDE
}
/// Value of the possession after a CARRY ends at `point`, fitted to where
/// real wide-lane carries (>= 12.5m off the centre line, 8-32m from the goal
/// line) end (StatsBomb Bundesliga 2023/24, 34 matches, future open-play xG
/// in the possession after the carry; `scripts/attack-gap/statsbomb/
/// carryend.py`): byline (<= 8m deep, >= 9.2m wide) 0.029 (n=159), box
/// channel 0.040 (n=195), wide outside the box 0.0255 (n=296), central
/// 16-25m 0.0226 (n=326), anything else 0.021 (n=1278); central box 0.145
/// (n=12, too few: the all-action fit above is kept there). The all-action
/// `continuation` (0.10*exp(-(d-10)/14)) is ~2x the real figure for a carry
/// ending central 16-25m out, which made every cut-in outscore a byline run
/// (real: 0.0226 vs 0.029).
fn carry_continuation(point: Vector3<f32>, goal: Vector3<f32>) -> f32 {
    let dep = (goal.x - point.x).abs() / UNITS_PER_METRE;
    let lat = (point.y - goal.y).abs() / UNITS_PER_METRE;
    let base = continuation(point, goal);
    if dep <= 16.5 && lat < 9.2 {
        return base;
    }
    let real = if dep <= 8.0 && lat >= 9.2 {
        0.029
    } else if dep <= 16.5 && lat < 20.2 {
        0.040
    } else if dep <= 16.5 {
        0.0255
    } else {
        0.0226
    };
    base.min(real)
}
/// Same-binary A/B toggle: set `CARRY_ENDVAL_ABLATE` (any value) to use the
/// all-action continuation for carry end points again.
fn carry_endval_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CARRY_ENDVAL_ABLATE").is_none())
}
/// OPT-IN (default off): set `CARRY_OPTIONS` (any value) to value a carry that
/// ends in the central box by the best pass from that spot instead of the
/// fixed 0.10 continuation. Measured 2026-10-06 (40 matches each, same
/// binary): no effect on winger cut-in ratio (slots 5/8 ~4:1 either way) and
/// byline arrivals 1.25 vs 1.65/match without it — so not enabled.
fn carry_options_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CARRY_OPTIONS").is_some())
}
/// Floor for the continuation of a carry that ends in the central box with no
/// teammate worth passing to: the real value of the least valuable zones
/// (StatsBomb carry-end table above: 0.021-0.029). Estimate (rounded).
const CARRY_OPTIONS_FLOOR: f32 = 0.02;
/// What the carrier could do from `point` besides shooting: the best pass to
/// a teammate (completion against defenders from `point`, box cap included,
/// teammate valued by `chance_at` where he stands), never below the floor.
/// Empty box or covered lanes -> the floor, so a cut-in into nothing is no
/// longer worth the fixed 0.10.
fn best_pass_from(
    ctx: &StateProcessingContext,
    point: Vector3<f32>,
    goal: Vector3<f32>,
    opponents: &[MatchPlayerLite],
) -> f32 {
    let mut best = CARRY_OPTIONS_FLOOR;
    for tm in ctx.players().teammates().nearby(PASS_MAX_RANGE) {
        if tm.tactical_positions.position_group() == PlayerFieldPositionGroup::Goalkeeper {
            continue;
        }
        if (tm.position - point).magnitude() < 5.0 {
            continue;
        }
        let mut p_c = 1.0 - intercept_risk(ctx, point, tm.position, opponents);
        if in_box(tm.position, goal) {
            let cap = if in_box(point, goal) {
                BOX_PASS_COMPLETION_FROM_INSIDE
            } else {
                BOX_PASS_COMPLETION_FROM_OUTSIDE
            };
            p_c = p_c.min(cap);
        }
        best = best.max(p_c * (chance_at(tm.position, goal, opponents) + RETAIN_BONUS));
    }
    best
}
/// `chance_at` for the end point of a carry.
fn carry_chance_at(
    ctx: &StateProcessingContext,
    point: Vector3<f32>,
    goal: Vector3<f32>,
    opponents: &[MatchPlayerLite],
) -> f32 {
    if !carry_endval_on() {
        return chance_at(point, goal, opponents);
    }
    let shoot = RECV_SHOT_FACTOR * shot_xg(point, goal, opponents);
    let cross = if in_crossing_area(point, goal) {
        CROSS_TYPICAL_EV * openness(point, opponents)
    } else {
        0.0
    };
    let dep = (goal.x - point.x).abs() / UNITS_PER_METRE;
    let lat = (point.y - goal.y).abs() / UNITS_PER_METRE;
    let cont = if carry_options_on() && dep <= 16.5 && lat < 9.2 {
        best_pass_from(ctx, point, goal, opponents)
    } else {
        carry_continuation(point, goal)
    };
    cont.max(shoot).max(cross)
}
/// Open-play cross completion.
const CROSS_COMPLETION: f32 = 0.35;
/// Share of completed crosses that become a shot (a shot follows 20.5% of
/// crosses within 4 events; 0.205/0.35).
const P_SHOT_GIVEN_REACH: f32 = 0.6;
/// Mean xG created per delivery (0.026-0.038 across origin depths).
const CROSS_TYPICAL_EV: f32 = 0.03;
/// Take-on success (StatsBomb Dribble complete / attempted).
const TAKE_ON_SUCCESS: f32 = crate::r#match::player::strategies::common::players::ops::tackle::TAKE_ON_BASE;
/// A carry or take-on delays the chance and keeps the ball at some risk
/// (estimate).
const CARRY_DISCOUNT: f32 = 0.9;
/// Extra value of KEEPING the ball, beyond the xG of this possession: a
/// retained possession also pays off in later possessions (territory,
/// no counter-attack), which the per-possession continuation value cannot
/// show. A shot spends the possession, so it gets none. Calibrated, not
/// measured: chosen so that the share of real centre-forward receptions
/// that end in a shot by shot quality (StatsBomb: 1% below 0.02 xG, 9% at
/// 0.02-0.04, 21% at 0.04-0.07, 28% at 0.07-0.12, 57% above 0.12) is
/// reproduced — see docs/box-attack-from-null.md.
const RETAIN_BONUS: f32 = 0.03;
/// A carrier with a defender at his shoulder loses the ball far more often
/// than one in space (real forwards lose 20-26% of final-third receptions
/// but take a defender on in only ~4%). Success of a carry is scaled down
/// by up to this fraction when the nearest opponent is on top of him
/// (fading out by 3m). Calibrated, not measured.
const CARRY_PRESSURE_LOSS: f32 = 0.35;
const CARRY_PRESSURE_RANGE_M: f32 = 3.0;
/// Standing still with the ball is worth almost nothing (estimate).
const HOLD_VALUE: f32 = 0.004;
/// A passer who just received from a teammate rarely returns it at once
/// (estimate; multiplier on the pass value).
const BACK_TO_PASSER_MULT: f32 = 0.6;

/// A crossing position is wide (>= 12.5m off the centre line) and within
/// 25m of the goal line (StatsBomb: ~76% of crosses start within 16.5m,
/// the rest deeper).
const CROSS_MIN_LATERAL: f32 = 100.0;
const CROSS_MAX_DEPTH: f32 = 200.0;
/// Reference for how open a delivery zone is (~7.5m).
const ZONE_OPEN_REF: f32 = 60.0;
/// How near a runner must be to his zone for the delivery to count as
/// arriving together (estimate, ~15m -> scales the arrival factor).
const ARRIVE_REF: f32 = 120.0;

/// Real completion of a pass that ends in the penalty box (regular play,
/// StatsBomb): 0.55 when played from outside the box, 0.42 from inside it
/// (n=28 per match, 0.46 overall). The engine's lane model says ~0.94, so a
/// pass to a teammate inside the box is capped at these values when
/// weighing it against a delivery or a shot.
const BOX_PASS_COMPLETION_FROM_OUTSIDE: f32 = 0.55;
const BOX_PASS_COMPLETION_FROM_INSIDE: f32 = 0.42;
/// Penalty area, in units (16.5m deep, 40.3m wide).
const BOX_DEPTH: f32 = 132.0;
const BOX_HALF_WIDTH: f32 = 161.0;

fn in_box(point: Vector3<f32>, goal: Vector3<f32>) -> bool {
    (goal.x - point.x).abs() <= BOX_DEPTH && (point.y - goal.y).abs() <= BOX_HALF_WIDTH
}

const PASS_MAX_RANGE: f32 = 320.0;
const SHOT_MAX_RANGE: f32 = 300.0;
const CARRY_STEP: f32 = 80.0;

/// Fair-carry step (2026-09-29): with `CARRY_FAIR_ABLATE` unset a final-third
/// carry may run up to 40m (was one 10m step), so a longer open run reaches
/// a better `chance_at` end point. Swept 160/240/320/440 against the sourced
/// target (StatsBomb Bundesliga 2023/24: 10.5% of wide final-third
/// receptions are followed by a carry >= 10m); 320 gave the best guards
/// (forward shot share 13% vs real 12.8%, spacing and defenders-at-a-shot
/// unchanged), 440 gave the same carry rate.
const FAIR_CARRY_STEP: f32 = 320.0;
const CARRY_HEADINGS: usize = 7;
/// Defenders within this of a carry's end point (6m) each have to be beaten.
const CARRY_CROWD_RADIUS: f32 = 48.0;
/// Defenders within this of a carry's path (4m: a defender this close steps
/// across and contests) each have to be beaten at the real take-on rate
/// 0.527 (the same contest model the duel physics uses). Without it a cut-in
/// "through" the gaps between standing defenders was free, since the lane
/// test only sees defenders within 2.5m. Same-binary A/B toggle:
/// `CARRY_PATH_ABLATE`.
const CARRY_PATH_CONTEST_RADIUS: f32 = 32.0;
fn carry_path_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CARRY_PATH_ABLATE").is_none())
}
fn dist_point_segment(p: Vector3<f32>, a: Vector3<f32>, b: Vector3<f32>) -> f32 {
    let ab = b - a;
    let l2 = ab.x * ab.x + ab.y * ab.y;
    let t = if l2 > 0.0 { (((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let q = Vector3::new(a.x + ab.x * t, a.y + ab.y * t, 0.0);
    ((p.x - q.x).powi(2) + (p.y - q.y).powi(2)).sqrt()
}
/// Run commitment for a flank run (wide-lane plan, 2026-10-04): a carrier who
/// is already moving along the flank toward the goal line keeps the run
/// unless a pass/delivery is clearly better, instead of re-deciding every
/// tick and giving the ball up after ~0.7s (measured: byline arrivals from
/// the wide lane ~1.1/match vs real 4.7). Size is a calibration to that real
/// arrival rate (swept via `BYLINE_STICKY`, default `BYLINE_STICKY_DEFAULT`).
const BYLINE_STICKY_DEFAULT: f32 = 0.5;
fn byline_sticky() -> f32 {
    static K: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("BYLINE_STICKY").ok().and_then(|v| v.parse().ok()).unwrap_or(BYLINE_STICKY_DEFAULT)
    })
}
/// Same-binary A/B toggle: set `CARRY_CROWD_ABLATE` (any value) to drop it.
fn carry_crowd_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CARRY_CROWD_ABLATE").is_none())
}
/// Six-yard box half-width (9.2m) and the minimum goal-line depth of a carry's
/// end point in front of goal (5.5m) / wide of it (3m), in units.
const CARRY_GOALMOUTH_HALF_WIDTH: f32 = 74.0;
const CARRY_FLOOR_CENTRAL: f32 = 44.0;
const CARRY_FLOOR_WIDE: f32 = 24.0;
/// Same-binary A/B toggle: set `CARRY_GOALLINE_ABLATE` (any value) to restore
/// the old end-point clamp.
fn carry_goalline_floor_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CARRY_GOALLINE_ABLATE").is_none())
}
const CARRY_FAN_HALF_DEG: f32 = 90.0;

// ---- Runs.

/// A player who can be sent to a zone.
#[derive(Clone, Copy)]
pub struct RunCandidate {
    pub id: u32,
    pub position: Vector3<f32>,
    pub group: PlayerFieldPositionGroup,
    /// True for wide-slotted roles (wingers, wide midfielders, full-backs).
    pub wide_tag: bool,
}

/// Distance of a zone from the goal line and its lateral offset toward
/// the carrier's side (units). Near post ~5.5m from the goal line just
/// inside the post, far post ~6m beyond the far post, penalty-spot depth
/// 11m, box edge 17m (StatsBomb: 50% of crosses land central at
/// six-yard/penalty-spot depth, 26% in the channels, 17% at the edge).
fn zone_geometry(zone: BoxZone) -> (f32, f32) {
    match zone {
        BoxZone::PenaltySpot => (88.0, 0.0),
        BoxZone::FarPost => (48.0, -37.0),
        BoxZone::Edge => (136.0, 0.0),
        BoxZone::NearPost => (44.0, 30.0),
    }
}

fn zone_point(zone: BoxZone, goal: Vector3<f32>, inward: f32, near_sign: f32) -> Vector3<f32> {
    let (depth, lateral) = zone_geometry(zone);
    Vector3::new(goal.x + inward * depth, goal.y + near_sign * lateral, 0.0)
}

/// Carrier within 25m of the goal line starts the runs; they stop once he
/// is beyond 29m (hysteresis so the plan does not flicker at the line).
const RUN_TRIGGER_DEPTH: f32 = 200.0;
const RUN_RELEASE_DEPTH: f32 = 232.0;
/// Runs also need the carrier to be in a crossing position — wide enough
/// that a delivery is a live option (>= 12.5m off the centre line to
/// start, released below 8.75m). With a central carrier the strikers stay
/// out as link players: real centre-forwards receive 56% of their
/// final-third touches at 25-36m from goal and only 27% in the box
/// (StatsBomb), and after a reception shoot 13% / pass back 26% / pass
/// sideways 15% / pass forward 19%.
const RUN_TRIGGER_LATERAL: f32 = 100.0;
const RUN_RELEASE_LATERAL: f32 = 70.0;
/// Bonus (units) for keeping a runner in the zone he already had.
const STICKY_BONUS: f32 = 60.0;
/// A runner nudges his point to the local gap: candidate offsets.
const NUDGE_LATERAL: f32 = 16.0;
const NUDGE_DEPTH: f32 = 12.0;

/// Assigns up to four attackers to box zones for the side attacking
/// `goal`, and records the plan on the context. Returns (id, point).
pub fn plan_box_runs(
    context: &mut MatchContext,
    side_index: usize,
    goal: Vector3<f32>,
    inward: f32,
    carrier_pos: Vector3<f32>,
    candidates: &[RunCandidate],
    opponents_outfield: &[Vector3<f32>],
) -> Vec<(u32, Vector3<f32>)> {
    let carrier_depth = (goal.x - carrier_pos.x).abs();
    let carrier_lateral = (carrier_pos.y - goal.y).abs();
    let was_active = context.box_plan[side_index].active;
    let threshold = if was_active { RUN_RELEASE_DEPTH } else { RUN_TRIGGER_DEPTH };
    let lat_threshold = if was_active { RUN_RELEASE_LATERAL } else { RUN_TRIGGER_LATERAL };
    if carrier_depth > threshold || carrier_lateral < lat_threshold {
        context.box_plan[side_index] = BoxPlan::default();
        return Vec::new();
    }
    let lat = carrier_pos.y - goal.y;
    let near_sign = if lat.abs() < 24.0 { 1.0 } else { lat.signum() };

    // Near-side wide players keep the flank (overlap / switch option).
    let eligible: Vec<&RunCandidate> = candidates
        .iter()
        .filter(|c| {
            matches!(
                c.group,
                PlayerFieldPositionGroup::Forward | PlayerFieldPositionGroup::Midfielder
            )
        })
        .filter(|c| {
            !(c.wide_tag
                && (c.position.y - goal.y).signum() == near_sign
                && (c.position.y - goal.y).abs() > 100.0)
        })
        .collect();

    let previous = context.box_plan[side_index].runners.clone();
    let mut used: Vec<u32> = Vec::new();
    let mut assigned: Vec<BoxRunner> = Vec::new();
    for zone in [BoxZone::PenaltySpot, BoxZone::FarPost, BoxZone::Edge, BoxZone::NearPost] {
        let base = zone_point(zone, goal, inward, near_sign);
        let mut best: Option<(f32, &RunCandidate)> = None;
        for c in eligible.iter().filter(|c| !used.contains(&c.id)) {
            let mut cost = (c.position - base).magnitude();
            if previous.iter().any(|r| r.id == c.id && r.zone == zone) {
                cost -= STICKY_BONUS;
            }
            if best.map_or(true, |(bc, _)| cost < bc) {
                best = Some((cost, c));
            }
        }
        let Some((_, cand)) = best else { continue };
        used.push(cand.id);
        // Nudge to the local gap between defenders, keeping clear of the
        // points already given to teammates.
        let mut best_pt = base;
        let mut best_score = f32::MIN;
        for dl in [-NUDGE_LATERAL, 0.0, NUDGE_LATERAL] {
            for dd in [-NUDGE_DEPTH, 0.0, NUDGE_DEPTH] {
                let pt = Vector3::new(base.x + inward * dd, base.y + dl, 0.0);
                let opp_d = opponents_outfield
                    .iter()
                    .map(|o| (o - pt).magnitude())
                    .fold(80.0f32, f32::min);
                let mate_d = assigned
                    .iter()
                    .map(|r| (r.point - pt).magnitude())
                    .fold(60.0f32, f32::min);
                let score = opp_d.min(80.0) + mate_d.min(60.0) - 0.3 * (dl.abs() + dd.abs());
                if score > best_score {
                    best_score = score;
                    best_pt = pt;
                }
            }
        }
        assigned.push(BoxRunner { id: cand.id, zone, point: best_pt });
    }
    let out = assigned.iter().map(|r| (r.id, r.point)).collect();
    context.box_plan[side_index] = BoxPlan { active: true, runners: assigned };
    out
}

// ---- On-ball decision.

fn in_crossing_area(point: Vector3<f32>, goal: Vector3<f32>) -> bool {
    (point.y - goal.y).abs() >= CROSS_MIN_LATERAL && (goal.x - point.x).abs() <= CROSS_MAX_DEPTH
}

/// Value of possessing the ball at `point` for a player who will act on it:
/// his own shot, a typical delivery if he is in a crossing position, or
/// carrying the possession on (`continuation`) — the best of the three.
///
/// `pub(crate)` (2026-09-27) so `off_ball_runs.rs` (domain 3.5's off-ball
/// run planner) can reuse this exact, already-sourced value function to
/// answer "what can a runner do once he's at this target" instead of
/// building a second, competing currency for the same question — the
/// same reuse discipline this module's own header already establishes
/// for domain 2's geometry helpers.
pub(crate) fn chance_at(point: Vector3<f32>, goal: Vector3<f32>, opponents: &[MatchPlayerLite]) -> f32 {
    let shoot = RECV_SHOT_FACTOR * shot_xg(point, goal, opponents);
    let cross = if in_crossing_area(point, goal) {
        CROSS_TYPICAL_EV * openness(point, opponents)
    } else {
        0.0
    };
    continuation(point, goal).max(shoot).max(cross)
}

/// Value of an attacker standing at `point` for the DEFENCE's threat
/// ranking (domain 1): the same xG currency the attack uses (continuation,
/// his own shot, a typical cross from a wide crossing position), without
/// defender-dependent terms. A corner-flag winger is worth ~0.03, a
/// central player 10m from goal ~0.10.
pub(crate) fn threat_value_at(point: Vector3<f32>, goal: Vector3<f32>) -> f32 {
    let shoot = RECV_SHOT_FACTOR * shot_xg(point, goal, &[]);
    let cross = if in_crossing_area(point, goal) { CROSS_TYPICAL_EV } else { 0.0 };
    continuation(point, goal).max(shoot).max(cross)
}

/// The final-third decision for the ball carrier. Always returns an
/// action (domain 4 governs the whole final third).
pub fn decide_final_third(
    ctx: &StateProcessingContext,
    from: Vector3<f32>,
    goal: Vector3<f32>,
    opponents: &[MatchPlayerLite],
) -> Action {
    let mut best = Action::Hold;
    let mut best_value = HOLD_VALUE;
    let (mut d_shoot, mut d_deliver, mut d_pass, mut d_carry, mut d_take) =
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut d_pf, mut d_pl, mut d_pb) = (0.0f32, 0.0f32, 0.0f32);
    let toward = (goal.x - from.x).signum();

    // Shoot.
    if (goal - from).magnitude() <= SHOT_MAX_RANGE {
        let v = shot_xg(from, goal, opponents);
        d_shoot = v;
        if v > best_value {
            best_value = v;
            best = Action::Shoot;
        }
    }

    // Deliver into a planned run zone (wide carriers).
    let side_index = match ctx.player.side {
        Some(crate::r#match::player::PlayerSide::Left) => 0,
        _ => 1,
    };
    let plan = &ctx.context.box_plan[side_index];
    if plan.active && in_crossing_area(from, goal) {
        let crossing = ctx.player.skills.technical.crossing.clamp(1.0, 20.0) / 20.0;
        let outfield: Vec<MatchPlayerLite> = opponents
            .iter()
            .filter(|o| o.tactical_positions.position_group() != PlayerFieldPositionGroup::Goalkeeper)
            .cloned()
            .collect();
        for r in plan.runners.iter().filter(|r| r.id != ctx.player.id) {
            let Some(runner) = ctx.players().teammates().all().find(|t| t.id == r.id) else {
                continue;
            };
            let d_zone = nearest_opponent_dist(r.point, &outfield);
            let open_mod = 0.55 + 0.9 * (d_zone.min(ZONE_OPEN_REF) / ZONE_OPEN_REF);
            let arrive =
                (1.0 - (runner.position - r.point).magnitude() / ARRIVE_REF).clamp(0.35, 1.0);
            let p_reach = CROSS_COMPLETION * (0.8 + 0.4 * crossing) * open_mod * arrive;
            let v = p_reach * P_SHOT_GIVEN_REACH * shot_xg(r.point, goal, opponents);
            d_deliver = d_deliver.max(v);
            if v > best_value {
                best_value = v;
                best = Action::Deliver { to: r.id, target: r.point };
            }
        }
    }

    // Pass.
    for tm in ctx.players().teammates().nearby(PASS_MAX_RANGE) {
        if tm.tactical_positions.position_group() == PlayerFieldPositionGroup::Goalkeeper {
            continue;
        }
        if (tm.position - from).magnitude() < 5.0 {
            continue;
        }
        let mut p_c = 1.0 - intercept_risk(ctx, from, tm.position, opponents);
        if in_box(tm.position, goal) {
            let cap = if in_box(from, goal) {
                BOX_PASS_COMPLETION_FROM_INSIDE
            } else {
                BOX_PASS_COMPLETION_FROM_OUTSIDE
            };
            p_c = p_c.min(cap);
        }
        let mut v = p_c * (chance_at(tm.position, goal, opponents) + RETAIN_BONUS);
        if ctx.tick_context.ball.last_owner == Some(tm.id) {
            v *= BACK_TO_PASSER_MULT;
        }
        d_pass = d_pass.max(v);
        let dxg = (tm.position.x - from.x) * toward;
        if dxg > 24.0 { d_pf = d_pf.max(v) } else if dxg < -24.0 { d_pb = d_pb.max(v) } else { d_pl = d_pl.max(v) }
        if v > best_value {
            best_value = v;
            best = Action::Pass(tm.id);
        }
        // Off-ball attacking runs, from null (`off_ball_runs.rs`): a
        // through ball into a teammate's frozen run target, on this
        // domain's own chance_at/RETAIN_BONUS currency, aimed at where
        // he is running into rather than where he stands. Reads
        // `ctx.tick_context.run_commitments`, NOT `ctx.context.players`
        // — see `GameTickContext::run_commitments`'s own doc comment.
        if let Some(&(_, target)) = ctx.tick_context.run_commitments.get(&tm.id) {
            if (target - from).magnitude() >= 5.0 {
                let p_c2 = 1.0 - intercept_risk(ctx, from, target, opponents);
                let tv = p_c2 * (chance_at(target, goal, opponents) + RETAIN_BONUS);
                if tv > best_value {
                    best_value = tv;
                    best = Action::Through { to: tm.id, target };
                }
            }
        }
    }

    #[cfg(feature = "match-logs")]
    let (mut dbg_best, mut dbg_by) = ((0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32), (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32));
    // Carry.
    let to_goal = goal - from;
    let forward = if to_goal.magnitude() > 1.0 {
        let n = to_goal.normalize();
        Vector3::new(n.x, n.y, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let opp_m = nearest_opponent_dist(from, opponents) / UNITS_PER_METRE;
    let press_keep = 1.0 - CARRY_PRESSURE_LOSS * (1.0 - opp_m / CARRY_PRESSURE_RANGE_M).clamp(0.0, 1.0);
    let half = CARRY_FAN_HALF_DEG.to_radians();
    for i in 0..CARRY_HEADINGS {
        let t = i as f32 / (CARRY_HEADINGS - 1) as f32;
        let heading = rotate_xy(forward, -half + 2.0 * half * t);
        let fair = carry_fair_scoring();
        let step_len = if fair { FAIR_CARRY_STEP } else { CARRY_STEP };
        let open = if fair {
            open_distance_capped(from, heading, opponents, step_len)
        } else {
            open_distance_in_heading(from, heading, opponents)
        };
        let travel = open.min(step_len);
        if travel < 15.0 {
            continue;
        }
        let mut target = from + heading * travel;
        let mut travel = travel;
        if fair {
            // A long fair-carry step can project past the goal line or off
            // the pitch, where chance_at means nothing: keep the end point
            // on the pitch (4u margin) and use the real distance.
            let w = ctx.context.field_size.width as f32;
            let h = ctx.context.field_size.height as f32;
            target.x = target.x.clamp(4.0, w - 4.0);
            target.y = target.y.clamp(4.0, h - 4.0);
            if carry_goalline_floor_on() {
                // A carry does not run into the goalmouth: the shot model
                // fitted on real shots says a point 0.7m from the goal line
                // beside the post is worth 0.25+, so the old clamp (4u from
                // the line) made "carry at the goal line" beat every real
                // option. Real carries from the crossing zone end within
                // 5.5m of the goal line 21% of the time but in the central
                // box 1% (StatsBomb). Keep the end point out of the six-yard
                // box (5.5m) when it is in front of goal, and 3m off the
                // line elsewhere (a byline carry).
                let lat = (target.y - goal.y).abs();
                let floor = if lat < CARRY_GOALMOUTH_HALF_WIDTH { CARRY_FLOOR_CENTRAL } else { CARRY_FLOOR_WIDE };
                if (goal.x - target.x).abs() < floor {
                    target.x = goal.x - toward * floor;
                }
            }
            travel = (target - from).magnitude();
            if travel < 15.0 {
                continue;
            }
        }
        // Ending the carry beside a defender is a take-on: success falls
        // from ~0.95 with room to the sourced 0.527 with none.
        let room = openness(target, opponents);
        let mut p_keep = 0.95 * (open / CARRY_STEP).min(1.0) * (TAKE_ON_SUCCESS + (1.0 - TAKE_ON_SUCCESS) * room);
        if carry_crowd_on() {
            // `room` only sees the single nearest opponent. A run that ends
            // among several defenders means beating each of them (real
            // take-on success 0.527 per defender, StatsBomb); the engine's
            // box is under-defended, so counting them is what stops a
            // winger "carrying" through a crowded area to a central point.
            let crowd = opponents
                .iter()
                .filter(|o| o.tactical_positions.position_group() != PlayerFieldPositionGroup::Goalkeeper)
                .filter(|o| (o.position - target).magnitude() <= CARRY_CROWD_RADIUS)
                .count();
            if crowd > 1 {
                p_keep *= TAKE_ON_SUCCESS.powi(crowd as i32 - 1);
            }
        }
        if carry_path_on() && fair {
            let on_path = opponents
                .iter()
                .filter(|o| o.tactical_positions.position_group() != PlayerFieldPositionGroup::Goalkeeper)
                .filter(|o| (o.position - target).magnitude() > CARRY_CROWD_RADIUS)
                .filter(|o| dist_point_segment(o.position, from, target) <= CARRY_PATH_CONTEST_RADIUS)
                .count();
            if on_path > 0 {
                p_keep *= TAKE_ON_SUCCESS.powi(on_path as i32);
            }
        }
        // Fair path: no flat delay discount. `p_keep` already prices the
        // take-on risk (0.527..0.95 by room), and a pass pays no equivalent
        // multiplier, so the old 0.9 double-counted risk against carries.
        let disc = if fair { 1.0 } else { CARRY_DISCOUNT };
        // Flank run in progress: the carrier is wide and the candidate
        // heading runs mostly toward the goal line (within ~37 degrees of
        // it) without cutting in by more than ~0.4 of its length. The end
        // points are 320u away, so a lateral-distance test on the end
        // point (the first version) almost never matched.
        let from_lat = (from.y - goal.y).abs();
        let inward_comp = -(from.y - goal.y).signum() * heading.y;
        let flank_run = fair
            && from_lat >= CROSS_MIN_LATERAL
            && heading.x * toward >= 0.80
            && inward_comp <= 0.40;
        let flank_mult = if flank_run && carry_sticky_mult(ctx.player.velocity, heading) > 1.0 {
            1.0 + byline_sticky()
        } else {
            1.0
        };
        let v = disc * p_keep * press_keep * (carry_chance_at(ctx, target, goal, opponents) + RETAIN_BONUS)
            * carry_sticky_mult(ctx.player.velocity, heading)
            * flank_mult;
        d_carry = d_carry.max(v);
        #[cfg(feature = "match-logs")]
        {
            let tdep = (goal.x - target.x).abs() / UNITS_PER_METRE;
            let tlat = (target.y - goal.y).abs() / UNITS_PER_METRE;
            let ch = chance_at(target, goal, opponents);
            if v > dbg_best.0 { dbg_best = (v, tdep, tlat, p_keep, ch, room); }
            if tdep <= 8.0 && tlat >= 12.5 && v > dbg_by.0 { dbg_by = (v, tdep, tlat, p_keep, ch, room); }
        }
        if v > best_value {
            best_value = v;
            best = Action::Carry(target);
        }
    }

    // Take-on.
    if let Some(blocker) = nearest_blocker(from, forward, opponents) {
        let beyond = blocker.position + forward * 30.0;
        use crate::r#match::player::strategies::common::players::ops::tackle as contest;
        let p_beat = contest::carrier_keep_probability(
            contest::carrier_skill_of(ctx, ctx.player.id),
            contest::defender_skill_of(ctx, blocker.id),
        );
        let v = CARRY_DISCOUNT * p_beat * press_keep * (chance_at(beyond, goal, opponents) + RETAIN_BONUS);
        d_take = v;
        if v > best_value {
            let lateral = Vector3::new(-forward.y, forward.x, 0.0);
            let side = if (blocker.position - from).dot(&lateral) >= 0.0 { -1.0 } else { 1.0 };
            best = Action::Dribble(beyond + lateral * side * 14.0);
        }
    }

    #[cfg(feature = "match-logs")]
    if (from.y - goal.y).abs() >= 100.0 && (goal.x - from.x).abs() <= 200.0 {
        crate::match_log_info!(
            "CARRYDIAG id={} depth={:.1} lat={:.1} best_v={:.3} best_dep={:.1} best_lat={:.1} best_pk={:.2} best_ch={:.3} best_room={:.2} by_v={:.3} by_dep={:.1} by_lat={:.1} by_pk={:.2} by_ch={:.3} by_room={:.2} deliver={:.3} pass={:.3} chosen={} tdep={:.1} tlat={:.1}",
            ctx.player.id, (goal.x - from.x).abs() / UNITS_PER_METRE, (from.y - goal.y).abs() / UNITS_PER_METRE,
            dbg_best.0, dbg_best.1, dbg_best.2, dbg_best.3, dbg_best.4, dbg_best.5,
            dbg_by.0, dbg_by.1, dbg_by.2, dbg_by.3, dbg_by.4, dbg_by.5, d_deliver, d_pass,
            match best { Action::Shoot => "shoot", Action::Deliver { .. } => "deliver", Action::Through { .. } => "through", Action::Pass(_) => "pass", Action::Carry(_) => "carry", Action::Dribble(_) => "take", _ => "hold" },
            match best { Action::Carry(t) => (goal.x - t.x).abs() / UNITS_PER_METRE, _ => -1.0 },
            match best { Action::Carry(t) => (t.y - goal.y).abs() / UNITS_PER_METRE, _ => -1.0 }
        );
    }
    #[cfg(feature = "match-logs")]
    crate::match_log_info!(
        "BOXDIAG id={} lat_m={:.1} depth_m={:.1} plan={} runners={} shoot={:.3} deliver={:.3} pass={:.3} carry={:.3} take={:.3} pf={:.3} pl={:.3} pb={:.3} opp_m={:.1} chose={}",
        ctx.player.id,
        (from.y - goal.y).abs() / UNITS_PER_METRE,
        (goal.x - from.x).abs() / UNITS_PER_METRE,
        plan.active,
        plan.runners.len(),
        d_shoot,
        d_deliver,
        d_pass,
        d_carry,
        d_take,
        d_pf,
        d_pl,
        d_pb,
        nearest_opponent_dist(from, opponents) / UNITS_PER_METRE,
        match best {
            Action::Shoot => "shoot",
            Action::Deliver { .. } => "deliver",
            Action::Through { .. } => "through",
            Action::Pass(_) => "pass",
            Action::Carry(_) => "carry",
            Action::Dribble(_) => "take",
            _ => "hold",
        }
    );

    best
}

/// True when the ball is in the final third, i.e. domain 4 governs.
pub fn governs(own_goal_dist: f32) -> bool {
    own_goal_dist >= FINAL_THIRD_BOUNDARY
}
