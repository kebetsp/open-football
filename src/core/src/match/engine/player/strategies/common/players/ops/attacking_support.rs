//! Off-ball attacking positioning/support — designed from null.
//!
//! `docs/decision-from-null.md` (domain 3) /
//! `docs/off-ball-attacking-positioning-from-null.md` (this domain's own
//! football reasoning, spec, and verification log — read that first).
//!
//! A single, team-wide, per-tick allocation of attacking duty
//! (rest-defense / width / advance / penetrate) across every outfield
//! teammate of the current ball carrier — the carrier's own decision
//! stays domain 2's (`on_ball_decision`). Deliberately calls no
//! pre-existing decision function (no `ForwardCreatingSpaceState`,
//! `MidfielderRunningState`'s off-ball branch, `DefenderPushingUpState`,
//! or any sibling) — every target formula here is derived fresh from
//! the football reasoning in the design doc, using only raw
//! `MatchField`/`MatchContext` data plus genuine measurements
//! (`max_speed_with_condition`, goal positions, Euclidean geometry).
//!
//! Process check (`docs/decision-from-null.md`'s required self-audit):
//! this module's call graph touches only `MatchField`/`MatchContext`/
//! `MatchPlayer` struct fields and `nalgebra::Vector3` geometry — a
//! measurement, encodes no judgment. Zero pre-existing decision-function
//! dependencies.

use crate::PlayerFieldPositionGroup;
use crate::club::player::ability::position::PlayerPositionType;
use crate::r#match::player::AttackDuty;
use crate::r#match::{MatchContext, MatchField, PlayerSide};
use nalgebra::Vector3;

/// Number of the deepest-positioned defenders held back for balance,
/// regardless of ball advancement — real rest-defense doctrine.
/// Reasoned estimate (category d): a small, fixed floor rather than a
/// fraction that could round to zero.
const REST_DEFENSE_COUNT: usize = 2;

/// Width target depth bounds (distance from own goal) and how strongly
/// it tracks the ball's own advancement. A wide player should already
/// hold a fairly advanced base position even during build-up (real
/// wingers don't retreat deep), and reach genuinely high, close to the
/// opponent's box, once the team is attacking. Reasoned estimate
/// (category d, no sourced exact figure) — ~50m min, ~92.5m max on a
/// 105m pitch.
const WIDTH_MIN_DEPTH: f32 = 400.0;
const WIDTH_MAX_DEPTH: f32 = 740.0;
const WIDTH_PUSH_FACTOR: f32 = 0.85;
/// How close to the touchline a width target sits.
const WIDTH_MARGIN: f32 = 24.0;

/// Advancing-line depth bounds for central players, the direct
/// attacking-phase counterpart of domain 1's own defensive-line
/// mechanism (`LINE_PUSH_FACTOR`/`LINE_MIN_DEPTH_UNITS`/
/// `LINE_MAX_DEPTH_UNITS` in `defensive_duty.rs`) — same idea, much
/// higher cap, since this governs players supporting an attack, not
/// holding a defensive shape. Reasoned estimate (category d) — ~37.5m
/// min, ~77.5m max: genuinely into the opponent's half, not stopping
/// just past halfway (the specific symptom Pavel reported).
const ADVANCE_MIN_DEPTH: f32 = 300.0;
const ADVANCE_MAX_DEPTH: f32 = 620.0;
const ADVANCE_PUSH_FACTOR: f32 = 0.80;

/// How much a central player's lateral position stays anchored to his
/// own formation slot vs. drifting — anti-clustering by construction
/// (different players have different slots).
const ADVANCE_WIDTH_PULL: f32 = 0.25;

/// Beyond this distance from a target, the nearest opponent doesn't
/// meaningfully contest it — genuine open space worth pushing further
/// into.
const PENETRATION_OPEN_THRESHOLD: f32 = 90.0;
/// How far a penetration bias pushes an already-advanced, genuinely
/// open target further toward the opponent's goal.
const PENETRATION_PUSH: f32 = 40.0;

#[derive(Clone, Copy)]
struct Cand {
    id: u32,
    team_id: u32,
    position: Vector3<f32>,
    position_type: PlayerPositionType,
    position_group: PlayerFieldPositionGroup,
    start_position: Vector3<f32>,
    side: Option<PlayerSide>,
    is_directed: bool,
}

struct Assignment {
    id: u32,
    duty: AttackDuty,
    target: Vector3<f32>,
}

fn is_wide(pt: PlayerPositionType) -> bool {
    matches!(
        pt,
        PlayerPositionType::DefenderLeft
            | PlayerPositionType::DefenderRight
            | PlayerPositionType::MidfielderLeft
            | PlayerPositionType::MidfielderRight
            | PlayerPositionType::AttackingMidfielderLeft
            | PlayerPositionType::AttackingMidfielderRight
            | PlayerPositionType::WingbackLeft
            | PlayerPositionType::WingbackRight
            | PlayerPositionType::ForwardLeft
            | PlayerPositionType::ForwardRight
    )
}

fn is_left_flank(pt: PlayerPositionType) -> bool {
    matches!(
        pt,
        PlayerPositionType::DefenderLeft
            | PlayerPositionType::MidfielderLeft
            | PlayerPositionType::AttackingMidfielderLeft
            | PlayerPositionType::WingbackLeft
            | PlayerPositionType::ForwardLeft
    )
}

/// Entry point — called once per full tick from `game_tick_inner`,
/// right after `assign_defensive_duties`. Resets every player's
/// `ai_attack_duty` to `None` each call, then, when a team clearly
/// holds the ball, assigns real duties to that team's non-carrier
/// outfielders. A loose ball leaves this system fully inactive (a
/// deliberate, deferred scope boundary — see the design doc).
pub fn assign_attacking_support(field: &mut MatchField, context: &MatchContext) {
    for p in field.players.iter_mut() {
        p.ai_attack_duty = AttackDuty::None;
    }

    let Some(carrier_id) = field.ball.current_owner else {
        return;
    };

    let cands: Vec<Cand> = field
        .players
        .iter()
        .filter(|p| !p.is_sent_off)
        .map(|p| Cand {
            id: p.id,
            team_id: p.team_id,
            position: p.position,
            position_type: p.tactical_position.current_position,
            position_group: p.tactical_position.current_position.position_group(),
            start_position: p.start_position,
            side: p.side,
            is_directed: p.behavioral_directive.is_some()
                || p.link_target.is_some()
                || p.supply_target.is_some(),
        })
        .collect();

    let Some(carrier) = cands.iter().find(|c| c.id == carrier_id).copied() else {
        return;
    };
    let Some(carrier_side) = carrier.side else {
        return;
    };

    let goal_for_side = |side: PlayerSide| match side {
        PlayerSide::Left => context.goal_positions.left,
        PlayerSide::Right => context.goal_positions.right,
    };
    let opposite_side = |side: PlayerSide| match side {
        PlayerSide::Left => PlayerSide::Right,
        PlayerSide::Right => PlayerSide::Left,
    };
    let own_goal = goal_for_side(carrier_side);
    let opponent_goal = goal_for_side(opposite_side(carrier_side));
    let attack_dir: f32 = match carrier_side {
        PlayerSide::Left => 1.0,
        PlayerSide::Right => -1.0,
    };
    let ball_dist_to_own_goal = (field.ball.position - own_goal).magnitude();

    let opponents: Vec<Vector3<f32>> = cands
        .iter()
        .filter(|c| c.team_id != carrier.team_id)
        .map(|c| c.position)
        .collect();
    let nearest_opp_dist = |p: Vector3<f32>| -> f32 {
        opponents
            .iter()
            .map(|o| (o - p).magnitude())
            .fold(f32::MAX, f32::min)
    };

    let pool: Vec<Cand> = cands
        .iter()
        .filter(|c| {
            c.team_id == carrier.team_id
                && c.id != carrier.id
                && c.position_group != PlayerFieldPositionGroup::Goalkeeper
                && !c.is_directed
        })
        .copied()
        .collect();

    let mut results: Vec<Assignment> = Vec::new();
    let mut assigned: Vec<u32> = Vec::new();

    // --- Step B: rest-defense floor -- the N deepest-positioned
    // defenders hold their own anchor, real football's deliberate
    // balance so a turnover doesn't instantly expose an empty defense.
    let mut defenders: Vec<Cand> = pool
        .iter()
        .filter(|c| c.position_group == PlayerFieldPositionGroup::Defender)
        .copied()
        .collect();
    defenders.sort_by(|a, b| {
        let da = (a.position - own_goal).magnitude();
        let db = (b.position - own_goal).magnitude();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    for d in defenders.iter().take(REST_DEFENSE_COUNT) {
        results.push(Assignment {
            id: d.id,
            duty: AttackDuty::RestDefense,
            target: d.start_position,
        });
        assigned.push(d.id);
    }

    // --- Step C: width -- any remaining wide-slotted player holds the
    // flank, tracking the ball's own advancement.
    let target_x_from_depth = |depth: f32| own_goal.x + attack_dir * depth;
    let wide_cands: Vec<Cand> = pool
        .iter()
        .filter(|c| !assigned.contains(&c.id) && is_wide(c.position_type))
        .copied()
        .collect();
    // Real football staggers multiple wide players on the SAME flank by
    // depth (a fullback overlaps/underlaps BEHIND the winger, doesn't
    // stand on top of him) -- without this, e.g. DefenderLeft and
    // ForwardLeft both wide-slotted collapsed onto the literal identical
    // target point (verified via raw trace, 2026-09-23: mean distance
    // from realized position to target was 167.8u for Width duty, far
    // worse than RestDefense/Advance's 37-43u, traced to exactly this
    // collision), the same anti-clustering violation domain 1's own
    // 2026-09-12 rollback exists to prevent.
    let base_depth = (ball_dist_to_own_goal * WIDTH_PUSH_FACTOR).clamp(WIDTH_MIN_DEPTH, WIDTH_MAX_DEPTH);
    for c in wide_cands.iter() {
        let (offset, min_d, max_d) = match c.position_group {
            PlayerFieldPositionGroup::Forward => (0.0, WIDTH_MIN_DEPTH, WIDTH_MAX_DEPTH),
            PlayerFieldPositionGroup::Midfielder => {
                (-90.0, WIDTH_MIN_DEPTH - 60.0, WIDTH_MAX_DEPTH - 90.0)
            }
            _ => (-180.0, WIDTH_MIN_DEPTH - 120.0, WIDTH_MAX_DEPTH - 180.0),
        };
        let depth = (base_depth + offset).clamp(min_d, max_d);
        let field_height = context.field_size.height as f32;
        let y = if is_left_flank(c.position_type) {
            WIDTH_MARGIN
        } else {
            field_height - WIDTH_MARGIN
        };
        let mut target = Vector3::new(target_x_from_depth(depth), y, 0.0);
        if nearest_opp_dist(target) > PENETRATION_OPEN_THRESHOLD {
            let to_goal = opponent_goal - target;
            let dist = to_goal.magnitude();
            if dist > 1.0 {
                target += (to_goal / dist) * PENETRATION_PUSH;
            }
        }
        results.push(Assignment {
            id: c.id,
            duty: AttackDuty::Width,
            target,
        });
        assigned.push(c.id);
    }

    // --- Step D: advancing line -- every remaining central player
    // (midfielders, and defenders beyond the rest-defense floor) tracks
    // the ball's depth as a coordinated line, the direct attacking-phase
    // counterpart of domain 1's own defensive-line formula.
    let depth = (ball_dist_to_own_goal * ADVANCE_PUSH_FACTOR).clamp(ADVANCE_MIN_DEPTH, ADVANCE_MAX_DEPTH);
    let target_x = target_x_from_depth(depth);
    for c in pool.iter().filter(|c| !assigned.contains(&c.id)) {
        let y = c.start_position.y * (1.0 - ADVANCE_WIDTH_PULL) + field.ball.position.y * ADVANCE_WIDTH_PULL;
        let mut target = Vector3::new(target_x, y, 0.0);
        if nearest_opp_dist(target) > PENETRATION_OPEN_THRESHOLD {
            let to_goal = opponent_goal - target;
            let dist = to_goal.magnitude();
            if dist > 1.0 {
                target += (to_goal / dist) * PENETRATION_PUSH;
            }
        }
        results.push(Assignment {
            id: c.id,
            duty: AttackDuty::Advance,
            target,
        });
    }

    #[cfg(feature = "match-logs")]
    for a in &results {
        if let Some(c) = pool.iter().find(|c| c.id == a.id) {
            crate::match_log_info!(
                "ATTACKSUPPORT t={} id={} pt={:?} duty={:?} pos_x={:.1} pos_y={:.1} target_x={:.1} target_y={:.1}",
                context.total_match_time,
                a.id,
                c.position_type,
                a.duty,
                c.position.x,
                c.position.y,
                a.target.x,
                a.target.y
            );
        }
    }

    for a in results {
        if let Some(idx) = field.player_index(a.id) {
            field.players[idx].ai_attack_duty = a.duty;
            field.players[idx].ai_attack_target = a.target;
        }
    }
}
