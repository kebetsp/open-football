//! Off-ball support spacing (§10.3, wishlist #17).
//!
//! A shared refinement pass over a state's own heuristic support target:
//! score candidate positions around the proposed target so support runs
//! spread out (maximise collectively covered space) and favour positions
//! that open a longer, cleaner passing lane from the ball carrier — rather
//! than bunching a trivial couple of metres from the ball. The proven
//! forward `find_optimal_free_zone` scoring terms are reused here in a
//! role-neutral form; the caller's heuristic target anchors the search so
//! each role's zone discipline and tactical intent stay intact.

use crate::{MatchTacticType, PlayerFieldPositionGroup, PlayerPositionType, TACTICS_POSITIONS};
use crate::r#match::player::strategies::common::players::ops::on_ball_value;
use crate::r#match::{
    BallSideZone, MatchPlayerLite, PitchLane, PlayerSide, StateProcessingContext,
};
use nalgebra::Vector3;

/// Milestone 7 (possession-decision-intelligence PRD) — which flank a
/// wide-slotted role belongs to. Central roles (including GK/sweeper/
/// defensive mid) return `None` — they don't participate in flank
/// rotation, only genuinely wide-slotted positions do.
#[derive(PartialEq, Clone, Copy)]
pub(crate) enum Side {
    Left,
    Right,
}

/// True when `tactic`'s own canonical position table (`TACTICS_POSITIONS`,
/// the engine's authoritative formation→slots mapping) fields both
/// `ForwardLeft` and `ForwardRight` with no `ForwardCenter`/`Striker`
/// between them — a genuine front-two of central strikers (T442, T352),
/// where `ForwardLeft`/`ForwardRight` denote which strike partner, not a
/// wide winger. Under a formation that also fields a central forward
/// (T433: `ForwardLeft`, `ForwardCenter`, `ForwardRight`), the same two
/// tags genuinely are wide wingers. `PlayerPositionType` has no separate
/// variant for "central striker, left/right of a strike partner" (only
/// Left/Center/Right for the whole Forward group), so this reads the
/// tactic's own table rather than hardcoding a tactic list — found via
/// raw position-data trace (docs/ball-distribution-fix-plan.md §10
/// thread 1): under T442, both teams' strikers were being pulled to the
/// touchline by every `flank_side` consumer (Step 1's `shape_rules`,
/// Step 2's stretch-credit, and the width-hold override), plausibly
/// explaining the goals/match drop found while re-verifying the override.
pub(crate) fn is_front_two_tactic(tactic: MatchTacticType) -> bool {
    let Some((_, positions)) = TACTICS_POSITIONS.iter().find(|(t, _)| *t == tactic) else {
        return false;
    };
    let has_left = positions.contains(&PlayerPositionType::ForwardLeft);
    let has_right = positions.contains(&PlayerPositionType::ForwardRight);
    let has_center_forward = positions.contains(&PlayerPositionType::ForwardCenter)
        || positions.contains(&PlayerPositionType::Striker);
    has_left && has_right && !has_center_forward
}

/// The player's own team's starting tactic, for disambiguating
/// `ForwardLeft`/`ForwardRight` in `flank_side` (see `is_front_two_tactic`).
/// Falls back to `T442` — the front-two case — on the (should-never-
/// happen in this pipeline) `None` case, so an unexpected gap in match
/// setup fails toward NOT dragging a central striker wide, not the other
/// way around.
pub(crate) fn own_tactic(ctx: &StateProcessingContext) -> MatchTacticType {
    let is_home = ctx.player.team_id == ctx.context.field_home_team_id;
    let starting = if is_home {
        ctx.context.starting_home_tactic
    } else {
        ctx.context.starting_away_tactic
    };
    starting.unwrap_or(MatchTacticType::T442)
}

/// `pub(crate)`: also reused by `flow::field::setup_player_on_field`
/// (docs/ball-distribution-fix-plan.md §9 Step 1) to identify which
/// wide-slotted roles get a default structural width rule, and by
/// `processor.rs`'s width-hold override (§10 thread 1). Behavior
/// unchanged for every existing caller for Defender/Midfielder/
/// AttackingMidfielder/Wingback — those are unambiguously wide in every
/// tactic this engine supports. Forward is the one ambiguous group; see
/// `is_front_two_tactic`.
pub(crate) fn flank_side(pt: PlayerPositionType, tactic: MatchTacticType) -> Option<Side> {
    match pt {
        PlayerPositionType::DefenderLeft
        | PlayerPositionType::MidfielderLeft
        | PlayerPositionType::AttackingMidfielderLeft
        | PlayerPositionType::WingbackLeft => Some(Side::Left),
        PlayerPositionType::DefenderRight
        | PlayerPositionType::MidfielderRight
        | PlayerPositionType::AttackingMidfielderRight
        | PlayerPositionType::WingbackRight => Some(Side::Right),
        PlayerPositionType::ForwardLeft if !is_front_two_tactic(tactic) => Some(Side::Left),
        PlayerPositionType::ForwardRight if !is_front_two_tactic(tactic) => Some(Side::Right),
        _ => None,
    }
}

/// The teammate who shares my flank (e.g. the winger ahead of a wide
/// fullback, or vice versa) — `None` if I'm not in a wide-slotted role
/// myself, or if nobody else on the team is on the same flank. Nearest
/// by distance is the tie-break when a formation happens to have more
/// than one same-side wide role.
fn flank_partner(ctx: &StateProcessingContext) -> Option<MatchPlayerLite> {
    let tactic = own_tactic(ctx);
    let my_side = flank_side(ctx.player.tactical_position.current_position, tactic)?;
    ctx.players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
        .filter(|t| flank_side(t.tactical_positions, tactic) == Some(my_side))
        .min_by(|a, b| {
            let da = (a.position - ctx.player.position).magnitude();
            let db = (b.position - ctx.player.position).magnitude();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Milestone 5 (possession-decision-intelligence PRD) — weight for the
/// reachability term in `score_candidate`. `pass_value_from`'s documented
/// ~[0,1.2] range times this weight lands in the same tens-scale ballpark
/// as the existing lane-clearance-plus-progress combination it
/// complements (~[-10,+35]) — a reasoned starting estimate, calibrated
/// against real match-logs traces before trusting it, same discipline as
/// every other unsourced weight in this file.
const REACHABILITY_WEIGHT: f32 = 40.0;

/// Milestone 6 — weight for the decoy-value term. `danger * marking_pressure`
/// is already a product of two [0,1] terms (typically smaller than
/// reachability alone), so this starts at the same tens-scale target.
const DECOY_WEIGHT: f32 = 35.0;

/// Milestone 7 — weight for the flank-rotation term: same tens-scale
/// target as Milestones 5/6's new terms. The "take the vacant wide role"
/// bonus is deliberately half this magnitude (a nudge, not a mandate) —
/// see `rotation_adjustment`.
const ROTATION_WEIGHT: f32 = 20.0;

/// Milestone 12 — weight for weak-side width-alignment. Smaller than
/// reachability/decoy (this is a positioning-readiness nudge, not a
/// primary value term) but still large enough to matter against the
/// tether/repulsion terms it competes with — same tens-scale reasoning
/// as the other milestone weights in this file.
const WEAK_SIDE_PATIENCE_WEIGHT: f32 = 18.0;

/// P1 (A1, `docs/attacking-defensive-decision-rebuild-plan.md` §2.1/
/// §2.2) — off-ball counterpart to `on_ball_value::lane_occupancy_
/// adjustment` (same rule — Guardiola's "no more than
/// `on_ball_value::LANE_OCCUPANCY_CAPACITY` teammates in the same
/// vertical lane" — reuses that shared capacity constant rather than
/// re-deriving it), scored on this file's own tens-scale rather than
/// `carry_value`'s roughly [-1,+1.5] scale. Tens-scale reasoning matches
/// `ROTATION_WEIGHT`/`WEAK_SIDE_PATIENCE_WEIGHT` above — a nudge among
/// several competing terms, not a dominant one.
const LANE_OCCUPANCY_PENALTY_WEIGHT: f32 = 22.0;
/// Reward for a candidate that vacates an already-at-capacity current
/// lane. Smaller than the penalty, same "nudge not mandate" reasoning
/// as `on_ball_value::LANE_VACATE_BONUS`.
const LANE_VACATE_BONUS_WEIGHT: f32 = 10.0;

/// D3 (`docs/attacking-defensive-decision-rebuild-plan.md` §2.3) —
/// central-corridor-priority default shape. Real doctrine (research
/// doc §9.1): the centre is defended harder than the flanks for a
/// geometric reason (360° vs. touchline-cut angle) — real defenders
/// default toward protecting the inner lanes independent of the ball's
/// current position. Implemented here as the value-comparison
/// counterpart to Phase 5's directive-only `ShapeAction::TuckInside`
/// (rather than retrofitting that position-mutation machinery to
/// auto-populate for every defender — a much larger-blast-radius
/// change touching the shared per-tick shape-rules loop in
/// `engine/run.rs` for a result this file's own existing architecture
/// already achieves more safely). This is the concrete mechanism for
/// "some general bias toward common football logic should remain"
/// (Pavel's own phrase in the rebuild plan's charter) — an explicit
/// player directive (`ShapeAction`/a future `defensiveBias` field) can
/// still amplify or override a defender's positioning; this only sets
/// the undirected default.
const CENTRAL_CORRIDOR_PRIORITY_WEIGHT: f32 = 15.0;

/// P5 (A3, `docs/attacking-defensive-decision-rebuild-plan.md` §2.1/
/// §2.2) — mobility/residency bonus. Real coaching drills "never
/// predictable or repetitive" (research doc §8.1) — this rewards a
/// candidate in a DIFFERENT lane than the one the player has held for
/// a while, using P5's new `MatchPlayer.lane_residency_ticks` counter
/// (nothing in the engine measured lane repetition before this).
/// Fades in over a real duration (no bonus until genuinely stale, full
/// bonus only after a further stretch) rather than firing the instant
/// residency ticks over 0 — brief, incidental lane occupancy shouldn't
/// trigger a "move now" signal.
/// Tick-to-real-time note (Core Rule 3): 1 engine tick = 10ms of engine
/// wall-clock, but the whole match is itself compressed ~9x (90 real
/// minutes -> ~10 engine minutes), so a tick count must be multiplied
/// by that 9x factor to read as REAL football time, not just divided
/// by 100. `THRESHOLD_TICKS`=300 -> 3 engine-seconds -> ~27 real
/// seconds; `FULL_TICKS`=900 -> ~81 real seconds (~1.35 real minutes)
/// before the bonus is fully ramped. An earlier version of this
/// comment mislabelled these as "~3s"/"~9s" — the exact unit-
/// conversion mistake this project's own Core Rule 3 warns about,
/// caught and corrected during this phase's own verification.
const MOBILITY_RESIDENCY_THRESHOLD_TICKS: u16 = 300;
const MOBILITY_RESIDENCY_FULL_TICKS: u16 = 900;
const MOBILITY_BONUS_WEIGHT: f32 = 12.0;

/// docs/principles-of-play-phase2-plan.md §4 (2026-08-31) — same-binary
/// ablation toggle for the "Mobility" principle's four terms
/// (`mobility_adjustment`, `rotation_adjustment`, `third_man_adjustment`,
/// the inline decoy-value term in `score_candidate`), so a clean before/
/// after comparison can run without a build-difference confound (same
/// idiom as `flow::context`'s `OF_SCORE_BLIND` env toggle). Diagnostic
/// only — inert unless `MOBILITY_ABLATE` is set, and not read anywhere
/// else in this file.
fn mobility_ablated() -> bool {
    use std::sync::OnceLock;
    static ABLATED: OnceLock<bool> = OnceLock::new();
    *ABLATED.get_or_init(|| std::env::var("MOBILITY_ABLATE").is_ok())
}

/// Scoped to non-defenders only — mirrors D3's defender-only gate, in
/// reverse. Measured (A/B, this phase's own verification): applying
/// this term to defenders too pushed goals/match to ~5.3-6.2, a real,
/// repeated regression against the 3.0-4.5 band. Root cause: a
/// defender holding a settled defensive lane for 27-81 real seconds is
/// tactically CORRECT discipline, not the "predictable, repetitive"
/// staleness §8.1's mobility doctrine actually targets (that doctrine
/// is about ATTACKING positional variety) — penalising it nudged
/// defenders out of their own defensive shape.
fn mobility_adjustment(ctx: &StateProcessingContext, candidate: Vector3<f32>) -> f32 {
    let is_defender = ctx
        .player
        .tactical_position
        .current_position
        .position_group()
        == PlayerFieldPositionGroup::Defender;
    if is_defender {
        return 0.0;
    }
    let field_h = ctx.context.field_size.height as f32;
    let candidate_lane_idx = PitchLane::for_y(field_h, candidate.y).index() as u8;
    if candidate_lane_idx == ctx.player.lane_residency_lane {
        return 0.0;
    }
    let residency = ctx.player.lane_residency_ticks as f32;
    let progress = ((residency - MOBILITY_RESIDENCY_THRESHOLD_TICKS as f32)
        / (MOBILITY_RESIDENCY_FULL_TICKS - MOBILITY_RESIDENCY_THRESHOLD_TICKS) as f32)
        .clamp(0.0, 1.0);
    // P6 — AI lever: `spatialBias.mobility`, 1.0 = neutral.
    progress * MOBILITY_BONUS_WEIGHT * ctx.player.spatial_mobility
}

/// P5 (A4) — vacated-space / third-man credit (research doc §8.5): a
/// decoy run drags a marker away, and a THIRD player — not the decoy
/// himself — exploits the space that opens up. Detects a genuinely
/// marked decoy among the OTHER teammates (an opponent glued within
/// close range), then rewards this candidate if it's a real attacking
/// threat (same danger geometry Milestone 6's own decoy term already
/// uses) with no local congestion — the space is open specifically
/// because a defender is tied up elsewhere, not because nobody bothered
/// marking anyone.
/// Measured (temporary diagnostic, removed after this fix): the
/// original 15.0u radius with no other qualifier made `has_marked_decoy`
/// true on 56.8% of all candidate evaluations (n=270,630) — "some
/// teammate somewhere on the pitch has an opponent nearby" is just
/// ordinary match congestion, not a genuine decoy event. Tightened to a
/// real tight-marking distance (matching the scale of other close-
/// engagement radii already used in this file/D1, ~1.25m) AND gated to
/// a teammate in the ATTACKING HALF — a marked defender in his own
/// third isn't "dragging a marker to create space for a third-man run"
/// in any real football sense.
const THIRD_MAN_MARK_RADIUS: f32 = 10.0;
const THIRD_MAN_LOCAL_CLEAR_RADIUS: f32 = 25.0;
const THIRD_MAN_WEIGHT: f32 = 20.0;

fn third_man_adjustment(
    ctx: &StateProcessingContext,
    candidate: Vector3<f32>,
    opponents: &[Vector3<f32>],
    teammates: &[(u32, Vector3<f32>)],
) -> f32 {
    let field_w = ctx.context.field_size.width as f32;
    let attacking_dir: f32 = match ctx.player.side {
        Some(PlayerSide::Left) => 1.0,
        Some(PlayerSide::Right) => -1.0,
        None => return 0.0,
    };
    let midfield_x = field_w / 2.0;
    let has_marked_decoy = teammates.iter().any(|(_, pos)| {
        let is_advanced = (pos.x - midfield_x) * attacking_dir > 0.0;
        is_advanced
            && opponents
                .iter()
                .any(|opp| (opp - pos).magnitude() < THIRD_MAN_MARK_RADIUS)
    });
    if !has_marked_decoy {
        return 0.0;
    }
    let locally_clear = !opponents
        .iter()
        .any(|opp| (opp - candidate).magnitude() < THIRD_MAN_LOCAL_CLEAR_RADIUS);
    if !locally_clear {
        return 0.0;
    }
    let gk_pos = ctx
        .players()
        .opponents()
        .goalkeeper()
        .next()
        .map(|g| g.position);
    let danger =
        (on_ball_value::effective_open_angle(ctx, candidate, gk_pos) / 1.31).clamp(0.0, 1.0);
    danger * THIRD_MAN_WEIGHT
}

/// Pressure-relief support ("show for the ball" — wishlist item
/// "pressure-sensitive spread distance", implemented here on top of the
/// possession-decision-intelligence architecture rather than the old
/// dead-code `is_ball_holder_under_pressure`/`CheckToFeet` scaffolding it
/// originally lived in). Weighted well above the other milestone terms —
/// measured via a match-logs score trace that a lower (tens-scale, ~30)
/// weight left close candidates scoring a mean of ~7-8 under genuine
/// pressure against ~27-34 for 80-200u candidates, i.e. structurally
/// unable to win even when a genuinely open near pocket existed.
const PRESSURE_RELIEF_WEIGHT: f32 = 70.0;

/// Range within which a close option can plausibly act as an immediate
/// out-ball. Deliberately wider than `SHORT_SUPPORT_RADIUS` (28u) so the
/// two terms overlap smoothly: inside 28u the redundancy penalty is
/// being relaxed at the same time the relief bonus ramps up, rather than
/// the bonus only starting exactly where the penalty stops.
const PRESSURE_RELIEF_RADIUS: f32 = 45.0;

/// Realism-bug (2026-08-03): a single opponent standing directly in the
/// ball holder's path forward (jockeying/blocking range, within a
/// forward-facing cone) — distinct from `pressure_intensity_for`'s
/// aggregate multi-opponent congestion measure, which needs roughly 3
/// close opponents to cross its own 0.5 engagement floor and therefore
/// never engages for a lone marker simply blocking the route forward.
/// This is the specific, everyday defensive pattern Pavel reported: the
/// carrier can't advance (blocked ahead) and can't retreat either, no
/// teammate ever drops behind the ball line to offer a reset, and the
/// stalemate resolves via an unrealistic full-power long clearance.
/// Deliberately purely geometric (no movement history) — the fix
/// attempted via `spacing_stats`/cluster-freeze detection (2026-07-30)
/// targeted near-zero-movement clusters and, per Pavel, missed exactly
/// this "barely wiggling, not literally frozen" case; a snapshot check
/// of "is someone blocking my forward cone right now" can't miss it
/// regardless of how much the carrier shuffles in place.
const BLOCK_RADIUS: f32 = 18.0;
/// cos(60°) ≈ 0.5 — the opponent must sit within roughly a 60° cone
/// dead ahead of the holder along the attacking direction, not merely
/// beside or behind him.
const BLOCK_FORWARD_COS: f32 = 0.5;

/// True when at least one opponent is within `BLOCK_RADIUS` of the ball
/// holder, roughly in front of him along his team's attacking direction
/// — i.e. genuinely blocking his route forward, not just nearby.
fn carrier_blocked_ahead(
    ctx: &StateProcessingContext,
    holder: &MatchPlayerLite,
    opponents: &[Vector3<f32>],
) -> bool {
    let attacking_dir: f32 = match ctx.player.side {
        Some(PlayerSide::Left) => 1.0,
        Some(PlayerSide::Right) => -1.0,
        None => return false,
    };
    opponents.iter().any(|opp| {
        let to_opp = opp - holder.position;
        let dist = (to_opp.x * to_opp.x + to_opp.y * to_opp.y).sqrt();
        if !(1.0..=BLOCK_RADIUS).contains(&dist) {
            return false;
        }
        (to_opp.x / dist) * attacking_dir > BLOCK_FORWARD_COS
    })
}

/// Radius inside which a candidate is penalised per nearby teammate —
/// same 90u repulsion radius the forward zone scorer uses.
const TEAMMATE_REPULSION_RADIUS: f32 = 90.0;
/// Weight of the summed linear teammate repulsion (forward scorer value).
const TEAMMATE_REPULSION_WEIGHT: f32 = 50.0;
/// Support closer to the holder than this is "short and redundant" —
/// deprioritised unless a genuine combination (link play) is in progress.
const SHORT_SUPPORT_RADIUS: f32 = 28.0;
/// Candidates further from the proposed target pay a mild tether cost so
/// the refinement repositions within the state's intent, never wanders.
const TETHER_WEIGHT: f32 = 0.15;

/// Step 2 (docs/ball-distribution-fix-plan.md §9, 2026-08-26) — same
/// scale family as `WEAK_SIDE_PATIENCE_WEIGHT` (a comparable "reward
/// realistic positioning" nudge), NOT `on_ball_value.rs`'s own [0,1.2]
/// scale — `central_congestion_discount`'s raw 0.08 would be negligible
/// on this file's own tens-scale scoring, the exact scale-mismatch trap
/// this project has hit before (`SHOOT_COMPARISON_SCALE` etc.).
const STRETCH_CREDIT_WEIGHT: f32 = 18.0;
/// Matches `teamplay::tactical::spatial_sensor::WIDE_MARKING_PROXIMITY_
/// RADIUS` (32.0) — a real spatial distance, not a value-scale term, so
/// reused directly rather than re-derived; hardcoded here (not imported)
/// to keep this addition self-contained and isolated.
const STRETCH_CREDIT_RADIUS: f32 = 32.0;
const WIDTH_PERSISTENCE_RADIUS: f32 = 40.0;
const WIDTH_PERSISTENCE_WEIGHT: f32 = 12.0;

/// §11.9 hard exclusion radius (~3m at the ~8u/m conversion): no two
/// teammates may target the same spot or the same small area around it.
/// Unlike the soft repulsion above, a candidate inside this radius of a
/// claimed point is EXCLUDED outright, not merely down-scored — the
/// penalty approach alone demonstrably failed to stop actual overlap.
pub const EXCLUSION_RADIUS: f32 = 24.0;
/// Run-destination projection horizon in engine ticks (~0.8s at 10ms/tick;
/// velocities are u/tick). A moving player "owns" the space he is running
/// into, not just the spot he stands on.
const PROJECTION_TICKS: f32 = 80.0;

/// Points already claimed by the rest of the team (§11.9): the ball
/// carrier's projected run destination, plus every other teammate's
/// current position AND projected destination.
pub fn claimed_points(ctx: &StateProcessingContext) -> Vec<Vector3<f32>> {
    let mut pts = Vec::with_capacity(24);
    if let Some(h) = ball_holder(ctx) {
        pts.push(h.position + h.velocity(ctx) * PROJECTION_TICKS);
    }
    for t in ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
    {
        let v = t.velocity(ctx);
        pts.push(t.position);
        pts.push(t.position + v * PROJECTION_TICKS);
    }
    pts
}

/// Diagnostic counters for the §11.9 exclusion (match-logs builds only) —
/// same opt-in pattern as `tackle_stats`. Proves the hard rule actually
/// fires at the mechanism level, since aggregate movement proxies are
/// dominated by paths this rule deliberately doesn't touch.
#[cfg(feature = "match-logs")]
pub mod spacing_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Candidates dropped by the hard exclusion.
    pub static EXCLUDED: AtomicU64 = AtomicU64::new(0);
    /// RunningInBehind targets shifted out of the carrier's space.
    pub static CARRIER_SHIFTS: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        EXCLUDED.store(0, Ordering::Relaxed);
        CARRIER_SHIFTS.store(0, Ordering::Relaxed);
    }

    pub fn snapshot() -> [u64; 2] {
        [
            EXCLUDED.load(Ordering::Relaxed),
            CARRIER_SHIFTS.load(Ordering::Relaxed),
        ]
    }
}

/// True when `candidate` falls inside the hard exclusion radius of any
/// claimed point.
pub fn violates_exclusion(candidate: Vector3<f32>, claimed: &[Vector3<f32>]) -> bool {
    let r_sq = EXCLUSION_RADIUS * EXCLUSION_RADIUS;
    let hit = claimed
        .iter()
        .any(|c| (c - candidate).norm_squared() < r_sq);
    #[cfg(feature = "match-logs")]
    if hit {
        spacing_stats::EXCLUDED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    hit
}

/// §11.9 carrier-space guard for direct run targets (e.g. RunningInBehind,
/// which doesn't go through candidate scoring): if `target` converges on
/// the space the ball carrier is already running into, shift it laterally
/// to just outside the exclusion radius. The carrier owns that space.
pub fn avoid_carrier_space(
    ctx: &StateProcessingContext,
    target: Vector3<f32>,
) -> Vector3<f32> {
    let Some(h) = ball_holder(ctx) else {
        return target;
    };
    if ctx.player.link_target == Some(h.id) {
        return target; // a one-two deliberately plays into the carrier's space
    }
    let carrier_dest = h.position + h.velocity(ctx) * PROJECTION_TICKS;
    let d = target - carrier_dest;
    let dist_sq = d.x * d.x + d.y * d.y;
    if dist_sq >= EXCLUSION_RADIUS * EXCLUSION_RADIUS {
        return target;
    }
    #[cfg(feature = "match-logs")]
    spacing_stats::CARRIER_SHIFTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let field_height = ctx.context.field_size.height as f32;
    let away: f32 = if target.y >= carrier_dest.y { 1.0 } else { -1.0 };
    let y = (carrier_dest.y + away * (EXCLUSION_RADIUS + 4.0)).clamp(15.0, field_height - 15.0);
    Vector3::new(target.x, y, 0.0)
}

/// §12.5 — universal off-ball separation, every phase (extends §11.9,
/// which only guarded attacking-support target selection). A gentle
/// repulsion velocity between same-team outfielders standing inside the
/// exclusion radius of each other, applied at the processor level so it
/// covers defensive shape, out-of-possession shifts, and every other
/// off-ball path without touching each state's own logic.
///
/// Deliberately NOT applied to: restarts/set-piece shapes (walls stand
/// 9u apart by design — gated on an OpenPlay origin), the ball carrier
/// or anyone within 60u of the ball (receivers, duels, pressing),
/// goalkeepers, and link-play pairs (a one-two needs close support).
/// Returns `(separation_velocity, blend_weight)`. The caller blends
/// `state_velocity * (1-w) + separation * w` — an additive-only nudge
/// demonstrably loses to the state's own convergence pull (two players
/// walking to near-identical anchors equilibrate a few units apart, or
/// fully overlapped), so at close range the separation must progressively
/// REPLACE the state's movement, not fight it. Asymmetric by id: the
/// higher-id player yields (full weight), the lower-id player mostly
/// holds — one of the two owns the contested zone, mirroring how real
/// players resolve a shared space.
pub fn separation_nudge(ctx: &StateProcessingContext) -> Option<(Vector3<f32>, f32)> {
    use crate::r#match::PassOriginRestart;

    let ball_meta = &ctx.tick_context.ball;
    // Skip only while a set piece is STAGED (restart origin + a taker
    // holding the ball): walls and blocks stand deliberately tight. A
    // restart origin alone isn't enough to skip — origins persist through
    // the delivery flight and loose scrambles for many seconds, which
    // disabled separation exactly when post-set-piece clusters form.
    if ball_meta.pass_origin_restart != PassOriginRestart::OpenPlay && ball_meta.is_owned {
        return None;
    }
    if ctx
        .player
        .tactical_position
        .current_position
        .is_goalkeeper()
    {
        return None;
    }
    if ball_meta.current_owner == Some(ctx.player.id) {
        return None;
    }
    if ctx.ball().distance() < 60.0 {
        return None;
    }

    /// Outward speed at full blend (~comfortable jog; velocities are u/tick).
    const SEPARATION_SPEED: f32 = 0.30;

    let mut dir_sum = Vector3::zeros();
    let mut w_max = 0.0f32;
    for t in ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
    {
        if ctx.player.link_target == Some(t.id) {
            continue;
        }
        if ball_meta.current_owner == Some(t.id) {
            continue;
        }
        let d = ctx.player.position - t.position;
        let dist = (d.x * d.x + d.y * d.y).sqrt();
        if dist >= EXCLUSION_RADIUS {
            continue;
        }
        // Real separation doctrine, not a straight-line retreat/advance
        // -- Pavel's direct report, 2026-09-24, same investigation as
        // the Action::Hold shielding fix (processor.rs): a real player
        // creates space from a crowded teammate by SIDESTEPPING, not by
        // walking straight backward (or forward) along the goal axis.
        // "Away from teammate X" has no notion of which direction that
        // happens to be, and a crowding teammate positioned ahead of
        // this player (the common case — teammates bunch up in the
        // direction of play) pushed him straight toward his own goal
        // for as long as the crowding persisted, at up to full
        // replacement weight (`w` reaching 1.0 inside ~8u). The pitch's
        // x-axis is the goal-to-goal (depth) axis throughout this
        // engine, y the lateral (width) axis — strip the x-component
        // from each teammate's "away from him" direction, keeping only
        // the lateral push; depth is the underlying state/duty's job,
        // not this anti-overlap mechanic's. Falls back to the same
        // deterministic id-based lateral split already used for an
        // exact overlap when the crowding teammate sits almost
        // directly on the goal axis (no real lateral escape either).
        let dir = if dist > 0.5 {
            let lateral = Vector3::new(0.0, d.y, 0.0);
            let lateral_dist = lateral.y.abs();
            if lateral_dist > 0.5 {
                lateral / lateral_dist
            } else if ctx.player.id > t.id {
                Vector3::new(0.0, 1.0, 0.0)
            } else {
                Vector3::new(0.0, -1.0, 0.0)
            }
        } else if ctx.player.id > t.id {
            // Deterministic split for exact overlap.
            Vector3::new(0.0, 1.0, 0.0)
        } else {
            Vector3::new(0.0, -1.0, 0.0)
        };
        // 1.5× slope: full takeover inside ~8u, tapering to zero at the
        // exclusion radius. Equilibrium against a typical 0.3 u/tick
        // state pull lands around 16u (~2m) — outside visual overlap.
        let closeness = (1.5 * (1.0 - dist / EXCLUSION_RADIUS)).clamp(0.0, 1.0);
        let yield_factor = if ctx.player.id > t.id { 1.0 } else { 0.4 };
        let w = closeness * yield_factor;
        dir_sum += dir * w;
        w_max = w_max.max(w);
    }
    if w_max < 1e-3 || dir_sum.norm_squared() < 1e-6 {
        return None;
    }
    Some((dir_sum.normalize() * SEPARATION_SPEED, w_max))
}

/// Refine a state's proposed off-ball support target by scoring candidates
/// around it. Returns the best-scoring position (possibly `proposed`
/// itself). Callers pass the target their own heuristics produced.
///
/// Skipped (returns `proposed` unchanged) when the player is in a genuine
/// combination with the ball holder (`link_target` pins them together) —
/// a one-two needs close support, and that case must not be broken.
pub fn refine_support_position(
    ctx: &StateProcessingContext,
    proposed: Vector3<f32>,
) -> Vector3<f32> {
    let holder = ball_holder(ctx);

    if let Some(h) = &holder {
        if ctx.player.link_target == Some(h.id) {
            return proposed;
        }
    }

    let field_width = ctx.context.field_size.width as f32;
    let field_height = ctx.context.field_size.height as f32;
    let goal_pos = ctx.player().opponent_goal_position();

    let opponents: Vec<Vector3<f32>> = ctx
        .players()
        .opponents()
        .all()
        .map(|o| o.position)
        .collect();
    let teammates: Vec<(u32, Vector3<f32>)> = ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
        .map(|t| (t.id, t.position))
        .collect();

    // §11.9 hard exclusion: candidates inside ~3m of the carrier's run
    // destination or any teammate's position/destination are dropped
    // outright before scoring. The soft repulsion below still shapes
    // preferences among the survivors.
    let claimed = claimed_points(ctx);

    // Candidate ring around the proposed target (25 positions), plus the
    // player's current position so "stay put" competes fairly.
    let mut best = proposed;
    let mut best_score = f32::MIN;
    // Fallback if EVERY candidate is excluded (dense box, all space
    // claimed): best-scoring candidate regardless of exclusion beats
    // returning an unvetted `proposed`.
    let mut best_any = proposed;
    let mut best_any_score = f32::MIN;
    let mut found_valid = false;
    let offsets: [f32; 5] = [-50.0, -25.0, 0.0, 25.0, 50.0];
    let mut consider = |candidate: Vector3<f32>| {
        let clamped = Vector3::new(
            candidate.x.clamp(20.0, field_width - 20.0),
            candidate.y.clamp(20.0, field_height - 20.0),
            0.0,
        );
        let score = score_candidate(
            ctx, clamped, proposed, &holder, goal_pos, &opponents, &teammates,
        );
        if score > best_any_score {
            best_any_score = score;
            best_any = clamped;
        }
        if violates_exclusion(clamped, &claimed) {
            return;
        }
        found_valid = true;
        if score > best_score {
            best_score = score;
            best = clamped;
        }
    };
    for dx in offsets {
        for dy in offsets {
            consider(proposed + Vector3::new(dx, dy, 0.0));
        }
    }
    consider(ctx.player.position);

    // Pressure-relief candidates: the caller's own baseline `proposed`
    // target is a SUPPORT/SPACING point (typically 80-180u from the ball
    // holder for these states — build-up outlets, box runs, wide
    // stretch), so the ±50u ring above almost never produces a candidate
    // close enough to the holder for the pressure-relief terms in
    // `score_candidate` to ever matter — confirmed empirically via a
    // match-logs trace (selected distance-to-holder was statistically
    // flat across every pressure bucket before this fix, ~130-140u
    // regardless of pressure). Mirrors `carry_candidates`'s own
    // byline-candidate precedent (2026-07-25): when a real pattern has
    // no candidate representing it, add one explicitly rather than hope
    // the general ring reaches it. Genuinely inert (loop doesn't run)
    // whenever the holder isn't under real pressure.
    // Realism-bug (2026-08-03): whether a lone opponent is blocking the
    // holder's route forward — see `carrier_blocked_ahead`'s own doc
    // comment. OR'd into the existing pressure gate below (rather than
    // replacing it) so both the pre-existing multi-opponent-congestion
    // case and this single-marker-block case reach the same relief
    // machinery.
    let blocked_ahead = holder
        .as_ref()
        .map(|h| carrier_blocked_ahead(ctx, h, &opponents))
        .unwrap_or(false);

    if let Some(h) = &holder {
        let holder_pressure = ctx.player().pressure().pressure_intensity_for(h.id);
        // Same 0.5 engagement floor as `score_candidate` — mild pressure
        // is common and shouldn't trigger extra candidate injection that
        // would only ever score at the (also-gated) baseline anyway.
        // `blocked_ahead` is a second, independent way in: a lone marker
        // directly blocking the route forward never crosses the 0.5
        // aggregate-pressure floor (it needs ~3 close opponents), so
        // without this the whole relief mechanism silently never
        // engaged for that specific, common case.
        if holder_pressure > 0.5 || blocked_ahead {
            // Two radii, not one: a "pressured" holder by definition has
            // opponents within ~30u (`pressure_intensity_for`'s own
            // scan radius), so a single fixed ring often lands inside
            // that same crowd too — measured via a match-logs trace
            // showing close
            // candidates scoring far below the eventual winner even with
            // the pressure-relief bonus applied. A tighter ring (genuine
            // one-touch range) and a wider one (just beyond the crowd,
            // the real "show on the far side of the marker" pocket) give
            // the scorer real open options to choose between instead of
            // one coarse sample per direction.
            const RELIEF_RING_DEG: [f32; 8] = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
            for &radius_frac in &[0.45, 0.9] {
                let relief_radius = PRESSURE_RELIEF_RADIUS * radius_frac;
                for &deg in &RELIEF_RING_DEG {
                    let rad = deg.to_radians();
                    let offset = Vector3::new(rad.cos(), rad.sin(), 0.0) * relief_radius;
                    consider(h.position + offset);
                }
            }
        }

        // Realism-bug (2026-08-03) — a genuine DEEP outlet, behind the
        // ball line, not just the close relief pocket above (which tops
        // out at 40.5u — still adjacent to the very marker doing the
        // blocking). Real football: when a lone defender cuts off the
        // route forward, a covering teammate (a deeper midfielder,
        // fullback, or centre-back) drops off to reset — a genuine
        // backward release, not a shuffle. Gated strictly on
        // `blocked_ahead` (not on aggregate pressure) since this is
        // specifically the "can't go forward" pattern, not "surrounded."
        if blocked_ahead {
            let attacking_dir: f32 = match ctx.player.side {
                Some(PlayerSide::Left) => 1.0,
                Some(PlayerSide::Right) => -1.0,
                None => 0.0,
            };
            const DEEP_OUTLET_DISTANCES: [f32; 2] = [55.0, 90.0];
            const DEEP_OUTLET_LATERAL: [f32; 3] = [-30.0, 0.0, 30.0];
            for &dist in &DEEP_OUTLET_DISTANCES {
                for &lat in &DEEP_OUTLET_LATERAL {
                    consider(h.position + Vector3::new(-attacking_dir * dist, lat, 0.0));
                }
            }
        }
    }

    if found_valid { best } else { best_any }
}

fn score_candidate(
    ctx: &StateProcessingContext,
    position: Vector3<f32>,
    proposed: Vector3<f32>,
    holder: &Option<MatchPlayerLite>,
    goal_pos: Vector3<f32>,
    opponents: &[Vector3<f32>],
    teammates: &[(u32, Vector3<f32>)],
) -> f32 {
    let mut score = 0.0f32;

    // Open space: congestion from opponents within 30u (forward scorer term).
    let mut congestion = 0.0f32;
    for opp in opponents {
        let d = (opp - position).magnitude();
        if d < 30.0 {
            congestion += (30.0 - d) / 30.0;
        }
    }
    score += (10.0 - congestion.min(10.0)) * 3.0;

    // Universal teammate repulsion — crowding ANY teammate wastes the
    // team's collective coverage (forward scorer weight).
    let crowd: f32 = teammates
        .iter()
        .map(|(_, t)| (t - position).magnitude())
        .filter(|&d| d < TEAMMATE_REPULSION_RADIUS)
        .map(|d| (TEAMMATE_REPULSION_RADIUS - d) / TEAMMATE_REPULSION_RADIUS)
        .sum::<f32>();
    score -= crowd * TEAMMATE_REPULSION_WEIGHT;

    // Pressure-relief support ("show for the ball" — wishlist item
    // "pressure-sensitive spread distance"): how surrounded the HOLDER
    // currently is, reusing the same crowding primitive Milestone 6's
    // `marking_pressure` already normalizes below — one formula, another
    // consumer, rather than inventing new proximity math. Computed once,
    // unconditionally (0.0 when there's no holder), so it can also drive
    // the tether blend at the bottom of this function, not just the
    // holder-relative terms below.
    let holder_pressure = holder
        .as_ref()
        .map(|h| ctx.player().pressure().pressure_intensity_for(h.id))
        .unwrap_or(0.0);

    // Engagement, not raw pressure: mild pressure (a single opponent
    // loosely tracking, `holder_pressure` in roughly 0.1-0.3) is by far
    // the MOST common reading in ordinary play — nudging the scoring
    // landscape on nearly every tick at that level measurably suppressed
    // goals in a regression gate (isolated via scoped git-stash: ~3.67
    // baseline vs. ~2.89 with an unconditional linear blend, consistent
    // across two independent batch pairs) even though it rarely flipped
    // the actual winning candidate — it just nudged close decisions.
    // `PRESSURE_ENGAGEMENT_FLOOR` keeps every term below it byte-for-byte
    // identical to no-pressure behaviour; only genuinely elevated
    // pressure (the same range the match-logs distance trace actually
    // showed moving selected positions) engages the mechanism at all.
    const PRESSURE_ENGAGEMENT_FLOOR: f32 = 0.5;
    let pressure_engagement =
        ((holder_pressure - PRESSURE_ENGAGEMENT_FLOOR) / (1.0 - PRESSURE_ENGAGEMENT_FLOOR))
            .clamp(0.0, 1.0);
    // Realism-bug (2026-08-03): a lone opponent directly blocking the
    // holder's route forward (see `carrier_blocked_ahead`) engages this
    // same relief machinery at full strength, independent of the
    // aggregate `pressure_engagement` above — that aggregate needs ~3
    // close opponents to ever leave 0.0, so a single blocking marker
    // (the reported case) never moved it. Full engagement here matches
    // the already-accepted design for genuine congestion: past the
    // floor, "the caller's tactical intent" shifts from holding shape
    // to getting the carrier an out-ball.
    let blocked_ahead = holder
        .as_ref()
        .map(|h| carrier_blocked_ahead(ctx, h, opponents))
        .unwrap_or(false);
    let relief_engagement = if blocked_ahead {
        1.0
    } else {
        pressure_engagement
    };

    if let Some(h) = holder {
        let holder_dist = (position - h.position).magnitude();

        // Short redundant support: a candidate a couple of metres from the
        // carrier offers a trivial pass — deprioritise it firmly. Relaxed
        // under genuine pressure: a close option stops being a redundant
        // trivial pass and becomes the actual point once the holder is
        // truly surrounded (matches the wishlist's own framing: "one
        // thing if your partner is surrounded... needs help... another
        // thing if there is space").
        if holder_dist < SHORT_SUPPORT_RADIUS {
            score -= (SHORT_SUPPORT_RADIUS - holder_dist) * 2.5 * (1.0 - relief_engagement);
        }

        // Pressure-relief bonus: reward a close, genuinely CLEAN outlet
        // specifically when the holder needs one — gated on both the
        // holder's own pressure and the candidate spot itself being
        // clear to receive at (`congestion`, already computed above for
        // the open-space term), so this never rewards converging into
        // another marked/crowded spot just because the holder is under
        // pressure elsewhere. Additive, never gates the reachability/
        // decoy/rotation/weak-side terms below — same "don't let one
        // term zero out another" discipline Milestone 6 already
        // established for reachability vs. decoy value. Zero whenever
        // pressure is below the engagement floor, so ordinary unpressed
        // circulation is untouched.
        if relief_engagement > 0.0 && holder_dist < PRESSURE_RELIEF_RADIUS {
            let closeness =
                ((PRESSURE_RELIEF_RADIUS - holder_dist) / PRESSURE_RELIEF_RADIUS).clamp(0.0, 1.0);
            let candidate_openness = (1.0 - congestion.min(10.0) / 10.0).clamp(0.0, 1.0);
            score += relief_engagement * closeness * candidate_openness * PRESSURE_RELIEF_WEIGHT;
        }

        // Passing-lane clearance from the holder, and lane QUALITY: a clear
        // lane scores, and a clear lane that progresses the ball toward goal
        // (longer, through the defence) scores more than a sideways one.
        let lane = position - h.position;
        let lane_len = lane.magnitude();
        if lane_len > 1.0 {
            let dir = lane / lane_len;
            let blocked = opponents.iter().any(|opp| {
                let to_opp = opp - h.position;
                let proj = to_opp.dot(&dir);
                if proj <= 0.0 || proj >= lane_len {
                    return false;
                }
                let lateral = to_opp - dir * proj;
                lateral.norm_squared() < 4.0 * 4.0
            });
            if blocked {
                score -= 10.0;
            } else {
                score += 15.0;
                let progress = (h.position - goal_pos).magnitude()
                    - (position - goal_pos).magnitude();
                score += (progress / 120.0).clamp(0.0, 1.0) * 20.0;
            }
        }

        // Width vs the holder stretches the shape (forward scorer term,
        // reduced — mids/defenders also stretch vertically via progress).
        let lateral_distance = (position.y - h.position.y).abs();
        if lateral_distance > 80.0 {
            score += 15.0;
        } else if lateral_distance > 40.0 {
            score += 8.0;
        }
    }

    // Milestone 5 (possession-decision-intelligence PRD) — reachability:
    // would the ACTUAL current ball holder's pass to a receiver AT THIS
    // POSITION genuinely be a good option, using the real completion-
    // probability-and-terminal-value reasoning from `on_ball_value`
    // (Milestone 3) rather than the binary 4u-corridor check above. This
    // is additive alongside that check, not a replacement — the corridor
    // check stays as a cheap hard signal, this adds the real probabilistic
    // one on top.
    if let Some(h) = holder {
        let hypothetical = MatchPlayerLite {
            id: ctx.player.id,
            position,
            tactical_positions: ctx.player.tactical_position.current_position,
        };
        let reachability = on_ball_value::pass_value_from(ctx, h.position, &hypothetical);
        score += reachability * REACHABILITY_WEIGHT;
    }

    // Milestone 6 — decoy value: credits a position for being a genuine
    // attacking threat that would force a nearby opponent to engage,
    // INDEPENDENT of whether a pass here is realistic right now (the
    // reachability term above) — "drag a defender away, even though I
    // won't receive it" as a legitimate value in its own right. Summed
    // into `score`, never gated by reachability, so a position with
    // near-zero reachability but genuine danger + marking pressure still
    // gets its full decoy contribution — this is deliberately the
    // opposite construction from `carry_value`'s MAX-not-SUM (Milestone
    // 3): here the two terms must NOT gate each other, or shipping
    // reachability scoring would mechanically zero out decoy-run value,
    // which is exactly the risk the PRD calls out.
    // §4 ablation gate (docs/principles-of-play-phase2-plan.md): the four
    // Mobility terms below (decoy, rotation, mobility/residency, third-man)
    // are skipped entirely when MOBILITY_ABLATE is set, for a clean
    // same-binary before/after comparison. Every other term in this
    // function is untouched either way.
    let mobility_ablated = mobility_ablated();

    if !mobility_ablated {
        let gk_pos = ctx
            .players()
            .opponents()
            .goalkeeper()
            .next()
            .map(|g| g.position);
        let danger = (on_ball_value::effective_open_angle(ctx, position, gk_pos) / 1.31).clamp(0.0, 1.0);
        let marking_pressure =
            (on_ball_value::congestion_risk(ctx, position) / on_ball_value::CONGESTION_CAP).clamp(0.0, 1.0);
        score += danger * marking_pressure * DECOY_WEIGHT;
    }

    // Milestone 7 — flank rotation: two teammates on the same flank (e.g.
    // a wide fullback and the winger ahead of him) should recognise which
    // of them currently holds the wide role, rather than both
    // independently reacting to ball-side and potentially stacking on the
    // same touchline. Scoped to genuinely wide candidates only — this is
    // purely about the wide-vs-tuck decision, not overall shape.
    if !mobility_ablated {
        score += rotation_adjustment(ctx, position);
    }

    // P1 (A1) — lane-occupancy-diversity: penalise a candidate that would
    // put more than `on_ball_value::LANE_OCCUPANCY_CAPACITY` teammates in
    // the same vertical lane, reward vacating a lane the player's own
    // current position already shares with that many or more. Distinct
    // from `rotation_adjustment` above (that's specifically the
    // wide-fullback/winger flank-role handoff); this is Guardiola's
    // general 5-lane diversity rule, applies to every candidate.
    score += lane_occupancy_adjustment(ctx, position, teammates);

    // D3 — central-corridor-priority default shape (defenders only,
    // zero for every other role — see the function's own doc comment).
    score += central_corridor_priority_adjustment(ctx, position);

    // P5 (A3) — mobility/residency: reward moving to a genuinely
    // different lane after holding the current one for a while.
    if !mobility_ablated {
        score += mobility_adjustment(ctx, position);
    }

    // P5 (A4) — vacated-space/third-man credit: a genuine attacking
    // threat with no local congestion, while some other teammate is
    // currently tying up a marker elsewhere.
    if !mobility_ablated {
        score += third_man_adjustment(ctx, position, opponents, teammates);
    }

    // Milestone 12 — weak-side off-ball patience: a candidate on the
    // opposite lateral side from the ball isn't judged on immediate
    // reachability (already near-zero there via the Milestone 5 term
    // above); its job is positioning for a LATER switch of play.
    score += weak_side_patience_adjustment(ctx, position);

    // Step 2 (docs/ball-distribution-fix-plan.md §9): the symmetric
    // stretch credit + width-persistence pair. See the two functions'
    // own doc comments for the full reasoning.
    let field_h = ctx.context.field_size.height as f32;
    score += stretch_credit(position, field_h, opponents);
    score += width_persistence_bonus(ctx, position, field_h);

    // Tether: stay recognisably within the caller's tactical intent —
    // blended toward the HOLDER's own position as pressure rises PAST the
    // engagement floor. Below the floor this is exactly `proposed`
    // (unchanged behaviour for every already-verified milestone: ordinary
    // spread/reachability/decoy/rotation/weak-side scoring never sees
    // this term move for mild, common pressure readings). Past the floor,
    // "the caller's tactical intent" itself shifts from "hold attacking
    // shape" to "get open near the outlet" — real football doesn't treat
    // an emergency out-ball as a deviation from shape, it treats it as
    // the actual priority. Necessary because the pressure-relief bonus
    // above was measured (match-logs trace) to be too small on its own to
    // overcome the tether pull toward a distant `proposed` target — the
    // tether itself has to relax, not just gain a competing bonus.
    let tether_target = match holder {
        Some(h) => proposed + (h.position - proposed) * relief_engagement,
        None => proposed,
    };
    score -= (position - tether_target).magnitude() * TETHER_WEIGHT;

    score
}

/// Milestone 7 — see `score_candidate`'s own call-site comment. `0.0` when
/// `candidate` isn't in the wide band (Milestone 4's established
/// `field_h*0.30/0.70` threshold), when the player has no flank partner,
/// or when the partner is on the opposite half of the pitch (a
/// side-mismatch guard — rotation only makes sense between two players
/// genuinely sharing the same flank). Otherwise: the partner already
/// holding width there is a real penalty (don't stack); the partner
/// having tucked in is a smaller bonus (a nudge to take the vacant wide
/// role, not a mandate — the existing width/repulsion terms still decide
/// the rest).
fn rotation_adjustment(ctx: &StateProcessingContext, candidate: Vector3<f32>) -> f32 {
    let field_h = ctx.context.field_size.height as f32;
    let is_wide_candidate = candidate.y < field_h * 0.30 || candidate.y > field_h * 0.70;
    if !is_wide_candidate {
        return 0.0;
    }
    let Some(partner) = flank_partner(ctx) else {
        return 0.0;
    };
    let candidate_left = candidate.y < field_h * 0.5;
    let partner_left = partner.position.y < field_h * 0.5;
    if candidate_left != partner_left {
        return 0.0; // partner is on the opposite flank right now — not a rotation conflict
    }
    let partner_is_wide = partner.position.y < field_h * 0.30 || partner.position.y > field_h * 0.70;
    if partner_is_wide {
        -ROTATION_WEIGHT
    } else {
        ROTATION_WEIGHT * 0.5
    }
}

/// Step 2 (docs/ball-distribution-fix-plan.md §9, 2026-08-26) — the
/// symmetric half of `on_ball_value.rs`'s `central_congestion_discount`
/// (P2). That mechanism already rewards the EXPLOITER of a stretch (a
/// teammate's central candidate gets a congestion discount when the
/// team has wide players currently marked elsewhere) but gives zero
/// credit to the CREATOR of that distortion — the width-holder
/// themselves. This is a live, per-candidate measurement of the same
/// underlying signal, not a forecast: is THIS wide candidate genuinely
/// close enough to a real opponent to be pulling them out of position,
/// using the same wide-band threshold `rotation_adjustment` above
/// already establishes for this file (0.0 for a central candidate —
/// "standing wide" only counts when it's actually distorting someone).
fn stretch_credit(candidate: Vector3<f32>, field_h: f32, opponents: &[Vector3<f32>]) -> f32 {
    let is_wide = candidate.y < field_h * 0.30 || candidate.y > field_h * 0.70;
    if !is_wide {
        return 0.0;
    }
    let marked = opponents
        .iter()
        .any(|opp| (opp - candidate).magnitude() <= STRETCH_CREDIT_RADIUS);
    if marked { STRETCH_CREDIT_WEIGHT } else { 0.0 }
}

/// Step 2's other half — width-holding persistence. Mirrors the
/// already-shipped byline-carry `momentum_bonus` in `on_ball_value.rs`'s
/// `carry_candidates`: reward continuity of an ESTABLISHED wide position
/// rather than letting a marginally-better central value flicker
/// re-decide it fresh every tick. Scoped to wide-to-wide continuity
/// only — a genuinely central candidate never gets this bonus just for
/// being near the player's current spot, and a player who currently
/// ISN'T wide gets no persistence credit for staying central either
/// (this is specifically an anti-abandonment term for width, not a
/// general inertia bonus).
fn width_persistence_bonus(
    ctx: &StateProcessingContext,
    candidate: Vector3<f32>,
    field_h: f32,
) -> f32 {
    let current = ctx.player.position;
    let current_is_wide = current.y < field_h * 0.30 || current.y > field_h * 0.70;
    let candidate_is_wide = candidate.y < field_h * 0.30 || candidate.y > field_h * 0.70;
    if !current_is_wide || !candidate_is_wide {
        return 0.0;
    }
    let dist = (candidate - current).magnitude();
    ((WIDTH_PERSISTENCE_RADIUS - dist).max(0.0) / WIDTH_PERSISTENCE_RADIUS)
        * WIDTH_PERSISTENCE_WEIGHT
}

/// D3 — central-corridor-priority default shape, defenders only. `0.0`
/// for every other role (midfielder `AttackSupporting` also calls
/// `score_candidate` — this term is deliberately scoped to defensive
/// positioning, not a general anti-width bias for the whole team,
/// which would fight A1/A2's own attacking-width work). Rewards
/// candidates closer to the pitch's own centre-line, continuous (no
/// hard cliff, matching this file's established convention), so a
/// defender still gets full credit for genuinely tracking a wide
/// attacker or covering a real threat — this is a DEFAULT preference
/// among otherwise-similar options, not an override of the state's own
/// zone/marking logic.
fn central_corridor_priority_adjustment(ctx: &StateProcessingContext, candidate: Vector3<f32>) -> f32 {
    let is_defender = ctx
        .player
        .tactical_position
        .current_position
        .position_group()
        == PlayerFieldPositionGroup::Defender;
    if !is_defender {
        return 0.0;
    }
    let field_h = ctx.context.field_size.height as f32;
    let center_y = field_h / 2.0;
    let centrality = (1.0 - (candidate.y - center_y).abs() / (field_h / 2.0)).clamp(0.0, 1.0);
    // P6 — AI lever: `defensiveBias.compactness`, 1.0 = neutral.
    centrality * CENTRAL_CORRIDOR_PRIORITY_WEIGHT * ctx.player.defensive_compactness
}

/// P1 (A1) — off-ball counterpart to `on_ball_value::lane_occupancy_
/// adjustment`. Uses this function's own already-fetched `teammates`
/// slice (id + position pairs, self already excluded by the caller)
/// instead of re-querying `ctx.players()` per candidate — cheaper than
/// the on-ball version, which has no such slice available to it.
fn lane_occupancy_adjustment(
    ctx: &StateProcessingContext,
    candidate: Vector3<f32>,
    teammates: &[(u32, Vector3<f32>)],
) -> f32 {
    let field_h = ctx.context.field_size.height as f32;
    let candidate_lane = PitchLane::for_y(field_h, candidate.y);
    let current_lane = PitchLane::for_y(field_h, ctx.player.position.y);

    // P6 — AI lever: `spatialBias.laneDiscipline`, 1.0 = neutral.
    let lane_discipline = ctx.player.spatial_lane_discipline;
    let mut adjustment = 0.0f32;
    let occupants = teammates
        .iter()
        .filter(|(_, pos)| PitchLane::for_y(field_h, pos.y) == candidate_lane)
        .count();
    if occupants >= on_ball_value::LANE_OCCUPANCY_CAPACITY {
        let over = (occupants - on_ball_value::LANE_OCCUPANCY_CAPACITY + 1) as f32;
        adjustment -= over * LANE_OCCUPANCY_PENALTY_WEIGHT * lane_discipline;
    }
    if candidate_lane != current_lane {
        let current_occupants = teammates
            .iter()
            .filter(|(_, pos)| PitchLane::for_y(field_h, pos.y) == current_lane)
            .count();
        if current_occupants >= on_ball_value::LANE_OCCUPANCY_CAPACITY {
            adjustment += LANE_VACATE_BONUS_WEIGHT * lane_discipline;
        }
    }
    adjustment
}

/// Milestone 12 (possession-decision-intelligence PRD) — weak-side
/// off-ball patience. A player positioned on the opposite lateral side
/// from the ball ("weak side") is, by construction, rarely a genuine
/// pass option right now — Milestone 5's reachability term already and
/// correctly scores that near zero at this range. Their real job in
/// real football is positioning for a LATER switch of play: hold
/// genuine width if the team wants a stretched picture for a diagonal
/// switch, or stay compact and ready to combine quickly if not —
/// following the team's own already-computed `team_width_target`
/// (0 = narrow, 1 = full width) rather than a fixed universal
/// preference. Returns `0.0` when the ball is central (no meaningful
/// "weak side" exists) or the candidate is on the ball's own side —
/// strong-side positioning is already densely handled by the
/// reachability/decoy/rotation terms above; this is additive, not a
/// replacement for any of them.
fn weak_side_patience_adjustment(ctx: &StateProcessingContext, candidate: Vector3<f32>) -> f32 {
    let field_h = ctx.context.field_size.height as f32;
    // `TeamTacticalState` is plain POD, `Copy` -- avoids a temporary-
    // borrow lifetime issue from `ctx.team()`'s owned return value.
    let tactical = *ctx.team().tactical();
    let ball_side = tactical.ball_side;
    if matches!(ball_side, BallSideZone::Center) {
        return 0.0;
    }
    let candidate_side = BallSideZone::for_y(field_h, candidate.y);
    let is_weak_side = matches!(
        (ball_side, candidate_side),
        (BallSideZone::Left, BallSideZone::Right) | (BallSideZone::Right, BallSideZone::Left)
    );
    if !is_weak_side {
        return 0.0;
    }

    // docs/positional-football-master-plan.md Stage 4, item 6
    // (2026-09-07): while a genuine `WideIsolation` is live on the
    // strong side, the weak side's real job in real football is to
    // hold MAXIMUM width -- keep the block stretched so the isolated
    // flank stays genuinely isolated. This is Part B of the reverted
    // rewrite's own original design for this exact pattern ("far-side
    // shape holds width to keep the block stretched... reuses
    // weak_side_patience_adjustment, already built") --
    // docs/positional-play-and-defensive-block-rewrite.md. Overrides
    // the general, ambient `team_width_target` specifically for this
    // situational case (still gated on the same contact-triggered arm
    // as every other consumer of `positional_attack_pattern` -- this
    // does NOT change when/whether a pattern starts, only what the
    // weak side does while one is genuinely live); falls back to the
    // ambient target the rest of the time.
    //
    // `OverloadAndSwitch` shares the identical doctrine (added same
    // session, same reasoning): the weak side isn't just "not crowded"
    // during a live overload, it's the literal TARGET of the switch
    // pass (`run.rs`'s own `OverloadAndSwitch` arm branch already
    // searches the opposite flank for the free man to switch to) --
    // real width there is what makes the switch dangerous once it
    // arrives, not just tidy off-ball shape.
    let width_target = if tactical.positional_attack_active()
        && matches!(
            tactical.positional_attack_pattern,
            Some(crate::r#match::AttackPattern::WideIsolation)
                | Some(crate::r#match::AttackPattern::OverloadAndSwitch)
        )
    {
        crate::match_log_info!(
            "WIDEISOSTRETCH id={} cand_y={:.1} ambient_target={:.2} pattern={:?}",
            ctx.player.id,
            candidate.y,
            tactical.team_width_target,
            tactical.positional_attack_pattern
        );
        1.0
    } else {
        tactical.team_width_target.clamp(0.0, 1.0)
    };
    let center_y = field_h / 2.0;
    let half_h = field_h / 2.0;
    let dist_from_center = ((candidate.y - center_y).abs() / half_h).clamp(0.0, 1.0);
    // How well this candidate's width matches the team's current width
    // preference — a wide candidate scores when width_target is high, a
    // central one scores when it's low.
    let alignment = (1.0 - (dist_from_center - width_target).abs()).max(0.0);
    alignment * WEAK_SIDE_PATIENCE_WEIGHT
}

fn ball_holder(ctx: &StateProcessingContext) -> Option<MatchPlayerLite> {
    let owner_id = ctx.ball().owner_id()?;
    ctx.players()
        .teammates()
        .all()
        .find(|t| t.id == owner_id)
}
