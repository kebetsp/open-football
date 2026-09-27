//! Off-ball attacking runs — designed from null. Read
//! `docs/off-ball-runs-from-null.md` first (Pavel's own verbatim request,
//! the good/bad lessons pulled from this project's own history, the
//! conflict check, and the design this module implements).
//!
//! **What this replaces.** For forwards, wingers/wide midfielders, and
//! attacking midfielders (not central midfielders or defenders) during
//! the Progression and Final phases of our own possession,
//! `attacking_support.rs` (domain 3) previously assigned a flat
//! `Width`/`Advance`/`Penetrate` target from `width_depth`/`advance_depth`
//! — a fixed depth by role and phase, with zero concept of "am I actually
//! open, and can the carrier realistically get me the ball." Pavel's own
//! report, verbatim: "forwards do not need to run into the box ONLY while
//! the ball carrier is wide and within 25m of the goal line... they
//! should evaluate their chance for success, and avoid running if it is
//! low. Otherwise, run." This module is that evaluation: several real
//! candidate actions per runner, each valued as P(the carrier can
//! realistically get the ball there) x (what the runner can do there once
//! he has it) — reusing `box_attack::chance_at`, the exact same,
//! already-sourced currency domain 4 already trusts for the identical
//! question, rather than inventing a second one.
//!
//! **What stays.** Domain 4's own box-zone assignment
//! (`box_attack::plan_box_runs`, gated on the carrier being wide and
//! close to the goal line) is deliberately left untouched — it is
//! already tuned and verified (sticky-zone bonus, local-gap nudging,
//! anti-clustering scoring) for exactly the moment a delivery is live,
//! and replacing it wholesale here would risk an already-working,
//! already-measured system for no reported gain. This module fills the
//! gap Pavel actually reported: everything OUTSIDE that narrow gate,
//! where domain 3's flat tables previously left runners standing on a
//! fixed spot regardless of whether it was open or reachable. Central
//! midfielders and defenders are untouched — they keep domain 3's own
//! Width/Advance tables at every phase.
//!
//! **Persistence.** A chosen run freezes for `COMMIT_TICKS` (the same
//! "freeze the target, don't re-derive it every tick" idiom already
//! proven by `byline_commitment_ticks`/`give_and_go_run_ticks` — this
//! project's own repeated lesson that a continuously re-evaluated target
//! either thrashes or, when chased via the ball's own velocity, wildly
//! overshoots). Staleness (`run_stale_ticks`, decremented/incremented by
//! the caller) makes a held spot's value decay over time, so a runner
//! genuinely alternates between checking short, running in behind, and
//! going to the byline, rather than freezing on one candidate forever.
//!
//! **Anti-clustering, built in from the first commit** (this project's own
//! hard-won lesson — the 2026-09-12 rollback: a global "chase the nearest
//! opponent" override moved every metric it was measured on while making
//! the pitch look wrong, because nothing checked whether two players
//! converged on the same spot). A candidate is rejected outright, not
//! merely discounted, if it sits within `MIN_RUN_SEPARATION` of another
//! runner's already-chosen target this tick, of an already-committed
//! teammate's frozen target, or inside the ball carrier's own direct lane
//! to goal; at most `MAX_IN_BEHIND_RUNNERS` runners may target genuinely
//! in-behind space at once.
//!
//! Process check (`docs/decision-from-null.md`'s required self-audit):
//! this module's call graph touches only raw position/skill data passed
//! in by the caller, `box_attack::chance_at` (an already-sourced value
//! function, not a decision), and `nalgebra::Vector3` geometry. No
//! pre-existing decision function (old `ForwardCreatingSpaceState`,
//! `MidfielderRunningState`'s off-ball branch, or any sibling) is called.

use crate::r#match::player::AttackDuty;
use crate::r#match::player::strategies::common::players::ops::box_attack::chance_at;
use crate::r#match::MatchPlayerLite;
use nalgebra::Vector3;

// ---- Constants (category d — reasoned estimates; no dataset publishes
// off-ball run geometry, per this project's own established sourcing
// hierarchy for this class of question). Flagged as a target for the raw-
// trace verification pass, not treated as final.

/// Real max range for a direct ball to a runner's chosen target — full
/// credit up to here, fading to zero further out. ~25m at this engine's
/// 8 units/metre scale.
const DIRECT_RANGE_FULL: f32 = 200.0;
/// Beyond this, a direct ball to the target isn't a realistic option at
/// all. ~45m.
const DIRECT_RANGE_ZERO: f32 = 360.0;
/// A teammate within this of the carrier is a genuine relay option —
/// Pavel's own framing: "teammates may find that passing line after
/// another small pass, or dribbling, or something else." ~20m. The
/// caller (`attacking_support.rs`) builds the `relays` list from this
/// radius before calling `plan_runs` — kept `pub(crate)` so both sides
/// use the exact same figure rather than risking two independently
/// hand-typed copies drifting apart.
pub(crate) const RELAY_RADIUS: f32 = 160.0;
/// A relayed reach is discounted — a second action has to happen first
/// before the ball can realistically arrive.
const RELAY_DISCOUNT: f32 = 0.55;
/// How far ahead of a real defender's own arrival a runner must be at his
/// target to count as meaningfully, not just technically, open there.
/// ~1.2s at this engine's 100-tick/s simulation rate.
const ARRIVAL_MARGIN_TICKS: f32 = 40.0;
/// A frozen commitment's real duration — long enough for a run to
/// actually develop, short enough that a stale one doesn't hang around
/// once it plainly isn't paying off. ~2s.
pub const COMMIT_TICKS: u16 = 200;
/// A held spot's value starts decaying once the runner has gone this long
/// without receiving the ball — the "back and forth" cycle Pavel
/// described (check short, run in behind, byline, back), not a run that
/// freezes on one spot forever. ~3s.
const STALE_ONSET_TICKS: f32 = 300.0;
/// Fraction of a "stay near where I already am" candidate's value knocked
/// off once fully stale.
const STALE_MAX_PENALTY: f32 = 0.6;
/// Within this of the runner's own current position, a candidate counts
/// as "the spot I already hold" for staleness purposes. ~5m.
const STALE_NEAR_RADIUS: f32 = 40.0;
/// Minimum separation between two runners' targets, and between a target
/// and an already-committed teammate's frozen target — anti-clustering.
/// ~6m.
const MIN_RUN_SEPARATION: f32 = 48.0;
/// How close to the direct carrier-to-goal line a candidate must sit to
/// count as "the carrier's own lane" — a runner should never crowd the
/// exact path the ball carrier himself is using. ~3.75m.
const CARRIER_LANE_RADIUS: f32 = 30.0;
/// At most this many runners target genuinely in-behind space at once — a
/// real forward line doesn't send every attacker through together.
const MAX_IN_BEHIND_RUNNERS: usize = 2;
/// How far beyond the last defender's line an in-behind target sits — the
/// offside-line hold clamp (which runs after every attacking-duty
/// override, last in the velocity chain) keeps the runner legal
/// regardless of this raw target, so it is deliberately placed genuinely
/// beyond the line, not pinned to its shoulder. ~5m.
const IN_BEHIND_PUSH: f32 = 40.0;
/// A byline run's depth short of the goal line — deep enough to be a real
/// cutback/byline threat, short of literally standing on the line. ~3m.
const BYLINE_DEPTH_MARGIN: f32 = 24.0;
/// A byline run's lateral margin from the touchline. ~2.5m.
const BYLINE_TOUCHLINE_MARGIN: f32 = 20.0;
/// How far ahead of the carrier's own current depth a flank-pocket
/// candidate sits. ~10m.
const FLANK_POCKET_AHEAD: f32 = 80.0;
/// A flank pocket sits this far in from the touchline (not literally on
/// it, unlike a byline run — it's meant as a live passing/relay option,
/// not a delivery-only run).
const FLANK_POCKET_TOUCHLINE_MARGIN: f32 = 40.0;
/// A check-short candidate approaches this fraction of the distance from
/// the runner's own position toward the carrier.
const CHECK_SHORT_APPROACH: f32 = 0.5;
/// ...capped at this absolute distance so he doesn't run all the way to
/// the carrier's feet. ~15m.
const CHECK_SHORT_MAX_CLOSE: f32 = 120.0;

fn direct_reach_factor(dist: f32) -> f32 {
    if dist <= DIRECT_RANGE_FULL {
        1.0
    } else if dist >= DIRECT_RANGE_ZERO {
        0.0
    } else {
        1.0 - (dist - DIRECT_RANGE_FULL) / (DIRECT_RANGE_ZERO - DIRECT_RANGE_FULL)
    }
}

/// P(the ball carrier can realistically get the ball to `target`) — a
/// direct ball within realistic range, or, if a teammate genuinely close
/// to the carrier could relay it on (`relays`, built by the caller from
/// teammates within `RELAY_RADIUS` of the carrier), the (discounted)
/// range from that relay instead. Deliberately NOT "is the passing lane
/// open" — Pavel's own correction, verbatim: "not if the passing line is
/// open but if the passing range is realistic."
fn reach_prob(carrier_pos: Vector3<f32>, target: Vector3<f32>, relays: &[Vector3<f32>]) -> f32 {
    let direct = direct_reach_factor((target - carrier_pos).magnitude());
    let relayed = relays
        .iter()
        .map(|r| RELAY_DISCOUNT * direct_reach_factor((target - r).magnitude()))
        .fold(0.0f32, f32::max);
    direct.max(relayed)
}

/// How genuinely, meaningfully open the runner will be at `target` by the
/// time he gets there, relative to the nearest real defender's own
/// arrival — a real race in time, using each side's own condition-
/// adjusted top speed (the caller already has this via
/// `max_speed_with_condition`, the same measurement this engine's kept
/// physics layer uses everywhere). No opponents in range at all reads as
/// fully open.
fn timing_margin(
    from: Vector3<f32>,
    target: Vector3<f32>,
    my_speed: f32,
    opponents_speed: &[(Vector3<f32>, f32)],
) -> f32 {
    let my_time = (target - from).magnitude() / my_speed.max(0.05);
    let mut opp_time = f32::MAX;
    for (pos, spd) in opponents_speed {
        let t = (target - pos).magnitude() / spd.max(0.05);
        opp_time = opp_time.min(t);
    }
    if opp_time >= f32::MAX {
        return 1.0;
    }
    ((opp_time - my_time) / ARRIVAL_MARGIN_TICKS).clamp(0.0, 1.0)
}

/// The value of a candidate held near the runner's own current spot
/// decays with dwell time without receiving the ball, so runs alternate
/// instead of freezing on one candidate forever.
fn staleness_penalty(candidate: Vector3<f32>, current_pos: Vector3<f32>, stale_ticks: u16) -> f32 {
    if (candidate - current_pos).magnitude() > STALE_NEAR_RADIUS {
        return 0.0;
    }
    (stale_ticks as f32 / STALE_ONSET_TICKS).clamp(0.0, 1.0) * STALE_MAX_PENALTY
}

fn too_close_to_others(pt: Vector3<f32>, taken: &[Vector3<f32>]) -> bool {
    taken.iter().any(|t| (pt - t).magnitude() < MIN_RUN_SEPARATION)
}

/// True when `pt` sits inside the ball carrier's own direct corridor to
/// goal — a runner should never target the exact lane the carrier
/// himself is using.
fn in_carrier_lane(pt: Vector3<f32>, carrier_pos: Vector3<f32>, opponent_goal: Vector3<f32>) -> bool {
    let seg = opponent_goal - carrier_pos;
    let len = seg.magnitude();
    if len < 1.0 {
        return false;
    }
    let dir = seg / len;
    let to_pt = pt - carrier_pos;
    let along = to_pt.dot(&dir);
    if along < 0.0 || along > len {
        return false;
    }
    let perp = (to_pt - dir * along).magnitude();
    perp < CARRIER_LANE_RADIUS
}

#[derive(Clone, Copy, PartialEq)]
enum RunKind {
    InBehind,
    FlankPocket,
    Byline,
    CheckShort,
    Hold,
}

/// One eligible off-ball runner's current state, as seen by the planner.
/// Built fresh each tick by the caller (`attacking_support.rs`) from
/// `MatchField`/`MatchPlayer` data.
pub struct RunnerInput {
    pub id: u32,
    pub position: Vector3<f32>,
    pub start_position: Vector3<f32>,
    pub max_speed: f32,
    /// True for a genuinely wide-tagged role (winger/wide midfielder, or
    /// a wide forward in a non-front-two formation) — only these get a
    /// byline candidate; a central forward/attacking midfielder does not.
    pub is_wide: bool,
    pub commit_ticks: u16,
    pub commit_target: Vector3<f32>,
    pub commit_duty: AttackDuty,
    pub stale_ticks: u16,
}

pub struct RunOutput {
    pub id: u32,
    pub duty: AttackDuty,
    pub target: Vector3<f32>,
    /// True when this tick chose a fresh candidate that should now be
    /// frozen for `COMMIT_TICKS` — the caller arms `run_commit_ticks`
    /// accordingly. False for a reused (already-committed) target, or
    /// for `Hold` (staying at a good-enough spot needs no freeze — it is
    /// re-evaluated fresh, and cheaply, every tick with no thrash risk).
    pub rearm: bool,
}

fn build_candidates(
    r: &RunnerInput,
    carrier_pos: Vector3<f32>,
    own_goal: Vector3<f32>,
    opponent_goal: Vector3<f32>,
    attack_dir: f32,
    opp_last_line_depth: Option<f32>,
    field_height: f32,
) -> Vec<(RunKind, Vector3<f32>)> {
    let mut out = vec![(RunKind::Hold, r.position)];
    let side_sign: f32 = if r.start_position.y < field_height * 0.5 { -1.0 } else { 1.0 };

    // In behind: his own channel (his start-position lane, so several
    // runners don't converge on one lane by construction — the same
    // anti-collapse idea domain 3's own Step D anchor-blend already
    // relies on), pushed genuinely beyond the last defender's line — the
    // offside-line hold clamp keeps this legal regardless.
    let channel_y = r.start_position.y;
    let in_behind_depth = opp_last_line_depth
        .map(|d| d + IN_BEHIND_PUSH)
        .unwrap_or_else(|| (opponent_goal.x - own_goal.x).abs() * 0.75);
    out.push((
        RunKind::InBehind,
        Vector3::new(own_goal.x + attack_dir * in_behind_depth, channel_y, 0.0),
    ));

    // Flank pocket: his own side, a genuine step ahead of the carrier's
    // own current depth, in from the touchline (a live relay option, not
    // a delivery-only run).
    let carrier_depth = (carrier_pos.x - own_goal.x) * attack_dir;
    let pocket_x = own_goal.x + attack_dir * (carrier_depth + FLANK_POCKET_AHEAD);
    let pocket_y = if side_sign < 0.0 {
        FLANK_POCKET_TOUCHLINE_MARGIN
    } else {
        field_height - FLANK_POCKET_TOUCHLINE_MARGIN
    };
    out.push((RunKind::FlankPocket, Vector3::new(pocket_x, pocket_y, 0.0)));

    // Byline: wingers/wide midfielders only. "Going deeper for wingers...
    // it is a must." (Pavel, verbatim.)
    if r.is_wide {
        let byline_x = opponent_goal.x - attack_dir * BYLINE_DEPTH_MARGIN;
        let byline_y = if side_sign < 0.0 {
            BYLINE_TOUCHLINE_MARGIN
        } else {
            field_height - BYLINE_TOUCHLINE_MARGIN
        };
        out.push((RunKind::Byline, Vector3::new(byline_x, byline_y, 0.0)));
    }

    // Check short: partway from the runner's own position toward the
    // carrier, capped so he doesn't run all the way to the carrier's
    // feet.
    let to_carrier = carrier_pos - r.position;
    let dist = to_carrier.magnitude();
    if dist > 5.0 {
        let step = (dist * CHECK_SHORT_APPROACH).min(CHECK_SHORT_MAX_CLOSE);
        out.push((RunKind::CheckShort, r.position + (to_carrier / dist) * step));
    }
    out
}

fn duty_for(kind: RunKind) -> AttackDuty {
    match kind {
        RunKind::InBehind | RunKind::FlankPocket => AttackDuty::Penetrate,
        RunKind::Byline | RunKind::CheckShort | RunKind::Hold => AttackDuty::Run,
    }
}

/// Plans a run for every eligible runner not already claimed by domain
/// 4's box-zone assignment. Runners already mid-commitment reuse their
/// frozen target unchanged (still seeding the anti-clustering/in-behind
/// count for teammates being freshly evaluated this same tick); everyone
/// else evaluates the real candidates above and either commits to the
/// best one or, if his current spot already scores as well as anything
/// else, holds.
pub fn plan_runs(
    runners: &[RunnerInput],
    carrier_pos: Vector3<f32>,
    relays: &[Vector3<f32>],
    own_goal: Vector3<f32>,
    opponent_goal: Vector3<f32>,
    attack_dir: f32,
    opponents_lite: &[MatchPlayerLite],
    opponents_speed: &[(Vector3<f32>, f32)],
    opp_last_line_depth: Option<f32>,
    field_height: f32,
) -> Vec<RunOutput> {
    let mut taken: Vec<Vector3<f32>> =
        runners.iter().filter(|r| r.commit_ticks > 0).map(|r| r.commit_target).collect();
    let mut in_behind_count = runners
        .iter()
        .filter(|r| r.commit_ticks > 0)
        .filter(|r| {
            let depth = (r.commit_target.x - own_goal.x) * attack_dir;
            opp_last_line_depth.is_some_and(|d| depth > d)
        })
        .count();

    let mut out = Vec::with_capacity(runners.len());
    for r in runners {
        if r.commit_ticks > 0 {
            out.push(RunOutput { id: r.id, duty: r.commit_duty, target: r.commit_target, rearm: false });
            continue;
        }
        let candidates = build_candidates(
            r,
            carrier_pos,
            own_goal,
            opponent_goal,
            attack_dir,
            opp_last_line_depth,
            field_height,
        );
        let mut best: Option<(f32, RunKind, Vector3<f32>)> = None;
        for (kind, pt) in candidates {
            if kind == RunKind::InBehind && in_behind_count >= MAX_IN_BEHIND_RUNNERS {
                continue;
            }
            if kind != RunKind::Hold
                && (too_close_to_others(pt, &taken) || in_carrier_lane(pt, carrier_pos, opponent_goal))
            {
                continue;
            }
            let reach = reach_prob(carrier_pos, pt, relays);
            let margin = timing_margin(r.position, pt, r.max_speed, opponents_speed);
            let chance = chance_at(pt, opponent_goal, opponents_lite);
            let mut value = reach * margin * chance;
            value *= 1.0 - staleness_penalty(pt, r.position, r.stale_ticks);
            if best.is_none_or(|(bv, ..)| value > bv) {
                best = Some((value, kind, pt));
            }
        }
        let Some((_, kind, pt)) = best else { continue };
        if kind == RunKind::InBehind {
            in_behind_count += 1;
        }
        if kind != RunKind::Hold {
            taken.push(pt);
        }
        out.push(RunOutput { id: r.id, duty: duty_for(kind), target: pt, rearm: kind != RunKind::Hold });
    }
    out
}
