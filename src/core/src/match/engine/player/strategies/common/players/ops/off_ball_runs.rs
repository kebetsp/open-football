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
//! realistically get the ball there) x (how genuinely open the
//! destination itself is, `space_value`).
//!
//! **Rewritten 2026-09-27, twice, same day (`docs/winger-space-and-
//! progressive-runs-fix-plan.md` §9 then §11).** The value was
//! originally `reach x margin x box_attack::chance_at`. §9 fixed
//! `chance_at` -> `space_value` (a run's worth is whether the
//! DESTINATION is genuinely open, not whether the runner's own eventual
//! shot/cross from there would score well — that's a separate, later
//! decision) and softened the raw footrace clamp (`reach`/`margin`) into
//! a sigmoid. Verified real, but Pavel watched the result and correctly
//! said it made no visible difference — a further raw trace (21k+ real
//! decisions, `docs/winger-space-and-progressive-runs-fix-plan.md`
//! §11.2) found `reach`/`margin` were STILL the dominant reason `Hold`
//! won ~97% of decisions even after `space_value` read correctly high
//! for every genuine candidate: both ask "would this exact pass be
//! deliverable, right now, from a standing start" — a defender is,
//! geometrically, almost always closer to any real attacking-third
//! target than an off-ball runner starting from a cold stop, so ANY
//! footrace framing (hard clamp or smooth sigmoid alike) structurally
//! disadvantages a genuine run relative to `Hold`'s free, no-travel-
//! needed "race." Pavel's correction, verbatim: "You see the space, you
//! use it. And that's it. As simple as can be... my old logic never had
//! this reach/margin rubbish." **`reach_prob`/`timing_margin` and every
//! constant/parameter that only fed them (`DIRECT_RANGE_FULL/ZERO`,
//! `RELAY_DISCOUNT`, `RELAY_RADIUS`, `TIMING_SIGMOID_SCALE_TICKS`,
//! the `relays`/`opponents_speed`/`max_speed` plumbing) are deleted
//! outright, not left dead** — confirmed with Pavel directly rather than
//! guessed at. The value is now `space_value(candidate) x (1 -
//! staleness_penalty)` — nothing else. Whether the ball can actually get
//! there at all is the CARRIER's own, separate, already-existing
//! decision (`on_ball_decision.rs`'s `pass_value`/`Action::Through`,
//! which independently checks progress/openness/risk before choosing to
//! pass) — a run-planner shouldn't pre-judge that before the run has
//! even developed.
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
//! in by the caller and `nalgebra::Vector3` geometry (`space_value` is a
//! plain nearest-opponent-distance measurement, a fact, not a judgment).
//! No pre-existing decision function (old `ForwardCreatingSpaceState`,
//! `MidfielderRunningState`'s off-ball branch, or any sibling) is called.

use crate::r#match::player::AttackDuty;
use crate::r#match::MatchPlayerLite;
use nalgebra::Vector3;

// ---- Constants (category d — reasoned estimates; no dataset publishes
// off-ball run geometry, per this project's own established sourcing
// hierarchy for this class of question). Flagged as a target for the raw-
// trace verification pass, not treated as final.

/// Nearest-opponent distance at which a run's DESTINATION reads as
/// "fully open" (space_value saturates at 1.0 beyond this) — the same
/// real-world reference (~7.5m, enough room to receive and turn
/// unpressured) `on_ball_decision.rs`'s own `OPEN_REFERENCE_DIST` uses
/// for the identical question on the carrier's side of the ball. Kept as
/// a separate local constant rather than importing that module's private
/// one — same number, same reasoning, no cross-module coupling for a
/// single shared magnitude.
const SPACE_REFERENCE_DIST: f32 = 60.0;
/// Two candidates within this much `value` of each other count as a
/// genuine tie, not a real difference in merit. Found via a raw trace
/// (2026-09-28, wide midfielders specifically): `InBehind` (generated
/// before `Byline` in `build_candidates`) was winning 76.3% of decisions
/// to `Byline`'s 7.3% -- despite `Byline`'s own mean `value` (0.998)
/// being HIGHER than `InBehind`'s (0.964), because 85.3% of the time the
/// two were within 0.02 of each other and a strict `>` comparison always
/// keeps whichever candidate was set first. `InBehind`'s target for a
/// wide player is only just beyond the last defender's line -- often far
/// short of the actual byline -- so this wasn't a football preference,
/// it was pure evaluation order silently suppressing the deeper option.
/// ~1.2 units (0.15m) of nearest-opponent-distance difference at
/// `SPACE_REFERENCE_DIST`'s own normalisation -- genuinely
/// indistinguishable openness, not a meaningful gap.
const TIE_BREAK_EPSILON: f32 = 0.02;

/// How genuinely open a candidate run TARGET is — nearest real opponent
/// distance, normalised. This is the term that replaced `box_attack::
/// chance_at` in `plan_runs`'s value formula (see this file's own header
/// for why): a run's worth is whether the space is really there, not
/// whether the runner's own shot/cross from that exact point would score
/// well — that's a separate, later decision. No opponents in range at
/// all reads as fully open.
fn space_value(pt: Vector3<f32>, opponents: &[MatchPlayerLite]) -> f32 {
    let nearest = opponents.iter().map(|o| (o.position - pt).magnitude()).fold(f32::MAX, f32::min);
    if nearest >= f32::MAX {
        return 1.0;
    }
    (nearest / SPACE_REFERENCE_DIST).clamp(0.0, 1.0)
}
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
    own_goal: Vector3<f32>,
    opponent_goal: Vector3<f32>,
    attack_dir: f32,
    opponents_lite: &[MatchPlayerLite],
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
        #[cfg(feature = "match-logs")]
        let (mut hold_val, mut best_alt_val, mut best_alt_kind): (f32, f32, u8) = (0.0, 0.0, 255);
        for (kind, pt) in candidates {
            if kind == RunKind::InBehind && in_behind_count >= MAX_IN_BEHIND_RUNNERS {
                #[cfg(feature = "match-logs")]
                crate::match_log_info!("RUNREJECTDIAG id={} kind={} reason=maxinbehind", r.id, kind as u8);
                continue;
            }
            if kind != RunKind::Hold
                && (too_close_to_others(pt, &taken) || in_carrier_lane(pt, carrier_pos, opponent_goal))
            {
                #[cfg(feature = "match-logs")]
                crate::match_log_info!(
                    "RUNREJECTDIAG id={} kind={} reason={}",
                    r.id,
                    kind as u8,
                    if too_close_to_others(pt, &taken) { "tooclose" } else { "carrierlane" }
                );
                continue;
            }
            let space = space_value(pt, opponents_lite);
            let value = space * (1.0 - staleness_penalty(pt, r.position, r.stale_ticks));
            #[cfg(feature = "match-logs")]
            crate::match_log_info!(
                "RUNCANDDIAG id={} kind={} space={:.4} val={:.4}",
                r.id,
                kind as u8,
                space,
                value
            );
            #[cfg(feature = "match-logs")]
            if kind == RunKind::Hold {
                hold_val = value;
            } else if value > best_alt_val {
                best_alt_val = value;
                best_alt_kind = kind as u8;
            }
            // On a genuine near-tie (within `TIE_BREAK_EPSILON`), prefer
            // whichever candidate is DEEPER -- a real football heuristic
            // (all else equal, take the more advanced option), not a
            // special case for any one `RunKind`. This is what actually
            // fixes the InBehind-over-Byline bias documented on
            // `TIE_BREAK_EPSILON`'s own doc comment: `Byline` is, by
            // construction, always the deepest candidate a wide player
            // can generate, so it now wins the near-ties it was already
            // scoring at least as well on, instead of losing them purely
            // to evaluation order.
            let better = match best {
                None => true,
                Some((bv, _, bpt)) => {
                    if value > bv + TIE_BREAK_EPSILON {
                        true
                    } else if value >= bv - TIE_BREAK_EPSILON {
                        let depth = (pt.x - own_goal.x) * attack_dir;
                        let best_depth = (bpt.x - own_goal.x) * attack_dir;
                        depth > best_depth
                    } else {
                        false
                    }
                }
            };
            if better {
                best = Some((value, kind, pt));
            }
        }
        #[cfg(feature = "match-logs")]
        crate::match_log_info!(
            "HOLDVSALTDIAG id={} hold_val={:.4} best_alt_val={:.4} best_alt_kind={} gap={:.4}",
            r.id,
            hold_val,
            best_alt_val,
            best_alt_kind,
            hold_val - best_alt_val
        );
        let Some((val, kind, pt)) = best else { continue };
        #[cfg(feature = "match-logs")]
        {
            let nearest_opp = opponents_lite
                .iter()
                .map(|o| (o.position - r.position).magnitude())
                .fold(f32::MAX, f32::min);
            crate::match_log_info!(
                "RUNWINDIAG id={} kind={} val={:.4} nearest_opp_to_runner={:.1}",
                r.id,
                kind as u8,
                val,
                nearest_opp
            );
        }
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
