use crate::club::player::events::PositionLoad;
use crate::club::player::traits::{BehavioralDirective, PlayerTrait};
use crate::r#match::PlayerMatchEndStats;
use crate::r#match::{BallZone, GamePhase, TeamTacticalState};
use crate::r#match::defenders::states::DefenderState;
use crate::r#match::engine::result::PlayerMatchPhysicalSnapshot;
use crate::r#match::engine::tactics::TacticalPositions;
use crate::r#match::events::EventCollection;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::goalkeepers::states::state::GoalkeeperState;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::player::memory::PlayerMemory;
use crate::r#match::player::state::{PlayerMatchState, PlayerState};
use crate::r#match::player::statistics::MatchPlayerStatistics;
use crate::r#match::player::waypoints::WaypointManager;
use crate::r#match::{GameTickContext, MatchContext, StateProcessingContext};
use crate::utils::DateUtils;
use crate::{
    PersonAttributes, Player, PlayerAttributes, PlayerFieldPositionGroup, PlayerPositionType,
    PlayerSkills,
};
use chrono::NaiveDate;
use nalgebra::Vector3;
use std::fmt::*;

#[cfg(debug_assertions)]
use log::debug;

/// Trigger half of a whole-team shape rule (plan Phase 5). Ball-lane
/// triggers are from the PLAYER's attacking perspective ("left" =
/// the manager's left wing), side-corrected each tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeTrigger {
    InPossession,
    OutOfPossession,
    BallLeft,
    BallCenter,
    BallRight,
    /// Score-state triggers (Phase 7.3), from this player's team's
    /// perspective: active while winning / level / losing.
    Leading,
    Level,
    Trailing,
    /// Elapsed-time trigger: active from this DISPLAYED minute onward
    /// (0-90 scale, mapped onto the engine's compressed match clock).
    AfterMinute(u32),
    /// Combined score x time (the wishlist #14 case: "when winning
    /// after 70 minutes, drop into a mid-block").
    LeadingAfterMinute(u32),
    TrailingAfterMinute(u32),
    /// Principles-of-Play plan (2026-08-30): active when this player's
    /// team does NOT have the ball and the ball is in this team's own
    /// defensive third (`BallZone::DefensiveThird`). The real-football
    /// case this exists for: a back line that should compress toward a
    /// shared depth/spread while defending deep, not one that only
    /// happens to compact as a side effect of individual duel behaviour.
    DefendingOwnThird,
    /// docs/positional-attack-phase-design.md, Stage 5: real, sourced
    /// build-out doctrine ("if they press with two, build with
    /// three... centre-backs split wide toward the corners of the
    /// box... the goalkeeper becomes the free man") — active while
    /// this team has the ball AND the OPPONENT is genuinely in a
    /// coach-driven high press (`GamePhase::HighPress`), not merely
    /// pressing incidentally.
    UnderHighPress,
    /// Rule 2.0: where the ball is, in the manager's attacking direction
    /// (`TeamTacticalState::ball_zone` of this player's own team).
    BallAttackingThird,
    BallMiddleThird,
    BallDefensiveThird,
}

/// Action half of a shape rule — computed relative to the player's
/// `shape_base` and side, so no absolute-coordinate mirroring bugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeAction {
    /// Fullback inverts: step up ~16% of the pitch and halfway toward
    /// the central lane.
    StepIntoMidfield,
    /// Pin to the nearest touchline, slightly advanced.
    HoldWidth,
    /// Come infield ~55% toward the central lane, same depth.
    TuckInside,
    /// Drop ~12% of the pitch toward own goal, same lane.
    DropDeeper,
    /// A short drop between the lines (~4% of the pitch, same lane) —
    /// what the `drop_deep` instruction means for a forward/attacker.
    DropShort,
    /// Principles-of-Play plan (2026-08-30): converge toward the team's
    /// shared back-line depth (`TeamTacticalState.defensive_line_x`,
    /// already computed and phase-aware, previously only read — softly
    /// blended — by `HoldingLineState`, one state among many a defender
    /// can be in) and squeeze this player's own lane toward the centre
    /// in proportion to `TeamTacticalState.compactness_target`. Unlike
    /// the other three actions this genuinely needs team-relational
    /// input, not just `base`/`side` — passed in as a cheap `Copy` PODs
    /// value rather than reaching into `field.players` from inside
    /// `ShapeRule::target`, which would need a very different signature
    /// (and, per this plan's own explicit design note, real per-teammate
    /// data isn't actually needed: the two team-level summaries already
    /// exist).
    CompactBackLine,
    /// Rule 2.0: a parametrised move (lane + side + depth shift). The
    /// five original lane/depth actions above are exactly expressible
    /// as presets of this (`MoveSpec::preset`), checked by a unit test.
    Move(MoveSpec),
    /// Rule 2.0 role interchange: while the rule holds, this player and
    /// `ShapeRule::swap_with` exchange position tags and anchors, so every
    /// duty system treats each as the other's role. Returns to normal after
    /// a minimum dwell once the rule stops holding.
    SwapRole,
}

impl ShapeAction {
    /// Whether this action's DEPTH is measured from a live reference (last defender,
    /// ball, an opponent) and so is an absolute position rather than an offset.
    pub fn depth_is_absolute(&self) -> bool {
        matches!(self, ShapeAction::Move(m) if !matches!(m.anchor, Anchor::Base))
    }

    /// Whether this action's LATERAL part is a pitch lane (absolute) rather
    /// than a nudge on where the duty system already wants the player.
    /// A lane is a fact about the pitch ("the half-space"); depth stays an
    /// offset because the duty systems own it (a fixed absolute depth sent a
    /// "drop deeper" back four to its own goal line, audit 2026-10-22).
    pub fn lateral_is_absolute(&self) -> bool {
        match self {
            ShapeAction::Move(m) => !matches!(m.lane, LaneRef::Keep) || matches!(m.anchor, Anchor::Opponent(_)),
            ShapeAction::StepIntoMidfield | ShapeAction::TuckInside | ShapeAction::HoldWidth => true,
            _ => false,
        }
    }
}

/// Rule 2.0 (docs/rule-language-conflict-check.md): where a `Move`
/// action puts the player ACROSS the pitch. Lanes follow the standard
/// five-lane model (flank / half-space / centre / half-space / flank),
/// each a fifth of the pitch width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneRef {
    /// Keep the player's current lateral position (depth-only move).
    Keep,
    /// The touchline lane (same point `HoldWidth` uses).
    Flank,
    /// The half-space (centre of the second-from-touchline lane).
    HalfSpace,
    /// The central lane.
    Centre,
    /// Go this many thousandths of the way from the base lateral
    /// position to the centre line (500 = halfway). Reproduces the
    /// original `StepIntoMidfield`/`TuckInside` actions exactly. Ignores
    /// `SideRef`.
    ToCentre(i16),
}

/// Which side of the pitch a `Move` lane refers to. Left/Right are the
/// MANAGER's left/right, corrected for the half-time side swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideRef {
    /// The side of the pitch the player's own base position is on.
    Own,
    /// The opposite side from the player's own.
    Opposite,
    Left,
    Right,
    /// The side the ball is on right now (own side when it is central).
    BallSide,
}

/// Reference point a `Move`'s depth (and, for an opponent, lateral position) is measured
/// from. `Base` is the player's own formation base, as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    Base,
    /// The opponent's deepest outfield player (the offside line).
    LastDefender,
    /// The ball's current depth.
    Ball,
    /// A named opponent (player id): depth from him, and with `LaneRef::Keep` also his lateral position.
    Opponent(u32),
}

/// Live reference positions a rule target may be measured from (computed each tick by the engine).
#[derive(Debug, Clone, Copy, Default)]
pub struct RuleRefs {
    pub last_def_x: Option<f32>,
    pub ball_x: f32,
    pub opponent: Option<(f32, f32)>,
}

/// A parametrised movement: lateral lane + side, plus a depth shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveSpec {
    pub lane: LaneRef,
    pub side: SideRef,
    /// Depth shift along the attack direction in thousandths of the
    /// pitch length; positive = toward the opponent goal.
    pub depth_permille: i16,
    pub anchor: Anchor,
}

impl MoveSpec {
    /// The original fixed actions expressed as `Move` parameters. Checked
    /// against `ShapeRule::target` by `rule2_presets_match_original_actions`.
    pub fn preset(action: ShapeAction) -> Option<MoveSpec> {
        let (lane, depth_permille) = match action {
            ShapeAction::StepIntoMidfield => (LaneRef::ToCentre(500), 160),
            ShapeAction::TuckInside => (LaneRef::ToCentre(550), 0),
            ShapeAction::HoldWidth => (LaneRef::Flank, 50),
            ShapeAction::DropDeeper => (LaneRef::Keep, -120),
            ShapeAction::DropShort => (LaneRef::Keep, -40),
            _ => return None,
        };
        Some(MoveSpec { lane, side: SideRef::Own, depth_permille, anchor: Anchor::Base })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeRule {
    pub trigger: ShapeTrigger,
    pub action: ShapeAction,
    /// Extra conditions that must ALL hold together with `trigger`
    /// ("in possession AND the ball on our left"). Rule 2.0.
    pub and: [Option<ShapeTrigger>; 2],
    /// Rule 2.0 limit: the rule is skipped while this many or more
    /// teammates (not himself) already stand within `ZONE_RADIUS` of its
    /// target ("fill the half-space only if fewer than two are there").
    pub max_in_zone: Option<u8>,
    /// For `SwapRole`: the id of the player whose role is taken.
    pub swap_with: Option<u32>,
}

/// Radius around a rule's target that counts as "the zone" for
/// `ShapeRule::max_in_zone` (100 u = 12.5 m).
pub const ZONE_RADIUS: f32 = 100.0;

impl ShapeRule {
    pub fn new(trigger: ShapeTrigger, action: ShapeAction) -> Self {
        Self { trigger, action, and: [None, None], max_in_zone: None, swap_with: None }
    }

    /// Whether this rule's trigger (and every extra condition) currently
    /// holds. `team` is this player's own team's already-computed
    /// tactical snapshot — cheap to pass (plain `Copy` POD), avoids
    /// re-deriving ball-zone/possession logic that already exists.
    pub fn is_active(
        &self,
        in_possession: bool,
        ball_owned: bool,
        lane: i8,
        goals_for: u8,
        goals_against: u8,
        minute_now: u32,
        team: TeamTacticalState,
        opponent_phase: GamePhase,
    ) -> bool {
        let holds = |t: ShapeTrigger| {
            Self::trigger_holds(
                t,
                in_possession,
                ball_owned,
                lane,
                goals_for,
                goals_against,
                minute_now,
                team,
                opponent_phase,
            )
        };
        holds(self.trigger) && self.and.iter().flatten().all(|t| holds(*t))
    }

    #[allow(clippy::too_many_arguments)]
    fn trigger_holds(
        trigger: ShapeTrigger,
        in_possession: bool,
        ball_owned: bool,
        lane: i8,
        goals_for: u8,
        goals_against: u8,
        minute_now: u32,
        team: TeamTacticalState,
        opponent_phase: GamePhase,
    ) -> bool {
        match trigger {
            ShapeTrigger::InPossession => in_possession,
            ShapeTrigger::OutOfPossession => !in_possession && ball_owned,
            ShapeTrigger::BallLeft => lane == -1,
            ShapeTrigger::BallCenter => lane == 0,
            ShapeTrigger::BallRight => lane == 1,
            ShapeTrigger::Leading => goals_for > goals_against,
            ShapeTrigger::Level => goals_for == goals_against,
            ShapeTrigger::Trailing => goals_for < goals_against,
            ShapeTrigger::AfterMinute(m) => minute_now >= m,
            ShapeTrigger::LeadingAfterMinute(m) => goals_for > goals_against && minute_now >= m,
            ShapeTrigger::TrailingAfterMinute(m) => {
                goals_for < goals_against && minute_now >= m
            }
            ShapeTrigger::DefendingOwnThird => {
                !in_possession && matches!(team.ball_zone, BallZone::DefensiveThird)
            }
            ShapeTrigger::BallAttackingThird => matches!(team.ball_zone, BallZone::AttackingThird),
            ShapeTrigger::BallMiddleThird => matches!(team.ball_zone, BallZone::MiddleThird),
            ShapeTrigger::BallDefensiveThird => matches!(team.ball_zone, BallZone::DefensiveThird),
            ShapeTrigger::UnderHighPress => {
                in_possession && matches!(opponent_phase, GamePhase::HighPress)
            }
        }
    }

    /// Target position for this rule's action, relative to `base`.
    /// `team` is this player's own team's tactical snapshot — only
    /// consulted by `CompactBackLine`, ignored by the other three.
    /// `ball_y` (raw field Y, not side-corrected) is likewise only
    /// consulted by `CompactBackLine`.
    pub fn target(
        &self,
        base: Vector3<f32>,
        side: PlayerSide,
        field_width: f32,
        field_height: f32,
        team: TeamTacticalState,
        ball_y: f32,
        refs: RuleRefs,
    ) -> Vector3<f32> {
        let dir = match side {
            PlayerSide::Left => 1.0,
            PlayerSide::Right => -1.0,
        };
        let centre = field_height / 2.0;
        match self.action {
            ShapeAction::StepIntoMidfield => Vector3::new(
                (base.x + dir * field_width * 0.16).clamp(20.0, field_width - 20.0),
                base.y + (centre - base.y) * 0.5,
                0.0,
            ),
            ShapeAction::HoldWidth => {
                let ty = if base.y < centre { 28.0 } else { field_height - 28.0 };
                Vector3::new(
                    (base.x + dir * field_width * 0.05).clamp(20.0, field_width - 20.0),
                    ty,
                    0.0,
                )
            }
            ShapeAction::TuckInside => {
                Vector3::new(base.x, base.y + (centre - base.y) * 0.55, 0.0)
            }
            ShapeAction::CompactBackLine => {
                // Depth converges to the team-shared line height
                // (already phase-aware: high in Attack/HighPress, deep
                // in LowBlock — see TeamTacticalState's own doc
                // comments). Lateral position squeezes this player's
                // own lane toward the centre in proportion to
                // `compactness_target` (0 = keep the formation's own
                // spread, ~0.6 = squeeze most of the way to the centre
                // at full compactness) — the same shape
                // `HoldingLineState::calculate_lateral_position` already
                // uses locally (`ball_offset * (0.08 + compactness *
                // 0.18)`), generalized here into a structural rule that
                // applies regardless of which state a defender happens
                // to be in, rather than only when one specific state's
                // own velocity code consults it.
                //
                // realism-bug (2026-08-30, deflection-rate investigation):
                // the squeeze-toward-centre above compacts the LANE
                // WIDTH, but on its own it re-centres the whole back
                // line on the pitch's fixed midline regardless of where
                // the ball actually is — dropping the other half of
                // `HoldingLineState`'s own pattern this action was meant
                // to generalize, the ball-side SHIFT
                // (`ball_offset * (0.08 + compactness * 0.18)`, added to
                // the anchor rather than used to compute a squeeze
                // fraction). A temporary diagnostic (`shot_dispatch_diag`,
                // player/events/players.rs) measured that at the instant
                // a shot is struck, opposing outfielders are generally
                // nearby (~4.7m on average — this squeeze IS working)
                // but almost never aligned with the actual shot corridor
                // — exactly what a back line compact around the pitch's
                // permanent centre, rather than shifted toward the
                // ball's actual side, would produce. Applied to the
                // squeezed lane position (not `base.y` directly) so the
                // compactness and ball-side effects compound the way
                // they already do in `HoldingLineState`.
                let lane_offset = base.y - centre;
                let squeeze = (0.15 + team.compactness_target * 0.45).clamp(0.15, 0.60);
                let squeezed_y = centre + lane_offset * (1.0 - squeeze);
                let ball_offset = ball_y - centre;
                let ball_shift = ball_offset * (0.08 + team.compactness_target * 0.18);
                Vector3::new(
                    team.defensive_line_x.clamp(20.0, field_width - 20.0),
                    (squeezed_y + ball_shift).clamp(20.0, field_height - 20.0),
                    0.0,
                )
            }
            ShapeAction::SwapRole => base,
            ShapeAction::Move(spec) => {
                // Manager's-left sign in raw field Y (Left-side teams
                // attack +x and their left is the low-Y touchline).
                let left_sign = match side {
                    PlayerSide::Left => -1.0,
                    PlayerSide::Right => 1.0,
                };
                let own_sign = if base.y < centre { -1.0 } else { 1.0 };
                let sign = match spec.side {
                    SideRef::Own => own_sign,
                    SideRef::Opposite => -own_sign,
                    SideRef::Left => left_sign,
                    SideRef::Right => -left_sign,
                    SideRef::BallSide => {
                        if (ball_y - centre).abs() < field_height * 0.05 {
                            own_sign
                        } else if ball_y < centre {
                            -1.0
                        } else {
                            1.0
                        }
                    }
                };
                let y = match spec.lane {
                    LaneRef::Keep => match (spec.anchor, refs.opponent) {
                        (Anchor::Opponent(_), Some((_, oy))) => oy,
                        _ => base.y,
                    },
                    LaneRef::Flank => centre + sign * (centre - 28.0),
                    LaneRef::HalfSpace => centre + sign * field_height * 0.2,
                    LaneRef::Centre => centre,
                    LaneRef::ToCentre(p) => base.y + (centre - base.y) * (p as f32 / 1000.0),
                };
                let anchor_x = match spec.anchor {
                    Anchor::Base => base.x,
                    Anchor::LastDefender => refs.last_def_x.unwrap_or(base.x),
                    Anchor::Ball => refs.ball_x,
                    Anchor::Opponent(_) => refs.opponent.map(|o| o.0).unwrap_or(base.x),
                };
                Vector3::new(
                    (anchor_x + dir * field_width * (spec.depth_permille as f32 / 1000.0))
                        .clamp(20.0, field_width - 20.0),
                    y.clamp(20.0, field_height - 20.0),
                    0.0,
                )
            }
            ShapeAction::DropShort => Vector3::new(
                (base.x - dir * field_width * 0.04).clamp(20.0, field_width - 20.0),
                base.y,
                0.0,
            ),
            ShapeAction::DropDeeper => Vector3::new(
                (base.x - dir * field_width * 0.12).clamp(20.0, field_width - 20.0),
                base.y,
                0.0,
            ),
        }
    }
}

/// Off-ball defensive duty, assigned team-wide once per tick by
/// `assign_defensive_duties` (docs/off-ball-defensive-positioning-from-
/// null.md — domain 1 of the from-null decision rewrite,
/// docs/decision-from-null.md). Replaces the old per-player, scan-from-
/// my-own-position marking/covering/pressing decisions for defenders and
/// midfielders out of possession against a clear opponent ball carrier.
/// `None` whenever the system isn't engaged this tick (team in
/// possession, ball loose, goalkeeper, forward — forwards' own
/// defensive duty is a distinct, not-yet-built domain).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DefensiveDuty {
    #[default]
    None,
    /// Closing down the ball carrier at a real jockeying distance.
    Press { opponent_id: u32 },
    /// Tracking a specific dangerous opponent goal-side, at a tightness
    /// that scales with how threatening their position is.
    Mark { opponent_id: u32 },
    /// Holding a goal-side position on the ball-to-goal line, ready to
    /// become the new presser instantly if the presser is beaten.
    Cover,
    /// Holding a coordinated zonal shape position — no live individual
    /// responsibility this tick.
    HoldShape,
    /// Box-emergency override: converge on genuine danger near our own
    /// goal, fanned out to avoid stacking on a teammate.
    Scramble,
}

/// Off-ball attacking positioning/support, assigned team-wide once per
/// tick by `assign_attacking_support` (docs/off-ball-attacking-
/// positioning-from-null.md — domain 3 of the from-null decision
/// rewrite, docs/decision-from-null.md). Governs every outfield
/// teammate of the current ball carrier (never the carrier himself —
/// his own decision is domain 2's `on_ball_decision`). `None` whenever
/// the system isn't engaged this tick (team out of possession, ball
/// loose, goalkeeper, directed player).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AttackDuty {
    #[default]
    None,
    /// A genuinely wide player holding the flank, tracking the ball's
    /// own advancement and this side's current possession phase
    /// (build-up/progression/final-third).
    Width,
    /// A central player advancing as part of a coordinated attacking
    /// line, tracking this side's current possession phase. Real
    /// football's "someone stays back for balance" property emerges
    /// from this duty's own per-role, per-phase depth caps (a central
    /// defender's cap stays shallow even in the final third; a
    /// wide/wingback defender's does not) — replacing the old fixed
    /// rest-defense headcount rule entirely (2026-09-24, phase-based
    /// redesign, `docs/off-ball-attacking-positioning-phase-redesign.md`),
    /// per Pavel's own framing: "we do not need 'stay back' — you play
    /// as the line, we go back as the line."
    Advance,
    /// An advanced player pushing into genuinely open space beyond the
    /// line position — a real forward/decoy run, or a forward's
    /// "in-behind" choice during the progression phase.
    Penetrate,
    /// Domain 4 (`box_attack.rs`): an attacker assigned to arrive in a
    /// specific penalty-area zone (near post, far post, penalty spot,
    /// box edge) while the carrier is within crossing/shooting range.
    BoxRun,
    /// The designated target of a pass already in flight
    /// (`Ball.pass_target_player_id`). Overrides whatever Width/Advance
    /// target this player would otherwise get — real football's most
    /// basic rule for a receiver: adjust to the ball actually coming to
    /// you, don't retreat away from it because an unrelated team-shape
    /// formula recomputed your position this tick. Added 2026-09-23,
    /// Pavel's direct, unhedged report: "#10 passes to #6 but #6
    /// retracts so the pass lands short... the opponent came into the
    /// space between them and took the ball."
    Receive,
    /// Off-ball attacking runs, from null
    /// (`docs/off-ball-runs-from-null.md`, `off_ball_runs.rs`): a
    /// forward/winger/attacking-midfielder committed to one of a real
    /// candidate run (in-behind, an open flank pocket, a wide-deep
    /// byline run, or checking short to link play) during the
    /// Progression/Final phases of our own possession — the replacement
    /// for those roles' flat `Width`/`Advance`/`Penetrate` targets in
    /// those phases. `BoxRun` stays the separate, existing duty for
    /// domain 4's own penalty-area zone assignment; this covers
    /// everything short of that.
    Run,
    /// A short-support option for a carrier pinned in the byline pocket:
    /// a nearby teammate comes to a safe passing position a few metres
    /// infield of him (away from his marker), so the carrier can pass
    /// instead of dithering. Assigned by `attacking_support.rs`.
    Support,
}

/// The phase of play an instruction is limited to ("in the build-up",
/// "in the attack"). Phases are the thirds of the pitch by the ball's
/// distance from the team's own goal: build-up = own third, middle = middle
/// third, attack = final third. `Any` = the whole match (no phase word).
/// The fade between phases lives in `directive_bias::phase_weight`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DirectivePhase {
    #[default]
    Any,
    BuildUp,
    Middle,
    Attack,
}

impl DirectivePhase {
    /// Wire name -> phase. Unknown names are None (caller logs and treats as Any).
    pub fn parse(name: &str) -> Option<DirectivePhase> {
        match name {
            "build_up" => Some(DirectivePhase::BuildUp),
            "middle" => Some(DirectivePhase::Middle),
            "attack" => Some(DirectivePhase::Attack),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatchPlayer {
    pub id: u32,
    pub position: Vector3<f32>,
    pub start_position: Vector3<f32>,
    pub attributes: PersonAttributes,
    pub team_id: u32,
    pub player_attributes: PlayerAttributes,
    pub skills: PlayerSkills,
    pub tactical_position: TacticalPositions,
    pub velocity: Vector3<f32>,
    pub side: Option<PlayerSide>,
    pub state: PlayerState,
    pub in_state_time: u64,
    pub statistics: MatchPlayerStatistics,
    pub use_extended_state_logging: bool,

    pub waypoint_manager: WaypointManager,

    pub memory: PlayerMemory,

    /// Accumulates fractional condition changes across ticks
    pub fatigue_accumulator: f32,

    /// Cached waypoint vectors (only changes on substitution/half-time swap)
    cached_waypoints: Vec<Vector3<f32>>,

    /// Signature moves (PPMs) — read by decision helpers to bias behaviour.
    pub traits: Vec<PlayerTrait>,

    /// Yellow cards accumulated in this match. 2 → red.
    pub yellow_cards: u8,
    /// Fouls committed in this match. Feeds end-of-match stats.
    pub fouls_committed: u8,
    /// Player has been sent off — skip state processing, treat as off field.
    pub is_sent_off: bool,
    /// Ticks of physical off-balance left after a tackle attempt (any
    /// outcome). While > 0 the player moves slowly and turns badly
    /// (`PlayerMatchState::process`), cannot start another challenge
    /// (`can_attempt_tackle`), reaches less far for the ball
    /// (`contact_reach`) and controls it worse. It ends on its own when
    /// balance is restored; there is no tackle cooldown. Replaces the old
    /// flat 30 s `tackle_cooldown` lockout (2026-10-24).
    pub off_balance_ticks: u16,
    /// The length this spell started at, so the recovery can be graded
    /// (fully off balance at the start, nearly recovered at the end).
    pub off_balance_total: u16,
    /// Consecutive ticks this player's final velocity has been ~zero.
    /// Reset to 0 the instant real movement is detected. Root-cause-
    /// agnostic safety net (2026-08-04) — see `PlayerMatchState::process`
    /// for the detection/recovery logic. Does not replace fixing the
    /// underlying state-machine gap when one is found; it's a backstop
    /// for whichever gap hasn't been found yet.
    pub stuck_ticks: u16,
    /// Tagged reason for the next Shoot event. Set by each transition
    /// point that routes into the Shooting state (e.g. "FWD_RUN_PRIO05",
    /// "FWD_POINT_BLANK", "MID_POINT_BLANK_RUN"). The Shooting state
    /// reads this and attaches it to the emitted Shoot event so the
    /// per-match shot-reason log shows which code path fired the shot.
    /// Cleared after consumption.
    pub pending_shot_reason: Option<&'static str>,
    /// Manager-issued per-match movement directive (at most one).
    /// Injected via the match API; checked by strategy states at their
    /// decision points. See `BehavioralDirective` in club/player/ability.
    /// May be swapped at runtime by a teammate trigger (see below);
    /// `directive_base` holds the static value it reverts to.
    pub behavioral_directive: Option<BehavioralDirective>,
    /// The statically-assigned directive (from the match API). The
    /// effective `behavioral_directive` reverts to this when a teammate
    /// trigger deactivates.
    pub directive_base: Option<BehavioralDirective>,
    /// Cross-player defensive assignment: press this opponent whenever
    /// they have the ball (velocity override in the state processor).
    pub press_target: Option<u32>,
    /// Cross-player defensive assignment: man-mark this opponent while
    /// our team is out of possession (goal-side stick, plus the press
    /// chase when they receive).
    pub mark_target: Option<u32>,
    /// This tick's off-ball defensive duty, from the from-null rewrite
    /// (see `DefensiveDuty`'s own doc comment). Distinct from
    /// `press_target`/`mark_target` above, which are manager-issued
    /// explicit directives and always take priority over this default
    /// AI assignment.
    pub ai_defensive_duty: DefensiveDuty,
    /// Where this tick's `ai_defensive_duty` says to stand — already
    /// computed by the assignment pass; the velocity override just
    /// steers there.
    pub ai_defensive_target: Vector3<f32>,
    /// True only for the assigned presser, only on the tick he commits
    /// to an actual tackle attempt (the from-null spec's Step H). The
    /// physical outcome of the attempt is a kept, not-rewritten
    /// mechanic (`DefenderTacklingState`/`MidfielderTacklingState`).
    pub ai_defensive_engage: bool,
    /// This tick's off-ball attacking positioning/support duty, from the
    /// from-null rewrite (see `AttackDuty`'s own doc comment).
    pub ai_attack_duty: AttackDuty,
    /// Where this tick's `ai_attack_duty` says to stand.
    pub ai_attack_target: Vector3<f32>,
    /// Off-ball attacking runs, from null
    /// (`docs/off-ball-runs-from-null.md`). Ticks remaining on a frozen
    /// run commitment (0 = inactive) — the same "freeze the target,
    /// don't re-derive it every tick" idiom already proven by
    /// `byline_commitment_ticks`/`give_and_go_run_ticks`. While active,
    /// `assign_attacking_support` reuses `run_commit_target`/
    /// `run_commit_duty` unchanged instead of re-evaluating candidates;
    /// decremented once per tick and released the instant this player
    /// actually receives the ball (`engine/run.rs` mirrors every other
    /// commitment's own decay site, and `off_ball_runs.rs` reads it as
    /// the "already claimed this run" fact when planning teammates).
    pub run_commit_ticks: u16,
    pub run_commit_target: Vector3<f32>,
    pub run_commit_duty: AttackDuty,
    /// Consecutive ticks since this player last had the ball — the
    /// staleness input that makes the value of a spot he already holds
    /// decay over time, so runs alternate (check short, run in behind,
    /// byline, back) instead of freezing on one spot forever. Resets to
    /// 0 the instant he receives the ball.
    pub run_stale_ticks: u16,
    /// Off-ball support spot (`off_ball_support.rs`): the spot this player
    /// last chose as his support position, where he has been standing
    /// (`support_anchor`) and since when — the dwell timer that forces a
    /// relocation instead of freezing on one spot.
    pub support_spot: Option<Vector3<f32>>,
    pub support_anchor: Vector3<f32>,
    pub support_anchor_since_ms: u64,
    /// Cross-player attacking trigger: when teammate `.0` has the ball,
    /// this player's effective directive becomes `.1` (evaluated per
    /// possession change in the engine loop).
    pub teammate_trigger: Option<(u32, BehavioralDirective)>,
    /// §11.5 kickoff short-pass window, in remaining ticks (0 = inactive).
    /// Set on the kickoff taker by `assign_kickoff`; while active and the
    /// player still owns the ball, the state processor forces the role's
    /// Passing state so the taker taps a short ball instead of setting
    /// off on a solo carry. Cleared in the engine loop the moment the
    /// taker releases the ball or the budget runs out.
    pub kickoff_pass_pending: u16,
    /// realism-bug (2026-07-25): opportunistic byline-run commitment, in
    /// remaining ticks (0 = inactive). Armed probabilistically (see
    /// `run.rs`) when a player gains fresh possession in a wide channel
    /// with room to advance — real football's winger-vs-isolated-back
    /// trigger, not something a single-tick value comparison could
    /// reliably sustain (the value function's own per-tick winner
    /// (`carry_candidates`) proved too fragile — a small geometry shift
    /// could flip it before a run ever developed). While active, the
    /// forward/midfielder Dribbling states bypass the generic exits that
    /// would otherwise cancel a genuine wide duel (dribble timeout,
    /// single-marker pass-out, premature should_cross) — mirroring the
    /// BylineAndCross DIRECTIVE's own already-proven bypass, just
    /// triggered for undirected play instead of an explicit instruction.
    /// Cleared on ball release, genuine two-man pressure, or budget
    /// expiry (engine loop, same pattern as the three fields above).
    pub byline_commitment_ticks: u16,
    /// Ticks this player has been the ball carrier inside the byline pocket
    /// (within 12.5m of the goal line, wide). Drives the stall escalation in
    /// `box_attack::decide_final_third` (a carrier pinned in the corner stops
    /// carrying and delivers/passes) and the short-support duty. Updated once
    /// per tick in `assign_attacking_support`.
    pub pocket_ticks: u16,
    /// realism-bug (2026-09-12): a forward's off-ball width is governed
    /// by a per-tick blend (`ForwardRunningState`/`ForwardCreatingSpace
    /// State`) whose every input (ball position, start-position anchor,
    /// slot quantile) is itself correlated with where forwards already
    /// tend to be — a self-reinforcing loop that two direct fixes to
    /// those blends only dented (raw position measurement: forward
    /// flank-time 2.4%->2.7-3.0%, still far below the ~25-28% seen for
    /// midfielders/defenders). Mirrors `byline_commitment_ticks`'s own
    /// proven pattern: a PERSISTENT, probabilistically-armed commitment
    /// that survives across ticks, immune to the same per-tick pull-back
    /// that neutralized a continuous side-density nudge (measured to
    /// make things WORSE, reverted same session). While active, forces
    /// the forward's off-ball target toward genuine flank depth for a
    /// bounded window (a real striker's occasional channel run to
    /// stretch play, not a permanent freeze) — armed rarely, only when
    /// not on the ball, team in settled possession, and genuine room
    /// exists; released on ball reception, genuine pressure, or budget
    /// expiry, same discipline as every other commitment field here.
    pub width_commitment_ticks: u16,
    /// realism-bug (2026-08-27): persistent assignment for
    /// `wide_run_track_override_velocity` (processor.rs) — the opponent
    /// this defender is currently shadowing on a wide run toward the
    /// byline, if any. Without persistence, the per-tick "who's the
    /// single closest defender" recomputation reassigns the tracker
    /// almost every tick as relative distances jitter slightly, so no
    /// one defender ever commits to a sustained chase long enough to
    /// close a real gap — the same thrashing `should_yield_takeball`'s
    /// own HYSTERESIS constant exists to prevent, applied here via an
    /// explicit persistent target instead.
    pub wide_run_track_target: Option<u32>,
    /// Ticks remaining on the above assignment — refreshed every tick
    /// the tracked opponent still satisfies the danger criteria, so a
    /// genuine sustained run keeps the same defender committed, while
    /// one that stops (loses the ball, gets tackled, moves away) decays
    /// within under a second with nothing else needed to clear it.
    pub wide_run_track_ticks: u16,
    /// Chunk C (docs/zonal-coverage-rotation-and-tackle-risk-design.md,
    /// 2026-09-15) — same persistence idiom as `wide_run_track_target`
    /// just above, for the same reason: a raw per-tick trace found the
    /// naive "is a teammate committing to engage RIGHT NOW" check
    /// flickering (median contiguous run length was 1 tick) because
    /// `DefenderState::Pressing`/`Tackling` combined with a real
    /// displacement-from-zone check doesn't hold steady tick to tick
    /// during a live duel. The teammate id this defender is currently
    /// rotating to cover for, if any.
    pub rotation_cover_target_id: Option<u32>,
    /// Ticks remaining on the above assignment — refreshed every tick
    /// the target teammate still qualifies as committing, decays
    /// otherwise (same shape as `wide_run_track_ticks`).
    pub rotation_cover_ticks: u16,
    /// P5 (A3, `docs/attacking-defensive-decision-rebuild-plan.md`
    /// §2.1/§2.2) — how many consecutive ticks this player has occupied
    /// their current `PitchLane`. Incremented each tick the lane stays
    /// the same, reset to 0 on a lane change (engine loop, same
    /// per-tick pattern as `byline_commitment_ticks`/`kickoff_pass_
    /// pending` above). Operationalizes §8.1's mobility principle
    /// ("never predictable or repetitive") — nothing in the engine
    /// previously measured lane repetition at all.
    pub lane_residency_ticks: u16,
    /// The `PitchLane` index (0-4) `lane_residency_ticks` is counted
    /// against — `u8::MAX` sentinel for "not yet initialised" so the
    /// very first tick doesn't read as a false 0-tick match.
    pub lane_residency_lane: u8,
    /// P6 (`docs/attacking-defensive-decision-rebuild-plan.md` §0.4/
    /// §2.2/§6) — AI-lever multipliers for P1-P5's attacking mechanisms,
    /// set identically on every player of a team by the API request
    /// layer (same convention as `attack_bias` above — a per-player
    /// copy of a genuinely team-level directive, not per-player state).
    /// 0-2 range, 1.0 = neutral (no directive). `spatial_width_emphasis`
    /// scales P0's `DISCOUNT_PER_WIDE_MARKED_PLAYER` (feeds A2's
    /// relational-width discount); `spatial_lane_discipline` scales A1's
    /// lane-occupancy penalty/bonus (both on-ball and off-ball);
    /// `spatial_mobility` scales A3's mobility/residency bonus.
    pub spatial_width_emphasis: f32,
    pub spatial_lane_discipline: f32,
    pub spatial_mobility: f32,
    /// P6 — defensive-side AI-lever multipliers, same convention.
    /// `defensive_compactness` scales D3's central-corridor-priority
    /// weight; `defensive_show_outside` (true = show outside, the real-
    /// doctrine default; false = show inside) flips D1's jockey-steering
    /// sign; `defensive_press_trap_emphasis` scales D2's cover-shadow
    /// radius bonus (read from the PRESSER's own copy, not the passer's
    /// — see `evaluator.rs::calculate_interception_risk`).
    pub defensive_compactness: f32,
    pub defensive_show_outside: bool,
    pub defensive_press_trap_emphasis: f32,
    /// realism-bug (2026-07-27): a genuine, clean 1v1-vs-GK opportunity had
    /// no code path at all that could choose "carry the ball around the
    /// keeper" — evaluate_forward_shot_decision only ever returns Shoot/
    /// Pass/Hold, and the existing GK-avoidance steering in
    /// ForwardDribblingState::velocity() only sidesteps to open a shooting
    /// ANGLE, never continues past him into the net. Same persistent-ticks
    /// pattern as byline_commitment_ticks: while active, ForwardDribblingState
    /// skips its normal shot-dispatch priorities and velocity() steers past
    /// the keeper's current position instead of at/around him for an angle.
    /// Cleared on ball release or on release-condition (carrier's position
    /// has advanced past the keeper toward goal).
    pub round_keeper_commitment_ticks: u16,
    /// realism-bug follow-up (2026-07-27): frozen at arm time (not
    /// recomputed live from the GK's current position). Raw tick trace
    /// showed the GK retreating toward his own goal in the SAME direction
    /// as the attacking carrier during the whole chase -- continuously
    /// re-targeting his live position meant the effective closing rate was
    /// only the DIFFERENCE of their speeds (~0.235u/tick measured, far
    /// below either player's own ~0.35u/tick absolute speed), so the gap
    /// never closed inside any reasonable ticks budget. A real striker
    /// commits to a route based on the keeper's position at the moment of
    /// decision, not a target that keeps receding as the keeper recovers.
    pub round_keeper_target_x: f32,
    pub round_keeper_target_y: f32,
    /// Chunk 5b (docs/zone-staged-possession-progression.md): the
    /// "give" half of a give-and-go started under
    /// `AttackPattern::UnderlapOverlapCombination` (`run.rs`'s arm
    /// branch, sibling to Chunk 5's own `WideIsolation` one) — biases
    /// THIS player's own pass-target preference toward the named
    /// teammate, mirroring `link_target`'s exact evaluator treatment
    /// (`evaluator.rs`, two sites) but as a separate, self-decaying
    /// field so a temporary tactical nudge can never clobber or be
    /// clobbered by a manager-issued "link with X" pairing on either
    /// player. Only the PASSER's own field is ever read (same as
    /// `link_target`), so arming this on the carrier alone is
    /// sufficient — no second-player mutation needed from the per-tick
    /// loop that arms it.
    pub combo_partner: Option<u32>,
    /// Ticks remaining on `combo_partner` (0 = inactive). Cleared
    /// immediately once this player no longer owns the ball (passed it
    /// away or lost it) — the bias is only meaningful while he's still
    /// deciding what to do WITH the ball.
    pub combo_partner_ticks: u16,
    /// Chunk 5b: the "go" half — ticks remaining on a committed
    /// off-ball run into the space the carrier was trying to exploit,
    /// armed at the same moment as `combo_partner`/`combo_partner_ticks`
    /// above but consumed AFTER the ball has left his feet (the
    /// override that reads this, `give_and_go_run_velocity` in
    /// `processor.rs`, only fires while he does NOT own the ball).
    /// Deliberately does not try to force a return pass — the runner
    /// becomes genuinely more open by making a real run, and the
    /// already-verified pass-value machinery
    /// (`pass_value_from`/`receiver_runway_value`) rewards feeding an
    /// open teammate on its own, the same "trust what's already built"
    /// reasoning Chunk 4's off-ball reaction used.
    pub give_and_go_run_ticks: u16,
    /// Frozen at arm time (not recomputed live), same pattern as
    /// `round_keeper_target_x/y` above and for the identical reason —
    /// a real player commits to a run decided at the moment he lays
    /// the ball off, not one that keeps drifting as he moves.
    pub give_and_go_run_target_x: f32,
    pub give_and_go_run_target_y: f32,
    /// docs/positional-football-master-plan.md Stage 2 item 3 (2026-09-07),
    /// GENERALIZED docs/positional-attack-phase-design.md Stage 3.5
    /// (2026-09-11): originally armed only for a byline-committed wide
    /// carrier meeting a genuine 2-man press (`close_defenders >= 2`
    /// within 8u) or one genuine blocker on the direct route
    /// (`single_blocker`) — real, verified, but narrow: Stage 3.5's own
    /// diagnosis named the missing piece directly — "the repeated
    /// probe → contained → retreat → probe-again shape at the center of
    /// Pavel's whole vision needs a genuine trigger for 'contained' that
    /// isn't RNG... generalizing beyond the one already-built byline
    /// case may have no real foundation without this." Now ALSO armed
    /// for ANY ball carrier, any role, any zone, who has been genuinely
    /// blocked ahead (see `carrier_containment_dwell_ticks` below) for
    /// a real, sustained moment — not just an instantaneous geometric
    /// coincidence. The original byline-specific instant-fire arm site
    /// is untouched (its own timing was already measured and verified;
    /// this generalization is purely additive, gated on the carrier NOT
    /// already being armed by it). A real "this attack just failed,
    /// look for a considered reset" moment that previously had no
    /// positive reward anywhere outside the byline case:
    /// `build_up_recycle_bonus` requires a CB/DM/GK receiver,
    /// `stalled_recycle_bonus` requires ~20s of continuous possession a
    /// quick contained dribble never reaches. Ticks remaining (0 =
    /// inactive); self-decaying like every other transient bias field
    /// on this struct, consumed by `evaluator.rs`'s
    /// `carrier_contained_reset_bonus`.
    pub carrier_contained_reset_ticks: u16,
    /// docs/positional-attack-phase-design.md, Stage 3.5 generalization
    /// (2026-09-11): consecutive ticks the current ball carrier has been
    /// genuinely blocked ahead by a close opponent (same geometry as the
    /// existing `single_blocker` check, generalized to any carrier, not
    /// just a byline-committed one). Resets to 0 the instant the
    /// carrier isn't blocked, loses the ball, or `carrier_contained_
    /// reset_ticks` is already armed (no need to keep counting once the
    /// reset bonus has fired). A real dwell requirement — not an
    /// instantaneous coincidence — so a single fleeting near-miss
    /// doesn't fire the reset; the carrier has to genuinely fail to
    /// progress for a real moment before "recognizing" containment.
    pub carrier_containment_dwell_ticks: u16,
    /// docs/positional-attack-phase-design.md, Stage 3: the forward
    /// drop-and-spin primitive. Armed while a Forward genuinely drops
    /// deep to receive/link up during a `patient_attack_phase_active()`
    /// window, decays only once the ball has left his feet (mirrors
    /// `give_and_go_run_ticks`' exact "counts down only after release"
    /// idiom, same reasoning: the run hasn't started yet while he's
    /// still deciding what to do with it). Consumed by
    /// `drop_and_spin_run_velocity` in `processor.rs`.
    pub drop_and_spin_run_ticks: u16,
    /// Frozen at arm time (same pattern as `give_and_go_run_target_x/y`
    /// and `round_keeper_target_x/y`) — "into the box" computed once
    /// from the state at the moment he received the ball deep, not
    /// recomputed live as he moves.
    pub drop_and_spin_target_x: f32,
    pub drop_and_spin_target_y: f32,
    /// realism-bug (2026-07-27): one-shot-per-possession lock so the
    /// round-keeper probability roll (in the engine loop) fires at most
    /// once per continuous spell of ball ownership, not every tick the
    /// clean-1v1 condition happens to hold (which would compound a small
    /// per-tick probability into near-certainty over a multi-second
    /// breakaway). Reset to false whenever the player doesn't own the ball.
    pub gk1v1_decision_made: bool,
    /// realism-bug (2026-07-19): throw-in taker's forced-release window,
    /// in remaining ticks (0 = inactive). Same pattern as
    /// `kickoff_pass_pending` — a throw-in is a single discrete motion
    /// in real football (the ball leaves the hands immediately; Law 15
    /// also bars the thrower touching it again until another player
    /// does), but nothing previously stopped the thrower's normal
    /// open-play state machine from dribbling him forward with the ball
    /// for seconds after the restart. Set on the thrower when the
    /// set-piece teleport applies; while active and he still owns the
    /// ball, the state processor forces the role's Passing state.
    /// Cleared the moment he releases the ball or the budget runs out.
    pub throw_in_pass_pending: u16,
    /// realism-bug (2026-07-20): free-kick taker's forced-release window,
    /// in remaining ticks (0 = inactive). Same pattern as
    /// `throw_in_pass_pending` above. Only armed for the outcomes that
    /// leave the taker to his normal open-play state machine —
    /// ShortRoutine/Recycle and the >280u "Deep" fallthrough in
    /// `resolve_free_kick` — never for DirectShot/BoxDelivery, which
    /// already dispatch a discrete Shoot/PassTo event immediately and
    /// bypass his AI entirely. Without this, a deep or short-routine
    /// free kick left the taker owning the ball like any open-play
    /// possession, free to dribble forward solo — unrealistic for a
    /// restart real teams take short and circulate, not a solo carry.
    /// Set in `resolve_free_kick`; while active and he still owns the
    /// ball, the state processor forces the role's Passing state.
    /// Cleared the moment he releases the ball or the budget runs out.
    pub free_kick_pass_pending: u16,
    /// Realism-bug 2026-07-26: set by the general (non-corner) aerial
    /// contest resolver — cross / free-kick cross / long high pass — the
    /// instant it decides this player won the header. Consumed by
    /// `ForwardHeadingState`/`MidfielderHeadingState`/`DefenderState`'s
    /// header carve-out (same clean-contact-only roll the corner-contest
    /// carve-out already uses via `is_team_attacking_corner()`) so the
    /// discretely-decided win isn't re-rolled as a second full duel.
    /// Ticks-remaining counter (not a plain bool) so it self-clears via
    /// the same per-tick decrement loop as `kickoff_pass_pending` et al.
    /// instead of needing bespoke reset logic.
    pub aerial_contest_won: u8,
    /// Realism-bug 2026-07-26 (passing follow-up): the lead-adjusted aim
    /// point of a pass currently in flight toward this player. Set (on
    /// the RECEIVER, not the passer) by `handle_pass_to_event` at pass
    /// dispatch time, using the exact same `ideal_target` the ball
    /// itself is aimed at — closing a coordination gap: previously
    /// `Ball.pending_pass_target` had zero readers anywhere in a
    /// receiving player's own movement code, so a receiver's run during
    /// the ball's flight was entirely independent of where the ball was
    /// actually headed (confirmed by a full-tree grep before this fix).
    /// `None` when no pass is currently incoming. Read by
    /// `incoming_pass_awareness_velocity` (processor.rs) as a BLENDED
    /// nudge — not a hard override — toward the target, weighted by the
    /// receiver's own off_ball_attack composite (off_the_ball/
    /// anticipation/decisions/pace) so a sharp off-ball mover reads and
    /// meets a leading ball far better than a poor one, preserving
    /// exactly the skill differentiation the passer's side already has
    /// (`anticipation`-scaled `lead_fraction` in `handle_pass_to_event`)
    /// instead of flattening every receiver into an equally good one.
    pub incoming_pass_target: Option<Vector3<f32>>,
    /// Ticks remaining on `incoming_pass_target` (0 = inactive). Self-
    /// clears via the same per-tick decrement loop as
    /// `kickoff_pass_pending` et al., sized to the passing lead
    /// formula's own flight-time ceiling (`flight_time_est.clamp(8.0,
    /// 80.0)` in `handle_pass_to_event`) plus margin, so it naturally
    /// expires around when the ball would have arrived even if nothing
    /// else clears it first.
    pub incoming_pass_ticks: u16,
    /// "Link with X" pair preference (wishlist #5): passes toward this
    /// teammate get a success/EV bias in the pass evaluator, and the
    /// pair's chemistry is pinned high at kickoff. Set symmetrically on
    /// both players by the API layer.
    pub link_target: Option<u32>,
    /// "Block passes into X" (wishlist #8): this player is assigned to
    /// cut out passes to the named OPPONENT — opposing passers see a
    /// success/selection penalty on passes to that target while this
    /// interceptor is within range of them.
    pub intercept_target: Option<u32>,
    /// "Feed X" directed supply (wishlist #9): stamped on every
    /// TEAMMATE of the named target — passes to the target are
    /// preferred, long balls especially.
    pub supply_target: Option<u32>,
    /// "Push the keeper up for corners late on" (wishlist #16): set on
    /// the GK only. From this displayed minute, own-team corners
    /// teleport him into the opposition box (no state override) and a
    /// velocity override keeps him there through the delivery, then
    /// sprints him home. The manager's risk is real — the goal is
    /// genuinely open while he's up.
    pub gk_up_after_minute: Option<u32>,
    /// §13.4: genuine restart-retreat steering target (foul wall/box-clear,
    /// corner defensive block/attacking push, goal-kick recovery shape).
    /// Set instead of an instant position snap so the player walks there
    /// tick-by-tick from wherever they actually are — a fast restart can
    /// then genuinely catch them short. Only ever read/written during a
    /// `dead_ball_retreat_active` window by the engine's light retreat
    /// tick (`FootballEngine::apply_restart_retreat_tick`, tick.rs) —
    /// never by the normal state processor, so it can't interact with
    /// open-play movement. Force-cleared on every player the instant a
    /// retreat window ends (early or at the cap), so a stale target can
    /// never leak into open play.
    pub restart_retreat_target: Option<Vector3<f32>>,
    /// Manager-issued start-position anchors (per axis, first-half
    /// coordinates). `setup_player_on_field` overwrites start_position
    /// from the formation table, which silently discarded the API's
    /// startX/startY overrides — these survive setup and are mirrored
    /// at the halftime swap.
    pub start_anchor_x: Option<f32>,
    pub start_anchor_y: Option<f32>,
    /// Whole-team shape rules (plan Phase 5): each tick the engine loop
    /// evaluates the first rule whose trigger holds and moves this
    /// player's `start_position` (the shape anchor every positional
    /// state blends toward — never a hard snap) to the action's target;
    /// with no active rule it reverts to `shape_base`.
    pub shape_rules: Vec<ShapeRule>,
    /// The formation-table start position (anchors included) captured
    /// after setup and re-captured after the halftime swap — the revert
    /// target for shape rules.
    pub shape_base: Option<Vector3<f32>>,
    /// True for the current tick iff some rule in `shape_rules` had an
    /// active trigger (i.e. `start_position` currently holds a computed
    /// rule target, not just the bare `shape_base`). Set alongside
    /// `start_position` in the engine loop's shape-rule pass
    /// (Principles-of-Play plan, 2026-08-30). Exists so
    /// `processor.rs::shape_rule_override_velocity` can tell "a
    /// structural rule wants this player somewhere specific right now"
    /// apart from "this player merely has a rule list" — without it, a
    /// generalized override reading `start_position` unconditionally
    /// would also force every rule-holding player back onto their bare
    /// formation anchor whenever no rule is active, which is much
    /// broader than intended.
    pub shape_rule_active: bool,
    /// Team-wide directional attack bias: -1 = manager's left, 1 =
    /// right, 0 = centre. Pass-evaluator weight toward that lateral
    /// third ("build down the left").
    pub attack_bias: Option<i8>,
    /// Phase limits for the per-player instruction fields (see `DirectivePhase`).
    pub attack_bias_phase: DirectivePhase,
    /// Optional second flank preference for another phase ("left in the
    /// build-up, right in the attack").
    pub attack_bias_b: Option<i8>,
    pub attack_bias_b_phase: DirectivePhase,
    pub supply_phase: DirectivePhase,
    pub link_phase: DirectivePhase,
    pub directive_phase: DirectivePhase,
    /// True when at least one of `shape_rules` was issued by the manager
    /// (request `shapeRules`), as opposed to the engine's own default
    /// rules. Manager rules are re-applied LAST in the velocity chain so the
    /// new defensive/attacking systems cannot overwrite them.
    pub manager_shape: bool,
    /// Number of leading entries of `shape_rules` that are manager-issued
    /// (they are pushed before the engine's defaults), and whether the rule
    /// active this tick is one of them.
    pub manager_rule_count: u8,
    pub shape_manager_active: bool,
    /// Rule 2.0: the active manager rule's lateral part is an ABSOLUTE
    /// lane (replaces the duty target's y) rather than an offset on it.
    pub shape_abs_y: bool,
    /// Same for depth when the rule anchors to a live reference (last defender, ball, opponent).
    pub shape_abs_x: bool,
    /// Rule 2.0: the ball lane (-1/0/1, manager's side) this player's
    /// ball-lane triggers last saw, kept so the lane only changes after the
    /// ball is clearly over the boundary (hysteresis against flicker).
    pub shape_lane_prev: i8,
    /// Manager counter-press (identical on every player of a team):
    /// for this many ticks after the team loses the ball, `counterpress_pressers`
    /// players (the normal presser plus tight marks on the carrier's nearest
    /// passing options) close the ball down. 0 ticks = off.
    /// Manager defensive line height, -1 (deep) .. +1 (high); 0 = default.
    pub defensive_line_height: f32,
    /// Manager press triggers (bitmask, identical on a team; 0 = press always):
    /// 1 carrier is a full-back, 2 centre-back, 4 goalkeeper, 8 back pass,
    /// 16 ball in their own third, 32 carrier is a winger. With a mask set the
    /// team presses only while a trigger holds (or in the counter-press window).
    pub press_trigger_mask: u8,
    pub counterpress_ticks: u16,
    pub counterpress_pressers: u8,
    /// Counter-press that ends in a tactical foul: if the window is nearly over and the
    /// carrier is breaking away toward our goal, the nearest player stops him with a foul.
    pub counterpress_foul: bool,
    /// Role interchange state (see `ShapeAction::SwapRole`).
    pub role_swap_wanted: Option<u32>,
    pub role_swapped_with: Option<u32>,
    pub role_swap_initiator: bool,
    pub role_swap_tick: u64,
    /// Manager opt-in: this player's off-ball runs are planned by `off_ball_runs::plan_runs`
    /// (valued in-behind / pocket / byline / check-short runs) instead of the flat duty tables.
    pub runs_enabled: bool,

    /// Manager-set build-up-vs-direct dial, -2 (possession) .. +2
    /// (direct), 0 = neutral. Pass-evaluator weight on long-pass
    /// selection and on the short recycle-to-the-back bonus — genuinely
    /// changes how often the team plays out from the back vs. goes
    /// long, not just a skill-stat flavour nudge.
    pub directness_bias: f32,

    /// Manager flag protecting this player from fatigue / development subs.
    /// Mirrored from `Player::is_force_match_selection` at squad-build time.
    pub is_force_match_selection: bool,

    /// Player's birth date, mirrored from the source `Player`. Read by the
    /// in-match substitution logic to apply age-appropriate protection
    /// thresholds for under-18 players (lower fatigue ceiling, condition
    /// floor that overrides the manager's force-selection flag).
    pub birth_date: NaiveDate,

    /// Match-time (in ms) at which the player entered the field.
    /// Starters are stamped with 0; substitutes are stamped with
    /// `context.total_match_time` at swap time. The end-of-match rating
    /// helper computes minutes-played as
    /// `(exit_or_now - entry_match_time_ms) / 60_000`, which fixes the
    /// previous behaviour where every player — even an 80th-minute sub —
    /// was credited with full match minutes.
    pub entry_match_time_ms: u64,
    /// Last tick at which a pressure event was credited for this
    /// player. Used as a per-player cooldown so a defender shadowing
    /// the carrier across many ticks racks up one pressure per "press
    /// burst" rather than one per tick.
    pub last_pressure_tick: u64,

    /// Condition the player carried onto the pitch — kickoff for
    /// starters, swap-time for substitutes. Stamped once at
    /// construction (or at the substitution swap) and never mutated
    /// after that. Read by the engine end-of-match path to build the
    /// `PlayerMatchPhysicalSnapshot` that drives the post-match
    /// condition-drop formula on the persisted `Player`.
    pub starting_condition: i16,

    /// Recovery debt the player walked onto the pitch with — copied
    /// from `Player::load::recovery_debt` at construction (or at the
    /// substitution swap) and never mutated during the match. Read by
    /// the in-match condition processor so a player with heavy legs
    /// tires faster than a fresh player with the same NF/stamina, and
    /// so a back-to-back schedule shows up in the tick-by-tick drain
    /// not just the post-match summary. Lives on MatchPlayer (rather
    /// than reaching back through `PlayerLoad`) so the hot per-tick
    /// path doesn't touch the persisted simulator data.
    pub starting_recovery_debt: f32,

    /// Home-crowd arousal multiplier applied inside `effective_skill`
    /// (1.0 = neutral). Stamped once at match start from the match
    /// environment: home players slightly above 1.0, away players
    /// slightly below, both scaled by `crowd_intensity ×
    /// home_advantage`. This is the play-quality half of home
    /// advantage (the documented crowd-arousal / travel-discomfort
    /// effect, worth ~+0.35 home goals at equal strength in real
    /// football); the officiating half lives in
    /// `RefereeProfile::home_bias`. Flows through every skill-mediated
    /// action via `effective_skill`, so it shifts duels, passing,
    /// saves and finishing continuously instead of dialling any single
    /// outcome.
    pub crowd_arousal: f32,
}

impl MatchPlayer {
    /// Age in whole years on `today`. Defined here (not on `Player`) so
    /// match-side code never has to reach back through the simulator
    /// data graph.
    #[inline]
    pub fn age_at(&self, today: NaiveDate) -> u8 {
        DateUtils::age(self.birth_date, today)
    }
}

impl MatchPlayer {
    /// Fast trait lookup used inside hot decision paths.
    #[inline]
    pub fn has_trait(&self, t: PlayerTrait) -> bool {
        self.traits.iter().any(|x| *x == t)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PlayerSide {
    Left,
    Right,
}

impl PlayerSide {
    /// Forward direction along the pitch's x-axis: +1 toward larger x
    /// for Left teams, -1 for Right. The math everywhere else in the
    /// engine uses this — never write `match side { Left => 1.0, ... }`
    /// inline or you'll inevitably miss a sign and produce the
    /// "right-side never reaches attacking third" bug we hit before.
    #[inline]
    pub fn forward_dir_x(self) -> f32 {
        match self {
            PlayerSide::Left => 1.0,
            PlayerSide::Right => -1.0,
        }
    }

    /// Attacking progress for an x-coordinate, normalised to [0.0, 1.0]:
    ///   0.0 = own goal line, 1.0 = opponent goal line.
    /// Use this everywhere a "what fraction of the pitch have we
    /// covered" check is needed (attacking third, defensive third,
    /// halfway thresholds). Replaces the legacy
    ///   `(x * forward_dir) / field_width`
    /// formula, which produced NEGATIVE values for right-side teams —
    /// that broke `> 0.66` (attacking third) tests by construction
    /// while leaving `< 0.33` (defensive third) accidentally correct.
    #[inline]
    pub fn attacking_progress_x(self, x: f32, field_width: f32) -> f32 {
        if field_width <= 0.0 {
            return 0.0;
        }
        let raw = match self {
            PlayerSide::Left => x / field_width,
            PlayerSide::Right => (field_width - x) / field_width,
        };
        raw.clamp(0.0, 1.0)
    }

    /// Forward delta along x: positive = toward opponent goal, negative
    /// = toward own goal. The right shape for "is this pass forward?",
    /// "did this player advance?", and any signed forward-progress
    /// arithmetic. Side-direction is baked in.
    #[inline]
    pub fn forward_delta(self, from_x: f32, to_x: f32) -> f32 {
        (to_x - from_x) * self.forward_dir_x()
    }

    /// Same as `forward_delta` but normalised by field width — gives a
    /// signed [-1.0, 1.0]-ish ratio for the cases that previously used
    /// `((to.x - from.x) * dir) / field_width`.
    #[inline]
    pub fn forward_delta_norm(self, from_x: f32, to_x: f32, field_width: f32) -> f32 {
        if field_width <= 0.0 {
            return 0.0;
        }
        self.forward_delta(from_x, to_x) / field_width
    }
}

impl MatchPlayer {
    pub fn from_player(
        team_id: u32,
        player: &Player,
        position: PlayerPositionType,
        use_extended_state_logging: bool,
    ) -> Self {
        MatchPlayer {
            id: player.id,
            position: Vector3::zeros(),
            start_position: Vector3::zeros(),
            attributes: player.attributes,
            team_id,
            player_attributes: player.player_attributes,
            skills: player.skills,
            velocity: Vector3::zeros(),
            tactical_position: TacticalPositions::new(position, None),
            side: None,
            state: Self::default_state(position),
            in_state_time: 0,
            statistics: MatchPlayerStatistics::new(),
            waypoint_manager: WaypointManager::new(),
            use_extended_state_logging,
            memory: PlayerMemory::new(),
            fatigue_accumulator: 0.0,
            cached_waypoints: Vec::new(),
            traits: player.traits.clone(),
            yellow_cards: 0,
            fouls_committed: 0,
            is_sent_off: false,
            off_balance_ticks: 0,
            off_balance_total: 0,
            stuck_ticks: 0,
            pending_shot_reason: None,
            behavioral_directive: None,
            directive_base: None,
            press_target: None,
            mark_target: None,
            ai_defensive_duty: DefensiveDuty::None,
            ai_defensive_target: Vector3::zeros(),
            ai_defensive_engage: false,
            ai_attack_duty: AttackDuty::None,
            ai_attack_target: Vector3::zeros(),
            run_commit_ticks: 0,
            run_commit_target: Vector3::zeros(),
            run_commit_duty: AttackDuty::None,
            run_stale_ticks: 0,
            support_spot: None,
            support_anchor: Vector3::zeros(),
            support_anchor_since_ms: 0,
            teammate_trigger: None,
            kickoff_pass_pending: 0,
            byline_commitment_ticks: 0,
            pocket_ticks: 0,
            width_commitment_ticks: 0,
            wide_run_track_target: None,
            wide_run_track_ticks: 0,
            rotation_cover_target_id: None,
            rotation_cover_ticks: 0,
            lane_residency_ticks: 0,
            lane_residency_lane: u8::MAX,
            spatial_width_emphasis: 1.0,
            spatial_lane_discipline: 1.0,
            spatial_mobility: 1.0,
            defensive_compactness: 1.0,
            defensive_show_outside: true,
            defensive_press_trap_emphasis: 1.0,
            round_keeper_commitment_ticks: 0,
            round_keeper_target_x: 0.0,
            round_keeper_target_y: 0.0,
            combo_partner: None,
            combo_partner_ticks: 0,
            give_and_go_run_ticks: 0,
            give_and_go_run_target_x: 0.0,
            give_and_go_run_target_y: 0.0,
            carrier_contained_reset_ticks: 0,
            carrier_containment_dwell_ticks: 0,
            drop_and_spin_run_ticks: 0,
            drop_and_spin_target_x: 0.0,
            drop_and_spin_target_y: 0.0,
            gk1v1_decision_made: false,
            throw_in_pass_pending: 0,
            free_kick_pass_pending: 0,
            aerial_contest_won: 0,
            incoming_pass_target: None,
            incoming_pass_ticks: 0,
            link_target: None,
            intercept_target: None,
            supply_target: None,
            gk_up_after_minute: None,
            restart_retreat_target: None,
            start_anchor_x: None,
            start_anchor_y: None,
            shape_rules: Vec::new(),
            shape_base: None,
            shape_rule_active: false,
            attack_bias: None,
            attack_bias_phase: DirectivePhase::Any,
            attack_bias_b: None,
            attack_bias_b_phase: DirectivePhase::Any,
            supply_phase: DirectivePhase::Any,
            link_phase: DirectivePhase::Any,
            directive_phase: DirectivePhase::Any,
            manager_shape: false,
            manager_rule_count: 0,
            shape_manager_active: false,
            shape_abs_y: false,
            shape_abs_x: false,
            shape_lane_prev: 0,
            defensive_line_height: 0.0,
            press_trigger_mask: 0,
            counterpress_ticks: 0,
            counterpress_pressers: 0,
            counterpress_foul: false,
            role_swap_wanted: None,
            role_swapped_with: None,
            role_swap_initiator: false,
            role_swap_tick: 0,
            runs_enabled: false,
            directness_bias: 0.0,
            is_force_match_selection: player.is_force_match_selection,
            birth_date: player.birth_date,
            entry_match_time_ms: 0,
            last_pressure_tick: 0,
            starting_condition: player.player_attributes.condition,
            starting_recovery_debt: player.load.recovery_debt,
            crowd_arousal: 1.0,
        }
    }

    /// Input-style constructor used by the distributed worker wire
    /// layer to rebuild a `MatchPlayer` from the bincode payload. Takes
    /// only the fields that meaningfully cross the network — engine
    /// runtime state (memory, waypoints, in-state timers, statistics,
    /// fatigue accumulator) is initialised to the same defaults
    /// `from_player` uses at kickoff, so the rebuilt player is
    /// indistinguishable from one freshly selected on the worker.
    #[allow(clippy::too_many_arguments)]
    pub fn from_inputs(
        id: u32,
        team_id: u32,
        position: [f32; 3],
        start_position: [f32; 3],
        attributes: PersonAttributes,
        player_attributes: PlayerAttributes,
        skills: PlayerSkills,
        tactical_position: PlayerPositionType,
        side: Option<PlayerSide>,
        traits: Vec<PlayerTrait>,
        birth_date: NaiveDate,
        is_force_match_selection: bool,
        starting_condition: i16,
        starting_recovery_debt: f32,
        use_extended_state_logging: bool,
    ) -> Self {
        MatchPlayer {
            id,
            position: Vector3::new(position[0], position[1], position[2]),
            start_position: Vector3::new(start_position[0], start_position[1], start_position[2]),
            attributes,
            team_id,
            player_attributes,
            skills,
            velocity: Vector3::zeros(),
            tactical_position: TacticalPositions::new(tactical_position, side),
            side,
            state: Self::default_state(tactical_position),
            in_state_time: 0,
            statistics: MatchPlayerStatistics::new(),
            waypoint_manager: WaypointManager::new(),
            use_extended_state_logging,
            memory: PlayerMemory::new(),
            fatigue_accumulator: 0.0,
            cached_waypoints: Vec::new(),
            traits,
            yellow_cards: 0,
            fouls_committed: 0,
            is_sent_off: false,
            off_balance_ticks: 0,
            off_balance_total: 0,
            stuck_ticks: 0,
            pending_shot_reason: None,
            behavioral_directive: None,
            directive_base: None,
            press_target: None,
            mark_target: None,
            ai_defensive_duty: DefensiveDuty::None,
            ai_defensive_target: Vector3::zeros(),
            ai_defensive_engage: false,
            ai_attack_duty: AttackDuty::None,
            ai_attack_target: Vector3::zeros(),
            run_commit_ticks: 0,
            run_commit_target: Vector3::zeros(),
            run_commit_duty: AttackDuty::None,
            run_stale_ticks: 0,
            support_spot: None,
            support_anchor: Vector3::zeros(),
            support_anchor_since_ms: 0,
            teammate_trigger: None,
            kickoff_pass_pending: 0,
            byline_commitment_ticks: 0,
            pocket_ticks: 0,
            width_commitment_ticks: 0,
            wide_run_track_target: None,
            wide_run_track_ticks: 0,
            rotation_cover_target_id: None,
            rotation_cover_ticks: 0,
            lane_residency_ticks: 0,
            lane_residency_lane: u8::MAX,
            spatial_width_emphasis: 1.0,
            spatial_lane_discipline: 1.0,
            spatial_mobility: 1.0,
            defensive_compactness: 1.0,
            defensive_show_outside: true,
            defensive_press_trap_emphasis: 1.0,
            round_keeper_commitment_ticks: 0,
            round_keeper_target_x: 0.0,
            round_keeper_target_y: 0.0,
            combo_partner: None,
            combo_partner_ticks: 0,
            give_and_go_run_ticks: 0,
            give_and_go_run_target_x: 0.0,
            give_and_go_run_target_y: 0.0,
            carrier_contained_reset_ticks: 0,
            carrier_containment_dwell_ticks: 0,
            drop_and_spin_run_ticks: 0,
            drop_and_spin_target_x: 0.0,
            drop_and_spin_target_y: 0.0,
            gk1v1_decision_made: false,
            throw_in_pass_pending: 0,
            free_kick_pass_pending: 0,
            aerial_contest_won: 0,
            incoming_pass_target: None,
            incoming_pass_ticks: 0,
            link_target: None,
            intercept_target: None,
            supply_target: None,
            gk_up_after_minute: None,
            restart_retreat_target: None,
            start_anchor_x: None,
            start_anchor_y: None,
            shape_rules: Vec::new(),
            shape_base: None,
            shape_rule_active: false,
            attack_bias: None,
            attack_bias_phase: DirectivePhase::Any,
            attack_bias_b: None,
            attack_bias_b_phase: DirectivePhase::Any,
            supply_phase: DirectivePhase::Any,
            link_phase: DirectivePhase::Any,
            directive_phase: DirectivePhase::Any,
            manager_shape: false,
            manager_rule_count: 0,
            shape_manager_active: false,
            shape_abs_y: false,
            shape_abs_x: false,
            shape_lane_prev: 0,
            defensive_line_height: 0.0,
            press_trigger_mask: 0,
            counterpress_ticks: 0,
            counterpress_pressers: 0,
            counterpress_foul: false,
            role_swap_wanted: None,
            role_swapped_with: None,
            role_swap_initiator: false,
            role_swap_tick: 0,
            runs_enabled: false,
            directness_bias: 0.0,
            is_force_match_selection,
            birth_date,
            entry_match_time_ms: 0,
            last_pressure_tick: 0,
            starting_condition,
            starting_recovery_debt,
            // Wire payloads predate the arousal field; the worker
            // re-stamps it at match start like the local path does.
            crowd_arousal: 1.0,
        }
    }

    /// Build the per-match end-of-game stats snapshot for this player.
    /// Owns the field-by-field mapping so the engine end-of-match path
    /// and the substitution snapshot path don't drift. `minutes_played`
    /// is computed by the caller from `entry_match_time_ms` and the
    /// exit / now match time — see `engine.rs` and `substitutions.rs`.
    pub fn to_match_end_stats(&self, minutes_played: u16) -> PlayerMatchEndStats {
        PlayerMatchEndStats {
            shots_on_target: self.memory.shots_on_target as u16,
            shots_total: self.memory.shots_taken as u16,
            passes_attempted: self.statistics.passes_attempted,
            passes_completed: self.statistics.passes_completed,
            tackles: self.statistics.tackles,
            interceptions: self.statistics.interceptions,
            saves: self.statistics.saves,
            shots_faced: self.statistics.shots_faced,
            goals: self.statistics.goals_count(),
            assists: self.statistics.assists_count(),
            match_rating: 0.0,
            raw_match_rating: 0.0,
            xg: self.memory.xg_total,
            position_group: self.tactical_position.current_position.position_group(),
            fouls: self.fouls_committed as u16,
            yellow_cards: self.statistics.yellow_cards_count(),
            red_cards: self.statistics.red_cards_count(),
            minutes_played,
            key_passes: self.statistics.key_passes,
            progressive_passes: self.statistics.progressive_passes,
            progressive_carries: self.statistics.progressive_carries,
            successful_dribbles: self.statistics.successful_dribbles,
            attempted_dribbles: self.statistics.attempted_dribbles,
            successful_pressures: self.statistics.successful_pressures,
            pressures: self.statistics.pressures,
            blocks: self.statistics.blocks,
            clearances: self.statistics.clearances,
            passes_into_box: self.statistics.passes_into_box,
            crosses_attempted: self.statistics.crosses_attempted,
            crosses_completed: self.statistics.crosses_completed,
            xg_chain: self.statistics.xg_chain,
            xg_buildup: self.statistics.xg_buildup,
            miscontrols: self.statistics.miscontrols,
            heavy_touches: self.statistics.heavy_touches,
            carry_distance: self.statistics.carry_distance,
            errors_leading_to_shot: self.statistics.errors_leading_to_shot,
            errors_leading_to_goal: self.statistics.errors_leading_to_goal,
            xg_prevented: self.statistics.xg_prevented,
            offsides: self.statistics.offsides,
            own_goals: self.statistics.own_goals_count(),
            zone_stats: self.statistics.zone_stats,
        }
    }

    /// Compute minutes spent on the pitch for this player given the
    /// current absolute match time (ms). Capped at 120 minutes so an
    /// extra-time match doesn't produce silly minute totals.
    pub fn minutes_played_at(&self, now_match_time_ms: u64) -> u16 {
        let elapsed = now_match_time_ms.saturating_sub(self.entry_match_time_ms);
        ((elapsed / 60_000) as u16).min(120)
    }

    /// Build the post-match physical snapshot for this player at the
    /// given absolute match time (substitution-off or full-time).
    /// Captures the starting tank, the current (drained) condition,
    /// and a high-intensity share that blends the position-group
    /// default with the player's actual high-intensity involvement
    /// (pressures, tackles, dribbles, crosses) so the persisted
    /// `Player::on_match_exertion` reflects how the player actually
    /// played, not just the position they nominally occupied. A
    /// fullback who pressed all match should bill more than one who
    /// sat in a low block.
    pub fn to_physical_snapshot(&self, now_match_time_ms: u64) -> PlayerMatchPhysicalSnapshot {
        let elapsed_ms = now_match_time_ms.saturating_sub(self.entry_match_time_ms);
        // Fractional minutes — `minutes_played_at` rounds down to a
        // u16 which loses information for cameo subs. The post-match
        // formula reads `duration = minutes / 90` so the fractional
        // resolution actually matters here.
        let minutes_played = (elapsed_ms as f32 / 60_000.0).min(120.0);
        let group = self.tactical_position.current_position.position_group();
        let position_default = PositionLoad::high_intensity_share(group);
        PlayerMatchPhysicalSnapshot {
            player_id: self.id,
            minutes_played,
            starting_condition: self.starting_condition,
            final_match_energy: self.player_attributes.condition,
            high_intensity_load_hint: Self::derive_high_intensity_hint(
                position_default,
                &self.statistics,
                minutes_played,
            ),
        }
    }

    /// Blend the position-group default high-intensity share with the
    /// observed action density (pressures, tackles, successful
    /// dribbles, crosses) so the post-match condition drop reflects
    /// how the player actually played. Keepers and defenders sitting
    /// deep stay near their position default; an attacking fullback
    /// who pressed every action will read materially higher.
    ///
    /// A "calibration baseline" of 0.50 actions/min maps to the
    /// position default; anything above lifts the hint linearly, up
    /// to a cap of 1.0 (the engine's mathematical ceiling). This is
    /// deliberately conservative — the position default is right for
    /// average involvement; behaviour-driven correction is a tilt,
    /// not a rewrite.
    pub(crate) fn derive_high_intensity_hint(
        position_default: f32,
        stats: &crate::r#match::engine::player::statistics::MatchPlayerStatistics,
        minutes_played: f32,
    ) -> f32 {
        if minutes_played < 1.0 {
            return position_default;
        }
        let hi_actions = stats.pressures as f32
            + stats.tackles as f32
            + stats.successful_dribbles as f32
            + stats.crosses_attempted as f32
            + stats.progressive_carries as f32;
        let per_min = hi_actions / minutes_played.max(1.0);
        // 0.50 actions/min ≈ the position default; deviation tilts the
        // hint by ±0.4 absolute at the extremes (per-min ≈ 0 → -0.4×,
        // per-min ≈ 1.5 → +0.4×). Clamped tightly so a stat-stuffer
        // can't pin the hint to 1.0 and a quiet game can't pin it to 0.
        const CAL_BASELINE: f32 = 0.50;
        const PER_MIN_GAIN: f32 = 0.40;
        let tilt = ((per_min - CAL_BASELINE) * PER_MIN_GAIN).clamp(-0.15, 0.20);
        (position_default + tilt).clamp(0.02, 1.0)
    }

    /// Counts the off-balance spell down by one tick. Called once per
    /// simulation tick from `update()`.
    #[inline]
    pub fn tick_off_balance(&mut self) {
        self.off_balance_ticks = self.off_balance_ticks.saturating_sub(1);
        if self.off_balance_ticks == 0 {
            self.off_balance_total = 0;
        }
    }

    /// Can this player start a challenge right now? Only once balance is
    /// restored, whichever state routed them toward Tackling.
    #[inline]
    pub fn can_attempt_tackle(&self) -> bool {
        self.off_balance_ticks == 0
    }

    #[inline]
    pub fn is_off_balance(&self) -> bool {
        self.off_balance_ticks > 0
    }

    /// Start (or extend) an off-balance spell after a tackle attempt.
    /// A longer spell is never shortened by a shorter one.
    #[inline]
    pub fn begin_off_balance(&mut self, ticks: u16) {
        if ticks > self.off_balance_ticks {
            self.off_balance_ticks = ticks;
            self.off_balance_total = ticks;
        }
    }

    /// 1.0 = fully balanced, 0.0 = as unbalanced as a spell gets. Within a
    /// spell it falls to its minimum straight after the lunge, holds there
    /// for the first part, then recovers linearly to 1.0 as balance
    /// returns.
    pub fn balance_factor(&self) -> f32 {
        if self.off_balance_ticks == 0 || self.off_balance_total == 0 {
            return 1.0;
        }
        let remaining = self.off_balance_ticks as f32 / self.off_balance_total as f32;
        // Worst for the first 40% of the spell, then a linear return.
        (1.0 - (remaining / 0.6).min(1.0)).clamp(0.0, 1.0)
    }

    pub fn rebuild_waypoint_cache(&mut self) {
        self.cached_waypoints = self
            .tactical_position
            .tactical_positions
            .iter()
            .filter(|tp| tp.position == self.tactical_position.current_position)
            .flat_map(|tp| &tp.waypoints)
            .map(|(x, y)| Vector3::new(*x, *y, 0.0))
            .collect();
    }

    pub fn update(
        &mut self,
        context: &MatchContext,
        tick_context: &GameTickContext,
        events: &mut EventCollection,
    ) {
        self.tick_off_balance();

        let player_events = PlayerMatchState::process(self, context, tick_context);

        events.add_from_collection(player_events);

        self.update_waypoint_index(tick_context);

        self.check_boundary_collision(context);
        self.move_to();
    }

    /// Movement-only update: apply existing velocity without re-evaluating AI state.
    #[inline]
    pub fn update_movement_only(&mut self, context: &MatchContext) {
        self.in_state_time += 1;
        self.check_boundary_collision(context);
        self.move_to();
    }

    #[inline]
    pub fn check_boundary_collision(&mut self, context: &MatchContext) {
        let field_width = context.field_size.width as f32 + 1.0;
        let field_height = context.field_size.height as f32 + 1.0;

        // Clamp position to field boundaries
        self.position.x = self.position.x.clamp(0.0, field_width);
        self.position.y = self.position.y.clamp(0.0, field_height);

        // Only stop velocity if player is trying to move OUT of bounds
        // Allow velocity that moves them back into the field
        if self.position.x <= 0.0 && self.velocity.x < 0.0 {
            // At left boundary, trying to move further left
            self.velocity.x = 0.0;
        } else if self.position.x >= field_width && self.velocity.x > 0.0 {
            // At right boundary, trying to move further right
            self.velocity.x = 0.0;
        }

        if self.position.y <= 0.0 && self.velocity.y < 0.0 {
            // At bottom boundary, trying to move further down
            self.velocity.y = 0.0;
        } else if self.position.y >= field_height && self.velocity.y > 0.0 {
            // At top boundary, trying to move further up
            self.velocity.y = 0.0;
        }
    }

    pub fn set_default_state(&mut self) {
        self.state = Self::default_state(self.tactical_position.current_position);
        self.rebuild_waypoint_cache();
    }

    fn default_state(position: PlayerPositionType) -> PlayerState {
        match position.position_group() {
            PlayerFieldPositionGroup::Goalkeeper => {
                PlayerState::Goalkeeper(GoalkeeperState::Standing)
            }
            PlayerFieldPositionGroup::Defender => PlayerState::Defender(DefenderState::Standing),
            PlayerFieldPositionGroup::Midfielder => {
                PlayerState::Midfielder(MidfielderState::Standing)
            }
            PlayerFieldPositionGroup::Forward => PlayerState::Forward(ForwardState::Standing),
        }
    }

    pub fn run_for_ball(&mut self) {
        self.state = match self.tactical_position.current_position.position_group() {
            PlayerFieldPositionGroup::Goalkeeper => {
                PlayerState::Goalkeeper(GoalkeeperState::TakeBall)
            }
            PlayerFieldPositionGroup::Defender => PlayerState::Defender(DefenderState::TakeBall),
            PlayerFieldPositionGroup::Midfielder => {
                PlayerState::Midfielder(MidfielderState::TakeBall)
            }
            PlayerFieldPositionGroup::Forward => PlayerState::Forward(ForwardState::TakeBall),
        }
    }

    #[inline]
    pub fn move_to(&mut self) {
        #[cfg(debug_assertions)]
        let old_position = self.position;

        // Apply velocity only if finite. `is_finite` rules out both NaN
        // and ±Infinity — either poisons the position, and a corrupt
        // position is excluded from the viewer recording so the player
        // literally disappears mid-match.
        if self.velocity.x.is_finite() {
            self.position.x += self.velocity.x;
        }

        if self.velocity.y.is_finite() {
            self.position.y += self.velocity.y;
        }

        // Last-resort salvage: if position is already corrupt from an
        // earlier tick (before the velocity guard was in place, or from
        // external code paths), reset to the player's tactical start
        // position. The player briefly teleports rather than vanishing.
        if !self.position.x.is_finite() || !self.position.y.is_finite() {
            self.position = self.start_position;
            self.velocity = Vector3::zeros();
        }

        #[cfg(debug_assertions)]
        {
            // Check for abnormally large position changes
            let position_delta = self.position - old_position;
            let position_change = position_delta.norm();

            const MAX_REASONABLE_POSITION_CHANGE: f32 = 20.0;

            if position_change > MAX_REASONABLE_POSITION_CHANGE {
                debug!(
                    "Player {:?} position jumped abnormally! {} from: ({:.2}, {:.2}) to: ({:.2}, {:.2}), delta: ({:.2}, {:.2}), distance: {:.2}, velocity: ({:.2}, {:.2})",
                    self.state,
                    self.id,
                    old_position.x,
                    old_position.y,
                    self.position.x,
                    self.position.y,
                    position_delta.x,
                    position_delta.y,
                    position_change,
                    self.velocity.x,
                    self.velocity.y
                );
            }
        }
    }

    pub fn heading(&self) -> f32 {
        self.velocity.y.atan2(self.velocity.x)
    }

    pub fn has_ball(&self, ctx: &StateProcessingContext<'_>) -> bool {
        ctx.ball().owner_id() == Some(self.id)
    }

    pub fn update_waypoint_index(&mut self, tick_context: &GameTickContext) {
        if self.cached_waypoints.is_empty() {
            self.rebuild_waypoint_cache();
        }
        self.waypoint_manager.update(
            &tick_context.positions.players.position(self.id),
            &self.cached_waypoints,
        );
    }

    pub fn get_waypoints_as_vectors(&self) -> &[Vector3<f32>] {
        &self.cached_waypoints
    }

    pub fn should_follow_waypoints(&self, ctx: &StateProcessingContext) -> bool {
        // Ball carrier doesn't follow waypoints — they move freely
        if self.has_ball(ctx) {
            return false;
        }

        // Best chaser pursues the ball, not waypoints
        if !ctx.ball().is_owned() && ctx.team().is_best_player_to_chase_ball() {
            return false;
        }

        // If any teammate is too close (< 12u, the natural "shoulder-
        // to-shoulder" bunching distance), follow waypoints back to
        // formation. This is the anti-grouping reinforcement: the
        // moment two of our players are crammed into one yard, one
        // of them peels off to their assigned tactical position. The
        // shorter of them (by id) peels, to avoid both trying to move
        // simultaneously.
        let me_id = self.id;
        let me_pos = self.position;
        let teammate_crowding = ctx.players().teammates().all().any(|t| {
            if t.id == me_id {
                return false;
            }
            let d_sq = (t.position - me_pos).norm_squared();
            if d_sq >= 144.0 {
                return false;
            } // 12² = 144
            // Only one of the pair peels (the lower id). Keeps the
            // behaviour deterministic per-tick and avoids both
            // leaving their post simultaneously.
            t.id > me_id
        });
        if teammate_crowding {
            return true;
        }

        // Everyone else follows waypoints to maintain tactical shape
        // Waypoints represent position-specific movement patterns that keep
        // formation spread and prevent clustering
        true
    }
}

#[derive(Copy, Clone)]
pub struct MatchPlayerLite {
    pub id: u32,
    pub position: Vector3<f32>,
    pub tactical_positions: PlayerPositionType,
}

impl MatchPlayerLite {
    pub fn has_ball(&self, ctx: &StateProcessingContext<'_>) -> bool {
        ctx.ball().owner_id() == Some(self.id)
    }

    pub fn velocity(&self, ctx: &StateProcessingContext<'_>) -> Vector3<f32> {
        ctx.tick_context.positions.players.velocity(self.id)
    }

    pub fn distance(&self, ctx: &StateProcessingContext<'_>) -> f32 {
        ctx.tick_context.grid.get(self.id, ctx.player.id)
    }
}

impl From<&MatchPlayer> for MatchPlayerLite {
    fn from(player: &MatchPlayer) -> Self {
        MatchPlayerLite {
            id: player.id,
            position: player.position,
            tactical_positions: player.tactical_position.current_position,
        }
    }
}

#[cfg(test)]
mod rule2_tests {
    use super::*;
    use crate::r#match::{GamePhase, TeamTacticalState};

    const W: f32 = 840.0;
    const H: f32 = 545.0;

    fn tgt(action: ShapeAction, base: Vector3<f32>, side: PlayerSide, ball_y: f32) -> Vector3<f32> {
        ShapeRule::new(ShapeTrigger::InPossession, action).target(
            base, side, W, H, TeamTacticalState::initial(), ball_y, RuleRefs::default(),
        )
    }

    #[test]
    fn rule2_presets_match_original_actions() {
        let actions = [
            ShapeAction::StepIntoMidfield,
            ShapeAction::TuckInside,
            ShapeAction::HoldWidth,
            ShapeAction::DropDeeper,
            ShapeAction::DropShort,
        ];
        for side in [PlayerSide::Left, PlayerSide::Right] {
            for bx in [60.0, 180.0, 420.0, 700.0] {
                for by in [40.0, 130.0, 272.5, 400.0, 500.0] {
                    let base = Vector3::new(bx, by, 0.0);
                    for a in actions {
                        let preset = ShapeAction::Move(MoveSpec::preset(a).unwrap());
                        let old = tgt(a, base, side, 272.5);
                        let new = tgt(preset, base, side, 272.5);
                        assert!((old - new).magnitude() < 0.01, "{:?} base=({},{}) {:?}: {:?} vs {:?}", a, bx, by, side, old, new);
                    }
                }
            }
        }
    }

    #[test]
    fn rule2_lanes_and_sides() {
        let base = Vector3::new(300.0, 300.0, 0.0);
        // Left-side team: manager's left is the low-Y touchline.
        let left_hs = ShapeAction::Move(MoveSpec { lane: LaneRef::HalfSpace, side: SideRef::Left, depth_permille: 0, anchor: Anchor::Base });
        let right_hs = ShapeAction::Move(MoveSpec { lane: LaneRef::HalfSpace, side: SideRef::Right, depth_permille: 0, anchor: Anchor::Base });
        assert!((tgt(left_hs, base, PlayerSide::Left, 272.5).y - (272.5 - 0.2 * H)).abs() < 0.01);
        assert!((tgt(right_hs, base, PlayerSide::Left, 272.5).y - (272.5 + 0.2 * H)).abs() < 0.01);
        // Right-side team attacks the other way: left flips to high Y.
        assert!((tgt(left_hs, base, PlayerSide::Right, 272.5).y - (272.5 + 0.2 * H)).abs() < 0.01);
        // Flank lane sits on the touchline inset, centre lane on the centre line.
        let flank = ShapeAction::Move(MoveSpec { lane: LaneRef::Flank, side: SideRef::Own, depth_permille: 0, anchor: Anchor::Base });
        assert!((tgt(flank, base, PlayerSide::Left, 272.5).y - (H - 28.0)).abs() < 0.01);
        let centre = ShapeAction::Move(MoveSpec { lane: LaneRef::Centre, side: SideRef::Own, depth_permille: 0, anchor: Anchor::Base });
        assert!((tgt(centre, base, PlayerSide::Left, 272.5).y - 272.5).abs() < 0.01);
        // Ball side follows the ball.
        let bs = ShapeAction::Move(MoveSpec { lane: LaneRef::Flank, side: SideRef::BallSide, depth_permille: 0, anchor: Anchor::Base });
        assert!(tgt(bs, base, PlayerSide::Left, 50.0).y < 100.0);
        assert!(tgt(bs, base, PlayerSide::Left, 500.0).y > 440.0);
        // Depth: positive = toward the opponent goal for each side.
        let fwd = ShapeAction::Move(MoveSpec { lane: LaneRef::Keep, side: SideRef::Own, depth_permille: 100, anchor: Anchor::Base });
        assert!(tgt(fwd, base, PlayerSide::Left, 272.5).x > base.x);
        assert!(tgt(fwd, base, PlayerSide::Right, 272.5).x < base.x);
    }

    #[test]
    fn rule2_conjunctive_triggers() {
        let team = TeamTacticalState::initial();
        let mut rule = ShapeRule::new(ShapeTrigger::InPossession, ShapeAction::HoldWidth);
        rule.and = [Some(ShapeTrigger::BallLeft), None];
        let active = |in_poss: bool, lane: i8| {
            rule.is_active(in_poss, true, lane, 0, 0, 10, team, GamePhase::MidBlock)
        };
        assert!(active(true, -1));
        assert!(!active(true, 0), "ball not on the left");
        assert!(!active(false, -1), "not in possession");
    }

    #[test]
    fn rule2_new_rule_has_no_limit_by_default() {
        let r = ShapeRule::new(ShapeTrigger::InPossession, ShapeAction::HoldWidth);
        assert_eq!(r.max_in_zone, None);
    }

    #[test]
    fn rule2_out_of_possession_needs_a_team_to_own_the_ball() {
        let team = TeamTacticalState::initial();
        let rule = ShapeRule::new(ShapeTrigger::OutOfPossession, ShapeAction::DropDeeper);
        assert!(rule.is_active(false, true, 0, 0, 0, 10, team, GamePhase::MidBlock));
        assert!(!rule.is_active(false, false, 0, 0, 0, 10, team, GamePhase::MidBlock));
    }
}

#[cfg(test)]
mod tests {
    use crate::club::player::builder::PlayerBuilder;
    use crate::r#match::MatchPlayer;
    use crate::shared::fullname::FullName;
    use crate::{
        PersonAttributes, PlayerAttributes, PlayerPosition, PlayerPositionType, PlayerPositions,
        PlayerSkills,
    };
    use chrono::NaiveDate;

    fn build_player(pos: PlayerPositionType) -> MatchPlayer {
        let player = PlayerBuilder::new()
            .id(1)
            .full_name(FullName::new("T".to_string(), "P".to_string()))
            .birth_date(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap())
            .country_id(1)
            .attributes(PersonAttributes::default())
            .skills(PlayerSkills::default())
            .positions(PlayerPositions {
                positions: vec![PlayerPosition {
                    position: pos,
                    level: 18,
                }],
            })
            .player_attributes(PlayerAttributes::default())
            .build()
            .unwrap();
        MatchPlayer::from_player(1, &player, pos, false)
    }

    #[test]
    fn starter_minutes_count_full_90() {
        let p = build_player(PlayerPositionType::DefenderCenter);
        // entry_match_time_ms defaults to 0 (kickoff). At full time
        // the helper returns the elapsed minutes.
        let minutes = p.minutes_played_at(90 * 60_000);
        assert_eq!(minutes, 90);
    }

    #[test]
    fn substituted_in_player_minutes_only_count_post_entry() {
        // Substitute came on at the 70th minute; at full time their
        // minutes-played should be 20, not 90 — the rating helper
        // damps cameo bonuses based on this number, so an 80th-minute
        // sub posting a key pass shouldn't be rated like a starter.
        let mut p = build_player(PlayerPositionType::ForwardCenter);
        p.entry_match_time_ms = 70 * 60_000;
        let minutes = p.minutes_played_at(90 * 60_000);
        assert_eq!(minutes, 20);
    }

    #[test]
    fn match_end_stats_carry_substituted_minutes() {
        // The end-of-match snapshot accepts the minutes count from the
        // caller — the substitution path computes it via
        // `minutes_played_at` and writes the right value into the
        // resulting `PlayerMatchEndStats`.
        let mut p = build_player(PlayerPositionType::MidfielderCenter);
        p.entry_match_time_ms = 60 * 60_000;
        let minutes = p.minutes_played_at(90 * 60_000);
        let stats = p.to_match_end_stats(minutes);
        assert_eq!(stats.minutes_played, 30);
    }

    #[test]
    fn minutes_played_at_caps_at_120_for_extra_time() {
        let p = build_player(PlayerPositionType::DefenderCenter);
        // Pathological — 200 minute match should clamp to 120.
        let minutes = p.minutes_played_at(200 * 60_000);
        assert_eq!(minutes, 120);
    }
}
