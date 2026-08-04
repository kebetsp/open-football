use crate::PlayerFieldPositionGroup;
use crate::club::player::skills::GoalkeeperSpeedContext;
use crate::r#match::defenders::states::DefenderState;
use crate::r#match::events::EventCollection;
use crate::r#match::forwarders::states::ForwardState;
use crate::r#match::goalkeepers::states::state::GoalkeeperState;
use crate::r#match::midfielders::states::MidfielderState;
use crate::r#match::player::strategies::players::ops::skill_composites as sc;
use crate::r#match::{GameTickContext, MatchContext, MatchPlayer};

use nalgebra::Vector3;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlayerState {
    Injured,
    Goalkeeper(GoalkeeperState),
    Defender(DefenderState),
    Midfielder(MidfielderState),
    Forward(ForwardState),
}

impl PlayerState {
    /// Cheap integer ID for fast dedup — avoids `to_string()` allocation.
    /// Each (outer variant, inner variant) pair maps to a unique u16.
    #[inline]
    pub fn compact_id(&self) -> u16 {
        match self {
            PlayerState::Injured => 0,
            PlayerState::Goalkeeper(s) => 100 + (*s as u16),
            PlayerState::Defender(s) => 200 + (*s as u16),
            PlayerState::Midfielder(s) => 300 + (*s as u16),
            PlayerState::Forward(s) => 400 + (*s as u16),
        }
    }
}

impl Display for PlayerState {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            PlayerState::Injured => write!(f, "Injured"),
            PlayerState::Goalkeeper(state) => write!(f, "Goalkeeper: {}", state),
            PlayerState::Defender(state) => write!(f, "Defender: {}", state),
            PlayerState::Midfielder(state) => write!(f, "Midfielder: {}", state),
            PlayerState::Forward(state) => write!(f, "Forward: {}", state),
        }
    }
}

// Root-cause-agnostic stuck-player safety net (2026-08-04). Two
// independent state-machine gaps (DefenderGuardingState had no
// has_ball check; MidfielderTacklingState had no "carrier not yet in
// range" fallback) both produced the exact same observable symptom: a
// player's finalized velocity converging to exactly zero and staying
// there indefinitely because nothing ever re-evaluates a new target.
// Rather than trust that those were the only two gaps of this shape,
// this watches the ONE thing every such bug has in common — the
// player simply never moves — regardless of which state or mechanism
// caused it. It does NOT replace fixing a root cause when one is
// found (see the guarding/tackling fixes committed alongside this);
// it's a backstop for whichever gap hasn't been found yet. Every
// trigger is logged so a firing safety net reads as "go investigate
// this," never as a silent, unnoticed fix.
//
// Thresholds are tiered by how plausible genuine, correct stillness
// is for the situation — a player holding the ball motionless is
// essentially never legitimate in real football; a player mid-tackle-
// attempt is meant to resolve within a tick or two; everything else
// (Guarding, Marking, Covering, HoldingLine, Standing, Returning,
// Walking, Pressing...) is a positional-holding state that can
// legitimately sit near-motionless for many seconds — measured
// directly (2026-08-04): an earlier design that also treated "near an
// opponent" as suspicious produced ~4 false triggers/match, almost all
// Guarding/Marking/Returning/Walking correctly shadowing a similarly-
// stationary opponent. Proximity doesn't predict stuckness; the state
// itself does.
const STUCK_EPSILON_SQ: f32 = 0.0004; // 0.02 u/tick — genuinely stopped, not just slow
const STUCK_THRESHOLD_HAS_BALL: u16 = 400; // 4s engine-time
const STUCK_THRESHOLD_TACKLING: u16 = 200; // 2s engine-time
const STUCK_THRESHOLD_FAR: u16 = 2500; // 25s engine-time
// Goalkeepers legitimately hold a static position for long real stretches
// whenever play is at the other end — measured directly (2026-08-04): a
// 40-match sanity batch with STUCK_THRESHOLD_FAR alone produced 22 false
// triggers, 100% of them goalkeepers with no opponent within radius at
// all (nearest_opp == f32::MAX). Outfield players never triggered the
// far tier in that same batch. GK gets a much longer grace period on
// this specific tier; the has_ball tier (a GK holding the ball forever
// IS still a real bug shape) stays untouched.
const STUCK_THRESHOLD_FAR_GK: u16 = 9000; // 90s engine-time
const STUCK_NEAR_PLAY_RADIUS: f32 = 60.0; // matches DUEL_DEBUG_RADIUS

pub struct PlayerMatchState;

impl PlayerMatchState {
    pub fn process(
        player: &mut MatchPlayer,
        context: &MatchContext,
        tick_context: &GameTickContext,
    ) -> EventCollection {
        // Decay memory every 100 ticks
        let current_tick = context.current_tick();
        if current_tick > 0 && current_tick % 100 == 0 {
            player.memory.decay(current_tick);
        }

        let player_position_group = player.tactical_position.current_position.position_group();

        let state_change_result =
            player_position_group.process(player.in_state_time, player, context, tick_context);

        if state_change_result.start_tackle_cooldown {
            player.start_tackle_cooldown();
        }

        // Stash the shot reason on the player. The Shooting state will
        // consume and clear this when it composes the Shoot event.
        if let Some(reason) = state_change_result.shot_reason {
            player.pending_shot_reason = Some(reason);
        }

        if let Some(state) = state_change_result.state {
            Self::change_state(player, state);
        } else {
            player.in_state_time += 1;
        }

        if let Some(velocity) = state_change_result.velocity {
            let mut max_speed = if player_position_group == PlayerFieldPositionGroup::Goalkeeper {
                let speed_context = match player.state {
                    PlayerState::Goalkeeper(GoalkeeperState::Diving)
                    | PlayerState::Goalkeeper(GoalkeeperState::PreparingForSave)
                    | PlayerState::Goalkeeper(GoalkeeperState::Jumping) => {
                        GoalkeeperSpeedContext::Explosive
                    }
                    PlayerState::Goalkeeper(GoalkeeperState::Catching)
                    | PlayerState::Goalkeeper(GoalkeeperState::ComingOut) => {
                        GoalkeeperSpeedContext::Active
                    }
                    PlayerState::Goalkeeper(GoalkeeperState::Standing)
                    | PlayerState::Goalkeeper(GoalkeeperState::ReturningToGoal) => {
                        GoalkeeperSpeedContext::Positioning
                    }
                    _ => GoalkeeperSpeedContext::Casual,
                };
                player
                    .skills
                    .goalkeeper_max_speed(player.player_attributes.condition, speed_context)
            } else {
                player
                    .skills
                    .max_speed_with_condition(player.player_attributes.condition)
            };

            // Ball-carrier speed multiplier. Real football: carrying
            // the ball costs ~15-25% of top sprint for an average
            // player — they keep the ball in stride, look up, protect
            // it. Elite carriers (Mbappé/Messi) lose almost nothing.
            //
            // Routes through `movement_speed_with_ball` so dribbling +
            // technique + pace + acceleration + agility + balance all
            // contribute, and so fatigue/late-game effects propagate
            // through `effective_skill`. Mapping per spec:
            //
            //   carry_mult = 0.78 + composite * 0.42
            //
            // Composite floor 0.05 → 0.80 (worst carrier under fatigue);
            // composite 1.00 → 1.20 (elite carrier — no realistic
            // penalty). Capped to existing `[0.75, 1.00]` band so the
            // model stays a CARRY COST: an elite carrier matches their
            // off-ball speed but doesn't go faster than it.
            if tick_context.ball.current_owner == Some(player.id)
                && player_position_group != PlayerFieldPositionGroup::Goalkeeper
            {
                let minute = sc::minute_from_ms(context.total_match_time);
                let composite = sc::movement_speed_with_ball(player, minute);
                let raw = 0.78 + composite * 0.42;
                max_speed *= raw.clamp(0.75, 1.00);
            }

            // NaN/Inf guard: state velocity functions compose many
            // divisions and normalizations, and any zero-magnitude vector
            // put through `.normalize()` anywhere upstream produces a
            // NaN that propagates into player.velocity → player.position
            // → the recording → the viewer renders nothing. Catch it
            // here at the single integration point so no state has to
            // remember to self-sanitize. Non-finite → zero this tick.
            let finite = velocity.x.is_finite() && velocity.y.is_finite() && velocity.z.is_finite();
            let velocity = if finite { velocity } else { Vector3::zeros() };

            let velocity_sq = velocity.norm_squared();
            let max_speed_sq = max_speed * max_speed;

            if velocity_sq > max_speed_sq && velocity_sq > 0.0 {
                let velocity_magnitude = velocity_sq.sqrt();
                player.velocity = velocity * (max_speed / velocity_magnitude);
            } else {
                player.velocity = velocity;
            }

            // Root-cause-agnostic stuck detector — see the module-level
            // doc comment. Judges the FINAL, already-clamped velocity
            // (the actual thing that will or won't move the player this
            // tick), so it's blind to which state or mechanism produced
            // it.
            if player.velocity.norm_squared() > STUCK_EPSILON_SQ {
                player.stuck_ticks = 0;
            } else {
                player.stuck_ticks = player.stuck_ticks.saturating_add(1);

                let has_ball = tick_context.ball.current_owner == Some(player.id);
                // Diagnostic only now (see below) — proximity to an
                // opponent turned out NOT to predict genuine stuckness:
                // a 40-match sanity batch with a distance-based "near
                // play" tier produced ~4 false triggers/match, almost
                // entirely Guarding/Marking/Returning/Walking — states
                // that correctly hold near-zero velocity for many
                // seconds while shadowing a similarly-stationary
                // opponent. The real signal is whether the STATE ITSELF
                // is supposed to resolve quickly, not how close anyone
                // is standing.
                let nearest_opp_dist = tick_context
                    .grid
                    .opponents(player.id, STUCK_NEAR_PLAY_RADIUS)
                    .map(|(_, d)| d)
                    .fold(f32::MAX, f32::min);

                // Tackling is the one state in this engine whose entire
                // premise is "resolve within a tick or two" (attempt,
                // succeed/fail/foul, or bounce out via a distance/
                // cooldown gate) — a sustained, motionless presence here
                // is anomalous almost immediately, regardless of role.
                // This is exactly bug #2's shape (MidfielderTacklingState
                // missing its too-far fallback).
                let is_tackling = matches!(
                    player.state,
                    PlayerState::Defender(DefenderState::Tackling)
                        | PlayerState::Midfielder(MidfielderState::Tackling)
                        | PlayerState::Forward(ForwardState::Tackling)
                        | PlayerState::Goalkeeper(GoalkeeperState::Tackling)
                );

                let threshold = if has_ball {
                    STUCK_THRESHOLD_HAS_BALL
                } else if is_tackling {
                    STUCK_THRESHOLD_TACKLING
                } else if player_position_group == PlayerFieldPositionGroup::Goalkeeper {
                    STUCK_THRESHOLD_FAR_GK
                } else {
                    STUCK_THRESHOLD_FAR
                };

                if player.stuck_ticks >= threshold {
                    log::warn!(
                        "STUCK_RECOVERY player={} team={} prev_state={} has_ball={} nearest_opp={:.1} stuck_ticks={}",
                        player.id,
                        player.team_id,
                        player.state,
                        has_ball,
                        nearest_opp_dist,
                        player.stuck_ticks
                    );
                    let recovery = Self::recovery_state(player_position_group, has_ball);
                    Self::change_state(player, recovery);
                    player.stuck_ticks = 0;
                }
            }
        }

        state_change_result.events
    }

    fn change_state(player: &mut MatchPlayer, state: PlayerState) {
        player.in_state_time = 0;
        player.state = state;
    }

    /// Safe, neutral re-entry state per role — lets the normal role-block
    /// decision logic reassign fresh duties next tick instead of leaving
    /// a broken state in place. Mirrors the existing idiom already used
    /// deliberately elsewhere in this codebase (e.g.
    /// `DefenderGuardingState`'s own crisis override: "Drop to Standing
    /// so the role block assigns fresh duties against the active
    /// threat").
    fn recovery_state(group: PlayerFieldPositionGroup, has_ball: bool) -> PlayerState {
        match group {
            PlayerFieldPositionGroup::Goalkeeper => PlayerState::Goalkeeper(GoalkeeperState::Standing),
            PlayerFieldPositionGroup::Defender => {
                if has_ball {
                    PlayerState::Defender(DefenderState::Running)
                } else {
                    PlayerState::Defender(DefenderState::Standing)
                }
            }
            PlayerFieldPositionGroup::Midfielder => {
                if has_ball {
                    PlayerState::Midfielder(MidfielderState::Running)
                } else {
                    PlayerState::Midfielder(MidfielderState::Standing)
                }
            }
            PlayerFieldPositionGroup::Forward => {
                if has_ball {
                    PlayerState::Forward(ForwardState::Running)
                } else {
                    PlayerState::Forward(ForwardState::Standing)
                }
            }
        }
    }
}
