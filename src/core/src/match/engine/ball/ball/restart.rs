//! Restart resolution: throw-ins on the touchlines and the lifetime
//! management of the in-flight offside snapshot. Endline restarts
//! (corner / goal kick) live in `goal.rs` because they share machinery
//! with the actual goal check.
//!
//! Real football rule: the ball is restarted by the team that did NOT
//! last touch the ball. If we have no record of a touch (kickoff /
//! genuine bug) we fall back to the team currently holding ownership;
//! if even that is missing, the side is decided by an unbiased coin
//! flip rather than left unresolved (see `check_throw_in`'s own comment
//! for why). This module never relies on `check_boundary_collision` as
//! a fallback — that function was removed entirely (engine audit
//! Module 1/2 rewrite, 2026-08-28) because it raced this exact logic;
//! every boundary crossing this module handles is now resolved
//! exhaustively, in one place, on the tick it happens.

use super::Ball;
use crate::PlayerFieldPositionGroup;
use crate::r#match::PassOriginRestart;
use crate::r#match::ball::events::BallEvent;
use crate::r#match::events::EventCollection;
use crate::r#match::{MatchContext, MatchPlayer, PlayerSide};
use nalgebra::Vector3;

impl Ball {
    /// Touchline check: if the ball crossed y<=0 or y>=field_height, set
    /// up a throw-in for the team that did NOT last touch it. Routes
    /// through `pending_set_piece_teleport` like corners / goal kicks so
    /// the engine can place the thrower onto the ball after the tick.
    pub(super) fn check_throw_in(
        &mut self,
        context: &mut MatchContext,
        players: &[MatchPlayer],
        events: &mut EventCollection,
    ) {
        // Already resolved this tick (goal scored, etc.).
        if self.goal_scored {
            return;
        }
        let field_height = context.field_size.height as f32;
        let crossed_top = self.position.y <= 0.0;
        let crossed_bottom = self.position.y >= field_height;
        if !crossed_top && !crossed_bottom {
            return;
        }
        crate::match_log_info!(
            "THROWINDIAG CROSSED y={:.1} last_touch={:?} prev_owner={:?} cur_owner={:?}",
            self.position.y,
            self.last_touch_player_id,
            self.previous_owner,
            self.current_owner
        );

        // Last toucher's side decides which team gets the throw-in.
        let last_toucher_side = self
            .last_touch_player_id
            .or(self.previous_owner)
            .or(self.current_owner)
            .and_then(|pid| players.iter().find(|p| p.id == pid))
            .and_then(|p| p.side);

        // realism-bug (2026-08-28, engine audit #2): the old `None => return`
        // here fell through to `check_boundary_collision`'s "safety net,"
        // which silently resets the ball's position and zeroes its velocity
        // with NO restart awarded — a genuine touchline crossing simply
        // vanishes, and play continues live where the Laws (15) require a
        // stoppage regardless of whose touch it was. `last_toucher_side`
        // can only legitimately be `None` when NONE of last_touch/prev/
        // current owner is set — every real player on the pitch gets
        // `side = Some(_)` unconditionally at setup and it's never cleared
        // to `None` afterward (swap_squads only flips an existing value),
        // so this path is reachable only in the brief post-kickoff/post-
        // goal/half-time window before anyone has touched the reset ball.
        // Rather than trust that this is provably unreachable, fail SAFE:
        // still award a real throw-in, side decided by an unbiased coin
        // flip (we have no signal on whose touch it "really" was, so
        // picking deterministically toward either team would be a worse,
        // silently-biased alternative to acknowledging the genuine
        // uncertainty).
        let throwing_side = match last_toucher_side {
            Some(PlayerSide::Left) => PlayerSide::Right,
            Some(PlayerSide::Right) => PlayerSide::Left,
            None => {
                let side = if context.rng.bernoulli(0.5) {
                    PlayerSide::Left
                } else {
                    PlayerSide::Right
                };
                crate::match_log_info!("THROWINDIAG BAIL_NO_SIDE_RECOVERED side={:?}", side);
                side
            }
        };

        // Inset the throw-in slightly inside the touchline so subsequent
        // physics doesn't immediately re-trigger a boundary cross.
        let field_width = context.field_size.width as f32;
        let throw_x = self.position.x.clamp(2.0, field_width - 2.0);
        let throw_y = if crossed_top { 2.0 } else { field_height - 2.0 };
        let throw_pos = Vector3::new(throw_x, throw_y, 0.0);

        let thrower = pick_thrower(players, throwing_side, throw_pos);
        let thrower_id = match thrower {
            Some(id) => id,
            None => return,
        };

        self.position = throw_pos;
        self.velocity = Vector3::zeros();

        self.previous_owner = self.current_owner;
        self.current_owner = Some(thrower_id);
        self.ownership_duration = 0;
        self.claim_cooldown = 45;
        self.flags.in_flight_state = 20;
        self.pass_target_player_id = None;
        self.recent_passers.clear();
        self.cached_shot_target = None;
        self.offside_snapshot = None;
        self.pass_origin_restart = PassOriginRestart::ThrowIn;
        self.restart_taker_lock = None; // new restart voids the old same-touch chain (§9.4.1)
        self.restart_pending_taker = Some(thrower_id);

        let team_id = players
            .iter()
            .find(|p| p.id == thrower_id)
            .map(|p| p.team_id)
            .unwrap_or(0);
        self.record_touch(thrower_id, team_id, context.current_tick(), true);

        // realism-bug (2026-07-18): previously the throw-in dead-ball
        // window never set `dead_ball_retreat_active`, so the tick loop
        // hit the total-freeze branch — no player AI, no movement, not
        // even the thrower's teammates/opponents reacting. Every other
        // restart (foul/corner/goal-kick) got a real retreat in §13.4;
        // throw-ins were the one restart type left out. Fix: the same
        // `pending_restart_teleports` -> `restart_retreat_target` ->
        // `dead_ball_retreat_active` machinery, populated by a throw-in-
        // specific (local, not full-formation) shape.
        let shape = crate::r#match::engine::set_pieces::throw_in_shape_targets(
            players,
            throwing_side,
            thrower_id,
            throw_pos,
            field_width,
            field_height,
        );
        crate::match_log_info!(
            "THROWIN_SHAPE thrower={} side={:?} throw_x={:.1} throw_y={:.1} shaped={}",
            thrower_id,
            throwing_side,
            throw_pos.x,
            throw_pos.y,
            shape.len()
        );
        for (player_id, pos) in &shape {
            if let Some(p) = players.iter().find(|p| p.id == *player_id) {
                crate::match_log_info!(
                    "THROWIN_TARGET player={} from=({:.1},{:.1}) to=({:.1},{:.1}) dist_from_throw={:.1}",
                    player_id,
                    p.position.x,
                    p.position.y,
                    pos.x,
                    pos.y,
                    (pos - throw_pos).magnitude()
                );
            }
        }
        // realism-bug (2026-07-28): `shape`'s first entry, when present, is
        // always the close-support attacker (`throw_in_shape_targets`
        // pushes close-support before deep-support/presser/marker) — the
        // specifically prepared, marker-evasion-adjusted option. Recorded
        // so the pass evaluator's selection multiplier can actually prefer
        // the receiver the shape mechanism just spent the dead-ball window
        // positioning, instead of a fully generic search that ignores it.
        self.throw_in_preferred_receiver = shape.first().map(|(id, _)| *id);
        self.pending_restart_teleports.extend(shape);

        context.dead_ball_min_ms = context.total_match_time + 300;
        context.dead_ball_until_ms = context.total_match_time + context.rng.range_u64(1, 3) * 1000;
        context.dead_ball_retreat_active = true;
        self.pending_set_piece_teleport = Some((thrower_id, throw_pos));
        events.add_ball_event(BallEvent::Claimed(thrower_id));
    }

    /// Drop the offside snapshot once its lifetime expires. Real-world
    /// passes that don't reach a receiver should end the offside
    /// pretence — anything older than ~220 ticks is stale.
    pub(super) fn expire_offside_snapshot(&mut self, context: &MatchContext) {
        const OFFSIDE_LIFETIME_TICKS: u64 = 220;
        if let Some(snap) = self.offside_snapshot {
            let now = context.current_tick();
            if now.saturating_sub(snap.set_tick) > OFFSIDE_LIFETIME_TICKS {
                self.offside_snapshot = None;
                if self.pass_origin_restart != PassOriginRestart::OpenPlay {
                    self.pass_origin_restart = PassOriginRestart::OpenPlay;
                }
            }
        }
    }
}

/// Score-based thrower selection.
///   0.35 distance (closer to throw point is better)
///   0.25 position fit (fullback / wing-back / wide mid preferred)
///   0.20 long_throws scaled
///   0.10 decisions scaled
///   0.05 technique scaled
///   0.05 strength scaled
fn pick_thrower(
    players: &[MatchPlayer],
    throwing_side: PlayerSide,
    throw_pos: Vector3<f32>,
) -> Option<u32> {
    // realism-bug (2026-08-28, engine audit Module 2): genuinely
    // exhaustive per Pavel's sign-off — a throw-in must always find a
    // taker if ANY eligible (not sent off) player of the throwing side
    // remains on the pitch. `allow_goalkeeper` runs the same scoring
    // twice only in the vanishingly rare case where every outfielder is
    // gone (e.g. mass red cards) — real football has no rule against a
    // goalkeeper taking a throw-in, it's just never the normal choice.
    pick_thrower_scored(players, throwing_side, throw_pos, false)
        .or_else(|| pick_thrower_scored(players, throwing_side, throw_pos, true))
}

fn pick_thrower_scored(
    players: &[MatchPlayer],
    throwing_side: PlayerSide,
    throw_pos: Vector3<f32>,
    allow_goalkeeper: bool,
) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    // Furthest-search radius — a player 200u from the touchline isn't
    // realistically the thrower. We still score them; this just bounds
    // the normalisation.
    const MAX_REASONABLE_DISTANCE: f32 = 200.0;

    for p in players {
        if p.side != Some(throwing_side) {
            continue;
        }
        if p.is_sent_off {
            continue;
        }
        if !allow_goalkeeper
            && p.tactical_position.current_position.position_group()
                == PlayerFieldPositionGroup::Goalkeeper
        {
            continue;
        }
        let dx = p.position.x - throw_pos.x;
        let dy = p.position.y - throw_pos.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let dist_score = (1.0 - (dist / MAX_REASONABLE_DISTANCE).min(1.0)).max(0.0);

        // Position fit: defenders + midfielders get a small bump for
        // throw-ins (fullbacks / wide mids most often take throws). The
        // exact PlayerPositionType subdivision (LB/RB/LM/RM) lives one
        // crate boundary away, so we lean on the position-group buckets
        // that are visible here.
        let position_fit = match p.tactical_position.current_position.position_group() {
            PlayerFieldPositionGroup::Defender => 1.0,
            PlayerFieldPositionGroup::Midfielder => 0.8,
            PlayerFieldPositionGroup::Forward => 0.5,
            PlayerFieldPositionGroup::Goalkeeper => 0.0,
        };

        let long_throws = (p.skills.technical.long_throws / 20.0).clamp(0.0, 1.0);
        let decisions = (p.skills.mental.decisions / 20.0).clamp(0.0, 1.0);
        let technique = (p.skills.technical.technique / 20.0).clamp(0.0, 1.0);
        let strength = (p.skills.physical.strength / 20.0).clamp(0.0, 1.0);

        let score = dist_score * 0.35
            + position_fit * 0.25
            + long_throws * 0.20
            + decisions * 0.10
            + technique * 0.05
            + strength * 0.05;

        let candidate = (p.id, score);
        match best {
            None => best = Some(candidate),
            Some((_, best_score)) if score > best_score => best = Some(candidate),
            _ => {}
        }
    }
    best.map(|(id, _)| id)
}

/// Throw-in delivery range, in field units. Used by tactical states that
/// pick a target from a throw-in.
///
/// 35u baseline + up to 35u extra at long_throws=20. Minimum useful
/// range 12u — anything shorter is a recycle pass.
#[allow(dead_code)]
pub fn throw_in_range(long_throws_skill: f32) -> (f32, f32) {
    let scaled = (long_throws_skill / 20.0).clamp(0.0, 1.0);
    let max_range = 35.0 + scaled * 35.0;
    (12.0, max_range)
}

/// Whether the player can deliver a "long throw into the box" — only
/// allowed in the attacking third with a strong long_throws skill.
#[allow(dead_code)]
pub fn can_long_throw_into_box(long_throws_skill: f32, in_attacking_third: bool) -> bool {
    in_attacking_third && long_throws_skill >= 14.0
}
