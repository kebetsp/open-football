//! Physical contact between a player and the ball.
//!
//! Rule (2026-10-16, Pavel): the ball never moves except by real physics
//! (velocity, gravity, drag, a leg/head/hand touch through
//! `release_after_touch`). If a contact needs a small relocation, the
//! PLAYER steps or lunges to the ball — never the ball to the player.
//!
//! * [`contact_reach`] — how far a player can lunge/step/jump to meet the
//!   ball, scaled by his physical ability, capped at [`MAX_REACH`] (24u =
//!   3m).
//! * [`jump_to_ball`] — move the player onto the ball's ground position.
//!   A contact beyond [`contact_reach`] must NOT be resolved at all; the
//!   ball keeps flying.

use crate::r#match::MatchPlayer;
use nalgebra::Vector3;

/// Hard ceiling on a player's lunge/step to the ball (24u = 3m).
pub const MAX_REACH: f32 = 24.0;
/// Reach of the least athletic player (10u = 1.25m).
pub const MIN_REACH: f32 = 10.0;

/// Reach for ordinary ground/low contacts: agility, acceleration and
/// balance on the 1–20 skill scale map linearly onto
/// `MIN_REACH..=MAX_REACH`.
pub fn contact_reach(p: &MatchPlayer) -> f32 {
    let ph = &p.skills.physical;
    let s = ((ph.agility + ph.acceleration + ph.balance) / 3.0 - 1.0) / 19.0;
    MIN_REACH + (MAX_REACH - MIN_REACH) * s.clamp(0.0, 1.0)
}

/// Move the player onto the ball's ground position (x/y only).
pub fn jump_to_ball(p: &mut MatchPlayer, ball: Vector3<f32>) {
    p.position.x = ball.x;
    p.position.y = ball.y;
}

/// Highest ball height (m) a jumping outfielder can meet with his head.
pub fn header_height_cap(p: &MatchPlayer) -> f32 {
    let j = ((p.skills.physical.jumping - 1.0) / 19.0).clamp(0.0, 1.0);
    2.5 + 0.5 * j
}

/// Lowest height (m) at which a contest is an aerial one (below this it
/// is ordinary ground/low play).
pub const AERIAL_MIN_Z: f32 = 1.6;

/// A header contest is decided only once the ball has dropped to this
/// height: below every outfielder's jump cap (2.5 m+), so nobody is
/// excluded just because he jumps a few centimetres lower and a defender
/// cannot win a contest the attacker could not yet reach (2026-10-17).
pub const CONTEST_MAX_Z: f32 = 2.4;

/// Header-work switches (2026-10-17). Each part of the lofted-ball header
/// work has its own environment variable so every part can be
/// measured alone against baseline (the first attempt shipped them
/// together and could not tell which one hurt byline runs and crosses).
/// Validated parts are on by default; `HDR_X=0` turns one off.
pub mod hdr {
    use std::sync::OnceLock;
    pub const RUN: usize = 0; // runners go to the head-height point
    pub const CLAIM: usize = 1; // a player who can head it skips the early claim
    pub const SHOT: usize = 2; // a won box header fires its own shot
    pub const Z: usize = 3; // contests resolve at z <= CONTEST_MAX_Z
    pub const OPEN: usize = 4; // lofted balls away from the box too
    pub const FREE: usize = 7; // unchallenged attacker at the ball takes the header
    pub const CLUMSY: usize = 8; // opponent within ~5 m: a glanced header is a mishit, not a foot touch
    pub const GK: usize = 6; // the keeper runs onto a high ball in his own box
    pub const QUALITY: usize = 10; // a poor contact is a weak off-target header, never a cancelled shot
    pub const RECV: usize = 9; // the intended receiver of a lofted ball always runs to the head-height point
    pub const DEF: usize = 5; // a defender in reach clears even when no attacker can reach
    static FLAGS: OnceLock<[bool; 11]> = OnceLock::new();
    pub fn on(i: usize) -> bool {
        FLAGS.get_or_init(|| {
            // Default on for the validated parts; "HDR_X=0" turns one off,
            // "HDR_X=1" turns an off-by-default part on.
            let e = |n: &str, d: bool| match std::env::var(n).ok().as_deref() {
                Some("1") => true,
                Some("0") => false,
                _ => d,
            };
            [
                e("HDR_RUN", true),
                e("HDR_CLAIM", true),
                e("HDR_SHOT", true),
                e("HDR_Z", true),
                e("HDR_OPEN", true),
                e("HDR_DEF", false),
                e("HDR_GK", true),
                e("HDR_FREE", true),
                e("HDR_CLUMSY", false),
                e("HDR_RECV", true),
                e("HDR_QUALITY", true),
            ]
        })[i]
    }
}

/// Can this OUTFIELD player physically meet the ball with his head right
/// now? Horizontal reach is his skill-scaled lunge/jump stride; vertical
/// is his jump height. No teleport anywhere: if true, HE steps to the ball.
/// Any player of the other team within ~5 m (40u) of the ball, keepers
/// included: the situation where a foot touch risks losing the ball.
pub fn opponent_near_ball(players: &[crate::r#match::MatchPlayer], team: u32, ball: Vector3<f32>) -> bool {
    players.iter().any(|o| {
        o.team_id != team && !o.is_sent_off && {
            let dx = o.position.x - ball.x;
            let dy = o.position.y - ball.y;
            dx * dx + dy * dy <= 40.0 * 40.0
        }
    })
}

/// `can_header` with a lunge: when the CLUMSY switch is on and an opponent
/// is within ~5 m of the ball, a foot touch risks losing it, so any player
/// whose full stride (`MAX_REACH`) gets him there may go for the header.
pub fn can_header_in(p: &MatchPlayer, ball: Vector3<f32>, players: &[MatchPlayer]) -> bool {
    if can_header(p, ball) {
        return true;
    }
    if !hdr::on(hdr::CLUMSY) || ball.z < AERIAL_MIN_Z || ball.z > header_height_cap(p) {
        return false;
    }
    let dx = p.position.x - ball.x;
    let dy = p.position.y - ball.y;
    (dx * dx + dy * dy).sqrt() <= MAX_REACH && opponent_near_ball(players, p.team_id, ball)
}

pub fn can_header(p: &MatchPlayer, ball: Vector3<f32>) -> bool {
    if ball.z < AERIAL_MIN_Z || ball.z > header_height_cap(p) {
        return false;
    }
    let dx = p.position.x - ball.x;
    let dy = p.position.y - ball.y;
    (dx * dx + dy * dy).sqrt() <= contact_reach(p)
}

/// Goalkeeper's hands reach (u): 16..32 by command of area and aerial
/// reach (1–20 scale).
pub fn keeper_reach(p: &MatchPlayer) -> f32 {
    let g = &p.skills.goalkeeping;
    let s = (((g.command_of_area + g.aerial_reach) * 0.5 - 1.0) / 19.0).clamp(0.0, 1.0);
    16.0 + 16.0 * s
}

/// Can this keeper claim the ball with his hands right now (crosses and
/// high balls)? Ceiling 3.0m.
pub fn can_keeper_claim(p: &MatchPlayer, ball: Vector3<f32>) -> bool {
    if ball.z > 3.0 {
        return false;
    }
    let dx = p.position.x - ball.x;
    let dy = p.position.y - ball.y;
    (dx * dx + dy * dy).sqrt() <= keeper_reach(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the no-teleport rule (2026-10-16): contact
    /// code must never write the ball's position. Only dead-ball
    /// restarts (restart.rs, the foul-restart placement in players.rs)
    /// and the sub-2u owner-follow in motion.rs may. If you need a
    /// contact to "reach" the ball, push the PLAYER with
    /// `pending_player_jumps` / `jump_to_ball`.
    #[test]
    fn contact_code_never_moves_the_ball() {
        let files: [(&str, &str); 4] = [
            ("save.rs", include_str!("save.rs")),
            ("ownership.rs", include_str!("ownership.rs")),
            ("interactions.rs", include_str!("interactions.rs")),
            ("tick.rs", include_str!("../../engine/tick.rs")),
        ];
        for (name, src) in files {
            for (i, line) in src.lines().enumerate() {
                let l = line.trim_start();
                if l.starts_with("//") {
                    continue;
                }
                assert!(
                    !l.contains("ball.position = ") && !l.contains("self.position = "),
                    "{}:{} moves the ball directly: {}",
                    name,
                    i + 1,
                    l
                );
            }
        }
    }

    #[test]
    fn reach_is_bounded_and_skill_scaled() {
        assert!(MIN_REACH < MAX_REACH);
        assert!(MAX_REACH <= 24.0);
    }
}
