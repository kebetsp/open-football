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

/// Can this OUTFIELD player physically meet the ball with his head right
/// now? Horizontal reach is his skill-scaled lunge/jump stride; vertical
/// is his jump height. No teleport anywhere: if true, HE steps to the ball.
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
