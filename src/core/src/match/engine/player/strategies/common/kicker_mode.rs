//! Kicker-mode diagnostic baseline (2026-09-12, Pavel's request).
//!
//! Everything this project has layered on top of the base state
//! machine over many sessions — positional-attack patterns, off-ball
//! mobility/rotation/decoy scoring, width-holding, wide-run tracking,
//! shot-reaction overrides, the carry-vs-pass-vs-shoot value comparison
//! — produced very little visible improvement, tried one at a time
//! against an already-complex baseline where several mechanisms compete
//! for control of the same decision. Pavel's own framing: "we need to
//! disable all the advanced features of how players move and challenge
//! the ball... something that looks similar to a kicker [tactics board]
//! - players move in their lines passing horizontally until they find a
//! gap to make a pass to the next line... there should be a distance
//! between forwards and defenders (delay principle)".
//!
//! Opt-in via the `KICKER_MODE` env var (any value), mirroring this
//! codebase's own established `OF_SCORE_BLIND`/`MOBILITY_ABLATE`
//! diagnostic-toggle idiom (`flow/context.rs`, `common/spacing.rs`).
//! Built as a toggle rather than a literal deletion specifically so
//! individual "advanced" mechanisms can be re-enabled one at a time
//! later, per Pavel's own explicit plan, without reconstructing them
//! from git history each time.
//!
//! What stays ON regardless (deliberately NOT gated, per Pavel's own
//! "things we cannot completely disable" list): loose-ball recovery
//! (`TakeBall`, `processor.rs`'s `PlayerFieldPositionGroup::process`),
//! every restart mechanic (corner/throw-in/free-kick/goal-kick/penalty
//! staging and their forced-release overrides), offside enforcement,
//! and goalkeeper shot-stopping/basic distribution. Shooting itself is
//! also left untouched — see `kicker_mode_pass_override`'s own doc
//! comment (processor.rs) for why that's a deliberate, narrow scope
//! choice, not an oversight.
//!
//! The two consuming halves:
//!   - `kicker_line_hold_velocity` (processor.rs) — off-ball
//!     positioning: hold one of three depth lines with an enforced
//!     separation from the opposing threat.
//!   - `kicker_mode_pass_override` (processor.rs) +
//!     `PassEvaluator::kicker_find_pass_option` (evaluator.rs) — on-ball
//!     decision: same-line first, then one line forward through an open
//!     lane, then recycle backward.

use crate::PlayerFieldPositionGroup;
use std::sync::OnceLock;

pub fn is_active() -> bool {
    static ACTIVE: OnceLock<bool> = OnceLock::new();
    *ACTIVE.get_or_init(|| std::env::var("KICKER_MODE").is_ok())
}

/// Ordinal position of a role's "line" in the buildup — back (0),
/// midfield (1), front (2). Goalkeeper has no line of its own; every
/// caller excludes goalkeepers before reaching here, so the returned
/// value for that variant is a safe, never-actually-used default.
pub fn line_index(group: PlayerFieldPositionGroup) -> i32 {
    match group {
        PlayerFieldPositionGroup::Defender => 0,
        PlayerFieldPositionGroup::Midfielder => 1,
        PlayerFieldPositionGroup::Forward => 2,
        PlayerFieldPositionGroup::Goalkeeper => 1,
    }
}
