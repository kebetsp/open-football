//! Match officiating and rules: the referee and set pieces.
//!
//! A fourth submodule, `management` (professional fouls, time-wasting,
//! stoppage time, home advantage — ~700 lines, sourced realism targets,
//! unit-tested), was deleted 2026-08-28 (engine audit #3): confirmed via
//! grep to have zero real call sites anywhere in the running simulation
//! — only a bare `pub use` re-export and its own unit tests referenced
//! it. It was never wired into the referee/tick-loop dispatch that would
//! have made it do anything. If this functionality (pro fouls, time
//! wasting, home bias) is wanted, it needs to be rebuilt wired-in from
//! the start, not resurrected as dead weight.

pub mod referee;
pub mod set_pieces;
