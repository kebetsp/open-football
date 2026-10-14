//! Feature-gated diagnostic logging for the match engine.
//!
//! Production runs (the web simulator, league batch simulation) don't
//! want per-shot / per-tick logs polluting stdout or burning CPU in
//! the log macros. Dev harnesses (`.dev/match`) DO want them for
//! analysis.
//!
//! All match-diagnostic sites go through `match_log_info!` /
//! `match_log_debug!` below. They expand to a real `log::info!` /
//! `log::debug!` only when the `match-logs` feature is enabled; with
//! the feature off they expand to nothing and compile out entirely.

#[macro_export]
macro_rules! match_log_info {
    ($($arg:tt)*) => {
        #[cfg(feature = "match-logs")]
        { log::info!($($arg)*); }
    };
}

#[macro_export]
macro_rules! match_log_debug {
    ($($arg:tt)*) => {
        #[cfg(feature = "match-logs")]
        { log::debug!($($arg)*); }
    };
}

/// TEMPORARY diagnostic (ball-teleport investigation, 2026-10-14):
/// every site that writes the ball's position leaves a tag; the main
/// loop logs a ball jump together with the tags that fired that tick.
#[cfg(feature = "match-logs")]
pub mod snapdiag {
    use std::sync::Mutex;
    static TAGS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    pub fn tag(s: &'static str) {
        TAGS.lock().unwrap().push(s);
    }
    pub fn take() -> Vec<&'static str> {
        std::mem::take(&mut *TAGS.lock().unwrap())
    }
}

#[macro_export]
macro_rules! snap_tag {
    ($s:expr) => {
        #[cfg(feature = "match-logs")]
        {
            $crate::r#match::logs::snapdiag::tag($s);
        }
    };
}
