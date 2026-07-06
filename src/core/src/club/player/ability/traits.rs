//! Preferred Player Moves (PPMs) — signature behaviours that give players
//! identity in the match engine and scouting reports. FM calls these
//! "Player Traits" or "Preferred Player Moves".
//!
//! Traits modulate decision weights in the match-engine state machines:
//! a player with `TriesThroughBalls` will bias toward risky passes, one
//! with `HugsLine` keeps a wider average x-position, etc.

use crate::club::player::position::{PlayerFieldPositionGroup, PlayerPosition};
use crate::club::player::skills::PlayerSkills;
use crate::utils::FloatUtils;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerTrait {
    // Attacking movement
    CutsInsideFromBothWings,
    HugsLine,
    RunsWithBallOften,
    RunsWithBallRarely,
    GetsIntoOppositionArea,
    ArrivesLateInOppositionArea,
    StaysBack,
    // Passing
    TriesThroughBalls,
    LikesToSwitchPlay,
    LooksForPassRatherThanAttemptShot,
    PlaysShortPasses,
    PlaysLongPasses,
    // Shooting
    ShootsFromDistance,
    PlacesShots,
    PowersShots,
    TriesLobs,
    // Set-piece / specialism
    CurlsBall,
    KnocksBallPast,
    KillerBallOften,
    // Defensive
    DivesIntoTackles,
    StaysOnFeet,
    MarkTightly,
    // Personality on-pitch
    Playmaker,
    Argues,
    WindsUpOpponents,
    // Technical flair
    TriesTricks,
    BackheelsRegularly,
    OneClubPlayer,
}

/// Manager-issued movement directive for one match. Unlike a
/// `PlayerTrait` (a permanent signature behaviour), a behavioural
/// directive is a per-match tactical instruction injected through the
/// match API and checked by the relevant strategy state at its decision
/// point. Typed enum on purpose: a mistyped directive string must fail
/// at the parse boundary (mapped to `None` + a warning), never reach a
/// state machine as a silent no-op. At most one movement directive per
/// player per match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BehavioralDirective {
    /// Stay wide, carry the ball along the touchline to the byline at
    /// the player's own channel, and deliver a cross from there —
    /// instead of the default cut-inside-and-pass-centrally pattern.
    BylineAndCross,
    /// Keep dribbles in the wide channel — never angle in toward goal
    /// centre with the ball. Softer than BylineAndCross: the normal
    /// pass/shot/cross decision tree still runs, only the carry route
    /// is shaped.
    StayWideNoCutInside,
    /// Action-selection bias toward shooting: the willingness roll in
    /// the shared shot-decision helper is scaled up, so the player
    /// pulls the trigger materially more often when a shot is live.
    /// Distinct from a finishing-skill nudge (which only changes how
    /// well the same shots go in).
    ShootOnSight,
    /// Make runs in behind far more readily (skill gate lowered); the
    /// run targets the player's own channel via the channel-aware
    /// logic already in RunningInBehind.
    RunChannelInBehind,
    /// Release the ball immediately on reception — first-touch lay-off
    /// to the best available teammate instead of turning or carrying.
    LayItOffFirstTouch,
    /// Conditional (Level 2): first-touch lay-off ONLY when the
    /// reception is a long ball (the pass travelled from a teammate far
    /// away). Normal short receptions behave exactly as without the
    /// directive — the condition gating is the point.
    LayOffOnLongBall,
    /// Conditional (Level 2): when there is clear space ahead (no
    /// opponent in the forward cone), carry the ball forward instead of
    /// releasing an early pass; under pressure behave as normal.
    CarryWhenSpaceAhead,
}

impl PlayerTrait {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlayerTrait::CutsInsideFromBothWings => "Cuts inside from both wings",
            PlayerTrait::HugsLine => "Hugs line",
            PlayerTrait::RunsWithBallOften => "Runs with ball often",
            PlayerTrait::RunsWithBallRarely => "Runs with ball rarely",
            PlayerTrait::GetsIntoOppositionArea => "Gets into opposition area",
            PlayerTrait::ArrivesLateInOppositionArea => "Arrives late in opposition area",
            PlayerTrait::StaysBack => "Stays back at all times",
            PlayerTrait::TriesThroughBalls => "Tries killer balls often",
            PlayerTrait::LikesToSwitchPlay => "Likes to switch play",
            PlayerTrait::LooksForPassRatherThanAttemptShot => "Looks for pass rather than shot",
            PlayerTrait::PlaysShortPasses => "Plays short passes",
            PlayerTrait::PlaysLongPasses => "Plays long passes",
            PlayerTrait::ShootsFromDistance => "Shoots from distance",
            PlayerTrait::PlacesShots => "Places shots",
            PlayerTrait::PowersShots => "Powers shots",
            PlayerTrait::TriesLobs => "Tries lobs",
            PlayerTrait::CurlsBall => "Curls ball",
            PlayerTrait::KnocksBallPast => "Knocks ball past opponent",
            PlayerTrait::KillerBallOften => "Plays killer balls",
            PlayerTrait::DivesIntoTackles => "Dives into tackles",
            PlayerTrait::StaysOnFeet => "Stays on feet",
            PlayerTrait::MarkTightly => "Marks opponent tightly",
            PlayerTrait::Playmaker => "Dictates tempo",
            PlayerTrait::Argues => "Argues with officials",
            PlayerTrait::WindsUpOpponents => "Winds up opponents",
            PlayerTrait::TriesTricks => "Tries tricks",
            PlayerTrait::BackheelsRegularly => "Tries backheels",
            PlayerTrait::OneClubPlayer => "One club player",
        }
    }

    /// Traits plausibly acquired by the player's position group.
    fn candidates_for(group: PlayerFieldPositionGroup) -> &'static [PlayerTrait] {
        match group {
            PlayerFieldPositionGroup::Goalkeeper => &[PlayerTrait::StaysBack],
            PlayerFieldPositionGroup::Defender => &[
                PlayerTrait::StaysBack,
                PlayerTrait::MarkTightly,
                PlayerTrait::StaysOnFeet,
                PlayerTrait::DivesIntoTackles,
                PlayerTrait::PlaysLongPasses,
                PlayerTrait::LikesToSwitchPlay,
            ],
            PlayerFieldPositionGroup::Midfielder => &[
                PlayerTrait::Playmaker,
                PlayerTrait::TriesThroughBalls,
                PlayerTrait::LikesToSwitchPlay,
                PlayerTrait::PlaysShortPasses,
                PlayerTrait::PlaysLongPasses,
                PlayerTrait::ShootsFromDistance,
                PlayerTrait::RunsWithBallOften,
                PlayerTrait::ArrivesLateInOppositionArea,
                PlayerTrait::CurlsBall,
                PlayerTrait::KillerBallOften,
                PlayerTrait::TriesTricks,
            ],
            PlayerFieldPositionGroup::Forward => &[
                PlayerTrait::CutsInsideFromBothWings,
                PlayerTrait::HugsLine,
                PlayerTrait::RunsWithBallOften,
                PlayerTrait::GetsIntoOppositionArea,
                PlayerTrait::ShootsFromDistance,
                PlayerTrait::PlacesShots,
                PlayerTrait::PowersShots,
                PlayerTrait::TriesLobs,
                PlayerTrait::KnocksBallPast,
                PlayerTrait::TriesTricks,
                PlayerTrait::BackheelsRegularly,
            ],
        }
    }
}

/// Roll traits for a new player based on their skills & position.
/// Better players get more traits and skill-biased selections.
pub fn generate_player_traits(
    skills: &PlayerSkills,
    positions: &[PlayerPosition],
    current_ability: u8,
) -> Vec<PlayerTrait> {
    // Trait count scales with ability: avg 0.4 traits at CA 40, ~2 at CA 150, up to 4 at CA 190+.
    let trait_count = if current_ability < 50 {
        if FloatUtils::random(0.0, 1.0) < 0.3 {
            1
        } else {
            0
        }
    } else if current_ability < 90 {
        1
    } else if current_ability < 140 {
        if FloatUtils::random(0.0, 1.0) < 0.4 {
            2
        } else {
            1
        }
    } else if current_ability < 170 {
        2
    } else if current_ability < 190 {
        3
    } else {
        4
    };

    if trait_count == 0 {
        return Vec::new();
    }

    let main_group = positions
        .first()
        .map(|p| p.position.position_group())
        .unwrap_or(PlayerFieldPositionGroup::Midfielder);

    let pool = PlayerTrait::candidates_for(main_group);
    if pool.is_empty() {
        return Vec::new();
    }

    let mut picked: Vec<PlayerTrait> = Vec::new();
    let mut attempts = 0;
    while picked.len() < trait_count && attempts < trait_count * 6 {
        attempts += 1;
        let idx = (FloatUtils::random(0.0, pool.len() as f32) as usize).min(pool.len() - 1);
        let candidate = pool[idx];

        if picked.contains(&candidate) {
            continue;
        }

        // Skill-gated filter: don't hand out "Shoots from distance" to a
        // midfielder with 5 Long Shots, or "Tries through balls" to a
        // 6 Passing CB.
        if !skill_supports_trait(&candidate, skills) {
            continue;
        }

        picked.push(candidate);
    }

    picked
}

fn skill_supports_trait(tr: &PlayerTrait, skills: &PlayerSkills) -> bool {
    // Thresholds (which skills, what minimum) live on the trait registry —
    // see `registry::TRAIT_REGISTRY`. Adding or rebalancing a trait now
    // means editing one row there instead of two locations.
    tr.skills_support(skills)
}
