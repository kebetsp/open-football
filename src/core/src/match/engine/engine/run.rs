use super::*;
use crate::r#match::engine::context::MatchEngineConfig;
use nalgebra::Vector3;
use crate::r#match::engine::rating::{RatingExpectationContext, TeamRatingSummary};
use crate::r#match::player::events::FoulResolver;
use crate::r#match::player::events::players::PlayerEventDispatcher;

impl<const W: usize, const H: usize> FootballEngine<W, H> {
    pub fn new() -> Self {
        FootballEngine {}
    }

    #[allow(unreachable_code)]
    pub fn play(
        left_squad: MatchSquad,
        right_squad: MatchSquad,
        match_recordings: bool,
        is_friendly: bool,
        is_knockout: bool,
    ) -> MatchResultRaw {
        let mut config = MatchEngineConfig::default();
        config.match_recordings = match_recordings;
        config.is_friendly = is_friendly;
        config.is_knockout = is_knockout;
        Self::play_with_config(left_squad, right_squad, config)
    }

    /// Seeded entry point. Compatibility wrapper around
    /// `play_with_config`. `seed = Some(_)` pins the engine's owned
    /// RNG (substitution timing, penalty shootout, foul card rolls,
    /// corner aerial contest, every converted player decision).
    /// `None` falls back to OS entropy, matching legacy behaviour.
    #[allow(unreachable_code)]
    pub fn play_seeded(
        left_squad: MatchSquad,
        right_squad: MatchSquad,
        match_recordings: bool,
        is_friendly: bool,
        is_knockout: bool,
        seed: Option<u64>,
    ) -> MatchResultRaw {
        let mut config = MatchEngineConfig::default();
        config.seed = seed;
        config.match_recordings = match_recordings;
        config.is_friendly = is_friendly;
        config.is_knockout = is_knockout;
        Self::play_with_config(left_squad, right_squad, config)
    }

    /// Full-config entry point. Lets the caller inject seed, fixture
    /// date, environment (weather/pitch/crowd/importance/derby),
    /// referee profile, friendly/knockout flags, and the
    /// match_recordings switch in one place — instead of patching the
    /// context after construction. Required by the calibration harness
    /// to run a real rainy match or a strict-referee fixture, and by
    /// any replay test that needs exact-seed control over today's
    /// date.
    #[allow(unreachable_code)]
    pub fn play_with_config(
        left_squad: MatchSquad,
        right_squad: MatchSquad,
        config: MatchEngineConfig,
    ) -> MatchResultRaw {
        // Profiling shortcut — see the `match-stub` feature in
        // `core/Cargo.toml`. Skips the simulation entirely and returns
        // a 0-0 result with just enough metadata (team IDs, player
        // IDs) for the surrounding pipeline to run.
        #[cfg(feature = "match-stub")]
        {
            let _ = &config;
            return Self::play_stub(left_squad, right_squad);
        }

        let perf = PerfCounters::instance();
        let match_start = Instant::now();
        let score = Score::new(left_squad.team_id, right_squad.team_id);

        // Snapshot starting tactics by team-id BEFORE the squads move
        // into `MatchField::new`. The first half always has the home
        // team on the left side, so left == home / right == away here.
        let starting_home_tactic = Some(left_squad.tactics.tactic_type);
        let starting_away_tactic = Some(right_squad.tactics.tactic_type);

        let players = MatchPlayerCollection::from_squads(&left_squad, &right_squad);

        let mut match_position_data = if !config.match_recordings {
            ResultMatchPositionData::empty()
        } else if MatchRuntime::events_mode() {
            ResultMatchPositionData::new_with_tracking()
        } else {
            ResultMatchPositionData::new()
        };

        let mut field = MatchField::new(W, H, left_squad, right_squad);

        // Dev testing knob (2026-07-19): red cards are naturally rare
        // (~0.15/match), too rare to reliably test the send-off/restart
        // fix by just playing matches. `GAFFER_FORCE_SECOND_YELLOW`
        // presets every outfield player's yellow-card count to 1 at
        // kickoff, so the first natural yellow anyone gets is a genuine
        // second-yellow-to-red dismissal. Opt-in, match-logs builds
        // only, zero effect unless the env var is set.
        #[cfg(feature = "match-logs")]
        if std::env::var("GAFFER_FORCE_SECOND_YELLOW").is_ok() {
            for p in field.players.iter_mut() {
                if p.tactical_position.current_position.position_group()
                    != PlayerFieldPositionGroup::Goalkeeper
                {
                    p.yellow_cards = 1;
                }
            }
        }
        #[cfg(feature = "match-logs")]
        crate::r#match::engine::player::events::players::debug_force_red::arm_from_env();

        let mut context = MatchContext::new_with_config(&field, players, score, &config);
        // Stash the starting tactics inside the context's match plan so
        // `build_result` can read them — no extra parameters threaded
        // through the state machine.
        context.starting_home_tactic = starting_home_tactic;
        context.starting_away_tactic = starting_away_tactic;

        // Seed the chemistry map from the kickoff XI of each side.
        // Pair scores stay constant for the match — live events could
        // adjust them, but the initial baseline is what feeds the pass
        // evaluator's one-touch bonus from the first whistle.
        let chemistry_roster: Vec<(u32, u32, PlayerFieldPositionGroup, f32, f32)> = field
            .players
            .iter()
            .map(|p| {
                (
                    p.id,
                    p.team_id,
                    p.tactical_position.current_position.position_group(),
                    p.position.y,
                    p.skills.mental.teamwork,
                )
            })
            .collect();
        let field_h = field.size.height as f32;
        context.chemistry.seed_from_roster(&chemistry_roster, field_h);

        // "Link with X" pairs: pin the pair's chemistry high so the
        // one-touch bonus (>0.65 gate) is guaranteed on top of the
        // evaluator's explicit link bias. Symmetric fields, so one
        // direction of the scan is enough.
        for p in field.players.iter() {
            if let Some(partner) = p.link_target {
                context.chemistry.set(p.id, partner, 0.90);
            }
        }

        // "Block passes into X" assignments: snapshot for the pass
        // evaluator (static pairing; positions are read live).
        for p in field.players.iter() {
            if let Some(target) = p.intercept_target {
                context.intercept_assignments.push((p.id, target));
            }
        }

        // Home-crowd arousal — the play-quality half of home advantage.
        // Stamped once on every match player (starters AND bench, so
        // substitutes carry it on) and consumed inside `effective_skill`.
        // Scaled by the environment so an empty-stadium friendly confers
        // ~nothing (matching the COVID ghost-game finding that home
        // advantage largely vanishes without crowds) and a packed derby
        // confers the full edge. At the default environment (crowd 0.55
        // × home_advantage 0.50 → edge 0.275) this is home ≈ +3.3% /
        // away ≈ −2.75% effective skill. Magnitude was titrated against
        // the engine's measured response: ±(+1.7/−1.4)% produced a
        // +5.5pp home-win gap at equal strength; the documented real
        // split (~45/25/30, +0.35 home goals) needs roughly double
        // that, landing here. The officiating half (referee marginal
        // calls) stacks on top via `RefereeProfile::home_bias`.
        let home_edge =
            (context.environment.crowd_intensity * context.environment.home_advantage).clamp(0.0, 1.0);
        let home_arousal = 1.0 + 0.12 * home_edge;
        let away_arousal = 1.0 - 0.07 * home_edge;
        let home_team_id = field.home_team_id;
        for p in field.players.iter_mut().chain(field.substitutes.iter_mut()) {
            p.crowd_arousal = if p.team_id == home_team_id {
                home_arousal
            } else {
                away_arousal
            };
        }

        if MatchRuntime::events_mode() {
            context.enable_logging();
        }

        let mut state_manager = StateManager::new();

        // Match kickoff — home team (playing Left in the first half)
        // starts the game with possession on the centre spot. Without
        // this the ball sits at centre until the emergency chaser
        // override fires, producing a ~14-second dead patch.
        assign_kickoff(&mut field, PlayerSide::Left);

        while let Some(state) = state_manager.next(&context.score, context.is_knockout) {
            context.state.set(state);

            let play_state_result = match state {
                MatchState::PenaltyShootout => {
                    Self::run_penalty_shootout(&mut field, &mut context);
                    PlayMatchStateResult::default()
                }
                _ => Self::play_inner(&mut field, &mut context, &mut match_position_data),
            };

            StateManager::handle_state_finish(&mut context, &mut field, play_state_result);
        }

        let result_start = Instant::now();
        let result = Self::build_result(field, context, match_position_data);
        perf.record_match_result_processing(result_start.elapsed());
        perf.record_match_total(match_start.elapsed());
        result
    }

    /// Stub match: skips the whole simulation and returns a 0-0
    /// scoreline with the minimum data downstream consumers expect
    /// (team IDs in `Score`, player IDs in the field squads). Gated
    /// on the `match-stub` Cargo feature; intended for profiling the
    /// pipeline around the engine.
    #[cfg(feature = "match-stub")]
    pub(super) fn play_stub(left_squad: MatchSquad, right_squad: MatchSquad) -> MatchResultRaw {
        use crate::r#match::engine::result::FieldSquad;

        let mut result = MatchResultRaw::with_match_time(90 * 60 * 1000);
        result.score = Some(Score::new(left_squad.team_id, right_squad.team_id));
        result.left_team_players = FieldSquad::from_team(&left_squad);
        result.right_team_players = FieldSquad::from_team(&right_squad);
        result
    }

    pub(super) fn build_result(
        field: MatchField,
        mut context: MatchContext,
        match_position_data: ResultMatchPositionData,
    ) -> MatchResultRaw {
        let mut result = MatchResultRaw::with_match_time(context.total_match_time);

        context.fill_details();

        result.second_half_start_ms = context.second_half_start_ms;
        result.additional_time_ms = context.additional_time_ms;
        result.penalty_shootout = context.penalty_shootout_kicks.clone();
        result.score = Some(context.score.clone());

        // Assign squads based on team IDs, not field positions
        let left_side_squad = field.left_side_players.expect("left team players");
        let right_side_squad = field.right_side_players.expect("right team players");

        // The engine swaps `left_team_tactics` / `right_team_tactics` on
        // every halftime swap, so at this point those fields track the
        // tactic the team currently ON the left / right is using —
        // including any mid-match shape change applied by
        // `evaluate_situational_shape`. Map them back to home / away
        // using the same team-id rule we use for player squads.
        let left_team_tac = field.left_team_tactics.tactic_type;
        let right_team_tac = field.right_team_tactics.tactic_type;

        if left_side_squad.team_id == field.home_team_id {
            result.left_team_players = left_side_squad;
            result.right_team_players = right_side_squad;
            result.final_home_tactic = Some(left_team_tac);
            result.final_away_tactic = Some(right_team_tac);
        } else {
            result.left_team_players = right_side_squad;
            result.right_team_players = left_side_squad;
            result.final_home_tactic = Some(right_team_tac);
            result.final_away_tactic = Some(left_team_tac);
        }
        result.starting_home_tactic = context.starting_home_tactic;
        result.starting_away_tactic = context.starting_away_tactic;
        result.shape_change_minute = context.first_shape_change_minute;

        // Copy substitution records to result
        for sub_record in &context.substitutions {
            if sub_record.team_id == result.left_team_players.team_id {
                result
                    .left_team_players
                    .mark_substitute_used(sub_record.player_in_id);
            } else {
                result
                    .right_team_players
                    .mark_substitute_used(sub_record.player_in_id);
            }

            result.substitutions.push(SubstitutionInfo {
                team_id: sub_record.team_id,
                player_out_id: sub_record.player_out_id,
                player_in_id: sub_record.player_in_id,
                match_time_ms: sub_record.match_time,
                reason: sub_record.reason,
            });
        }

        result.position_data = match_position_data;

        // Extract per-player stats and calculate match ratings.
        //
        // Two passes are needed because the contextual (public) rating
        // reads team-behaviour summaries folded from EVERY player's stat
        // line — who shot, who held the ball, who had to defend. So we
        // first materialise all the stat lines and physical snapshots
        // (subs included), then build the per-side summaries, then score
        // each player against their own / opponent summary plus their
        // own physical snapshot.
        let score_ref = result.score.as_ref().unwrap();
        let home_goals = score_ref.home_team.get();
        let away_goals = score_ref.away_team.get();
        let home_team_id = score_ref.home_team.team_id;

        // ── Pass 1: materialise stat lines + physical snapshots ──────────
        for player in &field.players {
            let minutes = player.minutes_played_at(context.total_match_time);
            let stats = player.to_match_end_stats(minutes);
            result.player_stats.insert(player.id, stats);

            // Final-whistle physical snapshot — every player still on the
            // pitch at full time. Captured here because `field.players`
            // is the canonical "who was on the pitch when the whistle
            // blew" view, and the engine's in-match condition drain
            // has been applied to each `MatchPlayer` by now. Players
            // who were substituted off are folded in below from
            // `context.substituted_out_physical_snapshots` so the same
            // player never gets two snapshots.
            let phys_snapshot = player.to_physical_snapshot(context.total_match_time);
            result.physical_snapshots.insert(player.id, phys_snapshot);
        }

        // Stat lines for substituted-out players (their snapshot was taken
        // at the swap minute; folded in just below). Ratings are filled
        // in Pass 2 alongside everyone else.
        for (player_id, stats) in context.substituted_out_stats.drain(..) {
            result.player_stats.insert(player_id, stats);
        }

        // Fold every "subbed-off" physical snapshot into the result. The
        // snapshot was taken at the swap minute, so the persisted
        // condition drop will use the right exit-time energy. A
        // substitute coming on later replaces the same shirt but has
        // their own snapshot logged at full time above — these two
        // snapshots are keyed by `player_id`, so they cannot collide.
        for snapshot in context.substituted_out_physical_snapshots.drain(..) {
            result
                .physical_snapshots
                .insert(snapshot.player_id, snapshot);
        }

        // ── Pass 2a: per-side behaviour summaries ────────────────────────
        // Fold each side's stat lines (starters + used subs) into the
        // team summary the expectation layer reads.
        let left_summary = {
            let side = &result.left_team_players;
            TeamRatingSummary::from_stats(
                side.main
                    .iter()
                    .chain(side.substitutes_used.iter())
                    .filter_map(|id| result.player_stats.get(id)),
            )
        };
        let right_summary = {
            let side = &result.right_team_players;
            TeamRatingSummary::from_stats(
                side.main
                    .iter()
                    .chain(side.substitutes_used.iter())
                    .filter_map(|id| result.player_stats.get(id)),
            )
        };
        let left_team_id = result.left_team_players.team_id;

        // ── Pass 2b: raw (stat-line) + contextual (public) ratings ───────
        // `raw_match_rating` stays the pure `calculate()` verdict; the
        // public `match_rating` seed layers Stage-2 (team expectation)
        // and Stage-3 (condition) on top. Reputation / personality
        // shaping is added later by the league pipeline.
        let rated_ids: Vec<u32> = result.player_stats.keys().copied().collect();
        for pid in rated_ids {
            let on_left = result.left_team_players.main.contains(&pid)
                || result.left_team_players.substitutes_used.contains(&pid);
            let team_id = if on_left {
                left_team_id
            } else {
                result.right_team_players.team_id
            };
            let (team_goals, opponent_goals) = if team_id == home_team_id {
                (home_goals, away_goals)
            } else {
                (away_goals, home_goals)
            };
            let (own_summary, opp_summary) = if on_left {
                (&left_summary, &right_summary)
            } else {
                (&right_summary, &left_summary)
            };
            let expectation = RatingExpectationContext::from_match(
                own_summary,
                opp_summary,
                team_goals,
                opponent_goals,
                result.physical_snapshots.get(&pid),
            );
            if let Some(stats) = result.player_stats.get_mut(&pid) {
                let (raw, contextual) = {
                    let ctx = RatingContext::new(stats, team_goals, opponent_goals);
                    (ctx.calculate(), ctx.calculate_contextual(&expectation))
                };
                stats.raw_match_rating = raw;
                stats.match_rating = contextual;
            }
        }

        // Blowout diagnostic — off by default (see `match-logs` feature).
        // The aggregation itself is O(players × stats), so we skip the
        // whole block in production. Enable with `--features match-logs`
        // in `.dev/match` to analyse shot / goal sources.
        #[cfg(feature = "match-logs")]
        {
            let total_goals = home_goals + away_goals;
            if total_goals >= 8 {
                let away_team_id = if field.home_team_id == home_team_id {
                    field.away_team_id
                } else {
                    field.home_team_id
                };
                Self::log_blowout_profile(
                    &field.players,
                    &context.substitutions,
                    &result,
                    home_team_id,
                    away_team_id,
                    home_goals,
                    away_goals,
                );
            }
        }

        result
    }

    /// Aggregate per-team skill dump for post-match analysis. Runs only
    /// on 8+ goal matches when the `match-logs` feature is enabled.
    /// Skipped entirely in production builds.
    #[cfg(feature = "match-logs")]
    pub(super) fn log_blowout_profile(
        players: &[MatchPlayer],
        substitutions: &[SubstitutionRecord],
        result: &MatchResultRaw,
        home_team_id: u32,
        away_team_id: u32,
        home_goals: u8,
        away_goals: u8,
    ) {
        struct TeamAgg {
            fwd_finishing: f32,
            fwd_technique: f32,
            fwd_composure: f32,
            fwd_count: u32,
            def_marking: f32,
            def_tackling: f32,
            def_positioning: f32,
            def_count: u32,
            gk_handling: f32,
            gk_reflexes: f32,
            gk_agility: f32,
            gk_count: u32,
            total_shots: u16,
            total_on_target: u16,
            passes_attempted: u32,
            passes_completed: u32,
            tackles: u32,
            interceptions: u32,
            saves: u32,
            fouls: u32,
            xg_total: f32,
        }
        impl TeamAgg {
            fn new() -> Self {
                Self {
                    fwd_finishing: 0.0,
                    fwd_technique: 0.0,
                    fwd_composure: 0.0,
                    fwd_count: 0,
                    def_marking: 0.0,
                    def_tackling: 0.0,
                    def_positioning: 0.0,
                    def_count: 0,
                    gk_handling: 0.0,
                    gk_reflexes: 0.0,
                    gk_agility: 0.0,
                    gk_count: 0,
                    total_shots: 0,
                    total_on_target: 0,
                    passes_attempted: 0,
                    passes_completed: 0,
                    tackles: 0,
                    interceptions: 0,
                    saves: 0,
                    fouls: 0,
                    xg_total: 0.0,
                }
            }
        }

        let mut home_agg = TeamAgg::new();
        let mut away_agg = TeamAgg::new();

        // Skill profile aggregation runs over the current XI only — sub-out
        // players' skills aren't retained on the field after they leave.
        // Raw `.skills.*` reads here are intentional: this aggregator
        // feeds the post-match log (a snapshot of "what skill profile
        // does this XI have on paper?"), not a live decision — fatigue
        // would only confuse the report.
        for player in players {
            let agg = if player.team_id == home_team_id {
                &mut home_agg
            } else {
                &mut away_agg
            };
            match player.tactical_position.current_position.position_group() {
                PlayerFieldPositionGroup::Forward => {
                    agg.fwd_finishing += player.skills.technical.finishing;
                    agg.fwd_technique += player.skills.technical.technique;
                    agg.fwd_composure += player.skills.mental.composure;
                    agg.fwd_count += 1;
                }
                PlayerFieldPositionGroup::Defender => {
                    agg.def_marking += player.skills.technical.marking;
                    agg.def_tackling += player.skills.technical.tackling;
                    agg.def_positioning += player.skills.mental.positioning;
                    agg.def_count += 1;
                }
                PlayerFieldPositionGroup::Goalkeeper => {
                    agg.gk_handling += player.skills.goalkeeping.handling;
                    agg.gk_reflexes += player.skills.goalkeeping.reflexes;
                    agg.gk_agility += player.skills.physical.agility;
                    agg.gk_count += 1;
                }
                _ => {}
            }
        }

        // Shot/OT aggregation must include sub-out players: their goals
        // count toward the team total, so their attempts must too.
        // Otherwise the log shows "7 goals from 2 OT" simply because the
        // hat-trick scorer was substituted off and their stats fell out
        // of the field.players iteration.
        let team_for = |player_id: u32| -> Option<u32> {
            players
                .iter()
                .find(|p| p.id == player_id)
                .map(|p| p.team_id)
                .or_else(|| {
                    substitutions
                        .iter()
                        .find(|s| s.player_out_id == player_id)
                        .map(|s| s.team_id)
                })
        };
        for (player_id, stats) in result.player_stats.iter() {
            let Some(team_id) = team_for(*player_id) else {
                continue;
            };
            let agg = if team_id == home_team_id {
                &mut home_agg
            } else {
                &mut away_agg
            };
            agg.total_shots += stats.shots_total;
            agg.total_on_target += stats.shots_on_target;
            agg.passes_attempted += stats.passes_attempted as u32;
            agg.passes_completed += stats.passes_completed as u32;
            agg.tackles += stats.tackles as u32;
            agg.interceptions += stats.interceptions as u32;
            agg.saves += stats.saves as u32;
            agg.fouls += stats.fouls as u32;
            agg.xg_total += stats.xg;
        }

        // Goal source breakdown — distinguishes "proper" shot goals from
        // own goals and from goals credited to a scorer who never actually
        // took a shot in the match (so the goal came from a pass, a
        // clearance-into-net, or a loose-ball scramble that the engine
        // credits via the `current_owner.or(previous_owner)` fallback).
        // The last bucket is the one we need to watch — goals should flow
        // almost exclusively from shots.
        let score_ref = result.score.as_ref().unwrap();
        let mut home_own: u16 = 0;
        let mut away_own: u16 = 0;
        let mut home_nonshot: u16 = 0;
        let mut away_nonshot: u16 = 0;
        for g in score_ref.detail() {
            if g.stat_type != MatchStatisticType::Goal {
                continue;
            }
            let scorer_team = team_for(g.player_id);
            let is_home_scorer = scorer_team == Some(home_team_id);
            if g.is_auto_goal {
                // Own goal — credited to opponent team
                if is_home_scorer {
                    away_own += 1;
                } else {
                    home_own += 1;
                }
                continue;
            }
            // Non-auto goal. Did this scorer take ANY shot in the match?
            let took_shot = result
                .player_stats
                .get(&g.player_id)
                .map(|s| s.shots_total > 0)
                .unwrap_or(false);
            if !took_shot {
                if is_home_scorer {
                    home_nonshot += 1;
                } else {
                    away_nonshot += 1;
                }
            }
        }

        let total_passes = (home_agg.passes_attempted + away_agg.passes_attempted).max(1) as f32;
        let home_possession = home_agg.passes_attempted as f32 / total_passes * 100.0;
        let away_possession = away_agg.passes_attempted as f32 / total_passes * 100.0;

        let fmt_team = |tag: &str,
                        team_id: u32,
                        goals: u8,
                        agg: &TeamAgg,
                        own_against: u16,
                        nonshot: u16,
                        possession: f32| {
            let fc = agg.fwd_count.max(1) as f32;
            let dc = agg.def_count.max(1) as f32;
            let gc = agg.gk_count.max(1) as f32;
            let pass_acc = if agg.passes_attempted > 0 {
                agg.passes_completed as f32 / agg.passes_attempted as f32 * 100.0
            } else {
                0.0
            };
            // xG overperformance: goals above what the shot quality
            // predicted. Real football: |diff| rarely exceeds xG by more
            // than ~1.5. Big positive means clinical finishing or a
            // generous finishing roll; big negative means wasted chances.
            let xg_delta = goals as f32 - agg.xg_total;
            // Shot volume per xG: a team that took 150 shots for only 4.5
            // xG was firing from impossible angles / long range — a
            // marker of blind shooting during desperation mode.
            let shots_per_xg = if agg.xg_total > 0.01 {
                agg.total_shots as f32 / agg.xg_total
            } else {
                0.0
            };
            format!(
                "{} team={} gls={} (own-ag={} non-shot={}) shots={} ot={} xG={:.2} (Δ{:+.2}, s/xG={:.0}) | poss={:.0}% pass={}/{} ({:.0}%) tck={} int={} sv={} fl={} | FWD fin={:.1} tec={:.1} com={:.1} | DEF mrk={:.1} tck={:.1} pos={:.1} | GK hnd={:.1} ref={:.1} agi={:.1}",
                tag,
                team_id,
                goals,
                own_against,
                nonshot,
                agg.total_shots,
                agg.total_on_target,
                agg.xg_total,
                xg_delta,
                shots_per_xg,
                possession,
                agg.passes_completed,
                agg.passes_attempted,
                pass_acc,
                agg.tackles,
                agg.interceptions,
                agg.saves,
                agg.fouls,
                agg.fwd_finishing / fc,
                agg.fwd_technique / fc,
                agg.fwd_composure / fc,
                agg.def_marking / dc,
                agg.def_tackling / dc,
                agg.def_positioning / dc,
                agg.gk_handling / gc,
                agg.gk_reflexes / gc,
                agg.gk_agility / gc,
            )
        };

        match_log_info!(
            "BLOWOUT {}-{} (total {}g)",
            home_goals,
            away_goals,
            home_goals + away_goals
        );
        // Notation:
        //   own-ag   own goals (our player into our own net)
        //   non-shot goals credited to a scorer who took zero shots — came
        //            via pass/scramble/deflection path
        //   xG       sum of expected-goals across all shots we took
        //   Δ        goals minus xG (overperformance if positive)
        //   s/xG     shots per xG — high = blind long-range spam
        //   poss     share of total pass attempts (possession proxy)
        //   pass     completed / attempted (acc %)
        //   tck/int/sv/fl  tackles / interceptions / saves / fouls
        match_log_info!(
            "  {}",
            fmt_team(
                "HOME",
                home_team_id,
                home_goals,
                &home_agg,
                home_own,
                home_nonshot,
                home_possession
            )
        );
        match_log_info!(
            "  {}",
            fmt_team(
                "AWAY",
                away_team_id,
                away_goals,
                &away_agg,
                away_own,
                away_nonshot,
                away_possession
            )
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Match state loop
    // ───────────────────────────────────────────────────────────────────────

    pub(super) fn play_inner(
        field: &mut MatchField,
        context: &mut MatchContext,
        match_data: &mut ResultMatchPositionData,
    ) -> PlayMatchStateResult {
        let result = PlayMatchStateResult::default();
        let inner_start = Instant::now();
        let mut tick_count: u64 = 0;

        let mut next_sub_time_ms: u64 = 0;
        let mut sub_times_initialized = false;
        let mut et_bonus_granted = false;

        let mut tick_ctx = GameTickContext::new(field);
        let mut events = EventCollection::with_capacity(10);

        let mut tick_parity: u32 = 0;
        let mut coach_eval_counter: u32 = 0;
        let mut tactical_eval_counter: u32 = 0;
        // Tactical refresh uses an adaptive cadence: BASE during stable
        // play, TRANSITION right after possession swings / set-piece
        // restarts / goals / coach-instruction changes / ball entering
        // or leaving the attacking third. Each "transition trigger"
        // opens a TRANSITION_WINDOW_TICKS window during which the
        // cheaper TRANSITION interval is used.
        const BASE_TACTICAL_INTERVAL_TICKS: u32 = 25;
        const TRANSITION_TACTICAL_INTERVAL_TICKS: u32 = 10;
        const TRANSITION_WINDOW_TICKS: u32 = 40;
        let mut transition_window_remaining: u32 = TRANSITION_WINDOW_TICKS;
        // Snapshots used to detect transition triggers between refresh
        // points without a per-tick walk over players.
        let mut last_owner_id: Option<u32> = field.ball.current_owner;
        let mut last_possession_team: Option<u32> = last_owner_id
            .and_then(|id| field.players.iter().find(|p| p.id == id).map(|p| p.team_id));
        let mut last_home_score: u8 = context.score.home_team.get();
        let mut last_away_score: u8 = context.score.away_team.get();
        let mut last_home_instruction = context.coach_home.instruction;
        let mut last_away_instruction = context.coach_away.instruction;
        let mut last_home_zone = context.tactical_home.ball_zone;
        let mut last_away_zone = context.tactical_away.ball_zone;
        // Position recording cursor — replaces the per-tick
        // `timestamp % POSITION_RECORD_INTERVAL_MS == 0` check. Round
        // the starting timestamp UP to the next multiple of the
        // recording interval so a half restart preserves the original
        // 30 ms cadence (the loop increments time *before* the body,
        // so we never see `t == 0`).
        let initial_t = context.total_match_time;
        let mut next_position_record_ms: u64 =
            (initial_t / Self::POSITION_RECORD_INTERVAL_MS + 1) * Self::POSITION_RECORD_INTERVAL_MS;
        let track_positions = match_data.is_tracking_positions();

        // Cross-player teammate triggers ("when the 6 receives, X makes a
        // run"): re-evaluated only when the ball owner changes. Skipped
        // entirely when no player carries a trigger assignment.
        let any_teammate_triggers = field.players.iter().any(|p| p.teammate_trigger.is_some());
        let mut last_trigger_owner: Option<u32> = None;
        // Whole-team shape rules (plan Phase 5): evaluated per tick for
        // ruled players only. Mutating start_position moves the anchor
        // every positional state blends toward — a walk/run to the new
        // spot, never a snap.
        let any_shape_rules = field.players.iter().any(|p| !p.shape_rules.is_empty());

        while context.increment_time() {
            tick_count += 1;

            // Post-goal dead time: only the match clock advances while
            // the players celebrate / walk back / wait for the restart
            // whistle. No ball physics, no AI, no events, no coach
            // evals — the world is already reset and frozen in
            // formation, so skipping the tick body IS the celebration.
            // See `MatchContext::dead_ball_until_ms` for why this pause
            // is load-bearing (it consumed the post-goal hot window
            // that made goals beget goals).
            if context.total_match_time < context.dead_ball_until_ms {
                // 2026-07-19 realism-bug: apply any card that was left
                // waiting for the game to actually stop (an advantage
                // that played out mid-play — see `awaiting_stoppage` on
                // MatchContext). This is the first tick of ANY real
                // stoppage (foul restart, corner, goal kick, throw-in,
                // goal, half-time all route through this branch), so a
                // delayed card is shown, and the player sent off,
                // exactly when a real referee would: at the pause, not
                // mid-play.
                if !context.awaiting_stoppage.is_empty() {
                    let awaiting = std::mem::take(&mut context.awaiting_stoppage);
                    for adv in awaiting {
                        let cards = PlayerEventDispatcher::apply_card_decision(
                            adv.fouler_id,
                            adv.severity,
                            adv.yellow_prob,
                            adv.red_prob,
                            field,
                            context,
                        );
                        FoulResolver::emit_card_events(adv.fouler_id, cards, &mut events);
                    }
                    if events.has_events() {
                        EventDispatcher::dispatch(&mut events, field, context, match_data, true);
                    }
                }
                // §13.4: foul/corner/goal-kick windows get a real retreat
                // instead of a total freeze — `dead_ball_retreat_active`
                // is never set for the post-goal kickoff freeze above, so
                // that pause stays byte-for-byte unchanged.
                if context.dead_ball_retreat_active {
                    Self::apply_restart_retreat_tick(field, context);
                    if track_positions && context.total_match_time >= next_position_record_ms {
                        Self::write_match_positions(field, context.total_match_time, match_data);
                        next_position_record_ms += Self::POSITION_RECORD_INTERVAL_MS;
                    }
                    // Real wait-or-exploit decision: once past the
                    // minimum reaction-time floor, each tick has a small
                    // chance to end the stoppage early — whoever hasn't
                    // reached their `restart_retreat_target` yet gets
                    // caught short. No sourced real-world quick-vs-patient
                    // split exists; this roll is an explicitly-flagged
                    // estimate (tune EARLY_RESTART_ROLL_PER_TICK if the
                    // resulting fast/patient mix needs adjusting). Real
                    // football treats a quick restart as the exception,
                    // not the rule — sized so a typical ~200-tick
                    // exploitable window (the gap between the minimum
                    // reaction floor and the maximum cap) has roughly a
                    // 20% chance of firing early, not a coin flip.
                    const EARLY_RESTART_ROLL_PER_TICK: u64 = 12; // /10000 ≈ 0.12%/tick

                    // realism-bug (2026-07-27): scale that flat baseline
                    // by the real team-shape advantage
                    // (`set_pieces::restart_timing_advantage`) — a team
                    // with more players still forward/ready than the
                    // defence has recovered should take it quickly
                    // (real doctrine: quick free kicks/corners/throw-ins
                    // exploit a disorganised defence); a team that isn't
                    // itself ready yet has no reason to rush. Penalties
                    // are excluded — a real penalty taker doesn't rush or
                    // delay the kick based on outfield shape, it's a
                    // fixed isolated procedure, so they keep the original
                    // flat baseline. The scale factor itself has no
                    // sourced real-world figure (same flagged-estimate
                    // status as EARLY_RESTART_ROLL_PER_TICK) — tune
                    // ADVANTAGE_ROLL_SCALE if the resulting fast/patient
                    // mix needs adjusting.
                    //
                    // realism-bug (2026-07-27, follow-up — Pavel): a direct
                    // free kick in genuine shooting/crossing range (≤280u —
                    // the same "crossable range" `award_restart_for_foul`
                    // already uses to decide whether to form a wall at all
                    // and stage the §11.7 attacking shape) was firing the
                    // quick-restart roll and rushing a low-value sideways
                    // pass before either the wall or the attacking shape
                    // had time to form. The counting metric alone can't
                    // tell this case apart from a genuine tactical
                    // opportunity: a defensive wall + markers structurally
                    // needs MORE players to organise than the ~4 the
                    // attacking side stages, so `defending_scrambling` is
                    // almost always higher right after the award — reading
                    // as "advantage" even though no real team would rush a
                    // promising direct free kick to exploit it. Real
                    // doctrine has no ambiguity here: a shootable/crossable
                    // free kick is patiently organised, full stop — a quick
                    // take only makes sense when a genuinely open,
                    // clearly-better-placed teammate exists right now,
                    // which isn't something this counting metric can judge
                    // (it counts headcount, not lane occlusion or shot
                    // quality). Per Pavel's own fallback, hard-disable the
                    // quick option in this zone instead of trying to model
                    // the narrower "is there truly a great pass on"
                    // condition — multiplier 0 means the early-exit roll
                    // can never fire, so the full natural stoppage window
                    // always plays out and `resolve_free_kick`'s real
                    // distance-based shot/cross model decides the outcome.
                    //
                    // realism-bug (2026-07-27, follow-up 2 — Pavel): the
                    // identical bias hits corners even harder — a corner's
                    // whole value is numbers in the box, and Pavel directly
                    // observed deliveries with 2 attackers vs 6 defenders
                    // because the corner fired before the attacking
                    // box-crowding shape (§9.3.3 pushed-up centre-backs/
                    // second wave) had time to arrive, while the defending
                    // six-slot block (naturally more bodies) read as
                    // "scrambling" and drove the multiplier up. Unlike a
                    // deep free kick, there's no "safe zone" for a corner —
                    // it's always taken right in the danger area by
                    // definition — so this is a blanket disable, same
                    // mechanism as the direct-FK case above.
                    // realism-bug (2026-07-27, follow-up 3 — Pavel): folded
                    // Penalty into the same shared classification
                    // (`set_pieces::is_patiently_organized_restart`) — a
                    // real penalty is never taken "quickly" based on
                    // outfield shape either, so it belongs in the same
                    // always-wait bucket as corners/close free kicks, not
                    // a separate flat-baseline case that could still
                    // occasionally roll an early exit.
                    let kicking_side = field
                        .ball
                        .current_owner
                        .and_then(|id| field.players.iter().find(|p| p.id == id))
                        .and_then(|p| p.side);
                    let dist_to_goal_if_free_kick = {
                        let goal_pos = match kicking_side {
                            Some(PlayerSide::Left) => context.goal_positions.right,
                            Some(PlayerSide::Right) => context.goal_positions.left,
                            None => context.goal_positions.right,
                        };
                        (field.ball.position - goal_pos).magnitude()
                    };
                    let never_rush = crate::r#match::engine::set_pieces::is_patiently_organized_restart(
                        field.ball.pass_origin_restart,
                        dist_to_goal_if_free_kick,
                    );
                    let advantage_multiplier: f32 = if never_rush {
                        0.0
                    } else {
                        match kicking_side {
                            Some(side) => {
                                let advantage = crate::r#match::engine::set_pieces::counterattack_advantage(
                                    &field.players,
                                    side,
                                    context.field_size.width as f32,
                                );
                                const ADVANTAGE_ROLL_SCALE: f32 = 0.35;
                                (1.0 + advantage as f32 * ADVANTAGE_ROLL_SCALE).clamp(0.15, 4.0)
                            }
                            None => 1.0,
                        }
                    };
                    let scaled_roll =
                        ((EARLY_RESTART_ROLL_PER_TICK as f32) * advantage_multiplier) as u64;
                    if context.total_match_time >= context.dead_ball_min_ms
                        && context.rng.range_u64(0, 10000) < scaled_roll
                    {
                        context.dead_ball_until_ms = context.total_match_time;
                    }
                    // realism-bug (2026-07-27, follow-up 3 — Pavel): for the
                    // always-patiently-organized types, don't make a viewer
                    // sit through the rest of a randomly-chosen stoppage
                    // window once the shape is ACTUALLY complete — end it
                    // the moment every player who was assigned a target has
                    // reached it. This is what actually realizes "don't
                    // waste time on a run that changes nothing": speeding
                    // up the walk (`apply_restart_retreat_tick`, tick.rs)
                    // alone would just mean everyone stands still for
                    // whatever's left of the random cap. Goal kicks/throw-
                    // ins/deep free kicks are untouched — they keep the
                    // natural randomized window, since that's the window
                    // `counterattack_advantage` is judging.
                    if never_rush
                        && context.total_match_time >= context.dead_ball_min_ms
                        && field.players.iter().all(|p| p.restart_retreat_target.is_none())
                    {
                        context.dead_ball_until_ms = context.total_match_time;
                    }
                }
                continue;
            }
            // §13.4: the retreat window (if any) just ended — either the
            // early-exploit roll fired above or the hard maximum was
            // reached. Sweep every player's target so a straggler who
            // never arrived can't keep drifting toward a stale anchor
            // once open play resumes; runs exactly once per window since
            // the flag is cleared in the same breath.
            if context.dead_ball_retreat_active {
                context.dead_ball_retreat_active = false;
                for player in field.players.iter_mut() {
                    player.restart_retreat_target = None;
                }
            }

            tick_parity += 1;
            coach_eval_counter += 1;
            tactical_eval_counter += 1;
            if transition_window_remaining > 0 {
                transition_window_remaining -= 1;
            }

            // Coach evaluates every 500 ticks (~5 seconds of match time)
            if coach_eval_counter >= 500 {
                coach_eval_counter = 0;
                Self::evaluate_coaches(field, context);
                // Once every coach-eval slice, also probe for situational
                // formation overrides — the manager swap to a chasing /
                // protecting shape based on score and minute. Cheap: a
                // single match arm and an equality check against the
                // current type per side.
                Self::evaluate_situational_shape(field, &mut *context);
                // Condition-trajectory sampling for the dev harness —
                // average condition per position group per 15-min band.
                // Rides the coach cadence so it costs one 22-player walk
                // every 5 sim-seconds, match-logs builds only.
                #[cfg(feature = "match-logs")]
                {
                    use crate::r#match::player::strategies::players::ops::forward_shot_decision::time_band_diag;
                    use std::sync::atomic::Ordering;
                    let band = time_band_diag::band_for_minute(
                        (context.total_match_time / 60_000) as u32,
                    );
                    for p in field.players.iter().filter(|p| !p.is_sent_off) {
                        let group = match p.tactical_position.current_position.position_group() {
                            crate::PlayerFieldPositionGroup::Goalkeeper => 0,
                            crate::PlayerFieldPositionGroup::Defender => 1,
                            crate::PlayerFieldPositionGroup::Midfielder => 2,
                            crate::PlayerFieldPositionGroup::Forward => 3,
                        };
                        time_band_diag::COND_SUM_BY_BAND_GROUP[band][group]
                            .fetch_add(p.player_attributes.condition.max(0) as u64, Ordering::Relaxed);
                        time_band_diag::COND_N_BY_BAND_GROUP[band][group]
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            // Team-level tactical state (phase, possession timers, line
            // height) used a fixed 10-tick cadence. Adaptive cadence:
            // stable possession uses BASE (25 ticks), while a 40-tick
            // window after any transition trigger drops to TRANSITION
            // (10 ticks) so phase/line-height/transition windows still
            // resolve crisply when the game state actually shifts.
            //
            // Triggers (each cheap, no per-tick player walks):
            //   • possession owner team changed
            //   • score changed (goal scored — handled via reset path)
            //   • coach instruction changed for either side
            //   • ball zone moved into / out of attacking third for
            //     either side
            //
            // Set-piece restarts are covered indirectly: kickoff /
            // corner / goal kick all reassign the ball owner, which
            // flips `last_possession_team` and re-opens the window.
            //
            // Cheap fast path: most ticks have the same `current_owner`
            // as the previous tick (passes/dribbles span many ticks).
            // Only re-resolve `team_id` via a 22-element scan when the
            // raw id actually changed since the last evaluation.
            let raw_owner = field.ball.current_owner;

            // Teammate triggers: on any owner change, actors with a
            // trigger switch their effective directive on (trigger
            // teammate on the ball) or revert to their base directive.
            if any_teammate_triggers && raw_owner != last_trigger_owner {
                last_trigger_owner = raw_owner;
                for p in field.players.iter_mut() {
                    if let Some((trigger_id, action)) = p.teammate_trigger {
                        p.behavioral_directive = if raw_owner == Some(trigger_id) {
                            Some(action)
                        } else {
                            p.directive_base
                        };
                    }
                }
            }

            // §11.5 kickoff short-pass window maintenance: the window
            // dies the moment the taker releases the ball (any ownership
            // change, including the pass leaving the foot) or the tick
            // budget runs out. At most one player can hold a window.
            // realism-bug (2026-07-19): throw_in_pass_pending follows the
            // identical rule — same loop, same iteration, so the taker
            // check isn't duplicated.
            for p in field.players.iter_mut() {
                if p.kickoff_pass_pending > 0 {
                    if raw_owner != Some(p.id) {
                        p.kickoff_pass_pending = 0;
                    } else {
                        p.kickoff_pass_pending -= 1;
                    }
                }
                if p.throw_in_pass_pending > 0 {
                    if raw_owner != Some(p.id) {
                        p.throw_in_pass_pending = 0;
                    } else {
                        p.throw_in_pass_pending -= 1;
                    }
                }
                if p.free_kick_pass_pending > 0 {
                    if raw_owner != Some(p.id) {
                        p.free_kick_pass_pending = 0;
                    } else {
                        p.free_kick_pass_pending -= 1;
                    }
                }
                if p.aerial_contest_won > 0 {
                    p.aerial_contest_won -= 1;
                }
                // Realism-bug 2026-07-26 (passing follow-up): self-clear
                // the incoming-pass-awareness target. Cleared immediately
                // once this player actually owns the ball (the "incoming"
                // state is over, their own on-ball logic takes over) —
                // otherwise decrements toward the same flight-time budget
                // `handle_pass_to_event` armed it with, so a pass that
                // gets intercepted before arrival still expires on its
                // own instead of lingering.
                if p.incoming_pass_ticks > 0 {
                    if raw_owner == Some(p.id) {
                        p.incoming_pass_ticks = 0;
                        p.incoming_pass_target = None;
                    } else {
                        p.incoming_pass_ticks -= 1;
                        if p.incoming_pass_ticks == 0 {
                            p.incoming_pass_target = None;
                        }
                    }
                }
            }

            // realism-bug (2026-07-25): opportunistic byline-run
            // commitment, armed here — this loop already has &mut
            // field.players + raw_owner (state handlers only ever see
            // &MatchPlayer, so this per-tick loop is the earliest
            // mutation-capable point after ownership resolves). Traced
            // via raw carry-trajectory data that a per-tick-recomputed
            // "is this the best candidate right now" check was too
            // fragile to sustain a real run — a small geometry shift
            // could flip the winner before the run developed. A
            // persistent, ticks-based commitment (same pattern as
            // kickoff/throw-in/free-kick above) survives exactly that.
            // Fresh possession (ownership_duration<=3, i.e. within the
            // first few ticks of receiving/winning the ball — the real
            // moment a winger decides whether to commit), wide channel,
            // and enough room ahead to make a run worthwhile (150-300u
            // from goal — not already in the box, not literally starting
            // from the halfway line; narrowed from an original 420u
            // ceiling after measuring that most armed commitments died
            // long before covering that much ground). Skill-scaled
            // probability: paced, good dribblers commit more often,
            // matching real doctrine.
            //
            // realism-bug follow-up: 28.6% of armed commitments died
            // within 100ms (measured) — almost certainly arming in
            // situations with no real space, e.g. a marker already
            // adjacent at the moment of reception. A real winger doesn't
            // decide to commit to a sprint down the line if he's already
            // being touch-tight marked; a snapshot of opponent positions
            // taken before this loop (can't borrow field.players
            // immutably while iterating it mutably) lets the arm gate
            // require genuine room to start the move.
            let opponent_positions: Vec<(u32, Vector3<f32>)> = field
                .players
                .iter()
                .map(|p| (p.team_id, p.position))
                .collect();
            // Chunk 5b: same "can't borrow field.players immutably
            // while iterating it mutably" constraint as
            // opponent_positions above, but needing the player id and
            // sent-off flag too (to find a specific nearby teammate to
            // lay off to, not just check whether anyone's nearby).
            let all_player_snapshot: Vec<(u32, u32, Vector3<f32>, bool)> = field
                .players
                .iter()
                .map(|p| (p.id, p.team_id, p.position, p.is_sent_off))
                .collect();
            // P5 (A3, docs/attacking-defensive-decision-rebuild-plan.md
            // §2.1/§2.2) — lane-residency tracking, a separate pass so
            // it runs for every player every tick regardless of the
            // byline-commitment loop's own early `continue`s below.
            let lane_residency_field_h = context.field_size.height as f32;
            for p in field.players.iter_mut() {
                let lane =
                    crate::r#match::PitchLane::for_y(lane_residency_field_h, p.position.y).index() as u8;
                if p.lane_residency_lane == lane {
                    p.lane_residency_ticks = p.lane_residency_ticks.saturating_add(1);
                } else {
                    // docs/principles-of-play-phase2-plan.md §4 (2026-08-31):
                    // log the dwell duration that just ended (not a periodic
                    // sample) -- gives a direct distribution of "how long did
                    // this player hold one lane before moving," the raw
                    // signal the Mobility ablation check is measuring.
                    let group = p.tactical_position.current_position.position_group();
                    if group != PlayerFieldPositionGroup::Goalkeeper {
                        match_log_info!(
                            "MOBILITYDIAG id={} group={:?} dwell_ticks={}",
                            p.id,
                            group,
                            p.lane_residency_ticks
                        );
                    }
                    p.lane_residency_lane = lane;
                    p.lane_residency_ticks = 0;
                }
            }
            for p in field.players.iter_mut() {
                // Chunk 5b: decay, unconditional every tick regardless
                // of any other commitment field's state below — kept
                // independent (not gated by byline_commitment_ticks'
                // own early `continue`) so an unrelated active
                // commitment on the same player can never accidentally
                // suppress this.
                if p.combo_partner_ticks > 0 {
                    if raw_owner != Some(p.id) {
                        // The bias is only meaningful while he's still
                        // deciding what to do WITH the ball — cleared
                        // immediately once it's left his feet (passed
                        // away or lost), whether or not the lay-off
                        // window had ticks left.
                        p.combo_partner = None;
                        p.combo_partner_ticks = 0;
                    } else {
                        p.combo_partner_ticks -= 1;
                        if p.combo_partner_ticks == 0 {
                            p.combo_partner = None;
                        }
                    }
                }
                if p.give_and_go_run_ticks > 0 && raw_owner != Some(p.id) {
                    // Only counts down once the ball has actually left
                    // his feet — the run hasn't started yet while he's
                    // still deciding whether/when to lay it off, so the
                    // budget shouldn't drain away before he's released
                    // it. Self-resolves harmlessly (never fires, per
                    // give_and_go_run_velocity's own ball-ownership
                    // gate) if he ends up never passing at all.
                    p.give_and_go_run_ticks -= 1;
                }
                // docs/positional-attack-phase-design.md, Stage 3: the
                // drop-and-spin run, same "only decays once the ball
                // has left his feet" reasoning as give_and_go above —
                // he hasn't started the run yet while still deciding
                // what to do with the deep touch.
                if p.drop_and_spin_run_ticks > 0 && raw_owner != Some(p.id) {
                    p.drop_and_spin_run_ticks -= 1;
                }
                // Stage 2 item 3 (docs/positional-football-master-plan.md,
                // 2026-09-07): decay, unconditional every tick, same
                // idiom/placement as combo_partner_ticks above (checked
                // BEFORE the byline_commitment_ticks `continue` below so
                // it isn't accidentally skipped while a commitment is
                // still active — a reset bias armed the instant an old
                // commitment aborts must survive into the very next tick,
                // not be starved by the early `continue`). Cleared
                // immediately once the ball has left this player's feet —
                // the bias only matters while he's still deciding what to
                // do WITH the ball.
                if p.carrier_contained_reset_ticks > 0 {
                    if raw_owner != Some(p.id) {
                        p.carrier_contained_reset_ticks = 0;
                    } else {
                        p.carrier_contained_reset_ticks -= 1;
                    }
                }
                // docs/positional-attack-phase-design.md, Stage 3.5
                // generalization (2026-09-11): the byline-specific block
                // below already recognizes "genuinely blocked, reset"
                // for a wide-committed carrier; this is the SAME
                // recognition for any OTHER ball carrier (a central
                // midfielder, a forward linking up, a winger who never
                // formally byline-committed) — Stage 3.5's own diagnosis
                // named this exact gap: "an attacker who runs into a
                // jockeying defender anywhere except the byline case
                // just keeps trying to carry forward into the same
                // defender, tick after tick... there's no deliberate
                // moment where the attacker recognizes he's stuck and
                // chooses to reset." Independent of, and never competing
                // with, the byline-specific arm just below: only
                // evaluated for players NOT in an active byline
                // commitment (that block ends in `continue`, so this
                // code is simply unreached for them), and both write the
                // same `carrier_contained_reset_ticks` field guarded by
                // "not already armed" — whichever fires first for a
                // given player on a given tick wins, harmlessly.
                // Reuses the exact same `single_blocker` geometry
                // (within 12u, genuinely ahead toward goal, y-aligned
                // within 6u) already measured and trusted for the
                // byline case — Pavel's own words: "just ONE genuinely
                // blocking the direct route forward" is real
                // containment, not a fabricated new threshold. The one
                // real addition is `carrier_containment_dwell_ticks`: a
                // genuine sustained-dwell requirement (~250ms) before
                // arming, since — unlike the byline case, which is
                // already narrowly scoped to a committed run — this
                // general population has no equivalent prior context
                // to rule out a single fleeting near-miss.
                if let Some(owner_id) = raw_owner {
                    if owner_id == p.id && p.byline_commitment_ticks == 0 {
                        let goal_x = match p.side {
                            Some(PlayerSide::Left) => context.field_size.width as f32,
                            _ => 0.0,
                        };
                        let to_goal_sign = (goal_x - p.position.x).signum();
                        let genuinely_blocked = to_goal_sign != 0.0
                            && opponent_positions.iter().any(|(team_id, pos)| {
                                *team_id != p.team_id
                                    && (*pos - p.position).magnitude() <= 12.0
                                    && (pos.x - p.position.x) * to_goal_sign > 0.0
                                    && (pos.y - p.position.y).abs() <= 6.0
                            });
                        if genuinely_blocked && p.carrier_contained_reset_ticks == 0 {
                            p.carrier_containment_dwell_ticks =
                                p.carrier_containment_dwell_ticks.saturating_add(1);
                            // TESTED and REJECTED 2026-09-11 (kept for the
                            // record): a GLOBAL instant-fire (threshold=1
                            // everywhere) was hypothesized to help by
                            // beating the defender's own tackle roll to
                            // the punch — measured, and it made both
                            // numbers worse (byline-reaching carries
                            // 8.04->5.32/match, recycle rate 5.0%->1.5%),
                            // because a defender merely being within 12u
                            // ANYWHERE during an otherwise-live run
                            // (not only near the byline) instantly
                            // aborted developing runs before they ever
                            // got there.
                            //
                            // FOLLOW-UP FIX, same day, per the corrected
                            // measurement (Follow-up #4): the global
                            // rejection doesn't mean 25 ticks is right
                            // EVERYWHERE — it means the two zones need
                            // different answers. Far from goal, a
                            // developing run needs the ~250ms tolerance
                            // to not be killed by a passing defender.
                            // Right at the byline, the carrier has
                            // nowhere further useful to go — waiting
                            // there only hands the very same close
                            // defender (within this engine's own
                            // TACKLE_ENGAGEMENT_DISTANCE=10.0, essentially
                            // the same radius as "genuinely_blocked"'s
                            // own 12u) repeated tackle rolls before
                            // containment is ever recognized. The
                            // measured 52.9% open-play byline-carry
                            // dispossession rate (vs. 31.4% recycled) is
                            // consistent with exactly this. React near-
                            // instantly once already in the danger zone;
                            // keep the slower, already-justified dwell
                            // everywhere else.
                            let depth_from_goal = (goal_x - p.position.x).abs();
                            const NEAR_BYLINE_DEPTH: f32 = 150.0;
                            let dwell_threshold: u16 = if depth_from_goal <= NEAR_BYLINE_DEPTH {
                                3
                            } else {
                                25
                            };
                            if p.carrier_containment_dwell_ticks >= dwell_threshold {
                                p.carrier_contained_reset_ticks = 90;
                                p.carrier_containment_dwell_ticks = 0;
                                match_log_info!(
                                    "CONTAINEDDIAG arm id={} general",
                                    p.id
                                );
                            }
                        } else {
                            p.carrier_containment_dwell_ticks = 0;
                        }
                    } else if p.byline_commitment_ticks == 0 {
                        p.carrier_containment_dwell_ticks = 0;
                    }
                } else {
                    p.carrier_containment_dwell_ticks = 0;
                }
                if p.byline_commitment_ticks > 0 {
                    if raw_owner != Some(p.id) {
                        // TEMP DIAG (2026-09-11, Follow-up #6 funnel trace):
                        // depth added to correlate against the ARM-time
                        // depth already logged above, to see whether loss
                        // is concentrated in deep-start runs or happens
                        // close to goal too. Remove once the funnel
                        // question is answered.
                        let goal_x_diag = match p.side {
                            Some(PlayerSide::Left) => context.field_size.width as f32,
                            _ => 0.0,
                        };
                        let depth_diag = (goal_x_diag - p.position.x).abs();
                        match_log_info!("CROSSDIAG LOSTBALL id={} ticks_left={} depth={:.1}", p.id, p.byline_commitment_ticks, depth_diag);
                        p.byline_commitment_ticks = 0;
                    } else {
                        // Stage 2 item 3: mirror the exact genuine-2-man-
                        // press condition ForwardDribblingState/
                        // MidfielderDribblingState themselves abort a
                        // byline-committed run on — `close_defenders >= 2`
                        // within 8u (the forward's own unconditional
                        // threshold; the midfielder's own
                        // `committing_to_byline` press_radius, same
                        // value). Armed here, one tick ahead of the state
                        // machine's own abort (which re-checks the
                        // identical live condition independently and will
                        // reach the same Passing transition this tick or
                        // next), so the very next pass decision already
                        // sees `carrier_contained_reset_ticks` set. A
                        // 1-tick lead/lag either way is harmless since the
                        // field decays over ~0.9s (90 ticks), not a single
                        // tick.
                        let close_defenders = opponent_positions
                            .iter()
                            .filter(|(team_id, _)| *team_id != p.team_id)
                            .filter(|(_, pos)| (*pos - p.position).magnitude() <= 8.0)
                            .count();
                        // docs/positional-attack-phase-design.md, Stage 2:
                        // Stage 0 measured this trigger firing ~3x below
                        // the sourced real-world "byline carry recycled
                        // without delivering" rate — but a genuine
                        // real-world recycle doesn't need two defenders
                        // converging, just ONE genuinely blocking the
                        // direct route forward. A purely geometric
                        // widening — deliberately does NOT touch the
                        // tackle/duel model itself (Part 3's guardrail;
                        // Stage 3.5's "contained dribble outcome" stays
                        // unbuilt/unscoped) — only changes when the
                        // already-shipped recycle reward arms.
                        let goal_x = match p.side {
                            Some(PlayerSide::Left) => context.field_size.width as f32,
                            _ => 0.0,
                        };
                        let to_goal_sign = (goal_x - p.position.x).signum();
                        let single_blocker = to_goal_sign != 0.0
                            && opponent_positions.iter().any(|(team_id, pos)| {
                                *team_id != p.team_id
                                    && (*pos - p.position).magnitude() <= 12.0
                                    && (pos.x - p.position.x) * to_goal_sign > 0.0
                                    && (pos.y - p.position.y).abs() <= 6.0
                            });
                        // A real bug, not just a measurement artifact:
                        // this used to re-arm (and re-log) on EVERY
                        // qualifying tick, not once per genuine episode
                        // — while a defender happened to stay in range
                        // for several consecutive ticks, the "same"
                        // decision moment was counted (and re-armed to
                        // its full 90-tick window) repeatedly. Gating on
                        // "not already armed" makes each arm a genuine,
                        // discrete episode — comparable to Stage 0's
                        // sourced real-world metric, which counts
                        // discrete carry-endings, not per-tick samples.
                        if p.carrier_contained_reset_ticks == 0
                            && (close_defenders >= 2 || single_blocker)
                        {
                            p.carrier_contained_reset_ticks = 90;
                            match_log_info!(
                                "BLOCKEDRESETDIAG arm id={} close_defenders={} single_blocker={}",
                                p.id,
                                close_defenders,
                                single_blocker
                            );
                        }
                        p.byline_commitment_ticks -= 1;
                        if p.byline_commitment_ticks == 0 {
                            match_log_info!("CROSSDIAG TIMEOUT id={}", p.id);
                        }
                    }
                    continue;
                }
                // Chunk 5 (docs/zone-staged-possession-progression.md):
                // a genuine "meeting a defender" contact event, already
                // classified as WideIsolation by
                // TeamTacticalState::refresh_positional_attack
                // (contact-gated, not continuous — see that function's
                // own doc comment), commits the CARRIER to trying to
                // beat his marker and reach the byline. Deliberately a
                // SEPARATE branch from the ambient arm gate below, not
                // an extra condition on it: that gate requires an EMPTY
                // flank (`has_room`, zero opponents within 9u) and can
                // therefore structurally never fire for a genuine 1v1
                // isolation — there IS a defender there, that's the
                // whole point of the pattern. Reuses the same downstream
                // field/mechanism (`byline_commitment_ticks`), not a
                // second, competing one.
                // Real, documented risk, guarded explicitly rather than
                // assumed harmless: the reverted rewrite's own fatal
                // regression was a corner-taker dribbling from the flag
                // instead of delivering. A corner/free-kick/throw-in/
                // penalty taker IS the ball owner and often IS in the
                // wide channel at the exact moment this branch would
                // otherwise check — `PassOriginRestart::OpenPlay` is
                // the same field `is_team_attacking_corner()` itself
                // reads (`common/ball.rs`), so this arm can only ever
                // fire during genuine open-play advancing, never during
                // any restart's own dedicated, already-tuned delivery
                // logic.
                if field.ball.pass_origin_restart == crate::r#match::PassOriginRestart::OpenPlay {
                if let Some(owner_id) = raw_owner {
                    if owner_id == p.id {
                        let team_tactical = if p.team_id == context.field_home_team_id {
                            &context.tactical_home
                        } else {
                            &context.tactical_away
                        };
                        if team_tactical.positional_attack_active()
                            && team_tactical.positional_attack_pattern
                                == Some(crate::r#match::AttackPattern::WideIsolation)
                        {
                            let group = p.tactical_position.current_position.position_group();
                            let h = context.field_size.height as f32;
                            let y = p.position.y;
                            let wide = y < h * 0.38 || y > h * 0.62;
                            if wide
                                && matches!(
                                    group,
                                    crate::PlayerFieldPositionGroup::Forward
                                        | crate::PlayerFieldPositionGroup::Midfielder
                                )
                            {
                                // Deliberately more modest than the
                                // ambient gate's 0.88-1.0 (a genuine 1v1
                                // against a real, present marker is
                                // harder to commit to than an empty
                                // flank) — reasoned estimate, same
                                // sourcing status as every other
                                // unsourced Zone-1/Chunk-5 constant in
                                // this codebase, flagged as such rather
                                // than invented silently.
                                let pace01 =
                                    ((p.skills.physical.pace - 1.0) / 19.0).clamp(0.0, 1.0);
                                let dribble01 =
                                    ((p.skills.technical.dribbling - 1.0) / 19.0).clamp(0.0, 1.0);
                                let prob = (0.45 + pace01 * 0.15 + dribble01 * 0.15)
                                    .clamp(0.45, 0.75);
                                if context.rng.range_u64(0, 10000) < (prob * 10000.0) as u64 {
                                    p.byline_commitment_ticks = 1200;
                                    match_log_info!(
                                        "STAGE0ARM kind=contact id={} team={} tick={} poss_ticks={}",
                                        p.id,
                                        p.team_id,
                                        context.current_tick(),
                                        team_tactical.possession_ticks
                                    );
                                    // Verification-only field, temporary:
                                    // confirms this arm is firing under
                                    // genuine defender proximity (the
                                    // whole point of the pattern), not
                                    // coincidentally an already-empty
                                    // flank the ambient gate below would
                                    // have covered anyway.
                                    let nearest_opp = opponent_positions
                                        .iter()
                                        .filter(|(team_id, _)| *team_id != p.team_id)
                                        .map(|(_, pos)| (*pos - p.position).magnitude())
                                        .fold(f32::MAX, f32::min);
                                    match_log_info!(
                                        "CHUNK5DIAG wide_isolation_arm id={} group={:?} y={:.1} nearest_opp={:.1}",
                                        p.id,
                                        group,
                                        y,
                                        nearest_opp
                                    );
                                    continue;
                                }
                            }
                        }
                        // Chunk 5b (docs/zone-staged-possession-
                        // progression.md): the give-and-go, sibling to
                        // the WideIsolation branch above — same contact
                        // trigger, different pattern, different action.
                        // "There is a lot of wide open space behind
                        // him... play a 1-2" (Pavel, 2026-09-02): the
                        // carrier lays it off to a genuinely close
                        // teammate (`combo_partner`, biases HIS OWN
                        // pass preference — evaluator.rs) then commits
                        // to a run into the space he was trying to
                        // exploit (`give_and_go_run_ticks`, consumed by
                        // `give_and_go_run_velocity` in processor.rs).
                        // Deliberately does not try to arm a symmetric
                        // "look for the return" bias on the teammate —
                        // that would need a second, simultaneous
                        // mutable borrow of another player from inside
                        // this same `iter_mut()` loop, which this
                        // codebase has no established safe pattern for
                        // yet; the run itself making him genuinely open
                        // is trusted to do that job instead (same
                        // "lean on already-verified machinery" call as
                        // Chunk 4's own off-ball reaction).
                        // realism-bug (caught via CHUNK5BDIAG raw data
                        // before shipping, same session): without the
                        // `combo_partner_ticks == 0` gate, this rolled
                        // fresh every tick the conditions kept holding
                        // — 665 arms across 15 matches, many for the
                        // SAME player on consecutive ticks — compounding
                        // a per-tick probability into near-certainty and
                        // continuously overwriting the supposedly-frozen
                        // run target, the exact class of bug
                        // `gk1v1_decision_made`'s own one-shot-per-
                        // possession lock exists to prevent elsewhere in
                        // this codebase.
                        if team_tactical.positional_attack_active()
                            && team_tactical.positional_attack_pattern
                                == Some(
                                    crate::r#match::AttackPattern::UnderlapOverlapCombination,
                                )
                            && p.combo_partner_ticks == 0
                        {
                            let group = p.tactical_position.current_position.position_group();
                            if matches!(
                                group,
                                crate::PlayerFieldPositionGroup::Forward
                                    | crate::PlayerFieldPositionGroup::Midfielder
                            ) {
                                let attacking_dir: f32 = match p.side {
                                    Some(crate::r#match::PlayerSide::Left) => 1.0,
                                    Some(crate::r#match::PlayerSide::Right) => -1.0,
                                    None => 0.0,
                                };
                                if attacking_dir != 0.0 {
                                    // A real lay-off is a few metres,
                                    // not a long diagonal switch —
                                    // deliberately much tighter than
                                    // Chunk 3's own
                                    // ZONE1_TEAMMATE_SEARCH_RADIUS
                                    // (200u). Reasoned estimate, not
                                    // sourced (no dataset exists for
                                    // "how far is a real give-and-go
                                    // lay-off").
                                    const COMBO_PARTNER_MAX_DIST: f32 = 70.0;
                                    let nearest_teammate = all_player_snapshot
                                        .iter()
                                        .filter(|&&(qid, qteam, _, qsent_off)| {
                                            qteam == p.team_id && qid != p.id && !qsent_off
                                        })
                                        .map(|&(qid, _, qpos, _)| {
                                            (qid, (qpos - p.position).magnitude())
                                        })
                                        .filter(|&(_, d)| d <= COMBO_PARTNER_MAX_DIST)
                                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                                    if let Some((partner_id, partner_dist)) = nearest_teammate {
                                        // Same reasoned-estimate status
                                        // as WideIsolation's own
                                        // probability above — this is a
                                        // passing+movement combination,
                                        // so scaled by passing/off-the-
                                        // ball skills instead of pace/
                                        // dribbling.
                                        let passing01 = ((p.skills.technical.passing - 1.0)
                                            / 19.0)
                                            .clamp(0.0, 1.0);
                                        let off_ball01 = ((p.skills.mental.off_the_ball
                                            - 1.0)
                                            / 19.0)
                                            .clamp(0.0, 1.0);
                                        let prob = (0.45
                                            + passing01 * 0.15
                                            + off_ball01 * 0.15)
                                            .clamp(0.45, 0.75);
                                        if context.rng.range_u64(0, 10000)
                                            < (prob * 10000.0) as u64
                                        {
                                            const COMBO_BIAS_TICKS: u16 = 300;
                                            const RUN_TICKS: u16 = 500;
                                            const RUN_FORWARD_DISTANCE: f32 = 70.0;
                                            const RUN_LATERAL_PULL: f32 = 0.35;
                                            p.combo_partner = Some(partner_id);
                                            p.combo_partner_ticks = COMBO_BIAS_TICKS;
                                            let center_y =
                                                context.field_size.height as f32 / 2.0;
                                            let field_w = context.field_size.width as f32;
                                            p.give_and_go_run_target_x = (p.position.x
                                                + attacking_dir * RUN_FORWARD_DISTANCE)
                                                .clamp(0.0, field_w);
                                            p.give_and_go_run_target_y = p.position.y
                                                + (center_y - p.position.y)
                                                    * RUN_LATERAL_PULL;
                                            p.give_and_go_run_ticks = RUN_TICKS;
                                            match_log_info!(
                                                "CHUNK5BDIAG combo_arm id={} partner={} partner_dist={:.1} target=({:.1},{:.1})",
                                                p.id,
                                                partner_id,
                                                partner_dist,
                                                p.give_and_go_run_target_x,
                                                p.give_and_go_run_target_y
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        // OverloadAndSwitch (docs/zone-staged-possession-
                        // progression.md): third sibling pattern — same
                        // contact trigger, `side_density` 4+ on the
                        // ball's own flank (a genuine overload of OUR
                        // OWN players there), real football response is
                        // to switch play to the underloaded far side,
                        // not force a combination through the crowd.
                        // Reuses `combo_partner`/`combo_partner_ticks`
                        // (Chunk 5b's own fields, not new plumbing —
                        // mutually exclusive with UnderlapOverlapCombination
                        // since only one AttackPattern is active per
                        // team at a time) to bias the carrier's OWN pass
                        // preference toward a genuine weak-side
                        // teammate. `evaluator.rs`'s own
                        // `underload_switch_bonus`/`overload_bonus`
                        // (Milestone 7/8, already shipped) already
                        // REWARD a real switch once a candidate is
                        // considered — this only makes sure a genuinely
                        // far switch target actually enters
                        // consideration at all (see
                        // `find_best_pass_option`'s own widened
                        // `max_distance` for `combo_partner`-armed
                        // players). Deliberately does NOT set
                        // `give_and_go_run_*` — a real switch of play
                        // doesn't have the passer follow the ball into
                        // space the way a short lay-off/1-2 does; the
                        // target is picked once, the bias alone does
                        // the rest via already-correct scoring.
                        if team_tactical.positional_attack_active()
                            && team_tactical.positional_attack_pattern
                                == Some(crate::r#match::AttackPattern::OverloadAndSwitch)
                            && p.combo_partner_ticks == 0
                        {
                            let group = p.tactical_position.current_position.position_group();
                            if matches!(
                                group,
                                crate::PlayerFieldPositionGroup::Forward
                                    | crate::PlayerFieldPositionGroup::Midfielder
                            ) {
                                let field_h = context.field_size.height as f32;
                                let carrier_side =
                                    crate::r#match::BallSideZone::for_y(field_h, p.position.y);
                                // realism-bug (caught via OASDIAG raw
                                // data before shipping): the first cut
                                // required only `!= carrier_side`, which
                                // a goalkeeper — almost always sitting
                                // dead-center on the goal line
                                // (`BallSideZone::Center`), unmarked —
                                // trivially satisfies every time. 6/6
                                // sampled arms picked the GK outright.
                                // The genuine football pattern is
                                // flank-to-flank, not "anywhere that
                                // isn't my own zone" — require the
                                // OPPOSITE flank specifically, which also
                                // structurally excludes Center (and
                                // hence the keeper) without needing a
                                // position-group check this raw
                                // per-tick loop doesn't otherwise have.
                                let Some(target_side) = (match carrier_side {
                                    crate::r#match::BallSideZone::Left => {
                                        Some(crate::r#match::BallSideZone::Right)
                                    }
                                    crate::r#match::BallSideZone::Right => {
                                        Some(crate::r#match::BallSideZone::Left)
                                    }
                                    // Unreachable in practice — the arm
                                    // gate (`refresh_positional_attack`)
                                    // already requires a non-Center
                                    // `ball_side` for the whole window —
                                    // but the carrier can drift slightly
                                    // between classification and this
                                    // tick, so handled explicitly rather
                                    // than assumed impossible.
                                    crate::r#match::BallSideZone::Center => None,
                                }) else {
                                    continue;
                                };
                                // Kept in sync with evaluator.rs's own
                                // OVERLOAD_SWITCH_MAX_DISTANCE — both
                                // answer the same "can this pass
                                // realistically reach" question.
                                const SWITCH_SEARCH_MAX_DIST: f32 = 450.0;
                                // "Free man on the far side" proxy — same
                                // order of magnitude as
                                // TeamTacticalState::COVER_CHECK_RADIUS
                                // (45.0, teamplay/tactical.rs), hardcoded
                                // locally rather than imported across
                                // modules since this loop only has raw
                                // field data, not a
                                // StateProcessingContext.
                                const NEARBY_OPPONENT_RADIUS: f32 = 45.0;
                                let target = all_player_snapshot
                                    .iter()
                                    .filter(|&&(qid, qteam, _, qsent_off)| {
                                        qteam == p.team_id && qid != p.id && !qsent_off
                                    })
                                    .map(|&(qid, _, qpos, _)| (qid, qpos))
                                    .filter(|&(_, qpos)| {
                                        let d = (qpos - p.position).magnitude();
                                        d > 1.0
                                            && d <= SWITCH_SEARCH_MAX_DIST
                                            && crate::r#match::BallSideZone::for_y(field_h, qpos.y)
                                                == target_side
                                    })
                                    .map(|(qid, qpos)| {
                                        let nearby_opp = opponent_positions
                                            .iter()
                                            .filter(|(oteam, opos)| {
                                                *oteam != p.team_id
                                                    && (*opos - qpos).magnitude()
                                                        <= NEARBY_OPPONENT_RADIUS
                                            })
                                            .count();
                                        (qid, nearby_opp)
                                    })
                                    .min_by_key(|&(_, nearby_opp)| nearby_opp);
                                if let Some((partner_id, nearby_opp)) = target {
                                    // Vision/passing-scaled — a genuine
                                    // long diagonal switch is a vision/
                                    // technique play, distinct from
                                    // WideIsolation's pace/dribbling and
                                    // UnderlapOverlapCombination's
                                    // passing/off-the-ball scaling. Same
                                    // reasoned-estimate status as every
                                    // other Chunk-5-family probability
                                    // constant.
                                    let vision01 =
                                        ((p.skills.mental.vision - 1.0) / 19.0).clamp(0.0, 1.0);
                                    let passing01 = ((p.skills.technical.passing - 1.0) / 19.0)
                                        .clamp(0.0, 1.0);
                                    let prob = (0.40 + vision01 * 0.20 + passing01 * 0.15)
                                        .clamp(0.40, 0.75);
                                    if context.rng.range_u64(0, 10000) < (prob * 10000.0) as u64 {
                                        const SWITCH_BIAS_TICKS: u16 = 300;
                                        let partner_pos = all_player_snapshot
                                            .iter()
                                            .find(|&&(qid, _, _, _)| qid == partner_id)
                                            .map(|&(_, _, qpos, _)| qpos);
                                        let dist = partner_pos
                                            .map(|qp| (qp - p.position).magnitude())
                                            .unwrap_or(-1.0);
                                        p.combo_partner = Some(partner_id);
                                        p.combo_partner_ticks = SWITCH_BIAS_TICKS;
                                        match_log_info!(
                                            "OASDIAG switch_arm id={} carrier_y={:.1} carrier_side={:?} partner={} partner_y={:.1} target_side={:?} dist={:.1} nearby_opp={}",
                                            p.id,
                                            p.position.y,
                                            carrier_side,
                                            partner_id,
                                            partner_pos.map(|qp| qp.y).unwrap_or(-1.0),
                                            target_side,
                                            dist,
                                            nearby_opp
                                        );
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
                }
                // docs/positional-attack-phase-design.md, Stage 3: the
                // forward drop-and-spin primitive. A Forward who
                // genuinely drops deep (out of his own attacking half
                // entirely — Pavel's own framing, "run back deeper to
                // the box," implies he really went missing from the
                // front line) to receive/link up during a genuine
                // settled positional-attack phase (Stage 1's detector)
                // commits to a run back into the box once he releases
                // the ball. Armed once per qualifying touch (the
                // `drop_and_spin_run_ticks == 0` guard, same debounce
                // idiom as Stage 2's widening above). `prob=0.5` is a
                // reasoned estimate, not sourced — no dataset gives a
                // real "does this striker spin a run" rate — flagged
                // as such rather than invented silently.
                if raw_owner == Some(p.id) && p.drop_and_spin_run_ticks == 0 {
                    let group = p.tactical_position.current_position.position_group();
                    if matches!(group, crate::PlayerFieldPositionGroup::Forward) {
                        let team_tactical = if p.team_id == context.field_home_team_id {
                            &context.tactical_home
                        } else {
                            &context.tactical_away
                        };
                        if team_tactical.patient_attack_phase_active() {
                            let goal_x = match p.side {
                                Some(PlayerSide::Left) => context.field_size.width as f32,
                                _ => 0.0,
                            };
                            let depth_from_goal = (goal_x - p.position.x).abs();
                            let h = context.field_size.height as f32;
                            // 0.42, not 0.5 (full own half) — real
                            // "drop deep to link up" more often means
                            // dropping into the middle third than all
                            // the way back to one's own half. Widened
                            // from an initial 0.5 after a 20-match
                            // measurement found only 3 arms (~0.15/
                            // match) — too rare to say anything about
                            // the pattern at all, not a case of chasing
                            // a target number.
                            if depth_from_goal > context.field_size.width as f32 * 0.42 {
                                let prob = 0.5;
                                if context.rng.range_u64(0, 10000) < (prob * 10000.0) as u64 {
                                    // ~60u (~7.5m) inside the goal line —
                                    // a near-six-yard-box run target, a
                                    // reasoned estimate like every other
                                    // unsourced radius/depth constant in
                                    // this codebase's off-ball runs.
                                    p.drop_and_spin_target_x = match p.side {
                                        Some(PlayerSide::Left) => goal_x - 60.0,
                                        _ => goal_x + 60.0,
                                    };
                                    // Pull halfway toward the central
                                    // channel from wherever he actually
                                    // dropped from — a striker's real
                                    // in-behind run targets the box, not
                                    // a literal straight-back-out replay
                                    // of an incidental wide drop.
                                    p.drop_and_spin_target_y =
                                        h * 0.5 + (p.position.y - h * 0.5) * 0.5;
                                    p.drop_and_spin_run_ticks = 500;
                                    match_log_info!(
                                        "STAGE3ARM id={} team={} depth={:.1} target_x={:.1} target_y={:.1}",
                                        p.id,
                                        p.team_id,
                                        depth_from_goal,
                                        p.drop_and_spin_target_x,
                                        p.drop_and_spin_target_y
                                    );
                                }
                            }
                        }
                    }
                }
                // realism-bug: arming at ownership_duration<=3 (within
                // 30ms of reception) meant committing before the ball is
                // genuinely settled — this engine's own shot logic treats
                // <300ms possession as "unsettled" for the same reason
                // (§12.6-adjacent has_settled gate). Measured many
                // commitments dying within 10 ticks regardless of arm-
                // rate/press-radius tuning, consistent with arming during
                // a volatile just-received moment rather than after the
                // carrier has actually secured it. Widened to 20-150
                // ticks (200ms-1.5s) — settled, but still an early
                // decision point in the possession.
                //
                // docs/positional-play-master-design.md, Follow-up #6
                // (2026-09-11, target: >=8 open-play byline-reaching
                // carries/match): the 150-tick (1.5s) ceiling meant a
                // real winger could only ever decide to commit to a
                // byline run in the first 1.5 seconds after FIRST
                // receiving the ball — a real player carrying/dribbling
                // for several seconds, finding themselves wide and open
                // LATER in that same unbroken possession, had no path to
                // commit at all. Real wingers commit to a run whenever
                // they find the right moment, not only in the opening
                // instant. Widened to 20-600 ticks (~6s) — still requires
                // the SAME player to have held the ball continuously
                // (a genuine, ongoing carry, not a stale flag), just no
                // longer artificially cuts that window down to 1.5s.
                if raw_owner != Some(p.id)
                    || !(20..600).contains(&field.ball.ownership_duration)
                {
                    continue;
                }
                let group = p.tactical_position.current_position.position_group();
                // docs/positional-play-master-design.md, Follow-up #6
                // (target: >=8/match): a real overlapping fullback/
                // wingback also makes this exact run in real football —
                // excluding Defender entirely meant this whole mechanism
                // was structurally invisible to roughly a third of the
                // outfield roster.
                if !matches!(
                    group,
                    crate::PlayerFieldPositionGroup::Forward
                        | crate::PlayerFieldPositionGroup::Midfielder
                        | crate::PlayerFieldPositionGroup::Defender
                ) {
                    continue;
                }
                let h = context.field_size.height as f32;
                let y = p.position.y;
                let wide = y < h * 0.38 || y > h * 0.62;
                if !wide {
                    continue;
                }
                let goal_x = match p.side {
                    Some(PlayerSide::Left) => context.field_size.width as f32,
                    _ => 0.0,
                };
                let depth = (goal_x - p.position.x).abs();
                // docs/positional-play-master-design.md, Follow-up #6
                // continued (2026-09-11): 100-450 was tried and directly
                // measured with a paired arm->outcome funnel trace (20
                // matches, depth logged at both arm and loss moments,
                // matched per-player in event order). Result: arms
                // starting at depth 100-250 reach the byline 31.5% of the
                // time (mean progress 41.8u before losing the ball) —
                // genuinely productive. Arms starting at 250-350 reach it
                // 0.0% of the time (mean progress only 22.0u — nowhere
                // near enough to matter), and 350-450 reach it 0.9% of
                // the time (mean progress 12.6u — essentially dying on
                // the spot). Over half of all arms (227/424 paired
                // episodes) were landing in these two unproductive
                // buckets, contributing volume to LOSTBALL but ~0 to the
                // actual target metric — a real, measured net negative
                // from the earlier widening, not a neutral no-op.
                // Narrowed to 100-280 (a small margin past the observed
                // 250 cliff, not exactly at it) to stop arming runs that
                // structurally cannot complete at realistic dribble
                // speed against this engine's own defensive-recovery
                // rate, while keeping the low-end widening (130->100),
                // which was never implicated by this measurement.
                if !(100.0..280.0).contains(&depth) {
                    continue;
                }
                // docs/positional-play-master-design.md, Follow-up #6
                // continued (2026-09-11): an isotropic "nobody within Xu"
                // check was ALWAYS the wrong shape for "has room to run"
                // — measured directly (temporary PREROOM diagnostic,
                // logged every tick a player was already wide + right
                // depth + settled: 133.7 such ticks/match, but only
                // 10.5% had an isotropic 5u radius clear). The underlying
                // situation (wide, right depth, settled possession) is
                // NOT rare; an isotropic radius check was discarding
                // ~90% of it, including every case where the nearby
                // opponent is BEHIND the carrier (already beaten — no
                // threat to a forward run at all) or directly alongside
                // (normal 1v1 defending a real winger runs at anyway).
                // Follow-up: a first cut reused the containment check's
                // own 12u ahead-radius verbatim — measured (30-match
                // funnel) to be a NET WASH vs. the old isotropic 5u
                // (arms 13.07->12.23/match, reached-byline 3.40->3.30):
                // most nearby opponents in this situation ARE ahead
                // (defenders sit goal-side by design), so exempting
                // behind/lateral cases was offset by widening the
                // ahead-radius from 5u to 12u for the dominant case.
                // Kept the directional exemption (behind/lateral no
                // longer disqualify at all — a real, defensible change
                // on its own) but shrunk the ahead-radius to 6u, smaller
                // than the isotropic 5u it replaces in effective area
                // once direction is accounted for, so this is now a
                // strict loosening in every direction rather than a
                // wash. has_room is still "not already genuinely
                // blocked," just tuned rather than borrowing the
                // containment check's own already-tuned distance
                // unverified.
                let to_goal_sign = (goal_x - p.position.x).signum();
                let has_room = to_goal_sign == 0.0
                    || !opponent_positions.iter().any(|(team_id, pos)| {
                        *team_id != p.team_id
                            && (*pos - p.position).magnitude() <= 6.0
                            && (pos.x - p.position.x) * to_goal_sign > 0.0
                            && (pos.y - p.position.y).abs() <= 6.0
                    });
                if !has_room {
                    continue;
                }
                let pace01 = ((p.skills.physical.pace - 1.0) / 19.0).clamp(0.0, 1.0);
                let dribble01 = ((p.skills.technical.dribbling - 1.0) / 19.0).clamp(0.0, 1.0);
                let prob = (0.88 + pace01 * 0.08 + dribble01 * 0.08).clamp(0.88, 1.0);
                if context.rng.range_u64(0, 10000) < (prob * 10000.0) as u64 {
                    // 1200 ticks (12s) — realistic dribble speed is
                    // ~0.4-0.6u/tick, so covering the full 380u worst-
                    // case trigger distance needs real time; the state-
                    // level timeouts are skipped while committed (see
                    // the dribbling states), so this is the actual bound.
                    p.byline_commitment_ticks = 1200;
                    match_log_info!(
                        "CROSSDIAG ARM id={} group={:?} y={:.1} depth={:.1}",
                        p.id,
                        group,
                        y,
                        depth
                    );
                    // docs/positional-attack-phase-design.md, Stage 0:
                    // does the byline probe cycle genuinely repeat
                    // within one continuous team possession, or is it
                    // structurally one-shot? `poss_ticks` lets a
                    // post-hoc script group arms by possession streak
                    // (a streak boundary is any tick where the team's
                    // own possession_ticks resets to a value smaller
                    // than the previous sample).
                    match_log_info!(
                        "STAGE0ARM kind=ambient id={} team={} tick={} poss_ticks={}",
                        p.id,
                        p.team_id,
                        context.current_tick(),
                        if p.team_id == context.field_home_team_id {
                            context.tactical_home.possession_ticks
                        } else {
                            context.tactical_away.possession_ticks
                        }
                    );
                }
            }

            // realism-bug (2026-08-27): persistent wide-run tracking
            // assignment. `processor.rs::wide_run_track_override_velocity`
            // does the actual per-tick STEERING, but without persistence
            // the "who's the single closest defender" recomputation
            // reassigns the tracker almost every tick as relative
            // distances jitter slightly, so no one defender ever commits
            // to a sustained chase (measured: a real, non-trivial but
            // modest effect on defensive presence in the byline-approach
            // zone before this fix). Mirrors byline_commitment_ticks'
            // own snapshot-then-mutate structure exactly — can't borrow
            // field.players immutably while iterating it mutably.
            let wrt_snapshot: Vec<(u32, u32, Vector3<f32>, Vector3<f32>, crate::PlayerFieldPositionGroup)> =
                field
                    .players
                    .iter()
                    .map(|p| {
                        (
                            p.id,
                            p.team_id,
                            p.position,
                            p.velocity,
                            p.tactical_position.current_position.position_group(),
                        )
                    })
                    .collect();
            const WRT_REFRESH_TICKS: u16 = 90; // ~0.9s — refreshed every
            // qualifying tick, so a genuine sustained run keeps the same
            // defender committed; one that stops decays within under a
            // second with nothing else needed to clear it.
            const WRT_ENGAGEMENT_DISTANCE: f32 = 20.0; // matches defensive.rs's ENGAGEMENT_DISTANCE
            let wrt_field_h = context.field_size.height as f32;
            let wrt_field_w = context.field_size.width as f32;
            let wrt_is_dangerous_run = |defender_pos: Vector3<f32>,
                                        defender_side: Option<PlayerSide>,
                                        opp_pos: Vector3<f32>,
                                        opp_vel: Vector3<f32>| {
                let wide = opp_pos.y < wrt_field_h * 0.35 || opp_pos.y > wrt_field_h * 0.65;
                if !wide {
                    return false;
                }
                let own_goal_x = match defender_side {
                    Some(PlayerSide::Left) => 0.0,
                    Some(PlayerSide::Right) => wrt_field_w,
                    None => 0.0,
                };
                let depth = (own_goal_x - opp_pos.x).abs();
                if !(20.0..420.0).contains(&depth) {
                    return false;
                }
                if opp_vel.norm() < 0.30 {
                    return false;
                }
                let toward_goal_x = if own_goal_x < wrt_field_w / 2.0 {
                    opp_vel.x < -0.1
                } else {
                    opp_vel.x > 0.1
                };
                if !toward_goal_x {
                    return false;
                }
                let _ = defender_pos;
                true
            };

            for p in field.players.iter_mut() {
                if let Some(target_id) = p.wide_run_track_target {
                    let still_valid = wrt_snapshot
                        .iter()
                        .find(|(id, team_id, _, _, _)| *id == target_id && *team_id != p.team_id)
                        .map(|(_, _, pos, vel, _)| wrt_is_dangerous_run(p.position, p.side, *pos, *vel))
                        .unwrap_or(false);
                    if still_valid {
                        p.wide_run_track_ticks = WRT_REFRESH_TICKS;
                        continue;
                    }
                    if p.wide_run_track_ticks > 0 {
                        p.wide_run_track_ticks -= 1;
                        continue;
                    }
                    p.wide_run_track_target = None;
                }

                if p.tactical_position.current_position.position_group()
                    != crate::PlayerFieldPositionGroup::Defender
                {
                    continue;
                }

                let mut best_id: Option<u32> = None;
                let mut best_dist = f32::MAX;
                for (opp_id, opp_team, opp_pos, opp_vel, _grp) in &wrt_snapshot {
                    if *opp_team == p.team_id {
                        continue;
                    }
                    if !wrt_is_dangerous_run(p.position, p.side, *opp_pos, *opp_vel) {
                        continue;
                    }
                    // Already engaged by another own-team defender/
                    // midfielder? Mirrors defensive.rs's
                    // is_opponent_being_engaged (raw snapshot version —
                    // that helper needs a full StateProcessingContext,
                    // not available in this raw per-tick loop).
                    let already_engaged = wrt_snapshot.iter().any(|(tid, tteam, tpos, _, tgrp)| {
                        *tid != p.id
                            && *tteam == p.team_id
                            && matches!(
                                tgrp,
                                crate::PlayerFieldPositionGroup::Defender
                                    | crate::PlayerFieldPositionGroup::Midfielder
                            )
                            && (*tpos - *opp_pos).magnitude() < WRT_ENGAGEMENT_DISTANCE
                    });
                    if already_engaged {
                        continue;
                    }
                    let d = (*opp_pos - p.position).magnitude();
                    if d < best_dist {
                        best_dist = d;
                        best_id = Some(*opp_id);
                    }
                }
                let Some(runner_id) = best_id else { continue };
                let runner_pos = wrt_snapshot
                    .iter()
                    .find(|(id, ..)| *id == runner_id)
                    .map(|(_, _, pos, ..)| *pos)
                    .unwrap();

                // Only the single nearest own-team defender tracks this
                // runner — otherwise multiple defenders converge on the
                // same man and the rest of the shape abandons its own
                // responsibilities.
                let am_closest = wrt_snapshot.iter().all(|(tid, tteam, tpos, _, tgrp)| {
                    *tid == p.id
                        || *tteam != p.team_id
                        || *tgrp != crate::PlayerFieldPositionGroup::Defender
                        || (*tpos - runner_pos).magnitude() >= best_dist
                });
                if !am_closest {
                    continue;
                }
                p.wide_run_track_target = Some(runner_id);
                p.wide_run_track_ticks = WRT_REFRESH_TICKS;
            }

            // realism-bug (2026-07-27): occasional "round the keeper"
            // decision for a genuine clean 1v1 vs. the GK — see
            // round_keeper_commitment_ticks' doc comment (player.rs).
            // One-shot-per-possession via gk1v1_decision_made: the roll
            // only happens once per continuous spell of ball ownership,
            // not every tick the clean-1v1 condition holds (which would
            // otherwise compound a small per-tick probability into
            // near-certainty over a multi-second breakaway).
            let gk_positions: Vec<(u32, PlayerSide, Vector3<f32>)> = field
                .players
                .iter()
                .filter(|p| {
                    matches!(
                        p.tactical_position.current_position.position_group(),
                        crate::PlayerFieldPositionGroup::Goalkeeper
                    )
                })
                .map(|p| (p.team_id, p.side.unwrap_or(PlayerSide::Left), p.position))
                .collect();
            for p in field.players.iter_mut() {
                if p.round_keeper_commitment_ticks > 0 {
                    if raw_owner != Some(p.id) {
                        p.round_keeper_commitment_ticks = 0;
                    } else {
                        p.round_keeper_commitment_ticks -= 1;
                    }
                    continue;
                }
                if raw_owner != Some(p.id) {
                    p.gk1v1_decision_made = false;
                    continue;
                }
                if p.gk1v1_decision_made {
                    continue;
                }
                if !matches!(
                    p.tactical_position.current_position.position_group(),
                    crate::PlayerFieldPositionGroup::Forward
                ) {
                    continue;
                }
                let Some((_, gk_side, gk_pos)) =
                    gk_positions.iter().find(|(team_id, _, _)| *team_id != p.team_id)
                else {
                    continue;
                };
                // No other outfield opponent within 40u -- genuinely
                // clean, matching the same radius the diagnostic
                // measurement used to define a "clean GK 1v1" baseline.
                let clean = !opponent_positions.iter().any(|(team_id, pos)| {
                    *team_id != p.team_id && (*pos - p.position).magnitude() < 40.0
                });
                if !clean {
                    continue;
                }
                let goal_x = match p.side {
                    Some(PlayerSide::Left) => context.field_size.width as f32,
                    _ => 0.0,
                };
                let dist_to_goal = (goal_x - p.position.x).abs();
                if !(40.0..220.0).contains(&dist_to_goal) {
                    continue;
                }
                // Keeper genuinely advanced off his own line -- real
                // coaching doctrine sourced during this investigation:
                // rounding the keeper is only viable when he's "several
                // yards outside the six-yard box" (totalfootballanalysis.
                // com). 44u = this engine's own six-yard-box-equivalent
                // depth (teamplay/zones.rs, already established elsewhere).
                let gk_own_goal_x = match gk_side {
                    PlayerSide::Left => 0.0,
                    PlayerSide::Right => context.field_size.width as f32,
                };
                let gk_off_line = (gk_pos.x - gk_own_goal_x).abs();
                if gk_off_line < 44.0 {
                    continue;
                }
                p.gk1v1_decision_made = true;
                // Rare by design -- real one-on-ones are overwhelmingly
                // finished with a direct/placed shot, not a dribble round
                // the keeper (sourced StatsBomb Bundesliga 2023/24 pull,
                // this investigation: only 1/23 foot-based one-on-one
                // shots used a Lob technique, the closest real-world
                // proxy). No public dataset isolates "round the keeper"
                // attempt rate specifically -- flagged estimate (category
                // d), skill-scaled.
                let composure01 = ((p.skills.mental.composure - 1.0) / 19.0).clamp(0.0, 1.0);
                let dribble01 = ((p.skills.technical.dribbling - 1.0) / 19.0).clamp(0.0, 1.0);
                let flair01 = ((p.skills.mental.flair - 1.0) / 19.0).clamp(0.0, 1.0);
                let prob =
                    (0.05 + composure01 * 0.05 + dribble01 * 0.05 + flair01 * 0.05).clamp(0.05, 0.20);
                let roll = context.rng.range_u64(0, 10000);
                if roll < (prob * 10000.0) as u64 {
                    // realism-bug follow-up: a raw tick trace showed the
                    // GK retreating toward his own goal in the SAME
                    // direction as the attacker for the whole chase --
                    // continuously re-targeting his live position meant
                    // the effective closing rate was only the difference
                    // of their speeds and the gap never closed. Freeze
                    // the target now, at decision time, exactly like a
                    // real striker committing to a route.
                    let goal_pos = Vector3::new(
                        goal_x,
                        context.field_size.height as f32 / 2.0,
                        0.0,
                    );
                    let to_goal = (goal_pos - p.position).normalize();
                    let lateral = Vector3::new(-to_goal.y, to_goal.x, 0.0);
                    let side = if p.position.y > gk_pos.y { -1.0 } else { 1.0 };
                    let target = gk_pos + lateral * side * 22.0 + to_goal * 20.0;
                    p.round_keeper_target_x = target.x.clamp(15.0, context.field_size.width as f32 - 15.0);
                    p.round_keeper_target_y = target.y.clamp(15.0, context.field_size.height as f32 - 15.0);
                    // 500 ticks (5s) -- the frozen target can still be
                    // 120-176u away at realistic ~0.35-0.6u/tick dribble
                    // speed, needing up to ~500 ticks worst case; a real
                    // end-to-end breakaway chase does take several
                    // seconds too.
                    p.round_keeper_commitment_ticks = 500;
                }
            }

            let current_owner_team = if raw_owner == last_owner_id {
                last_possession_team
            } else {
                last_owner_id = raw_owner;
                raw_owner
                    .and_then(|id| field.players.iter().find(|p| p.id == id).map(|p| p.team_id))
            };
            // Shape rules: first active rule wins; no active rule reverts
            // to the captured base. Ball-lane triggers are side-corrected
            // so "left" is always the manager's left wing.
            if any_shape_rules {
                let w = context.field_size.width as f32;
                let h = context.field_size.height as f32;
                let third = h / 3.0;
                let ball_y = field.ball.position.y;
                // Score/time trigger inputs (Phase 7.3). The displayed
                // minute maps the compressed match clock onto 0-90.
                let shape_home_id = field.home_team_id;
                let shape_home_goals = context.score.home_team.get();
                let shape_away_goals = context.score.away_team.get();
                let minute_now =
                    (context.total_match_time * 90 / super::MATCH_TIME_MS) as u32;
                for p in field.players.iter_mut() {
                    if p.shape_rules.is_empty() {
                        continue;
                    }
                    let Some(base) = p.shape_base else { continue };
                    let side = match p.side {
                        Some(s) => s,
                        None => continue,
                    };
                    let in_possession = current_owner_team == Some(p.team_id);
                    let raw_lane: i8 = if ball_y < third {
                        -1
                    } else if ball_y > h - third {
                        1
                    } else {
                        0
                    };
                    let lane = match side {
                        PlayerSide::Left => raw_lane,
                        PlayerSide::Right => -raw_lane,
                    };
                    let (goals_for, goals_against) = if p.team_id == shape_home_id {
                        (shape_home_goals, shape_away_goals)
                    } else {
                        (shape_away_goals, shape_home_goals)
                    };
                    // Principles-of-Play plan (2026-08-30): `CompactBackLine`
                    // needs this player's own team's tactical snapshot
                    // (`defensive_line_x`/`compactness_target`), so the
                    // team lookup happens once per player here rather
                    // than inside `ShapeRule` itself.
                    let team_tactical = *context.tactical_for_team(p.team_id);
                    // docs/positional-attack-phase-design.md, Stage 5:
                    // `UnderHighPress` needs to know the OPPONENT's
                    // phase, not just this player's own team's.
                    let opponent_team_id = if p.team_id == context.field_home_team_id {
                        context.field_away_team_id
                    } else {
                        context.field_home_team_id
                    };
                    let opponent_phase = context.tactical_for_team(opponent_team_id).phase;
                    #[cfg(feature = "match-logs")]
                    if matches!(opponent_phase, crate::r#match::GamePhase::HighPress)
                        && in_possession
                        && matches!(
                            p.tactical_position.current_position,
                            crate::PlayerPositionType::DefenderCenter
                                | crate::PlayerPositionType::DefenderCenterLeft
                                | crate::PlayerPositionType::DefenderCenterRight
                        )
                    {
                        match_log_info!(
                            "STAGE5CB cond=highpress id={} team={} y={:.1}",
                            p.id,
                            p.team_id,
                            p.position.y
                        );
                    } else if in_possession
                        && matches!(
                            p.tactical_position.current_position,
                            crate::PlayerPositionType::DefenderCenter
                                | crate::PlayerPositionType::DefenderCenterLeft
                                | crate::PlayerPositionType::DefenderCenterRight
                        )
                    {
                        match_log_info!(
                            "STAGE5CB cond=normal id={} team={} y={:.1}",
                            p.id,
                            p.team_id,
                            p.position.y
                        );
                    }
                    let ball_owned = current_owner_team.is_some();
                    let mut target = base;
                    let mut rule_active = false;
                    for rule in &p.shape_rules {
                        if rule.is_active(
                            in_possession,
                            ball_owned,
                            lane,
                            goals_for,
                            goals_against,
                            minute_now,
                            team_tactical,
                            opponent_phase,
                        ) {
                            target = rule.target(base, side, w, h, team_tactical, ball_y);
                            rule_active = true;
                            break;
                        }
                    }
                    p.start_position = target;
                    p.shape_rule_active = rule_active;
                }
            }

            // realism-bug (2026-09-12): forward off-ball width commitment
            // (see the field's own doc comment, player.rs) — a bounded,
            // probabilistic "hold genuine flank depth for a few seconds"
            // window, mirroring byline_commitment_ticks' own proven
            // persistence pattern. Decay/release first (unconditional,
            // same idiom as combo_partner_ticks above), then a fresh arm
            // check for anyone not currently committed.
            for p in field.players.iter_mut() {
                if p.width_commitment_ticks > 0 {
                    if raw_owner == Some(p.id) {
                        // Released the moment he actually receives the
                        // ball — on-ball decision logic takes over from
                        // there, this field only governs off-ball
                        // positioning.
                        p.width_commitment_ticks = 0;
                    } else {
                        p.width_commitment_ticks -= 1;
                    }
                    continue;
                }
                if raw_owner == Some(p.id) {
                    continue;
                }
                if p.tactical_position.current_position.position_group()
                    != crate::PlayerFieldPositionGroup::Forward
                {
                    continue;
                }
                // Team must control the ball -- a real striker doesn't
                // drift wide to stretch play while chasing the ball back.
                if current_owner_team != Some(p.team_id) {
                    continue;
                }
                // Settled possession only, not an urgent attacking
                // moment -- matches the real doctrine this whole fix is
                // built on (a striker's occasional channel run happens
                // during patient build-up/progression, not mid-transition
                // or already inside a live attacking sequence).
                let team_tactical_ref = if Some(p.team_id) == Some(field.home_team_id) {
                    &context.tactical_home
                } else {
                    &context.tactical_away
                };
                if !matches!(
                    team_tactical_ref.phase,
                    crate::r#match::GamePhase::BuildUp | crate::r#match::GamePhase::Progression
                ) {
                    continue;
                }
                let h = context.field_size.height as f32;
                // Aim at whichever touchline is nearer this forward's
                // current channel -- a real player drifts to HIS OWN
                // side, not an arbitrary one.
                let near_flank_y = if p.position.y < h * 0.5 { h * 0.08 } else { h * 0.92 };
                // Genuine room check: no opponent already camped near
                // that flank spot, roughly level with this player --
                // an occupied channel isn't a real opportunity to drift
                // into.
                let has_room = !opponent_positions.iter().any(|(opp_team, opp_pos)| {
                    *opp_team != p.team_id
                        && (opp_pos.y - near_flank_y).abs() < 40.0
                        && (opp_pos.x - p.position.x).abs() < 70.0
                });
                if !has_room {
                    continue;
                }
                // Retuned once (1-in-3000 -> 1-in-1100, 400 -> 550
                // ticks) to test whether raising the arm rate helped --
                // measured WORSE (4.1%->2.6%), not better. Given this is
                // a rare, probabilistic per-match event, a 20-match
                // sample's EPISODE count (not the huge per-tick sample
                // size) is the real, much smaller n, and high batch-to-
                // batch variance at that scale is exactly this project's
                // own well-documented pattern elsewhere. Reverted to the
                // original settings, the one configuration with a clear
                // measured gain, rather than chase a single noisier
                // reading. Still category (d), still tunable.
                if context.rng.range_u64(0, 3000) == 0 {
                    p.width_commitment_ticks = 400;
                }
            }

            let possession_changed =
                current_owner_team != last_possession_team && current_owner_team.is_some();
            let home_score_now = context.score.home_team.get();
            let away_score_now = context.score.away_team.get();
            let score_changed =
                home_score_now != last_home_score || away_score_now != last_away_score;
            let home_instr_now = context.coach_home.instruction;
            let away_instr_now = context.coach_away.instruction;
            let instr_changed =
                home_instr_now != last_home_instruction || away_instr_now != last_away_instruction;
            let home_zone_now = context.tactical_home.ball_zone;
            let away_zone_now = context.tactical_away.ball_zone;
            // Attacking-third entry/exit on either side.
            use crate::r#match::BallZone;
            let zone_changed = matches!(home_zone_now, BallZone::AttackingThird)
                != matches!(last_home_zone, BallZone::AttackingThird)
                || matches!(away_zone_now, BallZone::AttackingThird)
                    != matches!(last_away_zone, BallZone::AttackingThird);
            if possession_changed || score_changed || instr_changed || zone_changed {
                transition_window_remaining = TRANSITION_WINDOW_TICKS;
                if possession_changed {
                    last_possession_team = current_owner_team;
                }
                if score_changed {
                    last_home_score = home_score_now;
                    last_away_score = away_score_now;
                }
                if instr_changed {
                    last_home_instruction = home_instr_now;
                    last_away_instruction = away_instr_now;
                }
                if zone_changed {
                    last_home_zone = home_zone_now;
                    last_away_zone = away_zone_now;
                }
            }

            let tactical_interval = if transition_window_remaining > 0 {
                TRANSITION_TACTICAL_INTERVAL_TICKS
            } else {
                BASE_TACTICAL_INTERVAL_TICKS
            };
            if tactical_eval_counter >= tactical_interval {
                let interval = tactical_eval_counter;
                tactical_eval_counter = 0;
                Self::refresh_tactical_states(field, context, interval);
                // refresh_tactical_states may have repointed
                // ball_zone — re-snapshot to avoid spuriously
                // re-triggering the window on the next tick.
                last_home_zone = context.tactical_home.ball_zone;
                last_away_zone = context.tactical_away.ball_zone;
            }

            // Full tick: ball + player AI + events
            // Light tick: ball + player movement only (no AI re-evaluation)
            if tick_parity & 1 == 0 {
                Self::game_tick_light(field, context, match_data, &mut tick_ctx, &mut events);
            } else {
                Self::game_tick_inner(field, context, match_data, &mut tick_ctx, &mut events);
            }

            // Replay-position recording, gated by a cursor instead of
            // a per-tick modulo. Same 30 ms cadence as before; just one
            // u64 comparison + add per tick when nothing is being
            // tracked (the dominant production case).
            if track_positions && context.total_match_time >= next_position_record_ms {
                Self::write_match_positions(field, context.total_match_time, match_data);
                next_position_record_ms += Self::POSITION_RECORD_INTERVAL_MS;
            }

            // Substitutions allowed from the second half onwards, plus
            // extra time when we reach it in a knockout tie. First-half
            // subs in real football are reactive (injuries) — we defer
            // that to the injury/fitness pipeline rather than speculating
            // here. ET gets one bonus sub on entry (FIFA rule).
            let subs_enabled = matches!(
                context.state.match_state,
                MatchState::SecondHalf | MatchState::ExtraTime
            );

            if subs_enabled {
                // Grant the ET bonus once — bumps the cap by 1 for both
                // sides — but only when the active rule set allows it.
                // Friendlies (cap = usize::MAX) skip the increment.
                if context.state.match_state == MatchState::ExtraTime
                    && !et_bonus_granted
                    && context.allow_extra_time_extra_sub
                {
                    if context.max_substitutions_per_team < usize::MAX {
                        context.max_substitutions_per_team += 1;
                    }
                    et_bonus_granted = true;
                    // Reset the next-sub timer for the new period.
                    sub_times_initialized = false;
                }

                if !sub_times_initialized {
                    next_sub_time_ms = context.rng.range_u64(10, 20) * 60 * 1000;
                    sub_times_initialized = true;
                }

                let period_time = context.time.time;
                if period_time >= next_sub_time_ms {
                    // Deterministic "today" — captured at context
                    // construction. Used only for the youth-protection
                    // sub branch, where the comparison is age <= 17.
                    let today = context.today;
                    let per_pass_cap = context.max_substitutions_per_pass;
                    process_substitutions(field, context, per_pass_cap, today);
                    next_sub_time_ms = period_time + context.rng.range_u64(5, 15) * 60 * 1000;
                }
            }
        }

        PerfCounters::instance().record_play_inner(tick_count, inner_start.elapsed());
        result
    }
}
