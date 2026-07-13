use super::*;
use crate::r#match::engine::player::events::players::FoulResolver;
use nalgebra::Vector3;
#[cfg(feature = "match-logs")]
use std::sync::atomic::Ordering;

impl<const W: usize, const H: usize> FootballEngine<W, H> {
    // ───────────────────────────────────────────────────────────────────────
    // Tick processing
    // ───────────────────────────────────────────────────────────────────────

    pub fn game_tick(
        field: &mut MatchField,
        context: &mut MatchContext,
        match_data: &mut ResultMatchPositionData,
        tick_ctx: &mut GameTickContext,
    ) {
        let mut events = EventCollection::with_capacity(10);
        Self::game_tick_inner(field, context, match_data, tick_ctx, &mut events);
        // Keep this public single-tick wrapper self-contained — the
        // play_inner loop now gates position recording with a cursor
        // (`next_position_record_ms`) for efficiency, but external
        // callers of `game_tick` still expect each call to emit a
        // position sample when the timestamp is on the 30 ms cadence.
        Self::write_match_positions(field, context.total_match_time, match_data);
    }

    /// Light tick: full ball logic (physics, ownership, goals) but players only move.
    pub(super) fn game_tick_light(
        field: &mut MatchField,
        context: &mut MatchContext,
        match_data: &mut ResultMatchPositionData,
        tick_ctx: &mut GameTickContext,
        events: &mut EventCollection,
    ) {
        events.clear();

        field.ball.update_light(context, &field.players, events);
        Self::apply_pending_set_piece_teleport(field);
        Self::apply_pending_save_credit(field);

        // Shot-flight GK reactivity: normally light ticks skip player
        // AI to save CPU, but during a shot the keeper needs continuous
        // decisions to close on the intercept line. Run just the two
        // goalkeepers (cheap, ~2 of 22 players) when a shot is in
        // flight. Refresh the *existing* tick_ctx in place instead of
        // allocating a fresh GameTickContext (grid+space buffers) every
        // light tick during the shot window.
        if field.ball.cached_shot_target.is_some() {
            tick_ctx.update_for_goalkeeper_shot(field);
            Self::play_goalkeepers(field, context, tick_ctx, events);
        }

        // Skip sent-off players: they've been stashed at (-500, -500). A
        // boundary clamp here would drag them to (0, 0) — the pitch's
        // top-left corner — which then gets recorded as a ghost sample
        // by `write_match_positions`.
        for player in field.players.iter_mut().filter(|p| !p.is_sent_off) {
            player.check_boundary_collision(context);
            player.move_to();
        }

        if events.has_events() {
            EventDispatcher::dispatch(events, field, context, match_data, true);
            handle_goal_reset(field, context);
        }
    }

    pub(super) fn game_tick_inner(
        field: &mut MatchField,
        context: &mut MatchContext,
        match_data: &mut ResultMatchPositionData,
        tick_ctx: &mut GameTickContext,
        events: &mut EventCollection,
    ) {
        tick_ctx.update(field);

        events.clear();

        Self::play_ball(field, context, tick_ctx, events);
        Self::apply_pending_set_piece_teleport(field);
        Self::apply_pending_save_credit(field);
        Self::resolve_penalty_kick(field, context);
        Self::resolve_free_kick(field, context, match_data);
        Self::resolve_corner_contest(field, context);
        // Resolve any deferred-foul / advantage state. Cheap (one
        // Option read in the dominant no-advantage case) so we run it
        // every full tick rather than waiting for the next event.
        FoulResolver::tick_advantage(field, context, events);
        // Ownership may have changed inside play_ball (new claim, pass
        // target receive, etc.). Refresh the ball view so player state
        // dispatch sees the current owner — without this, the
        // TakeBall force-override fires for a player who already has
        // the ball.
        tick_ctx.refresh_ball(field);
        Self::play_players(field, context, tick_ctx, events);

        EventDispatcher::dispatch(events, field, context, match_data, true);

        handle_goal_reset(field, context);
    }

    /// Corner kicks and goal kicks rewrite ball ownership inside `ball.update`,
    /// but ball.rs only has `&[MatchPlayer]` — it can't teleport the designated
    /// taker to the ball. Instead it stashes the teleport intent on the Ball;
    /// we drain it here, now that we have `&mut field.players`. Without this,
    /// the ball sits at the corner flag / goal area with ownership assigned
    /// to a player 30-200 units away, and `move_to`'s 15-unit distance check
    /// nulls ownership on the very next tick — ball stalls for seconds.
    pub(super) fn apply_pending_set_piece_teleport(field: &mut MatchField) {
        if let Some((player_id, ball_pos)) = field.ball.pending_set_piece_teleport.take() {
            // Handing the ball to a staged taker starts a NEW restart —
            // any live same-touch lock belongs to a dead delivery chain
            // (§9.4.1). Without this, a taker whose previous restart went
            // straight out of play stays barred from his own new kick.
            field.ball.restart_taker_lock = None;
            if let Some(idx) = field.player_index(player_id) {
                let p = &mut field.players[idx];
                p.position = ball_pos;
                p.velocity = Vector3::zeros();
                p.in_state_time = 0;
            }
        }

        // Foul-restart set-up: place the free-kick wall / retreating
        // defenders, or clear the box for a penalty. No state override —
        // players resume normal positioning after the restart window
        // (there's no stoppage in the sim to walk the wall up during).
        if !field.ball.pending_restart_teleports.is_empty() {
            let teleports = std::mem::take(&mut field.ball.pending_restart_teleports);
            for (player_id, pos) in teleports {
                if let Some(idx) = field.player_index(player_id) {
                    let p = &mut field.players[idx];
                    p.position = pos;
                    p.velocity = Vector3::zeros();
                    p.in_state_time = 0;
                }
            }
        }

        // Corner dead-ball set-up: teleport the pushed-up centre-backs
        // into the box so they can attack the delivery (see
        // `Ball::pending_corner_teleports` — there's no stoppage in the
        // sim for them to walk up during, and they can't run the length
        // of the pitch inside the cross window).
        if !field.ball.pending_corner_teleports.is_empty() {
            use crate::r#match::defenders::states::DefenderState;
            use crate::r#match::player::state::PlayerState;
            let teleports = std::mem::take(&mut field.ball.pending_corner_teleports);
            for (player_id, pos) in teleports {
                if let Some(idx) = field.player_index(player_id) {
                    let p = &mut field.players[idx];
                    p.position = pos;
                    p.velocity = Vector3::zeros();
                    p.in_state_time = 0;
                    // Force the AttackingCorner state directly — the CB may
                    // have been in any defensive state when the corner was
                    // won, and not all of them carry the entry hook. This
                    // guarantees they attack the delivery.
                    p.state = PlayerState::Defender(DefenderState::AttackingCorner);
                }
            }
        }
    }

    /// Discrete corner aerial contest — fires once, the instant the corner
    /// cross is airborne. A played-out lofted corner can't thread the
    /// congested box to the pushed-up centre-back: the cross is always
    /// claimed/cleared mid-flight (`CB header chances` stayed 0 through
    /// every piecemeal GK / defender-duel fix). So we resolve ONE
    /// skill-weighted aerial contest — the best attacking header (a
    /// pushed-up CB or a forward) vs the defending line + GK command of
    /// area — and, if the attacker wins, drop the ball onto their head.
    /// Their existing heading state then strikes it on goal through the
    /// NORMAL shot/save pipeline, so the goal / shot / xG / save stats all
    /// credit correctly (no bespoke scoring path). The win chance is tuned
    /// (~0.30, modulated by the aerial mismatch and the keeper) so that —
    /// carried by a corner header's ~0.10-0.14 xG in the shot pipeline —
    /// only ~3-4% of corners end in a goal (real ≈ 3%), giving defenders
    /// their realistic set-piece share without inflating totals.
    /// §9.3.3 — discrete in-match penalty resolution. A penalty is a
    /// fixed, isolated 1v1 by definition, so instead of letting the
    /// taker's open-play state machine improvise from the spot (pass!
    /// dribble!), the kick resolves through the same skill model the
    /// shootout uses (`penalty_conversion_prob`), the moment the taker
    /// is staged on the ball after the §9.3.1 stoppage. Runs BEFORE
    /// `play_players` in the tick, so the taker's AI never gets a
    /// decision tick with the ball.
    ///
    /// Outcomes are then animated physically:
    ///   * goal   — ball launched into a corner on a high arc
    ///              (position.z > 2.5 is exempt from interception; the
    ///              claim cooldown covers the flight), resolved by the
    ///              normal `check_goal` path so score/stats credit
    ///              normally (no assist — `last_shot_assister_id` None).
    ///   * saved  — ball placed in the keeper's hands; save + on-target
    ///              stats credited via `pending_save_credit`.
    ///   * missed — ball sails over the bar; `check_over_goal` resolves
    ///              it into the normal goal-kick restart.
    pub(super) fn resolve_penalty_kick(field: &mut MatchField, context: &mut MatchContext) {
        use crate::r#match::PassOriginRestart;
        use crate::r#match::engine::set_pieces::{
            penalty_conversion_prob, score_keeper_save, score_penalty_taker,
        };
        use nalgebra::Vector3;

        if field.ball.pass_origin_restart != PassOriginRestart::Penalty {
            return;
        }
        let Some(taker_id) = field.ball.current_owner else {
            return;
        };
        // Staged: taker teleport has drained and he stands on the ball.
        let Some(taker) = field.players.iter().find(|p| p.id == taker_id) else {
            return;
        };
        let d = taker.position - field.ball.position;
        if (d.x * d.x + d.y * d.y).sqrt() > 6.0 {
            return;
        }

        // Which goal: the nearer one (the ball sits on the penalty spot,
        // 88u from the goal line — always unambiguous).
        let gl = context.goal_positions.left;
        let gr = context.goal_positions.right;
        let ball_pos = field.ball.position;
        let goal = if (ball_pos - gl).magnitude() < (ball_pos - gr).magnitude() {
            gl
        } else {
            gr
        };

        // Defending keeper: nearest opposing GK (placed on his line at
        // award time). May be absent after a red card with no sub — the
        // conversion model's keeper-score floor handles that.
        let taker_team = taker.team_id;
        let gk = field
            .players
            .iter()
            .find(|p| {
                p.team_id != taker_team
                    && !p.is_sent_off
                    && p.tactical_position.current_position.is_goalkeeper()
            })
            .map(|p| {
                let g = &p.skills.goalkeeping;
                let m = &p.skills.mental;
                (
                    p.id,
                    score_keeper_save(
                        g.reflexes,
                        p.skills.physical.agility,
                        g.handling,
                        m.anticipation,
                        p.attributes.pressure,
                        m.concentration,
                    )
                    .clamp(0.05, 1.0),
                )
            });
        let taker_score = {
            let t = &taker.skills.technical;
            let m = &taker.skills.mental;
            score_penalty_taker(
                t.penalty_taking,
                t.finishing,
                m.composure,
                taker.attributes.pressure,
                t.technique,
                0.0,
            )
            .clamp(0.05, 1.0)
        };
        let keeper_score = gk.map(|(_, s)| s).unwrap_or(0.05);

        // Late-match pressure bump, same spirit as the shootout's round
        // ramp but mild for in-play penalties.
        let minute = (context.total_match_time / 60_000) as u32;
        let pressure = if minute >= 75 { 0.55 } else { 0.35 };
        let p_goal = penalty_conversion_prob(taker_score, keeper_score, pressure, false);

        let tick = context.current_tick();
        let dir2d = {
            let v = Vector3::new(goal.x - ball_pos.x, goal.y - ball_pos.y, 0.0);
            if v.norm() > 0.01 {
                v.normalize()
            } else {
                Vector3::new(1.0, 0.0, 0.0)
            }
        };
        // Penalty xG is effectively fixed (~0.79) — stamp it so the GK
        // xg_prevented credit/debit works in both save and goal paths.
        field.ball.last_shot_xg = 0.79;
        field.ball.last_shot_shooter_id = Some(taker_id);
        field.ball.last_shot_assister_id = None;
        field.ball.pass_origin_restart = PassOriginRestart::OpenPlay;
        // §9.4.1 same-touch rule — covers the rebound case: a saved or
        // missed penalty must not be re-taken by the taker before any
        // other player touches it (the SAVE branch's keeper touch clears
        // the lock immediately, matching the law).
        field.ball.restart_taker_lock = Some(taker_id);

        if context.rng.bernoulli(p_goal) {
            // GOAL — flat driven shot into a corner. The keeper stands
            // at goal centre; the ball's closest approach to him is the
            // ~18u aim offset, outside every claim/intercept radius,
            // and the claim cooldown plus the absent `cached_shot_target`
            // keep the physics save & GK state machine out of it — the
            // outcome was already decided by the conversion model.
            let corner_y = if context.rng.bernoulli(0.5) { 18.0 } else { -18.0 };
            let aim = Vector3::new(goal.x, goal.y + corner_y, 0.0);
            let dir = {
                let v = Vector3::new(aim.x - ball_pos.x, aim.y - ball_pos.y, 0.0);
                v.normalize()
            };
            // Launch from the trajectory midpoint: the full 88u flight
            // at 4 u/tick gives the keeper ~22 ticks (~25u of lateral
            // cover — he eats the decided goal). From ~44u he gets ~12
            // ticks (~13u), inside the 18u corner offset, so the model's
            // verdict stands. Reads as an unstoppable strike in replay.
            let mid = Vector3::new(
                (ball_pos.x + aim.x) * 0.5,
                (ball_pos.y + aim.y) * 0.5,
                0.0,
            );
            field.ball.position = mid;
            field.ball.velocity = Vector3::new(dir.x * 4.0, dir.y * 4.0, 0.0);
            // Mark the ball as a live shot. `check_goal` only credits a
            // crossing when the scorer has recent shot MEMORY (invisible
            // here — it reads `context.players`, a copy that never sees
            // our `field` write) or when a shot is in flight via this
            // cache — the path every open-play goal takes. The
            // `physics_save_rolled` latch goes in pre-consumed so
            // `try_save_shot` can't re-roll an outcome the conversion
            // model already decided.
            {
                use crate::r#match::engine::ball::ball::ShotTarget;
                let defending_side = if goal.x < ball_pos.x {
                    PlayerSide::Left
                } else {
                    PlayerSide::Right
                };
                field.ball.cached_shot_target = Some(ShotTarget {
                    goal_line_y: aim.y,
                    goal_line_z: 0.5,
                    defending_side,
                    deflected: false,
                    physics_save_rolled: true,
                });
            }
            field.ball.previous_owner = Some(taker_id);
            field.ball.current_owner = None;
            field.ball.claim_cooldown = 80;
            field.ball.flags.in_flight_state = 80;
            if let Some(p) = field.get_player_mut(taker_id) {
                p.memory.record_shot(tick, true);
            }
        } else if context.rng.bernoulli(0.60) {
            // SAVED (60% of non-goals) — keeper holds it.
            if let Some((gk_id, _)) = gk {
                let gk_pos = field
                    .players
                    .iter()
                    .find(|p| p.id == gk_id)
                    .map(|p| p.position)
                    .unwrap_or(Vector3::new(goal.x, goal.y, 0.0));
                let gk_team = field
                    .players
                    .iter()
                    .find(|p| p.id == gk_id)
                    .map(|p| p.team_id)
                    .unwrap_or(0);
                field.ball.position = Vector3::new(gk_pos.x, gk_pos.y, 0.0);
                field.ball.velocity = Vector3::zeros();
                field.ball.previous_owner = Some(taker_id);
                field.ball.current_owner = Some(gk_id);
                field.ball.ownership_duration = 0;
                field.ball.claim_cooldown = 60;
                field.ball.flags.in_flight_state = 60;
                field.ball.pending_save_credit = Some((gk_id, taker_id));
                field.ball.record_touch(gk_id, gk_team, tick, true);
                if let Some(p) = field.get_player_mut(taker_id) {
                    p.memory.record_shot(tick, true);
                }
            } else {
                // No keeper to save it — treat as a miss over the bar.
                field.ball.velocity = Vector3::new(dir2d.x * 2.8, dir2d.y * 2.8, 3.0);
                field.ball.previous_owner = Some(taker_id);
                field.ball.current_owner = None;
                field.ball.claim_cooldown = 80;
                field.ball.flags.in_flight_state = 80;
                if let Some(p) = field.get_player_mut(taker_id) {
                    p.memory.record_shot(tick, false);
                }
            }
        } else {
            // MISSED — over the bar; check_over_goal turns it into a
            // goal kick.
            field.ball.velocity = Vector3::new(dir2d.x * 2.8, dir2d.y * 2.8, 3.0);
            field.ball.previous_owner = Some(taker_id);
            field.ball.current_owner = None;
            field.ball.claim_cooldown = 80;
            field.ball.flags.in_flight_state = 80;
            field.ball.last_shot_xg = 0.0;
            field.ball.last_shot_shooter_id = None;
            if let Some(p) = field.get_player_mut(taker_id) {
                p.memory.record_shot(tick, false);
            }
        }
    }

    /// §12.2 — dedicated free-kick-taking decision, distinct from the
    /// open-play shot/pass evaluation the staged taker would otherwise
    /// fall through to (which judged a 25-30 yard set-piece shot like a
    /// pressured open-play shot and always played the nearest-teammate
    /// pass). Runs once per staged direct free kick, right after the
    /// §9.3.1 stoppage ends: rolls the (previously dead) pure scoring
    /// model in `officiating::set_pieces::score_free_kick_choices` into
    /// a weighted choice and executes it through the NORMAL event
    /// pipeline — a `Shoot` (which already carries the direct-FK wall
    /// block, `wall_blocks_direct_fk`) or a lofted box-delivery `PassTo`
    /// at the §11.7-staged headers. Short/recycle outcomes leave the
    /// taker to his open-play state machine — the old behaviour, now the
    /// exception rather than the default. Events dispatch immediately so
    /// the taker's own AI never gets a decision tick with the ball
    /// (same reasoning as `resolve_penalty_kick`).
    pub(super) fn resolve_free_kick(
        field: &mut MatchField,
        context: &mut MatchContext,
        match_data: &mut ResultMatchPositionData,
    ) {
        use crate::r#match::PassOriginRestart;
        use crate::r#match::engine::events::dispatcher::Event;
        use crate::r#match::engine::set_pieces::{
            FreeKickBand, FreeKickChoice, score_free_kick_choices,
        };
        use crate::r#match::player::events::{
            PassingEventContext, PlayerEvent, ShootingEventContext,
        };
        use crate::r#match::player::strategies::players::ShotType;

        if field.ball.pass_origin_restart != PassOriginRestart::DirectFreeKick
            || field.ball.free_kick_decided
        {
            return;
        }
        let Some(taker_id) = field.ball.current_owner else {
            return;
        };
        if field.ball.restart_pending_taker != Some(taker_id) {
            return;
        }
        // Staged: the taker stands on the ball (mirrors the penalty gate).
        let Some(taker) = field.players.iter().find(|p| p.id == taker_id) else {
            return;
        };
        let d = taker.position - field.ball.position;
        if (d.x * d.x + d.y * d.y).sqrt() > 6.0 {
            return;
        }
        let Some(taker_side) = taker.side else {
            return;
        };

        let field_w = context.field_size.width as f32;
        let field_h = context.field_size.height as f32;
        let mid_y = field_h * 0.5;
        // Left defends x≈0 and attacks x=field_w (see award_restart_for_foul's
        // goal_x, which is the FOULER's own goal).
        let goal_x = match taker_side {
            PlayerSide::Left => field_w,
            PlayerSide::Right => 0.0,
        };
        let ball_pos = field.ball.position;
        let to_goal = Vector3::new(goal_x - ball_pos.x, mid_y - ball_pos.y, 0.0);
        let dist_goal = (to_goal.x * to_goal.x + to_goal.y * to_goal.y).sqrt();

        // Beyond crossing range (matches the §11.7 attacking-shape gate)
        // there is nothing set-piece-specific to decide — the deep-FK
        // short pass IS the realistic outcome. Latch and fall through.
        if dist_goal > 280.0 {
            field.ball.free_kick_decided = true;
            if context.logging_enabled {
                match_data.add_match_event(
                    context.total_match_time,
                    "player",
                    format!("FreeKickPlan({}, Deep, {})", taker_id, dist_goal as u32),
                );
            }
            return;
        }

        let band = FreeKickBand::from_distance(dist_goal);
        let taker_team = taker.team_id;
        let taker_fk = taker.skills.technical.free_kicks;
        let taker_crossing = taker.skills.technical.crossing;

        // Aerial advantage of the staged box targets vs the defenders
        // around the goal — the §11.7 shape put the two best headers at
        // the post spots, so compare the best two on each side inside
        // delivery range.
        let mut atk_heading: Vec<f32> = Vec::new();
        let mut def_heading: Vec<f32> = Vec::new();
        for p in field.players.iter() {
            if p.is_sent_off
                || p.id == taker_id
                || p.tactical_position.current_position.is_goalkeeper()
            {
                continue;
            }
            let dg = Vector3::new(goal_x - p.position.x, mid_y - p.position.y, 0.0);
            if (dg.x * dg.x + dg.y * dg.y).sqrt() > 200.0 {
                continue;
            }
            if p.team_id == taker_team {
                atk_heading.push(p.skills.technical.heading);
            } else {
                def_heading.push(p.skills.technical.heading);
            }
        }
        let top2 = |v: &mut Vec<f32>| -> f32 {
            v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            match v.len() {
                0 => 10.0,
                1 => v[0],
                _ => (v[0] + v[1]) * 0.5,
            }
        };
        let aerial_advantage =
            (0.5 + (top2(&mut atk_heading) - top2(&mut def_heading)) / 40.0).clamp(0.0, 1.0);

        // Match state for the scoring model.
        let minute = (context.total_match_time / 60_000) as u32;
        let home_goals = context.score.home_team.get() as i16;
        let away_goals = context.score.away_team.get() as i16;
        let own_diff = if taker_team == context.field_home_team_id {
            home_goals - away_goals
        } else {
            away_goals - home_goals
        };
        let chasing_late = minute >= 60 && own_diff < 0;
        let protecting_lead_late = minute >= 60 && own_diff > 0;

        let scores = score_free_kick_choices(
            band,
            false, // foul restarts are direct free kicks here
            taker_fk,
            taker_crossing,
            aerial_advantage,
            chasing_late,
            protecting_lead_late,
            &context.environment,
        );

        // From a wide position a direct shot isn't realistic — the whole
        // weight goes to the delivery/short options (§12.2).
        let is_wide_angle = (ball_pos.y - mid_y).abs() > dist_goal * 0.6;
        let shot_w = if is_wide_angle { 0.0 } else { scores.direct_shot };

        field.ball.free_kick_decided = true;
        let total = shot_w + scores.box_delivery + scores.short_routine + scores.recycle;
        if total <= 0.0 {
            return;
        }
        let mut roll = context.rng.unit_f32() * total;
        let choice = if roll < shot_w {
            FreeKickChoice::DirectShot
        } else if {
            roll -= shot_w;
            roll < scores.box_delivery
        } {
            FreeKickChoice::BoxDelivery
        } else {
            // ShortRoutine and Recycle both fall through to the taker's
            // normal state machine — the short pass IS that behaviour.
            FreeKickChoice::ShortRoutine
        };

        // Record-only plan tag (same pattern as CornerDelivery §10.2) so
        // batch harnesses can tally choices per distance band.
        if context.logging_enabled {
            match_data.add_match_event(
                context.total_match_time,
                "player",
                format!("FreeKickPlan({}, {:?}, {})", taker_id, choice, dist_goal as u32),
            );
        }

        match choice {
            FreeKickChoice::DirectShot => {
                // Aim inside a post (goal half-width 29u).
                let aim_off = 8.0 + context.rng.unit_f32() * 14.0;
                let aim_y = if context.rng.bernoulli(0.5) {
                    mid_y + aim_off
                } else {
                    mid_y - aim_off
                };
                // Shot power — same formula as `shoot_goal_power`, computed
                // here because there is no StateProcessingContext this deep
                // in the tick.
                let s = &taker.skills;
                let distance_blend = (dist_goal / (field_w * 0.3)).clamp(0.0, 1.0);
                let shot_skill = (s.technical.finishing / 20.0) * (1.0 - distance_blend)
                    + (s.technical.long_shots / 20.0) * distance_blend;
                let skill_multiplier = 0.2
                    + 0.8
                        * (shot_skill * 0.3
                            + (s.technical.technique / 20.0) * 0.25
                            + (s.physical.strength / 20.0) * 0.25
                            + (s.mental.composure / 20.0) * 0.2);
                let distance_factor = 1.0 + (dist_goal / field_w).clamp(0.0, 1.0) * 0.4;
                let condition_factor =
                    0.90 + (taker.player_attributes.condition as f32 / 10_000.0) * 0.10;
                let force =
                    (2.2 * skill_multiplier * distance_factor * condition_factor).clamp(1.4, 3.8);

                let mut evs = EventCollection::with_event(Event::PlayerEvent(PlayerEvent::Shoot(
                    ShootingEventContext {
                        from_player_id: taker_id,
                        target: Vector3::new(goal_x, aim_y, 0.0),
                        force: force as f64,
                        reason: "FK_DIRECT",
                        tick: context.current_tick(),
                        shot_type: ShotType::DirectFreeKick,
                    },
                )));
                EventDispatcher::dispatch(&mut evs, field, context, match_data, true);
                // The wall roll (if any) was consumed inside the shot
                // handler; decay the origin so a rebound shot doesn't get
                // a phantom second wall.
                field.ball.pass_origin_restart = PassOriginRestart::OpenPlay;
            }
            FreeKickChoice::BoxDelivery => {
                // Deliver at the best header staged in the box (§11.7 put
                // him at a post spot at penalty-spot depth). If nobody is
                // in delivery range, fall through to the state machine.
                let receiver = field
                    .players
                    .iter()
                    .filter(|p| {
                        p.team_id == taker_team
                            && p.id != taker_id
                            && !p.is_sent_off
                            && !p.tactical_position.current_position.is_goalkeeper()
                    })
                    .filter(|p| {
                        let dg = Vector3::new(goal_x - p.position.x, mid_y - p.position.y, 0.0);
                        (dg.x * dg.x + dg.y * dg.y).sqrt() < 200.0
                    })
                    .max_by(|a, b| {
                        a.skills
                            .technical
                            .heading
                            .partial_cmp(&b.skills.technical.heading)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|p| (p.id, p.position));
                let Some((receiver_id, receiver_pos)) = receiver else {
                    return;
                };
                // Pass power — same shape as `pass_teammate_power`.
                let s = &taker.skills;
                let skill_factor = (s.technical.passing / 20.0) * 0.35
                    + (s.technical.technique / 20.0) * 0.2
                    + (s.physical.strength / 20.0) * 0.15
                    + (s.mental.vision / 20.0) * 0.15
                    + (s.mental.composure / 20.0) * 0.15;
                let pass_dist = (receiver_pos - ball_pos).magnitude();
                let distance_factor = (pass_dist / (field_w * 0.8)).clamp(0.25, 1.0);
                let condition_factor =
                    0.92 + (taker.player_attributes.condition as f32 / 10_000.0) * 0.08;
                let pass_force = (0.5 + 1.5 * skill_factor * distance_factor) * condition_factor;

                let mut evs = EventCollection::with_event(Event::PlayerEvent(PlayerEvent::PassTo(
                    PassingEventContext {
                        from_player_id: taker_id,
                        to_player_id: receiver_id,
                        pass_target: receiver_pos,
                        pass_force,
                        reason: "FK_CROSS",
                    },
                )));
                EventDispatcher::dispatch(&mut evs, field, context, match_data, true);
            }
            _ => {}
        }
    }

    pub(super) fn resolve_corner_contest(field: &mut MatchField, context: &mut MatchContext) {
        use crate::r#match::PassOriginRestart;
        use nalgebra::Vector3;

        let ball = &field.ball;
        if ball.corner_contest_resolved || ball.pass_origin_restart != PassOriginRestart::Corner {
            return;
        }
        // [diag] reached with an armed Corner origin.
        #[cfg(feature = "match-logs")]
        crate::mid_run_diag::CORNER_CONTEST_SEEN.fetch_add(1, Ordering::Relaxed);
        // Only once the cross has actually left the taker and is airborne
        // (not the dead-ball set-up while the taker still holds it, and not
        // a short ground corner played along the floor).
        if ball.current_owner.is_some() {
            return;
        }
        // [diag] cross has left the taker (loose / in flight).
        #[cfg(feature = "match-logs")]
        crate::mid_run_diag::CORNER_CONTEST_FIRED.fetch_add(1, Ordering::Relaxed);
        if ball.position.z < 2.0 {
            return;
        }

        let minute = (context.total_match_time / 60_000) as u32;

        // The goal under attack is the one the corner is nearest to.
        let gl = context.goal_positions.left;
        let gr = context.goal_positions.right;
        let ball_pos = ball.position;
        let attacked_goal = if (ball_pos - gl).magnitude() < (ball_pos - gr).magnitude() {
            gl
        } else {
            gr
        };

        // Attacking team = the cross taker's team.
        let taker = ball.previous_owner.or(ball.current_owner);
        let att_team = match taker
            .and_then(|id| field.players.iter().find(|p| p.id == id))
            .map(|p| p.team_id)
        {
            Some(t) => t,
            None => {
                field.ball.corner_contest_resolved = true;
                return;
            }
        };

        // Best attacking header, best defending header, and GK command of
        // area — among the players inside the box (≈135u of the goal).
        let mut best_att: Option<(usize, f32)> = None;
        let mut best_def_score = 0.40_f32;
        let mut gk_command = 0.35_f32;
        for (i, p) in field.players.iter().enumerate() {
            if (p.position - attacked_goal).magnitude() > 135.0 {
                continue;
            }
            let is_gk = p.tactical_position.current_position.is_goalkeeper();
            if p.team_id == att_team {
                if is_gk {
                    continue;
                }
                let s = sc::aerial_outfield_attacker(p, minute);
                if best_att.map_or(true, |(_, bs)| s > bs) {
                    best_att = Some((i, s));
                }
            } else if is_gk {
                gk_command = (p.skills.goalkeeping.command_of_area * 0.6
                    + p.skills.goalkeeping.aerial_reach * 0.4)
                    / 20.0;
            } else {
                let s = sc::aerial_outfield_defender(p, minute);
                if s > best_def_score {
                    best_def_score = s;
                }
            }
        }

        let (att_idx, att_score) = match best_att {
            Some(v) => v,
            None => {
                field.ball.corner_contest_resolved = true;
                return;
            }
        };

        let att_win =
            (0.36 + (att_score - best_def_score) * 0.50 - gk_command * 0.18).clamp(0.10, 0.62);

        if context.rng.bernoulli(att_win) {
            #[cfg(feature = "match-logs")]
            crate::mid_run_diag::CORNER_CONTEST_WON.fetch_add(1, Ordering::Relaxed);
            // Attacker wins: drop the ball just behind them at head height,
            // moving goalward, so it reads as an incoming header to their
            // state (the CB's AttackingCorner, or a forward's run→heading).
            // Loose so they head it; keep the Corner origin so the CB stays
            // in AttackingCorner through the strike.
            //
            // Drop kinematics = apex-of-flick hang time. The previous
            // (z 2.2, vz −1.0, 4.0 u/tick drift) fell through the entire
            // heading band [1.4, 2.5] in ONE tick and drifted out of
            // 6u header reach almost as fast — so only a CB already in
            // AttackingCorner (whose same-tick path runs right after
            // this resolver) ever struck it; a FORWARD winner spent the
            // only valid tick transitioning Running→Heading and found
            // the ball below threshold, and the loose ball was then
            // vacuumed by the intercept gate (z ≤ 2.5). Real contested
            // headers hang ~0.3-0.4 s at the apex: z 2.55 (one tick
            // above the intercept window) with vz −0.35 and a modest
            // 1.8 u/tick goalward drift keeps the ball in the heading
            // band and within reach for ~3 ticks — enough for ANY
            // winner's state machine to strike, which is what the
            // contest already decided should happen.
            let winner_pos = field.players[att_idx].position;
            let to_goal = attacked_goal - winner_pos;
            let dir = if to_goal.magnitude() > 0.01 {
                to_goal.normalize()
            } else {
                Vector3::new(1.0, 0.0, 0.0)
            };
            let b = &mut field.ball;
            b.position =
                Vector3::new(winner_pos.x - dir.x * 2.0, winner_pos.y - dir.y * 2.0, 2.55);
            b.velocity = Vector3::new(dir.x * 1.8, dir.y * 1.8, -0.35);
            b.current_owner = None;
            b.previous_owner = taker;
            b.flags.in_flight_state = 1;
        }
        // Otherwise the cross plays out — the keeper claims or a defender
        // clears (the realistic majority outcome).

        // The contest IS the resolution of the delivery — clear the
        // stale cross-target so the original aim point (often the OTHER
        // pushed-up CB) can't auto-claim the dropped ball through the
        // 100u receiver-priority radius. Before this, won headers were
        // routinely converted into a different player's chest-trap →
        // slow foot-shot, and "lost" contests were caught by the
        // attacking CB instead of playing out as GK claims/clearances.
        field.ball.pass_target_player_id = None;
        field.ball.clear_pending_pass_metadata();

        // Persist this corner's routine + estimated xG into the team's
        // history so `pick_corner_routine` can vary future deliveries.
        // The xG used here is a rough estimate (att_win × generic
        // header xG); the precise xG is computed downstream when the
        // header actually fires through the shot pipeline. The history
        // only needs the *flavour* of "did this routine produce a
        // chance" to gate repeats, so an approximate value is fine.
        if let Some(routine) = field.ball.pending_corner_routine.take() {
            let estimated_xg = att_win * 0.12; // ~0.12 header xG ceiling × win prob
            let is_home_attacking = att_team == context.field_home_team_id;
            context
                .set_piece_history
                .record_corner(is_home_attacking, routine, estimated_xg);
        }

        field.ball.corner_contest_resolved = true;
    }

    /// Consume `Ball::pending_save_credit` left behind by the physics
    /// save (`try_save_shot`). When the keeper actually changed ball
    /// state mid-flight (catch, safe parry, dangerous parry), this fires
    /// the save stat for the keeper and the on-target stat for the
    /// shooter — matching the events the GK state machine would have
    /// emitted if the physics save hadn't pre-empted it.
    pub(super) fn apply_pending_save_credit(field: &mut MatchField) {
        let Some((keeper_id, shooter_id)) = field.ball.pending_save_credit.take() else {
            return;
        };
        // One pass over the 22-player list resolves both ids. The team-
        // mismatch guard is defence in depth against any accidental
        // same-team shooter — deflections through the save handler
        // should already have been filtered upstream.
        let Some((keeper_idx, shooter_idx)) = field.two_player_indices(keeper_id, shooter_id)
        else {
            return;
        };
        let keeper_team = field.players[keeper_idx].team_id;
        let shooter_team = field.players[shooter_idx].team_id;
        if keeper_team == shooter_team {
            return;
        }
        let shot_xg = field.ball.last_shot_xg;
        {
            let gk = &mut field.players[keeper_idx];
            gk.statistics.saves += 1;
            gk.statistics.shots_faced += 1;
            // The GK denied a shot worth `shot_xg` xG — full credit goes
            // to xG prevented. Saves an above-baseline keeper from being
            // capped by the synthetic-proxy fallback in the rating helper.
            if shot_xg > 0.0 {
                gk.statistics.record_xg_prevented(shot_xg);
            }
        }
        field.players[shooter_idx].memory.credit_shot_on_target();
        // Shot has resolved (saved). Drop the metadata so any
        // subsequent goal / save event can't double-credit.
        field.ball.clear_shot_metadata();
        field.ball.pending_error_to_shot_player_id = None;
        #[cfg(feature = "match-logs")]
        {
            use std::sync::atomic::Ordering;
            // Re-use the "catch" site bucket — physics-save outcomes are
            // catches, parries, and dangerous parries indistinguishably
            // from the stats viewpoint. The save_pipeline counters above
            // already separate them at the physics layer.
            save_accounting_stats::SAVES_CREDITED[1].fetch_add(1, Ordering::Relaxed);
            save_accounting_stats::ON_TARGET_PAIRED[1].fetch_add(1, Ordering::Relaxed);
        }
    }
}
