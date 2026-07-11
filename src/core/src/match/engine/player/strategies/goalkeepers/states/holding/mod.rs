use crate::r#match::goalkeepers::states::common::{ActivityIntensity, GoalkeeperCondition};
use crate::r#match::goalkeepers::states::state::GoalkeeperState;
use crate::r#match::{
    ConditionContext, StateChangeResult, StateProcessingContext, StateProcessingHandler,
};
use nalgebra::Vector3;

const MIN_HOLDING_DURATION: u64 = 80;
const MAX_HOLDING_DURATION: u64 = 160;
/// §9.2.2 — the keeper needs a beat to gather and look up before a
/// quick release is even physically plausible.
const QUICK_RELEASE_MIN_HOLD: u64 = 12;
/// §9.2.2 — per-tick probability of releasing early once a strong
/// quick-counter option exists. ~0.05/tick over a 80-160 tick hold
/// makes the quick release likely-but-not-certain when the option
/// persists, and rare when it flickers for a few ticks.
const QUICK_RELEASE_P_PER_TICK: f32 = 0.03;

#[derive(Default, Clone)]
pub struct GoalkeeperHoldingState {}

impl GoalkeeperHoldingState {
    /// §9.2.2 — scan for a strong quick-counter option: a teammate
    /// meaningfully upfield of the keeper with real open space (no
    /// opponent within 30u). Spirit of the forwards'
    /// `find_optimal_free_zone` scoring, reduced to the two terms that
    /// matter for a keeper's throw/kick decision: progress and space.
    /// Thresholds tuned empirically: at formation anchors the opposing
    /// back line sits ~50u off the forwards, so 90u of space means a
    /// runner who has genuinely broken free, and 300u of progress
    /// excludes midfielders standing in the recovery shape.
    fn quick_counter_option_exists(ctx: &StateProcessingContext) -> bool {
        let gk_pos = ctx.player.position;
        let goal_pos = ctx.player().opponent_goal_position();
        let upfield: f32 = if goal_pos.x > gk_pos.x { 1.0 } else { -1.0 };

        ctx.players().teammates().all().any(|t| {
            if t.id == ctx.player.id {
                return false;
            }
            // Meaningfully upfield of the keeper — a counter outlet,
            // not a centre-back standing next to the box.
            if (t.position.x - gk_pos.x) * upfield < 320.0 {
                return false;
            }
            // Genuinely open: no opponent within 30u.
            let nearest_opp = ctx
                .players()
                .opponents()
                .all()
                .map(|o| (o.position - t.position).magnitude())
                .fold(f32::MAX, f32::min);
            nearest_opp > 90.0
        })
    }
}

impl StateProcessingHandler for GoalkeeperHoldingState {
    fn process(&self, ctx: &StateProcessingContext) -> Option<StateChangeResult> {
        // If for some reason we no longer have the ball, return to standing
        if !ctx.player.has_ball(ctx) {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Standing,
            ));
        }

        // §9.2.2 release discretion: every held tick, scan for a strong
        // quick-counter option; when one exists, release early with some
        // probability instead of waiting out the recovery window.
        if ctx.in_state_time >= QUICK_RELEASE_MIN_HOLD
            && Self::quick_counter_option_exists(ctx)
            && ctx.context.rng.bernoulli(QUICK_RELEASE_P_PER_TICK)
        {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Distributing,
            ));
        }

        // Skill-based duration: better decision-makers distribute faster.
        // When losing, halve the wait — urgency to restart quickly.
        let decision = ctx.player.skills.mental.decisions / 20.0;
        let base_duration = MAX_HOLDING_DURATION
            - ((MAX_HOLDING_DURATION - MIN_HOLDING_DURATION) as f32 * decision) as u64;
        let holding_duration = if ctx.team().is_loosing() {
            base_duration / 2
        } else {
            base_duration
        };
        if ctx.in_state_time >= holding_duration {
            return Some(StateChangeResult::with_goalkeeper_state(
                GoalkeeperState::Distributing,
            ));
        }

        // No other transitions - goalkeeper should continue holding the ball
        // until ready to distribute it, should not try to catch the same ball
        // they already possess
        None
    }

    fn velocity(&self, _ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
        Some(Vector3::new(0.0, 0.0, 0.0))
    }

    fn process_conditions(&self, ctx: ConditionContext) {
        // Holding ball is a low intensity activity with minimal physical effort
        GoalkeeperCondition::new(ActivityIntensity::Low).process(ctx);
    }
}
