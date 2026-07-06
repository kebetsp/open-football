use crate::r#match::StateProcessingContext;
use nalgebra::Vector3;

/// Called from midfielder walking velocity() when the team is attacking.
///
/// Checks if the forward zone ahead is vacant (no forward is closer to the
/// opponent goal than this midfielder). If vacant and this midfielder is the
/// nearest to fill it, returns a target position in the forward zone.
/// Returns None if the zone is covered or another midfielder is closer.
pub fn find_vacant_forward_slot(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
    let field_width = ctx.context.field_size.width as f32;
    let field_height = ctx.context.field_size.height as f32;
    let pitch_center_x = field_width / 2.0;

    let goal = ctx.player().opponent_goal_position();
    let my_pos = ctx.player.position;
    let ball_x = ctx.tick_context.positions.ball.position.x;

    // Attacking direction: +1 = left-to-right, -1 = right-to-left
    let atk_dir: f32 = if goal.x > pitch_center_x { 1.0 } else { -1.0 };

    // Only activate when ball is in the attacking half
    if (ball_x - pitch_center_x) * atk_dir < 0.0 {
        return None;
    }

    let my_goal_dist = (goal - my_pos).magnitude();

    // Is any forward already ahead of me (closer to goal)?
    let forward_ahead = ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.tactical_positions.is_forward())
        .any(|t| (goal - t.position).magnitude() < my_goal_dist);

    if forward_ahead {
        return None;
    }

    // Am I the closest midfielder to the goal? Prevents two midfielders
    // both filling the same vacant forward slot simultaneously.
    let closer_mid = ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.tactical_positions.is_midfielder() && t.id != ctx.player.id)
        .any(|t| (goal - t.position).magnitude() < my_goal_dist - 15.0);

    if closer_mid {
        return None;
    }

    // Move into the forward zone: ~120u from goal, keeping my y-lane so
    // different midfielders spread across the width rather than converging.
    let center_y = field_height / 2.0;
    let target_x = (goal.x - atk_dir * 120.0).clamp(20.0, field_width - 20.0);
    let target_y = (my_pos.y * 0.6 + center_y * 0.4).clamp(20.0, field_height - 20.0);

    Some(Vector3::new(target_x, target_y, 0.0))
}

/// Called from defender/forward walking velocity() when the ball is in the
/// centre zone of the pitch.
///
/// Checks if the midfield zone is uncontested (no teammate midfielder is
/// within 150u of the ball). If uncontested and this player is the nearest
/// teammate to the ball, returns the ball position as the target to move
/// toward — filling the gap and contesting the area.
/// Returns None if midfield is covered or another player is closer.
pub fn find_vacant_midfielder_slot(ctx: &StateProcessingContext) -> Option<Vector3<f32>> {
    let field_width = ctx.context.field_size.width as f32;
    let pitch_center_x = field_width / 2.0;

    let goal = ctx.player().opponent_goal_position();
    let ball_pos = ctx.tick_context.positions.ball.position;
    let my_pos = ctx.player.position;

    let atk_dir: f32 = if goal.x > pitch_center_x { 1.0 } else { -1.0 };
    let ball_offset = (ball_pos.x - pitch_center_x) * atk_dir;

    // Only activate in the central zone (±field_width/6 from centre)
    if ball_offset.abs() > field_width / 6.0 {
        return None;
    }

    // Is any midfielder already near the ball?
    let midfielder_present = ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.tactical_positions.is_midfielder())
        .any(|t| (t.position - ball_pos).magnitude() < 150.0);

    if midfielder_present {
        return None;
    }

    // Am I the nearest teammate to the ball?
    let my_ball_dist = (my_pos - ball_pos).magnitude();
    let closer_exists = ctx
        .players()
        .teammates()
        .all()
        .filter(|t| t.id != ctx.player.id)
        .any(|t| (t.position - ball_pos).magnitude() < my_ball_dist - 15.0);

    if closer_exists {
        return None;
    }

    Some(ball_pos)
}
