use crate::r#match::StateProcessingContext;
use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct PassingEventContext {
    pub from_player_id: u32,
    pub to_player_id: u32,
    pub pass_target: Vector3<f32>,
    pub pass_force: f32,
    pub reason: &'static str,
}

impl PassingEventContext {
    pub fn new() -> PassingEventBuilder {
        PassingEventBuilder::new()
    }
}

pub struct PassingEventBuilder {
    from_player_id: Option<u32>,
    to_player_id: Option<u32>,
    pass_force: Option<f32>,
    pass_target: Option<Vector3<f32>>,
    reason: Option<&'static str>,
}

impl Default for PassingEventBuilder {
    fn default() -> Self {
        PassingEventBuilder::new()
    }
}

impl PassingEventBuilder {
    pub fn new() -> Self {
        PassingEventBuilder {
            from_player_id: None,
            to_player_id: None,
            pass_force: None,
            pass_target: None,
            reason: None,
        }
    }

    pub fn with_from_player_id(mut self, from_player_id: u32) -> Self {
        self.from_player_id = Some(from_player_id);
        self
    }

    pub fn with_to_player_id(mut self, to_player_id: u32) -> Self {
        self.to_player_id = Some(to_player_id);
        self
    }

    pub fn with_pass_force(mut self, pass_force: f32) -> Self {
        self.pass_force = Some(pass_force);
        self
    }

    /// Override the ball target position.  For corner deliveries the ball is
    /// aimed at the zone centre, not the receiver's current feet — so the
    /// ball reaches the intended area regardless of where the runner is
    /// at the moment of the kick.
    pub fn with_pass_target(mut self, target: Vector3<f32>) -> Self {
        self.pass_target = Some(target);
        self
    }

    pub fn with_reason(mut self, reason: &'static str) -> Self {
        self.reason = Some(reason);
        self
    }

    pub fn build(self, ctx: &StateProcessingContext) -> PassingEventContext {
        let to_player_id = self.to_player_id.unwrap();

        PassingEventContext {
            from_player_id: self.from_player_id.unwrap(),
            to_player_id,
            pass_target: self.pass_target.unwrap_or_else(|| {
                ctx.tick_context.positions.players.position(to_player_id)
            }),
            pass_force: self
                .pass_force
                .unwrap_or_else(|| ctx.player().pass_teammate_power(to_player_id)),
            reason: self.reason.unwrap_or("No reason specified"),
        }
    }
}
