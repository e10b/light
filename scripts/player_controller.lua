return {
  on_start = function(entity)
    entity:log("player_controller.lua active: WASD moves, mouse looks, Escape exits Play")
  end,
  on_update = function(entity, dt)
    local velocity = entity:velocity()
    local gravity = -24.0
    entity:set_velocity(velocity.x, velocity.y + gravity * dt, velocity.z)
  end,
}
