return {
  on_start = function(entity)
    entity:log("player_controller.lua active: WASD moves, mouse looks, Escape exits Play")
  end,
  on_update = function(entity, dt)
    local move_speed = 8.0
    local gravity = -24.0

    local velocity = entity:velocity()
    local forward = entity:forward_vector()
    local right = entity:right_vector()

    local move_x = 0.0
    local move_z = 0.0

    if entity:is_key_down("w") or entity:is_key_down("W") then
      move_z = move_z + 1.0
    end
    if entity:is_key_down("s") or entity:is_key_down("S") then
      move_z = move_z - 1.0
    end
    if entity:is_key_down("d") or entity:is_key_down("D") then
      move_x = move_x + 1.0
    end
    if entity:is_key_down("a") or entity:is_key_down("A") then
      move_x = move_x - 1.0
    end

    local target_vx = (forward.x * move_z + right.x * move_x) * move_speed
    local target_vz = (forward.z * move_z + right.z * move_x) * move_speed

    local mouse_dx, mouse_dy = entity:get_mouse_delta()
    if mouse_dx ~= 0 or mouse_dy ~= 0 then
      local sensitivity = 0.003
      entity:add_rotation(mouse_dx * sensitivity, mouse_dy * sensitivity)
    end

    local new_vy = velocity.y + (gravity * dt)
    local current_pos = entity:position()
    if current_pos.y <= -1.5 and new_vy < 0.0 then
      new_vy = 0.0
    end

    entity:set_velocity(target_vx, new_vy, target_vz)
  end,
}
