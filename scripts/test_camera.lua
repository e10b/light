local state = {
    t = 0.0,
}

return {
    on_start = function(entity)
        entity:log("test_camera.lua attaching Camera")
        entity:attach_camera()
    end,

    on_update = function(entity, dt)
        state.t = state.t + dt
        entity:set_position(math.sin(state.t) * 8.0, 4.0, 12.0 + math.cos(state.t) * 4.0)
        entity:set_rotation_euler(0.0, state.t * 0.25, 0.0)
    end,
}
