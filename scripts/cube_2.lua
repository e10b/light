local state = {
    t = 0.0,
}

return {
    on_start = function(entity)
        entity:log("Cube 2 script attached\")
    end,

    on_update = function(entity, dt)
        state.t = state.t + dt
        entity:set_rotation_euler(0.0, state.t * 1.5, 0.0)
    end,
}
