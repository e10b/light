local state = {
    t = 0.0,
}

return {
    on_start = function(entity)
        entity:log("test_lights.lua attaching all light component types")
        entity:attach_point_light(1.0, 0.72, 0.35, 2.5, 16.0)
        entity:attach_directional_light(1.0, 0.95, 0.82, 1.0)
        entity:attach_spot_light(0.7, 0.9, 1.0, 3.0, 18.0, 0.25, 0.65)
    end,

    on_update = function(entity, dt)
        state.t = state.t + dt
        entity:set_position(math.cos(state.t) * 8.0, 8.0 + math.sin(state.t * 2.0), math.sin(state.t) * 8.0)
    end,
}
