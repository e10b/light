local state = {
    t = 0.0,
}

return {
    on_start = function(entity)
        entity:log("test_components.lua attaching Mesh3d, Material3d, Physics, and PointLight")
        entity:attach_mesh3d(0)
        entity:attach_material3d(0, 0.2, 0.75, 1.0, 1.0, 0.28, 0.0, 0.0)
        entity:attach_physics()
        entity:attach_point_light(0.6, 0.85, 1.0, 1.8, 12.0)
    end,

    on_update = function(entity, dt)
        state.t = state.t + dt
        entity:set_rotation_euler(0.0, state.t * 1.2, 0.0)
        entity:set_scale(
            1.0 + math.sin(state.t * 2.0) * 0.15,
            1.0 + math.sin(state.t * 2.0) * 0.15,
            1.0 + math.sin(state.t * 2.0) * 0.15
        )
    end,
}
