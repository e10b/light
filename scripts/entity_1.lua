local state = {
    t = 0.0,
}

return {
    on_start = function(entity)
        entity:log("Building cube from blank entity: Transform + Mesh3d + Material3d")
        if not entity:has_transform() then
            entity:attach_transform(0.0, 3.0, 0.0)
        end
        entity:attach_mesh3d(0)
        entity:attach_material3d(0, 1.0, 0.1, 0.08, 1.0, 0.85, 0.0, 0.0)
        entity:show()
    end,

    on_update = function(entity, dt)
        state.t = state.t + dt
        entity:set_rotation_euler(0.0, state.t, 0.0)
    end,
}
