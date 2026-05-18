local state = {
    t = 0.0,
    visible = true,
}

return {
    on_start = function(entity)
        entity:log("test_visibility.lua toggling visibility")
        entity:show()
    end,

    on_update = function(entity, dt)
        state.t = state.t + dt
        if state.t > 1.0 then
            state.t = 0.0
            state.visible = not state.visible
            entity:set_visible(state.visible)
            entity:log("visible = " .. tostring(entity:is_visible()))
        end
    end,
}
