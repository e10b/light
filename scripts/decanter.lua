local state = {
    t = 0.0,
}

return {
    on_start = function(entity)
        entity:log("Decanter script attached")
    end,

    on_update = function(entity, dt)
        state.t = state.t + dt
    end,
}
