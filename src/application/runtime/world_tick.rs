use std::{cell::RefCell, rc::Rc};

use crate::{
    application::types::{Camera, LightObjectInstance, MeshObjectInstance},
    ecs::{ScriptEngine, World},
    scene_data::MainDatabase,
};

#[allow(clippy::too_many_arguments)]
pub fn tick_world_and_scripts(
    dt: f32,
    world: &Rc<RefCell<World>>,
    main_db: &mut MainDatabase,
    mesh_instances: &mut [MeshObjectInstance],
    light_instances: &mut [LightObjectInstance],
    camera: &mut Camera,
    script_engine: &mut ScriptEngine,
) -> bool {
    {
        use crate::application::ecs_sync::sync_runtime_to_ecs;
        let mut world_mut = world.borrow_mut();
        sync_runtime_to_ecs(
            &mut world_mut,
            main_db,
            mesh_instances,
            light_instances,
            camera,
        );
        world_mut.update_global_transforms_and_visibility();
    }

    script_engine.update(dt);

    {
        use crate::application::ecs_sync::sync_ecs_visibility_to_main;
        let mut world_mut = world.borrow_mut();
        world_mut.integrate_physics(dt);
        world_mut.resolve_collisions();
        world_mut.update_global_transforms_and_visibility();
        sync_ecs_visibility_to_main(&world_mut, main_db);
    }

    use crate::application::ecs_sync::sync_ecs_to_runtime;
    let world_ref = world.borrow();
    sync_ecs_to_runtime(&world_ref, main_db, mesh_instances, light_instances, camera)
}
