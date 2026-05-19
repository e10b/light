use crate::{
    ecs::{TransformComponent, World},
    scene_data::{Id, MainDatabase, Transform as DbTransform},
};

use super::types::{Camera, LightObjectInstance, MeshObjectInstance};

pub fn db_transform_to_ecs(transform: &DbTransform) -> TransformComponent {
    TransformComponent {
        translation: transform.location,
        rotation: transform.rotation,
        scale: transform.scale,
    }
}

pub fn register_object_entity(world: &mut World, main_db: &MainDatabase, object_id: Id) {
    if let Some(obj) = main_db.objects.get(&object_id) {
        world.register_existing(
            object_id,
            obj.name.clone(),
            db_transform_to_ecs(&obj.transform),
        );
        if obj.mesh_id.is_some() {
            world.attach_mesh(object_id, 0);
        }
    }
}

pub fn sync_ecs_to_runtime(
    world: &World,
    main_db: &mut MainDatabase,
    mesh_instances: &mut [MeshObjectInstance],
    light_instances: &mut [LightObjectInstance],
    camera: &mut Camera,
) -> bool {
    let mut geometry_changed = false;

    for (object_id, transform) in &world.global_transforms {
        if let Some(obj) = main_db.objects.get_mut(object_id) {
            obj.transform.location = transform.translation;
            obj.transform.rotation = transform.rotation;
            obj.transform.scale = transform.scale;
        }
        if let Some(inst) = mesh_instances
            .iter_mut()
            .find(|inst| inst.object_id == *object_id)
        {
            let next_translation = transform.translation - inst.pivot;
            if inst.translation != next_translation
                || inst.rotation != transform.rotation
                || inst.scale != transform.scale
            {
                inst.translation = next_translation;
                inst.rotation = transform.rotation;
                inst.scale = transform.scale;
                geometry_changed = true;
            }
        }
        if let Some(light) = light_instances
            .iter_mut()
            .find(|light| light.object_id == *object_id)
        {
            light.position = transform.translation;
            light.rotation = transform.rotation;
            light.scale = transform.scale;
        }
    }

    if let Some((object_id, _)) = world.cameras.iter().find(|(_, camera)| camera.active) {
        if let Some(transform) = world.global_transforms.get(object_id) {
            camera.pos = transform.translation;
        }
    }

    geometry_changed
}

pub fn sync_ecs_visibility_to_main(world: &World, main_db: &mut MainDatabase) {
    let scene_ids: Vec<Id> = main_db.scenes.keys().copied().collect();
    for scene_id in scene_ids {
        for entity in &world.entities {
            main_db.set_scene_base_visibility(scene_id, entity.id, world.is_visible(entity.id));
        }
    }
}

pub fn sync_runtime_to_ecs(
    world: &mut World,
    main_db: &MainDatabase,
    mesh_instances: &[MeshObjectInstance],
    light_instances: &[LightObjectInstance],
    camera: &Camera,
) {
    for entity in world.entities.clone() {
        if let Some(obj) = main_db.objects.get(&entity.id) {
            world.transforms.entry(entity.id).or_default().translation = obj.transform.location;
            if let Some(transform) = world.transforms.get_mut(&entity.id) {
                transform.rotation = obj.transform.rotation;
                transform.scale = obj.transform.scale;
            }
        }
    }

    for inst in mesh_instances {
        if let Some(transform) = world.transforms.get_mut(&inst.object_id) {
            transform.translation = inst.center();
            transform.rotation = inst.rotation;
            transform.scale = inst.scale;
        }
    }

    for light in light_instances {
        if let Some(transform) = world.transforms.get_mut(&light.object_id) {
            transform.translation = light.position;
            transform.rotation = light.rotation;
            transform.scale = light.scale;
        }
    }

    let active_camera_id = world
        .cameras
        .iter()
        .find_map(|(id, camera)| camera.active.then_some(*id));
    if let Some(id) = active_camera_id {
        if world
            .cameras
            .get(&id)
            .and_then(|camera| camera.attached_to)
            .is_some()
        {
            return;
        }
        if let Some(transform) = world.transforms.get_mut(&id) {
            transform.translation = camera.pos;
        }
    }
}
