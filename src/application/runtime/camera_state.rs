use crate::{
    application::types::{Camera, LightObjectInstance, MeshObjectInstance},
    scene_data::{Id, MainDatabase},
};

#[allow(clippy::too_many_arguments)]
pub fn write_runtime_back_to_database(
    main_db: &mut MainDatabase,
    sphere_obj_id: Id,
    decanter_obj_id: Id,
    wine_obj_id: Id,
    sun_obj_id: Id,
    spot_obj_id: Id,
    cornell_obj_id: Id,
    uniforms: &crate::application::types::SceneUniforms,
    sphere_rotation: glam::Quat,
    sphere_scale: glam::Vec3,
    decanter_center: glam::Vec3,
    decanter_translation: glam::Vec3,
    decanter_rotation: glam::Quat,
    decanter_scale: glam::Vec3,
    wine_center: glam::Vec3,
    wine_translation: glam::Vec3,
    wine_rotation: glam::Quat,
    wine_scale: glam::Vec3,
    stress_instance_count: usize,
    mesh_instances: &[MeshObjectInstance],
    sun_empty_position: glam::Vec3,
    sun_empty_rotation: glam::Quat,
    sun_empty_scale: glam::Vec3,
    light_instances: &[LightObjectInstance],
    spot_empty_position: glam::Vec3,
    spot_empty_rotation: glam::Quat,
    spot_empty_scale: glam::Vec3,
    active_center: glam::Vec3,
    cornell_translation: glam::Vec3,
    cornell_rotation: glam::Quat,
    cornell_scale: glam::Vec3,
) {
    if let Some(obj) = main_db.objects.get_mut(&sphere_obj_id) {
        obj.transform.location = glam::Vec3::new(
            uniforms.sphere_pos[0],
            uniforms.sphere_pos[1],
            uniforms.sphere_pos[2],
        );
        obj.transform.rotation = sphere_rotation;
        obj.transform.scale = sphere_scale;
    }
    if let Some(obj) = main_db.objects.get_mut(&decanter_obj_id) {
        obj.transform.location = decanter_center + decanter_translation;
        obj.transform.rotation = decanter_rotation;
        obj.transform.scale = decanter_scale;
    }
    if let Some(obj) = main_db.objects.get_mut(&wine_obj_id) {
        obj.transform.location = wine_center + wine_translation;
        obj.transform.rotation = wine_rotation;
        obj.transform.scale = wine_scale;
    }
    if stress_instance_count == 0 {
        for inst in mesh_instances {
            if let Some(obj) = main_db.objects.get_mut(&inst.object_id) {
                obj.transform.location = inst.center();
                obj.transform.rotation = inst.rotation;
                obj.transform.scale = inst.scale;
            }
        }
    }
    if let Some(obj) = main_db.objects.get_mut(&sun_obj_id) {
        obj.transform.location = sun_empty_position;
        obj.transform.rotation = sun_empty_rotation;
        obj.transform.scale = sun_empty_scale;
    }
    for light in light_instances {
        if let Some(obj) = main_db.objects.get_mut(&light.object_id) {
            obj.transform.location = light.position;
            obj.transform.rotation = light.rotation;
            obj.transform.scale = light.scale;
        }
    }
    if let Some(obj) = main_db.objects.get_mut(&spot_obj_id) {
        obj.transform.location = spot_empty_position;
        obj.transform.rotation = spot_empty_rotation;
        obj.transform.scale = spot_empty_scale;
    }
    if let Some(obj) = main_db.objects.get_mut(&cornell_obj_id) {
        obj.transform.location = active_center + cornell_translation;
        obj.transform.rotation = cornell_rotation;
        obj.transform.scale = cornell_scale;
    }
}
