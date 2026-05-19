use crate::{
    application::types::{LightObjectInstance, SceneUniforms, MAX_SUN_LIGHTS},
    scene_data::{Id, MainDatabase},
};

#[allow(clippy::too_many_arguments)]
pub fn update_sun_lights(
    uniforms: &mut SceneUniforms,
    main_db: &MainDatabase,
    decanter_scene_id: Id,
    current_scene_exists: bool,
    active_center: glam::Vec3,
    sun_lamp_pos: glam::Vec3,
    primary_sun_intensity: f32,
    light_instances: &[LightObjectInstance],
    sun_azimuth_deg: &mut f32,
    sun_elevation_deg: &mut f32,
) {
    uniforms.light_pos = {
        let d = (sun_lamp_pos - active_center).normalize_or_zero();
        *sun_azimuth_deg = d.z.atan2(d.x).to_degrees();
        let len_xz = (d.x * d.x + d.z * d.z).sqrt().max(1e-5);
        *sun_elevation_deg = d.y.atan2(len_xz).to_degrees();
        [d.x, d.y, d.z, primary_sun_intensity]
    };
    uniforms.sun_lights = [[0.0, 0.0, 0.0, 0.0]; MAX_SUN_LIGHTS];
    uniforms.sun_light_count = 0;
    if current_scene_exists {
        let visible = main_db.scene_visible_selectable_objects(decanter_scene_id);
        for light in light_instances
            .iter()
            .filter(|light| visible.contains(&light.object_id))
            .take(MAX_SUN_LIGHTS)
        {
            let d = (light.position - active_center).normalize_or_zero();
            let idx = uniforms.sun_light_count as usize;
            uniforms.sun_lights[idx] = [d.x, d.y, d.z, light.intensity.max(0.0)];
            uniforms.sun_light_count += 1;
        }
        if uniforms.sun_light_count == 0 {
            let d = (sun_lamp_pos - active_center).normalize_or_zero();
            uniforms.sun_lights[0] = [d.x, d.y, d.z, primary_sun_intensity];
            uniforms.sun_light_count = 1;
        }
    }
}
