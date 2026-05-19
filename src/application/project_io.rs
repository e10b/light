use std::collections::HashMap;

use crate::{
    prism_file::{load_prism_database, save_prism_file, MaterialData as PrismMaterialData},
    scene_data::{Id, MainDatabase},
    tooling::{
        materials::{make_glass_material, make_white_material},
        persistence::build_prism_database_from_main,
    },
};

use super::types::SceneUniforms;

pub struct ProjectIoContext<'a> {
    pub main_db: &'a mut MainDatabase,
    pub decanter_master: &'a mut Id,
    pub wine_master: &'a mut Id,
    pub cornell_master: &'a mut Id,
    pub decanter_scene_id: &'a mut Id,
    pub wine_scene_id: &'a mut Id,
    pub cornell_scene_id: &'a mut Id,
    pub object_material_names: &'a mut HashMap<Id, String>,
    pub material_library: &'a mut HashMap<String, PrismMaterialData>,
    pub sphere_obj_id: Id,
    pub decanter_obj_id: Id,
    pub wine_obj_id: Id,
    pub cornell_obj_id: Id,
    pub sun_obj_id: Id,
    pub spot_obj_id: Id,
    pub uniforms: &'a mut SceneUniforms,
    pub sphere_rotation: &'a mut glam::Quat,
    pub sphere_scale: &'a mut glam::Vec3,
    pub sphere_radius: f32,
    pub decanter_center: glam::Vec3,
    pub decanter_translation: &'a mut glam::Vec3,
    pub decanter_rotation: &'a mut glam::Quat,
    pub decanter_scale: &'a mut glam::Vec3,
    pub wine_center: glam::Vec3,
    pub wine_translation: &'a mut glam::Vec3,
    pub wine_rotation: &'a mut glam::Quat,
    pub wine_scale: &'a mut glam::Vec3,
    pub sun_empty_position: &'a mut glam::Vec3,
    pub sun_empty_rotation: &'a mut glam::Quat,
    pub sun_empty_scale: &'a mut glam::Vec3,
    pub spot_empty_position: &'a mut glam::Vec3,
    pub spot_empty_rotation: &'a mut glam::Quat,
    pub spot_empty_scale: &'a mut glam::Vec3,
    pub geometry_dirty: &'a mut bool,
    pub accumulation_dirty: &'a mut bool,
    pub project_status: &'a mut String,
}

pub fn draw_project_io_buttons(ui: &mut egui::Ui, ctx: ProjectIoContext<'_>) {
    if ui.button("Open").clicked() {
        match load_prism_database(std::path::Path::new("res/scenes.prism"), false) {
            Ok(loaded) => {
                ctx.main_db.collections.clear();
                ctx.main_db.scenes.clear();
                ctx.main_db.view_layers.clear();
                *ctx.decanter_master = Id(0);
                *ctx.wine_master = Id(0);
                *ctx.cornell_master = Id(0);
                *ctx.decanter_scene_id = Id(0);
                *ctx.wine_scene_id = Id(0);
                *ctx.cornell_scene_id = Id(0);
                ctx.object_material_names.clear();
                ctx.material_library.clear();
                ctx.material_library
                    .insert("White".to_string(), make_white_material());
                ctx.material_library
                    .insert("Glass".to_string(), make_glass_material());
                ctx.object_material_names
                    .insert(ctx.sphere_obj_id, "Glass".to_string());
                ctx.object_material_names
                    .insert(ctx.decanter_obj_id, "Glass".to_string());
                ctx.object_material_names
                    .insert(ctx.wine_obj_id, "Glass".to_string());
                ctx.object_material_names
                    .insert(ctx.cornell_obj_id, "Empty".to_string());
                for (_mh, mat) in &loaded.materials {
                    ctx.material_library.insert(mat.name.clone(), mat.clone());
                }
                for (_sh, scene) in &loaded.scenes {
                    let scene_name = scene.name.to_ascii_lowercase();
                    let local_master = ctx
                        .main_db
                        .create_collection(format!("{}Master", scene.name));
                    let local_scene = ctx.main_db.create_scene(&scene.name, local_master);
                    if scene_name.contains("decanter") || scene_name == "scene" {
                        *ctx.decanter_master = local_master;
                        *ctx.decanter_scene_id = local_scene;
                    } else if scene_name.contains("wine") {
                        *ctx.wine_master = local_master;
                        *ctx.wine_scene_id = local_scene;
                    } else if scene_name.contains("cornell") {
                        *ctx.cornell_master = local_master;
                        *ctx.cornell_scene_id = local_scene;
                    }
                    if let Some(master_col) = loaded.collections.get(scene.master_collection) {
                        for obj_handle in &master_col.objects {
                            if let Some(obj) = loaded.objects.get(*obj_handle) {
                                let name = obj.name.to_ascii_lowercase();
                                let oid = if name.contains("decanter") {
                                    Some(ctx.decanter_obj_id)
                                } else if name.contains("wine") {
                                    Some(ctx.wine_obj_id)
                                } else if name.contains("spot") {
                                    Some(ctx.spot_obj_id)
                                } else if name.contains("sun") {
                                    Some(ctx.sun_obj_id)
                                } else if name.contains("cornell") {
                                    Some(ctx.cornell_obj_id)
                                } else if name.contains("sphere") || name.contains("cube") {
                                    Some(ctx.sphere_obj_id)
                                } else {
                                    None
                                };
                                if let Some(local_obj_id) = oid {
                                    ctx.main_db
                                        .collection_link_object(local_master, local_obj_id);
                                    ctx.main_db.ensure_scene_base(
                                        local_scene,
                                        local_obj_id,
                                        true,
                                        true,
                                    );
                                }
                            }
                        }
                    }
                }
                for (_oh, obj) in &loaded.objects {
                    let m = glam::Mat4::from_cols_array(&obj.transform_matrix);
                    let (s, r, t) = m.to_scale_rotation_translation();
                    let lname = obj.name.to_ascii_lowercase();
                    if lname.contains("sphere") || lname.contains("cube") {
                        ctx.uniforms.sphere_pos[0] = t.x;
                        ctx.uniforms.sphere_pos[1] = t.y;
                        ctx.uniforms.sphere_pos[2] = t.z;
                        *ctx.sphere_rotation = r;
                        ctx.uniforms.sphere_rot = [
                            ctx.sphere_rotation.x,
                            ctx.sphere_rotation.y,
                            ctx.sphere_rotation.z,
                            ctx.sphere_rotation.w,
                        ];
                        *ctx.sphere_scale = s.max(glam::Vec3::splat(0.01));
                        ctx.uniforms.sphere_extent = [
                            ctx.sphere_radius * ctx.sphere_scale.x,
                            ctx.sphere_radius * ctx.sphere_scale.y,
                            ctx.sphere_radius * ctx.sphere_scale.z,
                            0.0,
                        ];
                    } else if lname.contains("decanter") {
                        *ctx.decanter_translation = t - ctx.decanter_center;
                        *ctx.decanter_rotation = r;
                        *ctx.decanter_scale = s;
                        *ctx.geometry_dirty = true;
                    } else if lname.contains("wine") {
                        *ctx.wine_translation = t - ctx.wine_center;
                        *ctx.wine_rotation = r;
                        *ctx.wine_scale = s;
                        *ctx.geometry_dirty = true;
                    } else if lname.contains("sun") {
                        *ctx.sun_empty_position = t;
                        *ctx.sun_empty_rotation = r;
                        *ctx.sun_empty_scale = s;
                    } else if lname.contains("spot") {
                        *ctx.spot_empty_position = t;
                        *ctx.spot_empty_rotation = r;
                        *ctx.spot_empty_scale = s;
                    }
                    if let Some(mh) = obj.material_link {
                        if let Some(mat) = loaded.materials.get(mh) {
                            ctx.material_library.insert(mat.name.clone(), mat.clone());
                            let target_id = if lname.contains("decanter") {
                                Some(ctx.decanter_obj_id)
                            } else if lname.contains("wine") {
                                Some(ctx.wine_obj_id)
                            } else if lname.contains("sphere") || lname.contains("cube") {
                                Some(ctx.sphere_obj_id)
                            } else if lname.contains("cornell") {
                                Some(ctx.cornell_obj_id)
                            } else {
                                None
                            };
                            if let Some(tid) = target_id {
                                ctx.object_material_names.insert(tid, mat.name.clone());
                            }
                        }
                    }
                }
                *ctx.accumulation_dirty = true;
                *ctx.project_status = "Opened: res/scenes.prism".to_string();
            }
            Err(e) => *ctx.project_status = format!("Open failed (res/scenes.prism): {e}"),
        }
    }

    if ui.button("Save").clicked() {
        let prism_db = build_prism_database_from_main(
            ctx.main_db,
            *ctx.decanter_scene_id,
            *ctx.wine_scene_id,
            *ctx.cornell_scene_id,
            ctx.object_material_names,
            ctx.material_library,
        );
        match save_prism_file(std::path::Path::new("res/scenes.prism"), &prism_db, false) {
            Ok(_) => *ctx.project_status = "Saved: res/scenes.prism".to_string(),
            Err(e) => *ctx.project_status = format!("Save failed: {e}"),
        }
    }
}
