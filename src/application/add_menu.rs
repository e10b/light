use std::{
    cell::RefCell,
    collections::HashMap,
    path::Path,
    rc::Rc,
};

use crate::{
    ecs::{CameraComponent, World},
    mesh::{load_gltf_mesh, MeshData},
    scene::SceneKind,
    scene_data::{Id, MainDatabase, Transform as DbTransform},
};

use super::{
    ecs_sync::register_object_entity,
    geometry::{
        append_object_mesh, make_cube_mesh, make_prism_mesh, mesh_bounds, orient_and_scale_mesh,
        place_instance_center, translate_mesh,
    },
    types::{GizmoTargetKind, LightObjectInstance, MeshObjectInstance},
};

pub struct AddMenuContext<'a> {
    pub scene_kind: SceneKind,
    pub decanter_scene_id: Id,
    pub decanter_master: Id,
    pub wine_master: Id,
    pub cornell_master: Id,
    pub wine_obj_id: Id,
    pub spot_obj_id: Id,
    pub cornell_obj_id: Id,
    pub sphere_obj_id: Id,
    pub active_center: glam::Vec3,
    pub wine_center: glam::Vec3,
    pub default_cube_mesh_id: Id,
    pub cornell_mesh_id: Id,
    pub camera_pos: glam::Vec3,
    pub sun_intensity: f32,
    pub decanter_path: &'a Path,
    pub wine_path: &'a Path,
    pub main_db: &'a mut MainDatabase,
    pub ecs_world: &'a Rc<RefCell<World>>,
    pub mesh: &'a mut MeshData,
    pub mesh_instances: &'a mut Vec<MeshObjectInstance>,
    pub light_instances: &'a mut Vec<LightObjectInstance>,
    pub object_target_by_id: &'a mut HashMap<Id, GizmoTargetKind>,
    pub object_material_names: &'a mut HashMap<Id, String>,
    pub model_idx: &'a mut Vec<u32>,
    pub selected_object_id: &'a mut Option<Id>,
    pub gizmo_target: &'a mut GizmoTargetKind,
    pub has_selection: &'a mut bool,
    pub gpu_mesh_dirty: &'a mut bool,
    pub geometry_dirty: &'a mut bool,
    pub accumulation_dirty: &'a mut bool,
    pub sun_empty_position: &'a mut glam::Vec3,
    pub project_status: &'a mut String,
    pub suppress_scene_click: &'a mut bool,
}

pub fn draw_add_menu(ui: &mut egui::Ui, ctx: AddMenuContext<'_>) {
    ui.menu_button("Add", |ui| {
        let scene_id = ctx.decanter_scene_id;
        match ctx.scene_kind {
            SceneKind::Decanter => {
                if ui.button("Empty Entity").clicked() {
                    let count = ctx
                        .main_db
                        .objects
                        .values()
                        .filter(|o| o.name.starts_with("Entity"))
                        .count()
                        + 1;
                    let obj_id = ctx.main_db.create_object(
                        format!("Entity {count}"),
                        None,
                        DbTransform::default(),
                    );
                    ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                    ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                    register_object_entity(&mut ctx.ecs_world.borrow_mut(), ctx.main_db, obj_id);
                    ctx.object_target_by_id
                        .insert(obj_id, GizmoTargetKind::Decanter);
                    *ctx.selected_object_id = Some(obj_id);
                    *ctx.gizmo_target = GizmoTargetKind::Decanter;
                    *ctx.has_selection = true;
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Camera").clicked() {
                    let count = ctx
                        .main_db
                        .objects
                        .values()
                        .filter(|o| o.name.starts_with("Camera"))
                        .count()
                        + 1;
                    let obj_id = ctx.main_db.create_object(
                        format!("Camera {count}"),
                        None,
                        DbTransform {
                            location: ctx.camera_pos,
                            rotation: glam::Quat::IDENTITY,
                            scale: glam::Vec3::ONE,
                        },
                    );
                    ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                    ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                    {
                        let mut world = ctx.ecs_world.borrow_mut();
                        register_object_entity(&mut world, ctx.main_db, obj_id);
                        world.attach_camera(obj_id, CameraComponent::default());
                    }
                    ctx.object_target_by_id
                        .insert(obj_id, GizmoTargetKind::Decanter);
                    *ctx.selected_object_id = Some(obj_id);
                    *ctx.gizmo_target = GizmoTargetKind::Decanter;
                    *ctx.has_selection = true;
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Cube").clicked() {
                    let count = ctx
                        .main_db
                        .objects
                        .values()
                        .filter(|o| o.name.starts_with("Cube"))
                        .count()
                        + 1;
                    let cube_center = ctx.active_center + glam::Vec3::new(count as f32 * 4.0, 0.5, 0.0);
                    let cube_mesh = make_cube_mesh(glam::Vec3::ZERO, 1.5);
                    let mesh_id = ctx.default_cube_mesh_id;
                    if let Some(mesh_db) = ctx.main_db.meshes.get_mut(&mesh_id) {
                        mesh_db.user_count = mesh_db.user_count.saturating_add(1);
                    }
                    let obj_id =
                        ctx.main_db
                            .create_object(format!("Cube {count}"), Some(mesh_id), DbTransform::default());
                    let mut inst = append_object_mesh(ctx.mesh, cube_mesh, obj_id, 0);
                    place_instance_center(&mut inst, cube_center);
                    ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                    ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                    register_object_entity(&mut ctx.ecs_world.borrow_mut(), ctx.main_db, obj_id);
                    ctx.object_target_by_id
                        .insert(obj_id, GizmoTargetKind::Decanter);
                    ctx.object_material_names
                        .insert(obj_id, "Glass".to_string());
                    ctx.mesh_instances.push(inst);
                    *ctx.model_idx = ctx.mesh.indices.clone();
                    *ctx.selected_object_id = Some(obj_id);
                    *ctx.gizmo_target = GizmoTargetKind::Decanter;
                    *ctx.has_selection = true;
                    *ctx.gpu_mesh_dirty = true;
                    *ctx.geometry_dirty = true;
                    *ctx.accumulation_dirty = true;
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Prism (Glass)").clicked() {
                    let count = ctx
                        .main_db
                        .objects
                        .values()
                        .filter(|o| o.name.starts_with("Prism"))
                        .count()
                        + 1;
                    let prism_center = ctx.active_center + glam::Vec3::new(count as f32 * 4.0, 1.0, -2.0);
                    let prism_mesh = make_prism_mesh(glam::Vec3::ZERO, 1.25, 3.2);
                    let mesh_id = ctx
                        .main_db
                        .create_mesh(format!("PrismMesh{count}"), prism_mesh.vertices.len());
                    let obj_id = ctx.main_db.create_object(
                        format!("Prism {count}"),
                        Some(mesh_id),
                        DbTransform::default(),
                    );
                    let mut inst = append_object_mesh(ctx.mesh, prism_mesh, obj_id, 5);
                    place_instance_center(&mut inst, prism_center);
                    ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                    ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                    register_object_entity(&mut ctx.ecs_world.borrow_mut(), ctx.main_db, obj_id);
                    ctx.object_target_by_id
                        .insert(obj_id, GizmoTargetKind::Decanter);
                    ctx.object_material_names
                        .insert(obj_id, "Glass".to_string());
                    ctx.mesh_instances.push(inst);
                    *ctx.model_idx = ctx.mesh.indices.clone();
                    *ctx.selected_object_id = Some(obj_id);
                    *ctx.gizmo_target = GizmoTargetKind::Decanter;
                    *ctx.has_selection = true;
                    *ctx.gpu_mesh_dirty = true;
                    *ctx.geometry_dirty = true;
                    *ctx.accumulation_dirty = true;
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Sun Lamp").clicked() {
                    let count = ctx
                        .main_db
                        .objects
                        .values()
                        .filter(|o| o.name.starts_with("Sun Lamp"))
                        .count()
                        + 1;
                    let obj_id = ctx.main_db.create_object(
                        format!("Sun Lamp {count}"),
                        None,
                        DbTransform::default(),
                    );
                    let pos = ctx.active_center + glam::Vec3::new(count as f32 * 3.0, 8.0, 10.0);
                    ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                    ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                    register_object_entity(&mut ctx.ecs_world.borrow_mut(), ctx.main_db, obj_id);
                    ctx.object_target_by_id
                        .insert(obj_id, GizmoTargetKind::SunLamp);
                    ctx.light_instances.push(LightObjectInstance {
                        object_id: obj_id,
                        position: pos,
                        rotation: glam::Quat::IDENTITY,
                        scale: glam::Vec3::ONE,
                        intensity: ctx.sun_intensity,
                    });
                    ctx.ecs_world
                        .borrow_mut()
                        .attach_light(obj_id, ctx.sun_intensity);
                    *ctx.selected_object_id = Some(obj_id);
                    *ctx.gizmo_target = GizmoTargetKind::SunLamp;
                    *ctx.has_selection = true;
                    *ctx.sun_empty_position = pos;
                    *ctx.accumulation_dirty = true;
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Decanter").clicked() {
                    let count = ctx
                        .main_db
                        .objects
                        .values()
                        .filter(|o| o.name.starts_with("Decanter"))
                        .count()
                        + 1;
                    match load_gltf_mesh(ctx.decanter_path) {
                        Ok(mut new_mesh) => {
                            translate_mesh(
                                &mut new_mesh,
                                glam::Vec3::new(count as f32 * 8.0, 0.0, 0.0),
                            );
                            let mesh_id = ctx
                                .main_db
                                .create_mesh("DecanterMesh", new_mesh.vertices.len());
                            let obj_id = ctx.main_db.create_object(
                                format!("Decanter {count}"),
                                Some(mesh_id),
                                DbTransform::default(),
                            );
                            let inst = append_object_mesh(ctx.mesh, new_mesh, obj_id, 1);
                            ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                            ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                            register_object_entity(&mut ctx.ecs_world.borrow_mut(), ctx.main_db, obj_id);
                            if let Some(transform) =
                                ctx.ecs_world.borrow_mut().transforms.get_mut(&obj_id)
                            {
                                transform.translation = inst.center();
                                transform.rotation = inst.rotation;
                                transform.scale = inst.scale;
                            }
                            ctx.object_target_by_id
                                .insert(obj_id, GizmoTargetKind::Decanter);
                            ctx.object_material_names
                                .insert(obj_id, "Glass".to_string());
                            ctx.mesh_instances.push(inst);
                            *ctx.model_idx = ctx.mesh.indices.clone();
                            *ctx.selected_object_id = Some(obj_id);
                            *ctx.gizmo_target = GizmoTargetKind::Decanter;
                            *ctx.has_selection = true;
                            *ctx.gpu_mesh_dirty = true;
                            *ctx.geometry_dirty = true;
                            *ctx.accumulation_dirty = true;
                        }
                        Err(e) => *ctx.project_status = format!("Add decanter failed: {e}"),
                    }
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Wine Glass").clicked() {
                    let count = ctx
                        .main_db
                        .objects
                        .values()
                        .filter(|o| o.name.starts_with("Wine Glass"))
                        .count()
                        + 1;
                    match load_gltf_mesh(ctx.wine_path) {
                        Ok(mut new_mesh) => {
                            let (c0, _, _, _) = mesh_bounds(&new_mesh.vertices);
                            orient_and_scale_mesh(
                                &mut new_mesh,
                                c0,
                                glam::Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                                25.0,
                            );
                            translate_mesh(
                                &mut new_mesh,
                                ctx.wine_center - c0 + glam::Vec3::new(count as f32 * 8.0, 0.0, 0.0),
                            );
                            let mesh_id = ctx
                                .main_db
                                .create_mesh("WineGlassMesh", new_mesh.vertices.len());
                            let obj_id = ctx.main_db.create_object(
                                format!("Wine Glass {count}"),
                                Some(mesh_id),
                                DbTransform::default(),
                            );
                            let inst = append_object_mesh(ctx.mesh, new_mesh, obj_id, 2);
                            ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                            ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                            register_object_entity(&mut ctx.ecs_world.borrow_mut(), ctx.main_db, obj_id);
                            if let Some(transform) =
                                ctx.ecs_world.borrow_mut().transforms.get_mut(&obj_id)
                            {
                                transform.translation = inst.center();
                                transform.rotation = inst.rotation;
                                transform.scale = inst.scale;
                            }
                            ctx.object_target_by_id
                                .insert(obj_id, GizmoTargetKind::WineGlass);
                            ctx.object_material_names
                                .insert(obj_id, "Glass".to_string());
                            ctx.mesh_instances.push(inst);
                            *ctx.model_idx = ctx.mesh.indices.clone();
                            *ctx.selected_object_id = Some(obj_id);
                            *ctx.gizmo_target = GizmoTargetKind::WineGlass;
                            *ctx.has_selection = true;
                            *ctx.gpu_mesh_dirty = true;
                            *ctx.geometry_dirty = true;
                            *ctx.accumulation_dirty = true;
                        }
                        Err(e) => *ctx.project_status = format!("Add wine glass failed: {e}"),
                    }
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Import GLB...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("glTF Binary", &["glb"])
                        .pick_file()
                    {
                        match load_gltf_mesh(&path) {
                            Ok(mut imported) => {
                                let (c0, size0, _, _) = mesh_bounds(&imported.vertices);
                                let fit = 8.0 / size0.max_element().max(0.001);
                                orient_and_scale_mesh(
                                    &mut imported,
                                    c0,
                                    glam::Quat::IDENTITY,
                                    fit,
                                );
                                let (c1, _, _, _) = mesh_bounds(&imported.vertices);
                                let offset_index = ctx.mesh_instances.len() as f32;
                                translate_mesh(
                                    &mut imported,
                                    ctx.active_center + glam::Vec3::new(offset_index * 4.0, 0.0, 0.0)
                                        - c1,
                                );
                                let stem = path
                                    .file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("Imported");
                                let mesh_id = ctx
                                    .main_db
                                    .create_mesh(format!("{stem}Mesh"), imported.vertices.len());
                                let obj_id = ctx.main_db.create_object(
                                    stem.to_string(),
                                    Some(mesh_id),
                                    DbTransform::default(),
                                );
                                let inst = append_object_mesh(ctx.mesh, imported, obj_id, 3);
                                ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                                ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                                register_object_entity(&mut ctx.ecs_world.borrow_mut(), ctx.main_db, obj_id);
                                if let Some(transform) =
                                    ctx.ecs_world.borrow_mut().transforms.get_mut(&obj_id)
                                {
                                    transform.translation = inst.center();
                                    transform.rotation = inst.rotation;
                                    transform.scale = inst.scale;
                                }
                                ctx.object_target_by_id
                                    .insert(obj_id, GizmoTargetKind::Decanter);
                                ctx.object_material_names
                                    .insert(obj_id, "Glass".to_string());
                                ctx.mesh_instances.push(inst);
                                *ctx.model_idx = ctx.mesh.indices.clone();
                                *ctx.selected_object_id = Some(obj_id);
                                *ctx.gizmo_target = GizmoTargetKind::Decanter;
                                *ctx.has_selection = true;
                                *ctx.gpu_mesh_dirty = true;
                                *ctx.geometry_dirty = true;
                                *ctx.accumulation_dirty = true;
                                *ctx.project_status = format!("Imported: {}", path.display());
                            }
                            Err(e) => {
                                *ctx.project_status = format!(
                                    "Import failed ({}): {e}",
                                    path.display()
                                );
                            }
                        }
                    }
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Cornell Box").clicked() {
                    let count = ctx
                        .main_db
                        .objects
                        .values()
                        .filter(|o| o.name.starts_with("Cornell Box"))
                        .count()
                        + 1;
                    let box_center = ctx.active_center + glam::Vec3::new(count as f32 * 4.0, 1.0, -2.0);
                    let box_mesh = make_cube_mesh(glam::Vec3::ZERO, 2.0);
                    let mesh_id = ctx.cornell_mesh_id;
                    if let Some(mesh_db) = ctx.main_db.meshes.get_mut(&mesh_id) {
                        mesh_db.user_count = mesh_db.user_count.saturating_add(1);
                    }
                    let obj_id = ctx.main_db.create_object(
                        format!("Cornell Box {count}"),
                        Some(mesh_id),
                        DbTransform::default(),
                    );
                    let mut inst = append_object_mesh(ctx.mesh, box_mesh, obj_id, 4);
                    place_instance_center(&mut inst, box_center);
                    ctx.main_db.collection_link_object(ctx.decanter_master, obj_id);
                    ctx.main_db.ensure_scene_base(scene_id, obj_id, true, true);
                    register_object_entity(&mut ctx.ecs_world.borrow_mut(), ctx.main_db, obj_id);
                    if let Some(transform) = ctx.ecs_world.borrow_mut().transforms.get_mut(&obj_id) {
                        transform.translation = inst.center();
                        transform.rotation = inst.rotation;
                        transform.scale = inst.scale;
                    }
                    ctx.object_target_by_id
                        .insert(obj_id, GizmoTargetKind::Decanter);
                    ctx.object_material_names.insert(obj_id, "Empty".to_string());
                    ctx.mesh_instances.push(inst);
                    *ctx.model_idx = ctx.mesh.indices.clone();
                    *ctx.selected_object_id = Some(obj_id);
                    *ctx.gizmo_target = GizmoTargetKind::Decanter;
                    *ctx.has_selection = true;
                    *ctx.gpu_mesh_dirty = true;
                    *ctx.geometry_dirty = true;
                    *ctx.accumulation_dirty = true;
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
            }
            SceneKind::Wine => {
                if ui.button("Wine Glass").clicked() {
                    ctx.main_db.collection_link_object(ctx.wine_master, ctx.wine_obj_id);
                    ctx.main_db
                        .ensure_scene_base(scene_id, ctx.wine_obj_id, true, true);
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Spotlight").clicked() {
                    ctx.main_db.collection_link_object(ctx.wine_master, ctx.spot_obj_id);
                    ctx.main_db
                        .ensure_scene_base(scene_id, ctx.spot_obj_id, true, true);
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
            }
            SceneKind::CornellBox => {
                if ui.button("Cornell Box").clicked() {
                    ctx.main_db
                        .collection_link_object(ctx.cornell_master, ctx.cornell_obj_id);
                    ctx.main_db
                        .ensure_scene_base(scene_id, ctx.cornell_obj_id, true, true);
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
                if ui.button("Cube").clicked() {
                    ctx.main_db
                        .collection_link_object(ctx.cornell_master, ctx.sphere_obj_id);
                    ctx.main_db
                        .ensure_scene_base(scene_id, ctx.sphere_obj_id, true, true);
                    *ctx.suppress_scene_click = true;
                    ui.close();
                }
            }
        }
    });
}
