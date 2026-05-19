use std::{cell::RefCell, collections::HashMap, rc::Rc};

use egui::Color32;
use egui_code_editor::{CodeEditor, ColorTheme, Syntax};

use crate::{
    ecs::{CameraComponent, PhysicsComponent, ScriptEngine, World},
    material_editor::{MaterialGraphEditor, RuntimeMaterialPreview},
    prism_file::MaterialData as PrismMaterialData,
    scene_data::{Id, MainDatabase},
    tooling::lua::{ensure_lua_editor_document, scripts_dir, write_lua_script},
};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum CameraProjectionKind {
    Perspective,
    Orthographic,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum GizmoModeKind {
    Translate,
    Rotate,
    Scale,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum RenderModeKind {
    Pathtraced,
    Raytraced,
    Rasterized,
}

#[derive(Default)]
pub struct PropertiesPanelOutput {
    pub accumulation_dirty: bool,
    pub visibility_changed: bool,
    pub selected_light_intensity: Option<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn draw_properties_panel(
    ctx: &egui::Context,
    main_db: &MainDatabase,
    ecs_world: &Rc<RefCell<World>>,
    script_engine: &mut ScriptEngine,
    lua_syntax: &Syntax,
    has_selection: bool,
    selected_object_id: Option<Id>,
    render_mode: &mut RenderModeKind,
    gizmo_mode: &mut GizmoModeKind,
    camera_projection_mode: &mut CameraProjectionKind,
    camera_near: &mut f32,
    camera_far: &mut f32,
    camera_fov_radians: &mut f32,
    camera_ortho_height: &mut f32,
    sun_azimuth_deg: &mut f32,
    sun_elevation_deg: &mut f32,
    sun_intensity: &mut f32,
    selected_light_intensity: Option<f32>,
    lua_editor_entity: &mut Option<Id>,
    lua_editor_path: &mut String,
    lua_editor_text: &mut String,
    lua_editor_status: &mut String,
    object_material_names: &mut HashMap<Id, String>,
    material_library: &mut HashMap<String, PrismMaterialData>,
    material_editor: &mut MaterialGraphEditor,
    material_runtime_overrides: &mut HashMap<String, RuntimeMaterialPreview>,
) -> PropertiesPanelOutput {
    let mut out = PropertiesPanelOutput::default();

    egui::SidePanel::right("properties")
        .resizable(true)
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.heading("Properties");
            ui.horizontal(|ui| {
                ui.label("Render");
                let path_clicked = ui
                    .selectable_value(render_mode, RenderModeKind::Pathtraced, "Pathtraced")
                    .changed();
                let ray_clicked = ui
                    .selectable_value(render_mode, RenderModeKind::Raytraced, "Raytraced")
                    .changed();
                let ras_clicked = ui
                    .selectable_value(render_mode, RenderModeKind::Rasterized, "Rasterized")
                    .changed();
                if path_clicked || ray_clicked || ras_clicked {
                    out.accumulation_dirty = true;
                }
            });
            ui.separator();

            let selected_label = selected_object_id
                .and_then(|id| main_db.objects.get(&id).map(|o| o.name.clone()))
                .unwrap_or_else(|| "None".to_string());
            ui.label(format!(
                "Selected: {}",
                if has_selection {
                    selected_label.as_str()
                } else {
                    "None"
                }
            ));

            if let Some(obj_id) = selected_object_id.filter(|_| has_selection) {
                ui.collapsing("Entity", |ui| {
                    let (
                        has_mesh,
                        has_camera,
                        has_physics,
                        visible,
                        inherited_visible,
                        view_visible,
                        script_path,
                        script_error,
                    ) = {
                        let world = ecs_world.borrow();
                        (
                            world.meshes.contains_key(&obj_id),
                            world.cameras.contains_key(&obj_id),
                            world.physics.contains_key(&obj_id),
                            world.visibility.get(&obj_id).copied().unwrap_or_default()
                                == crate::ecs::Visibility::Visible,
                            world
                                .inherited_visibility
                                .get(&obj_id)
                                .map(|v| v.visible)
                                .unwrap_or(true),
                            world
                                .view_visibility
                                .get(&obj_id)
                                .map(|v| v.visible)
                                .unwrap_or(true),
                            world.scripts.get(&obj_id).map(|s| s.path.clone()),
                            world
                                .scripts
                                .get(&obj_id)
                                .and_then(|s| s.last_error.clone()),
                        )
                    };
                    ui.label(format!(
                        "Components: transform, global transform, visibility{}{}{}{}",
                        if has_mesh { ", mesh" } else { "" },
                        if has_camera { ", camera" } else { "" },
                        if has_physics { ", physics" } else { "" },
                        if script_path.is_some() {
                            ", script"
                        } else {
                            ""
                        },
                    ));
                    let mut visible_edit = visible;
                    if ui.checkbox(&mut visible_edit, "Visible").changed() {
                        ecs_world.borrow_mut().set_visible(obj_id, visible_edit);
                        out.visibility_changed = true;
                        out.accumulation_dirty = true;
                    }
                    ui.label(format!(
                        "Inherited: {}  View: {}",
                        if inherited_visible {
                            "visible"
                        } else {
                            "hidden"
                        },
                        if view_visible { "visible" } else { "hidden" },
                    ));
                    ui.horizontal(|ui| {
                        if ui.button("Attach Physics").clicked() {
                            ecs_world
                                .borrow_mut()
                                .attach_physics(obj_id, PhysicsComponent::default());
                        }
                        if ui.button("Attach Camera").clicked() {
                            ecs_world
                                .borrow_mut()
                                .attach_camera(obj_id, CameraComponent::default());
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Attach Lua").clicked() {
                            let entity_name = main_db
                                .objects
                                .get(&obj_id)
                                .map(|o| o.name.as_str())
                                .unwrap_or("Entity");
                            ensure_lua_editor_document(
                                obj_id,
                                entity_name,
                                script_path.clone(),
                                lua_editor_entity,
                                lua_editor_path,
                                lua_editor_text,
                                lua_editor_status,
                            );
                            match write_lua_script(lua_editor_path, lua_editor_text) {
                                Ok(full_path) => {
                                    ecs_world
                                        .borrow_mut()
                                        .attach_script(obj_id, lua_editor_path.clone());
                                    script_engine.forget(obj_id);
                                    *lua_editor_status =
                                        format!("Attached: {}", full_path.display());
                                }
                                Err(e) => {
                                    *lua_editor_status = format!("Attach failed: {e}");
                                }
                            }
                        }
                        if ui.button("Use As Camera").clicked() {
                            let mut world = ecs_world.borrow_mut();
                            for camera in world.cameras.values_mut() {
                                camera.active = false;
                            }
                            world
                                .cameras
                                .entry(obj_id)
                                .or_insert_with(CameraComponent::default)
                                .active = true;
                        }
                    });
                    if let Some(path) = script_path {
                        ui.label(format!(
                            "Script: {path} ({})",
                            ecs_world.borrow().script_status(obj_id)
                        ));
                    }
                    if let Some(error) = script_error {
                        ui.colored_label(Color32::from_rgb(255, 120, 96), error);
                    }
                });

                ui.collapsing("Lua Editor", |ui| {
                    let entity_name = main_db
                        .objects
                        .get(&obj_id)
                        .map(|o| o.name.clone())
                        .unwrap_or_else(|| "Entity".to_string());
                    ensure_lua_editor_document(
                        obj_id,
                        &entity_name,
                        ecs_world
                            .borrow()
                            .scripts
                            .get(&obj_id)
                            .map(|s| s.path.clone()),
                        lua_editor_entity,
                        lua_editor_path,
                        lua_editor_text,
                        lua_editor_status,
                    );
                    ui.horizontal(|ui| {
                        ui.label("scripts/");
                        ui.text_edit_singleline(lua_editor_path);
                    });
                    let mut editor = CodeEditor::default()
                        .id_source(format!("lua_editor_{}", obj_id.0))
                        .with_rows(18)
                        .with_fontsize(13.0)
                        .with_theme(ColorTheme::GRUVBOX)
                        .with_numlines(true)
                        .desired_width(f32::INFINITY);
                    editor.show(ui, lua_editor_text, lua_syntax);
                    ui.horizontal(|ui| {
                        if ui.button("Save + Reload").clicked() {
                            let clean_path =
                                lua_editor_path.trim().trim_start_matches('/').to_string();
                            match write_lua_script(&clean_path, lua_editor_text) {
                                Ok(_) => {
                                    *lua_editor_path = clean_path.clone();
                                    {
                                        let mut world = ecs_world.borrow_mut();
                                        world.attach_script(obj_id, clean_path.clone());
                                        if let Some(script) = world.scripts.get_mut(&obj_id) {
                                            script.started = false;
                                            script.last_error = None;
                                        }
                                    }
                                    script_engine.forget(obj_id);
                                    *lua_editor_status = format!(
                                        "Saved: {}",
                                        scripts_dir().join(&clean_path).display()
                                    );
                                }
                                Err(e) => {
                                    *lua_editor_status = format!("Save failed: {e}");
                                }
                            }
                        }
                        if ui.button("Reload From Disk").clicked() {
                            let full_path = scripts_dir().join(lua_editor_path.trim());
                            match std::fs::read_to_string(&full_path) {
                                Ok(source) => {
                                    *lua_editor_text = source;
                                    *lua_editor_status =
                                        format!("Reloaded: {}", full_path.display());
                                }
                                Err(e) => {
                                    *lua_editor_status = format!("Reload failed: {e}");
                                }
                            }
                        }
                    });
                    if !lua_editor_status.is_empty() {
                        ui.label(lua_editor_status.as_str());
                    }
                });
            }

            ui.horizontal(|ui| {
                ui.selectable_value(gizmo_mode, GizmoModeKind::Translate, "Move");
                ui.selectable_value(gizmo_mode, GizmoModeKind::Rotate, "Rotate");
                ui.selectable_value(gizmo_mode, GizmoModeKind::Scale, "Scale");
            });
            ui.separator();
            ui.collapsing("Camera", |ui| {
                let mut projection_changed = false;
                ui.horizontal(|ui| {
                    projection_changed |= ui
                        .selectable_value(
                            camera_projection_mode,
                            CameraProjectionKind::Perspective,
                            "Perspective",
                        )
                        .changed();
                    projection_changed |= ui
                        .selectable_value(
                            camera_projection_mode,
                            CameraProjectionKind::Orthographic,
                            "Orthographic",
                        )
                        .changed();
                });
                projection_changed |= ui
                    .add(egui::Slider::new(camera_near, 0.01..=10.0).text("Near"))
                    .changed();
                projection_changed |= ui
                    .add(egui::Slider::new(camera_far, 10.0..=5000.0).text("Far"))
                    .changed();
                match camera_projection_mode {
                    CameraProjectionKind::Perspective => {
                        let mut fov_deg = camera_fov_radians.to_degrees();
                        if ui
                            .add(egui::Slider::new(&mut fov_deg, 10.0..=140.0).text("FOV"))
                            .changed()
                        {
                            *camera_fov_radians = fov_deg.to_radians();
                            projection_changed = true;
                        }
                    }
                    CameraProjectionKind::Orthographic => {
                        projection_changed |= ui
                            .add(egui::Slider::new(camera_ortho_height, 0.1..=200.0).text("Height"))
                            .changed();
                    }
                }
                if *camera_far <= *camera_near + 0.001 {
                    *camera_far = *camera_near + 0.001;
                }
                if projection_changed {
                    out.accumulation_dirty = true;
                }
            });
            ui.separator();
            ui.collapsing("Sun", |ui| {
                ui.add(egui::Slider::new(sun_azimuth_deg, -180.0..=180.0).text("Azimuth"));
                ui.add(egui::Slider::new(sun_elevation_deg, -10.0..=89.0).text("Elevation"));
                if let Some(mut intensity) = selected_light_intensity {
                    if ui
                        .add(egui::Slider::new(&mut intensity, 0.0..=5.0).text("Intensity"))
                        .changed()
                    {
                        out.selected_light_intensity = Some(intensity);
                        out.accumulation_dirty = true;
                    }
                } else if ui
                    .add(egui::Slider::new(sun_intensity, 0.0..=5.0).text("Intensity"))
                    .changed()
                {
                    out.accumulation_dirty = true;
                }
            });
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                ui.separator();
                ui.collapsing("Shader Graph", |ui| {
                    let material_object_id = if has_selection {
                        selected_object_id
                    } else {
                        None
                    };
                    if let Some(obj_id) = material_object_id {
                        let mut mat_name = object_material_names
                            .get(&obj_id)
                            .cloned()
                            .unwrap_or_else(|| "White".to_string());
                        egui::ComboBox::from_label("Material")
                            .selected_text(&mat_name)
                            .show_ui(ui, |ui| {
                                for key in material_library.keys() {
                                    ui.selectable_value(&mut mat_name, key.clone(), key);
                                }
                            });
                        object_material_names.insert(obj_id, mat_name.clone());
                        if let Some(mat) = material_library.get(&mat_name) {
                            let graph_key = mat_name.clone();
                            ui.label(format!("Graph: {}", mat.name));
                            material_editor.load_material(&graph_key, mat);
                            egui::Frame::default().show(ui, |ui| {
                                ui.set_min_height(280.0);
                                material_editor.show(ui);
                            });
                            if let Some(mat_mut) = material_library.get_mut(&mat_name) {
                                material_editor.commit_to_material(mat_mut);
                            }
                            material_runtime_overrides
                                .insert(mat_name.clone(), material_editor.runtime_preview());
                        } else {
                            ui.label("White fallback (no material graph).");
                        }
                    } else {
                        ui.label("No object selected.");
                    }
                });
            });
        });

    out
}
