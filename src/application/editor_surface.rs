use std::{cell::RefCell, collections::HashMap, rc::Rc};

use egui_code_editor::Syntax;

use crate::{
    ecs::{ScriptEngine, World},
    editor::panels::{
        draw_outliner_panel, draw_properties_panel, CameraProjectionKind, GizmoModeKind,
        OutlinerItem, RenderModeKind,
    },
    material_editor::{MaterialGraphEditor, RuntimeMaterialPreview},
    prism_file::MaterialData as PrismMaterialData,
    scene::SceneKind,
    scene_data::{Id, MainDatabase},
};

use super::{
    ecs_sync::sync_ecs_visibility_to_main,
    types::{
        default_target_for_scene, target_allowed_in_scene, GizmoTargetKind, LightObjectInstance,
    },
};

#[allow(clippy::too_many_arguments)]
pub fn draw_editor_surface(
    ctx: &egui::Context,
    main_db: &mut MainDatabase,
    scene_kind: SceneKind,
    project_status: &str,
    decanter_scene_id: Id,
    object_target_by_id: &HashMap<Id, GizmoTargetKind>,
    has_selection: &mut bool,
    selected_object_id: &mut Option<Id>,
    gizmo_target: &mut GizmoTargetKind,
    sphere_obj_id: Id,
    decanter_obj_id: Id,
    wine_obj_id: Id,
    cornell_obj_id: Id,
    sun_obj_id: Id,
    spot_obj_id: Id,
    light_instances: &mut [LightObjectInstance],
    ecs_world: &Rc<RefCell<World>>,
    script_engine: &mut ScriptEngine,
    lua_syntax: &Syntax,
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
    lua_editor_entity: &mut Option<Id>,
    lua_editor_path: &mut String,
    lua_editor_text: &mut String,
    lua_editor_status: &mut String,
    object_material_names: &mut HashMap<Id, String>,
    material_library: &mut HashMap<String, PrismMaterialData>,
    material_editor: &mut MaterialGraphEditor,
    material_runtime_overrides: &mut HashMap<String, RuntimeMaterialPreview>,
    material_script_name: &mut String,
    material_script_text: &mut String,
    material_script_status: &mut String,
    accumulation_dirty: &mut bool,
) {
    let scene_id = decanter_scene_id;
    let outliner_items: Vec<OutlinerItem> = main_db
        .scene_visible_selectable_objects(scene_id)
        .into_iter()
        .filter_map(|object_id| {
            let target = object_target_by_id.get(&object_id).copied()?;
            if !target_allowed_in_scene(scene_kind, target) {
                return None;
            }
            let label = main_db
                .objects
                .get(&object_id)
                .map(|o| o.name.clone())
                .unwrap_or_else(|| "Object".to_string());
            Some(OutlinerItem {
                object_id,
                label,
                selected: *has_selection && *selected_object_id == Some(object_id),
            })
        })
        .collect();
    if let Some(object_id) = draw_outliner_panel(ctx, &outliner_items, project_status) {
        if let Some(target) = object_target_by_id.get(&object_id).copied() {
            *gizmo_target = target;
            *selected_object_id = Some(object_id);
            *has_selection = true;
        }
    }

    if !target_allowed_in_scene(scene_kind, *gizmo_target) {
        *gizmo_target = default_target_for_scene(scene_kind);
        *selected_object_id = match *gizmo_target {
            GizmoTargetKind::Sphere => Some(sphere_obj_id),
            GizmoTargetKind::Decanter => Some(decanter_obj_id),
            GizmoTargetKind::WineGlass => Some(wine_obj_id),
            GizmoTargetKind::CornellBox => Some(cornell_obj_id),
            GizmoTargetKind::SunLamp => Some(sun_obj_id),
            GizmoTargetKind::WineSpotlight => Some(spot_obj_id),
            GizmoTargetKind::Camera => None,
        };
    }

    let selected_light_intensity = selected_object_id
        .and_then(|id| light_instances.iter().find(|l| l.object_id == id))
        .map(|l| l.intensity);
    let panel_output = draw_properties_panel(
        ctx,
        main_db,
        ecs_world,
        script_engine,
        lua_syntax,
        *has_selection,
        *selected_object_id,
        render_mode,
        gizmo_mode,
        camera_projection_mode,
        camera_near,
        camera_far,
        camera_fov_radians,
        camera_ortho_height,
        sun_azimuth_deg,
        sun_elevation_deg,
        sun_intensity,
        selected_light_intensity,
        lua_editor_entity,
        lua_editor_path,
        lua_editor_text,
        lua_editor_status,
        object_material_names,
        material_library,
        material_editor,
        material_runtime_overrides,
        material_script_name,
        material_script_text,
        material_script_status,
    );
    if let (Some(obj_id), Some(intensity)) =
        (*selected_object_id, panel_output.selected_light_intensity)
    {
        if let Some(light) = light_instances.iter_mut().find(|l| l.object_id == obj_id) {
            light.intensity = intensity;
        }
    }
    if panel_output.visibility_changed {
        sync_ecs_visibility_to_main(&ecs_world.borrow(), main_db);
    }
    if panel_output.accumulation_dirty {
        *accumulation_dirty = true;
    }
}
