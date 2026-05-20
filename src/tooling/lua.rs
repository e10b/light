use std::path::{Path, PathBuf};

use crate::ecs::script_path_for_entity_name;
use crate::scene_data::Id;

fn default_lua_script(entity_name: &str) -> String {
    let escaped_name = entity_name.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        r#"local state = {{
    t = 0.0,
}}

return {{
    on_start = function(entity)
        entity:log(\"{escaped_name} script attached\")
    end,

    on_update = function(entity, dt)
        state.t = state.t + dt
        entity:set_rotation_euler(0.0, state.t * 1.5, 0.0)
    end,
}}
"#
    )
}

fn project_path(path: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

pub fn scripts_dir() -> PathBuf {
    project_path("scripts")
}

pub fn material_scripts_dir() -> PathBuf {
    scripts_dir().join("materials")
}

pub fn write_lua_script(path: &str, source: &str) -> std::io::Result<PathBuf> {
    let clean_path = path.trim().trim_start_matches('/').to_string();
    let full_path = scripts_dir().join(clean_path);
    let parent = full_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(scripts_dir);
    std::fs::create_dir_all(parent)?;
    std::fs::write(&full_path, source)?;
    Ok(full_path)
}

pub fn ensure_lua_editor_document(
    entity_id: Id,
    entity_name: &str,
    script_path: Option<String>,
    editor_entity: &mut Option<Id>,
    editor_path: &mut String,
    editor_text: &mut String,
    editor_status: &mut String,
) {
    if *editor_entity == Some(entity_id) && !editor_path.is_empty() {
        return;
    }

    let path = script_path.unwrap_or_else(|| script_path_for_entity_name(entity_name));
    let full_path = scripts_dir().join(&path);
    let source =
        std::fs::read_to_string(&full_path).unwrap_or_else(|_| default_lua_script(entity_name));
    *editor_entity = Some(entity_id);
    *editor_path = path;
    *editor_text = source;
    *editor_status = String::new();
}
