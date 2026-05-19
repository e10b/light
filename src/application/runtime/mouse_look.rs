use std::collections::HashSet;

use crate::application::types::Camera;

pub fn handle_mouse_motion(
    delta: (f64, f64),
    egui_ctx: &egui::Context,
    keys_pressed: &HashSet<String>,
    mouse_speed: f32,
    camera: &mut Camera,
    accumulation_dirty: &mut bool,
) {
    if egui_ctx.is_pointer_over_egui() || !keys_pressed.contains("v") {
        return;
    }
    let (dx, dy) = delta;
    camera.yaw -= dx as f32 * mouse_speed;
    camera.pitch -= dy as f32 * mouse_speed;
    camera.pitch = camera.pitch.clamp(-1.45, 1.45);
    *accumulation_dirty = true;
}
