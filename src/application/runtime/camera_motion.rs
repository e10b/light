use std::collections::HashSet;

use crate::application::types::Camera;

#[allow(clippy::too_many_arguments)]
pub fn apply_fly_camera_motion(
    camera: &mut Camera,
    dt: f32,
    keys_pressed: &HashSet<String>,
    move_speed: f32,
    look_speed: f32,
    egui_ctx: &egui::Context,
) {
    let sprint = if keys_pressed.contains("Shift") {
        12.0
    } else {
        1.0
    };
    let wants_keyboard = egui_ctx.wants_keyboard_input();

    if !wants_keyboard && keys_pressed.contains("w") {
        camera.pos += camera.forward() * move_speed * sprint * dt;
    }
    if !wants_keyboard && keys_pressed.contains("s") {
        camera.pos -= camera.forward() * move_speed * sprint * dt;
    }
    if !wants_keyboard && keys_pressed.contains("a") {
        camera.pos -= camera.right() * move_speed * sprint * dt;
    }
    if !wants_keyboard && keys_pressed.contains("d") {
        camera.pos += camera.right() * move_speed * sprint * dt;
    }
    if !wants_keyboard && keys_pressed.contains("Space") {
        camera.pos.y += move_speed * sprint * dt;
    }
    if !wants_keyboard && keys_pressed.contains("Control") {
        camera.pos.y -= move_speed * sprint * dt;
    }
    if !wants_keyboard && keys_pressed.contains("ArrowUp") {
        camera.pitch += look_speed * dt;
        camera.pitch = camera.pitch.min(1.45);
    }
    if !wants_keyboard && keys_pressed.contains("ArrowDown") {
        camera.pitch -= look_speed * dt;
        camera.pitch = camera.pitch.max(-1.45);
    }
    if !wants_keyboard && keys_pressed.contains("ArrowLeft") {
        camera.yaw += look_speed * dt;
    }
    if !wants_keyboard && keys_pressed.contains("ArrowRight") {
        camera.yaw -= look_speed * dt;
    }
}
