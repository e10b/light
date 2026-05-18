pub fn update_fps_title(
    window: &winit::window::Window,
    frame_count: &mut u32,
    fps_display_time: &mut std::time::Instant,
) {
    *frame_count += 1;
    let now = std::time::Instant::now();
    let elapsed = now.duration_since(*fps_display_time).as_secs_f32();
    if elapsed >= 1.0 {
        let fps = *frame_count as f32 / elapsed;
        window.set_title(&format!("wgpu v0.29 ray tracing - {:.1} FPS", fps));
        *frame_count = 0;
        *fps_display_time = now;
    }
}
