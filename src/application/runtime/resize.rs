use crate::application::{types::SceneUniforms, view_math::camera_projection_matrix};
use crate::editor::panels::CameraProjectionKind;

#[allow(clippy::too_many_arguments)]
pub fn handle_surface_resize(
    size: winit::dpi::PhysicalSize<u32>,
    config: &mut wgpu::SurfaceConfiguration,
    camera_projection_mode: CameraProjectionKind,
    camera_fov_radians: f32,
    camera_ortho_height: f32,
    camera_near: f32,
    camera_far: f32,
    uniforms: &mut SceneUniforms,
    queue: &wgpu::Queue,
    ubuf: &wgpu::Buffer,
    surface: &wgpu::Surface,
    device: &wgpu::Device,
) {
    config.width = size.width;
    config.height = size.height;
    let projection = camera_projection_matrix(
        camera_projection_mode,
        camera_fov_radians,
        camera_ortho_height,
        camera_near,
        camera_far,
        config.width,
        config.height,
    );
    uniforms.proj_inv = projection.inverse().to_cols_array_2d();
    uniforms.frame = 0;
    queue.write_buffer(ubuf, 0, bytemuck::bytes_of(uniforms));
    surface.configure(device, config);
}
