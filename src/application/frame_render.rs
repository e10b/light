use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use wgpu::util::DeviceExt;

use crate::{
    compute_pass,
    editor::panels::RenderModeKind,
    photon_mapper::PhotonMapper,
    quad_pass, raster_pass,
    scene_data::{Id, MainDatabase},
};

use super::{
    geometry::build_photon_targets,
    types::{Camera, MeshObjectInstance, SceneUniforms, MAX_SUN_LIGHTS},
};

#[allow(clippy::too_many_arguments)]
pub fn render_frame_and_present(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    tex: wgpu::SurfaceTexture,
    egui_renderer: &mut EguiRenderer,
    textures_delta: &egui::TexturesDelta,
    clipped_primitives: &[egui::ClippedPrimitive],
    pixels_per_point: f32,
    render_mode: RenderModeKind,
    uniforms: &mut SceneUniforms,
    photons_per_frame: u32,
    main_db: &MainDatabase,
    decanter_scene_id: Id,
    mesh_instances: &[MeshObjectInstance],
    stress_instance_count: usize,
    photon_mapper: &mut PhotonMapper,
    photon_emitter_center: [f32; 4],
    sphere_visible_for_photons: bool,
    raster_pass: &mut raster_pass::RasterPass,
    raster_instance_count: &mut u32,
    raster_instance_buf: &mut wgpu::Buffer,
    projection: glam::Mat4,
    camera: &Camera,
    vbuf: &wgpu::Buffer,
    ibuf: &wgpu::Buffer,
    model_idx_len: usize,
    compute_pass: &compute_pass::ComputePass,
    ugroup: &wgpu::BindGroup,
    quad_pass: &quad_pass::QuadPass,
    mouse_left_clicked: &mut bool,
    mouse_left_down: bool,
    mouse_left_dragging: &mut bool,
) {
    let view = tex
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("enc"),
    });
    let screen_descriptor = ScreenDescriptor {
        size_in_pixels: [config.width, config.height],
        pixels_per_point,
    };
    for (id, image_delta) in &textures_delta.set {
        egui_renderer.update_texture(device, queue, *id, image_delta);
    }
    egui_renderer.update_buffers(
        device,
        queue,
        &mut encoder,
        clipped_primitives,
        &screen_descriptor,
    );

    if matches!(render_mode, RenderModeKind::Pathtraced) {
        let photon_light_pos = if uniforms.sun_light_count > 0 {
            let count = uniforms.sun_light_count.min(MAX_SUN_LIGHTS as u32);
            let idx = (uniforms.frame % count) as usize;
            uniforms.sun_lights[idx]
        } else {
            uniforms.light_pos
        };
        let photon_frame_count = photons_per_frame
            .saturating_mul(uniforms.sun_light_count.max(1))
            .min(1_000_000);
        let photon_visible_ids = main_db.scene_visible_selectable_objects(decanter_scene_id);
        let photon_targets = build_photon_targets(
            mesh_instances,
            &photon_visible_ids,
            stress_instance_count > 0,
        );
        photon_mapper.update(
            queue,
            photon_light_pos,
            photon_emitter_center,
            uniforms.sphere_pos,
            uniforms.sphere_rot,
            uniforms.sphere_extent,
            [
                uniforms.sphere_color[3],
                uniforms.sphere_params[1],
                uniforms.sphere_params[2],
                0.0,
            ],
            sphere_visible_for_photons,
            photon_frame_count,
            &photon_targets,
            uniforms.frame,
        );
        photon_mapper.emit_photons(&mut encoder, photon_frame_count);
        photon_mapper.build_spatial_structure(&mut encoder);
    }

    if matches!(render_mode, RenderModeKind::Rasterized) {
        if stress_instance_count > 0 {
            let instances: Vec<raster_pass::RasterInstance> = mesh_instances
                .iter()
                .map(|inst| {
                    let c = inst.center();
                    raster_pass::RasterInstance {
                        offset: [c.x, c.y, c.z, 0.0],
                    }
                })
                .collect();
            if *raster_instance_count != instances.len() as u32 {
                *raster_instance_count = instances.len() as u32;
                *raster_instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("raster_instance_buf"),
                    contents: bytemuck::cast_slice(&instances),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
            }
        } else if *raster_instance_count != 1 {
            *raster_instance_count = 1;
            *raster_instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("raster_instance_buf"),
                contents: bytemuck::cast_slice(&[raster_pass::RasterInstance {
                    offset: [0.0, 0.0, 0.0, 0.0],
                }]),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        }

        let view_proj = projection * camera.view_matrix();
        raster_pass.update_view_proj(queue, view_proj);
        let mut raster_rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raster_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            multiview_mask: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        raster_pass.render(
            &mut raster_rpass,
            vbuf,
            raster_instance_buf,
            *raster_instance_count,
            ibuf,
            model_idx_len as u32,
        );
    } else {
        compute_pass.record(
            &mut encoder,
            ugroup,
            match render_mode {
                RenderModeKind::Pathtraced => compute_pass::RenderPath::Pathtraced,
                RenderModeKind::Raytraced => compute_pass::RenderPath::Raytraced,
                RenderModeKind::Rasterized => unreachable!(),
            },
        );
        let mut present_rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            multiview_mask: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        quad_pass.render(&mut present_rpass);
    }

    {
        let ui_rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            multiview_mask: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        egui_renderer.render(
            &mut ui_rpass.forget_lifetime(),
            clipped_primitives,
            &screen_descriptor,
        );
    }

    queue.submit(Some(encoder.finish()));
    for id in &textures_delta.free {
        egui_renderer.free_texture(id);
    }
    *mouse_left_clicked = false;
    if !mouse_left_down {
        *mouse_left_dragging = false;
    }
    tex.present();
}
