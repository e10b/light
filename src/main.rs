use std::iter;
use std::path::Path;

use imgui::{Condition, FontConfig, FontSource};
use imgui_wgpu::{Renderer as ImguiRenderer, RendererConfig};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop, window};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

fn load_gltf_mesh(path: &Path) -> Result<(Vec<Vertex>, Vec<u32>), Box<dyn std::error::Error>> {
    let (document, buffers, _images) = gltf::import(path)?;

    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer_index| Some(&buffers[buffer_index.index()]));

            let start_vertex = all_vertices.len() as u32;
            let mut local_vertex_count = 0u32;

            if let Some(iter) = reader.read_positions() {
                for pos in iter {
                    all_vertices.push(Vertex { position: pos });
                    local_vertex_count += 1;
                }
            }

            if let Some(iter) = reader.read_indices() {
                match iter {
                    gltf::mesh::util::ReadIndices::U32(idx_iter) => {
                        for idx in idx_iter {
                            all_indices.push(start_vertex + idx);
                        }
                    }
                    gltf::mesh::util::ReadIndices::U16(idx_iter) => {
                        for idx in idx_iter {
                            all_indices.push(start_vertex + idx as u32);
                        }
                    }
                    gltf::mesh::util::ReadIndices::U8(idx_iter) => {
                        for idx in idx_iter {
                            all_indices.push(start_vertex + idx as u32);
                        }
                    }
                }
            } else {
                for idx in 0..local_vertex_count {
                    all_indices.push(start_vertex + idx);
                }
            }
        }
    }

    Ok((all_vertices, all_indices))
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    view_inv: [[f32; 4]; 4],
    proj_inv: [[f32; 4]; 4],
    light_pos: [f32; 4],
    sphere_pos: [f32; 4],
    sphere_color: [f32; 4],
    sun_intensity: f32,
    frame: u32,
    _pad: [u32; 2],
}

struct Camera {
    pos: glam::Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(pos: glam::Vec3) -> Self {
        Self {
            pos,
            yaw: std::f32::consts::PI,
            pitch: -0.12,
        }
    }

    fn forward(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }

    fn right(&self) -> glam::Vec3 {
        self.forward().cross(glam::Vec3::Y).normalize()
    }

    fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.pos, self.pos + self.forward(), glam::Vec3::Y)
    }
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = std::sync::Arc::new(
        event_loop
            .create_window(window::WindowAttributes::default().with_title("wgpu v0.29 ray tracing"))
            .expect("failed to create window"),
    );

    let size = window.inner_size();
    let instance = wgpu::Instance::default();
    let window_ref = window.clone();
    let surface = instance.create_surface(window_ref.as_ref()).unwrap();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("No adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Immediate,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 0,
    };
    surface.configure(&device, &config);

    let decanter_path = std::path::Path::new("res/wine_decanter_and_glass.glb");
    let (model_verts, model_idx) = load_gltf_mesh(decanter_path).expect("Failed to load decanter model");

    println!("Loaded {} vertices and {} indices from decanter", model_verts.len(), model_idx.len());

    let mut min_pos = glam::Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_pos = glam::Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for vert in &model_verts {
        let pos = glam::Vec3::from(vert.position);
        min_pos = min_pos.min(pos);
        max_pos = max_pos.max(pos);
    }
    let center = (min_pos + max_pos) * 0.5;
    let size = max_pos - min_pos;
    let _max_extent = size.max_element();

    println!("Model bounds: center={:?}, size={:?}", center, size);

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("model_vbuf"),
        contents: bytemuck::cast_slice(&model_verts),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("model_ibuf"),
        contents: bytemuck::cast_slice(&model_idx),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
    });

    let model_blas_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: model_verts.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(model_idx.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let model_blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("model_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![model_blas_desc.clone()],
        },
    );

    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("scene_tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });
    tlas[0] = Some(wgpu::TlasInstance::new(
        &model_blas,
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        0,
        0xff,
    ));

    let model_build = wgpu::BlasBuildEntry {
        blas: &model_blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
            size: &model_blas_desc,
            vertex_buffer: &vbuf,
            first_vertex: 0,
            vertex_stride: std::mem::size_of::<Vertex>() as u64,
            index_buffer: Some(&ibuf),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        }]),
    };

    let mut accel_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("accel") });
    accel_encoder.build_acceleration_structures([model_build].iter(), iter::once(&tlas));
    queue.submit(Some(accel_encoder.finish()));

    let mut camera = Camera::new(glam::Vec3::new(0.0, 1.1, 3.2));
    let projection = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_3 * 1.2, config.width as f32 / config.height as f32, 0.1, 1000.0);

    let sphere_pos = glam::Vec3::new(center.x + size.x * 0.6 + 2.0, center.y - size.y * 0.5 + 1.0, center.z);
    let sphere_radius = 6.0;
    let mut uniforms = SceneUniforms {
        view_inv: camera.view_matrix().inverse().to_cols_array_2d(),
        proj_inv: projection.inverse().to_cols_array_2d(),
        light_pos: [10.0, 8.0, 10.0, 1.0],
        sphere_pos: [sphere_pos.x, sphere_pos.y, sphere_pos.z, sphere_radius],
        sphere_color: [1.0, 0.1, 0.1, 0.0],
        sun_intensity: 0.8,
        frame: 0,
        _pad: [0, 0],
    };

    let mut sun_azimuth_deg = uniforms.light_pos[2].atan2(uniforms.light_pos[0]).to_degrees();
    let sun_len_xz = (uniforms.light_pos[0] * uniforms.light_pos[0] + uniforms.light_pos[2] * uniforms.light_pos[2]).sqrt();
    let mut sun_elevation_deg = uniforms.light_pos[1].atan2(sun_len_xz).to_degrees();
    let mut sun_intensity = uniforms.sun_intensity;

    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ubuf"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let ubind = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ubind"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
        ],
    });

    let ugroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ugroup"),
        layout: &ubind,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pathtracer"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sphere.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[Some(&ubind)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("rp"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);
    let mut platform = WinitPlatform::new(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Rounded);
    let hidpi_factor = window.scale_factor();
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(FontConfig {
            size_pixels: font_size,
            ..FontConfig::default()
        }),
    }]);
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    let renderer_config = RendererConfig {
        texture_format: config.format,
        ..RendererConfig::default()
    };
    let mut imgui_renderer = ImguiRenderer::new(&mut imgui, &device, &queue, renderer_config);

    let move_speed = 2.6;
    let look_speed = 0.28;
    let mouse_speed = 0.003;
    let mut keys_pressed = std::collections::HashSet::new();
    let mut frame_count = 0u32;
    let mut fps_display_time = std::time::Instant::now();
    let mut last_update = std::time::Instant::now();

    let _ = event_loop.run(move |event, active_loop| {
        active_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        platform.handle_event(imgui.io_mut(), &window, &event);
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => active_loop.exit(),
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event, .. },
                ..
            } => match event.state {
                ElementState::Pressed => {
                    if let winit::keyboard::Key::Character(c) = &event.logical_key {
                        keys_pressed.insert(c.to_lowercase().to_string());
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) {
                        keys_pressed.insert("Space".to_string());
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft)
                        || event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftRight)
                    {
                        keys_pressed.insert("Shift".to_string());
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlLeft)
                        || event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlRight)
                    {
                        keys_pressed.insert("Control".to_string());
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp) {
                        keys_pressed.insert("ArrowUp".to_string());
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown) {
                        keys_pressed.insert("ArrowDown".to_string());
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowLeft) {
                        keys_pressed.insert("ArrowLeft".to_string());
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowRight) {
                        keys_pressed.insert("ArrowRight".to_string());
                    }
                }
                ElementState::Released => {
                    if let winit::keyboard::Key::Character(c) = &event.logical_key {
                        keys_pressed.remove(&c.to_lowercase().to_string());
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) {
                        keys_pressed.remove("Space");
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft)
                        || event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftRight)
                    {
                        keys_pressed.remove("Shift");
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlLeft)
                        || event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlRight)
                    {
                        keys_pressed.remove("Control");
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp) {
                        keys_pressed.remove("ArrowUp");
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown) {
                        keys_pressed.remove("ArrowDown");
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowLeft) {
                        keys_pressed.remove("ArrowLeft");
                    } else if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowRight) {
                        keys_pressed.remove("ArrowRight");
                    }
                }
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                config.width = size.width;
                config.height = size.height;
                let projection = glam::Mat4::perspective_rh(
                    std::f32::consts::FRAC_PI_3 * 1.2,
                    config.width as f32 / config.height as f32,
                    0.1,
                    1000.0,
                );
                uniforms.proj_inv = projection.inverse().to_cols_array_2d();
                queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&uniforms));
                surface.configure(&device, &config);
            }
            Event::DeviceEvent {
                event: winit::event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if imgui.io().want_capture_mouse || !keys_pressed.contains("v") {
                    return;
                }
                let (dx, dy) = delta;
                camera.yaw -= dx as f32 * mouse_speed;
                camera.pitch -= dy as f32 * mouse_speed;
                camera.pitch = camera.pitch.clamp(-1.45, 1.45);
            }
            Event::NewEvents(start_cause) => match start_cause {
                winit::event::StartCause::Init | winit::event::StartCause::Poll => {
                    frame_count += 1;
                    let now = std::time::Instant::now();
                    let elapsed = now.duration_since(fps_display_time).as_secs_f32();
                    if elapsed >= 1.0 {
                        let fps = frame_count as f32 / elapsed;
                        window.set_title(&format!("wgpu v0.29 ray tracing - {:.1} FPS", fps));
                        frame_count = 0;
                        fps_display_time = now;
                    }

                    let now = std::time::Instant::now();
                    let dt = now.duration_since(last_update).as_secs_f32();
                    last_update = now;
                    let sprint = if keys_pressed.contains("Shift") { 3.0 } else { 1.0 };
                    let wants_keyboard = imgui.io().want_capture_keyboard;

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

                    uniforms.view_inv = camera.view_matrix().inverse().to_cols_array_2d();
                    uniforms.frame += 1;
                    queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&uniforms));

                    match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(tex) | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => {
                            imgui.io_mut().update_delta_time(std::time::Duration::from_secs_f32(dt.max(1.0 / 1000.0)));
                            if platform.prepare_frame(imgui.io_mut(), &window).is_err() {
                                tex.present();
                                return;
                            }
                            let ui = imgui.frame();

                            ui.window("Sun Controls")
                                .size([300.0, 160.0], Condition::FirstUseEver)
                                .build(|| {
                                    ui.slider_config("Azimuth (deg)", -180.0, 180.0).build(&mut sun_azimuth_deg);
                                    ui.slider_config("Elevation (deg)", -10.0, 89.0).build(&mut sun_elevation_deg);
                                    ui.slider_config("Intensity", 0.0, 5.0).build(&mut sun_intensity);
                                });

                            let sun_az = sun_azimuth_deg.to_radians();
                            let sun_el = sun_elevation_deg.to_radians();
                            let sun_dir = glam::Vec3::new(
                                sun_az.cos() * sun_el.cos(),
                                sun_el.sin(),
                                sun_az.sin() * sun_el.cos(),
                            )
                            .normalize_or_zero();
                            uniforms.light_pos = [sun_dir.x, sun_dir.y, sun_dir.z, 1.0];
                            uniforms.sun_intensity = sun_intensity.max(0.0);
                            queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&uniforms));

                            let view = tex.texture.create_view(&wgpu::TextureViewDescriptor::default());
                            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
                            {
                                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("rp"),
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
                                rpass.set_pipeline(&pipeline);
                                rpass.set_bind_group(0, &ugroup, &[]);
                                rpass.draw(0..3, 0..1);
                            }
                            platform.prepare_render(ui, &window);
                            let draw_data = imgui.render();
                            {
                                let mut ui_rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("imgui-pass"),
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
                                let _ = imgui_renderer.render(draw_data, &queue, &device, &mut ui_rpass);
                            }
                            queue.submit(Some(encoder.finish()));
                            tex.present();
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            _ => {}
        }
    });
}
