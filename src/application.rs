use std::iter;

use imgui::{Condition, FontConfig, FontSource};
use imgui_wgpu::{Renderer as ImguiRenderer, RendererConfig};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop};

use crate::{
    compute_pass,
    mesh::{load_gltf_mesh, Vertex},
    photon_mapper::PhotonMapper,
    quad_pass,
    scene::SceneKind,
    window::create_window,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    view_inv: [[f32; 4]; 4],
    proj_inv: [[f32; 4]; 4],
    light_pos: [f32; 4],
    sphere_pos: [f32; 4],
    sphere_color: [f32; 4],
    mesh_center: [f32; 4],
    sun_intensity: f32,
    frame: u32,
    scene_kind: u32,
    render_width: u32,
    render_height: u32,
    _pad: [u32; 7],
}

struct Camera {
    pos: glam::Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn look_at(pos: glam::Vec3, target: glam::Vec3) -> Self {
        let forward = (target - pos).normalize_or_zero();
        let yaw = forward.x.atan2(forward.z);
        let pitch = forward.y.asin();
        Self { pos, yaw, pitch }
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

pub async fn run() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = create_window(&event_loop, "wgpu v0.29 ray tracing");

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
            required_limits: wgpu::Limits::default()
                .using_minimum_supported_acceleration_structure_values(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats[0];

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
    let mesh = load_gltf_mesh(decanter_path).expect("Failed to load decanter model");
    let model_verts = mesh.vertices;
    let model_idx = mesh.indices;

    println!(
        "Loaded {} vertices and {} indices from decanter",
        model_verts.len(),
        model_idx.len()
    );

    let mut min_pos = glam::Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_pos = glam::Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for vert in &model_verts {
        let pos = glam::Vec3::from(vert.position);
        min_pos = min_pos.min(pos);
        max_pos = max_pos.max(pos);
    }
    let center = (min_pos + max_pos) * 0.5;
    let size = max_pos - min_pos;
    let max_extent = size.max_element();
    let render_width = 1280u32;
    let render_height = 720u32;

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
    let pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_pos_buf"),
        contents: bytemuck::cast_slice(&mesh.positions4),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let nrm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_nrm_buf"),
        contents: bytemuck::cast_slice(&mesh.normals4),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_idx_buf"),
        contents: bytemuck::cast_slice(&model_idx),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tri_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_tri_mat_buf"),
        contents: bytemuck::cast_slice(&mesh.triangle_material_ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_materials_buf"),
        contents: bytemuck::cast_slice(&mesh.materials),
        usage: wgpu::BufferUsages::STORAGE,
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
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
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

    let mut accel_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("accel"),
    });
    accel_encoder.build_acceleration_structures([model_build].iter(), iter::once(&tlas));
    queue.submit(Some(accel_encoder.finish()));

    let projection = glam::Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_3 * 1.2,
        config.width as f32 / config.height as f32,
        0.1,
        1000.0,
    );

    let mut scene_kind = SceneKind::Decanter;
    let sphere_radius = 6.0;
    let ground_y = -1.5;
    let sphere_pos = glam::Vec3::new(
        center.x + size.x * 0.6 + 2.0,
        ground_y + sphere_radius,
        center.z,
    );
    let (camera_pos, camera_target) = scene_kind.default_camera(center);
    let mut camera = Camera::look_at(camera_pos, camera_target);
    let mut uniforms = SceneUniforms {
        view_inv: camera.view_matrix().inverse().to_cols_array_2d(),
        proj_inv: projection.inverse().to_cols_array_2d(),
        light_pos: [10.0, 8.0, 10.0, 1.0],
        sphere_pos: [sphere_pos.x, sphere_pos.y, sphere_pos.z, sphere_radius],
        sphere_color: [0.98, 1.0, 1.0, 1.0],
        mesh_center: [center.x, center.y, center.z, 0.0],
        sun_intensity: 0.8,
        frame: 0,
        scene_kind: scene_kind.index(),
        render_width,
        render_height,
        _pad: [0; 7],
    };

    let mut sun_azimuth_deg = uniforms.light_pos[2]
        .atan2(uniforms.light_pos[0])
        .to_degrees();
    let sun_len_xz = (uniforms.light_pos[0] * uniforms.light_pos[0]
        + uniforms.light_pos[2] * uniforms.light_pos[2])
        .sqrt();
    let mut sun_elevation_deg = uniforms.light_pos[1].atan2(sun_len_xz).to_degrees();
    let mut sun_intensity = uniforms.sun_intensity;

    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ubuf"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let accum_byte_size = (render_width as u64) * (render_height as u64) * 16;
    let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accum_buf"),
        size: accum_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    {
        let zeros = vec![0u8; accum_byte_size as usize];
        queue.write_buffer(&accum_buf, 0, &zeros);
    }

    let ubind = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ubind"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let compute_pass = compute_pass::ComputePass::new(&device, &ubind, render_width, render_height);
    let quad_pass = quad_pass::QuadPass::new(&device, surface_format, compute_pass.output_view());
    let mut photon_mapper = PhotonMapper::new(
        &device,
        &queue,
        &tlas,
        &pos_buf,
        &nrm_buf,
        &idx_buf,
        &tri_mat_buf,
        &mat_buf,
    );

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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: accum_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: nrm_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: idx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: tri_mat_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: mat_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(compute_pass.output_view()),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: photon_mapper.photon_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: photon_mapper.hash_heads().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: photon_mapper.uniforms_buffer().as_entire_binding(),
            },
        ],
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
    let mut accumulation_dirty = true;

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
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space)
                    {
                        keys_pressed.insert("Space".to_string());
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft)
                        || event.physical_key
                            == winit::keyboard::PhysicalKey::Code(
                                winit::keyboard::KeyCode::ShiftRight,
                            )
                    {
                        keys_pressed.insert("Shift".to_string());
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlLeft)
                        || event.physical_key
                            == winit::keyboard::PhysicalKey::Code(
                                winit::keyboard::KeyCode::ControlRight,
                            )
                    {
                        keys_pressed.insert("Control".to_string());
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp)
                    {
                        keys_pressed.insert("ArrowUp".to_string());
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown)
                    {
                        keys_pressed.insert("ArrowDown".to_string());
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowLeft)
                    {
                        keys_pressed.insert("ArrowLeft".to_string());
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowRight)
                    {
                        keys_pressed.insert("ArrowRight".to_string());
                    }
                }
                ElementState::Released => {
                    if let winit::keyboard::Key::Character(c) = &event.logical_key {
                        keys_pressed.remove(&c.to_lowercase().to_string());
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space)
                    {
                        keys_pressed.remove("Space");
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft)
                        || event.physical_key
                            == winit::keyboard::PhysicalKey::Code(
                                winit::keyboard::KeyCode::ShiftRight,
                            )
                    {
                        keys_pressed.remove("Shift");
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlLeft)
                        || event.physical_key
                            == winit::keyboard::PhysicalKey::Code(
                                winit::keyboard::KeyCode::ControlRight,
                            )
                    {
                        keys_pressed.remove("Control");
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp)
                    {
                        keys_pressed.remove("ArrowUp");
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown)
                    {
                        keys_pressed.remove("ArrowDown");
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowLeft)
                    {
                        keys_pressed.remove("ArrowLeft");
                    } else if event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowRight)
                    {
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
                uniforms.frame = 0;
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
                accumulation_dirty = true;
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
                    let prev_cam_pos = camera.pos;
                    let prev_cam_yaw = camera.yaw;
                    let prev_cam_pitch = camera.pitch;
                    let sprint = if keys_pressed.contains("Shift") {
                        3.0
                    } else {
                        1.0
                    };
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
                    if camera.pos != prev_cam_pos
                        || camera.yaw != prev_cam_yaw
                        || camera.pitch != prev_cam_pitch
                    {
                        accumulation_dirty = true;
                    }

                    match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(tex)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => {
                            imgui
                                .io_mut()
                                .update_delta_time(std::time::Duration::from_secs_f32(
                                    dt.max(1.0 / 1000.0),
                                ));
                            if platform.prepare_frame(imgui.io_mut(), &window).is_err() {
                                tex.present();
                                return;
                            }
                            let ui = imgui.frame();

                            let mut requested_scene = scene_kind;

                            ui.window("Scene")
                                .size([220.0, 96.0], Condition::FirstUseEver)
                                .build(|| {
                                    if ui.button("Decanter") {
                                        requested_scene = SceneKind::Decanter;
                                    }
                                    ui.same_line();
                                    if ui.button("Cornell Box") {
                                        requested_scene = SceneKind::CornellBox;
                                    }
                                    ui.text(format!("Active: {}", scene_kind.label()));
                                });

                            ui.window("Sun Controls")
                                .size([300.0, 160.0], Condition::FirstUseEver)
                                .build(|| {
                                    ui.slider_config("Azimuth (deg)", -180.0, 180.0)
                                        .build(&mut sun_azimuth_deg);
                                    ui.slider_config("Elevation (deg)", -10.0, 89.0)
                                        .build(&mut sun_elevation_deg);
                                    ui.slider_config("Intensity", 0.0, 5.0)
                                        .build(&mut sun_intensity);
                                });

                            if requested_scene != scene_kind {
                                scene_kind = requested_scene;
                                uniforms.scene_kind = scene_kind.index();
                                let (camera_pos, camera_target) = scene_kind.default_camera(center);
                                camera = Camera::look_at(camera_pos, camera_target);
                                uniforms.view_inv =
                                    camera.view_matrix().inverse().to_cols_array_2d();
                                accumulation_dirty = true;
                            }

                            let sun_az = sun_azimuth_deg.to_radians();
                            let sun_el = sun_elevation_deg.to_radians();
                            let sun_dir = glam::Vec3::new(
                                sun_az.cos() * sun_el.cos(),
                                sun_el.sin(),
                                sun_az.sin() * sun_el.cos(),
                            )
                            .normalize_or_zero();
                            let old_light = uniforms.light_pos;
                            let old_intensity = uniforms.sun_intensity;
                            uniforms.light_pos = [sun_dir.x, sun_dir.y, sun_dir.z, 1.0];
                            uniforms.sun_intensity = sun_intensity.max(0.0);
                            uniforms.scene_kind = scene_kind.index();

                            let sun_changed = uniforms.light_pos != old_light
                                || (uniforms.sun_intensity - old_intensity).abs() > f32::EPSILON;
                            if accumulation_dirty || sun_changed {
                                uniforms.frame = 0;
                                let zeros = vec![0u8; accum_byte_size as usize];
                                queue.write_buffer(&accum_buf, 0, &zeros);
                                accumulation_dirty = false;
                            } else {
                                uniforms.frame = uniforms.frame.saturating_add(1);
                            }
                            queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&uniforms));

                            let view = tex
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("enc"),
                                });
                            photon_mapper.update(
                                &queue,
                                uniforms.light_pos,
                                [center.x, center.y, center.z, max_extent * 0.85],
                                uniforms.frame,
                            );
                            photon_mapper.emit_photons(&mut encoder, 262_144);
                            photon_mapper.build_spatial_structure(&mut encoder);
                            {
                                compute_pass.record(&mut encoder, &ugroup);
                            }
                            {
                                let mut present_rpass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("present_pass"),
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                depth_slice: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        depth_stencil_attachment: None,
                                        multiview_mask: None,
                                        occlusion_query_set: None,
                                        timestamp_writes: None,
                                    });
                                quad_pass.render(&mut present_rpass);
                            }
                            platform.prepare_render(ui, &window);
                            let draw_data = imgui.render();
                            {
                                let mut ui_rpass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("imgui-pass"),
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                depth_slice: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Load,
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        depth_stencil_attachment: None,
                                        multiview_mask: None,
                                        occlusion_query_set: None,
                                        timestamp_writes: None,
                                    });
                                let _ = imgui_renderer.render(
                                    draw_data,
                                    &queue,
                                    &device,
                                    &mut ui_rpass,
                                );
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
