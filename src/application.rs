use std::iter;

use imgui::{Condition, FontConfig, FontSource, MouseButton};
use imgui_wgpu::{Renderer as ImguiRenderer, RendererConfig};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use transform_gizmo::config::TransformPivotPoint;
use transform_gizmo::{math::Transform as GizmoTransform, prelude::*};
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop};

use crate::{
    compute_pass,
    mesh::{load_gltf_mesh, MeshData, Vertex},
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
    decanter_center: [f32; 4],
    sun_intensity: f32,
    frame: u32,
    scene_kind: u32,
    render_width: u32,
    render_height: u32,
    selected_object: u32,
    _pad: [u32; 6],
}

struct Camera {
    pos: glam::Vec3,
    yaw: f32,
    pitch: f32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum GizmoModeKind {
    Translate,
    Rotate,
    Scale,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum GizmoTargetKind {
    Sphere,
    Decanter,
    WineGlass,
    WineSpotlight,
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

fn mesh_bounds(vertices: &[Vertex]) -> (glam::Vec3, glam::Vec3, glam::Vec3, glam::Vec3) {
    let mut min_pos = glam::Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_pos = glam::Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for vert in vertices {
        let pos = glam::Vec3::from(vert.position);
        min_pos = min_pos.min(pos);
        max_pos = max_pos.max(pos);
    }
    let center = (min_pos + max_pos) * 0.5;
    let size = max_pos - min_pos;
    (center, size, min_pos, max_pos)
}

fn translate_mesh(mesh: &mut MeshData, offset: glam::Vec3) {
    for vertex in &mut mesh.vertices {
        let pos = glam::Vec3::from(vertex.position) + offset;
        vertex.position = pos.to_array();
    }
    for pos in &mut mesh.positions4 {
        pos[0] += offset.x;
        pos[1] += offset.y;
        pos[2] += offset.z;
    }
}

fn orient_and_scale_mesh(mesh: &mut MeshData, pivot: glam::Vec3, rotation: glam::Quat, scale: f32) {
    for vertex in &mut mesh.vertices {
        let pos = glam::Vec3::from(vertex.position);
        vertex.position = (pivot + rotation * ((pos - pivot) * scale)).to_array();
    }
    for pos in &mut mesh.positions4 {
        let p = glam::Vec3::new(pos[0], pos[1], pos[2]);
        let transformed = pivot + rotation * ((p - pivot) * scale);
        pos[0] = transformed.x;
        pos[1] = transformed.y;
        pos[2] = transformed.z;
    }
    for normal in &mut mesh.normals4 {
        let transformed = rotation * glam::Vec3::new(normal[0], normal[1], normal[2]);
        normal[0] = transformed.x;
        normal[1] = transformed.y;
        normal[2] = transformed.z;
    }
}

fn append_mesh(base: &mut MeshData, extra: MeshData) {
    let vertex_offset = base.positions4.len() as u32;
    let material_offset = base.materials.len() as u32;

    base.vertices.extend(extra.vertices);
    base.positions4.extend(extra.positions4);
    base.normals4.extend(extra.normals4);
    base.indices
        .extend(extra.indices.into_iter().map(|index| index + vertex_offset));
    base.triangle_material_ids.extend(
        extra
            .triangle_material_ids
            .into_iter()
            .map(|material_id| material_id + material_offset),
    );
    base.materials.extend(extra.materials);
}

fn sphere_position_for(center: glam::Vec3, size: glam::Vec3, radius: f32) -> glam::Vec3 {
    glam::Vec3::new(center.x + size.x * 0.6 + 2.0, -1.5 + radius, center.z)
}

fn scene_camera(scene_kind: SceneKind, center: glam::Vec3, size: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    if scene_kind == SceneKind::Wine {
        let distance = size.max_element().max(12.0) * 1.35;
        return (center + glam::Vec3::new(0.0, size.y * 0.2, distance), center);
    }
    scene_kind.default_camera(center)
}

fn wine_spotlight_position(
    center: glam::Vec3,
    azimuth_deg: f32,
    elevation_deg: f32,
    distance: f32,
) -> glam::Vec3 {
    let azimuth = azimuth_deg.to_radians();
    let elevation = elevation_deg.to_radians();
    let dir_from_target = glam::Vec3::new(
        azimuth.cos() * elevation.cos(),
        elevation.sin(),
        azimuth.sin() * elevation.cos(),
    )
    .normalize_or_zero();
    center + dir_from_target * distance.max(1.0)
}

fn world_ray_from_cursor(
    cursor: [f32; 2],
    viewport: [f32; 2],
    view_inv: glam::Mat4,
    proj_inv: glam::Mat4,
) -> (glam::Vec3, glam::Vec3) {
    let ndc_x = (cursor[0] / viewport[0]) * 2.0 - 1.0;
    let ndc_y = (1.0 - cursor[1] / viewport[1]) * 2.0 - 1.0;
    let cam_far = proj_inv * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let far_pos = cam_far.truncate() / cam_far.w.max(1e-6);
    let origin = (view_inv * glam::Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate();
    let far_world = (view_inv * glam::Vec4::new(far_pos.x, far_pos.y, far_pos.z, 1.0)).truncate();
    (origin, (far_world - origin).normalize_or_zero())
}

fn intersect_sphere(origin: glam::Vec3, dir: glam::Vec3, center: glam::Vec3, radius: f32) -> Option<f32> {
    let oc = origin - center;
    let a = dir.dot(dir);
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - a * c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t1 = (-b - sq) / a;
    let t2 = (-b + sq) / a;
    if t1 > 0.001 {
        Some(t1)
    } else if t2 > 0.001 {
        Some(t2)
    } else {
        None
    }
}

fn update_mesh_transform(
    mesh: &mut MeshData,
    model_verts: &mut [Vertex],
    start: usize,
    count: usize,
    base_positions: &[glam::Vec3],
    base_normals: &[glam::Vec3],
    pivot: glam::Vec3,
    scale: glam::Vec3,
    rotation: glam::Quat,
    translation: glam::Vec3,
) {
    for i in 0..count {
        let idx = start + i;
        let local = base_positions[i] - pivot;
        let scaled = glam::Vec3::new(local.x * scale.x, local.y * scale.y, local.z * scale.z);
        let p = pivot + rotation * scaled + translation;
        let n = (rotation * base_normals[i]).normalize_or_zero();
        model_verts[idx].position = p.to_array();
        mesh.positions4[idx][0] = p.x;
        mesh.positions4[idx][1] = p.y;
        mesh.positions4[idx][2] = p.z;
        mesh.normals4[idx][0] = n.x;
        mesh.normals4[idx][1] = n.y;
        mesh.normals4[idx][2] = n.z;
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
    let wine_path = std::path::Path::new("res/red_wine_glass.glb");
    let mut mesh = load_gltf_mesh(decanter_path).expect("Failed to load decanter model");
    let decanter_vertex_start = 0usize;
    let decanter_vertex_count = mesh.positions4.len();
    let decanter_base_positions: Vec<glam::Vec3> = mesh
        .positions4
        .iter()
        .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
        .collect();
    let decanter_base_normals: Vec<glam::Vec3> = mesh
        .normals4
        .iter()
        .map(|n| glam::Vec3::new(n[0], n[1], n[2]))
        .collect();
    let mut wine_mesh = load_gltf_mesh(wine_path).expect("Failed to load red wine model");
    let (decanter_center, decanter_size, _, decanter_max) = mesh_bounds(&mesh.vertices);
    let (wine_original_center, _, _, _) = mesh_bounds(&wine_mesh.vertices);
    orient_and_scale_mesh(
        &mut wine_mesh,
        wine_original_center,
        glam::Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        25.0,
    );
    let (wine_oriented_center, wine_size, wine_min, _) = mesh_bounds(&wine_mesh.vertices);
    let wine_target_center = glam::Vec3::new(
        decanter_max.x + wine_size.x * 0.5 + 36.0,
        wine_oriented_center.y + (-1.5 - wine_min.y),
        decanter_center.z,
    );
    translate_mesh(&mut wine_mesh, wine_target_center - wine_oriented_center);
    let (wine_center, wine_size, _, _) = mesh_bounds(&wine_mesh.vertices);
    let wine_vertex_start = mesh.positions4.len();
    let wine_vertex_count = wine_mesh.positions4.len();
    append_mesh(&mut mesh, wine_mesh);
    let wine_base_positions: Vec<glam::Vec3> =
        mesh.positions4[wine_vertex_start..wine_vertex_start + wine_vertex_count]
            .iter()
            .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
            .collect();
    let wine_base_normals: Vec<glam::Vec3> =
        mesh.normals4[wine_vertex_start..wine_vertex_start + wine_vertex_count]
            .iter()
            .map(|n| glam::Vec3::new(n[0], n[1], n[2]))
            .collect();

    let mut model_verts = mesh.vertices.clone();
    let model_idx = mesh.indices.clone();

    println!(
        "Loaded {} vertices and {} indices from decanter + wine",
        model_verts.len(),
        model_idx.len()
    );

    let (center, size, _, _) = mesh_bounds(&model_verts);
    let decanter_max_extent = decanter_size.max_element();
    let wine_max_extent = wine_size.max_element();
    let render_width = 1280u32;
    let render_height = 720u32;

    println!(
        "Scene bounds: decanter center={:?}, wine center={:?}, combined center={:?}, size={:?}",
        decanter_center, wine_center, center, size
    );

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("model_vbuf"),
        contents: bytemuck::cast_slice(&model_verts),
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("model_ibuf"),
        contents: bytemuck::cast_slice(&model_idx),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
    });
    let pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_pos_buf"),
        contents: bytemuck::cast_slice(&mesh.positions4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let nrm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_nrm_buf"),
        contents: bytemuck::cast_slice(&mesh.normals4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
    let mut active_center = decanter_center;
    let mut active_max_extent = decanter_max_extent;
    let sphere_pos = sphere_position_for(active_center, decanter_size, sphere_radius);
    let (camera_pos, camera_target) = scene_camera(scene_kind, active_center, decanter_size);
    let mut camera = Camera::look_at(camera_pos, camera_target);
    let mut uniforms = SceneUniforms {
        view_inv: camera.view_matrix().inverse().to_cols_array_2d(),
        proj_inv: projection.inverse().to_cols_array_2d(),
        light_pos: [10.0, 8.0, 10.0, 1.0],
        sphere_pos: [sphere_pos.x, sphere_pos.y, sphere_pos.z, sphere_radius],
        sphere_color: [0.98, 1.0, 1.0, 1.0],
        mesh_center: [wine_center.x, wine_center.y, wine_center.z, wine_max_extent * 0.8],
        decanter_center: [decanter_center.x, decanter_center.y, decanter_center.z, decanter_max_extent * 0.7],
        sun_intensity: 0.8,
        frame: 0,
        scene_kind: scene_kind.index(),
        render_width,
        render_height,
        selected_object: 1,
        _pad: [0; 6],
    };

    let mut sun_azimuth_deg = uniforms.light_pos[2]
        .atan2(uniforms.light_pos[0])
        .to_degrees();
    let sun_len_xz = (uniforms.light_pos[0] * uniforms.light_pos[0]
        + uniforms.light_pos[2] * uniforms.light_pos[2])
        .sqrt();
    let mut sun_elevation_deg = uniforms.light_pos[1].atan2(sun_len_xz).to_degrees();
    let mut sun_intensity = uniforms.sun_intensity;
    let mut wine_spotlight_azimuth_deg = -55.0;
    let mut wine_spotlight_elevation_deg = 54.0;
    let mut wine_spotlight_distance = wine_max_extent.max(10.0) * 1.4;

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
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let compute_pass = compute_pass::ComputePass::new(&device, &ubind, render_width, render_height);
    let quad_pass = quad_pass::QuadPass::new(
        &device,
        surface_format,
        compute_pass.output_view(),
        compute_pass.selection_mask_view(),
    );
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
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(compute_pass.selection_mask_view()),
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
    let mut gizmo = Gizmo::default();
    let mut gizmo_mode = GizmoModeKind::Translate;
    let mut gizmo_target = GizmoTargetKind::Sphere;
    let mut sphere_rotation = glam::Quat::IDENTITY;
    let mut sphere_radius_scale = 1.0f32;
    let mut decanter_rotation = glam::Quat::IDENTITY;
    let mut decanter_translation = glam::Vec3::ZERO;
    let mut decanter_scale = glam::Vec3::ONE;
    let mut wine_rotation = glam::Quat::IDENTITY;
    let mut wine_translation = glam::Vec3::ZERO;
    let mut wine_scale = glam::Vec3::ONE;
    let mut geometry_dirty = false;

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
                                    if ui.button("Wine") {
                                        requested_scene = SceneKind::Wine;
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

                            ui.window("Wine Spotlight")
                                .size([300.0, 150.0], Condition::FirstUseEver)
                                .build(|| {
                                    ui.slider_config("Azimuth (deg)", -180.0, 180.0)
                                        .build(&mut wine_spotlight_azimuth_deg);
                                    ui.slider_config("Elevation (deg)", 5.0, 85.0)
                                        .build(&mut wine_spotlight_elevation_deg);
                                    ui.slider_config("Distance", 2.0, wine_max_extent.max(10.0) * 4.0)
                                        .build(&mut wine_spotlight_distance);
                                });

                            ui.window("Transform Gizmo")
                                .size([330.0, 120.0], Condition::FirstUseEver)
                                .build(|| {
                                    ui.text("Target");
                                    if ui.radio_button_bool("Sphere", gizmo_target == GizmoTargetKind::Sphere) {
                                        gizmo_target = GizmoTargetKind::Sphere;
                                    }
                                    ui.same_line();
                                    if ui.radio_button_bool("Decanter", gizmo_target == GizmoTargetKind::Decanter) {
                                        gizmo_target = GizmoTargetKind::Decanter;
                                    }
                                    ui.same_line();
                                    if ui.radio_button_bool("Wine Glass", gizmo_target == GizmoTargetKind::WineGlass) {
                                        gizmo_target = GizmoTargetKind::WineGlass;
                                    }
                                    ui.same_line();
                                    if ui.radio_button_bool("Spotlight", gizmo_target == GizmoTargetKind::WineSpotlight) {
                                        gizmo_target = GizmoTargetKind::WineSpotlight;
                                    }

                                    ui.text("Mode");
                                    if ui.radio_button_bool(
                                        "Translate",
                                        gizmo_mode == GizmoModeKind::Translate,
                                    ) {
                                        gizmo_mode = GizmoModeKind::Translate;
                                    }
                                    ui.same_line();
                                    if ui.radio_button_bool("Rotate", gizmo_mode == GizmoModeKind::Rotate)
                                    {
                                        gizmo_mode = GizmoModeKind::Rotate;
                                    }
                                    ui.same_line();
                                    if ui.radio_button_bool("Scale", gizmo_mode == GizmoModeKind::Scale) {
                                        gizmo_mode = GizmoModeKind::Scale;
                                    }
                                });

                            if ui.is_mouse_clicked(MouseButton::Left) && !ui.io().want_capture_mouse {
                                let display_size = ui.io().display_size;
                                let (ro, rd) = world_ray_from_cursor(
                                    ui.io().mouse_pos,
                                    [display_size[0].max(1.0), display_size[1].max(1.0)],
                                    camera.view_matrix().inverse(),
                                    projection.inverse(),
                                );
                                let sphere_center =
                                    glam::Vec3::new(uniforms.sphere_pos[0], uniforms.sphere_pos[1], uniforms.sphere_pos[2]);
                                let decanter_center_now = decanter_center + decanter_translation;
                                let wine_center_now = wine_center + wine_translation;
                                let sphere_hit = intersect_sphere(ro, rd, sphere_center, uniforms.sphere_pos[3]);
                                let decanter_hit = intersect_sphere(
                                    ro,
                                    rd,
                                    decanter_center_now,
                                    (decanter_max_extent * decanter_scale.max_element() * 0.55).max(0.25),
                                );
                                let wine_hit = intersect_sphere(ro, rd, wine_center_now, (wine_max_extent * 0.55).max(0.25));
                                let mut best = gizmo_target;
                                let mut best_t = f32::INFINITY;
                                if let Some(t) = sphere_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = GizmoTargetKind::Sphere;
                                    }
                                }
                                if let Some(t) = decanter_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = GizmoTargetKind::Decanter;
                                    }
                                }
                                if let Some(t) = wine_hit {
                                    if t < best_t {
                                        best = GizmoTargetKind::WineGlass;
                                    }
                                }
                                gizmo_target = best;
                            }

                            if requested_scene != scene_kind {
                                scene_kind = requested_scene;
                                uniforms.scene_kind = scene_kind.index();
                                let (next_center, next_size, next_extent) = match scene_kind {
                                    SceneKind::Decanter => {
                                        (decanter_center, decanter_size, decanter_max_extent)
                                    }
                                    SceneKind::Wine => (wine_center, wine_size, wine_max_extent),
                                    SceneKind::CornellBox => {
                                        (glam::Vec3::ZERO, glam::Vec3::splat(2.0), 2.0)
                                    }
                                };
                                active_center = next_center;
                                active_max_extent = next_extent;
                                let sphere_pos =
                                    sphere_position_for(active_center, next_size, sphere_radius);
                                uniforms.sphere_pos =
                                    [sphere_pos.x, sphere_pos.y, sphere_pos.z, sphere_radius * sphere_radius_scale];
                                uniforms.mesh_center = [
                                    wine_center.x + wine_translation.x,
                                    wine_center.y + wine_translation.y,
                                    wine_center.z + wine_translation.z,
                                    wine_max_extent * wine_scale.max_element() * 0.8,
                                ];
                                let (camera_pos, camera_target) =
                                    scene_camera(scene_kind, active_center, next_size);
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
                            uniforms.light_pos = if scene_kind == SceneKind::Wine {
                                let spot_position = wine_spotlight_position(
                                    active_center,
                                    wine_spotlight_azimuth_deg,
                                    wine_spotlight_elevation_deg,
                                    wine_spotlight_distance,
                                );
                                [spot_position.x, spot_position.y, spot_position.z, -1.0]
                            } else {
                                [sun_dir.x, sun_dir.y, sun_dir.z, 1.0]
                            };

                            let view = camera.view_matrix();
                            let projection = glam::Mat4::perspective_rh(
                                std::f32::consts::FRAC_PI_3 * 1.2,
                                config.width as f32 / config.height as f32,
                                0.1,
                                1000.0,
                            );
                            let display_size = ui.io().display_size;
                            let viewport = Rect::from_min_max(
                                [0.0, 0.0].into(),
                                [display_size[0].max(1.0), display_size[1].max(1.0)].into(),
                            );
                            let cursor = ui.io().mouse_pos;
                            let interaction = GizmoInteraction {
                                cursor_pos: (cursor[0], cursor[1]),
                                hovered: !ui.io().want_capture_mouse,
                                drag_started: ui.is_mouse_clicked(MouseButton::Left),
                                dragging: ui.is_mouse_down(MouseButton::Left),
                            };
                            let gizmo_modes = match gizmo_mode {
                                GizmoModeKind::Translate => GizmoMode::all_translate(),
                                GizmoModeKind::Rotate => GizmoMode::all_rotate(),
                                GizmoModeKind::Scale => GizmoMode::all_scale(),
                            };
                            let view_cols = view.to_cols_array();
                            let proj_cols = projection.to_cols_array();
                            let view_matrix = transform_gizmo::math::DMat4::from_cols_array(&[
                                view_cols[0] as f64,
                                view_cols[1] as f64,
                                view_cols[2] as f64,
                                view_cols[3] as f64,
                                view_cols[4] as f64,
                                view_cols[5] as f64,
                                view_cols[6] as f64,
                                view_cols[7] as f64,
                                view_cols[8] as f64,
                                view_cols[9] as f64,
                                view_cols[10] as f64,
                                view_cols[11] as f64,
                                view_cols[12] as f64,
                                view_cols[13] as f64,
                                view_cols[14] as f64,
                                view_cols[15] as f64,
                            ]);
                            let projection_matrix = transform_gizmo::math::DMat4::from_cols_array(
                                &[
                                    proj_cols[0] as f64,
                                    proj_cols[1] as f64,
                                    proj_cols[2] as f64,
                                    proj_cols[3] as f64,
                                    proj_cols[4] as f64,
                                    proj_cols[5] as f64,
                                    proj_cols[6] as f64,
                                    proj_cols[7] as f64,
                                    proj_cols[8] as f64,
                                    proj_cols[9] as f64,
                                    proj_cols[10] as f64,
                                    proj_cols[11] as f64,
                                    proj_cols[12] as f64,
                                    proj_cols[13] as f64,
                                    proj_cols[14] as f64,
                                    proj_cols[15] as f64,
                                ],
                            );
                            gizmo.update_config(GizmoConfig {
                                view_matrix: view_matrix.into(),
                                projection_matrix: projection_matrix.into(),
                                viewport,
                                modes: gizmo_modes,
                                mode_override: None,
                                orientation: GizmoOrientation::Global,
                                pivot_point: TransformPivotPoint::MedianPoint,
                                snapping: false,
                                snap_angle: 15f32.to_radians(),
                                snap_distance: 0.5,
                                snap_scale: 0.1,
                                visuals: GizmoVisuals::default(),
                                pixels_per_point: ui.io().display_framebuffer_scale[0].max(1.0),
                            });

                            let target_transform = match gizmo_target {
                                GizmoTargetKind::Sphere => GizmoTransform::from_scale_rotation_translation(
                                    transform_gizmo::math::DVec3::new(
                                        sphere_radius_scale as f64,
                                        sphere_radius_scale as f64,
                                        sphere_radius_scale as f64,
                                    ),
                                    transform_gizmo::math::DQuat::from_xyzw(
                                        sphere_rotation.x as f64,
                                        sphere_rotation.y as f64,
                                        sphere_rotation.z as f64,
                                        sphere_rotation.w as f64,
                                    ),
                                    transform_gizmo::math::DVec3::new(
                                        uniforms.sphere_pos[0] as f64,
                                        uniforms.sphere_pos[1] as f64,
                                        uniforms.sphere_pos[2] as f64,
                                    ),
                                ),
                                GizmoTargetKind::Decanter => {
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            decanter_scale.x as f64,
                                            decanter_scale.y as f64,
                                            decanter_scale.z as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            decanter_rotation.x as f64,
                                            decanter_rotation.y as f64,
                                            decanter_rotation.z as f64,
                                            decanter_rotation.w as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            (decanter_center.x + decanter_translation.x) as f64,
                                            (decanter_center.y + decanter_translation.y) as f64,
                                            (decanter_center.z + decanter_translation.z) as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::WineGlass => {
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            wine_scale.x as f64,
                                            wine_scale.y as f64,
                                            wine_scale.z as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            wine_rotation.x as f64,
                                            wine_rotation.y as f64,
                                            wine_rotation.z as f64,
                                            wine_rotation.w as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            (wine_center.x + wine_translation.x) as f64,
                                            (wine_center.y + wine_translation.y) as f64,
                                            (wine_center.z + wine_translation.z) as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::WineSpotlight => {
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::ONE,
                                        transform_gizmo::math::DQuat::IDENTITY,
                                        transform_gizmo::math::DVec3::new(
                                            uniforms.light_pos[0] as f64,
                                            uniforms.light_pos[1] as f64,
                                            uniforms.light_pos[2] as f64,
                                        ),
                                    )
                                }
                            };

                            if let Some((_result, transforms)) = gizmo.update(interaction, &[target_transform]) {
                                let new_t = transforms[0];
                                let translation = glam::Vec3::new(
                                    new_t.translation.x as f32,
                                    new_t.translation.y as f32,
                                    new_t.translation.z as f32,
                                );
                                match gizmo_target {
                                    GizmoTargetKind::Sphere => {
                                        uniforms.sphere_pos[0] = translation.x;
                                        uniforms.sphere_pos[1] = translation.y;
                                        uniforms.sphere_pos[2] = translation.z;
                                        sphere_radius_scale = (new_t.scale.x as f32).clamp(0.15, 8.0);
                                        uniforms.sphere_pos[3] = sphere_radius * sphere_radius_scale;
                                        sphere_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                    }
                                    GizmoTargetKind::Decanter => {
                                        let new_center = glam::Vec3::new(translation.x, translation.y, translation.z);
                                        decanter_translation = new_center - decanter_center;
                                        decanter_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        decanter_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                        geometry_dirty = true;
                                    }
                                    GizmoTargetKind::WineGlass => {
                                        let new_center = glam::Vec3::new(translation.x, translation.y, translation.z);
                                        wine_translation = new_center - wine_center;
                                        wine_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        wine_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                        geometry_dirty = true;
                                    }
                                    GizmoTargetKind::WineSpotlight => {
                                        uniforms.light_pos[0] = translation.x;
                                        uniforms.light_pos[1] = translation.y;
                                        uniforms.light_pos[2] = translation.z;
                                    }
                                }
                                accumulation_dirty = true;
                            }

                            {
                                let draw_data = gizmo.draw();
                                let fg = ui.get_foreground_draw_list();
                                for idx in (0..draw_data.indices.len()).step_by(3) {
                                    let i0 = draw_data.indices[idx] as usize;
                                    let i1 = draw_data.indices[idx + 1] as usize;
                                    let i2 = draw_data.indices[idx + 2] as usize;
                                    let p0 = draw_data.vertices[i0];
                                    let p1 = draw_data.vertices[i1];
                                    let p2 = draw_data.vertices[i2];
                                    let c = draw_data.colors[i0];
                                    let color =
                                        imgui::ImColor32::from_rgba_f32s(c[0], c[1], c[2], c[3]);
                                    fg.add_triangle(
                                        [p0[0], p0[1]],
                                        [p1[0], p1[1]],
                                        [p2[0], p2[1]],
                                        color,
                                    )
                                    .filled(true)
                                    .build();
                                }
                            }
                            uniforms.sun_intensity = sun_intensity.max(0.0);
                            uniforms.scene_kind = scene_kind.index();
                            uniforms.mesh_center = [
                                wine_center.x + wine_translation.x,
                                wine_center.y + wine_translation.y,
                                wine_center.z + wine_translation.z,
                                wine_max_extent * wine_scale.max_element() * 0.8,
                            ];
                            uniforms.decanter_center = [
                                decanter_center.x + decanter_translation.x,
                                decanter_center.y + decanter_translation.y,
                                decanter_center.z + decanter_translation.z,
                                decanter_max_extent * decanter_scale.max_element() * 0.7,
                            ];
                            uniforms.selected_object = match gizmo_target {
                                GizmoTargetKind::Sphere => 1,
                                GizmoTargetKind::Decanter => 3,
                                GizmoTargetKind::WineGlass => 2,
                                GizmoTargetKind::WineSpotlight => 0,
                            };

                            let sun_changed = uniforms.light_pos != old_light
                                || (uniforms.sun_intensity - old_intensity).abs() > f32::EPSILON;
                            if accumulation_dirty || sun_changed {
                                if geometry_dirty {
                                    update_mesh_transform(
                                        &mut mesh,
                                        &mut model_verts,
                                        decanter_vertex_start,
                                        decanter_vertex_count,
                                        &decanter_base_positions,
                                        &decanter_base_normals,
                                        decanter_center,
                                        decanter_scale,
                                        decanter_rotation,
                                        decanter_translation,
                                    );
                                    update_mesh_transform(
                                        &mut mesh,
                                        &mut model_verts,
                                        wine_vertex_start,
                                        wine_vertex_count,
                                        &wine_base_positions,
                                        &wine_base_normals,
                                        wine_center,
                                        wine_scale,
                                        wine_rotation,
                                        wine_translation,
                                    );
                                    queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&model_verts));
                                    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&mesh.positions4));
                                    queue.write_buffer(&nrm_buf, 0, bytemuck::cast_slice(&mesh.normals4));

                                    let model_build = wgpu::BlasBuildEntry {
                                        blas: &model_blas,
                                        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                                            wgpu::BlasTriangleGeometry {
                                                size: &model_blas_desc,
                                                vertex_buffer: &vbuf,
                                                first_vertex: 0,
                                                vertex_stride: std::mem::size_of::<Vertex>() as u64,
                                                index_buffer: Some(&ibuf),
                                                first_index: Some(0),
                                                transform_buffer: None,
                                                transform_buffer_offset: None,
                                            },
                                        ]),
                                    };
                                    let mut accel_encoder =
                                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                            label: Some("accel_update"),
                                        });
                                    accel_encoder.build_acceleration_structures(
                                        [model_build].iter(),
                                        iter::once(&tlas),
                                    );
                                    queue.submit(Some(accel_encoder.finish()));
                                    geometry_dirty = false;
                                }
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
                                [
                                    active_center.x,
                                    active_center.y,
                                    active_center.z,
                                    active_max_extent * 0.85,
                                ],
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
