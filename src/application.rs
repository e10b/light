use std::iter;

use egui::{Color32, Pos2, Stroke, ViewportId};
use egui_wgpu::{Renderer as EguiRenderer, RendererOptions, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use transform_gizmo::config::TransformPivotPoint;
use transform_gizmo::{math::Transform as GizmoTransform, prelude::*};
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop};

use crate::{
    blender_data::{Id, MainDatabase, Transform as DbTransform},
    compute_pass,
    mesh::{load_gltf_mesh, MeshData, Vertex},
    prism_file::{
        load_prism_database, save_prism_file, CollectionData as PrismCollectionData,
        MaterialData as PrismMaterialData,
        MeshData as PrismMeshData, NodeProperties, NodeType, ObjectData as PrismObjectData,
        ObjectDataLink as PrismObjectDataLink, PrismDatabase, SceneData as PrismSceneData,
        ShaderNode,
    },
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
    cornell_center: [f32; 4],
    sun_intensity: f32,
    frame: u32,
    scene_kind: u32,
    render_width: u32,
    render_height: u32,
    selected_object: u32,
    mesh_enabled: u32,
    decanter_enabled: u32,
    wine_enabled: u32,
    cornell_enabled: u32,
    _pad: [u32; 2],
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
    CornellBox,
    SunLamp,
    WineSpotlight,
}

fn default_target_for_scene(scene_kind: SceneKind) -> GizmoTargetKind {
    match scene_kind {
        SceneKind::Decanter => GizmoTargetKind::Decanter,
        SceneKind::Wine => GizmoTargetKind::WineGlass,
        SceneKind::CornellBox => GizmoTargetKind::CornellBox,
    }
}

fn target_allowed_in_scene(scene_kind: SceneKind, target: GizmoTargetKind) -> bool {
    match scene_kind {
        SceneKind::Decanter => matches!(
            target,
            GizmoTargetKind::Sphere
                | GizmoTargetKind::Decanter
                | GizmoTargetKind::WineGlass
                | GizmoTargetKind::CornellBox
                | GizmoTargetKind::SunLamp
        ),
        SceneKind::Wine => matches!(target, GizmoTargetKind::WineGlass | GizmoTargetKind::WineSpotlight),
        SceneKind::CornellBox => matches!(target, GizmoTargetKind::CornellBox),
    }
}

fn target_label(target: GizmoTargetKind) -> &'static str {
    match target {
        GizmoTargetKind::Sphere => "Cube",
        GizmoTargetKind::Decanter => "Decanter",
        GizmoTargetKind::WineGlass => "Wine Glass",
        GizmoTargetKind::CornellBox => "Cornell Box",
        GizmoTargetKind::SunLamp => "Sun Lamp",
        GizmoTargetKind::WineSpotlight => "Spotlight",
    }
}

fn transform_to_matrix(t: &DbTransform) -> [f32; 16] {
    glam::Mat4::from_scale_rotation_translation(t.scale, t.rotation, t.location).to_cols_array()
}

fn build_prism_database_from_main(
    main_db: &MainDatabase,
    decanter_scene_id: Id,
    wine_scene_id: Id,
    cornell_scene_id: Id,
) -> PrismDatabase {
    let mut out = PrismDatabase::new();

    let mut mesh_map: std::collections::HashMap<Id, crate::prism_file::MeshHandle> =
        std::collections::HashMap::new();
    for (mid, mesh) in &main_db.meshes {
        let h = out.meshes.insert(PrismMeshData {
            vertices: vec![[0.0, 0.0, 0.0]; mesh.vertex_count],
            indices: Vec::new(),
            material_slots: Vec::new(),
        });
        mesh_map.insert(*mid, h);
    }

    let default_mat = out.materials.insert(PrismMaterialData {
        name: "DefaultMaterial".to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            let n_in = g.add_node(ShaderNode {
                node_type: NodeType::FloatInput,
                properties: NodeProperties {
                    float_value: Some(0.5),
                    vec3_value: None,
                },
            });
            let n_out = g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            g.add_edge(
                n_in,
                n_out,
                crate::prism_file::NodeLink {
                    output_socket: "Value".to_string(),
                    input_socket: "Surface".to_string(),
                },
            );
            g
        },
    });

    for mesh in out.meshes.values_mut() {
        mesh.material_slots.push(Some(default_mat));
    }

    let mut object_map: std::collections::HashMap<Id, crate::prism_file::ObjectHandle> =
        std::collections::HashMap::new();
    for (oid, obj) in &main_db.objects {
        let mesh_link = obj.mesh_id.and_then(|m| mesh_map.get(&m).copied());
        let h = out.objects.insert(PrismObjectData {
            name: obj.name.clone(),
            transform_matrix: transform_to_matrix(&obj.transform),
            data_link: mesh_link
                .map(PrismObjectDataLink::Mesh)
                .unwrap_or(PrismObjectDataLink::None),
        });
        object_map.insert(*oid, h);
    }

    let mut collection_map: std::collections::HashMap<Id, crate::prism_file::CollectionHandle> =
        std::collections::HashMap::new();
    for (cid, col) in &main_db.collections {
        let h = out.collections.insert(PrismCollectionData {
            name: col.name.clone(),
            objects: Vec::new(),
            children: Vec::new(),
        });
        collection_map.insert(*cid, h);
    }
    for (cid, col) in &main_db.collections {
        if let Some(ch) = collection_map.get(cid).copied() {
            if let Some(out_col) = out.collections.get_mut(ch) {
                out_col.objects = col
                    .object_ids
                    .iter()
                    .filter_map(|id| object_map.get(id).copied())
                    .collect();
                out_col.children = col
                    .child_collection_ids
                    .iter()
                    .filter_map(|id| collection_map.get(id).copied())
                    .collect();
            }
        }
    }

    for scene_id in [decanter_scene_id, wine_scene_id, cornell_scene_id] {
        if let Some(scene) = main_db.scenes.get(&scene_id) {
            if let Some(master) = collection_map.get(&scene.master_collection_id).copied() {
                out.scenes.insert(PrismSceneData {
                    name: scene.name.clone(),
                    master_collection: master,
                });
            }
        }
    }

    out
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

fn world_to_screen(
    point: glam::Vec3,
    view: glam::Mat4,
    proj: glam::Mat4,
    viewport: [f32; 2],
) -> Option<[f32; 2]> {
    let clip = proj * view * glam::Vec4::new(point.x, point.y, point.z, 1.0);
    if clip.w.abs() < 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z < -1.0 || ndc.z > 1.0 {
        return None;
    }
    let x = (ndc.x * 0.5 + 0.5) * viewport[0];
    let y = (1.0 - (ndc.y * 0.5 + 0.5)) * viewport[1];
    Some([x, y])
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

fn intersect_cube(origin: glam::Vec3, dir: glam::Vec3, center: glam::Vec3, half_extent: f32) -> Option<f32> {
    let min = center - glam::Vec3::splat(half_extent);
    let max = center + glam::Vec3::splat(half_extent);
    let inv = glam::Vec3::new(
        if dir.x.abs() > 1e-6 { 1.0 / dir.x } else { f32::INFINITY },
        if dir.y.abs() > 1e-6 { 1.0 / dir.y } else { f32::INFINITY },
        if dir.z.abs() > 1e-6 { 1.0 / dir.z } else { f32::INFINITY },
    );
    let t0 = (min - origin) * inv;
    let t1 = (max - origin) * inv;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let near = tmin.x.max(tmin.y).max(tmin.z);
    let far = tmax.x.min(tmax.y).min(tmax.z);
    if far < 0.0 || near > far {
        return None;
    }
    if near > 0.001 {
        Some(near)
    } else if far > 0.001 {
        Some(far)
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

    let mut main_db = MainDatabase::new();
    let decanter_mesh_id = main_db.create_mesh("DecanterMesh", decanter_vertex_count);
    let wine_mesh_id = main_db.create_mesh("WineGlassMesh", wine_vertex_count);
    let cornell_mesh_id = main_db.create_mesh("CornellBoxMesh", 0);
    let sphere_obj_id = main_db.create_object("Cube", None, DbTransform::default());
    let sun_obj_id = main_db.create_object("SunLamp", None, DbTransform::default());
    let spot_obj_id = main_db.create_object("Spotlight", None, DbTransform::default());
    let decanter_obj_id = main_db.create_object("Decanter", Some(decanter_mesh_id), DbTransform::default());
    let wine_obj_id = main_db.create_object("WineGlass", Some(wine_mesh_id), DbTransform::default());
    let cornell_obj_id = main_db.create_object("CornellBox", Some(cornell_mesh_id), DbTransform::default());

    let mut object_target_by_id: std::collections::HashMap<Id, GizmoTargetKind> =
        std::collections::HashMap::new();
    object_target_by_id.insert(sphere_obj_id, GizmoTargetKind::Sphere);
    object_target_by_id.insert(sun_obj_id, GizmoTargetKind::SunLamp);
    object_target_by_id.insert(spot_obj_id, GizmoTargetKind::WineSpotlight);
    object_target_by_id.insert(decanter_obj_id, GizmoTargetKind::Decanter);
    object_target_by_id.insert(wine_obj_id, GizmoTargetKind::WineGlass);
    object_target_by_id.insert(cornell_obj_id, GizmoTargetKind::CornellBox);

    let mut decanter_master = main_db.create_collection("SceneMaster");
    let mut wine_master = Id(0);
    let mut cornell_master = Id(0);
    let mut decanter_scene_id = main_db.create_scene("Scene", decanter_master);
    let mut wine_scene_id = Id(0);
    let mut cornell_scene_id = Id(0);
    main_db.collection_link_object(decanter_master, sphere_obj_id);
    main_db.ensure_scene_base(decanter_scene_id, sphere_obj_id, true, true);
    main_db.collection_link_object(decanter_master, sun_obj_id);
    main_db.ensure_scene_base(decanter_scene_id, sun_obj_id, true, true);
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
        cornell_center: [0.0, 0.5, -1.0, 1.0],
        sun_intensity: 0.8,
        frame: 0,
        scene_kind: scene_kind.index(),
        render_width,
        render_height,
        selected_object: 1,
        mesh_enabled: 0,
        decanter_enabled: 0,
        wine_enabled: 0,
        cornell_enabled: 0,
        _pad: [0; 2],
    };

    let mut sun_azimuth_deg = uniforms.light_pos[2]
        .atan2(uniforms.light_pos[0])
        .to_degrees();
    let sun_len_xz = (uniforms.light_pos[0] * uniforms.light_pos[0]
        + uniforms.light_pos[2] * uniforms.light_pos[2])
        .sqrt();
    let mut sun_elevation_deg = uniforms.light_pos[1].atan2(sun_len_xz).to_degrees();
    let mut sun_intensity = uniforms.sun_intensity;
    let mut sun_lamp_distance = decanter_max_extent.max(8.0) * 2.2;
    let mut sun_empty_rotation = glam::Quat::IDENTITY;
    let mut sun_empty_scale = glam::Vec3::ONE;
    let mut sun_empty_position = active_center
        + glam::Vec3::new(
            sun_azimuth_deg.to_radians().cos() * sun_elevation_deg.to_radians().cos(),
            sun_elevation_deg.to_radians().sin(),
            sun_azimuth_deg.to_radians().sin() * sun_elevation_deg.to_radians().cos(),
        )
        .normalize_or_zero()
            * sun_lamp_distance;
    let mut wine_spotlight_azimuth_deg = -55.0;
    let mut wine_spotlight_elevation_deg = 54.0;
    let mut wine_spotlight_distance = wine_max_extent.max(10.0) * 1.4;
    let mut spot_empty_rotation = glam::Quat::IDENTITY;
    let mut spot_empty_scale = glam::Vec3::ONE;
    let mut spot_empty_position = wine_spotlight_position(
        wine_center,
        wine_spotlight_azimuth_deg,
        wine_spotlight_elevation_deg,
        wine_spotlight_distance,
    );

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

    let egui_ctx = egui::Context::default();
    let mut egui_visuals = egui::Visuals::dark();
    egui_visuals.panel_fill = Color32::from_rgb(42, 43, 46);
    egui_visuals.window_fill = Color32::from_rgb(49, 50, 54);
    egui_visuals.extreme_bg_color = Color32::from_rgb(28, 29, 31);
    egui_ctx.set_visuals(egui_visuals);
    let mut egui_state = EguiWinitState::new(
        egui_ctx.clone(),
        ViewportId::ROOT,
        window.as_ref(),
        Some(window.scale_factor() as f32),
        window.theme(),
        None,
    );
    let mut egui_renderer = EguiRenderer::new(&device, config.format, RendererOptions::default());

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
    let mut gizmo_target = default_target_for_scene(scene_kind);
    let mut has_selection = true;
    let mut sphere_rotation = glam::Quat::IDENTITY;
    let mut sphere_radius_scale = 1.0f32;
    let mut decanter_rotation = glam::Quat::IDENTITY;
    let mut decanter_translation = glam::Vec3::ZERO;
    let mut decanter_scale = glam::Vec3::ONE;
    let mut cornell_rotation = glam::Quat::IDENTITY;
    let mut cornell_translation = glam::Vec3::ZERO;
    let mut cornell_scale = glam::Vec3::ONE;
    let mut wine_rotation = glam::Quat::IDENTITY;
    let mut wine_translation = glam::Vec3::ZERO;
    let mut wine_scale = glam::Vec3::ONE;
    let mut geometry_dirty = false;
    let mut project_status = String::new();
    let mut mouse_pos = [0.0f32, 0.0f32];
    let mut mouse_left_down = false;
    let mut mouse_left_clicked = false;
    let mut mouse_left_dragging = false;

    let _ = event_loop.run(move |event, active_loop| {
        active_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        if let Event::WindowEvent { event, .. } = &event {
            let _ = egui_state.on_window_event(window.as_ref(), event);
            match event {
                WindowEvent::CursorMoved { position, .. } => {
                    mouse_pos = [position.x as f32, position.y as f32];
                    if mouse_left_down {
                        mouse_left_dragging = true;
                    }
                }
                WindowEvent::MouseInput {
                    state,
                    button: winit::event::MouseButton::Left,
                    ..
                } => {
                    mouse_left_down = *state == ElementState::Pressed;
                    if *state == ElementState::Pressed {
                        mouse_left_clicked = true;
                        mouse_left_dragging = false;
                    }
                }
                _ => {}
            }
        }
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
                if egui_ctx.is_pointer_over_area() || !keys_pressed.contains("v") {
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
                            let raw_input = egui_state.take_egui_input(window.as_ref());
                            let mut sun_changed = false;
                            let full_output = egui_ctx.run(raw_input, |ctx| {
                                let mut requested_scene = scene_kind;
                                let current_scene_exists = match scene_kind {
                                    SceneKind::Decanter => decanter_scene_id.0 != 0 && main_db.scenes.contains_key(&decanter_scene_id),
                                    SceneKind::Wine => wine_scene_id.0 != 0 && main_db.scenes.contains_key(&wine_scene_id),
                                    SceneKind::CornellBox => cornell_scene_id.0 != 0 && main_db.scenes.contains_key(&cornell_scene_id),
                                };
                                let has_decanter = decanter_scene_id.0 != 0 && main_db.scenes.contains_key(&decanter_scene_id);
                                let has_wine = wine_scene_id.0 != 0 && main_db.scenes.contains_key(&wine_scene_id);
                                let has_cornell = cornell_scene_id.0 != 0 && main_db.scenes.contains_key(&cornell_scene_id);

                                egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.strong("Prism");
                                        ui.separator();
                                        if ui.button("New Cube Scene").clicked() {
                                            requested_scene = SceneKind::Decanter;
                                        }
                                        ui.menu_button("Add", |ui| {
                                            let scene_id = match scene_kind {
                                                SceneKind::Decanter => decanter_scene_id,
                                                SceneKind::Wine => wine_scene_id,
                                                SceneKind::CornellBox => cornell_scene_id,
                                            };
                                            match scene_kind {
                                                SceneKind::Decanter => {
                                                    if ui.button("Cube").clicked() {
                                                        main_db.collection_link_object(decanter_master, sphere_obj_id);
                                                        main_db.ensure_scene_base(scene_id, sphere_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                    if ui.button("Sun Lamp").clicked() {
                                                        main_db.collection_link_object(decanter_master, sun_obj_id);
                                                        main_db.ensure_scene_base(scene_id, sun_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                    if ui.button("Decanter").clicked() {
                                                        main_db.collection_link_object(decanter_master, decanter_obj_id);
                                                        main_db.ensure_scene_base(scene_id, decanter_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                    if ui.button("Wine Glass").clicked() {
                                                        main_db.collection_link_object(decanter_master, wine_obj_id);
                                                        main_db.ensure_scene_base(scene_id, wine_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                    if ui.button("Cornell Box").clicked() {
                                                        main_db.collection_link_object(decanter_master, cornell_obj_id);
                                                        main_db.ensure_scene_base(scene_id, cornell_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                }
                                                SceneKind::Wine => {
                                                    if ui.button("Wine Glass").clicked() {
                                                        main_db.collection_link_object(wine_master, wine_obj_id);
                                                        main_db.ensure_scene_base(scene_id, wine_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                    if ui.button("Spotlight").clicked() {
                                                        main_db.collection_link_object(wine_master, spot_obj_id);
                                                        main_db.ensure_scene_base(scene_id, spot_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                }
                                                SceneKind::CornellBox => {
                                                    if ui.button("Cornell Box").clicked() {
                                                        main_db.collection_link_object(cornell_master, cornell_obj_id);
                                                        main_db.ensure_scene_base(scene_id, cornell_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                    if ui.button("Cube").clicked() {
                                                        main_db.collection_link_object(cornell_master, sphere_obj_id);
                                                        main_db.ensure_scene_base(scene_id, sphere_obj_id, true, true);
                                                        ui.close();
                                                    }
                                                }
                                            }
                                        });
                                        if ui.button("Open").clicked() {
                                            match load_prism_database(std::path::Path::new("res/scenes.prism"), false) {
                                                Ok(loaded) => {
                                                    main_db.collections.clear();
                                                    main_db.scenes.clear();
                                                    main_db.view_layers.clear();
                                                    decanter_master = Id(0);
                                                    wine_master = Id(0);
                                                    cornell_master = Id(0);
                                                    decanter_scene_id = Id(0);
                                                    wine_scene_id = Id(0);
                                                    cornell_scene_id = Id(0);
                                                    for (_sh, scene) in loaded.scenes.iter() {
                                                        let scene_name = scene.name.to_ascii_lowercase();
                                                        let local_master = main_db.create_collection(format!("{}Master", scene.name));
                                                        let local_scene = main_db.create_scene(&scene.name, local_master);
                                                        if scene_name.contains("decanter") || scene_name == "scene" {
                                                            decanter_master = local_master;
                                                            decanter_scene_id = local_scene;
                                                        } else if scene_name.contains("wine") {
                                                            wine_master = local_master;
                                                            wine_scene_id = local_scene;
                                                        } else if scene_name.contains("cornell") {
                                                            cornell_master = local_master;
                                                            cornell_scene_id = local_scene;
                                                        }
                                                        if let Some(master_col) = loaded.collections.get(scene.master_collection) {
                                                            for obj_handle in &master_col.objects {
                                                                if let Some(obj) = loaded.objects.get(*obj_handle) {
                                                                    let name = obj.name.to_ascii_lowercase();
                                                                    let oid = if name.contains("decanter") {
                                                                        Some(decanter_obj_id)
                                                                    } else if name.contains("wine") {
                                                                        Some(wine_obj_id)
                                                                    } else if name.contains("spot") {
                                                                        Some(spot_obj_id)
                                                                    } else if name.contains("sun") {
                                                                        Some(sun_obj_id)
                                                                    } else if name.contains("cornell") {
                                                                        Some(cornell_obj_id)
                                                                    } else if name.contains("sphere") || name.contains("cube") {
                                                                        Some(sphere_obj_id)
                                                                    } else {
                                                                        None
                                                                    };
                                                                    if let Some(local_obj_id) = oid {
                                                                        main_db.collection_link_object(local_master, local_obj_id);
                                                                        main_db.ensure_scene_base(local_scene, local_obj_id, true, true);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    for (_oh, obj) in loaded.objects.iter() {
                                                        let m = glam::Mat4::from_cols_array(&obj.transform_matrix);
                                                        let (s, r, t) = m.to_scale_rotation_translation();
                                                        let lname = obj.name.to_ascii_lowercase();
                                                        if lname.contains("sphere") || lname.contains("cube") {
                                                            uniforms.sphere_pos[0] = t.x;
                                                            uniforms.sphere_pos[1] = t.y;
                                                            uniforms.sphere_pos[2] = t.z;
                                                            sphere_rotation = r;
                                                            sphere_radius_scale = s.x.max(0.01);
                                                            uniforms.sphere_pos[3] = sphere_radius * sphere_radius_scale;
                                                        } else if lname.contains("decanter") {
                                                            decanter_translation = t - decanter_center;
                                                            decanter_rotation = r;
                                                            decanter_scale = s;
                                                            geometry_dirty = true;
                                                        } else if lname.contains("wine") {
                                                            wine_translation = t - wine_center;
                                                            wine_rotation = r;
                                                            wine_scale = s;
                                                            geometry_dirty = true;
                                                        } else if lname.contains("sun") {
                                                            sun_empty_position = t;
                                                            sun_empty_rotation = r;
                                                            sun_empty_scale = s;
                                                        } else if lname.contains("spot") {
                                                            spot_empty_position = t;
                                                            spot_empty_rotation = r;
                                                            spot_empty_scale = s;
                                                        }
                                                    }
                                                    if decanter_scene_id.0 != 0 {
                                                        requested_scene = SceneKind::Decanter;
                                                    } else if wine_scene_id.0 != 0 {
                                                        requested_scene = SceneKind::Wine;
                                                    } else if cornell_scene_id.0 != 0 {
                                                        requested_scene = SceneKind::CornellBox;
                                                    }
                                                    accumulation_dirty = true;
                                                    project_status = "Opened: res/scenes.prism".to_string();
                                                }
                                                Err(e) => project_status = format!("Open failed (res/scenes.prism): {e}"),
                                            }
                                        }
                                        if ui.button("Save").clicked() {
                                            let prism_db = build_prism_database_from_main(
                                                &main_db,
                                                decanter_scene_id,
                                                wine_scene_id,
                                                cornell_scene_id,
                                            );
                                            match save_prism_file(std::path::Path::new("res/scenes.prism"), &prism_db, false) {
                                                Ok(_) => project_status = "Saved: res/scenes.prism".to_string(),
                                                Err(e) => project_status = format!("Save failed: {e}"),
                                            }
                                        }
                                    });
                                });

                                egui::SidePanel::left("outliner")
                                    .resizable(true)
                                    .default_width(230.0)
                                    .show(ctx, |ui| {
                                        ui.heading("Outliner");
                                        ui.horizontal(|ui| {
                                            if has_decanter && ui.selectable_label(scene_kind == SceneKind::Decanter, "Scene").clicked() {
                                                requested_scene = SceneKind::Decanter;
                                            }
                                            if has_wine && ui.selectable_label(scene_kind == SceneKind::Wine, "Wine").clicked() {
                                                requested_scene = SceneKind::Wine;
                                            }
                                            if has_cornell && ui.selectable_label(scene_kind == SceneKind::CornellBox, "Cornell").clicked() {
                                                requested_scene = SceneKind::CornellBox;
                                            }
                                        });
                                        ui.separator();
                                        let scene_id = match scene_kind {
                                            SceneKind::Decanter => decanter_scene_id,
                                            SceneKind::Wine => wine_scene_id,
                                            SceneKind::CornellBox => cornell_scene_id,
                                        };
                                        for object_id in main_db.scene_visible_selectable_objects(scene_id) {
                                            if let Some(target) = object_target_by_id.get(&object_id).copied() {
                                                if !target_allowed_in_scene(scene_kind, target) {
                                                    continue;
                                                }
                                                let label = main_db.objects.get(&object_id).map(|o| o.name.as_str()).unwrap_or("Object");
                                                if ui.selectable_label(has_selection && gizmo_target == target, label).clicked() {
                                                    gizmo_target = target;
                                                    has_selection = true;
                                                }
                                            }
                                        }
                                        if !project_status.is_empty() {
                                            ui.separator();
                                            ui.label(&project_status);
                                        }
                                    });

                                egui::SidePanel::right("properties")
                                    .resizable(true)
                                    .default_width(300.0)
                                    .show(ctx, |ui| {
                                        ui.heading("Properties");
                                        if !target_allowed_in_scene(scene_kind, gizmo_target) {
                                            gizmo_target = default_target_for_scene(scene_kind);
                                        }
                                        ui.label(format!("Selected: {}", if has_selection { target_label(gizmo_target) } else { "None" }));
                                        ui.horizontal(|ui| {
                                            ui.selectable_value(&mut gizmo_mode, GizmoModeKind::Translate, "Move");
                                            ui.selectable_value(&mut gizmo_mode, GizmoModeKind::Rotate, "Rotate");
                                            ui.selectable_value(&mut gizmo_mode, GizmoModeKind::Scale, "Scale");
                                        });
                                        ui.separator();
                                        ui.collapsing("Sun", |ui| {
                                            ui.add(egui::Slider::new(&mut sun_azimuth_deg, -180.0..=180.0).text("Azimuth"));
                                            ui.add(egui::Slider::new(&mut sun_elevation_deg, -10.0..=89.0).text("Elevation"));
                                            ui.add(egui::Slider::new(&mut sun_intensity, 0.0..=5.0).text("Intensity"));
                                        });
                                        ui.collapsing("Spotlight", |ui| {
                                            let az_changed = ui.add(egui::Slider::new(&mut wine_spotlight_azimuth_deg, -180.0..=180.0).text("Azimuth")).changed();
                                            let el_changed = ui.add(egui::Slider::new(&mut wine_spotlight_elevation_deg, 5.0..=85.0).text("Elevation")).changed();
                                            let dist_changed = ui.add(egui::Slider::new(&mut wine_spotlight_distance, 2.0..=wine_max_extent.max(10.0) * 4.0).text("Distance")).changed();
                                            if scene_kind == SceneKind::Wine && (az_changed || el_changed || dist_changed) {
                                                spot_empty_position = wine_spotlight_position(
                                                    active_center,
                                                    wine_spotlight_azimuth_deg,
                                                    wine_spotlight_elevation_deg,
                                                    wine_spotlight_distance,
                                                );
                                            }
                                        });
                                    });

                            let requested_scene_exists = match requested_scene {
                                SceneKind::Decanter => decanter_scene_id.0 != 0 && main_db.scenes.contains_key(&decanter_scene_id),
                                SceneKind::Wine => wine_scene_id.0 != 0 && main_db.scenes.contains_key(&wine_scene_id),
                                SceneKind::CornellBox => {
                                    cornell_scene_id.0 != 0 && main_db.scenes.contains_key(&cornell_scene_id)
                                }
                            };
                            if requested_scene != scene_kind && requested_scene_exists {
                                scene_kind = requested_scene;
                                gizmo_target = default_target_for_scene(scene_kind);
                                has_selection = true;
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
                                if scene_kind != SceneKind::Wine {
                                    let sun_dir_reset = glam::Vec3::new(
                                        sun_azimuth_deg.to_radians().cos()
                                            * sun_elevation_deg.to_radians().cos(),
                                        sun_elevation_deg.to_radians().sin(),
                                        sun_azimuth_deg.to_radians().sin()
                                            * sun_elevation_deg.to_radians().cos(),
                                    )
                                    .normalize_or_zero();
                                    sun_empty_position =
                                        active_center + sun_dir_reset * sun_lamp_distance.max(1.0);
                                } else {
                                    spot_empty_position = wine_spotlight_position(
                                        active_center,
                                        wine_spotlight_azimuth_deg,
                                        wine_spotlight_elevation_deg,
                                        wine_spotlight_distance,
                                    );
                                }
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
                            let sun_lamp_pos = if scene_kind == SceneKind::Wine {
                                active_center + sun_dir * sun_lamp_distance.max(1.0)
                            } else {
                                sun_empty_position
                            };
                            let old_light = uniforms.light_pos;
                            let old_intensity = uniforms.sun_intensity;
                            uniforms.light_pos = if scene_kind == SceneKind::Wine {
                                let to_spot = spot_empty_position - active_center;
                                let spot_len = to_spot.length().max(1.0);
                                let spot_dir = to_spot / spot_len;
                                wine_spotlight_distance = spot_len;
                                wine_spotlight_azimuth_deg = spot_dir.z.atan2(spot_dir.x).to_degrees();
                                let spot_len_xz = (spot_dir.x * spot_dir.x + spot_dir.z * spot_dir.z).sqrt().max(1e-5);
                                wine_spotlight_elevation_deg = spot_dir.y.atan2(spot_len_xz).to_degrees();
                                [spot_empty_position.x, spot_empty_position.y, spot_empty_position.z, -1.0]
                            } else {
                                let d = (sun_lamp_pos - active_center).normalize_or_zero();
                                sun_azimuth_deg = d.z.atan2(d.x).to_degrees();
                                let len_xz = (d.x * d.x + d.z * d.z).sqrt().max(1e-5);
                                sun_elevation_deg = d.y.atan2(len_xz).to_degrees();
                                [d.x, d.y, d.z, 1.0]
                            };

                            let view = camera.view_matrix();
                            let projection = glam::Mat4::perspective_rh(
                                std::f32::consts::FRAC_PI_3 * 1.2,
                                config.width as f32 / config.height as f32,
                                0.1,
                                1000.0,
                            );
                            let pixels_per_point = ctx.pixels_per_point().max(1.0);
                            let screen_rect = ctx.input(|i| i.screen_rect());
                            let display_size = [screen_rect.width().max(1.0), screen_rect.height().max(1.0)];
                            let pointer_pos = ctx
                                .input(|i| i.pointer.hover_pos())
                                .map(|p| [p.x, p.y])
                                .unwrap_or([mouse_pos[0] / pixels_per_point, mouse_pos[1] / pixels_per_point]);
                            let pointer_captured = ctx.is_pointer_over_area();
                            let viewport = Rect::from_min_max(
                                [0.0, 0.0].into(),
                                [display_size[0].max(1.0), display_size[1].max(1.0)].into(),
                            );
                            let interaction = GizmoInteraction {
                                cursor_pos: (pointer_pos[0], pointer_pos[1]),
                                hovered: !pointer_captured,
                                drag_started: mouse_left_clicked,
                                dragging: mouse_left_down,
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
                                pixels_per_point,
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
                                GizmoTargetKind::CornellBox => {
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            cornell_scale.x as f64,
                                            cornell_scale.y as f64,
                                            cornell_scale.z as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            cornell_rotation.x as f64,
                                            cornell_rotation.y as f64,
                                            cornell_rotation.z as f64,
                                            cornell_rotation.w as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            (active_center.x + cornell_translation.x) as f64,
                                            (active_center.y + cornell_translation.y) as f64,
                                            (active_center.z + cornell_translation.z) as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::SunLamp => {
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            sun_empty_scale.x as f64,
                                            sun_empty_scale.y as f64,
                                            sun_empty_scale.z as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            sun_empty_rotation.x as f64,
                                            sun_empty_rotation.y as f64,
                                            sun_empty_rotation.z as f64,
                                            sun_empty_rotation.w as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            sun_lamp_pos.x as f64,
                                            sun_lamp_pos.y as f64,
                                            sun_lamp_pos.z as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::WineSpotlight => {
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            spot_empty_scale.x as f64,
                                            spot_empty_scale.y as f64,
                                            spot_empty_scale.z as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            spot_empty_rotation.x as f64,
                                            spot_empty_rotation.y as f64,
                                            spot_empty_rotation.z as f64,
                                            spot_empty_rotation.w as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            spot_empty_position.x as f64,
                                            spot_empty_position.y as f64,
                                            spot_empty_position.z as f64,
                                        ),
                                    )
                                }
                            };

                            if has_selection {
                                if let Some((_result, transforms)) =
                                    gizmo.update(interaction, &[target_transform])
                                {
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
                                        let sx = new_t.scale.x.abs() as f32;
                                        let sy = new_t.scale.y.abs() as f32;
                                        let sz = new_t.scale.z.abs() as f32;
                                        sphere_radius_scale = sx.max(sy).max(sz).clamp(0.15, 8.0);
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
                                    GizmoTargetKind::CornellBox => {
                                        cornell_translation = translation - active_center;
                                        cornell_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        cornell_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                    }
                                    GizmoTargetKind::SunLamp => {
                                        sun_empty_position = translation;
                                        let to_sun = sun_empty_position - active_center;
                                        sun_lamp_distance = to_sun.length().max(1.0);
                                        sun_empty_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        sun_empty_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                    }
                                    GizmoTargetKind::WineSpotlight => {
                                        spot_empty_position = translation;
                                        spot_empty_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        spot_empty_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                    }
                                    }
                                    accumulation_dirty = true;
                                }
                            }

                            if mouse_left_clicked
                                && !pointer_captured
                                && !gizmo.is_focused()
                                && !mouse_left_dragging
                            {
                                let scene_id = match scene_kind {
                                    SceneKind::Decanter => decanter_scene_id,
                                    SceneKind::Wine => wine_scene_id,
                                    SceneKind::CornellBox => cornell_scene_id,
                                };
                                let selectable_ids = main_db.scene_visible_selectable_objects(scene_id);
                                let sphere_allowed = selectable_ids.contains(&sphere_obj_id);
                                let decanter_allowed = selectable_ids.contains(&decanter_obj_id);
                                let wine_allowed = selectable_ids.contains(&wine_obj_id);
                                let cornell_allowed = selectable_ids.contains(&cornell_obj_id);
                                let sun_allowed = selectable_ids.contains(&sun_obj_id);
                                let spot_allowed = selectable_ids.contains(&spot_obj_id);
                                let (ro, rd) = world_ray_from_cursor(
                                    pointer_pos,
                                    [display_size[0].max(1.0), display_size[1].max(1.0)],
                                    camera.view_matrix().inverse(),
                                    projection.inverse(),
                                );
                                let sphere_center = glam::Vec3::new(
                                    uniforms.sphere_pos[0],
                                    uniforms.sphere_pos[1],
                                    uniforms.sphere_pos[2],
                                );
                                let decanter_center_now = decanter_center + decanter_translation;
                                let wine_center_now = wine_center + wine_translation;
                                let sphere_hit = if scene_kind == SceneKind::Decanter && sphere_allowed {
                                    intersect_cube(ro, rd, sphere_center, uniforms.sphere_pos[3])
                                } else {
                                    None
                                };
                                let decanter_hit = if scene_kind == SceneKind::Decanter && decanter_allowed {
                                    intersect_sphere(
                                        ro,
                                        rd,
                                        decanter_center_now,
                                        (decanter_max_extent * decanter_scale.max_element() * 0.55)
                                            .max(0.25),
                                    )
                                } else {
                                    None
                                };
                                let wine_hit = if wine_allowed {
                                    intersect_sphere(
                                        ro,
                                        rd,
                                        wine_center_now,
                                        (wine_max_extent * 0.55).max(0.25),
                                    )
                                } else {
                                    None
                                };
                                let cornell_hit = if cornell_allowed {
                                    intersect_sphere(
                                        ro,
                                        rd,
                                        active_center + cornell_translation,
                                        (2.0 * cornell_scale.max_element()).max(0.25),
                                    )
                                } else {
                                    None
                                };
                                let sun_hit = if scene_kind == SceneKind::Decanter && sun_allowed {
                                    intersect_sphere(ro, rd, sun_empty_position, 1.2)
                                } else {
                                    None
                                };
                                let spot_hit = if scene_kind == SceneKind::Wine && spot_allowed {
                                    intersect_sphere(ro, rd, spot_empty_position, 1.2)
                                } else {
                                    None
                                };
                                let mut best: Option<GizmoTargetKind> = None;
                                let mut best_t = f32::INFINITY;
                                if let Some(t) = sphere_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = Some(GizmoTargetKind::Sphere);
                                    }
                                }
                                if let Some(t) = decanter_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = Some(GizmoTargetKind::Decanter);
                                    }
                                }
                                if let Some(t) = wine_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = Some(GizmoTargetKind::WineGlass);
                                    }
                                }
                                if let Some(t) = cornell_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = Some(GizmoTargetKind::CornellBox);
                                    }
                                }
                                if let Some(t) = sun_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = Some(GizmoTargetKind::SunLamp);
                                    }
                                }
                                if let Some(t) = spot_hit {
                                    if t < best_t {
                                        best = Some(GizmoTargetKind::WineSpotlight);
                                    }
                                }
                                if let Some(selected) = best {
                                    gizmo_target = selected;
                                    has_selection = true;
                                } else {
                                    has_selection = false;
                                }
                            }

                            if has_selection {
                                let draw_data = gizmo.draw();
                                let painter = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("gizmo_overlay"),
                                ));
                                for idx in (0..draw_data.indices.len()).step_by(3) {
                                    let i0 = draw_data.indices[idx] as usize;
                                    let i1 = draw_data.indices[idx + 1] as usize;
                                    let i2 = draw_data.indices[idx + 2] as usize;
                                    let p0 = draw_data.vertices[i0];
                                    let p1 = draw_data.vertices[i1];
                                    let p2 = draw_data.vertices[i2];
                                    let c = draw_data.colors[i0];
                                    let color = Color32::from_rgba_unmultiplied(
                                        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                                        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                                        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
                                        (c[3].clamp(0.0, 1.0) * 255.0) as u8,
                                    );
                                    painter.add(egui::Shape::convex_polygon(
                                        vec![
                                            Pos2::new(p0[0], p0[1]),
                                            Pos2::new(p1[0], p1[1]),
                                            Pos2::new(p2[0], p2[1]),
                                        ],
                                        color,
                                        Stroke::NONE,
                                    ));
                                }

                                if current_scene_exists && scene_kind != SceneKind::Wine {
                                    let display = [display_size[0].max(1.0), display_size[1].max(1.0)];
                                    let sun_screen = world_to_screen(sun_lamp_pos, view, projection, display);
                                    let center_screen = world_to_screen(active_center, view, projection, display);
                                    if let (Some(s), Some(cn)) = (sun_screen, center_screen) {
                                        let selected = gizmo_target == GizmoTargetKind::SunLamp;
                                        let line_color = if selected {
                                            Color32::from_rgb(255, 158, 38)
                                        } else {
                                            Color32::from_rgb(255, 242, 153)
                                        };
                                        painter.line_segment(
                                            [Pos2::new(s[0], s[1]), Pos2::new(cn[0], cn[1])],
                                            Stroke::new(2.0, line_color),
                                        );
                                        painter.circle_stroke(Pos2::new(s[0], s[1]), 8.0, Stroke::new(2.0, line_color));
                                        painter.circle_filled(Pos2::new(s[0], s[1]), 3.0, line_color);
                                    }
                                } else if current_scene_exists {
                                    let display = [display_size[0].max(1.0), display_size[1].max(1.0)];
                                    let spot_screen =
                                        world_to_screen(spot_empty_position, view, projection, display);
                                    let target_screen = world_to_screen(active_center, view, projection, display);
                                    if let (Some(s), Some(tg)) = (spot_screen, target_screen) {
                                        let selected = gizmo_target == GizmoTargetKind::WineSpotlight;
                                        let line_color = if selected {
                                            Color32::from_rgb(255, 158, 38)
                                        } else {
                                            Color32::from_rgb(255, 242, 153)
                                        };
                                        painter.line_segment(
                                            [Pos2::new(s[0], s[1]), Pos2::new(tg[0], tg[1])],
                                            Stroke::new(2.0, line_color),
                                        );
                                        painter.circle_stroke(Pos2::new(s[0], s[1]), 8.0, Stroke::new(2.0, line_color));
                                        painter.circle_filled(Pos2::new(s[0], s[1]), 3.0, line_color);
                                    }
                                }
                            }
                            uniforms.sun_intensity = sun_intensity.max(0.0);
                            uniforms.scene_kind = if current_scene_exists {
                                scene_kind.index()
                            } else {
                                99
                            };
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
                            uniforms.cornell_center = [
                                active_center.x + cornell_translation.x,
                                active_center.y + cornell_translation.y,
                                active_center.z + cornell_translation.z,
                                cornell_scale.max_element().max(0.1),
                            ];
                            uniforms.selected_object = if has_selection {
                                match gizmo_target {
                                    GizmoTargetKind::Sphere => 1,
                                    GizmoTargetKind::Decanter => 3,
                                    GizmoTargetKind::WineGlass => 2,
                                    GizmoTargetKind::CornellBox => 4,
                                    GizmoTargetKind::SunLamp => 0,
                                    GizmoTargetKind::WineSpotlight => 0,
                                }
                            } else {
                                0
                            };
                            let active_scene_id = match scene_kind {
                                SceneKind::Decanter => decanter_scene_id,
                                SceneKind::Wine => wine_scene_id,
                                SceneKind::CornellBox => cornell_scene_id,
                            };
                            let (decanter_visible, wine_visible, cornell_visible) = if current_scene_exists {
                                let visible = main_db.scene_visible_selectable_objects(active_scene_id);
                                (
                                    visible.contains(&decanter_obj_id),
                                    visible.contains(&wine_obj_id),
                                    visible.contains(&cornell_obj_id),
                                )
                            } else {
                                (false, false, false)
                            };
                            uniforms.decanter_enabled = if decanter_visible { 1 } else { 0 };
                            uniforms.wine_enabled = if wine_visible { 1 } else { 0 };
                            uniforms.cornell_enabled = if cornell_visible { 1 } else { 0 };
                            uniforms.mesh_enabled = if decanter_visible || wine_visible { 1 } else { 0 };

                            if let Some(obj) = main_db.objects.get_mut(&sphere_obj_id) {
                                obj.transform.location = glam::Vec3::new(
                                    uniforms.sphere_pos[0],
                                    uniforms.sphere_pos[1],
                                    uniforms.sphere_pos[2],
                                );
                                obj.transform.rotation = sphere_rotation;
                                obj.transform.scale = glam::Vec3::splat(sphere_radius_scale);
                            }
                            if let Some(obj) = main_db.objects.get_mut(&decanter_obj_id) {
                                obj.transform.location = decanter_center + decanter_translation;
                                obj.transform.rotation = decanter_rotation;
                                obj.transform.scale = decanter_scale;
                            }
                            if let Some(obj) = main_db.objects.get_mut(&wine_obj_id) {
                                obj.transform.location = wine_center + wine_translation;
                                obj.transform.rotation = wine_rotation;
                                obj.transform.scale = wine_scale;
                            }
                            if let Some(obj) = main_db.objects.get_mut(&sun_obj_id) {
                                obj.transform.location = sun_empty_position;
                                obj.transform.rotation = sun_empty_rotation;
                                obj.transform.scale = sun_empty_scale;
                            }
                            if let Some(obj) = main_db.objects.get_mut(&spot_obj_id) {
                                obj.transform.location = spot_empty_position;
                                obj.transform.rotation = spot_empty_rotation;
                                obj.transform.scale = spot_empty_scale;
                            }
                            if let Some(obj) = main_db.objects.get_mut(&cornell_obj_id) {
                                obj.transform.location = active_center + cornell_translation;
                                obj.transform.rotation = cornell_rotation;
                                obj.transform.scale = cornell_scale;
                            }

                            sun_changed = uniforms.light_pos != old_light
                                || (uniforms.sun_intensity - old_intensity).abs() > f32::EPSILON;
                            });
                            let egui::FullOutput {
                                platform_output,
                                textures_delta,
                                shapes,
                                pixels_per_point,
                                ..
                            } = full_output;
                            egui_state.handle_platform_output(window.as_ref(), platform_output);
                            let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);

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
                            let screen_descriptor = ScreenDescriptor {
                                size_in_pixels: [config.width, config.height],
                                pixels_per_point,
                            };
                            for (id, image_delta) in &textures_delta.set {
                                egui_renderer.update_texture(&device, &queue, *id, image_delta);
                            }
                            egui_renderer.update_buffers(
                                &device,
                                &queue,
                                &mut encoder,
                                &clipped_primitives,
                                &screen_descriptor,
                            );
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
                            {
                                let mut ui_rpass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("egui-pass"),
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
                                egui_renderer.render(
                                    &mut ui_rpass.forget_lifetime(),
                                    &clipped_primitives,
                                    &screen_descriptor,
                                );
                            }
                            queue.submit(Some(encoder.finish()));
                            for id in &textures_delta.free {
                                egui_renderer.free_texture(id);
                            }
                            mouse_left_clicked = false;
                            if !mouse_left_down {
                                mouse_left_dragging = false;
                            }
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
