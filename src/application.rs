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
    material_editor::{MaterialGraphEditor, RuntimeMaterialPreview},
    mesh::{load_gltf_mesh, MeshData, Vertex},
    photon_mapper::PhotonMapper,
    prism_file::{
        load_prism_database, save_prism_file, CollectionData as PrismCollectionData,
        MaterialData as PrismMaterialData, MeshData as PrismMeshData, NodeProperties, NodeType,
        ObjectData as PrismObjectData, ObjectDataLink as PrismObjectDataLink, PrismDatabase,
        SceneData as PrismSceneData, ShaderNode,
    },
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
    sphere_params: [f32; 4],
    sphere_rot: [f32; 4],
    sphere_extent: [f32; 4],
    lens_params: [f32; 4],
    mesh_center: [f32; 4],
    decanter_center: [f32; 4],
    cornell_center: [f32; 4],
    cornell_color: [f32; 4],
    cornell_params: [f32; 4],
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
    primitive_count: u32,
    camera_aperture: f32,
    photon_brightness: f32,
    ground_brightness: f32,
    _pad: [u32; 2],
}

const MAX_PRIMITIVES: usize = 64;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPrimitive {
    pos: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
    rot: [f32; 4],
    extent: [f32; 4],
    lens: [f32; 4],
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
enum RenderModeKind {
    Pathtraced,
    Raytraced,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum PrimitiveShape {
    Cube,
    Sphere,
    ParabolicMirror,
    SphericalLens,
    ImagePlane,
    HyperbolicMirror,
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
        SceneKind::Wine => matches!(
            target,
            GizmoTargetKind::WineGlass | GizmoTargetKind::WineSpotlight
        ),
        SceneKind::CornellBox => matches!(
            target,
            GizmoTargetKind::Sphere | GizmoTargetKind::CornellBox
        ),
    }
}

fn target_label(target: GizmoTargetKind) -> &'static str {
    match target {
        GizmoTargetKind::Sphere => "Primitive",
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

fn make_white_material() -> PrismMaterialData {
    PrismMaterialData {
        name: "White".to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            let n_out = g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            let n_in = g.add_node(ShaderNode {
                node_type: NodeType::FloatInput,
                properties: NodeProperties {
                    float_value: Some(1.0),
                    vec3_value: Some([1.0, 1.0, 1.0]),
                    roughness: None,
                    transmission: None,
                    ior: None,
                    bsdf_connected: None,
                },
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
    }
}

fn make_glass_material() -> PrismMaterialData {
    PrismMaterialData {
        name: "Glass".to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            let n_bsdf = g.add_node(ShaderNode {
                node_type: NodeType::PrincipledBSDF,
                properties: NodeProperties {
                    float_value: None,
                    vec3_value: Some([0.98, 1.0, 1.0]),
                    roughness: Some(0.02),
                    transmission: Some(1.0),
                    ior: Some(1.52),
                    bsdf_connected: Some(true),
                },
            });
            let n_out = g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            g.add_edge(
                n_bsdf,
                n_out,
                crate::prism_file::NodeLink {
                    output_socket: "BSDF".to_string(),
                    input_socket: "Surface".to_string(),
                },
            );
            g
        },
    }
}

fn make_mirror_material() -> PrismMaterialData {
    PrismMaterialData {
        name: "Mirror".to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            let n_bsdf = g.add_node(ShaderNode {
                node_type: NodeType::PrincipledBSDF,
                properties: NodeProperties {
                    float_value: None,
                    vec3_value: Some([0.93, 0.95, 0.98]),
                    roughness: Some(0.004),
                    transmission: Some(0.0),
                    ior: Some(1.0),
                    bsdf_connected: Some(true),
                },
            });
            let n_out = g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            g.add_edge(
                n_bsdf,
                n_out,
                crate::prism_file::NodeLink {
                    output_socket: "BSDF".to_string(),
                    input_socket: "Surface".to_string(),
                },
            );
            g
        },
    }
}

fn preview_from_material_data(material: Option<&PrismMaterialData>) -> RuntimeMaterialPreview {
    let mut out = RuntimeMaterialPreview::default();
    let Some(material) = material else {
        return out;
    };
    let mut bsdf_idx = None;
    let mut out_idx = None;
    for idx in material.graph.node_indices() {
        match material.graph[idx].node_type {
            NodeType::PrincipledBSDF => bsdf_idx = Some(idx),
            NodeType::MaterialOutput => out_idx = Some(idx),
            _ => {}
        }
    }
    if let Some(bi) = bsdf_idx {
        let props = &material.graph[bi].properties;
        if let Some(v) = props.vec3_value {
            out.base_color = v;
        }
        if let Some(v) = props.roughness {
            out.roughness = v;
        }
        if let Some(v) = props.transmission {
            out.transmission = v;
        }
        if let Some(v) = props.ior {
            out.ior = v;
        }
    }
    if let (Some(bi), Some(oi)) = (bsdf_idx, out_idx) {
        out.bsdf_connected = material.graph.edges_connecting(bi, oi).any(|edge| {
            edge.weight().output_socket == "BSDF" && edge.weight().input_socket == "Surface"
        });
        if let Some(v) = material.graph[bi].properties.bsdf_connected {
            out.bsdf_connected = v;
        }
    }
    out.roughness = out.roughness.clamp(0.001, 1.0);
    out.transmission = out.transmission.clamp(0.0, 1.0);
    out.ior = out.ior.max(1.0);
    out
}

fn build_prism_database_from_main(
    main_db: &MainDatabase,
    decanter_scene_id: Id,
    wine_scene_id: Id,
    cornell_scene_id: Id,
    object_material_names: &std::collections::HashMap<Id, String>,
    material_library: &std::collections::HashMap<String, PrismMaterialData>,
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

    let mut material_map: std::collections::HashMap<String, crate::prism_file::MaterialHandle> =
        std::collections::HashMap::new();
    for (name, material) in material_library {
        let h = out.materials.insert(material.clone());
        material_map.insert(name.clone(), h);
    }

    let mut object_map: std::collections::HashMap<Id, crate::prism_file::ObjectHandle> =
        std::collections::HashMap::new();
    for (oid, obj) in &main_db.objects {
        let mesh_link = obj.mesh_id.and_then(|m| mesh_map.get(&m).copied());
        let object_material = object_material_names
            .get(oid)
            .and_then(|name| material_map.get(name).copied());
        let h = out.objects.insert(PrismObjectData {
            name: obj.name.clone(),
            transform_matrix: transform_to_matrix(&obj.transform),
            data_link: mesh_link
                .map(PrismObjectDataLink::Mesh)
                .unwrap_or(PrismObjectDataLink::None),
            material_link: object_material,
        });
        object_map.insert(*oid, h);
        if let (Some(mesh_id), Some(mat_handle)) = (obj.mesh_id, object_material) {
            if let Some(mesh_h) = mesh_map.get(&mesh_id).copied() {
                if let Some(mesh) = out.meshes.get_mut(mesh_h) {
                    if mesh.material_slots.is_empty() {
                        mesh.material_slots.push(Some(mat_handle));
                    } else {
                        mesh.material_slots[0] = Some(mat_handle);
                    }
                }
            }
        }
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

fn primitive_spawn_transform(
    current_pos: glam::Vec3,
    instance_count: usize,
    scale: glam::Vec3,
    sphere_radius: f32,
) -> DbTransform {
    DbTransform {
        location: current_pos + glam::Vec3::X * (instance_count as f32 * sphere_radius * 0.35),
        rotation: glam::Quat::IDENTITY,
        scale,
    }
}

fn set_primitive_shape(
    main_db: &mut MainDatabase,
    sphere_obj_id: Id,
    primitive_shape: &mut PrimitiveShape,
    shape: PrimitiveShape,
    uniforms: &mut SceneUniforms,
) {
    *primitive_shape = shape;
    uniforms.sphere_params[3] = match shape {
        PrimitiveShape::Cube => 0.0,
        PrimitiveShape::Sphere => 1.0,
        PrimitiveShape::ParabolicMirror => 2.0,
        PrimitiveShape::SphericalLens => 3.0,
        PrimitiveShape::ImagePlane => 4.0,
        PrimitiveShape::HyperbolicMirror => 5.0,
    };
    if let Some(obj) = main_db.objects.get_mut(&sphere_obj_id) {
        obj.name = match shape {
            PrimitiveShape::Cube => "Cube",
            PrimitiveShape::Sphere => "Sphere",
            PrimitiveShape::ParabolicMirror => "Parabolic Mirror",
            PrimitiveShape::SphericalLens => "Spherical Lens",
            PrimitiveShape::ImagePlane => "Image",
            PrimitiveShape::HyperbolicMirror => "Hyperbolic Mirror",
        }
        .to_string();
    }
}

fn create_primitive_object(
    main_db: &mut MainDatabase,
    object_target_by_id: &mut std::collections::HashMap<Id, GizmoTargetKind>,
    primitive_shape_by_id: &mut std::collections::HashMap<Id, PrimitiveShape>,
    object_material_names: &mut std::collections::HashMap<Id, String>,
    shape: PrimitiveShape,
    material_name: &str,
    transform: DbTransform,
    sphere_radius: f32,
    primitive_shape: &mut PrimitiveShape,
    uniforms: &mut SceneUniforms,
) -> Id {
    let object_id = main_db.create_object("Primitive", None, transform.clone());
    object_target_by_id.insert(object_id, GizmoTargetKind::Sphere);
    primitive_shape_by_id.insert(object_id, shape);
    object_material_names.insert(object_id, material_name.to_string());
    set_primitive_shape(main_db, object_id, primitive_shape, shape, uniforms);
    uniforms.sphere_pos = [
        transform.location.x,
        transform.location.y,
        transform.location.z,
        sphere_radius,
    ];
    uniforms.sphere_rot = [
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ];
    uniforms.sphere_extent = [
        sphere_radius * transform.scale.x,
        sphere_radius * transform.scale.y,
        sphere_radius * transform.scale.z,
        0.0,
    ];
    object_id
}

fn include_photon_bounds(
    bounds_min: &mut glam::Vec3,
    bounds_max: &mut glam::Vec3,
    bounds_valid: &mut bool,
    center: glam::Vec3,
    radius: f32,
) {
    let radius = radius.max(0.05);
    let extent = glam::Vec3::splat(radius);
    if *bounds_valid {
        *bounds_min = bounds_min.min(center - extent);
        *bounds_max = bounds_max.max(center + extent);
    } else {
        *bounds_min = center - extent;
        *bounds_max = center + extent;
        *bounds_valid = true;
    }
}

fn scene_camera(
    scene_kind: SceneKind,
    center: glam::Vec3,
    size: glam::Vec3,
) -> (glam::Vec3, glam::Vec3) {
    if scene_kind == SceneKind::Wine {
        let distance = size.max_element().max(12.0) * 1.35;
        return (
            center + glam::Vec3::new(0.0, size.y * 0.2, distance),
            center,
        );
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

fn intersect_sphere(
    origin: glam::Vec3,
    dir: glam::Vec3,
    center: glam::Vec3,
    radius: f32,
) -> Option<f32> {
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

fn intersect_cube(
    origin: glam::Vec3,
    dir: glam::Vec3,
    center: glam::Vec3,
    half_extent: glam::Vec3,
) -> Option<f32> {
    let min = center - half_extent;
    let max = center + half_extent;
    let inv = glam::Vec3::new(
        if dir.x.abs() > 1e-6 {
            1.0 / dir.x
        } else {
            f32::INFINITY
        },
        if dir.y.abs() > 1e-6 {
            1.0 / dir.y
        } else {
            f32::INFINITY
        },
        if dir.z.abs() > 1e-6 {
            1.0 / dir.z
        } else {
            f32::INFINITY
        },
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
    let decanter_material_start = 0usize;
    let decanter_material_count = mesh.materials.len();
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
    let wine_material_start = decanter_material_start + decanter_material_count;
    let wine_material_count = mesh.materials.len().saturating_sub(wine_material_start);
    let wine_base_positions: Vec<glam::Vec3> = mesh.positions4
        [wine_vertex_start..wine_vertex_start + wine_vertex_count]
        .iter()
        .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
        .collect();
    let wine_base_normals: Vec<glam::Vec3> = mesh.normals4
        [wine_vertex_start..wine_vertex_start + wine_vertex_count]
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
    let decanter_obj_id =
        main_db.create_object("Decanter", Some(decanter_mesh_id), DbTransform::default());
    let wine_obj_id =
        main_db.create_object("WineGlass", Some(wine_mesh_id), DbTransform::default());
    let cornell_obj_id =
        main_db.create_object("CornellBox", Some(cornell_mesh_id), DbTransform::default());
    let mut material_library: std::collections::HashMap<String, PrismMaterialData> =
        std::collections::HashMap::new();
    material_library.insert("White".to_string(), make_white_material());
    material_library.insert("Glass".to_string(), make_glass_material());
    material_library.insert("Mirror".to_string(), make_mirror_material());
    let mut object_material_names: std::collections::HashMap<Id, String> =
        std::collections::HashMap::new();
    object_material_names.insert(sphere_obj_id, "Glass".to_string());
    object_material_names.insert(decanter_obj_id, "Glass".to_string());
    object_material_names.insert(wine_obj_id, "Glass".to_string());
    object_material_names.insert(cornell_obj_id, "White".to_string());
    let mut last_material_signature = String::new();

    let mut object_target_by_id: std::collections::HashMap<Id, GizmoTargetKind> =
        std::collections::HashMap::new();
    let mut primitive_shape_by_id: std::collections::HashMap<Id, PrimitiveShape> =
        std::collections::HashMap::new();
    let mut primitive_lens_params_by_id: std::collections::HashMap<Id, [f32; 4]> =
        std::collections::HashMap::new();
    object_target_by_id.insert(sphere_obj_id, GizmoTargetKind::Sphere);
    primitive_shape_by_id.insert(sphere_obj_id, PrimitiveShape::Cube);
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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

    let mut camera_fov_deg = 72.0_f32;
    let projection = glam::Mat4::perspective_rh(
        camera_fov_deg.to_radians(),
        config.width as f32 / config.height as f32,
        0.1,
        10_000.0,
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
        sphere_params: [0.02, 1.52, 1.0, 0.0],
        sphere_rot: [0.0, 0.0, 0.0, 1.0],
        sphere_extent: [sphere_radius, sphere_radius, sphere_radius, 0.0],
        lens_params: [
            sphere_radius * 1.8,
            sphere_radius * 1.8,
            sphere_radius * 0.5,
            0.0,
        ],
        mesh_center: [
            wine_center.x,
            wine_center.y,
            wine_center.z,
            wine_max_extent * 0.8,
        ],
        decanter_center: [
            decanter_center.x,
            decanter_center.y,
            decanter_center.z,
            decanter_max_extent * 0.7,
        ],
        cornell_center: [0.0, 0.5, -1.0, 1.0],
        cornell_color: [1.0, 1.0, 1.0, 0.0],
        cornell_params: [0.7, 1.0, 0.0, 0.0],
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
        primitive_count: 1,
        camera_aperture: 0.0,
        photon_brightness: 0.1,
        ground_brightness: 1.0,
        _pad: [0; 2],
    };
    primitive_lens_params_by_id.insert(sphere_obj_id, uniforms.lens_params);

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

    let puppy_image = image::open("res/puppy.jpg")
        .expect("failed to load res/puppy.jpg")
        .to_rgba8();
    let puppy_dimensions = puppy_image.dimensions();
    let puppy_texture_size = wgpu::Extent3d {
        width: puppy_dimensions.0.max(1),
        height: puppy_dimensions.1.max(1),
        depth_or_array_layers: 1,
    };
    let puppy_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("puppy_image_texture"),
        size: puppy_texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &puppy_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &puppy_image,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * puppy_dimensions.0),
            rows_per_image: Some(puppy_dimensions.1),
        },
        puppy_texture_size,
    );
    let puppy_texture_view = puppy_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let environment_image = image::open("res/sunflowers_puresky_4k.exr")
        .expect("failed to load sunflower environment")
        .to_rgba32f();
    let environment_dimensions = environment_image.dimensions();
    let environment_texture_size = wgpu::Extent3d {
        width: environment_dimensions.0.max(1),
        height: environment_dimensions.1.max(1),
        depth_or_array_layers: 1,
    };
    let environment_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sunflower_environment_texture"),
        size: environment_texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &environment_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(environment_image.as_raw()),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16 * environment_dimensions.0),
            rows_per_image: Some(environment_dimensions.1),
        },
        environment_texture_size,
    );
    let environment_texture_view =
        environment_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let primitive_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("primitive_instances_buf"),
        size: (std::mem::size_of::<GpuPrimitive>() * MAX_PRIMITIVES) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
            wgpu::BindGroupLayoutEntry {
                binding: 13,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 14,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 15,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
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
        &primitive_buffer,
        &puppy_texture_view,
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
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::TextureView(&puppy_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: primitive_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: wgpu::BindingResource::TextureView(&environment_texture_view),
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
    let mut render_mode = RenderModeKind::Pathtraced;
    let mut gizmo = Gizmo::default();
    let mut gizmo_mode = GizmoModeKind::Translate;
    let mut gizmo_target = default_target_for_scene(scene_kind);
    let mut has_selection = true;
    let mut primitive_shape = PrimitiveShape::Cube;
    let mut selected_primitive_id = sphere_obj_id;
    let mut sphere_rotation = glam::Quat::IDENTITY;
    let mut sphere_scale = glam::Vec3::ONE;
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
    let mut material_editor = MaterialGraphEditor::new();
    let mut material_runtime_overrides: std::collections::HashMap<String, RuntimeMaterialPreview> =
        std::collections::HashMap::new();
    let mut optical_trace_enabled = false;
    let mut optical_trace_rays = 9u32;
    let mut optical_trace_image_area = true;
    let mut cassegrain_focus_offset = 0.0_f32;

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
                    camera_fov_deg.to_radians(),
                    config.width as f32 / config.height as f32,
                    0.1,
                    10_000.0,
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
                        12.0
                    } else {
                        1.0
                    };
                    let wants_keyboard = egui_ctx.wants_keyboard_input();

                    if !wants_keyboard && keys_pressed.contains("r") {
                        gizmo_mode = GizmoModeKind::Rotate;
                        keys_pressed.remove("r");
                    }
                    if !wants_keyboard && keys_pressed.contains("g") {
                        gizmo_mode = GizmoModeKind::Translate;
                        keys_pressed.remove("g");
                    }

                    if !wants_keyboard && has_selection && keys_pressed.contains("x") {
                        let scene_id = match scene_kind {
                            SceneKind::Decanter => decanter_scene_id,
                            SceneKind::Wine => wine_scene_id,
                            SceneKind::CornellBox => cornell_scene_id,
                        };
                        let object_id = match gizmo_target {
                            GizmoTargetKind::Sphere => Some(selected_primitive_id),
                            GizmoTargetKind::Decanter => Some(decanter_obj_id),
                            GizmoTargetKind::WineGlass => Some(wine_obj_id),
                            GizmoTargetKind::CornellBox => Some(cornell_obj_id),
                            GizmoTargetKind::SunLamp => Some(sun_obj_id),
                            GizmoTargetKind::WineSpotlight => Some(spot_obj_id),
                        };
                        if let Some(object_id) = object_id {
                            if primitive_shape_by_id.contains_key(&object_id) {
                                main_db.delete_object(object_id);
                                object_target_by_id.remove(&object_id);
                                primitive_shape_by_id.remove(&object_id);
                                primitive_lens_params_by_id.remove(&object_id);
                                object_material_names.remove(&object_id);
                            } else {
                                main_db.unlink_object_from_scene(scene_id, object_id);
                            }
                            has_selection = false;
                            gizmo_target = default_target_for_scene(scene_kind);
                            accumulation_dirty = true;
                        }
                        keys_pressed.remove("x");
                    }

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
                            let mut photon_emitter_center = [
                                active_center.x,
                                active_center.y,
                                active_center.z,
                                active_max_extent * 0.85,
                            ];
                            let mut photons_per_frame = 0u32;
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
                                        if ui.button("Telescope Demo").clicked() {
                                            let (scene_id, master_id) = match scene_kind {
                                                SceneKind::Decanter => (decanter_scene_id, decanter_master),
                                                SceneKind::Wine => (wine_scene_id, wine_master),
                                                SceneKind::CornellBox => (cornell_scene_id, cornell_master),
                                            };
                                            if scene_id.0 != 0 && master_id.0 != 0 {
                                                let scene_objects = main_db.scene_objects_recursive(scene_id);
                                                for object_id in scene_objects {
                                                    if primitive_shape_by_id.contains_key(&object_id) {
                                                        main_db.delete_object(object_id);
                                                        object_target_by_id.remove(&object_id);
                                                        primitive_shape_by_id.remove(&object_id);
                                                        primitive_lens_params_by_id.remove(&object_id);
                                                        object_material_names.remove(&object_id);
                                                    } else {
                                                        main_db.unlink_object_from_scene(scene_id, object_id);
                                                    }
                                                }

                                                let bench_y = -1.5 + sphere_radius * 0.55;
                                                let bench_rot = glam::Quat::from_rotation_y(
                                                    std::f32::consts::FRAC_PI_2,
                                                );
                                                let image_aspect = puppy_dimensions.0 as f32
                                                    / puppy_dimensions.1.max(1) as f32;
                                                let image_scale = glam::Vec3::new(
                                                    image_aspect.max(0.1) * 0.72,
                                                    0.72,
                                                    1.0,
                                                );
                                                let objective_params = [18.0, 18.0, 0.9, 0.0];
                                                let eyepiece_params = [9.0, 9.0, 0.55, 0.0];

                                                let image_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::ImagePlane,
                                                    "White",
                                                    DbTransform {
                                                        location: glam::Vec3::new(-48.0, bench_y, 0.0),
                                                        rotation: bench_rot,
                                                        scale: image_scale,
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&image_id) {
                                                    obj.name = "Puppy Source".to_string();
                                                }
                                                main_db.collection_link_object(master_id, image_id);
                                                main_db.ensure_scene_base(scene_id, image_id, true, true);

                                                let objective_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::SphericalLens,
                                                    "Glass",
                                                    DbTransform {
                                                        location: glam::Vec3::new(0.0, bench_y, 0.0),
                                                        rotation: bench_rot,
                                                        scale: glam::Vec3::splat(0.55),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&objective_id) {
                                                    obj.name = "Objective Lens".to_string();
                                                }
                                                primitive_lens_params_by_id
                                                    .insert(objective_id, objective_params);
                                                main_db.collection_link_object(master_id, objective_id);
                                                main_db.ensure_scene_base(scene_id, objective_id, true, true);

                                                let eyepiece_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::SphericalLens,
                                                    "Glass",
                                                    DbTransform {
                                                        location: glam::Vec3::new(35.0, bench_y, 0.0),
                                                        rotation: bench_rot,
                                                        scale: glam::Vec3::splat(0.34),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&eyepiece_id) {
                                                    obj.name = "Eyepiece Lens".to_string();
                                                }
                                                primitive_lens_params_by_id
                                                    .insert(eyepiece_id, eyepiece_params);
                                                main_db.collection_link_object(master_id, eyepiece_id);
                                                main_db.ensure_scene_base(scene_id, eyepiece_id, true, true);

                                                let screen_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::Cube,
                                                    "White",
                                                    DbTransform {
                                                        location: glam::Vec3::new(54.0, bench_y, 0.0),
                                                        rotation: glam::Quat::IDENTITY,
                                                        scale: glam::Vec3::new(0.025, 0.45, 0.6),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&screen_id) {
                                                    obj.name = "Exit Screen".to_string();
                                                }
                                                main_db.collection_link_object(master_id, screen_id);
                                                main_db.ensure_scene_base(scene_id, screen_id, true, true);

                                                selected_primitive_id = objective_id;
                                                primitive_shape = PrimitiveShape::SphericalLens;
                                                uniforms.sphere_params[3] = 3.0;
                                                uniforms.lens_params = objective_params;
                                                uniforms.sphere_pos = [0.0, bench_y, 0.0, sphere_radius];
                                                uniforms.sphere_rot = [
                                                    bench_rot.x,
                                                    bench_rot.y,
                                                    bench_rot.z,
                                                    bench_rot.w,
                                                ];
                                                uniforms.sphere_extent = [
                                                    sphere_radius * 0.55,
                                                    sphere_radius * 0.55,
                                                    sphere_radius * 0.55,
                                                    0.0,
                                                ];
                                                sphere_rotation = bench_rot;
                                                sphere_scale = glam::Vec3::splat(0.55);
                                                gizmo_target = GizmoTargetKind::Sphere;
                                                has_selection = true;
                                                optical_trace_enabled = true;
                                                optical_trace_rays = 9;
                                                sun_azimuth_deg = 180.0;
                                                sun_elevation_deg = 70.0;
                                                sun_intensity = 1.5;
                                                sun_lamp_distance = sun_lamp_distance.max(80.0);
                                                let sun_elevation = sun_elevation_deg.to_radians();
                                                sun_empty_position = active_center
                                                    + glam::Vec3::new(
                                                        -sun_elevation.cos(),
                                                        sun_elevation.sin(),
                                                        0.0,
                                                    ) * sun_lamp_distance;
                                                sun_empty_rotation = glam::Quat::IDENTITY;
                                                camera = Camera::look_at(
                                                    glam::Vec3::new(2.0, bench_y + 8.0, 58.0),
                                                    glam::Vec3::new(3.0, bench_y, 0.0),
                                                );
                                                accumulation_dirty = true;
                                                project_status = "Telescope demo created".to_string();
                                            }
                                        }
                                        if ui.button("Newtonian Demo").clicked() {
                                            let (scene_id, master_id) = match scene_kind {
                                                SceneKind::Decanter => (decanter_scene_id, decanter_master),
                                                SceneKind::Wine => (wine_scene_id, wine_master),
                                                SceneKind::CornellBox => (cornell_scene_id, cornell_master),
                                            };
                                            if scene_id.0 != 0 && master_id.0 != 0 {
                                                let scene_objects = main_db.scene_objects_recursive(scene_id);
                                                for object_id in scene_objects {
                                                    if primitive_shape_by_id.contains_key(&object_id) {
                                                        main_db.delete_object(object_id);
                                                        object_target_by_id.remove(&object_id);
                                                        primitive_shape_by_id.remove(&object_id);
                                                        primitive_lens_params_by_id.remove(&object_id);
                                                        object_material_names.remove(&object_id);
                                                    } else {
                                                        main_db.unlink_object_from_scene(scene_id, object_id);
                                                    }
                                                }

                                                let bench_y = -1.5 + sphere_radius * 0.75;
                                                let tube_axis_rot =
                                                    glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
                                                let tube_axis = glam::Vec3::X;
                                                let desired_fold_axis = glam::Vec3::Z;
                                                let primary_center = glam::Vec3::new(0.0, bench_y, 0.0);
                                                let primary_radius = sphere_radius * 0.28;
                                                let primary_focal_length = primary_radius * 10.0;
                                                let primary_depth =
                                                    primary_radius * primary_radius / (8.0 * primary_focal_length);
                                                let primary_vertex =
                                                    primary_center - tube_axis * primary_depth;
                                                let puppy_distance = primary_focal_length * 7.0;
                                                let puppy_center =
                                                    primary_center + tube_axis * puppy_distance;
                                                let object_distance =
                                                    (puppy_center - primary_vertex).dot(tube_axis).max(
                                                        primary_focal_length + 0.01,
                                                    );
                                                let primary_image_distance = 1.0
                                                    / (1.0 / primary_focal_length
                                                        - 1.0 / object_distance)
                                                        .max(0.001);
                                                let primary_image_point =
                                                    primary_vertex + tube_axis * primary_image_distance;
                                                let folded_focus_distance = primary_radius * 1.15;
                                                let secondary_center =
                                                    primary_image_point - tube_axis * folded_focus_distance;
                                                let focuser_params = [6.0, 6.0, 0.25, 0.0];
                                                let focuser_ior = 1.52_f32;
                                                let focuser_power = (focuser_ior - 1.0)
                                                    * (1.0 / focuser_params[0]
                                                        + 1.0 / focuser_params[1]
                                                        - ((focuser_ior - 1.0) * focuser_params[2])
                                                            / (focuser_ior
                                                                * focuser_params[0]
                                                                * focuser_params[1]));
                                                let focuser_after_focus_distance =
                                                    (1.0 / focuser_power).clamp(1.0, 12.0);
                                                let central_incoming =
                                                    (primary_image_point - secondary_center).normalize();
                                                let secondary_normal =
                                                    (central_incoming - desired_fold_axis).normalize();
                                                let folded_axis = (central_incoming
                                                    - 2.0 * central_incoming.dot(secondary_normal)
                                                        * secondary_normal)
                                                    .normalize();
                                                let folded_focus =
                                                    secondary_center + folded_axis * folded_focus_distance;
                                                let focuser_center =
                                                    folded_focus + folded_axis * focuser_after_focus_distance;
                                                let cone_radius_at_secondary =
                                                    primary_radius * folded_focus_distance / primary_image_distance;
                                                let secondary_half_size =
                                                    (cone_radius_at_secondary * 1.28).clamp(0.18, 0.55);
                                                let secondary_rot = glam::Quat::from_rotation_arc(
                                                    glam::Vec3::Z,
                                                    secondary_normal,
                                                );
                                                let focuser_rot = glam::Quat::from_rotation_arc(
                                                    glam::Vec3::Z,
                                                    folded_axis,
                                                );
                                                let puppy_target_rot =
                                                    glam::Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
                                                let puppy_aspect = puppy_dimensions.0 as f32
                                                    / puppy_dimensions.1.max(1) as f32;

                                                let puppy_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::ImagePlane,
                                                    "White",
                                                    DbTransform {
                                                        location: puppy_center,
                                                        rotation: puppy_target_rot,
                                                        scale: glam::Vec3::new(
                                                            puppy_aspect.max(0.1) * 0.95,
                                                            0.95,
                                                            1.0,
                                                        ),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&puppy_id) {
                                                    obj.name = "Distant Puppy Target".to_string();
                                                }
                                                main_db.collection_link_object(master_id, puppy_id);
                                                main_db.ensure_scene_base(scene_id, puppy_id, true, true);

                                                let primary_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::ParabolicMirror,
                                                    "Mirror",
                                                    DbTransform {
                                                        location: primary_center,
                                                        rotation: tube_axis_rot,
                                                        scale: glam::Vec3::new(
                                                            primary_radius / sphere_radius,
                                                            primary_radius / sphere_radius,
                                                            primary_depth / sphere_radius,
                                                        ),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&primary_id) {
                                                    obj.name = "Primary Parabolic Mirror".to_string();
                                                }
                                                main_db.collection_link_object(master_id, primary_id);
                                                main_db.ensure_scene_base(scene_id, primary_id, true, true);

                                                let secondary_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::Cube,
                                                    "Mirror",
                                                    DbTransform {
                                                        location: secondary_center,
                                                        rotation: secondary_rot,
                                                        scale: glam::Vec3::new(
                                                            secondary_half_size / sphere_radius,
                                                            secondary_half_size / sphere_radius,
                                                            0.008,
                                                        ),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&secondary_id) {
                                                    obj.name = "45 Degree Secondary Mirror".to_string();
                                                }
                                                main_db.collection_link_object(master_id, secondary_id);
                                                main_db.ensure_scene_base(scene_id, secondary_id, true, true);

                                                let focuser_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::SphericalLens,
                                                    "Glass",
                                                    DbTransform {
                                                        location: focuser_center,
                                                        rotation: focuser_rot,
                                                        scale: glam::Vec3::splat(0.22),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&focuser_id) {
                                                    obj.name = "Focuser Lens".to_string();
                                                }
                                                primitive_lens_params_by_id
                                                    .insert(focuser_id, focuser_params);
                                                main_db.collection_link_object(master_id, focuser_id);
                                                main_db.ensure_scene_base(scene_id, focuser_id, true, true);

                                                selected_primitive_id = primary_id;
                                                primitive_shape = PrimitiveShape::ParabolicMirror;
                                                uniforms.sphere_params[3] = 2.0;
                                                uniforms.sphere_pos = [
                                                    primary_center.x,
                                                    primary_center.y,
                                                    primary_center.z,
                                                    sphere_radius,
                                                ];
                                                uniforms.sphere_rot = [
                                                    tube_axis_rot.x,
                                                    tube_axis_rot.y,
                                                    tube_axis_rot.z,
                                                    tube_axis_rot.w,
                                                ];
                                                uniforms.sphere_extent = [
                                                    primary_radius,
                                                    primary_radius,
                                                    primary_depth,
                                                    0.0,
                                                ];
                                                sphere_rotation = tube_axis_rot;
                                                sphere_scale = glam::Vec3::new(
                                                    primary_radius / sphere_radius,
                                                    primary_radius / sphere_radius,
                                                    primary_depth / sphere_radius,
                                                );
                                                gizmo_target = GizmoTargetKind::Sphere;
                                                has_selection = true;
                                                optical_trace_enabled = true;
                                                optical_trace_rays = 11;
                                                sun_azimuth_deg = 0.0;
                                                sun_elevation_deg = 70.0;
                                                sun_intensity = 1.5;
                                                sun_lamp_distance = sun_lamp_distance.max(80.0);
                                                let sun_elevation = sun_elevation_deg.to_radians();
                                                sun_empty_position = active_center
                                                    + glam::Vec3::new(
                                                        sun_elevation.cos(),
                                                        sun_elevation.sin(),
                                                        0.0,
                                                    ) * sun_lamp_distance;
                                                sun_empty_rotation = glam::Quat::IDENTITY;
                                                camera = Camera::look_at(
                                                    focuser_center + folded_axis * 8.0 + glam::Vec3::Y * 0.35,
                                                    focuser_center - folded_axis * 0.4,
                                                );
                                                accumulation_dirty = true;
                                                project_status = "Newtonian telescope demo created".to_string();
                                            }
                                        }
                                        if ui.button("Cassegrain Demo").clicked() {
                                            let (scene_id, master_id) = match scene_kind {
                                                SceneKind::Decanter => (decanter_scene_id, decanter_master),
                                                SceneKind::Wine => (wine_scene_id, wine_master),
                                                SceneKind::CornellBox => (cornell_scene_id, cornell_master),
                                            };
                                            if scene_id.0 != 0 && master_id.0 != 0 {
                                                cassegrain_focus_offset = 0.0;
                                                let scene_objects = main_db.scene_objects_recursive(scene_id);
                                                for object_id in scene_objects {
                                                    if primitive_shape_by_id.contains_key(&object_id) {
                                                        main_db.delete_object(object_id);
                                                        object_target_by_id.remove(&object_id);
                                                        primitive_shape_by_id.remove(&object_id);
                                                        primitive_lens_params_by_id.remove(&object_id);
                                                        object_material_names.remove(&object_id);
                                                    } else {
                                                        main_db.unlink_object_from_scene(scene_id, object_id);
                                                    }
                                                }

                                                let bench_y = -1.5 + sphere_radius * 1.35;
                                                let tube_axis = glam::Vec3::X;
                                                let primary_center = glam::Vec3::new(0.0, bench_y, 0.0);
                                                let primary_radius = sphere_radius * 0.38;
                                                let primary_focal_length = primary_radius * 8.0;
                                                let primary_depth =
                                                    primary_radius * primary_radius / (8.0 * primary_focal_length);
                                                let primary_vertex =
                                                    primary_center - tube_axis * primary_depth;
                                                let primary_rot =
                                                    glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
                                                let puppy_object_distance = primary_focal_length * 100.0;
                                                let primary_image_distance = 1.0
                                                    / (1.0 / primary_focal_length
                                                        - 1.0 / puppy_object_distance);
                                                let target_secondary_magnification = 2.5;
                                                let back_focus_distance = primary_radius * 1.25;
                                                let secondary_to_prime_focus =
                                                    (primary_image_distance + back_focus_distance)
                                                        / (target_secondary_magnification + 1.0);
                                                let secondary_vertex = primary_vertex
                                                    + tube_axis
                                                        * (primary_image_distance
                                                            - secondary_to_prime_focus);
                                                let rear_focus =
                                                    primary_vertex - tube_axis * back_focus_distance;
                                                let secondary_to_rear_focus =
                                                    (secondary_vertex - rear_focus).length();
                                                let prime_focus =
                                                    primary_vertex + tube_axis * primary_image_distance;
                                                let hyperbola_center = (prime_focus + rear_focus) * 0.5;
                                                let hyperbola_c =
                                                    (prime_focus - rear_focus).length() * 0.5;
                                                let hyperbola_a =
                                                    (secondary_vertex - hyperbola_center).length();
                                                let hyperbola_b = (hyperbola_c * hyperbola_c
                                                    - hyperbola_a * hyperbola_a)
                                                    .max(0.01)
                                                    .sqrt();
                                                let secondary_clear_radius = (primary_radius
                                                    * secondary_to_prime_focus
                                                    / primary_focal_length
                                                    * 1.18)
                                                    .clamp(0.18, primary_radius * 0.34);
                                                let beam_radius_at_primary_hole = secondary_clear_radius
                                                    * back_focus_distance
                                                    / secondary_to_rear_focus;
                                                let primary_hole_radius = (beam_radius_at_primary_hole * 1.45)
                                                    .clamp(primary_radius * 0.13, primary_radius * 0.28);
                                                let secondary_sag = hyperbola_a
                                                    * ((1.0
                                                        + secondary_clear_radius
                                                            * secondary_clear_radius
                                                            / (hyperbola_b * hyperbola_b))
                                                        .sqrt()
                                                        - 1.0);
                                                let secondary_center = secondary_vertex;
                                                let secondary_rot = primary_rot;
                                                let puppy_center =
                                                    primary_vertex + tube_axis * puppy_object_distance;
                                                let puppy_height = primary_radius * 2.4;
                                                let collimator_params = [6.0, 6.0, 0.25, 0.0];
                                                let collimator_ior = 1.52_f32;
                                                let collimator_power = (collimator_ior - 1.0)
                                                    * (1.0 / collimator_params[0]
                                                        + 1.0 / collimator_params[1]
                                                        - ((collimator_ior - 1.0)
                                                            * collimator_params[2])
                                                            / (collimator_ior
                                                                * collimator_params[0]
                                                                * collimator_params[1]));
                                                let collimator_focal_length =
                                                    (1.0 / collimator_power).clamp(1.0, 12.0);
                                                let rear_view_axis = -tube_axis;
                                                let collimator_center = rear_focus
                                                    + rear_view_axis * collimator_focal_length;
                                                let collimator_rot = glam::Quat::from_rotation_arc(
                                                    glam::Vec3::Z,
                                                    rear_view_axis,
                                                );
                                                let puppy_target_rot =
                                                    glam::Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
                                                let puppy_aspect = puppy_dimensions.0 as f32
                                                    / puppy_dimensions.1.max(1) as f32;

                                                let puppy_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::ImagePlane,
                                                    "White",
                                                    DbTransform {
                                                        location: puppy_center,
                                                        rotation: puppy_target_rot,
                                                        scale: glam::Vec3::new(
                                                            puppy_aspect.max(0.1) * puppy_height,
                                                            puppy_height,
                                                            1.0,
                                                        ),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&puppy_id) {
                                                    obj.name = "Distant Puppy Target".to_string();
                                                }
                                                main_db.collection_link_object(master_id, puppy_id);
                                                main_db.ensure_scene_base(scene_id, puppy_id, true, true);

                                                let primary_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::ParabolicMirror,
                                                    "Mirror",
                                                    DbTransform {
                                                        location: primary_center,
                                                        rotation: primary_rot,
                                                        scale: glam::Vec3::new(
                                                            primary_radius / sphere_radius,
                                                            primary_radius / sphere_radius,
                                                            primary_depth / sphere_radius,
                                                        ),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&primary_id) {
                                                    obj.name = "Cassegrain Annular Primary".to_string();
                                                }
                                                primitive_lens_params_by_id.insert(
                                                    primary_id,
                                                    [0.0, 0.0, 0.0, primary_hole_radius],
                                                );
                                                main_db.collection_link_object(master_id, primary_id);
                                                main_db.ensure_scene_base(scene_id, primary_id, true, true);

                                                let secondary_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::HyperbolicMirror,
                                                    "Mirror",
                                                    DbTransform {
                                                        location: secondary_center,
                                                        rotation: secondary_rot,
                                                        scale: glam::Vec3::new(
                                                            secondary_clear_radius / sphere_radius,
                                                            secondary_clear_radius / sphere_radius,
                                                            secondary_sag.max(0.02) / sphere_radius,
                                                        ),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&secondary_id) {
                                                    obj.name = "Cassegrain Secondary Mirror".to_string();
                                                }
                                                primitive_lens_params_by_id.insert(
                                                    secondary_id,
                                                    [
                                                        hyperbola_a,
                                                        hyperbola_b,
                                                        secondary_clear_radius,
                                                        0.0,
                                                    ],
                                                );
                                                main_db.collection_link_object(master_id, secondary_id);
                                                main_db.ensure_scene_base(scene_id, secondary_id, true, true);

                                                let collimator_id = create_primitive_object(
                                                    &mut main_db,
                                                    &mut object_target_by_id,
                                                    &mut primitive_shape_by_id,
                                                    &mut object_material_names,
                                                    PrimitiveShape::SphericalLens,
                                                    "Glass",
                                                    DbTransform {
                                                        location: collimator_center,
                                                        rotation: collimator_rot,
                                                        scale: glam::Vec3::splat(0.15),
                                                    },
                                                    sphere_radius,
                                                    &mut primitive_shape,
                                                    &mut uniforms,
                                                );
                                                if let Some(obj) = main_db.objects.get_mut(&collimator_id) {
                                                    obj.name = "Rear Collimator Lens".to_string();
                                                }
                                                primitive_lens_params_by_id
                                                    .insert(collimator_id, collimator_params);
                                                main_db.collection_link_object(master_id, collimator_id);
                                                main_db.ensure_scene_base(scene_id, collimator_id, true, true);

                                                selected_primitive_id = primary_id;
                                                primitive_shape = PrimitiveShape::ParabolicMirror;
                                                uniforms.sphere_params[3] = 2.0;
                                                uniforms.sphere_pos = [
                                                    primary_center.x,
                                                    primary_center.y,
                                                    primary_center.z,
                                                    sphere_radius,
                                                ];
                                                uniforms.sphere_rot = [
                                                    primary_rot.x,
                                                    primary_rot.y,
                                                    primary_rot.z,
                                                    primary_rot.w,
                                                ];
                                                uniforms.sphere_extent = [
                                                    primary_radius,
                                                    primary_radius,
                                                    primary_depth,
                                                    0.0,
                                                ];
                                                sphere_rotation = primary_rot;
                                                sphere_scale = glam::Vec3::new(
                                                    primary_radius / sphere_radius,
                                                    primary_radius / sphere_radius,
                                                    primary_depth / sphere_radius,
                                                );
                                                gizmo_target = GizmoTargetKind::Sphere;
                                                has_selection = true;
                                                optical_trace_enabled = true;
                                                optical_trace_rays = 11;
                                                sun_azimuth_deg = 0.0;
                                                sun_elevation_deg = 70.0;
                                                sun_intensity = 1.5;
                                                sun_lamp_distance = sun_lamp_distance.max(80.0);
                                                let sun_elevation = sun_elevation_deg.to_radians();
                                                sun_empty_position = active_center
                                                    + glam::Vec3::new(
                                                        sun_elevation.cos(),
                                                        sun_elevation.sin(),
                                                        0.0,
                                                    ) * sun_lamp_distance;
                                                sun_empty_rotation = glam::Quat::IDENTITY;
                                                camera = Camera::look_at(
                                                    collimator_center
                                                        + rear_view_axis * (primary_radius * 1.5),
                                                    collimator_center - rear_view_axis * primary_radius,
                                                );
                                                render_mode = RenderModeKind::Pathtraced;
                                                uniforms.camera_aperture = 0.0;
                                                accumulation_dirty = true;
                                                project_status =
                                                    "Cassegrain collimated-view demo created".to_string();
                                            }
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
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::Cube,
                                                            "Glass",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                glam::Vec3::ONE,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = glam::Vec3::ONE;
                                                        main_db.collection_link_object(decanter_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
                                                        ui.close();
                                                    }
                                                    if ui.button("Sphere").clicked() {
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::Sphere,
                                                            "Glass",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                glam::Vec3::ONE,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = glam::Vec3::ONE;
                                                        main_db.collection_link_object(decanter_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
                                                        ui.close();
                                                    }
                                                    if ui.button("Spherical Lens").clicked() {
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::SphericalLens,
                                                            "Glass",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                glam::Vec3::ONE,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        primitive_lens_params_by_id
                                                            .insert(object_id, uniforms.lens_params);
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = glam::Vec3::ONE;
                                                        main_db.collection_link_object(decanter_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
                                                        ui.close();
                                                    }
                                                    if ui.button("Image").clicked() {
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let image_aspect = puppy_dimensions.0 as f32
                                                            / puppy_dimensions.1.max(1) as f32;
                                                        let image_scale =
                                                            glam::Vec3::new(image_aspect.max(0.1), 1.0, 1.0);
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::ImagePlane,
                                                            "White",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                image_scale,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = image_scale;
                                                        uniforms.sphere_rot = [0.0, 0.0, 0.0, 1.0];
                                                        uniforms.sphere_extent = [
                                                            sphere_radius * sphere_scale.x,
                                                            sphere_radius * sphere_scale.y,
                                                            0.05,
                                                            0.0,
                                                        ];
                                                        main_db.collection_link_object(decanter_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
                                                        ui.close();
                                                    }
                                                    if ui.button("Parabolic Mirror").clicked() {
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::ParabolicMirror,
                                                            "Mirror",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                glam::Vec3::ONE,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = glam::Vec3::ONE;
                                                        main_db.collection_link_object(decanter_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
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
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::Cube,
                                                            "Glass",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                glam::Vec3::ONE,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = glam::Vec3::ONE;
                                                        main_db.collection_link_object(cornell_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
                                                        ui.close();
                                                    }
                                                    if ui.button("Sphere").clicked() {
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::Sphere,
                                                            "Glass",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                glam::Vec3::ONE,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = glam::Vec3::ONE;
                                                        main_db.collection_link_object(cornell_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
                                                        ui.close();
                                                    }
                                                    if ui.button("Spherical Lens").clicked() {
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::SphericalLens,
                                                            "Glass",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                glam::Vec3::ONE,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        primitive_lens_params_by_id
                                                            .insert(object_id, uniforms.lens_params);
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = glam::Vec3::ONE;
                                                        main_db.collection_link_object(cornell_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
                                                        ui.close();
                                                    }
                                                    if ui.button("Image").clicked() {
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let image_aspect = puppy_dimensions.0 as f32
                                                            / puppy_dimensions.1.max(1) as f32;
                                                        let image_scale =
                                                            glam::Vec3::new(image_aspect.max(0.1), 1.0, 1.0);
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::ImagePlane,
                                                            "White",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                image_scale,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = image_scale;
                                                        uniforms.sphere_rot = [0.0, 0.0, 0.0, 1.0];
                                                        uniforms.sphere_extent = [
                                                            sphere_radius * sphere_scale.x,
                                                            sphere_radius * sphere_scale.y,
                                                            0.05,
                                                            0.0,
                                                        ];
                                                        main_db.collection_link_object(cornell_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
                                                        ui.close();
                                                    }
                                                    if ui.button("Parabolic Mirror").clicked() {
                                                        let current_pos = glam::Vec3::new(
                                                            uniforms.sphere_pos[0],
                                                            uniforms.sphere_pos[1],
                                                            uniforms.sphere_pos[2],
                                                        );
                                                        let spawn_index = primitive_shape_by_id.len();
                                                        let object_id = create_primitive_object(
                                                            &mut main_db,
                                                            &mut object_target_by_id,
                                                            &mut primitive_shape_by_id,
                                                            &mut object_material_names,
                                                            PrimitiveShape::ParabolicMirror,
                                                            "Mirror",
                                                            primitive_spawn_transform(
                                                                current_pos,
                                                                spawn_index,
                                                                glam::Vec3::ONE,
                                                                sphere_radius,
                                                            ),
                                                            sphere_radius,
                                                            &mut primitive_shape,
                                                            &mut uniforms,
                                                        );
                                                        selected_primitive_id = object_id;
                                                        sphere_rotation = glam::Quat::IDENTITY;
                                                        sphere_scale = glam::Vec3::ONE;
                                                        main_db.collection_link_object(cornell_master, object_id);
                                                        main_db.ensure_scene_base(scene_id, object_id, true, true);
                                                        gizmo_target = GizmoTargetKind::Sphere;
                                                        has_selection = true;
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
                                                    object_material_names.clear();
                                                    material_library.clear();
                                                    material_library.insert("White".to_string(), make_white_material());
                                                    material_library.insert("Glass".to_string(), make_glass_material());
                                                    material_library.insert("Mirror".to_string(), make_mirror_material());
                                                    object_material_names.insert(sphere_obj_id, "Glass".to_string());
                                                    object_material_names.insert(decanter_obj_id, "Glass".to_string());
                                                    object_material_names.insert(wine_obj_id, "Glass".to_string());
                                                    object_material_names.insert(cornell_obj_id, "White".to_string());
                                                    for (_mh, mat) in loaded.materials.iter() {
                                                        material_library.insert(mat.name.clone(), mat.clone());
                                                    }
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
                                                                    } else if name.contains("image")
                                                                        || name.contains("lens")
                                                                        || name.contains("mirror")
                                                                        || name.contains("sphere")
                                                                        || name.contains("cube")
                                                                    {
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
                                                        if lname.contains("image")
                                                            || lname.contains("lens")
                                                            || lname.contains("mirror")
                                                            || lname.contains("sphere")
                                                            || lname.contains("cube")
                                                        {
                                                            let shape = if lname.contains("hyperbolic") {
                                                                PrimitiveShape::HyperbolicMirror
                                                            } else if lname.contains("image") {
                                                                PrimitiveShape::ImagePlane
                                                            } else if lname.contains("lens") {
                                                                PrimitiveShape::SphericalLens
                                                            } else if lname.contains("mirror") {
                                                                PrimitiveShape::ParabolicMirror
                                                            } else if lname.contains("sphere") {
                                                                PrimitiveShape::Sphere
                                                            } else {
                                                                PrimitiveShape::Cube
                                                            };
                                                            set_primitive_shape(
                                                                &mut main_db,
                                                                sphere_obj_id,
                                                                &mut primitive_shape,
                                                                shape,
                                                                &mut uniforms,
                                                            );
                                                            uniforms.sphere_pos[0] = t.x;
                                                            uniforms.sphere_pos[1] = t.y;
                                                            uniforms.sphere_pos[2] = t.z;
                                                            sphere_rotation = r;
                                                            uniforms.sphere_rot = [sphere_rotation.x, sphere_rotation.y, sphere_rotation.z, sphere_rotation.w];
                                                            sphere_scale = s.max(glam::Vec3::splat(0.01));
                                                            uniforms.sphere_extent = [
                                                                sphere_radius * sphere_scale.x,
                                                                sphere_radius * sphere_scale.y,
                                                                sphere_radius * sphere_scale.z,
                                                                0.0,
                                                            ];
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
                                                        if let Some(mh) = obj.material_link {
                                                            if let Some(mat) = loaded.materials.get(mh) {
                                                                material_library.insert(mat.name.clone(), mat.clone());
                                                                let target_id = if lname.contains("decanter") {
                                                                    Some(decanter_obj_id)
                                                                } else if lname.contains("wine") {
                                                                    Some(wine_obj_id)
                                                                } else if lname.contains("image")
                                                                    || lname.contains("lens")
                                                                    || lname.contains("mirror")
                                                                    || lname.contains("sphere")
                                                                    || lname.contains("cube")
                                                                {
                                                                    Some(sphere_obj_id)
                                                                } else if lname.contains("cornell") {
                                                                    Some(cornell_obj_id)
                                                                } else {
                                                                    None
                                                                };
                                                                if let Some(tid) = target_id {
                                                                    object_material_names
                                                                        .insert(tid, mat.name.clone());
                                                                }
                                                            }
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
                                                &object_material_names,
                                                &material_library,
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
                                        let scene_object_ids = main_db.scene_objects_recursive(scene_id);
                                        let mut delete_object_id = None;
                                        let mut visibility_change = None;
                                        for object_id in scene_object_ids {
                                            let Some(target) = object_target_by_id.get(&object_id).copied() else {
                                                continue;
                                            };
                                            let label = main_db
                                                .objects
                                                .get(&object_id)
                                                .map(|o| o.name.as_str())
                                                .unwrap_or("Object");
                                            let is_selectable = target_allowed_in_scene(scene_kind, target);
                                            let is_visible = main_db
                                                .scenes
                                                .get(&scene_id)
                                                .and_then(|scene| {
                                                    main_db.view_layers.get(&scene.view_layer_id)
                                                })
                                                .and_then(|view_layer| {
                                                    view_layer
                                                        .bases
                                                        .iter()
                                                        .find(|base| base.object_id == object_id)
                                                })
                                                .is_some_and(|base| base.visible);
                                            let is_selected = has_selection
                                                && if target == GizmoTargetKind::Sphere {
                                                    gizmo_target == GizmoTargetKind::Sphere
                                                        && selected_primitive_id == object_id
                                                } else {
                                                    gizmo_target == target
                                                };
                                            ui.horizontal(|ui| {
                                                let visibility_icon = if is_visible {
                                                    "\u{25c9}"
                                                } else {
                                                    "\u{25cb}"
                                                };
                                                if ui
                                                    .small_button(visibility_icon)
                                                    .on_hover_text(if is_visible {
                                                        "Hide object"
                                                    } else {
                                                        "Show object"
                                                    })
                                                    .clicked()
                                                {
                                                    visibility_change =
                                                        Some((object_id, !is_visible));
                                                }
                                                let clicked = ui
                                                    .add_enabled_ui(is_selectable, |ui| {
                                                        ui.selectable_label(is_selected, label).clicked()
                                                    })
                                                    .inner;
                                                if clicked {
                                                    gizmo_target = target;
                                                    has_selection = true;
                                                    if target == GizmoTargetKind::Sphere {
                                                        selected_primitive_id = object_id;
                                                        if let Some(shape) =
                                                            primitive_shape_by_id.get(&object_id).copied()
                                                        {
                                                            primitive_shape = shape;
                                                            uniforms.sphere_params[3] = match shape {
                                                                PrimitiveShape::Cube => 0.0,
                                                                PrimitiveShape::Sphere => 1.0,
                                                                PrimitiveShape::ParabolicMirror => 2.0,
                                                                PrimitiveShape::SphericalLens => 3.0,
                                                                PrimitiveShape::ImagePlane => 4.0,
                                                                PrimitiveShape::HyperbolicMirror => 5.0,
                                                            };
                                                            if matches!(
                                                                shape,
                                                                PrimitiveShape::SphericalLens
                                                                    | PrimitiveShape::ParabolicMirror
                                                                    | PrimitiveShape::HyperbolicMirror
                                                            ) {
                                                                let lens_params =
                                                                    *primitive_lens_params_by_id
                                                                        .entry(object_id)
                                                                        .or_insert(uniforms.lens_params);
                                                                uniforms.lens_params = lens_params;
                                                            }
                                                        }
                                                        if let Some(obj) = main_db.objects.get(&object_id) {
                                                            uniforms.sphere_pos = [
                                                                obj.transform.location.x,
                                                                obj.transform.location.y,
                                                                obj.transform.location.z,
                                                                sphere_radius,
                                                            ];
                                                            sphere_rotation = obj.transform.rotation;
                                                            sphere_scale = obj.transform.scale;
                                                            uniforms.sphere_rot = [
                                                                sphere_rotation.x,
                                                                sphere_rotation.y,
                                                                sphere_rotation.z,
                                                                sphere_rotation.w,
                                                            ];
                                                            uniforms.sphere_extent = [
                                                                sphere_radius * sphere_scale.x,
                                                                sphere_radius * sphere_scale.y,
                                                                sphere_radius * sphere_scale.z,
                                                                0.0,
                                                            ];
                                                        }
                                                    }
                                                }
                                                if ui.small_button("X").clicked() {
                                                    delete_object_id = Some(object_id);
                                                }
                                            });
                                        }
                                        if let Some((object_id, visible)) = visibility_change {
                                            main_db.set_scene_base_visibility(
                                                scene_id,
                                                object_id,
                                                visible,
                                            );
                                            let hidden_was_selected = has_selection
                                                && object_target_by_id
                                                    .get(&object_id)
                                                    .copied()
                                                    .is_some_and(|target| {
                                                        if target == GizmoTargetKind::Sphere {
                                                            gizmo_target
                                                                == GizmoTargetKind::Sphere
                                                                && selected_primitive_id
                                                                    == object_id
                                                        } else {
                                                            gizmo_target == target
                                                        }
                                                    });
                                            if !visible && hidden_was_selected {
                                                has_selection = false;
                                                gizmo_target = default_target_for_scene(scene_kind);
                                            }
                                            accumulation_dirty = true;
                                        }
                                        if let Some(object_id) = delete_object_id {
                                            let was_selected = has_selection
                                                && if object_id == selected_primitive_id {
                                                    gizmo_target == GizmoTargetKind::Sphere
                                                } else {
                                                    object_target_by_id
                                                        .get(&object_id)
                                                        .copied()
                                                        .is_some_and(|target| target == gizmo_target)
                                                };
                                            if primitive_shape_by_id.contains_key(&object_id) {
                                                main_db.delete_object(object_id);
                                                object_target_by_id.remove(&object_id);
                                                primitive_shape_by_id.remove(&object_id);
                                                primitive_lens_params_by_id.remove(&object_id);
                                                object_material_names.remove(&object_id);
                                            } else {
                                                main_db.unlink_object_from_scene(scene_id, object_id);
                                            }
                                            if was_selected {
                                                has_selection = false;
                                                gizmo_target = default_target_for_scene(scene_kind);
                                            }
                                            accumulation_dirty = true;
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
                                        ui.horizontal(|ui| {
                                            ui.label("Render");
                                            let path_clicked = ui
                                                .selectable_value(
                                                    &mut render_mode,
                                                    RenderModeKind::Pathtraced,
                                                    "Pathtraced",
                                                )
                                                .changed();
                                            let ray_clicked = ui
                                                .selectable_value(
                                                    &mut render_mode,
                                                    RenderModeKind::Raytraced,
                                                    "Raytraced",
                                                )
                                                .changed();
                                            if path_clicked || ray_clicked {
                                                accumulation_dirty = true;
                                            }
                                        });
                                        ui.separator();
                                        if !target_allowed_in_scene(scene_kind, gizmo_target) {
                                            gizmo_target = default_target_for_scene(scene_kind);
                                        }
                                        let selected_label = if has_selection {
                                            match gizmo_target {
                                                GizmoTargetKind::Sphere => main_db
                                                    .objects
                                                    .get(&selected_primitive_id)
                                                    .map(|obj| obj.name.as_str())
                                                    .unwrap_or(target_label(gizmo_target)),
                                                _ => target_label(gizmo_target),
                                            }
                                        } else {
                                            "None"
                                        };
                                        ui.label(format!("Selected: {}", selected_label));
                                        ui.horizontal(|ui| {
                                            ui.selectable_value(&mut gizmo_mode, GizmoModeKind::Translate, "Move");
                                            ui.selectable_value(&mut gizmo_mode, GizmoModeKind::Rotate, "Rotate");
                                            ui.selectable_value(&mut gizmo_mode, GizmoModeKind::Scale, "Scale");
                                        });
                                        if has_selection
                                            && gizmo_target == GizmoTargetKind::Sphere
                                            && matches!(
                                                primitive_shape,
                                                PrimitiveShape::SphericalLens
                                                    | PrimitiveShape::ParabolicMirror
                                                    | PrimitiveShape::HyperbolicMirror
                                            )
                                        {
                                            ui.separator();
                                            ui.collapsing("Optical Trace", |ui| {
                                                ui.checkbox(&mut optical_trace_enabled, "Show rays");
                                                ui.checkbox(
                                                    &mut optical_trace_image_area,
                                                    "Trace image area",
                                                );
                                                ui.add(
                                                    egui::Slider::new(&mut optical_trace_rays, 3..=21)
                                                        .text("Rays"),
                                                );
                                                if optical_trace_rays % 2 == 0 {
                                                    optical_trace_rays += 1;
                                                }
                                            });
                                        }
                                        if has_selection
                                            && gizmo_target == GizmoTargetKind::Sphere
                                            && primitive_shape == PrimitiveShape::SphericalLens
                                        {
                                            ui.separator();
                                            ui.collapsing("Lens", |ui| {
                                                let mut lens_changed = false;
                                                lens_changed |= ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut uniforms.lens_params[0],
                                                            0.25..=64.0,
                                                        )
                                                        .text("Front radius"),
                                                    )
                                                    .changed();
                                                lens_changed |= ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut uniforms.lens_params[1],
                                                            0.25..=64.0,
                                                        )
                                                        .text("Back radius"),
                                                    )
                                                    .changed();
                                                lens_changed |= ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut uniforms.lens_params[2],
                                                            0.05..=24.0,
                                                        )
                                                        .text("Thickness"),
                                                    )
                                                    .changed();
                                                uniforms.lens_params[0] = uniforms.lens_params[0].max(0.25);
                                                uniforms.lens_params[1] = uniforms.lens_params[1].max(0.25);
                                                uniforms.lens_params[2] = uniforms.lens_params[2].max(0.05);
                                                if lens_changed {
                                                    primitive_lens_params_by_id.insert(
                                                        selected_primitive_id,
                                                        uniforms.lens_params,
                                                    );
                                                    accumulation_dirty = true;
                                                }
                                            });
                                        }
                                        if has_selection
                                            && gizmo_target == GizmoTargetKind::Sphere
                                            && matches!(
                                                primitive_shape,
                                                PrimitiveShape::ParabolicMirror
                                                    | PrimitiveShape::HyperbolicMirror
                                            )
                                        {
                                            ui.separator();
                                            ui.collapsing("Cassegrain Optics", |ui| {
                                                let mut optics_changed = false;

                                                ui.label("Exact position");
                                                optics_changed |= ui
                                                    .add(
                                                        egui::DragValue::new(&mut uniforms.sphere_pos[0])
                                                            .speed(0.02)
                                                            .prefix("X  "),
                                                    )
                                                    .changed();
                                                optics_changed |= ui
                                                    .add(
                                                        egui::DragValue::new(&mut uniforms.sphere_pos[1])
                                                            .speed(0.02)
                                                            .prefix("Y  "),
                                                    )
                                                    .changed();
                                                optics_changed |= ui
                                                    .add(
                                                        egui::DragValue::new(&mut uniforms.sphere_pos[2])
                                                            .speed(0.02)
                                                            .prefix("Z  "),
                                                    )
                                                    .changed();

                                                match primitive_shape {
                                                    PrimitiveShape::ParabolicMirror => {
                                                        let mut aperture =
                                                            sphere_radius * sphere_scale.x.max(0.01);
                                                        let depth =
                                                            sphere_radius * sphere_scale.z.max(0.0001);
                                                        let mut focal_length =
                                                            aperture * aperture / (8.0 * depth);
                                                        let mut hole_radius =
                                                            uniforms.lens_params[3].max(0.0);

                                                        ui.separator();
                                                        ui.label("Parabolic primary");
                                                        optics_changed |= ui
                                                            .add(
                                                                egui::DragValue::new(&mut aperture)
                                                                    .speed(0.02)
                                                                    .range(0.1..=128.0)
                                                                    .prefix("Aperture radius  "),
                                                            )
                                                            .changed();
                                                        optics_changed |= ui
                                                            .add(
                                                                egui::DragValue::new(&mut focal_length)
                                                                    .speed(0.05)
                                                                    .range(0.25..=1024.0)
                                                                    .prefix("Focal length  "),
                                                            )
                                                            .changed();
                                                        optics_changed |= ui
                                                            .add(
                                                                egui::DragValue::new(&mut hole_radius)
                                                                    .speed(0.01)
                                                                    .range(0.0..=aperture * 0.9)
                                                                    .prefix("Central hole radius  "),
                                                            )
                                                            .changed();

                                                        aperture = aperture.max(0.1);
                                                        focal_length = focal_length.max(0.25);
                                                        hole_radius = hole_radius.clamp(0.0, aperture * 0.9);
                                                        sphere_scale.x = aperture / sphere_radius;
                                                        sphere_scale.y = aperture / sphere_radius;
                                                        sphere_scale.z = aperture * aperture
                                                            / (8.0 * focal_length * sphere_radius);
                                                        uniforms.lens_params[3] = hole_radius;
                                                        let primary_f_number =
                                                            focal_length / (2.0 * aperture);
                                                        ui.label(format!(
                                                            "Primary: f/{:.2}",
                                                            primary_f_number
                                                        ));
                                                        if let Some(target_position) =
                                                            primitive_shape_by_id.iter().find_map(
                                                                |(id, shape)| {
                                                                    if *shape
                                                                        == PrimitiveShape::ImagePlane
                                                                    {
                                                                        main_db.objects
                                                                            .get(id)
                                                                            .map(|obj| {
                                                                                obj.transform.location
                                                                            })
                                                                    } else {
                                                                        None
                                                                    }
                                                                },
                                                            )
                                                        {
                                                            let primary_axis = (sphere_rotation
                                                                * glam::Vec3::Z)
                                                                .normalize_or_zero();
                                                            let primary_center = glam::Vec3::new(
                                                                uniforms.sphere_pos[0],
                                                                uniforms.sphere_pos[1],
                                                                uniforms.sphere_pos[2],
                                                            );
                                                            let vertex = primary_center
                                                                - primary_axis
                                                                    * (sphere_radius
                                                                        * sphere_scale.z);
                                                            let object_distance = (target_position
                                                                - vertex)
                                                                .dot(primary_axis)
                                                                .abs()
                                                                .max(focal_length + 0.01);
                                                            let finite_image_distance = 1.0
                                                                / (1.0 / focal_length
                                                                    - 1.0 / object_distance);
                                                            ui.label(format!(
                                                                "Target: {:.1} focal lengths away (focus shift {:.3})",
                                                                object_distance / focal_length,
                                                                finite_image_distance - focal_length
                                                            ));
                                                        }
                                                        if let Some(secondary_params) =
                                                            primitive_shape_by_id.iter().find_map(
                                                                |(id, shape)| {
                                                                    if *shape
                                                                        == PrimitiveShape::HyperbolicMirror
                                                                    {
                                                                        primitive_lens_params_by_id
                                                                            .get(id)
                                                                            .copied()
                                                                    } else {
                                                                        None
                                                                    }
                                                                },
                                                            )
                                                        {
                                                            let a = secondary_params[0].max(0.01);
                                                            let b = secondary_params[1].max(0.01);
                                                            let c = (a * a + b * b).sqrt();
                                                            let secondary_magnification =
                                                                ((c + a) / (c - a).max(0.01))
                                                                    .max(1.0);
                                                            ui.label(format!(
                                                                "Effective system: f/{:.2} ({:.2}x secondary)",
                                                                primary_f_number
                                                                    * secondary_magnification,
                                                                secondary_magnification
                                                            ));
                                                        }
                                                    }
                                                    PrimitiveShape::HyperbolicMirror => {
                                                        let mut hyperbola_a =
                                                            uniforms.lens_params[0].max(0.01);
                                                        let mut hyperbola_b =
                                                            uniforms.lens_params[1].max(0.01);
                                                        let mut clear_radius =
                                                            uniforms.lens_params[2].max(0.05);

                                                        ui.separator();
                                                        ui.label("Hyperbolic secondary");
                                                        optics_changed |= ui
                                                            .add(
                                                                egui::DragValue::new(&mut hyperbola_a)
                                                                    .speed(0.02)
                                                                    .range(0.01..=512.0)
                                                                    .prefix("Hyperbola a  "),
                                                            )
                                                            .changed();
                                                        optics_changed |= ui
                                                            .add(
                                                                egui::DragValue::new(&mut hyperbola_b)
                                                                    .speed(0.02)
                                                                    .range(0.01..=512.0)
                                                                    .prefix("Hyperbola b  "),
                                                            )
                                                            .changed();
                                                        optics_changed |= ui
                                                            .add(
                                                                egui::DragValue::new(&mut clear_radius)
                                                                    .speed(0.01)
                                                                    .range(0.05..=64.0)
                                                                    .prefix("Clear radius  "),
                                                            )
                                                            .changed();

                                                        let c = (hyperbola_a * hyperbola_a
                                                            + hyperbola_b * hyperbola_b)
                                                            .sqrt();
                                                        let sag = hyperbola_a
                                                            * ((1.0
                                                                + clear_radius * clear_radius
                                                                    / (hyperbola_b * hyperbola_b))
                                                                .sqrt()
                                                                - 1.0);
                                                        uniforms.lens_params = [
                                                            hyperbola_a,
                                                            hyperbola_b,
                                                            clear_radius,
                                                            0.0,
                                                        ];
                                                        sphere_scale.x = clear_radius / sphere_radius;
                                                        sphere_scale.y = clear_radius / sphere_radius;
                                                        sphere_scale.z = sag.max(0.001) / sphere_radius;
                                                        ui.label(format!(
                                                            "Focus spacing: {:.3}   Sag: {:.3}",
                                                            2.0 * c,
                                                            sag
                                                        ));
                                                    }
                                                    _ => {}
                                                }

                                                if optics_changed {
                                                    primitive_lens_params_by_id.insert(
                                                        selected_primitive_id,
                                                        uniforms.lens_params,
                                                    );
                                                    uniforms.sphere_extent = [
                                                        sphere_radius * sphere_scale.x,
                                                        sphere_radius * sphere_scale.y,
                                                        sphere_radius * sphere_scale.z,
                                                        0.0,
                                                    ];
                                                    accumulation_dirty = true;
                                                }
                                            });
                                        }
                                        ui.separator();
                                        ui.collapsing("Sun", |ui| {
                                            ui.add(egui::Slider::new(&mut sun_azimuth_deg, -180.0..=180.0).text("Azimuth"));
                                            ui.add(egui::Slider::new(&mut sun_elevation_deg, -10.0..=89.0).text("Elevation"));
                                            ui.add(egui::Slider::new(&mut sun_intensity, 0.0..=5.0).text("Intensity"));
                                        });
                                        ui.collapsing("Photon Map", |ui| {
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut uniforms.photon_brightness,
                                                        0.0..=10.0,
                                                    )
                                                    .text("Brightness"),
                                                )
                                                .changed()
                                            {
                                                accumulation_dirty = true;
                                            }
                                        });
                                        egui::CollapsingHeader::new("Ground")
                                            .default_open(true)
                                            .show(ui, |ui| {
                                                if ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut uniforms.ground_brightness,
                                                            0.0..=3.0,
                                                        )
                                                        .text("Brightness"),
                                                    )
                                                    .changed()
                                                {
                                                    accumulation_dirty = true;
                                                }
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
                                        ui.collapsing("Camera", |ui| {
                                            let fov_changed = ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut camera_fov_deg,
                                                        10.0..=120.0,
                                                    )
                                                    .text("Field of view")
                                                    .suffix(" deg"),
                                                )
                                                .changed();
                                            if fov_changed {
                                                let projection = glam::Mat4::perspective_rh(
                                                    camera_fov_deg.to_radians(),
                                                    config.width as f32 / config.height.max(1) as f32,
                                                    0.1,
                                                    10_000.0,
                                                );
                                                uniforms.proj_inv =
                                                    projection.inverse().to_cols_array_2d();
                                                accumulation_dirty = true;
                                            }
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut uniforms.camera_aperture,
                                                        0.0..=0.8,
                                                    )
                                                    .text("Pupil radius"),
                                                )
                                                .changed()
                                            {
                                                accumulation_dirty = true;
                                            }
                                            let rear_collimator_id = main_db.objects.iter().find_map(
                                                |(id, obj)| {
                                                    if obj.name == "Rear Collimator Lens" {
                                                        Some(*id)
                                                    } else {
                                                        None
                                                    }
                                                },
                                            );
                                            if let Some(collimator_id) = rear_collimator_id {
                                                ui.separator();
                                                let previous_focus = cassegrain_focus_offset;
                                                let mut focus_changed = ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut cassegrain_focus_offset,
                                                            -2.0..=2.0,
                                                        )
                                                        .text("Cassegrain focus")
                                                        .suffix(" units"),
                                                    )
                                                    .changed();
                                                if ui.button("Reset focus").clicked() {
                                                    cassegrain_focus_offset = 0.0;
                                                    focus_changed = true;
                                                }
                                                if focus_changed {
                                                    let focus_delta =
                                                        cassegrain_focus_offset - previous_focus;
                                                    if let Some(obj) =
                                                        main_db.objects.get_mut(&collimator_id)
                                                    {
                                                        let focus_axis = (obj.transform.rotation
                                                            * glam::Vec3::Z)
                                                            .normalize_or_zero();
                                                        obj.transform.location +=
                                                            focus_axis * focus_delta;
                                                        if selected_primitive_id == collimator_id {
                                                            uniforms.sphere_pos[0] =
                                                                obj.transform.location.x;
                                                            uniforms.sphere_pos[1] =
                                                                obj.transform.location.y;
                                                            uniforms.sphere_pos[2] =
                                                                obj.transform.location.z;
                                                        }
                                                    }
                                                    accumulation_dirty = true;
                                                }
                                            }
                                        });
                                        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                                            ui.separator();
                                            ui.collapsing("Shader Graph", |ui| {
                                                let selected_object_id = if has_selection {
                                                    match gizmo_target {
                                                        GizmoTargetKind::Sphere => Some(selected_primitive_id),
                                                        GizmoTargetKind::Decanter => Some(decanter_obj_id),
                                                        GizmoTargetKind::WineGlass => Some(wine_obj_id),
                                                        GizmoTargetKind::CornellBox => Some(cornell_obj_id),
                                                        GizmoTargetKind::SunLamp => Some(sun_obj_id),
                                                        GizmoTargetKind::WineSpotlight => Some(spot_obj_id),
                                                    }
                                                } else {
                                                    None
                                                };
                                                if let Some(obj_id) = selected_object_id {
                                                    let mut mat_name = object_material_names
                                                        .get(&obj_id)
                                                        .cloned()
                                                        .unwrap_or_else(|| "White".to_string());
                                                    egui::ComboBox::from_label("Material")
                                                        .selected_text(&mat_name)
                                                        .show_ui(ui, |ui| {
                                                            for key in material_library.keys() {
                                                                ui.selectable_value(&mut mat_name, key.clone(), key);
                                                            }
                                                        });
                                                    object_material_names.insert(obj_id, mat_name.clone());
                                                    if let Some(mat) = material_library.get(&mat_name) {
                                                        let graph_key = mat_name.clone();
                                                        ui.label(format!("Graph: {}", mat.name));
                                                        material_editor.load_material(&graph_key, mat);
                                                        egui::Frame::default().show(ui, |ui| {
                                                            ui.set_min_height(280.0);
                                                            material_editor.show(ui);
                                                        });
                                                        if let Some(mat_mut) = material_library.get_mut(&mat_name) {
                                                            material_editor.commit_to_material(mat_mut);
                                                        }
                                                        material_runtime_overrides
                                                            .insert(mat_name.clone(), material_editor.runtime_preview());
                                                    } else {
                                                        ui.label("White fallback (no material graph).");
                                                    }
                                                } else {
                                                    ui.label("No object selected.");
                                                }
                                            });
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
                                    [sphere_pos.x, sphere_pos.y, sphere_pos.z, sphere_radius];
                                uniforms.sphere_extent = [
                                    sphere_radius * sphere_scale.x,
                                    sphere_radius * sphere_scale.y,
                                    sphere_radius * sphere_scale.z,
                                    0.0,
                                ];
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
                                camera_fov_deg.to_radians(),
                                config.width as f32 / config.height as f32,
                                0.1,
                                10_000.0,
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
                                        sphere_scale.x as f64,
                                        sphere_scale.y as f64,
                                        sphere_scale.z as f64,
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
                                    let mut translation = glam::Vec3::new(
                                        new_t.translation.x as f32,
                                        new_t.translation.y as f32,
                                        new_t.translation.z as f32,
                                    );
                                    // If Shift is held, make gizmo translations faster (scale deltas)
                                    if keys_pressed.contains("Shift") {
                                        match gizmo_target {
                                            GizmoTargetKind::Sphere => {
                                                let cur = glam::Vec3::new(
                                                    uniforms.sphere_pos[0],
                                                    uniforms.sphere_pos[1],
                                                    uniforms.sphere_pos[2],
                                                );
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::Decanter => {
                                                let cur = decanter_center + decanter_translation;
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::WineGlass => {
                                                let cur = wine_center + wine_translation;
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::CornellBox => {
                                                let cur = active_center + cornell_translation;
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::SunLamp => {
                                                let cur = sun_empty_position;
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::WineSpotlight => {
                                                let cur = spot_empty_position;
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                        }
                                    }
                                    match gizmo_target {
                                    GizmoTargetKind::Sphere => {
                                        uniforms.sphere_pos[0] = translation.x;
                                        uniforms.sphere_pos[1] = translation.y;
                                        uniforms.sphere_pos[2] = translation.z;
                                        let sx = new_t.scale.x.abs() as f32;
                                        let sy = new_t.scale.y.abs() as f32;
                                        let sz = new_t.scale.z.abs() as f32;
                                        sphere_scale = glam::Vec3::new(
                                            sx.clamp(0.15, 8.0),
                                            sy.clamp(0.15, 8.0),
                                            sz.clamp(0.15, 8.0),
                                        );
                                        uniforms.sphere_extent = [
                                            sphere_radius * sphere_scale.x,
                                            sphere_radius * sphere_scale.y,
                                            sphere_radius * sphere_scale.z,
                                            0.0,
                                        ];
                                        sphere_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        uniforms.sphere_rot = [
                                            sphere_rotation.x,
                                            sphere_rotation.y,
                                            sphere_rotation.z,
                                            sphere_rotation.w,
                                        ];
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
                                let sphere_hit = if scene_kind != SceneKind::Wine && sphere_allowed {
                                    match primitive_shape {
                                        PrimitiveShape::Cube => intersect_cube(
                                            ro,
                                            rd,
                                            sphere_center,
                                            glam::Vec3::new(
                                                uniforms.sphere_extent[0],
                                                uniforms.sphere_extent[1],
                                                uniforms.sphere_extent[2],
                                            ),
                                        ),
                                        PrimitiveShape::Sphere => intersect_sphere(
                                            ro,
                                            rd,
                                            sphere_center,
                                            uniforms.sphere_extent[0]
                                                .max(uniforms.sphere_extent[1])
                                                .max(uniforms.sphere_extent[2]),
                                        ),
                                        PrimitiveShape::ParabolicMirror => intersect_sphere(
                                            ro,
                                            rd,
                                            sphere_center,
                                            uniforms.sphere_extent[0]
                                                .max(uniforms.sphere_extent[1])
                                                .max(uniforms.sphere_extent[2]),
                                        ),
                                        PrimitiveShape::SphericalLens => intersect_sphere(
                                            ro,
                                            rd,
                                            sphere_center,
                                            uniforms.sphere_extent[0]
                                                .max(uniforms.sphere_extent[1])
                                                .max(uniforms.sphere_extent[2]),
                                        ),
                                        PrimitiveShape::HyperbolicMirror => intersect_sphere(
                                            ro,
                                            rd,
                                            sphere_center,
                                            uniforms.sphere_extent[0]
                                                .max(uniforms.sphere_extent[1])
                                                .max(uniforms.sphere_extent[2]),
                                        ),
                                        PrimitiveShape::ImagePlane => intersect_cube(
                                            ro,
                                            rd,
                                            sphere_center,
                                            glam::Vec3::new(
                                                uniforms.sphere_extent[0],
                                                uniforms.sphere_extent[1],
                                                0.05,
                                            ),
                                        ),
                                    }
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

                                if optical_trace_enabled
                                    && gizmo_target == GizmoTargetKind::Sphere
                                    && matches!(
                                        primitive_shape,
                                        PrimitiveShape::SphericalLens | PrimitiveShape::ParabolicMirror
                                            | PrimitiveShape::HyperbolicMirror
                                    )
                                {
                                    let display = [display_size[0].max(1.0), display_size[1].max(1.0)];
                                    let primitive_center = glam::Vec3::new(
                                        uniforms.sphere_pos[0],
                                        uniforms.sphere_pos[1],
                                        uniforms.sphere_pos[2],
                                    );
                                    let axis = (sphere_rotation * glam::Vec3::Z).normalize_or_zero();
                                    let tangent = (sphere_rotation * glam::Vec3::Y).normalize_or_zero();
                                    let aperture = uniforms.sphere_extent[0]
                                        .min(uniforms.sphere_extent[1])
                                        .max(0.1)
                                        * 0.82;
                                    let ray_count = optical_trace_rays.max(3);
                                    let denom = (ray_count - 1).max(1) as f32;
                                    let incoming_color = Color32::from_rgb(255, 230, 120);
                                    let outgoing_color = Color32::from_rgb(105, 205, 255);
                                    let focus_color = Color32::from_rgb(255, 125, 72);
                                    let draw_segment = |a: glam::Vec3,
                                                        b: glam::Vec3,
                                                        color: Color32,
                                                        width: f32| {
                                        if let (Some(pa), Some(pb)) = (
                                            world_to_screen(a, view, projection, display),
                                            world_to_screen(b, view, projection, display),
                                        ) {
                                            painter.line_segment(
                                                [Pos2::new(pa[0], pa[1]), Pos2::new(pb[0], pb[1])],
                                                Stroke::new(width, color),
                                            );
                                        }
                                    };

                                    match primitive_shape {
                                        PrimitiveShape::SphericalLens => {
                                            let r1 = uniforms.lens_params[0].max(0.25);
                                            let r2 = uniforms.lens_params[1].max(0.25);
                                            let thickness = uniforms.lens_params[2].max(0.05);
                                            let ior = uniforms.sphere_params[1].max(1.01);
                                            let power = (ior - 1.0)
                                                * (1.0 / r1
                                                    + 1.0 / r2
                                                    - ((ior - 1.0) * thickness)
                                                        / (ior * r1 * r2).max(1e-4));
                                            let focal_length = if power.abs() > 1e-4 {
                                                (1.0 / power).clamp(0.5, 120.0)
                                            } else {
                                                120.0
                                            };
                                            let is_newtonian_focuser = main_db
                                                .objects
                                                .get(&selected_primitive_id)
                                                .is_some_and(|obj| obj.name == "Focuser Lens");
                                            let image_source = if is_newtonian_focuser {
                                                None
                                            } else {
                                                main_db
                                                    .scene_visible_selectable_objects(match scene_kind {
                                                        SceneKind::Decanter => decanter_scene_id,
                                                        SceneKind::Wine => wine_scene_id,
                                                        SceneKind::CornellBox => cornell_scene_id,
                                                    })
                                                    .into_iter()
                                                    .filter(|id| *id != selected_primitive_id)
                                                    .find_map(|id| {
                                                        if primitive_shape_by_id.get(&id).copied()
                                                            != Some(PrimitiveShape::ImagePlane)
                                                        {
                                                            return None;
                                                        }
                                                        main_db.objects.get(&id).map(|obj| {
                                                            (
                                                                obj.transform.location,
                                                                obj.transform.rotation,
                                                                obj.transform.scale,
                                                            )
                                                        })
                                                    })
                                            };
                                            if let Some((image_center, image_rotation, image_scale)) = image_source {
                                                let scene_id = match scene_kind {
                                                    SceneKind::Decanter => decanter_scene_id,
                                                    SceneKind::Wine => wine_scene_id,
                                                    SceneKind::CornellBox => cornell_scene_id,
                                                };
                                                let visible_ids =
                                                    main_db.scene_visible_selectable_objects(scene_id);
                                                let mut lens_sequence = Vec::new();
                                                for object_id in visible_ids {
                                                    if primitive_shape_by_id.get(&object_id).copied()
                                                        != Some(PrimitiveShape::SphericalLens)
                                                    {
                                                        continue;
                                                    }
                                                    let Some(obj) = main_db.objects.get(&object_id) else {
                                                        continue;
                                                    };
                                                    let z = (obj.transform.location - image_center).dot(axis);
                                                    if z <= 0.05 {
                                                        continue;
                                                    }
                                                    let lens_params = primitive_lens_params_by_id
                                                        .get(&object_id)
                                                        .copied()
                                                        .unwrap_or(uniforms.lens_params);
                                                    let mat_name = object_material_names
                                                        .get(&object_id)
                                                        .cloned()
                                                        .unwrap_or_else(|| "Glass".to_string());
                                                    let preview = material_runtime_overrides
                                                        .get(&mat_name)
                                                        .copied()
                                                        .unwrap_or_else(|| {
                                                            preview_from_material_data(
                                                                material_library.get(&mat_name),
                                                            )
                                                        });
                                                    let ior = preview.ior.max(1.01);
                                                    let r1 = lens_params[0].max(0.25);
                                                    let r2 = lens_params[1].max(0.25);
                                                    let thickness = lens_params[2].max(0.05);
                                                    let power = (ior - 1.0)
                                                        * (1.0 / r1
                                                            + 1.0 / r2
                                                            - ((ior - 1.0) * thickness)
                                                                / (ior * r1 * r2).max(1e-4));
                                                    let focal = if power.abs() > 1e-4 {
                                                        (1.0 / power).clamp(0.5, 240.0)
                                                    } else {
                                                        240.0
                                                    };
                                                    let aperture = sphere_radius
                                                        * obj.transform.scale.x.min(obj.transform.scale.y).max(0.01)
                                                        * 0.82;
                                                    lens_sequence.push((
                                                        z,
                                                        obj.transform.location,
                                                        aperture.max(0.1),
                                                        focal,
                                                    ));
                                                }
                                                lens_sequence.sort_by(|a, b| {
                                                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                                                });
                                                if !lens_sequence.is_empty() {
                                                    let image_tangent =
                                                        (image_rotation * glam::Vec3::Y)
                                                            .normalize_or_zero();
                                                    let source_half_height =
                                                        (sphere_radius * image_scale.y).max(0.1);
                                                    let source_samples = [
                                                        (-0.65_f32, Color32::from_rgb(255, 105, 95)),
                                                        (0.0_f32, Color32::from_rgb(255, 238, 140)),
                                                        (0.65_f32, Color32::from_rgb(110, 210, 255)),
                                                    ];
                                                    for (source_t, color) in source_samples {
                                                        let source_offset =
                                                            image_tangent * (source_t * source_half_height);
                                                        let source_point = image_center + source_offset;
                                                        for i in 0..ray_count {
                                                            let t = i as f32 / denom;
                                                            let first_aperture = lens_sequence[0].2;
                                                            let first_lens_z = lens_sequence[0].0;
                                                            let mut current_z = 0.0_f32;
                                                            let mut current_y =
                                                                (source_point - image_center).dot(tangent);
                                                            let first_y = (t * 2.0 - 1.0) * first_aperture;
                                                            let mut angle =
                                                                (first_y - current_y) / first_lens_z.max(0.001);
                                                            let mut prev_point = source_point;
                                                            let mut last_point = source_point;
                                                            for (lens_z, lens_center, lens_aperture, focal) in
                                                                &lens_sequence
                                                            {
                                                                let distance = (*lens_z - current_z).max(0.001);
                                                                let mut lens_y = current_y + angle * distance;
                                                                lens_y = lens_y.clamp(-*lens_aperture, *lens_aperture);
                                                                let hit = *lens_center + tangent * lens_y;
                                                                draw_segment(prev_point, hit, color, 1.1);
                                                                angle -= lens_y / *focal;
                                                                current_y = lens_y;
                                                                current_z = *lens_z;
                                                                prev_point = hit;
                                                                last_point = hit;
                                                            }
                                                            let focus_distance = if angle.abs() > 1e-4 {
                                                                (-current_y / angle).clamp(0.5, 240.0)
                                                            } else {
                                                                80.0
                                                            };
                                                            let image_point = last_point
                                                                + axis * focus_distance
                                                                + tangent * (current_y + angle * focus_distance);
                                                            draw_segment(last_point, image_point, color, 1.3);
                                                            if i == ray_count / 2 {
                                                                if let Some(fp) = world_to_screen(
                                                                    image_point,
                                                                    view,
                                                                    projection,
                                                                    display,
                                                                ) {
                                                                    painter.circle_filled(
                                                                        Pos2::new(fp[0], fp[1]),
                                                                        3.5,
                                                                        color,
                                                                    );
                                                                }
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    let front_plane = primitive_center - axis * (thickness * 0.5);
                                                    let back_plane = primitive_center + axis * (thickness * 0.5);
                                                    let focus = primitive_center + axis * focal_length;
                                                    if let Some(fp) =
                                                        world_to_screen(focus, view, projection, display)
                                                    {
                                                        painter.circle_filled(
                                                            Pos2::new(fp[0], fp[1]),
                                                            4.0,
                                                            focus_color,
                                                        );
                                                    }
                                                    for i in 0..ray_count {
                                                        let t = i as f32 / denom;
                                                        let offset = (t * 2.0 - 1.0) * aperture;
                                                        let front_hit = front_plane + tangent * offset;
                                                        let back_hit = back_plane + tangent * offset;
                                                        let start =
                                                            front_hit - axis * (aperture * 2.5 + thickness);
                                                        draw_segment(start, front_hit, incoming_color, 1.4);
                                                        draw_segment(front_hit, back_hit, incoming_color, 1.0);
                                                        draw_segment(back_hit, focus, outgoing_color, 1.6);
                                                    }
                                                }
                                            } else {
                                                if is_newtonian_focuser {
                                                    let focus = primitive_center - axis * focal_length;
                                                    if let Some(fp) =
                                                        world_to_screen(focus, view, projection, display)
                                                    {
                                                        painter.circle_filled(
                                                            Pos2::new(fp[0], fp[1]),
                                                            4.0,
                                                            focus_color,
                                                        );
                                                    }
                                                    let front_plane =
                                                        primitive_center - axis * (thickness * 0.5);
                                                    let back_plane =
                                                        primitive_center + axis * (thickness * 0.5);
                                                    for i in 0..ray_count {
                                                        let t = i as f32 / denom;
                                                        let offset = (t * 2.0 - 1.0) * aperture;
                                                        let front_hit = front_plane + tangent * offset;
                                                        let back_hit = back_plane + tangent * offset;
                                                        let start = focus;
                                                        let out = back_hit + axis * (aperture * 3.0 + focal_length);
                                                        draw_segment(
                                                            start,
                                                            front_hit,
                                                            Color32::from_rgb(255, 170, 90),
                                                            1.5,
                                                        );
                                                        draw_segment(front_hit, back_hit, incoming_color, 1.0);
                                                        draw_segment(back_hit, out, outgoing_color, 1.5);
                                                    }
                                                } else {
                                                let front_plane = primitive_center - axis * (thickness * 0.5);
                                                let back_plane = primitive_center + axis * (thickness * 0.5);
                                                let focus = primitive_center + axis * focal_length;
                                                if let Some(fp) = world_to_screen(focus, view, projection, display) {
                                                    painter.circle_filled(Pos2::new(fp[0], fp[1]), 4.0, focus_color);
                                                }
                                                for i in 0..ray_count {
                                                    let t = i as f32 / denom;
                                                    let offset = (t * 2.0 - 1.0) * aperture;
                                                    let front_hit = front_plane + tangent * offset;
                                                    let back_hit = back_plane + tangent * offset;
                                                    let start = front_hit - axis * (aperture * 2.5 + thickness);
                                                    draw_segment(start, front_hit, incoming_color, 1.4);
                                                    draw_segment(front_hit, back_hit, incoming_color, 1.0);
                                                    draw_segment(back_hit, focus, outgoing_color, 1.6);
                                                }
                                                }
                                            }
                                        }
                                        PrimitiveShape::ParabolicMirror => {
                                            let radius = uniforms.sphere_extent[0]
                                                .min(uniforms.sphere_extent[1])
                                                .max(0.1);
                                            let depth = uniforms.sphere_extent[2].max(0.1);
                                            let focal_length = (radius * radius / (8.0 * depth)).max(0.1);
                                            let vertex = primitive_center - axis * depth;
                                            let scene_id = match scene_kind {
                                                SceneKind::Decanter => decanter_scene_id,
                                                SceneKind::Wine => wine_scene_id,
                                                SceneKind::CornellBox => cornell_scene_id,
                                            };
                                            let visible_ids =
                                                main_db.scene_visible_selectable_objects(scene_id);
                                            let focus = visible_ids
                                                .iter()
                                                .find_map(|id| {
                                                    if primitive_shape_by_id.get(id).copied()
                                                        != Some(PrimitiveShape::ImagePlane)
                                                    {
                                                        return None;
                                                    }
                                                    let obj = main_db.objects.get(id)?;
                                                    let object_distance =
                                                        (obj.transform.location - vertex).dot(axis);
                                                    if object_distance <= focal_length + 0.01 {
                                                        return None;
                                                    }
                                                    let image_distance = 1.0
                                                        / (1.0 / focal_length
                                                            - 1.0 / object_distance)
                                                            .max(0.001);
                                                    Some(vertex + axis * image_distance)
                                                })
                                                .unwrap_or(vertex + axis * focal_length);
                                            let secondary = visible_ids.iter().find_map(|id| {
                                                let obj = main_db.objects.get(id)?;
                                                if obj.name.contains("Secondary Mirror") {
                                                    Some((
                                                        obj.transform.location,
                                                        (obj.transform.rotation * glam::Vec3::Z)
                                                            .normalize_or_zero(),
                                                    ))
                                                } else {
                                                    None
                                                }
                                            });
                                            let focuser = visible_ids.iter().find_map(|id| {
                                                let obj = main_db.objects.get(id)?;
                                                if obj.name == "Focuser Lens" {
                                                    Some(obj.transform.location)
                                                } else {
                                                    None
                                                }
                                            });
                                            let cassegrain_secondary = visible_ids.iter().find_map(|id| {
                                                if primitive_shape_by_id.get(id).copied()
                                                    != Some(PrimitiveShape::HyperbolicMirror)
                                                {
                                                    return None;
                                                }
                                                let obj = main_db.objects.get(id)?;
                                                let params = primitive_lens_params_by_id.get(id).copied()?;
                                                Some((
                                                    obj.transform.location,
                                                    (obj.transform.rotation * glam::Vec3::Z)
                                                        .normalize_or_zero(),
                                                    params,
                                                ))
                                            });
                                            let rear_collimator = visible_ids.iter().find_map(|id| {
                                                let obj = main_db.objects.get(id)?;
                                                if obj.name != "Rear Collimator Lens" {
                                                    return None;
                                                }
                                                let params = primitive_lens_params_by_id
                                                    .get(id)
                                                    .copied()
                                                    .unwrap_or(uniforms.lens_params);
                                                let mat_name = object_material_names
                                                    .get(id)
                                                    .cloned()
                                                    .unwrap_or_else(|| "Glass".to_string());
                                                let preview = material_runtime_overrides
                                                    .get(&mat_name)
                                                    .copied()
                                                    .unwrap_or_else(|| {
                                                        preview_from_material_data(
                                                            material_library.get(&mat_name),
                                                        )
                                                    });
                                                let ior = preview.ior.max(1.01);
                                                let power = (ior - 1.0)
                                                    * (1.0 / params[0].max(0.25)
                                                        + 1.0 / params[1].max(0.25)
                                                        - ((ior - 1.0) * params[2].max(0.05))
                                                            / (ior
                                                                * params[0].max(0.25)
                                                                * params[1].max(0.25)));
                                                Some((
                                                    obj.transform.location,
                                                    (obj.transform.rotation * glam::Vec3::Z)
                                                        .normalize_or_zero(),
                                                    if power.abs() > 1e-4 {
                                                        1.0 / power
                                                    } else {
                                                        120.0
                                                    },
                                                ))
                                            });
                                            let puppy_source = visible_ids.iter().find_map(|id| {
                                                if primitive_shape_by_id.get(id).copied()
                                                    != Some(PrimitiveShape::ImagePlane)
                                                {
                                                    return None;
                                                }
                                                let obj = main_db.objects.get(id)?;
                                                Some((
                                                    obj.transform.location,
                                                    (obj.transform.rotation * glam::Vec3::Y)
                                                        .normalize_or_zero(),
                                                    (sphere_radius * obj.transform.scale.y).max(0.1),
                                                ))
                                            });
                                            if let Some(fp) = world_to_screen(focus, view, projection, display) {
                                                painter.circle_filled(Pos2::new(fp[0], fp[1]), 4.0, focus_color);
                                            }
                                            let cassegrain_bundle = if optical_trace_image_area {
                                                match (
                                                    cassegrain_secondary,
                                                    rear_collimator,
                                                    puppy_source,
                                                ) {
                                                    (Some(secondary), Some(collimator), Some(source)) => {
                                                        Some((secondary, collimator, source))
                                                    }
                                                    _ => None,
                                                }
                                            } else {
                                                None
                                            };
                                            if let Some((
                                                    (secondary_center, secondary_axis, hyperbola),
                                                    (collimator_center, collimator_axis, collimator_focal),
                                                    (source_center, _source_tangent, source_half_height),
                                                )) = cassegrain_bundle
                                            {
                                                let hyperbola_a = hyperbola[0].max(0.01);
                                                let hyperbola_b = hyperbola[1].max(0.01);
                                                let hyperbola_c = (hyperbola_a * hyperbola_a
                                                    + hyperbola_b * hyperbola_b)
                                                    .sqrt();
                                                let prime_focus = secondary_center
                                                    + secondary_axis * (hyperbola_c - hyperbola_a);
                                                let rear_focus = secondary_center
                                                    - secondary_axis * (hyperbola_c + hyperbola_a);
                                                let object_distance =
                                                    (source_center - vertex).dot(axis).abs().max(0.1);
                                                let primary_image_distance =
                                                    (prime_focus - vertex).dot(axis).abs().max(0.1);
                                                let secondary_magnification =
                                                    ((rear_focus - secondary_center).length()
                                                        / (prime_focus - secondary_center).length().max(0.01))
                                                        .max(0.1);
                                                let area_samples = [
                                                    (-0.65_f32, Color32::from_rgb(255, 95, 90)),
                                                    (0.0_f32, Color32::from_rgb(255, 230, 105)),
                                                    (0.65_f32, Color32::from_rgb(90, 205, 255)),
                                                ];

                                                for (source_t, color) in area_samples {
                                                    let source_offset = source_t * source_half_height;
                                                    let incoming_direction = (-axis
                                                        - tangent
                                                            * (source_offset / object_distance))
                                                        .normalize_or_zero();
                                                    let prime_offset =
                                                        -source_offset * primary_image_distance / object_distance;
                                                    let prime_image = prime_focus + tangent * prime_offset;
                                                    let rear_offset =
                                                        prime_offset * secondary_magnification;
                                                    let rear_image = rear_focus + tangent * rear_offset;

                                                    for i in 0..ray_count {
                                                        let t = i as f32 / denom;
                                                        let offset = (t * 2.0 - 1.0) * aperture;
                                                        let radial = offset.abs().min(radius * 0.98);
                                                        let z = -depth
                                                            + (radial * radial) / (4.0 * focal_length);
                                                        let mirror_hit = primitive_center
                                                            + axis * z
                                                            + tangent * offset;
                                                        let toward_prime = prime_image - mirror_hit;
                                                        let plane_denom = toward_prime.dot(secondary_axis);
                                                        if plane_denom.abs() < 1e-5 {
                                                            continue;
                                                        }
                                                        let secondary_t = (secondary_center - mirror_hit)
                                                            .dot(secondary_axis)
                                                            / plane_denom;
                                                        if secondary_t <= 0.0 || secondary_t >= 1.0 {
                                                            continue;
                                                        }
                                                        let secondary_hit =
                                                            mirror_hit + toward_prime * secondary_t;
                                                        let after_secondary =
                                                            (rear_image - secondary_hit).normalize_or_zero();
                                                        let collimator_denom =
                                                            after_secondary.dot(collimator_axis);
                                                        if collimator_denom.abs() < 1e-5 {
                                                            continue;
                                                        }
                                                        let collimator_t = (collimator_center
                                                            - secondary_hit)
                                                            .dot(collimator_axis)
                                                            / collimator_denom;
                                                        if collimator_t <= 0.0 {
                                                            continue;
                                                        }
                                                        let collimator_hit = secondary_hit
                                                            + after_secondary * collimator_t;
                                                        let collimated_dir = (collimator_axis
                                                            - tangent
                                                                * (rear_offset
                                                                    / collimator_focal.abs().max(0.1)))
                                                            .normalize_or_zero();
                                                        let out = collimator_hit
                                                            + collimated_dir * (radius * 2.5);

                                                        let ray_start = mirror_hit
                                                            - incoming_direction * object_distance;
                                                        draw_segment(
                                                            ray_start,
                                                            mirror_hit,
                                                            color,
                                                            1.0,
                                                        );
                                                        draw_segment(
                                                            mirror_hit,
                                                            secondary_hit,
                                                            color,
                                                            1.4,
                                                        );
                                                        draw_segment(
                                                            secondary_hit,
                                                            rear_image,
                                                            color,
                                                            1.6,
                                                        );
                                                        draw_segment(
                                                            rear_image,
                                                            collimator_hit,
                                                            color,
                                                            1.4,
                                                        );
                                                        draw_segment(collimator_hit, out, color, 1.6);
                                                    }

                                                    if let Some(fp) = world_to_screen(
                                                        rear_image,
                                                        view,
                                                        projection,
                                                        display,
                                                    ) {
                                                        painter.circle_filled(
                                                            Pos2::new(fp[0], fp[1]),
                                                            3.5,
                                                            color,
                                                        );
                                                    }
                                                }
                                            } else {
                                            for i in 0..ray_count {
                                                let t = i as f32 / denom;
                                                let offset = (t * 2.0 - 1.0) * aperture;
                                                let radial = offset.abs().min(radius * 0.98);
                                                let z = -depth + (radial * radial) / (4.0 * focal_length);
                                                let mirror_hit = primitive_center + axis * z + tangent * offset;
                                                let start = mirror_hit + axis * (aperture * 2.5 + depth);
                                                draw_segment(start, mirror_hit, incoming_color, 1.4);
                                                if let (Some((secondary_center, secondary_normal)), Some(focuser_pos)) =
                                                    (secondary, focuser)
                                                {
                                                    let to_focus = focus - mirror_hit;
                                                    let secondary_denom = to_focus.dot(secondary_normal);
                                                    let secondary_t = if secondary_denom.abs() > 1e-4 {
                                                        (secondary_center - mirror_hit)
                                                            .dot(secondary_normal)
                                                            / secondary_denom
                                                    } else {
                                                        1.0
                                                    }
                                                    .clamp(0.0, 1.0);
                                                    let secondary_hit =
                                                        mirror_hit + to_focus * secondary_t;
                                                    let incoming_dir = to_focus.normalize_or_zero();
                                                    let face_n = if incoming_dir.dot(secondary_normal) > 0.0 {
                                                        -secondary_normal
                                                    } else {
                                                        secondary_normal
                                                    };
                                                    let folded_dir = (incoming_dir
                                                        - 2.0 * incoming_dir.dot(face_n) * face_n)
                                                        .normalize_or_zero();
                                                    let remaining = (focus - secondary_hit).length();
                                                    let folded_focus =
                                                        secondary_hit + folded_dir * remaining;
                                                    draw_segment(
                                                        mirror_hit,
                                                        secondary_hit,
                                                        outgoing_color,
                                                        1.6,
                                                    );
                                                    draw_segment(
                                                        secondary_hit,
                                                        folded_focus,
                                                        Color32::from_rgb(255, 170, 90),
                                                        1.6,
                                                    );
                                                    draw_segment(
                                                        folded_focus,
                                                        focuser_pos,
                                                        Color32::from_rgb(170, 120, 255),
                                                        1.4,
                                                    );
                                                } else {
                                                    draw_segment(mirror_hit, focus, outgoing_color, 1.6);
                                                }
                                            }
                                            }
                                        }
                                        _ => {}
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
                            let sphere_mat = object_material_names
                                .get(&selected_primitive_id)
                                .cloned()
                                .unwrap_or_else(|| "White".to_string());
                            let sphere_preview = material_runtime_overrides
                                .get(&sphere_mat)
                                .copied()
                                .unwrap_or_else(|| preview_from_material_data(material_library.get(&sphere_mat)));
                            uniforms.sphere_color = [
                                sphere_preview.base_color[0],
                                sphere_preview.base_color[1],
                                sphere_preview.base_color[2],
                                if sphere_preview.bsdf_connected {
                                    sphere_preview.transmission
                                } else {
                                    0.0
                                },
                            ];
                            uniforms.sphere_params = [
                                sphere_preview.roughness,
                                sphere_preview.ior,
                                if sphere_preview.bsdf_connected { 1.0 } else { 0.0 },
                                uniforms.sphere_params[3],
                            ];

                            let decanter_mat = object_material_names
                                .get(&decanter_obj_id)
                                .cloned()
                                .unwrap_or_else(|| "White".to_string());
                            let decanter_preview = material_runtime_overrides
                                .get(&decanter_mat)
                                .copied()
                                .unwrap_or_else(|| preview_from_material_data(material_library.get(&decanter_mat)));
                            let wine_mat = object_material_names
                                .get(&wine_obj_id)
                                .cloned()
                                .unwrap_or_else(|| "White".to_string());
                            let wine_preview = material_runtime_overrides
                                .get(&wine_mat)
                                .copied()
                                .unwrap_or_else(|| preview_from_material_data(material_library.get(&wine_mat)));
                            let cornell_mat = object_material_names
                                .get(&cornell_obj_id)
                                .cloned()
                                .unwrap_or_else(|| "White".to_string());
                            let cornell_preview = material_runtime_overrides
                                .get(&cornell_mat)
                                .copied()
                                .unwrap_or_else(|| preview_from_material_data(material_library.get(&cornell_mat)));
                            uniforms.cornell_color = [
                                cornell_preview.base_color[0],
                                cornell_preview.base_color[1],
                                cornell_preview.base_color[2],
                                if cornell_preview.bsdf_connected {
                                    cornell_preview.transmission
                                } else {
                                    0.0
                                },
                            ];
                            uniforms.cornell_params = [
                                cornell_preview.roughness,
                                cornell_preview.ior,
                                if cornell_preview.bsdf_connected { 1.0 } else { 0.0 },
                                0.0,
                            ];
                            let material_signature = format!(
                                "{sphere_mat}:{sphere_preview:?}|{decanter_mat}:{decanter_preview:?}|{wine_mat}:{wine_preview:?}|{cornell_mat}:{cornell_preview:?}"
                            );
                            if material_signature != last_material_signature {
                                let set_range = |materials: &mut [crate::mesh::GpuMaterial],
                                                 start: usize,
                                                 count: usize,
                                                 preview: RuntimeMaterialPreview,
                                                 wine_style: bool| {
                                    let end = (start + count).min(materials.len());
                                    for m in &mut materials[start..end] {
                                        if preview.bsdf_connected {
                                            if wine_style {
                                                m.base_color = [
                                                    preview.base_color[0],
                                                    preview.base_color[1],
                                                    preview.base_color[2],
                                                    0.78,
                                                ];
                                                m.params = [
                                                    0.0,
                                                    preview.roughness.min(0.06),
                                                    preview.transmission.max(0.72),
                                                    preview.ior.max(1.0),
                                                ];
                                            } else {
                                                m.base_color = [
                                                    preview.base_color[0],
                                                    preview.base_color[1],
                                                    preview.base_color[2],
                                                    1.0,
                                                ];
                                                m.params = [
                                                    0.0,
                                                    preview.roughness,
                                                    preview.transmission,
                                                    preview.ior,
                                                ];
                                            }
                                        } else {
                                            m.base_color = [1.0, 1.0, 1.0, 1.0];
                                            m.params = [0.0, 0.65, 0.0, 1.0];
                                        }
                                    }
                                };
                                set_range(
                                    &mut mesh.materials,
                                    decanter_material_start,
                                    decanter_material_count,
                                    decanter_preview,
                                    false,
                                );
                                set_range(
                                    &mut mesh.materials,
                                    wine_material_start,
                                    wine_material_count,
                                    wine_preview,
                                    true,
                                );
                                queue.write_buffer(&mat_buf, 0, bytemuck::cast_slice(&mesh.materials));
                                last_material_signature = material_signature;
                                accumulation_dirty = true;
                            }
                            if let Some(obj) = main_db.objects.get_mut(&selected_primitive_id) {
                                obj.transform.location = glam::Vec3::new(
                                    uniforms.sphere_pos[0],
                                    uniforms.sphere_pos[1],
                                    uniforms.sphere_pos[2],
                                );
                                obj.transform.rotation = sphere_rotation;
                                obj.transform.scale = sphere_scale;
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
                            let mut primitive_instances = [GpuPrimitive {
                                pos: [0.0; 4],
                                color: [0.0; 4],
                                params: [0.0; 4],
                                rot: [0.0, 0.0, 0.0, 1.0],
                                extent: [0.0; 4],
                                lens: [0.0; 4],
                            }; MAX_PRIMITIVES];
                            let primitive_scene_id = match scene_kind {
                                SceneKind::Decanter => decanter_scene_id,
                                SceneKind::Wine => wine_scene_id,
                                SceneKind::CornellBox => cornell_scene_id,
                            };
                            let current_scene_exists_for_photons =
                                primitive_scene_id.0 != 0 && main_db.scenes.contains_key(&primitive_scene_id);
                            let (decanter_visible_for_photons, wine_visible_for_photons, cornell_visible_for_photons) =
                                if current_scene_exists_for_photons {
                                    let visible =
                                        main_db.scene_visible_selectable_objects(primitive_scene_id);
                                    (
                                        visible.contains(&decanter_obj_id),
                                        visible.contains(&wine_obj_id),
                                        visible.contains(&cornell_obj_id),
                                    )
                                } else {
                                    (false, false, false)
                                };
                            let visible_primitive_ids = main_db.scene_visible_selectable_objects(primitive_scene_id);
                            let mut photon_bounds_min = glam::Vec3::splat(f32::INFINITY);
                            let mut photon_bounds_max = glam::Vec3::splat(f32::NEG_INFINITY);
                            let mut photon_bounds_valid = false;
                            if decanter_visible_for_photons {
                                include_photon_bounds(
                                    &mut photon_bounds_min,
                                    &mut photon_bounds_max,
                                    &mut photon_bounds_valid,
                                    glam::Vec3::new(
                                        uniforms.decanter_center[0],
                                        uniforms.decanter_center[1],
                                        uniforms.decanter_center[2],
                                    ),
                                    uniforms.decanter_center[3],
                                );
                            }
                            if wine_visible_for_photons {
                                include_photon_bounds(
                                    &mut photon_bounds_min,
                                    &mut photon_bounds_max,
                                    &mut photon_bounds_valid,
                                    glam::Vec3::new(
                                        uniforms.mesh_center[0],
                                        uniforms.mesh_center[1],
                                        uniforms.mesh_center[2],
                                    ),
                                    uniforms.mesh_center[3],
                                );
                            }
                            if cornell_visible_for_photons {
                                include_photon_bounds(
                                    &mut photon_bounds_min,
                                    &mut photon_bounds_max,
                                    &mut photon_bounds_valid,
                                    glam::Vec3::new(
                                        uniforms.cornell_center[0],
                                        uniforms.cornell_center[1],
                                        uniforms.cornell_center[2],
                                    ),
                                    uniforms.cornell_center[3],
                                );
                            }
                            let mut primitive_count = 0usize;
                            for object_id in visible_primitive_ids {
                                if primitive_count >= MAX_PRIMITIVES {
                                    break;
                                }
                                let Some(shape) = primitive_shape_by_id.get(&object_id).copied() else {
                                    continue;
                                };
                                let Some(obj) = main_db.objects.get(&object_id) else {
                                    continue;
                                };
                                let mat_name = object_material_names
                                    .get(&object_id)
                                    .cloned()
                                    .unwrap_or_else(|| "White".to_string());
                                let preview = material_runtime_overrides
                                    .get(&mat_name)
                                    .copied()
                                    .unwrap_or_else(|| preview_from_material_data(material_library.get(&mat_name)));
                                let mut color = [
                                    preview.base_color[0],
                                    preview.base_color[1],
                                    preview.base_color[2],
                                    if preview.bsdf_connected {
                                        preview.transmission
                                    } else {
                                        0.0
                                    },
                                ];
                                if shape == PrimitiveShape::ImagePlane {
                                    color[3] = 0.0;
                                }
                                let mut params = [
                                    preview.roughness,
                                    preview.ior,
                                    if preview.bsdf_connected { 1.0 } else { 0.0 },
                                    match shape {
                                        PrimitiveShape::Cube => 0.0,
                                        PrimitiveShape::Sphere => 1.0,
                                        PrimitiveShape::ParabolicMirror => 2.0,
                                        PrimitiveShape::SphericalLens => 3.0,
                                        PrimitiveShape::ImagePlane => 4.0,
                                        PrimitiveShape::HyperbolicMirror => 5.0,
                                    },
                                ];
                                if shape == PrimitiveShape::ImagePlane {
                                    params[2] = 1.0;
                                }
                                let lens_params = primitive_lens_params_by_id
                                    .get(&object_id)
                                    .copied()
                                    .unwrap_or(uniforms.lens_params);
                                let primitive_center = obj.transform.location;
                                let primitive_radius = sphere_radius
                                    * obj.transform.scale.max_element().max(0.01);
                                include_photon_bounds(
                                    &mut photon_bounds_min,
                                    &mut photon_bounds_max,
                                    &mut photon_bounds_valid,
                                    primitive_center,
                                    primitive_radius,
                                );
                                primitive_instances[primitive_count] = GpuPrimitive {
                                    pos: [
                                        obj.transform.location.x,
                                        obj.transform.location.y,
                                        obj.transform.location.z,
                                        sphere_radius,
                                    ],
                                    color,
                                    params,
                                    rot: [
                                        obj.transform.rotation.x,
                                        obj.transform.rotation.y,
                                        obj.transform.rotation.z,
                                        obj.transform.rotation.w,
                                    ],
                                    extent: [
                                        sphere_radius * obj.transform.scale.x,
                                        sphere_radius * obj.transform.scale.y,
                                        if shape == PrimitiveShape::ImagePlane {
                                            0.05
                                        } else {
                                            sphere_radius * obj.transform.scale.z
                                        },
                                        0.0,
                                    ],
                                    lens: lens_params,
                                };
                                primitive_count += 1;
                            }
                            uniforms.primitive_count = primitive_count as u32;
                            if current_scene_exists_for_photons && photon_bounds_valid {
                                let photon_center = (photon_bounds_min + photon_bounds_max) * 0.5;
                                let photon_radius =
                                    (photon_bounds_max - photon_bounds_min).length() * 0.55;
                                photon_emitter_center = [
                                    photon_center.x,
                                    photon_center.y,
                                    photon_center.z,
                                    photon_radius.max(0.25),
                                ];
                                photons_per_frame = 262_144;
                            }
                            queue.write_buffer(
                                &primitive_buffer,
                                0,
                                bytemuck::cast_slice(&primitive_instances),
                            );
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
                                photon_emitter_center,
                                uniforms.frame,
                                photons_per_frame,
                                uniforms.primitive_count,
                                (if decanter_visible_for_photons { 1 } else { 0 })
                                    | (if wine_visible_for_photons { 2 } else { 0 }),
                                uniforms.decanter_center,
                                uniforms.mesh_center,
                            );
                            photon_mapper.emit_photons(&mut encoder, photons_per_frame);
                            photon_mapper.build_spatial_structure(&mut encoder);
                            {
                                compute_pass.record(
                                    &mut encoder,
                                    &ugroup,
                                    match render_mode {
                                        RenderModeKind::Pathtraced => compute_pass::RenderPath::Pathtraced,
                                        RenderModeKind::Raytraced => compute_pass::RenderPath::Raytraced,
                                    },
                                );
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
