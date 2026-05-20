use crate::{
    mesh::{MeshData, Vertex},
    photon_mapper::PhotonTarget,
    scene_data::Id,
};

use super::types::{MeshObjectInstance, MAX_PHOTON_TARGETS};

pub fn make_prism_mesh(center: glam::Vec3, radius: f32, height: f32) -> MeshData {
    let half_h = height * 0.5;
    let angles = [
        0.0f32,
        std::f32::consts::TAU / 3.0,
        2.0 * std::f32::consts::TAU / 3.0,
    ];
    let mut vertices = Vec::new();
    let mut positions4 = Vec::new();

    for angle in angles {
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        let p = center + glam::Vec3::new(x, -half_h, z);
        vertices.push(Vertex {
            position: p.to_array(),
        });
        positions4.push([p.x, p.y, p.z, 0.0]);
    }
    for angle in angles {
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        let p = center + glam::Vec3::new(x, half_h, z);
        vertices.push(Vertex {
            position: p.to_array(),
        });
        positions4.push([p.x, p.y, p.z, 0.0]);
    }

    let indices: Vec<u32> = vec![
        0, 2, 1, 3, 4, 5, 0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4, 2, 0, 3, 2, 3, 5,
    ];

    let mut normals = vec![glam::Vec3::ZERO; vertices.len()];
    for tri in indices.chunks_exact(3) {
        let p0 = glam::Vec3::from(vertices[tri[0] as usize].position);
        let p1 = glam::Vec3::from(vertices[tri[1] as usize].position);
        let p2 = glam::Vec3::from(vertices[tri[2] as usize].position);
        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        for idx in tri {
            normals[*idx as usize] += n;
        }
    }

    MeshData {
        vertices,
        indices,
        positions4,
        normals4: normals
            .into_iter()
            .map(|n| {
                let n = n.normalize_or_zero();
                [n.x, n.y, n.z, 0.0]
            })
            .collect(),
        triangle_material_ids: vec![0; 8],
        materials: vec![crate::mesh::GpuMaterial {
            base_color: [0.98, 1.0, 1.0, 1.0],
            params: [0.0, 0.015, 1.0, 1.52],
        }],
    }
}

pub fn mesh_bounds(vertices: &[Vertex]) -> (glam::Vec3, glam::Vec3, glam::Vec3, glam::Vec3) {
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

pub fn translate_mesh(mesh: &mut MeshData, offset: glam::Vec3) {
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

pub fn orient_and_scale_mesh(
    mesh: &mut MeshData,
    pivot: glam::Vec3,
    rotation: glam::Quat,
    scale: f32,
) {
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

pub fn append_mesh(base: &mut MeshData, extra: MeshData) {
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

pub fn make_cube_mesh(center: glam::Vec3, half_extent: f32) -> MeshData {
    let corners = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let indices: Vec<u32> = vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5,
        3, 0, 4, 3, 4, 7,
    ];
    let mut vertices = Vec::new();
    let mut positions4 = Vec::new();
    for c in corners {
        let p = center + glam::Vec3::new(c[0], c[1], c[2]) * half_extent;
        vertices.push(Vertex {
            position: p.to_array(),
        });
        positions4.push([p.x, p.y, p.z, 0.0]);
    }
    let mut normals = vec![glam::Vec3::ZERO; vertices.len()];
    for tri in indices.chunks_exact(3) {
        let p0 = glam::Vec3::from(vertices[tri[0] as usize].position);
        let p1 = glam::Vec3::from(vertices[tri[1] as usize].position);
        let p2 = glam::Vec3::from(vertices[tri[2] as usize].position);
        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        for idx in tri {
            normals[*idx as usize] += n;
        }
    }
    MeshData {
        vertices,
        indices,
        positions4,
        normals4: normals
            .into_iter()
            .map(|n| {
                let n = n.normalize_or_zero();
                [n.x, n.y, n.z, 0.0]
            })
            .collect(),
        triangle_material_ids: vec![0; 12],
        materials: vec![crate::mesh::GpuMaterial {
            base_color: [0.98, 1.0, 1.0, 1.0],
            params: [0.0, 0.02, 1.0, 1.52],
        }],
    }
}

pub fn make_plane_mesh(center: glam::Vec3, half_extent: f32) -> MeshData {
    let corners = [
        [-1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ];
    let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];
    let mut vertices = Vec::new();
    let mut positions4 = Vec::new();
    for c in corners {
        let p = center + glam::Vec3::new(c[0], c[1], c[2]) * half_extent;
        vertices.push(Vertex {
            position: p.to_array(),
        });
        positions4.push([p.x, p.y, p.z, 0.0]);
    }
    let normals4 = vec![[0.0, 1.0, 0.0, 0.0]; 4];
    MeshData {
        vertices,
        indices,
        positions4,
        normals4,
        triangle_material_ids: vec![0; 2],
        materials: vec![crate::mesh::GpuMaterial {
            base_color: [0.5, 0.5, 0.5, -1.0],
            params: [0.0, 0.9, 0.0, 1.0],
        }],
    }
}

pub fn append_object_mesh(
    combined: &mut MeshData,
    extra: MeshData,
    object_id: Id,
    mesh_asset_id: u32,
) -> MeshObjectInstance {
    let (pivot, size, _, _) = mesh_bounds(&extra.vertices);
    let vertex_start = combined.positions4.len();
    let vertex_count = extra.positions4.len();
    let index_start = combined.indices.len();
    let index_count = extra.indices.len();
    let material_start = combined.materials.len();
    let material_count = extra.materials.len();
    let base_positions: Vec<glam::Vec3> = extra
        .positions4
        .iter()
        .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
        .collect();
    let base_normals: Vec<glam::Vec3> = extra
        .normals4
        .iter()
        .map(|n| glam::Vec3::new(n[0], n[1], n[2]))
        .collect();
    append_mesh(combined, extra);
    MeshObjectInstance {
        object_id,
        mesh_asset_id,
        vertex_start,
        vertex_count,
        index_start,
        index_count,
        material_start,
        material_count,
        base_positions,
        base_normals,
        pivot,
        max_extent: size.max_element().max(0.1),
        rotation: glam::Quat::IDENTITY,
        translation: glam::Vec3::ZERO,
        scale: glam::Vec3::ONE,
    }
}

pub fn place_instance_center(instance: &mut MeshObjectInstance, center: glam::Vec3) {
    instance.translation = center - instance.pivot;
}

pub fn visible_render_geometry(
    mesh: &MeshData,
    instances: &[MeshObjectInstance],
    visible_ids: &[Id],
) -> (
    Vec<Vertex>,
    Vec<u32>,
    Vec<[f32; 4]>,
    Vec<[f32; 4]>,
    Vec<u32>,
) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut triangle_material_ids = Vec::new();

    for inst in instances {
        if !visible_ids.contains(&inst.object_id) {
            continue;
        }
        let vertex_offset = verts.len() as u32;
        let vertex_end = inst.vertex_start + inst.vertex_count;
        positions.extend_from_slice(&mesh.positions4[inst.vertex_start..vertex_end]);
        normals.extend_from_slice(&mesh.normals4[inst.vertex_start..vertex_end]);
        verts.extend(
            mesh.positions4[inst.vertex_start..vertex_end]
                .iter()
                .map(|p| Vertex {
                    position: [p[0], p[1], p[2]],
                }),
        );

        let index_end = inst.index_start + inst.index_count;
        indices.extend(
            mesh.indices[inst.index_start..index_end]
                .iter()
                .map(|index| index - inst.vertex_start as u32 + vertex_offset),
        );

        let tri_start = inst.index_start / 3;
        let tri_count = inst.index_count / 3;
        let tri_end = tri_start + tri_count;
        triangle_material_ids.extend_from_slice(&mesh.triangle_material_ids[tri_start..tri_end]);
    }

    if verts.is_empty() {
        verts = vec![
            Vertex {
                position: [1.0e6, 1.0e6, 1.0e6],
            },
            Vertex {
                position: [1.0e6 + 1.0, 1.0e6, 1.0e6],
            },
            Vertex {
                position: [1.0e6, 1.0e6 + 1.0, 1.0e6],
            },
        ];
        indices = vec![0, 1, 2];
        positions = verts
            .iter()
            .map(|v| [v.position[0], v.position[1], v.position[2], 0.0])
            .collect();
        normals = vec![[0.0, 1.0, 0.0, 0.0]; 3];
        triangle_material_ids = vec![0];
    }

    (verts, indices, positions, normals, triangle_material_ids)
}

pub fn build_photon_targets(
    instances: &[MeshObjectInstance],
    visible_ids: &[Id],
    include_all: bool,
) -> Vec<PhotonTarget> {
    let mut cumulative_area = 0.0f32;
    let mut targets = Vec::new();
    for inst in instances {
        if !include_all && !visible_ids.contains(&inst.object_id) {
            continue;
        }
        if targets.len() >= MAX_PHOTON_TARGETS {
            break;
        }
        let center = inst.center();
        let radius = (inst.max_extent * inst.scale.max_element() * 0.6).max(0.25);
        cumulative_area += 4.0 * std::f32::consts::PI * radius * radius;
        targets.push(PhotonTarget {
            center_radius: [center.x, center.y, center.z, radius],
            cumulative_area: [cumulative_area, 0.0, 0.0, 0.0],
        });
    }
    targets
}

pub fn sphere_position_for(center: glam::Vec3, size: glam::Vec3, radius: f32) -> glam::Vec3 {
    glam::Vec3::new(center.x + size.x * 0.6 + 2.0, -1.5 + radius, center.z)
}

pub fn update_mesh_transform(
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
