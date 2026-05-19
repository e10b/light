use std::path::Path;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],
    pub params: [f32; 4], // metallic, roughness, transmission, ior
}

pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub positions4: Vec<[f32; 4]>,
    pub normals4: Vec<[f32; 4]>,
    pub triangle_material_ids: Vec<u32>,
    pub materials: Vec<GpuMaterial>,
}

pub fn load_gltf_mesh(path: &Path) -> Result<MeshData, Box<dyn std::error::Error>> {
    let (document, buffers, _images) = gltf::import(path)?;
    let path_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_red_wine_asset = path_name.contains("red_wine");

    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let mut all_positions4 = Vec::new();
    let mut all_normals = Vec::new();
    let mut all_triangle_material_ids = Vec::new();

    let mut materials = Vec::new();
    let mut any_likely_glass = false;
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let mut base_color = pbr.base_color_factor();
        let metallic = pbr.metallic_factor();
        let roughness = pbr.roughness_factor();
        let mat_name = mat.name().unwrap_or("").to_ascii_lowercase();
        let looks_glass = mat_name.contains("glass") || mat_name.contains("decanter");
        let looks_wine = is_red_wine_asset || mat_name.contains("wine");
        if looks_wine {
            base_color = [0.42, 0.015, 0.025, 0.78];
        }
        let transmission = if looks_glass || base_color[3] < 0.99 {
            1.0
        } else {
            0.0
        };
        if transmission > 0.5 {
            any_likely_glass = true;
        }
        let ior = if transmission > 0.0 { 1.52 } else { 1.0 };
        materials.push(GpuMaterial {
            base_color,
            params: [
                metallic,
                roughness.max(if looks_wine { 0.006 } else { 0.02 }),
                transmission,
                if looks_wine { 1.36 } else { ior },
            ],
        });
    }
    if materials.is_empty() {
        materials.push(GpuMaterial {
            base_color: [1.0, 1.0, 1.0, 1.0],
            params: [0.0, 0.03, 1.0, 1.52],
        });
        any_likely_glass = true;
    }
    if !any_likely_glass {
        for m in &mut materials {
            m.params[2] = 1.0;
            m.params[3] = 1.52;
            m.params[1] = m.params[1].min(0.08).max(0.01);
        }
    }

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer_index| Some(&buffers[buffer_index.index()]));

            let start_vertex = all_vertices.len() as u32;
            let mut local_positions: Vec<[f32; 3]> = Vec::new();
            let mut local_indices: Vec<u32> = Vec::new();

            if let Some(iter) = reader.read_positions() {
                for pos in iter {
                    local_positions.push(pos);
                }
            }
            let local_vertex_count = local_positions.len() as u32;
            for pos in &local_positions {
                all_vertices.push(Vertex { position: *pos });
                all_positions4.push([pos[0], pos[1], pos[2], 0.0]);
            }

            let mut local_normals = vec![[0.0f32; 3]; local_positions.len()];
            if let Some(iter) = reader.read_normals() {
                for (i, n) in iter.enumerate() {
                    if i < local_normals.len() {
                        local_normals[i] = n;
                    }
                }
            }

            if let Some(iter) = reader.read_indices() {
                match iter {
                    gltf::mesh::util::ReadIndices::U32(idx_iter) => {
                        for idx in idx_iter {
                            local_indices.push(idx);
                        }
                    }
                    gltf::mesh::util::ReadIndices::U16(idx_iter) => {
                        for idx in idx_iter {
                            local_indices.push(idx as u32);
                        }
                    }
                    gltf::mesh::util::ReadIndices::U8(idx_iter) => {
                        for idx in idx_iter {
                            local_indices.push(idx as u32);
                        }
                    }
                }
            } else {
                for idx in 0..local_vertex_count {
                    local_indices.push(idx);
                }
            }

            if reader.read_normals().is_none() {
                for tri in local_indices.chunks_exact(3) {
                    let i0 = tri[0] as usize;
                    let i1 = tri[1] as usize;
                    let i2 = tri[2] as usize;
                    if i0 < local_positions.len()
                        && i1 < local_positions.len()
                        && i2 < local_positions.len()
                    {
                        let p0 = glam::Vec3::from(local_positions[i0]);
                        let p1 = glam::Vec3::from(local_positions[i1]);
                        let p2 = glam::Vec3::from(local_positions[i2]);
                        let fnorm = (p1 - p0).cross(p2 - p0);
                        if fnorm.length_squared() > 1e-20 {
                            let n = fnorm.normalize();
                            for idx in [i0, i1, i2] {
                                let old = glam::Vec3::from(local_normals[idx]);
                                let sum = old + n;
                                local_normals[idx] = sum.to_array();
                            }
                        }
                    }
                }
            }
            for n in &local_normals {
                let nn = glam::Vec3::from(*n).normalize_or_zero();
                all_normals.push([nn.x, nn.y, nn.z, 0.0]);
            }

            let mat_id = primitive.material().index().unwrap_or(0) as u32;
            for tri in local_indices.chunks_exact(3) {
                all_indices.push(start_vertex + tri[0]);
                all_indices.push(start_vertex + tri[1]);
                all_indices.push(start_vertex + tri[2]);
                all_triangle_material_ids.push(mat_id);
            }
        }
    }

    Ok(MeshData {
        vertices: all_vertices,
        indices: all_indices,
        positions4: all_positions4,
        normals4: all_normals,
        triangle_material_ids: all_triangle_material_ids,
        materials,
    })
}
