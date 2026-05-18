use crate::{application::{geometry::make_cube_mesh, types::MeshObjectInstance}, scene_data::Id};

#[allow(clippy::too_many_arguments)]
pub fn maybe_build_stress_scene(
    stress_test_requested: &mut bool,
    mesh: &mut crate::mesh::MeshData,
    model_idx: &mut Vec<u32>,
    mesh_instances: &mut Vec<MeshObjectInstance>,
    stress_instance_count: &mut usize,
    gpu_mesh_dirty: &mut bool,
    geometry_dirty: &mut bool,
    accumulation_dirty: &mut bool,
    project_status: &mut String,
    mesh_asset_count: usize,
) {
    if !*stress_test_requested {
        return;
    }

    *stress_test_requested = false;
    let cube_mesh = make_cube_mesh(glam::Vec3::ZERO, 1.5);
    *mesh = cube_mesh;
    *model_idx = mesh.indices.clone();
    mesh_instances.clear();
    let side = 1000usize;
    let total = side * side;
    let spacing = 3.2f32;
    let half = side as f32 * 0.5;
    for i in 0..total {
        let x = (i % side) as f32;
        let z = (i / side) as f32;
        let center = glam::Vec3::new((x - half) * spacing, 0.0, (z - half) * spacing);
        mesh_instances.push(MeshObjectInstance {
            object_id: Id(10_000_000 + i as u64),
            mesh_asset_id: 0,
            vertex_start: 0,
            vertex_count: mesh.positions4.len(),
            index_start: 0,
            index_count: mesh.indices.len(),
            material_start: 0,
            material_count: mesh.materials.len(),
            base_positions: mesh
                .positions4
                .iter()
                .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
                .collect(),
            base_normals: mesh
                .normals4
                .iter()
                .map(|n| glam::Vec3::new(n[0], n[1], n[2]))
                .collect(),
            pivot: glam::Vec3::ZERO,
            max_extent: 3.0,
            rotation: glam::Quat::IDENTITY,
            translation: center,
            scale: glam::Vec3::ONE,
        });
    }
    *stress_instance_count = mesh_instances.len();
    *gpu_mesh_dirty = true;
    *geometry_dirty = true;
    *accumulation_dirty = true;
    *project_status = format!(
        "Stress scene armed: {} cubes (TLAS instances), {} mesh assets",
        *stress_instance_count,
        mesh_asset_count
    );
}
