use std::collections::HashMap;

use crate::scene_data::{Id, MainDatabase, Transform as DbTransform};
use crate::prism_file::{
    CollectionData as PrismCollectionData, MaterialData as PrismMaterialData,
    MeshData as PrismMeshData, ObjectData as PrismObjectData, ObjectDataLink as PrismObjectDataLink,
    PrismDatabase, SceneData as PrismSceneData,
};

fn transform_to_matrix(t: &DbTransform) -> [f32; 16] {
    glam::Mat4::from_scale_rotation_translation(t.scale, t.rotation, t.location).to_cols_array()
}

pub fn build_prism_database_from_main(
    main_db: &MainDatabase,
    decanter_scene_id: Id,
    wine_scene_id: Id,
    cornell_scene_id: Id,
    object_material_names: &HashMap<Id, String>,
    material_library: &HashMap<String, PrismMaterialData>,
) -> PrismDatabase {
    let mut out = PrismDatabase::new();

    let mut mesh_map: HashMap<Id, crate::prism_file::MeshHandle> = HashMap::new();
    for (mid, mesh) in &main_db.meshes {
        let h = out.meshes.insert(PrismMeshData {
            vertices: vec![[0.0, 0.0, 0.0]; mesh.vertex_count],
            indices: Vec::new(),
            material_slots: Vec::new(),
        });
        mesh_map.insert(*mid, h);
    }

    let mut material_map: HashMap<String, crate::prism_file::MaterialHandle> = HashMap::new();
    for (name, material) in material_library {
        let h = out.materials.insert(material.clone());
        material_map.insert(name.clone(), h);
    }

    let mut object_map: HashMap<Id, crate::prism_file::ObjectHandle> = HashMap::new();
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

    let mut collection_map: HashMap<Id, crate::prism_file::CollectionHandle> = HashMap::new();
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
