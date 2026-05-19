#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use petgraph::graph::DiGraph;
use serde::{Deserialize, Serialize};
use slotmap::{new_key_type, SlotMap};

new_key_type! {
    pub struct ObjectHandle;
    pub struct MeshHandle;
    pub struct MaterialHandle;
    pub struct SceneHandle;
    pub struct CollectionHandle;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ObjectData {
    pub name: String,
    pub transform_matrix: [f32; 16],
    pub data_link: ObjectDataLink,
    pub material_link: Option<MaterialHandle>,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ObjectDataLink {
    Mesh(MeshHandle),
    Camera,
    Light,
    None,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MeshData {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub material_slots: Vec<Option<MaterialHandle>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MaterialData {
    pub name: String,
    pub graph: DiGraph<ShaderNode, NodeLink>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ShaderNode {
    pub node_type: NodeType,
    pub properties: NodeProperties,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NodeLink {
    pub output_socket: String,
    pub input_socket: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeType {
    FloatInput,
    VectorMath,
    PrincipledBSDF,
    MaterialOutput,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct NodeProperties {
    pub float_value: Option<f32>,
    pub vec3_value: Option<[f32; 3]>,
    pub roughness: Option<f32>,
    pub transmission: Option<f32>,
    pub ior: Option<f32>,
    pub bsdf_connected: Option<bool>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CollectionData {
    pub name: String,
    pub objects: Vec<ObjectHandle>,
    pub children: Vec<CollectionHandle>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SceneData {
    pub name: String,
    pub master_collection: CollectionHandle,
}

pub struct PrismDatabase {
    pub objects: SlotMap<ObjectHandle, ObjectData>,
    pub meshes: SlotMap<MeshHandle, MeshData>,
    pub materials: SlotMap<MaterialHandle, MaterialData>,
    pub scenes: SlotMap<SceneHandle, SceneData>,
    pub collections: SlotMap<CollectionHandle, CollectionData>,
}

impl PrismDatabase {
    pub fn new() -> Self {
        Self {
            objects: SlotMap::with_key(),
            meshes: SlotMap::with_key(),
            materials: SlotMap::with_key(),
            scenes: SlotMap::with_key(),
            collections: SlotMap::with_key(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct PrismFileBlock {
    pub block_type: String,
    pub data_payload: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct IndexEntry {
    key: String,
    offset: u64,
    size: u64,
}

#[derive(Serialize, Deserialize)]
struct StableObjectData {
    id: u32,
    name: String,
    transform_matrix: [f32; 16],
    data_link: StableObjectDataLink,
    material_link: Option<u32>,
}

#[derive(Serialize, Deserialize)]
enum StableObjectDataLink {
    Mesh(u32),
    Camera,
    Light,
    None,
}

#[derive(Serialize, Deserialize)]
struct StableMeshData {
    id: u32,
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
    material_slots: Vec<Option<u32>>,
}

#[derive(Serialize, Deserialize)]
struct StableMaterialData {
    id: u32,
    name: String,
    graph: DiGraph<ShaderNode, NodeLink>,
}

#[derive(Serialize, Deserialize)]
struct StableCollectionData {
    id: u32,
    name: String,
    objects: Vec<u32>,
    children: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct StableSceneData {
    id: u32,
    name: String,
    master_collection: u32,
}

#[derive(Default)]
struct SaveContext {
    object_ids: HashMap<ObjectHandle, u32>,
    mesh_ids: HashMap<MeshHandle, u32>,
    material_ids: HashMap<MaterialHandle, u32>,
    scene_ids: HashMap<SceneHandle, u32>,
    collection_ids: HashMap<CollectionHandle, u32>,
}

fn build_save_context(db: &PrismDatabase) -> SaveContext {
    let mut ctx = SaveContext::default();
    for (i, (h, _)) in db.objects.iter().enumerate() {
        ctx.object_ids.insert(h, i as u32);
    }
    for (i, (h, _)) in db.meshes.iter().enumerate() {
        ctx.mesh_ids.insert(h, i as u32);
    }
    for (i, (h, _)) in db.materials.iter().enumerate() {
        ctx.material_ids.insert(h, i as u32);
    }
    for (i, (h, _)) in db.collections.iter().enumerate() {
        ctx.collection_ids.insert(h, i as u32);
    }
    for (i, (h, _)) in db.scenes.iter().enumerate() {
        ctx.scene_ids.insert(h, i as u32);
    }
    ctx
}

pub fn save_prism_file(path: &Path, db: &PrismDatabase, compress: bool) -> std::io::Result<()> {
    let ctx = build_save_context(db);
    let mut blocks: Vec<PrismFileBlock> = Vec::new();

    let stable_meshes: Vec<StableMeshData> = db
        .meshes
        .iter()
        .map(|(h, m)| StableMeshData {
            id: ctx.mesh_ids[&h],
            vertices: m.vertices.clone(),
            indices: m.indices.clone(),
            material_slots: m
                .material_slots
                .iter()
                .map(|slot| slot.and_then(|mh| ctx.material_ids.get(&mh).copied()))
                .collect(),
        })
        .collect();
    blocks.push(PrismFileBlock {
        block_type: "MESH".to_string(),
        data_payload: bincode::serialize(&stable_meshes).expect("serialize meshes"),
    });

    let stable_materials: Vec<StableMaterialData> = db
        .materials
        .iter()
        .map(|(h, m)| StableMaterialData {
            id: ctx.material_ids[&h],
            name: m.name.clone(),
            graph: m.graph.clone(),
        })
        .collect();
    blocks.push(PrismFileBlock {
        block_type: "MATR".to_string(),
        data_payload: bincode::serialize(&stable_materials).expect("serialize materials"),
    });

    let stable_objects: Vec<StableObjectData> = db
        .objects
        .iter()
        .map(|(h, o)| StableObjectData {
            id: ctx.object_ids[&h],
            name: o.name.clone(),
            transform_matrix: o.transform_matrix,
            data_link: match o.data_link {
                ObjectDataLink::Mesh(mh) => StableObjectDataLink::Mesh(ctx.mesh_ids[&mh]),
                ObjectDataLink::Camera => StableObjectDataLink::Camera,
                ObjectDataLink::Light => StableObjectDataLink::Light,
                ObjectDataLink::None => StableObjectDataLink::None,
            },
            material_link: o
                .material_link
                .and_then(|mh| ctx.material_ids.get(&mh).copied()),
        })
        .collect();
    blocks.push(PrismFileBlock {
        block_type: "OBJD".to_string(),
        data_payload: bincode::serialize(&stable_objects).expect("serialize objects"),
    });

    let stable_collections: Vec<StableCollectionData> = db
        .collections
        .iter()
        .map(|(h, c)| StableCollectionData {
            id: ctx.collection_ids[&h],
            name: c.name.clone(),
            objects: c.objects.iter().map(|oh| ctx.object_ids[oh]).collect(),
            children: c.children.iter().map(|ch| ctx.collection_ids[ch]).collect(),
        })
        .collect();
    blocks.push(PrismFileBlock {
        block_type: "COLL".to_string(),
        data_payload: bincode::serialize(&stable_collections).expect("serialize collections"),
    });

    let stable_scenes: Vec<StableSceneData> = db
        .scenes
        .iter()
        .map(|(h, s)| StableSceneData {
            id: ctx.scene_ids[&h],
            name: s.name.clone(),
            master_collection: ctx.collection_ids[&s.master_collection],
        })
        .collect();
    blocks.push(PrismFileBlock {
        block_type: "SCEN".to_string(),
        data_payload: bincode::serialize(&stable_scenes).expect("serialize scenes"),
    });

    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(b"PRISM");
    raw.extend_from_slice(b"010");
    raw.push(0x01);

    let mut index: Vec<IndexEntry> = Vec::new();
    for block in &blocks {
        let encoded = bincode::serialize(block).expect("serialize block");
        let offset = raw.len() as u64;
        raw.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        raw.extend_from_slice(&encoded);
        index.push(IndexEntry {
            key: block.block_type.clone(),
            offset,
            size: encoded.len() as u64,
        });
    }

    let index_bytes = bincode::serialize(&index).expect("serialize index");
    raw.extend_from_slice(&index_bytes);
    raw.extend_from_slice(&(index_bytes.len() as u64).to_le_bytes());

    if compress {
        let mut out = File::create(path)?;
        let mut encoder = zstd::Encoder::new(&mut out, 3)?;
        encoder.write_all(&raw)?;
        encoder.finish()?;
        Ok(())
    } else {
        let mut out = File::create(path)?;
        out.write_all(&raw)
    }
}

pub fn load_prism_file(path: &Path, compressed: bool) -> std::io::Result<Vec<PrismFileBlock>> {
    let mut bytes = Vec::new();
    if compressed {
        let file = File::open(path)?;
        let mut decoder = zstd::Decoder::new(file)?;
        decoder.read_to_end(&mut bytes)?;
    } else {
        let mut file = File::open(path)?;
        file.read_to_end(&mut bytes)?;
    }

    if bytes.len() < 9 || &bytes[0..5] != b"PRISM" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid .prism magic",
        ));
    }

    let mut cursor = std::io::Cursor::new(bytes);
    cursor.seek(SeekFrom::End(-8))?;
    let mut len_buf = [0u8; 8];
    cursor.read_exact(&mut len_buf)?;
    let index_len = u64::from_le_bytes(len_buf);
    cursor.seek(SeekFrom::End(-(8 + index_len as i64)))?;
    let mut index_bytes = vec![0u8; index_len as usize];
    cursor.read_exact(&mut index_bytes)?;
    let index: Vec<IndexEntry> = bincode::deserialize(&index_bytes).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("index decode: {e}"))
    })?;

    let data = cursor.into_inner();
    let mut blocks = Vec::new();
    for entry in index {
        let start = entry.offset as usize;
        let len_start = start;
        let data_start = len_start + 4;
        let data_end = data_start + entry.size as usize;
        if data_end > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "block out of bounds",
            ));
        }
        let block: PrismFileBlock =
            bincode::deserialize(&data[data_start..data_end]).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("block decode: {e}"))
            })?;
        blocks.push(block);
    }
    Ok(blocks)
}

pub fn load_prism_database(path: &Path, compressed: bool) -> std::io::Result<PrismDatabase> {
    let blocks = load_prism_file(path, compressed)?;
    let mut db = PrismDatabase::new();

    let mut mesh_by_id: HashMap<u32, MeshHandle> = HashMap::new();
    let mut mat_by_id: HashMap<u32, MaterialHandle> = HashMap::new();
    let mut obj_by_id: HashMap<u32, ObjectHandle> = HashMap::new();
    let mut col_by_id: HashMap<u32, CollectionHandle> = HashMap::new();
    let mut scene_by_id: HashMap<u32, SceneHandle> = HashMap::new();

    let mut stable_meshes: Vec<StableMeshData> = Vec::new();
    let mut stable_materials: Vec<StableMaterialData> = Vec::new();
    let mut stable_objects: Vec<StableObjectData> = Vec::new();
    let mut stable_collections: Vec<StableCollectionData> = Vec::new();
    let mut stable_scenes: Vec<StableSceneData> = Vec::new();

    for block in blocks {
        match block.block_type.as_str() {
            "MESH" => {
                stable_meshes = bincode::deserialize(&block.data_payload).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("MESH decode: {e}"))
                })?;
            }
            "MATR" => {
                stable_materials = bincode::deserialize(&block.data_payload).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("MATR decode: {e}"))
                })?;
            }
            "OBJD" => {
                stable_objects = bincode::deserialize(&block.data_payload).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("OBJD decode: {e}"))
                })?;
            }
            "COLL" => {
                stable_collections = bincode::deserialize(&block.data_payload).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("COLL decode: {e}"))
                })?;
            }
            "SCEN" => {
                stable_scenes = bincode::deserialize(&block.data_payload).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("SCEN decode: {e}"))
                })?;
            }
            _ => {}
        }
    }

    for m in &stable_meshes {
        let h = db.meshes.insert(MeshData {
            vertices: m.vertices.clone(),
            indices: m.indices.clone(),
            material_slots: Vec::new(),
        });
        mesh_by_id.insert(m.id, h);
    }

    for m in &stable_materials {
        let h = db.materials.insert(MaterialData {
            name: m.name.clone(),
            graph: m.graph.clone(),
        });
        mat_by_id.insert(m.id, h);
    }

    for m in &stable_meshes {
        if let Some(mh) = mesh_by_id.get(&m.id).copied() {
            if let Some(mesh) = db.meshes.get_mut(mh) {
                mesh.material_slots = m
                    .material_slots
                    .iter()
                    .map(|slot| slot.and_then(|mid| mat_by_id.get(&mid).copied()))
                    .collect();
            }
        }
    }

    for o in &stable_objects {
        let link = match o.data_link {
            StableObjectDataLink::Mesh(mid) => mesh_by_id
                .get(&mid)
                .copied()
                .map(ObjectDataLink::Mesh)
                .unwrap_or(ObjectDataLink::None),
            StableObjectDataLink::Camera => ObjectDataLink::Camera,
            StableObjectDataLink::Light => ObjectDataLink::Light,
            StableObjectDataLink::None => ObjectDataLink::None,
        };
        let h = db.objects.insert(ObjectData {
            name: o.name.clone(),
            transform_matrix: o.transform_matrix,
            data_link: link,
            material_link: o.material_link.and_then(|mid| mat_by_id.get(&mid).copied()),
        });
        obj_by_id.insert(o.id, h);
    }

    for c in &stable_collections {
        let h = db.collections.insert(CollectionData {
            name: c.name.clone(),
            objects: Vec::new(),
            children: Vec::new(),
        });
        col_by_id.insert(c.id, h);
    }

    for c in &stable_collections {
        if let Some(ch) = col_by_id.get(&c.id).copied() {
            if let Some(col) = db.collections.get_mut(ch) {
                col.objects = c
                    .objects
                    .iter()
                    .filter_map(|id| obj_by_id.get(id).copied())
                    .collect();
                col.children = c
                    .children
                    .iter()
                    .filter_map(|id| col_by_id.get(id).copied())
                    .collect();
            }
        }
    }

    for s in &stable_scenes {
        if let Some(master) = col_by_id.get(&s.master_collection).copied() {
            let sh = db.scenes.insert(SceneData {
                name: s.name.clone(),
                master_collection: master,
            });
            scene_by_id.insert(s.id, sh);
        }
    }

    Ok(db)
}
