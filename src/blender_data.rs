#![allow(dead_code)]

use std::collections::HashMap;

use glam::{Quat, Vec3};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Id(pub u64);

#[derive(Clone, Debug)]
pub struct Transform {
    pub location: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MeshDataBlock {
    pub id: Id,
    pub name: String,
    pub vertex_count: usize,
    pub user_count: u32,
}

#[derive(Clone, Debug)]
pub struct ObjectDataBlock {
    pub id: Id,
    pub name: String,
    pub transform: Transform,
    pub mesh_id: Option<Id>,
}

#[derive(Clone, Debug)]
pub struct CollectionDataBlock {
    pub id: Id,
    pub name: String,
    pub object_ids: Vec<Id>,
    pub child_collection_ids: Vec<Id>,
}

#[derive(Clone, Debug)]
pub struct SceneDataBlock {
    pub id: Id,
    pub name: String,
    pub master_collection_id: Id,
    pub view_layer_id: Id,
}

#[derive(Clone, Debug)]
pub struct BaseData {
    pub object_id: Id,
    pub visible: bool,
    pub selectable: bool,
}

#[derive(Clone, Debug)]
pub struct ViewLayerDataBlock {
    pub id: Id,
    pub name: String,
    pub bases: Vec<BaseData>,
}

#[derive(Default)]
pub struct MainDatabase {
    next_id: u64,
    pub meshes: HashMap<Id, MeshDataBlock>,
    pub objects: HashMap<Id, ObjectDataBlock>,
    pub collections: HashMap<Id, CollectionDataBlock>,
    pub scenes: HashMap<Id, SceneDataBlock>,
    pub view_layers: HashMap<Id, ViewLayerDataBlock>,
}

impl MainDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_id(&mut self) -> Id {
        self.next_id += 1;
        Id(self.next_id)
    }

    pub fn create_mesh(&mut self, name: impl Into<String>, vertex_count: usize) -> Id {
        let id = self.alloc_id();
        self.meshes.insert(
            id,
            MeshDataBlock {
                id,
                name: name.into(),
                vertex_count,
                user_count: 0,
            },
        );
        id
    }

    pub fn create_object(
        &mut self,
        name: impl Into<String>,
        mesh_id: Option<Id>,
        transform: Transform,
    ) -> Id {
        let id = self.alloc_id();
        if let Some(mid) = mesh_id {
            if let Some(mesh) = self.meshes.get_mut(&mid) {
                mesh.user_count = mesh.user_count.saturating_add(1);
            }
        }
        self.objects.insert(
            id,
            ObjectDataBlock {
                id,
                name: name.into(),
                transform,
                mesh_id,
            },
        );
        id
    }

    pub fn create_collection(&mut self, name: impl Into<String>) -> Id {
        let id = self.alloc_id();
        self.collections.insert(
            id,
            CollectionDataBlock {
                id,
                name: name.into(),
                object_ids: Vec::new(),
                child_collection_ids: Vec::new(),
            },
        );
        id
    }

    pub fn create_scene(&mut self, name: impl Into<String>, master_collection_id: Id) -> Id {
        let view_layer_id = self.create_view_layer("ViewLayer");
        let id = self.alloc_id();
        self.scenes.insert(
            id,
            SceneDataBlock {
                id,
                name: name.into(),
                master_collection_id,
                view_layer_id,
            },
        );
        id
    }

    pub fn create_view_layer(&mut self, name: impl Into<String>) -> Id {
        let id = self.alloc_id();
        self.view_layers.insert(
            id,
            ViewLayerDataBlock {
                id,
                name: name.into(),
                bases: Vec::new(),
            },
        );
        id
    }

    pub fn collection_link_object(&mut self, collection_id: Id, object_id: Id) {
        if let Some(col) = self.collections.get_mut(&collection_id) {
            if !col.object_ids.contains(&object_id) {
                col.object_ids.push(object_id);
            }
        }
    }

    pub fn collection_link_child(&mut self, parent_collection_id: Id, child_collection_id: Id) {
        if let Some(col) = self.collections.get_mut(&parent_collection_id) {
            if !col.child_collection_ids.contains(&child_collection_id) {
                col.child_collection_ids.push(child_collection_id);
            }
        }
    }

    pub fn scene_objects_recursive(&self, scene_id: Id) -> Vec<Id> {
        fn walk(db: &MainDatabase, cid: Id, out: &mut Vec<Id>) {
            if let Some(col) = db.collections.get(&cid) {
                out.extend(col.object_ids.iter().copied());
                for child in &col.child_collection_ids {
                    walk(db, *child, out);
                }
            }
        }
        let mut out = Vec::new();
        if let Some(scene) = self.scenes.get(&scene_id) {
            walk(self, scene.master_collection_id, &mut out);
        }
        out
    }

    pub fn ensure_scene_base(&mut self, scene_id: Id, object_id: Id, visible: bool, selectable: bool) {
        if let Some(scene) = self.scenes.get(&scene_id) {
            if let Some(vl) = self.view_layers.get_mut(&scene.view_layer_id) {
                if !vl.bases.iter().any(|b| b.object_id == object_id) {
                    vl.bases.push(BaseData {
                        object_id,
                        visible,
                        selectable,
                    });
                }
            }
        }
    }

    pub fn set_scene_base_visibility(&mut self, scene_id: Id, object_id: Id, visible: bool) {
        if let Some(scene) = self.scenes.get(&scene_id) {
            if let Some(vl) = self.view_layers.get_mut(&scene.view_layer_id) {
                if let Some(base) = vl.bases.iter_mut().find(|b| b.object_id == object_id) {
                    base.visible = visible;
                }
            }
        }
    }

    pub fn set_scene_base_selectable(&mut self, scene_id: Id, object_id: Id, selectable: bool) {
        if let Some(scene) = self.scenes.get(&scene_id) {
            if let Some(vl) = self.view_layers.get_mut(&scene.view_layer_id) {
                if let Some(base) = vl.bases.iter_mut().find(|b| b.object_id == object_id) {
                    base.selectable = selectable;
                }
            }
        }
    }

    pub fn scene_visible_selectable_objects(&self, scene_id: Id) -> Vec<Id> {
        let allowed = self.scene_objects_recursive(scene_id);
        let mut out = Vec::new();
        if let Some(scene) = self.scenes.get(&scene_id) {
            if let Some(vl) = self.view_layers.get(&scene.view_layer_id) {
                for base in &vl.bases {
                    if base.visible && base.selectable && allowed.contains(&base.object_id) {
                        out.push(base.object_id);
                    }
                }
            }
        }
        out
    }
}
