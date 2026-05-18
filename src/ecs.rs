#![allow(dead_code)]

use std::{
    cell::RefCell,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
};

use glam::{Quat, Vec3};
use mlua::{AnyUserData, Lua, Table, UserData, UserDataMethods, Value};

use crate::blender_data::Id;

#[derive(Clone, Debug)]
pub struct NameComponent {
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct TransformComponent {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for TransformComponent {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GlobalTransformComponent {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for GlobalTransformComponent {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl GlobalTransformComponent {
    fn from_transform(transform: &TransformComponent) -> Self {
        Self {
            translation: transform.translation,
            rotation: transform.rotation,
            scale: transform.scale,
        }
    }

    fn combine(parent: &GlobalTransformComponent, child: &TransformComponent) -> Self {
        let scaled = Vec3::new(
            child.translation.x * parent.scale.x,
            child.translation.y * parent.scale.y,
            child.translation.z * parent.scale.z,
        );
        Self {
            translation: parent.translation + parent.rotation * scaled,
            rotation: parent.rotation * child.rotation,
            scale: parent.scale * child.scale,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Visibility {
    Visible,
    Hidden,
}

impl Default for Visibility {
    fn default() -> Self {
        Self::Visible
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct InheritedVisibility {
    pub visible: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ViewVisibility {
    pub visible: bool,
}

#[derive(Copy, Clone, Debug)]
pub struct ParentComponent {
    pub parent: Id,
}

#[derive(Clone, Debug)]
pub struct MeshComponent {
    pub mesh_asset_id: u32,
}

#[derive(Clone, Debug)]
pub struct CameraComponent {
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    pub active: bool,
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self {
            fov_y_radians: std::f32::consts::FRAC_PI_3,
            near: 0.1,
            far: 1000.0,
            active: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhysicsComponent {
    pub velocity: Vec3,
    pub mass: f32,
    pub dynamic: bool,
}

impl Default for PhysicsComponent {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            mass: 1.0,
            dynamic: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LightComponent {
    pub intensity: f32,
}

#[derive(Clone, Debug)]
pub struct ScriptComponent {
    pub path: String,
    pub enabled: bool,
    pub started: bool,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct EntityRecord {
    pub id: Id,
}

#[derive(Default)]
pub struct World {
    next_detached_id: u64,
    pub entities: Vec<EntityRecord>,
    pub names: HashMap<Id, NameComponent>,
    pub transforms: HashMap<Id, TransformComponent>,
    pub global_transforms: HashMap<Id, GlobalTransformComponent>,
    pub parents: HashMap<Id, ParentComponent>,
    pub visibility: HashMap<Id, Visibility>,
    pub inherited_visibility: HashMap<Id, InheritedVisibility>,
    pub view_visibility: HashMap<Id, ViewVisibility>,
    pub meshes: HashMap<Id, MeshComponent>,
    pub cameras: HashMap<Id, CameraComponent>,
    pub physics: HashMap<Id, PhysicsComponent>,
    pub lights: HashMap<Id, LightComponent>,
    pub scripts: HashMap<Id, ScriptComponent>,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn(&mut self, name: impl Into<String>) -> Id {
        self.next_detached_id = self.next_detached_id.max(100_000);
        self.next_detached_id += 1;
        let id = Id(self.next_detached_id);
        self.register_existing(id, name, TransformComponent::default());
        id
    }

    pub fn register_existing(
        &mut self,
        id: Id,
        name: impl Into<String>,
        transform: TransformComponent,
    ) {
        if !self.entities.iter().any(|entity| entity.id == id) {
            self.entities.push(EntityRecord { id });
        }
        self.names.insert(id, NameComponent { name: name.into() });
        self.global_transforms
            .insert(id, GlobalTransformComponent::from_transform(&transform));
        self.transforms.insert(id, transform);
        self.visibility.entry(id).or_insert(Visibility::Visible);
        self.inherited_visibility
            .entry(id)
            .or_insert(InheritedVisibility { visible: true });
        self.view_visibility
            .entry(id)
            .or_insert(ViewVisibility { visible: true });
    }

    pub fn despawn(&mut self, id: Id) {
        self.entities.retain(|entity| entity.id != id);
        self.names.remove(&id);
        self.transforms.remove(&id);
        self.global_transforms.remove(&id);
        self.parents.remove(&id);
        self.parents.retain(|_, parent| parent.parent != id);
        self.visibility.remove(&id);
        self.inherited_visibility.remove(&id);
        self.view_visibility.remove(&id);
        self.meshes.remove(&id);
        self.cameras.remove(&id);
        self.physics.remove(&id);
        self.lights.remove(&id);
        self.scripts.remove(&id);
    }

    pub fn attach_mesh(&mut self, id: Id, mesh_asset_id: u32) {
        self.meshes.insert(id, MeshComponent { mesh_asset_id });
    }

    pub fn detach_mesh(&mut self, id: Id) {
        self.meshes.remove(&id);
    }

    pub fn attach_camera(&mut self, id: Id, camera: CameraComponent) {
        self.cameras.insert(id, camera);
    }

    pub fn attach_physics(&mut self, id: Id, physics: PhysicsComponent) {
        self.physics.insert(id, physics);
    }

    pub fn attach_light(&mut self, id: Id, intensity: f32) {
        self.lights.insert(id, LightComponent { intensity });
    }

    pub fn attach_script(&mut self, id: Id, path: impl Into<String>) {
        self.scripts.insert(
            id,
            ScriptComponent {
                path: path.into(),
                enabled: true,
                started: false,
                last_error: None,
            },
        );
    }

    pub fn set_parent(&mut self, id: Id, parent: Id) {
        if id != parent && self.entities.iter().any(|entity| entity.id == parent) {
            self.parents.insert(id, ParentComponent { parent });
        }
    }

    pub fn clear_parent(&mut self, id: Id) {
        self.parents.remove(&id);
    }

    pub fn set_visible(&mut self, id: Id, visible: bool) {
        self.visibility.insert(
            id,
            if visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
        );
    }

    pub fn is_visible(&self, id: Id) -> bool {
        self.visibility.get(&id).copied().unwrap_or_default() == Visibility::Visible
            && self
                .inherited_visibility
                .get(&id)
                .map(|v| v.visible)
                .unwrap_or(true)
            && self
                .view_visibility
                .get(&id)
                .map(|v| v.visible)
                .unwrap_or(true)
    }

    pub fn script_status(&self, id: Id) -> &'static str {
        match self.scripts.get(&id) {
            Some(script) if !script.enabled => "disabled",
            Some(script) if script.last_error.is_some() => "error",
            Some(_) => "ready",
            None => "none",
        }
    }

    pub fn integrate_physics(&mut self, dt: f32) {
        let ids: Vec<Id> = self.physics.keys().copied().collect();
        for id in ids {
            let Some(physics) = self.physics.get(&id).cloned() else {
                continue;
            };
            if !physics.dynamic {
                continue;
            }
            if let Some(transform) = self.transforms.get_mut(&id) {
                transform.translation += physics.velocity * dt;
            }
        }
    }

    pub fn update_global_transforms_and_visibility(&mut self) {
        let ids: Vec<Id> = self.entities.iter().map(|entity| entity.id).collect();
        for id in &ids {
            let global = self
                .transforms
                .get(id)
                .map(GlobalTransformComponent::from_transform)
                .unwrap_or_default();
            self.global_transforms.insert(*id, global);
            let local_visible = self.visibility.get(id).copied().unwrap_or_default();
            self.inherited_visibility.insert(
                *id,
                InheritedVisibility {
                    visible: local_visible == Visibility::Visible,
                },
            );
        }

        for _ in 0..ids.len() {
            for id in &ids {
                let Some(parent_id) = self.parents.get(id).map(|p| p.parent) else {
                    continue;
                };
                let Some(parent_global) = self.global_transforms.get(&parent_id).cloned() else {
                    continue;
                };
                let Some(local) = self.transforms.get(id) else {
                    continue;
                };
                self.global_transforms.insert(
                    *id,
                    GlobalTransformComponent::combine(&parent_global, local),
                );

                let parent_visible = self
                    .inherited_visibility
                    .get(&parent_id)
                    .map(|v| v.visible)
                    .unwrap_or(true);
                let local_visible = self.visibility.get(id).copied().unwrap_or_default();
                self.inherited_visibility.insert(
                    *id,
                    InheritedVisibility {
                        visible: parent_visible && local_visible == Visibility::Visible,
                    },
                );
            }
        }

        for id in ids {
            let inherited = self
                .inherited_visibility
                .get(&id)
                .map(|v| v.visible)
                .unwrap_or(true);
            self.view_visibility
                .insert(id, ViewVisibility { visible: inherited });
        }
    }

    pub fn entity_ids_with_scripts(&self) -> Vec<Id> {
        self.scripts
            .iter()
            .filter_map(|(id, script)| script.enabled.then_some(*id))
            .collect()
    }
}

#[derive(Clone)]
struct LuaEntity {
    id: Id,
    world: Rc<RefCell<World>>,
}

impl LuaEntity {
    fn vec3_table(lua: &Lua, v: Vec3) -> mlua::Result<Table> {
        let table = lua.create_table()?;
        table.set("x", v.x)?;
        table.set("y", v.y)?;
        table.set("z", v.z)?;
        Ok(table)
    }
}

impl UserData for LuaEntity {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("id", |_, this, ()| Ok(this.id.0));
        methods.add_method("name", |_, this, ()| {
            Ok(this
                .world
                .borrow()
                .names
                .get(&this.id)
                .map(|n| n.name.clone())
                .unwrap_or_else(|| format!("Entity {}", this.id.0)))
        });
        methods.add_method_mut("set_name", |_, this, name: String| {
            this.world
                .borrow_mut()
                .names
                .insert(this.id, NameComponent { name });
            Ok(())
        });
        methods.add_method("position", |lua, this, ()| {
            let pos = this
                .world
                .borrow()
                .transforms
                .get(&this.id)
                .map(|t| t.translation)
                .unwrap_or(Vec3::ZERO);
            LuaEntity::vec3_table(lua, pos)
        });
        methods.add_method_mut("set_position", |_, this, (x, y, z): (f32, f32, f32)| {
            this.world
                .borrow_mut()
                .transforms
                .entry(this.id)
                .or_default()
                .translation = Vec3::new(x, y, z);
            Ok(())
        });
        methods.add_method_mut("translate", |_, this, (x, y, z): (f32, f32, f32)| {
            this.world
                .borrow_mut()
                .transforms
                .entry(this.id)
                .or_default()
                .translation += Vec3::new(x, y, z);
            Ok(())
        });
        methods.add_method("scale", |lua, this, ()| {
            let scale = this
                .world
                .borrow()
                .transforms
                .get(&this.id)
                .map(|t| t.scale)
                .unwrap_or(Vec3::ONE);
            LuaEntity::vec3_table(lua, scale)
        });
        methods.add_method_mut("set_scale", |_, this, (x, y, z): (f32, f32, f32)| {
            this.world
                .borrow_mut()
                .transforms
                .entry(this.id)
                .or_default()
                .scale = Vec3::new(x, y, z);
            Ok(())
        });
        methods.add_method_mut(
            "set_rotation_euler",
            |_, this, (x, y, z): (f32, f32, f32)| {
                this.world
                    .borrow_mut()
                    .transforms
                    .entry(this.id)
                    .or_default()
                    .rotation = Quat::from_euler(glam::EulerRot::XYZ, x, y, z);
                Ok(())
            },
        );
        methods.add_method_mut("rotate_y", |_, this, radians: f32| {
            let mut world = this.world.borrow_mut();
            let transform = world.transforms.entry(this.id).or_default();
            transform.rotation *= Quat::from_rotation_y(radians);
            Ok(())
        });
        methods.add_method("is_visible", |_, this, ()| {
            Ok(this.world.borrow().is_visible(this.id))
        });
        methods.add_method_mut("set_visible", |_, this, visible: bool| {
            this.world.borrow_mut().set_visible(this.id, visible);
            Ok(())
        });
        methods.add_method_mut("show", |_, this, ()| {
            this.world.borrow_mut().set_visible(this.id, true);
            Ok(())
        });
        methods.add_method_mut("hide", |_, this, ()| {
            this.world.borrow_mut().set_visible(this.id, false);
            Ok(())
        });
        methods.add_method("has_mesh", |_, this, ()| {
            Ok(this.world.borrow().meshes.contains_key(&this.id))
        });
        methods.add_method_mut("attach_mesh", |_, this, mesh_asset_id: u32| {
            this.world.borrow_mut().attach_mesh(this.id, mesh_asset_id);
            Ok(())
        });
        methods.add_method_mut("detach_mesh", |_, this, ()| {
            this.world.borrow_mut().detach_mesh(this.id);
            Ok(())
        });
        methods.add_method("has_camera", |_, this, ()| {
            Ok(this.world.borrow().cameras.contains_key(&this.id))
        });
        methods.add_method_mut("attach_camera", |_, this, ()| {
            this.world
                .borrow_mut()
                .attach_camera(this.id, CameraComponent::default());
            Ok(())
        });
        methods.add_method("has_physics", |_, this, ()| {
            Ok(this.world.borrow().physics.contains_key(&this.id))
        });
        methods.add_method_mut("attach_physics", |_, this, ()| {
            this.world
                .borrow_mut()
                .attach_physics(this.id, PhysicsComponent::default());
            Ok(())
        });
        methods.add_method_mut("set_velocity", |_, this, (x, y, z): (f32, f32, f32)| {
            this.world
                .borrow_mut()
                .physics
                .entry(this.id)
                .or_default()
                .velocity = Vec3::new(x, y, z);
            Ok(())
        });
        methods.add_method("log", |_, this, message: String| {
            let name = this
                .world
                .borrow()
                .names
                .get(&this.id)
                .map(|n| n.name.clone())
                .unwrap_or_else(|| format!("Entity {}", this.id.0));
            println!("[lua:{name}] {message}");
            Ok(())
        });
    }
}

struct LoadedScript {
    path: String,
    table: Table,
}

pub struct ScriptEngine {
    lua: Lua,
    scripts: HashMap<Id, LoadedScript>,
    world: Rc<RefCell<World>>,
    script_root: PathBuf,
}

impl ScriptEngine {
    pub fn new(world: Rc<RefCell<World>>, script_root: impl Into<PathBuf>) -> mlua::Result<Self> {
        let lua = Lua::new();
        let globals = lua.globals();
        globals.set(
            "prism_log",
            lua.create_function(|_, message: String| {
                println!("[lua] {message}");
                Ok(())
            })?,
        )?;
        Ok(Self {
            lua,
            scripts: HashMap::new(),
            world,
            script_root: script_root.into(),
        })
    }

    pub fn update(&mut self, dt: f32) {
        let ids = self.world.borrow().entity_ids_with_scripts();
        for id in ids {
            if let Err(err) = self.update_entity(id, dt) {
                if let Some(script) = self.world.borrow_mut().scripts.get_mut(&id) {
                    script.last_error = Some(err.to_string());
                }
            }
        }
    }

    pub fn forget(&mut self, id: Id) {
        self.scripts.remove(&id);
    }

    fn update_entity(&mut self, id: Id, dt: f32) -> mlua::Result<()> {
        let (path, started) = {
            let world = self.world.borrow();
            let Some(script) = world.scripts.get(&id) else {
                return Ok(());
            };
            (script.path.clone(), script.started)
        };

        self.ensure_loaded(id, &path)?;
        let entity = self.lua.create_userdata(LuaEntity {
            id,
            world: Rc::clone(&self.world),
        })?;

        if !started {
            self.call_optional(id, "on_start", (entity.clone(),))?;
            if let Some(script) = self.world.borrow_mut().scripts.get_mut(&id) {
                script.started = true;
                script.last_error = None;
            }
        }

        self.call_optional(id, "on_update", (entity, dt))?;
        if let Some(script) = self.world.borrow_mut().scripts.get_mut(&id) {
            script.last_error = None;
        }
        Ok(())
    }

    fn ensure_loaded(&mut self, id: Id, path: &str) -> mlua::Result<()> {
        if self
            .scripts
            .get(&id)
            .map(|script| script.path == path)
            .unwrap_or(false)
        {
            return Ok(());
        }

        let resolved = self.resolve_script_path(path);
        let source = fs::read_to_string(&resolved).map_err(|err| {
            mlua::Error::external(format!("{}: {err}", resolved.display()))
        })?;
        let table = match self.lua.load(&source).set_name(path).eval::<Value>()? {
            Value::Table(table) => table,
            Value::Nil => self.lua.create_table()?,
            other => {
                return Err(mlua::Error::external(format!(
                    "script {} returned {}, expected table",
                    resolved.display(),
                    other.type_name()
                )))
            }
        };
        self.scripts.insert(
            id,
            LoadedScript {
                path: path.to_string(),
                table,
            },
        );
        Ok(())
    }

    fn resolve_script_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.script_root.join(path)
        }
    }

    fn call_optional<A>(&self, id: Id, name: &str, args: A) -> mlua::Result<()>
    where
        A: mlua::IntoLuaMulti,
    {
        let Some(script) = self.scripts.get(&id) else {
            return Ok(());
        };
        match script.table.get::<Value>(name)? {
            Value::Function(function) => function.call::<()>(args),
            Value::Nil => Ok(()),
            _ => Err(mlua::Error::external(format!(
                "{} must be a function when present",
                name
            ))),
        }
    }
}

pub fn script_path_for_entity_name(name: &str) -> String {
    let slug: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("{slug}.lua")
}

pub fn entity_userdata_id(userdata: AnyUserData) -> mlua::Result<u64> {
    let entity = userdata.borrow::<LuaEntity>()?;
    Ok(entity.id.0)
}
