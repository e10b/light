use crate::{mesh::MeshData, scene::SceneKind, scene_data::Id};

pub const MAX_SUN_LIGHTS: usize = 8;
pub const MAX_PHOTON_TARGETS: usize = 4096;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    pub view_inv: [[f32; 4]; 4],
    pub proj_inv: [[f32; 4]; 4],
    pub light_pos: [f32; 4],
    pub sphere_pos: [f32; 4],
    pub sphere_color: [f32; 4],
    pub sphere_params: [f32; 4],
    pub sphere_rot: [f32; 4],
    pub sphere_extent: [f32; 4],
    pub mesh_center: [f32; 4],
    pub decanter_center: [f32; 4],
    pub cornell_center: [f32; 4],
    pub cornell_color: [f32; 4],
    pub cornell_params: [f32; 4],
    pub sun_lights: [[f32; 4]; MAX_SUN_LIGHTS],
    pub sun_intensity: f32,
    pub frame: u32,
    pub scene_kind: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub selected_object: u32,
    pub mesh_enabled: u32,
    pub decanter_enabled: u32,
    pub wine_enabled: u32,
    pub cornell_enabled: u32,
    pub sun_light_count: u32,
    pub _pad: [u32; 1],
}

pub struct Camera {
    pub pos: glam::Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum GizmoTargetKind {
    Sphere,
    Decanter,
    WineGlass,
    CornellBox,
    SunLamp,
    WineSpotlight,
}

#[derive(Clone)]
pub struct MeshObjectInstance {
    pub object_id: Id,
    pub mesh_asset_id: u32,
    pub vertex_start: usize,
    pub vertex_count: usize,
    pub index_start: usize,
    pub index_count: usize,
    pub material_start: usize,
    pub material_count: usize,
    pub base_positions: Vec<glam::Vec3>,
    pub base_normals: Vec<glam::Vec3>,
    pub pivot: glam::Vec3,
    pub max_extent: f32,
    pub rotation: glam::Quat,
    pub translation: glam::Vec3,
    pub scale: glam::Vec3,
}

impl MeshObjectInstance {
    pub fn center(&self) -> glam::Vec3 {
        self.pivot + self.translation
    }
}

pub struct MeshAsset {
    pub asset_id: u32,
    pub name: String,
    pub mesh: MeshData,
}

#[derive(Clone)]
pub struct LightObjectInstance {
    pub object_id: Id,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub intensity: f32,
}

pub fn default_target_for_scene(_scene_kind: SceneKind) -> GizmoTargetKind {
    GizmoTargetKind::Decanter
}

pub fn target_allowed_in_scene(_scene_kind: SceneKind, _target: GizmoTargetKind) -> bool {
    true
}

pub fn target_label(target: GizmoTargetKind) -> &'static str {
    match target {
        GizmoTargetKind::Sphere => "Cube",
        GizmoTargetKind::Decanter => "Decanter",
        GizmoTargetKind::WineGlass => "Wine Glass",
        GizmoTargetKind::CornellBox => "Cornell Box",
        GizmoTargetKind::SunLamp => "Sun Lamp",
        GizmoTargetKind::WineSpotlight => "Spotlight",
    }
}

impl Camera {
    pub fn look_at(pos: glam::Vec3, target: glam::Vec3) -> Self {
        let forward = (target - pos).normalize_or_zero();
        let yaw = forward.x.atan2(forward.z);
        let pitch = forward.y.asin();
        Self { pos, yaw, pitch }
    }

    pub fn forward(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }

    pub fn right(&self) -> glam::Vec3 {
        self.forward().cross(glam::Vec3::Y).normalize()
    }

    pub fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.pos, self.pos + self.forward(), glam::Vec3::Y)
    }
}
