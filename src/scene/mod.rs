use glam::Vec3;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SceneKind {
    Decanter,
    Wine,
    CornellBox,
}

impl SceneKind {
    pub fn label(self) -> &'static str {
        match self {
            SceneKind::Decanter => "Decanter",
            SceneKind::Wine => "Wine",
            SceneKind::CornellBox => "Cornell Box",
        }
    }

    pub fn index(self) -> u32 {
        match self {
            SceneKind::Decanter => 0,
            SceneKind::Wine => 2,
            SceneKind::CornellBox => 1,
        }
    }

    pub fn default_camera(self, scene_center: Vec3) -> (Vec3, Vec3) {
        match self {
            SceneKind::Decanter => (Vec3::new(3.0, 23.0, 40.0), scene_center),
            SceneKind::Wine => (scene_center + Vec3::new(0.0, 7.0, 18.0), scene_center),
            SceneKind::CornellBox => (Vec3::new(0.0, 1.0, 3.6), Vec3::new(0.0, 1.0, -1.0)),
        }
    }
}
