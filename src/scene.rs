use glam::Vec3;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SceneKind {
    Decanter,
    CornellBox,
}

impl SceneKind {
    pub fn label(self) -> &'static str {
        match self {
            SceneKind::Decanter => "Decanter",
            SceneKind::CornellBox => "Cornell Box",
        }
    }

    pub fn index(self) -> u32 {
        match self {
            SceneKind::Decanter => 0,
            SceneKind::CornellBox => 1,
        }
    }

    pub fn default_camera(self, decanter_center: Vec3) -> (Vec3, Vec3) {
        match self {
            SceneKind::Decanter => (
                Vec3::new(3.0, 23.0, 40.0),
                decanter_center,
            ),
            SceneKind::CornellBox => (
                Vec3::new(0.0, 1.0, 3.6),
                Vec3::new(0.0, 1.0, -1.0),
            ),
        }
    }
}
