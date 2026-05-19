use std::{cell::RefCell, collections::HashSet, rc::Rc};

use crate::{
    application::types::Camera,
    ecs::{
        CameraComponent, ColliderComponent, ColliderShape, PhysicsComponent, TransformComponent,
        World,
    },
    scene_data::{Id, MainDatabase},
};

const GROUND_Y: f32 = -1.5;
pub const PLAYER_HALF_EXTENTS: glam::Vec3 = glam::Vec3::new(0.35, 0.9, 0.35);
const EYE_HEIGHT: f32 = 1.55;

pub struct PlayerInputBindings {
    pub move_forward: &'static str,
    pub move_backward: &'static str,
    pub move_left: &'static str,
    pub move_right: &'static str,
    pub sprint: &'static str,
}

impl Default for PlayerInputBindings {
    fn default() -> Self {
        Self {
            move_forward: "w",
            move_backward: "s",
            move_left: "a",
            move_right: "d",
            sprint: "Shift",
        }
    }
}

#[derive(Default)]
pub struct PlayMode {
    pub active: bool,
    saved_camera: Option<Camera>,
    yaw: f32,
    pitch: f32,
    input: PlayerInputBindings,
}

impl PlayMode {
    pub fn start(
        &mut self,
        world: &Rc<RefCell<World>>,
        main_db: &mut MainDatabase,
        player_id: Id,
        camera_id: Id,
        camera: &Camera,
    ) {
        if !self.active {
            self.saved_camera = Some(Camera {
                pos: camera.pos,
                yaw: camera.yaw,
                pitch: camera.pitch,
            });
        }
        self.active = true;
        self.yaw = camera.yaw;
        self.pitch = camera.pitch;

        let mut spawn_pos = camera.pos;
        spawn_pos.y = (GROUND_Y + PLAYER_HALF_EXTENTS.y).max(spawn_pos.y);

        if let Some(obj) = main_db.objects.get_mut(&player_id) {
            obj.transform.location = spawn_pos;
            obj.transform.rotation = glam::Quat::from_rotation_y(self.yaw);
            obj.transform.scale = player_cube_scale();
        }
        if let Some(obj) = main_db.objects.get_mut(&camera_id) {
            obj.transform.location = glam::Vec3::new(0.0, EYE_HEIGHT, 0.0);
            obj.transform.rotation = glam::Quat::IDENTITY;
            obj.transform.scale = glam::Vec3::ONE;
        }

        let mut world = world.borrow_mut();
        world.transforms.insert(
            player_id,
            TransformComponent {
                translation: spawn_pos,
                rotation: glam::Quat::from_rotation_y(self.yaw),
                scale: player_cube_scale(),
            },
        );
        world.transforms.insert(
            camera_id,
            TransformComponent {
                translation: glam::Vec3::new(0.0, EYE_HEIGHT, 0.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
        );
        world.set_parent(camera_id, player_id);
        world.attach_physics(
            player_id,
            PhysicsComponent {
                velocity: glam::Vec3::ZERO,
                mass: 1.0,
                dynamic: true,
            },
        );
        world.attach_collider(
            player_id,
            ColliderComponent {
                shape: ColliderShape::Box {
                    half_extents: PLAYER_HALF_EXTENTS,
                },
            },
        );
        world.attach_collider(
            Id(0),
            ColliderComponent {
                shape: ColliderShape::Plane {
                    normal: glam::Vec3::Y,
                    offset: GROUND_Y,
                },
            },
        );
        world.attach_script(player_id, "player_controller.lua");
        for camera in world.cameras.values_mut() {
            camera.active = false;
        }
        world.cameras.insert(
            camera_id,
            CameraComponent {
                active: true,
                attached_to: Some(player_id),
                ..CameraComponent::default()
            },
        );
        world.update_global_transforms_and_visibility();
    }

    pub fn stop(&mut self, world: &Rc<RefCell<World>>, camera_id: Id, camera: &mut Camera) {
        if let Some(saved) = self.saved_camera.take() {
            *camera = saved;
        }
        self.active = false;
        let mut world = world.borrow_mut();
        world.clear_parent(camera_id);
        if let Some(camera_component) = world.cameras.get_mut(&camera_id) {
            camera_component.active = false;
            camera_component.attached_to = None;
        }
    }

    pub fn trigger_look_action(&mut self, delta: (f64, f64), mouse_speed: f32) {
        if !self.active {
            return;
        }
        self.yaw -= delta.0 as f32 * mouse_speed;
        self.pitch -= delta.1 as f32 * mouse_speed;
        self.pitch = self.pitch.clamp(-1.45, 1.45);
    }

    pub fn apply_movement_input(
        &self,
        world: &Rc<RefCell<World>>,
        player_id: Id,
        keys_pressed: &HashSet<String>,
        move_speed: f32,
    ) {
        if !self.active {
            return;
        }

        let sprint = if keys_pressed.contains(self.input.sprint) {
            2.2
        } else {
            1.0
        };
        let forward = glam::Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos()).normalize_or_zero();
        let right = glam::Vec3::Y.cross(forward).normalize_or_zero();
        let mut wish = glam::Vec3::ZERO;
        if keys_pressed.contains(self.input.move_forward) {
            wish += forward;
        }
        if keys_pressed.contains(self.input.move_backward) {
            wish -= forward;
        }
        if keys_pressed.contains(self.input.move_right) {
            wish += right;
        }
        if keys_pressed.contains(self.input.move_left) {
            wish -= right;
        }

        let mut world = world.borrow_mut();
        if let Some(transform) = world.transforms.get_mut(&player_id) {
            transform.rotation = glam::Quat::from_rotation_y(self.yaw);
        }
        let velocity = world
            .physics
            .entry(player_id)
            .or_insert_with(PhysicsComponent::default)
            .velocity;
        let horizontal = if wish.length_squared() > 0.0 {
            wish.normalize() * move_speed * sprint
        } else {
            glam::Vec3::ZERO
        };
        if let Some(physics) = world.physics.get_mut(&player_id) {
            physics.velocity = glam::Vec3::new(horizontal.x, velocity.y, horizontal.z);
        }
    }

    pub fn sync_camera_from_player(
        &self,
        world: &Rc<RefCell<World>>,
        camera_id: Id,
        camera: &mut Camera,
    ) {
        if !self.active {
            return;
        }

        let mut world = world.borrow_mut();
        world.transforms.entry(camera_id).or_default().translation =
            glam::Vec3::new(0.0, EYE_HEIGHT, 0.0);
        world.update_global_transforms_and_visibility();

        if let Some(camera_transform) = world.global_transforms.get(&camera_id) {
            camera.pos = camera_transform.translation;
        }
        camera.yaw = self.yaw;
        camera.pitch = self.pitch;
    }
}

pub fn player_cube_scale() -> glam::Vec3 {
    PLAYER_HALF_EXTENTS / 1.5
}

pub fn play_ground_y() -> f32 {
    GROUND_Y
}
