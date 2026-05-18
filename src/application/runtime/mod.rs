mod camera_motion;
mod camera_state;
mod engine;
mod fps;
mod input;
mod mouse_look;
mod resize;
mod stress;
mod sun;
mod world_tick;

pub struct Runtime;

impl Runtime {
    pub async fn run() {
        engine::run().await;
    }
}

pub async fn run() {
    Runtime::run().await;
}
