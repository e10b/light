mod add_menu;
mod ecs_sync;
mod editor_surface;
mod geometry;
mod runtime;
mod types;
mod view_math;

pub struct Application;

impl Application {
    pub async fn run() {
        runtime::run().await;
    }

    pub const fn module_count() -> usize {
        7
    }
}

pub async fn run() {
    Application::run().await;
}
