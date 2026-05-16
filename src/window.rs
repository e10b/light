use std::sync::Arc;

use winit::{event_loop::EventLoop, window::Window};

pub fn create_window(event_loop: &EventLoop<()>, title: &str) -> Arc<Window> {
    Arc::new(
        event_loop
            .create_window(winit::window::WindowAttributes::default().with_title(title))
            .expect("failed to create window"),
    )
}
