mod application;
mod compute_pass;
mod quad_pass;
mod scene;
mod mesh;
mod window;

fn main() {
    pollster::block_on(application::run());
}
