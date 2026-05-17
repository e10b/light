mod application;
mod compute_pass;
mod mesh;
mod photon_mapper;
mod quad_pass;
mod scene;
mod window;

fn main() {
    pollster::block_on(application::run());
}
