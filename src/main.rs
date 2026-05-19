mod application;
mod compute_pass;
mod ecs;
mod editor;
mod material_editor;
mod mesh;
mod photon_mapper;
mod prism_file;
mod quad_pass;
mod raster_pass;
mod scene;
mod scene_data;
mod tooling;
mod window;

fn main() {
    pollster::block_on(application::run());
}
