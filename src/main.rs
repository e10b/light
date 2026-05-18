mod blender_data;
mod prism_file;
mod application;
mod compute_pass;
mod ecs;
mod material_editor;
mod mesh;
mod photon_mapper;
mod quad_pass;
mod raster_pass;
mod scene;
mod window;

fn main() {
    pollster::block_on(application::run());
}
