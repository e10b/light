mod application;
mod blender_data;
mod compute_pass;
mod material_editor;
mod mesh;
mod photon_mapper;
mod prism_file;
mod quad_pass;
mod scene;
mod window;

fn main() {
    pollster::block_on(application::run());
}
