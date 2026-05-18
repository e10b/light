mod blender_data;
mod prism_file;
mod application;
mod compute_pass;
mod ecs;
mod editor_ui;
mod material_editor;
mod mesh;
mod photon_mapper;
mod quad_pass;
mod raster_pass;
mod scene;
mod tooling_lua;
mod tooling_materials;
mod tooling_persistence;
mod window;

fn main() {
    pollster::block_on(application::run());
}
