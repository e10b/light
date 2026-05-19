use std::iter;

use egui::{Color32, Pos2, Stroke, ViewportId};
use egui_code_editor::Syntax;
use egui_wgpu::{Renderer as EguiRenderer, RendererOptions, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use transform_gizmo::config::TransformPivotPoint;
use transform_gizmo::{math::Transform as GizmoTransform, prelude::*};
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop};

use crate::{
    compute_pass,
    ecs::{
        CameraComponent, ColliderComponent, ColliderShape, PhysicsComponent, ScriptEngine, World,
    },
    editor::panels::{CameraProjectionKind, GizmoModeKind, RenderModeKind},
    material_editor::{MaterialGraphEditor, RuntimeMaterialPreview},
    mesh::{load_gltf_mesh, Vertex},
    photon_mapper::PhotonMapper,
    prism_file::MaterialData as PrismMaterialData,
    quad_pass, raster_pass,
    scene::SceneKind,
    scene_data::{Id, MainDatabase, Transform as DbTransform},
    tooling::lua::scripts_dir,
    tooling::materials::{
        make_empty_material, make_glass_material, make_white_material, preview_from_material_data,
    },
    window::create_window,
};

use super::super::{
    add_menu::{draw_add_menu, AddMenuContext},
    ecs_sync::register_object_entity,
    editor_surface::draw_editor_surface,
    frame_render::render_frame_and_present,
    geometry::{
        append_object_mesh, build_photon_targets, make_cube_mesh, make_prism_mesh, mesh_bounds,
        orient_and_scale_mesh, place_instance_center, sphere_position_for, translate_mesh,
        update_mesh_transform, visible_render_geometry,
    },
    gpu_scene::sync_accumulation_and_geometry,
    project_io::{draw_project_io_buttons, ProjectIoContext},
    types::{
        default_target_for_scene, target_allowed_in_scene, Camera, GizmoTargetKind,
        LightObjectInstance, MeshAsset, MeshObjectInstance, SceneUniforms, MAX_SUN_LIGHTS,
    },
    view_math::{
        camera_projection_matrix, gizmo_projection_matrix, intersect_cube, intersect_sphere,
        scene_camera, wine_spotlight_position, world_ray_from_cursor, world_to_screen,
    },
};
use super::input::{handle_keyboard_input, handle_pointer_window_event};
use super::{
    camera_motion::apply_fly_camera_motion,
    camera_state::write_runtime_back_to_database,
    fps::update_fps_title,
    mouse_look::handle_mouse_motion,
    play_mode::{play_ground_y, player_cube_scale, PlayMode, PLAYER_HALF_EXTENTS},
    resize::handle_surface_resize,
    stress::maybe_build_stress_scene,
    sun::update_sun_lights,
    world_tick::tick_world_and_scripts,
};

pub async fn run() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = create_window(&event_loop, "wgpu v0.29 ray tracing");

    let size = window.inner_size();
    let instance = wgpu::Instance::default();
    let window_ref = window.clone();
    let surface = instance.create_surface(window_ref.as_ref()).unwrap();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("No adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: wgpu::Limits::default()
                .using_minimum_supported_acceleration_structure_values(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats[0];

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Immediate,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 0,
    };
    surface.configure(&device, &config);

    let decanter_path = std::path::Path::new("res/wine_decanter_and_glass.glb");
    let wine_path = std::path::Path::new("res/red_wine_glass.glb");

    let sphere_radius = 6.0;
    let default_cube_center = glam::Vec3::new(0.0, -1.5 + sphere_radius, 0.0);
    let default_cube_mesh = make_cube_mesh(glam::Vec3::ZERO, 1.5);
    let default_cube_vertex_len = default_cube_mesh.vertices.len();
    let (cube_center, cube_size, _, _) = mesh_bounds(&default_cube_mesh.vertices);
    let cube_max_extent = cube_size.max_element().max(0.1);
    let mut mesh = default_cube_mesh;

    // No preloaded scene-specific geometry: start from one mesh object path only.
    let decanter_material_start = 0usize;
    let decanter_material_count = 0usize;
    let decanter_vertex_start = 0usize;
    let decanter_vertex_count = 0usize;
    let decanter_index_start = 0usize;
    let decanter_index_count = 0usize;
    let decanter_base_positions: Vec<glam::Vec3> = Vec::new();
    let decanter_base_normals: Vec<glam::Vec3> = Vec::new();
    let decanter_center = cube_center;
    let decanter_size = cube_size;

    let wine_center = cube_center;
    let wine_size = cube_size;
    let wine_vertex_start = 0usize;
    let wine_vertex_count = 0usize;
    let wine_index_start = 0usize;
    let wine_index_count = 0usize;
    let wine_material_start = 0usize;
    let wine_material_count = 0usize;
    let wine_base_positions: Vec<glam::Vec3> = Vec::new();
    let wine_base_normals: Vec<glam::Vec3> = Vec::new();

    let (center, size, _, _) = mesh_bounds(&mesh.vertices);
    let decanter_max_extent = cube_max_extent;
    let wine_max_extent = cube_max_extent;
    let mesh_assets = vec![
        MeshAsset {
            asset_id: 0,
            name: "Cube".to_string(),
            mesh: make_cube_mesh(glam::Vec3::ZERO, 1.5),
        },
        MeshAsset {
            asset_id: 4,
            name: "CornellBox".to_string(),
            mesh: make_cube_mesh(glam::Vec3::ZERO, 2.0),
        },
    ];
    let render_width = 1280u32;
    let render_height = 720u32;

    let mut editor_db = MainDatabase::new();
    let mut play_db: Option<MainDatabase> = None;
    let decanter_mesh_id = editor_db.create_mesh("DecanterMesh", decanter_vertex_count);
    let wine_mesh_id = editor_db.create_mesh("WineGlassMesh", wine_vertex_count);
    let cornell_mesh_id = editor_db.create_mesh(
        "CornellBoxMesh",
        make_cube_mesh(glam::Vec3::ZERO, 2.0).vertices.len(),
    );
    let sphere_obj_id = editor_db.create_object("Cube", None, DbTransform::default());
    let sun_obj_id = editor_db.create_object("SunLamp", None, DbTransform::default());
    let spot_obj_id = editor_db.create_object("Spotlight", None, DbTransform::default());
    let decanter_obj_id =
        editor_db.create_object("Decanter", Some(decanter_mesh_id), DbTransform::default());
    let wine_obj_id =
        editor_db.create_object("WineGlass", Some(wine_mesh_id), DbTransform::default());
    let cornell_obj_id =
        editor_db.create_object("CornellBox", Some(cornell_mesh_id), DbTransform::default());
    let player_start = glam::Vec3::new(0.0, play_ground_y() + 0.9, 5.0);
    let player_obj_id = editor_db.create_object(
        "Player",
        None,
        DbTransform {
            location: player_start,
            rotation: glam::Quat::IDENTITY,
            scale: player_cube_scale(),
        },
    );
    let player_camera_obj_id = editor_db.create_object(
        "Player Camera",
        None,
        DbTransform {
            location: glam::Vec3::new(0.0, 1.55, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        },
    );
    let ecs_world = std::rc::Rc::new(std::cell::RefCell::new(World::new()));
    {
        let mut world = ecs_world.borrow_mut();
        for object_id in [
            sphere_obj_id,
            sun_obj_id,
            spot_obj_id,
            decanter_obj_id,
            wine_obj_id,
            cornell_obj_id,
            player_obj_id,
            player_camera_obj_id,
        ] {
            register_object_entity(&mut world, &editor_db, object_id);
        }
        world.attach_light(sun_obj_id, 0.8);
        world.attach_light(spot_obj_id, 1.0);
        world.attach_camera(
            player_camera_obj_id,
            CameraComponent {
                active: false,
                attached_to: None,
                ..CameraComponent::default()
            },
        );
        world.attach_physics(player_obj_id, PhysicsComponent::default());
        world.attach_collider(
            player_obj_id,
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
                    offset: play_ground_y(),
                },
            },
        );
    }
    let mut script_engine = ScriptEngine::new(std::rc::Rc::clone(&ecs_world), scripts_dir())
        .expect("create Lua engine");
    let mut material_library: std::collections::HashMap<String, PrismMaterialData> =
        std::collections::HashMap::new();
    material_library.insert("White".to_string(), make_white_material());
    material_library.insert("Empty".to_string(), make_empty_material());
    material_library.insert("Glass".to_string(), make_glass_material());
    let mut object_material_names: std::collections::HashMap<Id, String> =
        std::collections::HashMap::new();
    object_material_names.insert(sphere_obj_id, "Glass".to_string());
    object_material_names.insert(decanter_obj_id, "Glass".to_string());
    object_material_names.insert(wine_obj_id, "Glass".to_string());
    object_material_names.insert(cornell_obj_id, "Empty".to_string());
    object_material_names.insert(player_obj_id, "White".to_string());
    let mut last_material_signature = String::new();
    let default_cube_mesh_id = editor_db.create_mesh("CubeMesh", default_cube_vertex_len);
    if let Some(mesh_db) = editor_db.meshes.get_mut(&default_cube_mesh_id) {
        mesh_db.user_count = 2;
    }
    if let Some(obj) = editor_db.objects.get_mut(&sphere_obj_id) {
        obj.mesh_id = Some(default_cube_mesh_id);
    }
    if let Some(obj) = editor_db.objects.get_mut(&player_obj_id) {
        obj.mesh_id = Some(default_cube_mesh_id);
    }
    ecs_world.borrow_mut().attach_mesh(player_obj_id, 0);
    let default_cube_instance = MeshObjectInstance {
        object_id: sphere_obj_id,
        mesh_asset_id: 0,
        vertex_start: 0,
        vertex_count: mesh.positions4.len(),
        index_start: 0,
        index_count: mesh.indices.len(),
        material_start: 0,
        material_count: mesh.materials.len(),
        base_positions: mesh
            .positions4
            .iter()
            .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
            .collect(),
        base_normals: mesh
            .normals4
            .iter()
            .map(|n| glam::Vec3::new(n[0], n[1], n[2]))
            .collect(),
        pivot: cube_center,
        max_extent: cube_max_extent,
        rotation: glam::Quat::IDENTITY,
        translation: default_cube_center - cube_center,
        scale: glam::Vec3::ONE,
    };
    let player_cube_instance = MeshObjectInstance {
        object_id: player_obj_id,
        mesh_asset_id: 0,
        vertex_start: 0,
        vertex_count: mesh.positions4.len(),
        index_start: 0,
        index_count: mesh.indices.len(),
        material_start: 0,
        material_count: mesh.materials.len(),
        base_positions: mesh
            .positions4
            .iter()
            .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
            .collect(),
        base_normals: mesh
            .normals4
            .iter()
            .map(|n| glam::Vec3::new(n[0], n[1], n[2]))
            .collect(),
        pivot: cube_center,
        max_extent: cube_max_extent,
        rotation: glam::Quat::IDENTITY,
        translation: player_start - cube_center,
        scale: player_cube_scale(),
    };

    let mut model_verts = mesh.vertices.clone();
    let mut model_idx = mesh.indices.clone();
    println!(
        "Loaded {} vertices and {} indices from default cube mesh object",
        model_verts.len(),
        model_idx.len()
    );

    let mut mesh_instances = vec![default_cube_instance, player_cube_instance];

    let mut object_target_by_id: std::collections::HashMap<Id, GizmoTargetKind> =
        std::collections::HashMap::new();
    object_target_by_id.insert(sphere_obj_id, GizmoTargetKind::Decanter);
    object_target_by_id.insert(sun_obj_id, GizmoTargetKind::SunLamp);
    object_target_by_id.insert(spot_obj_id, GizmoTargetKind::WineSpotlight);
    object_target_by_id.insert(decanter_obj_id, GizmoTargetKind::Decanter);
    object_target_by_id.insert(wine_obj_id, GizmoTargetKind::WineGlass);
    object_target_by_id.insert(cornell_obj_id, GizmoTargetKind::CornellBox);
    object_target_by_id.insert(player_obj_id, GizmoTargetKind::Decanter);
    object_target_by_id.insert(player_camera_obj_id, GizmoTargetKind::Camera);

    let mut decanter_master = editor_db.create_collection("SceneMaster");
    let mut wine_master = Id(0);
    let mut cornell_master = Id(0);
    let mut decanter_scene_id = editor_db.create_scene("Scene", decanter_master);
    let mut wine_scene_id = Id(0);
    let mut cornell_scene_id = Id(0);
    editor_db.collection_link_object(decanter_master, sphere_obj_id);
    editor_db.ensure_scene_base(decanter_scene_id, sphere_obj_id, true, true);
    editor_db.collection_link_object(decanter_master, sun_obj_id);
    editor_db.ensure_scene_base(decanter_scene_id, sun_obj_id, true, true);
    editor_db.collection_link_object(decanter_master, player_obj_id);
    editor_db.ensure_scene_base(decanter_scene_id, player_obj_id, true, true);
    editor_db.collection_link_object(decanter_master, player_camera_obj_id);
    editor_db.ensure_scene_base(decanter_scene_id, player_camera_obj_id, true, true);
    println!(
        "Scene bounds: decanter center={:?}, wine center={:?}, combined center={:?}, size={:?}",
        decanter_center, wine_center, center, size
    );

    let mut vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("model_vbuf"),
        contents: bytemuck::cast_slice(&model_verts),
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
    });
    let mut ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("model_ibuf"),
        contents: bytemuck::cast_slice(&model_idx),
        usage: wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::COPY_DST,
    });
    let mut pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_pos_buf"),
        contents: bytemuck::cast_slice(&mesh.positions4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let mut nrm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_nrm_buf"),
        contents: bytemuck::cast_slice(&mesh.normals4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let mut idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_idx_buf"),
        contents: bytemuck::cast_slice(&model_idx),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let mut tri_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_tri_mat_buf"),
        contents: bytemuck::cast_slice(&mesh.triangle_material_ids),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let mut mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_materials_buf"),
        contents: bytemuck::cast_slice(&mesh.materials),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let mut model_blas_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: model_verts.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(model_idx.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let mut model_blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("model_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![model_blas_desc.clone()],
        },
    );

    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("scene_tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: mesh_instances.len().max(1) as u32,
    });
    for (i, inst) in mesh_instances.iter().enumerate() {
        let c = inst.center();
        tlas[i] = Some(wgpu::TlasInstance::new(
            &model_blas,
            [1.0, 0.0, 0.0, c.x, 0.0, 1.0, 0.0, c.y, 0.0, 0.0, 1.0, c.z],
            0,
            0xff,
        ));
    }

    let model_build = wgpu::BlasBuildEntry {
        blas: &model_blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
            size: &model_blas_desc,
            vertex_buffer: &vbuf,
            first_vertex: 0,
            vertex_stride: std::mem::size_of::<Vertex>() as u64,
            index_buffer: Some(&ibuf),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        }]),
    };

    let mut accel_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("accel"),
    });
    accel_encoder.build_acceleration_structures([model_build].iter(), iter::once(&tlas));
    queue.submit(Some(accel_encoder.finish()));

    let mut camera_projection_mode = CameraProjectionKind::Perspective;
    let mut camera_fov_radians = std::f32::consts::FRAC_PI_3 * 1.2;
    let mut camera_ortho_height = 12.0f32;
    let mut camera_near = 0.1f32;
    let mut camera_far = 1000.0f32;
    let projection = camera_projection_matrix(
        camera_projection_mode,
        camera_fov_radians,
        camera_ortho_height,
        camera_near,
        camera_far,
        config.width,
        config.height,
    );

    let scene_kind = SceneKind::Decanter;
    let active_center = decanter_center;
    let active_max_extent = decanter_max_extent;
    let (camera_pos, camera_target) = scene_camera(scene_kind, active_center, decanter_size);
    let mut camera = Camera::look_at(camera_pos, camera_target);
    let mut uniforms = SceneUniforms {
        view_inv: camera.view_matrix().inverse().to_cols_array_2d(),
        proj_inv: projection.inverse().to_cols_array_2d(),
        light_pos: [10.0, 8.0, 10.0, 1.0],
        sphere_pos: [1.0e9, 1.0e9, 1.0e9, 0.001],
        sphere_color: [0.98, 1.0, 1.0, 1.0],
        sphere_params: [0.02, 1.52, 1.0, 0.0],
        sphere_rot: [0.0, 0.0, 0.0, 1.0],
        sphere_extent: [0.001, 0.001, 0.001, 0.0],
        mesh_center: [
            wine_center.x,
            wine_center.y,
            wine_center.z,
            wine_max_extent * 0.8,
        ],
        decanter_center: [
            decanter_center.x,
            decanter_center.y,
            decanter_center.z,
            decanter_max_extent * 0.7,
        ],
        cornell_center: [0.0, 0.5, -1.0, 1.0],
        cornell_color: [1.0, 1.0, 1.0, 0.0],
        cornell_params: [0.7, 1.0, 0.0, 0.0],
        sun_lights: [[0.0, 0.0, 0.0, 0.0]; MAX_SUN_LIGHTS],
        sun_intensity: 0.8,
        frame: 0,
        scene_kind: scene_kind.index(),
        render_width,
        render_height,
        selected_object: 1,
        mesh_enabled: 0,
        decanter_enabled: 0,
        wine_enabled: 0,
        cornell_enabled: 0,
        sun_light_count: 0,
        _pad: [0; 1],
    };

    let mut sun_azimuth_deg = uniforms.light_pos[2]
        .atan2(uniforms.light_pos[0])
        .to_degrees();
    let sun_len_xz = (uniforms.light_pos[0] * uniforms.light_pos[0]
        + uniforms.light_pos[2] * uniforms.light_pos[2])
        .sqrt();
    let mut sun_elevation_deg = uniforms.light_pos[1].atan2(sun_len_xz).to_degrees();
    let mut sun_intensity = uniforms.sun_intensity;
    let mut sun_lamp_distance = decanter_max_extent.max(8.0) * 2.2;
    let mut sun_empty_rotation = glam::Quat::IDENTITY;
    let mut sun_empty_scale = glam::Vec3::ONE;
    let mut sun_empty_position = active_center
        + glam::Vec3::new(
            sun_azimuth_deg.to_radians().cos() * sun_elevation_deg.to_radians().cos(),
            sun_elevation_deg.to_radians().sin(),
            sun_azimuth_deg.to_radians().sin() * sun_elevation_deg.to_radians().cos(),
        )
        .normalize_or_zero()
            * sun_lamp_distance;
    let mut light_instances = vec![LightObjectInstance {
        object_id: sun_obj_id,
        position: sun_empty_position,
        rotation: sun_empty_rotation,
        scale: sun_empty_scale,
        intensity: sun_intensity,
    }];
    let wine_spotlight_azimuth_deg = -55.0;
    let wine_spotlight_elevation_deg = 54.0;
    let wine_spotlight_distance = wine_max_extent.max(10.0) * 1.4;
    let mut spot_empty_rotation = glam::Quat::IDENTITY;
    let mut spot_empty_scale = glam::Vec3::ONE;
    let mut spot_empty_position = wine_spotlight_position(
        wine_center,
        wine_spotlight_azimuth_deg,
        wine_spotlight_elevation_deg,
        wine_spotlight_distance,
    );

    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ubuf"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let accum_byte_size = (render_width as u64) * (render_height as u64) * 16;
    let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accum_buf"),
        size: accum_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    {
        let zeros = vec![0u8; accum_byte_size as usize];
        queue.write_buffer(&accum_buf, 0, &zeros);
    }

    let ubind = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ubind"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let compute_pass = compute_pass::ComputePass::new(&device, &ubind, render_width, render_height);
    let quad_pass = quad_pass::QuadPass::new(
        &device,
        surface_format,
        compute_pass.output_view(),
        compute_pass.selection_mask_view(),
    );
    let mut raster_pass = raster_pass::RasterPass::new(&device, surface_format);
    let mut raster_instance_count: u32 = 0;
    let mut raster_instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("raster_instance_buf"),
        contents: bytemuck::cast_slice(&[raster_pass::RasterInstance {
            offset: [0.0, 0.0, 0.0, 0.0],
        }]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let mut photon_mapper = PhotonMapper::new(
        &device,
        &queue,
        &tlas,
        &pos_buf,
        &nrm_buf,
        &idx_buf,
        &tri_mat_buf,
        &mat_buf,
    );

    let mut ugroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ugroup"),
        layout: &ubind,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: accum_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: nrm_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: idx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: tri_mat_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: mat_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(compute_pass.output_view()),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: photon_mapper.photon_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: photon_mapper.hash_heads().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: photon_mapper.uniforms_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(compute_pass.selection_mask_view()),
            },
        ],
    });

    let egui_ctx = egui::Context::default();
    let mut egui_visuals = egui::Visuals::dark();
    egui_visuals.panel_fill = Color32::from_rgb(42, 43, 46);
    egui_visuals.window_fill = Color32::from_rgb(49, 50, 54);
    egui_visuals.extreme_bg_color = Color32::from_rgb(28, 29, 31);
    egui_ctx.set_visuals(egui_visuals);
    let mut egui_state = EguiWinitState::new(
        egui_ctx.clone(),
        ViewportId::ROOT,
        window.as_ref(),
        Some(window.scale_factor() as f32),
        window.theme(),
        None,
    );
    let mut egui_renderer = EguiRenderer::new(&device, config.format, RendererOptions::default());

    let move_speed = 2.6;
    let look_speed = 0.28;
    let mouse_speed = 0.003;
    let mut keys_pressed = std::collections::HashSet::new();
    let mut frame_count = 0u32;
    let mut fps_display_time = std::time::Instant::now();
    let mut last_update = std::time::Instant::now();
    let mut accumulation_dirty = true;
    let mut render_mode = RenderModeKind::Pathtraced;
    let mut gizmo = Gizmo::default();
    let mut show_editor_ui = true;
    let mut gizmo_mode = GizmoModeKind::Translate;
    let mut gizmo_target = default_target_for_scene(scene_kind);
    let mut selected_object_id = Some(decanter_obj_id);
    let mut has_selection = true;
    let mut sphere_rotation = glam::Quat::IDENTITY;
    let mut sphere_scale = glam::Vec3::ONE;
    let mut decanter_rotation = glam::Quat::IDENTITY;
    let mut decanter_translation = glam::Vec3::ZERO;
    let mut decanter_scale = glam::Vec3::ONE;
    let mut cornell_rotation = glam::Quat::IDENTITY;
    let mut cornell_translation = glam::Vec3::ZERO;
    let mut cornell_scale = glam::Vec3::ONE;
    let mut wine_rotation = glam::Quat::IDENTITY;
    let mut wine_translation = glam::Vec3::ZERO;
    let mut wine_scale = glam::Vec3::ONE;
    let mut geometry_dirty = true;
    let mut gpu_mesh_dirty = false;
    let mut stress_test_requested = std::env::var("PRISM_STRESS_1M")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut stress_instance_count = 0usize;
    let mut project_status = String::new();
    let mut mouse_pos = [0.0f32, 0.0f32];
    let mut mouse_delta = (0.0f64, 0.0f64);
    let mut mouse_left_down = false;
    let mut mouse_left_clicked = false;
    let mut mouse_left_dragging = false;
    let mut material_editor = MaterialGraphEditor::new();
    let mut material_runtime_overrides: std::collections::HashMap<String, RuntimeMaterialPreview> =
        std::collections::HashMap::new();
    let lua_syntax = Syntax::lua();
    let mut lua_editor_entity: Option<Id> = None;
    let mut lua_editor_path = String::new();
    let mut lua_editor_text = String::new();
    let mut lua_editor_status = String::new();
    let mut play_mode = PlayMode::default();
    let mut show_editor_ui_before_play = show_editor_ui;

    let _ = event_loop.run(move |event, active_loop| {
        active_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        if let Event::WindowEvent { event, .. } = &event {
            let _ = egui_state.on_window_event(window.as_ref(), event);
            handle_pointer_window_event(
                event,
                &mut mouse_pos,
                &mut mouse_left_down,
                &mut mouse_left_clicked,
                &mut mouse_left_dragging,
            );
        }
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => active_loop.exit(),
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event, .. },
                ..
            } => {
                let escape_pressed = event.state == winit::event::ElementState::Pressed
                    && !event.repeat
                    && event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape);
                if escape_pressed && play_mode.active {
                    play_mode.stop(
                        &ecs_world,
                        &editor_db,
                        &mut mesh_instances,
                        &mut light_instances,
                        player_camera_obj_id,
                        &mut camera,
                    );
                    play_db = None;
                    show_editor_ui = show_editor_ui_before_play;
                    window.set_cursor_visible(true);
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    geometry_dirty = true;
                    accumulation_dirty = true;
                } else {
                    handle_keyboard_input(&event, &mut show_editor_ui, &mut keys_pressed);
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                handle_surface_resize(
                    size,
                    &mut config,
                    camera_projection_mode,
                    camera_fov_radians,
                    camera_ortho_height,
                    camera_near,
                    camera_far,
                    &mut uniforms,
                    &queue,
                    &ubuf,
                    &surface,
                    &device,
                );
            }
            Event::DeviceEvent {
                event: winit::event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                mouse_delta = delta;
                if play_mode.active {
                    play_mode.trigger_look_action(delta, mouse_speed);
                    accumulation_dirty = true;
                } else {
                    handle_mouse_motion(
                        delta,
                        &egui_ctx,
                        &keys_pressed,
                        mouse_speed,
                        &mut camera,
                        &mut accumulation_dirty,
                    );
                }
            }
            Event::NewEvents(start_cause) => match start_cause {
                winit::event::StartCause::Init | winit::event::StartCause::Poll => {
                    update_fps_title(window.as_ref(), &mut frame_count, &mut fps_display_time);

                    let now = std::time::Instant::now();
                    let dt = now.duration_since(last_update).as_secs_f32();
                    last_update = now;
                    let prev_cam_pos = camera.pos;
                    let prev_cam_yaw = camera.yaw;
                    let prev_cam_pitch = camera.pitch;
                    if !play_mode.active {
                        apply_fly_camera_motion(
                            &mut camera,
                            dt,
                            &keys_pressed,
                            move_speed,
                            look_speed,
                            &egui_ctx,
                        );
                    }

                    if play_mode.active {
                        play_mode.apply_movement_input(
                            &ecs_world,
                            player_obj_id,
                            &keys_pressed,
                            move_speed,
                        );
                    }

                    script_engine.update_input(&keys_pressed, mouse_delta);

                    {
                        let main_db = if play_mode.active {
                            play_db.as_mut().expect("PIE db missing")
                        } else {
                            &mut editor_db
                        };
                        if tick_world_and_scripts(
                            dt,
                            &ecs_world,
                            main_db,
                            &mut mesh_instances,
                            &mut light_instances,
                            &mut camera,
                            &mut script_engine,
                            play_mode.active,
                        ) {
                            geometry_dirty = true;
                            accumulation_dirty = true;
                        }
                    }
                    mouse_delta = (0.0, 0.0);
                    if play_mode.active {
                        play_mode.sync_camera_from_player(
                            &ecs_world,
                            player_camera_obj_id,
                            &mut camera,
                        );
                    }

                    uniforms.view_inv = camera.view_matrix().inverse().to_cols_array_2d();
                    let projection = camera_projection_matrix(
                        camera_projection_mode,
                        camera_fov_radians,
                        camera_ortho_height,
                        camera_near,
                        camera_far,
                        config.width,
                        config.height,
                    );
                    uniforms.proj_inv = projection.inverse().to_cols_array_2d();
                    if camera.pos != prev_cam_pos
                        || camera.yaw != prev_cam_yaw
                        || camera.pitch != prev_cam_pitch
                    {
                        accumulation_dirty = true;
                    }

                    match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(tex)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => {
                            let raw_input = egui_state.take_egui_input(window.as_ref());
                            let mut sun_changed = false;
                            let mut photon_emitter_center = [
                                active_center.x,
                                active_center.y,
                                active_center.z,
                                active_max_extent * 0.85,
                            ];
                            let mut photons_per_frame = 0u32;
                            let mut sphere_visible_for_photons = false;
                            let mut start_play_requested = false;
                            let full_output = {
                                let main_db = if play_mode.active {
                                    play_db.as_mut().expect("PIE db missing")
                                } else {
                                    &mut editor_db
                                };
                                egui_ctx.run(raw_input, |ctx| {
                                let mut suppress_scene_click = false;
                                let current_scene_exists =
                                    decanter_scene_id.0 != 0 && main_db.scenes.contains_key(&decanter_scene_id);

                                if show_editor_ui {
                                egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.strong("Prism");
                                        ui.separator();
                                        let _ = ui.button("New Cube Scene");
                                        if ui
                                            .add_enabled(!play_mode.active, egui::Button::new("Play"))
                                            .clicked()
                                        {
                                            start_play_requested = true;
                                        }
                                        if ui.button("Stress 1M Cubes").clicked() {
                                            stress_test_requested = true;
                                        }
                                        draw_add_menu(
                                            ui,
                                            AddMenuContext {
                                                scene_kind,
                                                decanter_scene_id,
                                                decanter_master,
                                                wine_master,
                                                cornell_master,
                                                wine_obj_id,
                                                spot_obj_id,
                                                cornell_obj_id,
                                                sphere_obj_id,
                                                active_center,
                                                wine_center,
                                                default_cube_mesh_id,
                                                cornell_mesh_id,
                                                camera_pos: camera.pos,
                                                sun_intensity,
                                                decanter_path,
                                                wine_path,
                                                main_db,
                                                ecs_world: &ecs_world,
                                                mesh: &mut mesh,
                                                mesh_instances: &mut mesh_instances,
                                                light_instances: &mut light_instances,
                                                object_target_by_id: &mut object_target_by_id,
                                                object_material_names: &mut object_material_names,
                                                model_idx: &mut model_idx,
                                                selected_object_id: &mut selected_object_id,
                                                gizmo_target: &mut gizmo_target,
                                                has_selection: &mut has_selection,
                                                gpu_mesh_dirty: &mut gpu_mesh_dirty,
                                                geometry_dirty: &mut geometry_dirty,
                                                accumulation_dirty: &mut accumulation_dirty,
                                                sun_empty_position: &mut sun_empty_position,
                                                project_status: &mut project_status,
                                                suppress_scene_click: &mut suppress_scene_click,
                                            },
                                        );
                                        draw_project_io_buttons(
                                            ui,
                                            ProjectIoContext {
                                                main_db,
                                                decanter_master: &mut decanter_master,
                                                wine_master: &mut wine_master,
                                                cornell_master: &mut cornell_master,
                                                decanter_scene_id: &mut decanter_scene_id,
                                                wine_scene_id: &mut wine_scene_id,
                                                cornell_scene_id: &mut cornell_scene_id,
                                                object_material_names: &mut object_material_names,
                                                material_library: &mut material_library,
                                                sphere_obj_id,
                                                decanter_obj_id,
                                                wine_obj_id,
                                                cornell_obj_id,
                                                sun_obj_id,
                                                spot_obj_id,
                                                uniforms: &mut uniforms,
                                                sphere_rotation: &mut sphere_rotation,
                                                sphere_scale: &mut sphere_scale,
                                                sphere_radius,
                                                decanter_center,
                                                decanter_translation: &mut decanter_translation,
                                                decanter_rotation: &mut decanter_rotation,
                                                decanter_scale: &mut decanter_scale,
                                                wine_center,
                                                wine_translation: &mut wine_translation,
                                                wine_rotation: &mut wine_rotation,
                                                wine_scale: &mut wine_scale,
                                                sun_empty_position: &mut sun_empty_position,
                                                sun_empty_rotation: &mut sun_empty_rotation,
                                                sun_empty_scale: &mut sun_empty_scale,
                                                spot_empty_position: &mut spot_empty_position,
                                                spot_empty_rotation: &mut spot_empty_rotation,
                                                spot_empty_scale: &mut spot_empty_scale,
                                                geometry_dirty: &mut geometry_dirty,
                                                accumulation_dirty: &mut accumulation_dirty,
                                                project_status: &mut project_status,
                                            },
                                        );
                                    });
                                });

                                draw_editor_surface(
                                    ctx,
                                    main_db,
                                    scene_kind,
                                    &project_status,
                                    decanter_scene_id,
                                    &object_target_by_id,
                                    &mut has_selection,
                                    &mut selected_object_id,
                                    &mut gizmo_target,
                                    sphere_obj_id,
                                    decanter_obj_id,
                                    wine_obj_id,
                                    cornell_obj_id,
                                    sun_obj_id,
                                    spot_obj_id,
                                    &mut light_instances,
                                    &ecs_world,
                                    &mut script_engine,
                                    &lua_syntax,
                                    &mut render_mode,
                                    &mut gizmo_mode,
                                    &mut camera_projection_mode,
                                    &mut camera_near,
                                    &mut camera_far,
                                    &mut camera_fov_radians,
                                    &mut camera_ortho_height,
                                    &mut sun_azimuth_deg,
                                    &mut sun_elevation_deg,
                                    &mut sun_intensity,
                                    &mut lua_editor_entity,
                                    &mut lua_editor_path,
                                    &mut lua_editor_text,
                                    &mut lua_editor_status,
                                    &mut object_material_names,
                                    &mut material_library,
                                    &mut material_editor,
                                    &mut material_runtime_overrides,
                                    &mut accumulation_dirty,
                                );
                                }

                            maybe_build_stress_scene(
                                &mut stress_test_requested,
                                &mut mesh,
                                &mut model_idx,
                                &mut mesh_instances,
                                &mut stress_instance_count,
                                &mut gpu_mesh_dirty,
                                &mut geometry_dirty,
                                &mut accumulation_dirty,
                                &mut project_status,
                                mesh_assets.len(),
                            );

                            let sun_lamp_pos = sun_empty_position;
                            let primary_sun_intensity = light_instances
                                .iter()
                                .find(|l| l.object_id == sun_obj_id)
                                .map(|l| l.intensity)
                                .unwrap_or(sun_intensity)
                                .max(0.0);
                            let old_light = uniforms.light_pos;
                            let old_intensity = uniforms.sun_intensity;
                            update_sun_lights(
                                &mut uniforms,
                                main_db,
                                decanter_scene_id,
                                current_scene_exists,
                                active_center,
                                sun_lamp_pos,
                                primary_sun_intensity,
                                &light_instances,
                                &mut sun_azimuth_deg,
                                &mut sun_elevation_deg,
                            );

                            let view = camera.view_matrix();
                            let projection = camera_projection_matrix(
                                camera_projection_mode,
                                camera_fov_radians,
                                camera_ortho_height,
                                camera_near,
                                camera_far,
                                config.width,
                                config.height,
                            );
                            let gizmo_projection = gizmo_projection_matrix(
                                camera_projection_mode,
                                camera_fov_radians,
                                camera_ortho_height,
                                camera_near,
                                camera_far,
                                config.width,
                                config.height,
                            );
                            if show_editor_ui {
                            let pixels_per_point = ctx.pixels_per_point().max(1.0);
                            let screen_rect = ctx.input(|i| i.screen_rect());
                            let display_size = [screen_rect.width().max(1.0), screen_rect.height().max(1.0)];
                            let pointer_pos = ctx
                                .input(|i| i.pointer.hover_pos())
                                .map(|p| [p.x, p.y])
                                .unwrap_or([mouse_pos[0] / pixels_per_point, mouse_pos[1] / pixels_per_point]);
                            let pointer_captured = ctx.is_pointer_over_area();
                            let viewport = Rect::from_min_max(
                                [0.0, 0.0].into(),
                                [display_size[0].max(1.0), display_size[1].max(1.0)].into(),
                            );
                            let interaction = GizmoInteraction {
                                cursor_pos: (pointer_pos[0], pointer_pos[1]),
                                hovered: !pointer_captured,
                                drag_started: mouse_left_clicked,
                                dragging: mouse_left_down,
                            };
                            let gizmo_modes = match gizmo_mode {
                                GizmoModeKind::Translate => GizmoMode::all_translate(),
                                GizmoModeKind::Rotate => GizmoMode::all_rotate(),
                                GizmoModeKind::Scale => GizmoMode::all_scale(),
                            };
                            let view_cols = view.to_cols_array();
                            let proj_cols = gizmo_projection.to_cols_array();
                            let view_matrix = transform_gizmo::math::DMat4::from_cols_array(&[
                                view_cols[0] as f64,
                                view_cols[1] as f64,
                                view_cols[2] as f64,
                                view_cols[3] as f64,
                                view_cols[4] as f64,
                                view_cols[5] as f64,
                                view_cols[6] as f64,
                                view_cols[7] as f64,
                                view_cols[8] as f64,
                                view_cols[9] as f64,
                                view_cols[10] as f64,
                                view_cols[11] as f64,
                                view_cols[12] as f64,
                                view_cols[13] as f64,
                                view_cols[14] as f64,
                                view_cols[15] as f64,
                            ]);
                            let projection_matrix = transform_gizmo::math::DMat4::from_cols_array(
                                &[
                                    proj_cols[0] as f64,
                                    proj_cols[1] as f64,
                                    proj_cols[2] as f64,
                                    proj_cols[3] as f64,
                                    proj_cols[4] as f64,
                                    proj_cols[5] as f64,
                                    proj_cols[6] as f64,
                                    proj_cols[7] as f64,
                                    proj_cols[8] as f64,
                                    proj_cols[9] as f64,
                                    proj_cols[10] as f64,
                                    proj_cols[11] as f64,
                                    proj_cols[12] as f64,
                                    proj_cols[13] as f64,
                                    proj_cols[14] as f64,
                                    proj_cols[15] as f64,
                                ],
                            );
                            gizmo.update_config(GizmoConfig {
                                view_matrix: view_matrix.into(),
                                projection_matrix: projection_matrix.into(),
                                viewport,
                                modes: gizmo_modes,
                                mode_override: None,
                                orientation: GizmoOrientation::Global,
                                pivot_point: TransformPivotPoint::MedianPoint,
                                snapping: false,
                                snap_angle: 15f32.to_radians(),
                                snap_distance: 0.5,
                                snap_scale: 0.1,
                                visuals: GizmoVisuals::default(),
                                pixels_per_point,
                            });

                            let target_transform = match gizmo_target {
                                GizmoTargetKind::Sphere => GizmoTransform::from_scale_rotation_translation(
                                    transform_gizmo::math::DVec3::new(
                                        sphere_scale.x as f64,
                                        sphere_scale.y as f64,
                                        sphere_scale.z as f64,
                                    ),
                                    transform_gizmo::math::DQuat::from_xyzw(
                                        sphere_rotation.x as f64,
                                        sphere_rotation.y as f64,
                                        sphere_rotation.z as f64,
                                        sphere_rotation.w as f64,
                                    ),
                                    transform_gizmo::math::DVec3::new(
                                        uniforms.sphere_pos[0] as f64,
                                        uniforms.sphere_pos[1] as f64,
                                        uniforms.sphere_pos[2] as f64,
                                    ),
                                ),
                                GizmoTargetKind::Decanter => {
                                    let inst = selected_object_id
                                        .and_then(|id| mesh_instances.iter().find(|inst| inst.object_id == id))
                                        .or_else(|| mesh_instances.iter().find(|inst| inst.object_id == decanter_obj_id));
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            inst.map(|i| i.scale.x).unwrap_or(decanter_scale.x) as f64,
                                            inst.map(|i| i.scale.y).unwrap_or(decanter_scale.y) as f64,
                                            inst.map(|i| i.scale.z).unwrap_or(decanter_scale.z) as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            inst.map(|i| i.rotation.x).unwrap_or(decanter_rotation.x) as f64,
                                            inst.map(|i| i.rotation.y).unwrap_or(decanter_rotation.y) as f64,
                                            inst.map(|i| i.rotation.z).unwrap_or(decanter_rotation.z) as f64,
                                            inst.map(|i| i.rotation.w).unwrap_or(decanter_rotation.w) as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            inst.map(|i| i.center().x).unwrap_or(decanter_center.x + decanter_translation.x) as f64,
                                            inst.map(|i| i.center().y).unwrap_or(decanter_center.y + decanter_translation.y) as f64,
                                            inst.map(|i| i.center().z).unwrap_or(decanter_center.z + decanter_translation.z) as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::WineGlass => {
                                    let inst = selected_object_id
                                        .and_then(|id| mesh_instances.iter().find(|inst| inst.object_id == id))
                                        .or_else(|| mesh_instances.iter().find(|inst| inst.object_id == wine_obj_id));
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            inst.map(|i| i.scale.x).unwrap_or(wine_scale.x) as f64,
                                            inst.map(|i| i.scale.y).unwrap_or(wine_scale.y) as f64,
                                            inst.map(|i| i.scale.z).unwrap_or(wine_scale.z) as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            inst.map(|i| i.rotation.x).unwrap_or(wine_rotation.x) as f64,
                                            inst.map(|i| i.rotation.y).unwrap_or(wine_rotation.y) as f64,
                                            inst.map(|i| i.rotation.z).unwrap_or(wine_rotation.z) as f64,
                                            inst.map(|i| i.rotation.w).unwrap_or(wine_rotation.w) as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            inst.map(|i| i.center().x).unwrap_or(wine_center.x + wine_translation.x) as f64,
                                            inst.map(|i| i.center().y).unwrap_or(wine_center.y + wine_translation.y) as f64,
                                            inst.map(|i| i.center().z).unwrap_or(wine_center.z + wine_translation.z) as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::CornellBox => {
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            cornell_scale.x as f64,
                                            cornell_scale.y as f64,
                                            cornell_scale.z as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            cornell_rotation.x as f64,
                                            cornell_rotation.y as f64,
                                            cornell_rotation.z as f64,
                                            cornell_rotation.w as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            (active_center.x + cornell_translation.x) as f64,
                                            (active_center.y + cornell_translation.y) as f64,
                                            (active_center.z + cornell_translation.z) as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::SunLamp => {
                                    let light = selected_object_id
                                        .and_then(|id| light_instances.iter().find(|l| l.object_id == id))
                                        .or_else(|| light_instances.iter().find(|l| l.object_id == sun_obj_id));
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            light.map(|l| l.scale.x).unwrap_or(sun_empty_scale.x) as f64,
                                            light.map(|l| l.scale.y).unwrap_or(sun_empty_scale.y) as f64,
                                            light.map(|l| l.scale.z).unwrap_or(sun_empty_scale.z) as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            light.map(|l| l.rotation.x).unwrap_or(sun_empty_rotation.x) as f64,
                                            light.map(|l| l.rotation.y).unwrap_or(sun_empty_rotation.y) as f64,
                                            light.map(|l| l.rotation.z).unwrap_or(sun_empty_rotation.z) as f64,
                                            light.map(|l| l.rotation.w).unwrap_or(sun_empty_rotation.w) as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            light.map(|l| l.position.x).unwrap_or(sun_lamp_pos.x) as f64,
                                            light.map(|l| l.position.y).unwrap_or(sun_lamp_pos.y) as f64,
                                            light.map(|l| l.position.z).unwrap_or(sun_lamp_pos.z) as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::WineSpotlight => {
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            spot_empty_scale.x as f64,
                                            spot_empty_scale.y as f64,
                                            spot_empty_scale.z as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            spot_empty_rotation.x as f64,
                                            spot_empty_rotation.y as f64,
                                            spot_empty_rotation.z as f64,
                                            spot_empty_rotation.w as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            spot_empty_position.x as f64,
                                            spot_empty_position.y as f64,
                                            spot_empty_position.z as f64,
                                        ),
                                    )
                                }
                                GizmoTargetKind::Camera => {
                                    let transform = selected_object_id
                                        .and_then(|id| ecs_world.borrow().global_transforms.get(&id).cloned());
                                    GizmoTransform::from_scale_rotation_translation(
                                        transform_gizmo::math::DVec3::new(
                                            transform.as_ref().map(|t| t.scale.x).unwrap_or(1.0) as f64,
                                            transform.as_ref().map(|t| t.scale.y).unwrap_or(1.0) as f64,
                                            transform.as_ref().map(|t| t.scale.z).unwrap_or(1.0) as f64,
                                        ),
                                        transform_gizmo::math::DQuat::from_xyzw(
                                            transform.as_ref().map(|t| t.rotation.x).unwrap_or(0.0) as f64,
                                            transform.as_ref().map(|t| t.rotation.y).unwrap_or(0.0) as f64,
                                            transform.as_ref().map(|t| t.rotation.z).unwrap_or(0.0) as f64,
                                            transform.as_ref().map(|t| t.rotation.w).unwrap_or(1.0) as f64,
                                        ),
                                        transform_gizmo::math::DVec3::new(
                                            transform.as_ref().map(|t| t.translation.x).unwrap_or(camera.pos.x) as f64,
                                            transform.as_ref().map(|t| t.translation.y).unwrap_or(camera.pos.y) as f64,
                                            transform.as_ref().map(|t| t.translation.z).unwrap_or(camera.pos.z) as f64,
                                        ),
                                    )
                                }
                            };

                            if has_selection {
                                if let Some((_result, transforms)) =
                                    gizmo.update(interaction, &[target_transform])
                                {
                                    let new_t = transforms[0];
                                    let mut translation = glam::Vec3::new(
                                        new_t.translation.x as f32,
                                        new_t.translation.y as f32,
                                        new_t.translation.z as f32,
                                    );
                                    // If Shift is held, make gizmo translations faster (scale deltas)
                                    if keys_pressed.contains("Shift") {
                                        match gizmo_target {
                                            GizmoTargetKind::Sphere => {
                                                let cur = glam::Vec3::new(
                                                    uniforms.sphere_pos[0],
                                                    uniforms.sphere_pos[1],
                                                    uniforms.sphere_pos[2],
                                                );
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::Decanter => {
                                                let cur = selected_object_id
                                                    .and_then(|id| mesh_instances.iter().find(|inst| inst.object_id == id))
                                                    .map(|inst| inst.center())
                                                    .unwrap_or(decanter_center + decanter_translation);
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::WineGlass => {
                                                let cur = selected_object_id
                                                    .and_then(|id| mesh_instances.iter().find(|inst| inst.object_id == id))
                                                    .map(|inst| inst.center())
                                                    .unwrap_or(wine_center + wine_translation);
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::CornellBox => {
                                                let cur = active_center + cornell_translation;
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::SunLamp => {
                                                let cur = selected_object_id
                                                    .and_then(|id| light_instances.iter().find(|l| l.object_id == id))
                                                    .map(|l| l.position)
                                                    .unwrap_or(sun_empty_position);
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::WineSpotlight => {
                                                let cur = spot_empty_position;
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                            GizmoTargetKind::Camera => {
                                                let cur = selected_object_id
                                                    .and_then(|id| ecs_world.borrow().global_transforms.get(&id).cloned())
                                                    .map(|t| t.translation)
                                                    .unwrap_or(camera.pos);
                                                translation = cur + (translation - cur) * 3.0;
                                            }
                                        }
                                    }
                                    match gizmo_target {
                                    GizmoTargetKind::Sphere => {
                                        uniforms.sphere_pos[0] = translation.x;
                                        uniforms.sphere_pos[1] = translation.y;
                                        uniforms.sphere_pos[2] = translation.z;
                                        let sx = new_t.scale.x.abs() as f32;
                                        let sy = new_t.scale.y.abs() as f32;
                                        let sz = new_t.scale.z.abs() as f32;
                                        sphere_scale = glam::Vec3::new(
                                            sx.clamp(0.15, 8.0),
                                            sy.clamp(0.15, 8.0),
                                            sz.clamp(0.15, 8.0),
                                        );
                                        uniforms.sphere_extent = [
                                            sphere_radius * sphere_scale.x,
                                            sphere_radius * sphere_scale.y,
                                            sphere_radius * sphere_scale.z,
                                            0.0,
                                        ];
                                        sphere_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        uniforms.sphere_rot = [
                                            sphere_rotation.x,
                                            sphere_rotation.y,
                                            sphere_rotation.z,
                                            sphere_rotation.w,
                                        ];
                                    }
                                    GizmoTargetKind::Decanter => {
                                        let new_center = glam::Vec3::new(translation.x, translation.y, translation.z);
                                        let new_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        let new_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                        if let Some(inst) = selected_object_id
                                            .and_then(|id| mesh_instances.iter_mut().find(|inst| inst.object_id == id))
                                        {
                                            inst.translation = new_center - inst.pivot;
                                            inst.rotation = new_rotation;
                                            inst.scale = new_scale;
                                            if inst.object_id == decanter_obj_id {
                                                decanter_translation = inst.translation;
                                                decanter_rotation = inst.rotation;
                                                decanter_scale = inst.scale;
                                            }
                                        } else {
                                            decanter_translation = new_center - decanter_center;
                                            decanter_rotation = new_rotation;
                                            decanter_scale = new_scale;
                                        }
                                        geometry_dirty = true;
                                    }
                                    GizmoTargetKind::WineGlass => {
                                        let new_center = glam::Vec3::new(translation.x, translation.y, translation.z);
                                        let new_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        let new_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                        if let Some(inst) = selected_object_id
                                            .and_then(|id| mesh_instances.iter_mut().find(|inst| inst.object_id == id))
                                        {
                                            inst.translation = new_center - inst.pivot;
                                            inst.rotation = new_rotation;
                                            inst.scale = new_scale;
                                            if inst.object_id == wine_obj_id {
                                                wine_translation = inst.translation;
                                                wine_rotation = inst.rotation;
                                                wine_scale = inst.scale;
                                            }
                                        } else {
                                            wine_translation = new_center - wine_center;
                                            wine_rotation = new_rotation;
                                            wine_scale = new_scale;
                                        }
                                        geometry_dirty = true;
                                    }
                                    GizmoTargetKind::CornellBox => {
                                        cornell_translation = translation - active_center;
                                        cornell_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        cornell_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                    }
                                    GizmoTargetKind::SunLamp => {
                                        if let Some(light) = selected_object_id
                                            .and_then(|id| light_instances.iter_mut().find(|l| l.object_id == id))
                                        {
                                            light.position = translation;
                                            light.rotation = glam::Quat::from_array([
                                                new_t.rotation.v.x as f32,
                                                new_t.rotation.v.y as f32,
                                                new_t.rotation.v.z as f32,
                                                new_t.rotation.s as f32,
                                            ]);
                                            light.scale = glam::Vec3::new(
                                                (new_t.scale.x as f32).clamp(0.1, 8.0),
                                                (new_t.scale.y as f32).clamp(0.1, 8.0),
                                                (new_t.scale.z as f32).clamp(0.1, 8.0),
                                            );
                                            sun_empty_position = light.position;
                                            sun_empty_rotation = light.rotation;
                                            sun_empty_scale = light.scale;
                                        } else {
                                            sun_empty_position = translation;
                                            sun_empty_rotation = glam::Quat::from_array([
                                                new_t.rotation.v.x as f32,
                                                new_t.rotation.v.y as f32,
                                                new_t.rotation.v.z as f32,
                                                new_t.rotation.s as f32,
                                            ]);
                                            sun_empty_scale = glam::Vec3::new(
                                                (new_t.scale.x as f32).clamp(0.1, 8.0),
                                                (new_t.scale.y as f32).clamp(0.1, 8.0),
                                                (new_t.scale.z as f32).clamp(0.1, 8.0),
                                            );
                                        }
                                        let to_sun = sun_empty_position - active_center;
                                        sun_lamp_distance = to_sun.length().max(1.0);
                                    }
                                    GizmoTargetKind::WineSpotlight => {
                                        spot_empty_position = translation;
                                        spot_empty_rotation = glam::Quat::from_array([
                                            new_t.rotation.v.x as f32,
                                            new_t.rotation.v.y as f32,
                                            new_t.rotation.v.z as f32,
                                            new_t.rotation.s as f32,
                                        ]);
                                        spot_empty_scale = glam::Vec3::new(
                                            (new_t.scale.x as f32).clamp(0.1, 8.0),
                                            (new_t.scale.y as f32).clamp(0.1, 8.0),
                                            (new_t.scale.z as f32).clamp(0.1, 8.0),
                                        );
                                    }
                                    GizmoTargetKind::Camera => {
                                        if let Some(id) = selected_object_id {
                                            let rotation = glam::Quat::from_array([
                                                new_t.rotation.v.x as f32,
                                                new_t.rotation.v.y as f32,
                                                new_t.rotation.v.z as f32,
                                                new_t.rotation.s as f32,
                                            ]);
                                            let scale = glam::Vec3::new(
                                                (new_t.scale.x as f32).clamp(0.1, 8.0),
                                                (new_t.scale.y as f32).clamp(0.1, 8.0),
                                                (new_t.scale.z as f32).clamp(0.1, 8.0),
                                            );
                                            let mut world = ecs_world.borrow_mut();
                                            let transform = world.transforms.entry(id).or_default();
                                            transform.translation = translation;
                                            transform.rotation = rotation;
                                            transform.scale = scale;
                                            world.update_global_transforms_and_visibility();
                                            drop(world);
                                            if let Some(obj) = main_db.objects.get_mut(&id) {
                                                obj.transform.location = translation;
                                                obj.transform.rotation = rotation;
                                                obj.transform.scale = scale;
                                            }
                                        }
                                    }
                                    }
                                    accumulation_dirty = true;
                                }
                            }

                            if mouse_left_clicked
                                && !suppress_scene_click
                                && !pointer_captured
                                && !gizmo.is_focused()
                                && !mouse_left_dragging
                            {
                                let scene_id = decanter_scene_id;
                                let selectable_ids = main_db.scene_visible_selectable_objects(scene_id);
                                let sphere_allowed = selectable_ids.contains(&sphere_obj_id);
                                let cornell_allowed = selectable_ids.contains(&cornell_obj_id);
                                let spot_allowed = selectable_ids.contains(&spot_obj_id);
                                let (ro, rd) = world_ray_from_cursor(
                                    pointer_pos,
                                    [display_size[0].max(1.0), display_size[1].max(1.0)],
                                    camera.view_matrix().inverse(),
                                    projection.inverse(),
                                );
                                let sphere_center = glam::Vec3::new(
                                    uniforms.sphere_pos[0],
                                    uniforms.sphere_pos[1],
                                    uniforms.sphere_pos[2],
                                );
                                let sphere_hit = if sphere_allowed {
                                    intersect_cube(
                                        ro,
                                        rd,
                                        sphere_center,
                                        glam::Vec3::new(
                                            uniforms.sphere_extent[0],
                                            uniforms.sphere_extent[1],
                                            uniforms.sphere_extent[2],
                                        ),
                                    )
                                } else {
                                    None
                                };
                                let cornell_hit = if cornell_allowed {
                                    intersect_sphere(
                                        ro,
                                        rd,
                                        active_center + cornell_translation,
                                        (2.0 * cornell_scale.max_element()).max(0.25),
                                    )
                                } else {
                                    None
                                };
                                let spot_hit = if spot_allowed {
                                    intersect_sphere(ro, rd, spot_empty_position, 1.2)
                                } else {
                                    None
                                };
                                let mut best: Option<(GizmoTargetKind, Id)> = None;
                                let mut best_t = f32::INFINITY;
                                if let Some(t) = sphere_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = Some((GizmoTargetKind::Sphere, sphere_obj_id));
                                    }
                                }
                                for inst in &mesh_instances {
                                    if !selectable_ids.contains(&inst.object_id) {
                                        continue;
                                    }
                                    let Some(target) = object_target_by_id.get(&inst.object_id).copied() else {
                                        continue;
                                    };
                                    if matches!(target, GizmoTargetKind::SunLamp | GizmoTargetKind::WineSpotlight) {
                                        continue;
                                    }
                                    if let Some(t) = intersect_sphere(
                                        ro,
                                        rd,
                                        inst.center(),
                                        (inst.max_extent * inst.scale.max_element() * 0.55).max(0.25),
                                    ) {
                                        if t < best_t {
                                            best_t = t;
                                            best = Some((target, inst.object_id));
                                        }
                                    }
                                }
                                if let Some(t) = cornell_hit {
                                    if t < best_t {
                                        best_t = t;
                                        best = Some((GizmoTargetKind::CornellBox, cornell_obj_id));
                                    }
                                }
                                for light in &light_instances {
                                    if !selectable_ids.contains(&light.object_id) {
                                        continue;
                                    }
                                    if let Some(t) = intersect_sphere(ro, rd, light.position, 1.2) {
                                        if t < best_t {
                                            best_t = t;
                                            best = Some((GizmoTargetKind::SunLamp, light.object_id));
                                        }
                                    }
                                }
                                if let Some(t) = spot_hit {
                                    if t < best_t {
                                        best = Some((GizmoTargetKind::WineSpotlight, spot_obj_id));
                                    }
                                }
                                if let Some(camera_transform) = ecs_world
                                    .borrow()
                                    .global_transforms
                                    .get(&player_camera_obj_id)
                                    .cloned()
                                {
                                    if let Some(t) = intersect_sphere(
                                        ro,
                                        rd,
                                        camera_transform.translation,
                                        0.85,
                                    ) {
                                        if t < best_t {
                                            best = Some((GizmoTargetKind::Camera, player_camera_obj_id));
                                        }
                                    }
                                }
                                if let Some((selected, object_id)) = best {
                                    gizmo_target = selected;
                                    selected_object_id = Some(object_id);
                                    has_selection = true;
                                } else {
                                    has_selection = false;
                                    selected_object_id = None;
                                }
                            }

                            if has_selection {
                                let draw_data = gizmo.draw();
                                let painter = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("gizmo_overlay"),
                                ));
                                for idx in (0..draw_data.indices.len()).step_by(3) {
                                    let i0 = draw_data.indices[idx] as usize;
                                    let i1 = draw_data.indices[idx + 1] as usize;
                                    let i2 = draw_data.indices[idx + 2] as usize;
                                    let p0 = draw_data.vertices[i0];
                                    let p1 = draw_data.vertices[i1];
                                    let p2 = draw_data.vertices[i2];
                                    let c = draw_data.colors[i0];
                                    let color = Color32::from_rgba_unmultiplied(
                                        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                                        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                                        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
                                        (c[3].clamp(0.0, 1.0) * 255.0) as u8,
                                    );
                                    painter.add(egui::Shape::convex_polygon(
                                        vec![
                                            Pos2::new(p0[0], p0[1]),
                                            Pos2::new(p1[0], p1[1]),
                                            Pos2::new(p2[0], p2[1]),
                                        ],
                                        color,
                                        Stroke::NONE,
                                    ));
                                }

                                if current_scene_exists && scene_kind != SceneKind::Wine {
                                    let display = [display_size[0].max(1.0), display_size[1].max(1.0)];
                                    let center_screen = world_to_screen(active_center, view, projection, display);
                                    if let Some(cn) = center_screen {
                                        for light in &light_instances {
                                            let Some(s) = world_to_screen(light.position, view, projection, display) else {
                                                continue;
                                            };
                                            let selected = selected_object_id == Some(light.object_id);
                                            let line_color = if selected {
                                                Color32::from_rgb(255, 158, 38)
                                            } else {
                                                Color32::from_rgb(255, 242, 153)
                                            };
                                            painter.line_segment(
                                                [Pos2::new(s[0], s[1]), Pos2::new(cn[0], cn[1])],
                                                Stroke::new(2.0, line_color),
                                            );
                                            painter.circle_stroke(Pos2::new(s[0], s[1]), 8.0, Stroke::new(2.0, line_color));
                                            painter.circle_filled(Pos2::new(s[0], s[1]), 3.0, line_color);
                                        }
                                    }
                                } else if current_scene_exists {
                                    let display = [display_size[0].max(1.0), display_size[1].max(1.0)];
                                    let spot_screen =
                                        world_to_screen(spot_empty_position, view, projection, display);
                                    let target_screen = world_to_screen(active_center, view, projection, display);
                                    if let (Some(s), Some(tg)) = (spot_screen, target_screen) {
                                        let selected = gizmo_target == GizmoTargetKind::WineSpotlight;
                                        let line_color = if selected {
                                            Color32::from_rgb(255, 158, 38)
                                        } else {
                                            Color32::from_rgb(255, 242, 153)
                                        };
                                        painter.line_segment(
                                            [Pos2::new(s[0], s[1]), Pos2::new(tg[0], tg[1])],
                                            Stroke::new(2.0, line_color),
                                        );
                                        painter.circle_stroke(Pos2::new(s[0], s[1]), 8.0, Stroke::new(2.0, line_color));
                                        painter.circle_filled(Pos2::new(s[0], s[1]), 3.0, line_color);
                                    }
                                }
                                if current_scene_exists {
                                    let display = [display_size[0].max(1.0), display_size[1].max(1.0)];
                                    let world_ref = ecs_world.borrow();
                                    for (camera_id, _) in &world_ref.cameras {
                                        let Some(transform) = world_ref.global_transforms.get(camera_id) else {
                                            continue;
                                        };
                                        let Some(s) = world_to_screen(transform.translation, view, projection, display) else {
                                            continue;
                                        };
                                        let selected = selected_object_id == Some(*camera_id);
                                        let color = if selected {
                                            Color32::from_rgb(86, 190, 255)
                                        } else {
                                            Color32::from_rgb(135, 220, 255)
                                        };
                                        let p = Pos2::new(s[0], s[1]);
                                        painter.add(egui::Shape::convex_polygon(
                                            vec![
                                                Pos2::new(p.x, p.y - 11.0),
                                                Pos2::new(p.x + 12.0, p.y),
                                                Pos2::new(p.x, p.y + 11.0),
                                                Pos2::new(p.x - 12.0, p.y),
                                            ],
                                            Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 36),
                                            Stroke::new(2.0, color),
                                        ));
                                        painter.line_segment(
                                            [p, Pos2::new(p.x + 16.0, p.y - 10.0)],
                                            Stroke::new(2.0, color),
                                        );
                                        painter.line_segment(
                                            [p, Pos2::new(p.x + 16.0, p.y + 10.0)],
                                            Stroke::new(2.0, color),
                                        );
                                    }
                                }
                            }
                            if !has_selection && current_scene_exists {
                                let painter = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("always_visible_scene_icons"),
                                ));
                                let display = [display_size[0].max(1.0), display_size[1].max(1.0)];
                                let center_screen = world_to_screen(active_center, view, projection, display);
                                for light in &light_instances {
                                    let Some(s) = world_to_screen(light.position, view, projection, display) else {
                                        continue;
                                    };
                                    let color = Color32::from_rgb(255, 242, 153);
                                    if let Some(cn) = center_screen {
                                        painter.line_segment(
                                            [Pos2::new(s[0], s[1]), Pos2::new(cn[0], cn[1])],
                                            Stroke::new(1.5, color),
                                        );
                                    }
                                    painter.circle_stroke(
                                        Pos2::new(s[0], s[1]),
                                        8.0,
                                        Stroke::new(2.0, color),
                                    );
                                    painter.circle_filled(Pos2::new(s[0], s[1]), 3.0, color);
                                }
                                let world_ref = ecs_world.borrow();
                                for (camera_id, _) in &world_ref.cameras {
                                    let Some(transform) = world_ref.global_transforms.get(camera_id) else {
                                        continue;
                                    };
                                    let Some(s) = world_to_screen(transform.translation, view, projection, display) else {
                                        continue;
                                    };
                                    let color = Color32::from_rgb(135, 220, 255);
                                    let p = Pos2::new(s[0], s[1]);
                                    painter.add(egui::Shape::convex_polygon(
                                        vec![
                                            Pos2::new(p.x, p.y - 11.0),
                                            Pos2::new(p.x + 12.0, p.y),
                                            Pos2::new(p.x, p.y + 11.0),
                                            Pos2::new(p.x - 12.0, p.y),
                                        ],
                                        Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 36),
                                        Stroke::new(2.0, color),
                                    ));
                                    painter.line_segment(
                                        [p, Pos2::new(p.x + 16.0, p.y - 10.0)],
                                        Stroke::new(2.0, color),
                                    );
                                    painter.line_segment(
                                        [p, Pos2::new(p.x + 16.0, p.y + 10.0)],
                                        Stroke::new(2.0, color),
                                    );
                                }
                            }
                            }
                            uniforms.sun_intensity = sun_intensity.max(0.0);
                            uniforms.scene_kind = if current_scene_exists {
                                scene_kind.index()
                            } else {
                                99
                            };
                            uniforms.mesh_center = [
                                wine_center.x + wine_translation.x,
                                wine_center.y + wine_translation.y,
                                wine_center.z + wine_translation.z,
                                wine_max_extent * wine_scale.max_element() * 0.8,
                            ];
                            uniforms.decanter_center = [
                                decanter_center.x + decanter_translation.x,
                                decanter_center.y + decanter_translation.y,
                                decanter_center.z + decanter_translation.z,
                                decanter_max_extent * decanter_scale.max_element() * 0.7,
                            ];
                            uniforms.cornell_center = [
                                active_center.x + cornell_translation.x,
                                active_center.y + cornell_translation.y,
                                active_center.z + cornell_translation.z,
                                cornell_scale.max_element().max(0.1),
                            ];
                            if let Some(selected_inst) = selected_object_id
                                .and_then(|id| mesh_instances.iter().find(|inst| inst.object_id == id))
                            {
                                let center = selected_inst.center();
                                uniforms.decanter_center = [
                                    center.x,
                                    center.y,
                                    center.z,
                                    selected_inst.max_extent * selected_inst.scale.max_element() * 0.9,
                                ];
                                if gizmo_target == GizmoTargetKind::WineGlass {
                                    uniforms.mesh_center = [
                                        center.x,
                                        center.y,
                                        center.z,
                                        selected_inst.max_extent * selected_inst.scale.max_element() * 0.8,
                                    ];
                                }
                            }
                            uniforms.selected_object = if has_selection {
                                match gizmo_target {
                                    GizmoTargetKind::Sphere => 1,
                                    GizmoTargetKind::Decanter => 3,
                                    GizmoTargetKind::WineGlass => 2,
                                    GizmoTargetKind::CornellBox => 4,
                                    GizmoTargetKind::SunLamp => 0,
                                    GizmoTargetKind::WineSpotlight => 0,
                                    GizmoTargetKind::Camera => 0,
                                }
                            } else {
                                0
                            };
                            let active_scene_id = decanter_scene_id;
                            let (sphere_visible, decanter_visible, wine_visible, cornell_visible) = if current_scene_exists {
                                let visible = main_db.scene_visible_selectable_objects(active_scene_id);
                                (
                                    visible.contains(&sphere_obj_id),
                                    visible.contains(&decanter_obj_id),
                                    visible.contains(&wine_obj_id),
                                    visible.contains(&cornell_obj_id),
                                )
                            } else {
                                (false, false, false, false)
                            };
                            sphere_visible_for_photons = sphere_visible;
                            uniforms.decanter_enabled = if decanter_visible { 1 } else { 0 };
                            uniforms.wine_enabled = if wine_visible { 1 } else { 0 };
                            uniforms.cornell_enabled = if cornell_visible { 1 } else { 0 };
                            let any_mesh_visible = if stress_instance_count > 0 {
                                true
                            } else if current_scene_exists {
                                let visible = main_db.scene_visible_selectable_objects(active_scene_id);
                                mesh_instances.iter().any(|inst| visible.contains(&inst.object_id))
                            } else {
                                false
                            };
                            uniforms.mesh_enabled = if any_mesh_visible { 1 } else { 0 };
                            let sphere_mat = object_material_names
                                .get(&sphere_obj_id)
                                .cloned()
                                .unwrap_or_else(|| "White".to_string());
                            let sphere_preview = material_runtime_overrides
                                .get(&sphere_mat)
                                .copied()
                                .unwrap_or_else(|| preview_from_material_data(material_library.get(&sphere_mat)));
                            uniforms.sphere_color = [
                                sphere_preview.base_color[0],
                                sphere_preview.base_color[1],
                                sphere_preview.base_color[2],
                                if sphere_preview.bsdf_connected {
                                    sphere_preview.transmission
                                } else {
                                    0.0
                                },
                            ];
                            uniforms.sphere_params = [
                                sphere_preview.roughness,
                                sphere_preview.ior,
                                if sphere_preview.bsdf_connected { 1.0 } else { 0.0 },
                                0.0,
                            ];

                            let decanter_mat = object_material_names
                                .get(&decanter_obj_id)
                                .cloned()
                                .unwrap_or_else(|| "White".to_string());
                            let decanter_preview = material_runtime_overrides
                                .get(&decanter_mat)
                                .copied()
                                .unwrap_or_else(|| preview_from_material_data(material_library.get(&decanter_mat)));
                            let wine_mat = object_material_names
                                .get(&wine_obj_id)
                                .cloned()
                                .unwrap_or_else(|| "White".to_string());
                            let wine_preview = material_runtime_overrides
                                .get(&wine_mat)
                                .copied()
                                .unwrap_or_else(|| preview_from_material_data(material_library.get(&wine_mat)));
                            let cornell_mat = object_material_names
                                .get(&cornell_obj_id)
                                .cloned()
                                .unwrap_or_else(|| "White".to_string());
                            let cornell_preview = material_runtime_overrides
                                .get(&cornell_mat)
                                .copied()
                                .unwrap_or_else(|| preview_from_material_data(material_library.get(&cornell_mat)));
                            uniforms.cornell_color = [
                                cornell_preview.base_color[0],
                                cornell_preview.base_color[1],
                                cornell_preview.base_color[2],
                                if cornell_preview.bsdf_connected {
                                    cornell_preview.transmission
                                } else {
                                    0.0
                                },
                            ];
                            uniforms.cornell_params = [
                                cornell_preview.roughness,
                                cornell_preview.ior,
                                if cornell_preview.bsdf_connected { 1.0 } else { 0.0 },
                                0.0,
                            ];
                            if stress_instance_count == 0 {
                                let mut material_signature = format!(
                                    "{sphere_mat}:{sphere_preview:?}|{decanter_mat}:{decanter_preview:?}|{wine_mat}:{wine_preview:?}|{cornell_mat}:{cornell_preview:?}"
                                );
                                for inst in &mesh_instances {
                                    let mat = object_material_names
                                        .get(&inst.object_id)
                                        .cloned()
                                        .unwrap_or_else(|| "Glass".to_string());
                                    material_signature.push_str(&format!("|{}:{mat}", inst.object_id.0));
                                }
                                if material_signature != last_material_signature {
                                let set_range = |materials: &mut [crate::mesh::GpuMaterial],
                                                 start: usize,
                                                 count: usize,
                                                 preview: RuntimeMaterialPreview,
                                                 wine_style: bool| {
                                    let end = (start + count).min(materials.len());
                                    for m in &mut materials[start..end] {
                                        if preview.bsdf_connected {
                                            if wine_style {
                                                m.base_color = [
                                                    preview.base_color[0],
                                                    preview.base_color[1],
                                                    preview.base_color[2],
                                                    0.78,
                                                ];
                                                m.params = [
                                                    0.0,
                                                    preview.roughness.min(0.06),
                                                    preview.transmission.max(0.72),
                                                    preview.ior.max(1.0),
                                                ];
                                            } else {
                                                m.base_color = [
                                                    preview.base_color[0],
                                                    preview.base_color[1],
                                                    preview.base_color[2],
                                                    1.0,
                                                ];
                                                m.params = [
                                                    0.0,
                                                    preview.roughness,
                                                    preview.transmission,
                                                    preview.ior,
                                                ];
                                            }
                                        } else {
                                            m.base_color = [1.0, 1.0, 1.0, 1.0];
                                            m.params = [0.0, 0.65, 0.0, 1.0];
                                        }
                                    }
                                };
                                set_range(
                                    &mut mesh.materials,
                                    decanter_material_start,
                                    decanter_material_count,
                                    decanter_preview,
                                    false,
                                );
                                set_range(
                                    &mut mesh.materials,
                                    wine_material_start,
                                    wine_material_count,
                                    wine_preview,
                                    true,
                                );
                                for inst in &mesh_instances {
                                    let mat = object_material_names
                                        .get(&inst.object_id)
                                        .cloned()
                                        .unwrap_or_else(|| "Glass".to_string());
                                    let preview = material_runtime_overrides
                                        .get(&mat)
                                        .copied()
                                        .unwrap_or_else(|| preview_from_material_data(material_library.get(&mat)));
                                    let wine_style = object_target_by_id
                                        .get(&inst.object_id)
                                        .copied()
                                        == Some(GizmoTargetKind::WineGlass);
                                    set_range(
                                        &mut mesh.materials,
                                        inst.material_start,
                                        inst.material_count,
                                        preview,
                                        wine_style,
                                    );
                                }
                                if !gpu_mesh_dirty {
                                    queue.write_buffer(
                                        &mat_buf,
                                        0,
                                        bytemuck::cast_slice(&mesh.materials),
                                    );
                                }
                                    last_material_signature = material_signature;
                                    accumulation_dirty = true;
                                }
                            }
                            if current_scene_exists && stress_instance_count == 0 {
                                let visible_ids = main_db.scene_visible_selectable_objects(active_scene_id);
                                let mut visible_mesh_instances: Vec<&MeshObjectInstance> = mesh_instances
                                    .iter()
                                    .filter(|inst| visible_ids.contains(&inst.object_id))
                                    .collect();
                                if !visible_mesh_instances.is_empty() {
                                    let mut min_p = glam::Vec3::splat(f32::INFINITY);
                                    let mut max_p = glam::Vec3::splat(f32::NEG_INFINITY);
                                    for inst in &visible_mesh_instances {
                                        let c = inst.center();
                                        let r = (inst.max_extent * inst.scale.max_element() * 0.6).max(0.25);
                                        let e = glam::Vec3::splat(r);
                                        min_p = min_p.min(c - e);
                                        max_p = max_p.max(c + e);
                                    }
                                    let c = (min_p + max_p) * 0.5;
                                    let r = ((max_p - min_p).length() * 0.6).max(0.5);
                                    photon_emitter_center = [c.x, c.y, c.z, r];
                                    photons_per_frame = 262_144;
                                }
                            }

                            if !start_play_requested {
                                write_runtime_back_to_database(
                                    main_db,
                                    sphere_obj_id,
                                    decanter_obj_id,
                                    wine_obj_id,
                                    sun_obj_id,
                                    spot_obj_id,
                                    cornell_obj_id,
                                    &uniforms,
                                    sphere_rotation,
                                    sphere_scale,
                                    decanter_center,
                                    decanter_translation,
                                    decanter_rotation,
                                    decanter_scale,
                                    wine_center,
                                    wine_translation,
                                    wine_rotation,
                                    wine_scale,
                                    stress_instance_count,
                                    &mesh_instances,
                                    sun_empty_position,
                                    sun_empty_rotation,
                                    sun_empty_scale,
                                    &light_instances,
                                    spot_empty_position,
                                    spot_empty_rotation,
                                    spot_empty_scale,
                                    active_center,
                                    cornell_translation,
                                    cornell_rotation,
                                    cornell_scale,
                                );
                            }

                            sun_changed = uniforms.light_pos != old_light
                                || (uniforms.sun_intensity - old_intensity).abs() > f32::EPSILON;
                            })
                            };
                            if start_play_requested {
                                play_db = Some(editor_db.clone());
                                show_editor_ui_before_play = show_editor_ui;
                                play_mode.start(
                                    &ecs_world,
                                    play_db.as_ref().expect("PIE db missing"),
                                    player_obj_id,
                                    player_camera_obj_id,
                                    &camera,
                                );
                                show_editor_ui = false;
                                window.set_cursor_visible(false);
                                let _ = window
                                    .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                    .or_else(|_| {
                                        window.set_cursor_grab(
                                            winit::window::CursorGrabMode::Confined,
                                        )
                                    });
                                accumulation_dirty = true;
                            }
                            let egui::FullOutput {
                                platform_output,
                                textures_delta,
                                shapes,
                                pixels_per_point,
                                ..
                            } = full_output;
                            egui_state.handle_platform_output(window.as_ref(), platform_output);
                            let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);

                            let main_db = if play_mode.active {
                                play_db.as_mut().expect("PIE db missing")
                            } else {
                                &mut editor_db
                            };

                            sync_accumulation_and_geometry(
                                &mut accumulation_dirty,
                                sun_changed,
                                &mut geometry_dirty,
                                &mut uniforms,
                                accum_byte_size,
                                &queue,
                                &ubuf,
                                &accum_buf,
                                &mut mesh,
                                &mut model_verts,
                                stress_instance_count,
                                &mesh_instances,
                                &mesh_assets,
                                main_db,
                                decanter_scene_id,
                                &mut model_idx,
                                &mut gpu_mesh_dirty,
                                &mut vbuf,
                                &mut ibuf,
                                &mut pos_buf,
                                &mut nrm_buf,
                                &mut idx_buf,
                                &mut tri_mat_buf,
                                &mut mat_buf,
                                &mut model_blas_desc,
                                &mut model_blas,
                                &mut tlas,
                                &mut photon_mapper,
                                &mut ugroup,
                                &device,
                                &ubind,
                                &compute_pass,
                            );
                            queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&uniforms));

                            render_frame_and_present(
                                &device,
                                &queue,
                                &config,
                                tex,
                                &mut egui_renderer,
                                &textures_delta,
                                &clipped_primitives,
                                pixels_per_point,
                                render_mode,
                                &mut uniforms,
                                photons_per_frame,
                                main_db,
                                decanter_scene_id,
                                &mesh_instances,
                                stress_instance_count,
                                &mut photon_mapper,
                                photon_emitter_center,
                                sphere_visible_for_photons,
                                &mut raster_pass,
                                &mut raster_instance_count,
                                &mut raster_instance_buf,
                                projection,
                                &camera,
                                &vbuf,
                                &ibuf,
                                model_idx.len(),
                                &compute_pass,
                                &ugroup,
                                &quad_pass,
                                &mut mouse_left_clicked,
                                mouse_left_down,
                                &mut mouse_left_dragging,
                            );
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            _ => {}
        }
    });
}
