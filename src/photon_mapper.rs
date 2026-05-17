use wgpu::util::DeviceExt;

const MAX_PHOTONS: u32 = 1_000_000;
const VOXEL_SIZE: f32 = 0.12;
const HASH_TABLE_SIZE: u32 = 1_048_576; // 2^20

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PhotonMapUniforms {
    light_pos: [f32; 4],
    emitter_center: [f32; 4],
    sphere_pos: [f32; 4],
    sphere_rot: [f32; 4],
    sphere_extent: [f32; 4],
    sphere_material: [f32; 4],
    sphere_enabled: [u32; 4],
    photon_count: u32,
    voxel_size: f32,
    hash_table_size: u32,
    frame: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Photon {
    position: [f32; 3],
    wavelength_nm: f32,
    direction: [f32; 3],
    power: f32,
    next: u32,
    _pad3: [u32; 3],
}

pub struct PhotonMapper {
    emission_pipeline: wgpu::ComputePipeline,
    emission_bind_group: wgpu::BindGroup,
    hash_pipeline: wgpu::ComputePipeline,
    hash_bind_group: wgpu::BindGroup,
    photon_buffer: wgpu::Buffer,
    photon_counter: wgpu::Buffer,
    hash_heads: wgpu::Buffer,
    uniforms_buffer: wgpu::Buffer,
    photon_count: u32,
}

impl PhotonMapper {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        tlas: &wgpu::Tlas,
        mesh_pos_buf: &wgpu::Buffer,
        mesh_nrm_buf: &wgpu::Buffer,
        mesh_idx_buf: &wgpu::Buffer,
        mesh_tri_mat_buf: &wgpu::Buffer,
        mesh_mat_buf: &wgpu::Buffer,
    ) -> Self {
        let photon_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon_buffer"),
            size: (MAX_PHOTONS as u64) * std::mem::size_of::<Photon>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let photon_counter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("photon_counter"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let hash_heads = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photon_hash_heads"),
            size: (HASH_TABLE_SIZE as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = PhotonMapUniforms {
            light_pos: [10.0, 8.0, 10.0, 1.0],
            emitter_center: [7.2949142, 15.422569, 0.0, 24.0],
            sphere_pos: [0.0, 0.0, 0.0, 1.0],
            sphere_rot: [0.0, 0.0, 0.0, 1.0],
            sphere_extent: [1.0, 1.0, 1.0, 0.0],
            sphere_material: [0.0, 1.0, 0.0, 0.0],
            sphere_enabled: [0; 4],
            photon_count: 0,
            voxel_size: VOXEL_SIZE,
            hash_table_size: HASH_TABLE_SIZE,
            frame: 0,
        };

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("photon_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let emission_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("photon_emission_layout"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, true),
            ],
        });

        let hash_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("photon_hash_layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        });

        let emission_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("photon_emission_bind_group"),
            layout: &emission_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::AccelerationStructure(tlas),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: photon_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: photon_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mesh_pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mesh_nrm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: mesh_idx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: mesh_tri_mat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: mesh_mat_buf.as_entire_binding(),
                },
            ],
        });

        let hash_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("photon_hash_bind_group"),
            layout: &hash_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: photon_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: hash_heads.as_entire_binding(),
                },
            ],
        });

        let emission_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("photon_emission"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/photon_emission.wgsl").into(),
            ),
        });
        let hash_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("photon_hash"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/photon_hash.wgsl").into()),
        });

        let emission_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("photon_emission_pipeline_layout"),
                bind_group_layouts: &[Some(&emission_layout)],
                immediate_size: 0,
            });
        let hash_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("photon_hash_pipeline_layout"),
            bind_group_layouts: &[Some(&hash_layout)],
            immediate_size: 0,
        });

        let emission_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("photon_emission_pipeline"),
            layout: Some(&emission_pipeline_layout),
            module: &emission_shader,
            entry_point: Some("emit_photons"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let hash_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("photon_hash_pipeline"),
            layout: Some(&hash_pipeline_layout),
            module: &hash_shader,
            entry_point: Some("compute_hash"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            emission_pipeline,
            emission_bind_group,
            hash_pipeline,
            hash_bind_group,
            photon_buffer,
            photon_counter,
            hash_heads,
            uniforms_buffer,
            photon_count: 0,
        }
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        light_pos: [f32; 4],
        emitter_center: [f32; 4],
        sphere_pos: [f32; 4],
        sphere_rot: [f32; 4],
        sphere_extent: [f32; 4],
        sphere_material: [f32; 4],
        sphere_enabled: bool,
        frame: u32,
    ) {
        let uniforms = PhotonMapUniforms {
            light_pos,
            emitter_center,
            sphere_pos,
            sphere_rot,
            sphere_extent,
            sphere_material,
            sphere_enabled: [if sphere_enabled { 1 } else { 0 }, 0, 0, 0],
            photon_count: self.photon_count,
            voxel_size: VOXEL_SIZE,
            hash_table_size: HASH_TABLE_SIZE,
            frame,
        };
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn emit_photons(&mut self, encoder: &mut wgpu::CommandEncoder, photons_per_frame: u32) {
        self.photon_count = photons_per_frame.min(MAX_PHOTONS);
        encoder.clear_buffer(&self.photon_counter, 0, None);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("photon_emission_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.emission_pipeline);
        pass.set_bind_group(0, &self.emission_bind_group, &[]);
        pass.dispatch_workgroups(photons_per_frame.div_ceil(256), 1, 1);
        drop(pass);
    }

    pub fn build_spatial_structure(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.photon_count == 0 {
            return;
        }

        encoder.clear_buffer(&self.hash_heads, 0, None);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("photon_hash_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.hash_pipeline);
        pass.set_bind_group(0, &self.hash_bind_group, &[]);
        pass.dispatch_workgroups(self.photon_count.div_ceil(256), 1, 1);
    }

    pub fn photon_buffer(&self) -> &wgpu::Buffer {
        &self.photon_buffer
    }

    pub fn hash_heads(&self) -> &wgpu::Buffer {
        &self.hash_heads
    }

    pub fn uniforms_buffer(&self) -> &wgpu::Buffer {
        &self.uniforms_buffer
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
