use crate::mesh::Vertex;
use wgpu::util::DeviceExt;

const CASCADE_COUNT: usize = 4;
const SHADOW_MAP_SIZE: u32 = 2048;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const FRAME_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RasterInstance {
    pub offset: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RasterUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    light_view_proj: [[[f32; 4]; 4]; CASCADE_COUNT],
    cascade_splits: [f32; 4],
    light_dir: [f32; 4],
    shadow_texel_size: [f32; 4],
    camera_pos: [f32; 4],
}

pub struct RasterPass {
    env_pipeline: wgpu::RenderPipeline,
    pipeline: wgpu::RenderPipeline,
    shadow_pipelines: Vec<wgpu::RenderPipeline>,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    shadow_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    shadow_texture: wgpu::Texture,
    #[allow(dead_code)]
    shadow_view: wgpu::TextureView,
    shadow_layer_views: Vec<wgpu::TextureView>,
    #[allow(dead_code)]
    shadow_sampler: wgpu::Sampler,
    frame_depth_texture: Option<wgpu::Texture>,
    frame_depth_view: Option<wgpu::TextureView>,
    frame_depth_size: (u32, u32),
}

impl RasterPass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/raster.wgsl").into()),
        });
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raster_uniforms"),
            contents: bytemuck::bytes_of(&RasterUniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                inv_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                light_view_proj: [glam::Mat4::IDENTITY.to_cols_array_2d(); CASCADE_COUNT],
                cascade_splits: [8.0, 24.0, 72.0, 220.0],
                light_dir: [0.35, 0.85, 0.25, 0.0],
                shadow_texel_size: [1.0 / SHADOW_MAP_SIZE as f32, 0.0, 0.0, 0.0],
                camera_pos: [0.0, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raster_shadow_texture"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: CASCADE_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("raster_shadow_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(CASCADE_COUNT as u32),
            ..Default::default()
        });
        let shadow_layer_views = (0..CASCADE_COUNT)
            .map(|cascade| {
                shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("raster_shadow_layer_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: cascade as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raster_shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster_shadow_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster_shadow_bg"),
            layout: &shadow_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raster_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raster_shadow_pl"),
            bind_group_layouts: &[Some(&shadow_bgl)],
            immediate_size: 0,
        });
        let env_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster_env_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_env"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_env"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: FRAME_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<RasterInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 1,
                        }],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: FRAME_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let shadow_pipelines = (0..CASCADE_COUNT)
            .map(|cascade| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("raster_shadow_pipeline"),
                    layout: Some(&shadow_pl),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(match cascade {
                            0 => "vs_shadow0",
                            1 => "vs_shadow1",
                            2 => "vs_shadow2",
                            _ => "vs_shadow3",
                        }),
                        buffers: &[
                            wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<Vertex>() as u64,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &[wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 0,
                                    shader_location: 0,
                                }],
                            },
                            wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<RasterInstance>() as u64,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &[wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x4,
                                    offset: 0,
                                    shader_location: 1,
                                }],
                            },
                        ],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: None,
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: SHADOW_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState {
                            constant: 2,
                            slope_scale: 2.0,
                            clamp: 0.0,
                        },
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            })
            .collect();
        Self {
            env_pipeline,
            pipeline,
            shadow_pipelines,
            uniform_buf,
            bind_group,
            shadow_bind_group,
            shadow_texture,
            shadow_view,
            shadow_layer_views,
            shadow_sampler,
            frame_depth_texture: None,
            frame_depth_view: None,
            frame_depth_size: (0, 0),
        }
    }

    pub fn update_view_proj(
        &self,
        queue: &wgpu::Queue,
        projection: glam::Mat4,
        view: glam::Mat4,
        camera_pos: glam::Vec3,
        light_dir: glam::Vec3,
    ) {
        let view_proj = projection * view;
        let light_dir = light_dir.normalize_or_zero();
        let (light_view_proj, cascade_splits) = build_cascades(projection, view, light_dir);
        let data = RasterUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            light_view_proj,
            cascade_splits,
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            shadow_texel_size: [1.0 / SHADOW_MAP_SIZE as f32, 0.0, 0.0, 0.0],
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&data));
    }

    pub fn ensure_frame_depth(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.frame_depth_size != (width, height) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("raster_frame_depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FRAME_DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.frame_depth_texture = Some(texture);
            self.frame_depth_view = Some(view);
            self.frame_depth_size = (width, height);
        }
    }

    pub fn frame_depth_view(&self) -> &wgpu::TextureView {
        self.frame_depth_view.as_ref().unwrap()
    }

    pub fn render_shadow_maps<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        vbuf: &'a wgpu::Buffer,
        instance_buf: &'a wgpu::Buffer,
        instance_count: u32,
        ibuf: &'a wgpu::Buffer,
        index_count: u32,
    ) {
        for (layer_view, pipeline) in self
            .shadow_layer_views
            .iter()
            .zip(self.shadow_pipelines.iter())
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raster_shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                multiview_mask: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &self.shadow_bind_group, &[]);
            rpass.set_vertex_buffer(0, vbuf.slice(..));
            rpass.set_vertex_buffer(1, instance_buf.slice(..));
            rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..index_count, 0, 0..instance_count);
        }
    }

    pub fn render<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        vbuf: &'a wgpu::Buffer,
        instance_buf: &'a wgpu::Buffer,
        instance_count: u32,
        ibuf: &'a wgpu::Buffer,
        index_count: u32,
    ) {
        rpass.set_pipeline(&self.env_pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1);

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, vbuf.slice(..));
        rpass.set_vertex_buffer(1, instance_buf.slice(..));
        rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..index_count, 0, 0..instance_count);
    }
}

fn build_cascades(
    projection: glam::Mat4,
    view: glam::Mat4,
    light_dir: glam::Vec3,
) -> ([[[f32; 4]; 4]; CASCADE_COUNT], [f32; 4]) {
    let splits = [8.0, 24.0, 72.0, 220.0];
    let mut matrices = [glam::Mat4::IDENTITY.to_cols_array_2d(); CASCADE_COUNT];
    let inv_view_proj = (projection * view).inverse();
    let light_dir = light_dir.normalize_or_zero();
    let up = if light_dir.y.abs() > 0.96 {
        glam::Vec3::Z
    } else {
        glam::Vec3::Y
    };

    let mut prev_split = 0.1;
    for (i, split) in splits.iter().copied().enumerate() {
        let corners = frustum_slice_corners(inv_view_proj, projection, prev_split, split);
        let center = corners
            .iter()
            .copied()
            .fold(glam::Vec3::ZERO, |acc, p| acc + p)
            / corners.len() as f32;
        let radius = corners
            .iter()
            .map(|p| (*p - center).length())
            .fold(0.0f32, f32::max)
            .max(1.0);
        let texel_world = (radius * 2.0) / SHADOW_MAP_SIZE as f32;
        let light_view = glam::Mat4::look_at_rh(center + light_dir * radius * 2.5, center, up);
        let snapped_center_ls = light_view.transform_point3(center);
        let snapped_center_ls = (snapped_center_ls / texel_world).floor() * texel_world;
        let snapped_center_ws = light_view.inverse().transform_point3(snapped_center_ls);
        let light_view = glam::Mat4::look_at_rh(
            snapped_center_ws + light_dir * radius * 2.5,
            snapped_center_ws,
            up,
        );
        let light_proj =
            glam::Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, radius * 5.0);
        matrices[i] = (light_proj * light_view).to_cols_array_2d();
        prev_split = split;
    }

    (matrices, splits)
}

fn frustum_slice_corners(
    inv_view_proj: glam::Mat4,
    projection: glam::Mat4,
    near_dist: f32,
    far_dist: f32,
) -> [glam::Vec3; 8] {
    let near_z = ndc_z_for_view_distance(projection, near_dist);
    let far_z = ndc_z_for_view_distance(projection, far_dist);
    let mut corners = [glam::Vec3::ZERO; 8];
    let mut idx = 0;
    for z in [near_z, far_z] {
        for y in [-1.0, 1.0] {
            for x in [-1.0, 1.0] {
                let world = inv_view_proj * glam::Vec4::new(x, y, z, 1.0);
                corners[idx] = world.truncate() / world.w;
                idx += 1;
            }
        }
    }
    corners
}

fn ndc_z_for_view_distance(projection: glam::Mat4, dist: f32) -> f32 {
    let clip = projection * glam::Vec4::new(0.0, 0.0, -dist.max(0.001), 1.0);
    (clip.z / clip.w).clamp(0.0, 1.0)
}
