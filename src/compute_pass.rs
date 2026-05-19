pub struct ComputePass {
    pathtraced_pipeline: wgpu::ComputePipeline,
    raytraced_pipeline: wgpu::ComputePipeline,
    _color_texture: wgpu::Texture,
    color_texture_view: wgpu::TextureView,
    _selection_mask_texture: wgpu::Texture,
    selection_mask_texture_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum RenderPath {
    Pathtraced,
    Raytraced,
}

impl ComputePass {
    pub fn new(device: &wgpu::Device, bind_group_layout: &wgpu::BindGroupLayout, width: u32, height: u32) -> Self {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compute_output_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_texture_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let selection_mask_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("selection_mask_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let selection_mask_texture_view =
            selection_mask_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let pathtraced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_pathtracer"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pathtraced.wgsl").into()),
        });
        let raytraced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_raytracer"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/raytraced.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_pl"),
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        let pathtraced_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pathtraced_pipeline"),
            layout: Some(&pipeline_layout),
            module: &pathtraced_shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let raytraced_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_raytraced_pipeline"),
            layout: Some(&pipeline_layout),
            module: &raytraced_shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pathtraced_pipeline,
            raytraced_pipeline,
            _color_texture: color_texture,
            color_texture_view,
            _selection_mask_texture: selection_mask_texture,
            selection_mask_texture_view,
            width,
            height,
        }
    }

    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.color_texture_view
    }

    pub fn selection_mask_view(&self) -> &wgpu::TextureView {
        &self.selection_mask_texture_view
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        render_path: RenderPath,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        let pipeline = match render_path {
            RenderPath::Pathtraced => &self.pathtraced_pipeline,
            RenderPath::Raytraced => &self.raytraced_pipeline,
        };
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        let workgroup_size = 8;
        let dispatch_x = self.width.div_ceil(workgroup_size);
        let dispatch_y = self.height.div_ceil(workgroup_size);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}
