use std::iter;

use wgpu::util::DeviceExt;

use crate::{
    compute_pass,
    mesh::{MeshData, Vertex},
    photon_mapper::PhotonMapper,
    scene_data::{Id, MainDatabase},
};

use super::{
    geometry::{update_mesh_transform, visible_render_geometry},
    types::{MeshAsset, MeshObjectInstance, SceneUniforms},
};

#[allow(clippy::too_many_arguments)]
pub fn sync_accumulation_and_geometry(
    accumulation_dirty: &mut bool,
    sun_changed: bool,
    geometry_dirty: &mut bool,
    uniforms: &mut SceneUniforms,
    accum_byte_size: u64,
    queue: &wgpu::Queue,
    ubuf: &wgpu::Buffer,
    accum_buf: &wgpu::Buffer,
    mesh: &mut MeshData,
    model_verts: &mut Vec<Vertex>,
    stress_instance_count: usize,
    mesh_instances: &[MeshObjectInstance],
    mesh_assets: &[MeshAsset],
    main_db: &MainDatabase,
    decanter_scene_id: Id,
    model_idx: &mut Vec<u32>,
    gpu_mesh_dirty: &mut bool,
    vbuf: &mut wgpu::Buffer,
    ibuf: &mut wgpu::Buffer,
    pos_buf: &mut wgpu::Buffer,
    nrm_buf: &mut wgpu::Buffer,
    idx_buf: &mut wgpu::Buffer,
    tri_mat_buf: &mut wgpu::Buffer,
    mat_buf: &mut wgpu::Buffer,
    model_blas_desc: &mut wgpu::BlasTriangleGeometrySizeDescriptor,
    model_blas: &mut wgpu::Blas,
    tlas: &mut wgpu::Tlas,
    photon_mapper: &mut PhotonMapper,
    ugroup: &mut wgpu::BindGroup,
    device: &wgpu::Device,
    ubind: &wgpu::BindGroupLayout,
    compute_pass: &compute_pass::ComputePass,
) {
    if *accumulation_dirty || sun_changed {
        if *geometry_dirty {
            *model_verts = mesh.vertices.clone();
            if stress_instance_count == 0 {
                for inst in mesh_instances {
                    update_mesh_transform(
                        mesh,
                        model_verts,
                        inst.vertex_start,
                        inst.vertex_count,
                        &inst.base_positions,
                        &inst.base_normals,
                        inst.pivot,
                        inst.scale,
                        inst.rotation,
                        inst.translation,
                    );
                }
            }

            let (
                render_verts,
                render_indices,
                render_positions,
                render_normals,
                render_triangle_material_ids,
            ) = if stress_instance_count > 0 {
                let asset_mesh = &mesh_assets[0].mesh;
                (
                    asset_mesh.vertices.clone(),
                    asset_mesh.indices.clone(),
                    asset_mesh.positions4.clone(),
                    asset_mesh.normals4.clone(),
                    asset_mesh.triangle_material_ids.clone(),
                )
            } else {
                let visible_render_ids =
                    main_db.scene_visible_selectable_objects(decanter_scene_id);
                visible_render_geometry(mesh, mesh_instances, &visible_render_ids)
            };

            let vbuf_bytes = std::mem::size_of_val(render_verts.as_slice()) as u64;
            let ibuf_bytes = std::mem::size_of_val(render_indices.as_slice()) as u64;
            let pos_buf_bytes = std::mem::size_of_val(render_positions.as_slice()) as u64;
            let nrm_buf_bytes = std::mem::size_of_val(render_normals.as_slice()) as u64;
            let tri_mat_buf_bytes =
                std::mem::size_of_val(render_triangle_material_ids.as_slice()) as u64;
            let mat_buf_bytes = std::mem::size_of_val(mesh.materials.as_slice()) as u64;
            if vbuf_bytes > vbuf.size()
                || ibuf_bytes > ibuf.size()
                || pos_buf_bytes > pos_buf.size()
                || nrm_buf_bytes > nrm_buf.size()
                || tri_mat_buf_bytes > tri_mat_buf.size()
                || mat_buf_bytes > mat_buf.size()
            {
                *gpu_mesh_dirty = true;
            }

            *model_idx = render_indices;
            if *gpu_mesh_dirty {
                *vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("model_vbuf"),
                    contents: bytemuck::cast_slice(&render_verts),
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::BLAS_INPUT
                        | wgpu::BufferUsages::COPY_DST,
                });
                *ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("model_ibuf"),
                    contents: bytemuck::cast_slice(model_idx),
                    usage: wgpu::BufferUsages::INDEX
                        | wgpu::BufferUsages::BLAS_INPUT
                        | wgpu::BufferUsages::COPY_DST,
                });
                *pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_pos_buf"),
                    contents: bytemuck::cast_slice(&render_positions),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                *nrm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_nrm_buf"),
                    contents: bytemuck::cast_slice(&render_normals),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                *idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_idx_buf"),
                    contents: bytemuck::cast_slice(model_idx),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                *tri_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_tri_mat_buf"),
                    contents: bytemuck::cast_slice(&render_triangle_material_ids),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                *mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_materials_buf"),
                    contents: bytemuck::cast_slice(&mesh.materials),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                *model_blas_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: render_verts.len() as u32,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(model_idx.len() as u32),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                };
                *model_blas = device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("model_blas"),
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![model_blas_desc.clone()],
                    },
                );
                let tlas_instances: u32 = if stress_instance_count > 0 {
                    stress_instance_count.max(1)
                } else {
                    1
                } as u32;
                *tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
                    label: Some("scene_tlas"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    max_instances: tlas_instances,
                });
                if stress_instance_count > 0 {
                    for (i, inst) in mesh_instances.iter().enumerate() {
                        let c = inst.center();
                        tlas[i] = Some(wgpu::TlasInstance::new(
                            model_blas,
                            [1.0, 0.0, 0.0, c.x, 0.0, 1.0, 0.0, c.y, 0.0, 0.0, 1.0, c.z],
                            0,
                            0xff,
                        ));
                    }
                } else {
                    tlas[0] = Some(wgpu::TlasInstance::new(
                        model_blas,
                        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        0,
                        0xff,
                    ));
                }
                *photon_mapper = PhotonMapper::new(
                    device,
                    queue,
                    tlas,
                    pos_buf,
                    nrm_buf,
                    idx_buf,
                    tri_mat_buf,
                    mat_buf,
                );
                *ugroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ugroup"),
                    layout: ubind,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ubuf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::AccelerationStructure(tlas),
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
                            resource: wgpu::BindingResource::TextureView(
                                compute_pass.output_view(),
                            ),
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
                            resource: wgpu::BindingResource::TextureView(
                                compute_pass.selection_mask_view(),
                            ),
                        },
                    ],
                });
                *gpu_mesh_dirty = false;
            } else {
                queue.write_buffer(vbuf, 0, bytemuck::cast_slice(&render_verts));
                queue.write_buffer(ibuf, 0, bytemuck::cast_slice(model_idx));
                queue.write_buffer(pos_buf, 0, bytemuck::cast_slice(&render_positions));
                queue.write_buffer(nrm_buf, 0, bytemuck::cast_slice(&render_normals));
                queue.write_buffer(idx_buf, 0, bytemuck::cast_slice(model_idx));
                queue.write_buffer(
                    tri_mat_buf,
                    0,
                    bytemuck::cast_slice(&render_triangle_material_ids),
                );
            }

            let model_build = wgpu::BlasBuildEntry {
                blas: model_blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: model_blas_desc,
                        vertex_buffer: vbuf,
                        first_vertex: 0,
                        vertex_stride: std::mem::size_of::<Vertex>() as u64,
                        index_buffer: Some(ibuf),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            };
            let mut accel_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("accel_update"),
                });
            accel_encoder.build_acceleration_structures([model_build].iter(), iter::once(&*tlas));
            queue.submit(Some(accel_encoder.finish()));
            *geometry_dirty = false;
        }

        uniforms.frame = 0;
        let zeros = vec![0u8; accum_byte_size as usize];
        queue.write_buffer(accum_buf, 0, &zeros);
        *accumulation_dirty = false;
    } else {
        uniforms.frame = uniforms.frame.saturating_add(1);
    }
}
