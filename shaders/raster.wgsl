struct RasterUniforms {
  view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: RasterUniforms;

struct VsOut {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) instance_offset: vec4<f32>) -> VsOut {
  var out: VsOut;
  let offset = vec3<f32>(instance_offset.x, instance_offset.y, instance_offset.z);
  let world = position + offset;
  out.world_pos = world;
  out.position = uniforms.view_proj * vec4<f32>(world, 1.0);
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let base = vec3<f32>(0.74, 0.8, 0.9);
  let tint = 0.15 * sin(in.world_pos * 0.15);
  return vec4<f32>(base + tint, 1.0);
}
