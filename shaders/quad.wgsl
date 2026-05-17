struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0)
var present_tex: texture_2d<f32>;

@group(0) @binding(1)
var present_sampler: sampler;

@group(0) @binding(2)
var selection_mask_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
  var out: VertexOut;
  let x = f32((vertex_index << 1u) & 2u);
  let y = f32(vertex_index & 2u);
  out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  out.uv = vec2<f32>(x, y);
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let color = textureSample(present_tex, present_sampler, in.uv);
  let dims = vec2<f32>(vec2<u32>(textureDimensions(selection_mask_tex)));
  let texel = vec2<f32>(1.0 / max(dims.x, 1.0), 1.0 / max(dims.y, 1.0));

  let center = textureSample(selection_mask_tex, present_sampler, in.uv).r;
  let left = textureSample(selection_mask_tex, present_sampler, in.uv + vec2<f32>(-texel.x, 0.0)).r;
  let right = textureSample(selection_mask_tex, present_sampler, in.uv + vec2<f32>(texel.x, 0.0)).r;
  let up = textureSample(selection_mask_tex, present_sampler, in.uv + vec2<f32>(0.0, texel.y)).r;
  let down = textureSample(selection_mask_tex, present_sampler, in.uv + vec2<f32>(0.0, -texel.y)).r;
  let nmax = max(max(left, right), max(up, down));
  let outside_edge = (1.0 - step(0.5, center)) * step(0.5, nmax);
  let orange = vec3<f32>(1.0, 0.5, 0.0);
  let out_rgb = mix(color.rgb, orange, outside_edge);
  return vec4<f32>(out_rgb, 1.0);
}
