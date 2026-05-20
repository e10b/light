struct RasterUniforms {
  view_proj: mat4x4<f32>,
  inv_view_proj: mat4x4<f32>,
  light_view_proj: array<mat4x4<f32>, 4>,
  cascade_splits: vec4<f32>,
  light_dir: vec4<f32>,
  shadow_texel_size: vec4<f32>,
  camera_pos: vec4<f32>,
  checker_color_a: vec4<f32>,
  checker_color_b: vec4<f32>,
  checker_params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: RasterUniforms;
@group(0) @binding(1)
var shadow_map: texture_depth_2d_array;
@group(0) @binding(2)
var shadow_sampler: sampler_comparison;
struct VsOut {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
};

struct EnvVsOut {
  @builtin(position) position: vec4<f32>,
  @location(0) ndc_xy: vec2<f32>,
};

struct EnvFsOut {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_env(@builtin(vertex_index) vi: u32) -> EnvVsOut {
  var out: EnvVsOut;
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  let p = pos[vi];
  out.position = vec4<f32>(p, 0.0, 1.0);
  out.ndc_xy = p;
  return out;
}

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) instance_offset: vec4<f32>) -> VsOut {
  var out: VsOut;
  let offset = vec3<f32>(instance_offset.x, instance_offset.y, instance_offset.z);
  let world = position + offset;
  out.world_pos = world;
  out.position = uniforms.view_proj * vec4<f32>(world, 1.0);
  return out;
}

fn shadow_position(position: vec3<f32>, instance_offset: vec4<f32>, cascade: u32) -> vec4<f32> {
  let offset = vec3<f32>(instance_offset.x, instance_offset.y, instance_offset.z);
  let world = position + offset;
  return uniforms.light_view_proj[cascade] * vec4<f32>(world, 1.0);
}

@vertex
fn vs_shadow0(@location(0) position: vec3<f32>, @location(1) instance_offset: vec4<f32>) -> @builtin(position) vec4<f32> {
  return shadow_position(position, instance_offset, 0u);
}

@vertex
fn vs_shadow1(@location(0) position: vec3<f32>, @location(1) instance_offset: vec4<f32>) -> @builtin(position) vec4<f32> {
  return shadow_position(position, instance_offset, 1u);
}

@vertex
fn vs_shadow2(@location(0) position: vec3<f32>, @location(1) instance_offset: vec4<f32>) -> @builtin(position) vec4<f32> {
  return shadow_position(position, instance_offset, 2u);
}

@vertex
fn vs_shadow3(@location(0) position: vec3<f32>, @location(1) instance_offset: vec4<f32>) -> @builtin(position) vec4<f32> {
  return shadow_position(position, instance_offset, 3u);
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
  let t = clamp(0.5 * (dir.y + 1.0), 0.0, 1.0);
  let horizon = vec3<f32>(0.78, 0.86, 0.95);
  let zenith = vec3<f32>(0.36, 0.56, 0.82);
  var rgb = mix(horizon, zenith, pow(t, 1.3));

  let sun_dir = normalize(vec3<f32>(0.35, 0.85, 0.25));
  let sun_amt = pow(max(dot(dir, sun_dir), 0.0), 512.0);
  rgb += vec3<f32>(1.0, 0.96, 0.88) * sun_amt * 0.75;
  return rgb;
}

fn checker_at(world_pos: vec3<f32>, scale: f32, color_a: vec3<f32>, color_b: vec3<f32>) -> vec3<f32> {
  let gx = i32(floor(world_pos.x / scale)) & 1;
  let gz = i32(floor(world_pos.z / scale)) & 1;
  return select(color_a, color_b, (gx ^ gz) == 0);
}

fn cascade_index(world_pos: vec3<f32>) -> u32 {
  let camera_dist = length(world_pos - uniforms.camera_pos.xyz);
  if (camera_dist < uniforms.cascade_splits.x) { return 0u; }
  if (camera_dist < uniforms.cascade_splits.y) { return 1u; }
  if (camera_dist < uniforms.cascade_splits.z) { return 2u; }
  return 3u;
}

fn shadow_factor(world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
  let cascade = cascade_index(world_pos);
  let light_clip = uniforms.light_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
  let light_ndc = light_clip.xyz / light_clip.w;
  let uv = light_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || light_ndc.z < 0.0 || light_ndc.z > 1.0) {
    return 1.0;
  }

  let ndotl = max(dot(normalize(normal), normalize(uniforms.light_dir.xyz)), 0.0);
  let bias = max(0.0008 * (1.0 - ndotl), 0.00025);
  let texel = uniforms.shadow_texel_size.x;
  var lit = 0.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      let offset = vec2<f32>(f32(x), f32(y)) * texel;
      lit += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, i32(cascade), light_ndc.z - bias);
    }
  }
  return lit / 9.0;
}

@fragment
fn fs_env(in: EnvVsOut) -> EnvFsOut {
  let near_clip = vec4<f32>(in.ndc_xy, 0.0, 1.0);
  let far_clip = vec4<f32>(in.ndc_xy, 1.0, 1.0);
  let near_world4 = uniforms.inv_view_proj * near_clip;
  let far_world4 = uniforms.inv_view_proj * far_clip;
  let near_world = near_world4.xyz / near_world4.w;
  let far_world = far_world4.xyz / far_world4.w;
  let ro = near_world;
  let rd = normalize(far_world - near_world);

  let ground_y = -1.5;
  if (rd.y < -1e-4) {
    let t = (ground_y - ro.y) / rd.y;
    if (t > 0.0) {
      let p = ro + rd * t;
      let near_a = vec3<f32>(0.52, 0.52, 0.54);
      let near_b = vec3<f32>(0.29, 0.29, 0.30);
      let checker_scale = 1.0;
      var ground_col = checker_at(p, checker_scale, near_a, near_b);
      let shadow = shadow_factor(p, vec3<f32>(0.0, 1.0, 0.0));
      ground_col *= mix(0.36, 1.0, shadow);
      let fade = exp(-length(p.xz) * 0.03);
      ground_col *= mix(0.72, 1.0, fade);
      let clip = uniforms.view_proj * vec4<f32>(p, 1.0);
      return EnvFsOut(vec4<f32>(ground_col, 1.0), clamp(clip.z / clip.w, 0.0, 1.0));
    }
  }

  return EnvFsOut(vec4<f32>(sky(rd), 1.0), 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  var base = vec3<f32>(0.74, 0.8, 0.9);
  var tint = 0.1 * sin(in.world_pos * 0.12);
  var normal = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
  let view_dir = normalize(uniforms.camera_pos.xyz - in.world_pos);
  normal = select(-normal, normal, dot(normal, view_dir) >= 0.0);
  let checker_plane_y = uniforms.checker_params.x;
  let checker_y_band = uniforms.checker_params.y;
  let checker_normal_thresh = uniforms.checker_params.z;
  if (
    uniforms.checker_color_b.w > 0.5 &&
    normal.y > checker_normal_thresh &&
    abs(in.world_pos.y - checker_plane_y) < checker_y_band
  ) {
    let checker_scale = max(uniforms.checker_color_a.w, 0.05);
    let color_a = uniforms.checker_color_a.rgb;
    let color_b = uniforms.checker_color_b.rgb;
    base = checker_at(in.world_pos, checker_scale, color_a, color_b);
    tint = vec3<f32>(0.0);
  } else {
    base = vec3<f32>(0.74, 0.8, 0.9);
  }
  let nl = max(dot(normal, normalize(uniforms.light_dir.xyz)), 0.0);
  let shadow = shadow_factor(in.world_pos, normal);
  let direct = vec3<f32>(1.0, 0.95, 0.84) * nl * shadow;
  let ambient = vec3<f32>(0.28, 0.34, 0.42);
  return vec4<f32>((base + tint) * (ambient + direct), 1.0);
}
