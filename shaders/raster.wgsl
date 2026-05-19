struct RasterUniforms {
  view_proj: mat4x4<f32>,
  inv_view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: RasterUniforms;

struct VsOut {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
};

struct EnvVsOut {
  @builtin(position) position: vec4<f32>,
  @location(0) ndc_xy: vec2<f32>,
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

@fragment
fn fs_env(in: EnvVsOut) -> @location(0) vec4<f32> {
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
      let checker_scale = 2.0;
      let gx = i32(floor(p.x / checker_scale)) & 1;
      let gz = i32(floor(p.z / checker_scale)) & 1;
      let near_a = vec3<f32>(0.52, 0.52, 0.54);
      let near_b = vec3<f32>(0.29, 0.29, 0.30);
      var ground_col = select(near_b, near_a, (gx ^ gz) == 0);
      let fade = exp(-length(p.xz) * 0.03);
      ground_col *= mix(0.72, 1.0, fade);
      return vec4<f32>(ground_col, 1.0);
    }
  }

  return vec4<f32>(sky(rd), 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let base = vec3<f32>(0.74, 0.8, 0.9);
  let tint = 0.15 * sin(in.world_pos * 0.15);
  return vec4<f32>(base + tint, 1.0);
}
