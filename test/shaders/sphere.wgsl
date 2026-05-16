enable wgpu_ray_query;

struct Uniforms {
  view_inv: mat4x4<f32>,
  proj_inv: mat4x4<f32>,
  light_pos: vec4<f32>,
  sphere_pos: vec4<f32>,
  sphere_color: vec4<f32>,
  frame: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var acc_struct: acceleration_structure;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
  var result: VertexOut;
  let x = i32(vertex_index) / 2;
  let y = i32(vertex_index) & 1;
  let tc = vec2<f32>(f32(x) * 2.0, f32(y) * 2.0);
  result.position = vec4<f32>(tc.x * 2.0 - 1.0, 1.0 - tc.y * 2.0, 0.0, 1.0);
  result.tex_coords = tc;
  return result;
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
  let up = normalize(dir);
  let t = 0.5 * (up.y + 1.0);
  return mix(vec3<f32>(0.05, 0.08, 0.15), vec3<f32>(0.15, 0.35, 0.8), t);
}

// PCG-ish hash for RNG
fn hash(x: u32) -> u32 {
  var v = x;
  v = ((v >> 16u) ^ v) * 0x45d9f3bu;
  v = ((v >> 16u) ^ v) * 0x45d9f3bu;
  v = (v >> 16u) ^ v;
  return v;
}

fn randu(seed: u32) -> u32 {
  return hash(seed);
}

// Ground plane at y = -1.5
fn ground_plane_intersection(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
  let ground_y = -1.5;
  if abs(direction.y) > 0.0001 {
    let t = (ground_y - origin.y) / direction.y;
    if t > 0.001 {
      return t;
    }
  }
  return 1e38;
}

fn trace_ray(origin: vec3<f32>, direction: vec3<f32>, seed_in: u32) -> vec3<f32> {
  // Path tracing loop (cosine-weighted hemisphere sampling)
  var L = vec3<f32>(0.0);
  var throughput = vec3<f32>(1.0);
  var rng_seed = seed_in;
  var ro = origin;
  var rd = direction;
  let max_bounces = 4u;
  var bounce: u32 = 0u;
  loop {
    if (bounce >= max_bounces) { break; }
    bounce = bounce + 1u;

    // Scene intersections: sphere, triangles (ray query), ground
    // Sphere intersection
    let sph = uniforms.sphere_pos;
    let sphere_center = sph.xyz;
    let sphere_r = sph.w;
    let oc = ro - sphere_center;
    let a = dot(rd, rd);
    let b = dot(oc, rd);
    let c = dot(oc, oc) - sphere_r * sphere_r;
    var t_sphere = 1e38;
    let disc = b * b - a * c;
    if (disc > 0.0) {
      let sq = sqrt(disc);
      let t1 = (-b - sq) / a;
      let t2 = (-b + sq) / a;
      if (t1 > 0.001) { t_sphere = t1; } else if (t2 > 0.001) { t_sphere = t2; }
    }

    // Triangle / mesh intersection via ray query
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 1000.0, ro, rd));
    rayQueryProceed(&rq);
    let tri_hit = rayQueryGetCommittedIntersection(&rq);
    var t_tri = 1e38;
    if (tri_hit.kind != RAY_QUERY_INTERSECTION_NONE) { t_tri = tri_hit.t; }

    // Ground plane
    let t_ground = ground_plane_intersection(ro, rd);

    // Choose nearest
    var hit_t = 1e38;
    var hit_type = 0u; // 0=none,1=sphere,2=tri,3=ground
    if (t_sphere < hit_t) { hit_t = t_sphere; hit_type = 1u; }
    if (t_tri < hit_t) { hit_t = t_tri; hit_type = 2u; }
    if (t_ground < hit_t) { hit_t = t_ground; hit_type = 3u; }

    if (hit_type == 0u) {
      // Miss -> add environment
      L = L + throughput * sky(rd);
      break;
    }

    

    let hit_pos = ro + rd * hit_t;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var albedo = vec3<f32>(0.8);

    if (hit_type == 1u) {
      // Sphere
      normal = normalize(hit_pos - sphere_center);
      albedo = uniforms.sphere_color.xyz;
    } else if (hit_type == 2u) {
      // Triangle: approximate normal using screen-space method
      let tangent = normalize(cross(rd, vec3<f32>(0.0, 1.0, 0.0)));
      normal = normalize(cross(tangent, rd));
      albedo = vec3<f32>(0.8);
    } else {
      // Ground
      normal = vec3<f32>(0.0, 1.0, 0.0);
      let grid_scale = 2.0;
      let grid_x = i32(floor(hit_pos.x / grid_scale)) & 1;
      let grid_z = i32(floor(hit_pos.z / grid_scale)) & 1;
      let is_white = (grid_x ^ grid_z) == 0;
      albedo = select(vec3<f32>(0.3), vec3<f32>(0.7), is_white);
    }

    // Direct emission from light (simple lambert to point light)
    let to_light = normalize(uniforms.light_pos.xyz - hit_pos);
    // Shadow test
    var shadow_rq: ray_query;
    let shadow_origin = hit_pos + normal * 0.01;
    rayQueryInitialize(&shadow_rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100.0, shadow_origin, to_light));
    rayQueryProceed(&shadow_rq);
    let shadow_hit = rayQueryGetCommittedIntersection(&shadow_rq);
    let visible = shadow_hit.kind == RAY_QUERY_INTERSECTION_NONE;
    if (visible) {
      let nl = max(dot(normal, to_light), 0.0);
      let attenuation = 1.0 / (length(uniforms.light_pos.xyz - hit_pos) * length(uniforms.light_pos.xyz - hit_pos));
      L = L + throughput * vec3<f32>(1.0) * nl * attenuation * 20.0;
    }

    // Multiply throughput by albedo and cosine
    // Sample new direction: cosine-weighted hemisphere
    rng_seed = randu(rng_seed);
    let r1 = f32(rng_seed & 0x00FFFFFFu) / 16777215.0;
    rng_seed = randu(rng_seed + 1u);
    let r2 = f32(rng_seed & 0x00FFFFFFu) / 16777215.0;
    rng_seed = rng_seed + u32(bounce) * 9781u;
    let phi = 2.0 * 3.141592653589793 * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let local_dir = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    // Build TBN
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) >= 0.999) { up = vec3<f32>(1.0, 0.0, 0.0); }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    let new_dir = normalize(tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z);

    throughput = throughput * albedo;
    // Russian roulette
    if (bounce > 2u) {
      let p = max(max(throughput.x, throughput.y), throughput.z);
      rng_seed = randu(rng_seed + 7u);
      if (f32(rng_seed & 0x00FFFFFFu) / 16777215.0 > p) { break; }
      throughput = throughput * (1.0 / p);
    }

    // Setup next ray
    ro = hit_pos + normal * 0.001;
    rd = new_dir;
  }

  return L;
}

@fragment
fn fs_main(vertex: VertexOut) -> @location(0) vec4<f32> {
  // Normalize screen coordinates to [-1, 1], flip Y to fix upside-down rendering
  let ndc = vec3<f32>(vertex.tex_coords.x * 2.0 - 1.0, (1.0 - vertex.tex_coords.y) * 2.0 - 1.0, 0.5);
  
  // Unproject to camera space
  let cam_near = uniforms.proj_inv * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
  let cam_far = uniforms.proj_inv * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  
  // Perspective divide
  let near_pos = cam_near.xyz / cam_near.w;
  let far_pos = cam_far.xyz / cam_far.w;
  
  // Convert to world space
  let origin = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
  let far_world = (uniforms.view_inv * vec4<f32>(far_pos, 1.0)).xyz;
  let direction = normalize(far_world - origin);
  
  // Seed RNG with pixel coords and frame (use builtin position from vertex)
  let frag_coord = vertex.position;
  let seed = u32(uniforms.frame) * 1973u + u32(frag_coord.x) * 9277u + u32(frag_coord.y) * 7013u + 1u;
  let color = trace_ray(origin, direction, seed);
  return vec4<f32>(sqrt(color), 1.0);
}