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
  let PI = 3.141592653589793;
  let up = vec3<f32>(0.0, 1.0, 0.0);
  let sun_dir = normalize(uniforms.light_pos.xyz);
  let view_dir = normalize(dir);

  let cos_theta = clamp(dot(view_dir, up), 0.0, 1.0);
  let theta = acos(cos_theta);
  let cos_theta_s = clamp(dot(sun_dir, up), 0.001, 1.0);
  let theta_s = acos(cos_theta_s);
  let cos_gamma = clamp(dot(view_dir, sun_dir), -1.0, 1.0);
  let gamma = acos(cos_gamma);

  let T = 3.0;
  let T2 = T * T;

  let Ay = 0.1787 * T - 1.4630;
  let By = -0.3554 * T + 0.4275;
  let Cy = -0.0227 * T + 5.3251;
  let Dy = 0.1206 * T - 2.5771;
  let Ey = -0.0670 * T + 0.3703;

  let Ax = -0.0193 * T - 0.2592;
  let Bx = -0.0665 * T + 0.0008;
  let Cx = -0.0004 * T + 0.2125;
  let Dx = -0.0641 * T - 0.8989;
  let Ex = -0.0033 * T + 0.0452;

  let Az = -0.0167 * T - 0.2608;
  let Bz = -0.0950 * T + 0.0092;
  let Cz = -0.0079 * T + 0.2102;
  let Dz = -0.0441 * T - 1.6537;
  let Ez = -0.0109 * T + 0.0529;

  let chi = (4.0 / 9.0 - T / 120.0) * (PI - 2.0 * theta_s);
  let Yz = (4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192;
  let xz = (0.00165 * theta_s * theta_s * theta_s - 0.00374 * theta_s * theta_s + 0.00208 * theta_s) * T2 +
           (-0.02902 * theta_s * theta_s * theta_s + 0.06377 * theta_s * theta_s - 0.03202 * theta_s + 0.00394) * T +
           (0.11693 * theta_s * theta_s * theta_s - 0.21196 * theta_s * theta_s + 0.06052 * theta_s + 0.25886);
  let yz = (0.00275 * theta_s * theta_s * theta_s - 0.00610 * theta_s * theta_s + 0.00316 * theta_s) * T2 +
           (-0.04214 * theta_s * theta_s * theta_s + 0.08970 * theta_s * theta_s - 0.04153 * theta_s + 0.00515) * T +
           (0.15346 * theta_s * theta_s * theta_s - 0.26756 * theta_s * theta_s + 0.06669 * theta_s + 0.26688);

  let Fy = preetham_perez(cos_theta, gamma, cos_gamma, Ay, By, Cy, Dy, Ey);
  let Fx = preetham_perez(cos_theta, gamma, cos_gamma, Ax, Bx, Cx, Dx, Ex);
  let Fz = preetham_perez(cos_theta, gamma, cos_gamma, Az, Bz, Cz, Dz, Ez);
  let Fy0 = preetham_perez(cos_theta_s, 0.0, 1.0, Ay, By, Cy, Dy, Ey);
  let Fx0 = preetham_perez(cos_theta_s, 0.0, 1.0, Ax, Bx, Cx, Dx, Ex);
  let Fz0 = preetham_perez(cos_theta_s, 0.0, 1.0, Az, Bz, Cz, Dz, Ez);

  let Y = max(Yz * Fy / max(Fy0, 1e-4), 0.0);
  let x = clamp(xz * Fx / max(Fx0, 1e-4), 0.001, 0.999);
  let y = clamp(yz * Fz / max(Fz0, 1e-4), 0.001, 0.999);
  let X = (x / y) * Y;
  let Z = ((1.0 - x - y) / y) * Y;

  let rgb = vec3<f32>(
    3.2406 * X - 1.5372 * Y - 0.4986 * Z,
   -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
    0.0557 * X - 0.2040 * Y + 1.0570 * Z
  );

  let sky_rgb = max(rgb * 0.06, vec3<f32>(0.0));
  let sun_disk = smoothstep(cos(0.27 * PI / 180.0) - 0.0008, cos(0.27 * PI / 180.0) + 0.0002, cos_gamma);
  return sky_rgb + vec3<f32>(1.0, 0.97, 0.9) * sun_disk * 0.35;
}

fn preetham_perez(cos_t: f32, g: f32, cos_g: f32, a: f32, b: f32, c: f32, d: f32, e: f32) -> f32 {
  let ct = max(cos_t, 0.01);
  return (1.0 + a * exp(b / ct)) * (1.0 + c * exp(d * g) + e * cos_g * cos_g);
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

    // Directional sun lamp
    let sun_dir = normalize(uniforms.light_pos.xyz);
    let to_light = sun_dir;
    // Shadow test
    var shadow_rq: ray_query;
    let shadow_origin = hit_pos + normal * 0.01;
    rayQueryInitialize(&shadow_rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 10000.0, shadow_origin, to_light));
    rayQueryProceed(&shadow_rq);
    let shadow_hit = rayQueryGetCommittedIntersection(&shadow_rq);
    let visible = shadow_hit.kind == RAY_QUERY_INTERSECTION_NONE;
    if (visible) {
      let nl = max(dot(normal, to_light), 0.0);
      let sun_color = vec3<f32>(1.0, 0.98, 0.93);
      let sun_intensity = 2.5;
      L = L + throughput * sun_color * nl * sun_intensity;
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
