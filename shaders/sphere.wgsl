enable wgpu_ray_query;

struct Uniforms {
  view_inv: mat4x4<f32>,
  proj_inv: mat4x4<f32>,
  light_pos: vec4<f32>,
  sphere_pos: vec4<f32>,
  sphere_color: vec4<f32>,
  mesh_center: vec4<f32>,
  sun_intensity: f32,
  frame: u32,
  scene_kind: u32,
  render_width: u32,
  render_height: u32,
  pad: vec3<u32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var acc_struct: acceleration_structure;

@group(0) @binding(2)
var<storage, read_write> accum: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read> mesh_positions: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> mesh_normals: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read> mesh_indices: array<u32>;

@group(0) @binding(6)
var<storage, read> mesh_triangle_material: array<u32>;

struct MaterialData {
  base_color: vec4<f32>,
  params: vec4<f32>, // metallic, roughness, transmission, ior
}

@group(0) @binding(7)
var<storage, read> materials: array<MaterialData>;

@group(0) @binding(8)
var output_image: texture_storage_2d<rgba8unorm, write>;

struct Photon {
  position: vec3<f32>,
  pad0: f32,
  direction: vec3<f32>,
  pad1: f32,
  power: vec3<f32>,
  pad2: f32,
  next: u32,
  pad3: vec3<u32>,
};

struct PhotonMapUniforms {
  light_pos: vec4<f32>,
  emitter_center: vec4<f32>,
  photon_count: u32,
  voxel_size: f32,
  hash_table_size: u32,
  frame: u32,
};

@group(0) @binding(9)
var<storage, read> photons: array<Photon>;

@group(0) @binding(10)
var<storage, read> photon_hash_heads: array<u32>;

@group(0) @binding(11)
var<uniform> photon_uniforms: PhotonMapUniforms;

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

fn rand01(seed: u32) -> f32 {
  return f32(randu(seed) & 0x00FFFFFFu) / 16777215.0;
}

fn photon_spatial_hash(cell: vec3<i32>) -> u32 {
  let x = u32(cell.x) * 73856093u;
  let y = u32(cell.y) * 19349663u;
  let z = u32(cell.z) * 83492791u;
  return (x ^ y ^ z) % max(photon_uniforms.hash_table_size, 1u);
}

fn estimate_photon_density(position: vec3<f32>, normal: vec3<f32>, radius: f32) -> vec3<f32> {
  let count = photon_uniforms.photon_count;
  if (count == 0u) {
    return vec3<f32>(0.0);
  }

  let base_cell = vec3<i32>(floor(position / photon_uniforms.voxel_size));
  let radius2 = radius * radius;
  var flux = vec3<f32>(0.0);

  for (var oz = -1; oz <= 1; oz = oz + 1) {
    for (var oy = -1; oy <= 1; oy = oy + 1) {
      for (var ox = -1; ox <= 1; ox = ox + 1) {
        var node = photon_hash_heads[photon_spatial_hash(base_cell + vec3<i32>(ox, oy, oz))];
        var visited = 0u;
        loop {
          if (node == 0u || visited >= 128u) {
            break;
          }
          let photon = photons[node - 1u];
          let delta = photon.position - position;
          let d2 = dot(delta, delta);
          let same_side = dot(normal, photon.direction) > 0.0;
          if (d2 <= radius2 && same_side) {
            let kernel = 1.0 - d2 / max(radius2, 1e-5);
            flux = flux + photon.power * kernel;
          }
          node = photon.next;
          visited = visited + 1u;
        }
      }
    }
  }

  let area = 3.141592653589793 * radius2;
  return flux / max(area, 1e-4);
}

fn wl(lambda_nm: f32) -> vec3<f32> {
  let t = clamp((lambda_nm - 380.0) / 400.0, 0.0, 1.0);
  let r = smoothstep(0.45, 0.85, t) + (1.0 - smoothstep(0.0, 0.15, t)) * 0.35;
  let g = smoothstep(0.1, 0.45, t) * (1.0 - smoothstep(0.65, 0.9, t));
  let b = (1.0 - smoothstep(0.2, 0.55, t)) + smoothstep(0.88, 1.0, t) * 0.2;
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn snell_ior_for_wavelength(lambda_nm: f32, dispersion: f32) -> f32 {
  let x = (lambda_nm - 550.0) / 170.0;
  return 1.5 + dispersion * (-x + 0.2 * x * x);
}

fn schlick(cos_theta: f32, eta_i: f32, eta_t: f32) -> f32 {
  let r0 = pow((eta_i - eta_t) / (eta_i + eta_t), 2.0);
  return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
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

fn sphere_intersection_t(origin: vec3<f32>, direction: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
  let oc = origin - center;
  let a = dot(direction, direction);
  let b = dot(oc, direction);
  let c = dot(oc, oc) - radius * radius;
  let disc = b * b - a * c;
  if (disc <= 0.0) {
    return 1e38;
  }
  let sq = sqrt(disc);
  let t1 = (-b - sq) / a;
  let t2 = (-b + sq) / a;
  if (t1 > 0.0) { return t1; }
  if (t2 > 0.0) { return t2; }
  return 1e38;
}

fn trace_cornell(origin: vec3<f32>, direction: vec3<f32>, seed_in: u32) -> vec3<f32> {
  var L = vec3<f32>(0.0);
  var throughput = vec3<f32>(1.0);
  var ro = origin;
  var rd = direction;
  var rng_seed = seed_in;

  let room_min = vec3<f32>(-1.0, 0.0, -2.0);
  let room_max = vec3<f32>(1.0, 2.0, 0.0);
  let sphere_center = vec3<f32>(0.35, 0.35, -1.05);
  let sphere_radius = 0.35;

  var bounce: u32 = 0u;
  loop {
    if (bounce >= 10u) { break; }
    bounce = bounce + 1u;

    var hit_t = 1e38;
    var normal = vec3<f32>(0.0);
    var albedo = vec3<f32>(0.9);
    var emissive = vec3<f32>(0.0);

    if (abs(rd.x) > 0.0001) {
      let t_left = (room_min.x - ro.x) / rd.x;
      if (t_left > 0.001) {
        let p = ro + rd * t_left;
        if (p.y >= room_min.y && p.y <= room_max.y && p.z >= room_min.z && p.z <= room_max.z && t_left < hit_t) {
          hit_t = t_left;
          normal = vec3<f32>(1.0, 0.0, 0.0);
          albedo = vec3<f32>(0.75, 0.14, 0.14);
        }
      }
      let t_right = (room_max.x - ro.x) / rd.x;
      if (t_right > 0.001) {
        let p = ro + rd * t_right;
        if (p.y >= room_min.y && p.y <= room_max.y && p.z >= room_min.z && p.z <= room_max.z && t_right < hit_t) {
          hit_t = t_right;
          normal = vec3<f32>(-1.0, 0.0, 0.0);
          albedo = vec3<f32>(0.14, 0.75, 0.14);
        }
      }
    }

    if (abs(rd.y) > 0.0001) {
      let t_floor = (room_min.y - ro.y) / rd.y;
      if (t_floor > 0.001) {
        let p = ro + rd * t_floor;
        if (p.x >= room_min.x && p.x <= room_max.x && p.z >= room_min.z && p.z <= room_max.z && t_floor < hit_t) {
          hit_t = t_floor;
          normal = vec3<f32>(0.0, 1.0, 0.0);
          albedo = vec3<f32>(0.82, 0.82, 0.82);
        }
      }
      let t_ceiling = (room_max.y - ro.y) / rd.y;
      if (t_ceiling > 0.001) {
        let p = ro + rd * t_ceiling;
        if (p.x >= room_min.x && p.x <= room_max.x && p.z >= room_min.z && p.z <= room_max.z && t_ceiling < hit_t) {
          hit_t = t_ceiling;
          normal = vec3<f32>(0.0, -1.0, 0.0);
          albedo = vec3<f32>(0.86, 0.86, 0.86);
          if (abs(p.x) < 0.32 && abs(p.z + 1.0) < 0.32) {
            emissive = vec3<f32>(11.5, 10.8, 9.8);
          }
        }
      }
    }

    if (abs(rd.z) > 0.0001) {
      let t_back = (room_min.z - ro.z) / rd.z;
      if (t_back > 0.001) {
        let p = ro + rd * t_back;
        if (p.x >= room_min.x && p.x <= room_max.x && p.y >= room_min.y && p.y <= room_max.y && t_back < hit_t) {
          hit_t = t_back;
          normal = vec3<f32>(0.0, 0.0, 1.0);
          albedo = vec3<f32>(0.84, 0.84, 0.84);
        }
      }
    }

    let t_sphere = sphere_intersection_t(ro, rd, sphere_center, sphere_radius);
    if (t_sphere < hit_t) {
      hit_t = t_sphere;
      let hit_pos = ro + rd * hit_t;
      normal = normalize(hit_pos - sphere_center);
      albedo = vec3<f32>(0.88, 0.88, 0.9);
    }

    if (hit_t >= 1e37) {
      break;
    }

    let hit_pos = ro + rd * hit_t;
    if (max(max(emissive.x, emissive.y), emissive.z) > 0.0) {
      L = L + throughput * emissive;
      break;
    }

    let n = normalize(normal);
    let jitter = vec3<f32>(
      rand01(rng_seed ^ (bounce * 1231u + 11u)),
      rand01(rng_seed ^ (bounce * 1867u + 17u)),
      rand01(rng_seed ^ (bounce * 2017u + 23u))
    ) * 2.0 - 1.0;
    rd = normalize(n + jitter);
    ro = hit_pos + n * 0.001;
    throughput = throughput * albedo;

    if (bounce > 2u) {
      let p = max(max(throughput.x, throughput.y), throughput.z);
      rng_seed = randu(rng_seed + 7u);
      if (rand01(rng_seed) > p) { break; }
      throughput = throughput * (1.0 / max(p, 1e-4));
    }

    if (max(max(throughput.x, throughput.y), throughput.z) < 0.01) {
      break;
    }
  }

  return L;
}

fn trace_ray(origin: vec3<f32>, direction: vec3<f32>, seed_in: u32) -> vec3<f32> {
  if (uniforms.scene_kind == 1u) {
    return trace_cornell(origin, direction, seed_in);
  }
  let is_wine_scene = uniforms.scene_kind == 2u;

  var L = vec3<f32>(0.0);
  var throughput = vec3<f32>(1.0);
  var rng_seed = seed_in;
  let lambda_nm = 380.0 + 400.0 * rand01(seed_in ^ 0x9e3779b9u);
  let spectral_weight = wl(lambda_nm);
  let dispersion = 0.12;
  var ro = origin;
  var rd = direction;
  let max_bounces = 16u;
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
    if (!is_wine_scene && disc > 0.0) {
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
      // Wine is a black studio scene; decanter keeps the procedural sky.
      if (!is_wine_scene) {
        L = L + throughput * sky(rd) * spectral_weight;
      }
      break;
    }

    

    let hit_pos = ro + rd * hit_t;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var albedo = vec3<f32>(0.8);

    var metallic = 0.0;
    var roughness = 0.2;
    var transmission = 0.0;
    var ior = 1.5;

    if (hit_type == 1u) {
      // Sphere: allow glass behavior via sphere_color.w toggle
      normal = normalize(hit_pos - sphere_center);
      albedo = max(uniforms.sphere_color.xyz, vec3<f32>(0.001));
      metallic = 0.0;
      roughness = 0.0025;
      transmission = clamp(uniforms.sphere_color.w, 0.0, 1.0);
      ior = 1.52;
    } else if (hit_type == 2u) {
      // True triangle normal/material from ray-query primitive + barycentrics.
      let prim = tri_hit.primitive_index;
      let i0 = mesh_indices[prim * 3u + 0u];
      let i1 = mesh_indices[prim * 3u + 1u];
      let i2 = mesh_indices[prim * 3u + 2u];
      let bary = tri_hit.barycentrics;
      let w = 1.0 - bary.x - bary.y;
      let n0 = mesh_normals[i0].xyz;
      let n1 = mesh_normals[i1].xyz;
      let n2 = mesh_normals[i2].xyz;
      normal = normalize(n0 * w + n1 * bary.x + n2 * bary.y);
      let mid = mesh_triangle_material[prim];
      let m = materials[mid];
      albedo = m.base_color.rgb;
      metallic = clamp(m.params.x, 0.0, 1.0);
      roughness = clamp(m.params.y, 0.001, 1.0);
      transmission = clamp(m.params.z, 0.0, 1.0);
      ior = max(m.params.w, 1.0);
      // Decanter path: force true dielectric behavior even when source material metadata is weak.
      transmission = max(transmission, 0.98);
      roughness = min(roughness, 0.003);
      albedo = mix(albedo, vec3<f32>(1.0), 0.85);
    } else {
      // Ground
      normal = vec3<f32>(0.0, 1.0, 0.0);
      if (is_wine_scene) {
        albedo = vec3<f32>(0.035, 0.03, 0.024);
      } else {
        let grid_scale = 2.0;
        let grid_x = i32(floor(hit_pos.x / grid_scale)) & 1;
        let grid_z = i32(floor(hit_pos.z / grid_scale)) & 1;
        let is_white = (grid_x ^ grid_z) == 0;
        albedo = select(vec3<f32>(0.3), vec3<f32>(0.7), is_white);
      }
      metallic = 0.0;
      roughness = 0.9;
      transmission = 0.0;
      ior = 1.0;
    }

    // Decanter uses directional sun; Wine uses a local spotlight aimed at the glass.
    let spot_position = uniforms.light_pos.xyz;
    let spot_target = uniforms.mesh_center.xyz;
    let spot_to_hit = hit_pos - spot_position;
    let spot_distance = length(spot_to_hit);
    let spot_axis = normalize(spot_target - spot_position);
    let spot_cos = dot(normalize(spot_to_hit), spot_axis);
    let spot_shape = smoothstep(cos(24.0 * 3.141592653589793 / 180.0), cos(8.0 * 3.141592653589793 / 180.0), spot_cos);
    let wine_to_light = normalize(spot_position - hit_pos);
    let sun_dir = normalize(uniforms.light_pos.xyz);
    let to_light = select(sun_dir, wine_to_light, is_wine_scene);
    let light_tmax = select(10000.0, max(spot_distance - 0.05, 0.05), is_wine_scene);
    var shadow_rq: ray_query;
    let shadow_origin = hit_pos + normal * 0.02;
    rayQueryInitialize(&shadow_rq, acc_struct, RayDesc(0u, 0xFFu, 0.02, light_tmax, shadow_origin, to_light));
    rayQueryProceed(&shadow_rq);
    let shadow_hit = rayQueryGetCommittedIntersection(&shadow_rq);
    var sphere_shadow_t = 1e38;
    if (!is_wine_scene) {
      sphere_shadow_t = sphere_intersection_t(
        shadow_origin,
        to_light,
        uniforms.sphere_pos.xyz,
        uniforms.sphere_pos.w,
      );
    }
    let visible = (shadow_hit.kind == RAY_QUERY_INTERSECTION_NONE) && (sphere_shadow_t >= 1e37);
    let receives_spot_pool = is_wine_scene && hit_type == 3u;
    if ((visible || receives_spot_pool) && transmission < 0.5) {
      let nl = max(dot(normal, to_light), 0.0);
      let base = select(vec3<f32>(0.08), vec3<f32>(0.05), hit_type == 1u);
      let light_color = select(vec3<f32>(1.0, 0.94, 0.82), vec3<f32>(1.0, 0.82, 0.58) * spot_shape * 7.5, is_wine_scene);
      let photon_indirect = estimate_photon_density(hit_pos, normal, photon_uniforms.voxel_size * 1.5);
      if (is_wine_scene && hit_type == 3u) {
        L = L + throughput * (photon_indirect * 8.0 + albedo * light_color * nl * uniforms.sun_intensity) * spectral_weight;
      } else {
        L = L + throughput * (base + photon_indirect * albedo + albedo * light_color * nl * uniforms.sun_intensity) * spectral_weight;
      }
      break;
    }

    if (hit_type == 2u || transmission >= 0.5) {
      if (is_wine_scene && visible) {
        let half_vec = normalize(to_light - rd);
        let spec = pow(max(dot(normal, half_vec), 0.0), 96.0);
        let rim = pow(1.0 - max(dot(-rd, normal), 0.0), 3.0);
        L = L + throughput * vec3<f32>(1.0, 0.55, 0.35) * spot_shape * (spec * 2.5 + rim * 0.08);
      }
      // Spectral glass transport (faithful style to main branch)
      let entering = dot(rd, normal) < 0.0;
      let n = select(-normal, normal, entering);
      let local_dispersion = select(dispersion, 0.08, hit_type == 1u);
      let glass_ior = ior + (snell_ior_for_wavelength(lambda_nm, local_dispersion) - 1.5);
      let eta_i = select(glass_ior, 1.0, entering);
      let eta_t = select(1.0, glass_ior, entering);
      let eta = eta_i / eta_t;
      let cos_i = clamp(dot(-rd, n), 0.0, 1.0);
      let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
      let tir = sin2_t > 1.0;
      let fresnel = select(schlick(cos_i, eta_i, eta_t), 1.0, tir);
      let choose = rand01(rng_seed ^ (0xa511e9b3u + bounce * 977u));
      let next_dir = select(refract(rd, n, eta), reflect(rd, n), choose < fresnel || tir);
      if (roughness > 0.0) {
        let j = normalize(
          n + vec3<f32>(
            rand01(rng_seed ^ (bounce * 1231u + 11u)),
            rand01(rng_seed ^ (bounce * 1867u + 17u)),
            rand01(rng_seed ^ (bounce * 2017u + 23u))
          ) * 2.0 - 1.0
        );
        rd = normalize(mix(next_dir, j, roughness));
      } else {
        rd = normalize(next_dir);
      }
      throughput *= mix(albedo * 0.985, vec3<f32>(1.0), vec3<f32>(fresnel));
      rng_seed = randu(rng_seed + bounce * 26699u);
      ro = hit_pos + rd * 0.002;
      if (max(max(throughput.x, throughput.y), throughput.z) < 0.01) { break; }
      continue;
    }

    // If diffuse surface is shadowed, keep only small ambient and terminate.
    if (transmission < 0.5) {
      let photon_indirect = estimate_photon_density(hit_pos, normal, photon_uniforms.voxel_size * 1.5);
      if (is_wine_scene && hit_type == 3u) {
        L = L + throughput * photon_indirect * 8.0 * spectral_weight;
      } else {
        L = L + throughput * ((vec3<f32>(0.04) + photon_indirect) * albedo) * spectral_weight;
      }
      break;
    }

    // Fallback (shouldn't hit with current material split)
    throughput = throughput * albedo;
    if (bounce > 2u) {
      let p = max(max(throughput.x, throughput.y), throughput.z);
      rng_seed = randu(rng_seed + 7u);
      if (rand01(rng_seed) > p) { break; }
      throughput = throughput * (1.0 / max(p, 1e-4));
    }
    ro = hit_pos + normal * 0.001;
    rd = normalize(reflect(rd, normal));
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
  let uv = vec2<f32>(
    0.5 * (ndc.x + 1.0),
    0.5 * (1.0 - ndc.y)
  );
  let px = u32(clamp(floor(uv.x * f32(uniforms.render_width)), 0.0, f32(uniforms.render_width - 1u)));
  let py = u32(clamp(floor(uv.y * f32(uniforms.render_height)), 0.0, f32(uniforms.render_height - 1u)));
  let idx = py * uniforms.render_width + px;

  let seed = u32(uniforms.frame) * 1973u + px * 9277u + py * 7013u + 1u;
  let sample_color = trace_ray(origin, direction, seed);

  var accum_color = sample_color;
  if (uniforms.frame > 0u) {
    let prev = accum[idx].rgb;
    let n = f32(uniforms.frame + 1u);
    accum_color = prev + (sample_color - prev) / n;
  }

  accum[idx] = vec4<f32>(accum_color, 1.0);
  return vec4<f32>(sqrt(max(accum_color, vec3<f32>(0.0))), 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= uniforms.render_width || gid.y >= uniforms.render_height) {
    return;
  }

  let px = gid.x;
  let py = gid.y;
  let idx = py * uniforms.render_width + px;

  let uv = vec2<f32>(
    (f32(px) + 0.5) / f32(uniforms.render_width),
    (f32(py) + 0.5) / f32(uniforms.render_height)
  );
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.5);

  let cam_far = uniforms.proj_inv * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
  let far_pos = cam_far.xyz / cam_far.w;

  let origin = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
  let far_world = (uniforms.view_inv * vec4<f32>(far_pos, 1.0)).xyz;
  let direction = normalize(far_world - origin);

  let seed = uniforms.frame * 1973u + px * 9277u + py * 7013u + 1u;
  let sample_color = trace_ray(origin, direction, seed);

  var accum_color = sample_color;
  if (uniforms.frame > 0u) {
    let prev = accum[idx].rgb;
    let n = f32(uniforms.frame + 1u);
    accum_color = prev + (sample_color - prev) / n;
  }

  accum[idx] = vec4<f32>(accum_color, 1.0);
  textureStore(output_image, vec2<i32>(i32(px), i32(py)), vec4<f32>(sqrt(max(accum_color, vec3<f32>(0.0))), 1.0));
}
