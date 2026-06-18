enable wgpu_ray_query;

struct Uniforms {
  view_inv: mat4x4<f32>,
  proj_inv: mat4x4<f32>,
  light_pos: vec4<f32>,
  sphere_pos: vec4<f32>,
  sphere_color: vec4<f32>,
  sphere_params: vec4<f32>,
  sphere_rot: vec4<f32>,
  sphere_extent: vec4<f32>,
  lens_params: vec4<f32>,
  mesh_center: vec4<f32>,
  decanter_center: vec4<f32>,
  cornell_center: vec4<f32>,
  cornell_color: vec4<f32>,
  cornell_params: vec4<f32>,
  sun_intensity: f32,
  frame: u32,
  scene_kind: u32,
  render_width: u32,
  render_height: u32,
  selected_object: u32,
  mesh_enabled: u32,
  decanter_enabled: u32,
  wine_enabled: u32,
  cornell_enabled: u32,
  primitive_count: u32,
  camera_aperture: f32,
  photon_brightness: f32,
  ground_brightness: f32,
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
  wavelength_nm: f32,
  direction: vec3<f32>,
  power: f32,
  color: vec3<f32>,
  next: u32,
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

@group(0) @binding(12)
var selection_mask_out: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(13)
var image_texture: texture_2d<f32>;

struct PrimitiveData {
  pos: vec4<f32>,
  color: vec4<f32>,
  params: vec4<f32>,
  rot: vec4<f32>,
  extent: vec4<f32>,
  lens: vec4<f32>,
};

struct PrimitiveBlock {
  items: array<PrimitiveData, 64>,
};

@group(0) @binding(14)
var<uniform> primitive_block: PrimitiveBlock;

@group(0) @binding(15)
var environment_texture: texture_2d<f32>;

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
  let view_dir = normalize(dir);
  let uv = vec2<f32>(
    fract(0.5 + atan2(view_dir.z, view_dir.x) / (2.0 * PI)),
    acos(clamp(view_dir.y, -1.0, 1.0)) / PI
  );
  let dims = textureDimensions(environment_texture);
  let pixel = vec2<i32>(
    clamp(i32(uv.x * f32(dims.x)), 0, i32(dims.x) - 1),
    clamp(i32(uv.y * f32(dims.y)), 0, i32(dims.y) - 1)
  );
  let radiance = max(textureLoad(environment_texture, pixel, 0).rgb * 0.8, vec3<f32>(0.0));
  let mapped = clamp(
    (radiance * (2.51 * radiance + vec3<f32>(0.03))) /
      (radiance * (2.43 * radiance + vec3<f32>(0.59)) + vec3<f32>(0.14)),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
  let luma = dot(mapped, vec3<f32>(0.2126, 0.7152, 0.0722));
  return clamp(
    mix(vec3<f32>(luma), mapped, 1.12),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
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
          let same_side = abs(dot(normal, photon.direction)) > 0.0;
          if (d2 <= radius2 && same_side) {
            let kernel = 1.0 - d2 / max(radius2, 1e-5);
            flux = flux + photon.color * photon.power * kernel;
          }
          node = photon.next;
          visited = visited + 1u;
        }
      }
    }
  }

  let area = 3.141592653589793 * radius2;
  return flux / max(area, 1e-4) * uniforms.photon_brightness;
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

fn cube_intersection_t(origin: vec3<f32>, direction: vec3<f32>, center: vec3<f32>, half_extent: vec3<f32>) -> f32 {
  let bmin = center - half_extent;
  let bmax = center + half_extent;
  let inv_dir = 1.0 / max(abs(direction), vec3<f32>(1e-6)) * sign(direction);
  let t0 = (bmin - origin) * inv_dir;
  let t1 = (bmax - origin) * inv_dir;
  let tmin3 = min(t0, t1);
  let tmax3 = max(t0, t1);
  let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
  let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
  if (tmax < 0.0 || tmin > tmax) {
    return 1e38;
  }
  if (tmin > 0.001) {
    return tmin;
  }
  if (tmax > 0.001) {
    return tmax;
  }
  return 1e38;
}

fn cube_normal(hit_pos: vec3<f32>, center: vec3<f32>, half_extent: vec3<f32>) -> vec3<f32> {
  let p = (hit_pos - center) / max(half_extent, vec3<f32>(1e-6));
  let ax = abs(p.x);
  let ay = abs(p.y);
  let az = abs(p.z);
  if (ax > ay && ax > az) {
    return vec3<f32>(sign(p.x), 0.0, 0.0);
  }
  if (ay > az) {
    return vec3<f32>(0.0, sign(p.y), 0.0);
  }
  return vec3<f32>(0.0, 0.0, sign(p.z));
}

// Rotate vector `v` by quaternion `q` (q = [xyz, w])
fn quat_mul_vec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let qv = q.xyz;
  let t = 2.0 * cross(qv, v);
  return v + q.w * t + cross(qv, t);
}

fn primitive_radius(half_extent: vec3<f32>) -> f32 {
  return max(max(half_extent.x, half_extent.y), half_extent.z);
}

fn primitive_shape() -> u32 {
  if (uniforms.sphere_params.w >= 4.5) {
    return 5u;
  }
  if (uniforms.sphere_params.w >= 3.5) {
    return 4u;
  }
  if (uniforms.sphere_params.w >= 2.5) {
    return 3u;
  }
  if (uniforms.sphere_params.w >= 1.5) {
    return 2u;
  }
  if (uniforms.sphere_params.w >= 0.5) {
    return 1u;
  }
  return 0u;
}

fn parabolic_mirror_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>, hole_radius: f32) -> f32 {
  let radius = max(max(half_extent.x, half_extent.y), 1e-4);
  let hole2 = max(hole_radius, 0.0) * max(hole_radius, 0.0);
  let depth = max(half_extent.z, 1e-4);
  let focal_length = (radius * radius) / (8.0 * depth);
  let a = direction.x * direction.x + direction.y * direction.y;
  let b = 2.0 * (origin.x * direction.x + origin.y * direction.y) - 4.0 * focal_length * direction.z;
  let c = origin.x * origin.x + origin.y * origin.y - 4.0 * focal_length * (origin.z + depth);

  var best_t = 1e38;
  if (abs(a) < 1e-6) {
    if (abs(b) > 1e-6) {
      let t = -c / b;
      let p = origin + direction * t;
      let radial2 = p.x * p.x + p.y * p.y;
      if (t > 0.001 && radial2 >= hole2 && radial2 <= radius * radius && p.z >= -depth && p.z <= depth) {
        best_t = t;
      }
    }
  } else {
    let disc = b * b - 4.0 * a * c;
    if (disc > 0.0) {
      let sq = sqrt(disc);
      let t0 = (-b - sq) / (2.0 * a);
      let t1 = (-b + sq) / (2.0 * a);
      let p0 = origin + direction * t0;
      let r0 = p0.x * p0.x + p0.y * p0.y;
      if (t0 > 0.001 && r0 >= hole2 && r0 <= radius * radius && p0.z >= -depth && p0.z <= depth) {
        best_t = t0;
      }
      let p1 = origin + direction * t1;
      let r1 = p1.x * p1.x + p1.y * p1.y;
      if (t1 > 0.001 && t1 < best_t && r1 >= hole2 && r1 <= radius * radius && p1.z >= -depth && p1.z <= depth) {
        best_t = t1;
      }
    }
  }
  return best_t;
}

fn hyperbolic_mirror_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>, lens: vec4<f32>) -> f32 {
  let a = max(abs(lens.x), 1e-4);
  let b = max(abs(lens.y), 1e-4);
  let a2 = a * a;
  let b2 = b * b;
  let shifted_z = origin.z + a;
  let qa = direction.z * direction.z / a2
    - (direction.x * direction.x + direction.y * direction.y) / b2;
  let qb = 2.0 * (shifted_z * direction.z / a2
    - (origin.x * direction.x + origin.y * direction.y) / b2);
  let qc = shifted_z * shifted_z / a2
    - (origin.x * origin.x + origin.y * origin.y) / b2
    - 1.0;
  let disc = qb * qb - 4.0 * qa * qc;
  if (disc < 0.0 || abs(qa) < 1e-7) { return 1e38; }
  let root = sqrt(disc);
  let t0 = (-qb - root) / (2.0 * qa);
  let t1 = (-qb + root) / (2.0 * qa);
  let radius2 = max(max(half_extent.x, half_extent.y), 1e-4);
  let aperture2 = radius2 * radius2;
  let max_sag = max(half_extent.z * 1.5, 0.02);
  var best_t = 1e38;
  let p0 = origin + direction * t0;
  let r0 = dot(p0.xy, p0.xy);
  if (t0 > 0.001 && p0.z >= -0.001 && p0.z <= max_sag && r0 <= aperture2) {
    best_t = t0;
  }
  let p1 = origin + direction * t1;
  let r1 = dot(p1.xy, p1.xy);
  if (t1 > 0.001 && t1 < best_t && p1.z >= -0.001 && p1.z <= max_sag && r1 <= aperture2) {
    best_t = t1;
  }
  return best_t;
}

fn image_plane_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>) -> f32 {
  if (abs(direction.z) <= 1e-6) {
    return 1e38;
  }
  let t = -origin.z / direction.z;
  let p = origin + direction * t;
  if (t > 0.001 && abs(p.x) <= half_extent.x && abs(p.y) <= half_extent.y) {
    return t;
  }
  return 1e38;
}

fn image_plane_color(local_hit: vec3<f32>, half_extent: vec3<f32>) -> vec3<f32> {
  let dims = textureDimensions(image_texture);
  let uv = clamp(
    vec2<f32>(
      local_hit.x / max(half_extent.x * 2.0, 1e-4) + 0.5,
      0.5 - local_hit.y / max(half_extent.y * 2.0, 1e-4)
    ),
    vec2<f32>(0.0),
    vec2<f32>(1.0)
  );
  let texel = uv * vec2<f32>(f32(dims.x), f32(dims.y)) - vec2<f32>(0.5);
  let base = floor(texel);
  let frac = fract(texel);
  let p00 = vec2<i32>(
    i32(clamp(base.x, 0.0, f32(dims.x - 1u))),
    i32(clamp(base.y, 0.0, f32(dims.y - 1u)))
  );
  let p10 = vec2<i32>(min(p00.x + 1, i32(dims.x) - 1), p00.y);
  let p01 = vec2<i32>(p00.x, min(p00.y + 1, i32(dims.y) - 1));
  let p11 = vec2<i32>(min(p00.x + 1, i32(dims.x) - 1), min(p00.y + 1, i32(dims.y) - 1));
  let c00 = textureLoad(image_texture, p00, 0).rgb;
  let c10 = textureLoad(image_texture, p10, 0).rgb;
  let c01 = textureLoad(image_texture, p01, 0).rgb;
  let c11 = textureLoad(image_texture, p11, 0).rgb;
  return mix(mix(c00, c10, frac.x), mix(c01, c11, frac.x), frac.y);
}

fn spherical_lens_edge_radius(front_radius: f32, back_radius: f32, half_thickness: f32, max_aperture: f32) -> f32 {
  let max_radius = min(max_aperture, min(front_radius, back_radius) * 0.999);
  var lo = 0.0;
  var hi = max(max_radius, 1e-4);
  for (var i = 0; i < 18; i = i + 1) {
    let mid = (lo + hi) * 0.5;
    let r2 = mid * mid;
    let front_sag = sqrt(max(front_radius * front_radius - r2, 0.0));
    let back_sag = sqrt(max(back_radius * back_radius - r2, 0.0));
    let gap = 2.0 * half_thickness - front_radius - back_radius + front_sag + back_sag;
    if (gap > 0.0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return max(lo, 1e-4);
}

fn spherical_lens_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>) -> f32 {
  let base_aperture = max(max(half_extent.x, half_extent.y), 1e-4);
  let front_radius = max(abs(uniforms.lens_params.x), 1e-4);
  let back_radius = max(abs(uniforms.lens_params.y), 1e-4);
  let half_thickness = min(max(uniforms.lens_params.z * 0.5, 0.025), min(front_radius, back_radius) * 0.95);
  let aperture = spherical_lens_edge_radius(front_radius, back_radius, half_thickness, base_aperture);
  let front_center = vec3<f32>(0.0, 0.0, -half_thickness + front_radius);
  let back_center = vec3<f32>(0.0, 0.0, half_thickness - back_radius);
  let aperture2 = aperture * aperture;

  var best_t = 1e38;
  let t_front = sphere_intersection_t(origin, direction, front_center, front_radius);
  if (t_front < 1e37) {
    let p = origin + direction * t_front;
    let r2 = p.x * p.x + p.y * p.y;
    if (r2 <= aperture2 && p.z >= -half_thickness && p.z <= half_thickness) {
      best_t = t_front;
    }
  }
  let t_back = sphere_intersection_t(origin, direction, back_center, back_radius);
  if (t_back < best_t) {
    let p = origin + direction * t_back;
    let r2 = p.x * p.x + p.y * p.y;
    if (r2 <= aperture2 && p.z >= -half_thickness && p.z <= half_thickness) {
      best_t = t_back;
    }
  }

  let front_edge_z = front_center.z - sqrt(max(front_radius * front_radius - aperture2, 0.0));
  let back_edge_z = back_center.z + sqrt(max(back_radius * back_radius - aperture2, 0.0));
  let side_min_z = min(front_edge_z, back_edge_z);
  let side_max_z = max(front_edge_z, back_edge_z);
  let a = direction.x * direction.x + direction.y * direction.y;
  if (a > 1e-6 && side_max_z - side_min_z > 1e-4) {
    let b = 2.0 * (origin.x * direction.x + origin.y * direction.y);
    let c = origin.x * origin.x + origin.y * origin.y - aperture2;
    let disc = b * b - 4.0 * a * c;
    if (disc > 0.0) {
      let sq = sqrt(disc);
      let t0 = (-b - sq) / (2.0 * a);
      let p0 = origin + direction * t0;
      if (t0 > 0.001 && t0 < best_t && p0.z >= side_min_z && p0.z <= side_max_z) {
        best_t = t0;
      }
      let t1 = (-b + sq) / (2.0 * a);
      let p1 = origin + direction * t1;
      if (t1 > 0.001 && t1 < best_t && p1.z >= side_min_z && p1.z <= side_max_z) {
        best_t = t1;
      }
    }
  }
  return best_t;
}

fn primitive_intersection_t(origin: vec3<f32>, direction: vec3<f32>, center: vec3<f32>, half_extent: vec3<f32>) -> f32 {
  let shape = primitive_shape();
  if (shape == 4u) {
    return image_plane_intersection_t(origin - center, direction, half_extent);
  }
  if (shape == 3u) {
    return spherical_lens_intersection_t(origin - center, direction, half_extent);
  }
  if (shape == 2u) {
    return parabolic_mirror_intersection_t(origin - center, direction, half_extent, 0.0);
  }
  if (shape == 1u) {
    return sphere_intersection_t(origin, direction, center, primitive_radius(half_extent));
  }
  return cube_intersection_t(origin, direction, center, half_extent);
}

fn primitive_normal(hit_pos: vec3<f32>, center: vec3<f32>, half_extent: vec3<f32>) -> vec3<f32> {
  let shape = primitive_shape();
  if (shape == 4u) {
    return vec3<f32>(0.0, 0.0, select(-1.0, 1.0, hit_pos.z >= center.z));
  }
  if (shape == 3u) {
    let local = hit_pos - center;
    let base_aperture = max(max(half_extent.x, half_extent.y), 1e-4);
    let front_radius = max(abs(uniforms.lens_params.x), 1e-4);
    let back_radius = max(abs(uniforms.lens_params.y), 1e-4);
    let half_thickness = min(max(uniforms.lens_params.z * 0.5, 0.025), min(front_radius, back_radius) * 0.95);
    let aperture = spherical_lens_edge_radius(front_radius, back_radius, half_thickness, base_aperture);
    let front_center = vec3<f32>(0.0, 0.0, -half_thickness + front_radius);
    let back_center = vec3<f32>(0.0, 0.0, half_thickness - back_radius);
    let front_edge_z = front_center.z - sqrt(max(front_radius * front_radius - aperture * aperture, 0.0));
    let back_edge_z = back_center.z + sqrt(max(back_radius * back_radius - aperture * aperture, 0.0));
    let side_min_z = min(front_edge_z, back_edge_z);
    let side_max_z = max(front_edge_z, back_edge_z);
    let radial = length(local.xy);
    let front_error = abs(length(local - front_center) - front_radius);
    let back_error = abs(length(local - back_center) - back_radius);
    let side_error = abs(radial - aperture);
    if (local.z >= side_min_z && local.z <= side_max_z && side_error <= min(front_error, back_error)) {
      return normalize(vec3<f32>(local.x, local.y, 0.0));
    }
    if (front_error <= back_error) {
      return normalize(local - front_center);
    }
    return normalize(local - back_center);
  }
  if (shape == 2u) {
    let local = hit_pos - center;
    let radius = max(max(half_extent.x, half_extent.y), 1e-4);
    let depth = max(half_extent.z, 1e-4);
    let focal_length = (radius * radius) / (8.0 * depth);
    return normalize(vec3<f32>(2.0 * local.x, 2.0 * local.y, -4.0 * focal_length));
  }
  if (shape == 1u) {
    return normalize(hit_pos - center);
  }
  return cube_normal(hit_pos, center, half_extent);
}

fn primitive_shape_for(params: vec4<f32>) -> u32 {
  if (params.w >= 4.5) { return 5u; }
  if (params.w >= 3.5) { return 4u; }
  if (params.w >= 2.5) { return 3u; }
  if (params.w >= 1.5) { return 2u; }
  if (params.w >= 0.5) { return 1u; }
  return 0u;
}

fn spherical_lens_intersection_t_for(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>, lens: vec4<f32>) -> f32 {
  let base_aperture = max(max(half_extent.x, half_extent.y), 1e-4);
  let front_radius = max(abs(lens.x), 1e-4);
  let back_radius = max(abs(lens.y), 1e-4);
  let half_thickness = min(max(lens.z * 0.5, 0.025), min(front_radius, back_radius) * 0.95);
  let aperture = spherical_lens_edge_radius(front_radius, back_radius, half_thickness, base_aperture);
  let front_center = vec3<f32>(0.0, 0.0, -half_thickness + front_radius);
  let back_center = vec3<f32>(0.0, 0.0, half_thickness - back_radius);
  let aperture2 = aperture * aperture;
  var best_t = 1e38;
  let t_front = sphere_intersection_t(origin, direction, front_center, front_radius);
  if (t_front < 1e37) {
    let p = origin + direction * t_front;
    let r2 = p.x * p.x + p.y * p.y;
    if (r2 <= aperture2 && p.z >= -half_thickness && p.z <= half_thickness) { best_t = t_front; }
  }
  let t_back = sphere_intersection_t(origin, direction, back_center, back_radius);
  if (t_back < best_t) {
    let p = origin + direction * t_back;
    let r2 = p.x * p.x + p.y * p.y;
    if (r2 <= aperture2 && p.z >= -half_thickness && p.z <= half_thickness) { best_t = t_back; }
  }
  let front_edge_z = front_center.z - sqrt(max(front_radius * front_radius - aperture2, 0.0));
  let back_edge_z = back_center.z + sqrt(max(back_radius * back_radius - aperture2, 0.0));
  let side_min_z = min(front_edge_z, back_edge_z);
  let side_max_z = max(front_edge_z, back_edge_z);
  let a = direction.x * direction.x + direction.y * direction.y;
  if (a > 1e-6 && side_max_z - side_min_z > 1e-4) {
    let b = 2.0 * (origin.x * direction.x + origin.y * direction.y);
    let c = origin.x * origin.x + origin.y * origin.y - aperture2;
    let disc = b * b - 4.0 * a * c;
    if (disc > 0.0) {
      let sq = sqrt(disc);
      let t0 = (-b - sq) / (2.0 * a);
      let p0 = origin + direction * t0;
      if (t0 > 0.001 && t0 < best_t && p0.z >= side_min_z && p0.z <= side_max_z) { best_t = t0; }
      let t1 = (-b + sq) / (2.0 * a);
      let p1 = origin + direction * t1;
      if (t1 > 0.001 && t1 < best_t && p1.z >= side_min_z && p1.z <= side_max_z) { best_t = t1; }
    }
  }
  return best_t;
}

fn primitive_intersection_t_for(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>, params: vec4<f32>, lens: vec4<f32>) -> f32 {
  let shape = primitive_shape_for(params);
  if (shape == 5u) { return hyperbolic_mirror_intersection_t(origin, direction, half_extent, lens); }
  if (shape == 4u) { return image_plane_intersection_t(origin, direction, half_extent); }
  if (shape == 3u) { return spherical_lens_intersection_t_for(origin, direction, half_extent, lens); }
  if (shape == 2u) { return parabolic_mirror_intersection_t(origin, direction, half_extent, lens.w); }
  if (shape == 1u) { return sphere_intersection_t(origin, direction, vec3<f32>(0.0), primitive_radius(half_extent)); }
  return cube_intersection_t(origin, direction, vec3<f32>(0.0), half_extent);
}

fn primitive_normal_for(local_hit: vec3<f32>, half_extent: vec3<f32>, params: vec4<f32>, lens: vec4<f32>) -> vec3<f32> {
  let shape = primitive_shape_for(params);
  if (shape == 5u) {
    let a = max(abs(lens.x), 1e-4);
    let b = max(abs(lens.y), 1e-4);
    return normalize(vec3<f32>(
      -local_hit.x / (b * b),
      -local_hit.y / (b * b),
      (local_hit.z + a) / (a * a)
    ));
  }
  if (shape == 4u) { return vec3<f32>(0.0, 0.0, select(-1.0, 1.0, local_hit.z >= 0.0)); }
  if (shape == 3u) {
    let base_aperture = max(max(half_extent.x, half_extent.y), 1e-4);
    let front_radius = max(abs(lens.x), 1e-4);
    let back_radius = max(abs(lens.y), 1e-4);
    let half_thickness = min(max(lens.z * 0.5, 0.025), min(front_radius, back_radius) * 0.95);
    let aperture = spherical_lens_edge_radius(front_radius, back_radius, half_thickness, base_aperture);
    let front_center = vec3<f32>(0.0, 0.0, -half_thickness + front_radius);
    let back_center = vec3<f32>(0.0, 0.0, half_thickness - back_radius);
    let front_edge_z = front_center.z - sqrt(max(front_radius * front_radius - aperture * aperture, 0.0));
    let back_edge_z = back_center.z + sqrt(max(back_radius * back_radius - aperture * aperture, 0.0));
    let side_min_z = min(front_edge_z, back_edge_z);
    let side_max_z = max(front_edge_z, back_edge_z);
    let radial = length(local_hit.xy);
    let front_error = abs(length(local_hit - front_center) - front_radius);
    let back_error = abs(length(local_hit - back_center) - back_radius);
    let side_error = abs(radial - aperture);
    if (local_hit.z >= side_min_z && local_hit.z <= side_max_z && side_error <= min(front_error, back_error)) {
      return normalize(vec3<f32>(local_hit.x, local_hit.y, 0.0));
    }
    if (front_error <= back_error) { return normalize(local_hit - front_center); }
    return normalize(local_hit - back_center);
  }
  if (shape == 2u) {
    let radius = max(max(half_extent.x, half_extent.y), 1e-4);
    let depth = max(half_extent.z, 1e-4);
    let focal_length = (radius * radius) / (8.0 * depth);
    return normalize(vec3<f32>(2.0 * local_hit.x, 2.0 * local_hit.y, -4.0 * focal_length));
  }
  if (shape == 1u) { return normalize(local_hit); }
  return cube_normal(local_hit, vec3<f32>(0.0), half_extent);
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
      L = L + throughput * sky(rd);
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
  if (uniforms.scene_kind == 99u) {
    return vec3<f32>(0.0);
  }
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

    // Scene intersections: procedural primitives, triangles (ray query), ground
    var cube_center = uniforms.sphere_pos.xyz;
    var q = uniforms.sphere_rot;
    var q_inv = vec4<f32>(-q.xyz, q.w);
    var hit_primitive = primitive_block.items[0u];
    var hit_primitive_index = 0u;
    var t_cube = 1e38;
    if (!is_wine_scene) {
      let primitive_limit = min(uniforms.primitive_count, 64u);
      for (var pi = 0u; pi < primitive_limit; pi = pi + 1u) {
        let prim = primitive_block.items[pi];
        let prim_center = prim.pos.xyz;
        let prim_q = prim.rot;
        let prim_q_inv = vec4<f32>(-prim_q.xyz, prim_q.w);
        let local_ro = quat_mul_vec(prim_q_inv, ro - prim_center);
        let local_rd = quat_mul_vec(prim_q_inv, rd);
        let t_local = primitive_intersection_t_for(local_ro, local_rd, prim.extent.xyz, prim.params, prim.lens);
        if (t_local < t_cube) {
          t_cube = t_local;
          cube_center = prim_center;
          q = prim_q;
          q_inv = prim_q_inv;
          hit_primitive = prim;
          hit_primitive_index = pi;
        }
      }
    }

    // Triangle / mesh intersection via ray query
    var t_tri = 1e38;
    var tri_prim = 0u;
    var tri_bary = vec2<f32>(0.0);
    if (uniforms.mesh_enabled != 0u) {
      var rq: ray_query;
      rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 1000.0, ro, rd));
      rayQueryProceed(&rq);
      let tri_hit = rayQueryGetCommittedIntersection(&rq);
      if (tri_hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        t_tri = tri_hit.t;
        tri_prim = tri_hit.primitive_index;
        tri_bary = tri_hit.barycentrics;
      }
    }
    if (t_tri < 1e37) {
      let tri_pos = ro + rd * t_tri;
      let in_wine = uniforms.wine_enabled != 0u && distance(tri_pos, uniforms.mesh_center.xyz) <= uniforms.mesh_center.w;
      let in_decanter = uniforms.decanter_enabled != 0u && distance(tri_pos, uniforms.decanter_center.xyz) <= uniforms.decanter_center.w;
      if (!in_wine && !in_decanter) {
        t_tri = 1e38;
      }
    }

    var t_cornell = 1e38;
    if (!is_wine_scene && uniforms.cornell_enabled != 0u) {
      t_cornell = cube_intersection_t(ro, rd, uniforms.cornell_center.xyz, vec3<f32>(uniforms.cornell_center.w));
    }

    // Ground plane
    let t_ground = ground_plane_intersection(ro, rd);

    // Choose nearest
    var hit_t = 1e38;
    var hit_type = 0u; // 0=none,1=cube,2=tri,3=ground,4=cornell object
    if (t_cube < hit_t) { hit_t = t_cube; hit_type = 1u; }
    if (t_tri < hit_t) { hit_t = t_tri; hit_type = 2u; }
    if (t_cornell < hit_t) { hit_t = t_cornell; hit_type = 4u; }
    if (t_ground < hit_t) { hit_t = t_ground; hit_type = 3u; }

    if (hit_type == 0u) {
      L = L + throughput * sky(rd);
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
      // Cube: allow glass behavior via sphere_color.w toggle
      let local_hit = quat_mul_vec(q_inv, hit_pos - cube_center);
      let local_n = primitive_normal_for(local_hit, hit_primitive.extent.xyz, hit_primitive.params, hit_primitive.lens);
      normal = quat_mul_vec(q, local_n);
      albedo = max(hit_primitive.color.xyz, vec3<f32>(0.001));
      metallic = 0.0;
      roughness = hit_primitive.params.x;
      transmission = clamp(hit_primitive.color.w, 0.0, 1.0);
      ior = max(hit_primitive.params.y, 1.0);
      if (hit_primitive.params.z < 0.5) {
        albedo = vec3<f32>(1.0);
        roughness = 0.65;
        transmission = 0.0;
        ior = 1.0;
      }
      if (primitive_shape_for(hit_primitive.params) == 2u ||
          primitive_shape_for(hit_primitive.params) == 5u) {
        albedo = max(hit_primitive.color.xyz, vec3<f32>(0.001));
        roughness = 0.0;
        transmission = 0.0;
      }
      if (primitive_shape_for(hit_primitive.params) == 3u) {
        roughness = 0.0;
      }
      if (primitive_shape_for(hit_primitive.params) == 4u) {
        albedo = image_plane_color(local_hit, hit_primitive.extent.xyz);
        L = L + throughput * albedo;
        break;
      }
    } else if (hit_type == 2u) {
      // True triangle normal/material from ray-query primitive + barycentrics.
      let prim = tri_prim;
      let i0 = mesh_indices[prim * 3u + 0u];
      let i1 = mesh_indices[prim * 3u + 1u];
      let i2 = mesh_indices[prim * 3u + 2u];
      let bary = tri_bary;
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
      if (is_wine_scene) {
        let wine_tint = vec3<f32>(0.62, 0.11, 0.16);
        let is_wine_tinted = transmission > 0.02;
        albedo = select(albedo, wine_tint, is_wine_tinted);
        transmission = max(transmission, 0.72);
        roughness = min(roughness, 0.012);
        ior = select(ior, 1.36, is_wine_tinted);
      } else {
        // Decanter path: force true dielectric behavior even when source material metadata is weak.
        transmission = max(transmission, 0.98);
        roughness = min(roughness, 0.003);
        albedo = mix(albedo, vec3<f32>(1.0), 0.85);
      }
    } else if (hit_type == 4u) {
      normal = cube_normal(hit_pos, uniforms.cornell_center.xyz, vec3<f32>(uniforms.cornell_center.w));
      albedo = uniforms.cornell_color.xyz;
      metallic = 0.0;
      roughness = uniforms.cornell_params.x;
      transmission = clamp(uniforms.cornell_color.w, 0.0, 1.0);
      ior = max(uniforms.cornell_params.y, 1.0);
      if (uniforms.cornell_params.z < 0.5) {
        albedo = vec3<f32>(1.0);
        roughness = 0.65;
        transmission = 0.0;
        ior = 1.0;
      }
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
      albedo = albedo * uniforms.ground_brightness;
      metallic = 0.0;
      roughness = 0.9;
      transmission = 0.0;
      ior = 1.0;
    }

    let primitive_is_mirror = hit_type == 1u &&
      (primitive_shape_for(hit_primitive.params) == 2u ||
       primitive_shape_for(hit_primitive.params) == 5u ||
       (hit_primitive.params.z >= 0.5 && transmission < 0.05 && roughness <= 0.05));
    if (primitive_is_mirror) {
      let face_n = select(normal, -normal, dot(rd, normal) > 0.0);
      let mirror_dir = reflect(rd, normalize(face_n));
      if (roughness > 0.0) {
        let jitter = normalize(
          face_n + vec3<f32>(
            rand01(rng_seed ^ (bounce * 3011u + 41u)),
            rand01(rng_seed ^ (bounce * 3511u + 43u)),
            rand01(rng_seed ^ (bounce * 4013u + 47u))
          ) * 2.0 - 1.0
        );
        rd = normalize(mix(mirror_dir, jitter, roughness));
      } else {
        rd = normalize(mirror_dir);
      }
      ro = hit_pos + rd * 0.002;
      throughput = throughput * albedo;
      rng_seed = randu(rng_seed + bounce * 1699u);
      if (max(max(throughput.x, throughput.y), throughput.z) < 0.01) { break; }
      continue;
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
    var cube_shadow_t = 1e38;
    if (!is_wine_scene) {
      let primitive_limit = min(uniforms.primitive_count, 64u);
      for (var spi = 0u; spi < primitive_limit; spi = spi + 1u) {
        if (hit_type == 1u && spi == hit_primitive_index) {
          continue;
        }
        let prim = primitive_block.items[spi];
        let qsh_inv = vec4<f32>(-prim.rot.xyz, prim.rot.w);
        let local_shadow_origin = quat_mul_vec(qsh_inv, shadow_origin - prim.pos.xyz);
        let local_to_light = quat_mul_vec(qsh_inv, to_light);
        let t_local_sh = primitive_intersection_t_for(local_shadow_origin, local_to_light, prim.extent.xyz, prim.params, prim.lens);
        if (t_local_sh < cube_shadow_t) { cube_shadow_t = t_local_sh; }
      }
    }
    let visible = ((uniforms.mesh_enabled == 0u) || shadow_hit.kind == RAY_QUERY_INTERSECTION_NONE) && (cube_shadow_t >= 1e37);
    let receives_spot_pool = is_wine_scene && hit_type == 3u;
    if ((visible || receives_spot_pool) && transmission < 0.5) {
      let nl = max(dot(normal, to_light), 0.0);
      let base = select(vec3<f32>(0.04), vec3<f32>(0.025), hit_type == 1u)
        + sky(normal) * 0.1;
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
      let is_optical_lens = hit_type == 1u && primitive_shape_for(hit_primitive.params) == 3u;
      let local_dispersion = select(dispersion, 0.01, is_optical_lens);
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
        L = L + throughput * ((vec3<f32>(0.02) + sky(normal) * 0.08 + photon_indirect) * albedo) * spectral_weight;
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

fn selection_mask_ray(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
  if (uniforms.scene_kind == 99u) {
    return 0.0;
  }
  if (uniforms.selected_object == 0u) {
    return 0.0;
  }
  let is_wine_scene = uniforms.scene_kind == 2u;
  var ro = origin;
  let rd = direction;

  let sph = uniforms.sphere_pos;
  let cube_center = sph.xyz;
  let cube_half_extent = sph.w;
  let q = uniforms.sphere_rot;
  let q_inv = vec4<f32>(-q.xyz, q.w);
  let cube_half_vec_main = uniforms.sphere_extent.xyz;
  var t_cube = 1e38;
  if (!is_wine_scene) {
    let local_ro = quat_mul_vec(q_inv, ro - cube_center);
    let local_rd = quat_mul_vec(q_inv, rd);
    let t_local = primitive_intersection_t(local_ro, local_rd, vec3<f32>(0.0), cube_half_vec_main);
    if (t_local < 1e37) { t_cube = t_local; }
  }

  var t_tri = 1e38;
  if (uniforms.mesh_enabled != 0u) {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 1000.0, ro, rd));
    rayQueryProceed(&rq);
    let tri_hit = rayQueryGetCommittedIntersection(&rq);
    if (tri_hit.kind != RAY_QUERY_INTERSECTION_NONE) { t_tri = tri_hit.t; }
  }
  if (t_tri < 1e37) {
    let tri_pos = ro + rd * t_tri;
    let in_wine = uniforms.wine_enabled != 0u && distance(tri_pos, uniforms.mesh_center.xyz) <= uniforms.mesh_center.w;
    let in_decanter = uniforms.decanter_enabled != 0u && distance(tri_pos, uniforms.decanter_center.xyz) <= uniforms.decanter_center.w;
    if (!in_wine && !in_decanter) {
      t_tri = 1e38;
    }
  }
  var t_cornell = 1e38;
  if (!is_wine_scene && uniforms.cornell_enabled != 0u) {
    t_cornell = cube_intersection_t(ro, rd, uniforms.cornell_center.xyz, vec3<f32>(uniforms.cornell_center.w));
  }

  let t_ground = ground_plane_intersection(ro, rd);
  var hit_t = 1e38;
  var hit_type = 0u;
  if (t_cube < hit_t) { hit_t = t_cube; hit_type = 1u; }
  if (t_tri < hit_t) { hit_t = t_tri; hit_type = 2u; }
  if (t_cornell < hit_t) { hit_t = t_cornell; hit_type = 4u; }
  if (t_ground < hit_t) { hit_t = t_ground; hit_type = 3u; }
  if (hit_type == 0u || hit_type == 3u) {
    return 0.0;
  }
  if (hit_type == 1u) {
    return select(0.0, 1.0, uniforms.selected_object == 1u);
  }
  if (hit_type == 4u) {
    return select(0.0, 1.0, uniforms.selected_object == 4u);
  }

  let hit_pos = ro + rd * hit_t;
  let within_wine = distance(hit_pos, uniforms.mesh_center.xyz) <= uniforms.mesh_center.w;
  let within_decanter = distance(hit_pos, uniforms.decanter_center.xyz) <= uniforms.decanter_center.w;
  if (uniforms.selected_object == 2u && within_wine) {
    return 1.0;
  }
  if (uniforms.selected_object == 3u && within_decanter) {
    return 1.0;
  }
  return 0.0;
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
  let camera_origin = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
  let far_world = (uniforms.view_inv * vec4<f32>(far_pos, 1.0)).xyz;
  
  // Seed RNG with pixel coords and frame (use builtin position from vertex)
  let uv = vec2<f32>(
    0.5 * (ndc.x + 1.0),
    0.5 * (1.0 - ndc.y)
  );
  let px = u32(clamp(floor(uv.x * f32(uniforms.render_width)), 0.0, f32(uniforms.render_width - 1u)));
  let py = u32(clamp(floor(uv.y * f32(uniforms.render_height)), 0.0, f32(uniforms.render_height - 1u)));
  let idx = py * uniforms.render_width + px;

  let seed = u32(uniforms.frame) * 1973u + px * 9277u + py * 7013u + 1u;
  let pupil_r = sqrt(rand01(seed ^ 0x68bc21ebu)) * uniforms.camera_aperture;
  let pupil_theta = rand01(seed ^ 0x02e5be93u) * 6.28318530718;
  let camera_right = normalize((uniforms.view_inv * vec4<f32>(1.0, 0.0, 0.0, 0.0)).xyz);
  let camera_up = normalize((uniforms.view_inv * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
  let origin = camera_origin + camera_right * (cos(pupil_theta) * pupil_r)
    + camera_up * (sin(pupil_theta) * pupil_r);
  let direction = normalize(far_world - camera_origin);
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

  let camera_origin = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
  let far_world = (uniforms.view_inv * vec4<f32>(far_pos, 1.0)).xyz;

  let seed = uniforms.frame * 1973u + px * 9277u + py * 7013u + 1u;
  let pupil_r = sqrt(rand01(seed ^ 0x68bc21ebu)) * uniforms.camera_aperture;
  let pupil_theta = rand01(seed ^ 0x02e5be93u) * 6.28318530718;
  let camera_right = normalize((uniforms.view_inv * vec4<f32>(1.0, 0.0, 0.0, 0.0)).xyz);
  let camera_up = normalize((uniforms.view_inv * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
  let origin = camera_origin + camera_right * (cos(pupil_theta) * pupil_r)
    + camera_up * (sin(pupil_theta) * pupil_r);
  let direction = normalize(far_world - camera_origin);
  let sample_color = trace_ray(origin, direction, seed);
  let selected_mask = selection_mask_ray(camera_origin, direction);

  var accum_color = sample_color;
  if (uniforms.frame > 0u) {
    let prev = accum[idx].rgb;
    let n = f32(uniforms.frame + 1u);
    accum_color = prev + (sample_color - prev) / n;
  }

  accum[idx] = vec4<f32>(accum_color, 1.0);
  textureStore(output_image, vec2<i32>(i32(px), i32(py)), vec4<f32>(sqrt(max(accum_color, vec3<f32>(0.0))), 1.0));
  textureStore(selection_mask_out, vec2<i32>(i32(px), i32(py)), vec4<f32>(selected_mask, 0.0, 0.0, 1.0));
}
