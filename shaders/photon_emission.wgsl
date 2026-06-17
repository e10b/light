enable wgpu_ray_query;

struct PhotonMapUniforms {
  light_pos: vec4<f32>,
  emitter_center: vec4<f32>,
  photon_count: u32,
  voxel_size: f32,
  hash_table_size: u32,
  frame: u32,
  primitive_count: u32,
  pad: vec3<u32>,
};

struct Photon {
  position: vec3<f32>,
  wavelength_nm: f32,
  direction: vec3<f32>,
  power: f32,
  color: vec3<f32>,
  next: u32,
};

struct MaterialData {
  base_color: vec4<f32>,
  params: vec4<f32>,
};

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

@group(0) @binding(0) var<uniform> uniforms: PhotonMapUniforms;
@group(0) @binding(1) var acc_struct: acceleration_structure;
@group(0) @binding(2) var<storage, read_write> photons: array<Photon>;
@group(0) @binding(3) var<storage, read_write> photon_counter: atomic<u32>;
@group(0) @binding(4) var<storage, read> mesh_positions: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> mesh_normals: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> mesh_indices: array<u32>;
@group(0) @binding(7) var<storage, read> mesh_triangle_material: array<u32>;
@group(0) @binding(8) var<storage, read> materials: array<MaterialData>;
@group(0) @binding(9) var<uniform> primitive_block: PrimitiveBlock;
@group(0) @binding(10) var image_texture: texture_2d<f32>;

const MAX_PHOTONS: u32 = 1000000u;
const PI: f32 = 3.141592653589793;

fn hash(x: u32) -> u32 {
  var v = x;
  v = ((v >> 16u) ^ v) * 0x45d9f3bu;
  v = ((v >> 16u) ^ v) * 0x45d9f3bu;
  v = (v >> 16u) ^ v;
  return v;
}

fn rand01(seed: u32) -> f32 {
  return f32(hash(seed) & 0x00ffffffu) / 16777215.0;
}

fn disk_sample(seed: u32, radius: f32) -> vec2<f32> {
  let r = sqrt(rand01(seed ^ 0x51ed270bu)) * radius;
  let phi = 2.0 * PI * rand01(seed ^ 0x3f84d5b5u);
  return vec2<f32>(cos(phi), sin(phi)) * r;
}

fn wl(lambda_nm: f32) -> vec3<f32> {
  let t = clamp((lambda_nm - 380.0) / 400.0, 0.0, 1.0);
  let r = smoothstep(0.45, 0.85, t) + (1.0 - smoothstep(0.0, 0.15, t)) * 0.35;
  let g = smoothstep(0.1, 0.45, t) * (1.0 - smoothstep(0.65, 0.9, t));
  let b = (1.0 - smoothstep(0.2, 0.55, t)) + smoothstep(0.88, 1.0, t) * 0.2;
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn snell_ior_for_wavelength(lambda_nm: f32, base_ior: f32, dispersion: f32) -> f32 {
  let x = (lambda_nm - 550.0) / 170.0;
  return base_ior + dispersion * (-x + 0.2 * x * x);
}

fn sample_image_texture(uv: vec2<f32>) -> vec3<f32> {
  let dims = textureDimensions(image_texture);
  let texel = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(f32(dims.x), f32(dims.y)) - vec2<f32>(0.5);
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

fn write_photon(slot: u32, position: vec3<f32>, direction: vec3<f32>, wavelength_nm: f32, power: f32, color: vec3<f32>) {
  photons[slot].position = position;
  photons[slot].wavelength_nm = wavelength_nm;
  photons[slot].direction = direction;
  photons[slot].power = power;
  photons[slot].color = color;
  photons[slot].next = 0u;
}

fn ground_plane_intersection(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
  if (abs(direction.y) <= 0.0001) { return 1e38; }
  let t = (-1.5 - origin.y) / direction.y;
  return select(1e38, t, t > 0.001);
}

fn quat_mul_vec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let qv = q.xyz;
  let t = 2.0 * cross(qv, v);
  return v + q.w * t + cross(qv, t);
}

fn primitive_shape_for(params: vec4<f32>) -> u32 {
  if (params.w >= 3.5) { return 4u; }
  if (params.w >= 2.5) { return 3u; }
  if (params.w >= 1.5) { return 2u; }
  if (params.w >= 0.5) { return 1u; }
  return 0u;
}

fn cube_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>) -> f32 {
  let inv = 1.0 / direction;
  let t0 = (-half_extent - origin) * inv;
  let t1 = (half_extent - origin) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let near_t = max(max(tmin.x, tmin.y), tmin.z);
  let far_t = min(min(tmax.x, tmax.y), tmax.z);
  if (far_t < max(near_t, 0.001)) { return 1e38; }
  return select(far_t, near_t, near_t > 0.001);
}

fn cube_normal(local_hit: vec3<f32>, half_extent: vec3<f32>) -> vec3<f32> {
  let p = local_hit / max(half_extent, vec3<f32>(1e-4));
  let ax = abs(p.x);
  let ay = abs(p.y);
  let az = abs(p.z);
  if (ax > ay && ax > az) { return vec3<f32>(sign(p.x), 0.0, 0.0); }
  if (ay > az) { return vec3<f32>(0.0, sign(p.y), 0.0); }
  return vec3<f32>(0.0, 0.0, sign(p.z));
}

fn sphere_intersection_t(origin: vec3<f32>, direction: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
  let oc = origin - center;
  let a = dot(direction, direction);
  let b = 2.0 * dot(oc, direction);
  let c = dot(oc, oc) - radius * radius;
  let disc = b * b - 4.0 * a * c;
  if (disc < 0.0) { return 1e38; }
  let sq = sqrt(disc);
  let t0 = (-b - sq) / (2.0 * a);
  let t1 = (-b + sq) / (2.0 * a);
  if (t0 > 0.001) { return t0; }
  if (t1 > 0.001) { return t1; }
  return 1e38;
}

fn image_plane_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>) -> f32 {
  if (abs(direction.z) <= 1e-6) { return 1e38; }
  let t = -origin.z / direction.z;
  let p = origin + direction * t;
  if (t > 0.001 && abs(p.x) <= half_extent.x && abs(p.y) <= half_extent.y) { return t; }
  return 1e38;
}

fn parabolic_mirror_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>) -> f32 {
  let radius = max(max(half_extent.x, half_extent.y), 1e-4);
  let depth = max(half_extent.z, 1e-4);
  let focal_length = (radius * radius) / (8.0 * depth);
  let a = direction.x * direction.x + direction.y * direction.y;
  let b = 2.0 * (origin.x * direction.x + origin.y * direction.y) - 4.0 * focal_length * direction.z;
  let c = origin.x * origin.x + origin.y * origin.y - 4.0 * focal_length * (origin.z + depth);
  if (abs(a) < 1e-6) { return 1e38; }
  let disc = b * b - 4.0 * a * c;
  if (disc <= 0.0) { return 1e38; }
  var best_t = 1e38;
  let sq = sqrt(disc);
  let t0 = (-b - sq) / (2.0 * a);
  let p0 = origin + direction * t0;
  if (t0 > 0.001 && dot(p0.xy, p0.xy) <= radius * radius && p0.z >= -depth && p0.z <= depth) { best_t = t0; }
  let t1 = (-b + sq) / (2.0 * a);
  let p1 = origin + direction * t1;
  if (t1 > 0.001 && t1 < best_t && dot(p1.xy, p1.xy) <= radius * radius && p1.z >= -depth && p1.z <= depth) { best_t = t1; }
  return best_t;
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
    if (gap > 0.0) { lo = mid; } else { hi = mid; }
  }
  return max(lo, 1e-4);
}

fn spherical_lens_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>, lens: vec4<f32>) -> f32 {
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
    if (dot(p.xy, p.xy) <= aperture2 && p.z >= -half_thickness && p.z <= half_thickness) { best_t = t_front; }
  }
  let t_back = sphere_intersection_t(origin, direction, back_center, back_radius);
  if (t_back < best_t) {
    let p = origin + direction * t_back;
    if (dot(p.xy, p.xy) <= aperture2 && p.z >= -half_thickness && p.z <= half_thickness) { best_t = t_back; }
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

fn primitive_intersection_t(origin: vec3<f32>, direction: vec3<f32>, half_extent: vec3<f32>, params: vec4<f32>, lens: vec4<f32>) -> f32 {
  let shape = primitive_shape_for(params);
  if (shape == 4u) { return image_plane_intersection_t(origin, direction, half_extent); }
  if (shape == 3u) { return spherical_lens_intersection_t(origin, direction, half_extent, lens); }
  if (shape == 2u) { return parabolic_mirror_intersection_t(origin, direction, half_extent); }
  if (shape == 1u) { return sphere_intersection_t(origin, direction, vec3<f32>(0.0), max(max(half_extent.x, half_extent.y), half_extent.z)); }
  return cube_intersection_t(origin, direction, half_extent);
}

fn primitive_normal(local_hit: vec3<f32>, half_extent: vec3<f32>, params: vec4<f32>, lens: vec4<f32>) -> vec3<f32> {
  let shape = primitive_shape_for(params);
  if (shape == 4u) { return vec3<f32>(0.0, 0.0, select(-1.0, 1.0, local_hit.z >= 0.0)); }
  if (shape == 3u) {
    let base_aperture = max(max(half_extent.x, half_extent.y), 1e-4);
    let front_radius = max(abs(lens.x), 1e-4);
    let back_radius = max(abs(lens.y), 1e-4);
    let half_thickness = min(max(lens.z * 0.5, 0.025), min(front_radius, back_radius) * 0.95);
    let aperture = spherical_lens_edge_radius(front_radius, back_radius, half_thickness, base_aperture);
    let front_center = vec3<f32>(0.0, 0.0, -half_thickness + front_radius);
    let back_center = vec3<f32>(0.0, 0.0, half_thickness - back_radius);
    let front_error = abs(length(local_hit - front_center) - front_radius);
    let back_error = abs(length(local_hit - back_center) - back_radius);
    let side_error = abs(length(local_hit.xy) - aperture);
    if (side_error <= min(front_error, back_error)) { return normalize(vec3<f32>(local_hit.x, local_hit.y, 0.0)); }
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
  return cube_normal(local_hit, half_extent);
}

@compute @workgroup_size(256, 1, 1)
fn emit_photons(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= uniforms.photon_count) { return; }

  let center = uniforms.emitter_center.xyz;
  let radius = max(uniforms.emitter_center.w, 1.0);
  let disk = disk_sample(gid.x * 9781u + uniforms.frame * 6271u, radius);

  var ro = vec3<f32>(0.0);
  var rd = vec3<f32>(0.0, 0.0, 1.0);
  var photon_color = vec3<f32>(1.0);
  var image_emitter_found = false;
  let primitive_limit = min(uniforms.primitive_count, 64u);
  for (var pi = 0u; pi < primitive_limit; pi = pi + 1u) {
    let prim_data = primitive_block.items[pi];
    if (primitive_shape_for(prim_data.params) == 4u && !image_emitter_found) {
      let image_x = quat_mul_vec(prim_data.rot, vec3<f32>(1.0, 0.0, 0.0));
      let image_y = quat_mul_vec(prim_data.rot, vec3<f32>(0.0, 1.0, 0.0));
      let image_forward = normalize(quat_mul_vec(prim_data.rot, vec3<f32>(0.0, 0.0, 1.0)));
      let ux = rand01(gid.x * 3911u + uniforms.frame * 197u + 3u);
      let vy = rand01(gid.x * 4721u + uniforms.frame * 251u + 5u);
      let u = (ux * 2.0 - 1.0) * prim_data.extent.x;
      let v = (vy * 2.0 - 1.0) * prim_data.extent.y;
      photon_color = sample_image_texture(vec2<f32>(ux, 1.0 - vy));
      ro = prim_data.pos.xyz + image_x * u + image_y * v + image_forward * 0.03;
      rd = image_forward;

      var lens_target = ro + image_forward * 30.0;
      var nearest_optic_z = 1e38;
      for (var li = 0u; li < primitive_limit; li = li + 1u) {
        let lens_data = primitive_block.items[li];
        let target_shape = primitive_shape_for(lens_data.params);
        if (target_shape == 2u || target_shape == 3u) {
          let optic_z = dot(lens_data.pos.xyz - ro, image_forward);
          if (optic_z > 0.05 && optic_z < nearest_optic_z) {
            let lens_x = quat_mul_vec(lens_data.rot, vec3<f32>(1.0, 0.0, 0.0));
            let lens_y = quat_mul_vec(lens_data.rot, vec3<f32>(0.0, 1.0, 0.0));
            let aperture_radius = max(max(lens_data.extent.x, lens_data.extent.y) * 0.82, 0.05);
            let aperture_sample = disk_sample(gid.x * 6553u + uniforms.frame * 379u + li * 17u, aperture_radius);
            lens_target = lens_data.pos.xyz + lens_x * aperture_sample.x + lens_y * aperture_sample.y;
            nearest_optic_z = optic_z;
          }
        }
      }
      rd = normalize(lens_target - ro);
      image_emitter_found = true;
    }
  }
  if (!image_emitter_found) {
    let is_spotlight = uniforms.light_pos.w < 0.0;
    let sun_to_scene = -normalize(uniforms.light_pos.xyz);
    let spot_position = uniforms.light_pos.xyz;
    let spot_axis = normalize(center - spot_position);
    let light_axis = select(sun_to_scene, spot_axis, is_spotlight);
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(light_axis.y) < 0.95);
    let tangent = normalize(cross(up, light_axis));
    let bitangent = cross(light_axis, tangent);
    let aperture = disk * select(1.0, 0.08, is_spotlight);
    ro = select(center - light_axis * 70.0 + tangent * disk.x + bitangent * disk.y, spot_position, is_spotlight);
    rd = normalize(select(light_axis, center + tangent * aperture.x + bitangent * aperture.y - spot_position, is_spotlight));
  }
  let lambda_nm = 380.0 + 400.0 * rand01(gid.x * 8191u + uniforms.frame * 131u + 17u);
  if (!image_emitter_found) {
    photon_color = wl(lambda_nm);
  }
  var power = select(0.035, 0.08, image_emitter_found);
  var passed_glass = false;
  write_photon(gid.x, center, vec3<f32>(0.0, 1.0, 0.0), lambda_nm, 0.0, photon_color);

  for (var bounce = 0u; bounce < 8u; bounce = bounce + 1u) {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xffu, 0.001, 1000.0, ro, rd));
    rayQueryProceed(&rq);
    let tri_hit = rayQueryGetCommittedIntersection(&rq);
    let tri_t = select(1e38, tri_hit.t, tri_hit.kind != RAY_QUERY_INTERSECTION_NONE);
    let ground_t = ground_plane_intersection(ro, rd);

    var primitive_t = 1e38;
    var primitive_index = 0u;
    var primitive_local_ro = vec3<f32>(0.0);
    var primitive_local_rd = vec3<f32>(0.0, 0.0, 1.0);
    let primitive_limit = min(uniforms.primitive_count, 64u);
    for (var pi = 0u; pi < primitive_limit; pi = pi + 1u) {
      let prim_data = primitive_block.items[pi];
      let q_inv = vec4<f32>(-prim_data.rot.xyz, prim_data.rot.w);
      let local_ro = quat_mul_vec(q_inv, ro - prim_data.pos.xyz);
      let local_rd = quat_mul_vec(q_inv, rd);
      let t_local = primitive_intersection_t(local_ro, local_rd, prim_data.extent.xyz, prim_data.params, prim_data.lens);
      if (t_local < primitive_t) {
        primitive_t = t_local;
        primitive_index = pi;
        primitive_local_ro = local_ro;
        primitive_local_rd = local_rd;
      }
    }

    if (ground_t < tri_t && ground_t < primitive_t) {
      if (passed_glass) {
        let hit_pos = ro + rd * ground_t;
        write_photon(gid.x, hit_pos, -rd, lambda_nm, power, photon_color);
      }
      break;
    }

    if (primitive_t < tri_t) {
      let prim_data = primitive_block.items[primitive_index];
      let local_hit = primitive_local_ro + primitive_local_rd * primitive_t;
      let shape = primitive_shape_for(prim_data.params);
      var normal = normalize(quat_mul_vec(prim_data.rot, primitive_normal(local_hit, prim_data.extent.xyz, prim_data.params, prim_data.lens)));
      let hit_pos = ro + rd * primitive_t;
      let transmission = clamp(prim_data.color.w, 0.0, 1.0);

      if (shape == 2u) {
        let face_n = select(normal, -normal, dot(rd, normal) > 0.0);
        rd = normalize(reflect(rd, face_n));
        ro = hit_pos + rd * 0.01;
        passed_glass = true;
        continue;
      }

      if (prim_data.params.z >= 0.5 && transmission < 0.05 && prim_data.params.x <= 0.05) {
        let face_n = select(normal, -normal, dot(rd, normal) > 0.0);
        rd = normalize(reflect(rd, face_n));
        ro = hit_pos + rd * 0.01;
        passed_glass = true;
        power = power * 0.94;
        continue;
      }

      if (shape == 3u || transmission >= 0.5) {
        let ior = max(snell_ior_for_wavelength(lambda_nm, max(prim_data.params.y, 1.01), 0.12), 1.01);
        let entering = dot(rd, normal) < 0.0;
        normal = select(-normal, normal, entering);
        let eta = select(ior, 1.0 / ior, entering);
        var next_dir = refract(rd, normal, eta);
        if (dot(next_dir, next_dir) < 0.0001) {
          next_dir = reflect(rd, normal);
        }
        passed_glass = true;
        power = power * 0.96;
        rd = normalize(next_dir);
        ro = hit_pos + rd * 0.01;
        continue;
      }

      if (passed_glass) {
        write_photon(gid.x, hit_pos, -rd, lambda_nm, power, photon_color);
      }
      break;
    }

    if (tri_t >= 1e37) { break; }

    let hit_pos = ro + rd * tri_t;
    let prim = tri_hit.primitive_index;
    let i0 = mesh_indices[prim * 3u + 0u];
    let i1 = mesh_indices[prim * 3u + 1u];
    let i2 = mesh_indices[prim * 3u + 2u];
    let bary = tri_hit.barycentrics;
    let w = 1.0 - bary.x - bary.y;
    var normal = normalize(mesh_normals[i0].xyz * w + mesh_normals[i1].xyz * bary.x + mesh_normals[i2].xyz * bary.y);
    let mat = materials[mesh_triangle_material[prim]];
    let ior = max(snell_ior_for_wavelength(lambda_nm, mat.params.w, 0.12), 1.01);

    let entering = dot(rd, normal) < 0.0;
    normal = select(-normal, normal, entering);
    let eta = select(ior, 1.0 / ior, entering);
    var next_dir = refract(rd, normal, eta);
    if (dot(next_dir, next_dir) < 0.0001) {
      next_dir = reflect(rd, normal);
    }

    passed_glass = true;
    let spectral_filter = dot(max(mat.base_color.rgb, vec3<f32>(0.05)), wl(lambda_nm)) / max(dot(vec3<f32>(1.0), wl(lambda_nm)), 0.001);
    power = power * mix(0.9, clamp(spectral_filter, 0.05, 1.0), 0.3);
    rd = normalize(next_dir);
    ro = hit_pos + rd * 0.01;
  }
}
