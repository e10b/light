enable wgpu_ray_query;

struct PhotonMapUniforms {
  light_pos: vec4<f32>,
  emitter_center: vec4<f32>,
  sphere_pos: vec4<f32>,
  sphere_rot: vec4<f32>,
  sphere_extent: vec4<f32>,
  sphere_material: vec4<f32>,
  sphere_enabled: vec4<u32>,
  photon_count: u32,
  voxel_size: f32,
  hash_table_size: u32,
  frame: u32,
};

struct Photon {
  position: vec3<f32>,
  wavelength_nm: f32,
  direction: vec3<f32>,
  power: f32,
  next: u32,
  pad3: vec3<u32>,
};

struct MaterialData {
  base_color: vec4<f32>,
  params: vec4<f32>,
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

fn write_photon(slot: u32, position: vec3<f32>, direction: vec3<f32>, wavelength_nm: f32, power: f32) {
  photons[slot].position = position;
  photons[slot].wavelength_nm = wavelength_nm;
  photons[slot].direction = direction;
  photons[slot].power = power;
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
  if (tmax < 0.0 || tmin > tmax) { return 1e38; }
  if (tmin > 0.001) { return tmin; }
  if (tmax > 0.001) { return tmax; }
  return 1e38;
}

fn cube_normal(hit_pos: vec3<f32>, center: vec3<f32>, half_extent: vec3<f32>) -> vec3<f32> {
  let p = (hit_pos - center) / max(half_extent, vec3<f32>(1e-6));
  let ax = abs(p.x);
  let ay = abs(p.y);
  let az = abs(p.z);
  if (ax > ay && ax > az) { return vec3<f32>(sign(p.x), 0.0, 0.0); }
  if (ay > az) { return vec3<f32>(0.0, sign(p.y), 0.0); }
  return vec3<f32>(0.0, 0.0, sign(p.z));
}

@compute @workgroup_size(256, 1, 1)
fn emit_photons(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= uniforms.photon_count) { return; }

  let center = uniforms.emitter_center.xyz;
  let radius = max(uniforms.emitter_center.w, 1.0);
  let disk = disk_sample(gid.x * 9781u + uniforms.frame * 6271u, radius);

  let is_spotlight = uniforms.light_pos.w < 0.0;
  let sun_to_scene = -normalize(uniforms.light_pos.xyz);
  let spot_position = uniforms.light_pos.xyz;
  let spot_axis = normalize(center - spot_position);
  let light_axis = select(sun_to_scene, spot_axis, is_spotlight);
  let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(light_axis.y) < 0.95);
  let tangent = normalize(cross(up, light_axis));
  let bitangent = cross(light_axis, tangent);
  let aperture = disk * select(1.0, 0.08, is_spotlight);

  var ro = select(center - light_axis * 70.0 + tangent * disk.x + bitangent * disk.y, spot_position, is_spotlight);
  var rd = normalize(select(light_axis, center + tangent * aperture.x + bitangent * aperture.y - spot_position, is_spotlight));
  let lambda_nm = 380.0 + 400.0 * rand01(gid.x * 8191u + uniforms.frame * 131u + 17u);
  var power = 0.035;
  var passed_glass = false;
  write_photon(gid.x, center, vec3<f32>(0.0, 1.0, 0.0), lambda_nm, 0.0);

  for (var bounce = 0u; bounce < 8u; bounce = bounce + 1u) {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xffu, 0.001, 1000.0, ro, rd));
    rayQueryProceed(&rq);
    let tri_hit = rayQueryGetCommittedIntersection(&rq);
    let tri_t = select(1e38, tri_hit.t, tri_hit.kind != RAY_QUERY_INTERSECTION_NONE);
    let ground_t = ground_plane_intersection(ro, rd);
    var cube_t = 1e38;
    let cube_enabled = uniforms.sphere_enabled.x != 0u;
    let cube_center = uniforms.sphere_pos.xyz;
    let cube_q = uniforms.sphere_rot;
    let cube_q_inv = vec4<f32>(-cube_q.xyz, cube_q.w);
    if (cube_enabled) {
      let local_ro = quat_mul_vec(cube_q_inv, ro - cube_center);
      let local_rd = quat_mul_vec(cube_q_inv, rd);
      cube_t = cube_intersection_t(local_ro, local_rd, vec3<f32>(0.0), uniforms.sphere_extent.xyz);
    }

    if (ground_t < tri_t && ground_t < cube_t) {
      if (passed_glass) {
        let hit_pos = ro + rd * ground_t;
        write_photon(gid.x, hit_pos, -rd, lambda_nm, power);
      }
      break;
    }

    if (cube_t < tri_t) {
      let hit_pos = ro + rd * cube_t;
      let local_hit = quat_mul_vec(cube_q_inv, hit_pos - cube_center);
      var normal = quat_mul_vec(cube_q, cube_normal(local_hit, vec3<f32>(0.0), uniforms.sphere_extent.xyz));
      let transmission = clamp(uniforms.sphere_material.x, 0.0, 1.0);
      let has_bsdf = uniforms.sphere_material.z > 0.5;
      if (!(has_bsdf && transmission > 0.01)) {
        if (passed_glass) {
          write_photon(gid.x, hit_pos, -rd, lambda_nm, power);
        }
        break;
      }
      let ior = max(snell_ior_for_wavelength(lambda_nm, uniforms.sphere_material.y, 0.12), 1.01);
      let entering = dot(rd, normal) < 0.0;
      normal = select(-normal, normal, entering);
      let eta = select(ior, 1.0 / ior, entering);
      var next_dir = refract(rd, normal, eta);
      if (dot(next_dir, next_dir) < 0.0001) {
        next_dir = reflect(rd, normal);
      }
      passed_glass = true;
      power = power * transmission;
      rd = normalize(next_dir);
      ro = hit_pos + rd * 0.01;
      continue;
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
    let transmission = clamp(mat.params.z, 0.0, 1.0);
    if (transmission <= 0.01) {
      if (passed_glass) {
        write_photon(gid.x, hit_pos, -rd, lambda_nm, power);
      }
      break;
    }
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
    power = power * transmission * mix(0.9, clamp(spectral_filter, 0.05, 1.0), 0.3);
    rd = normalize(next_dir);
    ro = hit_pos + rd * 0.01;
  }
}
