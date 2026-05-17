enable wgpu_ray_query;

struct PhotonMapUniforms {
  light_pos: vec4<f32>,
  emitter_center: vec4<f32>,
  photon_count: u32,
  voxel_size: f32,
  hash_table_size: u32,
  frame: u32,
};

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

fn write_photon(slot: u32, position: vec3<f32>, direction: vec3<f32>, power: vec3<f32>) {
  photons[slot].position = position;
  photons[slot].direction = direction;
  photons[slot].power = power;
  photons[slot].next = 0u;
}

fn ground_plane_intersection(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
  if (abs(direction.y) <= 0.0001) { return 1e38; }
  let t = (-1.5 - origin.y) / direction.y;
  return select(1e38, t, t > 0.001);
}

@compute @workgroup_size(256, 1, 1)
fn emit_photons(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= uniforms.photon_count) { return; }

  let sun_to_scene = -normalize(uniforms.light_pos.xyz);
  let center = uniforms.emitter_center.xyz;
  let radius = max(uniforms.emitter_center.w, 1.0);
  let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(sun_to_scene.y) < 0.95);
  let tangent = normalize(cross(up, sun_to_scene));
  let bitangent = cross(sun_to_scene, tangent);
  let disk = disk_sample(gid.x * 9781u + uniforms.frame * 6271u, radius);

  var ro = center - sun_to_scene * 70.0 + tangent * disk.x + bitangent * disk.y;
  var rd = sun_to_scene;
  var power = vec3<f32>(1.0, 0.9, 0.72) * 0.035;
  var passed_glass = false;
  write_photon(gid.x, center, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0));

  for (var bounce = 0u; bounce < 8u; bounce = bounce + 1u) {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xffu, 0.001, 1000.0, ro, rd));
    rayQueryProceed(&rq);
    let tri_hit = rayQueryGetCommittedIntersection(&rq);
    let tri_t = select(1e38, tri_hit.t, tri_hit.kind != RAY_QUERY_INTERSECTION_NONE);
    let ground_t = ground_plane_intersection(ro, rd);

    if (ground_t < tri_t) {
      if (passed_glass) {
        let hit_pos = ro + rd * ground_t;
        write_photon(gid.x, hit_pos, -rd, power);
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
    let ior = max(mat.params.w, 1.01);

    let entering = dot(rd, normal) < 0.0;
    normal = select(-normal, normal, entering);
    let eta = select(ior, 1.0 / ior, entering);
    var next_dir = refract(rd, normal, eta);
    if (dot(next_dir, next_dir) < 0.0001) {
      next_dir = reflect(rd, normal);
    }

    passed_glass = true;
    power = power * mix(vec3<f32>(0.9), max(mat.base_color.rgb, vec3<f32>(0.85)), 0.3);
    rd = normalize(next_dir);
    ro = hit_pos + rd * 0.01;
  }
}
