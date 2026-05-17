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

@group(0) @binding(0) var<uniform> uniforms: PhotonMapUniforms;
@group(0) @binding(1) var<storage, read_write> photons: array<Photon>;
@group(0) @binding(2) var<storage, read_write> hash_heads: array<atomic<u32>>;

fn spatial_hash(cell: vec3<i32>) -> u32 {
  let x = u32(cell.x) * 73856093u;
  let y = u32(cell.y) * 19349663u;
  let z = u32(cell.z) * 83492791u;
  return (x ^ y ^ z) % uniforms.hash_table_size;
}

@compute @workgroup_size(256, 1, 1)
fn compute_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= uniforms.photon_count) { return; }

  let cell = vec3<i32>(floor(photons[id].position / uniforms.voxel_size));
  let key = spatial_hash(cell);
  let previous_head = atomicExchange(&hash_heads[key], id + 1u);
  photons[id].next = previous_head;
}
