struct Globals {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var albedo_array: texture_2d_array<f32>;
@group(0) @binding(2) var albedo_sampler: sampler;

struct VSIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tex_layer: f32,
}

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) wnormal: vec3<f32>,
    @location(1) wpos: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tex_layer: f32,
}

@vertex
fn vs_main(in: VSIn) -> VSOut {
    let world_pos = (globals.model * vec4<f32>(in.position, 1.0)).xyz;
    let world_nrm = normalize((globals.model * vec4<f32>(in.normal, 0.0)).xyz);
    var out: VSOut;
    out.clip_pos = globals.view_proj * vec4<f32>(world_pos, 1.0);
    out.wnormal = world_nrm;
    out.wpos = world_pos;
    out.uv = in.uv;
    out.tex_layer = in.tex_layer;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.wnormal);
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.35));
    var ndl = dot(n, light_dir);
    ndl = max(ndl, 0.12);

    let layer = i32(floor(in.tex_layer + 0.5));
    let albedo = textureSample(albedo_array, albedo_sampler, in.uv, u32(max(layer, 0))).rgb;

    let sky = vec3<f32>(0.55, 0.72, 0.95);
    let ground = vec3<f32>(0.22, 0.24, 0.28);
    let hemi = mix(ground, sky, n.y * 0.5 + 0.5);
    let lit = albedo * hemi * ndl + vec3<f32>(0.03);
    let dist = length(in.wpos);
    let fog = clamp(1.0 - (dist - 120.0) / 800.0, 0.0, 1.0);
    let fog_color = vec3<f32>(0.45, 0.62, 0.88);
    return vec4<f32>(mix(fog_color, lit, fog), 1.0);
}
