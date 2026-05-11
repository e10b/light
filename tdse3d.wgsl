// 3D TDSE volume ray-marcher
// Storage layout:
//   binding 1: waveB — array<vec2f> (re, im) — written by compute, read here
//   binding 2: potBuf — array<f32>  — potential (read-only)
// Grid index: iz*N*N + iy*N + ix

struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct Tdse3dUniform {
    invViewProj:    mat4x4f,
    camPos:         vec4f,    // xyz=cam, w=time
    params:         vec4f,    // x=gridSize, y=domainHalf, z=intensityScale, w=intensityRange
    render:         vec4f,    // x=colorMode, y=aspect, z=showPotential, w=sliceAxis(-1=vol)
    march:          vec4f,    // x=stepCount, y=alphaScale, z=slicePos, w=reserved
};

@group(0) @binding(0) var<uniform>         u:       Tdse3dUniform;
@group(0) @binding(1) var<storage, read>   waveB:   array<u32>;     // (re, im) packed
@group(0) @binding(2) var<storage, read>   potBuf:  array<f32>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(input.position, 1.0);
    out.uv = input.position.xy;
    return out;
}

// ---------------------------------------------------------------------------
fn gridIndex(ix: i32, iy: i32, iz: i32, N: i32) -> i32 {
    let cx = clamp(ix, 0, N-1);
    let cy = clamp(iy, 0, N-1);
    let cz = clamp(iz, 0, N-1);
    return cz*N*N + cy*N + cx;
}

fn worldToCell(px: f32, py: f32, pz: f32, N: i32, h: f32) -> vec3f {
    return vec3f(
        (px / h * 0.5 + 0.5) * f32(N-1),
        (py / h * 0.5 + 0.5) * f32(N-1),
        (pz / h * 0.5 + 0.5) * f32(N-1)
    );
}

fn sampleRho(px: f32, py: f32, pz: f32, N: i32, h: f32) -> f32 {
    let nx = px / h * 0.5 + 0.5;
    let ny = py / h * 0.5 + 0.5;
    let nz = pz / h * 0.5 + 0.5;
    if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0 || nz < 0.0 || nz > 1.0) { return 0.0; }
    let fc = vec3f(nx, ny, nz) * f32(N-1);
    let ic = vec3i(i32(floor(fc.x)), i32(floor(fc.y)), i32(floor(fc.z)));
    let t  = fc - floor(fc);

    // trilinear interpolation of rho = |ψ|²
    var rho = 0.0f;
    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let w = unpack2x16float(waveB[gridIndex(ic.x+dx, ic.y+dy, ic.z+dz, N)]);
                let r = w.x*w.x + w.y*w.y;
                let wt = select(1.0-t.x, t.x, dx==1)
                       * select(1.0-t.y, t.y, dy==1)
                       * select(1.0-t.z, t.z, dz==1);
                rho += wt * r;
            }
        }
    }
    return rho;
}

fn samplePhase(px: f32, py: f32, pz: f32, N: i32, h: f32) -> f32 {
    let nx = clamp(px / h * 0.5 + 0.5, 0.0, 1.0);
    let ny = clamp(py / h * 0.5 + 0.5, 0.0, 1.0);
    let nz = clamp(pz / h * 0.5 + 0.5, 0.0, 1.0);
    let ix = clamp(i32(nx * f32(N-1)), 0, N-1);
    let iy = clamp(i32(ny * f32(N-1)), 0, N-1);
    let iz = clamp(i32(nz * f32(N-1)), 0, N-1);
    let w  = unpack2x16float(waveB[gridIndex(ix, iy, iz, N)]);
    var p  = atan2(w.y, w.x) / (2.0 * 3.14159265);
    if (p < 0.0) { p += 1.0; }
    return p;
}

fn samplePot(px: f32, py: f32, pz: f32, N: i32, h: f32) -> f32 {
    let nx = clamp(px / h * 0.5 + 0.5, 0.0, 1.0);
    let ny = clamp(py / h * 0.5 + 0.5, 0.0, 1.0);
    let nz = clamp(pz / h * 0.5 + 0.5, 0.0, 1.0);
    let ix = clamp(i32(nx * f32(N-1)), 0, N-1);
    let iy = clamp(i32(ny * f32(N-1)), 0, N-1);
    let iz = clamp(i32(nz * f32(N-1)), 0, N-1);
    return potBuf[gridIndex(ix, iy, iz, N)];
}

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------
fn hsvToRgb(h: f32, s: f32, v: f32) -> vec3f {
    let hue = fract(h) * 6.0;
    let c = v * s;
    let x = c * (1.0 - abs(fract(hue * 0.5) * 2.0 - 1.0));
    let m = v - c;
    if (hue < 1.0) { return vec3f(c+m, x+m, m); }
    if (hue < 2.0) { return vec3f(x+m, c+m, m); }
    if (hue < 3.0) { return vec3f(m, c+m, x+m); }
    if (hue < 4.0) { return vec3f(m, x+m, c+m); }
    if (hue < 5.0) { return vec3f(x+m, m, c+m); }
    return vec3f(c+m, m, x+m);
}

fn pal6(x: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f) -> vec3f {
    let u = clamp(x,0.0,1.0)*5.0;
    if (u<1.0){return mix(c0,c1,u);}
    if (u<2.0){return mix(c1,c2,u-1.0);}
    if (u<3.0){return mix(c2,c3,u-2.0);}
    if (u<4.0){return mix(c3,c4,u-3.0);}
    return mix(c4,c5,u-4.0);
}

fn mapColor(t: f32, colorMode: i32, phase01: f32) -> vec3f {
    let x = clamp(t, 0.0, 1.0);
    if (colorMode == 0) { return pal6(x,vec3f(0.0015,0.0005,0.0139),vec3f(0.1462,0.0449,0.3374),vec3f(0.3904,0.1004,0.5019),vec3f(0.6663,0.1819,0.3698),vec3f(0.9018,0.4251,0.1081),vec3f(0.9884,0.9984,0.6449)); }
    if (colorMode == 1) { return pal6(x,vec3f(0.0015,0.0005,0.0139),vec3f(0.1717,0.0673,0.3708),vec3f(0.4452,0.1227,0.5069),vec3f(0.7164,0.2150,0.4753),vec3f(0.9440,0.3776,0.3651),vec3f(0.9871,0.9914,0.7495)); }
    if (colorMode == 2) { return pal6(x,vec3f(0.0504,0.0298,0.5280),vec3f(0.4176,0.0006,0.6584),vec3f(0.6928,0.1651,0.5645),vec3f(0.8814,0.3925,0.3832),vec3f(0.9883,0.6523,0.2114),vec3f(0.9400,0.9752,0.1313)); }
    if (colorMode == 3) { return pal6(x,vec3f(0.2670,0.0049,0.3294),vec3f(0.2539,0.2653,0.5300),vec3f(0.1636,0.4711,0.5581),vec3f(0.1347,0.6586,0.5176),vec3f(0.4775,0.8214,0.3182),vec3f(0.9932,0.9062,0.1439)); }
    if (colorMode == 6) { return vec3f(x); }
    return hsvToRgb(phase01, 0.92, 0.18 + 0.82 * x);
}

// ---------------------------------------------------------------------------
// Box intersection
// ---------------------------------------------------------------------------
fn boxIntersect(ro: vec3f, rd: vec3f, bMin: vec3f, bMax: vec3f) -> vec2f {
    let invD = 1.0 / rd;
    let t0 = (bMin - ro) * invD;
    let t1 = (bMax - ro) * invD;
    let tmin = max(max(min(t0,t1).x, min(t0,t1).y), min(t0,t1).z);
    let tmax = min(min(max(t0,t1).x, max(t0,t1).y), max(t0,t1).z);
    return vec2f(tmin, tmax);
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let N          = max(4, i32(round(u.params.x)));
    let domainHalf = max(u.params.y, 0.5);
    let iScale     = max(u.params.z, 0.0001);
    let iRange     = max(u.params.w, 0.0001);
    let colorMode  = i32(round(u.render.x));
    let showPot    = u.render.z > 0.5;
    let sliceAxis  = i32(round(u.render.w));
    let slicePos   = u.march.z;
    let steps      = max(4, i32(round(u.march.x)));
    let alphaScale = max(u.march.y, 0.001);
    let h          = domainHalf;
    let bg         = vec3f(0.01, 0.01, 0.02);

    // ---- Slice mode ----
    if (sliceAxis >= 0) {
        let aspect = max(u.render.y, 0.001);
        var uvs = input.uv;
        uvs.x *= aspect;
        var px = 0.0; var py = 0.0; var pz = 0.0;
        if (sliceAxis == 0) { px = slicePos; py = uvs.y*h; pz = uvs.x*h; }
        else if (sliceAxis == 1) { px = uvs.x*h; py = slicePos; pz = uvs.y*h; }
        else { px = uvs.x*h; py = uvs.y*h; pz = slicePos; }
        let rho   = sampleRho(px, py, pz, N, h);
        let phase = samplePhase(px, py, pz, N, h);
        let pot   = samplePot(px, py, pz, N, h);
        let sc    = rho * iScale * 200.0;
        let br    = clamp(sc / (sc + iRange), 0.0, 1.0);
        var col   = mapColor(br, colorMode, phase);
        if (showPot) {
            let pc = mix(vec3f(0.15,0.45,0.95), vec3f(0.95,0.25,0.10), pot);
            col = mix(col, pc, 0.35 * (0.2 + 0.8 * pot));
        }
        return vec4f(mix(bg, col, br), 1.0);
    }

    // ---- Volume ray-march ----
    let ndcPos = vec4f(input.uv.x, input.uv.y, 1.0, 1.0);
    let worldH = u.invViewProj * ndcPos;
    let worldPt = worldH.xyz / worldH.w;
    let camPos  = u.camPos.xyz;
    let rd = normalize(worldPt - camPos);
    let ro = camPos;

    let hit = boxIntersect(ro, rd, vec3f(-h), vec3f(h));
    if (hit.y < hit.x || hit.y < 0.0) { return vec4f(bg, 1.0); }

    let tStart   = max(hit.x, 0.0);
    let tEnd     = hit.y;
    let stepSize = (tEnd - tStart) / f32(steps);

    var accColor = vec3f(0.0);
    var accAlpha = 0.0;
    var t = tStart + stepSize * 0.5;

    for (var i = 0; i < steps; i++) {
        if (accAlpha >= 0.99) { break; }
        let p     = ro + rd * t;
        let rho   = sampleRho(p.x, p.y, p.z, N, h);
        let phase = samplePhase(p.x, p.y, p.z, N, h);
        let pot   = samplePot(p.x, p.y, p.z, N, h);

        let sc     = rho * iScale * 200.0;
        let bright = clamp(sc / (sc + iRange), 0.0, 1.0);

        var col = mapColor(bright, colorMode, phase);
        if (showPot) {
            let pc = mix(vec3f(0.15,0.45,0.95), vec3f(0.95,0.25,0.10), pot);
            col = mix(col, pc, 0.25 * pot);
        }

        let sa = clamp(bright * alphaScale * stepSize, 0.0, 1.0);
        accColor += col * sa * (1.0 - accAlpha);
        accAlpha += sa * (1.0 - accAlpha);
        t += stepSize;
    }

    let finalColor = mix(bg, accColor / max(accAlpha, 0.001), accAlpha);
    return vec4f(finalColor, 1.0);
}
