// 3D TDSE FDTD compute shader — GPU ping-pong integration
//
// Binding layout (group 0):
//   0 : uniform   Tdse3dParams
//   1 : storage   waveA  (read-write) — "current"  psi (re, im)
//   2 : storage   waveB  (read-write) — "next"     psi / rhs scratch
//   3 : storage   potBuf (read-only)  — potential V per cell (f32)
//
// Storage index: idx = iz*N*N + iy*N + ix   (matches CPU layout)
// Workgroup = (4,4,4) = 64 threads → dispatch ceil(N/4)^3

struct Tdse3dParams {
    gridN        : u32,
    mode         : u32,   // 0=Euler, 1=CN
    dt           : f32,
    absorbWidth  : f32,
    absorbStr    : f32,
    dx           : f32,
    _pad0        : f32,
    _pad1        : f32,
};

@group(0) @binding(0) var<uniform>             params  : Tdse3dParams;
@group(0) @binding(1) var<storage, read_write> waveA   : array<u32>; // (re,im) packed current / iterate
@group(0) @binding(2) var<storage, read_write> waveB   : array<u32>; // (re,im) packed next / rhs

fn readWaveA(i: u32) -> vec2f { return unpack2x16float(waveA[i]); }
fn writeWaveA(i: u32, v: vec2f) { waveA[i] = pack2x16float(v); }
fn readWaveB(i: u32) -> vec2f { return unpack2x16float(waveB[i]); }
fn writeWaveB(i: u32, v: vec2f) { waveB[i] = pack2x16float(v); }
@group(0) @binding(3) var<storage, read>       potBuf  : array<f32>;

// ---------------------------------------------------------------------------
fn idx3(ix: i32, iy: i32, iz: i32, N: i32) -> u32 {
    let cx = clamp(ix, 0, N-1);
    let cy = clamp(iy, 0, N-1);
    let cz = clamp(iz, 0, N-1);
    return u32(cz*N*N + cy*N + cx);
}

fn lapA(ix: i32, iy: i32, iz: i32, N: i32) -> vec2f {
    return readWaveA(idx3(ix-1,iy,iz,N)) + readWaveA(idx3(ix+1,iy,iz,N))
         + readWaveA(idx3(ix,iy-1,iz,N)) + readWaveA(idx3(ix,iy+1,iz,N))
         + readWaveA(idx3(ix,iy,iz-1,N)) + readWaveA(idx3(ix,iy,iz+1,N))
         - 6.0 * readWaveA(idx3(ix,iy,iz,N));
}

// i dψ/dt = (-½∇² + V)ψ  =>  dψ/dt.re = +½ lap.im - V*im
//                             dψ/dt.im = -½ lap.re + V*re
fn schrodRHS(psi: vec2f, lap: vec2f, V: f32) -> vec2f {
    return vec2f(0.5 * lap.y - V * psi.y,
                -0.5 * lap.x + V * psi.x);
}

fn absorbDamp(ix: i32, iy: i32, iz: i32, N: i32) -> f32 {
    let edge = f32(min(min(min(ix, N-1-ix), iy), min(N-1-iy, min(iz, N-1-iz)))) * params.dx;
    if (edge >= params.absorbWidth || params.absorbWidth <= 0.0) { return 1.0; }
    let s = 1.0 - edge / params.absorbWidth;
    return exp(-params.absorbStr * s * s * params.dt);
}

fn isBoundary(ix: i32, iy: i32, iz: i32, N: i32) -> bool {
    return ix == 0 || ix == N-1 || iy == 0 || iy == N-1 || iz == 0 || iz == N-1;
}

// ---------------------------------------------------------------------------
// Euler: waveA → waveB
// ---------------------------------------------------------------------------
@compute @workgroup_size(4, 4, 4)
fn euler(@builtin(global_invocation_id) gid: vec3u) {
    let N  = i32(params.gridN);
    let ix = i32(gid.x); let iy = i32(gid.y); let iz = i32(gid.z);
    if (ix >= N || iy >= N || iz >= N) { return; }
    let i = idx3(ix, iy, iz, N);
    if (isBoundary(ix, iy, iz, N)) { writeWaveB(i, vec2f(0.0)); return; }

    let psi  = readWaveA(i);
    let lap  = lapA(ix, iy, iz, N);
    let V    = potBuf[i];
    let drhs = schrodRHS(psi, lap, V);
    let damp = absorbDamp(ix, iy, iz, N);
    writeWaveB(i, (psi + params.dt * drhs) * damp);
}

// ---------------------------------------------------------------------------
// CN step 1 — build RHS from waveA into waveB
// ---------------------------------------------------------------------------
@compute @workgroup_size(4, 4, 4)
fn cn_rhs(@builtin(global_invocation_id) gid: vec3u) {
    let N  = i32(params.gridN);
    let ix = i32(gid.x); let iy = i32(gid.y); let iz = i32(gid.z);
    if (ix >= N || iy >= N || iz >= N) { return; }
    let i = idx3(ix, iy, iz, N);
    if (isBoundary(ix, iy, iz, N)) { writeWaveB(i, vec2f(0.0)); return; }

    let psi  = readWaveA(i);
    let lap  = lapA(ix, iy, iz, N);
    let V    = potBuf[i];
    // RHS = ψ_old + (dt/2) * F(ψ_old)
    writeWaveB(i, psi + (params.dt * 0.5) * schrodRHS(psi, lap, V));
}

// ---------------------------------------------------------------------------
// CN step 2 — fixed-point iterate: waveA ← RHS(waveB) + (dt/2)*F(waveA)
// Dispatch this 4 times for convergence.
// ---------------------------------------------------------------------------
@compute @workgroup_size(4, 4, 4)
fn cn_iter(@builtin(global_invocation_id) gid: vec3u) {
    let N  = i32(params.gridN);
    let ix = i32(gid.x); let iy = i32(gid.y); let iz = i32(gid.z);
    if (ix >= N || iy >= N || iz >= N) { return; }
    let i = idx3(ix, iy, iz, N);
    if (isBoundary(ix, iy, iz, N)) { writeWaveA(i, vec2f(0.0)); return; }

    let rhsVal = readWaveB(i);
    let lap    = lapA(ix, iy, iz, N);   // Laplacian of current waveA guess
    let V      = potBuf[i];
    writeWaveA(i, rhsVal + (params.dt * 0.5) * schrodRHS(readWaveA(i), lap, V));
}

// ---------------------------------------------------------------------------
// CN step 3 — apply absorbing layer to waveA, copy result to waveB
// After this, waveB holds the completed new ψ.
// ---------------------------------------------------------------------------
@compute @workgroup_size(4, 4, 4)
fn cn_finish(@builtin(global_invocation_id) gid: vec3u) {
    let N  = i32(params.gridN);
    let ix = i32(gid.x); let iy = i32(gid.y); let iz = i32(gid.z);
    if (ix >= N || iy >= N || iz >= N) { return; }
    let i = idx3(ix, iy, iz, N);
    writeWaveB(i, readWaveA(i) * absorbDamp(ix, iy, iz, N));
}
