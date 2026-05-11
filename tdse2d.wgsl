struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct TwoDUniform {
    orbital: vec4f, // unused in TDSE mode
    tuning: vec4f,  // x:intensityScale, y:intensityRange, z:zoom, w:unused
    render: vec4f,  // x:time, y:aspect ratio, z:phase speed, w:unused
    pan: vec4f,     // x:panX, y:panY, z:unused, w:unused
    tdse: vec4f,    // x:gridSize, y:domainHalfExtent, z:potentialOverlay, w:reserved
};

@group(0) @binding(0) var<uniform> u: TwoDUniform;

struct WaveCell {
    value: vec4f,
};

@group(0) @binding(1) var<storage, read> waveGrid: array<WaveCell>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(input.position, 1.0);
    out.uv = input.position.xy;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let intensityScale = max(u.tuning.x, 0.0001);
    let intensityRange = max(u.tuning.y, 0.0001);
    let zoom = max(u.tuning.z, 0.0001);
    let aspect = max(u.render.y, 0.0001);
    let phaseSpeed = u.render.z;
    let time = u.render.x;
    let colorMode = i32(round(u.orbital.w));
    let gridSize = max(2, i32(round(u.tdse.x)));
    let domainHalf = max(u.tdse.y, 0.001);
    let showPotentialOverlay = u.tdse.z > 0.5;

    var uv = input.uv;
    uv.x *= aspect;
    uv /= zoom;
    uv -= u.pan.xy;

    let nx = 0.5 * (uv.x / domainHalf + 1.0);
    let ny = 0.5 * (uv.y / domainHalf + 1.0);
    if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0) {
        return vec4f(0.01, 0.01, 0.02, 1.0);
    }

    let ix = clamp(i32(floor(nx * f32(gridSize - 1))), 0, gridSize - 1);
    let iy = clamp(i32(floor(ny * f32(gridSize - 1))), 0, gridSize - 1);
    let idx = iy * gridSize + ix;
    let cell = waveGrid[idx].value;
    let rho = max(cell.x, 0.0);
    let phase01 = fract(cell.y);
    let potential01 = clamp(cell.z, 0.0, 1.0);

    let scaled = rho * max(intensityScale * 120.0, 0.0001);
    let bright = clamp(scaled / (scaled + intensityRange), 0.0, 1.0);

    var color = vec3f(0.0);
    
    if (colorMode == 10 || colorMode == 9) {
        color = hsvToRgb(phase01, 0.92, 0.20 + 0.80 * bright);
    } else {
        color = mapColor(bright, colorMode, vec3f(uv.x, uv.y, 0.0), time * phaseSpeed, 1, 0, 1.0);
    }

    if (showPotentialOverlay) {
        let potColor = mix(vec3f(0.15, 0.45, 0.95), vec3f(0.95, 0.25, 0.10), potential01);
        color = mix(color, potColor, 0.32 * (0.25 + 0.75 * potential01));
    }

    let bg = vec3f(0.01, 0.01, 0.02);

    return vec4f(mix(bg, color, bright), 1.0);
}

fn hsvToRgb(h: f32, s: f32, v: f32) -> vec3f {
    let hue = fract(h);
    let hh = hue * 6.0;
    let c = v * s;
    let x = c * (1.0 - abs(fract(hh) * 2.0 - 1.0));
    let m = v - c;
    if (hh < 1.0) { return vec3f(c + m, x + m, m); }
    if (hh < 2.0) { return vec3f(x + m, c + m, m); }
    if (hh < 3.0) { return vec3f(m, c + m, x + m); }
    if (hh < 4.0) { return vec3f(m, x + m, c + m); }
    if (hh < 5.0) { return vec3f(x + m, m, c + m); }
    return vec3f(c + m, m, x + m);
}

fn samplePalette6(x: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f) -> vec3f {
    let u = clamp(x, 0.0, 1.0) * 5.0;
    if (u < 1.0) { return c0 + u * (c1 - c0); }
    if (u < 2.0) { return c1 + (u - 1.0) * (c2 - c1); }
    if (u < 3.0) { return c2 + (u - 2.0) * (c3 - c2); }
    if (u < 4.0) { return c3 + (u - 3.0) * (c4 - c3); }
    return c4 + (u - 4.0) * (c5 - c4);
}

fn mapColor(t: f32, colorMode: i32, samplePos: vec3f, simulationTime: f32, n: i32, m: i32, localOmega: f32) -> vec3f {
    let x = clamp(t, 0.0, 1.0);
    if (colorMode == 0) {
        return samplePalette6(x, vec3f(0.0015, 0.0005, 0.0139), vec3f(0.1462, 0.0449, 0.3374), vec3f(0.3904, 0.1004, 0.5019), vec3f(0.6663, 0.1819, 0.3698), vec3f(0.9018, 0.4251, 0.1081), vec3f(0.9884, 0.9984, 0.6449));
    }
    if (colorMode == 1) {
        return samplePalette6(x, vec3f(0.0015, 0.0005, 0.0139), vec3f(0.1717, 0.0673, 0.3708), vec3f(0.4452, 0.1227, 0.5069), vec3f(0.7164, 0.2150, 0.4753), vec3f(0.9440, 0.3776, 0.3651), vec3f(0.9871, 0.9914, 0.7495));
    }
    if (colorMode == 2) {
        return samplePalette6(x, vec3f(0.0504, 0.0298, 0.5280), vec3f(0.4176, 0.0006, 0.6584), vec3f(0.6928, 0.1651, 0.5645), vec3f(0.8814, 0.3925, 0.3832), vec3f(0.9883, 0.6523, 0.2114), vec3f(0.9400, 0.9752, 0.1313));
    }
    if (colorMode == 3) {
        return samplePalette6(x, vec3f(0.2670, 0.0049, 0.3294), vec3f(0.2539, 0.2653, 0.5300), vec3f(0.1636, 0.4711, 0.5581), vec3f(0.1347, 0.6586, 0.5176), vec3f(0.4775, 0.8214, 0.3182), vec3f(0.9932, 0.9062, 0.1439));
    }
    if (colorMode == 4) {
        return samplePalette6(x, vec3f(0.0000, 0.1262, 0.3015), vec3f(0.2081, 0.2666, 0.4622), vec3f(0.3390, 0.4306, 0.5270), vec3f(0.4887, 0.5864, 0.5053), vec3f(0.6785, 0.7335, 0.3791), vec3f(0.9957, 0.9093, 0.2178));
    }
    if (colorMode == 5) {
        return samplePalette6(x, vec3f(0.1900, 0.0718, 0.2322), vec3f(0.2511, 0.2524, 0.6337), vec3f(0.2763, 0.4774, 0.9308), vec3f(0.1637, 0.7170, 0.8030), vec3f(0.7560, 0.8940, 0.2260), vec3f(0.9840, 0.4920, 0.1280));
    }
    if (colorMode == 6) {
        return vec3f(x, x, x);
    }
    if (colorMode == 8) {
        return vec3f(0.2 + 0.8 * x, 1.0 - 0.6 * x, 1.0);
    }
    if (colorMode == 9) {
        let basePhi = atan2(samplePos.y, samplePos.x) / 6.28318530717958647692;
        let hue = fract(basePhi + simulationTime * (0.06 * localOmega));
        return hsvToRgb(hue, 0.92, 0.22 + 0.78 * x);
    }
    if (colorMode == 10) {
        let basePhi = f32(m) * atan2(samplePos.y, samplePos.x) / 6.28318530717958647692;
        let energyOmega = 1.0 / max(f32(n * n), 1.0);
        let hue = fract(basePhi + simulationTime * (0.9 * energyOmega));
        return hsvToRgb(hue, 0.88, 0.24 + 0.76 * x);
    }
    return samplePalette6(x, vec3f(0.0, 0.0, 0.0), vec3f(0.5, 0.0, 0.99), vec3f(0.8, 0.0, 0.0), vec3f(1.0, 0.5, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(1.0, 1.0, 1.0));
}
