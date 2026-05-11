struct VertexInput {
    @location(0) position: vec3f,
    @builtin(vertex_index) vertexIndex: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) visible: f32,
};

struct OrbitalUniform {
    viewProj: mat4x4f,
    clipOrigin: vec4f,
    quantum: vec4f, // x:n, y:l, z:m, w:colorMode
    render: vec4f,  // x:removedOctant, y:intensityScale, z:simulationTime, w:seed
};

@group(0) @binding(0) var<uniform> orbital: OrbitalUniform;

const RADIAL_BINS: i32 = 384;
const THETA_BINS: i32 = 256;

fn hash32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

fn rand01(state: ptr<function, u32>) -> f32 {
    (*state) = hash32((*state) + 0x9e3779b9u);
    let mantissa = (*state) & 0x00ffffffu;
    return f32(mantissa) / 16777216.0;
}

fn associatedLaguerre(k: i32, alpha: i32, x: f32) -> f32 {
    if (k <= 0) {
        return 1.0;
    }
    var lm2 = 1.0;
    var lm1 = 1.0 + f32(alpha) - x;
    if (k == 1) {
        return lm1;
    }

    var l = lm1;
    var j = 2;
    loop {
        if (j > k) {
            break;
        }
        l = ((f32(2 * j - 1 + alpha) - x) * lm1 - f32(j - 1 + alpha) * lm2) / f32(j);
        lm2 = lm1;
        lm1 = l;
        j += 1;
    }
    return l;
}

fn associatedLegendre(l: i32, mAbs: i32, x: f32) -> f32 {
    var pmm = 1.0;
    if (mAbs > 0) {
        let somx2 = sqrt(max(0.0, (1.0 - x) * (1.0 + x)));
        var fact = 1.0;
        var j = 1;
        loop {
            if (j > mAbs) {
                break;
            }
            pmm = pmm * (-fact) * somx2;
            fact += 2.0;
            j += 1;
        }
    }

    if (l == mAbs) {
        return pmm;
    }

    var pm1m = x * f32(2 * mAbs + 1) * pmm;
    if (l == mAbs + 1) {
        return pm1m;
    }

    var pll = pm1m;
    var ll = mAbs + 2;
    loop {
        if (ll > l) {
            break;
        }
        pll = ((f32(2 * ll - 1) * x * pm1m) - (f32(ll + mAbs - 1) * pmm)) / f32(ll - mAbs);
        pmm = pm1m;
        pm1m = pll;
        ll += 1;
    }
    return pll;
}

fn factorialInt(v: i32) -> f32 {
    if (v <= 1) {
        return 1.0;
    }
    var out = 1.0;
    var i = 2;
    loop {
        if (i > v) {
            break;
        }
        out *= f32(i);
        i += 1;
    }
    return out;
}

fn radialPdf(r: f32, n: i32, l: i32) -> f32 {
    let nn = max(f32(n), 1.0);
    let rho = 2.0 * r / nn;
    let k = n - l - 1;
    let alpha = 2 * l + 1;
    let lag = associatedLaguerre(max(k, 0), alpha, rho);
    let num = factorialInt(max(n - l - 1, 0));
    let den = factorialInt(max(n + l, 0));
    let norm = pow(2.0 / nn, 3.0) * num / (2.0 * nn * max(den, 1.0));
    let radial = sqrt(max(norm, 0.0)) * exp(-rho * 0.5) * pow(max(rho, 0.0001), f32(l)) * lag;
    return max(0.0, r * r * radial * radial);
}

fn thetaPdf(theta: f32, l: i32, mAbs: i32) -> f32 {
    let x = cos(theta);
    let plm = associatedLegendre(l, mAbs, x);
    return max(0.0, sin(theta) * plm * plm);
}

fn sampleRFromCDF(u: f32, n: i32, l: i32) -> f32 {
    let nF = max(f32(n), 1.0);
    let lF = max(f32(l), 0.0);
    let expectedR = max(0.5 * (3.0 * nF * nF - lF * (lF + 1.0)), 0.0);
    let rMax = 12.0 * nF * nF;
    let dr = rMax / f32(RADIAL_BINS - 1);
    var cdf: array<f32, 384>;
    var pdfMax = 0.0;

    var i = 0;
    loop {
        if (i >= RADIAL_BINS) {
            break;
        }
        let r = f32(i) * dr;
        pdfMax = max(pdfMax, radialPdf(r, n, l));
        i += 1;
    }

    if (pdfMax <= 1e-20) {
        return clamp(expectedR, 0.0, rMax);
    }

    var sum = 0.0;
    i = 0;
    loop {
        if (i >= RADIAL_BINS) {
            break;
        }
        let r = f32(i) * dr;
        sum += radialPdf(r, n, l) / pdfMax;
        cdf[i] = sum;
        i += 1;
    }

    if (sum <= 1e-8) {
        return clamp(expectedR, 0.0, rMax);
    }

    let sampleTarget = u * sum;
    var lo = 0;
    var hi = RADIAL_BINS - 1;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2;
        if (cdf[mid] < sampleTarget) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo <= 0) {
        return 0.0;
    }

    let prev = lo - 1;
    let c0 = cdf[prev];
    let c1 = cdf[lo];
    let denom = max(c1 - c0, 1e-8);
    let t = clamp((sampleTarget - c0) / denom, 0.0, 1.0);
    return (f32(prev) + t) * dr;
}

fn sampleThetaFromCDF(u: f32, l: i32, mAbs: i32) -> f32 {
    let dTheta = 3.14159265358979323846 / f32(THETA_BINS - 1);
    var cdf: array<f32, 256>;
    var pdfMax = 0.0;

    var i = 0;
    loop {
        if (i >= THETA_BINS) {
            break;
        }
        let theta = f32(i) * dTheta;
        pdfMax = max(pdfMax, thetaPdf(theta, l, mAbs));
        i += 1;
    }

    if (pdfMax <= 1e-20) {
        return 0.5 * 3.14159265358979323846;
    }

    var sum = 0.0;
    i = 0;
    loop {
        if (i >= THETA_BINS) {
            break;
        }
        let theta = f32(i) * dTheta;
        sum += thetaPdf(theta, l, mAbs) / pdfMax;
        cdf[i] = sum;
        i += 1;
    }

    if (sum <= 1e-8) {
        return 0.5 * 3.14159265358979323846;
    }

    let sampleTarget = u * sum;
    var lo = 0;
    var hi = THETA_BINS - 1;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2;
        if (cdf[mid] < sampleTarget) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo <= 0) {
        return 0.0;
    }

    let prev = lo - 1;
    let c0 = cdf[prev];
    let c1 = cdf[lo];
    let denom = max(c1 - c0, 1e-8);
    let t = clamp((sampleTarget - c0) / denom, 0.0, 1.0);
    return (f32(prev) + t) * dTheta;
}

fn sphericalToCartesian(r: f32, theta: f32, phi: f32) -> vec3f {
    return vec3f(
        r * sin(theta) * cos(phi),
        r * cos(theta),
        r * sin(theta) * sin(phi)
    );
}

fn inRemovedOctant(pos: vec3f, origin: vec3f, removedOctant: i32) -> bool {
    let v = pos - origin;
    if (removedOctant == 1) { return v.x >= 0.0 && v.y >= 0.0 && v.z >= 0.0; }
    if (removedOctant == 2) { return v.x < 0.0 && v.y >= 0.0 && v.z >= 0.0; }
    if (removedOctant == 3) { return v.x < 0.0 && v.y < 0.0 && v.z >= 0.0; }
    if (removedOctant == 4) { return v.x >= 0.0 && v.y < 0.0 && v.z >= 0.0; }
    if (removedOctant == 5) { return v.x >= 0.0 && v.y >= 0.0 && v.z < 0.0; }
    if (removedOctant == 6) { return v.x < 0.0 && v.y >= 0.0 && v.z < 0.0; }
    if (removedOctant == 7) { return v.x < 0.0 && v.y < 0.0 && v.z < 0.0; }
    return v.x >= 0.0 && v.y < 0.0 && v.z < 0.0;
}

fn intensityAt(pos: vec3f, n: i32, l: i32, m: i32, intensityScale: f32, intensityRange: f32) -> f32 {
    let r = length(pos);
    let theta = acos(clamp(pos.y / max(r, 0.0001), -1.0, 1.0));

    let rho = 2.0 * r / max(f32(n), 1.0);
    let k = n - l - 1;
    let alpha = 2 * l + 1;
    let laguerre = associatedLaguerre(max(k, 0), alpha, rho);
    let nn = max(f32(n), 1.0);
    let num = factorialInt(max(n - l - 1, 0));
    let den = factorialInt(max(n + l, 0));
    let norm = pow(2.0 / nn, 3.0) * num / (2.0 * nn * max(den, 1.0));
    let radial = sqrt(max(norm, 0.0)) * exp(-rho * 0.5) * pow(max(rho, 0.0001), f32(l)) * laguerre;
    let radialP = radial * radial;

    let x = cos(theta);
    let mAbs = abs(m);
    let plm = associatedLegendre(l, mAbs, x);
    let angNum = factorialInt(max(l - mAbs, 0));
    let angDen = max(factorialInt(max(l + mAbs, 0)), 1e-8);
    let yNorm = (f32(2 * l + 1) / (4.0 * 3.14159265358979323846)) * (angNum / angDen);
    let angular = yNorm * plm * plm;

    let raw = radialP * angular;
    let nNorm = pow(max(f32(n), 1.0), 3.0);
    let scaled = raw * max(intensityScale, 0.0001) * 30.0 * nNorm;
    return clamp(scaled / (scaled + max(intensityRange, 0.0001)), 0.0, 1.0);
}

fn samplePalette6(x: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f) -> vec3f {
    let u = clamp(x, 0.0, 1.0) * 5.0;
    if (u < 1.0) { return c0 + u * (c1 - c0); }
    if (u < 2.0) { return c1 + (u - 1.0) * (c2 - c1); }
    if (u < 3.0) { return c2 + (u - 2.0) * (c3 - c2); }
    if (u < 4.0) { return c3 + (u - 3.0) * (c4 - c3); }
    return c4 + (u - 4.0) * (c5 - c4);
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

fn mapColor(t: f32, colorMode: i32, samplePos: vec3f, simulationTime: f32, n: i32, m: i32, localOmega: f32) -> vec3f {
    let x = clamp(t, 0.0, 1.0);
    if (colorMode == 0) {
        return samplePalette6(x,
            vec3f(0.0015, 0.0005, 0.0139),
            vec3f(0.1462, 0.0449, 0.3374),
            vec3f(0.3904, 0.1004, 0.5019),
            vec3f(0.6663, 0.1819, 0.3698),
            vec3f(0.9018, 0.4251, 0.1081),
            vec3f(0.9884, 0.9984, 0.6449));
    }
    if (colorMode == 1) {
        return samplePalette6(x,
            vec3f(0.0015, 0.0005, 0.0139),
            vec3f(0.1717, 0.0673, 0.3708),
            vec3f(0.4452, 0.1227, 0.5069),
            vec3f(0.7164, 0.2150, 0.4753),
            vec3f(0.9440, 0.3776, 0.3651),
            vec3f(0.9871, 0.9914, 0.7495));
    }
    if (colorMode == 2) {
        return samplePalette6(x,
            vec3f(0.0504, 0.0298, 0.5280),
            vec3f(0.4176, 0.0006, 0.6584),
            vec3f(0.6928, 0.1651, 0.5645),
            vec3f(0.8814, 0.3925, 0.3832),
            vec3f(0.9883, 0.6523, 0.2114),
            vec3f(0.9400, 0.9752, 0.1313));
    }
    if (colorMode == 3) {
        return samplePalette6(x,
            vec3f(0.2670, 0.0049, 0.3294),
            vec3f(0.2539, 0.2653, 0.5300),
            vec3f(0.1636, 0.4711, 0.5581),
            vec3f(0.1347, 0.6586, 0.5176),
            vec3f(0.4775, 0.8214, 0.3182),
            vec3f(0.9932, 0.9062, 0.1439));
    }
    if (colorMode == 4) {
        return samplePalette6(x,
            vec3f(0.0000, 0.1262, 0.3015),
            vec3f(0.2081, 0.2666, 0.4622),
            vec3f(0.3390, 0.4306, 0.5270),
            vec3f(0.4887, 0.5864, 0.5053),
            vec3f(0.6785, 0.7335, 0.3791),
            vec3f(0.9957, 0.9093, 0.2178));
    }
    if (colorMode == 5) {
        return samplePalette6(x,
            vec3f(0.1900, 0.0718, 0.2322),
            vec3f(0.2511, 0.2524, 0.6337),
            vec3f(0.2763, 0.4774, 0.9308),
            vec3f(0.1637, 0.7170, 0.8030),
            vec3f(0.7560, 0.8940, 0.2260),
            vec3f(0.9840, 0.4920, 0.1280));
    }
    if (colorMode == 6) {
        return vec3f(x, x, x);
    }
    if (colorMode == 8) {
        return vec3f(0.2 + 0.8 * x, 1.0 - 0.6 * x, 1.0);
    }
    if (colorMode == 9) {
        let basePhi = atan2(samplePos.z, samplePos.x) / 6.28318530717958647692;
        let hue = fract(basePhi + simulationTime * (0.06 * localOmega));
        return hsvToRgb(hue, 0.92, 0.22 + 0.78 * x);
    }
    if (colorMode == 10) {
        let basePhi = f32(m) * atan2(samplePos.z, samplePos.x) / 6.28318530717958647692;
        let energyOmega = 1.0 / max(f32(n * n), 1.0);
        let hue = fract(basePhi + simulationTime * (0.9 * energyOmega));
        return hsvToRgb(hue, 0.88, 0.24 + 0.76 * x);
    }

    return samplePalette6(x,
        vec3f(0.0, 0.0, 0.0),
        vec3f(0.5, 0.0, 0.99),
        vec3f(0.8, 0.0, 0.0),
        vec3f(1.0, 0.5, 0.0),
        vec3f(1.0, 1.0, 0.0),
        vec3f(1.0, 1.0, 1.0));
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output : VertexOutput;

    let directMode = orbital.quantum.x < 0.0;

    if (directMode) {
        let samplePos = input.position;
        let colorMode = i32(round(orbital.quantum.w));
        let removedOctant = i32(round(orbital.render.x));
        let simulationTime = orbital.render.z;

        let hidden = inRemovedOctant(samplePos, orbital.clipOrigin.xyz, removedOctant);
        let r = length(samplePos);
        let t = clamp(exp(-0.06 * r * r), 0.0, 1.0);
        let omega = 0.5 / max(r, 0.1);

        output.position = orbital.viewProj * vec4f(samplePos, 1.0);
        output.color = mapColor(t, colorMode, samplePos, simulationTime, 1, 0, omega);
        output.visible = select(1.0, 0.0, hidden);
        return output;
    }

    let n = max(1, i32(round(input.position.x)));
    let lRaw = i32(round(input.position.y));
    let l = clamp(lRaw, 0, n - 1);
    let mRaw = i32(round(input.position.z));
    let m = clamp(mRaw, -l, l);
    let colorMode = i32(round(orbital.quantum.w));

    let removedOctant = i32(round(orbital.render.x));
    let intensityScale = orbital.render.y;
    let simulationTime = orbital.render.z;
    let seed = u32(max(1.0, orbital.render.w));
    let intensityRange = orbital.clipOrigin.w;

    var rngState = hash32(input.vertexIndex ^ seed ^ 0x68bc21ebu);
    let r = sampleRFromCDF(rand01(&rngState), n, l);
    let theta = sampleThetaFromCDF(rand01(&rngState), l, abs(m));
    let phi = rand01(&rngState) * 6.28318530717958647692;

    let safeR = max(r, 0.04);
    let safeSin = max(abs(sin(theta)), 0.04);
    let omega = f32(m) / (safeR * safeSin);
    let advectedPhi = phi + simulationTime * omega;
    let samplePos = sphericalToCartesian(r, theta, advectedPhi);

    let hidden = inRemovedOctant(samplePos, orbital.clipOrigin.xyz, removedOctant);
    let t = intensityAt(samplePos, n, l, m, intensityScale, intensityRange);

    output.position = orbital.viewProj * vec4f(samplePos, 1.0);
    output.color = mapColor(t, colorMode, samplePos, simulationTime, n, m, omega);
    output.visible = select(1.0, 0.0, hidden);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if (input.visible < 0.5) {
        discard;
    }
    return vec4f(input.color, 1.0);
}
