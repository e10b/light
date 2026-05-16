struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct Params {
    orbital: vec4f, // x:maxBounces, y:spp, z:dispersion, w:roughness
    tuning: vec4f,  // x:camPos.x, y:camPos.y, z:camPos.z, w:exposure
    render: vec4f,  // x:time, y:aspect, z:yaw, w:pitch
    pan: vec4f,     // x:envBrightness, y:envMode, z:envRotation
    tdse: vec4f,
};

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var envTex: texture_2d<f32>;
@group(0) @binding(2) var envSamp: sampler;
struct BvhNode {
    bminLeft: vec4f,  // xyz: min, w: leftFirst
    bmaxCount: vec4f, // xyz: max, w: count (leaf if > 0)
};
struct Tri {
    v0: vec4f,
    v1: vec4f,
    v2: vec4f,
};
@group(0) @binding(3) var<storage, read> bvhNodes: array<BvhNode>;
@group(0) @binding(4) var<storage, read> bvhTris: array<Tri>;
@group(0) @binding(5) var<storage, read_write> accumA: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> accumB: array<vec4f>;
struct AccumParams { data: vec4f, };
@group(0) @binding(7) var<uniform> accum: AccumParams;
@group(0) @binding(8) var<storage, read_write> photonA: array<atomic<u32>>;
struct PhotonParams { data: vec4f, };
@group(0) @binding(9) var<uniform> photon: PhotonParams;
@group(0) @binding(10) var<storage, read_write> photonB: array<atomic<u32>>;

const PI: f32 = 3.14159265359;
const GROUND_Y: f32 = -0.62;
const PHOTON_MIN_XZ: vec2f = vec2f(-2.8, -2.8);
const PHOTON_MAX_XZ: vec2f = vec2f(2.8, 2.8);

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(input.position, 1.0);
    out.uv = input.position.xy;
    return out;
}

fn hash12(p: vec2f) -> f32 {
    let h = dot(p, vec2f(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn hash13(p: vec3f) -> f32 {
    let h = dot(p, vec3f(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

fn acesToneMap(color: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
}

fn wavelengthToRgb(lambdaNm: f32) -> vec3f {
    let t = clamp((lambdaNm - 380.0) / 400.0, 0.0, 1.0);
    let r = smoothstep(0.45, 0.85, t) + (1.0 - smoothstep(0.0, 0.15, t)) * 0.35;
    let g = smoothstep(0.1, 0.45, t) * (1.0 - smoothstep(0.65, 0.9, t));
    let b = (1.0 - smoothstep(0.2, 0.55, t)) + smoothstep(0.88, 1.0, t) * 0.2;
    return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
}

fn preethamPerez(cosTheta: f32, gamma: f32, cosGamma: f32, a: f32, b: f32, c: f32, d: f32, e: f32) -> f32 {
    let ct = max(cosTheta, 0.01);
    return (1.0 + a * exp(b / ct)) * (1.0 + c * exp(d * gamma) + e * cosGamma * cosGamma);
}

fn xyzToSrgb(xyz: vec3f) -> vec3f {
    let rgb = mat3x3f(
        3.2406, -0.9689,  0.0557,
       -1.5372,  1.8758, -0.2040,
       -0.4986,  0.0415,  1.0570
    ) * xyz;
    return max(rgb, vec3f(0.0));
}

fn envSky(d: vec3f) -> vec3f {
    let dir = normalize(d);
    let up = vec3f(0.0, 1.0, 0.0);
    let sunDir = sunDirection();
    let cosTheta = clamp(dot(dir, up), 0.0, 1.0);
    let theta = acos(clamp(cosTheta, 0.0, 1.0));
    let cosThetaS = clamp(dot(sunDir, up), 0.001, 1.0);
    let thetaS = acos(cosThetaS);
    let cosGamma = clamp(dot(dir, sunDir), -1.0, 1.0);
    let gamma = acos(cosGamma);

    let T = 3.0;
    let T2 = T * T;

    let Ay = 0.1787 * T - 1.4630;
    let By = -0.3554 * T + 0.4275;
    let Cy = -0.0227 * T + 5.3251;
    let Dy = 0.1206 * T - 2.5771;
    let Ey = -0.0670 * T + 0.3703;

    let Ax = -0.0193 * T - 0.2592;
    let Bx = -0.0665 * T + 0.0008;
    let Cx = -0.0004 * T + 0.2125;
    let Dx = -0.0641 * T - 0.8989;
    let Ex = -0.0033 * T + 0.0452;

    let Az = -0.0167 * T - 0.2608;
    let Bz = -0.0950 * T + 0.0092;
    let Cz = -0.0079 * T + 0.2102;
    let Dz = -0.0441 * T - 1.6537;
    let Ez = -0.0109 * T + 0.0529;

    let chi = (4.0 / 9.0 - T / 120.0) * (PI - 2.0 * thetaS);
    let Yz = (4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192;
    let xz = (0.00165 * thetaS * thetaS * thetaS - 0.00374 * thetaS * thetaS + 0.00208 * thetaS) * T2 +
             (-0.02902 * thetaS * thetaS * thetaS + 0.06377 * thetaS * thetaS - 0.03202 * thetaS + 0.00394) * T +
             (0.11693 * thetaS * thetaS * thetaS - 0.21196 * thetaS * thetaS + 0.06052 * thetaS + 0.25886);
    let yz = (0.00275 * thetaS * thetaS * thetaS - 0.00610 * thetaS * thetaS + 0.00316 * thetaS) * T2 +
             (-0.04214 * thetaS * thetaS * thetaS + 0.08970 * thetaS * thetaS - 0.04153 * thetaS + 0.00515) * T +
             (0.15346 * thetaS * thetaS * thetaS - 0.26756 * thetaS * thetaS + 0.06669 * thetaS + 0.26688);

    let FyThetaGamma = preethamPerez(cosTheta, gamma, cosGamma, Ay, By, Cy, Dy, Ey);
    let FxThetaGamma = preethamPerez(cosTheta, gamma, cosGamma, Ax, Bx, Cx, Dx, Ex);
    let FzThetaGamma = preethamPerez(cosTheta, gamma, cosGamma, Az, Bz, Cz, Dz, Ez);

    let FySun = preethamPerez(cosThetaS, 0.0, 1.0, Ay, By, Cy, Dy, Ey);
    let FxSun = preethamPerez(cosThetaS, 0.0, 1.0, Ax, Bx, Cx, Dx, Ex);
    let FzSun = preethamPerez(cosThetaS, 0.0, 1.0, Az, Bz, Cz, Dz, Ez);

    let Y = max(Yz * FyThetaGamma / max(FySun, 1e-4), 0.0);
    let x = clamp(xz * FxThetaGamma / max(FxSun, 1e-4), 0.001, 0.999);
    let y = clamp(yz * FzThetaGamma / max(FzSun, 1e-4), 0.001, 0.999);

    let X = (x / y) * Y;
    let Z = ((1.0 - x - y) / y) * Y;
    var sky = xyzToSrgb(vec3f(X, Y, Z)) * 0.12;

    let sunCos = cos(0.53 * PI / 180.0 * 0.5);
    let sunDisk = smoothstep(sunCos - 0.0008, sunCos + 0.0002, cosGamma);
    sky += vec3f(1.0, 0.98, 0.93) * sunDisk * max(u.tdse.z, 0.0) * 0.4;
    return sky;
}

fn envSunset(d: vec3f) -> vec3f {
    let t = clamp(0.5 * (d.y + 1.0), 0.0, 1.0);
    let low = vec3f(0.95, 0.33, 0.14);
    let mid = vec3f(0.95, 0.64, 0.30);
    let top = vec3f(0.12, 0.25, 0.52);
    let warm = mix(low, mid, smoothstep(0.0, 0.35, t));
    let sky = mix(warm, top, smoothstep(0.35, 1.0, t));
    let sun = vec3f(1.0, 0.8, 0.52) * pow(max(dot(d, normalize(vec3f(-0.1, 0.25, -0.95))), 0.0), 700.0);
    return sky + sun;
}

fn envHdrLatLong(d: vec3f) -> vec3f {
    let rot = u.pan.z;
    let cr = cos(rot);
    let sr = sin(rot);
    let dr = vec3f(
        d.x * cr - d.z * sr,
        d.y,
        d.x * sr + d.z * cr
    );
    let phi = atan2(dr.z, dr.x);
    let theta = acos(clamp(dr.y, -1.0, 1.0));
    let uv = vec2f((phi + PI) / (2.0 * PI), theta / PI);
    return textureSample(envTex, envSamp, uv).rgb;
}

fn sampleEnvironment(d: vec3f) -> vec3f {
    let mode = i32(clamp(u.pan.y, 0.0, 2.0));
    if (mode == 0) {
        return envHdrLatLong(d);
    }
    if (mode == 1) {
        return envSky(d);
    }
    return envSunset(d);
}

fn sunDirection() -> vec3f {
    let az = u.tdse.x;
    let el = clamp(u.tdse.y, 0.001, 1.56);
    let ce = cos(el);
    return normalize(vec3f(
        cos(az) * ce,
        sin(el),
        sin(az) * ce
    ));
}

fn spotlightDirection() -> vec3f {
    // Rotation-controlled spotlight orientation.
    let az = u.tdse.x;
    let el = clamp(u.tdse.y, 0.001, 1.56);
    let ce = cos(el);
    return normalize(vec3f(
        cos(az) * ce,
        -sin(el),
        sin(az) * ce
    ));
}

fn spotlightPosition() -> vec3f {
    // Position the spotlight above and to the side of the scene
    return vec3f(1.5, 2.5, 1.0);
}

fn spotLight(p: vec3f, n: vec3f) -> vec3f {
    let spotPos = spotlightPosition();
    let spotDir = spotlightDirection();
    let intensity = max(u.tdse.z, 0.0);
    let softness = max(u.tdse.w, 16.0);
    
    // Vector from surface point to light
    let toLight = spotPos - p;
    let dist = length(toLight);
    let L = normalize(toLight);
    
    // Spotlight cone angle (cosine of the cone angle)
    let coneAngle = 0.85; // ~32 degrees
    let coneSoftness = 0.75; // Soft edge
    
    // Check if point is within spotlight cone
    let spotEffect = dot(-L, spotDir);
    let spotAttenuation = smoothstep(coneSoftness, coneAngle, spotEffect);
    
    // Distance attenuation (inverse square with some artistic control)
    let distAttenuation = 1.0 / (1.0 + 0.1 * dist + 0.01 * dist * dist);
    
    // Lambertian diffuse
    let ndotl = max(dot(n, L), 0.0);
    
    // Shadow test - trace through glass with refraction
    let shadowBiasN = select(-n, n, dot(n, L) >= 0.0);
    let shadowRayOrigin = p + shadowBiasN * 0.01;
    
    // Check sphere first
    let sphereCenter = vec3f(-0.8, -0.12, 0.5);
    let sphereRadius = 0.5;
    let sphereShadow = intersectSphere(shadowRayOrigin, L, sphereCenter, sphereRadius);
    let prismShadow = intersectPrism(shadowRayOrigin, L);
    
    // Check mesh
    let meshShadow = intersectMesh(shadowRayOrigin, L);
    
    var shadowFactor = 1.0;
    
    // If we hit glass, we need to trace THROUGH it with refraction
    // This is where caustics come from!
    let hitGlass = (meshShadow.w > 0.0 && meshShadow.w < (dist - 0.01)) ||
                   (sphereShadow.w > 0.0 && sphereShadow.w < (dist - 0.01)) ||
                   (prismShadow.w > 0.0 && prismShadow.w < (dist - 0.01));
    
    if (hitGlass) {
        // Glass blocks direct light - it will be added via refracted paths
        shadowFactor = 0.0;
    }
    
    // Warm white light color
    let lightColor = vec3f(1.0, 0.98, 0.95);
    
    return lightColor * intensity * spotAttenuation * distAttenuation * ndotl * shadowFactor;
}

fn sunLamp(d: vec3f) -> vec3f {
    // Disabled - using spotlight instead
    return vec3f(0.0);
}

fn traceShadowThroughGlass(origin: vec3f, dir: vec3f, maxDist: f32) -> f32 {
    // Trace shadow ray through glass with refraction
    // This creates caustics when wavelength-dependent refraction focuses light
    
    var pos = origin;
    var direction = dir;
    var transmittance = 1.0;
    var traveled = 0.0;
    
    let sphereCenter = vec3f(-0.8, -0.12, 0.5);
    let sphereRadius = 0.5;
    
    // Trace through glass (up to 2 refractions: enter and exit)
    for (var i = 0; i < 2; i += 1) {
        let meshHit = intersectMesh(pos, direction);
        let sphereHit = intersectSphere(pos, direction, sphereCenter, sphereRadius);
        let prismHit = intersectPrism(pos, direction);
        
        // Find closest hit
        var hit = vec4f(0.0, 0.0, 0.0, -1.0);
        if (meshHit.w > 0.0 && sphereHit.w > 0.0) {
            hit = select(sphereHit, meshHit, meshHit.w < sphereHit.w);
        } else if (meshHit.w > 0.0) {
            hit = meshHit;
        } else if (sphereHit.w > 0.0) {
            hit = sphereHit;
        }
        
        if (hit.w <= 0.0) {
            // No more glass - check if we reached light
            return select(0.0, transmittance, traveled < maxDist);
        }
        
        traveled += hit.w;
        if (traveled > maxDist) {
            return 0.0;
        }
        
        // Hit glass - refract through it
        let hitPos = pos + direction * hit.w;
        let nRaw = hit.xyz;
        let entering = dot(direction, nRaw) < 0.0;
        let n = select(-nRaw, nRaw, entering);
        
        // Use wavelength-dependent IOR for caustics!
        // We don't have the wavelength here, so use a middle value
        // The spectral path tracer will sample different wavelengths
        let glassIor = 1.5;
        let etaI = select(glassIor, 1.0, entering);
        let etaT = select(1.0, glassIor, entering);
        let eta = etaI / etaT;
        
        let cosI = clamp(dot(-direction, n), 0.0, 1.0);
        let sin2T = eta * eta * (1.0 - cosI * cosI);
        
        if (sin2T > 1.0) {
            return 0.0; // TIR blocks light
        }
        
        let fresnel = schlick(cosI, etaI, etaT);
        transmittance *= (1.0 - fresnel);
        
        direction = refract(direction, n, eta);
        pos = hitPos + direction * 0.002;
        
        if (transmittance < 0.01) {
            return 0.0;
        }
    }
    
    return select(0.0, transmittance, traveled < maxDist);
}

fn intersectAabb(ro: vec3f, invRd: vec3f, bmin: vec3f, bmax: vec3f, tMax: f32) -> bool {
    let t0 = (bmin - ro) * invRd;
    let t1 = (bmax - ro) * invRd;
    let tsm = min(t0, t1);
    let tbg = max(t0, t1);
    let tn = max(max(tsm.x, tsm.y), max(tsm.z, 0.0));
    let tf = min(min(tbg.x, tbg.y), min(tbg.z, tMax));
    return tn <= tf;
}

fn intersectTri(ro: vec3f, rd: vec3f, tri: Tri, tBest: f32) -> vec4f {
    let v0 = tri.v0.xyz;
    let v1 = tri.v1.xyz;
    let v2 = tri.v2.xyz;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = cross(rd, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-7) {
        return vec4f(-1.0);
    }
    let invDet = 1.0 / det;
    let tvec = ro - v0;
    let u = dot(tvec, p) * invDet;
    if (u < 0.0 || u > 1.0) {
        return vec4f(-1.0);
    }
    let q = cross(tvec, e1);
    let v = dot(rd, q) * invDet;
    if (v < 0.0 || u + v > 1.0) {
        return vec4f(-1.0);
    }
    let t = dot(e2, q) * invDet;
    if (t <= 0.0005 || t >= tBest) {
        return vec4f(-1.0);
    }
    let n = normalize(cross(e1, e2));
    return vec4f(n, t);
}

fn intersectSphere(ro: vec3f, rd: vec3f, center: vec3f, radius: f32) -> vec4f {
    let oc = ro - center;
    let a = dot(rd, rd);
    let b = 2.0 * dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if (discriminant < 0.0) {
        return vec4f(0.0, 0.0, 0.0, -1.0);
    }
    
    let sqrtDisc = sqrt(discriminant);
    var t = (-b - sqrtDisc) / (2.0 * a);
    
    if (t < 0.0005) {
        t = (-b + sqrtDisc) / (2.0 * a);
    }
    
    if (t < 0.0005) {
        return vec4f(0.0, 0.0, 0.0, -1.0);
    }
    
    let hitPos = ro + rd * t;
    let normal = normalize(hitPos - center);
    return vec4f(normal, t);
}

fn intersectPrism(ro: vec3f, rd: vec3f) -> vec4f {
    // Triangular prism for visible dispersion.
    // Stand on the SMALLEST face (triangular end cap): triangle lies in XZ, extruded along Y.
    let c = vec2f(1.65, 1.20); // (x, z) center on ground area
    let v0 = c + vec2f(0.00, 0.42);
    let v1 = c + vec2f(-0.34, -0.16);
    let v2 = c + vec2f(0.34, -0.16);
    let yMin = GROUND_Y;
    let yMax = GROUND_Y + 0.70;

    var tEnter = -1e20;
    var tExit = 1e20;
    var nEnter = vec3f(0.0, 1.0, 0.0);
    var nExit = vec3f(0.0, -1.0, 0.0);
    let eps = 1e-6;

    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;
    let n0xz = normalize(vec2f(e0.y, -e0.x));
    let n1xz = normalize(vec2f(e1.y, -e1.x));
    let n2xz = normalize(vec2f(e2.y, -e2.x));
    let n0 = vec3f(n0xz.x, 0.0, n0xz.y);
    let n1 = vec3f(n1xz.x, 0.0, n1xz.y);
    let n2 = vec3f(n2xz.x, 0.0, n2xz.y);
    let d0 = -dot(n0xz, v0);
    let d1 = -dot(n1xz, v1);
    let d2 = -dot(n2xz, v2);

    let ny0 = vec3f(0.0, 1.0, 0.0);
    let dy0 = -yMax;
    let ny1 = vec3f(0.0, -1.0, 0.0);
    let dy1 = yMin;

    {
        let f = dot(n0, ro) + d0;
        let den = dot(n0, rd);
        if (abs(den) < eps) { if (f > 0.0) { return vec4f(0.0, 0.0, 0.0, -1.0); } }
        else {
            let t = -f / den;
            if (den < 0.0) { if (t > tEnter) { tEnter = t; nEnter = n0; } }
            else { if (t < tExit) { tExit = t; nExit = n0; } }
            if (tEnter > tExit) { return vec4f(0.0, 0.0, 0.0, -1.0); }
        }
    }
    {
        let f = dot(n1, ro) + d1;
        let den = dot(n1, rd);
        if (abs(den) < eps) { if (f > 0.0) { return vec4f(0.0, 0.0, 0.0, -1.0); } }
        else {
            let t = -f / den;
            if (den < 0.0) { if (t > tEnter) { tEnter = t; nEnter = n1; } }
            else { if (t < tExit) { tExit = t; nExit = n1; } }
            if (tEnter > tExit) { return vec4f(0.0, 0.0, 0.0, -1.0); }
        }
    }
    {
        let f = dot(n2, ro) + d2;
        let den = dot(n2, rd);
        if (abs(den) < eps) { if (f > 0.0) { return vec4f(0.0, 0.0, 0.0, -1.0); } }
        else {
            let t = -f / den;
            if (den < 0.0) { if (t > tEnter) { tEnter = t; nEnter = n2; } }
            else { if (t < tExit) { tExit = t; nExit = n2; } }
            if (tEnter > tExit) { return vec4f(0.0, 0.0, 0.0, -1.0); }
        }
    }
    {
        let f = dot(ny0, ro) + dy0;
        let den = dot(ny0, rd);
        if (abs(den) < eps) { if (f > 0.0) { return vec4f(0.0, 0.0, 0.0, -1.0); } }
        else {
            let t = -f / den;
            if (den < 0.0) { if (t > tEnter) { tEnter = t; nEnter = ny0; } }
            else { if (t < tExit) { tExit = t; nExit = ny0; } }
            if (tEnter > tExit) { return vec4f(0.0, 0.0, 0.0, -1.0); }
        }
    }
    {
        let f = dot(ny1, ro) + dy1;
        let den = dot(ny1, rd);
        if (abs(den) < eps) { if (f > 0.0) { return vec4f(0.0, 0.0, 0.0, -1.0); } }
        else {
            let t = -f / den;
            if (den < 0.0) { if (t > tEnter) { tEnter = t; nEnter = ny1; } }
            else { if (t < tExit) { tExit = t; nExit = ny1; } }
            if (tEnter > tExit) { return vec4f(0.0, 0.0, 0.0, -1.0); }
        }
    }

    if (tExit <= 0.0005) { return vec4f(0.0, 0.0, 0.0, -1.0); }
    if (tEnter > 0.0005) { return vec4f(nEnter, tEnter); }
    return vec4f(nExit, tExit);
}

fn intersectMesh(ro: vec3f, rd: vec3f) -> vec4f {
    var bestT = 1e20;
    var bestN = vec3f(0.0);

    var stack: array<i32, 64>;
    var sp = 0;
    stack[sp] = 0;
    sp += 1;
    let invRd = 1.0 / select(vec3f(1e-6), rd, abs(rd) > vec3f(1e-6));

    loop {
        if (sp <= 0) { break; }
        sp -= 1;
        let ni = stack[sp];
        if (ni < 0) { continue; }
        let node = bvhNodes[u32(ni)];
        let bmin = node.bminLeft.xyz;
        let bmax = node.bmaxCount.xyz;
        if (!intersectAabb(ro, invRd, bmin, bmax, bestT)) {
            continue;
        }
        let leftFirst = i32(node.bminLeft.w + 0.5);
        let enc = node.bmaxCount.w;
        let count = i32(enc + 0.5);
        if (count > 0) {
            for (var i = 0; i < count; i += 1) {
                let tri = bvhTris[u32(leftFirst + i)];
                let hit = intersectTri(ro, rd, tri, bestT);
                if (hit.w > 0.0) {
                    bestT = hit.w;
                    bestN = hit.xyz;
                }
            }
        } else {
            let right = i32(-enc - 1.0 + 0.5);
            if (sp + 2 < 64) {
                stack[sp] = right;
                sp += 1;
                stack[sp] = leftFirst;
                sp += 1;
            }
        }
    }
    if (bestT < 1e19) {
        return vec4f(bestN, bestT);
    }
    return vec4f(0.0, 0.0, 0.0, -1.0);
}

fn glassTransmissionToSun(p: vec3f, sunDir: vec3f, lambdaNm: f32) -> f32 {
    // First intersection from ground toward sun
    let h1 = intersectMesh(p + sunDir * 0.003, sunDir);
    if (h1.w <= 0.0) {
        return 1.0;
    }

    // Second intersection approximates glass thickness along this path
    let pInside = p + sunDir * (h1.w + 0.006);
    let h2 = intersectMesh(pInside, sunDir);
    if (h2.w <= 0.0) {
        return 0.25;
    }

    let thickness = max(h2.w, 0.0);
    // Slightly stronger absorption for short wavelengths to mimic tinted dispersion.
    let t = clamp((lambdaNm - 380.0) / 400.0, 0.0, 1.0);
    let sigma = mix(0.9, 0.45, t);
    return exp(-sigma * thickness * 6.0);
}

fn snellIorForWavelength(lambdaNm: f32, dispersion: f32) -> f32 {
    let etaD = 1.50;
    let x = (lambdaNm - 550.0) / 170.0;
    return etaD + dispersion * (-x + 0.2 * x * x);
}

fn schlick(cosTheta: f32, etaI: f32, etaT: f32) -> f32 {
    let r0 = pow((etaI - etaT) / (etaI + etaT), 2.0);
    return r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
}

fn cosineHemisphere(n: vec3f, seed: vec2f) -> vec3f {
    let r1 = fract(seed.x);
    let r2 = fract(seed.y);
    let phi = 2.0 * PI * r1;
    let r = sqrt(r2);
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(1.0 - r2, 0.0));

    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * x + b * y + n * z);
}

fn sampleConeDirection(axis: vec3f, seed: vec2f, coneCos: f32) -> vec3f {
    let r1 = fract(seed.x);
    let r2 = fract(seed.y);
    let phi = 2.0 * PI * r2;
    let cosTheta = mix(coneCos, 1.0, r1);
    let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(axis.y) < 0.999);
    let t = normalize(cross(up, axis));
    let b = cross(axis, t);
    return normalize(t * (cos(phi) * sinTheta) + b * (sin(phi) * sinTheta) + axis * cosTheta);
}

fn photonGridDims() -> vec2u {
    return vec2u(
        max(u32(photon.data.x + 0.5), 1u),
        max(u32(photon.data.y + 0.5), 1u)
    );
}

fn photonIndexFromXZ(xz: vec2f) -> i32 {
    let dims = photonGridDims();
    let span = PHOTON_MAX_XZ - PHOTON_MIN_XZ;
    let uv = (xz - PHOTON_MIN_XZ) / span;
    if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) {
        return -1;
    }
    let ix = min(u32(uv.x * f32(dims.x)), dims.x - 1u);
    let iy = min(u32(uv.y * f32(dims.y)), dims.y - 1u);
    return i32(iy * dims.x + ix);
}

fn writePhoton(xz: vec2f, energy: vec3f) {
    let idx = photonIndexFromXZ(xz);
    if (idx < 0) {
        return;
    }
    let q = vec3u(
        u32(clamp(energy.r * 4096.0, 0.0, 65535.0)),
        u32(clamp(energy.g * 4096.0, 0.0, 65535.0)),
        u32(clamp(energy.b * 4096.0, 0.0, 65535.0))
    );
    if (q.r == 0u && q.g == 0u && q.b == 0u) {
        return;
    }
    let base = u32(idx) * 3u;
    if (photon.data.z > 0.5) {
        atomicAdd(&photonB[base + 0u], q.r);
        atomicAdd(&photonB[base + 1u], q.g);
        atomicAdd(&photonB[base + 2u], q.b);
    } else {
        atomicAdd(&photonA[base + 0u], q.r);
        atomicAdd(&photonA[base + 1u], q.g);
        atomicAdd(&photonA[base + 2u], q.b);
    }
}

fn readPhoton(idx: u32) -> u32 {
    if (photon.data.z > 0.5) {
        return atomicLoad(&photonA[idx]);
    }
    return atomicLoad(&photonB[idx]);
}

fn readPhotonRgb(cellIndex: u32) -> vec3f {
    let base = cellIndex * 3u;
    return vec3f(
        f32(readPhoton(base + 0u)),
        f32(readPhoton(base + 1u)),
        f32(readPhoton(base + 2u))
    );
}

fn samplePhotonGrid(xz: vec2f) -> vec3f {
    let dims = photonGridDims();
    let span = PHOTON_MAX_XZ - PHOTON_MIN_XZ;
    let uv = (xz - PHOTON_MIN_XZ) / span;
    if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) {
        return vec3f(0.0);
    }

    let gx = clamp(uv.x * f32(dims.x) - 0.5, 0.0, f32(dims.x) - 1.0);
    let gy = clamp(uv.y * f32(dims.y) - 0.5, 0.0, f32(dims.y) - 1.0);
    let ix = u32(gx);
    let iy = u32(gy);
    let ix1 = min(ix + 1u, dims.x - 1u);
    let iy1 = min(iy + 1u, dims.y - 1u);
    let fx = fract(gx);
    let fy = fract(gy);

    let i00 = iy * dims.x + ix;
    let i10 = iy * dims.x + ix1;
    let i01 = iy1 * dims.x + ix;
    let i11 = iy1 * dims.x + ix1;

    let p00 = readPhotonRgb(i00);
    let p10 = readPhotonRgb(i10);
    let p01 = readPhotonRgb(i01);
    let p11 = readPhotonRgb(i11);

    let a = mix(p00, p10, fx);
    let b = mix(p01, p11, fx);

    return mix(a, b, fy) * photon.data.w;
}

fn tracePhotonFromLight(seed: vec3f, lambdaNm: f32) {
    let sunAxis = -sunDirection();
    let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunAxis.y) < 0.95);
    let basisX = normalize(cross(upRef, sunAxis));
    let basisY = normalize(cross(sunAxis, basisX));
    let u0 = hash13(seed + vec3f(0.13, 0.71, 1.11));
    let v0 = hash13(seed + vec3f(0.47, 0.23, 2.73));
    let span = 3.2;
    var pos = vec3f(0.0, 1.4, 0.2) - sunAxis * 5.5 + basisX * mix(-span, span, u0) + basisY * mix(-span, span, v0);
    var direction = sunAxis;

    var power = 1.0;
    let spectralColor = wavelengthToRgb(lambdaNm);
    var seenGlass = false;
    let sphereCenter = vec3f(-0.8, -0.12, 0.5);
    let sphereRadius = 0.5;

    for (var step = 0; step < 10; step += 1) {
        let meshHit = intersectMesh(pos, direction);
        let sphereHit = intersectSphere(pos, direction, sphereCenter, sphereRadius);
        let prismHit = intersectPrism(pos, direction);

        let tGround = (GROUND_Y - pos.y) / direction.y;
        let hitGround = tGround > 0.0005;
        let hitMesh = meshHit.w > 0.0;
        let hitSphere = sphereHit.w > 0.0;
        let hitPrism = prismHit.w > 0.0;

        var tClosest = 1e20;
        var hitType = -1.0;
        var nRaw = vec3f(0.0);

        if (hitSphere && sphereHit.w < tClosest) {
            tClosest = sphereHit.w;
            hitType = 1.0;
            nRaw = sphereHit.xyz;
        }
        if (hitMesh && meshHit.w < tClosest) {
            tClosest = meshHit.w;
            hitType = 1.0;
            nRaw = meshHit.xyz;
        }
        if (hitPrism && prismHit.w < tClosest) {
            tClosest = prismHit.w;
            hitType = 1.0;
            nRaw = prismHit.xyz;
        }
        if (hitGround && tGround < tClosest) {
            tClosest = tGround;
            hitType = 0.0;
            nRaw = vec3f(0.0, 1.0, 0.0);
        }

        if (hitType < -0.5) {
            break;
        }

        let hitPos = pos + direction * tClosest;
        if (hitType < 0.5) {
            if (seenGlass) {
                writePhoton(hitPos.xz, spectralColor * power);
            }
            break;
        }

        seenGlass = true;
        let n = select(-nRaw, nRaw, dot(direction, nRaw) < 0.0);
        let glassIor = snellIorForWavelength(lambdaNm, clamp(u.orbital.z, 0.0, 0.2));
        let entering = dot(direction, nRaw) < 0.0;
        let etaI = select(glassIor, 1.0, entering);
        let etaT = select(1.0, glassIor, entering);
        let eta = etaI / etaT;

        let cosI = clamp(dot(-direction, n), 0.0, 1.0);
        let sin2T = eta * eta * (1.0 - cosI * cosI);
        let cannotRefract = sin2T > 1.0;
        let fresnel = select(schlick(cosI, etaI, etaT), 1.0, cannotRefract);
        let r = hash13(seed + vec3f(f32(step) * 2.37, f32(step) * 0.91, u.render.x));

        var nextDir = reflect(direction, n);
        if (r > fresnel && !cannotRefract) {
            nextDir = refract(direction, n, eta);
            power *= 0.985;
        } else {
            power *= 0.96;
        }

        if (power < 0.01) {
            break;
        }

        pos = hitPos + nextDir * 0.002;
        direction = nextDir;
    }
}

fn trace(ro: vec3f, rd: vec3f, pixelJitter: vec2f, lambdaNm: f32) -> vec3f {
    var origin = ro;
    var dir = rd;
    var throughput = vec3f(1.0);
    var radiance = vec3f(0.0);

    let maxBounces = i32(clamp(u.orbital.x, 1.0, 32.0));
    let roughness = clamp(u.orbital.w, 0.0, 0.2);
    let dispersion = clamp(u.orbital.z, 0.0, 0.2);
    let spectralWeight = wavelengthToRgb(lambdaNm);

    // Light-traced pass: emit spectral photons from directional sun into a floor grid.
    let photonRaysPerSample = 4;
    for (var pi = 0; pi < photonRaysPerSample; pi += 1) {
        let pf = f32(pi);
        let photonSeed = vec3f(
            pixelJitter.x + pf * 0.37,
            pixelJitter.y + pf * 0.61,
            u.render.x + pf * 1.17
        );
        let photonLambda = 380.0 + 400.0 * hash13(photonSeed + vec3f(9.3, 4.1, 2.7));
        tracePhotonFromLight(photonSeed, photonLambda);
    }

    var etaCurrent = 1.0;

    for (var bounce = 0; bounce < maxBounces; bounce += 1) {
        let meshHit = intersectMesh(origin, dir);
        
        // Add glass sphere at position on the ground
        let sphereCenter = vec3f(-0.8, -0.12, 0.5);
        let sphereRadius = 0.5;
        let sphereHit = intersectSphere(origin, dir, sphereCenter, sphereRadius);
        let prismHit = intersectPrism(origin, dir);
        
        let tGround = (-0.62 - origin.y) / dir.y;
        let hitGround = (tGround > 0.0005);
        let hitMesh = (meshHit.w > 0.0);
        let hitSphere = (sphereHit.w > 0.0);
        let hitPrism = (prismHit.w > 0.0);

        var hitMat = -1.0;
        var hitPos = vec3f(0.0);
        var nRaw = vec3f(0.0);
        
        // Find closest hit
        var tClosest = 1e20;
        
        if (hitSphere && sphereHit.w < tClosest) {
            tClosest = sphereHit.w;
            hitMat = 1.0; // Glass
            hitPos = origin + dir * sphereHit.w;
            nRaw = sphereHit.xyz;
        }
        
        if (hitMesh && meshHit.w < tClosest) {
            tClosest = meshHit.w;
            hitMat = 1.0; // Glass
            hitPos = origin + dir * meshHit.w;
            nRaw = meshHit.xyz;
        }
        
        if (hitPrism && prismHit.w < tClosest) {
            tClosest = prismHit.w;
            hitMat = 1.0; // Glass prism
            hitPos = origin + dir * prismHit.w;
            nRaw = prismHit.xyz;
        }
        
        if (hitGround && tGround < tClosest) {
            tClosest = tGround;
            hitMat = 0.0; // Ground
            hitPos = origin + dir * tGround;
            nRaw = vec3f(0.0, 1.0, 0.0);
        }
        
        if (hitMat < -0.5) {
            // Hit environment/sky
            radiance += throughput * sampleEnvironment(dir) * u.pan.x * spectralWeight;
            break;
        }

        let n = select(-nRaw, nRaw, dot(dir, nRaw) < 0.0);

        if (hitMat < 0.5) {
            // Ground material - DIFFUSE surface
            // This is where we do Next Event Estimation (shadow ray to light)
            let base = vec3f(0.08, 0.08, 0.09);
            let albedo = vec3f(0.85, 0.85, 0.85);  // More reflective for brighter caustics
            
            // Ambient from sky
            let ambient = vec3f(0.3, 0.35, 0.4) * 0.2;
            
            // Next Event Estimation: sample directional sun directly
            let sunDir = sunDirection();
            let sunIntensity = max(u.tdse.z, 0.0);
            
            var lightContrib = vec3f(0.0);
            
            if (sunIntensity > 0.0) {
                let L = sunDir;
                let ndotl = max(dot(n, L), 0.0);

                // Shadow test along sun direction (infinite directional light)
                let shadowBiasN = select(-n, n, dot(n, L) >= 0.0);
                let shadowOrigin = hitPos + shadowBiasN * 0.01;
                let meshShadow = intersectMesh(shadowOrigin, L);
                let sphereCenter = vec3f(-0.8, -0.12, 0.5);
                let sphereRadius = 0.5;
                let sphereShadow = intersectSphere(shadowOrigin, L, sphereCenter, sphereRadius);
                let prismShadow = intersectPrism(shadowOrigin, L);

                // If glass is in the way, direct sun is blocked.
                let blocked = (meshShadow.w > 0.0) || (sphereShadow.w > 0.0) || (prismShadow.w > 0.0);

                if (!blocked) {
                    let lightColor = vec3f(1.0, 0.98, 0.95);
                    lightContrib = lightColor * sunIntensity * ndotl;
                }
            }

            // Photon-map caustics from light-traced paths through glass.
            let photonColor = samplePhotonGrid(hitPos.xz);
            let causticColor = photonColor * max(u.tdse.z, 0.0);
            
            radiance += throughput * (base + albedo * (ambient + lightContrib + causticColor)) * spectralWeight;
            
            // CONTINUE BOUNCING from ground (diffuse reflection)
            // This allows paths like: Camera -> Ground -> Glass -> Light (caustics!)
            let diffuseDir = cosineHemisphere(n, vec2f(
                hash13(vec3f(hitPos.xy, f32(bounce) * 3.14)),
                hash13(vec3f(hitPos.xz, f32(bounce) * 2.71))
            ));
            
            throughput *= albedo;
            origin = hitPos + diffuseDir * 0.002;
            dir = diffuseDir;
            
            // Continue to next bounce instead of breaking!
        } else {
            // Glass material - specular reflection/refraction
            let glassIor = snellIorForWavelength(lambdaNm, dispersion);
            let entering = dot(dir, nRaw) < 0.0;
            let etaI = select(glassIor, 1.0, entering);
            let etaT = select(1.0, glassIor, entering);

            let eta = etaI / etaT;
            let cosI = clamp(dot(-dir, n), 0.0, 1.0);
            let sin2T = eta * eta * (1.0 - cosI * cosI);
            let cannotRefract = sin2T > 1.0;

            let fresnel = select(schlick(cosI, etaI, etaT), 1.0, cannotRefract);
            let r = hash13(vec3f(pixelJitter, f32(bounce) * 17.0 + u.render.x));

            var nextDir = reflect(dir, n);
            if (r > fresnel && !cannotRefract) {
                nextDir = refract(dir, n, eta);
                etaCurrent = etaT;
            }

            if (roughness > 0.0) {
                let jitter = cosineHemisphere(nextDir, vec2f(
                    hash13(vec3f(hitPos.xy, f32(bounce) * 2.71 + 11.0)),
                    hash13(vec3f(hitPos.zy, f32(bounce) * 4.31 + 29.0))
                ));
                nextDir = normalize(mix(nextDir, jitter, roughness));
            }

            throughput *= mix(vec3f(0.98), vec3f(1.0), vec3f(fresnel));
            origin = hitPos + nextDir * 0.002;
            dir = nextDir;
        }

        // Russian Roulette path termination (but keep going if throughput is high)
        let maxThroughput = max(throughput.r, max(throughput.g, throughput.b));
        if (bounce > 3 && maxThroughput < 0.1) {
            let rrProb = maxThroughput;
            let rrRand = hash13(vec3f(pixelJitter, f32(bounce) * 23.0 + u.render.x));
            if (rrRand > rrProb) {
                break;
            }
            throughput /= rrProb;
        }

        if (max(throughput.r, max(throughput.g, throughput.b)) < 0.01) {
            break;
        }

        _ = etaCurrent;
    }

    return radiance;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let aspect = max(u.render.y, 0.0001);
    let time = u.render.x;
    let spp = i32(clamp(u.orbital.y, 1.0, 256.0));

    let camPos = u.tuning.xyz;
    let exposure = max(u.tuning.w, 0.01);

    let yaw = u.render.z;
    let pitch = u.render.w;
    let eye = camPos;
    let forward = normalize(vec3f(
        cos(pitch) * sin(yaw),
        sin(pitch),
        cos(pitch) * cos(yaw)));
    let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
    let up = cross(right, forward);

    let uv = vec2f(input.uv.x * aspect, input.uv.y);

    var color = vec3f(0.0);
    for (var i = 0; i < spp; i += 1) {
        let fi = f32(i);
        let jitter = vec2f(
            hash12(uv + vec2f(fi, time * 0.17)),
            hash12(uv.yx + vec2f(time * 0.23, fi * 1.7))
        ) - 0.5;

        let pixelUv = uv + jitter * (0.0018 + 0.0007 * hash12(vec2f(fi, time)));
        let rd = normalize(forward + pixelUv.x * right + pixelUv.y * up);

        let lambda = 380.0 + 400.0 * hash12(vec2f(fi * 13.1 + uv.x * 91.0, uv.y * 73.0 + time * 0.31));
        color += trace(eye, rd, jitter + vec2f(fi * 0.03), lambda);
    }

    color /= f32(spp);
    color *= exposure;

    let w = max(i32(accum.data.x + 0.5), 1);
    let h = max(i32(accum.data.y + 0.5), 1);
    let xi = clamp(i32(input.position.x), 0, w - 1);
    let yi = clamp(i32(input.position.y), 0, h - 1);
    let idx = u32(yi * w + xi);
    let useA = accum.data.z > 0.5;
    let reset = accum.data.w > 0.5;

    let prev = select(accumB[idx], accumA[idx], useA);
    let prevSum = select(prev.rgb, vec3f(0.0), reset);
    let prevCount = select(prev.w, 0.0, reset);
    let sum = prevSum + color;
    let count = prevCount + 1.0;
    let outVal = vec4f(sum, count);
    if (useA) {
        accumB[idx] = outVal;
    } else {
        accumA[idx] = outVal;
    }

    let avg = sum / max(count, 1.0);
    let mapped = pow(acesToneMap(avg), vec3f(1.0 / 2.2));
    return vec4f(mapped, 1.0);
}
