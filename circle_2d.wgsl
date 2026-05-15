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

const PI: f32 = 3.14159265359;

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

fn envSky(d: vec3f) -> vec3f {
    // Simple blue sky gradient - darker at horizon, lighter at zenith
    let t = clamp(0.5 * (d.y + 1.0), 0.0, 1.0);
    let horizon = vec3f(0.4, 0.6, 0.9);  // Light blue at horizon
    let zenith = vec3f(0.2, 0.4, 0.8);   // Deeper blue at top
    return mix(horizon, zenith, pow(t, 0.8));
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
    // Point the spotlight toward the center of the decanters
    let spotPos = spotlightPosition();
    let targetPos = vec3f(0.0, 0.2, 0.0); // Center of glass objects
    return normalize(targetPos - spotPos);
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
    
    // Shadow test - check if light ray is blocked by glass
    let shadowRayOrigin = p + L * 0.01;
    let meshShadow = intersectMesh(shadowRayOrigin, L);
    
    // Also check sphere
    let sphereCenter = vec3f(-0.8, -0.12, 0.5);
    let sphereRadius = 0.5;
    let sphereShadow = intersectSphere(shadowRayOrigin, L, sphereCenter, sphereRadius);
    
    var shadowFactor = 1.0;
    
    // If blocked by glass, create shadow
    if ((meshShadow.w > 0.0 && meshShadow.w < (dist - 0.01)) ||
        (sphereShadow.w > 0.0 && sphereShadow.w < (dist - 0.01))) {
        shadowFactor = 0.15; // Soft shadow
    }
    
    // Warm white light color
    let lightColor = vec3f(1.0, 0.98, 0.95);
    
    return lightColor * intensity * spotAttenuation * distAttenuation * ndotl * shadowFactor;
}

fn sunLamp(d: vec3f) -> vec3f {
    // Disabled - using spotlight instead
    return vec3f(0.0);
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

fn trace(ro: vec3f, rd: vec3f, pixelJitter: vec2f, lambdaNm: f32) -> vec3f {
    var origin = ro;
    var dir = rd;
    var throughput = vec3f(1.0);
    var radiance = vec3f(0.0);

    let maxBounces = i32(clamp(u.orbital.x, 1.0, 32.0));
    let roughness = clamp(u.orbital.w, 0.0, 0.2);
    let dispersion = clamp(u.orbital.z, 0.0, 0.2);
    let spectralWeight = wavelengthToRgb(lambdaNm);

    var etaCurrent = 1.0;

    for (var bounce = 0; bounce < maxBounces; bounce += 1) {
        let meshHit = intersectMesh(origin, dir);
        
        // Add glass sphere at position (-0.8, -0.12, 0.5) with radius 0.5
        let sphereCenter = vec3f(-0.8, -0.12, 0.5);
        let sphereRadius = 0.5;
        let sphereHit = intersectSphere(origin, dir, sphereCenter, sphereRadius);
        
        let tGround = (-0.62 - origin.y) / dir.y;
        let hitGround = (tGround > 0.0005);
        let hitMesh = (meshHit.w > 0.0);
        let hitSphere = (sphereHit.w > 0.0);

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
        
        if (hitGround && tGround < tClosest) {
            tClosest = tGround;
            hitMat = 0.0; // Ground
            hitPos = origin + dir * tGround;
            nRaw = vec3f(0.0, 1.0, 0.0);
        }
        
        if (hitMat < -0.5) {
            radiance += throughput * sampleEnvironment(dir) * u.pan.x * spectralWeight;
            break;
        }

        let n = select(-nRaw, nRaw, dot(dir, nRaw) < 0.0);

        if (hitMat < 0.5) {
            // Ground material with basic lighting
            let base = vec3f(0.08, 0.08, 0.09);
            let albedo = vec3f(0.7, 0.72, 0.74);
            
            // Direct spotlight contribution (with shadow)
            let spotContrib = spotLight(hitPos, n);
            
            // Ambient from sky
            let ambient = vec3f(0.3, 0.35, 0.4) * 0.2;
            
            radiance += throughput * (base + albedo * (ambient + spotContrib)) * spectralWeight;
            break;
        }

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
