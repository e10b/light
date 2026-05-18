use crate::{editor::panels::CameraProjectionKind, scene::SceneKind};

pub fn scene_camera(
    scene_kind: SceneKind,
    center: glam::Vec3,
    size: glam::Vec3,
) -> (glam::Vec3, glam::Vec3) {
    if scene_kind == SceneKind::Wine {
        let distance = size.max_element().max(12.0) * 1.35;
        return (
            center + glam::Vec3::new(0.0, size.y * 0.2, distance),
            center,
        );
    }
    scene_kind.default_camera(center)
}

pub fn camera_projection_matrix(
    mode: CameraProjectionKind,
    fov_radians: f32,
    ortho_height: f32,
    near: f32,
    far: f32,
    width: u32,
    height: u32,
) -> glam::Mat4 {
    let aspect = width.max(1) as f32 / height.max(1) as f32;
    match mode {
        CameraProjectionKind::Perspective => {
            glam::Mat4::perspective_rh(fov_radians, aspect, near, far)
        }
        CameraProjectionKind::Orthographic => {
            let half_h = (ortho_height * 0.5).max(0.001);
            let half_w = (half_h * aspect).max(0.001);
            glam::Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, near, far)
        }
    }
}

pub fn gizmo_projection_matrix(
    mode: CameraProjectionKind,
    fov_radians: f32,
    ortho_height: f32,
    near: f32,
    far: f32,
    width: u32,
    height: u32,
) -> glam::Mat4 {
    let aspect = width.max(1) as f32 / height.max(1) as f32;
    match mode {
        CameraProjectionKind::Perspective => {
            glam::Mat4::perspective_rh_gl(fov_radians, aspect, near, far)
        }
        CameraProjectionKind::Orthographic => {
            let half_h = (ortho_height * 0.5).max(0.001);
            let half_w = (half_h * aspect).max(0.001);
            glam::Mat4::orthographic_rh_gl(-half_w, half_w, -half_h, half_h, near, far)
        }
    }
}

pub fn wine_spotlight_position(
    center: glam::Vec3,
    azimuth_deg: f32,
    elevation_deg: f32,
    distance: f32,
) -> glam::Vec3 {
    let azimuth = azimuth_deg.to_radians();
    let elevation = elevation_deg.to_radians();
    let dir_from_target = glam::Vec3::new(
        azimuth.cos() * elevation.cos(),
        elevation.sin(),
        azimuth.sin() * elevation.cos(),
    )
    .normalize_or_zero();
    center + dir_from_target * distance.max(1.0)
}

pub fn world_ray_from_cursor(
    cursor: [f32; 2],
    viewport: [f32; 2],
    view_inv: glam::Mat4,
    proj_inv: glam::Mat4,
) -> (glam::Vec3, glam::Vec3) {
    let ndc_x = (cursor[0] / viewport[0]) * 2.0 - 1.0;
    let ndc_y = (1.0 - cursor[1] / viewport[1]) * 2.0 - 1.0;
    let cam_far = proj_inv * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let far_pos = cam_far.truncate() / cam_far.w.max(1e-6);
    let origin = (view_inv * glam::Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate();
    let far_world = (view_inv * glam::Vec4::new(far_pos.x, far_pos.y, far_pos.z, 1.0)).truncate();
    (origin, (far_world - origin).normalize_or_zero())
}

pub fn world_to_screen(
    point: glam::Vec3,
    view: glam::Mat4,
    proj: glam::Mat4,
    viewport: [f32; 2],
) -> Option<[f32; 2]> {
    let clip = proj * view * glam::Vec4::new(point.x, point.y, point.z, 1.0);
    if clip.w.abs() < 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z < -1.0 || ndc.z > 1.0 {
        return None;
    }
    let x = (ndc.x * 0.5 + 0.5) * viewport[0];
    let y = (1.0 - (ndc.y * 0.5 + 0.5)) * viewport[1];
    Some([x, y])
}

pub fn intersect_sphere(
    origin: glam::Vec3,
    dir: glam::Vec3,
    center: glam::Vec3,
    radius: f32,
) -> Option<f32> {
    let oc = origin - center;
    let a = dir.dot(dir);
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - a * c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t1 = (-b - sq) / a;
    let t2 = (-b + sq) / a;
    if t1 > 0.001 {
        Some(t1)
    } else if t2 > 0.001 {
        Some(t2)
    } else {
        None
    }
}

pub fn intersect_cube(
    origin: glam::Vec3,
    dir: glam::Vec3,
    center: glam::Vec3,
    half_extent: glam::Vec3,
) -> Option<f32> {
    let min = center - half_extent;
    let max = center + half_extent;
    let inv = glam::Vec3::new(
        if dir.x.abs() > 1e-6 {
            1.0 / dir.x
        } else {
            f32::INFINITY
        },
        if dir.y.abs() > 1e-6 {
            1.0 / dir.y
        } else {
            f32::INFINITY
        },
        if dir.z.abs() > 1e-6 {
            1.0 / dir.z
        } else {
            f32::INFINITY
        },
    );
    let t0 = (min - origin) * inv;
    let t1 = (max - origin) * inv;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let near = tmin.x.max(tmin.y).max(tmin.z);
    let far = tmax.x.min(tmax.y).min(tmax.z);
    if far < 0.0 || near > far {
        return None;
    }
    if near > 0.001 {
        Some(near)
    } else if far > 0.001 {
        Some(far)
    } else {
        None
    }
}
