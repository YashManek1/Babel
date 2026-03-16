use crate::world::spatial_grid::SpatialGrid;
use crate::world::voxel::{ShapeType, Voxel};
use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use std::collections::{HashMap, HashSet};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
const SOLVER_ITERATIONS: usize = 10;
const MAX_CORRECTION_PER_ITER: f32 = 0.22;
const MAX_ACCUM_CORRECTION_PER_ITER: f32 = 0.65;
const MAX_DISPLACEMENT_PER_FRAME: f32 = 1.25;
const MAX_VELOCITY: f32 = 25.0;
const FLAT_CONTACT_Y_THRESHOLD: f32 = 0.98;
const UPRIGHT_SETTLE_RATE: f32 = 0.2;
const LINEAR_SLEEP_SPEED: f32 = 0.08;
const POSITION_VELOCITY_DAMPING: f32 = 0.94;
const STATIC_FRICTION_SPEED: f32 = 0.12;

#[derive(Clone, Copy)]
struct Contact {
    normal: Vec3,
    penetration: f32,
    #[allow(dead_code)]
    contact_point: Vec3,
}

#[derive(Clone)]
pub struct BodySnapshot {
    pub predicted_position: Vec3,
    pub inv_mass: f32,
    pub shape: ShapeType,
    pub sphere_radius: f32,
}

#[derive(Resource)]
pub struct PhysicsSettings {
    pub global_floor_y: Option<f32>,
}
impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            global_floor_y: Some(-0.5),
        }
    }
}

#[derive(Resource, Default)]
pub struct SolverBuffers {
    pub snapshots: HashMap<Entity, BodySnapshot>,
    pub pos_corrections: HashMap<Entity, Vec3>,
    pub normal_accum: HashMap<Entity, Vec3>,
    pub normal_count: HashMap<Entity, u32>,
    pub processed_pairs: HashSet<(Entity, Entity)>,
}

fn make_pair(a: Entity, b: Entity) -> (Entity, Entity) {
    if a.index() < b.index() {
        (a, b)
    } else {
        (b, a)
    }
}

fn compute_cube_wedge_contact(self_pos: Vec3, wedge_pos: Vec3) -> Option<Contact> {
    let local = self_pos - wedge_pos;
    let intersect = Vec3::ONE - local.abs();
    if intersect.x <= 0.0 || intersect.y <= 0.0 || intersect.z <= 0.0 {
        return None;
    }

    let slope_normal = Vec3::new(0.0, 0.70710678, 0.70710678);
    let dist = local.dot(slope_normal);
    let pen_slope = 0.70710678 - dist;
    if pen_slope <= 0.0 {
        return None;
    }

    let mut min_pen = pen_slope;
    let mut normal = slope_normal;

    if intersect.x < min_pen && local.y < 0.25 {
        min_pen = intersect.x;
        normal = Vec3::new(if local.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0);
    }
    if intersect.z < min_pen && local.z < -0.25 {
        min_pen = intersect.z;
        normal = Vec3::new(0.0, 0.0, -1.0);
    }
    if intersect.y < min_pen && local.y < -0.25 {
        min_pen = intersect.y;
        normal = Vec3::new(0.0, -1.0, 0.0);
    }

    let slope_z = local.z.clamp(-0.5, 0.5);
    let contact_point = wedge_pos + Vec3::new(local.x.clamp(-0.5, 0.5), -slope_z, slope_z);
    Some(Contact {
        normal,
        penetration: min_pen,
        contact_point,
    })
}

fn compute_aabb_aabb_contact(
    self_pos: Vec3,
    self_half: Vec3,
    other_pos: Vec3,
    other_half: Vec3,
) -> Option<Contact> {
    let delta = self_pos - other_pos;
    let overlap = self_half + other_half - delta.abs();
    if overlap.x <= 0.0 || overlap.y <= 0.0 || overlap.z <= 0.0 {
        return None;
    }
    let mut normal = Vec3::new(if delta.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0);
    let mut penetration = overlap.x;
    if overlap.y < penetration {
        penetration = overlap.y;
        normal = Vec3::new(0.0, if delta.y >= 0.0 { 1.0 } else { -1.0 }, 0.0);
    }
    if overlap.z < penetration {
        penetration = overlap.z;
        normal = Vec3::new(0.0, 0.0, if delta.z >= 0.0 { 1.0 } else { -1.0 });
    }
    Some(Contact {
        normal,
        penetration,
        contact_point: self_pos - normal * penetration * 0.5,
    })
}

fn compute_sphere_sphere_contact(
    self_pos: Vec3,
    self_radius: f32,
    other_pos: Vec3,
    other_radius: f32,
) -> Option<Contact> {
    let delta = self_pos - other_pos;
    let distance = delta.length();
    let radius_sum = self_radius + other_radius;
    if distance >= radius_sum {
        return None;
    }
    let normal = if distance > 1e-6 {
        delta / distance
    } else {
        Vec3::Y
    };
    Some(Contact {
        normal,
        penetration: radius_sum - distance,
        contact_point: self_pos - normal * self_radius,
    })
}

fn compute_sphere_aabb_contact(
    sphere_pos: Vec3,
    sphere_radius: f32,
    box_pos: Vec3,
    box_half: Vec3,
) -> Option<Contact> {
    let local = sphere_pos - box_pos;
    let closest = Vec3::new(
        local.x.clamp(-box_half.x, box_half.x),
        local.y.clamp(-box_half.y, box_half.y),
        local.z.clamp(-box_half.z, box_half.z),
    );
    let delta = local - closest;
    let dist_sq = delta.length_squared();
    if dist_sq >= sphere_radius * sphere_radius {
        return None;
    }
    let distance = dist_sq.sqrt();
    let normal = if distance > 1e-6 {
        delta / distance
    } else {
        Vec3::Y
    };
    Some(Contact {
        normal,
        penetration: sphere_radius - distance,
        contact_point: box_pos + closest,
    })
}

fn compute_contact(
    self_pos: Vec3,
    self_shape: &ShapeType,
    self_radius: f32,
    other_pos: Vec3,
    other_shape: &ShapeType,
    other_radius: f32,
) -> Option<Contact> {
    match (self_shape, other_shape) {
        (ShapeType::Cube, ShapeType::Cube) => {
            compute_aabb_aabb_contact(self_pos, Vec3::splat(0.5), other_pos, Vec3::splat(0.5))
        }
        (ShapeType::Cube, ShapeType::Wedge) => compute_cube_wedge_contact(self_pos, other_pos),
        (ShapeType::Wedge, ShapeType::Cube) => {
            compute_cube_wedge_contact(other_pos, self_pos).map(|mut c| {
                c.normal = -c.normal;
                c
            })
        }
        (ShapeType::Wedge, ShapeType::Wedge) => {
            compute_aabb_aabb_contact(self_pos, Vec3::splat(0.5), other_pos, Vec3::splat(0.5))
        }
        (ShapeType::Sphere, ShapeType::Sphere) => {
            compute_sphere_sphere_contact(self_pos, self_radius, other_pos, other_radius)
        }
        (ShapeType::Sphere, ShapeType::Cube) | (ShapeType::Sphere, ShapeType::Wedge) => {
            compute_sphere_aabb_contact(self_pos, self_radius, other_pos, Vec3::splat(0.5))
        }
        (ShapeType::Cube, ShapeType::Sphere) | (ShapeType::Wedge, ShapeType::Sphere) => {
            compute_sphere_aabb_contact(other_pos, other_radius, self_pos, Vec3::splat(0.5)).map(
                |mut c| {
                    c.normal = -c.normal;
                    c
                },
            )
        }
    }
}

pub fn integrate_system(mut query: Query<&mut Voxel>) {
    for mut voxel in query.iter_mut() {
        voxel.contact_normal_accum = Vec3::ZERO;
        voxel.contact_count = 0;
        if voxel.inv_mass == 0.0 {
            continue;
        }
        voxel.velocity += GRAVITY * DT;
        voxel.velocity = voxel.velocity.clamp_length_max(MAX_VELOCITY);
        voxel.predicted_position = voxel.position + (voxel.velocity * DT);
    }
}

pub fn solve_constraints_system(
    mut query: Query<(Entity, &mut Voxel)>,
    grid: Res<SpatialGrid>,
    settings: Res<PhysicsSettings>,
    mut buffers: ResMut<SolverBuffers>,
) {
    for _ in 0..SOLVER_ITERATIONS {
        buffers.snapshots.clear();
        buffers.pos_corrections.clear();
        buffers.normal_accum.clear();
        buffers.normal_count.clear();
        buffers.processed_pairs.clear();

        for (entity, voxel) in query.iter() {
            buffers.snapshots.insert(
                entity,
                BodySnapshot {
                    predicted_position: voxel.predicted_position,
                    inv_mass: voxel.inv_mass,
                    shape: voxel.shape.clone(),
                    sphere_radius: voxel.sphere_radius,
                },
            );
        }

        for (entity, voxel) in query.iter() {
            if let Some(floor_y) = settings.global_floor_y {
                let mut min_local_y = 0.0;
                let extents: Vec<Vec3> = if voxel.shape == ShapeType::Sphere {
                    vec![Vec3::new(0.0, -voxel.sphere_radius, 0.0)]
                } else {
                    vec![
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(-0.5, 0.5, 0.5),
                        Vec3::new(0.5, -0.5, 0.5),
                        Vec3::new(-0.5, -0.5, 0.5),
                        Vec3::new(0.5, 0.5, -0.5),
                        Vec3::new(-0.5, 0.5, -0.5),
                        Vec3::new(0.5, -0.5, -0.5),
                        Vec3::new(-0.5, -0.5, -0.5),
                    ]
                };
                for v in extents {
                    let r = voxel.rotation * v;
                    if r.y < min_local_y {
                        min_local_y = r.y;
                    }
                }
                let lowest = voxel.predicted_position.y + min_local_y;
                if lowest < floor_y {
                    let pen = floor_y - lowest;
                    *buffers.pos_corrections.entry(entity).or_insert(Vec3::ZERO) +=
                        Vec3::new(0.0, pen, 0.0);
                    *buffers.normal_accum.entry(entity).or_insert(Vec3::ZERO) += Vec3::Y;
                    *buffers.normal_count.entry(entity).or_insert(0) += 1;
                }
            }

            for (other_e, _other_s, grid_pos) in grid.get_neighbors(voxel.predicted_position) {
                if entity == other_e {
                    continue;
                }
                let pair = make_pair(entity, other_e);
                if !buffers.processed_pairs.insert(pair) {
                    continue;
                }

                let Some(self_snap) = buffers.snapshots.get(&entity).cloned() else {
                    continue;
                };
                let Some(other_snap) = buffers.snapshots.get(&other_e).cloned() else {
                    continue;
                };

                let other_pos = if other_snap.inv_mass == 0.0 {
                    Vec3::new(grid_pos[0] as f32, grid_pos[1] as f32, grid_pos[2] as f32)
                } else {
                    other_snap.predicted_position
                };

                let Some(c) = compute_contact(
                    self_snap.predicted_position,
                    &self_snap.shape,
                    self_snap.sphere_radius,
                    other_pos,
                    &other_snap.shape,
                    other_snap.sphere_radius,
                ) else {
                    continue;
                };

                let inv_mass_sum = self_snap.inv_mass + other_snap.inv_mass;
                if inv_mass_sum <= 0.0 {
                    continue;
                }

                let base_corr =
                    (c.normal * c.penetration).clamp_length_max(MAX_CORRECTION_PER_ITER);

                let self_is_wedge = self_snap.shape == ShapeType::Wedge;
                let other_is_wedge = other_snap.shape == ShapeType::Wedge;
                let mut self_share = self_snap.inv_mass / inv_mass_sum;
                let mut other_share = other_snap.inv_mass / inv_mass_sum;
                if self_is_wedge ^ other_is_wedge {
                    if self_is_wedge {
                        self_share *= 0.20;
                        other_share = 1.0 - self_share;
                    } else {
                        other_share *= 0.20;
                        self_share = 1.0 - other_share;
                    }
                }

                if self_snap.inv_mass > 0.0 {
                    let delta = base_corr * self_share;
                    *buffers.pos_corrections.entry(entity).or_insert(Vec3::ZERO) =
                        (*buffers.pos_corrections.entry(entity).or_insert(Vec3::ZERO) + delta)
                            .clamp_length_max(MAX_ACCUM_CORRECTION_PER_ITER);
                    *buffers.normal_accum.entry(entity).or_insert(Vec3::ZERO) += c.normal;
                    *buffers.normal_count.entry(entity).or_insert(0) += 1;
                }
                if other_snap.inv_mass > 0.0 {
                    let delta = base_corr * -other_share;
                    *buffers.pos_corrections.entry(other_e).or_insert(Vec3::ZERO) =
                        (*buffers.pos_corrections.entry(other_e).or_insert(Vec3::ZERO) + delta)
                            .clamp_length_max(MAX_ACCUM_CORRECTION_PER_ITER);
                    *buffers.normal_accum.entry(other_e).or_insert(Vec3::ZERO) -= c.normal;
                    *buffers.normal_count.entry(other_e).or_insert(0) += 1;
                }
            }
        }

        for (entity, corr) in buffers.pos_corrections.iter() {
            if let Ok((_, mut v)) = query.get_mut(*entity) {
                v.predicted_position += *corr;
            }
        }
        for (entity, norm) in buffers.normal_accum.iter() {
            if let Ok((_, mut v)) = query.get_mut(*entity) {
                v.contact_normal_accum += *norm;
            }
        }
        for (entity, count) in buffers.normal_count.iter() {
            if let Ok((_, mut v)) = query.get_mut(*entity) {
                v.contact_count += *count;
            }
        }
    }
}

pub fn update_velocities_system(mut query: Query<&mut Voxel>) {
    for mut voxel in query.iter_mut() {
        if voxel.inv_mass == 0.0 {
            continue;
        }
        let frame_delta = (voxel.predicted_position - voxel.position)
            .clamp_length_max(MAX_DISPLACEMENT_PER_FRAME);
        voxel.predicted_position = voxel.position + frame_delta;

        let mut derived_velocity = (voxel.predicted_position - voxel.position) / DT;
        if voxel.contact_count > 0 {
            derived_velocity *= POSITION_VELOCITY_DAMPING;
        }
        voxel.velocity = derived_velocity.clamp_length_max(MAX_VELOCITY);

        if voxel.contact_count > 0 {
            let avg_normal =
                (voxel.contact_normal_accum / voxel.contact_count as f32).normalize_or_zero();
            if avg_normal != Vec3::ZERO {
                let axis = Vec3::Y.cross(avg_normal);
                let angle = Vec3::Y.angle_between(avg_normal);
                let target_rot = if angle > 1e-4 {
                    Quat::from_axis_angle(axis.normalize(), angle)
                } else {
                    Quat::IDENTITY
                };
                voxel.rotation = voxel.rotation.slerp(target_rot, 0.25);
                voxel.angular_velocity = Vec3::ZERO;

                let v_n_mag = voxel.velocity.dot(avg_normal);
                let v_n = avg_normal * v_n_mag;
                let v_t = voxel.velocity - v_n;
                let resolved_v_n = if v_n_mag < 0.0 {
                    v_n * voxel.restitution
                } else {
                    v_n
                };

                let v_t_len = v_t.length();
                if v_t_len > 1e-5 {
                    let normal_support = avg_normal.y.max(0.0);
                    let friction_drop = 9.81 * DT * voxel.friction * normal_support;
                    if avg_normal.y >= FLAT_CONTACT_Y_THRESHOLD && v_t_len <= STATIC_FRICTION_SPEED
                    {
                        voxel.velocity = resolved_v_n;
                    } else {
                        let new_vt_len = (v_t_len - friction_drop).max(0.0);
                        voxel.velocity = resolved_v_n + v_t * (new_vt_len / v_t_len);
                    }
                } else {
                    voxel.velocity = resolved_v_n;
                }

                if avg_normal.y >= FLAT_CONTACT_Y_THRESHOLD {
                    voxel.rotation = voxel.rotation.slerp(Quat::IDENTITY, UPRIGHT_SETTLE_RATE);
                    if voxel.velocity.y.abs() < 0.2 {
                        voxel.velocity.y = 0.0;
                    }
                    if voxel.velocity.length() < LINEAR_SLEEP_SPEED {
                        voxel.velocity = Vec3::ZERO;
                    }
                }
            }
        } else {
            voxel.rotation = voxel.rotation.slerp(Quat::IDENTITY, 0.05);
        }
        voxel.position = voxel.predicted_position;
    }
}
