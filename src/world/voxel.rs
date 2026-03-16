use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

#[derive(Clone, Debug, PartialEq)]
pub enum ShapeType {
    Cube,
    Wedge,
    Sphere,
}

#[derive(Component, Debug)]
pub struct Voxel {
    pub position: Vec3,
    pub predicted_position: Vec3,
    pub velocity: Vec3,
    pub shape: ShapeType,
    pub sphere_radius: f32,
    pub inv_mass: f32,
    pub friction: f32,
    pub restitution: f32,
    pub rotation: Quat,
    pub angular_velocity: Vec3,
    pub contact_normal_accum: Vec3,
    pub contact_count: u32,
}

impl Voxel {
    pub fn new(x: f32, y: f32, z: f32, shape: ShapeType, is_static: bool) -> Self {
        // LEARNING: Smaller inv_mass = Heavier object.
        // Construction wedges should be very heavy to remain stable platforms.
        let dynamic_inv_mass = match shape {
            ShapeType::Cube => 1.0,
            ShapeType::Wedge => 0.002, // Wedge is now 500x heavier than a cube
            ShapeType::Sphere => 1.0,
        };

        Self {
            position: Vec3::new(x, y, z),
            predicted_position: Vec3::new(x, y, z),
            velocity: Vec3::ZERO,
            shape,
            sphere_radius: 0.5,
            inv_mass: if is_static { 0.0 } else { dynamic_inv_mass },
            friction: 0.8, // High friction for construction materials
            restitution: 0.0,
            rotation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
            contact_normal_accum: Vec3::ZERO,
            contact_count: 0,
        }
    }

    pub fn new_sphere(x: f32, y: f32, z: f32, radius: f32, is_static: bool) -> Self {
        let mut voxel = Self::new(x, y, z, ShapeType::Sphere, is_static);
        voxel.sphere_radius = radius.max(0.2);
        voxel
    }
}
