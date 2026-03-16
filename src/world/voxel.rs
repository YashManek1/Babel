// =============================================================================
// src/world/voxel.rs  —  Operation Babel: Voxel Component Definition
// =============================================================================
//
// LEARNING TOPIC: Bevy ECS Components
// ------------------------------------
// In Bevy's Entity Component System, a "Component" is pure data — no logic.
// The Voxel struct below is attached to an Entity (a unique ID number).
// Systems then query for all entities that have a Voxel component and
// process them in tight, cache-friendly loops.
//
// This is the opposite of Object-Oriented Programming where data and logic
// live together. ECS separates them, giving massive performance gains because
// all Voxel data is stored in contiguous memory ("archetypes"), not scattered
// across the heap as individual objects.
// =============================================================================

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

// =============================================================================
// LEARNING TOPIC: Enum-Driven Polymorphism (Zero-Cost)
// =====================================================
// Instead of runtime polymorphism (trait objects / vtables), we use an enum.
// Pattern matching on an enum is a single CPU branch instruction — faster than
// a vtable lookup and allows the compiler to exhaustively verify all cases.
//
// ShapeType determines:
//   - Which collision algorithm to use (AABB, SAT+slope, sphere-AABB)
//   - Which vertex data to send to the GPU for rendering
//   - What physics mass to assign on creation
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeType {
    Cube,
    Wedge,
    Sphere,
}

// =============================================================================
// LEARNING TOPIC: The Voxel Component — All Physics State in One Place
// ====================================================================
// Every dynamic and static object in the simulation is a Voxel.
// The Component derive macro tells Bevy's ECS this struct can be attached
// to entities and queried by systems.
//
// Why store both `position` AND `predicted_position`?
// This is the fundamental XPBD trick:
//   - `position`:           where the block WAS at the end of last frame
//   - `predicted_position`: where the block WANTS to go (before collision resolution)
// The solver iterates, nudging `predicted_position` out of overlaps, then
// commits it back to `position`. This "predict → correct → commit" cycle is
// what makes XPBD stable.
#[derive(Component, Debug)]
pub struct Voxel {
    /// The block's authoritative world position (committed at end of each frame)
    pub position: Vec3,

    /// Where the block predicts it will be (modified by the XPBD solver each iteration)
    pub predicted_position: Vec3,

    /// Current linear velocity in world units per second
    pub velocity: Vec3,

    /// The geometric shape — determines collision algorithm and render mesh
    pub shape: ShapeType,

    /// Only used for Sphere shape: the sphere's radius in world units
    pub sphere_radius: f32,

    // =========================================================================
    // LEARNING TOPIC: Inverse Mass — The Physics Engine's Core Trick
    // ================================================================
    // Why store 1/mass instead of mass?
    //
    // In all XPBD constraint math, mass always appears as 1/mass (inverse mass).
    // Pre-computing it avoids a division every physics step — and crucially,
    // setting inv_mass = 0.0 makes an object STATIC (immovable) without any
    // special-case code:
    //
    //   correction_share = inv_mass_a / (inv_mass_a + inv_mass_b)
    //
    // If inv_mass_a = 0.0 → share = 0 → object doesn't move.
    // If inv_mass_b = 0.0 → share = 1 → object absorbs 100% of the correction.
    //
    // This elegantly handles floor, walls, and any static platform without
    // separate code paths — just set inv_mass = 0.0.
    pub inv_mass: f32,

    /// Coefficient of friction (0.0 = frictionless ice, 1.0 = rubber on concrete)
    /// 0.8 is realistic for rough stone construction blocks.
    pub friction: f32,

    /// Coefficient of restitution (0.0 = dead stop, 1.0 = perfectly elastic bounce)
    /// 0.0 for construction blocks — they should thud down and stay, not bounce.
    pub restitution: f32,

    // =========================================================================
    // LEARNING TOPIC: Quaternions for Rotation
    // ==========================================
    // We use Quat (quaternion) instead of Euler angles (pitch/yaw/roll) because:
    //
    // 1. GIMBAL LOCK: Euler angles have a degenerate state where two rotation
    //    axes align and you lose a degree of freedom. This causes blocks to
    //    "snap" or rotate incorrectly. Quaternions have no gimbal lock.
    //
    // 2. SLERP: Quaternion spherical interpolation produces smooth, shortest-path
    //    rotation. Interpolating Euler angles creates weird spiraling arcs.
    //
    // 3. EFFICIENCY: A quaternion is 4 floats. A rotation matrix is 9 floats.
    //    Composing quaternions (q1 * q2) is cheaper than matrix multiplication.
    //
    // Quat::IDENTITY represents "no rotation" (aligned with world axes).
    pub rotation: Quat,

    /// Angular velocity vector (radians per second around each world axis).
    /// Not fully used in the current constraint solver (angular XPBD is the
    /// next sprint), but stored for future rotational dynamics.
    pub angular_velocity: Vec3,

    /// Accumulated contact normals this frame (sum of all surface normals that
    /// pushed this block). Used by update_velocities_system to:
    ///   1. Compute the "average up direction" for friction decomposition
    ///   2. Apply upright-settle rotation (lerp toward the surface's normal)
    pub contact_normal_accum: Vec3,

    /// Number of contacts this frame — used to normalize contact_normal_accum
    /// into an average rather than a sum.
    pub contact_count: u32,
}

impl Voxel {
    pub fn new(x: f32, y: f32, z: f32, shape: ShapeType, is_static: bool) -> Self {
        // =====================================================================
        // LEARNING TOPIC: Physics-Derived Mass Values
        // ============================================
        //
        // BUG FIX #2 (Wedge Stability) — Physics Calculation:
        //
        // A cube (1 kg) dropped from height h = 10.0 m arrives with velocity:
        //   v_impact = sqrt(2 * g * h) = sqrt(2 * 9.81 * 10) ≈ 14.0 m/s
        //
        // Impact momentum:
        //   p = m_cube * v_impact = 1.0 * 14.0 = 14.0 kg·m/s
        //
        // For the wedge to not visibly move, its velocity change must be tiny:
        //   Δv_wedge = p / m_wedge < 0.01 m/s  (imperceptible threshold)
        //   m_wedge > p / 0.01 = 14.0 / 0.01 = 1400 kg
        //
        // We use m_wedge = 2000 kg (generous safety factor ≈ 1.4×):
        //   inv_mass_wedge = 1 / 2000 = 0.0005
        //
        // Additionally, in xpbd.rs we pin the wedge's correction share to
        // near-zero when it's settled on the ground, making it behave as a
        // static platform for collision response (the cube slides up the slope
        // instead of sending the wedge flying backward).
        //
        // OLD VALUE: 0.002  (mass = 500 kg → Δv ≈ 0.028 m/s → visible wobble)
        // NEW VALUE: 0.0005 (mass = 2000 kg → Δv ≈ 0.007 m/s → imperceptible)
        // =====================================================================
        let dynamic_inv_mass = match shape {
            // Cube: 1 kg, inv_mass = 1.0
            // Feels light but sturdy — standard construction block.
            ShapeType::Cube => 1.0,

            // Wedge: 2000 kg, inv_mass = 0.0005
            // PHYSICS DERIVATION: Must resist impulse from cube dropped at 14 m/s.
            // A 2000 kg wedge will only shift 0.007 m/s when hit — imperceptible.
            // Combined with the WEDGE_SETTLED_SHARE_CAP in xpbd.rs, the wedge
            // behaves as a rock-solid ramp once settled on the ground.
            ShapeType::Wedge => 0.0005,

            // Sphere: 1 kg, inv_mass = 1.0
            // Same as cube for now — spheres roll via XPBD contact normals.
            ShapeType::Sphere => 1.0,
        };

        Self {
            position: Vec3::new(x, y, z),
            predicted_position: Vec3::new(x, y, z),
            velocity: Vec3::ZERO,
            shape,
            sphere_radius: 0.5,
            inv_mass: if is_static { 0.0 } else { dynamic_inv_mass },
            // 0.8 friction: rough stone — blocks grip each other and don't slide off
            // high towers easily. Good for construction stability.
            friction: 0.8,
            // 0.0 restitution: no bounce. A construction block landing on a ramp
            // should stop and slide, not ricochet off.
            restitution: 0.0,
            rotation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
            contact_normal_accum: Vec3::ZERO,
            contact_count: 0,
        }
    }

    // =========================================================================
    // LEARNING TOPIC: Builder Pattern for Variant Construction
    // =========================================================
    // Instead of overloading `new()` (Rust doesn't have function overloading),
    // we provide a separate named constructor for specialized configurations.
    // new_sphere() creates a sphere voxel with a custom radius — it delegates
    // to new() then overrides just the sphere-specific field.
    // =========================================================================
    pub fn new_sphere(x: f32, y: f32, z: f32, radius: f32, is_static: bool) -> Self {
        let mut voxel = Self::new(x, y, z, ShapeType::Sphere, is_static);
        // Enforce minimum radius: a sphere of radius < 0.2 would be smaller than
        // a grid cell, causing it to phase through geometry between frames.
        voxel.sphere_radius = radius.max(0.2);
        voxel
    }
}
