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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShapeType {
    Cube,
    Wedge,
    Sphere,
}

// =============================================================================
// SPRINT 3 + 4: MaterialType — Drives All Physics Properties
// =============================================================================
//
// LEARNING: This enum is the single source of truth for material behavior.
// Rather than letting Python pass arbitrary floats for mass/friction/adhesion
// (error-prone), we encode the valid combinations here and derive them.
// The Voxel::new_with_material() constructor reads this enum and sets all
// physics fields automatically — no manual tuning per spawn call.
//
// MATERIAL PHYSICS PROFILES:
//
//   Steel    — heavy (mass=800kg), high adhesion (can overhang 4+ blocks)
//              High adhesion_strength means the mortar bond survives large
//              breaking force checks. Good for bridges and arches.
//
//   Wood     — light (mass=1kg), medium adhesion (overhang ~1 block)
//              Easy to carry, moderate adhesion. Useful for scaffolding
//              and light structures. Bond breaks under moderate load.
//
//   Stone    — medium mass (mass=50kg), ZERO adhesion, high friction
//              Stacks only — no side bonding at all. Best for foundations
//              and ground-level walls where pure stacking is sufficient.
//
//   Scaffold — SPRINT 4: ultra-light (mass=0.5kg), medium adhesion, removable.
//              Temporary support during arch/bridge construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaterialType {
    /// Heavy, rigid, high adhesion. Ideal for structural spans.
    /// inv_mass = 1/800 = 0.00125
    /// adhesion_strength = 8.0  (can resist 8× its own weight laterally)
    /// friction = 0.6
    Steel,

    /// Light, medium adhesion. Ideal for scaffolding and light walls.
    /// inv_mass = 1/1 = 1.0  (standard reference mass)
    /// adhesion_strength = 2.0  (can resist 2× its own weight laterally)
    /// friction = 0.7
    Wood,

    /// Medium weight, zero adhesion, high friction. Stack only.
    /// inv_mass = 1/50 = 0.02
    /// adhesion_strength = 0.0  (never bonds to side-neighbors)
    /// friction = 0.95
    Stone,

    // =========================================================================
    // SPRINT 4: Scaffold Material
    // =========================================================================
    //
    // LEARNING TOPIC: Why Scaffolding Is Needed for Arch Construction
    // ----------------------------------------------------------------
    // An arch is structurally self-supporting ONCE COMPLETE — the stones push
    // outward against the abutments (walls) and the keystone locks them in place.
    // But during construction, the arch blocks on each side have NO keystone yet.
    // Without something to hold them up, they fall inward before the arch closes.
    //
    // In Operation Babel, the agent faces the same problem. To build:
    //   Step 1: Place left arch block (held in place by scaffold below it)
    //   Step 2: Place right arch block (held in place by scaffold below it)
    //   Step 3: Place keystone at top — arch becomes self-supporting via mortar
    //   Step 4: Remove scaffold → arch stands alone
    //
    // PHYSICS PROPERTIES OF SCAFFOLD:
    //   - Ultra-light (0.5 kg): doesn't add stress to the structure it supports
    //   - Medium adhesion (2.0): bonds to arch blocks to hold them laterally
    //   - High friction (0.8): grips the ground well
    //   - is_scaffold flag: marks it for mass removal by despawn_scaffolding()
    //   - Stress capacity = 5×: turns red quickly as a "remove me" warning
    //
    // inv_mass = 1/0.5 = 2.0
    // adhesion_strength = 2.0  (same as wood — moderate lateral hold)
    // friction = 0.8
    /// Ultra-light temporary support. Place during arch construction, remove after.
    Scaffold,
}

impl MaterialType {
    // =========================================================================
    // Physics profile derivation — one source of truth
    // =========================================================================

    /// Inverse mass (1/kg). Used directly in XPBD constraint math.
    /// Static objects use 0.0 regardless of material.
    pub fn inv_mass(self) -> f32 {
        match self {
            MaterialType::Steel => 1.0 / 800.0, // 800 kg — very heavy
            MaterialType::Wood => 1.0,          // 1 kg   — reference mass
            MaterialType::Stone => 1.0 / 50.0,  // 50 kg  — medium
            // SPRINT 4: Scaffold is ultra-light so it adds minimal load to the
            // structure it supports. 0.5 kg means even a 10-block scaffold
            // column adds only 5 kg to the base — negligible for stress purposes.
            MaterialType::Scaffold => 1.0 / 0.5, // 0.5 kg — ultra-light
        }
    }

    /// Friction coefficient (0 = frictionless ice, 1 = rubber on concrete).
    pub fn friction(self) -> f32 {
        match self {
            MaterialType::Steel => 0.6,
            MaterialType::Wood => 0.7,
            MaterialType::Stone => 0.95,   // roughest — grips stack well
            MaterialType::Scaffold => 0.8, // good grip — scaffold must not slide
        }
    }

    /// Restitution (bounciness). Construction blocks should not bounce.
    pub fn restitution(self) -> f32 {
        match self {
            MaterialType::Steel => 0.05,   // tiny bounce (metal ring)
            MaterialType::Wood => 0.0,     // dead stop
            MaterialType::Stone => 0.02,   // almost nothing
            MaterialType::Scaffold => 0.0, // scaffold should thud and stay put
        }
    }

    /// How strongly side-bonds resist breaking.
    ///
    /// adhesion_strength is a DIMENSIONLESS MULTIPLIER in world-unit-stretch space.
    /// In break_overloaded_bonds_system the survival check is:
    ///   survives = adhesion_strength >= tension (bond stretch in world units)
    ///
    /// Stone returns 0.0 — it never forms side bonds at all.
    pub fn adhesion_strength(self) -> f32 {
        match self {
            MaterialType::Steel => 8.0, // survives up to 8 world-units of stretch
            MaterialType::Wood => 2.0,  // survives up to 2 world-units of stretch
            MaterialType::Stone => 0.0, // no adhesion — stack only
            // SPRINT 4: Scaffold has the same adhesion as wood — it needs to bond
            // laterally to arch blocks to hold them in position.
            MaterialType::Scaffold => 2.0, // same as wood — moderate hold
        }
    }

    /// Maximum overhang this material can support (in block units).
    /// Purely informational — the breaking force check enforces this at runtime.
    pub fn max_overhang_blocks(self) -> u32 {
        match self {
            MaterialType::Steel => 4,
            MaterialType::Wood => 1,
            MaterialType::Stone => 0,
            MaterialType::Scaffold => 1, // same as wood
        }
    }

    /// Whether this material type is scaffold (removable temporary support).
    ///
    /// SPRINT 4 LEARNING: We use a method rather than a separate field so we
    /// don't need to pass `is_scaffold` through every constructor call. Any code
    /// that needs to know "is this removable?" just calls voxel.material.is_scaffold().
    pub fn is_scaffold(self) -> bool {
        matches!(self, MaterialType::Scaffold)
    }
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
//
// SPRINT 4.5 ADDITION: floor_contact field
// -----------------------------------------
// Previously the sleep system only checked contact_count (block-block contacts).
// Blocks resting on the floor had contact_count = 0 because the floor constraint
// is applied analytically (not through the block-block pair solver). This meant
// blocks on the floor NEVER entered the sleep state — wasting CPU and causing
// the test suite's sleep tests to fail.
//
// floor_contact is set to true in solve_constraints_system when the floor
// constraint fires, reset to false in integrate_system each frame.
// The sleep check in update_velocities_system reads EITHER contact_count > 0
// OR floor_contact to decide if a block is "grounded enough to sleep".
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
    pub friction: f32,

    /// Coefficient of restitution (0.0 = dead stop, 1.0 = perfectly elastic bounce)
    /// Construction blocks should thud down and stay, not bounce.
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
    /// Stored for future full angular XPBD dynamics (Sprint 5+).
    pub angular_velocity: Vec3,

    /// Accumulated contact normals this frame (sum of all surface normals that
    /// pushed this block). Used by update_velocities_system to:
    ///   1. Compute the "average up direction" for friction decomposition
    ///   2. Apply upright-settle rotation (lerp toward the surface's normal)
    pub contact_normal_accum: Vec3,

    /// Number of BLOCK-BLOCK contacts this frame — used to normalize
    /// contact_normal_accum into an average rather than a sum.
    ///
    /// LEARNING: This is ONLY block-block contacts, NOT floor contacts.
    /// The distinction matters for the sleep system — see floor_contact below.
    /// The floor also contributes to contact_normal_accum and contact_count
    /// through the normal_accum buffer in the solver, but floor_contact is the
    /// dedicated boolean flag for clean sleep detection.
    pub contact_count: u32,

    /// Sleep flag for broad-phase/integration skipping.
    /// Sleeping blocks are treated as temporarily static until a contact
    /// correction wakes them up.
    pub is_sleeping: bool,

    // =========================================================================
    // SPRINT 4.5 NEW FIELD: floor_contact
    // =========================================================================
    //
    // LEARNING: Why is this separate from contact_count?
    //
    // contact_count tracks contacts resolved in the XPBD pair solver
    // (Step B2/C of solve_constraints_system). It is also incremented by the
    // floor constraint (which adds Y to normal_accum and increments normal_count
    // → contact_count). However, a block at EXACTLY floor height has
    // lowest_point_y == floor_y, so the condition `lowest_point_y < floor_y`
    // is FALSE. No correction fires, no floor contribution to contact_count.
    //
    // floor_contact is set to true when the block's lowest point is AT OR
    // NEAR (within a small epsilon) the floor. This catches both:
    //   - Blocks actively being corrected upward (penetrating)
    //   - Blocks resting exactly on the floor (no penetration needed)
    //
    // Without floor_contact, a block resting precisely on the floor had
    // contact_count = 0 → sleep NEVER triggered → test failures.
    pub floor_contact: bool,

    // =========================================================================
    // SPRINT 3 NEW FIELDS
    // =========================================================================
    /// Which material this block is made of.
    /// Determines adhesion_strength, mass, friction, restitution at spawn.
    /// SPRINT 4: Also encodes whether this block is scaffold (removable).
    pub material: MaterialType,

    /// Adhesion strength — how much lateral bond stretch this block can resist.
    ///
    /// Static blocks carry their material's adhesion_strength so they can act
    /// as valid bond anchors for dynamic neighbors.
    /// The inv_mass = 0.0 field prevents them from being moved by the bond solver.
    pub adhesion_strength: f32,
}

impl Voxel {
    // =========================================================================
    // Original constructor — defaults to Wood material for backward compatibility
    // =========================================================================
    pub fn new(x: f32, y: f32, z: f32, shape: ShapeType, is_static: bool) -> Self {
        Self::new_with_material(x, y, z, shape, MaterialType::Wood, is_static)
    }

    // =========================================================================
    // SPRINT 3: Material-aware constructor
    // =========================================================================
    //
    // is_static = true → inv_mass forced to 0.0 (immovable) regardless of material.
    //             adhesion_strength is KEPT from the material so static blocks
    //             act as valid bond anchors for dynamic neighbors.
    pub fn new_with_material(
        x: f32,
        y: f32,
        z: f32,
        shape: ShapeType,
        material: MaterialType,
        is_static: bool,
    ) -> Self {
        // =====================================================================
        // LEARNING TOPIC: Wedge Mass Override
        // ====================================
        // The Wedge shape needs a very high mass (2000 kg, inv_mass=0.0005)
        // regardless of material, to prevent it from flying when a cube lands
        // on it at high velocity. See xpbd.rs BUG FIX #2 for the physics math.
        // =====================================================================
        let dynamic_inv_mass = if shape == ShapeType::Wedge {
            0.0005 // Wedge stability fix — 2000 kg to resist impact momentum
        } else {
            material.inv_mass()
        };

        Self {
            position: Vec3::new(x, y, z),
            predicted_position: Vec3::new(x, y, z),
            velocity: Vec3::ZERO,
            shape,
            sphere_radius: 0.5,
            inv_mass: if is_static { 0.0 } else { dynamic_inv_mass },
            friction: material.friction(),
            restitution: material.restitution(),
            rotation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
            contact_normal_accum: Vec3::ZERO,
            contact_count: 0,
            is_sleeping: false,
            floor_contact: false, // SPRINT 4.5: initialized to false, set each frame by solver
            material,
            adhesion_strength: material.adhesion_strength(),
        }
    }

    // =========================================================================
    // Sphere constructor — delegates to new() then overrides sphere_radius
    // =========================================================================
    pub fn new_sphere(x: f32, y: f32, z: f32, radius: f32, is_static: bool) -> Self {
        let mut voxel = Self::new(x, y, z, ShapeType::Sphere, is_static);
        // Enforce minimum radius: a sphere of radius < 0.2 would be smaller than
        // a grid cell, causing it to phase through geometry between frames.
        voxel.sphere_radius = radius.max(0.2);
        voxel
    }

    // =========================================================================
    // SPRINT 3: Sphere with material
    // =========================================================================
    pub fn new_sphere_with_material(
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        material: MaterialType,
        is_static: bool,
    ) -> Self {
        let mut voxel = Self::new_with_material(x, y, z, ShapeType::Sphere, material, is_static);
        voxel.sphere_radius = radius.max(0.2);
        voxel
    }
}
