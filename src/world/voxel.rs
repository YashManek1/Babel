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
// SPRINT 3: MaterialType — Drives All Physics Properties
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
//   Steel  — heavy (mass=800kg), high adhesion (can overhang 4+ blocks)
//             High adhesion_strength means the mortar bond survives large
//             breaking force checks. Good for bridges and arches.
//
//   Wood   — light (mass=1kg), medium adhesion (overhang ~1 block)
//             Easy to carry, moderate adhesion. Useful for scaffolding
//             and light structures. Bond breaks under moderate load.
//
//   Stone  — medium mass (mass=50kg), ZERO adhesion, high friction
//             Stacks only — no side bonding at all. Best for foundations
//             and ground-level walls where pure stacking is sufficient.
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
        }
    }

    /// Friction coefficient (0 = frictionless, 1 = rubber on concrete).
    pub fn friction(self) -> f32 {
        match self {
            MaterialType::Steel => 0.6,
            MaterialType::Wood => 0.7,
            MaterialType::Stone => 0.95, // roughest — grips stack well
        }
    }

    /// Restitution (bounciness). Construction blocks should not bounce.
    pub fn restitution(self) -> f32 {
        match self {
            MaterialType::Steel => 0.05, // tiny bounce (metal ring)
            MaterialType::Wood => 0.0,   // dead stop
            MaterialType::Stone => 0.02, // almost nothing
        }
    }

    /// How strongly side-bonds resist breaking.
    ///
    /// SPRINT 3 BUG FIX — Unit Clarification:
    /// ----------------------------------------
    /// adhesion_strength is a DIMENSIONLESS MULTIPLIER, not raw Newtons.
    ///
    /// In break_overloaded_bonds_system the survival check is:
    ///   survives = adhesion_strength >= breaking_force_normalized
    ///
    /// where breaking_force_normalized = tension (bond stretch in world units,
    /// typically 0.0 to ~0.5 before the bond breaks geometrically).
    ///
    /// So adhesion_strength = 2.0 means "survive up to 2 world-units of stretch."
    /// Since tension rarely exceeds 0.5 under normal loads, Wood bonds are stable
    /// for light structures, and Steel bonds are nearly unbreakable in practice.
    ///
    /// Stone returns 0.0 — it never forms side bonds at all.
    pub fn adhesion_strength(self) -> f32 {
        match self {
            MaterialType::Steel => 8.0, // survives up to 8 world-units of stretch
            MaterialType::Wood => 2.0,  // survives up to 2 world-units of stretch
            MaterialType::Stone => 0.0, // no adhesion — stack only
        }
    }

    /// Maximum overhang this material can support (in block units).
    /// Purely informational — the breaking force check enforces this at runtime.
    /// Useful for UI display and agent heuristics.
    pub fn max_overhang_blocks(self) -> u32 {
        match self {
            MaterialType::Steel => 4,
            MaterialType::Wood => 1,
            MaterialType::Stone => 0,
        }
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

    /// Sleep flag for broad-phase/integration skipping.
    /// Sleeping blocks are treated as temporarily static until a contact
    /// correction wakes them up.
    pub is_sleeping: bool,

    // =========================================================================
    // SPRINT 3 NEW FIELDS
    // =========================================================================
    /// Which material this block is made of.
    /// Determines adhesion_strength, mass, friction, restitution at spawn.
    pub material: MaterialType,

    /// Adhesion strength — how much lateral bond stretch this block can resist.
    ///
    /// SPRINT 3 BUG FIX — Static Block Bond Registration:
    /// ---------------------------------------------------
    /// Previously this was set to 0.0 for static blocks. That caused
    /// try_register_bonds() to abort early, so a dynamic block with adhesion
    /// could never bond to a static wall that was spawned after it.
    ///
    /// FIX: Static blocks now carry their material's adhesion_strength.
    /// The mortar system uses this when the NEIGHBOR block is dynamic —
    /// the bond forms using min(self.adhesion, neighbor.adhesion), so static
    /// blocks act as valid anchor points for dynamic block bonds.
    ///
    /// The inv_mass = 0.0 field already handles the "static blocks don't move"
    /// invariant in XPBD. adhesion_strength is purely for bond eligibility.
    pub adhesion_strength: f32,
}

impl Voxel {
    // =========================================================================
    // Original constructor — defaults to Wood material for backward compatibility
    // =========================================================================
    //
    // LEARNING: All existing calls to Voxel::new() (from lib.rs, test_engine.py,
    // babel_gym_env.py) continue to work unchanged. They get Wood material which
    // has the same mass as the old default (inv_mass = 1.0). No behavior change
    // for existing tests.
    pub fn new(x: f32, y: f32, z: f32, shape: ShapeType, is_static: bool) -> Self {
        Self::new_with_material(x, y, z, shape, MaterialType::Wood, is_static)
    }

    // =========================================================================
    // SPRINT 3: Material-aware constructor
    // =========================================================================
    //
    // This is the preferred constructor going forward. The agent will call this
    // via a new spawn_block_with_material() Python method (added to lib.rs).
    //
    // is_static = true → inv_mass forced to 0.0 (immovable) regardless of material.
    //             adhesion_strength is KEPT from the material — see field comment
    //             above for why this matters for bond registration.
    pub fn new_with_material(
        x: f32,
        y: f32,
        z: f32,
        shape: ShapeType,
        material: MaterialType,
        is_static: bool,
    ) -> Self {
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
        //
        // SPRINT 3: Wedge special case — must be heavy enough to not fly when hit.
        // We keep the Sprint 2 fix (0.0005) only for the Wedge shape,
        // overriding whatever the material says for mass.
        // Rationale: a wedge is a ramp — its stability matters more than
        // matching the material's weight. This can be revisited in Sprint 5
        // when humanoid locomotion needs climbing on wedge ramps.
        // =====================================================================
        let dynamic_inv_mass = if shape == ShapeType::Wedge {
            0.0005 // Sprint 2 wedge stability fix — keep unchanged
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
            material,
            // =================================================================
            // SPRINT 3 BUG FIX: Do NOT zero out adhesion_strength for static
            // blocks. Static blocks serve as valid bond anchors for dynamic
            // neighbors. Their inv_mass = 0.0 already prevents them from being
            // pushed by the bond solver — adhesion_strength is orthogonal.
            //
            // Stone material returns 0.0 regardless, so Stone static walls
            // still won't bond to anything. This is the correct behavior.
            // =================================================================
            adhesion_strength: material.adhesion_strength(),
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
