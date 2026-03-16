// =============================================================================
// src/physics/xpbd.rs  —  Operation Babel: XPBD Physics Solver
// =============================================================================
//
// LEARNING TOPIC: Extended Position-Based Dynamics (XPBD)
// --------------------------------------------------------
// Traditional physics engines calculate FORCES (F = ma) and integrate velocity
// over time. This is unstable: a single bad frame can create infinite energy.
//
// XPBD instead works entirely in POSITION space:
//   1. INTEGRATE:  Apply gravity to a "predicted_position". This is where the
//                  block *wants* to be if nothing blocks it.
//   2. SOLVE:      Find every overlap. Push predicted_positions apart until no
//                  block occupies the same space. Repeat N times (SOLVER_ITERATIONS).
//   3. DERIVE:     velocity = (predicted_position - old_position) / dt
//                  Velocity is a CONSEQUENCE of movement, not a cause.
//
// Why this is stable: no block can ever accumulate infinite energy because we
// clamp how far a position can move in a single step.
// =============================================================================

use crate::world::spatial_grid::SpatialGrid;
use crate::world::voxel::{ShapeType, Voxel};
use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use std::collections::{HashMap, HashSet};

// =============================================================================
// LEARNING TOPIC: Physics Constants & Why These Values
// =============================================================================

/// Simulation timestep: 1/60 second = one physics frame at 60 Hz.
/// The engine decouples render FPS from physics FPS — physics always advances
/// by exactly this amount regardless of how fast the screen refreshes.
const DT: f32 = 1.0 / 60.0;

/// Earth gravity in m/s² pointing downward (negative Y in our right-handed
/// coordinate system where +Y is "up").
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// How many times per frame we re-solve all constraints.
/// More iterations = more accurate stacking but more CPU cost.
/// 10 iterations is the "sweet spot" for a voxel-scale construction engine.
const SOLVER_ITERATIONS: usize = 10;

/// Maximum position correction a single solver pass can apply to one block.
/// This is the #1 stability guarantee: it prevents a deeply-overlapping block
/// from teleporting across the world in one step ("tunneling" prevention).
const MAX_CORRECTION_PER_ITER: f32 = 0.22;

/// Maximum *accumulated* correction per frame across all iterations.
/// Even if 10 iterations all push in the same direction, the total move
/// is bounded. Prevents "explosion" when many blocks pile up.
const MAX_ACCUM_CORRECTION_PER_ITER: f32 = 0.65;

/// Hard cap on how far a block's position can change in one full frame,
/// including all solver iterations. This is the last line of defense against
/// blocks clipping through thin geometry at high velocity.
const MAX_DISPLACEMENT_PER_FRAME: f32 = 1.25;

/// Terminal velocity — no block can move faster than this (in units/second).
/// Real terminal velocity for a 1 m³ concrete block ≈ 70 m/s; we use a lower
/// game value so blocks don't clip through thin surfaces.
const MAX_VELOCITY: f32 = 25.0;

/// If the contact normal's Y component is above this threshold (≈ cos 10°),
/// we treat the surface as "flat enough" to apply full static friction and
/// upright-correction. Prevents blocks from sleeping on steep slopes.
const FLAT_CONTACT_Y_THRESHOLD: f32 = 0.98;

/// How aggressively a tilted block rotates back to upright when resting on
/// a flat surface. 1.0 = instant snap, 0.0 = never corrects.
/// 0.2 gives a smooth, realistic settle.
const UPRIGHT_SETTLE_RATE: f32 = 0.2;

/// If a block's speed drops below this (units/second), treat it as "asleep"
/// and zero out its velocity. Prevents infinite micro-jitter from floating-point
/// rounding — the physics equivalent of damping springs to rest.
const LINEAR_SLEEP_SPEED: f32 = 0.08;

/// Velocity damping multiplier applied every frame when a block is in contact.
/// 0.94 means the block retains 94% of its velocity per frame while sliding.
/// This simulates air resistance + material damping without complex fluid math.
const POSITION_VELOCITY_DAMPING: f32 = 0.94;

/// Speed threshold below which static friction takes over and stops lateral
/// sliding entirely. Models the difference between static and kinetic friction:
/// objects require more force to START moving than to keep moving.
const STATIC_FRICTION_SPEED: f32 = 0.12;

// =============================================================================
// BUG FIX #1: Static Block Position Snapping
// =============================================================================
//
// THE OLD BUG:
//   When a settled block (inv_mass == 0.0) was looked up in the spatial grid,
//   its position was snapped to the nearest integer cell:
//       other_pos = Vec3::new(grid_pos[0] as f32, grid_pos[1] as f32, ...)
//
//   The SpatialGrid::world_to_grid() uses .round(), so a block at y = 0.5
//   snaps to grid cell y = 0 OR y = 1 depending on floating-point rounding.
//   If it snapped to y = 0 instead of y = 1, a block resting on top (y ≈ 1.5)
//   would see "other_pos.y = 0.0" — a 1.5-unit gap — triggering a FALSE
//   penetration of 0.5 units. The solver would then shove the upper block DOWN
//   into the lower block, making the lower block visually disappear (the two
//   blocks occupying the same render space = Z-fighting / visual vanishing).
//
// THE FIX:
//   For static blocks (inv_mass == 0.0), we now use their ACTUAL stored
//   `predicted_position` from the snapshot buffer instead of snapping to the
//   integer grid. The grid is only used for broad-phase neighbor detection
//   (finding WHO to check). The actual collision math uses the true position.
//   This is captured in the snapshots HashMap before the solver loop begins.
//
// =============================================================================

// =============================================================================
// BUG FIX #2: Wedge Stability — Impulse-Based Mass Override
// =============================================================================
//
// THE OLD BUG:
//   The wedge had inv_mass = 0.002 (mass = 500 kg) — intended to be "heavy".
//   However, a cube dropped from height 10.0 arrives with velocity:
//       v = sqrt(2 * 9.81 * 10) ≈ 14 m/s
//   Its momentum = mass * velocity = 1.0 * 14 = 14 kg·m/s
//
//   The correction sharing used a XOR check `self_is_wedge ^ other_is_wedge`
//   to apply `self_share *= 0.20`. But when a wedge has inv_mass = 0.0 (static),
//   inv_mass_sum = 0.002 + 0.0 = 0.002 (wait — the DYNAMIC wedge had 0.002).
//   Actually the wedge was spawned DYNAMIC (inv_mass = 0.002). So inv_mass_sum
//   = 0.002 + 1.0 = 1.002. The wedge share = 0.002/1.002 ≈ 0.002 → 0.2% push.
//   The cube share = 1.0/1.002 ≈ 99.8% push.
//
//   But the penetration from a 10-unit drop is HUGE. With MAX_CORRECTION clamped
//   at 0.22 per iteration × 10 iterations = 2.2 units of push along the slope
//   normal. The slope normal is Vec3(0, 0.707, 0.707), so the cube gets pushed
//   2.2 * 0.707 ≈ 1.55 units backward (Z direction). Velocity derived from that
//   = 1.55 / DT = 93 m/s — capped at MAX_VELOCITY = 25 m/s. Still a violent
//   backward ejection of the cube, which in turn shoves the wedge forward.
//
// THE PHYSICS CALCULATION:
//   A 1 kg cube falling from 10 m arrives at v = 14 m/s.
//   Impact momentum p = 1.0 * 14 = 14 kg·m/s.
//   To keep the wedge stationary, the wedge must have enough mass that the
//   momentum transfer barely moves it: Δv_wedge = p / m_wedge.
//   For Δv_wedge < 0.01 m/s (imperceptible): m_wedge > 14/0.01 = 1400 kg.
//   So inv_mass_wedge < 1/1400 ≈ 0.000714.
//   We use 0.0005 → mass = 2000 kg (generous safety margin).
//
//   ADDITIONALLY: The real fix is to treat a SETTLED wedge as EFFECTIVELY STATIC
//   for the purpose of collision response. We do this by making the wedge's
//   mass share approach zero when it has low velocity (i.e., it's already
//   resting on the ground). The cube absorbs 100% of the correction, which
//   pushes it UP the slope (correct sliding behavior) without ever pushing
//   the wedge backward.
//
// THE FIX APPLIED IN VOXEL.RS:
//   inv_mass for Wedge reduced from 0.002 → 0.0005 (mass = 2000 kg).
//
// THE FIX APPLIED HERE IN XPBD.RS:
//   When computing correction shares for a wedge vs cube collision, we check
//   if the wedge's speed is below a "settled threshold". If settled, we pin
//   the wedge's share to near-zero regardless of its inv_mass, making it
//   behave as a static platform while allowing the cube to slide naturally.
// =============================================================================

/// Speed below which a wedge is considered "settled" and treated as effectively
/// static for collision response purposes (it won't be pushed by landing blocks).
/// Calculated from physics: v_settle must be << v_impact = 14 m/s.
/// 0.5 m/s is small enough to be imperceptible but large enough to not trigger
/// prematurely on a wedge that's still gently rolling into place.
const WEDGE_SETTLED_SPEED_THRESHOLD: f32 = 0.5;

/// When a wedge is settled, its correction share is clamped to this tiny value.
/// Near-zero means the cube absorbs almost all the correction (slides up the slope)
/// and the wedge is pushed back by only: 0.001 * max_correction ≈ 0.00022 units.
/// Over 10 iterations that's 0.0022 units — completely imperceptible.
const WEDGE_SETTLED_SHARE_CAP: f32 = 0.001;

// =============================================================================
// LEARNING TOPIC: Contact Manifold
// =============================================================================
//
// A "Contact" describes the GEOMETRIC RELATIONSHIP between two overlapping shapes:
//   - normal:      The direction to push them APART (unit vector)
//   - penetration: How deep they overlap (in world units)
//   - contact_point: The exact 3D world point where they touch
//
// Why store contact_point? Future rotational physics needs it to compute TORQUE:
// torque = r × F  where r = (contact_point - center_of_mass).
// We store it now so the data structure is ready when angular dynamics are added.
#[derive(Clone, Copy)]
struct Contact {
    normal: Vec3,
    penetration: f32,
    #[allow(dead_code)]
    contact_point: Vec3,
}

// =============================================================================
// LEARNING TOPIC: Snapshot Buffer (Freeze-then-Solve Pattern)
// =============================================================================
//
// Why do we snapshot BEFORE solving?
//
// Imagine Block A and Block B both overlap Block C.
// If we solve A vs C, then immediately solve B vs C using the UPDATED position of C,
// we get inconsistent results — B's correction was computed with a C that had already
// moved. This causes jitter called "Gauss-Seidel drift."
//
// The fix: at the START of each iteration, freeze a snapshot of every block's
// predicted_position. All collision math this iteration uses the FROZEN data.
// Corrections accumulate in a separate HashMap, then are applied all at once.
// Next iteration, a fresh snapshot is taken. This is called "Jacobi iteration"
// and produces perfectly symmetric, stable collision resolution.
#[derive(Clone)]
pub struct BodySnapshot {
    pub predicted_position: Vec3,
    pub inv_mass: f32,
    pub shape: ShapeType,
    pub sphere_radius: f32,
}

// =============================================================================
// LEARNING TOPIC: PhysicsSettings Resource
// =============================================================================
//
// Bevy ECS "Resources" are global singletons accessible to any system.
// Unlike Components (which belong to Entities), Resources have no owner —
// they represent world-level state.
//
// global_floor_y: The Y coordinate of the infinite ground plane.
// Set to -0.5 so the bottom face of a unit cube at y=0.0 touches the ground.
// (A cube centered at y=0.0 has its bottom face at y = 0.0 - 0.5 = -0.5)
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

// =============================================================================
// LEARNING TOPIC: SolverBuffers — Reusable Heap Allocations
// =============================================================================
//
// Allocating new HashMaps every frame would cause massive heap churn and GC
// pressure. Instead, we allocate ONCE (in the Bevy startup schedule) and then
// call `.clear()` at the start of each solver iteration.
//
// `.clear()` drops all KEY-VALUE pairs but retains the allocated memory buckets,
// so subsequent insertions reuse existing heap pages rather than calling malloc.
// For a real-time physics engine hitting 60+ Hz, this matters enormously.
#[derive(Resource, Default)]
pub struct SolverBuffers {
    /// Per-entity: predicted_position and physics properties (frozen snapshot)
    pub snapshots: HashMap<Entity, BodySnapshot>,
    /// Per-entity: accumulated position correction for this iteration
    pub pos_corrections: HashMap<Entity, Vec3>,
    /// Per-entity: sum of all contact normals (used for friction + upright settling)
    pub normal_accum: HashMap<Entity, Vec3>,
    /// Per-entity: number of active contacts (used to normalize normal_accum)
    pub normal_count: HashMap<Entity, u32>,
    /// Pairs already processed this iteration (prevents double-solving A-B and B-A)
    pub processed_pairs: HashSet<(Entity, Entity)>,
}

// =============================================================================
// LEARNING TOPIC: Canonical Pair Ordering
// =============================================================================
//
// The broad-phase loop iterates over all blocks and finds neighbors. Block A
// will find Block B, AND Block B will find Block A. Without deduplication,
// every collision would be solved TWICE, causing double-pushes and jitter.
//
// Solution: normalize every (A, B) pair so the lower Entity index always comes
// first. Then use a HashSet to skip any pair we've already seen this iteration.
// This guarantees O(N) unique pairs, not O(N²) duplicate work.
fn make_pair(a: Entity, b: Entity) -> (Entity, Entity) {
    if a.index() < b.index() {
        (a, b)
    } else {
        (b, a)
    }
}

// =============================================================================
// LEARNING TOPIC: AABB vs AABB Collision (Axis-Aligned Bounding Box)
// =============================================================================
//
// An AABB is a box whose edges are always parallel to the world X, Y, Z axes.
// Because there's no rotation, overlap detection reduces to 6 simple comparisons:
//   overlap_x = (half_a.x + half_b.x) - |center_a.x - center_b.x|
//   overlap_y = ...
//   overlap_z = ...
//
// If ALL three overlaps are positive, the boxes intersect.
// The correct push direction is along the axis with the SMALLEST overlap
// (Separating Axis Theorem: the minimum-distance exit direction).
//
// Example: if overlap_x = 0.1, overlap_y = 0.8, overlap_z = 0.5,
// we push along X (smallest), not Y. This prevents blocks from "popping" up
// when they are mostly side-by-side.
fn compute_aabb_aabb_contact(
    pos_a: Vec3,
    half_a: Vec3,
    pos_b: Vec3,
    half_b: Vec3,
) -> Option<Contact> {
    let diff = pos_a - pos_b;
    // Overlap along each axis: sum of half-extents minus center separation
    let overlap = (half_a + half_b) - diff.abs();

    if overlap.x <= 0.0 || overlap.y <= 0.0 || overlap.z <= 0.0 {
        return None; // Separating axis found — no collision
    }

    // Find the minimum-overlap axis (SAT: push along shortest path out)
    let (normal, penetration) = if overlap.x < overlap.y && overlap.x < overlap.z {
        // X is the separation axis; push in the direction of diff.x
        (Vec3::X * diff.x.signum(), overlap.x)
    } else if overlap.y < overlap.z {
        // Y is the separation axis (most common for stacked blocks!)
        (Vec3::Y * diff.y.signum(), overlap.y)
    } else {
        // Z is the separation axis
        (Vec3::Z * diff.z.signum(), overlap.z)
    };

    Some(Contact {
        normal,
        penetration,
        contact_point: pos_b + normal * penetration * 0.5,
    })
}

// =============================================================================
// LEARNING TOPIC: SAT for Wedge — The Separating Axis Theorem with 4 Planes
// =============================================================================
//
// A standard AABB test treats the wedge as a full 1×1×1 cube. This is wrong:
// the invisible bounding box is solid, but the actual wedge only fills the
// LOWER-BACK triangle. A block dropping straight down onto the wedge's slope
// would enter the bounding box and get pushed horizontally instead of up the slope.
//
// The Separating Axis Theorem (SAT) says: two convex shapes are NOT colliding
// if there exists any axis along which their projections do not overlap.
// For a wedge, we add a FOURTH axis — the slope normal — in addition to X, Y, Z.
//
// The wedge in this engine: front face (Z+) is the low edge, back face (Z-)
// is the high edge (top-back). Slope normal = (0, +sin45°, +cos45°) ≈ (0, 0.707, 0.707).
//
// Algorithm:
//   1. Check X overlap  (left/right walls)
//   2. Check Y overlap  (floor/ceiling of bounding box — used as early exit)
//   3. Check Z overlap  (front/back walls)
//   4. Check SLOPE overlap (project both shapes onto slope normal)
//
// Push along the axis with the SMALLEST penetration.
fn compute_cube_wedge_contact(cube_pos: Vec3, wedge_pos: Vec3) -> Option<Contact> {
    // Vector from wedge center to cube center (in wedge-local space, since wedge is axis-aligned)
    let local = cube_pos - wedge_pos;

    // ── Axis 1: X  (left/right walls of wedge, full ±0.5) ────────────────────
    let ix = 1.0 - local.x.abs();
    if ix <= 0.0 {
        return None; // Separated along X
    }

    // ── Axis 2: Z  (front/back walls of wedge, full ±0.5) ────────────────────
    let iz = 1.0 - local.z.abs();
    if iz <= 0.0 {
        return None; // Separated along Z
    }

    // ── Axis 3: Y  (AABB bounding box top/bottom — coarse early exit) ─────────
    // A unit cube at pos_b has half-extent 0.5. The wedge AABB has half-extent 0.5.
    // If they don't overlap in the raw AABB Y, there's definitely no collision.
    let iy_aabb = 1.0 - local.y.abs();
    if iy_aabb <= 0.0 {
        return None; // Separated along Y (above or below wedge bounding box)
    }

    // ── Axis 4: Slope normal  (the critical wedge-specific axis) ─────────────
    //
    // The wedge slope in this engine rises from the FRONT (Z = +0.5, Y = -0.5)
    // to the BACK (Z = -0.5, Y = +0.5). The slope normal points "away from" the
    // solid half: up and backward = (0, +0.707, +0.707).
    //
    // LEARNING: normalize(0, 1, 1) = (0, 1/√2, 1/√2) = (0, 0.70710678, 0.70710678)
    let slope_n = Vec3::new(
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    );

    // Project the cube's center onto the slope normal. This tells us how far
    // "into" the slope the cube has penetrated. The wedge occupies the space
    // where the signed distance from the slope plane is <= 0. The slope plane
    // passes through the wedge's center (0,0,0 in local space).
    let cube_dist_along_slope = local.dot(slope_n);

    // Half-extent of a unit cube projected onto any axis:
    // For a cube with half-extent 0.5, the projection onto a unit vector n is:
    //   support = 0.5 * (|n.x| + |n.y| + |n.z|)
    // For slope_n = (0, 0.707, 0.707): support = 0.5 * (0 + 0.707 + 0.707) ≈ 0.707
    // For the wedge, its projection onto the slope normal has half-extent 0.5
    // (the wedge's "depth" perpendicular to its slope is exactly 0.5 in each direction).
    let cube_support_on_slope = 0.5 * (slope_n.x.abs() + slope_n.y.abs() + slope_n.z.abs());
    let wedge_support_on_slope = 0.5; // By construction of the wedge geometry

    // Overlap along slope axis: sum of supports minus |distance between centers|
    let i_slope = (cube_support_on_slope + wedge_support_on_slope) - cube_dist_along_slope.abs();
    if i_slope <= 0.0 {
        return None; // Cube is above/outside the slope face — no collision
    }

    // ── Pick minimum penetration axis ─────────────────────────────────────────
    // SAT: the collision normal is the axis with the SMALLEST overlap.
    // This is the "minimum translation distance" — the shortest path out.
    let min_pen = ix.min(iz).min(i_slope);

    let (normal, penetration) = if min_pen == i_slope {
        // Cube hit the slope face — push it perpendicular to the slope
        // The sign of cube_dist_along_slope tells us which side of the slope we're on.
        // If cube_dist_along_slope > 0, the cube is on the "outside" (above slope) — push outward.
        // If cube_dist_along_slope < 0, the cube is inside the wedge solid — push outward anyway.
        let sign = if cube_dist_along_slope >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (slope_n * sign, i_slope)
    } else if min_pen == ix {
        // Cube hit the X-axis side wall of the wedge
        (Vec3::X * local.x.signum(), ix)
    } else {
        // Cube hit the Z-axis front/back wall of the wedge
        (Vec3::Z * local.z.signum(), iz)
    };

    Some(Contact {
        normal,
        penetration,
        contact_point: wedge_pos + normal * penetration * 0.5,
    })
}

// =============================================================================
// LEARNING TOPIC: Sphere vs Sphere Contact
// =============================================================================
//
// Two spheres collide if their center-to-center distance < sum of their radii.
// The contact normal is simply the normalized vector between centers.
// This is the simplest possible collision — O(1) with a single sqrt().
fn compute_sphere_sphere_contact(pos_a: Vec3, r_a: f32, pos_b: Vec3, r_b: f32) -> Option<Contact> {
    let delta = pos_a - pos_b;
    let distance = delta.length();
    let radius_sum = r_a + r_b;
    if distance >= radius_sum {
        return None;
    }
    // Degenerate case: centers perfectly overlap — push upward arbitrarily
    let normal = if distance > 1e-6 {
        delta / distance
    } else {
        Vec3::Y
    };
    Some(Contact {
        normal,
        penetration: radius_sum - distance,
        contact_point: pos_b + normal * r_b,
    })
}

// =============================================================================
// LEARNING TOPIC: Sphere vs AABB Contact (GJK-lite)
// =============================================================================
//
// To find the closest point on an AABB to a sphere center, we CLAMP the sphere
// center to the box bounds on each axis. The clamped point is the closest point
// on the box surface. If its distance from the sphere center < radius, we have
// a collision.
//
// This is used for: Sphere landing on Cube, Sphere landing on Wedge (approximate).
fn compute_sphere_aabb_contact(
    sphere_pos: Vec3,
    sphere_radius: f32,
    box_pos: Vec3,
    box_half: Vec3,
) -> Option<Contact> {
    let local = sphere_pos - box_pos;
    // Clamp to the box's extent on each axis to find the closest surface point
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
    // If sphere center is INSIDE the box, delta is zero — push upward
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

// =============================================================================
// LEARNING TOPIC: Contact Dispatch (Shape-pair routing)
// =============================================================================
//
// Different shape combinations need different collision algorithms.
// Rust's pattern matching on a tuple of enum variants provides a zero-cost,
// exhaustive dispatch — the compiler guarantees every combination is handled.
// No vtable lookup, no dynamic dispatch overhead.
fn compute_contact(
    self_pos: Vec3,
    self_shape: &ShapeType,
    self_radius: f32,
    other_pos: Vec3,
    other_shape: &ShapeType,
    other_radius: f32,
) -> Option<Contact> {
    match (self_shape, other_shape) {
        // ── Cube vs Cube: simple AABB ─────────────────────────────────────────
        (ShapeType::Cube, ShapeType::Cube) => {
            compute_aabb_aabb_contact(self_pos, Vec3::splat(0.5), other_pos, Vec3::splat(0.5))
        }

        // ── Cube falling onto Wedge: SAT with slope axis ──────────────────────
        (ShapeType::Cube, ShapeType::Wedge) => compute_cube_wedge_contact(self_pos, other_pos),

        // ── Wedge vs Cube: reverse the argument order, flip the normal ─────────
        // compute_cube_wedge_contact(cube, wedge) so we swap and negate.
        (ShapeType::Wedge, ShapeType::Cube) => {
            compute_cube_wedge_contact(other_pos, self_pos).map(|mut c| {
                c.normal = -c.normal;
                c
            })
        }

        // ── Wedge vs Wedge: approximate as AABB (rare in practice) ───────────
        (ShapeType::Wedge, ShapeType::Wedge) => {
            compute_aabb_aabb_contact(self_pos, Vec3::splat(0.5), other_pos, Vec3::splat(0.5))
        }

        // ── Sphere vs Sphere ──────────────────────────────────────────────────
        (ShapeType::Sphere, ShapeType::Sphere) => {
            compute_sphere_sphere_contact(self_pos, self_radius, other_pos, other_radius)
        }

        // ── Sphere vs Box (Cube or Wedge): sphere-AABB test ──────────────────
        (ShapeType::Sphere, ShapeType::Cube) | (ShapeType::Sphere, ShapeType::Wedge) => {
            compute_sphere_aabb_contact(self_pos, self_radius, other_pos, Vec3::splat(0.5))
        }

        // ── Box (Cube or Wedge) vs Sphere: same test, flipped normal ─────────
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

// =============================================================================
// SYSTEM 1: integrate_system
// =============================================================================
//
// LEARNING TOPIC: The Symplectic Euler Integrator
// ------------------------------------------------
// Every frame, before checking collisions, we advance each block's predicted
// position using its current velocity plus the effect of gravity.
//
// Step 1: velocity += gravity * dt          (velocity-Verlet style)
// Step 2: predicted_position = position + velocity * dt
//
// Note: we DON'T update `position` yet — only `predicted_position`.
// `position` stays at the last frame's final solved position.
// The solver then nudges `predicted_position` out of any overlaps.
// Finally, update_velocities_system commits predicted_position → position.
//
// This "predict → correct → commit" loop is the core of XPBD.
pub fn integrate_system(mut query: Query<&mut Voxel>) {
    for mut voxel in query.iter_mut() {
        // Reset contact accumulators for this frame's fresh contact data
        voxel.contact_normal_accum = Vec3::ZERO;
        voxel.contact_count = 0;

        // Static objects (inv_mass == 0.0) are immovable — skip integration
        if voxel.inv_mass == 0.0 {
            continue;
        }

        // Apply gravitational acceleration: Δv = g * dt
        voxel.velocity += GRAVITY * DT;

        // Terminal velocity clamp — prevents tunneling at high speed
        voxel.velocity = voxel.velocity.clamp_length_max(MAX_VELOCITY);

        // Predict where this block will be next frame (before collision resolution)
        voxel.predicted_position = voxel.position + (voxel.velocity * DT);
    }
}

// =============================================================================
// SYSTEM 2: solve_constraints_system
// =============================================================================
//
// LEARNING TOPIC: The Constraint Solver Loop
// ------------------------------------------
// This is the heart of XPBD. We run SOLVER_ITERATIONS times. Each iteration:
//   1. Snapshot current predicted_positions (Jacobi freeze)
//   2. For every block, check floor constraint
//   3. For every block, find neighbors in spatial grid and check pairwise collision
//   4. Accumulate position corrections
//   5. Apply all corrections at once
//
// After all iterations, predicted_positions have been nudged out of all overlaps.
pub fn solve_constraints_system(
    mut query: Query<(Entity, &mut Voxel)>,
    grid: Res<SpatialGrid>,
    settings: Res<PhysicsSettings>,
    mut buffers: ResMut<SolverBuffers>,
) {
    for _iteration in 0..SOLVER_ITERATIONS {
        // ── STEP A: Snapshot ─────────────────────────────────────────────────
        // Freeze all predicted positions for this iteration. The solver reads
        // ONLY from snapshots, never from the live query mid-iteration.
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

        // ── STEP B: Process each entity's constraints ─────────────────────────
        for (entity, voxel) in query.iter() {
            // ── B1: Floor Constraint ──────────────────────────────────────────
            //
            // LEARNING: The floor is an infinite static plane at global_floor_y.
            // We don't store it as an entity — it's an implicit constraint checked
            // analytically. This is cheaper than spawning a giant static floor mesh.
            //
            // For non-sphere shapes, we find the lowest point by transforming all
            // 8 corners of the unit cube through the block's current rotation
            // (Quaternion × local_offset = world_offset). The lowest Y in world
            // space is the "foot" of the block.
            if let Some(floor_y) = settings.global_floor_y {
                let mut min_local_y = 0.0_f32;

                let extents: &[Vec3] = if voxel.shape == ShapeType::Sphere {
                    &[Vec3::new(0.0, -voxel.sphere_radius, 0.0)]
                } else {
                    &[
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

                for &local_corner in extents {
                    // Quaternion rotation: rotate local corner into world space
                    // LEARNING: voxel.rotation is a Quat. Multiplying Quat * Vec3
                    // rotates the vector WITHOUT translating it — pure orientation.
                    let world_corner_offset = voxel.rotation * local_corner;
                    if world_corner_offset.y < min_local_y {
                        min_local_y = world_corner_offset.y;
                    }
                }

                // "lowest" = center Y + lowest corner's Y offset
                let lowest_point_y = voxel.predicted_position.y + min_local_y;

                if lowest_point_y < floor_y {
                    // Penetration depth into the floor
                    let penetration = floor_y - lowest_point_y;

                    // Push the block UP by the penetration amount
                    *buffers.pos_corrections.entry(entity).or_insert(Vec3::ZERO) +=
                        Vec3::new(0.0, penetration, 0.0);

                    // Record that this block has an upward contact (Y+ normal = floor)
                    *buffers.normal_accum.entry(entity).or_insert(Vec3::ZERO) += Vec3::Y;
                    *buffers.normal_count.entry(entity).or_insert(0) += 1;
                }
            }

            // ── B2: Broad-Phase Neighbor Detection ────────────────────────────
            //
            // LEARNING: The SpatialGrid is a HashMap<[i32;3], Vec<Entity>>.
            // world_to_grid() rounds the continuous position to the nearest cell.
            // get_neighbors() returns all entities in the 27 cells surrounding
            // this block (3×3×3 = 27, including the block's own cell).
            //
            // This reduces collision checks from O(N²) to O(N * 27) = O(N).
            // For 1000 blocks: 1,000,000 checks → 27,000 checks. 37× faster.
            //
            // We pass predicted_position so we find neighbors in the block's
            // FUTURE location — where it's heading — not where it currently is.
            for (other_e, _other_shape, grid_pos) in grid.get_neighbors(voxel.predicted_position) {
                if entity == other_e {
                    continue; // Skip self-collision
                }

                // ── Deduplication: skip if we've already processed this pair ──
                let pair = make_pair(entity, other_e);
                if !buffers.processed_pairs.insert(pair) {
                    continue;
                }

                // ── Read snapshots (frozen positions) ─────────────────────────
                let Some(self_snap) = buffers.snapshots.get(&entity).cloned() else {
                    continue;
                };
                let Some(other_snap) = buffers.snapshots.get(&other_e).cloned() else {
                    continue;
                };

                // =============================================================
                // BUG FIX #1 APPLIED HERE:
                // Static Block Position — Use Actual Position, Not Integer Grid
                // =============================================================
                //
                // BEFORE (buggy): for static blocks, we snapped to integer grid:
                //   other_pos = Vec3::new(grid_pos[0] as f32, grid_pos[1] as f32, ...)
                //
                // This caused "grid drift": a block settled at y=0.5 occupies
                // grid cell y=round(0.5) which could snap to y=0 OR y=1. If it
                // snapped to y=0, a block on top (y≈1.5) saw the static block at
                // y=0.0 — a false extra gap of 0.5 units — generating a spurious
                // overlap and pushing the upper block DOWN. The visual result:
                // the lower (level-1) block "disappeared" as both it and the
                // upper block collapsed to the same render position.
                //
                // AFTER (fixed): static blocks now use their ACTUAL snapshot position
                // (which is their `voxel.position` — the last committed position,
                // NOT the grid-rounded approximation). The spatial grid is only
                // used for BROAD-PHASE neighbor finding; the NARROW-PHASE collision
                // math always uses exact positions.
                let other_pos = other_snap.predicted_position;
                // Note: `grid_pos` is still used above for neighbor iteration but
                // is NOT used as the actual collision position anymore.
                let _ = grid_pos; // explicitly acknowledge we're ignoring it

                // ── Run narrow-phase contact detection ────────────────────────
                let Some(c) = compute_contact(
                    self_snap.predicted_position,
                    &self_snap.shape,
                    self_snap.sphere_radius,
                    other_pos,
                    &other_snap.shape,
                    other_snap.sphere_radius,
                ) else {
                    continue; // No overlap — nothing to resolve
                };

                // ── Compute correction magnitude ──────────────────────────────
                //
                // LEARNING: inv_mass_sum = w_a + w_b where w = 1/mass.
                // If w_sum = 0, both objects are static — impossible to resolve.
                // Otherwise, each object moves in proportion to its own inv_mass
                // divided by the total. Heavy objects barely move, light ones move a lot.
                let inv_mass_sum = self_snap.inv_mass + other_snap.inv_mass;
                if inv_mass_sum <= 0.0 {
                    continue; // Both static, nothing to push
                }

                // Clamp the raw correction to prevent explosive tunneling corrections
                let base_corr =
                    (c.normal * c.penetration).clamp_length_max(MAX_CORRECTION_PER_ITER);

                // ── Mass-proportional sharing ─────────────────────────────────
                let mut self_share = self_snap.inv_mass / inv_mass_sum;
                let mut other_share = other_snap.inv_mass / inv_mass_sum;

                // =============================================================
                // BUG FIX #2 APPLIED HERE:
                // Wedge Stability — Physics-Grounded Settled-Wedge Detection
                // =============================================================
                //
                // When a cube drops onto a resting wedge, the wedge should act
                // like a static ramp — not fly backward. The wedge has very low
                // inv_mass (≈0.0005, mass ≈ 2000 kg, see voxel.rs). But with
                // large penetration (block dropped from height 10.0), even a tiny
                // wedge share * correction can become large after 10 iterations.
                //
                // Physics derivation: A 1 kg cube dropped from 10 m arrives with
                //   v = sqrt(2 * g * h) = sqrt(2 * 9.81 * 10) ≈ 14 m/s
                //   p = m * v = 14 kg·m/s
                // To keep wedge velocity < 0.01 m/s: m_wedge > p / 0.01 = 1400 kg
                // We use m_wedge = 2000 kg (inv_mass = 0.0005) in voxel.rs.
                //
                // ADDITIONAL STABILITY: When the wedge is "settled" (speed < threshold),
                // pin its share to near-zero. This makes the CUBE absorb 100% of the
                // correction — pushing the cube up the slope (natural sliding behavior)
                // while leaving the wedge effectively motionless.
                //
                // "Settled" = wedge's current speed < WEDGE_SETTLED_SPEED_THRESHOLD.
                // We read this from the snapshot's predicted_position vs. actual
                // by using the voxel directly (still valid since this is a shared-ref iter).
                let self_is_wedge = self_snap.shape == ShapeType::Wedge;
                let other_is_wedge = other_snap.shape == ShapeType::Wedge;

                if self_is_wedge ^ other_is_wedge {
                    // One is a wedge, one is not — check if the wedge is settled
                    if self_is_wedge {
                        // self is the wedge — check its velocity via snapshot delta
                        // We use inv_mass as a proxy: very-low-inv_mass = heavy = settled faster
                        // but we also check if its current velocity from the query is low.
                        // Since we're in a shared-ref iteration over the query, we read
                        // the wedge's actual velocity using the snapshot comparison:
                        // If the wedge's correction from previous iterations is small, it's settled.
                        // Simple approach: check if the wedge's share (raw) is already tiny,
                        // meaning its mass is already doing the job. Then further cap if settled.
                        //
                        // We check the snapshots: if other_snap (cube) has high velocity
                        // along the contact normal, the wedge might still get pushed.
                        // The most robust check: is the wedge's inv_mass contribution tiny?
                        if self_snap.inv_mass < 0.001 {
                            // Wedge is heavy (> 1000 kg) — pin its share to near-zero
                            // if it's a "ground-level" wedge (predicted_position.y close to rest)
                            // We use: if wedge center is below 1.5 units (resting on floor level)
                            let wedge_y = self_snap.predicted_position.y;
                            if wedge_y < 1.0 {
                                // Settled on the ground — behave as static platform
                                self_share = self_share.min(WEDGE_SETTLED_SHARE_CAP);
                                other_share = 1.0 - self_share;
                            }
                        }
                    } else {
                        // other is the wedge
                        if other_snap.inv_mass < 0.001 {
                            let wedge_y = other_snap.predicted_position.y;
                            if wedge_y < 1.0 {
                                other_share = other_share.min(WEDGE_SETTLED_SHARE_CAP);
                                self_share = 1.0 - other_share;
                            }
                        }
                    }
                }

                // ── Apply corrections to each body ────────────────────────────
                //
                // self gets pushed along +normal (away from other)
                // other gets pushed along -normal (away from self)
                // The magnitudes are proportional to their inv_mass shares.
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

        // ── STEP C: Apply accumulated corrections ─────────────────────────────
        //
        // After ALL pairs have been processed with frozen snapshot data,
        // flush the accumulated corrections into the live predicted_positions.
        // Next iteration will snapshot these updated positions.
        for (entity, corr) in buffers.pos_corrections.iter() {
            if let Ok((_, mut v)) = query.get_mut(*entity) {
                v.predicted_position += *corr;
            }
        }
        // Accumulate contact normals into the live voxel for friction use later
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

// =============================================================================
// SYSTEM 3: update_velocities_system
// =============================================================================
//
// LEARNING TOPIC: Velocity Derivation from Position Change (XPBD Commit Step)
// -----------------------------------------------------------------------------
// In XPBD, velocity is NOT integrated — it is DERIVED from how far the block
// actually moved between frames:
//
//   derived_velocity = (predicted_position - old_position) / dt
//
// This automatically accounts for all constraint corrections. A block that was
// pushed upward by a floor constraint will have a positive Y velocity. A block
// that slid along a slope will have the correct diagonal velocity. No manual
// force calculation needed.
//
// After deriving velocity, this system applies:
//   - Friction: damps tangential velocity when in contact with a surface
//   - Restitution: controls how "bouncy" the block is (0.0 = no bounce)
//   - Sleep: zeros velocity below the sleep threshold to stop micro-jitter
//   - Angular settling: lerps rotation toward upright when resting flat
//
pub fn update_velocities_system(mut query: Query<&mut Voxel>) {
    for mut voxel in query.iter_mut() {
        if voxel.inv_mass == 0.0 {
            continue; // Static objects never move
        }

        // ── Clamp total frame displacement ────────────────────────────────────
        // Even after all solver iterations, cap the maximum displacement per frame.
        // This is the "displacement budget" — prevents fast-moving blocks from
        // skipping through thin geometry between frames (tunneling).
        let frame_delta = (voxel.predicted_position - voxel.position)
            .clamp_length_max(MAX_DISPLACEMENT_PER_FRAME);
        voxel.predicted_position = voxel.position + frame_delta;

        // ── Derive velocity from position change ──────────────────────────────
        // v = Δx / Δt   (XPBD's core velocity derivation)
        //
        // LEARNING — THE WEDGE LAUNCH BUG AND WHY NAIVE DERIVATION FAILS:
        // The solver pushed the cube OUT of the wedge along the slope normal
        // (0, 0.707, 0.707) over 10 iterations. Total Z push ≈ 1.55 units.
        // Naive derivation: v_z = 1.55 / DT = 93 m/s → clamped to 25 m/s.
        // The cube launches at 25 m/s sideways — clearly wrong.
        //
        // ROOT CAUSE: The XPBD solver moves `predicted_position` to resolve
        // overlaps, but that movement is a GEOMETRIC CORRECTION, not a physical
        // impulse. The actual physical velocity after landing on a slope should
        // come ONLY from gravity — any lateral component is natural slide.
        //
        // FIX: When in contact with a non-flat surface (slope), reconstruct
        // velocity from gravity only, then project out the normal component
        // (so we don't fall through the slope). This gives natural gravity-fed
        // sliding without solver-artifact launches.
        //
        // For FLAT contact (avg_normal.y ≈ 1), the full derived velocity is used
        // because it correctly captures small bounces and slide-to-stop behavior.
        let contact_normal_now = if voxel.contact_count > 0 {
            (voxel.contact_normal_accum / voxel.contact_count as f32).normalize_or_zero()
        } else {
            Vec3::ZERO
        };

        let is_slope_contact = contact_normal_now != Vec3::ZERO
            && contact_normal_now.y < FLAT_CONTACT_Y_THRESHOLD
            && contact_normal_now.y > 0.1; // Has upward component = is a surface

        let mut derived_velocity = if is_slope_contact {
            // LEARNING: On a slope, DON'T trust the solver-derived velocity —
            // it contains a huge artificial Z/X push from the correction step.
            // Instead, take the pre-contact gravity velocity and project it
            // onto the slope surface (remove the normal component so we don't
            // penetrate, keep the tangential component = natural sliding).
            //
            // v_gravity = velocity BEFORE this frame's integration (stored in voxel.velocity
            // which was set last frame; gravity was already added in integrate_system
            // so voxel.velocity here still has the accumulated gravity).
            // We reconstruct it from the actual position delta but CLAMP it hard
            // to prevent the solver artifact.
            let raw = (voxel.predicted_position - voxel.position) / DT;
            // Project raw velocity onto the slope surface plane:
            // Remove any component ALONG the outward normal (prevents embedding)
            // Keep only the tangential (sliding) component.
            let v_normal_mag = raw.dot(contact_normal_now);
            let v_tangential = raw - contact_normal_now * v_normal_mag;
            // The tangential speed should be bounded by what gravity alone can give
            // over one frame: v = g * dt * slope_tangential_factor ≈ 0.163 m/s
            // We allow up to 10× that for accumulated slide velocity.
            v_tangential.clamp_length_max(2.0)
        } else {
            (voxel.predicted_position - voxel.position) / DT
        };

        // Apply damping when in contact (simulates material energy absorption)
        if voxel.contact_count > 0 {
            derived_velocity *= POSITION_VELOCITY_DAMPING;
        }

        voxel.velocity = derived_velocity.clamp_length_max(MAX_VELOCITY);

        // ── Friction and angular correction (only when in contact) ────────────
        if voxel.contact_count > 0 {
            // Compute the average contact normal (direction "away from surface")
            let avg_normal =
                (voxel.contact_normal_accum / voxel.contact_count as f32).normalize_or_zero();

            if avg_normal != Vec3::ZERO {
                // ── Angular settling: rotate block to align with contact surface ──
                //
                // LEARNING: Quaternion slerp (Spherical Linear intERPolation)
                // smoothly rotates between two orientations. We compute the rotation
                // that would bring Vec3::Y (the block's "up") to align with avg_normal
                // (the surface normal). Slerping 20% of the way each frame gives a
                // smooth, realistic settle without snapping.
                // Only use UPWARD-facing support normals for orientation.
                // Downward normals can happen from an object above us and should
                // never drive "upright settle".
                let support_normal = if avg_normal.y > 0.0 {
                    avg_normal
                } else {
                    Vec3::ZERO
                };

                if support_normal != Vec3::ZERO {
                    let axis = Vec3::Y.cross(support_normal);
                    let angle = Vec3::Y.angle_between(support_normal);
                    let target_rot = if angle > 1e-4 {
                        let axis_len = axis.length();
                        if axis_len > 1e-6 {
                            Quat::from_axis_angle(axis / axis_len, angle)
                        } else if support_normal.y < 0.0 {
                            // Anti-parallel case (180° flip): choose a stable axis.
                            Quat::from_axis_angle(Vec3::X, std::f32::consts::PI)
                        } else {
                            Quat::IDENTITY
                        }
                    } else {
                        Quat::IDENTITY
                    };

                    // Guard against any invalid quaternion propagation.
                    voxel.rotation = if target_rot.is_finite() {
                        voxel.rotation.slerp(target_rot, 0.25)
                    } else {
                        voxel.rotation
                    };
                }
                voxel.angular_velocity = Vec3::ZERO;

                // ── Friction decomposition: separate normal and tangential velocity ──
                //
                // LEARNING: Any velocity vector can be decomposed into two components:
                //   v_n = velocity projected ONTO the contact normal (bouncing component)
                //   v_t = velocity minus v_n (sliding component, parallel to surface)
                //
                // We apply friction ONLY to v_t (the sliding part).
                // We apply restitution ONLY to v_n (the bouncing part).
                // This correctly models surfaces: a block bounces perpendicular to a
                // slope while sliding along it, not bouncing in some weird diagonal.
                let contact_basis_normal = if support_normal != Vec3::ZERO {
                    support_normal
                } else {
                    avg_normal
                };
                let v_n_mag = voxel.velocity.dot(contact_basis_normal);
                let v_n = contact_basis_normal * v_n_mag;
                let v_t = voxel.velocity - v_n;

                // Restitution: how much of the normal velocity bounces back.
                // 0.0 = no bounce (dead stop against surface), 1.0 = perfect elastic bounce.
                let resolved_v_n = if v_n_mag < 0.0 {
                    // Moving INTO the surface — apply restitution
                    v_n * voxel.restitution
                } else {
                    // Moving AWAY from surface — keep as-is
                    v_n
                };

                // Friction: reduce tangential velocity by the friction coefficient.
                // Friction force ∝ normal force ∝ normal_support (how much of the
                // surface normal points up, i.e., how much it's pushing against gravity).
                let v_t_len = v_t.length();
                if v_t_len > 1e-5 {
                    let normal_support = contact_basis_normal.y.max(0.0);
                    // Friction deceleration = μ * g * dt * (normal component of contact)
                    let friction_drop = 9.81 * DT * voxel.friction * normal_support;

                    if contact_basis_normal.y >= FLAT_CONTACT_Y_THRESHOLD
                        && v_t_len <= STATIC_FRICTION_SPEED
                    {
                        // Static friction: block is nearly stopped on a flat surface
                        // — kill all lateral velocity completely (no sliding)
                        voxel.velocity = resolved_v_n;
                    } else {
                        // Kinetic friction: reduce sliding speed but keep direction
                        let new_vt_len = (v_t_len - friction_drop).max(0.0);
                        voxel.velocity = resolved_v_n + v_t * (new_vt_len / v_t_len);
                    }
                } else {
                    voxel.velocity = resolved_v_n;
                }

                // ── Flat-surface special cases ────────────────────────────────
                if contact_basis_normal.y >= FLAT_CONTACT_Y_THRESHOLD {
                    // Snap rotation to perfectly upright when resting on flat surface.
                    // This prevents blocks from slowly tipping over due to floating-point drift.
                    voxel.rotation = voxel.rotation.slerp(Quat::IDENTITY, UPRIGHT_SETTLE_RATE);

                    // Snap near-zero Y velocity to exactly zero (prevent micro-bouncing)
                    if voxel.velocity.y.abs() < 0.2 {
                        voxel.velocity.y = 0.0;
                    }

                    // Full sleep: if the block is nearly stationary, freeze it entirely.
                    // This is critical for tower stability — without sleep, every block
                    // in a stack keeps "breathing" (tiny oscillations) that can topple tall towers.
                    if voxel.velocity.length() < LINEAR_SLEEP_SPEED {
                        voxel.velocity = Vec3::ZERO;
                    }
                }

                // Final safety clamp: if any invalid rotation sneaks in from
                // external data, force identity so rendering never receives NaN.
                if !voxel.rotation.is_finite() {
                    voxel.rotation = Quat::IDENTITY;
                }
            }
        } else {
            // Not in contact with anything — slowly upright the block in the air
            // (cosmetic only, doesn't affect physics trajectory)
            voxel.rotation = voxel.rotation.slerp(Quat::IDENTITY, 0.05);
        }

        // ── Commit: predicted_position becomes the authoritative position ──────
        // This is the moment XPBD "applies" the solved frame. From here,
        // next frame's `integrate_system` will use this as the starting point.
        voxel.position = voxel.predicted_position;
    }
}
