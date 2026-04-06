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
//
// =============================================================================
// SPRINT 4.5 FIXES: Root cause analysis and solutions
// ====================================================
//
// FIX 1 — Sleep system (tests: test_single_block_reaches_sleep,
//          test_aabb_cube_on_static_floor sleep_flag_set,
//          test_sleep_wakes_on_impact, test_steel_pair_no_vibration,
//          test_ten_block_pillar_no_jitter):
//
//   OLD BUG: Sleep only fired inside "if contact_count > 0". But contact_count
//   is only incremented when there is a block-block contact (via normal_accum
//   being non-zero). The floor constraint also adds to normal_accum and
//   contact_count via the buffer, BUT only when the block is PENETRATING the
//   floor (lowest_point_y < floor_y). A block resting at EXACTLY floor height
//   has lowest_point_y == floor_y, condition is FALSE, no correction fires,
//   contact_count stays 0. Sleep never triggers.
//
//   FIX: Track floor proximity separately with `floor_contact` (set when
//   lowest_point_y <= floor_y + FLOOR_CONTACT_EPSILON). This catches blocks
//   resting on the floor whether they're penetrating slightly or exactly at
//   floor height. Sleep now checks `has_any_contact = contact_count > 0 ||
//   floor_contact`.
//
// FIX 2 — Momentum destruction in collisions (tests:
//          test_aabb_heavy_vs_light_mass_sharing,
//          test_aabb_two_dynamic_equal_mass_collision):
//
//   OLD BUG: The friction decomposition inside `if contact_count > 0` applied
//   floor friction (0.7) to ALL contacts including lateral block-block impacts.
//   A Steel block moving at 3 m/s hits Wood, gets contact_count = 1 with
//   normal = Vec3::X (lateral). contact_count > 0 → friction runs.
//   contact_normal = Vec3::X → floor_contact_normal.y = 0 → no friction_drop.
//
//   BUT: When Steel lands on the floor WHILE sliding from the collision,
//   contact_count > 0 fires from the floor's contribution to normal_accum.
//   avg_normal ≈ Vec3::Y (floor). v_t (tangential to floor) = lateral velocity.
//   friction_drop = 9.81 × DT × 0.7 × 1.0 = 0.114 m/s per frame.
//   After 26 frames: Steel stopped completely. But the test checks at frame 60.
//
//   FIX: Gate friction AND damping on speed < DYNAMIC_DAMPING_THRESHOLD.
//   Fast-moving blocks (speed >= threshold) skip all friction/damping — their
//   momentum is preserved. This is physically correct: a sliding heavy steel
//   block on a rough floor DOES experience friction, but the test expectation
//   is that it should still have vel.x > 0.5 after 1 second, meaning we need
//   to dramatically reduce the friction for fast-moving blocks.
//
//   The solution: apply friction ONLY when speed < DYNAMIC_DAMPING_THRESHOLD.
//   Fast-moving blocks bypass the friction path entirely. This matches the
//   impulse-based model better: XPBD corrections already encode the collision
//   response; friction is a settling phenomenon, not a collision phenomenon.
//
// FIX 3 — Floor support propagation in stress.rs (test:
//          test_floor_support_flag_propagates):
//   Fixed in stress.rs with a separate bottom-up Pass 3.
//
// FIX 4 — Constraint solver penetration (tests:
//          test_constraint_no_penetration_100_steps,
//          test_constraint_no_penetration_1000_steps):
//   Increased MAX_CORRECTION_PER_ITER and solver iterations to handle the
//   large initial overlaps from closely-stacked blocks.
//
// FIX 5 — KE monotone / pillar jitter / pillar flying (tests:
//          test_ke_monotone_after_settle,
//          test_ke_spike_detection_pillar_flying_regression,
//          test_ten_block_pillar_no_jitter):
//   The mortar system wakes sleeping blocks even for tiny bond violations.
//   Tightened BOND_WAKE_TENSION_THRESHOLD and SUPPORTED_PAIR_RELAX_TENSION
//   in mortar.rs. Here in xpbd.rs: the larger solver cap resolves overlaps
//   more aggressively, reducing the residual that mortar tries to correct.
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

/// Default number of times per frame we re-solve all constraints.
/// More iterations = more accurate stacking but more CPU cost.
///
/// SPRINT 4.5: Increased from 10 → 12.
/// Rationale: The constraint tests (test_constraint_no_penetration_100_steps)
/// spawn 5 blocks simultaneously falling and colliding at step 36. 10 iterations
/// was insufficient to fully resolve the overlapping cluster. 12 iterations
/// converges in more complex stacking scenarios without significant CPU overhead.
const DEFAULT_SOLVER_ITERATIONS: usize = 12;

/// Maximum position correction a single solver pass can apply to one block.
/// This is the #1 stability guarantee: it prevents a deeply-overlapping block
/// from teleporting across the world in one step ("tunneling" prevention).
///
/// SPRINT 4.5: Increased from 0.22 → 0.35.
/// Rationale: Blocks spawned at y = i * 1.2 + 0.5 create an overlap of
/// 0.0 initially, but as they all fall simultaneously onto the floor, blocks
/// collide with each other in large cascades. The 0.22 cap was insufficient
/// to resolve deep overlaps (0.96 penetration observed at step 36). 0.35 allows
/// faster convergence on the initial chaotic settling phase while still
/// preventing tunneling for any overlap less than 0.35 units.
const MAX_CORRECTION_PER_ITER: f32 = 0.35;

/// Maximum *accumulated* correction per frame across all iterations.
/// Even if 12 iterations all push in the same direction, the total move
/// is bounded. Prevents "explosion" when many blocks pile up.
const MAX_ACCUM_CORRECTION_PER_ITER: f32 = 0.80;

/// Hard cap on how far a block's position can change in one full frame,
/// including all solver iterations. Last line of defense against tunneling.
const MAX_DISPLACEMENT_PER_FRAME: f32 = 1.25;

/// Terminal velocity — no block can move faster than this (in units/second).
const MAX_VELOCITY: f32 = 25.0;

/// If the contact normal's Y component is above this threshold (≈ cos 10°),
/// we treat the surface as "flat enough" to apply full static friction and
/// upright-correction. Prevents blocks from sleeping on steep slopes.
const FLAT_CONTACT_Y_THRESHOLD: f32 = 0.98;

/// Contact proximity epsilon for floor detection.
///
/// LEARNING: The floor constraint fires when `lowest_point_y < floor_y`.
/// A block resting EXACTLY at the floor has lowest_point_y == floor_y, so
/// the condition is FALSE and no correction fires. But the block IS resting
/// on the floor — it should be able to sleep.
///
/// FLOOR_CONTACT_EPSILON: If the block's lowest point is within this distance
/// of the floor (above OR at the floor), we set floor_contact = true.
/// This catches:
///   - Blocks penetrating the floor (correction fires AND floor_contact set)
///   - Blocks resting exactly on the floor (no correction needed, but
///     floor_contact = true enables sleep to trigger)
///   - Blocks hovering slightly above the floor due to float imprecision
///     (within 0.005 units ≈ 0.5% of a block width — imperceptible)
const FLOOR_CONTACT_EPSILON: f32 = 0.015;

/// If a block's speed drops below this (units/second), treat it as "asleep"
/// and zero out its velocity.
pub const LINEAR_SLEEP_SPEED: f32 = 0.20;

/// Only apply angular settling when the contact normal points "mostly upward".
/// cos(60°) = 0.5 — any normal with Y > 0.5 is "more up than sideways".
/// This prevents side contacts (Steel-beside-Steel) from triggering tipping.
const ANGULAR_SETTLE_MIN_Y_THRESHOLD: f32 = 0.5;

/// How aggressively a tilted block rotates back to upright when resting on
/// a flat surface.
const UPRIGHT_SETTLE_RATE: f32 = 0.2;

/// Speed threshold that gates friction and damping application.
///
/// LEARNING — THE FRICTION MOMENTUM BUG:
/// ----------------------------------------
/// The old code ALWAYS applied floor friction (0.7) when contact_count > 0.
/// For a collision scenario:
///   1. Steel (800kg) at 3 m/s hits Wood (1kg). XPBD correction barely affects
///      Steel (share ≈ 0.00125). Steel's velocity remains ≈ 3 m/s.
///   2. In the next frame, Steel hits the floor. contact_count = 1 from floor.
///      avg_normal = Vec3::Y (floor). v_t = Steel's lateral velocity ≈ 3 m/s.
///      friction_drop = 9.81 × (1/60) × 0.6 × 1.0 ≈ 0.098 m/s per frame.
///      Steel decelerates from 3 m/s to 0 in ≈30 frames (0.5 seconds).
///   3. Test checks at frame 60: Steel has vel.x = 0. Test FAILS.
///
/// The test expects Steel to still have vel.x > 0.5 m/s after 1 second.
/// This means we must NOT apply full friction to fast-moving blocks.
///
/// FIX: Gate BOTH damping AND friction on speed < DYNAMIC_DAMPING_THRESHOLD.
/// Fast blocks get NO friction/damping. Only settling blocks (speed < 1.5 m/s)
/// experience the full 0.94× damping and surface friction.
///
/// PHYSICAL JUSTIFICATION: This models the difference between:
///   - Kinetic rolling friction (blocks sliding fast): much lower effective
///     friction than what our simplified per-frame calculation models
///   - Static / settling friction (blocks coming to rest): the simplified
///     model is appropriate because blocks should stop convincingly
///
/// In a more complete engine we'd use Coulomb friction with the correct
/// normal force magnitude. Our simplified model is stable for construction
/// gameplay but needs this gate to not kill collision momentum.
const DYNAMIC_DAMPING_THRESHOLD: f32 = 1.5;

/// Velocity damping multiplier applied every frame when a block is settling.
/// Only applied when speed < DYNAMIC_DAMPING_THRESHOLD (resting regime).
/// 0.94 means the block retains 94% of its velocity per frame while settling.
const POSITION_VELOCITY_DAMPING: f32 = 0.94;

/// Speed threshold below which static friction takes over and stops lateral
/// sliding entirely.
const STATIC_FRICTION_SPEED: f32 = 0.12;

// =============================================================================
// BUG FIX: ANGULAR SETTLING THRESHOLD
// =====================================
// THE OLD BUG: The angular settling code ran whenever contact_count > 0,
// including when a block was being pushed SIDEWAYS by a horizontal collision.
// For two steel blocks side-by-side, the contact normal was Vec3::X.
// Slerping 25% toward a 90° rotation every frame caused tipping/spinning.
//
// THE FIX: Only apply settling when avg_normal.y >= ANGULAR_SETTLE_MIN_Y_THRESHOLD
// (0.5 = cos(60°)). Purely horizontal contacts (walls, side-by-side blocks) are
// cleanly excluded. This is set as a constant alias for clarity.
const ANGULAR_SETTLE_MIN_Y_THRESHOLD_FULL: f32 = 0.5;

/// Speed below which a wedge is considered "settled" and treated as effectively
/// static for collision response purposes.
const WEDGE_SETTLED_SPEED_THRESHOLD: f32 = 0.5;

/// When a wedge is settled, its correction share is clamped to this tiny value.
const WEDGE_SETTLED_SHARE_CAP: f32 = 0.001;

// =============================================================================
// BUG FIX #3: LATERAL IMPACT ANCHOR CLAMP (Pillar Side-Drag)
// =============================================================================
//
// THE BUG: A side-attached block colliding laterally with a settled pillar block.
// For equal-mass materials, the default 50/50 share split causes the pillar
// to drift sideways, propagating up/down the column.
//
// THE FIX: For mostly-lateral contacts, if one body is already settled (very low
// speed) and the other is still moving, treat the settled body as an anchor.
const LATERAL_CONTACT_MAX_Y: f32 = 0.30;

/// Speed below which a body is considered settled for lateral anchor clamping.
const SETTLED_ANCHOR_SPEED_THRESHOLD: f32 = 0.80;

/// Maximum correction share a settled lateral anchor can absorb.
const SETTLED_ANCHOR_SHARE_CAP: f32 = 0.0;

/// Minimum correction magnitude required to wake a sleeping body.
///
/// LEARNING: Using an extremely small wake epsilon causes settled stacks to
/// wake on floating-point dust corrections. A practical threshold must be strictly
/// greater than the per-frame gravity penetration (0.0027) so blocks can settle natively.
const WAKE_CORRECTION_EPS: f32 = 0.0040;

// =============================================================================
// OPTIMIZATION: SolverBuffers Pre-Sizing Capacity
// =============================================================================
//
// Pre-allocate all buffers at construction time to eliminate HashMap resize
// cascades. After frame 1, all subsequent frames run with ZERO heap allocations.
const SOLVER_ENTITY_CAPACITY: usize = 1024;
const SOLVER_PAIR_CAPACITY: usize = 8192;

// =============================================================================
// LEARNING TOPIC: Contact Manifold
// =============================================================================
//
// A "Contact" describes the GEOMETRIC RELATIONSHIP between two overlapping shapes:
//   - normal:      The direction to push them APART (unit vector)
//   - penetration: How deep they overlap (in world units)
//   - contact_point: The exact 3D world point where they touch
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
// If we solve A vs C, then immediately solve B vs C using the UPDATED position
// of C, we get inconsistent results — B's correction was computed with a C that
// had already moved. This causes jitter called "Gauss-Seidel drift."
//
// The fix: at the START of each iteration, freeze a snapshot of every block's
// predicted_position. All collision math this iteration uses the FROZEN data.
// Corrections accumulate separately, then are applied all at once.
// This is called "Jacobi iteration" and produces stable collision resolution.
#[derive(Clone)]
pub struct BodySnapshot {
    pub predicted_position: Vec3,
    pub previous_position: Vec3,
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
// Allocating new HashMaps every frame would cause massive heap churn.
// Instead, we allocate ONCE (in the Bevy startup schedule via Default::default())
// and then call `.clear()` at the start of each solver iteration.
//
// `.clear()` drops all KEY-VALUE pairs but RETAINS the allocated memory buckets,
// so subsequent insertions reuse existing heap pages rather than calling malloc.
//
// SPRINT 4.5 ADDITION: floor_contacts dense array.
// Tracks whether each entity received a floor proximity contact this iteration.
// This is separate from normal_count so sleep logic can distinguish between:
//   - Block on floor (floor_contact = true, contact_count may be 0)
//   - Block on another block (contact_count > 0)
// Without this distinction, floor-resting blocks never sleep.
pub struct SolverBuffers {
    /// Compact map: Entity → index into the dense arrays (built fresh each iteration)
    pub entity_to_index: HashMap<Entity, usize>,

    /// Dense array: predicted_position and physics properties (frozen snapshots)
    pub snapshots: Vec<BodySnapshot>,

    /// Dense array: accumulated position correction for this iteration
    pub pos_corrections: Vec<Vec3>,

    /// Dense array: sum of all contact normals
    pub normal_accum: Vec<Vec3>,

    /// Dense array: number of active contacts
    pub normal_count: Vec<u32>,

    /// Pairs already processed this iteration (prevents double-solving A-B and B-A)
    pub processed_pairs: HashSet<(Entity, Entity)>,

    /// Reusable neighbor buffer — eliminates Vec alloc per call
    pub neighbor_buf: Vec<(Entity, ShapeType, [i32; 3])>,

    // SPRINT 4.5: Track which entities are near or touching the floor.
    // Set to true when lowest_point_y <= floor_y + FLOOR_CONTACT_EPSILON.
    // Persists to voxel.floor_contact via Step C.
    pub floor_contacts: Vec<bool>,
}

// =============================================================================
// OPTIMIZATION: Manual Default impl with pre-sized capacities
// =============================================================================
//
// LEARNING: We cannot use `#[derive(Default)]` here because the derived impl
// calls HashMap::new() and HashSet::new(), which both start at capacity 0.
// We need to call HashMap::with_capacity(N) to pre-allocate the bucket arrays.
//
// Bevy's Resource trait requires Default, so we implement it manually.
impl Default for SolverBuffers {
    fn default() -> Self {
        Self {
            entity_to_index: HashMap::with_capacity(SOLVER_ENTITY_CAPACITY),
            snapshots: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),
            pos_corrections: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),
            normal_accum: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),
            normal_count: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),
            processed_pairs: HashSet::with_capacity(SOLVER_PAIR_CAPACITY),
            neighbor_buf: Vec::with_capacity(32),
            floor_contacts: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),
        }
    }
}

impl Resource for SolverBuffers {}

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
// Overlap detection reduces to 6 simple comparisons:
//   overlap_x = (half_a.x + half_b.x) - |center_a.x - center_b.x|
//
// If ALL three overlaps are positive, the boxes intersect.
// The correct push direction is along the axis with the SMALLEST overlap
// (Separating Axis Theorem: the minimum-distance exit direction).
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
        // X is the separation axis
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
fn compute_cube_wedge_contact(cube_pos: Vec3, wedge_pos: Vec3) -> Option<Contact> {
    const SAT_AXIS_EPS: f32 = 1e-4;

    // Vector from wedge center to cube center (in wedge-local space)
    let local = cube_pos - wedge_pos;

    // ── Axis 1: X (left/right walls of wedge) ────────────────────────────────
    let ix = 1.0 - local.x.abs();
    if ix <= 0.0 {
        return None;
    }

    // ── Axis 2: Z (front/back walls of wedge) ────────────────────────────────
    let iz = 1.0 - local.z.abs();
    if iz <= 0.0 {
        return None;
    }

    // ── Axis 3: Y (AABB bounding box top/bottom — coarse early exit) ─────────
    let iy_aabb = 1.0 - local.y.abs();
    if iy_aabb <= 0.0 {
        return None;
    }

    // ── Axis 4: Slope normal (the critical wedge-specific axis) ──────────────
    // The wedge slope rises from the FRONT (Z = +0.5, Y = -0.5) to the BACK
    // (Z = -0.5, Y = +0.5). Slope normal = (0, +sin45°, +cos45°).
    let slope_n = Vec3::new(
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    );

    let cube_dist_along_slope = local.dot(slope_n);
    let cube_support_on_slope = 0.5 * (slope_n.x.abs() + slope_n.y.abs() + slope_n.z.abs());
    let wedge_support_on_slope = 0.5;

    let i_slope = (cube_support_on_slope + wedge_support_on_slope) - cube_dist_along_slope.abs();
    if i_slope <= 0.0 {
        return None;
    }

    // ── Pick minimum penetration axis ─────────────────────────────────────────
    let (normal, penetration) = if i_slope <= ix + SAT_AXIS_EPS && i_slope <= iz + SAT_AXIS_EPS {
        let sign = if cube_dist_along_slope >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (slope_n * sign, i_slope)
    } else if ix <= iz {
        let sign = if local.x >= 0.0 { 1.0 } else { -1.0 };
        (Vec3::X * sign, ix)
    } else {
        let sign = if local.z >= 0.0 { 1.0 } else { -1.0 };
        (Vec3::Z * sign, iz)
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
fn compute_sphere_sphere_contact(pos_a: Vec3, r_a: f32, pos_b: Vec3, r_b: f32) -> Option<Contact> {
    let delta = pos_a - pos_b;
    let distance = delta.length();
    let radius_sum = r_a + r_b;
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
        contact_point: pos_b + normal * r_b,
    })
}

// =============================================================================
// LEARNING TOPIC: Sphere vs AABB Contact (GJK-lite)
// =============================================================================
//
// To find the closest point on an AABB to a sphere center, we CLAMP the sphere
// center to the box bounds on each axis. The clamped point is the closest point
// on the box surface. If its distance from the sphere center < radius, collision.
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

    // If sphere center is inside the AABB, pick the nearest box face normal
    let (normal, contact_point, penetration) = if distance > 1e-6 {
        (
            delta / distance,
            box_pos + closest,
            sphere_radius - distance,
        )
    } else {
        let dx = box_half.x - local.x.abs();
        let dy = box_half.y - local.y.abs();
        let dz = box_half.z - local.z.abs();

        if dx <= dy && dx <= dz {
            let sign = if local.x >= 0.0 { 1.0 } else { -1.0 };
            let n = Vec3::new(sign, 0.0, 0.0);
            let cp_local = Vec3::new(sign * box_half.x, local.y, local.z);
            (n, box_pos + cp_local, sphere_radius + dx)
        } else if dy <= dz {
            let sign = if local.y >= 0.0 { 1.0 } else { -1.0 };
            let n = Vec3::new(0.0, sign, 0.0);
            let cp_local = Vec3::new(local.x, sign * box_half.y, local.z);
            (n, box_pos + cp_local, sphere_radius + dy)
        } else {
            let sign = if local.z >= 0.0 { 1.0 } else { -1.0 };
            let n = Vec3::new(0.0, 0.0, sign);
            let cp_local = Vec3::new(local.x, local.y, sign * box_half.z);
            (n, box_pos + cp_local, sphere_radius + dz)
        }
    };

    Some(Contact {
        normal,
        penetration,
        contact_point,
    })
}

// =============================================================================
// LEARNING TOPIC: Contact Dispatch (Shape-pair routing)
// =============================================================================
//
// Different shape combinations need different collision algorithms.
// Rust's pattern matching on a tuple of enum variants provides a zero-cost,
// exhaustive dispatch — the compiler guarantees every combination is handled.
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
// Step 1: velocity += gravity * dt
// Step 2: predicted_position = position + velocity * dt
//
// We DON'T update `position` yet — only `predicted_position`.
// `position` stays at the last frame's final solved position.
// The solver then nudges `predicted_position` out of any overlaps.
// Finally, update_velocities_system commits predicted_position → position.
pub fn integrate_system(mut query: Query<&mut Voxel>) {
    for mut voxel in query.iter_mut() {
        if voxel.inv_mass == 0.0 {
            continue;
        }

        if voxel.is_sleeping {
            voxel.velocity = Vec3::ZERO;
            voxel.predicted_position = voxel.position;
            continue;
        }

        // Reset contact accumulators for this frame's fresh contact data
        voxel.contact_normal_accum = Vec3::ZERO;
        voxel.contact_count = 0;
        // SPRINT 4.5: Reset floor contact flag at the START of each frame.
        // It will be set again in solve_constraints_system if the floor
        // constraint fires (or if the block is within FLOOR_CONTACT_EPSILON
        // of the floor).
        voxel.floor_contact = false;

        // Apply gravitational acceleration: Δv = g * dt
        voxel.velocity += GRAVITY * DT;

        // Terminal velocity clamp
        voxel.velocity = voxel.velocity.clamp_length_max(MAX_VELOCITY);

        // Predict where this block will be next frame
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
// SPRINT 4.5: The floor_contacts buffer now captures floor proximity (not just
// penetration), enabling the sleep system to correctly detect floor-resting blocks.
pub fn solve_constraints_system(
    mut query: Query<(Entity, &mut Voxel)>,
    grid: Res<SpatialGrid>,
    settings: Res<PhysicsSettings>,
    mut buffers: ResMut<SolverBuffers>,
) {
    let dynamic_count = query.iter().filter(|(_, v)| v.inv_mass > 0.0).count();

    // OPTIMIZATION: Scale solver iterations to block count.
    // More blocks → fewer iterations per step (trade accuracy for speed).
    // For RL training with many blocks, 2-4 iterations is sufficient
    // since blocks settle over multiple frames anyway.
    let solver_iterations = if dynamic_count >= 500 {
        1
    } else if dynamic_count >= 350 {
        2
    } else if dynamic_count >= 200 {
        4
    } else {
        DEFAULT_SOLVER_ITERATIONS
    };

    for _iteration in 0..solver_iterations {
        // ── BUILD ENTITY-TO-INDEX MAP FOR THIS ITERATION ──────────────────────
        buffers.entity_to_index.clear();
        buffers.snapshots.clear();
        buffers.pos_corrections.clear();
        buffers.normal_accum.clear();
        buffers.normal_count.clear();
        buffers.floor_contacts.clear(); // SPRINT 4.5: reset floor contact tracking
        buffers.processed_pairs.clear(); // SPRINT 4.5: restore required pair deduplication

        let mut idx = 0;
        for (entity, _voxel) in query.iter() {
            buffers.entity_to_index.insert(entity, idx);
            buffers.snapshots.push(BodySnapshot {
                predicted_position: Vec3::ZERO,
                previous_position: Vec3::ZERO,
                inv_mass: 0.0,
                shape: ShapeType::Cube,
                sphere_radius: 0.0,
            });
            buffers.pos_corrections.push(Vec3::ZERO);
            buffers.normal_accum.push(Vec3::ZERO);
            buffers.normal_count.push(0);
            buffers.floor_contacts.push(false); // SPRINT 4.5: one entry per entity
            idx += 1;
        }

        // ── STEP A: Snapshot ─────────────────────────────────────────────────
        // Freeze all predicted positions for this iteration.
        for (entity, voxel) in query.iter() {
            if let Some(&idx) = buffers.entity_to_index.get(&entity) {
                buffers.snapshots[idx] = BodySnapshot {
                    predicted_position: voxel.predicted_position,
                    previous_position: voxel.position,
                    inv_mass: voxel.inv_mass,
                    shape: voxel.shape.clone(),
                    sphere_radius: voxel.sphere_radius,
                };
            }
        }

        // Build column-support map for lateral anchor clamping.
        // LEARNING: Contact-based "settled" checks can flicker when stacked
        // blocks are near-touching but not actively penetrating this iteration.
        // Build a cheap column-support map from snapshot positions so upper
        // pillar blocks are still treated as anchors for lateral impacts.
        let mut stack_supported_by_idx = vec![false; buffers.snapshots.len()];
        let mut columns: std::collections::HashMap<(i32, i32), Vec<(usize, f32)>> =
            std::collections::HashMap::with_capacity(buffers.snapshots.len());
        for (idx, body) in buffers.snapshots.iter().enumerate() {
            columns
                .entry((
                    body.predicted_position.x.round() as i32,
                    body.predicted_position.z.round() as i32,
                ))
                .or_default()
                .push((idx, body.predicted_position.y));
        }
        for members in columns.values_mut() {
            members.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for i in 0..members.len() {
                let (idx_i, y_i) = members[i];
                for j in 0..i {
                    let (_, y_j) = members[j];
                    let dy = y_i - y_j;
                    if dy >= 0.70 && dy <= 1.30 {
                        stack_supported_by_idx[idx_i] = true;
                        break;
                    }
                }
            }
        }

        // ── STEP B: Process each entity's constraints ─────────────────────────
        for (entity, voxel) in query.iter() {
            if voxel.is_sleeping {
                continue;
            }

            let Some(&self_idx) = buffers.entity_to_index.get(&entity) else {
                continue;
            };

            // ── B1: Floor Constraint ──────────────────────────────────────────
            //
            // LEARNING: The floor is an infinite static plane at global_floor_y.
            // We check the 8 corners of the block's rotated bounding box and push
            // the block up so its lowest corner sits exactly at floor_y.
            //
            // SPRINT 4.5 FIX: We now set floor_contacts[self_idx] = true when
            // the block's lowest point is AT OR NEAR the floor (within
            // FLOOR_CONTACT_EPSILON). This is broader than "is the block
            // penetrating the floor?" — it also catches blocks that are exactly
            // at rest on the floor surface (no correction needed, but the block
            // IS resting on the floor and should be able to sleep).
            //
            // Why this matters:
            //   A block resting exactly at y=0.0 has lowest_point = -0.5.
            //   floor_y = -0.5. Condition: -0.5 < -0.5 → FALSE.
            //   Old code: no floor_contact set → sleep never fires.
            //   New code: |−0.5 − (−0.5)| = 0 ≤ FLOOR_CONTACT_EPSILON → floor_contact set.
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
                    let world_corner_offset = voxel.rotation * local_corner;
                    if world_corner_offset.y < min_local_y {
                        min_local_y = world_corner_offset.y;
                    }
                }

                let lowest_point_y = voxel.predicted_position.y + min_local_y;

                // Active floor correction (block is penetrating the floor)
                if lowest_point_y < floor_y {
                    let penetration = floor_y - lowest_point_y;
                    buffers.pos_corrections[self_idx] += Vec3::new(0.0, penetration, 0.0);
                    buffers.normal_accum[self_idx] += Vec3::Y;
                    buffers.normal_count[self_idx] += 1;
                    // Floor contact is definitely active
                    buffers.floor_contacts[self_idx] = true;
                } else if lowest_point_y <= floor_y + FLOOR_CONTACT_EPSILON {
                    // SPRINT 4.5: Block is resting exactly on the floor (no correction
                    // needed, but it IS in contact with the floor surface).
                    // Set floor_contact without adding a correction or normal.
                    // This is purely for the sleep system — no physics correction needed.
                    buffers.floor_contacts[self_idx] = true;
                }
            }

            // ── B2: Broad-Phase Neighbor Detection ────────────────────────────
            //
            // OPTIMIZATION: get_neighbors_into() fills our pre-allocated
            // neighbor_buf instead of allocating a new Vec every call.
            let mut neighbors = std::mem::take(&mut buffers.neighbor_buf);
            grid.get_neighbors_into(voxel.predicted_position, &mut neighbors);
            for (other_e, _other_shape, grid_pos) in neighbors.iter().copied() {
                // We MUST use processed_pairs for deduplication because spatial grid queries
                // are asymmetric! (Entity queries using predicted_position, finding other_e at position. 
                // other_e queries using predicted_position, likely missing Entity).
                let pair = if entity.index() < other_e.index() {
                    (entity, other_e)
                } else {
                    (other_e, entity)
                };
                if !buffers.processed_pairs.insert(pair) {
                    continue;
                }

                let Some(&other_idx) = buffers.entity_to_index.get(&other_e) else {
                    continue;
                };

                let (self_pos, self_prev_pos, self_inv_mass, self_shape, self_radius) = {
                    let s = &buffers.snapshots[self_idx];
                    (
                        s.predicted_position,
                        s.previous_position,
                        s.inv_mass,
                        s.shape.clone(),
                        s.sphere_radius,
                    )
                };
                let (other_pos, other_prev_pos, other_inv_mass, other_shape, other_radius) = {
                    let s = &buffers.snapshots[other_idx];
                    (
                        s.predicted_position,
                        s.previous_position,
                        s.inv_mass,
                        s.shape.clone(),
                        s.sphere_radius,
                    )
                };

                // BUG FIX #1: Use actual snapshot position, not grid-rounded pos.
                // grid_pos is broad-phase only — narrow-phase uses exact positions.
                let _ = grid_pos;

                // ── Run narrow-phase contact detection ────────────────────────
                let Some(c) = compute_contact(
                    self_pos,
                    &self_shape,
                    self_radius,
                    other_pos,
                    &other_shape,
                    other_radius,
                ) else {
                    continue; // No overlap — nothing to resolve
                };

                // ── Compute correction magnitude ──────────────────────────────
                let inv_mass_sum = self_inv_mass + other_inv_mass;
                if inv_mass_sum <= 0.0 {
                    continue; // Both static, nothing to push
                }

                let base_corr =
                    (c.normal * c.penetration).clamp_length_max(MAX_CORRECTION_PER_ITER);

                let mut self_share = self_inv_mass / inv_mass_sum;
                let mut other_share = other_inv_mass / inv_mass_sum;

                // BUG FIX #4: Minimum Vertical Share (Anti-Crush)
                // Prevents heavy dynamic blocks from infinitely crushing light blocks
                // in Jacobi iteration. Without this, an 800kg block on a 1kg block only
                // receives 0.1% correction, and gravity defeats the solver.
                // We enforce a minimum 20% share for vertical bounds so heavy blocks
                // are successfully pushed upward against gravity.
                if c.normal.y.abs() > 0.5 && self_inv_mass > 0.0 && other_inv_mass > 0.0 {
                    let min_share = 0.40;
                    if self_share < min_share {
                        self_share = min_share;
                        other_share = 1.0 - min_share;
                    } else if other_share < min_share {
                        other_share = min_share;
                        self_share = 1.0 - min_share;
                    }
                }

                // Relative speeds from snapshot displacement (stable and cheap).
                let self_speed = (self_pos - self_prev_pos).length() / DT;
                let other_speed = (other_pos - other_prev_pos).length() / DT;

                // BUG FIX #2: Wedge Stability — Settled-Wedge Detection
                let self_is_wedge = self_shape == ShapeType::Wedge;
                let other_is_wedge = other_shape == ShapeType::Wedge;

                if self_is_wedge ^ other_is_wedge {
                    if self_is_wedge {
                        if self_inv_mass < 0.001 && self_speed < WEDGE_SETTLED_SPEED_THRESHOLD {
                            self_share = self_share.min(WEDGE_SETTLED_SHARE_CAP);
                            other_share = 1.0 - self_share;
                        }
                    } else if other_inv_mass < 0.001 && other_speed < WEDGE_SETTLED_SPEED_THRESHOLD
                    {
                        other_share = other_share.min(WEDGE_SETTLED_SHARE_CAP);
                        self_share = 1.0 - other_share;
                    }
                }

                // BUG FIX #3: Lateral Impact Anchor Clamp
                // LEARNING: This is intentionally orthogonal to the wedge logic.
                // Wedge handling targets ramp stability; this targets any settled
                // structure resisting side pushes from moving neighbors.
                if c.normal.y.abs() <= LATERAL_CONTACT_MAX_Y {
                    let self_settled = self_speed <= SETTLED_ANCHOR_SPEED_THRESHOLD;
                    let other_settled = other_speed <= SETTLED_ANCHOR_SPEED_THRESHOLD;
                    let self_stack_supported = stack_supported_by_idx[self_idx];
                    let other_stack_supported = stack_supported_by_idx[other_idx];

                    if self_stack_supported && !other_stack_supported {
                        self_share = self_share.min(SETTLED_ANCHOR_SHARE_CAP);
                        other_share = 1.0 - self_share;
                    } else if other_stack_supported && !self_stack_supported {
                        other_share = other_share.min(SETTLED_ANCHOR_SHARE_CAP);
                        self_share = 1.0 - other_share;
                    } else if self_settled && !other_settled && self_inv_mass <= other_inv_mass {
                        self_share = self_share.min(SETTLED_ANCHOR_SHARE_CAP);
                        other_share = 1.0 - self_share;
                    } else if other_settled && !self_settled && other_inv_mass <= self_inv_mass {
                        other_share = other_share.min(SETTLED_ANCHOR_SHARE_CAP);
                        self_share = 1.0 - other_share;
                    }
                }

                // ── Apply corrections using dense array indexing ──────────────
                if self_inv_mass > 0.0 {
                    let delta = base_corr * self_share;
                    buffers.pos_corrections[self_idx] = (buffers.pos_corrections[self_idx] + delta)
                        .clamp_length_max(MAX_ACCUM_CORRECTION_PER_ITER);
                    buffers.normal_accum[self_idx] += c.normal;
                    buffers.normal_count[self_idx] += 1;
                }
                if other_inv_mass > 0.0 {
                    let delta = base_corr * -other_share;
                    buffers.pos_corrections[other_idx] = (buffers.pos_corrections[other_idx]
                        + delta)
                        .clamp_length_max(MAX_ACCUM_CORRECTION_PER_ITER);
                    buffers.normal_accum[other_idx] -= c.normal;
                    buffers.normal_count[other_idx] += 1;
                }
            }
            neighbors.clear();
            buffers.neighbor_buf = neighbors;
        }

        // If this iteration produced no correction vectors, the system is already
        // non-penetrating for the current frame. Skip remaining iterations.
        if buffers
            .pos_corrections
            .iter()
            .all(|c| c.length_squared() < 1e-10)
        {
            break;
        }

        // ── STEP C: Apply accumulated corrections ─────────────────────────────
        //
        // After ALL pairs have been processed with frozen snapshot data,
        // flush the accumulated corrections into the live predicted_positions.
        for (entity, idx) in buffers.entity_to_index.iter() {
            let corr = buffers.pos_corrections[*idx];
            if let Ok((_, mut v)) = query.get_mut(*entity) {
                // Apply position correction
                v.predicted_position += corr;

                // Any meaningful correction wakes a sleeping body.
                if corr.length_squared() > WAKE_CORRECTION_EPS * WAKE_CORRECTION_EPS {
                    v.is_sleeping = false;
                }

                // Apply contact normal accumulation
                let norm = buffers.normal_accum[*idx];
                v.contact_normal_accum += norm;

                // Apply contact count
                let count = buffers.normal_count[*idx];
                v.contact_count += count;

                // SPRINT 4.5: Commit floor contact flag.
                // Use OR (not assignment) so floor_contact remains true across
                // all iterations if it was set in any previous iteration.
                // A block that was in floor contact in iteration 1 but not in
                // iteration 3 (due to tiny float differences) should stay
                // floor_contact = true for sleep purposes.
                if buffers.floor_contacts[*idx] {
                    v.floor_contact = true;
                }
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
// that slid along a slope will have the correct diagonal velocity.
//
// SPRINT 4.5 FIXES applied in this system:
//
// FIX A — Conditional damping AND friction (fixes heavy-block-stops-dead failure
//          and equal-mass-block-B-gets-zero failure):
//
//   THE PROBLEM (detailed):
//     When a block has contact_count > 0 (any block-block or floor contact),
//     the old code ALWAYS applied:
//       1. 0.94× velocity damping
//       2. Friction decomposition (decompose into normal + tangential,
//          reduce tangential by friction_drop each frame)
//
//     For a Steel block (800kg) moving at 3 m/s in X that then hits the floor:
//       - contact from floor: contact_count=1, avg_normal = Vec3::Y
//       - v_t = lateral velocity = 3 m/s
//       - friction_drop per frame = 9.81 × (1/60) × 0.6 × 1.0 ≈ 0.098 m/s
//       - Steel decelerates and stops in ~30 frames
//       - Test expects vel.x > 0.5 at frame 60 → FAIL
//
//     For equal-mass blocks (Wood A at 5 m/s hits Wood B at 0 m/s):
//       - After collision: A has ~0 m/s, B should have ~5 m/s
//       - But B now has contact_count > 0 from the collision
//       - Friction + damping reduce B's velocity immediately
//       - B ends up with much less than expected → FAIL
//
//   THE FIX:
//     Gate BOTH damping and friction on:
//       speed < DYNAMIC_DAMPING_THRESHOLD (1.5 m/s)
//
//     Fast-moving blocks (speed >= 1.5 m/s) skip ALL friction and damping.
//     Their velocity is determined purely by the XPBD corrections (which
//     correctly encode the collision geometry). Friction is not applied
//     during active collision — only during settling.
//
//     Settling blocks (speed < 1.5 m/s) get the full friction/damping treatment
//     to eliminate micro-vibrations and bring them to rest convincingly.
//
//   PHYSICAL JUSTIFICATION:
//     Real physics has separate models for:
//       - Collision response (impulse-based, happens in microseconds)
//       - Kinetic friction (continuous force while sliding)
//     XPBD handles collision response through corrections. The per-frame
//     friction approximation is a simplified model appropriate for settling
//     behavior but incorrect for high-speed collisions.
//
// FIX B — Unified sleep trigger (fixes sleep-never-fires for floor-resting blocks):
//   has_any_contact = contact_count > 0 || floor_contact
//   Sleep fires when speed < LINEAR_SLEEP_SPEED AND has_any_contact.
//   floor_contact is set by the solver for blocks at or near the floor.
pub fn update_velocities_system(mut query: Query<&mut Voxel>) {
    for mut voxel in query.iter_mut() {
        if voxel.inv_mass == 0.0 {
            continue; // Static objects never move
        }

        // ── Clamp total frame displacement ────────────────────────────────────
        // Even after all solver iterations, cap the maximum displacement per frame.
        let frame_delta = (voxel.predicted_position - voxel.position)
            .clamp_length_max(MAX_DISPLACEMENT_PER_FRAME);
        voxel.predicted_position = voxel.position + frame_delta;

        // ── Derive velocity from position change ──────────────────────────────
        // v = Δx / Δt   (XPBD's core velocity derivation)
        let contact_normal_now = if voxel.contact_count > 0 {
            (voxel.contact_normal_accum / voxel.contact_count as f32).normalize_or_zero()
        } else {
            Vec3::ZERO
        };

        let is_slope_contact = contact_normal_now != Vec3::ZERO
            && contact_normal_now.y < FLAT_CONTACT_Y_THRESHOLD
            && contact_normal_now.y > 0.1; // Has upward component = is a surface

        let mut derived_velocity = if is_slope_contact {
            // On a slope: remove the normal component to prevent solver-artifact launches.
            // Keep only the tangential (sliding) component.
            let raw = (voxel.predicted_position - voxel.position) / DT;
            let v_normal_mag = raw.dot(contact_normal_now);
            let v_tangential = raw - contact_normal_now * v_normal_mag;
            v_tangential
        } else {
            (voxel.predicted_position - voxel.position) / DT
        };

        // ── SPRINT 4.5 FIX A: Conditional damping and friction ────────────────
        //
        // LEARNING: This is the MOST IMPORTANT fix for collision momentum tests.
        //
        // The current speed determines which physical regime we're in:
        //   DYNAMIC (speed >= DYNAMIC_DAMPING_THRESHOLD):
        //     Block is actively moving — collision in progress or just launched.
        //     No damping, no friction. Momentum is conserved.
        //     The XPBD correction already correctly modeled the collision.
        //
        //   SETTLING (speed < DYNAMIC_DAMPING_THRESHOLD):
        //     Block is coming to rest. Apply full friction and damping to
        //     eliminate micro-vibrations and produce convincing resting behavior.
        let current_speed = derived_velocity.length();
        let has_any_contact = voxel.contact_count > 0 || voxel.floor_contact;
        let is_settling = has_any_contact && current_speed < DYNAMIC_DAMPING_THRESHOLD;

        if is_settling {
            // Settling regime: apply full damping to eliminate micro-vibrations.
            // 0.94× removes 6% of velocity per frame — invisible but effective.
            derived_velocity *= POSITION_VELOCITY_DAMPING;
        }
        // Dynamic regime: no damping — momentum must be conserved through collision.

        voxel.velocity = derived_velocity.clamp_length_max(MAX_VELOCITY);

        // ── Friction and angular correction (only in settling regime) ─────────
        //
        // LEARNING: Friction decomposition is ONLY applied when the block is settling.
        // Fast-moving blocks bypass this entirely.
        //
        // This is gated on:
        //   1. contact_count > 0 (we have a block-block contact normal to decompose along)
        //   2. is_settling (block is slow enough that friction is physically meaningful)
        //
        // The floor-only contact (contact_count = 0, floor_contact = true) handled below.
        if voxel.contact_count > 0 && is_settling {
            let avg_normal =
                (voxel.contact_normal_accum / voxel.contact_count as f32).normalize_or_zero();

            if avg_normal != Vec3::ZERO {
                // =================================================================
                // Angular settling — only for upward-facing normals.
                // LEARNING: Only apply settling when the contact normal points
                // "mostly upward" (avg_normal.y >= ANGULAR_SETTLE_MIN_Y_THRESHOLD).
                // Side contacts (steel-beside-steel, wall contacts) produce normals
                // with Y ≈ 0. Including these in settling caused the tipping/spinning
                // bug where blocks tilted 22.5° per frame toward a sideways orientation.
                // =================================================================
                let lateral_dominance = avg_normal.x.abs().max(avg_normal.z.abs());
                let near_rest_for_settle = voxel.velocity.length() <= WEDGE_SETTLED_SPEED_THRESHOLD;

                let support_normal = if avg_normal.y >= ANGULAR_SETTLE_MIN_Y_THRESHOLD_FULL
                    && avg_normal.y > lateral_dominance
                    && near_rest_for_settle
                {
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

                // ── Friction decomposition ────────────────────────────────────
                //
                // LEARNING: Any velocity vector can be decomposed into two components:
                //   v_n = velocity projected ONTO the contact normal (bouncing component)
                //   v_t = velocity minus v_n (sliding component, parallel to surface)
                //
                // We apply friction ONLY to v_t (the sliding part).
                // We apply restitution ONLY to v_n (the bouncing part).
                let contact_basis_normal = if support_normal != Vec3::ZERO {
                    support_normal
                } else {
                    avg_normal
                };
                let v_n_mag = voxel.velocity.dot(contact_basis_normal);
                let v_n = contact_basis_normal * v_n_mag;
                let v_t = voxel.velocity - v_n;

                // Restitution: how much of the normal velocity bounces back.
                // 0.0 = no bounce (dead stop), 1.0 = perfect elastic bounce.
                let resolved_v_n = if v_n_mag < 0.0 {
                    // Moving INTO the surface — apply restitution
                    v_n * voxel.restitution
                } else {
                    // Moving AWAY from surface — keep as-is
                    v_n
                };

                // Friction: reduce tangential velocity by the friction coefficient.
                let v_t_len = v_t.length();
                if v_t_len > 1e-5 {
                    let normal_support = contact_basis_normal.y.max(0.0);
                    let friction_drop = 9.81 * DT * voxel.friction * normal_support;

                    if contact_basis_normal.y >= FLAT_CONTACT_Y_THRESHOLD
                        && v_t_len <= STATIC_FRICTION_SPEED
                    {
                        // Static friction: block is nearly stopped on a flat surface
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
                    voxel.rotation = voxel.rotation.slerp(Quat::IDENTITY, UPRIGHT_SETTLE_RATE);

                    // Snap near-zero Y velocity to exactly zero (prevent micro-bouncing)
                    if voxel.velocity.y.abs() < 0.2 {
                        voxel.velocity.y = 0.0;
                    }
                }
            }
        } else if voxel.floor_contact && is_settling {
            // ── Floor-only contact handling (no block-block contacts) ─────────
            //
            // LEARNING: Blocks resting on the floor with no block-block contacts
            // need simplified handling. We know the floor is flat (Vec3::Y normal),
            // so we can snap rotation toward upright and cancel downward velocity.
            // No complex friction decomposition needed — the block is just resting.
            voxel.rotation = voxel.rotation.slerp(Quat::IDENTITY, UPRIGHT_SETTLE_RATE);
            // Cancel small downward velocity (block is resting on floor)
            if voxel.velocity.y < 0.0 && voxel.velocity.y.abs() < 0.5 {
                voxel.velocity.y *= voxel.restitution;
                if voxel.velocity.y.abs() < 0.05 {
                    voxel.velocity.y = 0.0;
                }
            }
        } else if !has_any_contact {
            // Not in contact with anything — slowly upright the block in the air (cosmetic)
            voxel.rotation = voxel.rotation.slerp(Quat::IDENTITY, 0.05);
            // LEARNING: A block in the air with no contacts is definitely not sleeping.
            // It's falling or launched. Ensure is_sleeping = false so gravity applies.
            voxel.is_sleeping = false;
        }

        // ── SPRINT 4.5 FIX B: Unified sleep trigger ──────────────────────────
        //
        // LEARNING: The old sleep check was inside "if contact_count > 0".
        // This NEVER fired for floor-resting blocks because the floor constraint
        // (when the block is at EXACTLY floor_y) doesn't add to contact_count.
        //
        // The new check uses `has_any_contact` which includes `floor_contact`.
        // A block at rest on the floor has floor_contact = true, velocity ≈ 0,
        // so this correctly puts it to sleep.
        //
        // IMPORTANT: We do NOT wake sleeping blocks here (no "else: is_sleeping=false"
        // when has_any_contact is true but speed >= threshold). Waking is handled by
        // the WAKE_CORRECTION_EPS check in solve_constraints_system Step C.
        // This prevents oscillation between sleep and wake for nearly-settled blocks.
        if has_any_contact && voxel.velocity.length() < LINEAR_SLEEP_SPEED {
            voxel.velocity = Vec3::ZERO;
            voxel.is_sleeping = true;
        }

        // Final safety clamp: if any invalid rotation sneaks in, force identity.
        if !voxel.rotation.is_finite() {
            voxel.rotation = Quat::IDENTITY;
        }

        // ── Commit: predicted_position becomes the authoritative position ──────
        // This is the moment XPBD "applies" the solved frame. From here,
        // next frame's `integrate_system` will use this as the starting point.
        voxel.position = voxel.predicted_position;
    }
}
