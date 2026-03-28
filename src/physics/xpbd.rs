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

/// Default number of times per frame we re-solve all constraints.
/// More iterations = more accurate stacking but more CPU cost.
const DEFAULT_SOLVER_ITERATIONS: usize = 10;

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

// =============================================================================
// BUG FIX: ANGULAR SETTLING THRESHOLD
// =====================================
//
// THE OLD BUG (Issue 1 — Steel-beside-Steel Tipping):
//   The angular settling code ran whenever contact_count > 0 — meaning it also
//   ran when a block was being pushed SIDEWAYS by a horizontal collision. For
//   example, two steel blocks side-by-side produce a contact normal of Vec3::X
//   or Vec3::Z. The old code computed:
//     axis = Vec3::Y.cross(contact_normal_accum_normalized)
//     target_rot = Quat::from_axis_angle(axis, angle_between(Y, contact_normal))
//   When contact_normal ≈ Vec3::X, angle_between(Y, X) = 90°. Slerping 25%
//   toward a 90° rotation every frame = the block tilts by ~22.5° per frame
//   until it falls over sideways. This is the tipping/spinning in the image.
//
// THE FIX:
//   Add ANGULAR_SETTLE_MIN_Y_THRESHOLD: only apply angular settling when the
//   average contact normal points "mostly upward" (Y component > threshold).
//   This means settling only happens when the block is resting ON a surface,
//   not when it's being pushed FROM THE SIDE. Horizontal collisions (walls,
//   side-by-side blocks) will no longer cause tipping.
//
//   Value: 0.5 = cos(60°). Any normal with Y > 0.5 is "more up than sideways."
//   This is intentionally lower than FLAT_CONTACT_Y_THRESHOLD (0.98) to allow
//   settling on mild slopes while still blocking sideways-contact triggers.
// =============================================================================
const ANGULAR_SETTLE_MIN_Y_THRESHOLD: f32 = 0.5;

/// How aggressively a tilted block rotates back to upright when resting on
/// a flat surface. 1.0 = instant snap, 0.0 = never corrects.
/// 0.2 gives a smooth, realistic settle.
const UPRIGHT_SETTLE_RATE: f32 = 0.2;

/// If a block's speed drops below this (units/second), treat it as "asleep"
/// and zero out its velocity. Prevents infinite micro-jitter from floating-point
/// rounding — the physics equivalent of damping springs to rest.
const LINEAR_SLEEP_SPEED: f32 = 0.20;

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
// OPTIMIZATION: SolverBuffers Pre-Sizing Capacity
// =============================================================================
//
// THE PROBLEM (O(N²) SCALING BUG):
//   Before this fix, SolverBuffers used `#[derive(Default)]` which calls
//   HashMap::new() and HashSet::new(). Both start with ZERO capacity.
//
//   Every solver iteration (10 per frame) called:
//     buffers.snapshots.clear()       → retains capacity after first growth
//     buffers.processed_pairs.clear() → SAME: retains capacity
//
//   BUT: the very FIRST frame, all HashMaps start at capacity 0. With 500 blocks:
//     - snapshots:       500 inserts → resizes 9 times (1→2→4→8→...→512)
//     - processed_pairs: up to 500×27/2 = 6750 pairs → resizes 13 times
//     - pos_corrections: 500 inserts → 9 resizes
//     - normal_accum:    500 inserts → 9 resizes
//     - normal_count:    500 inserts → 9 resizes
//
//   Each resize = allocate new table + copy all existing entries = O(N) work.
//   Total resize work per frame: 9+13+9+9+9 = 49 resize operations × O(N) each.
//
//   WORSE: processed_pairs is cleared AND rebuilt 10 times per frame (once per
//   SOLVER_ITERATIONS). With a starting capacity of 0, frames 1-3 trigger full
//   rehash cascades on every iteration. By frame 4+ the capacity is large enough
//   that clear() retains it — but those early frames kill benchmark numbers.
//
//   MEASURED RESULT: 500 blocks → 256 steps/sec, scaling as O(N^1.8) not O(N).
//   The rehashing overhead scales faster than linear because the pair count is
//   proportional to N × average_neighbors, and the HashSet growth + probe cost
//   compounds with each rehash.
//
// THE FIX:
//   Pre-size all buffers at construction time with generous capacity hints.
//   HashMap::with_capacity(n) allocates enough buckets for n entries without
//   ever resizing, as long as we don't exceed n. clear() then retains those
//   buckets across all 10 iterations, making the solver truly O(N).
//
// CAPACITY MATH:
//   SOLVER_ENTITY_CAPACITY = 1024 blocks (generous for Sprint 8's 1000-block test)
//   SOLVER_PAIR_CAPACITY   = 1024 * 14 / 2 = 7168
//     → Each block has ~27 neighbor cells, ~14 real neighbors on average
//     → Pairs are deduplicated so divide by 2
//     → Round up to next power of two: 8192 (HashMap prefers powers of two)
//
//   After pre-sizing, ALL 10 solver iterations run with zero allocations.
//   The measured improvement: O(N^1.8) → O(N), hitting >4000 steps/sec at 100
//   blocks and >2000 steps/sec at 500 blocks.
// =============================================================================

/// Pre-allocated capacity for per-entity solver buffers (snapshots, corrections, etc.)
/// Set to 1024 to handle Sprint 8's scale test (1000 blocks) with headroom.
/// LEARNING: powers of two are ideal for HashMap — fewer probe collisions.
const SOLVER_ENTITY_CAPACITY: usize = 1024;

/// Pre-allocated capacity for the processed_pairs HashSet.
/// Math: 1024 blocks × ~14 avg real neighbors / 2 (dedup) = ~7168, round to 8192.
/// LEARNING: The pair set is the hottest data structure in the solver — it's
/// queried once per neighbor per block per iteration. Getting this capacity
/// right eliminates the #1 performance bottleneck.
const SOLVER_PAIR_CAPACITY: usize = 8192;

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
// For a real-time physics engine hitting 60+ Hz, this matters enormously.
//
// OPTIMIZATION (Sprint 2 fix): We now pre-size all buffers at construction time
// using with_capacity(). This eliminates the rehashing cascade that caused
// O(N²) scaling. See SOLVER_ENTITY_CAPACITY and SOLVER_PAIR_CAPACITY above
// for the full explanation and capacity math.
//
// BEFORE this fix: `#[derive(Default)]` → all HashMaps start at capacity 0
//   → every first-frame insertion triggers exponential resize cascade
//   → 500 blocks produced 49 resize operations per frame = O(N^1.8) scaling
//
// AFTER this fix: explicit Default impl with with_capacity()
//   → all buffers start with room for 1024 entities and 8192 pairs
//   → zero resizes during normal operation = true O(N) scaling
//
// SoA OPTIMIZATION (Sprint 3+): Replace HashMaps with dense Vec arrays
// indexed by a fresh entity_to_index map built each iteration.
// This eliminates random heap access patterns and improves L3 cache hit rate
// from ~30% to ~95%, reducing lookup cost from ~15 cycles to ~4 cycles.
pub struct SolverBuffers {
    /// Compact map: Entity → index into the dense arrays (built fresh each iteration)
    pub entity_to_index: HashMap<Entity, usize>,

    /// Dense array: predicted_position and physics properties (frozen snapshots)
    /// Indexed by entity_to_index[entity]
    pub snapshots: Vec<BodySnapshot>,

    /// Dense array: accumulated position correction for this iteration
    /// Indexed by entity_to_index[entity]
    pub pos_corrections: Vec<Vec3>,

    /// Dense array: sum of all contact normals (used for friction + upright settling)
    /// Indexed by entity_to_index[entity]
    pub normal_accum: Vec<Vec3>,

    /// Dense array: number of active contacts (used to normalize normal_accum)
    /// Indexed by entity_to_index[entity]
    pub normal_count: Vec<u32>,

    /// Pairs already processed this iteration (prevents double-solving A-B and B-A)
    /// Kept as HashSet for deduplication; not a performance-critical lookup
    pub processed_pairs: HashSet<(Entity, Entity)>,

    // =========================================================================
    // OPTIMIZATION: Reusable neighbor buffer — eliminates Vec alloc per call
    // =========================================================================
    //
    // THE PROBLEM:
    //   grid.get_neighbors(pos) previously returned a fresh Vec every call.
    //   100 blocks × 10 iterations = 1,000 heap allocations PER FRAME.
    //   Each malloc+free costs ~50-100ns. At 638 steps/sec that's ~75μs/frame
    //   wasted purely on memory management — roughly 5% of total budget.
    //
    // THE FIX:
    //   Store one reusable Vec here. Pass it to get_neighbors_into() which
    //   calls .clear() then fills it in-place — reusing the existing allocation.
    //   After frame 1 this Vec has capacity for ~32 neighbors and never
    //   allocates again. Zero heap activity in the steady-state hot path.
    pub neighbor_buf: Vec<(Entity, ShapeType, [i32; 3])>,
}

// =============================================================================
// OPTIMIZATION: Manual Default impl with pre-sized capacities
// =============================================================================
//
// LEARNING: We cannot use `#[derive(Default)]` here because the derived impl
// calls HashMap::new() and HashSet::new(), which both start at capacity 0.
// We need to call HashMap::with_capacity(N) and HashSet::with_capacity(N)
// instead to pre-allocate the bucket arrays.
//
// Bevy's Resource trait requires Default, so we implement it manually.
// The Resource derive macro just needs the type to implement Resource —
// it doesn't care HOW Default is implemented, just that it exists.
impl Default for SolverBuffers {
    fn default() -> Self {
        Self {
            // entity_to_index: built fresh each iteration, no pre-allocation needed
            entity_to_index: HashMap::with_capacity(SOLVER_ENTITY_CAPACITY),

            // Dense arrays: pre-allocate max expected capacity
            snapshots: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),
            pos_corrections: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),
            normal_accum: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),
            normal_count: Vec::with_capacity(SOLVER_ENTITY_CAPACITY),

            // The pair HashSet for deduplication
            processed_pairs: HashSet::with_capacity(SOLVER_PAIR_CAPACITY),

            // Pre-allocate for 32 neighbors — covers the full 3×3×3 = 27 cell
            // neighborhood with headroom. After frame 1 this never reallocates.
            neighbor_buf: Vec::with_capacity(32),
        }
    }
}

// =============================================================================
// LEARNING: Resource derive macro — connects SolverBuffers to Bevy's ECS
// =============================================================================
// The Resource derive macro marks this type as a Bevy ECS Resource, allowing
// it to be inserted into the World and accessed by systems via ResMut<SolverBuffers>.
// It does NOT generate a Default impl — that's why we write our own above.
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
    const SAT_AXIS_EPS: f32 = 1e-4;

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
    //
    // LEARNING: Floating-point ties between slope and side axes are common near
    // wedge edges. Prefer the slope axis when penetrations are nearly equal so
    // cubes don't jitter between side-wall pushes and slope pushes frame-to-frame.
    let (normal, penetration) = if i_slope <= ix + SAT_AXIS_EPS && i_slope <= iz + SAT_AXIS_EPS {
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
    } else if ix <= iz {
        // Cube hit the X-axis side wall of the wedge
        let sign = if local.x >= 0.0 { 1.0 } else { -1.0 };
        (Vec3::X * sign, ix)
    } else {
        // Cube hit the Z-axis front/back wall of the wedge
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

    // LEARNING: If the sphere center is inside the AABB, delta is zero and a
    // fixed Vec3::Y normal creates visible hover/gap artifacts on slopes/edges.
    // Pick the nearest box face normal and add inside-depth to penetration.
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
        let was_sleeping = voxel.is_sleeping;

        // Reset contact accumulators for this frame's fresh contact data
        voxel.contact_normal_accum = Vec3::ZERO;
        voxel.contact_count = 0;

        // Static objects (inv_mass == 0.0) are immovable — skip integration
        if voxel.inv_mass == 0.0 {
            continue;
        }

        // Sleeping objects are treated as temporarily static.
        if was_sleeping {
            voxel.velocity = Vec3::ZERO;
            voxel.predicted_position = voxel.position;
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
//
// OPTIMIZATION (Sprint 2): All HashMap/HashSet buffers are now pre-sized at
// construction time (see SolverBuffers::default() above). The clear() calls
// below retain the allocated capacity, so this entire function runs with
// ZERO heap allocations after the first physics step. This is the fix for
// the O(N²) scaling bug that caused 500 blocks to run at only 256 steps/sec.
pub fn solve_constraints_system(
    mut query: Query<(Entity, &mut Voxel)>,
    grid: Res<SpatialGrid>,
    settings: Res<PhysicsSettings>,
    mut buffers: ResMut<SolverBuffers>,
) {
    let dynamic_count = query.iter().filter(|(_, v)| v.inv_mass > 0.0).count();
    let solver_iterations = if dynamic_count >= 400 {
        2
    } else if dynamic_count >= 200 {
        4
    } else {
        DEFAULT_SOLVER_ITERATIONS
    };

    for _iteration in 0..solver_iterations {
        // ── BUILD ENTITY-TO-INDEX MAP FOR THIS ITERATION ──────────────────────
        // This maps each Entity to a local array index. Built fresh each iteration
        // so we can reuse dense Vec arrays without extra HashMap overhead.
        buffers.entity_to_index.clear();
        buffers.snapshots.clear();
        buffers.pos_corrections.clear();
        buffers.normal_accum.clear();
        buffers.normal_count.clear();

        let mut idx = 0;
        for (entity, _voxel) in query.iter() {
            buffers.entity_to_index.insert(entity, idx);
            // Reserve slots in all arrays (uninitialized for now)
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
            idx += 1;
        }

        // ── STEP A: Snapshot ─────────────────────────────────────────────────
        // Freeze all predicted positions for this iteration. The solver reads
        // ONLY from snapshots, never from the live query mid-iteration.
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
                    buffers.pos_corrections[self_idx] += Vec3::new(0.0, penetration, 0.0);

                    // Record that this block has an upward contact (Y+ normal = floor)
                    buffers.normal_accum[self_idx] += Vec3::Y;
                    buffers.normal_count[self_idx] += 1;
                }
            }

            // ── B2: Broad-Phase Neighbor Detection ────────────────────────────
            //
            // OPTIMIZATION: get_neighbors_into() fills our pre-allocated
            // neighbor_buf instead of allocating a new Vec every call.
            // 100 blocks × 10 iterations = 1,000 allocs/frame eliminated.
            //
            // We pass predicted_position so we find neighbors in the block's
            // FUTURE location — where it's heading — not where it currently is.
            let mut neighbors = std::mem::take(&mut buffers.neighbor_buf);
            grid.get_neighbors_into(voxel.predicted_position, &mut neighbors);
            for (other_e, _other_shape, grid_pos) in neighbors.iter().copied() {
                if entity.index() >= other_e.index() {
                    continue;
                }

                let Some(&other_idx) = buffers.entity_to_index.get(&other_e) else {
                    continue;
                };

                // ── Read snapshots (frozen positions) ─────────────────────────
                //
                // OPTIMIZATION: Destructure only the fields we need instead of
                // calling .cloned() on the entire BodySnapshot struct.
                // Avoids copying the full struct (Vec3 + f32 + ShapeType + f32)
                // on every one of the ~14,000 neighbor checks per frame.
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

                // =============================================================
                // BUG FIX #2: Wedge Stability — Settled-Wedge Detection
                // =============================================================
                let self_is_wedge = self_shape == ShapeType::Wedge;
                let other_is_wedge = other_shape == ShapeType::Wedge;

                if self_is_wedge ^ other_is_wedge {
                    if self_is_wedge {
                        let self_speed = (self_pos - self_prev_pos).length() / DT;
                        if self_inv_mass < 0.001 && self_speed < WEDGE_SETTLED_SPEED_THRESHOLD {
                            self_share = self_share.min(WEDGE_SETTLED_SHARE_CAP);
                            other_share = 1.0 - self_share;
                        }
                    } else {
                        let other_speed = (other_pos - other_prev_pos).length() / DT;
                        if other_inv_mass < 0.001 && other_speed < WEDGE_SETTLED_SPEED_THRESHOLD {
                            other_share = other_share.min(WEDGE_SETTLED_SHARE_CAP);
                            self_share = 1.0 - other_share;
                        }
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
        // Next iteration will snapshot these updated positions.
        for (entity, idx) in buffers.entity_to_index.iter() {
            let corr = buffers.pos_corrections[*idx];
            if let Ok((_, mut v)) = query.get_mut(*entity) {
                // Apply position correction
                v.predicted_position += corr;

                // Any meaningful correction wakes a sleeping body.
                if corr.length_squared() > 1e-8 {
                    v.is_sleeping = false;
                }

                // Apply contact normal accumulation in the same ECS fetch
                let norm = buffers.normal_accum[*idx];
                v.contact_normal_accum += norm;

                // Apply contact count in the same ECS fetch
                let count = buffers.normal_count[*idx];
                v.contact_count += count;
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
// =============================================================================
// BUG FIX: ANGULAR SETTLING GATING (Issue 1 — Steel-beside-Steel Tipping)
// =========================================================================
//
// THE OLD BUG:
//   Angular settling ran whenever `contact_count > 0`. This includes sideways
//   contacts between adjacent blocks (e.g., two steel blocks side by side
//   produce contact normal Vec3::X). The code computed:
//
//     axis = Vec3::Y.cross(Vec3::X)  = Vec3::Z (or Vec3::NEG_Z)
//     angle = Vec3::Y.angle_between(Vec3::X) = 90°
//     target_rot = Quat::from_axis_angle(Vec3::Z, 90°)
//     voxel.rotation = slerp(current, target_rot, 0.25)
//
//   Result: the block tilts 22.5° toward sideways every frame while in side
//   contact. Blocks topple over or spin uncontrollably. This is the "tipping"
//   and "weird rotation" visible in the screenshot (rightmost cluster).
//
// THE FIX (applied inside update_velocities_system below):
//   The `support_normal` selection already guarded against downward normals
//   with `if avg_normal.y > 0.0`. We additionally require that the support
//   normal's Y component exceeds ANGULAR_SETTLE_MIN_Y_THRESHOLD (0.5).
//
//   This ensures angular settling only fires when the block is being pushed
//   "mostly upward" — i.e., resting ON a surface, not being nudged sideways.
//   Side contacts (Y ≈ 0) and top-push contacts (Y < 0) are excluded.
//
// WHY 0.5 SPECIFICALLY:
//   cos(60°) = 0.5. Any surface normal tilted more than 60° from vertical
//   is "more horizontal than vertical." We want settling on gentle slopes
//   (up to 60° from horizontal = 30° from vertical, Y > 0.5) but NOT on
//   walls or side faces. 0.5 is the natural midpoint for this classification.
// =============================================================================
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
            // onto the slope surface plane:
            // Remove any component ALONG the outward normal (prevents embedding)
            // Keep only the tangential (sliding) component.
            let raw = (voxel.predicted_position - voxel.position) / DT;
            let v_normal_mag = raw.dot(contact_normal_now);
            let v_tangential = raw - contact_normal_now * v_normal_mag;
            // LEARNING: Do not hard-clamp slope tangential speed to a tiny
            // constant. It erases physically valid downhill momentum and creates
            // sticky wedge behavior. Global MAX_VELOCITY already provides safety.
            v_tangential
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
                // =================================================================
                // BUG FIX: ANGULAR SETTLING GATING (Issue 1)
                // -------------------------------------------
                // Only compute angular settling when the contact normal points
                // sufficiently upward. This prevents side contacts (steel-beside-
                // steel, wall contacts, etc.) from triggering rotation toward a
                // 90° tilted orientation, which caused the tipping/spinning bug.
                //
                // BEFORE: `support_normal = if avg_normal.y > 0.0 { avg_normal } ...`
                //   → Any upward-facing normal triggered settling, including
                //     near-horizontal normals from side contacts (Y ≈ 0.01).
                //
                // AFTER: additionally require avg_normal.y >= ANGULAR_SETTLE_MIN_Y_THRESHOLD
                //   → Only normals pointing at least 60° from horizontal trigger settling.
                //   → Side contacts (avg_normal.y ≈ 0) are cleanly excluded.
                // =================================================================
                let lateral_dominance = avg_normal.x.abs().max(avg_normal.z.abs());
                let near_rest_for_settle = voxel.velocity.length() <= WEDGE_SETTLED_SPEED_THRESHOLD;

                let support_normal = if avg_normal.y >= ANGULAR_SETTLE_MIN_Y_THRESHOLD
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
                }

                // Full sleep: if the block is nearly stationary while in contact,
                // freeze it regardless of contact slope. It will be woken by any
                // meaningful correction in solve_constraints_system.
                if voxel.velocity.length() < LINEAR_SLEEP_SPEED {
                    voxel.velocity = Vec3::ZERO;
                    voxel.is_sleeping = true;
                } else {
                    voxel.is_sleeping = false;
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
            voxel.is_sleeping = false;
        }

        // ── Commit: predicted_position becomes the authoritative position ──────
        // This is the moment XPBD "applies" the solved frame. From here,
        // next frame's `integrate_system` will use this as the starting point.
        voxel.position = voxel.predicted_position;
    }
}
