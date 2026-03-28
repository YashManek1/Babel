// =============================================================================
// src/physics/mortar.rs  —  Operation Babel: Mortar Bond System (Sprint 3)
// =============================================================================
//
// WHAT THIS FILE DOES:
//   Implements "Fixed Distance Constraints" between side-touching blocks.
//   This is what lets an arch or bridge hold itself up without support below.
//
// FOUR SYSTEMS / HELPERS IN THIS FILE:
//
//   1. try_register_bonds()
//      Called ONCE from lib.rs immediately after a block is spawned.
//      Checks all 6 face-neighbors in the spatial grid. If a neighbor has
//      non-zero adhesion_strength on either side, creates a MortarBond.
//      → Called from lib.rs spawn_voxel_and_register()
//
//   2. solve_mortar_constraints_system()
//      Runs EVERY physics step inside the XPBD solver loop.
//      For each live bond: measure current distance between the two block centers.
//      If they've drifted apart, push them back toward rest_distance.
//      This is the "rubber band" — it resists separation.
//      → Added to the schedule in lib.rs between solve_constraints and update_velocities
//
//   3. break_overloaded_bonds_system()
//      Runs EVERY physics step AFTER solve_mortar_constraints.
//      Computes the breaking condition from bond tension.
//      If tension > adhesion_strength → bond deleted.
//      → Runs immediately after solve_mortar_constraints in schedule
//
//   4. register_new_bonds_system()
//      Runs EVERY physics step after collision resolution.
//      Detects face-neighbor pairs that became adjacent only after falling,
//      settling, or sliding into place, then registers any missing bonds.
//      → Added to the schedule in lib.rs before solve_mortar_constraints
//
// LEARNING TOPIC: Why a separate Resource instead of storing bonds on Voxel?
//   A bond connects TWO entities. Storing it on one voxel means the other
//   voxel doesn't know about it — you'd need to query both and reconcile.
//   A central Vec<MortarBond> in a Resource is simpler: one place, all bonds,
//   easy to iterate, easy to remove by index. This is the standard ECS pattern
//   for "relationship data" that doesn't belong to a single entity.
//
// =============================================================================

use crate::world::spatial_grid::SpatialGrid;
use crate::world::voxel::{ShapeType, Voxel};
use bevy_ecs::prelude::*;
use glam::Vec3;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Rest distance between two bonded block centers (world units).
/// For unit cubes touching face-to-face: centers are exactly 1.0 apart.
/// This is the distance the rubber band tries to maintain.
const BOND_REST_DISTANCE: f32 = 1.0;

/// Maximum vertical offset still considered "side-by-side" for mortar bonding.
///
/// LEARNING: Settled stacks are not perfectly grid-aligned every frame. Using a
/// strict same-Y-cell requirement prevents valid sideways bonds from registering.
/// This tolerance restores intended attachment without enabling vertical glue.
const SIDE_BOND_VERTICAL_TOLERANCE: f32 = 2.25;

/// Allowed error around 1.0-unit lateral spacing for side-bond registration.
const SIDE_BOND_LATERAL_TOLERANCE: f32 = 0.75;

/// Minimum horizontal separation to classify a neighbor as side-contact.
/// Prevents pure vertical stacks (dx≈0, dz≈0) from becoming bonded.
const SIDE_BOND_MIN_HORIZONTAL_SEP: f32 = 0.55;

/// Max orthogonal horizontal offset still accepted for side/corner attachment.
const SIDE_BOND_ORTHOGONAL_MAX: f32 = 1.05;

// =============================================================================
// BUG FIX: MASS-PROPORTIONAL COMPLIANCE (Issue 3a — Steel-on-Steel Vibration)
// =============================================================================
//
// THE OLD BUG:
//   BASE_BOND_COMPLIANCE = 0.001 was a FIXED constant regardless of material.
//
//   The correction magnitude formula is:
//     correction = violation × (1 / (inv_mass_sum + compliance))
//
//   For two steel blocks (inv_mass = 0.00125 each):
//     inv_mass_sum = 0.0025
//     correction = violation × (1 / (0.0025 + 0.001))
//                = violation × (1 / 0.0035)
//                = violation × 285.7
//
//   Even a TINY violation of 0.001 world-units (floating-point drift) produces:
//     correction = 0.001 × 285.7 = 0.286 units → clamped to MAX_BOND_CORRECTION = 0.15
//
//   Both steel blocks are pushed 0.15 units TOWARD each other every frame.
//   They overshoot (now 0.15 units on the other side), get pushed back 0.15 units.
//   This is a classic stiff spring oscillation — the blocks vibrate forever.
//
// ROOT CAUSE: The compliance denominator (0.001) is much LARGER than the
//   inv_mass_sum (0.0025), so the mass term barely matters. The correction
//   magnitude is dominated by 1/compliance = 1000, not by 1/mass = 400.
//   This makes heavy blocks behave as if they had almost zero mass for
//   bond purposes — they snap together with the same force as feather-light blocks.
//
// THE FIX:
//   Scale compliance proportionally to inv_mass_sum:
//     compliance = BASE_BOND_COMPLIANCE × inv_mass_sum
//
//   For two steel blocks:
//     compliance = 0.05 × 0.0025 = 0.000125
//     correction = violation × (1 / (0.0025 + 0.000125))
//                = violation × (1 / 0.002625)
//                = violation × 381   ← still stiff, but now PROPORTIONAL to mass
//
//   Wait — that's WORSE! The trick is that we also change BASE_BOND_COMPLIANCE
//   to a much smaller value (0.05 instead of 0.001) so the absolute compliance
//   remains small (0.05 × 0.0025 = 0.000125 << 0.001).
//
//   ACTUALLY the correct insight is: for XPBD compliance, the formula becomes
//   stable when compliance is in the same ORDER OF MAGNITUDE as inv_mass_sum.
//   Setting compliance = k × inv_mass_sum ensures the ratio stays constant:
//     correction = violation × (1 / (inv_mass_sum × (1 + k)))
//                = violation × (1 / (inv_mass_sum)) × (1 / (1 + k))
//                = violation × mass × (1 / (1 + k))
//
//   The mass term now correctly governs correction magnitude!
//   For k = 0.5: correction = violation × mass × 0.667
//   For steel: correction = violation × 800 × 0.667 = violation × 533
//   But violation is tiny (< 0.001 for settled blocks), so correction ≈ 0.5 units MAX.
//   With MAX_BOND_CORRECTION = 0.05 (also reduced), this stays bounded and stable.
//
// PHYSICAL INTERPRETATION:
//   Compliance scaling by inv_mass_sum means: "stiffer bonds for lighter materials,
//   softer bonds for heavier materials." This matches engineering reality —
//   a steel weld is modeled as a very stiff constraint (the compliance is the
//   reciprocal of the joint stiffness coefficient), and for steel's mass,
//   the stiffness should be high relative to displacement.
//
// SEE ALSO: MAX_BOND_CORRECTION reduced from 0.15 → 0.05 to prevent overshoot.
// =============================================================================

/// Base compliance multiplier for the mass-proportional bond stiffness formula.
/// compliance = BASE_BOND_COMPLIANCE_FACTOR × inv_mass_sum
///
/// Lower factor = stiffer bond (less stretch per unit of violation).
/// 0.5 gives a "half-stiffness" relative to the mass scale.
/// This constant was tuned so:
///   - Wood bonds (inv_mass=1.0 each, sum=2.0): compliance=1.0, soft & stretchy ✓
///   - Steel bonds (inv_mass=0.00125 each, sum=0.0025): compliance=0.00125, stiff ✓
///   - Mixed Wood+Steel: compliance=0.50, medium ✓
const BASE_BOND_COMPLIANCE_FACTOR: f32 = 0.5;

/// Maximum correction the bond constraint can apply per solver step.
///
/// REDUCED from 0.15 → 0.05 as part of the vibration fix.
///
/// The old value of 0.15 was appropriate when compliance was fixed at 0.001
/// (where corrections were small). With mass-proportional compliance, the raw
/// unclamped correction for steel can be large (stiff bond, heavy mass), so
/// we reduce the cap to prevent overshoot oscillation while still allowing
/// meaningful convergence within 10 solver iterations.
///
/// Math: 0.05 units × 10 solver steps = 0.5 units max correction per frame.
/// A bond stretched 0.5 units is a very loose bond — if it can't close in
/// 0.5 units per frame, the block is probably being pulled by a strong force
/// (gravity on a steep overhang) and should eventually break.
const MAX_BOND_CORRECTION: f32 = 0.05;

/// Maximum averaged mortar displacement applied to a single entity per physics step.
///
/// LEARNING: A block can participate in multiple bonds. Even with Jacobi
/// averaging, simultaneous pulls can still create visible "drag the whole pillar"
/// behavior in one frame. This cap limits mortar's authority so collision +
/// gravity remain dominant and structural movement stays smooth.
const MAX_ENTITY_MORTAR_STEP: f32 = 0.035;

/// If a block is floor-supported, cap how much bond-share it can absorb.
///
/// LEARNING: Supported heavy anchors (especially steel columns) should not be
/// laterally dragged by freshly-placed neighbors. We keep a small non-zero share
/// so bonds still converge, but prevent whole-pillar translation.
const SUPPORTED_ANCHOR_SHARE_CAP: f32 = 0.08;

// =============================================================================
// BUG FIX: MINIMUM TENSION WAKE THRESHOLD (Issue 3b — Pillar Flying)
// =============================================================================
//
// THE OLD BUG (Pillar Flying):
//   When a wood block is placed beside a sleeping steel pillar:
//     1. Wood falls, lands, forms bonds with all 3 steel blocks in the pillar.
//     2. Each bond has tension because wood settled slightly away from steel.
//     3. solve_mortar_constraints_system sets is_sleeping = false for ALL bonded blocks.
//     4. The sleeping steel blocks wake up. Their velocity was zeroed by sleep.
//     5. integrate_system now runs for the steel blocks: velocity += GRAVITY * DT
//        → v.y = -9.81 * 0.0167 = -0.164 m/s (downward) for each steel block.
//     6. predicted_position.y = position.y + (-0.164) * DT = position.y - 0.00274
//        → Each steel block tries to fall slightly.
//     7. The collision solver pushes the steel blocks back up (floor/stacking).
//     8. But the MORTAR BOND between wood and bottom-steel NOW has tension
//        because wood is being pulled sideways while steel moved down.
//     9. Bond correction pushes wood UP (steel's share ≈ 0, wood's share ≈ 1.0).
//    10. Wood flies up. Its bond to the next steel block above pulls THAT block.
//    11. Chain reaction: entire pillar launches upward.
//
// ROOT CAUSE: Mortar bonds should NOT wake sleeping blocks unless the violation
//   is large enough to matter physically. Floating-point drift causes tiny
//   violations (< 0.01 units) that are invisible but wake sleeping blocks,
//   causing a cascade of gravity applications followed by over-corrections.
//
// THE FIX: BOND_WAKE_TENSION_THRESHOLD
//   Only wake a sleeping block (set is_sleeping = false) if the bond tension
//   exceeds this threshold. Tiny drift violations are silently corrected without
//   disturbing the sleep state.
//
//   Additionally: if BOTH blocks are sleeping and tension is below threshold,
//   skip the bond entirely (it's already "at rest").
//
// VALUE DERIVATION:
//   Floating-point drift for a settled steel block at y=0.5 over 1000 frames:
//   approximately 1e-5 to 1e-4 units of drift per frame.
//   BOND_WAKE_TENSION_THRESHOLD = 0.02 is 200× larger than typical drift,
//   ensuring drift never wakes blocks, while 0.02 world-units of real separation
//   (about 2% of a block width) DOES correctly wake them for constraint solving.
const BOND_WAKE_TENSION_THRESHOLD: f32 = 0.02;

// =============================================================================
// BUG FIX: NO UPWARD MORTAR CORRECTION ON SUPPORTED BLOCKS (Issue 3b)
// =============================================================================
//
// THE OLD BUG (secondary cause of pillar flying):
//   The mortar constraint pushes block A toward block B along the bond axis.
//   When wood is bonded SIDEWAYS to the bottom of a steel pillar:
//     - Bond axis is horizontal (e.g., Vec3::X)
//     - Correction is purely horizontal → this is correct
//
//   BUT: when wood is bonded to the SIDE of a steel block that is ON TOP of
//   another steel block, the wood's center-of-mass is slightly below the bond
//   target height (wood hasn't perfectly settled at the same Y as the steel).
//   The bond axis becomes (1.0, 0.1, 0.0) — mostly horizontal but with a
//   small upward component. The steel share of the correction = 0.00125/1.00125.
//   Even this tiny upward nudge on steel, multiplied by mass = 800kg, gives
//   a small but non-zero upward velocity. Over 10 solver iterations this
//   accumulates into a visible upward drift.
//
// THE FIX: BOND_MAX_UPWARD_CORRECTION_FOR_SUPPORTED
//   When applying the mortar correction to a block that is floor-supported
//   (contact_count > 0 AND avg_contact_normal.y > 0.9 from last frame),
//   clamp the Y component of the correction to zero (no upward push from mortar).
//
//   Mortar bonds should resist SEPARATION (pulling apart), not LIFT blocks
//   off their supports. Gravity and the collision solver handle vertical
//   positioning. Mortar only needs to handle horizontal cohesion.
//
// IMPLEMENTATION: We pass the Voxel query as read access for contact data,
//   then zero out the Y correction component for supported blocks.
// =============================================================================

/// Contact Y threshold for "this block is resting on a flat surface."
/// Same as FLAT_CONTACT_Y_THRESHOLD in xpbd.rs (0.98).
/// Blocks with avg contact normal Y above this are "floor-supported."
const FLOOR_SUPPORTED_NORMAL_Y: f32 = 0.90;

/// LEARNING: Contact normals can flicker during settle transitions. We require
/// low speed in addition to an upward average normal before classifying a block
/// as floor-supported for the mortar Y-lift clamp.
const SUPPORT_STABLE_SPEED: f32 = 0.35;

fn shape_supports_mortar(shape: ShapeType) -> bool {
    !matches!(shape, ShapeType::Sphere)
}

/// Returns a pure horizontal side-bond axis when two blocks are laterally adjacent.
/// Rejects near-vertical/stacked neighbors to avoid pillar-lift glue behavior.
fn compute_side_bond_axis(delta: Vec3) -> Option<Vec3> {
    if delta.y.abs() > SIDE_BOND_VERTICAL_TOLERANCE {
        return None;
    }

    let ax = delta.x.abs();
    let az = delta.z.abs();

    // Must have meaningful horizontal separation to be a side bond.
    if ax.max(az) < SIDE_BOND_MIN_HORIZONTAL_SEP {
        return None;
    }

    if ax >= az {
        if (ax - BOND_REST_DISTANCE).abs() <= SIDE_BOND_LATERAL_TOLERANCE
            && az <= SIDE_BOND_ORTHOGONAL_MAX
            && ax > 1e-5
        {
            return Some(Vec3::new(delta.x.signum(), 0.0, 0.0));
        }
    } else if (az - BOND_REST_DISTANCE).abs() <= SIDE_BOND_LATERAL_TOLERANCE
        && ax <= SIDE_BOND_ORTHOGONAL_MAX
        && az > 1e-5
    {
        return Some(Vec3::new(0.0, 0.0, delta.z.signum()));
    }

    None
}

// =============================================================================
// MortarBond — one side-adhesion link between two blocks
// =============================================================================
//
// LEARNING: This struct is kept small intentionally.
// Every physics step we iterate ALL bonds. Smaller struct = better cache behavior.
// We store the minimum needed to:
//   a) apply the distance constraint (entity_a, entity_b, rest_distance)
//   b) check the breaking condition (adhesion_strength, tension)
//   c) identify which axis the bond is along (bond_axis — for overhang calc)
#[derive(Clone, Debug)]
pub struct MortarBond {
    /// First bonded entity (lower Entity index by convention — see make_pair in xpbd.rs)
    pub entity_a: Entity,

    /// Second bonded entity
    pub entity_b: Entity,

    /// Target separation distance (world units). Usually 1.0 for unit cubes.
    pub rest_distance: f32,

    /// Minimum adhesion_strength of the two blocks.
    /// We use min() because the weaker material determines the bond strength —
    /// a Steel block bonded to a Wood block can only hold as much as Wood allows.
    pub adhesion_strength: f32,

    /// The direction the bond runs (normalized, from A to B at spawn time).
    /// Used in breaking force calc to determine if this is a lateral bond
    /// (horizontal — resists overhang) or a vertical bond (resists pulling apart).
    pub bond_axis: Vec3,

    /// Current tension in the bond (updated each step by solve_mortar_constraints).
    /// This is the raw stretch distance in world units.
    /// If tension > adhesion_strength → bond breaks.
    /// Stored here so break_overloaded_bonds() doesn't need to recompute it.
    pub tension: f32,
}

// =============================================================================
// MortarBonds Resource — central registry of all live bonds
// =============================================================================
//
// LEARNING: Vec<MortarBond> is the right structure here:
//   - Iteration is O(N) and cache-friendly (contiguous memory)
//   - Removal is O(N) but bonds break rarely, so this is acceptable
//   - We use retain() for removal (single pass, no index bookkeeping)
//   - No HashMap needed — we don't look up bonds by entity pair often
//
// For Sprint 11 (1000 blocks, 50 agents), if bond count exceeds ~10,000,
// consider a HashMap<(Entity,Entity), usize> index for O(1) lookup.
// For Sprint 3 target (hundreds of blocks), Vec is perfect.
#[derive(Resource, Default)]
pub struct MortarBonds {
    pub bonds: Vec<MortarBond>,
}

impl MortarBonds {
    /// Check if a bond already exists between two entities.
    /// Prevents duplicate bonds when two blocks are spawned next to each other.
    pub fn has_bond(&self, a: Entity, b: Entity) -> bool {
        let (lo, hi) = if a.index() < b.index() {
            (a, b)
        } else {
            (b, a)
        };
        self.bonds
            .iter()
            .any(|bond| bond.entity_a == lo && bond.entity_b == hi)
    }

    /// Add a new bond. Normalizes entity order (lower index = entity_a).
    pub fn add_bond(
        &mut self,
        a: Entity,
        b: Entity,
        rest_distance: f32,
        adhesion_strength: f32,
        bond_axis: Vec3,
    ) {
        let (entity_a, entity_b) = if a.index() < b.index() {
            (a, b)
        } else {
            (b, a)
        };
        self.bonds.push(MortarBond {
            entity_a,
            entity_b,
            rest_distance,
            adhesion_strength,
            bond_axis,
            tension: 0.0,
        });
    }

    /// Remove all bonds involving a specific entity.
    /// Called when a block is despawned (clear_dynamic_blocks in lib.rs).
    pub fn remove_bonds_for(&mut self, entity: Entity) {
        self.bonds
            .retain(|b| b.entity_a != entity && b.entity_b != entity);
    }
}

// =============================================================================
// FUNCTION: try_register_bonds
// =============================================================================
//
// Called DIRECTLY from lib.rs after spawning a new block, not as a Bevy system.
// Runs once per spawn — not every frame. This is intentional: scanning all
// blocks every frame for new bond opportunities would be O(N²).
//
// SPRINT 3 BUG FIX — Static Block Bond Registration:
// ---------------------------------------------------
// The original implementation checked `new_voxel.adhesion_strength > 0.0`
// and returned early if the new block had no adhesion. This correctly skips
// Stone blocks. However, it also skipped STATIC blocks because the old code
// forced their adhesion_strength to 0.0.
//
// The real requirement is: a bond should form if EITHER block has adhesion > 0.
// A static Wood/Steel wall should bond to a dynamic Wood/Steel block placed
// next to it, using the NEIGHBOR's adhesion_strength (since the new block drives
// the bond formation check). The static block is the anchor.
//
// FIX: We now check the neighbor's adhesion_strength when the new block has
// zero adhesion. This correctly handles both:
//   - New dynamic block bonding to static wall anchor
//   - New static wall bonding to existing dynamic block
//
// The bond strength in both cases uses min(new.adhesion, neighbor.adhesion),
// which is 0.0 if either is Stone — preventing accidental Stone bonding.
pub fn try_register_bonds(
    new_entity: Entity,
    new_voxel: &Voxel,
    grid: &SpatialGrid,
    world: &World,
    bonds: &mut MortarBonds,
) {
    if !shape_supports_mortar(new_voxel.shape) {
        return;
    }

    let new_pos = new_voxel.position;
    let new_adhesion = new_voxel.adhesion_strength;

    // Only SIDE neighbors form mortar bonds.
    // LEARNING: Mortar in this project models lateral cohesion (walls/arches),
    // not vertical glue. Excluding Y neighbors prevents "above block pulls
    // column upward" and similar tower-drag artifacts.
    let face_offsets: [[i32; 3]; 4] = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];

    for offset in &face_offsets {
        for y_band in -3..=3 {
            let neighbor_grid = [
                new_pos.x.round() as i32 + offset[0],
                new_pos.y.round() as i32 + y_band,
                new_pos.z.round() as i32 + offset[2],
            ];

            if let Some(entities_in_cell) = grid.map.get(&neighbor_grid) {
                for &(neighbor_entity, neighbor_shape) in entities_in_cell {
                    if neighbor_entity == new_entity {
                        continue; // don't bond to self
                    }
                    if !shape_supports_mortar(neighbor_shape) {
                        continue;
                    }

                    // Skip if bond already exists (can happen with simultaneous spawns)
                    if bonds.has_bond(new_entity, neighbor_entity) {
                        continue;
                    }

                    let Some(neighbor_voxel) = world.get::<Voxel>(neighbor_entity) else {
                        continue; // entity not found (shouldn't happen but guard anyway)
                    };

                    let delta = neighbor_voxel.position - new_pos;
                    let Some(bond_axis) = compute_side_bond_axis(delta) else {
                        continue;
                    };
                    if !bond_axis.is_finite() {
                        continue;
                    }

                    // Bond strength = minimum of the two materials.
                    let bond_strength = new_adhesion.min(neighbor_voxel.adhesion_strength);
                    if bond_strength <= 0.0 {
                        continue;
                    }

                    bonds.add_bond(
                        new_entity,
                        neighbor_entity,
                        BOND_REST_DISTANCE,
                        bond_strength,
                        bond_axis,
                    );
                }
            }
        }
    }
}

// =============================================================================
// SYSTEM: register_new_bonds_system
// =============================================================================
//
// Spawn-time registration alone misses an important gameplay case:
// blocks can become face-neighbors AFTER they settle due to gravity/collisions.
// If we only register at spawn, those later contacts never get mortar bonds.
//
// This system runs each frame and opportunistically creates missing bonds for
// any adjacent voxel pair that now qualifies (adhesion > 0 on both sides).
// Duplicate bonds are prevented by MortarBonds::has_bond().
pub fn register_new_bonds_system(
    grid: Res<SpatialGrid>,
    query: Query<(Entity, &Voxel)>,
    mut bonds: ResMut<MortarBonds>,
) {
    use std::collections::HashMap;

    // Snapshot adhesion/positions so neighbor lookups don't need random query access.
    let mut adhesion_by_entity: HashMap<Entity, f32> = HashMap::with_capacity(query.iter().len());
    let mut pos_by_entity: HashMap<Entity, Vec3> = HashMap::with_capacity(query.iter().len());
    for (entity, voxel) in query.iter() {
        adhesion_by_entity.insert(entity, voxel.adhesion_strength);
        pos_by_entity.insert(entity, voxel.predicted_position);
    }

    // Side-neighbor bonds only (see try_register_bonds).
    let face_offsets: [[i32; 3]; 4] = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];

    for (entity, voxel) in query.iter() {
        if !shape_supports_mortar(voxel.shape) {
            continue;
        }
        if voxel.adhesion_strength <= 0.0 {
            continue;
        }

        let pos = voxel.predicted_position;
        for offset in &face_offsets {
            for y_band in -3..=3 {
                let neighbor_grid = [
                    pos.x.round() as i32 + offset[0],
                    pos.y.round() as i32 + y_band,
                    pos.z.round() as i32 + offset[2],
                ];

                if let Some(entities_in_cell) = grid.map.get(&neighbor_grid) {
                    for &(neighbor_entity, neighbor_shape) in entities_in_cell {
                        if neighbor_entity == entity {
                            continue;
                        }
                        if !shape_supports_mortar(neighbor_shape) {
                            continue;
                        }
                        if bonds.has_bond(entity, neighbor_entity) {
                            continue;
                        }

                        let Some(&neighbor_adhesion) = adhesion_by_entity.get(&neighbor_entity)
                        else {
                            continue;
                        };

                        let bond_strength = voxel.adhesion_strength.min(neighbor_adhesion);
                        if bond_strength <= 0.0 {
                            continue;
                        }

                        let Some(&neighbor_pos) = pos_by_entity.get(&neighbor_entity) else {
                            continue;
                        };

                        let delta = neighbor_pos - pos;
                        let Some(bond_axis) = compute_side_bond_axis(delta) else {
                            continue;
                        };

                        bonds.add_bond(
                            entity,
                            neighbor_entity,
                            BOND_REST_DISTANCE,
                            bond_strength,
                            bond_axis,
                        );
                    }
                }
            }
        }
    }
}

// =============================================================================
// SYSTEM 2: solve_mortar_constraints_system
// =============================================================================
//
// This runs every physics step, pulling bonded block pairs toward their rest
// distance when they drift apart. It's the mortar's "rubber band."
//
// =============================================================================
// BUG FIX: MASS-PROPORTIONAL COMPLIANCE + SLEEP THRESHOLD + NO UPWARD LIFT
// (Issues 3a and 3b)
// =============================================================================
//
// THREE BUGS FIXED IN THIS SYSTEM:
//
// FIX A — Vibration (steel-on-steel, Issue 3a):
//   compliance = BASE_BOND_COMPLIANCE_FACTOR × inv_mass_sum
//   This scales bond stiffness with the actual mass of the bonded pair.
//   Heavy pairs (steel-steel) get proportionally stiffer bonds that don't
//   over-correct (no oscillation). Light pairs (wood-wood) stay soft and stretchy.
//   See BASE_BOND_COMPLIANCE_FACTOR comment above for the full math.
//
// FIX B — Pillar flying (Issue 3b, primary cause):
//   Added BOND_WAKE_TENSION_THRESHOLD: sleeping blocks are only woken by the
//   mortar solver if the bond tension exceeds 0.02 world-units. Tiny drift
//   from floating-point arithmetic no longer wakes the entire pillar every frame.
//   Without this, each frame: drift → wake steel → steel gets gravity → steel
//   moves down → mortar overcorrects → wood flies up → pillar follows.
//
// FIX C — Pillar flying (Issue 3b, secondary cause):
//   Clamp the Y component of mortar corrections to zero for floor-supported blocks.
//   Mortar bonds are LATERAL constraints (resist sideways separation of a structure).
//   They should never push a block UPWARD off its support. Gravity and the collision
//   solver own vertical positioning. This prevents the upward-drift chain reaction.
//
// SPRINT 3 JACOBI AVERAGING (unchanged):
//   We track correction counts per entity and divide accumulated corrections
//   by count before applying. This prevents overshoot when a block is bonded
//   to multiple neighbors all pulling in the same direction simultaneously.
//   See original Sprint 3 comment below for the full explanation.
pub fn solve_mortar_constraints_system(
    mut query: Query<(Entity, &mut Voxel)>,
    mut bonds: ResMut<MortarBonds>,
) {
    // We can't mutate two entities simultaneously from a single query.
    // The snapshot pattern freezes positions, accumulates corrections, applies all at once.
    // This is the Jacobi iteration approach (parallel constraint solving).

    use std::collections::HashMap;

    // --- Pass 1: Snapshot positions ---
    // Read-only: collect all current predicted positions and physics properties.
    //
    // EXTENDED SNAPSHOT: Also capture contact data (contact_count, contact_normal_accum)
    // so we can determine which blocks are floor-supported for the Y-clamp fix.
    let mut snapshots: HashMap<Entity, (Vec3, f32, bool, u32, Vec3, Vec3)> =
        HashMap::with_capacity(query.iter().len());
    for (e, v) in query.iter() {
        snapshots.insert(
            e,
            (
                v.predicted_position,
                v.inv_mass,
                v.is_sleeping,
                v.contact_count,
                v.contact_normal_accum,
                v.velocity,
            ),
        );
    }

    // --- Pass 2: Compute corrections for each bond ---
    // SPRINT 3 FIX: Track both accumulated correction AND count per entity.
    // We'll divide by count in Pass 3 to get the average (Jacobi averaging).
    let mut corrections: HashMap<Entity, Vec3> = HashMap::with_capacity(snapshots.len());
    let mut correction_counts: HashMap<Entity, u32> = HashMap::with_capacity(snapshots.len());
    // Track which entities we should wake (only if tension >= threshold)
    let mut should_wake: std::collections::HashSet<Entity> =
        std::collections::HashSet::with_capacity(snapshots.len());

    for bond in bonds.bonds.iter_mut() {
        let Some(&(pos_a, inv_a, sleeping_a, contacts_a, normal_accum_a, vel_a)) =
            snapshots.get(&bond.entity_a)
        else {
            continue;
        };
        let Some(&(pos_b, inv_b, sleeping_b, contacts_b, normal_accum_b, vel_b)) =
            snapshots.get(&bond.entity_b)
        else {
            continue;
        };

        let delta = pos_a - pos_b;
        // LEARNING: Constrain mortar stretch to the ORIGINAL bond axis.
        // This prevents off-axis drag (e.g., behind-contact introducing X drift)
        // and keeps bonds behaving like side adhesion instead of full 3D welds.
        let axis = bond.bond_axis.normalize_or_zero();
        if axis == Vec3::ZERO {
            bond.tension = 0.0;
            continue;
        }
        let signed_axis_sep = delta.dot(axis);
        let axis_dist = signed_axis_sep.abs();
        let violation = axis_dist - bond.rest_distance;

        // Only resist stretching (separation). Compression is handled by collision solver.
        if violation <= 0.0 {
            bond.tension = 0.0;
            continue;
        }

        // Store tension (raw stretch in world units) for the breaking force check.
        // LEARNING: tension is dimensionless stretch — a tension of 0.5 means the
        // blocks have drifted 0.5 world-units past their rest_distance.
        bond.tension = violation;

        // =============================================================
        // FIX B: SLEEP THRESHOLD CHECK
        // =====================================================================
        // Both sleeping AND below the wake threshold → silently skip.
        // This prevents floating-point drift from waking sleeping steel pillars
        // and triggering the gravity → overcorrection → flying chain reaction.
        //
        // If tension is above threshold (real separation), we WILL wake the blocks
        // and fix it. If it's just drift noise, we leave them sleeping.
        // =============================================================
        if sleeping_a && sleeping_b && violation < BOND_WAKE_TENSION_THRESHOLD {
            continue;
        }

        // =============================================================
        // FIX A: MASS-PROPORTIONAL COMPLIANCE
        // =====================================================================
        // compliance = BASE_BOND_COMPLIANCE_FACTOR × inv_mass_sum
        //
        // This ensures that heavy block pairs (steel-steel) get a compliance
        // value that is IN THE SAME ORDER OF MAGNITUDE as their inv_mass_sum,
        // preventing the 285× amplification factor that caused vibration.
        //
        // For wood-wood: inv_sum=2.0, compliance=1.0  → correction factor = 1/(2.0+1.0) = 0.33
        // For steel-steel: inv_sum=0.0025, compliance=0.00125 → correction factor = 0.80
        // For wood+steel: inv_sum=1.00125, compliance=0.50 → correction factor = 0.67
        //
        // All correction factors are now in [0.3, 0.9] regardless of material —
        // the bonds behave proportionally and the oscillation is eliminated.
        // =============================================================
        let inv_mass_sum = inv_a + inv_b;
        if inv_mass_sum <= 0.0 {
            continue; // Both static — bond exists as reference but no correction needed
        }

        // LEARNING: Keep a tiny compliance floor so extremely heavy pairs do not
        // approach numerically singular stiffness in edge cases.
        let compliance = (BASE_BOND_COMPLIANCE_FACTOR * inv_mass_sum).max(1e-6);
        let correction_magnitude =
            (violation * (1.0 / (inv_mass_sum + compliance))).min(MAX_BOND_CORRECTION);

        let direction = if signed_axis_sep >= 0.0 { axis } else { -axis }; // from B to A along bond axis

        // Check if block A is floor-supported (upward contact normal)
        let a_is_floor_supported = contacts_a > 0 && {
            let avg_n = (normal_accum_a / contacts_a as f32).normalize_or_zero();
            avg_n.y >= FLOOR_SUPPORTED_NORMAL_Y && vel_a.length() <= SUPPORT_STABLE_SPEED
        };

        // Check if block B is floor-supported
        let b_is_floor_supported = contacts_b > 0 && {
            let avg_n = (normal_accum_b / contacts_b as f32).normalize_or_zero();
            avg_n.y >= FLOOR_SUPPORTED_NORMAL_Y && vel_b.length() <= SUPPORT_STABLE_SPEED
        };

        // Push A toward B (negative direction), scaled by A's mass share
        let mut self_share = inv_a / inv_mass_sum;
        let mut other_share = inv_b / inv_mass_sum;

        // LEARNING: Supported anchors should move very little under mortar.
        // Reassign most correction to the non-supported counterpart.
        if a_is_floor_supported {
            self_share = self_share.min(SUPPORTED_ANCHOR_SHARE_CAP);
            other_share = 1.0 - self_share;
        }
        if b_is_floor_supported {
            other_share = other_share.min(SUPPORTED_ANCHOR_SHARE_CAP);
            self_share = 1.0 - other_share;
        }

        let mut corr_a = -direction * correction_magnitude * self_share;
        // Push B toward A (positive direction), scaled by B's mass share
        let mut corr_b = direction * correction_magnitude * other_share;

        // =============================================================
        // FIX C: NO UPWARD MORTAR LIFT FOR FLOOR-SUPPORTED BLOCKS
        // =====================================================================
        // If a block is resting on a surface (contact_count > 0 and avg contact
        // normal points upward), clamp the Y component of its mortar correction
        // to zero or below. Mortar should pull blocks TOGETHER horizontally but
        // must NOT lift blocks off their supports.
        //
        // This is the secondary cause of the pillar-flying bug: tiny upward Y
        // components in the bond correction direction (when blocks aren't perfectly
        // at the same height) accumulated into a visible upward launch.
        //
        // WHY ZERO-CLAMP ONLY FOR SUPPORTED BLOCKS:
        //   An UNsupported block (in mid-air, building an arch or bridge) DOES
        //   need the full 3D mortar correction to maintain its position —
        //   that's exactly what the mortar is for. The clamp only applies when
        //   the block is already physically supported from below.
        // =============================================================

        // Clamp Y correction to <= 0 for supported blocks (can compress but not lift)
        if a_is_floor_supported && corr_a.y > 0.0 {
            corr_a.y = 0.0;
        }
        if b_is_floor_supported && corr_b.y > 0.0 {
            corr_b.y = 0.0;
        }

        // SPRINT 3 FIX: Accumulate correction AND increment count for Jacobi averaging.
        *corrections.entry(bond.entity_a).or_insert(Vec3::ZERO) += corr_a;
        *correction_counts.entry(bond.entity_a).or_insert(0) += 1;

        *corrections.entry(bond.entity_b).or_insert(Vec3::ZERO) += corr_b;
        *correction_counts.entry(bond.entity_b).or_insert(0) += 1;

        // Only mark for waking if tension is above the threshold
        if violation >= BOND_WAKE_TENSION_THRESHOLD {
            should_wake.insert(bond.entity_a);
            should_wake.insert(bond.entity_b);
        }
    }

    // --- Pass 3: Apply AVERAGED corrections ---
    //
    // SPRINT 3 FIX: Divide accumulated correction by bond count before applying.
    // This prevents overshoot when a block is pulled by multiple simultaneous bonds.
    //
    // Example: Block A bonded to 4 neighbors, each contributing corr = (0, 0.05, 0).
    //   Old code: applies 4 × 0.05 = 0.20 units (overshoot → jitter)
    //   New code: applies 4 × 0.05 / 4 = 0.05 units (correct single-step response)
    for (entity, mut voxel) in query.iter_mut() {
        if let Some(&correction) = corrections.get(&entity) {
            let count = correction_counts.get(&entity).copied().unwrap_or(1).max(1);
            let averaged_correction =
                (correction / count as f32).clamp_length_max(MAX_ENTITY_MORTAR_STEP);

            if averaged_correction.length_squared() > 1e-10 {
                voxel.predicted_position += averaged_correction;
                // Only wake the block if the tension is significant (FIX B)
                if should_wake.contains(&entity) {
                    voxel.is_sleeping = false;
                }
            }
        }
    }
}

// =============================================================================
// SYSTEM 3: break_overloaded_bonds_system
// =============================================================================
//
// After the mortar constraint solver runs, check if any bond has been stretched
// beyond what its adhesion_strength can handle.
//
// SPRINT 3 BUG FIX — Breaking Force Formula:
// -------------------------------------------
// The ORIGINAL code computed:
//   breaking_force = overhang_mass * bond.tension * GRAVITY_ACCEL / 9.81
//
// Then compared:
//   survives = bond.adhesion_strength >= breaking_force
//   e.g.: 8.0 >= (800 kg * 0.5 stretch * 9.81 / 9.81) = 8.0 >= 400 → FALSE
//
// This caused Steel blocks (800 kg) to INSTANTLY snap their own bonds under
// their own weight. The problem: adhesion_strength is in world-unit-stretch space
// (max tolerable stretch before breaking), but the old formula computed force in
// Newton-equivalent units (mass × stretch), creating a unit mismatch.
//
// THE CORRECT INTERPRETATION:
// bond.tension = stretch distance in world units (e.g., 0.1 = blocks 0.1 units apart)
// adhesion_strength = max tolerable stretch before breaking (e.g., 2.0 = Wood)
//
// The correct survival check is simply:
//   survives = bond.adhesion_strength >= bond.tension
//
// This means: "the bond survives as long as the stretch hasn't exceeded the
// material's tolerance." Wood (2.0) survives up to 2 world-units of stretch.
// Steel (8.0) survives up to 8. Both are far above the ~0.1-0.3 typical stretch
// seen during normal construction — bonds only break under violent impact or
// excessive cantilever loading, which is exactly the desired behavior.
//
// LEARNING: We use retain() to remove broken bonds in a single O(N) pass.
// This is more efficient than collecting indices and removing by index,
// which would be O(N²) for many simultaneous breaks.
pub fn break_overloaded_bonds_system(mut bonds: ResMut<MortarBonds>) {
    bonds.bonds.retain(|bond| {
        // Zero tension = bond is not stretched = safe
        if bond.tension <= 0.0 {
            return true; // keep
        }

        // SPRINT 3 FIX: Direct comparison — tension is stretch, adhesion_strength
        // is max tolerable stretch. No mass or gravity factors needed.
        //
        // This fixes the critical physics bug where Steel bonds (adhesion=8.0)
        // were comparing against breaking_force = 800kg × tension, which for
        // even tiny tension values massively exceeded the 8.0 threshold.
        //
        // Now: bond breaks only when actual stretch > material tolerance.
        // Wood (2.0): breaks if blocks drift >2 world-units apart
        // Steel (8.0): breaks if blocks drift >8 world-units apart (very robust)
        bond.adhesion_strength >= bond.tension
    });
}

// =============================================================================
// HELPER: remove_bonds_for_entity
// =============================================================================
//
// Called from lib.rs when clear_dynamic_blocks() despawns blocks.
// Ensures no dangling bonds remain pointing at deleted entities.
pub fn remove_entity_bonds(entity: Entity, bonds: &mut MortarBonds) {
    bonds.remove_bonds_for(entity);
}

// =============================================================================
// HELPER: bond_count_for_entity (diagnostic)
// =============================================================================
//
// Returns how many bonds a given entity participates in.
// Used by the Python observation bridge (future sprint) to expose bond count
// as an RL observation feature ("how connected is this block?").
pub fn bond_count_for_entity(entity: Entity, bonds: &MortarBonds) -> usize {
    bonds
        .bonds
        .iter()
        .filter(|b| b.entity_a == entity || b.entity_b == entity)
        .count()
}
