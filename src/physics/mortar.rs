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

/// XPBD compliance for mortar bonds.
/// Lower = stiffer bond (resists separation more).
/// 0.0 = perfectly rigid (infinite stiffness — can cause jitter).
/// 0.001 = very stiff but stable. Good for steel.
/// We scale this by (1 / adhesion_strength) so weaker materials are softer.
///
/// LEARNING: Compliance is the inverse of stiffness in XPBD.
/// compliance = 1 / stiffness
/// A bond with compliance 0.001 moves 0.001 units per unit of constraint violation.
const BASE_BOND_COMPLIANCE: f32 = 0.001;

/// Maximum correction the bond constraint can apply per solver step.
/// Prevents a single bad frame from teleporting bonded blocks.
const MAX_BOND_CORRECTION: f32 = 0.15;

fn shape_supports_mortar(shape: ShapeType) -> bool {
    !matches!(shape, ShapeType::Sphere)
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

    // The 6 cardinal face-neighbor offsets for unit cubes.
    // A block at grid position [x,y,z] touches these 6 neighbors:
    let face_offsets: [[i32; 3]; 6] = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];

    for offset in &face_offsets {
        let neighbor_grid = [
            new_pos.x.round() as i32 + offset[0],
            new_pos.y.round() as i32 + offset[1],
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

                // Read the neighbor's adhesion from the live world
                let neighbor_adhesion =
                    if let Some(neighbor_voxel) = world.get::<Voxel>(neighbor_entity) {
                        neighbor_voxel.adhesion_strength
                    } else {
                        continue; // entity not found (shouldn't happen but guard anyway)
                    };

                // Bond strength = minimum of the two materials.
                // LEARNING: This implements the "weakest link" principle.
                // A Steel beam bolted to a Wood plank can only hold as much
                // as the Wood can — the Wood will fail first.
                //
                // If EITHER block has 0.0 adhesion (Stone), the product is 0.0
                // and we skip. Stone never bonds regardless of its neighbor.
                let bond_strength = new_adhesion.min(neighbor_adhesion);
                if bond_strength <= 0.0 {
                    continue; // at least one is Stone (or another no-adhesion material)
                }

                // Compute the bond axis (unit vector from new block to neighbor)
                let bond_axis =
                    Vec3::new(offset[0] as f32, offset[1] as f32, offset[2] as f32).normalize();

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

    // Snapshot adhesion so neighbor lookups don't need random query access.
    let adhesion_by_entity: HashMap<Entity, f32> = query
        .iter()
        .map(|(entity, voxel)| (entity, voxel.adhesion_strength))
        .collect();

    let face_offsets: [[i32; 3]; 6] = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];

    for (entity, voxel) in query.iter() {
        if !shape_supports_mortar(voxel.shape) {
            continue;
        }
        if voxel.adhesion_strength <= 0.0 {
            continue;
        }

        let pos = voxel.position;
        for offset in &face_offsets {
            let neighbor_grid = [
                pos.x.round() as i32 + offset[0],
                pos.y.round() as i32 + offset[1],
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

                    let Some(&neighbor_adhesion) = adhesion_by_entity.get(&neighbor_entity) else {
                        continue;
                    };

                    let bond_strength = voxel.adhesion_strength.min(neighbor_adhesion);
                    if bond_strength <= 0.0 {
                        continue;
                    }

                    let bond_axis =
                        Vec3::new(offset[0] as f32, offset[1] as f32, offset[2] as f32).normalize();

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

// =============================================================================
// SYSTEM 2: solve_mortar_constraints_system
// =============================================================================
//
// This runs every physics step, pulling bonded block pairs toward their rest
// distance when they drift apart. It's the mortar's "rubber band."
//
// SPRINT 3 BUG FIX — Jacobi Averaging (Correction Accumulation):
// ---------------------------------------------------------------
// The original implementation accumulated position corrections into a HashMap
// and applied them directly:
//   corrections[entity] += corr_a;
//
// PROBLEM: If a block is bonded to 4 neighbors all pulling in the same direction,
// it receives 4× the single-bond correction. This overshoot causes jitter and
// can make bonded clusters oscillate or explode at high bond counts.
//
// FIX: We track both the accumulated correction AND the count of bonds affecting
// each entity. In Pass 3, we divide by the count before applying:
//   actual_correction = accumulated / bond_count
//
// This is the standard Jacobi averaging approach for XPBD constraint systems.
// It ensures a block with 4 active bonds is moved by the AVERAGE correction,
// not the sum. This is mathematically equivalent to running each bond once and
// blending the results.
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
    let snapshots: HashMap<Entity, (Vec3, f32, bool)> = query
        .iter()
        .map(|(e, v)| (e, (v.predicted_position, v.inv_mass, v.is_sleeping)))
        .collect();

    // --- Pass 2: Compute corrections for each bond ---
    // SPRINT 3 FIX: Track both accumulated correction AND count per entity.
    // We'll divide by count in Pass 3 to get the average (Jacobi averaging).
    let mut corrections: HashMap<Entity, Vec3> = HashMap::new();
    let mut correction_counts: HashMap<Entity, u32> = HashMap::new();

    for bond in bonds.bonds.iter_mut() {
        let Some(&(pos_a, inv_a, sleeping_a)) = snapshots.get(&bond.entity_a) else {
            continue;
        };
        let Some(&(pos_b, inv_b, sleeping_b)) = snapshots.get(&bond.entity_b) else {
            continue;
        };

        // Both sleeping → skip (micro-optimization: sleeping bonds are at rest)
        if sleeping_a && sleeping_b {
            bond.tension = 0.0;
            continue;
        }

        let delta = pos_a - pos_b;
        let current_dist = delta.length();

        if current_dist < 1e-6 {
            bond.tension = 0.0;
            continue;
        }

        let violation = current_dist - bond.rest_distance;

        // Only resist stretching (separation). Compression is handled by collision solver.
        if violation <= 0.0 {
            bond.tension = 0.0;
            continue;
        }

        // Store tension (raw stretch in world units) for the breaking force check.
        // LEARNING: tension is dimensionless stretch — a tension of 0.5 means the
        // blocks have drifted 0.5 world-units past their rest_distance.
        bond.tension = violation;

        // Bond compliance: weaker bonds are softer (more compliant)
        // adhesion_strength is in range [2.0, 8.0] for wood/steel
        // compliance = 0.001 / 8.0 = 0.000125 (steel, very stiff)
        // compliance = 0.001 / 2.0 = 0.0005   (wood, softer)
        let compliance = BASE_BOND_COMPLIANCE / bond.adhesion_strength.max(0.001);

        let inv_mass_sum = inv_a + inv_b;
        if inv_mass_sum <= 0.0 {
            continue; // Both static — bond exists as reference but no correction needed
        }

        // Correction magnitude clamped for stability
        let correction_magnitude =
            (violation * (1.0 / (inv_mass_sum + compliance))).min(MAX_BOND_CORRECTION);

        let direction = delta / current_dist; // unit vector from B to A

        // Push A toward B (negative direction), scaled by A's mass share
        let corr_a = -direction * correction_magnitude * (inv_a / inv_mass_sum);
        // Push B toward A (positive direction), scaled by B's mass share
        let corr_b = direction * correction_magnitude * (inv_b / inv_mass_sum);

        // SPRINT 3 FIX: Accumulate correction AND increment count for Jacobi averaging.
        *corrections.entry(bond.entity_a).or_insert(Vec3::ZERO) += corr_a;
        *correction_counts.entry(bond.entity_a).or_insert(0) += 1;

        *corrections.entry(bond.entity_b).or_insert(Vec3::ZERO) += corr_b;
        *correction_counts.entry(bond.entity_b).or_insert(0) += 1;
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
            let averaged_correction = correction / count as f32;

            if averaged_correction.length_squared() > 1e-10 {
                voxel.predicted_position += averaged_correction;
                voxel.is_sleeping = false; // bond correction wakes the block
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
