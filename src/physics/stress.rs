// =============================================================================
// src/physics/stress.rs  —  Operation Babel: Structural Stress Calculation
// =============================================================================
//
// SPRINT 4: Stress Calculation System
//
// LEARNING TOPIC: What Is Structural Stress?
// -------------------------------------------
// In the real world, every block in a structure bears LOAD — the total weight
// of everything above it plus any lateral (sideways) forces from overhangs.
//
// There are two kinds of stress we care about:
//
//   COMPRESSION STRESS: Vertical squashing force.
//     A block at the bottom of a 10-block pillar supports the weight of
//     all 9 blocks above it. If each block weighs 1kg, the bottom block
//     is under 9× more compression than the top block.
//     Formula: compression = Σ(mass_of_all_blocks_above) × gravity
//
//   SHEAR STRESS: Lateral (sideways) force from overhanging blocks.
//     An arch's keystone is pulled sideways by the weight × horizontal
//     distance from the nearest vertical support. A block cantilevered
//     3 units out from a wall is under 3× more shear than a block 1 unit out.
//     Formula: shear = mass × gravity × horizontal_distance_from_support
//
// LEARNING TOPIC: How We Propagate Stress Through the Structure
// -------------------------------------------------------------
// We use a TWO-PASS approach each frame:
//
//   PASS 1 — Build the load graph:
//     For every block, find which blocks are directly above it (within 1.5 units,
//     with a net upward contact normal from the lower block's perspective).
//     A block "supports" another if it sits beneath it and they are in contact.
//
//   PASS 2 — Propagate weight downward:
//     Starting from the highest blocks (no blocks above them, weight = own mass),
//     propagate accumulated load downward. Each block receives the sum of all
//     loads from the blocks it directly supports, plus its own weight.
//
// WHY THIS ORDER:
//     Processing top-down ensures that when we process block B, all blocks
//     ABOVE B have already been assigned their load. B's load = own_weight +
//     sum(load of all blocks B directly supports).
//
// LEARNING TOPIC: Why Not Just Use contact_count?
// ------------------------------------------------
// contact_count from the XPBD solver tells us HOW MANY contacts a block has,
// but not the DIRECTION (above vs below vs side) or the MAGNITUDE of force.
// We need to actually trace the support graph — who is sitting on top of whom —
// to correctly assign compression loads.
//
// The spatial grid gives us O(1) neighbor lookup, so the full graph traversal
// is O(N) where N = number of blocks.
//
// LEARNING TOPIC: Normalized Stress for Rendering
// ------------------------------------------------
// Raw stress is in kg (total supported mass). For the heatmap shader we need
// a value in [0.0, 1.0] where:
//   0.0 = perfectly safe (no load beyond own weight)
//   0.5 = approaching limit (structure is stressed but stable)
//   1.0 = critical (at or beyond the material's rated capacity)
//
// Normalization: stress_normalized = compression / (mass × STRESS_CAPACITY_MULTIPLIER)
//   where STRESS_CAPACITY_MULTIPLIER is how many times its own weight a block
//   can safely support before reaching critical stress.
//
// WHY PER-MATERIAL CAPACITY:
//   Steel can support 400× its own weight (structural steel is incredibly strong).
//   Wood can support ~10× its own weight before risk of failure.
//   Stone can support ~20× its own weight in pure compression.
//   These ratios produce meaningful visual differentiation in the heatmap.
//
// =============================================================================

use crate::world::spatial_grid::SpatialGrid;
use crate::world::voxel::{MaterialType, Voxel};
use bevy_ecs::prelude::*;
use glam::Vec3;
use std::collections::HashMap;

// =============================================================================
// CONSTANTS
// =============================================================================

/// The Y-separation threshold for classifying one block as "directly above" another.
/// A block at y=1.5 is "above" a block at y=0.5 (separation = 1.0 = one block width).
/// We use 1.4 as the threshold: close enough to be in contact, but not so tight
/// that floating-point drift causes misclassification.
const SUPPORT_Y_SEPARATION_MIN: f32 = 0.6;
const SUPPORT_Y_SEPARATION_MAX: f32 = 1.4;

/// Maximum horizontal offset for a block to still count as "directly above".
/// A block sitting precisely on top should have X,Z separation < 0.6 units.
/// This prevents diagonal neighbors from being counted as "above".
const SUPPORT_XZ_MAX_OFFSET: f32 = 0.65;

// =============================================================================
// LEARNING TOPIC: Per-Material Stress Capacity
// ============================================
// The stress capacity multiplier defines how many times a block's own weight
// it can support before we color it as "critical red" in the heatmap.
//
// These values are physically informed but tuned for visual clarity:
//   - Steel (800 kg/block): strong, but finite in gameplay scale (24x)
//   - Wood (1 kg/block):    moderate, rated at 15× own weight
//   - Stone (50 kg/block):  good in compression, rated at 30× own weight
//   - Scaffold (0.5 kg/block): intentionally weak, only 5× own weight — scaffold
//     turns red quickly to signal "you need to remove me before I fails"
//
// LEARNING: Real structural engineering uses safety factors of 2-10× the
// working load. We use larger values here because our "blocks" are massive
// simplified objects, not individual atoms of material.
// =============================================================================
fn stress_capacity_for_material(material: MaterialType) -> f32 {
    match material {
        MaterialType::Steel => 24.0, // strong, but should still show load under tall pillars
        MaterialType::Wood => 15.0,  // wood fails relatively easily
        MaterialType::Stone => 30.0, // stone is strong in pure compression
        MaterialType::Scaffold => 5.0, // scaffold turns red fast — a visual warning
    }
}

// =============================================================================
// StressInfo: per-block stress data computed each frame
// =============================================================================
//
// LEARNING: We store stress separately from the Voxel component to keep the
// hot physics path (XPBD solver) from carrying extra data it doesn't need.
// Stress is read ONLY by the rendering system and the reward function —
// never during collision solving. Keeping it separate allows the stress
// system to run AFTER physics commits positions, with no interference.
#[derive(Clone, Debug, Default)]
pub struct StressInfo {
    /// Total mass this block supports above it (in kg, not including own mass).
    /// Used for compression stress calculation.
    pub supported_mass: f32,

    /// Normalized stress value in [0.0, 1.0] for the heatmap shader.
    ///   0.0 = green (safe)
    ///   0.5 = yellow (stressed)
    ///   1.0 = red (critical / failure imminent)
    pub stress_normalized: f32,

    /// Whether this block is directly floor-supported (rests on ground or
    /// on a floor-supported block). Non-supported blocks in mid-air are
    /// colored differently (they rely entirely on mortar bonds).
    pub is_supported_from_below: bool,
}

// =============================================================================
// StressMap Resource — central registry of all per-block stress data
// =============================================================================
//
// LEARNING: Using a Resource (global singleton) instead of a Component lets
// the stress system write results AFTER the physics systems have committed
// their final positions, without needing to interleave with the XPBD solver.
// The render system then reads from StressMap to color each block.
//
// We use HashMap<Entity, StressInfo> because:
//   - Blocks can be despawned at any time; a dense Vec would have holes
//   - Lookup by Entity is O(1) average — fast enough for the render loop
//   - We clear and rebuild the entire map every frame anyway (stress changes
//     every step as the structure evolves)
#[derive(Resource)]
pub struct StressMap {
    pub data: HashMap<Entity, StressInfo>,
    pub frame_counter: u64,
    pub min_update_interval: u64,
}

impl Default for StressMap {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
            frame_counter: 0,
            min_update_interval: 1,
        }
    }
}

// =============================================================================
// SYSTEM: compute_stress_system
// =============================================================================
//
// LEARNING TOPIC: Two-Pass Load Propagation Algorithm
// ---------------------------------------------------
//
// This system runs ONCE PER FRAME, after update_velocities_system has committed
// all final positions. It performs two passes:
//
// PASS 1 — Support Graph Construction:
//   For each block B, check the 3×3 column of grid cells directly above B.
//   If a block A is found there with the right Y-separation and XZ proximity,
//   record that "B supports A" (B bears A's weight).
//   Result: HashMap<Entity, Vec<Entity>> = "what does each block support above it?"
//
// PASS 2 — Downward Load Propagation:
//   Sort all blocks by Y position (highest first).
//   For each block in top-down order:
//     block.supported_mass = own_mass + sum(children.supported_mass + children.own_mass)
//
//   Processing top-down guarantees that when we reach block B, ALL blocks
//   above B (its "children" in the load tree) have already computed their
//   supported_mass. B can then sum them all up correctly.
//
// NORMALIZATION:
//   stress_normalized = supported_mass / (own_mass × capacity_multiplier)
//   Clamped to [0.0, 1.0] for the shader.
//
// FLOOR SUPPORT:
//   A block is floor-supported if it rests within SUPPORT_Y_SEPARATION_MAX
//   of the floor (y ≈ 0.5) OR if any block below it is floor-supported.
//   This propagates upward through the stack.
pub fn compute_stress_system(
    query: Query<(Entity, &Voxel)>,
    grid: Res<SpatialGrid>,
    mut stress_map: ResMut<StressMap>,
) {
    let block_count = query.iter().count();
    if block_count == 0 {
        stress_map.data.clear();
        return;
    }

    // Adaptive cadence keeps stress heatmap responsive for small scenes while
    // reducing overhead in high-block-count worlds.
    stress_map.frame_counter = stress_map.frame_counter.wrapping_add(1);
    let adaptive_interval = if block_count >= 400 {
        4
    } else if block_count >= 200 {
        2
    } else {
        1
    };
    let interval = stress_map.min_update_interval.max(adaptive_interval).max(1);
    if stress_map.frame_counter % interval != 0 {
        return;
    }

    // Clear previous frame's stress data.
    // LEARNING: clear() retains HashMap capacity (no reallocation), so after
    // the first frame this is a fast O(N) element-drop with zero heap ops.
    stress_map.data.clear();

    // =========================================================================
    // PASS 1: Build the support graph
    // =========================================================================
    //
    // LEARNING: We build two maps simultaneously:
    //   supports_above[B] = list of blocks A that B directly supports (A is above B)
    //   block_data[entity] = (position, mass, material) snapshot for pass 2
    //
    // We snapshot block data here so pass 2 doesn't need to re-query the ECS.
    // This is the same "freeze then process" pattern used in the XPBD solver.

    // Map: entity → blocks it directly supports above it
    let mut supports_above: HashMap<Entity, Vec<Entity>> = HashMap::with_capacity(block_count);

    // Snapshot: entity → (position, inv_mass, material, is_dynamic)
    let mut block_snapshot: HashMap<Entity, (Vec3, f32, MaterialType)> =
        HashMap::with_capacity(block_count);

    for (entity, voxel) in query.iter() {
        block_snapshot.insert(entity, (voxel.position, voxel.inv_mass, voxel.material));
        supports_above.insert(entity, Vec::new());
    }

    // For each block, look ABOVE it in the spatial grid to find what it supports.
    // LEARNING: We check grid cells at Y+1 relative to this block's rounded grid Y.
    // The 3×3 XZ neighborhood (center ± 1) catches slightly off-center stacks.
    for (entity_b, (pos_b, _, _)) in &block_snapshot {
        let grid_b = SpatialGrid::world_to_grid(*pos_b);

        // Check the column directly above this block (same XZ, Y+1)
        for dx in -1..=1i32 {
            for dz in -1..=1i32 {
                let check_cell = [grid_b[0] + dx, grid_b[1] + 1, grid_b[2] + dz];

                if let Some(neighbors) = grid.map.get(&check_cell) {
                    for (entity_a, _shape) in neighbors {
                        if entity_a == entity_b {
                            continue;
                        }

                        let Some((pos_a, _, _)) = block_snapshot.get(entity_a) else {
                            continue;
                        };

                        // Verify A is genuinely above B with the right separation.
                        // LEARNING: We check the actual world-space Y separation
                        // rather than trusting the grid cell alone, because grid cells
                        // round positions and a block at y=0.999 and y=1.001 are
                        // both in the "y=1" cell but are actually 0.002 apart.
                        let dy = pos_a.y - pos_b.y;
                        let dx_world = (pos_a.x - pos_b.x).abs();
                        let dz_world = (pos_a.z - pos_b.z).abs();

                        if dy >= SUPPORT_Y_SEPARATION_MIN
                            && dy <= SUPPORT_Y_SEPARATION_MAX
                            && dx_world <= SUPPORT_XZ_MAX_OFFSET
                            && dz_world <= SUPPORT_XZ_MAX_OFFSET
                        {
                            // B supports A (A is directly above B)
                            if let Some(list) = supports_above.get_mut(entity_b) {
                                list.push(*entity_a);
                            }
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // PASS 2: Propagate load downward (top-down traversal)
    // =========================================================================
    //
    // LEARNING: Sort entities by Y position descending (highest first).
    // This is the KEY insight: when we process block B, every block above B
    // has already had its supported_mass computed. B can sum them all up
    // in a single pass with no recursion.
    //
    // Example for a 3-block pillar (A on top, B in middle, C on bottom):
    //   Order: A, B, C
    //   A: supported_mass = 0 (nothing above A)
    //   B: supported_mass = mass_A + A.supported_mass = mass_A + 0 = mass_A
    //   C: supported_mass = mass_B + B.supported_mass = mass_B + mass_A
    //
    // This correctly gives C the total load of A+B above it.

    let mut sorted_entities: Vec<(Entity, f32)> = block_snapshot
        .iter()
        .map(|(e, (pos, _, _))| (*e, pos.y))
        .collect();
    // Sort highest Y first (top-down traversal)
    // LEARNING: Use total_cmp for floats so ordering remains valid even when
    // NaN sneaks in from unstable transient states. partial_cmp + fallback can
    // violate strict weak ordering and panic inside Rust's sort implementation.
    sorted_entities.sort_by(|a, b| b.1.total_cmp(&a.1));

    // supported_mass[entity] = total mass this block supports above it
    let mut supported_mass: HashMap<Entity, f32> = HashMap::with_capacity(block_snapshot.len());

    // Initialize all blocks to 0 supported mass
    for (entity, _) in &sorted_entities {
        supported_mass.insert(*entity, 0.0);
    }

    // Floor Y threshold: blocks within this height of the ground are floor-supported.
    // A block at y=0.5 is resting directly on the ground (ground is at y=-0.5, block
    // center is at 0.5, bottom face at 0.0 which is above the floor).
    const FLOOR_SUPPORTED_Y_MAX: f32 = 0.75;

    // Track which entities are floor-supported (ground or stacked on ground-supported block)
    let mut is_floor_supported: HashMap<Entity, bool> =
        HashMap::with_capacity(block_snapshot.len());

    // Process top-down
    for (entity_b, _y) in &sorted_entities {
        let Some((pos_b, inv_mass_b, material_b)) = block_snapshot.get(entity_b) else {
            continue;
        };

        // Own mass from inv_mass: mass = 1 / inv_mass (static blocks have inv_mass=0)
        let own_mass_b = if *inv_mass_b > 0.0 {
            1.0 / inv_mass_b
        } else {
            // Static blocks: treat as having their material's mass for stress purposes
            // (they don't receive stress — they ARE the foundation)
            1.0 / material_b.inv_mass().max(1e-6)
        };

        // Sum up the load from all blocks this block directly supports above it.
        // LEARNING: Because we process top-down, supported_mass[A] is already
        // finalized when we reach B. This makes the sum correct in one pass.
        let mut load_from_above = 0.0f32;
        if let Some(supported) = supports_above.get(entity_b) {
            for entity_a in supported {
                let mass_a = if let Some((_, inv_a, material_a)) = block_snapshot.get(entity_a) {
                    if *inv_a > 0.0 {
                        1.0 / inv_a
                    } else {
                        1.0 / material_a.inv_mass().max(1e-6)
                    }
                } else {
                    0.0
                };
                let load_a = supported_mass.get(entity_a).copied().unwrap_or(0.0);
                // B bears A's own weight PLUS A's supported load
                load_from_above += mass_a + load_a;
            }
        }

        supported_mass.insert(*entity_b, load_from_above);

        // Determine floor support.
        // LEARNING: A block is floor-supported if:
        //   1. Its Y position is near the ground (y < FLOOR_SUPPORTED_Y_MAX), OR
        //   2. Any block directly below it is floor-supported.
        // This propagates the "supported" flag upward through stacks.
        let self_on_floor = pos_b.y <= FLOOR_SUPPORTED_Y_MAX;
        let below_is_supported = if self_on_floor {
            false
        } else {
            let grid_b = SpatialGrid::world_to_grid(*pos_b);
            let mut found_support = false;
            'outer: for dx in -1..=1i32 {
                for dz in -1..=1i32 {
                    let below_cell = [grid_b[0] + dx, grid_b[1] - 1, grid_b[2] + dz];
                    if let Some(neighbors) = grid.map.get(&below_cell) {
                        for (entity_below, _) in neighbors {
                            if is_floor_supported
                                .get(entity_below)
                                .copied()
                                .unwrap_or(false)
                            {
                                found_support = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
            found_support
        };
        is_floor_supported.insert(*entity_b, self_on_floor || below_is_supported);

        // Compute normalized stress for the heatmap.
        // LEARNING: stress_normalized = supported_mass / rated_capacity
        //   rated_capacity = own_mass × capacity_multiplier
        //   This means the heatmap turns red when a block supports
        //   capacity_multiplier × its own weight.
        let capacity = own_mass_b * stress_capacity_for_material(*material_b);
        let stress_normalized = if capacity > 0.0 {
            (load_from_above / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };

        stress_map.data.insert(
            *entity_b,
            StressInfo {
                supported_mass: load_from_above,
                stress_normalized,
                is_supported_from_below: self_on_floor || below_is_supported,
            },
        );
    }
}
