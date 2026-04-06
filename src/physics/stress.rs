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
// We use a THREE-PASS approach each frame:
//
//   PASS 1 — Build the load graph:
//     For every block, find which blocks are directly above it (within 1.5 units).
//     A block "supports" another if it sits beneath it and they are in contact.
//
//   PASS 2 — Propagate weight downward (TOP-DOWN):
//     Sort entities highest Y first. Process in that order.
//     Each block's supported_mass = sum of (mass + supported_mass) for all blocks
//     it directly supports above it. Correct because when we process block B,
//     all blocks ABOVE B have already been processed and their loads computed.
//
//   PASS 3 — Propagate floor support upward (BOTTOM-UP): ← SPRINT 4.5 FIX
//     Sort entities lowest Y first. Process in that order.
//     A block is floor-supported if its Y is near the ground, OR any block
//     directly below it is floor-supported. Correct because when we process
//     block B at y=1.0, the block at y=0.0 below it has ALREADY been processed
//     (bottom-up order) and its is_supported_from_below = true. So B correctly
//     inherits the floor-support flag.
//
// =============================================================================
// SPRINT 4.5 FIX: Floor Support Propagation Bug
// ===============================================
//
// THE OLD BUG (test: test_floor_support_flag_propagates, 3/4 blocks unsupported):
//
//   The original code tried to compute floor support INLINE during Pass 2
//   (top-down traversal). When processing block B at y=1.0, it looked for
//   block A at y=0.0 in the spatial grid and checked is_floor_supported[A].
//
//   But A HASN'T been processed yet! Pass 2 processes blocks top-down, so
//   block A (lowest Y) comes LAST. When B checks A, A's floor-support flag
//   is still false (just initialized). So B is marked "not floor-supported".
//
//   Result: For a 4-block pillar [y=0, y=1, y=2, y=3]:
//     Pass 2 order: y=3, y=2, y=1, y=0
//     y=3: no blocks above, check below... y=2 not processed yet → unsupported
//     y=2: check below... y=1 not processed yet → unsupported
//     y=1: check below... y=0 not processed yet → unsupported
//     y=0: y <= FLOOR_SUPPORTED_Y_MAX → floor_supported = true ✓
//     Only 1/4 blocks floor-supported. Test failed with 3/4 unsupported.
//
// THE FIX: Separate Pass 3 with BOTTOM-UP traversal.
//
//   Pass 3 order (lowest Y first): y=0, y=1, y=2, y=3
//   y=0: y <= FLOOR_SUPPORTED_Y_MAX → floor_supported = true (base case)
//   y=1: lookup y=0 → is_floor_supported=true → y=1 is also floor-supported ✓
//   y=2: lookup y=1 → is_floor_supported=true → y=2 is also floor-supported ✓
//   y=3: lookup y=2 → is_floor_supported=true → y=3 is also floor-supported ✓
//   All 4 blocks correctly floor-supported. Test passes.
//
// LEARNING: This is a classic "topological sort direction" problem in graph theory.
//
//   The load graph has edges pointing DOWN (weight flows from top to bottom).
//   To propagate load DOWNWARD correctly, process TOP-DOWN (so parents are
//   computed before children in the load-flow direction).
//
//   The support graph has edges pointing UP (support flows from ground to top).
//   To propagate support UPWARD correctly, process BOTTOM-UP (so the ground-
//   level blocks are processed first, then each block inherits from below).
//
//   These two propagation directions are OPPOSITES, requiring separate passes.
//   Trying to do both in a single pass in either direction will get one wrong.
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
const SUPPORT_XZ_MAX_OFFSET: f32 = 0.65;

// =============================================================================
// LEARNING TOPIC: Per-Material Stress Capacity
// ============================================
// The stress capacity multiplier defines how many times a block's own weight
// it can support before we color it as "critical red" in the heatmap.
//
// These values are physically informed but tuned for visual clarity:
//   - Steel (800 kg/block): strong, but finite in gameplay scale (24×)
//   - Wood (1 kg/block):    moderate, rated at 15× own weight
//   - Stone (50 kg/block):  good in compression, rated at 30× own weight
//   - Scaffold (0.5 kg/block): intentionally weak, only 5× own weight —
//     scaffold turns red quickly to signal "you need to remove me"
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
// never during collision solving.
#[derive(Clone, Debug, Default)]
pub struct StressInfo {
    /// Total mass this block supports above it (in kg, not including own mass).
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
//   - We clear and rebuild the entire map every frame anyway
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
// LEARNING TOPIC: Three-Pass Load Propagation Algorithm
// ---------------------------------------------------
//
// This system runs ONCE PER FRAME, after update_velocities_system has committed
// all final positions. It performs three passes:
//
// PASS 1 — Support Graph Construction:
//   For each block B, check the 3×3 column of grid cells directly above B.
//   If a block A is found there with the right Y-separation and XZ proximity,
//   record that "B supports A" (B bears A's weight).
//   Result: HashMap<Entity, Vec<Entity>> = "what does each block support above it?"
//
// PASS 2 — Downward Load Propagation (TOP-DOWN):
//   Sort all blocks by Y position (highest first).
//   For each block in top-down order:
//     block.supported_mass = sum over each block it supports of
//       (block_above.own_mass + block_above.supported_mass)
//   Processing top-down guarantees that when we reach block B, ALL blocks
//   above B have already computed their supported_mass.
//
// PASS 3 — Floor Support Propagation (BOTTOM-UP): ← SPRINT 4.5 FIX
//   Sort all blocks by Y position (lowest first).
//   For each block in bottom-up order:
//     If y <= FLOOR_SUPPORTED_Y_MAX → floor_supported = true (direct floor contact)
//     Else: check spatial grid for blocks directly below → inherit their flag
//   Processing bottom-up guarantees that when we check block B, all blocks
//   BELOW B have already been processed and their floor support flags are correct.
//
// NORMALIZATION:
//   stress_normalized = supported_mass / (own_mass × capacity_multiplier)
//   Clamped to [0.0, 1.0] for the shader.
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
    // Map: entity → list of blocks it directly supports (blocks above it)
    let mut supports_above: HashMap<Entity, Vec<Entity>> = HashMap::with_capacity(block_count);

    // Snapshot: entity → (position, inv_mass, material)
    // We snapshot here so pass 2 and 3 don't need to re-query the ECS.
    let mut block_snapshot: HashMap<Entity, (Vec3, f32, MaterialType)> =
        HashMap::with_capacity(block_count);

    for (entity, voxel) in query.iter() {
        block_snapshot.insert(entity, (voxel.position, voxel.inv_mass, voxel.material));
        supports_above.insert(entity, Vec::new());
    }

    // For each block, look ABOVE it in the spatial grid to find what it supports.
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
    // PASS 2: Propagate load downward (TOP-DOWN traversal)
    // =========================================================================
    //
    // LEARNING: Sort entities by Y position DESCENDING (highest first).
    //
    // This is the KEY insight for load propagation:
    //   When we process block B, every block ABOVE B (which B supports) has
    //   already had its supported_mass computed. B can sum them all correctly.
    //
    // Example for a 3-block pillar (A on top, B in middle, C on bottom):
    //   Top-down order: A, B, C
    //   A: supported_mass = 0 (nothing above A)
    //   B: supported_mass = mass_A + A.supported_mass = mass_A + 0 = mass_A
    //   C: supported_mass = mass_B + B.supported_mass = mass_B + mass_A
    //   ✓ C correctly bears the total weight of A+B above it.
    let mut sorted_by_y_desc: Vec<(Entity, f32)> = block_snapshot
        .iter()
        .map(|(e, (pos, _, _))| (*e, pos.y))
        .collect();
    // Sort highest first — use total_cmp to avoid NaN panics
    sorted_by_y_desc.sort_by(|a, b| b.1.total_cmp(&a.1));

    // supported_mass[entity] = total mass this block supports above it
    let mut supported_mass: HashMap<Entity, f32> = HashMap::with_capacity(block_snapshot.len());
    for (entity, _) in &sorted_by_y_desc {
        supported_mass.insert(*entity, 0.0);
    }

    // Y threshold for "resting directly on the ground" (base case for floor support)
    const FLOOR_SUPPORTED_Y_MAX: f32 = 0.75;

    for (entity_b, _y) in &sorted_by_y_desc {
        let Some((pos_b, inv_mass_b, material_b)) = block_snapshot.get(entity_b) else {
            continue;
        };

        // Own mass from inv_mass: mass = 1 / inv_mass (static blocks have inv_mass=0)
        let own_mass_b = if *inv_mass_b > 0.0 {
            1.0 / inv_mass_b
        } else {
            // Static blocks: treat as having their material's mass for stress purposes
            1.0 / material_b.inv_mass().max(1e-6)
        };

        // Sum up the load from all blocks this block directly supports above it.
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

        // Compute normalized stress for the heatmap.
        let capacity = own_mass_b * stress_capacity_for_material(*material_b);
        let stress_normalized = if capacity > 0.0 {
            (load_from_above / capacity).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Insert with base-case floor support (near the ground = directly supported).
        // Pass 3 will correct this for blocks higher up in the stack.
        let is_at_floor = pos_b.y <= FLOOR_SUPPORTED_Y_MAX;
        stress_map.data.insert(
            *entity_b,
            StressInfo {
                supported_mass: load_from_above,
                stress_normalized,
                is_supported_from_below: is_at_floor, // Pass 3 will update non-floor blocks
            },
        );
    }

    // =========================================================================
    // PASS 3: Propagate floor support UPWARD (BOTTOM-UP traversal)
    // =========================================================================
    //
    // SPRINT 4.5 FIX: This is the critical fix for test_floor_support_flag_propagates.
    //
    // LEARNING: Why must this be a SEPARATE bottom-up pass?
    //
    //   If we try to compute floor support DURING Pass 2 (top-down), we look
    //   for blocks below us to check their floor-support flag. But those blocks
    //   come LATER in the top-down order and haven't been processed yet.
    //   Their is_supported_from_below is still false (just initialized).
    //   So ALL blocks except the bottom one would be marked "not supported".
    //
    //   By doing a SEPARATE bottom-up pass:
    //     Order: y=0, y=1, y=2, y=3 (lowest to highest)
    //     y=0: y <= 0.75 → floor_supported = true (set in Pass 2 base case)
    //     y=1: look below at y=0 → is_supported_from_below=true → set true ✓
    //     y=2: look below at y=1 → is_supported_from_below=true → set true ✓
    //     y=3: look below at y=2 → is_supported_from_below=true → set true ✓
    //
    // Build a sorted-by-Y-ascending list for bottom-up traversal
    let mut sorted_by_y_asc: Vec<(Entity, f32)> = block_snapshot
        .iter()
        .map(|(e, (pos, _, _))| (*e, pos.y))
        .collect();
    // Sort lowest first
    sorted_by_y_asc.sort_by(|a, b| a.1.total_cmp(&b.1));

    for (entity_b, _y) in &sorted_by_y_asc {
        // Skip blocks already marked floor-supported (the base case, set in Pass 2).
        // These are blocks with y <= FLOOR_SUPPORTED_Y_MAX. They don't need to look
        // below themselves — they're already at the ground.
        if stress_map
            .data
            .get(entity_b)
            .map(|s| s.is_supported_from_below)
            .unwrap_or(false)
        {
            continue;
        }

        let Some((pos_b, _, _)) = block_snapshot.get(entity_b) else {
            continue;
        };

        // Look at grid cells directly BELOW this block to find floor-supported neighbors.
        // LEARNING: We check the cell one Y unit below (grid_b[1] - 1) and a ±1 XZ
        // neighborhood because grid rounding means the block below might not be in
        // exactly the same XZ cell.
        let grid_b = SpatialGrid::world_to_grid(*pos_b);
        let mut found_support = false;

        'outer: for dx in -1..=1i32 {
            for dz in -1..=1i32 {
                let below_cell = [grid_b[0] + dx, grid_b[1] - 1, grid_b[2] + dz];
                if let Some(neighbors) = grid.map.get(&below_cell) {
                    for (entity_below, _shape) in neighbors {
                        if entity_below == entity_b {
                            continue;
                        }

                        let Some((pos_below, _, _)) = block_snapshot.get(entity_below) else {
                            continue;
                        };

                        // Verify the block below is actually below us (not just in the
                        // same grid cell due to rounding).
                        let dy = pos_b.y - pos_below.y;
                        if dy >= SUPPORT_Y_SEPARATION_MIN && dy <= SUPPORT_Y_SEPARATION_MAX {
                            // LEARNING: Because we process bottom-up, entity_below was
                            // already processed in this pass. Its is_supported_from_below
                            // is CORRECTLY SET (either true from base case or true from
                            // a previous iteration of this loop). So this lookup is valid.
                            if stress_map
                                .data
                                .get(entity_below)
                                .map(|s| s.is_supported_from_below)
                                .unwrap_or(false)
                            {
                                found_support = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        // Propagate the support flag upward through the stack.
        if found_support {
            if let Some(info) = stress_map.data.get_mut(entity_b) {
                info.is_supported_from_below = true;
            }
        }
    }
}
