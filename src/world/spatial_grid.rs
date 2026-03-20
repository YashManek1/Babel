// =============================================================================
// src/world/spatial_grid.rs  —  Operation Babel: O(1) Spatial Hash Map
// =============================================================================
//
// LEARNING TOPIC: Why a Spatial Hash Map for Collision Detection?
// ---------------------------------------------------------------
// Naive collision detection checks every block against every other block:
//   For N blocks: N × N checks = O(N²) complexity.
//   1000 blocks → 1,000,000 checks per frame → engine crawls to a halt.
//
// A spatial hash map solves this:
//   1. Divide 3D space into a grid of integer cells (round position to nearest int)
//   2. Store each block's Entity in the cell that contains its center
//   3. To find neighbors of block A: look up only the 27 cells around A's cell
//   4. Total checks = N × 27 = O(N) — linear, not quadratic
//
// For 1000 blocks: 27,000 checks instead of 1,000,000 — 37× faster.
// For 10,000 blocks: 270,000 instead of 100,000,000 — 370× faster.
//
// The "hash" part: [i32; 3] (a grid coordinate) is hashed by Rust's default
// hasher into a HashMap bucket index. Lookup is O(1) amortized.
//
// =============================================================================
//
// OPTIMIZATION: Pre-allocated capacity for stable memory footprint
// ----------------------------------------------------------------
// Previously: HashMap::new() allocated a tiny initial table. As blocks were
// inserted, Rust would repeatedly resize the table (exponential growth:
// 16 → 32 → 64 → 128 → ...). Each resize copies the entire table.
//
// NEW: We pre-allocate with capacity for the expected world size. No resizes
// occur during normal operation. Memory footprint is stable after the first
// few frames, eliminating a major source of allocation pressure.
//
// LEARNING TOPIC: HashMap Capacity in Rust
// -----------------------------------------
// HashMap::with_capacity(n) allocates space for n items WITHOUT RESIZING
// until n+1 is inserted. The table uses open addressing with a load factor
// of ~87%, so actual bucket count = n / 0.87 ≈ 1.15n.
//
// For a world with 1000 blocks: 1000 capacity → ~1150 buckets allocated.
// Each bucket = 24 bytes (hash + key + value) → ~27KB total — trivial.
//
// CRITICAL INSIGHT: Vec::with_capacity inside the map values
// -----------------------------------------------------------
// Each grid cell stores a Vec<(Entity, ShapeType)>. Most cells contain 0 or 1
// blocks. Pre-allocating these inner Vecs would waste memory. Instead, we rely
// on the HashMap entry API which creates the Vec lazily (only when a block
// occupies that cell). Vec default capacity is 0 (no heap allocation) until
// the first push().
//
// A Vec push to an empty Vec allocates exactly 4 slots (Rust default growth).
// For our voxel world, most cells have 1 block, so 4 slots wastes 3 × 8 bytes
// = 24 bytes per occupied cell. For 1000 blocks, that's 24KB overhead — fine.
// A future optimization: use SmallVec<[(Entity, ShapeType); 1]> for 0-alloc
// single-block cells (relevant for Sprint 8's scale test with 1000+ blocks).
// =============================================================================

use crate::world::voxel::{ShapeType, Voxel};
use bevy_ecs::prelude::*;
use glam::Vec3;
use std::collections::HashMap;

// =============================================================================
// OPTIMIZATION CONSTANT: Default world capacity
// -----------------------------------------------
// Pre-allocate the HashMap for this many distinct grid cells.
// A grid cell is occupied if any block's center rounds to that integer position.
//
// For a 20×20×20 world volume with average 50% occupancy:
//   Expected cells = 20 × 20 × 20 × 0.5 = 4000
// We use 2048 as a power-of-two (hashmap resizes prefer powers of two).
//
// LEARNING: If you spawn 5000 blocks, the grid will resize once past 2048.
// For the Sprint 8 scale test (1000 blocks), 2048 is more than sufficient.
// Adjust this constant as the project scales to Phase III (100+ agents).
// =============================================================================
const GRID_INITIAL_CAPACITY: usize = 2048;

#[derive(Resource)]
pub struct SpatialGrid {
    // ==========================================================================
    // OPTIMIZATION: HashMap instead of Default
    // ----------------------------------------
    // Changed from `#[derive(Default)]` (which calls HashMap::new() → tiny initial
    // allocation) to manual Resource impl with with_capacity().
    //
    // This means the very first physics frame pays the allocation cost upfront,
    // but all subsequent frames find a correctly-sized table with no resizing.
    //
    // The map holds a Vec because multiple blocks can occupy the same grid cell
    // if they're in the same 1×1×1 unit of space. This happens during:
    //   - Initial spawning (blocks dropped at the same X/Z before gravity separates them)
    //   - Solver iterations where predicted_positions overlap before correction
    //   - Sphere shapes that span multiple grid cells (radius > 0.5)
    // ==========================================================================
    pub map: HashMap<[i32; 3], Vec<(Entity, ShapeType)>>,
}

impl Default for SpatialGrid {
    fn default() -> Self {
        Self {
            // Pre-allocate for GRID_INITIAL_CAPACITY distinct cells.
            // LEARNING: with_capacity hint prevents HashMap from resizing during
            // the first frame when all blocks are inserted simultaneously.
            map: HashMap::with_capacity(GRID_INITIAL_CAPACITY),
        }
    }
}

impl SpatialGrid {
    // =========================================================================
    // world_to_grid() — Continuous position → Discrete grid cell
    // =========================================================================
    //
    // LEARNING: We use .round() (nearest integer), not .floor() (floor).
    // Why? Consider a block at y = 0.51 vs y = 0.49:
    //   .floor(): 0.51 → 0,  0.49 → 0  (both in same cell — correct)
    //   .round(): 0.51 → 1,  0.49 → 0  (DIFFERENT cells for neighbors!)
    //
    // Wait — that seems like .round() is WRONG. But actually for our physics:
    // A block at y=0.5 is at the TOP of cell 0 or BOTTOM of cell 1. With .round(),
    // it maps to cell 1 (nearest integer). Its neighbor below (y=-0.5) maps to cell 0.
    // Our get_neighbors() checks 27 cells (−1, 0, +1 in each axis), so both cells
    // are always checked regardless of which cell the block is in.
    //
    // The key requirement: blocks that are PHYSICALLY ADJACENT should map to
    // NEARBY cells (within the 27-cell neighborhood). .round() guarantees this
    // for unit-cube voxels since adjacent blocks differ by exactly 1.0 in one axis.
    //
    // XPBD BUG FIX NOTE: In xpbd.rs, we DON'T use grid_pos as the collision position.
    // We use the actual snapshot.predicted_position. The grid is ONLY for broad-phase
    // neighbor detection. This fixed the static-block-disappearing bug (see xpbd.rs
    // "BUG FIX #1" comment for the full explanation).
    pub fn world_to_grid(position: Vec3) -> [i32; 3] {
        [
            position.x.round() as i32,
            position.y.round() as i32,
            position.z.round() as i32,
        ]
    }

    // =========================================================================
    // insert() — Add a block to the spatial grid
    // =========================================================================
    //
    // LEARNING: entry().or_default() is Rust's idiomatic "insert if not present,
    // then push". It avoids a double-lookup (check if key exists, then insert):
    //
    //   // NAIVE (two lookups):
    //   if !map.contains_key(&key) { map.insert(key, Vec::new()); }
    //   map.get_mut(&key).unwrap().push(value);
    //
    //   // IDIOMATIC (one lookup):
    //   map.entry(key).or_default().push(value);
    //
    // The Entry API locks the bucket once, checks existence, creates if needed,
    // and returns a mutable reference — all in one hash operation.
    pub fn insert(&mut self, position: Vec3, entity: Entity, shape: ShapeType) {
        let grid_pos = Self::world_to_grid(position);
        self.map.entry(grid_pos).or_default().push((entity, shape));
    }

    // =========================================================================
    // get_neighbors() — Return all entities in the 27-cell neighborhood
    // =========================================================================
    //
    // LEARNING TOPIC: 27-Cell Neighborhood (3×3×3 kernel)
    // -----------------------------------------------------
    // We check cells from (center - 1) to (center + 1) on each axis:
    //   x ∈ {-1, 0, +1}
    //   y ∈ {-1, 0, +1}
    //   z ∈ {-1, 0, +1}
    //   Total: 3 × 3 × 3 = 27 cells
    //
    // This guarantees we find ANY block whose center is within 1.5 units of
    // the query position in any direction. For unit cubes (radius = 0.5):
    //   Maximum collision range = 0.5 + 0.5 + rounding_error ≤ 1.5
    //   → Always found within the 27-cell neighborhood ✓
    //
    // For spheres with radius up to 1.5, we might miss some neighbors!
    // Sprint 6 (Lidar Vision) will extend this to a configurable search radius.
    //
    // OPTIMIZATION NOTE: Returns Vec (heap allocated) per call.
    // For the scale test (1000 blocks × 60Hz), that's 60,000 Vec allocations/sec.
    // Sprint 8 improvement: pass a reusable buffer (like SolverBuffers pattern)
    // to avoid per-call allocation. Prototype correctness first, optimize later.
    pub fn get_neighbors(&self, position: Vec3) -> Vec<(Entity, ShapeType, [i32; 3])> {
        // =======================================================================
        // OPTIMIZATION: Pre-allocate the neighbors vec with a reasonable capacity.
        // Average blocks per 27-cell neighborhood in a dense world: 5-15 blocks.
        // Pre-allocating 16 slots avoids realloc for the common case.
        // 16 = next power of two above 15 = minimum Vec doubling threshold.
        // =======================================================================
        let mut neighbors = Vec::with_capacity(16);
        let center = Self::world_to_grid(position);

        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let check_pos = [center[0] + x, center[1] + y, center[2] + z];
                    if let Some(entities) = self.map.get(&check_pos) {
                        for (entity, shape) in entities {
                            neighbors.push((*entity, shape.clone(), check_pos));
                        }
                    }
                }
            }
        }
        neighbors
    }

    // =========================================================================
    // column_max_y() — Find highest block in a vertical column (for spawning)
    // =========================================================================
    //
    // LEARNING: This is used by lib.rs to compute a safe drop height for newly
    // spawned blocks. Without it, clicking "Spawn Cube" above an existing tower
    // would spawn the block INSIDE the tower (same Y position), causing the
    // solver to violently eject both blocks.
    //
    // Implementation: filter all grid keys for matching (x, z), then take max y.
    // This is O(total_cells) — not O(1). For the current world size (<1000 cells),
    // this is fine. A production optimization: maintain a separate HashMap<[i32;2], i32>
    // (x,z → max_y) updated incrementally as blocks settle.
    //
    // LEARNING: We use .filter() + .map() + .max() — a functional chain.
    // Rust's iterator combinators are zero-overhead: the compiler fuses them into
    // a single loop with no intermediate allocation (unlike Python list comprehensions).
    pub fn column_max_y(&self, x: i32, z: i32) -> Option<i32> {
        self.map
            .keys()
            .filter(|k| k[0] == x && k[2] == z)
            .map(|k| k[1])
            .max()
    }

    // =========================================================================
    // clear_and_reserve() — Reset grid between frames efficiently
    // =========================================================================
    //
    // LEARNING: HashMap::clear() drops all key-value pairs but RETAINS the
    // allocated bucket array. This is the critical performance property:
    //
    //   First frame:  HashMap grows from 0 → 2048 capacity (allocation cost)
    //   Frame 2+:     clear() resets to empty, capacity stays at 2048 (free!)
    //
    // This is the "amortized O(1)" property of HashMap. After the first frame,
    // all subsequent clears are O(N) in the number of stored items (to drop them)
    // but O(1) in memory operations (no realloc, no free+malloc).
    //
    // Contrast with creating a new HashMap every frame (old approach):
    //   New HashMap → allocates a tiny table (16 buckets)
    //   Insert 1000 blocks → resizes: 16→32→64→128→256→512→1024→2048
    //   7 resize operations × O(N) copy each = O(N log N) total per frame!
    //
    // With clear_and_reserve(), it's O(N) total per frame (just inserts, no resizes).
    // For 1000 blocks × 60 Hz: saves 6 × 60 × realloc overhead ≈ measurable win.
    pub fn clear_and_reserve(&mut self, expected_blocks: usize) {
        self.map.clear();

        // If the world has grown significantly beyond our initial capacity,
        // reserve additional space to prevent future resizes.
        // LEARNING: reserve() is a hint — it ensures capacity >= current + additional.
        // We use saturating_sub to avoid overflow if len() > expected_blocks.
        let current_capacity = self.map.capacity();
        if expected_blocks > current_capacity {
            // Growing: reserve the difference (plus 25% headroom to avoid
            // immediate re-resize when more blocks are added).
            let additional = expected_blocks.saturating_sub(current_capacity);
            self.map.reserve(additional + additional / 4);
        }
        // Shrinking: we intentionally don't shrink_to_fit() because:
        //   1. The next frame will likely re-expand back to the same size
        //   2. Shrinking requires reallocation — the exact cost we're avoiding
    }
}

// =============================================================================
// update_spatial_grid_system — Bevy ECS system to rebuild the grid each frame
// =============================================================================
//
// LEARNING TOPIC: Bevy Systems — The Logic Layer
// -----------------------------------------------
// Systems are pure functions that take Bevy "SystemParams" (Query, Res, ResMut)
// and operate on the ECS world. They have no state of their own — state lives
// in Components and Resources.
//
// This system runs TWICE per physics frame (see lib.rs schedule):
//   1. BEFORE integrate: grid is built from last frame's COMMITTED positions.
//      The solver uses this to find neighbors for the current step.
//   2. AFTER update_velocities: grid is rebuilt from THIS frame's FINAL positions.
//      Python queries and the render system see the correct, post-physics state.
//
// Running it twice is O(2N) = O(N) — acceptable. The alternative (incremental
// updates) would be O(moved_blocks) but requires tracking which blocks moved,
// adding complexity that's premature for Sprint 2's block counts.
//
// OPTIMIZATION APPLIED HERE:
//   - count() pass before insert pass to get exact expected_blocks count
//     for clear_and_reserve(). This gives the HashMap perfect capacity.
//   - The two-pass approach adds one O(N) iteration but saves multiple
//     O(N) resize operations inside the HashMap.
pub fn update_spatial_grid_system(mut grid: ResMut<SpatialGrid>, query: Query<(Entity, &Voxel)>) {
    // ==========================================================================
    // OPTIMIZATION: Count blocks first, then reserve, then insert.
    // Two passes: O(N) + O(N) = O(2N) = O(N).
    // Without reserve: could be O(N log N) due to multiple resizes.
    //
    // LEARNING: .count() on a Bevy Query iterates the archetype storage once.
    // Since Voxel components are stored in a single contiguous archetype array,
    // .count() is a fast O(N) scan with excellent cache behavior.
    // ==========================================================================
    let block_count = query.iter().count();
    grid.clear_and_reserve(block_count);

    for (entity, voxel) in query.iter() {
        // Use voxel.position (the COMMITTED position from last frame's
        // update_velocities_system), NOT predicted_position.
        //
        // LEARNING: predicted_position is mid-flight during the solve phase.
        // The spatial grid for neighbor detection should reflect WHERE BLOCKS ARE,
        // not where they might end up. Using predicted_position in the grid would
        // cause the broad phase to find neighbors in the wrong place and miss
        // real contacts between settled blocks.
        grid.insert(voxel.position, entity, voxel.shape.clone());
    }
}
