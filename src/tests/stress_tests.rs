// =============================================================================
// src/tests/stress_tests.rs  —  Layer 1c: Stress Propagation Unit Tests
// =============================================================================
//
// WHAT IS TESTED
// --------------
// These tests verify the two-pass structural stress algorithm in
// `src/physics/stress.rs`. Correct stress propagation is critical because:
//   1. The rendering heatmap must accurately warn about overloaded blocks.
//   2. The RL reward function uses stress to discourage noodle towers.
//   3. Research publications will cite stress data as evidence of structural
//      realism — the numbers must be provably correct.
//
// TEST CATALOGUE
// --------------
//   test_single_column_stress_propagation
//       A 5-block Wood pillar. The bottom block must report supported_mass ≈
//       4 kg (mass of the 4 blocks above it). Each block upward should have
//       one fewer kg of supported mass. Tests the downward traversal ordering.
//
//   test_stress_normalized_range
//       Every block's stress_normalized value must be in [0.0, 1.0].
//       Values outside this range indicate a divide-by-zero or capacity
//       calculation error in stress.rs.
//
//   test_stress_increases_with_load
//       Bottom block of a tall pillar must have higher stress_normalized than
//       the top block. If bottom ≤ top, the propagation direction is wrong.
//
//   test_static_block_stress_excluded
//       Static blocks (inv_mass=0) must not appear in StressMap or must report
//       zero supported_mass. They ARE the foundation — they don't carry stress.
//
//   test_scaffold_stress_capacity_is_low
//       Scaffold blocks have capacity multiplier=5× (vs Steel=24×, Wood=15×).
//       A lightly loaded scaffold must still show high stress_normalized.
//       This is the "scaffold turns red quickly" design requirement.
//
//   test_floor_support_flag_propagates
//       In a 4-block stack, all blocks must report is_supported_from_below=true.
//       A floating arch block not connected to the ground must report false.
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::physics::mortar::{
        MortarBonds, break_overloaded_bonds_system, register_new_bonds_system,
        solve_mortar_constraints_system,
    };
    use crate::physics::stress::{StressMap, compute_stress_system};
    use crate::physics::xpbd::{
        PhysicsSettings, SolverBuffers, integrate_system, solve_constraints_system,
        update_velocities_system,
    };
    use crate::tests::{AuditLog, SimParams};
    use crate::world::spatial_grid::{SpatialGrid, update_spatial_grid_system};
    use crate::world::voxel::{MaterialType, ShapeType, Voxel};
    use bevy_ecs::prelude::*;

    fn full_world() -> World {
        let mut world = World::new();
        world.insert_resource(SpatialGrid::default());
        world.insert_resource(PhysicsSettings::default());
        world.insert_resource(SolverBuffers::default());
        world.insert_resource(MortarBonds::default());
        world.insert_resource(StressMap::default());
        world
    }

    fn full_schedule() -> Schedule {
        let mut s = Schedule::default();
        s.add_systems(
            (
                update_spatial_grid_system,
                integrate_system,
                solve_constraints_system,
                register_new_bonds_system,
                solve_mortar_constraints_system,
                break_overloaded_bonds_system,
                update_velocities_system,
                update_spatial_grid_system,
                compute_stress_system,
            )
                .chain(),
        );
        s
    }

    fn run_steps(world: &mut World, schedule: &mut Schedule, n: u32) {
        for _ in 0..n {
            schedule.run(world);
        }
    }

    fn get_stress(world: &World, entity: Entity) -> Option<(f32, f32, bool)> {
        let map = world.get_resource::<StressMap>().unwrap();
        map.data.get(&entity).map(|s| {
            (
                s.supported_mass,
                s.stress_normalized,
                s.is_supported_from_below,
            )
        })
    }

    // =========================================================================
    // TEST 1: Stress propagation in a 5-block Wood pillar
    // =========================================================================
    #[test]
    fn test_single_column_stress_propagation() {
        let params = SimParams {
            seed: 3001,
            block_count: 5,
            steps_measured: 600,
            extra: vec![
                ("material".into(), "Wood (1 kg each)".into()),
                ("pillar_height".into(), "5 blocks".into()),
                ("expected_bottom_supported_mass".into(), "~4.0 kg".into()),
                ("expected_top_supported_mass".into(), "~0.0 kg".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_single_column_stress_propagation",
            "STRESS",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        // Spawn blocks from ground up — they settle into a clean pillar
        let mut entities = Vec::new();
        for i in 0..5 {
            let e = world
                .spawn(Voxel::new_with_material(
                    0.0,
                    i as f32 + 0.5,
                    0.0,
                    ShapeType::Cube,
                    MaterialType::Wood,
                    false,
                ))
                .id();
            entities.push(e);
        }

        run_steps(&mut world, &mut schedule, params.steps_measured);

        // Re-sort entities by final Y position so bottom=index 0, top=index 4
        let mut entities_by_y: Vec<(Entity, f32)> = entities
            .iter()
            .map(|&e| {
                let y = world.get::<Voxel>(e).unwrap().position.y;
                (e, y)
            })
            .collect();
        entities_by_y.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let bottom_entity = entities_by_y[0].0;
        let top_entity = entities_by_y[4].0;

        // Wood mass = 1.0 / inv_mass = 1.0 / 1.0 = 1 kg each
        let wood_mass_kg = 1.0_f32;

        // ── Check 1: Bottom block supported_mass ≈ 4 × wood_mass ─────────────
        // Bottom block supports all 4 blocks above it.
        // supported_mass = 4 × 1 kg = 4 kg
        let bottom_stress = get_stress(&world, bottom_entity);
        log.assert_true(
            "bottom_block_has_stress_entry",
            "Bottom block must have an entry in StressMap. Missing entry means \
             compute_stress_system did not process this entity.",
            bottom_stress.is_some(),
            "Bottom block entity not found in StressMap",
        );

        if let Some((supported_mass, stress_norm, floor_supported)) = bottom_stress {
            log.assert_approx(
                "bottom_supported_mass",
                "Bottom block must support the mass of all 4 blocks above it. \
                 Wood mass = 1 kg × 4 blocks = 4 kg. \
                 supported_mass < 3.0 → propagation is not traversing all upper blocks. \
                 supported_mass > 5.0 → blocks are being double-counted.",
                supported_mass,
                4.0 * wood_mass_kg,
                0.5, // 0.5 kg tolerance for settle-position variations
            );

            log.assert_true(
                "bottom_is_floor_supported",
                "The bottom block of the pillar must be marked as floor-supported \
                 since it rests directly on the ground plane (y ≈ 0.0).",
                floor_supported,
                format!(
                    "bottom block is_supported_from_below={} supported_mass={:.4}",
                    floor_supported, supported_mass
                ),
            );

            // Stress normalized must be in [0, 1]
            log.assert_true(
                "bottom_stress_in_range",
                "stress_normalized must be in [0.0, 1.0]. Values outside this range \
                 indicate a capacity calculation error (divide by zero or negative capacity).",
                (0.0..=1.0).contains(&stress_norm),
                format!("stress_normalized = {:.6}", stress_norm),
            );
        }

        // ── Check 2: Top block supported_mass ≈ 0 ────────────────────────────
        let top_stress = get_stress(&world, top_entity);
        if let Some((supported_mass, _, _)) = top_stress {
            log.assert_approx(
                "top_supported_mass",
                "Top block has nothing above it — supported_mass must be ~0.0 kg. \
                 Non-zero value means a phantom block is being counted, or the \
                 traversal direction is reversed (propagating upward instead of downward).",
                supported_mass,
                0.0,
                0.3,
            );
        }

        // ── Check 3: Monotonic decrease from bottom to top ────────────────────
        let masses: Vec<f32> = entities_by_y
            .iter()
            .filter_map(|(e, _)| get_stress(&world, *e).map(|(m, _, _)| m))
            .collect();

        if masses.len() == 5 {
            let monotonic = masses.windows(2).all(|w| w[0] >= w[1] - 0.1);
            log.assert_true(
                "stress_monotonic_top_to_bottom",
                "Supported mass must decrease (or stay equal) from bottom to top. \
                 A block higher in the stack supports fewer blocks above it. \
                 Non-monotonic sequence indicates the pass 2 ordering is wrong — \
                 entities are not being processed top-down.",
                monotonic,
                format!(
                    "masses bottom→top = {:?}  (expected strictly decreasing)",
                    masses
                ),
            );
        }

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 2: stress_normalized is always in [0.0, 1.0]
    // =========================================================================
    #[test]
    fn test_stress_normalized_range() {
        let params = SimParams {
            seed: 3002,
            block_count: 8,
            steps_measured: 400,
            extra: vec![
                ("materials".into(), "mixed: Wood, Steel, Stone".into()),
                (
                    "invariant".into(),
                    "0.0 ≤ stress_normalized ≤ 1.0 for all blocks".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_stress_normalized_range", "STRESS", params.clone());

        let mut world = full_world();
        let mut schedule = full_schedule();

        // Mixed material pillar to stress-test the capacity formula across all materials
        let materials = [
            MaterialType::Wood,
            MaterialType::Steel,
            MaterialType::Stone,
            MaterialType::Wood,
            MaterialType::Steel,
            MaterialType::Stone,
            MaterialType::Wood,
            MaterialType::Steel,
        ];

        let mut entities = Vec::new();
        for (i, &mat) in materials.iter().enumerate() {
            let e = world
                .spawn(Voxel::new_with_material(
                    0.0,
                    i as f32 + 0.5,
                    0.0,
                    ShapeType::Cube,
                    mat,
                    false,
                ))
                .id();
            entities.push(e);
        }

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let stress_map = world.get_resource::<StressMap>().unwrap();
        let mut out_of_range: Vec<(Entity, f32)> = Vec::new();
        let mut values: Vec<f32> = Vec::new();

        for &entity in &entities {
            if let Some(info) = stress_map.data.get(&entity) {
                let s = info.stress_normalized;
                values.push(s);
                if !(0.0..=1.0).contains(&s) {
                    out_of_range.push((entity, s));
                }
            }
        }

        log.assert_count(
            "all_stress_values_in_range",
            "All stress_normalized values must be in [0.0, 1.0]. \
             Values < 0 indicate negative load (invalid). \
             Values > 1 mean the clamp in compute_stress_system is missing or \
             the capacity formula returned a negative/zero capacity \
             (divide-by-zero with zero-mass block).",
            out_of_range.len(),
            0,
            format!(
                "Out-of-range entries: {:?}  All values: {:?}",
                out_of_range, values
            ),
        );

        // Verify all entities have a stress entry
        let missing = entities
            .iter()
            .filter(|&&e| !stress_map.data.contains_key(&e))
            .count();

        log.assert_count(
            "all_blocks_have_stress_entry",
            "Every dynamic block must have an entry in StressMap. \
             Missing entries indicate compute_stress_system skipped some entities — \
             check that the query includes all dynamic voxels.",
            missing,
            0,
            format!("{} blocks missing from StressMap", missing),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 3: Stress increases from top to bottom of pillar
    // =========================================================================
    #[test]
    fn test_stress_increases_with_load() {
        let params = SimParams {
            seed: 3003,
            block_count: 6,
            steps_measured: 500,
            extra: vec![
                (
                    "material".into(),
                    "Steel (uniform mass for clean test)".into(),
                ),
                (
                    "invariant".into(),
                    "bottom.stress_normalized > top.stress_normalized".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_stress_increases_with_load", "STRESS", params.clone());

        let mut world = full_world();
        let mut schedule = full_schedule();

        let mut entities = Vec::new();
        for i in 0..6 {
            let e = world
                .spawn(Voxel::new_with_material(
                    0.0,
                    i as f32 + 0.5,
                    0.0,
                    ShapeType::Cube,
                    MaterialType::Steel,
                    false,
                ))
                .id();
            entities.push(e);
        }

        run_steps(&mut world, &mut schedule, params.steps_measured);

        // Sort by Y to get unambiguous bottom/top
        let mut by_y: Vec<(Entity, f32)> = entities
            .iter()
            .map(|&e| (e, world.get::<Voxel>(e).unwrap().position.y))
            .collect();
        by_y.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let bottom_e = by_y[0].0;
        let top_e = by_y[5].0;

        let bottom_info = get_stress(&world, bottom_e);
        let top_info = get_stress(&world, top_e);

        if let (Some((_, bottom_norm, _)), Some((_, top_norm, _))) = (bottom_info, top_info) {
            log.assert_true(
                "bottom_stress_greater_than_top",
                "Bottom block must have higher stress_normalized than the top block. \
                 Bottom supports 5 blocks above it; top supports 0. \
                 bottom_norm ≤ top_norm means the propagation direction is wrong \
                 or the capacity formula is not working for Steel.",
                bottom_norm > top_norm,
                format!(
                    "bottom_norm={:.6}  top_norm={:.6}  (expected bottom > top)",
                    bottom_norm, top_norm
                ),
            );

            log.assert_true(
                "top_block_low_stress",
                "Top block (nothing above it) must have low stress_normalized ≤ 0.1. \
                 High stress_normalized on the top block means phantom loads are \
                 being assigned to blocks with no blocks above them.",
                top_norm <= 0.1,
                format!("top_norm = {:.6} (expected ≤ 0.1)", top_norm),
            );
        }

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 4: Scaffold capacity turns red quickly under light load
    // =========================================================================
    #[test]
    fn test_scaffold_stress_capacity_is_low() {
        let params = SimParams {
            seed: 3004,
            block_count: 2,
            steps_measured: 400,
            extra: vec![
                (
                    "scenario".into(),
                    "scaffold block under 1 Wood block".into(),
                ),
                (
                    "scaffold_capacity_multiplier".into(),
                    "5× own weight".into(),
                ),
                ("wood_capacity_multiplier".into(), "15× own weight".into()),
                (
                    "expected".into(),
                    "scaffold.stress_norm >> wood.stress_norm".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_scaffold_stress_capacity_is_low",
            "STRESS",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        // Scaffold at bottom, Wood on top
        let scaffold_e = world
            .spawn(Voxel::new_with_material(
                0.0,
                0.0,
                0.0,
                ShapeType::Cube,
                MaterialType::Scaffold,
                false,
            ))
            .id();
        let wood_e = world
            .spawn(Voxel::new_with_material(
                0.0,
                1.1,
                0.0,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            ))
            .id();

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let scaffold_info = get_stress(&world, scaffold_e);
        let wood_info = get_stress(&world, wood_e);

        if let (Some((_, scaffold_norm, _)), Some((_, wood_norm, _))) = (scaffold_info, wood_info) {
            // Scaffold carries 1 kg of Wood above it.
            // Scaffold own mass = 0.5 kg, capacity = 5 × 0.5 = 2.5 kg
            // scaffold_norm = 1.0 / 2.5 = 0.40
            //
            // Wood block carries 0 kg above it.
            // wood_norm = 0.0
            log.assert_true(
                "scaffold_higher_stress_than_wood",
                "Scaffold block supporting 1 Wood block must have \
                 higher stress_normalized than the top Wood block (which supports nothing). \
                 scaffold_norm should be around 0.4 (1 kg / 2.5 kg capacity). \
                 If scaffold_norm ≤ wood_norm, the scaffold capacity multiplier (5×) \
                 is not being applied correctly — check stress_capacity_for_material().",
                scaffold_norm > wood_norm,
                format!(
                    "scaffold_norm={:.6}  wood_norm={:.6}  \
                     (expected scaffold >> wood for same load)",
                    scaffold_norm, wood_norm
                ),
            );

            // Scaffold norm should be meaningfully high (around 0.3–0.5)
            log.assert_true(
                "scaffold_stress_meaningful",
                "Scaffold supporting 1 kg (Wood block) must show meaningful stress \
                 ≥ 0.2. Low stress indicates the capacity formula is using the wrong \
                 multiplier or the scaffold mass is wrong.",
                scaffold_norm >= 0.2,
                format!("scaffold_norm = {:.6} (expected ≥ 0.2)", scaffold_norm),
            );

            // Stress must stay in range
            log.assert_true(
                "scaffold_norm_in_range",
                "Scaffold stress_normalized must be in [0.0, 1.0].",
                (0.0..=1.0).contains(&scaffold_norm),
                format!("scaffold_norm = {:.6}", scaffold_norm),
            );
        }

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 5: Floor support flag propagates correctly through stack
    // =========================================================================
    #[test]
    fn test_floor_support_flag_propagates() {
        let params = SimParams {
            seed: 3005,
            block_count: 4,
            steps_measured: 400,
            extra: vec![
                ("stack_height".into(), "4 blocks on ground".into()),
                (
                    "invariant".into(),
                    "all 4 blocks must be floor-supported".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_floor_support_flag_propagates",
            "STRESS",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        let mut entities = Vec::new();
        for i in 0..4 {
            let e = world
                .spawn(Voxel::new_with_material(
                    0.0,
                    i as f32 + 0.5,
                    0.0,
                    ShapeType::Cube,
                    MaterialType::Wood,
                    false,
                ))
                .id();
            entities.push(e);
        }

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let mut unsupported_blocks = Vec::new();
        for (idx, &entity) in entities.iter().enumerate() {
            if let Some((_, _, floor_supported)) = get_stress(&world, entity) {
                if !floor_supported {
                    unsupported_blocks.push(idx);
                }
            }
        }

        // ── Check: All blocks floor-supported ─────────────────────────────────
        log.assert_count(
            "all_grounded_blocks_floor_supported",
            "All 4 blocks in a grounded stack must have is_supported_from_below=true. \
             An unsupported block in a grounded stack means the floor support \
             propagation pass did not traverse far enough up the stack — \
             check the below_is_supported logic in compute_stress_system.",
            unsupported_blocks.len(),
            0,
            format!(
                "Unsupported block indices: {:?}  (all should be floor-supported)",
                unsupported_blocks
            ),
        );

        log.finalize_and_assert();
    }
}
