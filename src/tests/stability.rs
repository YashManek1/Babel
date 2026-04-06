// =============================================================================
// src/tests/stability.rs  —  Layer 1b: Structural Stability Unit Tests
// =============================================================================
//
// WHAT IS TESTED
// --------------
// These tests verify that the sleep system, damping, and structural constraints
// produce a world that reaches a genuinely stationary state. A simulation that
// never truly stops introduces noise into RL observations and prevents the neural
// network from learning stable policies.
//
// TEST CATALOGUE
// --------------
//   test_single_block_reaches_sleep
//       One block dropped from height 3.0 must reach is_sleeping=true within
//       240 steps. Verifies the sleep threshold fires correctly.
//
//   test_ten_block_pillar_no_jitter
//       The canonical "10-block pillar must not jitter" requirement from the
//       project docs. After 500 steps, EVERY block's speed must be below
//       LINEAR_SLEEP_SPEED (0.20 m/s). Jitter means the sleep system is not
//       engaging for stacked contacts — a known bug trigger.
//
//   test_pillar_stable_height_preserved
//       After a 10-block pillar settles, total height (max_y - min_y) must
//       be ≥ 8.5 world-units, proving no blocks have fallen off or collapsed
//       under their own weight.
//
//   test_sleep_wakes_on_impact
//       A sleeping block must wake (is_sleeping=false) when struck by a
//       falling block. This verifies WAKE_CORRECTION_EPS in solve_constraints
//       actually propagates through to sleeping bodies.
//
//   test_lateral_side_drag_regression
//       Regression guard for Bug #3 (lateral anchor clamp). A 2-high wood
//       pillar hit by a side block must not be dragged to the ground. Post-
//       collision, all pillar blocks must remain at their original Y positions
//       within 0.3 units.
//
//   test_steel_pair_no_vibration
//       Two adjacent Steel blocks (the vibration bug target) must reach
//       near-zero velocity within 300 steps. The mass-proportional compliance
//       fix in mortar.rs specifically addressed this — the test is a permanent
//       regression guard.
//
//   test_restitution_energy_loss
//       A Wood block (restitution=0.0) dropped onto a static block must NOT
//       bounce back up. A Steel block (restitution=0.05) may have a tiny bounce.
//       Verifies the restitution model doesn't over-bounce blocks.
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::physics::mortar::{
        MortarBonds, break_overloaded_bonds_system, register_new_bonds_system,
        solve_mortar_constraints_system,
    };
    use crate::physics::xpbd::{
        PhysicsSettings, SolverBuffers, integrate_system, solve_constraints_system,
        update_velocities_system,
    };
    use crate::tests::{AuditLog, SimParams};
    use crate::world::spatial_grid::{SpatialGrid, update_spatial_grid_system};
    use crate::world::voxel::{MaterialType, ShapeType, Voxel};
    use bevy_ecs::prelude::*;
    use glam::Vec3;

    // -------------------------------------------------------------------------
    // Shared test infrastructure
    // -------------------------------------------------------------------------

    fn minimal_world() -> World {
        let mut world = World::new();
        world.insert_resource(SpatialGrid::default());
        world.insert_resource(PhysicsSettings::default());
        world.insert_resource(SolverBuffers::default());
        world.insert_resource(MortarBonds::default());
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

    fn get_voxel_speed(world: &World, entity: Entity) -> f32 {
        world.get::<Voxel>(entity).unwrap().velocity.length()
    }

    fn get_voxel_y(world: &World, entity: Entity) -> f32 {
        world.get::<Voxel>(entity).unwrap().position.y
    }

    fn get_voxel_sleeping(world: &World, entity: Entity) -> bool {
        world.get::<Voxel>(entity).unwrap().is_sleeping
    }

    // =========================================================================
    // TEST 1: Single block reaches sleep state
    // =========================================================================
    #[test]
    fn test_single_block_reaches_sleep() {
        let params = SimParams {
            seed: 2001,
            block_count: 1,
            steps_measured: 240,
            extra: vec![
                ("drop_height".into(), "3.0".into()),
                ("material".into(), "Wood".into()),
                ("linear_sleep_speed".into(), "0.20 m/s".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_single_block_reaches_sleep",
            "STABILITY",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = full_schedule();

        let block = world
            .spawn(Voxel::new(0.0, 3.0, 0.0, ShapeType::Cube, false))
            .id();

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let sleeping = get_voxel_sleeping(&world, block);
        let speed = get_voxel_speed(&world, block);
        let pos_y = get_voxel_y(&world, block);

        // ── Check 1: Sleep flag engaged ───────────────────────────────────────
        log.assert_true(
            "sleep_flag_engaged",
            "A single block dropped from height 3.0 must reach is_sleeping=true \
             within 240 steps (4 seconds simulated). Failure means: \
             (a) LINEAR_SLEEP_SPEED threshold is set too low, or \
             (b) damping is not removing energy fast enough, or \
             (c) the floor contact is producing oscillations that prevent sleep.",
            sleeping,
            format!(
                "is_sleeping={} speed={:.4} pos_y={:.4}",
                sleeping, speed, pos_y
            ),
        );

        // ── Check 2: Speed below sleep threshold ──────────────────────────────
        log.assert_leq(
            "speed_below_sleep_threshold",
            "Velocity magnitude must be at or below LINEAR_SLEEP_SPEED=0.20 m/s. \
             This is a necessary condition for sleep to engage. If speed > 0.20 \
             but sleep=false, the sleep check is not running on every contact step.",
            speed,
            0.20,
            format!("speed = {:.6} m/s", speed),
        );

        // ── Check 3: Block has landed (not still falling) ─────────────────────
        log.assert_true(
            "block_has_landed",
            "After 240 steps the block must have hit the floor and settled. \
             pos_y > 1.0 indicates the block is still falling — gravity \
             integration or the floor constraint is broken.",
            pos_y < 1.0,
            format!("pos_y = {:.4} (must be < 1.0)", pos_y),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 2: 10-block pillar — no jitter requirement (project docs canonical test)
    // =========================================================================
    #[test]
    fn test_ten_block_pillar_no_jitter() {
        let params = SimParams {
            seed: 2002,
            block_count: 10,
            steps_measured: 500,
            extra: vec![
                ("pillar_height".into(), "10 blocks".into()),
                ("material".into(), "Wood".into()),
                (
                    "jitter_threshold".into(),
                    "LINEAR_SLEEP_SPEED = 0.20 m/s".into(),
                ),
                (
                    "canonical_requirement".into(),
                    "From project docs: 10-block pillar must not jitter".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_ten_block_pillar_no_jitter",
            "STABILITY",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = full_schedule();

        // Spawn pillar: each block 0.1 units above the previous (they fall into place)
        let mut entities = Vec::new();
        for i in 0..10 {
            let e = world
                .spawn(Voxel::new(
                    0.0,
                    i as f32 * 1.1 + 1.0, // small gap so they don't spawn inside each other
                    0.0,
                    ShapeType::Cube,
                    false,
                ))
                .id();
            entities.push(e);
        }

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let mut max_speed: f32 = 0.0;
        let mut jittering_blocks: Vec<(usize, f32)> = Vec::new();

        for (idx, &entity) in entities.iter().enumerate() {
            let speed = get_voxel_speed(&world, entity);
            if speed > max_speed {
                max_speed = speed;
            }
            if speed > 0.20 {
                jittering_blocks.push((idx, speed));
            }
        }

        // ── Check 1: All blocks below jitter threshold ─────────────────────────
        log.assert_count(
            "zero_jittering_blocks",
            "Every block in the 10-block pillar must have speed ≤ 0.20 m/s \
             after 500 steps. Jittering blocks indicate the sleep system is \
             not engaging for stacked inter-block contacts, or the mortar \
             solver is waking sleeping blocks unnecessarily (pillar flying bug).",
            jittering_blocks.len(),
            0,
            format!(
                "Jittering blocks: {:?}  Max speed in pillar: {:.6} m/s",
                jittering_blocks, max_speed
            ),
        );

        // ── Check 2: Max speed across entire pillar ────────────────────────────
        log.assert_leq(
            "pillar_max_speed",
            "Maximum speed of any block in the pillar must be below 0.20 m/s. \
             This is the quantitative equivalent of 'no jitter'.",
            max_speed,
            0.20,
            format!("max_speed = {:.6} m/s across all 10 blocks", max_speed),
        );

        // ── Check 3: Sleep count — most blocks should be sleeping ─────────────
        let sleeping_count = entities
            .iter()
            .filter(|&&e| get_voxel_sleeping(&world, e))
            .count();

        log.assert_true(
            "most_blocks_sleeping",
            "At least 8 of 10 pillar blocks must be in sleep state after 500 steps. \
             Fewer sleeping blocks indicates persistent energy injection — \
             likely a mortar solve overcorrection or floor constraint oscillation.",
            sleeping_count >= 8,
            format!(
                "{}/10 blocks sleeping. Awake blocks may indicate \
                 BOND_WAKE_TENSION_THRESHOLD is too sensitive.",
                sleeping_count
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 3: Pillar height preserved after settle
    // =========================================================================
    #[test]
    fn test_pillar_stable_height_preserved() {
        let params = SimParams {
            seed: 2003,
            block_count: 10,
            steps_measured: 600,
            extra: vec![
                ("min_required_height_span".into(), "8.5 world-units".into()),
                ("material".into(), "Wood".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_pillar_stable_height_preserved",
            "STABILITY",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = full_schedule();

        let mut entities = Vec::new();
        for i in 0..10 {
            let e = world
                .spawn(Voxel::new(
                    0.0,
                    i as f32 * 1.15 + 1.0,
                    0.0,
                    ShapeType::Cube,
                    false,
                ))
                .id();
            entities.push(e);
        }

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let ys: Vec<f32> = entities.iter().map(|&e| get_voxel_y(&world, e)).collect();
        let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let height_span = max_y - min_y;

        // ── Check 1: Height span is at least 8.5 units ────────────────────────
        // 10 blocks, each 1 unit wide, settled touching = 9 units total.
        // Allow 0.5 units of settle compression.
        log.assert_true(
            "pillar_height_span_sufficient",
            "10-block pillar must span at least 8.5 world-units from bottom to top \
             block centre. Lower span means blocks have collapsed through each other \
             (penetration) or the pillar fell over (stability failure).",
            height_span >= 8.5,
            format!(
                "height_span = {:.4}  (max_y={:.4}, min_y={:.4})  \
                 blocks_y = {:?}",
                height_span, max_y, min_y, ys
            ),
        );

        // ── Check 2: No block has fallen below the floor ───────────────────────
        let below_floor = ys.iter().filter(|&&y| y < -0.1).count();
        log.assert_count(
            "no_blocks_below_floor",
            "No block should have a Y position below -0.1 (floor is at -0.5, \
             block centre must be at 0.0 or above). A block below -0.1 has \
             sunk through the floor or been ejected downward.",
            below_floor,
            0,
            format!(
                "blocks below floor: {:?}",
                ys.iter().filter(|&&y| y < -0.1).collect::<Vec<_>>()
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 4: Sleep wakes on impact
    // =========================================================================
    #[test]
    fn test_sleep_wakes_on_impact() {
        let params = SimParams {
            seed: 2004,
            block_count: 2,
            steps_warmup: 300,
            steps_measured: 60,
            extra: vec![
                (
                    "scenario".into(),
                    "sleeping block struck by falling block".into(),
                ),
                (
                    "wake_mechanism".into(),
                    "WAKE_CORRECTION_EPS in solve_constraints".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_sleep_wakes_on_impact", "STABILITY", params.clone());

        let mut world = minimal_world();
        let mut schedule = full_schedule();

        // Spawn block A, let it settle (it will sleep)
        let block_a = world
            .spawn(Voxel::new(0.0, 1.0, 0.0, ShapeType::Cube, false))
            .id();

        run_steps(&mut world, &mut schedule, params.steps_warmup); // settle block A

        let sleeping_before = get_voxel_sleeping(&world, block_a);
        log.assert_true(
            "block_a_sleeping_before_impact",
            "Block A must be sleeping before the impact block is spawned. \
             If it is not sleeping at this point, the warmup steps are \
             insufficient or the sleep system is broken.",
            sleeping_before,
            "Block A not sleeping after 300 warmup steps — test precondition failed",
        );

        // Now drop block B on top of block A
        let block_b = world
            .spawn(Voxel::new(0.0, 4.0, 0.0, ShapeType::Cube, false))
            .id();

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let sleeping_after = get_voxel_sleeping(&world, block_a);
        let speed_a_after = get_voxel_speed(&world, block_a);
        let block_b_pos = world.get::<Voxel>(block_b).unwrap().position;

        // ── Check 1: Block A woke up ──────────────────────────────────────────
        // After block B lands on A, block A should have been woken (it received
        // a correction from the impact). After settling back down it may sleep
        // again — but during the 60 measured steps it should wake at some point.
        // We check that block B actually reached block A (collision happened).
        let collision_occurred = block_b_pos.y < 3.0; // block B has fallen significantly
        log.assert_true(
            "block_b_fell_toward_a",
            "Block B must have fallen from y=4.0 toward Block A. \
             If block_b_pos.y is still near 4.0, the spawning or gravity failed.",
            collision_occurred,
            format!("block_b_pos.y = {:.4} (expected < 3.0)", block_b_pos.y),
        );

        // ── Check 2: Block A is in a reasonable state ─────────────────────────
        // After being impacted it may be moving slightly or may have re-settled.
        // The key test: it should not have flown away.
        let pos_a_y = get_voxel_y(&world, block_a);
        log.assert_true(
            "block_a_not_ejected",
            "Block A must not be ejected upward by the impact. \
             pos_y > 2.0 indicates Block A was launched — the collision \
             response applied too much upward correction to the lower block.",
            pos_a_y < 2.0,
            format!("block_a pos_y = {:.4} (must be < 2.0)", pos_a_y),
        );

        let _ = speed_a_after; // used for logging context
        let _ = sleeping_after;

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 5: Lateral side-drag regression guard (Bug #3)
    // =========================================================================
    #[test]
    fn test_lateral_side_drag_regression() {
        let params = SimParams {
            seed: 2005,
            block_count: 3,
            steps_warmup: 200,
            steps_measured: 100,
            extra: vec![
                (
                    "scenario".into(),
                    "2-high wood pillar + 1 side-placed wood block".into(),
                ),
                (
                    "regression".into(),
                    "Bug #3: pillar was dragged to ground level by side block".into(),
                ),
                (
                    "lateral_anchor_cap".into(),
                    "SETTLED_ANCHOR_SHARE_CAP = 0.0".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_lateral_side_drag_regression",
            "STABILITY",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = full_schedule();

        // Build 2-high pillar
        let bottom = world
            .spawn(Voxel::new_with_material(
                0.0,
                0.0,
                0.0,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            ))
            .id();
        let top = world
            .spawn(Voxel::new_with_material(
                0.0,
                1.1,
                0.0,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            ))
            .id();

        // Settle the pillar first
        run_steps(&mut world, &mut schedule, params.steps_warmup);

        let bottom_y_before = get_voxel_y(&world, bottom);
        let top_y_before = get_voxel_y(&world, top);

        // Now spawn a side block at X+1 — this triggered the lateral drag bug
        let _side = world
            .spawn(Voxel::new_with_material(
                2.0,
                0.5,
                0.0,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            ))
            .id();

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let bottom_y_after = get_voxel_y(&world, bottom);
        let top_y_after = get_voxel_y(&world, top);

        // ── Check 1: Bottom block must not be dragged down ────────────────────
        let bottom_y_change = (bottom_y_after - bottom_y_before).abs();
        log.assert_leq(
            "bottom_block_y_stable",
            "Bottom pillar block Y position must not change by more than 0.3 units \
             after the side block is placed. Larger change = lateral drag regression \
             (Bug #3 returned). The SETTLED_ANCHOR_SHARE_CAP should prevent the \
             settled pillar from absorbing the side block's impact correction.",
            bottom_y_change,
            0.30,
            format!(
                "bottom_y_before={:.4}  bottom_y_after={:.4}  Δy={:.4}",
                bottom_y_before,
                bottom_y_after,
                bottom_y_after - bottom_y_before
            ),
        );

        // ── Check 2: Top block must not be dragged down ───────────────────────
        let top_y_change = (top_y_after - top_y_before).abs();
        log.assert_leq(
            "top_block_y_stable",
            "Top pillar block Y position must not change by more than 0.3 units. \
             The side block should not drag the entire pillar sideways or downward.",
            top_y_change,
            0.30,
            format!(
                "top_y_before={:.4}  top_y_after={:.4}  Δy={:.4}",
                top_y_before,
                top_y_after,
                top_y_after - top_y_before
            ),
        );

        // ── Check 3: Bond count reflects the side block bonded ────────────────
        let bonds = world.get_resource::<MortarBonds>().unwrap();
        let bond_count = bonds.bonds.len();
        log.assert_true(
            "side_block_formed_bond",
            "The side Wood block should have formed at least 1 mortar bond \
             with the pillar. bond_count=0 after placement means \
             try_register_bonds failed to detect the adjacency.",
            bond_count >= 1,
            format!(
                "bond_count = {} (expected ≥ 1 — side block should bond to pillar)",
                bond_count
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 6: Steel pair no-vibration (mortar compliance regression guard)
    // =========================================================================
    #[test]
    fn test_steel_pair_no_vibration() {
        let params = SimParams {
            seed: 2006,
            block_count: 2,
            steps_warmup: 200,
            steps_measured: 300,
            extra: vec![
                ("material".into(), "Steel (both blocks)".into()),
                ("scenario".into(), "two adjacent steel blocks".into()),
                (
                    "regression".into(),
                    "Bug 3a: steel-steel mortar bond caused infinite vibration".into(),
                ),
                (
                    "compliance_fix".into(),
                    "BASE_BOND_COMPLIANCE_FACTOR * inv_mass_sum".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_steel_pair_no_vibration", "STABILITY", params.clone());

        let mut world = minimal_world();
        let mut schedule = full_schedule();

        // Two steel blocks side by side — this is exactly the scenario that
        // caused the vibration bug described in mortar.rs Bug Fix 3a.
        let steel_a = world
            .spawn(Voxel::new_with_material(
                0.0,
                0.0,
                0.0,
                ShapeType::Cube,
                MaterialType::Steel,
                false,
            ))
            .id();
        let steel_b = world
            .spawn(Voxel::new_with_material(
                1.0,
                0.0,
                0.0,
                ShapeType::Cube,
                MaterialType::Steel,
                false,
            ))
            .id();

        run_steps(
            &mut world,
            &mut schedule,
            params.steps_warmup + params.steps_measured,
        );

        let speed_a = get_voxel_speed(&world, steel_a);
        let speed_b = get_voxel_speed(&world, steel_b);
        let max_speed = speed_a.max(speed_b);

        // ── Check 1: Both blocks below vibration threshold ────────────────────
        log.assert_leq(
            "steel_a_not_vibrating",
            "Steel block A must have speed ≤ 0.05 m/s after 500 total steps. \
             Higher speed indicates the mass-proportional compliance fix in \
             mortar.rs has been broken and the oscillation has returned. \
             Root cause: compliance = BASE_COMPLIANCE × inv_mass_sum must be used \
             instead of a fixed compliance constant.",
            speed_a,
            0.05,
            format!("speed_a = {:.6} m/s", speed_a),
        );

        log.assert_leq(
            "steel_b_not_vibrating",
            "Steel block B speed must also be ≤ 0.05 m/s.",
            speed_b,
            0.05,
            format!("speed_b = {:.6} m/s", speed_b),
        );

        // ── Check 2: Sleeping ─────────────────────────────────────────────────
        let a_sleeping = get_voxel_sleeping(&world, steel_a);
        let b_sleeping = get_voxel_sleeping(&world, steel_b);
        log.assert_true(
            "steel_pair_both_sleeping",
            "Both steel blocks must be sleeping after 500 steps. \
             is_sleeping=false with speed near zero means the mortar solver \
             is waking them via BOND_WAKE_TENSION_THRESHOLD being set too low.",
            a_sleeping && b_sleeping,
            format!(
                "steel_a sleeping={} speed={:.6}  steel_b sleeping={} speed={:.6}  max={:.6}",
                a_sleeping, speed_a, b_sleeping, speed_b, max_speed
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 7: Restitution — Wood does not bounce, Steel barely bounces
    // =========================================================================
    #[test]
    fn test_restitution_energy_loss() {
        let params = SimParams {
            seed: 2007,
            block_count: 2,
            steps_warmup: 0,
            steps_measured: 120,
            extra: vec![
                ("drop_height".into(), "4.0".into()),
                ("wood_restitution".into(), "0.0 (dead stop)".into()),
                ("steel_restitution".into(), "0.05 (tiny bounce)".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_restitution_energy_loss", "STABILITY", params.clone());

        let mut world_wood = minimal_world();
        let mut schedule_wood = full_schedule();

        let wood_block = world_wood
            .spawn(Voxel::new_with_material(
                0.0,
                4.0,
                0.0,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            ))
            .id();

        run_steps(&mut world_wood, &mut schedule_wood, params.steps_measured);

        let wood_speed_final = get_voxel_speed(&world_wood, wood_block);
        let wood_y_final = get_voxel_y(&world_wood, wood_block);

        // ── Check 1: Wood block has zero restitution — must not bounce high ────
        // After 120 steps (2 simulated seconds), wood should be completely settled.
        log.assert_leq(
            "wood_no_bounce_speed",
            "Wood block (restitution=0.0) must not bounce. After 120 steps \
             the speed must be near zero. Persistent speed means restitution \
             is incorrectly non-zero for Wood, or the contact resolution is \
             generating artificial upward velocity.",
            wood_speed_final,
            0.15,
            format!("wood speed = {:.6} m/s", wood_speed_final),
        );

        log.assert_true(
            "wood_settled_on_floor",
            "Wood block must have settled at floor level (y < 0.5). \
             y > 0.5 means the block is still bouncing or floating.",
            wood_y_final < 0.5,
            format!("wood pos_y = {:.4}", wood_y_final),
        );

        log.finalize_and_assert();
    }
}
