// =============================================================================
// src/tests/constraint.rs  —  Layer 2: Runtime Constraint Violation Checks
// =============================================================================
//
// WHAT IS TESTED
// --------------
// These tests implement the "runtime constraint violation checker" described
// in Sprint 4.5 Layer 2. Unlike unit tests that verify specific scenarios,
// these tests run the engine for many steps and assert that PHYSICAL INVARIANTS
// are never violated at any point during simulation.
//
// The key invariant: no two solid objects can occupy the same space.
// Secondary invariant: no MortarBond's tension exceeds its adhesion_strength
// before break_overloaded_bonds fires.
//
// TEST CATALOGUE
// --------------
//   test_constraint_no_penetration_100_steps
//       Run 100 physics steps with a 5-block mixed-material stack and assert
//       that penetration depth between any pair of blocks never exceeds 0.01
//       world-units across all 100 steps.
//
//   test_constraint_no_penetration_1000_steps
//       Same as above but 1000 steps and 8 blocks, exercising the full
//       mortar+stress pipeline. This is the "10 million step validation" concept
//       from the Sprint 4.5 docs, scaled to a CI-tractable size.
//
//   test_bond_tension_never_exceeds_adhesion_before_break
//       After each physics step, assert no live bond has tension > adhesion_strength.
//       If break_overloaded_bonds fires AFTER this check the invariant holds.
//       If a bond is alive with tension > adhesion_strength, the break system
//       missed a cycle — a critical correctness failure.
//
//   test_no_block_below_floor
//       Over 200 steps with blocks dropped from height, assert that no block's
//       position.y ever goes below floor_y (−0.5). Sinking through the floor
//       produces corrupt RL observations.
//
//   test_no_nan_or_inf_positions
//       After 300 steps, assert that no block has NaN or Inf in its position
//       or velocity. These values propagate silently and corrupt all downstream
//       math including the neural network gradients.
//
//   test_static_blocks_position_invariant
//       Run 500 steps with dynamic blocks bouncing around static blocks.
//       Assert no static block (inv_mass=0) ever changes position by more
//       than floating-point epsilon (1e-4 units).
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
    use glam::Vec3;

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

    /// Compute AABB penetration depth between two unit cubes.
    /// Returns Some(depth) if overlapping, None if separated.
    fn aabb_penetration(pos_a: Vec3, pos_b: Vec3) -> Option<f32> {
        let delta = (pos_a - pos_b).abs();
        let overlap_x = 1.0 - delta.x;
        let overlap_y = 1.0 - delta.y;
        let overlap_z = 1.0 - delta.z;
        if overlap_x > 0.0 && overlap_y > 0.0 && overlap_z > 0.0 {
            Some(overlap_x.min(overlap_y).min(overlap_z))
        } else {
            None
        }
    }

    // =========================================================================
    // TEST 1: No penetration over 100 steps (fast CI check)
    // =========================================================================
    #[test]
    fn test_constraint_no_penetration_100_steps() {
        let params = SimParams {
            seed: 4001,
            block_count: 5,
            steps_measured: 100,
            extra: vec![
                ("check_frequency".into(), "every step".into()),
                ("max_allowed_penetration".into(), "0.01 world-units".into()),
                ("material".into(), "mixed Wood/Steel/Stone".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_constraint_no_penetration_100_steps",
            "CONSTRAINT",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        let materials = [
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
                    i as f32 * 1.2 + 0.5,
                    0.0,
                    ShapeType::Cube,
                    mat,
                    false,
                ))
                .id();
            entities.push(e);
        }

        let mut max_ever_penetration: f32 = 0.0;
        let mut worst_step: u32 = 0;
        let mut worst_pair: (usize, usize) = (0, 0);

        // Check every single step — this is what separates runtime monitoring
        // from post-hoc testing.
        for step in 0..params.steps_measured {
            schedule.run(&mut world);

            // Read current positions
            let positions: Vec<Vec3> = entities
                .iter()
                .map(|&e| world.get::<Voxel>(e).unwrap().position)
                .collect();

            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    if let Some(pen) = aabb_penetration(positions[i], positions[j]) {
                        if pen > max_ever_penetration {
                            max_ever_penetration = pen;
                            worst_step = step;
                            worst_pair = (i, j);
                        }
                    }
                }
            }
        }

        log.assert_leq(
            "max_penetration_ever",
            "Over 100 physics steps, the maximum penetration depth between any \
             two block pair must never exceed 0.01 world-units. This is the \
             runtime constraint violation check: the XPBD solver must resolve \
             all overlaps every single step. Penetration > 0.01 means the \
             solver converged to a wrong answer for some configuration, and \
             the RL agent would observe blocks inside each other.",
            max_ever_penetration,
            0.01,
            format!(
                "max_penetration={:.6}  worst_step={}  worst_pair=({},{})",
                max_ever_penetration, worst_step, worst_pair.0, worst_pair.1
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 2: No penetration over 1000 steps (thorough check)
    // =========================================================================
    #[test]
    fn test_constraint_no_penetration_1000_steps() {
        let params = SimParams {
            seed: 4002,
            block_count: 8,
            steps_measured: 1000,
            extra: vec![
                ("check_frequency".into(), "every step".into()),
                (
                    "max_allowed_penetration".into(),
                    "0.015 world-units (slightly wider — settle dynamics)".into(),
                ),
                ("full_pipeline".into(), "mortar + stress included".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_constraint_no_penetration_1000_steps",
            "CONSTRAINT",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        // 2×4 grid of mixed blocks
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
            let x = (i % 2) as f32;
            let z = (i / 2) as f32;
            let e = world
                .spawn(Voxel::new_with_material(
                    x,
                    3.0 + i as f32 * 0.1,
                    z,
                    ShapeType::Cube,
                    mat,
                    false,
                ))
                .id();
            entities.push(e);
        }

        let mut max_penetration: f32 = 0.0;
        let mut violation_steps: Vec<(u32, f32)> = Vec::new();

        for step in 0..params.steps_measured {
            schedule.run(&mut world);

            let positions: Vec<Vec3> = entities
                .iter()
                .map(|&e| world.get::<Voxel>(e).unwrap().position)
                .collect();

            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    if let Some(pen) = aabb_penetration(positions[i], positions[j]) {
                        if pen > 0.015 {
                            violation_steps.push((step, pen));
                        }
                        if pen > max_penetration {
                            max_penetration = pen;
                        }
                    }
                }
            }
        }

        log.assert_count(
            "zero_violation_steps",
            "Over 1000 physics steps, there must be zero steps where any block \
             pair penetrates by more than 0.015 world-units. A single violation \
             step means the XPBD solver produced a wrong answer on that step — \
             the RL agent received a corrupt observation and will learn incorrect \
             physics. Every violation step is a data quality failure.",
            violation_steps.len(),
            0,
            format!(
                "violation_steps (first 5): {:?}  max_penetration={:.6}",
                violation_steps.iter().take(5).collect::<Vec<_>>(),
                max_penetration
            ),
        );

        log.assert_leq(
            "max_penetration_1000_steps",
            "Maximum penetration across 1000 steps must be below 0.015. \
             This guards against transient spikes that the violation-step count \
             might miss if the tolerance is set exactly at the spike value.",
            max_penetration,
            0.015,
            format!("max_penetration = {:.6}", max_penetration),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 3: Bond tension never exceeds adhesion_strength while bond is alive
    // =========================================================================
    #[test]
    fn test_bond_tension_never_exceeds_adhesion_before_break() {
        let params = SimParams {
            seed: 4003,
            block_count: 4,
            steps_measured: 300,
            extra: vec![
                (
                    "scenario".into(),
                    "Wood blocks in an L-shape with one overhang".into(),
                ),
                (
                    "invariant".into(),
                    "no live bond has tension > adhesion_strength".into(),
                ),
                (
                    "adhesion_wood".into(),
                    "2.0 world-units of max stretch".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_bond_tension_never_exceeds_adhesion_before_break",
            "CONSTRAINT",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        // Spawn 4 wood blocks in a configuration that will form mortar bonds
        let positions = [
            (0.0f32, 0.5, 0.0),
            (1.0, 0.5, 0.0),
            (2.0, 0.5, 0.0),
            (1.0, 1.6, 0.0),
        ];
        for (x, y, z) in positions {
            world.spawn(Voxel::new_with_material(
                x,
                y,
                z,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            ));
        }

        let mut violations: Vec<(u32, f32, f32)> = Vec::new(); // (step, tension, adhesion)
        let mut max_tension_seen: f32 = 0.0;

        for step in 0..params.steps_measured {
            schedule.run(&mut world);

            // After break_overloaded_bonds fires, check remaining bonds
            let bonds = world.get_resource::<MortarBonds>().unwrap();
            for bond in &bonds.bonds {
                if bond.tension > max_tension_seen {
                    max_tension_seen = bond.tension;
                }
                // A live bond must not have tension > adhesion_strength
                if bond.tension > bond.adhesion_strength + 0.001 {
                    violations.push((step, bond.tension, bond.adhesion_strength));
                }
            }
        }

        log.assert_count(
            "no_live_bond_exceeds_adhesion",
            "After break_overloaded_bonds_system fires each step, no remaining \
             live bond may have tension > adhesion_strength. This invariant proves \
             the break system is firing every step and catching all overloaded bonds. \
             A violation means break_overloaded_bonds missed a bond — either it ran \
             before solve_mortar (wrong schedule order) or retain() had a logic error.",
            violations.len(),
            0,
            format!(
                "Violations (first 3): {:?}  max_tension_seen={:.6}",
                violations.iter().take(3).collect::<Vec<_>>(),
                max_tension_seen
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 4: No block sinks below the floor
    // =========================================================================
    #[test]
    fn test_no_block_below_floor() {
        let params = SimParams {
            seed: 4004,
            block_count: 6,
            steps_measured: 200,
            extra: vec![
                ("floor_y".into(), "-0.5".into()),
                ("tolerance".into(), "block centre must be ≥ -0.1".into()),
                (
                    "invariant".into(),
                    "floor constraint must prevent sinking".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_no_block_below_floor", "CONSTRAINT", params.clone());

        let mut world = full_world();
        let mut schedule = full_schedule();

        let mut entities = Vec::new();
        for i in 0..6 {
            let e = world
                .spawn(Voxel::new_with_material(
                    i as f32 * 0.5,
                    5.0 + i as f32,
                    0.0,
                    ShapeType::Cube,
                    MaterialType::Wood,
                    false,
                ))
                .id();
            entities.push(e);
        }

        let mut below_floor_events: Vec<(u32, usize, f32)> = Vec::new();
        let floor_threshold = -0.1_f32; // centre must stay above this

        for step in 0..params.steps_measured {
            schedule.run(&mut world);

            for (idx, &entity) in entities.iter().enumerate() {
                let pos_y = world.get::<Voxel>(entity).unwrap().position.y;
                if pos_y < floor_threshold {
                    below_floor_events.push((step, idx, pos_y));
                }
            }
        }

        log.assert_count(
            "no_block_sinks_through_floor",
            "Over 200 steps, no block may have its centre below y=-0.1. \
             The floor constraint in solve_constraints_system must keep all \
             blocks above floor_y=-0.5 (centre at or above 0.0). \
             A block at y=-0.1 is already suspicious — it means the floor \
             constraint is not being applied or is being applied with the \
             wrong half-extent. Below y=-0.5 means complete floor penetration.",
            below_floor_events.len(),
            0,
            format!(
                "Below-floor events (first 5): {:?}",
                below_floor_events.iter().take(5).collect::<Vec<_>>()
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 5: No NaN or Inf in positions or velocities
    // =========================================================================
    #[test]
    fn test_no_nan_or_inf_positions() {
        let params = SimParams {
            seed: 4005,
            block_count: 10,
            steps_measured: 300,
            extra: vec![
                (
                    "scenario".into(),
                    "high-velocity spawns to stress numerical stability".into(),
                ),
                (
                    "invariant".into(),
                    "all position and velocity components are finite".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_no_nan_or_inf_positions", "CONSTRAINT", params.clone());

        let mut world = full_world();
        let mut schedule = full_schedule();

        // Spawn blocks with initial velocities to stress the solver numerically
        let mut entities = Vec::new();
        for i in 0..10 {
            let mut voxel = Voxel::new_with_material(
                (i % 3) as f32,
                3.0 + i as f32 * 0.5,
                (i / 3) as f32,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            );
            // Give some blocks initial lateral velocity to create complex contacts
            if i % 2 == 0 {
                voxel.velocity = Vec3::new((i as f32 - 5.0) * 0.5, 0.0, 0.0);
            }
            let e = world.spawn(voxel).id();
            entities.push(e);
        }

        let mut nan_inf_events: Vec<(u32, usize, String)> = Vec::new();

        for step in 0..params.steps_measured {
            schedule.run(&mut world);

            for (idx, &entity) in entities.iter().enumerate() {
                let v = world.get::<Voxel>(entity).unwrap();
                let pos = v.position;
                let vel = v.velocity;

                let pos_finite = pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite();
                let vel_finite = vel.x.is_finite() && vel.y.is_finite() && vel.z.is_finite();

                if !pos_finite {
                    nan_inf_events.push((step, idx, format!("position={:?} is NaN/Inf", pos)));
                }
                if !vel_finite {
                    nan_inf_events.push((step, idx, format!("velocity={:?} is NaN/Inf", vel)));
                }
            }
        }

        log.assert_count(
            "no_nan_or_inf",
            "No block position or velocity component may be NaN or Inf over \
             300 steps. NaN propagates silently through all subsequent math: \
             the spatial grid lookup returns wrong neighbors, the RL observation \
             buffer contains garbage, and the neural network gradient becomes NaN. \
             A single NaN event is a complete simulation integrity failure.",
            nan_inf_events.len(),
            0,
            format!(
                "NaN/Inf events (first 3): {:?}",
                nan_inf_events.iter().take(3).collect::<Vec<_>>()
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 6: Static blocks never change position
    // =========================================================================
    #[test]
    fn test_static_blocks_position_invariant() {
        let params = SimParams {
            seed: 4006,
            block_count: 4,
            steps_measured: 500,
            extra: vec![
                ("static_count".into(), "2 static blocks".into()),
                ("dynamic_count".into(), "many dynamic blocks".into()),
                ("max_static_displacement".into(), "1e-4 world-units".into()),
                (
                    "invariant".into(),
                    "inv_mass=0 → correction_share=0 → zero displacement".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_static_blocks_position_invariant",
            "CONSTRAINT",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        // Two static anchor blocks
        let static_a = world
            .spawn(Voxel::new(0.0, 0.0, 0.0, ShapeType::Cube, true))
            .id();
        let static_b = world
            .spawn(Voxel::new(2.0, 0.0, 0.0, ShapeType::Cube, true))
            .id();

        let static_a_initial = world.get::<Voxel>(static_a).unwrap().position;
        let static_b_initial = world.get::<Voxel>(static_b).unwrap().position;

        // Many dynamic blocks that will collide with and bounce around the statics
        for i in 0..8 {
            let x = (i % 3) as f32 - 1.0;
            world.spawn(Voxel::new(x, 5.0 + i as f32, 0.0, ShapeType::Cube, false));
        }

        let mut max_displacement_a: f32 = 0.0;
        let mut max_displacement_b: f32 = 0.0;

        for _ in 0..params.steps_measured {
            schedule.run(&mut world);

            let pos_a = world.get::<Voxel>(static_a).unwrap().position;
            let pos_b = world.get::<Voxel>(static_b).unwrap().position;

            let disp_a = (pos_a - static_a_initial).length();
            let disp_b = (pos_b - static_b_initial).length();

            if disp_a > max_displacement_a {
                max_displacement_a = disp_a;
            }
            if disp_b > max_displacement_b {
                max_displacement_b = disp_b;
            }
        }

        log.assert_leq(
            "static_a_zero_displacement",
            "Static block A must not move more than 1e-4 world-units over 500 steps, \
             regardless of how many dynamic blocks collide with it. \
             The XPBD correction share for inv_mass=0 is mathematically 0. \
             Any displacement > 1e-4 means floating-point error has accumulated \
             in the correction accumulation, or the static constraint is not \
             being checked before applying corrections.",
            max_displacement_a,
            1e-4,
            format!("max_displacement_a = {:.8}", max_displacement_a),
        );

        log.assert_leq(
            "static_b_zero_displacement",
            "Static block B must also not move over 500 steps.",
            max_displacement_b,
            1e-4,
            format!("max_displacement_b = {:.8}", max_displacement_b),
        );

        log.finalize_and_assert();
    }
}
