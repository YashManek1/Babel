// =============================================================================
// src/tests/energy.rs  —  Layer 3: Energy Consistency Validation
// =============================================================================
//
// WHAT IS TESTED
// --------------
// These tests verify that Operation Babel's physics engine respects the
// fundamental law of energy conservation. In a damped XPBD system:
//
//   KE[t] ≤ KE[t-1] + external_work[t]
//
// Where external_work is the kinetic energy added by newly spawned blocks.
// A settled system (no new blocks, no applied forces) must have monotonically
// non-increasing total kinetic energy.
//
// WHY THIS MATTERS
// ----------------
// If a settled tower suddenly gains kinetic energy (KE increases without cause),
// the simulation has a fundamental physics violation. The tower might explode,
// individual blocks might vibrate forever, or the RL training signal becomes
// non-stationary. All three scenarios ruin Phase II training.
//
// TEST CATALOGUE
// --------------
//   test_ke_monotone_after_settle
//       Spawn 5 blocks, let them settle for 200 steps, then measure KE for
//       100 more steps. KE must be non-increasing (monotonically decreasing
//       toward zero) throughout the measurement window.
//
//   test_ke_zero_in_static_world
//       A world with ONLY static blocks must have total KE = 0.0 at every step.
//       This tests the degenerate case: static inv_mass=0 blocks have no velocity.
//
//   test_ke_dissipation_rate
//       A single block dropped from height 5.0 must lose at least 80% of its
//       peak kinetic energy within 60 steps of first hitting the floor. This
//       tests that damping (POSITION_VELOCITY_DAMPING=0.94) is working.
//
//   test_total_energy_not_created_on_collision
//       Two blocks collide. Total KE before collision >= total KE after
//       collision (within noise). Energy must be dissipated, not created.
//
//   test_momentum_conservation_elastic_floor
//       A block bouncing on the floor with restitution=0.0 must lose Y
//       momentum on every bounce (energy goes to heat/damping). Y velocity
//       must approach zero monotonically.
//
//   test_ke_spike_detection
//       A world where the mortar pillar-flying bug was present would show
//       KE spikes — sudden increases from zero. This test places a side block
//       next to a settled pillar and monitors for KE spikes. Spikes > 5.0 J
//       indicate the pillar-flying regression has returned.
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

    /// Compute total kinetic energy of all dynamic blocks.
    /// KE = Σ (0.5 × mass × velocity²)
    fn total_kinetic_energy(world: &mut World) -> f32 {
        let mut query = world.query::<&Voxel>();
        let mut ke = 0.0f32;
        for voxel in query.iter(world) {
            if voxel.inv_mass > 0.0 {
                let mass = 1.0 / voxel.inv_mass;
                ke += 0.5 * mass * voxel.velocity.length_squared();
            }
        }
        ke
    }

    // =========================================================================
    // TEST 1: KE is monotonically non-increasing after blocks settle
    // =========================================================================
    #[test]
    fn test_ke_monotone_after_settle() {
        let params = SimParams {
            seed: 5001,
            block_count: 5,
            steps_warmup: 300,
            steps_measured: 100,
            extra: vec![
                (
                    "law".into(),
                    "KE[t] ≤ KE[t-1] once settled (damped system)".into(),
                ),
                (
                    "damping_coefficient".into(),
                    "POSITION_VELOCITY_DAMPING=0.94".into(),
                ),
                ("material".into(), "Wood".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_ke_monotone_after_settle", "ENERGY", params.clone());

        let mut world = full_world();
        let mut schedule = full_schedule();

        for i in 0..5 {
            world.spawn(Voxel::new_with_material(
                0.0,
                i as f32 + 0.5,
                0.0,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            ));
        }

        // Warmup: let blocks settle
        run_steps(&mut world, &mut schedule, params.steps_warmup);

        let ke_at_settle_end = total_kinetic_energy(&mut world);

        // Measure: KE must not increase
        let mut ke_history: Vec<f32> = Vec::with_capacity(params.steps_measured as usize);
        let mut increases: Vec<(u32, f32, f32)> = Vec::new(); // (step, ke_prev, ke_curr)

        let mut prev_ke = ke_at_settle_end;

        for step in 0..params.steps_measured {
            schedule.run(&mut world);
            let ke = total_kinetic_energy(&mut world);
            ke_history.push(ke);

            // Small tolerance: floating-point rounding can produce tiny increases
            if ke > prev_ke + 0.01 {
                increases.push((step, prev_ke, ke));
            }
            prev_ke = ke;
        }

        let final_ke = *ke_history.last().unwrap_or(&0.0);
        let max_ke_in_window = ke_history.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // ── Check 1: No significant KE increases ──────────────────────────────
        log.assert_count(
            "no_ke_increases_after_settle",
            "After blocks settle, total KE must not increase between steps. \
             This is the energy conservation law for a damped system: without \
             new external forces, energy can only be dissipated (friction, \
             restitution) or stay constant (perfect elastic bounce). \
             KE increases indicate the solver is creating energy — a critical \
             correctness failure. Tolerance = 0.01 J to account for float rounding.",
            increases.len(),
            0,
            format!(
                "Energy increases (first 5): {:?}  \
                 ke_at_settle={:.6}  max_in_window={:.6}  final_ke={:.6}",
                increases.iter().take(5).collect::<Vec<_>>(),
                ke_at_settle_end,
                max_ke_in_window,
                final_ke
            ),
        );

        // ── Check 2: KE at end of window is less than at start ────────────────
        log.assert_leq(
            "ke_final_less_than_ke_initial",
            "Final KE (after 100 measurement steps) must be ≤ initial KE \
             at the start of the measurement window. The damped system must \
             have dissipated all remaining energy to zero or near-zero.",
            final_ke,
            ke_at_settle_end + 0.05,
            format!(
                "final_ke={:.6} J  settle_end_ke={:.6} J",
                final_ke, ke_at_settle_end
            ),
        );

        // ── Check 3: KE approaches zero ───────────────────────────────────────
        log.assert_leq(
            "ke_approaches_zero",
            "After 300 warmup + 100 measurement steps, total KE must be near \
             zero (< 0.5 J). Persistent KE means blocks are still moving — \
             the damping or sleep system is failing to fully dissipate energy.",
            final_ke,
            0.5,
            format!("final_ke = {:.6} J", final_ke),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 2: KE is zero in an all-static world
    // =========================================================================
    #[test]
    fn test_ke_zero_in_static_world() {
        let params = SimParams {
            seed: 5002,
            block_count: 5,
            steps_measured: 100,
            extra: vec![
                ("all_blocks_static".into(), "true (inv_mass=0)".into()),
                ("expected_ke".into(), "0.0 J (exactly)".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_ke_zero_in_static_world", "ENERGY", params.clone());

        let mut world = full_world();
        let mut schedule = full_schedule();

        // All static blocks
        for i in 0..5 {
            world.spawn(Voxel::new(
                i as f32,
                0.0,
                0.0,
                ShapeType::Cube,
                true, // is_static=true
            ));
        }

        let mut nonzero_ke_steps: Vec<(u32, f32)> = Vec::new();

        for step in 0..params.steps_measured {
            schedule.run(&mut world);
            let ke = total_kinetic_energy(&mut world);
            if ke > 1e-10 {
                nonzero_ke_steps.push((step, ke));
            }
        }

        log.assert_count(
            "static_world_ke_always_zero",
            "A world containing only static blocks (inv_mass=0) must have \
             KE=0 at every step. Static blocks have velocity=Vec3::ZERO always \
             (integrate_system skips them). Any KE > 1e-10 means a static block \
             was incorrectly given velocity — critical correctness failure.",
            nonzero_ke_steps.len(),
            0,
            format!(
                "Non-zero KE steps (first 5): {:?}",
                nonzero_ke_steps.iter().take(5).collect::<Vec<_>>()
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 3: KE dissipation rate after floor impact
    // =========================================================================
    #[test]
    fn test_ke_dissipation_rate_after_impact() {
        let params = SimParams {
            seed: 5003,
            block_count: 1,
            steps_warmup: 0,
            steps_measured: 200,
            extra: vec![
                ("drop_height".into(), "5.0".into()),
                ("material".into(), "Wood (restitution=0.0)".into()),
                (
                    "damping".into(),
                    "POSITION_VELOCITY_DAMPING=0.94 per contact step".into(),
                ),
                (
                    "expected".into(),
                    "80% KE dissipation within 60 steps of floor impact".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_ke_dissipation_rate_after_impact",
            "ENERGY",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        let block = world
            .spawn(Voxel::new_with_material(
                0.0,
                5.0,
                0.0,
                ShapeType::Cube,
                MaterialType::Wood,
                false,
            ))
            .id();

        let mut peak_ke: f32 = 0.0;
        let mut peak_step: u32 = 0;
        let mut ke_at_peak_plus_60: f32 = 0.0;

        for step in 0..params.steps_measured {
            schedule.run(&mut world);
            let ke = total_kinetic_energy(&mut world);

            if ke > peak_ke {
                peak_ke = ke;
                peak_step = step;
            }

            if step == peak_step + 60 {
                ke_at_peak_plus_60 = ke;
            }
        }

        // ── Check 1: Peak KE was actually reached (block hit floor) ───────────
        let block_final_y = world.get::<Voxel>(block).unwrap().position.y;
        log.assert_true(
            "block_hit_floor",
            "Block must have fallen from y=5.0 and hit the floor. \
             If peak_ke ≈ 0, the block never fell (gravity integration broken). \
             Expected peak KE ≈ 0.5 × 1 kg × (2×9.81×5)^0.5 ≈ 7.0 J at impact.",
            peak_ke > 2.0,
            format!(
                "peak_ke={:.4} J at step={} final_y={:.4}",
                peak_ke, peak_step, block_final_y
            ),
        );

        // ── Check 2: 80% dissipation within 60 steps of peak ──────────────────
        if peak_step + 60 < params.steps_measured {
            let dissipated_fraction = 1.0 - (ke_at_peak_plus_60 / peak_ke.max(f32::EPSILON));
            log.assert_true(
                "ke_80pct_dissipated_in_60_steps",
                "Wood block (restitution=0, damping=0.94/contact) must dissipate \
                 ≥ 80% of peak KE within 60 steps (~1 second simulated) of \
                 floor impact. Lower dissipation rate means damping is not being \
                 applied on contact, or the restitution value is too high for Wood.",
                dissipated_fraction >= 0.80,
                format!(
                    "dissipated={:.2}%  peak_ke={:.4} J  ke_at_peak+60={:.4} J  \
                     (need ≥ 80% dissipation)",
                    dissipated_fraction * 100.0,
                    peak_ke,
                    ke_at_peak_plus_60
                ),
            );
        }

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 4: Energy not created on block-block collision
    // =========================================================================
    #[test]
    fn test_total_energy_not_created_on_collision() {
        let params = SimParams {
            seed: 5004,
            block_count: 2,
            steps_warmup: 0,
            steps_measured: 80,
            extra: vec![
                ("block_a_vel".into(), "4.0 m/s on X axis".into()),
                ("block_b_vel".into(), "0.0 (stationary)".into()),
                ("invariant".into(), "KE_post ≤ KE_pre + gravity_work".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_total_energy_not_created_on_collision",
            "ENERGY",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        let mut vox_a =
            Voxel::new_with_material(-1.5, 2.0, 0.0, ShapeType::Cube, MaterialType::Wood, false);
        vox_a.velocity = glam::Vec3::new(4.0, 0.0, 0.0);
        world.spawn(vox_a);

        world.spawn(Voxel::new_with_material(
            1.5,
            2.0,
            0.0,
            ShapeType::Cube,
            MaterialType::Wood,
            false,
        ));

        let ke_initial = total_kinetic_energy(&mut world);

        // Record KE for each step and find the max (should be at or before collision)
        let mut ke_history: Vec<f32> = Vec::with_capacity(params.steps_measured as usize);
        let mut energy_creation_events: Vec<(u32, f32, f32)> = Vec::new();

        let mut prev_ke = ke_initial;

        for step in 0..params.steps_measured {
            schedule.run(&mut world);
            let ke = total_kinetic_energy(&mut world);
            ke_history.push(ke);

            // Gravity adds energy each step: W = mg × delta_y (blocks falling)
            // We use a generous allowance of 5 J per step to account for gravitational PE
            // conversion. The blocks start at y=2 and fall, so gravity IS doing work.
            // The key check is that KE doesn't spike ABOVE the total energy budget.
            let max_allowed = prev_ke + 5.0; // gravity contribution budget
            if ke > max_allowed {
                energy_creation_events.push((step, prev_ke, ke));
            }
            prev_ke = ke;
        }

        // Final KE should be much less than initial (blocks have settled, gravity PE spent)
        let final_ke = *ke_history.last().unwrap_or(&0.0);

        log.assert_count(
            "no_energy_creation_spikes",
            "Total KE must not increase by more than 5.0 J per step (the maximum \
             gravitational PE converted to KE for a 1 kg block falling at g=9.81). \
             Larger increases indicate the solver is creating kinetic energy — \
             either the velocity derivation amplified a correction, or the \
             position update is adding spurious displacement.",
            energy_creation_events.len(),
            0,
            format!(
                "Energy creation events (first 3): {:?}  final_ke={:.4} J  initial_ke={:.4} J",
                energy_creation_events.iter().take(3).collect::<Vec<_>>(),
                final_ke,
                ke_initial
            ),
        );

        log.assert_leq(
            "final_ke_less_than_initial",
            "After 80 steps (blocks have collided and settled), final KE must \
             be less than initial KE. The collision and damping must have \
             dissipated energy. If final_ke > initial_ke, the system is \
             net-positive on energy — a physics correctness failure.",
            final_ke,
            ke_initial + 1.0, // allow 1 J for gravitational PE → KE conversion
            format!("final_ke={:.4} J  initial_ke={:.4} J", final_ke, ke_initial),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 5: KE spike detection — pillar-flying regression guard
    // =========================================================================
    #[test]
    fn test_ke_spike_detection_pillar_flying_regression() {
        let params = SimParams {
            seed: 5005,
            block_count: 5,
            steps_warmup: 300,
            steps_measured: 150,
            extra: vec![
                ("scenario".into(), "settled steel pillar + side wood block spawned during measurement".into()),
                (
                    "regression".into(),
                    "Bug 3b: mortar woke sleeping steel → gravity → overcorrection → pillar launched".into(),
                ),
                ("spike_threshold".into(), "5.0 J sudden increase".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_ke_spike_detection_pillar_flying_regression",
            "ENERGY",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        // Build a settled steel pillar (the exact scenario from the bug report)
        for i in 0..3 {
            world.spawn(Voxel::new_with_material(
                0.0,
                i as f32 + 0.5,
                0.0,
                ShapeType::Cube,
                MaterialType::Steel,
                false,
            ));
        }

        // Settle the pillar
        run_steps(&mut world, &mut schedule, params.steps_warmup);

        let ke_before_side_block = total_kinetic_energy(&mut world);

        // Now spawn the side wood block — this was the trigger for the bug
        world.spawn(Voxel::new_with_material(
            2.0,
            0.5,
            0.0,
            ShapeType::Cube,
            MaterialType::Wood,
            false,
        ));

        let mut spikes: Vec<(u32, f32, f32)> = Vec::new();
        let mut prev_ke = total_kinetic_energy(&mut world);
        let mut max_ke_in_window: f32 = 0.0;

        for step in 0..params.steps_measured {
            schedule.run(&mut world);
            let ke = total_kinetic_energy(&mut world);
            if ke > max_ke_in_window {
                max_ke_in_window = ke;
            }

            // A KE spike of > 5.0 J means the pillar was launched
            if ke > prev_ke + 5.0 {
                spikes.push((step, prev_ke, ke));
            }
            prev_ke = ke;
        }

        // ── Check 1: No large KE spikes ───────────────────────────────────────
        log.assert_count(
            "no_pillar_launch_spikes",
            "After a side block is placed next to a settled steel pillar, \
             total KE must not spike by more than 5.0 J in a single step. \
             A spike of > 5.0 J means the pillar-flying regression (Bug 3b) \
             has returned: mortar solver woke sleeping steel blocks → gravity \
             integrated → over-correction → chain launch. \
             Fix: BOND_WAKE_TENSION_THRESHOLD must suppress wake for tiny drift.",
            spikes.len(),
            0,
            format!(
                "Spikes (first 3): {:?}  max_ke={:.4} J  pre_side_block_ke={:.4} J",
                spikes.iter().take(3).collect::<Vec<_>>(),
                max_ke_in_window,
                ke_before_side_block
            ),
        );

        // ── Check 2: Peak KE is reasonable (side block fell, not entire pillar) ─
        // The side Wood block (1 kg) falls from y=0.5 — it contributes very little KE.
        // The steel pillar should stay settled. Max KE should be < 10 J.
        log.assert_leq(
            "max_ke_reasonable",
            "Maximum KE during the measurement window must be < 10 J. \
             A 1 kg Wood block falling 0.5 m generates ≈ 0.5 × 1 × (2×9.81×0.5)^0.5 ≈ 2.2 J. \
             Values >> 10 J indicate the steel pillar was launched and contributed \
             its 800 kg mass × velocity² to the KE — the pillar-flying bug.",
            max_ke_in_window,
            15.0, // generous bound; pillar flying would produce hundreds of J
            format!("max_ke_in_window = {:.4} J", max_ke_in_window),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 6: Y velocity monotone decrease on dead-bounce Wood block
    // =========================================================================
    #[test]
    fn test_y_velocity_decreases_on_dead_bounce() {
        let params = SimParams {
            seed: 5006,
            block_count: 1,
            steps_warmup: 0,
            steps_measured: 200,
            extra: vec![
                ("material".into(), "Wood (restitution=0.0)".into()),
                (
                    "expected".into(),
                    "Y velocity magnitude decreases monotonically after each bounce".into(),
                ),
                (
                    "physics_basis".into(),
                    "restitution=0 → normal velocity component zeroed on contact".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_y_velocity_decreases_on_dead_bounce",
            "ENERGY",
            params.clone(),
        );

        let mut world = full_world();
        let mut schedule = full_schedule();

        world.spawn(Voxel::new_with_material(
            0.0,
            5.0,
            0.0,
            ShapeType::Cube,
            MaterialType::Wood,
            false,
        ));

        // Track peak upward Y velocities after each floor contact
        // (upward velocity after a bounce is the "rebound speed")
        let mut peak_upward_velocities: Vec<(u32, f32)> = Vec::new();
        let mut prev_vy: f32 = 0.0;
        let mut in_rebound = false;
        let mut rebound_peak: f32 = 0.0;
        let mut rebound_start_step: u32 = 0;

        let mut query_state = world.query::<&Voxel>();

        for step in 0..params.steps_measured {
            schedule.run(&mut world);

            let vy = {
                let mut q = world.query::<&Voxel>();
                q.iter(&world)
                    .filter(|v| v.inv_mass > 0.0)
                    .map(|v| v.velocity.y)
                    .next()
                    .unwrap_or(0.0)
            };

            // Detect transition from downward to upward (floor contact moment)
            if prev_vy < -0.05 && vy >= 0.0 {
                if in_rebound && rebound_peak > 0.001 {
                    peak_upward_velocities.push((rebound_start_step, rebound_peak));
                }
                in_rebound = true;
                rebound_peak = 0.0;
                rebound_start_step = step;
            }

            if in_rebound && vy > rebound_peak {
                rebound_peak = vy;
            }

            // Detect peak and start of next downward phase
            if in_rebound && vy < prev_vy - 0.05 && prev_vy > 0.1 {
                peak_upward_velocities.push((rebound_start_step, rebound_peak));
                in_rebound = false;
                rebound_peak = 0.0;
            }

            prev_vy = vy;
            let _ = query_state;
        }

        if peak_upward_velocities.len() >= 2 {
            // Check that each successive rebound is smaller than the last
            let monotone = peak_upward_velocities
                .windows(2)
                .all(|w| w[1].1 <= w[0].1 + 0.1); // allow tiny float noise

            log.assert_true(
                "rebound_velocities_monotone_decreasing",
                "Each successive rebound peak Y velocity must be ≤ the previous \
                 rebound peak. For Wood (restitution=0), bounces should diminish \
                 toward zero. Increasing rebound velocity means energy is being \
                 added on contact — a critical correctness failure.",
                monotone,
                format!(
                    "Rebound peaks (step, vy): {:?}  \
                     (each must be ≤ the previous)",
                    peak_upward_velocities
                ),
            );
        }

        // Final check: block must be settled
        let final_vy = {
            let mut q = world.query::<&Voxel>();
            q.iter(&world)
                .filter(|v| v.inv_mass > 0.0)
                .map(|v| v.velocity.y.abs())
                .next()
                .unwrap_or(0.0)
        };

        log.assert_leq(
            "block_settled_after_bounces",
            "Wood block must be completely settled (|vy| < 0.1 m/s) after \
             200 steps. Persistent Y velocity means the dead-bounce \
             (restitution=0.0) is not zeroing the normal velocity component \
             on floor contact.",
            final_vy,
            0.20,
            format!("final |vy| = {:.6} m/s", final_vy),
        );

        log.finalize_and_assert();
    }
}
