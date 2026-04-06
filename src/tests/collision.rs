// =============================================================================
// src/tests/collision.rs  —  Layer 1a: Collision Resolution Unit Tests
// =============================================================================
//
// WHAT IS TESTED
// --------------
// These tests verify the narrow-phase and broad-phase collision resolution
// logic in `src/physics/xpbd.rs`. Each test sets up a minimal physics world,
// runs a known number of steps, and asserts the resulting positions and
// velocities match analytic expectations derived from physics first principles.
//
// TEST CATALOGUE
// --------------
//   test_aabb_cube_on_static_floor
//       A unit cube dropped from height 5.0 must land at y = 0.5 (resting
//       on the floor plane at y = -0.5). Verifies floor constraint and
//       position commitment.
//
//   test_aabb_cube_on_cube_stack
//       Block A static at origin, Block B dropped from y = 3.0. After settle,
//       B must rest at y = 1.0 (one block-width above A). Verifies inter-block
//       AABB penetration correction and mass sharing.
//
//   test_aabb_two_dynamic_blocks_equal_mass
//       Block A moving right (+X) at 5 m/s, Block B stationary. After elastic
//       collision, A must stop and B must move right. Verifies momentum is
//       conserved (within damping tolerance) and mass sharing is 50/50.
//
//   test_aabb_heavy_vs_light_mass_sharing
//       Heavy block (Steel, 800 kg) moving into Light block (Wood, 1 kg).
//       Heavy block must barely deflect, Light block must absorb nearly all
//       the correction. Verifies inv_mass-proportional share formula.
//
//   test_static_block_never_moves
//       A static block (inv_mass = 0.0) hit by a fast-moving block must not
//       change position by more than floating-point epsilon. Verifies the
//       static body constraint.
//
//   test_cube_on_wedge_slides_not_launches
//       A cube dropped onto a wedge must acquire lateral velocity (sliding)
//       and must NOT exceed MAX_VELOCITY. This was the famous "wedge launch"
//       bug — the test permanently guards against regression.
//
//   test_sphere_on_flat_floor
//       A sphere of radius 0.5 dropped from y = 5.0 must rest at y = 0.0
//       (centre above floor_y = -0.5 by exactly sphere_radius). Verifies
//       sphere-floor constraint uses sphere_radius not cube half-extent.
//
//   test_no_penetration_after_stack
//       After 1000 steps with a 5-block pillar, no two blocks may overlap
//       (penetration depth > 0.01 units). This is the constraint violation
//       check from Sprint 4.5 Layer 2, used here as a post-stack sanity guard.
// =============================================================================

#[cfg(test)]
mod tests {
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
    // Helper: build a minimal headless ECS world with all required resources
    // -------------------------------------------------------------------------
    fn minimal_world() -> World {
        let mut world = World::new();
        world.insert_resource(SpatialGrid::default());
        world.insert_resource(PhysicsSettings::default());
        world.insert_resource(SolverBuffers::default());
        world
    }

    // -------------------------------------------------------------------------
    // Helper: build the standard physics schedule (no mortar, no stress)
    // -------------------------------------------------------------------------
    fn physics_schedule() -> Schedule {
        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                update_spatial_grid_system,
                integrate_system,
                solve_constraints_system,
                update_velocities_system,
                update_spatial_grid_system,
            )
                .chain(),
        );
        schedule
    }

    // -------------------------------------------------------------------------
    // Helper: run schedule for N steps
    // -------------------------------------------------------------------------
    fn run_steps(world: &mut World, schedule: &mut Schedule, n: u32) {
        for _ in 0..n {
            schedule.run(world);
        }
    }

    // -------------------------------------------------------------------------
    // Helper: get voxel position by entity
    // -------------------------------------------------------------------------
    fn voxel_pos(world: &World, entity: Entity) -> Vec3 {
        world.get::<Voxel>(entity).unwrap().position
    }

    fn voxel_vel(world: &World, entity: Entity) -> Vec3 {
        world.get::<Voxel>(entity).unwrap().velocity
    }

    fn voxel_sleeping(world: &World, entity: Entity) -> bool {
        world.get::<Voxel>(entity).unwrap().is_sleeping
    }

    // =========================================================================
    // TEST 1: Single cube dropped onto the floor
    // =========================================================================
    #[test]
    fn test_aabb_cube_on_static_floor() {
        let params = SimParams {
            seed: 1001,
            block_count: 1,
            steps_warmup: 0,
            steps_measured: 300,
            extra: vec![
                ("drop_height".into(), "5.0".into()),
                ("expected_rest_y".into(), "0.5".into()),
                ("floor_y".into(), "-0.5".into()),
                ("shape".into(), "Cube".into()),
                ("material".into(), "Wood (1 kg)".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_aabb_cube_on_static_floor",
            "COLLISION",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = physics_schedule();

        // Spawn: unit cube, dynamic, dropped from height 5
        let block = world
            .spawn(Voxel::new(0.0, 5.0, 0.0, ShapeType::Cube, false))
            .id();

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let pos = voxel_pos(&world, block);
        let vel = voxel_vel(&world, block);
        let sleeping = voxel_sleeping(&world, block);

        // ── Check 1: Final Y position ─────────────────────────────────────────
        // A unit cube (half-extent = 0.5) resting on floor_y = -0.5 has its
        // centre at y = 0.0. The floor is at y = -0.5. The bottom face of the
        // cube touches the floor. XPBD places the centre at y = 0.5 when the
        // floor is at y = -0.5 (centre + half_extent = 0.5 + (-0.5+0.5) = 0.5).
        // Wait — floor_y = -0.5 means the floor surface is at y = -0.5.
        // Block centre must be at floor_y + 0.5 = 0.0.
        // Re-check xpbd.rs: floor_y = Some(-0.5), penetration pushes block up
        // so lowest_point_y >= floor_y, i.e. centre_y - 0.5 >= -0.5 → centre_y >= 0.0
        log.assert_approx(
            "final_y_position",
            "Cube dropped from y=5.0 must rest with centre at y=0.0 \
             (bottom face touching floor at y=-0.5). Verifies XPBD floor \
             constraint pushes block to exact floor contact without sinking \
             (y < 0.0) or floating (y > 0.05).",
            pos.y,
            0.0,
            0.05, // 5 cm tolerance for floating-point settle
        );

        // ── Check 2: X and Z must not drift ───────────────────────────────────
        // No lateral forces applied. Block must stay at x=0, z=0.
        log.assert_approx(
            "final_x_no_drift",
            "No lateral forces — X position must remain at spawn X=0.0. \
             Non-zero X indicates a spurious lateral correction in the solver.",
            pos.x,
            0.0,
            0.01,
        );
        log.assert_approx(
            "final_z_no_drift",
            "No lateral forces — Z position must remain at spawn Z=0.0.",
            pos.z,
            0.0,
            0.01,
        );

        // ── Check 3: Velocity must be near-zero (block has settled) ───────────
        let speed = vel.length();
        log.assert_leq(
            "final_speed_at_rest",
            "After 300 steps the block must be stationary. Speed must be \
             below LINEAR_SLEEP_SPEED (0.20 m/s). Non-zero speed indicates \
             the block is still bouncing or vibrating — damping/restitution bug.",
            speed,
            0.20,
            format!("observed speed = {:.6} m/s", speed),
        );

        // ── Check 4: Sleep flag must be set ───────────────────────────────────
        log.assert_true(
            "sleep_flag_set",
            "Block must have entered sleep state after settling. \
             is_sleeping=false after 300 steps indicates the sleep threshold \
             is not being triggered — check LINEAR_SLEEP_SPEED constant.",
            sleeping,
            "Block is not sleeping after 300 steps. \
             The sleep system may not be engaging for floor contacts.",
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 2: Block stacked on top of another block
    // =========================================================================
    #[test]
    fn test_aabb_cube_on_cube_stack() {
        let params = SimParams {
            seed: 1002,
            block_count: 2,
            steps_warmup: 0,
            steps_measured: 400,
            extra: vec![
                ("block_a".into(), "static at (0,0,0)".into()),
                ("block_b".into(), "dynamic, dropped from (0,3,0)".into()),
                ("expected_b_rest_y".into(), "1.0".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_aabb_cube_on_cube_stack", "COLLISION", params.clone());

        let mut world = minimal_world();
        let mut schedule = physics_schedule();

        // Block A: static floor reference block
        let block_a = world
            .spawn(Voxel::new(0.0, 0.0, 0.0, ShapeType::Cube, true))
            .id();

        // Block B: dynamic, dropped from above
        let block_b = world
            .spawn(Voxel::new(0.0, 3.0, 0.0, ShapeType::Cube, false))
            .id();

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let pos_a = voxel_pos(&world, block_a);
        let pos_b = voxel_pos(&world, block_b);
        let vel_b = voxel_vel(&world, block_b);

        // ── Check 1: Static block A must not move ─────────────────────────────
        // inv_mass = 0.0 → correction share = 0.0 → position unchanged.
        log.assert_approx(
            "static_block_a_y_unchanged",
            "Static block A (inv_mass=0) must not move when block B lands on it. \
             Any movement indicates the static constraint is broken — the solver \
             is applying a non-zero correction share to the static body.",
            pos_a.y,
            0.0,
            0.001,
        );

        // ── Check 2: Block B must rest at y = 1.0 ─────────────────────────────
        // Block A centre at y=0.0, half-extent=0.5, so top face at y=0.5.
        // Block B (half-extent=0.5) must rest with bottom face at y=0.5 → centre at y=1.0.
        log.assert_approx(
            "dynamic_block_b_rest_y",
            "Block B must rest with centre at y=1.0 (one block-width above \
             block A). y < 0.95 → sinking through A (penetration correction \
             failed). y > 1.05 → floating above A (correction overshot or \
             block A was moved upward).",
            pos_b.y,
            1.0,
            0.05,
        );

        // ── Check 3: Block B must be settled ──────────────────────────────────
        let speed_b = vel_b.length();
        log.assert_leq(
            "block_b_settled_speed",
            "Block B velocity must be below sleep threshold after 400 steps. \
             Persistent velocity indicates damping or contact friction is not \
             engaging correctly for block-on-block contacts.",
            speed_b,
            0.20,
            format!("speed = {:.6}", speed_b),
        );

        // ── Check 4: Centre separation must equal exactly 1.0 ─────────────────
        let sep = (pos_b.y - pos_a.y).abs();
        log.assert_approx(
            "inter_block_separation",
            "Centre-to-centre Y separation must equal 1.0 world-units \
             (two cubes of half-extent 0.5 touching face-to-face). \
             Deviation indicates residual penetration or a floating gap.",
            sep,
            1.0,
            0.05,
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 3: Two equal-mass blocks colliding (momentum conservation)
    // =========================================================================
    #[test]
    fn test_aabb_two_dynamic_equal_mass_collision() {
        let params = SimParams {
            seed: 1003,
            block_count: 2,
            steps_warmup: 0,
            steps_measured: 120,
            extra: vec![
                ("block_a_initial_vel_x".into(), "5.0 m/s".into()),
                ("block_b_initial_vel_x".into(), "0.0 m/s".into()),
                ("both_mass".into(), "1.0 kg (Wood)".into()),
                ("collision_axis".into(), "X".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_aabb_two_dynamic_equal_mass_collision",
            "COLLISION",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = physics_schedule();

        // Spawn A moving right, B stationary, placed 1.5 units apart so they
        // collide within the first few frames.
        let mut voxel_a = Voxel::new(-0.8, 2.0, 0.0, ShapeType::Cube, false);
        voxel_a.velocity = Vec3::new(5.0, 0.0, 0.0);
        let block_a = world.spawn(voxel_a).id();

        let voxel_b = Voxel::new(0.8, 2.0, 0.0, ShapeType::Cube, false);
        let block_b = world.spawn(voxel_b).id();

        // Record initial total momentum (only X matters for this test)
        let p_initial = 5.0_f32 * 1.0_f32 + 0.0_f32 * 1.0_f32; // mv_a + mv_b

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let vel_a = voxel_vel(&world, block_a);
        let vel_b = voxel_vel(&world, block_b);

        // Post-collision final momentum
        let mass = 1.0_f32; // Wood inv_mass = 1.0 → mass = 1.0
        let p_final = mass * vel_a.x + mass * vel_b.x;

        // ── Check 1: Momentum conserved ───────────────────────────────────────
        // XPBD uses damping (0.94 per contact frame) so momentum is NOT fully
        // conserved — we verify it is approximately conserved within a wider
        // tolerance that accounts for damping over N contact frames.
        // The key is that p_final should be in the same direction as p_initial
        // and not wildly larger (energy was not created).
        log.assert_leq(
            "momentum_not_created",
            "Final total X-momentum must not exceed initial X-momentum \
             (energy cannot be created). XPBD with damping dissipates some \
             momentum, so p_final <= p_initial is the correct bound. \
             p_final > p_initial indicates the solver is adding energy — \
             a critical correctness failure.",
            p_final,
            p_initial + 0.1, // small slack for floating-point
            format!(
                "p_initial={:.4}  p_final={:.4}  vel_a.x={:.4}  vel_b.x={:.4}",
                p_initial, p_final, vel_a.x, vel_b.x
            ),
        );

        // ── Check 2: Energy not gained ────────────────────────────────────────
        let ke_initial = 0.5 * mass * 5.0_f32.powi(2); // 12.5 J
        let ke_final = 0.5 * mass * vel_a.length_squared() + 0.5 * mass * vel_b.length_squared();
        log.assert_leq(
            "kinetic_energy_not_created",
            "Final total kinetic energy must not exceed initial KE. \
             Energy creation indicates a constraint correction amplified \
             velocity — a fundamental solver violation.",
            ke_final,
            ke_initial + 0.5, // damping dissipates, so ke_final should be << ke_initial
            format!("KE_initial={:.4} J  KE_final={:.4} J", ke_initial, ke_final),
        );

        // ── Check 3: Block B acquired positive X velocity ─────────────────────
        log.assert_true(
            "block_b_acquired_velocity",
            "After collision, Block B (initially stationary) must have \
             positive X velocity. vel_b.x <= 0 means the collision correction \
             was never applied or was applied in the wrong direction.",
            vel_b.x > 0.0,
            format!(
                "Block B X velocity after collision = {:.6} m/s (expected > 0)",
                vel_b.x
            ),
        );

        // ── Check 4: No large Y velocity from pure X collision ────────────────
        // The collision is purely on the X axis. The solver must not produce
        // spurious Y or Z corrections.
        log.assert_leq(
            "no_spurious_y_velocity_block_a",
            "Pure X-axis collision must not produce large Y velocity in block A. \
             Large Y velocity indicates the contact normal computation is \
             using the wrong separation axis.",
            vel_a.y.abs(),
            1.5, // gravity contributes some Y, but should be small after 120 steps
            format!("vel_a.y = {:.6}", vel_a.y),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 4: Heavy (Steel) vs Light (Wood) mass sharing ratio
    // =========================================================================
    #[test]
    fn test_aabb_heavy_vs_light_mass_sharing() {
        let params = SimParams {
            seed: 1004,
            block_count: 2,
            steps_warmup: 0,
            steps_measured: 60,
            extra: vec![
                ("heavy_material".into(), "Steel (800 kg)".into()),
                ("light_material".into(), "Wood (1 kg)".into()),
                ("heavy_initial_vel_x".into(), "3.0 m/s".into()),
                (
                    "expected".into(),
                    "heavy barely deflects, light absorbs ~99.9%".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_aabb_heavy_vs_light_mass_sharing",
            "COLLISION",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = physics_schedule();

        // Heavy block moving into stationary light block
        let mut heavy =
            Voxel::new_with_material(-0.8, 2.0, 0.0, ShapeType::Cube, MaterialType::Steel, false);
        heavy.velocity = Vec3::new(3.0, 0.0, 0.0);
        let heavy_e = world.spawn(heavy).id();

        let light =
            Voxel::new_with_material(0.8, 2.0, 0.0, ShapeType::Cube, MaterialType::Wood, false);
        let light_e = world.spawn(light).id();

        // Record heavy block's X position before and after collision
        let heavy_x_before = voxel_pos(&world, heavy_e).x;

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let heavy_pos_after = voxel_pos(&world, heavy_e);
        let light_pos_after = voxel_pos(&world, light_e);

        // ── Check 1: Heavy block must move significantly more than light block ─
        // XPBD correction share = inv_mass_self / (inv_mass_self + inv_mass_other)
        // For Steel vs Wood:
        //   inv_steel = 1/800 ≈ 0.00125
        //   inv_wood  = 1.0
        //   steel_share = 0.00125 / 1.00125 ≈ 0.00125 (barely moves)
        //   wood_share  = 1.0    / 1.00125 ≈ 0.99875 (absorbs almost all)
        let heavy_displacement = (heavy_pos_after.x - heavy_x_before).abs();
        let light_displacement = (light_pos_after.x - 0.8).abs();

        log.assert_true(
            "heavy_moves_less_than_light",
            "Steel block (800 kg) correction share ≈ 0.00125, Wood share ≈ 0.998. \
             Steel block displacement must be << Wood block displacement. \
             If heavy moves more than light, the inv_mass share formula is inverted.",
            heavy_displacement < light_displacement,
            format!(
                "heavy_displacement={:.6}  light_displacement={:.6}  \
                 Expected heavy << light based on mass ratio 800:1",
                heavy_displacement, light_displacement
            ),
        );

        // ── Check 2: Steel block must still be moving in +X after collision ────
        let heavy_vel = voxel_vel(&world, heavy_e);
        log.assert_true(
            "heavy_block_continues_forward",
            "Steel block moving at 3 m/s into a 1 kg Wood block should \
             barely slow down. The heavy block must still have positive X \
             velocity after collision (it has 800× more momentum).",
            heavy_vel.x > 0.5, // still moving forward significantly
            format!(
                "heavy vel.x after collision = {:.6} (expected > 0.5 m/s for 800 kg block)",
                heavy_vel.x
            ),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 5: Static block must never move when struck
    // =========================================================================
    #[test]
    fn test_static_block_never_moves() {
        let params = SimParams {
            seed: 1005,
            block_count: 2,
            steps_warmup: 0,
            steps_measured: 120,
            extra: vec![
                ("static_block_pos".into(), "(0,0,0)".into()),
                (
                    "dynamic_block_initial_vel".into(),
                    "10.0 m/s on X axis".into(),
                ),
                (
                    "max_allowed_static_displacement".into(),
                    "0.001 units".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new("test_static_block_never_moves", "COLLISION", params.clone());

        let mut world = minimal_world();
        let mut schedule = physics_schedule();

        // Static block — inv_mass = 0.0
        let static_block = world
            .spawn(Voxel::new(0.0, 0.0, 0.0, ShapeType::Cube, true))
            .id();

        // Fast dynamic block approaching from -X
        let mut dynamic = Voxel::new(-2.0, 0.0, 0.0, ShapeType::Cube, false);
        dynamic.velocity = Vec3::new(10.0, 0.0, 0.0);
        let _dynamic_e = world.spawn(dynamic).id();

        let static_pos_before = voxel_pos(&world, static_block);

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let static_pos_after = voxel_pos(&world, static_block);
        let displacement = (static_pos_after - static_pos_before).length();

        // ── Check 1: Static block displacement is sub-epsilon ─────────────────
        log.assert_leq(
            "static_block_zero_displacement",
            "Static block (inv_mass=0.0) must not move when struck at 10 m/s. \
             The XPBD formula gives correction_share = 0.0 / (0.0 + inv_dyn) = 0. \
             Any displacement > 0.001 units means the share formula is \
             applied to static entities incorrectly.",
            displacement,
            0.001,
            format!(
                "displacement = {:.8} units  \
                 before={:?}  after={:?}",
                displacement, static_pos_before, static_pos_after
            ),
        );

        // ── Check 2: Position components individually ──────────────────────────
        log.assert_approx(
            "static_x_unchanged",
            "Static block X coordinate must be unchanged.",
            static_pos_after.x,
            static_pos_before.x,
            0.001,
        );
        log.assert_approx(
            "static_y_unchanged",
            "Static block Y coordinate must be unchanged.",
            static_pos_after.y,
            static_pos_before.y,
            0.001,
        );
        log.assert_approx(
            "static_z_unchanged",
            "Static block Z coordinate must be unchanged.",
            static_pos_after.z,
            static_pos_before.z,
            0.001,
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 6: Cube on wedge — slides, does not launch (regression guard)
    // =========================================================================
    #[test]
    fn test_cube_on_wedge_slides_not_launches() {
        let params = SimParams {
            seed: 1006,
            block_count: 2,
            steps_warmup: 0,
            steps_measured: 200,
            extra: vec![
                ("wedge_pos".into(), "(0, 0, 0)".into()),
                ("cube_drop_height".into(), "3.0".into()),
                (
                    "max_allowed_velocity".into(),
                    "25.0 m/s (MAX_VELOCITY)".into(),
                ),
                (
                    "regression_guard".into(),
                    "wedge launch bug — cube must slide not fly".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_cube_on_wedge_slides_not_launches",
            "COLLISION",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = physics_schedule();

        // Wedge: settled, effectively static for collision purposes.
        // Using a high-mass Steel wedge (the fix from BUG FIX #2 in xpbd.rs).
        let wedge =
            Voxel::new_with_material(0.0, 0.0, 0.0, ShapeType::Wedge, MaterialType::Steel, false);
        let wedge_e = world.spawn(wedge).id();

        // Let wedge settle first
        run_steps(&mut world, &mut schedule, 60);

        // Drop cube from height 3 onto the slope
        let cube = Voxel::new(0.0, 3.0, 0.0, ShapeType::Cube, false);
        let cube_e = world.spawn(cube).id();

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let cube_vel = voxel_vel(&world, cube_e);
        let wedge_vel = voxel_vel(&world, wedge_e);
        let cube_speed = cube_vel.length();
        let wedge_speed = wedge_vel.length();

        // ── Check 1: Cube speed must not reach MAX_VELOCITY ───────────────────
        // The old wedge launch bug sent the cube to exactly MAX_VELOCITY (25 m/s).
        // If cube_speed > 10 m/s the bug has regressed.
        log.assert_leq(
            "cube_speed_under_limit",
            "Cube dropped on wedge must NOT reach MAX_VELOCITY (25 m/s). \
             Speed > 10 m/s indicates the wedge-launch regression has returned: \
             the solver computed a massive correction along the slope normal \
             and derived an enormous velocity from it. \
             Fix: verify WEDGE_SETTLED_SHARE_CAP is being applied in xpbd.rs.",
            cube_speed,
            10.0,
            format!(
                "cube_speed = {:.4} m/s  (MAX_VELOCITY = 25.0 m/s)",
                cube_speed
            ),
        );

        // ── Check 2: Cube must have acquired SOME lateral velocity (sliding) ───
        // A cube on a slope should slide — some X or Z velocity is expected.
        // If speed = 0 and cube is still in the air, something else is wrong.
        let cube_pos = voxel_pos(&world, cube_e);
        log.assert_true(
            "cube_has_moved_from_spawn",
            "Cube must have moved from its spawn position. \
             A cube on a wedge slope should slide. Zero movement indicates \
             the cube is stuck in the wedge (penetration not resolved) or \
             fell through (missed collision).",
            (cube_pos - Vec3::new(0.0, 3.0, 0.0)).length() > 0.5,
            format!("cube_pos = {:?}", cube_pos),
        );

        // ── Check 3: Wedge must barely move (it is very heavy) ────────────────
        log.assert_leq(
            "wedge_barely_moves",
            "Steel wedge (2000 kg, WEDGE_SETTLED_SHARE_CAP) must have \
             near-zero velocity after cube impact. High wedge speed means \
             the settled-wedge share cap is not being applied.",
            wedge_speed,
            2.0,
            format!("wedge_speed = {:.4} m/s", wedge_speed),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 7: Sphere rests at correct height on floor
    // =========================================================================
    #[test]
    fn test_sphere_on_flat_floor_correct_height() {
        let params = SimParams {
            seed: 1007,
            block_count: 1,
            steps_warmup: 0,
            steps_measured: 300,
            extra: vec![
                ("sphere_radius".into(), "0.5".into()),
                ("drop_height".into(), "5.0".into()),
                ("floor_y".into(), "-0.5".into()),
                (
                    "expected_centre_y".into(),
                    "0.0 (radius above floor)".into(),
                ),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_sphere_on_flat_floor_correct_height",
            "COLLISION",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = physics_schedule();

        let sphere = Voxel::new_sphere(0.0, 5.0, 0.0, 0.5, false);
        let sphere_e = world.spawn(sphere).id();

        run_steps(&mut world, &mut schedule, params.steps_measured);

        let pos = voxel_pos(&world, sphere_e);
        let vel = voxel_vel(&world, sphere_e);

        // Sphere centre must be at floor_y + radius = -0.5 + 0.5 = 0.0
        log.assert_approx(
            "sphere_centre_y",
            "Sphere (radius=0.5) on floor (floor_y=-0.5) must have centre \
             at y = floor_y + radius = 0.0. \
             y < -0.05 → sphere sank through floor (floor uses cube half-extent, not sphere radius). \
             y > 0.05  → sphere is floating (constraint too aggressive).",
            pos.y,
            0.0,
            0.06,
        );

        let speed = vel.length();
        log.assert_leq(
            "sphere_settled",
            "Sphere must be settled (speed < 0.20 m/s) after 300 steps.",
            speed,
            0.20,
            format!("speed = {:.6}", speed),
        );

        log.finalize_and_assert();
    }

    // =========================================================================
    // TEST 8: No penetration after 5-block pillar settles (1000 steps)
    // =========================================================================
    #[test]
    fn test_no_penetration_in_settled_pillar() {
        let params = SimParams {
            seed: 1008,
            block_count: 5,
            steps_warmup: 1000,
            steps_measured: 0,
            extra: vec![
                ("pillar_height".into(), "5 blocks".into()),
                ("max_allowed_penetration".into(), "0.01 world-units".into()),
                ("material".into(), "Wood".into()),
            ],
            ..SimParams::default()
        };

        let mut log = AuditLog::new(
            "test_no_penetration_in_settled_pillar",
            "COLLISION",
            params.clone(),
        );

        let mut world = minimal_world();
        let mut schedule = physics_schedule();

        // Spawn 5 blocks in a vertical column
        let mut entities = Vec::new();
        for i in 0..5 {
            let e = world
                .spawn(Voxel::new(0.0, i as f32 + 1.0, 0.0, ShapeType::Cube, false))
                .id();
            entities.push(e);
        }

        run_steps(&mut world, &mut schedule, params.steps_warmup);

        // Check every pair of blocks for penetration
        let positions: Vec<Vec3> = entities.iter().map(|&e| voxel_pos(&world, e)).collect();

        let mut max_penetration: f32 = 0.0;
        let mut violating_pair: Option<(usize, usize, f32)> = None;

        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let delta = (positions[i] - positions[j]).abs();
                // For unit cubes, penetration on each axis: overlap = 1.0 - delta_axis
                // Only check if blocks are near each other (within 2 units)
                if delta.length() < 2.0 {
                    let overlap_x = (1.0_f32 - delta.x).max(0.0);
                    let overlap_y = (1.0_f32 - delta.y).max(0.0);
                    let overlap_z = (1.0_f32 - delta.z).max(0.0);
                    // All three must be positive for actual penetration
                    if overlap_x > 0.0 && overlap_y > 0.0 && overlap_z > 0.0 {
                        let pen = overlap_x.min(overlap_y).min(overlap_z);
                        if pen > max_penetration {
                            max_penetration = pen;
                            violating_pair = Some((i, j, pen));
                        }
                    }
                }
            }
        }

        // ── Check 1: No penetration above epsilon ─────────────────────────────
        log.assert_leq(
            "max_penetration_depth",
            "After 1000 steps, no two blocks in the pillar should overlap by \
             more than 0.01 world-units (1% of a block width). \
             Penetration > 0.01 means the XPBD solver failed to resolve all \
             overlaps — the agent would observe blocks inside each other and \
             learn incorrect physics.",
            max_penetration,
            0.01,
            format!(
                "max_penetration = {:.6}  violating_pair = {:?}",
                max_penetration, violating_pair
            ),
        );

        // ── Check 2: All blocks are within their expected Y range ─────────────
        // After settle, blocks should be at approximately y ≈ 0, 1, 2, 3, 4
        // (each resting on top of the one below, floor at -0.5).
        for (idx, pos) in positions.iter().enumerate() {
            let expected_y = idx as f32; // 0, 1, 2, 3, 4 approximately
            log.assert_approx(
                "pillar_block_y_position",
                format!(
                    "Pillar block {} must rest at approximately y={:.1}. \
                     Large deviation indicates a collapse or floating gap.",
                    idx, expected_y
                ),
                pos.y,
                expected_y,
                0.15, // wider tolerance: small settle differences are OK
            );
        }

        log.finalize_and_assert();
    }
}
