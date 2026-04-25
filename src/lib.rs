// =============================================================================
// src/lib.rs  —  Operation Babel: PyO3 Entry-Point & Engine Orchestrator
// =============================================================================
//
// LEARNING TOPIC: The PyO3 Extension Module Pattern
// --------------------------------------------------
// This file is the "glue layer" between Python and Rust. When Python imports
// `babel_engine`, PyO3 calls `babel_engine()` at the bottom of this file to
// register all exposed classes and functions.
//
// The #[pyclass(unsendable)] attribute tells PyO3 that BabelEngine is NOT
// thread-safe (it owns a winit EventLoop which must live on the main thread).
// Python will enforce that it's only used from the thread that created it.
//
// Architecture overview:
//   Python (brain/training loop)
//       │  env.step_batch(n)   ← zero-copy observation tensor returned
//       │  env.render()        ← optional visual window (decoupled from physics)
//       │  env.pump_os_events()← keeps the OS window alive
//       ▼
//   BabelEngine (this file) — owns ECS World + Schedule + EventLoop
//       │
//       ├─► ECS Schedule (physics pipeline, runs N times per step_batch call)
//       │       spatial_grid → integrate → solve_constraints →
//       │       register_new_bonds → solve_mortar → break_bonds →
//       │       update_velocities → spatial_grid
//       │
//       ├─► RenderContext (WGPU window, only updated when render() is called)
//       │
//       └─► SpatialGrid, PhysicsSettings, SolverBuffers, MortarBonds (Resources)
//
// =============================================================================
//
// OPTIMIZATION: HEADLESS BATCHING (Sprint 2 — Audit Target: >2000 steps/sec)
// ---------------------------------------------------------------------------
// The previous design called step() once per Python loop iteration and slept
// for 1/60 seconds (60 Hz cap). This is correct for rendering but terrible for
// headless AI training where we want the physics to run at full CPU speed.
//
// The new step_batch(n) method runs the full physics schedule N times in a
// tight Rust loop with zero Python round-trips. Each Python call has a fixed
// overhead of ~1-5 μs for PyO3 crossing; by batching 64-256 steps per call,
// we amortize that overhead and let Rust/CPU cache stay hot.
//
// BENCHMARK RESULTS (expected on a modern CPU):
//   step() called from Python × 2000:   ~800 steps/sec  (PyO3 crossing overhead)
//   step_batch(2000) from Python × 1:  ~8000+ steps/sec (near-native Rust speed)
//
// =============================================================================
//
// OPTIMIZATION: ZERO-COPY NUMPY BRIDGE
// ------------------------------------
// The observation vector (block positions, velocities, contact info) is the
// data PyTorch's neural network will read every training step.
//
// SLOW (old approach): build a Vec<f32>, return it to Python, Python copies it
//   into a NumPy array. That's TWO allocations and TWO copies per step.
//
// FAST (new approach): allocate a numpy array ONCE in Python, pass it to Rust
//   as a mutable reference. Rust writes directly into that array's memory.
//   Zero copies. Zero allocations per step. The neural network reads the
//   same physical memory address that Rust just wrote into.
//
// This is the "Zero-Copy" principle from the FFI theory section of the docs:
//   "Rust sends a pointer. Python's numpy reads directly from Rust's memory.
//    Not a single byte is copied." — from the project's Info section.
//
// Implementation uses PyO3's numpy integration:
//   PyReadwriteArray1<f32> — a mutable reference to a numpy array, borrowed
//   from Python and written to in-place by Rust.
// =============================================================================
//
// OPTIMIZATION: HEADLESS FLAG (Sprint 2 — Window Throttling Fix)
// --------------------------------------------------------------
// PROBLEM IDENTIFIED (March 22):
//   The zero-copy benchmark section called BabelEngine::new(), which always
//   calls RenderContext::new() → creates a WGPU window via winit.
//
//   On Windows, an OS window that never receives pump_app_events() is treated
//   as "Not Responding." Windows starts rate-limiting the process's CPU time
//   as a resource-management measure. This bled into benchmark timing, making
//   the zero-copy section show artificially low step rates.
//
//   Confirmed: the "not responding" window visible after benchmark output was
//   exactly this — a WGPU window created by BabelEngine::new() inside the
//   zero-copy test, never pumped, causing OS throttling.
//
// FIX:
//   Add BabelEngine::new_headless() — same as new() but skips RenderContext
//   entirely. renderer = None. No window is created. No OS throttling occurs.
//   The EventLoop is still created (required by winit for pump_app_events)
//   but without a Surface/Window attached it has zero overhead.
//
//   Use new_headless() for: benchmark tests, RL training, any headless workload.
//   Use new() for: interactive debugging with the visual 3D window.
// =============================================================================

use bevy_ecs::prelude::*;
use glam::Vec3;
use numpy::{PyArray1, PyArrayMethods, PyReadwriteArray1};
use pyo3::prelude::*;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    platform::pump_events::EventLoopExtPumpEvents,
    window::WindowId,
};

mod agent;
mod physics;
mod render;
mod world;

// =============================================================================
// LEARNING: Bringing bridge module into scope
// The bridge module contains the zero-copy gym environment wrapper.
// We declare it here so it compiles as part of this crate.
// =============================================================================
mod bridge;
// Bring the `tests` module (src/tests/mod.rs) into scope ONLY when testing.
// This prevents the test harness from bloating the Python production build.
#[cfg(test)]
mod tests;

use agent::{
    BodySegment, HumanoidPosePreset, HumanoidRegistry, HumanoidRig, HumanoidSensors,
    JOINT_OBSERVATION_STRIDE, SENSOR_OBSERVATION_STRIDE, SpawnHumanoidParams,
    apply_joint_torques_system, apply_torque_to_joint, collect_humanoid_observation,
    despawn_all_humanoids, move_humanoid, solve_joint_constraints_system, spawn_humanoid,
    update_humanoid_rigs_system, update_humanoid_sensors_system,
};
use physics::mortar::{
    MortarBonds, break_overloaded_bonds_system, register_new_bonds_system,
    solve_mortar_constraints_system, try_register_bonds,
};
use physics::stress::{StressMap, compute_stress_system};
use physics::xpbd::{PhysicsSettings, SolverBuffers};
use render::wgpu_view::{RenderContext, UiCommand};
use world::spatial_grid::SpatialGrid;
use world::voxel::{MaterialType, ShapeType, Voxel};

// =============================================================================
// SPRINT 4: ScaffoldTracker Resource
// =============================================================================
//
// LEARNING TOPIC: Why Track Scaffold Entities Separately?
// -------------------------------------------------------
// Mass scaffold removal should be O(N_scaffold), not O(N_world).
// A HashSet<Entity> lets us despawn all scaffold blocks directly without
// scanning every voxel in the ECS world.
#[derive(Resource, Default)]
pub struct ScaffoldTracker {
    pub entities: HashSet<Entity>,
}

// =============================================================================
// OPTIMIZATION CONSTANT: Maximum Stack Height Cap
// ------------------------------------------------
// When spawning blocks from the UI or Python, we read the SpatialGrid to find
// the highest existing block in that column and drop the new block 5 units above.
// Without a cap, if the grid somehow contains corrupt data at very high Y values,
// the engine would spawn blocks at y=1000+ which immediately explodes the solver.
//
// Capped at 16: this is the "Tutorial" and "Benchmark" height range from the
// Operation Babel design doc. Higher construction requires the multi-agent
// Bucket Brigade (Phase IV).
// =============================================================================
const MAX_STACK_HEIGHT: i32 = 16;

/// Extra Y clearance used when spawning above an existing column.
///
/// LEARNING: A very large drop height injects unnecessary impact energy,
/// which appears as bounce/jitter on tall pillars (especially steel). Keeping
/// a small but safe clearance preserves placement reliability without harsh
/// collision impulses.
const OCCUPIED_COLUMN_SPAWN_CLEARANCE: f32 = 1.25;

/// Extra Y clearance for empty columns that are adjacent to existing blocks.
/// Keeps side placements close to local ledge height so they attach with
/// lower impact energy instead of hammering short pillars from above.
const ADJACENT_COLUMN_SPAWN_CLEARANCE: f32 = 0.35;

fn material_from_id(material_id: u8) -> MaterialType {
    match material_id {
        0 => MaterialType::Wood,
        1 => MaterialType::Steel,
        2 => MaterialType::Stone,
        3 => MaterialType::Scaffold,
        _ => MaterialType::Wood,
    }
}

fn safe_spawn_height_for_grid(grid: &SpatialGrid, gx: i32, gz: i32) -> f32 {
    match grid.column_max_y(gx, gz) {
        Some(y) => {
            let capped = y.min(MAX_STACK_HEIGHT);
            capped as f32 + OCCUPIED_COLUMN_SPAWN_CLEARANCE
        }
        None => {
            // When the target column is empty but sits next to an existing
            // structure, spawn near the tallest local neighbor so side drops
            // engage upper blocks instead of bonding only at ground level.
            let mut local_neighbor_top: Option<i32> = None;
            for nx in -1..=1 {
                for nz in -1..=1 {
                    if nx == 0 && nz == 0 {
                        continue;
                    }
                    if let Some(ny) = grid.column_max_y(gx + nx, gz + nz) {
                        local_neighbor_top = Some(match local_neighbor_top {
                            Some(current) => current.max(ny),
                            None => ny,
                        });
                    }
                }
            }

            match local_neighbor_top {
                Some(y) => y.min(MAX_STACK_HEIGHT) as f32 + ADJACENT_COLUMN_SPAWN_CLEARANCE,
                None => 5.0,
            }
        }
    }
}

// =============================================================================
// OPTIMIZATION CONSTANT: Observation Vector Layout
// -------------------------------------------------
// Each block contributes OBSERVATION_STRIDE floats to the observation vector.
// Layout per block:
//   [0]:  position.x
//   [1]:  position.y
//   [2]:  position.z
//   [3]:  velocity.x
//   [4]:  velocity.y
//   [5]:  velocity.z
//   [6]:  contact_normal_accum.x  (average contact surface direction)
//   [7]:  contact_normal_accum.y
//   [8]:  contact_normal_accum.z
//   [9]:  contact_count (cast to f32, tells RL agent if block is in contact)
//   [10]: is_static (1.0 = static/floor, 0.0 = dynamic)
//   [11]: shape_id  (0.0=Cube, 1.0=Wedge, 2.0=Sphere)
//
// This flat layout is PyTorch-friendly: the RL network can slice obs[i*12:(i+1)*12]
// to extract features for block i without any fancy indexing.
//
// LEARNING: Why fixed stride? If we used variable-length data, the neural network
// would need a different architecture for each world state size. Fixed-stride
// packing lets a single Linear(n_blocks * 12, hidden_size) layer handle any world.
// =============================================================================
pub const OBSERVATION_STRIDE: usize = 12;

/// Total humanoid observation size exposed to Python:
/// 9 body segments × 15 floats each + 18 aggregate sensor floats.
pub const HUMANOID_OBSERVATION_SIZE: usize =
    BodySegment::COUNT * JOINT_OBSERVATION_STRIDE + SENSOR_OBSERVATION_STRIDE;

fn latest_humanoid_agent_id(ecs_world: &World) -> Option<u32> {
    let registry = ecs_world.get_resource::<HumanoidRegistry>()?;
    registry.torso_entity_by_agent_id.keys().max().copied()
}

fn default_swing_axis_for_segment(segment: BodySegment) -> Vec3 {
    match segment {
        BodySegment::LeftThigh | BodySegment::RightThigh => Vec3::Z,
        BodySegment::LeftUpperArm => -Vec3::X,
        BodySegment::RightUpperArm => Vec3::X,
        _ => Vec3::Y,
    }
}

fn sync_renderer_agent_panel(renderer: &mut RenderContext, ecs_world: &World) {
    let Some(agent_id) = latest_humanoid_agent_id(ecs_world) else {
        renderer.agent_panel.display_agent_count = 0;
        renderer.agent_panel.display_balance_error = 0.0;
        renderer.agent_panel.display_com_height = 0.0;
        renderer.agent_panel.display_left_foot_contact = false;
        renderer.agent_panel.display_right_foot_contact = false;
        renderer.agent_panel.joint_angles_degrees = [0.0; BodySegment::COUNT];
        return;
    };

    let registry = ecs_world.get_resource::<HumanoidRegistry>().unwrap();
    renderer.agent_panel.display_agent_count = registry.agent_count();

    let Some(torso_entity) = registry.torso_entity(agent_id) else {
        return;
    };

    if let Some(rig) = ecs_world.get::<HumanoidRig>(torso_entity) {
        renderer.agent_panel.display_balance_error = rig.balance_error;
        renderer.agent_panel.display_com_height = rig.center_of_mass.y;

        for segment in [
            BodySegment::LeftThigh,
            BodySegment::LeftShin,
            BodySegment::RightThigh,
            BodySegment::RightShin,
            BodySegment::LeftUpperArm,
            BodySegment::LeftForearm,
            BodySegment::RightUpperArm,
            BodySegment::RightForearm,
        ] {
            if let Some(segment_entity) = rig.segment_entity(segment) {
                if let Some(joint_constraint) =
                    ecs_world.get::<agent::JointConstraint>(segment_entity)
                {
                    renderer.agent_panel.joint_angles_degrees[segment.index()] =
                        joint_constraint.target_angle_radians.to_degrees();
                }
            }
        }
    }

    if let Some(sensors) = ecs_world.get::<HumanoidSensors>(torso_entity) {
        renderer.agent_panel.display_left_foot_contact = sensors.left_foot_contact;
        renderer.agent_panel.display_right_foot_contact = sensors.right_foot_contact;
    }
}

// =============================================================================
// RenderEventPump: thin ApplicationHandler wrapper for winit's pump_events API
// =============================================================================
//
// LEARNING TOPIC: Winit's Event Model
// ------------------------------------
// Winit uses a callback-based event loop. The "ApplicationHandler" trait is the
// interface: Winit calls window_event() and device_event() when OS events fire.
//
// We use pump_events() with a timeout of Duration::ZERO, which processes ALL
// pending events without blocking. This is the non-blocking "polling" mode,
// contrasted with run() which blocks until the application exits.
//
// The engine calls pump_os_events() once per render frame to drain the OS queue
// (mouse clicks, key presses, window close) without stalling the Python loop.
struct RenderEventPump<'a> {
    renderer: &'a mut Option<RenderContext>,
}

impl ApplicationHandler for RenderEventPump<'_> {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(renderer) = self.renderer.as_mut() {
            if renderer.window.id() == window_id {
                if matches!(event, WindowEvent::CloseRequested) {
                    // ==========================================================
                    // LEARNING: Why std::process::exit() instead of just
                    // setting renderer = None?
                    //
                    // When the user clicks X, the window is gone but Python's
                    // loop is still running (it doesn't know the window closed).
                    // If we just drop the renderer, the next render() call would
                    // panic trying to access a destroyed surface.
                    //
                    // std::process::exit(0) is the nuclear option: it immediately
                    // terminates the OS process. This is acceptable here because
                    // Operation Babel is a training tool, not a production server.
                    // A production implementation would signal the Python loop via
                    // a shared AtomicBool and let Python shut down gracefully.
                    // ==========================================================
                    *self.renderer = None;
                    event_loop.exit();
                    std::process::exit(0);
                }
                renderer.handle_window_event(&event);
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // =======================================================================
        // LEARNING: DeviceEvent vs WindowEvent for Mouse
        // -----------------------------------------------
        // WindowEvent::CursorMoved fires relative to the window's client area.
        //   → Gives absolute pixel position. Good for UI raycasting.
        //   → STOPS updating when cursor leaves the window.
        //
        // DeviceEvent::MouseMotion gives RAW DELTA from the hardware sensor.
        //   → Continues updating even when cursor is at the screen edge.
        //   → Not affected by OS acceleration curves (pure physical movement).
        //   → This is what you want for camera orbit: smooth 360° rotation
        //     without the cursor "hitting the wall" at the screen border.
        // =======================================================================
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.handle_device_event(&event);
        }
    }
}

// =============================================================================
// BabelEngine: the main Python-facing struct
// =============================================================================
#[pyclass(unsendable)]
pub struct BabelEngine {
    ecs_world: World,
    schedule: Schedule,
    event_loop: EventLoop<()>,
    renderer: Option<RenderContext>,
    last_frame: Instant,

    // =========================================================================
    // OPTIMIZATION: Step counter for headless performance measurement
    // ---------------------------------------------------------------
    // Tracks total physics steps executed. Used by get_benchmark_stats() to
    // report real steps/second without the overhead of calling Instant::now()
    // inside the hot step loop.
    //
    // LEARNING: Calling Instant::now() has a cost (~10-50ns per call on Linux
    // via clock_gettime syscall). Inside a loop running 100,000+ steps, that
    // adds up to 1-5ms of wasted time per second — 0.5% overhead that we avoid
    // by only sampling time at batch boundaries.
    // =========================================================================
    step_count: u64,
    benchmark_start: Instant,
}

// =============================================================================
// LEARNING: Shared constructor logic extracted into a free function
// =============================================================================
//
// Both new() and new_headless() need to build the ECS world and schedule.
// We extract that shared logic here so it doesn't need to be duplicated.
// This is the "Extract Method" refactor pattern — keep constructors DRY.
fn build_ecs_and_schedule() -> (World, Schedule) {
    let mut ecs_world = World::new();

    // =======================================================================
    // LEARNING: Bevy ECS Resources vs Components
    // -------------------------------------------
    // Resources are global singletons (one per World, no Entity owner).
    // We pre-insert them here so systems can always assume they exist.
    //
    // If a system tries to access a Resource that wasn't inserted, Bevy
    // panics immediately with a clear error — much better than a null pointer.
    //
    // SolverBuffers uses ResMut (mutable resource) — it's the reusable
    // HashMap/HashSet pool that prevents per-frame heap allocations in the
    // XPBD constraint solver.
    // =======================================================================
    ecs_world.insert_resource(SpatialGrid::default());
    ecs_world.insert_resource(PhysicsSettings::default());
    ecs_world.insert_resource(SolverBuffers::default());
    ecs_world.insert_resource(MortarBonds::default()); // SPRINT 3: bond registry
    ecs_world.insert_resource(StressMap::default());
    ecs_world.insert_resource(ScaffoldTracker::default());
    ecs_world.insert_resource(HumanoidRegistry::default());

    // =======================================================================
    // LEARNING: Bevy's System Scheduling and .chain()
    // ------------------------------------------------
    // .chain() enforces strict sequential ordering within the tuple.
    // Without it, Bevy might parallelize systems that have data dependencies.
    //
    // SPRINT 3 SCHEDULE ORDER (updated):
    //   1. update_spatial_grid          ← pre-step grid from committed positions
    //   2. integrate                    ← apply gravity, compute predicted_position
    //   3. solve_constraints            ← push blocks out of overlaps (XPBD)
    //   4. register_new_bonds           ← create missing bonds for new face-neighbors
    //   5. solve_mortar_constraints     ← pull bonded blocks back together ← SPRINT 3
    //   6. break_overloaded_bonds       ← snap bonds that exceed breaking force ← SPRINT 3
    //   7. update_velocities            ← commit positions, derive velocity, sleep
    //   8. update_spatial_grid          ← post-step grid for spawning/UI/Python queries
    //
    // WHY THIS ORDER:
    //   Mortar runs AFTER the collision solver so blocks are already out of each
    //   other's volume before we pull them together. This prevents the mortar
    //   correction from pushing blocks back INTO a collision.
    //
    //   Break runs AFTER mortar so we can compute tension from the just-solved
    //   stretch distance. Running break before mortar would snap bonds at their
    //   pre-correction tension, which is artificially high (not yet pulled back).
    // =======================================================================
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            world::spatial_grid::update_spatial_grid_system,
            physics::xpbd::integrate_system,
            physics::xpbd::solve_constraints_system,
            solve_joint_constraints_system,
            apply_joint_torques_system,
            register_new_bonds_system,
            solve_mortar_constraints_system, // SPRINT 3
            break_overloaded_bonds_system,   // SPRINT 3
            physics::xpbd::update_velocities_system,
            world::spatial_grid::update_spatial_grid_system,
            compute_stress_system,
            update_humanoid_rigs_system,
            update_humanoid_sensors_system,
        )
            .chain(),
    );

    (ecs_world, schedule)
}

// =============================================================================
// INTERNAL HELPER: spawn a voxel and immediately register mortar bonds
// =============================================================================
//
// LEARNING: We extract spawn + bond-registration into one helper so both
// the UI path (render()) and the Python API (spawn_block, spawn_block_with_material)
// go through identical logic. No path forgets to register bonds.
//
// The spatial grid must be UP TO DATE before calling this, otherwise
// try_register_bonds() won't find neighbors. The schedule now rebuilds the
// grid both before and after each physics step, so UI and Python spawns both
// see current committed positions.
//
// SPRINT 3 NOTE: The benchmark previously bypassed this function and called
// ecs_world.spawn(Voxel::new_with_material(...)) directly, which meant zero
// bonds were ever registered and the mortar systems iterated over empty data.
// All spawn paths now go through this function for honest benchmark numbers.
fn spawn_voxel_and_register(
    ecs_world: &mut World,
    x: f32,
    y: f32,
    z: f32,
    shape: ShapeType,
    material: MaterialType,
    is_static: bool,
) -> Entity {
    let is_scaffold = material.is_scaffold();
    let voxel = Voxel::new_with_material(x, y, z, shape, material, is_static);
    let entity = ecs_world.spawn(voxel).id();

    // Immediately check 6 face-neighbors and register mortar bonds.
    // We need to read: the voxel we just spawned, the spatial grid, all voxels,
    // and write to: MortarBonds. Bevy's borrow checker requires careful ordering.
    //
    // Strategy: take the resources out, do the work, put them back.
    // This is the standard "non-conflicting mutable access" pattern in Bevy.
    let mut grid = ecs_world.remove_resource::<SpatialGrid>().unwrap();
    let mut bonds = ecs_world.remove_resource::<MortarBonds>().unwrap();

    // Get the spawned voxel's data for bond registration.
    // SAFETY: We just spawned this entity, it definitely exists.
    let new_voxel_ref = ecs_world.get::<Voxel>(entity).unwrap();

    // try_register_bonds checks the neighbor's adhesion too, so we pass it
    // even if the new block itself has 0.0 adhesion (e.g., Stone). The function
    // handles that case by checking both sides before creating a bond.
    try_register_bonds(entity, new_voxel_ref, &grid, ecs_world, &mut bonds);

    // Keep grid immediately consistent so back-to-back spawns in the same frame
    // (or benchmark spawn loops) can discover freshly spawned neighbors.
    grid.insert(new_voxel_ref.position, entity, new_voxel_ref.shape);

    ecs_world.insert_resource(grid);
    ecs_world.insert_resource(bonds);

    if is_scaffold {
        let mut tracker = ecs_world.remove_resource::<ScaffoldTracker>().unwrap();
        tracker.entities.insert(entity);
        ecs_world.insert_resource(tracker);
    }

    entity
}

fn spawn_sphere_and_register(
    ecs_world: &mut World,
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
    material: MaterialType,
    is_static: bool,
) -> Entity {
    let voxel = Voxel::new_sphere_with_material(x, y, z, radius, material, is_static);
    let entity = ecs_world.spawn(voxel).id();

    let mut grid = ecs_world.remove_resource::<SpatialGrid>().unwrap();
    if let Some(new_voxel_ref) = ecs_world.get::<Voxel>(entity) {
        grid.insert(new_voxel_ref.position, entity, new_voxel_ref.shape);
    }
    ecs_world.insert_resource(grid);

    entity
}

#[pymethods]
impl BabelEngine {
    // =========================================================================
    // new() — Full engine with WGPU window (interactive mode)
    // =========================================================================
    //
    // LEARNING: Use this constructor when you want the visual 3D window —
    // for interactive block spawning, debugging physics, watching agents train.
    //
    // Do NOT use this for benchmarks or headless RL training. The WGPU window
    // it creates will cause OS throttling if pump_os_events() isn't called
    // regularly. Use new_headless() instead for any non-interactive use.
    #[new]
    pub fn new() -> Self {
        let (ecs_world, schedule) = build_ecs_and_schedule();
        let event_loop = EventLoop::new().unwrap();

        // RenderContext::new() creates: winit Window + WGPU device + surface +
        // shader compilation + egui init. Takes ~500ms-2s on first call.
        let renderer = RenderContext::new(&event_loop);

        Self {
            ecs_world,
            schedule,
            event_loop,
            renderer,
            last_frame: Instant::now(),
            step_count: 0,
            benchmark_start: Instant::now(),
        }
    }

    // =========================================================================
    // new_headless() — Physics-only engine, NO window (benchmark / RL training)
    // =========================================================================
    //
    // LEARNING TOPIC: Why the window causes throttling
    // -------------------------------------------------
    // On Windows (and macOS/Linux to a lesser extent), the OS tracks whether
    // GUI windows are "responsive" by monitoring whether they process messages
    // from the message queue. If a window doesn't process messages for >5 seconds,
    // Windows marks it "Not Responding" and can start throttling its CPU time.
    //
    // In our benchmark:
    //   BabelEngine::new()   → creates a WGPU window via winit
    //   step_batch(N)        → runs physics in a tight loop, never pumps events
    //   result printed       → window is still there, OS has been throttling us
    //
    // new_headless() sets renderer = None, which means:
    //   - No winit Window is created
    //   - No WGPU Surface/Device/Queue is allocated
    //   - No shader compilation
    //   - No OS message queue for the throttler to monitor
    //   - pump_os_events() becomes a no-op (renderer is None)
    //   - render() becomes a no-op (renderer is None)
    #[staticmethod]
    pub fn new_headless() -> Self {
        let (ecs_world, schedule) = build_ecs_and_schedule();
        let event_loop = EventLoop::new().unwrap();

        // renderer = None — NO window created, NO WGPU device, NO OS throttling.
        Self {
            ecs_world,
            schedule,
            event_loop,
            renderer: None,
            last_frame: Instant::now(),
            step_count: 0,
            benchmark_start: Instant::now(),
        }
    }

    // =========================================================================
    // step() — Single physics tick (kept for compatibility and debugging)
    // =========================================================================
    //
    // LEARNING: This is the original API. For rendering-synchronized training
    // (where you want to see every physics frame), call step() once per render().
    // For headless AI training (maximum speed), use step_batch() instead.
    //
    // Returns the current block count so Python can quickly check world state.
    pub fn step(&mut self) -> PyResult<usize> {
        self.schedule.run(&mut self.ecs_world);
        self.step_count += 1;

        let mut query = self.ecs_world.query::<&Voxel>();
        let count = query.iter(&self.ecs_world).count();
        Ok(count)
    }

    // =========================================================================
    // step_batch() — Run N physics ticks in a tight Rust loop (HEADLESS SPEED)
    // =========================================================================
    //
    // LEARNING TOPIC: Why Batching Eliminates PyO3 Overhead
    // -------------------------------------------------------
    // Every Python→Rust call through PyO3 has a fixed overhead cost:
    //   - Python GIL release/acquire: ~500ns
    //   - PyO3 argument marshalling: ~200ns
    //   - Rust function call + return: ~100ns
    //   Total: ~1 μs per call
    //
    // With step_batch(256) from Python, Rust runs 256 steps with ZERO Python
    // interruptions:
    //   Rust loop = 256 × 100μs = 25.6ms of pure Rust
    //   PyO3 overhead = 1μs (paid ONCE per batch, not per step)
    //   Max speed = 256 / (25.6ms + 0.001ms) ≈ 10,000+ steps/sec
    //
    // Returns (steps_completed, block_count) so Python can monitor training.
    pub fn step_batch(&mut self, n: usize) -> PyResult<(usize, usize)> {
        for _ in 0..n {
            self.schedule.run(&mut self.ecs_world);
        }
        self.step_count += n as u64;

        let mut query = self.ecs_world.query::<&Voxel>();
        let count = query.iter(&self.ecs_world).count();
        Ok((n, count))
    }

    // =========================================================================
    // get_observation_into() — ZERO-COPY observation write into Python array
    // =========================================================================
    //
    // LEARNING TOPIC: Zero-Copy FFI — The "Pointer Not Copy" Philosophy
    // ------------------------------------------------------------------
    // Traditional Python↔Rust data transfer:
    //   1. Rust allocates Vec<f32>           (heap allocation #1)
    //   2. PyO3 converts Vec → Python list   (allocation #2 + copy #1)
    //   3. Python converts list → np.array   (allocation #3 + copy #2)
    //
    // ZERO-COPY approach:
    //   Python allocates np.zeros(N, dtype=np.float32) ONCE before training.
    //   Each step: Rust receives a mutable reference to that array's raw memory.
    //   Rust writes directly into it. PyTorch can use torch.from_numpy() to get
    //   a tensor that SHARES the same physical memory.
    //
    //   Total copies: ZERO. Total allocations per step: ZERO.
    pub fn get_observation_into<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<usize> {
        let mut rw: PyReadwriteArray1<f32> = obs.readwrite();
        let slice: &mut [f32] = rw.as_slice_mut()?;

        let mut query = self.ecs_world.query::<&Voxel>();
        let mut write_idx = 0usize;
        let mut block_count = 0usize;

        for voxel in query.iter(&self.ecs_world) {
            if voxel.inv_mass == 0.0 {
                continue;
            }

            let end = write_idx + OBSERVATION_STRIDE;
            if end > slice.len() {
                break;
            }

            // =================================================================
            // DIRECT MEMORY WRITE — this is the zero-copy moment.
            // We're writing into the numpy array's raw f32 buffer.
            // No Python objects are created. No heap allocations occur.
            // The data appears in `obs` immediately after this function returns,
            // ready for torch.from_numpy() to create a zero-copy tensor view.
            // =================================================================
            slice[write_idx] = voxel.position.x;
            slice[write_idx + 1] = voxel.position.y;
            slice[write_idx + 2] = voxel.position.z;
            slice[write_idx + 3] = voxel.velocity.x;
            slice[write_idx + 4] = voxel.velocity.y;
            slice[write_idx + 5] = voxel.velocity.z;
            slice[write_idx + 6] = voxel.contact_normal_accum.x;
            slice[write_idx + 7] = voxel.contact_normal_accum.y;
            slice[write_idx + 8] = voxel.contact_normal_accum.z;
            slice[write_idx + 9] = voxel.contact_count as f32;
            slice[write_idx + 10] = 0.0; // is_static: dynamic block = 0.0
            slice[write_idx + 11] = match voxel.shape {
                ShapeType::Cube => 0.0,
                ShapeType::Wedge => 1.0,
                ShapeType::Sphere => 2.0,
            };

            write_idx += OBSERVATION_STRIDE;
            block_count += 1;
        }

        let _ = py;
        Ok(block_count)
    }

    // =========================================================================
    // observation_stride() — tells Python the per-block stride constant
    // =========================================================================
    //
    // LEARNING: Python needs to know how many floats per block to correctly
    // slice the observation array. Exposing this as a method keeps Python and
    // Rust in sync even if OBSERVATION_STRIDE changes in future sprints.
    //
    // Usage: stride = engine.observation_stride()  # → 12
    pub fn observation_stride(&self) -> usize {
        OBSERVATION_STRIDE
    }

    // =========================================================================
    // Sprint 5: Humanoid observation + control API
    // =========================================================================
    pub fn humanoid_observation_size(&self) -> usize {
        HUMANOID_OBSERVATION_SIZE
    }

    pub fn humanoid_joint_count(&self) -> usize {
        8
    }

    pub fn latest_humanoid_agent_id(&self) -> Option<u32> {
        latest_humanoid_agent_id(&self.ecs_world)
    }

    pub fn spawn_humanoid_worker(&mut self, world_x: f32, world_z: f32) -> PyResult<u32> {
        let agent_id = spawn_humanoid(
            &mut self.ecs_world,
            SpawnHumanoidParams::new(world_x, world_z),
        );
        Ok(agent_id)
    }

    pub fn despawn_all_humanoid_workers(&mut self) -> PyResult<usize> {
        Ok(despawn_all_humanoids(&mut self.ecs_world))
    }

    pub fn set_humanoid_joint_target(
        &mut self,
        agent_id: u32,
        joint_index: usize,
        target_angle_radians: f32,
    ) -> PyResult<bool> {
        let joint_segment = match joint_index {
            0 => BodySegment::LeftThigh,
            1 => BodySegment::LeftShin,
            2 => BodySegment::RightThigh,
            3 => BodySegment::RightShin,
            4 => BodySegment::LeftUpperArm,
            5 => BodySegment::LeftForearm,
            6 => BodySegment::RightUpperArm,
            7 => BodySegment::RightForearm,
            _ => return Ok(false),
        };

        Ok(apply_torque_to_joint(
            &mut self.ecs_world,
            agent_id,
            joint_segment,
            target_angle_radians,
            default_swing_axis_for_segment(joint_segment),
        ))
    }

    pub fn set_humanoid_joint_targets(
        &mut self,
        agent_id: u32,
        joint_targets_radians: Vec<f32>,
    ) -> PyResult<usize> {
        let segment_order = [
            BodySegment::LeftThigh,
            BodySegment::LeftShin,
            BodySegment::RightThigh,
            BodySegment::RightShin,
            BodySegment::LeftUpperArm,
            BodySegment::LeftForearm,
            BodySegment::RightUpperArm,
            BodySegment::RightForearm,
        ];

        let mut applied_joint_count = 0usize;
        for (joint_index, target_angle_radians) in joint_targets_radians.iter().enumerate() {
            let Some(&segment) = segment_order.get(joint_index) else {
                break;
            };
            if apply_torque_to_joint(
                &mut self.ecs_world,
                agent_id,
                segment,
                *target_angle_radians,
                default_swing_axis_for_segment(segment),
            ) {
                applied_joint_count += 1;
            }
        }

        Ok(applied_joint_count)
    }

    pub fn apply_humanoid_pose_preset(
        &mut self,
        agent_id: u32,
        preset_name: String,
    ) -> PyResult<usize> {
        let preset = match preset_name.to_ascii_lowercase().as_str() {
            "tstance" | "t_stance" | "tee_stance" | "t-stance" => HumanoidPosePreset::TeeStance,
            "standing" => HumanoidPosePreset::Standing,
            "walk" | "walkmidstride" | "walk_stride" | "walk-stride" => {
                HumanoidPosePreset::WalkMidStride
            }
            "squat" | "deepsquat" | "deep_squat" | "deep-squat" => HumanoidPosePreset::Squat,
            "armraise" | "arm_raise" | "arm-raise" => HumanoidPosePreset::ArmRaise,
            _ => HumanoidPosePreset::Standing,
        };

        let mut applied_joint_count = 0usize;
        for &(segment, target_angle_radians) in preset.joint_angles() {
            if apply_torque_to_joint(
                &mut self.ecs_world,
                agent_id,
                segment,
                target_angle_radians,
                default_swing_axis_for_segment(segment),
            ) {
                applied_joint_count += 1;
            }
        }

        Ok(applied_joint_count)
    }

    pub fn move_humanoid_worker(
        &mut self,
        agent_id: u32,
        direction_x: f32,
        direction_z: f32,
    ) -> PyResult<bool> {
        Ok(move_humanoid(
            &mut self.ecs_world,
            agent_id,
            Vec3::new(direction_x, 0.0, direction_z),
        ))
    }

    pub fn get_humanoid_observation_into<'py>(
        &mut self,
        py: Python<'py>,
        agent_id: u32,
        obs: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<usize> {
        let mut readwrite_observation: PyReadwriteArray1<f32> = obs.readwrite();
        let output_slice: &mut [f32] = readwrite_observation.as_slice_mut()?;

        if output_slice.len() < HUMANOID_OBSERVATION_SIZE {
            return Ok(0);
        }

        let segments_written = collect_humanoid_observation(
            &self.ecs_world,
            agent_id,
            &mut output_slice[..BodySegment::COUNT * JOINT_OBSERVATION_STRIDE],
        );
        if segments_written == 0 {
            return Ok(0);
        }

        let registry = self.ecs_world.get_resource::<HumanoidRegistry>().unwrap();
        let Some(torso_entity) = registry.torso_entity(agent_id) else {
            return Ok(0);
        };
        let Some(rig) = self.ecs_world.get::<HumanoidRig>(torso_entity) else {
            return Ok(0);
        };
        let Some(sensors) = self.ecs_world.get::<HumanoidSensors>(torso_entity) else {
            return Ok(0);
        };
        let torso_rotation = rig
            .segment_entity(BodySegment::Torso)
            .and_then(|entity| self.ecs_world.get::<Voxel>(entity))
            .map(|torso_voxel| torso_voxel.rotation)
            .unwrap_or_default();

        sensors.pack_into_buffer(
            rig.center_of_mass,
            rig.center_of_mass_velocity,
            rig.balance_error,
            torso_rotation,
            &mut output_slice[BodySegment::COUNT * JOINT_OBSERVATION_STRIDE
                ..BodySegment::COUNT * JOINT_OBSERVATION_STRIDE + SENSOR_OBSERVATION_STRIDE],
        );

        let _ = py;
        Ok(HUMANOID_OBSERVATION_SIZE)
    }

    // =========================================================================
    // SPRINT 4: get_stress_data() — Stress values for Python reward shaping
    // =========================================================================
    pub fn get_stress_data(&self) -> PyResult<(f32, f32, usize)> {
        let stress_map = self.ecs_world.get_resource::<StressMap>().unwrap();

        if stress_map.data.is_empty() {
            return Ok((0.0, 0.0, 0));
        }

        let mut max_stress = 0.0f32;
        let mut sum_stress = 0.0f32;
        let mut n_critical = 0usize;

        for info in stress_map.data.values() {
            let s = info.stress_normalized;
            if s > max_stress {
                max_stress = s;
            }
            sum_stress += s;
            if s > 0.8 {
                n_critical += 1;
            }
        }

        let avg_stress = sum_stress / stress_map.data.len() as f32;
        Ok((max_stress, avg_stress, n_critical))
    }

    // =========================================================================
    // get_benchmark_stats() — Real steps/second measurement
    // =========================================================================
    //
    // LEARNING: Measure performance FROM INSIDE RUST to avoid PyO3 timing bias.
    // If Python measured time around step_batch(), it would include:
    //   - Python interpreter overhead
    //   - GIL acquire/release
    //   - Any Python code between calls
    //
    // Returns: (total_steps, elapsed_seconds, steps_per_second)
    pub fn get_benchmark_stats(&self) -> PyResult<(u64, f64, f64)> {
        let elapsed = self.benchmark_start.elapsed().as_secs_f64();
        let steps_per_sec = if elapsed > 0.0 {
            self.step_count as f64 / elapsed
        } else {
            0.0
        };
        Ok((self.step_count, elapsed, steps_per_sec))
    }

    // =========================================================================
    // reset_benchmark() — restart the timing window
    // =========================================================================
    //
    // LEARNING: Call this at the start of a training episode to get clean
    // per-episode performance numbers, without contamination from engine
    // initialization time (which includes WGPU setup, window creation, etc.).
    pub fn reset_benchmark(&mut self) {
        self.step_count = 0;
        self.benchmark_start = Instant::now();
    }

    // =========================================================================
    // SPRINT 3: get_safe_spawn_height() — column-aware spawn height for Python
    // =========================================================================
    //
    // SPRINT 3 BUG FIX — Hardcoded Spawn Height:
    // -------------------------------------------
    // babel_gym_env.py previously had:
    //   spawn_y = 10.0  # TODO: query engine...
    //
    // This caused blocks to spawn INSIDE the tower once it exceeded Y=10.
    // The XPBD solver would detect a massive penetration depth and launch blocks
    // at extreme velocities, corrupting the RL gradient and terminating episodes.
    //
    // FIX: Expose the column_max_y lookup from the spatial grid to Python.
    // Python calls this before each spawn to get a safe drop height.
    //
    // WHY SPATIAL GRID, NOT QUERY:
    //   The spatial grid already maps (x,z) → max occupied y in O(N) via
    //   column_max_y(). We don't need to query all voxels and filter by column.
    //   This is the same lookup the UI's render() path has always used —
    //   now exposed to Python as a proper API.
    //
    // Returns: safe Y coordinate to spawn at (column top + 5 units clearance)
    // The +5 gives enough fall height to let the block settle into position
    // without spawning too far above (which wastes physics time falling).
    pub fn get_safe_spawn_height(&self, world_x: f32, world_z: f32) -> PyResult<f32> {
        let gx = world_x.round() as i32;
        let gz = world_z.round() as i32;

        let grid = self.ecs_world.get_resource::<SpatialGrid>().unwrap();
        let height = safe_spawn_height_for_grid(grid, gx, gz);
        Ok(height)
    }

    // =========================================================================
    // spawn_block() — Original API, defaults to Wood (backward compatible)
    // =========================================================================
    pub fn spawn_block(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        shape_id: u8,
        is_static: bool,
    ) -> PyResult<()> {
        let shape = match shape_id {
            0 => ShapeType::Cube,
            1 => ShapeType::Wedge,
            2 => ShapeType::Sphere,
            _ => ShapeType::Cube,
        };
        // Default to Wood — same inv_mass as old behavior (inv_mass = 1.0)
        spawn_voxel_and_register(
            &mut self.ecs_world,
            x,
            y,
            z,
            shape,
            MaterialType::Wood,
            is_static,
        );
        Ok(())
    }

    // =========================================================================
    // SPRINT 3: spawn_block_with_material() — Material-aware spawn
    // =========================================================================
    //
    // material_id:
    //   0 = Wood  (light, medium adhesion)
    //   1 = Steel (heavy, high adhesion)
    //   2 = Stone (medium weight, no adhesion, high friction)
    pub fn spawn_block_with_material(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        shape_id: u8,
        material_id: u8,
        is_static: bool,
    ) -> PyResult<()> {
        let shape = match shape_id {
            0 => ShapeType::Cube,
            1 => ShapeType::Wedge,
            2 => ShapeType::Sphere,
            _ => ShapeType::Cube,
        };
        let material = material_from_id(material_id);
        spawn_voxel_and_register(&mut self.ecs_world, x, y, z, shape, material, is_static);
        Ok(())
    }

    // =========================================================================
    // SPRINT 4: spawn_scaffold() — Convenience API for scaffold placement
    // =========================================================================
    pub fn spawn_scaffold(&mut self, x: f32, y: f32, z: f32) -> PyResult<usize> {
        spawn_voxel_and_register(
            &mut self.ecs_world,
            x,
            y,
            z,
            ShapeType::Cube,
            MaterialType::Scaffold,
            false,
        );
        let tracker = self.ecs_world.get_resource::<ScaffoldTracker>().unwrap();
        Ok(tracker.entities.len())
    }

    // =========================================================================
    // SPRINT 4: despawn_scaffolding() — Remove ALL scaffold blocks at once
    // =========================================================================
    pub fn despawn_scaffolding(&mut self) -> PyResult<usize> {
        let scaffold_entities: Vec<Entity> = {
            let tracker = self.ecs_world.get_resource::<ScaffoldTracker>().unwrap();
            tracker.entities.iter().copied().collect()
        };

        let count = scaffold_entities.len();

        {
            let mut bonds = self.ecs_world.get_resource_mut::<MortarBonds>().unwrap();
            for &entity in &scaffold_entities {
                bonds.remove_bonds_for(entity);
            }
        }

        for &entity in &scaffold_entities {
            self.ecs_world.despawn(entity);
        }

        {
            let mut tracker = self
                .ecs_world
                .get_resource_mut::<ScaffoldTracker>()
                .unwrap();
            tracker.entities.clear();
        }

        {
            let mut stress_map = self.ecs_world.get_resource_mut::<StressMap>().unwrap();
            for &entity in &scaffold_entities {
                stress_map.data.remove(&entity);
            }
        }

        Ok(count)
    }

    // =========================================================================
    // SPRINT 3+: spawn_sphere_with_material() — Material-aware sphere spawn
    // =========================================================================
    //
    // Spheres inherit material-dependent mass/friction/restitution, but the
    // mortar system only creates bonds for face-neighbor voxel blocks.
    pub fn spawn_sphere_with_material(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        material_id: u8,
        is_static: bool,
    ) -> PyResult<()> {
        let material = material_from_id(material_id);
        spawn_sphere_and_register(&mut self.ecs_world, x, y, z, radius, material, is_static);
        Ok(())
    }

    // =========================================================================
    // SPRINT 3: bond_count() — How many mortar bonds currently exist
    // =========================================================================
    //
    // Useful for:
    //   - Debugging: watch bonds form and break in the terminal
    //   - Python reward shaping: reward agent for building connected structures
    //   - Stress testing: confirm bonds don't accumulate endlessly
    pub fn bond_count(&self) -> PyResult<usize> {
        let bonds = self.ecs_world.get_resource::<MortarBonds>().unwrap();
        Ok(bonds.bonds.len())
    }

    // =========================================================================
    // SPRINT 4: scaffold_count() — How many scaffold blocks currently exist
    // =========================================================================
    pub fn scaffold_count(&self) -> PyResult<usize> {
        let tracker = self.ecs_world.get_resource::<ScaffoldTracker>().unwrap();
        Ok(tracker.entities.len())
    }

    // =========================================================================
    // clear_dynamic_blocks() — SPRINT 3: also clears all mortar bonds
    // =========================================================================
    pub fn clear_dynamic_blocks(&mut self) -> PyResult<usize> {
        let humanoids_cleared = despawn_all_humanoids(&mut self.ecs_world);
        let mut query = self.ecs_world.query::<(Entity, &Voxel)>();
        let to_despawn: Vec<Entity> = query
            .iter(&self.ecs_world)
            .filter(|(_, v)| v.inv_mass > 0.0)
            .map(|(e, _)| e)
            .collect();

        let count = to_despawn.len() + humanoids_cleared;

        // SPRINT 3: Remove bonds before despawning entities
        // (prevents dangling entity references in MortarBonds)
        {
            let mut bonds = self.ecs_world.get_resource_mut::<MortarBonds>().unwrap();
            for &entity in &to_despawn {
                bonds.remove_bonds_for(entity);
            }
        }

        for entity in to_despawn {
            // LEARNING: despawn() removes the entity AND all its components.
            // Bevy's archetype system immediately reclaims the memory slot,
            // making it available for future spawn() calls without reallocation.
            self.ecs_world.despawn(entity);
        }

        {
            let mut tracker = self
                .ecs_world
                .get_resource_mut::<ScaffoldTracker>()
                .unwrap();
            tracker.entities.clear();
        }

        {
            let mut stress_map = self.ecs_world.get_resource_mut::<StressMap>().unwrap();
            stress_map.data.clear();
        }

        Ok(count)
    }

    // =========================================================================
    // block_count() — fast world size query without full observation build
    // =========================================================================
    pub fn block_count(&mut self) -> PyResult<(usize, usize)> {
        let mut query = self.ecs_world.query::<&Voxel>();
        let mut dynamic = 0usize;
        let mut static_count = 0usize;
        for v in query.iter(&self.ecs_world) {
            if v.inv_mass > 0.0 {
                dynamic += 1;
            } else {
                static_count += 1;
            }
        }
        Ok((dynamic, static_count))
    }

    // =========================================================================
    // render() — visual frame update (DECOUPLED from physics speed)
    // =========================================================================
    //
    // LEARNING TOPIC: Decoupled Rendering Architecture
    // -------------------------------------------------
    // In traditional game engines, rendering and physics are tightly coupled:
    //   game loop: update physics → render → sleep(16ms) → repeat
    //   → Physics speed is LOCKED to 60 FPS
    //
    // Operation Babel decouples them:
    //   Training mode: step_batch(N) as fast as possible, never render()
    //   Debug mode:    step_batch(1) then render() at 60 Hz
    //
    // In headless mode (renderer = None), this is a complete no-op.
    // No GPU work, no window update, no blocking — purely returns immediately.
    pub fn render(&mut self) {
        if self.renderer.is_none() {
            return;
        }

        let now = Instant::now();
        let dt = self.last_frame.elapsed().as_secs_f32().min(0.1);
        self.last_frame = now;
        let mut commands_to_execute: (Vec<UiCommand>, Vec<agent::UiAgentCommand>) =
            (Vec::new(), Vec::new());
        if let Some(renderer) = &mut self.renderer {
            renderer.camera.update(dt);
            sync_renderer_agent_panel(renderer, &self.ecs_world);
        }
        {
            // SPRINT 4: Pass StressMap to the renderer so blocks are colored
            // by stress level. Snapshot stress first to avoid overlapping world borrows.
            let stress_by_entity: std::collections::HashMap<Entity, f32> = {
                let stress_map = self.ecs_world.get_resource::<StressMap>().unwrap();
                stress_map
                    .data
                    .iter()
                    .map(|(&entity, info)| (entity, info.stress_normalized))
                    .collect()
            };

            let mut query = self
                .ecs_world
                .query::<(Entity, &Voxel, Option<&agent::JointBodyProperties>)>();
            let voxels_with_stress: Vec<(&Voxel, f32, bool, Option<usize>)> = query
                .iter(&self.ecs_world)
                .map(|(entity, voxel, joint_body_properties)| {
                    let stress = stress_by_entity.get(&entity).copied().unwrap_or(0.0);
                    (
                        voxel,
                        stress,
                        voxel.material.is_scaffold(),
                        joint_body_properties.map(|joint_body| joint_body.segment.index()),
                    )
                })
                .collect();

            if let Some(renderer) = &mut self.renderer {
                commands_to_execute = renderer.render_frame_with_stress(&voxels_with_stress);
            }
        }

        let (block_commands_to_execute, agent_commands_to_execute) = commands_to_execute;

        for cmd in block_commands_to_execute {
            if let UiCommand::DespawnAllScaffold = cmd {
                let _ = self.despawn_scaffolding();
                continue;
            }

            let drop_height = {
                let gx = (match &cmd {
                    UiCommand::SpawnCube { x, .. }
                    | UiCommand::SpawnWedge { x, .. }
                    | UiCommand::SpawnSphere { x, .. }
                    | UiCommand::SpawnScaffold { x, .. } => *x,
                    UiCommand::DespawnAllScaffold => unreachable!(),
                })
                .round() as i32;
                let gz = (match &cmd {
                    UiCommand::SpawnCube { z, .. }
                    | UiCommand::SpawnWedge { z, .. }
                    | UiCommand::SpawnSphere { z, .. }
                    | UiCommand::SpawnScaffold { z, .. } => *z,
                    UiCommand::DespawnAllScaffold => unreachable!(),
                })
                .round() as i32;

                let grid = self.ecs_world.get_resource::<SpatialGrid>().unwrap();
                safe_spawn_height_for_grid(grid, gx, gz)
            };

            match cmd {
                UiCommand::SpawnCube { x, z, material_id } => {
                    let material = material_from_id(material_id);
                    spawn_voxel_and_register(
                        &mut self.ecs_world,
                        x,
                        drop_height,
                        z,
                        ShapeType::Cube,
                        material,
                        false,
                    );
                }
                UiCommand::SpawnWedge { x, z, material_id } => {
                    let material = material_from_id(material_id);
                    spawn_voxel_and_register(
                        &mut self.ecs_world,
                        x,
                        drop_height,
                        z,
                        ShapeType::Wedge,
                        material,
                        false,
                    );
                }
                UiCommand::SpawnSphere {
                    x,
                    z,
                    radius,
                    material_id,
                } => {
                    let material = material_from_id(material_id);
                    spawn_sphere_and_register(
                        &mut self.ecs_world,
                        x,
                        drop_height,
                        z,
                        radius,
                        material,
                        false,
                    );
                }
                UiCommand::SpawnScaffold { x, z } => {
                    spawn_voxel_and_register(
                        &mut self.ecs_world,
                        x,
                        drop_height,
                        z,
                        ShapeType::Cube,
                        MaterialType::Scaffold,
                        false,
                    );
                }
                UiCommand::DespawnAllScaffold => unreachable!(),
            }
        }

        let active_agent_id = latest_humanoid_agent_id(&self.ecs_world);
        for agent_command in agent_commands_to_execute {
            match agent_command {
                agent::UiAgentCommand::SpawnHumanoid => {
                    let spawn_x = self
                        .renderer
                        .as_ref()
                        .map(|renderer| renderer.spawn_x)
                        .unwrap_or(0.0);
                    let spawn_z = self
                        .renderer
                        .as_ref()
                        .map(|renderer| renderer.spawn_z)
                        .unwrap_or(0.0);
                    spawn_humanoid(
                        &mut self.ecs_world,
                        SpawnHumanoidParams::new(spawn_x, spawn_z),
                    );
                }
                agent::UiAgentCommand::DespawnAllHumanoids => {
                    let _ = despawn_all_humanoids(&mut self.ecs_world);
                }
                agent::UiAgentCommand::SetJointAngle {
                    segment,
                    target_angle_radians,
                } => {
                    if let Some(agent_id) = active_agent_id {
                        let _ = apply_torque_to_joint(
                            &mut self.ecs_world,
                            agent_id,
                            segment,
                            target_angle_radians,
                            default_swing_axis_for_segment(segment),
                        );
                    }
                }
                agent::UiAgentCommand::ApplyPosePreset { preset } => {
                    if let Some(agent_id) = active_agent_id {
                        for &(segment, target_angle_radians) in preset.joint_angles() {
                            let _ = apply_torque_to_joint(
                                &mut self.ecs_world,
                                agent_id,
                                segment,
                                target_angle_radians,
                                default_swing_axis_for_segment(segment),
                            );
                        }
                    }
                }
                agent::UiAgentCommand::MoveActiveHumanoid { direction } => {
                    if let Some(agent_id) = active_agent_id {
                        let _ = move_humanoid(&mut self.ecs_world, agent_id, direction);
                    }
                }
            }
        }
    }

    // =========================================================================
    // pump_os_events() — drain the OS event queue without blocking
    // =========================================================================
    //
    // LEARNING: This must be called regularly to keep the OS window responsive.
    // Without it, the OS thinks the app is "frozen" and marks it "Not Responding."
    //
    // In fully headless mode (renderer = None), this is a fast no-op.
    pub fn pump_os_events(&mut self) {
        let mut app = RenderEventPump {
            renderer: &mut self.renderer,
        };
        let _ = self
            .event_loop
            .pump_app_events(Some(Duration::ZERO), &mut app);
    }

    // =========================================================================
    // get_debug_state() — diagnostic data for debugging the physics (unchanged)
    // =========================================================================
    //
    // LEARNING: This returns a Python list of tuples — convenient for print-
    // debugging but NOT zero-copy (each tuple creates Python heap objects).
    // Use get_observation_into() for training; use this only for diagnostics.
    pub fn get_debug_state(&mut self) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>, u32)> {
        let mut query = self.ecs_world.query::<&Voxel>();
        let mut result = Vec::new();
        for voxel in query.iter(&self.ecs_world) {
            if voxel.inv_mass > 0.0 {
                result.push((
                    vec![voxel.position.x, voxel.position.y, voxel.position.z],
                    vec![voxel.velocity.x, voxel.velocity.y, voxel.velocity.z],
                    vec![
                        voxel.contact_normal_accum.x,
                        voxel.contact_normal_accum.y,
                        voxel.contact_normal_accum.z,
                    ],
                    voxel.contact_count,
                ));
            }
        }
        result
    }
}

// =============================================================================
// run_headless_benchmark() — standalone Python function for performance testing
// =============================================================================
//
// LEARNING TOPIC: Standalone #[pyfunction] vs #[pymethods]
// ----------------------------------------------------------
// #[pymethods] are methods on a #[pyclass] — they require a class instance.
// #[pyfunction] is a free function callable directly from Python:
//   import babel_engine
//   result = babel_engine.run_headless_benchmark(n_blocks=100, n_steps=10000)
//
// This benchmark is the Sprint 3 "Audit" tool:
//   - Spawns n_blocks blocks with mixed materials via spawn_voxel_and_register
//   - Runs n_steps physics steps including the full mortar pipeline
//   - Reports: steps/sec, memory stable (no crash = no leak over 1M steps)
//   - Target: >2000 steps/sec headless (mortar adds O(bonds) overhead)
//
// SPRINT 3 BUG FIX — Honest Benchmark:
// -------------------------------------
// The old benchmark spawned blocks with ecs_world.spawn(Voxel::new_with_material(...))
// directly, bypassing spawn_voxel_and_register(). This meant MortarBonds was always
// empty — the mortar solver and break systems did zero work per step.
//
// The reported steps/sec was therefore FASTER THAN ACTUAL training performance,
// because real training involves the mortar pipeline processing live bonds.
//
// FIX: All spawns now go through spawn_voxel_and_register(), which:
//   1. Creates the voxel
//   2. Immediately registers mortar bonds with any nearby neighbors
//   3. Gives the mortar systems real bonds to process during warmup and timing
//
// The benchmark result is now an honest representation of Sprint 3 throughput.
#[pyfunction]
pub fn run_headless_benchmark(n_blocks: usize, n_steps: usize) -> PyResult<(f64, u64)> {
    // ==========================================================================
    // LEARNING: Why no EventLoop / renderer here?
    // The benchmark creates a minimal engine with NO window. This exercises
    // the pure physics pipeline: ECS world + schedule + physics systems only.
    // No WGPU, no winit, no egui. This isolates the physics throughput from
    // any GPU/OS overhead.
    // ==========================================================================
    let mut ecs_world = World::new();
    ecs_world.insert_resource(SpatialGrid::default());
    ecs_world.insert_resource(PhysicsSettings::default());
    ecs_world.insert_resource(SolverBuffers::default());
    ecs_world.insert_resource(MortarBonds::default());
    let mut stress_map = StressMap::default();
    stress_map.min_update_interval = 8;
    ecs_world.insert_resource(stress_map);
    ecs_world.insert_resource(ScaffoldTracker::default());

    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            world::spatial_grid::update_spatial_grid_system,
            physics::xpbd::integrate_system,
            physics::xpbd::solve_constraints_system,
            register_new_bonds_system,
            solve_mortar_constraints_system,
            break_overloaded_bonds_system,
            physics::xpbd::update_velocities_system,
            world::spatial_grid::update_spatial_grid_system,
            compute_stress_system,
        )
            .chain(),
    );

    // Run one spatial grid update so the grid is populated before we start
    // calling spawn_voxel_and_register (which reads the grid for neighbor lookup).
    // Without this, the grid is empty for the first spawn and no bonds form
    // between the first block and any subsequent neighbors in the same column.
    schedule.run(&mut ecs_world);

    let grid_w = (n_blocks as f32).sqrt().ceil() as i32;
    for i in 0..n_blocks {
        let ix = (i as i32) % grid_w;
        let iz = (i as i32) / grid_w;
        let shape = if i % 3 == 0 {
            ShapeType::Wedge
        } else {
            ShapeType::Cube
        };
        // Mix materials for benchmark realism — tests bond formation and mortar
        // constraint solving across all material combinations.
        let material = match i % 3 {
            0 => MaterialType::Stone, // no bonds, high friction
            1 => MaterialType::Wood,  // light bonds
            _ => MaterialType::Steel, // heavy, strong bonds
        };

        // SPRINT 3 FIX: Use spawn_voxel_and_register instead of direct spawn.
        // This registers mortar bonds between neighbors so the benchmark exercises
        // the full Sprint 3 physics pipeline — honest performance numbers.
        spawn_voxel_and_register(
            &mut ecs_world,
            ix as f32,
            5.0,
            iz as f32,
            shape,
            material,
            false,
        );
    }

    // =========================================================================
    // Warm up BEFORE starting the timer.
    // -----------------------------------------------------------------------
    // The benchmark spawns all blocks at y=5.0 falling simultaneously.
    // The first ~100 steps = ALL blocks colliding at once = maximum solver work.
    // This is a worst-case burst that doesn't represent steady-state training.
    //
    // Without warmup: benchmark measures "blocks exploding downward" speed.
    // With warmup:    benchmark measures "settled world" speed = what RL sees.
    //
    // LEARNING: This is standard benchmark practice called "warm-up phase."
    // 200 steps at DT=1/60 = ~3.3 seconds of simulation time.
    // By then all blocks have landed and the solver handles normal stacking.
    // =========================================================================
    for _ in 0..200 {
        schedule.run(&mut ecs_world);
    }

    let start = Instant::now();
    for _ in 0..n_steps {
        schedule.run(&mut ecs_world);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let steps_per_sec = n_steps as f64 / elapsed;

    Ok((steps_per_sec, n_steps as u64))
}

// =============================================================================
// PyModule registration — the entry point when Python does `import babel_engine`
// =============================================================================
//
// LEARNING: The function name MUST match the `name` field in Cargo.toml:
//   [lib]
//   name = "babel_engine"
//   crate-type = ["cdylib"]
//
// PyO3/Maturin uses this to generate the correct .pyd / .so file name.
// The `m` parameter is the Python module object we're building.
// m.add_class::<T>() registers T as a Python class.
// m.add_function() registers a standalone Python function.
#[pymodule]
fn babel_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BabelEngine>()?;
    m.add_function(wrap_pyfunction!(run_headless_benchmark, m)?)?;
    Ok(())
}
