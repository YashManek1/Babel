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
//       │       spatial_grid → integrate → solve_constraints → update_velocities → spatial_grid
//       │
//       ├─► RenderContext (WGPU window, only updated when render() is called)
//       │
//       └─► SpatialGrid, PhysicsSettings, SolverBuffers (Bevy ECS Resources)
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

use bevy_ecs::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadwriteArray1};
use pyo3::prelude::*;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    platform::pump_events::EventLoopExtPumpEvents,
    window::WindowId,
};

mod physics;
mod render;
mod world;

// =============================================================================
// LEARNING: Bringing bridge module into scope
// The bridge module contains the zero-copy gym environment wrapper.
// We declare it here so it compiles as part of this crate.
// =============================================================================
mod bridge;

use physics::xpbd::{PhysicsSettings, SolverBuffers};
use render::wgpu_view::{RenderContext, UiCommand};
use world::spatial_grid::SpatialGrid;
use world::voxel::{ShapeType, Voxel};

// =============================================================================
// OPTIMIZATION CONSTANT: Maximum Stack Height Cap
// ------------------------------------------------
// When spawning blocks from the UI, we read the SpatialGrid to find the highest
// existing block in that column and drop the new block 5 units above it.
// Without a cap, if the grid somehow contains corrupt data at very high Y values,
// the engine would spawn blocks at y=1000+ which immediately explodes the solver.
//
// Capped at 16: this is the "Tutorial" and "Benchmark" height range from the
// Operation Babel design doc. Higher construction requires the multi-agent
// Bucket Brigade (Phase IV).
// =============================================================================
const MAX_STACK_HEIGHT: i32 = 16;

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

#[pymethods]
impl BabelEngine {
    #[new]
    pub fn new() -> Self {
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

        // =======================================================================
        // LEARNING: Bevy's System Scheduling and .chain()
        // ------------------------------------------------
        // .chain() enforces strict sequential ordering within the tuple.
        // Without it, Bevy might parallelize systems that have data dependencies.
        //
        // Our pipeline has strict dependencies:
        //   1. update_spatial_grid   ← must run FIRST (builds the grid from
        //                              current positions for neighbor lookups)
        //   2. integrate             ← writes predicted_position (reads velocity)
        //   3. solve_constraints     ← reads spatial grid, writes predicted_pos
        //   4. update_velocities     ← commits predicted_pos → position, derives v
        //   5. update_spatial_grid   ← SECOND run: rebuild grid from FINAL positions
        //                              (so render and Python queries see correct state)
        //
        // The second spatial grid update is crucial: without it, the grid would
        // contain positions from BEFORE constraint resolution, causing the next
        // frame's broad-phase to find false-positive overlaps.
        // =======================================================================
        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                world::spatial_grid::update_spatial_grid_system,
                physics::xpbd::integrate_system,
                physics::xpbd::solve_constraints_system,
                physics::xpbd::update_velocities_system,
                world::spatial_grid::update_spatial_grid_system,
            )
                .chain(),
        );

        let event_loop = EventLoop::new().unwrap();
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
    // At 2000 steps/sec target, if we call step() once per Python iteration:
    //   Overhead = 2000 × 1μs = 2ms/sec = 0.2% overhead (acceptable)
    //
    // But our goal is to hit 2000+ steps/sec. The ACTUAL physics is ~100μs
    // per step. So with step():
    //   Max speed = 1 / (100μs + 1μs overhead) = ~9900 steps/sec
    //   (Sounds fine, but Python's sleep/loop overhead adds another 10-50μs)
    //   Realistic: ~2000-5000 steps/sec with step() from a tight Python loop
    //
    // With step_batch(256) from Python, Rust runs 256 steps with ZERO Python
    // interruptions:
    //   Rust loop = 256 × 100μs = 25.6ms of pure Rust
    //   PyO3 overhead = 1μs (paid ONCE per batch, not per step)
    //   Max speed = 256 / (25.6ms + 0.001ms) ≈ 10,000+ steps/sec
    //
    // This is the core insight: Python should be the DIRECTOR (train/evaluate),
    // not the EXECUTOR (inner physics loop). Let Rust execute the inner loop.
    //
    // Returns (steps_completed, block_count) so Python can monitor training.
    pub fn step_batch(&mut self, n: usize) -> PyResult<(usize, usize)> {
        // Run the full physics schedule N times with no Python intervention.
        // The CPU cache stays hot: all ECS component data accessed in iteration 1
        // is still in L2/L3 cache for iteration 2 (same memory addresses).
        //
        // LEARNING: This is cache locality in practice. The Voxel components are
        // stored in contiguous memory by Bevy's archetype system. Running the
        // schedule N times without any intervening Python allocation keeps that
        // data resident in CPU cache, giving near-linear speedup with batch size.
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
    //   4. PyTorch converts array → tensor   (possibly allocation #4 + copy #3)
    //
    // ZERO-COPY approach:
    //   Python allocates np.zeros(N, dtype=np.float32) ONCE before training.
    //   Each step: Rust receives a mutable reference to that array's raw memory.
    //   Rust writes directly into it. PyTorch can use torch.from_numpy() to get
    //   a tensor that SHARES the same physical memory.
    //
    //   Total copies: ZERO. Total allocations per step: ZERO.
    //
    // USAGE FROM PYTHON:
    //   obs = np.zeros(engine.max_observation_size(), dtype=np.float32)
    //   n_blocks = engine.get_observation_into(obs)
    //   tensor = torch.from_numpy(obs[:n_blocks * 12])  # zero-copy view!
    //
    // LEARNING: PyReadwriteArray1<f32> is PyO3's way of borrowing a mutable
    // reference to a numpy array's underlying data buffer. The 'py lifetime
    // ensures Python's GIL is held for the duration of the write, preventing
    // another Python thread from resizing the array while Rust is writing into it.
    //
    // Returns: number of DYNAMIC blocks written (static blocks are skipped —
    // their positions never change so no RL agent needs to observe them).
    pub fn get_observation_into<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<usize> {
        // Borrow the numpy array as a writable Rust slice.
        // This is a zero-cost borrow — no data is copied.
        // PyO3 validates that the array is:
        //   - contiguous in memory (C-order, not Fortran-order)
        //   - dtype == float32 (matches our Vertex/physics data)
        //   - not currently borrowed by another Rust function
        let mut rw: PyReadwriteArray1<f32> = obs.readwrite();
        let slice: &mut [f32] = rw.as_slice_mut()?;

        let mut query = self.ecs_world.query::<&Voxel>();
        let mut write_idx = 0usize;
        let mut block_count = 0usize;

        for voxel in query.iter(&self.ecs_world) {
            // Skip static objects — they never move, neural network doesn't
            // need to observe them (their positions are architectural constants).
            if voxel.inv_mass == 0.0 {
                continue;
            }

            // Bounds check: don't overflow the pre-allocated array.
            // LEARNING: In a well-designed training loop, Python allocates
            // obs = np.zeros(max_blocks * STRIDE) before the episode starts.
            // If the world has more blocks than max_blocks, we silently truncate.
            // This is intentional — the RL agent has a fixed input size.
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

        // Suppress unused variable warning for 'py — it's needed to prove
        // the GIL is held during the write (PyO3 lifetime-based safety).
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
    // get_benchmark_stats() — Real steps/second measurement
    // =========================================================================
    //
    // LEARNING: Measure performance FROM INSIDE RUST to avoid PyO3 timing bias.
    // If Python measured time around step_batch(), it would include:
    //   - Python interpreter overhead
    //   - GIL acquire/release
    //   - Any Python code between calls
    //
    // Measuring from Rust gives the TRUE physics throughput, independent of
    // Python overhead. This matches the project's performance target:
    //   "The engine must run at >2000 steps per second (headless)"
    //   — from the Criteria & Requirements section.
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
    // spawn_block() — programmatic block spawning from Python (for RL reset)
    // =========================================================================
    //
    // LEARNING: In RL training, episodes start by calling env.reset().
    // Reset needs to spawn blocks into specific positions (the "blueprint").
    // This method allows Python to spawn blocks at exact positions without
    // going through the UI command queue.
    //
    // shape_id: 0=Cube, 1=Wedge, 2=Sphere
    // is_static: true = immovable (foundation), false = dynamic (interactive)
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
            _ => ShapeType::Cube, // Default to cube for unknown IDs
        };
        self.ecs_world.spawn(Voxel::new(x, y, z, shape, is_static));
        Ok(())
    }

    // =========================================================================
    // clear_world() — remove all dynamic blocks (RL episode reset)
    // =========================================================================
    //
    // LEARNING TOPIC: Bevy ECS Despawn Strategy
    // ------------------------------------------
    // In RL training, each episode starts with a fresh world state.
    // We have two options:
    //
    // Option A: Create a new World() each reset.
    //   → Clean state guaranteed
    //   → But: re-inserting all Resources, re-building all schedules
    //   → Cost: ~1-5ms per reset (acceptable for >1 second episodes)
    //
    // Option B: Despawn only dynamic entities, keep the World alive.
    //   → Resources (SpatialGrid, SolverBuffers) are already allocated
    //   → Their internal HashMaps retain their allocated memory (no realloc)
    //   → Cost: O(N) entity despawn, ~10-100μs for 1000 blocks
    //   → BETTER for fast episode cycling (e.g., 10-second episodes)
    //
    // We implement Option B. Static entities (inv_mass == 0.0) are kept
    // because they represent the environment structure (floors, walls).
    // Dynamic entities (the blocks being stacked) are cleared for reset.
    pub fn clear_dynamic_blocks(&mut self) -> PyResult<usize> {
        // Collect entities to despawn (can't despawn while iterating)
        let mut query = self.ecs_world.query::<(Entity, &Voxel)>();
        let to_despawn: Vec<Entity> = query
            .iter(&self.ecs_world)
            .filter(|(_, v)| v.inv_mass > 0.0) // Only dynamic blocks
            .map(|(e, _)| e)
            .collect();

        let count = to_despawn.len();
        for entity in to_despawn {
            // LEARNING: despawn() removes the entity AND all its components.
            // Bevy's archetype system immediately reclaims the memory slot,
            // making it available for future spawn() calls without reallocation.
            self.ecs_world.despawn(entity);
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
    // The render() call is expensive (~5-15ms for GPU submit + vsync wait).
    // The step_batch() call is fast (~1-25ms for 256 physics steps).
    // By separating them, the RL trainer can run at 10,000+ steps/sec
    // while the human observer can call render() every 256 steps to watch
    // at an effective "video" of the training at 30-60 FPS.
    //
    // This is also why BabelEngine owns BOTH the physics World and the
    // RenderContext: they share ECS data but are called on separate cadences.
    pub fn render(&mut self) {
        let now = Instant::now();
        let dt = self.last_frame.elapsed().as_secs_f32().min(0.1);
        self.last_frame = now;
        let mut commands_to_execute = Vec::new();
        if let Some(renderer) = &mut self.renderer {
            renderer.camera.update(dt);
        }
        {
            let mut query = self.ecs_world.query::<&Voxel>();
            let voxels: Vec<&Voxel> = query.iter(&self.ecs_world).collect();
            if let Some(renderer) = &mut self.renderer {
                commands_to_execute = renderer.render_frame(&voxels);
            }
        }

        for cmd in commands_to_execute {
            // =================================================================
            // OPTIMIZATION: Column-height lookup before spawning
            // -------------------------------------------------
            // We read the SpatialGrid (already built by the last physics step)
            // to find the highest existing block at the spawn X/Z column.
            // This prevents new blocks from spawning INSIDE existing towers.
            //
            // The cap at MAX_STACK_HEIGHT prevents runaway heights from a bad
            // physics state (e.g., a block launched upward by solver instability
            // registering as the "highest" block in its column).
            // =================================================================
            let drop_height = {
                let gx = (match &cmd {
                    UiCommand::SpawnCube { x, .. }
                    | UiCommand::SpawnWedge { x, .. }
                    | UiCommand::SpawnSphere { x, .. } => *x,
                })
                .round() as i32;
                let gz = (match &cmd {
                    UiCommand::SpawnCube { z, .. }
                    | UiCommand::SpawnWedge { z, .. }
                    | UiCommand::SpawnSphere { z, .. } => *z,
                })
                .round() as i32;

                let grid = self.ecs_world.get_resource::<SpatialGrid>().unwrap();
                match grid.column_max_y(gx, gz) {
                    Some(y) => {
                        let capped = y.min(MAX_STACK_HEIGHT);
                        capped as f32 + 5.0
                    }
                    None => 10.0,
                }
            };

            match cmd {
                UiCommand::SpawnCube { x, z } => {
                    self.ecs_world
                        .spawn(Voxel::new(x, drop_height, z, ShapeType::Cube, false));
                }
                UiCommand::SpawnWedge { x, z } => {
                    self.ecs_world
                        .spawn(Voxel::new(x, drop_height, z, ShapeType::Wedge, false));
                }
                UiCommand::SpawnSphere { x, z, radius } => {
                    self.ecs_world
                        .spawn(Voxel::new_sphere(x, drop_height, z, radius, false));
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
    // Even in headless training, if a window exists, it must receive events.
    //
    // In fully headless mode (no window), this is a no-op since renderer = None.
    // The EventLoop still processes events but they all go nowhere.
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
// This benchmark is the Sprint 2 "Audit" tool:
//   - Spawns n_blocks blocks in a tower formation
//   - Runs n_steps physics steps with NO rendering
//   - Reports: steps/sec, memory stable (no crash = no leak over 1M steps)
//   - Target: >2000 steps/sec headless
//
// MEMORY LEAK DETECTION STRATEGY:
//   If step_batch() had a memory leak (e.g., forgot to clear a Vec inside
//   SolverBuffers), repeated calls would grow RSS memory monotonically.
//   The benchmark runs long enough to trigger OS memory pressure if leaking.
//   A stable step rate (not slowing down over time) implies stable memory.
//   For precise leak detection, run under `valgrind` or `heaptrack` in release.
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

    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            world::spatial_grid::update_spatial_grid_system,
            physics::xpbd::integrate_system,
            physics::xpbd::solve_constraints_system,
            physics::xpbd::update_velocities_system,
            world::spatial_grid::update_spatial_grid_system,
        )
            .chain(),
    );

    // Spawn blocks in a flat grid layout (avoids immediate-collapse instability
    // from a pre-stacked tower that hasn't settled yet).
    // A 10×10 grid fits 100 blocks; larger grids spread further.
    let grid_w = (n_blocks as f32).sqrt().ceil() as i32;
    for i in 0..n_blocks {
        let ix = (i as i32) % grid_w;
        let iz = (i as i32) / grid_w;
        // Interleave cubes and wedges for a stress-test of collision dispatch variety
        let shape = if i % 3 == 0 {
            ShapeType::Wedge
        } else {
            ShapeType::Cube
        };
        ecs_world.spawn(Voxel::new(
            ix as f32,
            5.0, // drop from height 5 — they fall and settle during the benchmark
            iz as f32, shape, false,
        ));
    }

    // =========================================================================
    // OPTIMIZATION: Use Instant outside the loop to measure PURE physics time.
    // The loop itself has zero overhead: no print, no Python calls, no sleep.
    // This is as close to native Rust loop speed as possible from PyO3.
    // =========================================================================
    let start = Instant::now();

    // BATCH SIZE TUNING:
    // Running all n_steps as one loop gives maximum cache locality.
    // In practice, training loops call step_batch(64) or step_batch(256)
    // per Python iteration. Both achieve similar throughput due to Bevy's
    // archetype-based memory layout keeping Voxel data cache-hot.
    for _ in 0..n_steps {
        schedule.run(&mut ecs_world);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let steps_per_sec = n_steps as f64 / elapsed;

    // Returns (steps_per_second, total_steps) — Python can print and compare
    // against the >2000 steps/sec target from the project requirements.
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

    // Register the standalone benchmark function so Python can call it as:
    //   babel_engine.run_headless_benchmark(n_blocks=100, n_steps=10000)
    m.add_function(wrap_pyfunction!(run_headless_benchmark, m)?)?;

    Ok(())
}
