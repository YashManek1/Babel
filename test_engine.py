# =============================================================================
# test_engine.py  —  Operation Babel: Interactive Engine Test + Benchmark
# =============================================================================
#
# LEARNING TOPIC: Separation of Rendering and Physics
# ---------------------------------------------------
# This script demonstrates the decoupled architecture:
#   - Normal mode:    step() + render() at 60 Hz (visual debugging)
#   - Benchmark mode: step_batch(N) only, no rendering (maximum physics speed)
#
# Usage:
#   python test_engine.py           → visual mode (original behavior)
#   → headless benchmark mode
#   python test_engine.py --bench --blocks 500 --steps 20000
#
# =============================================================================

import babel_engine
import time
import sys

# =============================================================================
# LEARNING: sys.argv argument parsing (no argparse to keep imports minimal)
# =============================================================================
bench_mode  = "--bench"  in sys.argv
n_blocks    = 100
n_steps     = 10_000

# Parse optional --blocks and --steps arguments
for i, arg in enumerate(sys.argv):
    if arg == "--blocks" and i + 1 < len(sys.argv):
        n_blocks = int(sys.argv[i + 1])
    if arg == "--steps" and i + 1 < len(sys.argv):
        n_steps = int(sys.argv[i + 1])

# =============================================================================
# BENCHMARK MODE — headless, maximum physics throughput
# =============================================================================
if bench_mode:
    print("=" * 60)
    print("  Operation Babel — Headless Benchmark")
    print("=" * 60)
    print(f"  Blocks:     {n_blocks}")
    print(f"  Steps:      {n_steps:,}")
    print()

    # ==========================================================================
    # LEARNING: run_headless_benchmark() is a standalone pyfunction (not a method
    # on BabelEngine). It creates its own minimal ECS World with no WGPU window,
    # runs N steps, then returns performance stats.
    #
    # This is the cleanest way to measure pure physics throughput:
    #   No window creation overhead
    #   No event loop overhead
    #   No render pipeline overhead
    #   JUST the physics schedule: spatial_grid → integrate → solve → velocity
    # ==========================================================================
    print("Running standalone headless benchmark...")
    steps_per_sec, total_steps = babel_engine.run_headless_benchmark(n_blocks, n_steps)

    print(f"  Steps/second:  {steps_per_sec:>10,.0f}")
    print(f"  Target:        {2000:>10,} steps/sec")
    print(f"  Status:        {'✓ PASS' if steps_per_sec >= 2000 else '✗ FAIL (need optimization)'}")
    print()

    # ==========================================================================
    # LEARNING: Zero-copy observation benchmark
    # The second test measures observation extraction performance — the bottleneck
    # between physics and the neural network's forward pass.
    #
    # We create a full BabelEngine (with window suppressed by checking None renderer)
    # and use get_observation_into() to write directly into a numpy array.
    # ==========================================================================
    try:
        import numpy as np
        print("Testing zero-copy observation bridge...")

        # Pre-allocate the observation array ONCE (simulates what the RL trainer does)
        # LEARNING: dtype=np.float32 is essential — Rust writes f32, not f64.
        # If dtype is wrong, PyO3 will raise an error or silently misinterpret bytes.
        stride = 12  # OBSERVATION_STRIDE from lib.rs
        obs = np.zeros(n_blocks * stride, dtype=np.float32)

        engine = babel_engine.BabelEngine()
        engine.reset_benchmark()

        # Spawn test blocks
        import math
        grid_w = math.ceil(math.sqrt(n_blocks))
        for i in range(n_blocks):
            ix = i % grid_w
            iz = i // grid_w
            engine.spawn_block(float(ix), 5.0, float(iz), 0, False)  # shape_id=0 = Cube

        # Warm up (let physics settle before measuring)
        engine.step_batch(100)
        engine.reset_benchmark()

        # Benchmark: N steps + observation extraction per step
        obs_start = time.perf_counter()
        batch_size = 64
        batches = n_steps // batch_size

        for _ in range(batches):
            engine.step_batch(batch_size)
            # ZERO-COPY WRITE: Rust writes directly into obs.data buffer
            # No Python objects created, no heap allocations
            n_written = engine.get_observation_into(obs)

        obs_elapsed = time.perf_counter() - obs_start
        total, elapsed, sps = engine.get_benchmark_stats()

        print(f"  Observation size:  {n_written} blocks × {stride} floats = {n_written * stride} floats")
        print(f"  Steps with obs/s:  {total / obs_elapsed:>10,.0f}")
        print(f"  Physics only /s:   {sps:>10,.0f}")
        print(f"  Obs overhead:      {((total / obs_elapsed - sps) / sps * 100):+.1f}%")
        print()
        print("  Tip: torch.from_numpy(obs[:n_written * 12]) gives a ZERO-COPY tensor!")

    except ImportError:
        print("  numpy not found — skipping zero-copy bridge test")
        print("  Install with: pip install numpy")

    print()
    print("Benchmark complete.")
    sys.exit(0)

# =============================================================================
# VISUAL MODE — original interactive mode with rendering
# =============================================================================
print("Engine running! Click the ground to target, and the UI to spawn.")
print("Right-click + drag to orbit | Scroll to zoom | WASD to pan")
print("Run with --bench for headless benchmark mode")
print()

engine = babel_engine.BabelEngine()
engine.reset_benchmark()

frame_count = 0
report_interval = 300  # Print perf stats every 300 frames (~5 seconds at 60fps)

while True:
    # ==========================================================================
    # LEARNING: step() vs step_batch() in visual mode
    # Using step() here (single step per frame) keeps physics synchronized with
    # rendering at 60 Hz. If we used step_batch(10), the physics would run 10×
    # faster than what the user sees — blocks would teleport rather than fall.
    #
    # For visual debugging: step() once per render frame.
    # For AI training:      step_batch(N) as fast as possible, render() occasionally.
    # ==========================================================================
    engine.step()
    engine.pump_os_events()
    engine.render()

    frame_count += 1

    # Periodic performance report to console
    if frame_count % report_interval == 0:
        total, elapsed, sps = engine.get_benchmark_stats()
        dynamic, static = engine.block_count()
        print(f"Frame {frame_count:6d} | "
              f"Blocks: {dynamic:3d} dynamic, {static:2d} static | "
              f"Physics: {sps:.0f} steps/sec")

    time.sleep(1 / 60)  # Target 60 FPS visual frame rate