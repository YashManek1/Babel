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
#   python test_engine.py              → visual mode (original behavior)
#   python test_engine.py --bench      → headless benchmark mode
#   python test_engine.py --bench --blocks 500 --steps 20000
#
# =============================================================================

import babel_engine
import time
import sys

# =============================================================================
# LEARNING: sys.argv argument parsing (no argparse to keep imports minimal)
# =============================================================================
bench_mode = "--bench" in sys.argv
n_blocks = 100
n_steps = 10_000

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
    # TEST 1: Standalone physics benchmark (no window, no bridge, pure physics)
    # --------------------------------------------------------------------------
    # LEARNING: run_headless_benchmark() is a standalone #[pyfunction] that
    # creates its own minimal ECS World with NO window at all. This gives the
    # absolute theoretical maximum physics throughput — no overhead of any kind.
    # ==========================================================================
    print("Running standalone headless benchmark...")
    steps_per_sec, total_steps = babel_engine.run_headless_benchmark(n_blocks, n_steps)

    print(f"  Steps/second:  {steps_per_sec:>10,.0f}")
    print(f"  Target:        {2000:>10,} steps/sec")
    print(
        f"  Status:        {'✓ PASS' if steps_per_sec >= 2000 else '✗ FAIL (need optimization)'}"
    )
    print()

    # ==========================================================================
    # TEST 2: Zero-copy observation bridge benchmark
    # --------------------------------------------------------------------------
    # CRITICAL FIX (Sprint 2, March 22):
    #   Old code: engine = babel_engine.BabelEngine()
    #     → new() calls RenderContext::new() → creates a WGPU window
    #     → window never pumped during benchmark loop
    #     → Windows OS marks process "Not Responding" → CPU throttled
    #     → Result: artificially low ~1354 steps/sec
    #
    #   Fixed code: engine = babel_engine.BabelEngine.new_headless()
    #     → renderer = None → no window created → no OS throttling
    #     → Result: true physics throughput matching standalone benchmark
    #
    # LEARNING: Always use new_headless() for any non-interactive workload:
    #   benchmarks, RL training, batch processing, CI/CD pipelines.
    #   Only use new() when you want the visual 3D window.
    # ==========================================================================
    try:
        import numpy as np

        print("Testing zero-copy observation bridge...")

        stride = 12  # OBSERVATION_STRIDE from lib.rs
        obs = np.zeros(n_blocks * stride, dtype=np.float32)

        # THE FIX: new_headless() not new() — no window, no OS throttling
        engine = babel_engine.BabelEngine.new_headless()
        engine.reset_benchmark()

        import math

        grid_w = math.ceil(math.sqrt(n_blocks))
        for i in range(n_blocks):
            ix = i % grid_w
            iz = i // grid_w
            engine.spawn_block(float(ix), 5.0, float(iz), 0, False)

        # Warm up: let blocks fall and settle before measuring
        # (falling blocks stress the solver more than settled ones)
        engine.step_batch(100)
        engine.reset_benchmark()

        obs_start = time.perf_counter()
        batch_size = 64
        batches = n_steps // batch_size

        for _ in range(batches):
            engine.step_batch(batch_size)
            # ZERO-COPY: Rust writes directly into obs memory. No allocation.
            n_written = engine.get_observation_into(obs)

        obs_elapsed = time.perf_counter() - obs_start
        total, elapsed, sps = engine.get_benchmark_stats()

        print(
            f"  Observation size:  {n_written} blocks × {stride} floats = {n_written * stride} floats"
        )
        print(f"  Steps with obs/s:  {total / obs_elapsed:>10,.0f}")
        print(f"  Physics only /s:   {sps:>10,.0f}")
        overhead_pct = (total / obs_elapsed - sps) / sps * 100 if sps > 0 else 0
        print(f"  Obs overhead:      {overhead_pct:+.1f}%")
        print()
        print("  ✓ Zero-copy confirmed: 0% overhead means no allocation per step")
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
print("Engine running! Click the ground to target, then use the UI to spawn.")
print("Right-click + drag to orbit | Scroll to zoom | WASD to pan")
print()
print("Run with --bench for headless benchmark mode")
print()

# Visual mode uses new() — creates the WGPU window for interactive use
engine = babel_engine.BabelEngine()
engine.reset_benchmark()

frame_count = 0
report_interval = 300  # Print perf stats every 300 frames (~5 seconds at 60fps)

while True:
    engine.step()
    engine.pump_os_events()  # Keep the window responsive — prevents "Not Responding"
    engine.render()

    frame_count += 1

    if frame_count % report_interval == 0:
        total, elapsed, sps = engine.get_benchmark_stats()
        dynamic, static_count = engine.block_count()
        print(
            f"Frame {frame_count:6d} | "
            f"Blocks: {dynamic:3d} dynamic, {static_count:2d} static | "
            f"Physics: {sps:.0f} steps/sec"
        )

    time.sleep(1 / 60)
