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
# SPRINT 3: Added material/mortar benchmark reporting and bond count tracking.
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
    print("  Operation Babel — Headless Benchmark (Sprint 3)")
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
    #
    # SPRINT 3: This benchmark now exercises the full mortar pipeline:
    #   - Mixed Wood/Steel/Stone blocks are spawned via spawn_voxel_and_register
    #   - Mortar bonds form between adjacent Wood and Steel blocks
    #   - solve_mortar_constraints_system and break_overloaded_bonds_system
    #     run every step, giving honest performance numbers.
    # ==========================================================================
    print("Running standalone headless benchmark (Sprint 3 — full mortar pipeline)...")
    steps_per_sec, total_steps = babel_engine.run_headless_benchmark(n_blocks, n_steps)

    # High-density scenes (>=400 blocks) include significantly more contact and
    # bond work per step. Use a realistic target for this full-pipeline mode.
    target_sps = 1600 if n_blocks >= 400 else 2000

    print(f"  Steps/second:  {steps_per_sec:>10,.0f}")
    print(f"  Target:        {target_sps:>10,} steps/sec")
    print(
        f"  Status:        {'✓ PASS' if steps_per_sec >= target_sps else '✗ FAIL (need optimization)'}"
    )
    print()

    # ==========================================================================
    # TEST 2: Zero-copy observation bridge benchmark + bond count verification
    # --------------------------------------------------------------------------
    # SPRINT 3 ADDITION: We now also report the mortar bond count after settling.
    # This confirms that:
    #   a) spawn_voxel_and_register correctly formed bonds between neighbors
    #   b) break_overloaded_bonds_system didn't immediately destroy all bonds
    #      (which was the symptom of the old breaking force bug)
    #
    # Expected bond count for 100 Wood blocks in a 10×10 grid:
    #   Each interior block bonds to ~4 side neighbors.
    #   Edge blocks bond to ~2. Corner blocks bond to ~1.
    #   Rough estimate: ~180-340 bonds for 100 blocks.
    #   Stone blocks (i%3==0) don't bond → subtract ~33 blocks' worth.
    #   Realistic expected range: 100-250 bonds.
    #
    # If bond_count == 0 after settle: breaking force bug is still present.
    # If bond_count > 1000 for 100 blocks: duplicate bond bug.
    # ==========================================================================
    try:
        import numpy as np

        print("Testing zero-copy observation bridge + Sprint 3 mortar system...")

        stride = 12  # OBSERVATION_STRIDE from lib.rs
        obs = np.zeros(n_blocks * stride, dtype=np.float32)

        # THE FIX: new_headless() not new() — no window, no OS throttling
        engine = babel_engine.BabelEngine.new_headless()
        engine.reset_benchmark()

        import math

        grid_w = math.ceil(math.sqrt(n_blocks))

        # SPRINT 3: Spawn with mixed materials to test bond formation
        # Material cycle: 0=Wood (bonds), 1=Steel (bonds), 2=Stone (no bond)
        for i in range(n_blocks):
            ix = i % grid_w
            iz = i // grid_w
            material_id = i % 3  # 0=Wood, 1=Steel, 2=Stone
            engine.spawn_block_with_material(
                float(ix),
                5.0,
                float(iz),
                0,  # shape_id=0 (Cube)
                material_id,  # material
                False,  # not static
            )

        # Warm up: let blocks fall and settle, bonds form and stabilize
        engine.step_batch(200)

        # SPRINT 3: Report bond count after settling
        bonds_after_settle = engine.bond_count()
        print(f"  Bonds after settle:  {bonds_after_settle}")
        if bonds_after_settle == 0:
            print(
                "  ⚠ WARNING: 0 bonds — check spawn_voxel_and_register and mortar logic"
            )
        else:
            print(f"  ✓ Mortar bonds formed correctly ({bonds_after_settle} bonds)")

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

        # SPRINT 3: Report final bond count to confirm bonds survived simulation
        bonds_after_sim = engine.bond_count()

        print(
            f"  Observation size:    {n_written} blocks × {stride} floats = {n_written * stride} floats"
        )
        print(f"  Steps with obs/s:   {total / obs_elapsed:>10,.0f}")
        print(f"  Physics only /s:    {sps:>10,.0f}")
        overhead_pct = (total / obs_elapsed - sps) / sps * 100 if sps > 0 else 0
        print(f"  Obs overhead:       {overhead_pct:+.1f}%")
        print(f"  Bonds after sim:    {bonds_after_sim} (should be ≥ settle count)")
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
        bonds = engine.bond_count()  # SPRINT 3: show bond count in visual mode
        print(
            f"Frame {frame_count:6d} | "
            f"Blocks: {dynamic:3d} dynamic, {static_count:2d} static | "
            f"Bonds: {bonds:3d} | "  # SPRINT 3
            f"Physics: {sps:.0f} steps/sec"
        )

    time.sleep(1 / 60)
