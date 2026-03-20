// =============================================================================
// src/bridge/mod.rs  —  Operation Babel: Python/Rust Bridge Module
// =============================================================================
//
// LEARNING TOPIC: Rust Module System
// ------------------------------------
// Rust's module system is hierarchical. `mod gym_env;` here tells the compiler
// to look for `src/bridge/gym_env.rs` and compile it as a submodule of `bridge`.
//
// The bridge module is the boundary between:
//   - The Rust engine (physics, ECS, rendering)
//   - The Python training code (RL algorithms, neural networks)
//
// All FFI-facing code that isn't part of the core BabelEngine struct lives here.
// This keeps lib.rs focused on orchestration and keeps bridge concerns isolated.
//
// SPRINT ROADMAP for this module:
//   Sprint 2 (now):  gym_env.rs — Gymnasium-compatible environment wrapper
//   Sprint 7 (later): zero-copy shared memory tensors for parallel envs
//   Sprint 11:        Axon distributed training hooks
// =============================================================================

pub mod gym_env;
