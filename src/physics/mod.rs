// =============================================================================
// src/physics/mod.rs  —  Operation Babel: Physics Module Registry
// =============================================================================
//
// LEARNING TOPIC: Rust Module System
// ------------------------------------
// Each `pub mod` declaration here tells the compiler to look for a corresponding
// .rs file in src/physics/ and compile it as a submodule.
//
// The physics pipeline systems that run every frame (in order, via .chain()):
//   mortar.rs  — mortar bond constraint solving and breaking
//   stress.rs  — SPRINT 4: structural stress propagation and heatmap data
//   xpbd.rs    — Extended Position-Based Dynamics core solver
// =============================================================================

pub mod mortar;
pub mod stress; // SPRINT 4: structural analytics
pub mod xpbd;
