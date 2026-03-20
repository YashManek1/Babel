// =============================================================================
// src/bridge/gym_env.rs  —  Operation Babel: Farama Gymnasium Bridge
// =============================================================================
//
// LEARNING TOPIC: The Gymnasium Standard (formerly OpenAI Gym)
// ------------------------------------------------------------
// Farama Gymnasium defines the universal interface for RL environments:
//
//   env.reset()  → (observation, info)
//   env.step(action) → (observation, reward, terminated, truncated, info)
//
// Every RL library (StableBaselines3, CleanRL, RLlib, etc.) can train on ANY
// environment that implements this interface. By making Operation Babel
// Gymnasium-compatible, we get access to the entire RL ecosystem for FREE.
//
// This file defines the Rust-side data structures and helper functions that
// support the Python-side Gymnasium wrapper (in brain/babel_gym_env.py).
//
// ARCHITECTURE: Why not implement Gymnasium entirely in Rust?
// -----------------------------------------------------------
// The Gymnasium interface is Python-native (it uses Python dicts, numpy arrays,
// gym.spaces objects). Implementing it fully in Rust would require PyO3 wrappers
// for every Gymnasium concept — more complexity than value.
//
// Instead, we use a HYBRID approach:
//   Rust: fast physics, observation packing (zero-copy), reward calculation
//   Python: Gymnasium wrapper, action parsing, episode management
//
// The Python wrapper calls Rust functions that do the heavy lifting.
// Python is the "thin shell" — the RL algorithm talks to Python Gymnasium,
// Python delegates computation to Rust via this bridge.
//
// =============================================================================
//
// LEARNING TOPIC: Reward Signal Design
// -------------------------------------
// The reward function is the most important design decision in RL.
// A bad reward → the agent finds unexpected ways to "hack" the score.
//   (Classic example: the BoatRace agent learned to spin in circles collecting
//    power-ups instead of racing because spinning gave more reward per second.)
//
// Operation Babel's reward hierarchy (from the project docs):
//   1. HEIGHT reward:   +1.0 per block above the previous max height
//      → Encourages building tall
//   2. STABILITY bonus: × (1 - total_stress / max_strength)
//      → Discourages "noodle towers" — encourages wide stable bases
//   3. MATERIAL bonus:  small positive for using appropriate materials
//   4. TIME penalty:    -0.001 per step → agent learns to be efficient
//   5. COLLAPSE penalty: -10.0 when tower falls below previous max height
//      → Strong signal to avoid instability
//
// These are computed in calculate_reward() below using data from the physics sim.
// =============================================================================

use crate::world::voxel::Voxel;
use bevy_ecs::prelude::*;

// =============================================================================
// ObservationConfig: defines the observation space for the RL agent
// =============================================================================
//
// LEARNING: The observation space tells the RL algorithm what it will receive
// as input. Different tasks might need different observation configurations:
//
// Phase II (God Brain): Full world state — all blocks' positions and velocities.
//   obs_size = max_blocks * OBSERVATION_STRIDE (potentially thousands of floats)
//
// Phase IV (Emergent Communication): Partial observation — each agent only sees
//   blocks within its "Lidar Vision" range (Sprint 6: 64-ray distance sensing).
//   obs_size = 64 (lidar) + 12 * nearby_blocks
//
// We expose this config to Python so the RL framework can build the correct
// gym.spaces.Box(low=..., high=..., shape=(obs_size,), dtype=np.float32).
pub struct ObservationConfig {
    /// Maximum number of blocks the agent can observe per step.
    /// Must match the pre-allocated numpy array size on the Python side.
    pub max_blocks: usize,
    /// Floats per block in the observation vector (see OBSERVATION_STRIDE in lib.rs)
    pub stride: usize,
    /// Total floats in the observation array = max_blocks × stride
    pub total_size: usize,
}

impl ObservationConfig {
    pub fn new(max_blocks: usize, stride: usize) -> Self {
        Self {
            max_blocks,
            stride,
            total_size: max_blocks * stride,
        }
    }
}

// =============================================================================
// EpisodeState: tracks per-episode progress for reward calculation
// =============================================================================
//
// LEARNING: RL episodes need to track "what happened before this step" to
// calculate things like:
//   - Did the tower just collapse? (current_max_height < previous_max_height)
//   - Is the agent making progress? (current_max_height > best_height)
//   - How many steps have elapsed? (for truncation / time limit)
//
// This struct is owned by the Python Gymnasium wrapper, which passes relevant
// fields to calculate_reward() each step. It's kept simple (plain f32/u32) so
// it crosses the FFI boundary cheaply.
pub struct EpisodeState {
    /// The highest block Y position achieved so far in this episode.
    /// Updated each step with the current max block height.
    pub max_height_achieved: f32,

    /// Number of physics steps elapsed in this episode.
    pub step_count: u32,

    /// Maximum allowed steps before episode truncation (time limit).
    /// Prevents infinite episodes when the agent gets stuck.
    pub max_steps: u32,

    /// Running sum of total stress across all blocks last step.
    /// Used to detect tower collapse (stress → 0 means blocks scattered).
    pub prev_total_stress: f32,
}

impl Default for EpisodeState {
    fn default() -> Self {
        Self {
            max_height_achieved: 0.0,
            step_count: 0,
            max_steps: 1000, // Default: 1000 physics steps per episode
            prev_total_stress: 0.0,
        }
    }
}

// =============================================================================
// calculate_reward() — Compute the RL reward signal from world state
// =============================================================================
//
// LEARNING TOPIC: Reward Shaping
// --------------------------------
// Raw task reward ("did you build a 10-story tower? +100") is too sparse.
// The agent rarely stumbles upon the solution by chance, so it gets almost
// no gradient signal to learn from.
//
// "Dense reward" provides feedback EVERY STEP:
//   - +0.1 for each block above ground level (continuous progress signal)
//   - -0.01 for each step taken (encourages efficiency)
//   - +1.0 bonus when a new height record is set
//   - -10.0 when tower collapses below previous max
//
// This is exactly the "dense reward based on structural match percentage and
// stress reduction" mentioned in Sprint 9 of the project doc.
//
// Parameters:
//   world:       the live Bevy ECS world (mutable reference so queries may borrow)
//   state:       mutable episode state (updated by this call)
//   time_step:   fraction of max_steps elapsed (0.0 → 1.0) for time penalty
//
// Returns: (reward: f32, terminated: bool, info: RewardInfo)
//   terminated = true when the episode should end (collapse or goal achieved)
pub fn calculate_reward(
    world: &mut World,
    state: &mut EpisodeState,
    _time_step_fraction: f32,
) -> (f32, bool, RewardInfo) {
    let mut query_state = world.query::<&Voxel>();
    let mut reward = 0.0f32;
    let mut current_max_y = -100.0f32;
    let mut block_count = 0u32;
    let mut total_stress_approx = 0.0f32;
    let mut terminated = false;

    for voxel in query_state.iter(world) {
        if voxel.inv_mass == 0.0 {
            continue;
        } // Skip static
        block_count += 1;

        // Track the highest block position in the world
        if voxel.position.y > current_max_y {
            current_max_y = voxel.position.y;
        }

        // Approximate structural stress from contact_count.
        // LEARNING: contact_count > 0 means a block is being supported or loaded.
        // High contact counts on lower blocks indicate load transfer (good stability).
        // Zero contact count on a block that should be supported = free-floating (bad).
        // This is a proxy for the real stress heatmap in Sprint 4 (which will
        // compute actual compression forces through the structural graph).
        total_stress_approx += voxel.contact_count as f32;
    }

    if block_count == 0 {
        // Edge case: world is empty, no reward
        return (0.0, false, RewardInfo::default());
    }

    // =========================================================================
    // REWARD COMPONENT 1: Height progress
    // ----------------------------------------
    // Reward = current_max_height (normalized to block units)
    // This is always positive and grows as the tower grows.
    // Small multiplier (0.1) to keep rewards in a well-scaled range for PPO.
    //
    // LEARNING: Reward scaling matters enormously for neural network training.
    // If rewards are too large (e.g., 1000 for a tall tower), gradient updates
    // become huge and training diverges. If too small, learning is slow.
    // PPO works best with rewards in the range [-1, +1] per step.
    // We use 0.1 × height so a 10-block tower gives +1.0 reward/step (ideal).
    // =========================================================================
    let height_reward = current_max_y.max(0.0) * 0.1;
    reward += height_reward;

    // =========================================================================
    // REWARD COMPONENT 2: New height record bonus
    // -------------------------------------------
    // Extra +1.0 when the agent achieves a new maximum height.
    // This is the "milestone" reward that prevents the agent from being content
    // to maintain an existing height rather than building higher.
    // =========================================================================
    if current_max_y > state.max_height_achieved {
        reward += 1.0;
        state.max_height_achieved = current_max_y;
    }

    // =========================================================================
    // REWARD COMPONENT 3: Collapse penalty
    // --------------------------------------
    // If current_max_y dropped significantly from the achieved max, the tower
    // collapsed. Give a strong negative reward to teach the agent that collapse
    // is very bad, then terminate the episode.
    //
    // Threshold: 2.0 blocks drop = collapse (not just settling/bouncing).
    // =========================================================================
    if current_max_y < state.max_height_achieved - 2.0 {
        reward -= 10.0;
        terminated = true; // End episode immediately on collapse
    }

    // =========================================================================
    // REWARD COMPONENT 4: Stability bonus (anti-noodle-tower)
    // --------------------------------------------------------
    // From the project docs:
    //   "Reward = Height × (1 - Total_Stress / Max_Strength)"
    //
    // We approximate this as:
    //   stability_ratio = avg_contact_count / max_possible_contacts
    //   stability_bonus = height_reward × stability_ratio
    //
    // Higher contact counts = more structural connections = more stable.
    // A single-block-wide "noodle tower" has 1 contact per block.
    // A wide pyramid base has 3-4 contacts per lower block.
    //
    // This bonus naturally favors wide, stable bases over unstable pillars.
    // =========================================================================
    let avg_contacts = total_stress_approx / block_count as f32;
    let stability_ratio = (avg_contacts / 4.0).min(1.0); // 4 contacts = fully stable
    reward *= 0.5 + 0.5 * stability_ratio; // Scale height reward by stability

    // =========================================================================
    // REWARD COMPONENT 5: Time penalty
    // ----------------------------------
    // -0.001 per step encourages the agent to build efficiently.
    // Without this, the agent might "wait" (do nothing) and still get
    // height rewards from previous blocks remaining in place.
    // =========================================================================
    reward -= 0.001;

    state.step_count += 1;
    state.prev_total_stress = total_stress_approx;

    // Truncate episode if time limit reached (not a collapse — no penalty)
    let truncated = state.step_count >= state.max_steps;

    let info = RewardInfo {
        height_reward,
        stability_ratio,
        block_count,
        current_max_height: current_max_y,
        total_reward: reward,
    };

    (reward, terminated || truncated, info)
}

// =============================================================================
// RewardInfo: diagnostic breakdown of the reward components
// =============================================================================
//
// LEARNING: Returning only the scalar reward hides information that's useful
// for debugging training. By returning a struct with all reward components,
// you can plot them separately in TensorBoard/WandB and understand WHICH
// signal the agent is responding to. Critical for reward shaping iteration.
#[derive(Debug, Default)]
pub struct RewardInfo {
    pub height_reward: f32,
    pub stability_ratio: f32,
    pub block_count: u32,
    pub current_max_height: f32,
    pub total_reward: f32,
}

// =============================================================================
// ActionParser: converts flat action arrays from RL policy → engine commands
// =============================================================================
//
// LEARNING TOPIC: Action Spaces in Construction RL
// ------------------------------------------------
// The RL agent's "action" at each step is a vector of floats (continuous) or
// integers (discrete). For construction tasks, common choices are:
//
// DISCRETE: agent chooses from N pre-defined actions
//   [0: place cube at +X, 1: place cube at -X, 2: place cube at +Z, ...]
//   Simple. Limited expressiveness (can't place diagonally, etc.)
//
// CONTINUOUS: agent outputs (dx, dz, shape_id) as raw floats
//   dx ∈ [-1, 1]: normalized X offset from current spawn point
//   dz ∈ [-1, 1]: normalized Z offset from current spawn point
//   shape_id ∈ [0, 3]: rounded to nearest int for cube/wedge/sphere
//   Flexible. Harder to train (larger action space).
//
// We expose a continuous action parser here, matching the project's goal of
// training a "Universal Builder" that can place any block anywhere.
//
// The Python Gymnasium wrapper provides:
//   action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
//   action = policy.predict(observation)   # → [dx, dz, shape_type]
//   engine.apply_action(action)            # → parsed block spawn
pub struct ParsedAction {
    pub world_x: f32,
    pub world_z: f32,
    pub shape_id: u8, // 0=Cube, 1=Wedge, 2=Sphere
    pub spawn_height: f32,
}

/// Parse a 3-element continuous action vector from the RL policy.
///
/// `action_vec`:  [dx_normalized, dz_normalized, shape_continuous]
/// `world_center`: the X/Z position the agent is currently building around
/// `spawn_radius`: the maximum block placement distance from center (in world units)
/// `spawn_height`: pre-computed drop height (from column max Y lookup)
pub fn parse_action(
    action_vec: &[f32; 3],
    world_center: (f32, f32),
    spawn_radius: f32,
    spawn_height: f32,
) -> ParsedAction {
    // Denormalize: action[-1, 1] → world_units centered at world_center
    let world_x = world_center.0 + action_vec[0] * spawn_radius;
    let world_z = world_center.1 + action_vec[1] * spawn_radius;

    // Discretize the continuous shape selection:
    // action_vec[2] ∈ [-1, 1] → shape_id ∈ {0, 1, 2}
    // Map: [-1, -0.33) → 0 (Cube), [-0.33, 0.33) → 1 (Wedge), [0.33, 1] → 2 (Sphere)
    let shape_id = if action_vec[2] < -0.33 {
        0u8 // Cube: the workhorse, most placed
    } else if action_vec[2] < 0.33 {
        1u8 // Wedge: for structural angles
    } else {
        2u8 // Sphere: for decorative / arch keystone
    };

    ParsedAction {
        world_x: world_x.round(), // Snap to grid (voxel world = integer positions)
        world_z: world_z.round(),
        shape_id,
        spawn_height,
    }
}

// =============================================================================
// LEARNING: Why is this file in bridge/ rather than lib.rs?
// =============================================================================
// Separation of concerns: lib.rs orchestrates the engine lifecycle (init,
// step, render, event handling). gym_env.rs defines the RL-specific layer
// (reward functions, action parsing, episode management).
//
// In Sprint 7, gym_env.rs will grow to include:
//   - Parallel environment management (multiple World instances, one per CPU core)
//   - Zero-copy shared memory tensors using PyTorch's CUDA IPC mechanism
//   - The Farama Gymnasium VectorEnv interface for batched training
//
// Keeping it separate from lib.rs makes those expansions easier to isolate,
// test, and document without tangling with the rendering/event loop code.
