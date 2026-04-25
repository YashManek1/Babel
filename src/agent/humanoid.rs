// =============================================================================
// src/agent/humanoid.rs  —  Operation Babel: Box-Model Humanoid Rig
// =============================================================================
//
// LEARNING TOPIC: Box-Model "Worker" Agent
// ----------------------------------------
// The humanoid is built from 9 rectangular voxel blocks connected by 8 XPBD joints.
// This "box-model" approach (used by early MuJoCo humanoids and Minecraft-style agents)
// trades anatomical accuracy for simulation speed and training stability.
//
// SEGMENT DIMENSIONS (in world units, where 1 unit = 1 voxel block):
//
//   Torso:       0.6 × 1.0 × 0.4  (wide, tall, shallow)
//   Thigh:       0.3 × 0.5 × 0.3  (cylinder-like box)
//   Shin:        0.25 × 0.5 × 0.25
//   Upper Arm:   0.2 × 0.45 × 0.2
//   Forearm:     0.18 × 0.4 × 0.18
//
// JOINT CONFIGURATION (8 joints):
//
//   Left Hip:       Ball Socket, 75° cone, torso ↔ left thigh
//   Right Hip:      Ball Socket, 75° cone, torso ↔ right thigh
//   Left Knee:      Hinge, 0°–120°, left thigh ↔ left shin
//   Right Knee:     Hinge, 0°–120°, right thigh ↔ right shin
//   Left Shoulder:  Ball Socket, 90° cone, torso ↔ left upper arm
//   Right Shoulder: Ball Socket, 90° cone, torso ↔ right upper arm
//   Left Elbow:     Hinge, 0°–145°, left upper arm ↔ left forearm
//   Right Elbow:    Hinge, 0°–145°, right upper arm ↔ right forearm
//
// COORDINATE SYSTEM:
//   +Y = up (same as world)
//   +X = to the agent's right (in its local frame)
//   +Z = forward (direction agent faces at spawn)
//
// LEARNING TOPIC: Spawn Position Strategy
// -----------------------------------------
// The torso spawns at (x, y + 1.5, z) — raised 1.5 units above the target ground
// position. The leg segments are pre-positioned below the torso. All segments are
// given zero initial velocity and land together under gravity. The XPBD joint
// constraints pull them into anatomical configuration within ~30 physics steps.
//
// This "drop-and-settle" approach is much simpler than analytically computing the
// exact anatomical configuration in world space — the physics engine does the work.
// =============================================================================

use crate::agent::joints::{
    BodySegment, JOINT_OBSERVATION_STRIDE, JointBodyProperties, JointConstraint,
};
use crate::agent::sensors::HumanoidSensors;
use crate::world::voxel::{MaterialType, Voxel};
use bevy_ecs::prelude::*;
use glam::Vec3;
use std::collections::HashMap;

// =============================================================================
// HUMANOID PHYSICS CONSTANTS
// =============================================================================

/// Torso segment half-extents (world units).
/// Full size: 0.6 × 1.0 × 0.4.
const TORSO_HALF_EXTENTS: Vec3 = Vec3::new(0.3, 0.5, 0.2);

/// Thigh segment half-extents.
const THIGH_HALF_EXTENTS: Vec3 = Vec3::new(0.15, 0.25, 0.15);

/// Shin segment half-extents.
const SHIN_HALF_EXTENTS: Vec3 = Vec3::new(0.125, 0.25, 0.125);

/// Upper arm segment half-extents.
const UPPER_ARM_HALF_EXTENTS: Vec3 = Vec3::new(0.10, 0.225, 0.10);

/// Forearm segment half-extents.
const FOREARM_HALF_EXTENTS: Vec3 = Vec3::new(0.09, 0.20, 0.09);

/// Global floor plane used by the XPBD solver.
///
/// Normal 1x1 blocks sit with their center at y=0.0 and bottom at y=-0.5.
/// The humanoid uses the same floor so the shins begin in contact instead of
/// dropping from the air and shocking the joint solver.
const HUMANOID_FLOOR_Y: f32 = -0.5;

/// Hip joint ball socket cone limit (radians). ~75 degrees.
const HIP_CONE_LIMIT_RADIANS: f32 = 1.309; // 75°

/// Shoulder joint ball socket cone limit (radians). ~90 degrees.
const SHOULDER_CONE_LIMIT_RADIANS: f32 = 1.571; // 90°

/// Knee minimum angle (radians). Can't hyperextend backward.
const KNEE_MIN_ANGLE_RADIANS: f32 = -0.05; // ~-3° for a tiny bit of backward flex

/// Knee maximum angle (radians). ~120 degrees of flex.
const KNEE_MAX_ANGLE_RADIANS: f32 = 2.094; // 120°

/// Elbow minimum angle (radians).
const ELBOW_MIN_ANGLE_RADIANS: f32 = 0.0;

/// Elbow maximum angle (radians). ~145 degrees of flex.
const ELBOW_MAX_ANGLE_RADIANS: f32 = 2.531; // 145°

/// Inverse mass for torso (mass ≈ 15 kg for a construction worker).
const TORSO_INV_MASS: f32 = 1.0 / 15.0;

/// Inverse mass for thigh segment (mass ≈ 8 kg each).
const THIGH_INV_MASS: f32 = 1.0 / 8.0;

/// Inverse mass for shin segment (mass ≈ 5 kg each).
const SHIN_INV_MASS: f32 = 1.0 / 5.0;

/// Inverse mass for upper arm (mass ≈ 3 kg each).
const UPPER_ARM_INV_MASS: f32 = 1.0 / 3.0;

/// Inverse mass for forearm (mass ≈ 2 kg each).
const FOREARM_INV_MASS: f32 = 1.0 / 2.0;

/// High-level movement speed for GUI/manual Worker locomotion.
const HUMANOID_MANUAL_MOVE_SPEED: f32 = 1.8;

// =============================================================================
// HumanoidRig Component — Root entity of the humanoid, tracks all segment entities
// =============================================================================
//
// LEARNING: We use a "root entity" pattern where the torso entity carries the
// HumanoidRig component as an index into all other segments. This makes it easy to:
//   - Despawn the entire agent (iterate segment_entities and despawn each)
//   - Find all agents via `Query<&HumanoidRig>`
//   - Look up any segment by index without searching the entire world
//
// Alternative: a central HumanoidRegistry resource. We use the component approach
// because it integrates naturally with Bevy's entity system — despawning the torso
// entity automatically removes the HumanoidRig component, and we can catch that
// to trigger cleanup of all other segments.
#[derive(Component, Clone, Debug)]
pub struct HumanoidRig {
    /// Unique identifier for this agent (assigned at spawn, monotonically increasing).
    pub agent_id: u32,

    /// Entity for each named segment. Indexed by BodySegment::index().
    /// LEARNING: Fixed-size array of OPTIONS because segments may not all exist
    /// simultaneously (future: partial humanoid rigs for damaged agents).
    pub segment_entities: [Option<Entity>; BodySegment::COUNT],

    /// World position of the center of mass (updated each frame by sensors system).
    pub center_of_mass: Vec3,

    /// World velocity of the center of mass (derived from segment velocities).
    pub center_of_mass_velocity: Vec3,

    /// Balance metric: 0.0 = perfectly balanced, 1.0 = about to fall.
    pub balance_error: f32,

    /// Current facing direction (unit vector in XZ plane).
    pub facing_direction: Vec3,

    /// Whether this agent is currently alive and simulated.
    pub is_alive: bool,
}

impl HumanoidRig {
    /// Create a new humanoid rig with all segment entities initially None.
    pub fn new(agent_id: u32) -> Self {
        Self {
            agent_id,
            segment_entities: [None; BodySegment::COUNT],
            center_of_mass: Vec3::ZERO,
            center_of_mass_velocity: Vec3::ZERO,
            balance_error: 0.0,
            facing_direction: Vec3::Z,
            is_alive: true,
        }
    }

    /// Get the entity for a specific body segment, if it exists.
    pub fn segment_entity(&self, segment: BodySegment) -> Option<Entity> {
        self.segment_entities[segment.index()]
    }

    /// Set the entity for a specific body segment.
    pub fn set_segment_entity(&mut self, segment: BodySegment, entity: Entity) {
        self.segment_entities[segment.index()] = Some(entity);
    }
}

// =============================================================================
// HumanoidRegistry Resource — Global registry mapping agent_id → torso entity
// =============================================================================
//
// LEARNING: The registry allows O(1) lookup from Python's agent_id (a u32)
// to the Bevy Entity that holds the HumanoidRig component.
//
// Without this, Python would have to pass a full entity ID (which is not stable
// across save/load and is ugly in Python APIs). The agent_id is a clean
// monotonically increasing integer that the Python RL agent can use directly.
#[derive(Resource, Default)]
pub struct HumanoidRegistry {
    /// Map from agent_id to the torso entity (which holds HumanoidRig).
    pub torso_entity_by_agent_id: HashMap<u32, Entity>,

    /// Next agent_id to assign on spawn.
    next_agent_id: u32,
}

impl HumanoidRegistry {
    /// Assign a new unique agent ID and register the torso entity.
    pub fn register_new_agent(&mut self, torso_entity: Entity) -> u32 {
        let agent_id = self.next_agent_id;
        self.next_agent_id += 1;
        self.torso_entity_by_agent_id.insert(agent_id, torso_entity);
        agent_id
    }

    /// Remove an agent from the registry when despawned.
    pub fn unregister_agent(&mut self, agent_id: u32) {
        self.torso_entity_by_agent_id.remove(&agent_id);
    }

    /// Look up the torso entity for a given agent_id.
    pub fn torso_entity(&self, agent_id: u32) -> Option<Entity> {
        self.torso_entity_by_agent_id.get(&agent_id).copied()
    }

    /// Total number of active agents.
    pub fn agent_count(&self) -> usize {
        self.torso_entity_by_agent_id.len()
    }
}

// =============================================================================
// SpawnHumanoidParams — Parameters for spawning a new humanoid agent
// =============================================================================
//
// LEARNING: Using a dedicated struct for spawn parameters instead of a long
// argument list follows the "Builder" pattern without the boilerplate of a
// full builder. It's also more amenable to serialization for replay recording
// (Sprint 9: Replay System).
#[derive(Clone, Debug)]
pub struct SpawnHumanoidParams {
    /// World position of the agent's feet (the humanoid spawns above this).
    pub ground_position: Vec3,

    /// Initial facing direction (unit vector in XZ plane).
    pub facing_direction: Vec3,
}

impl SpawnHumanoidParams {
    pub fn new(ground_x: f32, ground_z: f32) -> Self {
        Self {
            ground_position: Vec3::new(ground_x, 0.0, ground_z),
            facing_direction: Vec3::Z,
        }
    }
}

// =============================================================================
// FUNCTION: spawn_humanoid
// =============================================================================
//
// LEARNING TOPIC: Anatomical Segment Placement
// ---------------------------------------------
// We compute each segment's initial world position relative to the ground point.
// The convention is Y = 0 at ground level, +Y upward.
//
// Assembly from ground up:
//   Shin bottom at Y = 0.0 (feet on ground)
//   Shin center at Y = 0.25
//   Thigh center at Y = 0.75  (shin top at 0.5 + thigh half = 0.25)
//   Torso center at Y = 1.5   (thigh top at 1.0 + torso half = 0.5)
//   Upper arm at Y = 1.5 (shoulder height, same as torso center)
//   Forearm at Y = 1.0   (hanging down from upper arm)
//
// Segments start in their anatomical rest pose with the shin bottoms exactly on
// the floor plane. This avoids the old "drop-and-settle" shock that launched
// the rig before the joint solver had a chance to stabilize it.
pub fn spawn_humanoid(ecs_world: &mut World, params: SpawnHumanoidParams) -> u32 {
    let ground = find_clear_humanoid_spawn_ground(ecs_world, params.ground_position);

    // ── Helper closure: create a Voxel with humanoid-specific physics ─────────
    //
    // LEARNING: Humanoid segments use a different material profile than voxel blocks.
    // We use MaterialType::Wood as the base (inv_mass ≈ 1.0) and override inv_mass
    // directly after creation. This reuses all of Wood's friction/restitution
    // values which are appropriate for a soft-body humanoid agent.
    let make_segment_voxel =
        |position: Vec3, half_extents: Vec3, override_inv_mass: f32| -> Voxel {
            let mut voxel = Voxel::new_box_with_material(
                position.x,
                position.y,
                position.z,
                half_extents,
                MaterialType::Wood,
                false, // dynamic
            );
            voxel.friction = 0.95;
            voxel.restitution = 0.0;
            voxel.inv_mass = override_inv_mass;
            voxel
        };

    // ── Compute world-space positions for each segment ──────────────────────
    //
    // LEARNING: These Y offsets match the segment dimensions defined above.
    // The torso center is at height 1.0 above the default block floor.
    // All positions are relative to `ground` for easy repositioning.
    let floor_y = ground.y + HUMANOID_FLOOR_Y;
    let torso_position = Vec3::new(ground.x, floor_y + 1.5, ground.z);

    // Legs: symmetric about X=0 and exactly stacked from the floor up.
    let hip_x = 0.21;
    let left_thigh_position = Vec3::new(ground.x - hip_x, floor_y + 0.75, ground.z);
    let left_shin_position = Vec3::new(ground.x - hip_x, floor_y + 0.25, ground.z);
    let right_thigh_position = Vec3::new(ground.x + hip_x, floor_y + 0.75, ground.z);
    let right_shin_position = Vec3::new(ground.x + hip_x, floor_y + 0.25, ground.z);

    // Arms: hang close to the torso with side anchors at the shoulders.
    let arm_x = TORSO_HALF_EXTENTS.x + UPPER_ARM_HALF_EXTENTS.x + 0.02;
    let shoulder_height = 1.30;
    let left_upper_arm_position = Vec3::new(
        ground.x - arm_x,
        floor_y + shoulder_height - UPPER_ARM_HALF_EXTENTS.y,
        ground.z,
    );
    let left_forearm_position = Vec3::new(
        ground.x - arm_x,
        floor_y + shoulder_height - UPPER_ARM_HALF_EXTENTS.y * 2.0 - FOREARM_HALF_EXTENTS.y,
        ground.z,
    );
    let right_upper_arm_position = Vec3::new(
        ground.x + arm_x,
        floor_y + shoulder_height - UPPER_ARM_HALF_EXTENTS.y,
        ground.z,
    );
    let right_forearm_position = Vec3::new(
        ground.x + arm_x,
        floor_y + shoulder_height - UPPER_ARM_HALF_EXTENTS.y * 2.0 - FOREARM_HALF_EXTENTS.y,
        ground.z,
    );

    // ── Spawn all segment entities ───────────────────────────────────────────
    //
    // LEARNING: We spawn each segment as a Voxel (so existing collision and
    // render systems handle them automatically) and add the humanoid-specific
    // components alongside.

    let torso_entity = ecs_world
        .spawn((
            make_segment_voxel(torso_position, TORSO_HALF_EXTENTS, TORSO_INV_MASS),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::Torso,
                inv_inertia: TORSO_INV_MASS * 6.0, // 6.0 = inertia-to-mass ratio for box
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
        ))
        .id();

    let left_thigh_entity = ecs_world
        .spawn((
            make_segment_voxel(left_thigh_position, THIGH_HALF_EXTENTS, THIGH_INV_MASS),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::LeftThigh,
                inv_inertia: THIGH_INV_MASS * 6.0,
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
        ))
        .id();

    let left_shin_entity = ecs_world
        .spawn((
            make_segment_voxel(left_shin_position, SHIN_HALF_EXTENTS, SHIN_INV_MASS),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::LeftShin,
                inv_inertia: SHIN_INV_MASS * 6.0,
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
            JointConstraint::new_hinge(
                left_thigh_entity,
                Vec3::new(0.0, -THIGH_HALF_EXTENTS.y, 0.0), // bottom of thigh
                Vec3::new(0.0, SHIN_HALF_EXTENTS.y, 0.0),   // top of shin
                Vec3::X,                                    // hinge axis: left-right
                KNEE_MIN_ANGLE_RADIANS,
                KNEE_MAX_ANGLE_RADIANS,
            ),
        ))
        .id();

    let right_thigh_entity = ecs_world
        .spawn((
            make_segment_voxel(right_thigh_position, THIGH_HALF_EXTENTS, THIGH_INV_MASS),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::RightThigh,
                inv_inertia: THIGH_INV_MASS * 6.0,
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
        ))
        .id();

    let right_shin_entity = ecs_world
        .spawn((
            make_segment_voxel(right_shin_position, SHIN_HALF_EXTENTS, SHIN_INV_MASS),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::RightShin,
                inv_inertia: SHIN_INV_MASS * 6.0,
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
            JointConstraint::new_hinge(
                right_thigh_entity,
                Vec3::new(0.0, -THIGH_HALF_EXTENTS.y, 0.0),
                Vec3::new(0.0, SHIN_HALF_EXTENTS.y, 0.0),
                Vec3::X,
                KNEE_MIN_ANGLE_RADIANS,
                KNEE_MAX_ANGLE_RADIANS,
            ),
        ))
        .id();

    let left_upper_arm_entity = ecs_world
        .spawn((
            make_segment_voxel(
                left_upper_arm_position,
                UPPER_ARM_HALF_EXTENTS,
                UPPER_ARM_INV_MASS,
            ),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::LeftUpperArm,
                inv_inertia: UPPER_ARM_INV_MASS * 6.0,
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
        ))
        .id();

    let left_forearm_entity = ecs_world
        .spawn((
            make_segment_voxel(
                left_forearm_position,
                FOREARM_HALF_EXTENTS,
                FOREARM_INV_MASS,
            ),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::LeftForearm,
                inv_inertia: FOREARM_INV_MASS * 6.0,
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
            JointConstraint::new_hinge(
                left_upper_arm_entity,
                Vec3::new(0.0, -UPPER_ARM_HALF_EXTENTS.y, 0.0),
                Vec3::new(0.0, FOREARM_HALF_EXTENTS.y, 0.0),
                Vec3::X,
                ELBOW_MIN_ANGLE_RADIANS,
                ELBOW_MAX_ANGLE_RADIANS,
            ),
        ))
        .id();

    let right_upper_arm_entity = ecs_world
        .spawn((
            make_segment_voxel(
                right_upper_arm_position,
                UPPER_ARM_HALF_EXTENTS,
                UPPER_ARM_INV_MASS,
            ),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::RightUpperArm,
                inv_inertia: UPPER_ARM_INV_MASS * 6.0,
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
        ))
        .id();

    let right_forearm_entity = ecs_world
        .spawn((
            make_segment_voxel(
                right_forearm_position,
                FOREARM_HALF_EXTENTS,
                FOREARM_INV_MASS,
            ),
            JointBodyProperties {
                agent_id: 0,
                segment: BodySegment::RightForearm,
                inv_inertia: FOREARM_INV_MASS * 6.0,
                applied_torque: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
            JointConstraint::new_hinge(
                right_upper_arm_entity,
                Vec3::new(0.0, -UPPER_ARM_HALF_EXTENTS.y, 0.0),
                Vec3::new(0.0, FOREARM_HALF_EXTENTS.y, 0.0),
                Vec3::X,
                ELBOW_MIN_ANGLE_RADIANS,
                ELBOW_MAX_ANGLE_RADIANS,
            ),
        ))
        .id();

    // ── Add Ball-Socket (hip and shoulder) joints to thigh and upper arm ────
    //
    // LEARNING: Ball socket joints must be added AFTER spawning their parent entities
    // (torso) so we have the parent entity IDs. We use Bevy's world.entity_mut().insert()
    // to add the JointConstraint component to already-spawned entities.

    // Left hip: connects left thigh to torso
    ecs_world
        .entity_mut(left_thigh_entity)
        .insert(JointConstraint::new_ball_socket(
            torso_entity,
            Vec3::new(-TORSO_HALF_EXTENTS.x * 0.7, -TORSO_HALF_EXTENTS.y, 0.0), // bottom-left torso
            Vec3::new(0.0, THIGH_HALF_EXTENTS.y, 0.0),                          // top of thigh
            HIP_CONE_LIMIT_RADIANS,
        ));

    // Right hip: connects right thigh to torso
    ecs_world
        .entity_mut(right_thigh_entity)
        .insert(JointConstraint::new_ball_socket(
            torso_entity,
            Vec3::new(TORSO_HALF_EXTENTS.x * 0.7, -TORSO_HALF_EXTENTS.y, 0.0),
            Vec3::new(0.0, THIGH_HALF_EXTENTS.y, 0.0),
            HIP_CONE_LIMIT_RADIANS,
        ));

    // Left shoulder: connects left upper arm to torso
    ecs_world
        .entity_mut(left_upper_arm_entity)
        .insert(JointConstraint::new_ball_socket(
            torso_entity,
            Vec3::new(-TORSO_HALF_EXTENTS.x, 0.3, 0.0), // upper-left side of torso
            Vec3::new(UPPER_ARM_HALF_EXTENTS.x, UPPER_ARM_HALF_EXTENTS.y, 0.0),
            SHOULDER_CONE_LIMIT_RADIANS,
        ));

    // Right shoulder: connects right upper arm to torso
    ecs_world
        .entity_mut(right_upper_arm_entity)
        .insert(JointConstraint::new_ball_socket(
            torso_entity,
            Vec3::new(TORSO_HALF_EXTENTS.x, 0.3, 0.0),
            Vec3::new(-UPPER_ARM_HALF_EXTENTS.x, UPPER_ARM_HALF_EXTENTS.y, 0.0),
            SHOULDER_CONE_LIMIT_RADIANS,
        ));

    // ── Build the HumanoidRig and attach to the torso entity ─────────────────
    let mut rig = HumanoidRig::new(0); // agent_id assigned below after registration
    rig.set_segment_entity(BodySegment::Torso, torso_entity);
    rig.set_segment_entity(BodySegment::LeftThigh, left_thigh_entity);
    rig.set_segment_entity(BodySegment::LeftShin, left_shin_entity);
    rig.set_segment_entity(BodySegment::RightThigh, right_thigh_entity);
    rig.set_segment_entity(BodySegment::RightShin, right_shin_entity);
    rig.set_segment_entity(BodySegment::LeftUpperArm, left_upper_arm_entity);
    rig.set_segment_entity(BodySegment::LeftForearm, left_forearm_entity);
    rig.set_segment_entity(BodySegment::RightUpperArm, right_upper_arm_entity);
    rig.set_segment_entity(BodySegment::RightForearm, right_forearm_entity);

    // Attach HumanoidRig and HumanoidSensors to the torso entity.
    ecs_world
        .entity_mut(torso_entity)
        .insert((rig.clone(), HumanoidSensors::default()));

    // Register with the global registry and get the agent_id.
    let mut registry = ecs_world.remove_resource::<HumanoidRegistry>().unwrap();
    let agent_id = registry.register_new_agent(torso_entity);
    ecs_world.insert_resource(registry);

    // Update the rig's agent_id now that we have it.
    ecs_world
        .entity_mut(torso_entity)
        .get_mut::<HumanoidRig>()
        .unwrap()
        .agent_id = agent_id;

    for segment_entity in [
        torso_entity,
        left_thigh_entity,
        left_shin_entity,
        right_thigh_entity,
        right_shin_entity,
        left_upper_arm_entity,
        left_forearm_entity,
        right_upper_arm_entity,
        right_forearm_entity,
    ] {
        if let Some(mut joint_body) = ecs_world.get_mut::<JointBodyProperties>(segment_entity) {
            joint_body.agent_id = agent_id;
        }
    }

    agent_id
}

fn find_clear_humanoid_spawn_ground(ecs_world: &World, requested_ground: Vec3) -> Vec3 {
    let mut occupied_xz = Vec::new();
    if let Some(registry) = ecs_world.get_resource::<HumanoidRegistry>() {
        for torso_entity in registry.torso_entity_by_agent_id.values().copied() {
            if let Some(torso_voxel) = ecs_world.get::<Voxel>(torso_entity) {
                occupied_xz.push(Vec3::new(
                    torso_voxel.position.x,
                    0.0,
                    torso_voxel.position.z,
                ));
            }
        }
    }

    let candidate_offsets = [
        Vec3::ZERO,
        Vec3::new(1.4, 0.0, 0.0),
        Vec3::new(-1.4, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.4),
        Vec3::new(0.0, 0.0, -1.4),
        Vec3::new(1.4, 0.0, 1.4),
        Vec3::new(-1.4, 0.0, 1.4),
        Vec3::new(1.4, 0.0, -1.4),
        Vec3::new(-1.4, 0.0, -1.4),
        Vec3::new(2.8, 0.0, 0.0),
        Vec3::new(-2.8, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 2.8),
        Vec3::new(0.0, 0.0, -2.8),
    ];

    for offset in candidate_offsets {
        let candidate = requested_ground + offset;
        let candidate_xz = Vec3::new(candidate.x, 0.0, candidate.z);
        if occupied_xz
            .iter()
            .all(|occupied| occupied.distance(candidate_xz) >= 1.15)
        {
            return candidate;
        }
    }

    requested_ground
        + Vec3::new(
            occupied_xz.len() as f32 * 1.4,
            0.0,
            (occupied_xz.len() % 2) as f32 * 1.4,
        )
}

pub fn move_humanoid(ecs_world: &mut World, agent_id: u32, desired_direction: Vec3) -> bool {
    let horizontal_direction = Vec3::new(desired_direction.x, 0.0, desired_direction.z);
    let Some(move_direction) = horizontal_direction.try_normalize() else {
        return false;
    };

    let segment_entities: Vec<Entity> = {
        let Some(registry) = ecs_world.get_resource::<HumanoidRegistry>() else {
            return false;
        };
        let Some(torso_entity) = registry.torso_entity(agent_id) else {
            return false;
        };
        let Some(rig) = ecs_world.get::<HumanoidRig>(torso_entity) else {
            return false;
        };
        rig.segment_entities.iter().flatten().copied().collect()
    };

    for segment_entity in segment_entities {
        if let Some(mut voxel) = ecs_world.get_mut::<Voxel>(segment_entity) {
            let y_velocity = voxel.velocity.y.min(0.0);
            voxel.velocity = move_direction * HUMANOID_MANUAL_MOVE_SPEED + Vec3::Y * y_velocity;
            voxel.is_sleeping = false;
        }
    }

    true
}

// =============================================================================
// FUNCTION: despawn_humanoid
// =============================================================================
//
// LEARNING: Despawning a humanoid requires:
//   1. Finding all segment entities via the HumanoidRig component on the torso
//   2. Removing all joint constraints (they reference now-deleted parent entities)
//   3. Despawning each segment entity
//   4. Unregistering from the HumanoidRegistry
//
// We do NOT call remove_bonds_for() because humanoid joints are stored as
// JointConstraint components (not in MortarBonds), so they're automatically
// despawned with their entities.
pub fn despawn_humanoid(ecs_world: &mut World, agent_id: u32) -> bool {
    // Look up the torso entity.
    let torso_entity = {
        let registry = ecs_world.get_resource::<HumanoidRegistry>().unwrap();
        match registry.torso_entity(agent_id) {
            Some(e) => e,
            None => return false, // Agent not found
        }
    };

    // Collect all segment entities from the rig.
    let segment_entities: Vec<Entity> = {
        let rig = match ecs_world.get::<HumanoidRig>(torso_entity) {
            Some(r) => r,
            None => return false, // Torso entity exists but has no rig (shouldn't happen)
        };
        rig.segment_entities.iter().flatten().copied().collect()
    };

    // Despawn all segments.
    for segment_entity in segment_entities {
        ecs_world.despawn(segment_entity);
    }

    // Unregister from the registry.
    let mut registry = ecs_world.remove_resource::<HumanoidRegistry>().unwrap();
    registry.unregister_agent(agent_id);
    ecs_world.insert_resource(registry);

    true
}

// =============================================================================
// FUNCTION: despawn_all_humanoids
// =============================================================================
//
// Called from `clear_dynamic_blocks()` and episode `reset()` in lib.rs.
// Returns the count of humanoids that were despawned.
pub fn despawn_all_humanoids(ecs_world: &mut World) -> usize {
    let agent_ids: Vec<u32> = {
        let registry = ecs_world.get_resource::<HumanoidRegistry>().unwrap();
        registry.torso_entity_by_agent_id.keys().copied().collect()
    };

    let count = agent_ids.len();
    for agent_id in agent_ids {
        despawn_humanoid(ecs_world, agent_id);
    }
    count
}

// =============================================================================
// FUNCTION: apply_torque_to_joint
// =============================================================================
//
// LEARNING TOPIC: Torque Application to XPBD Joints
// --------------------------------------------------
// The Python RL agent outputs a torque vector for each joint. Internally, we
// represent this as a "target angle" on the JointConstraint component.
//
// Conversion from torque → target angle:
//   δθ = torque_magnitude * DT * inv_inertia
//   new_target = current_target + δθ (for velocity-based control)
//   OR
//   new_target = torque_magnitude (for position-based control)
//
// We use POSITION-BASED CONTROL (direct angle setting) for Phase II RL training
// because it gives cleaner gradient signals to the neural network. The agent
// outputs desired joint angles, not forces. The XPBD solver handles the physics
// of how fast the joints reach those angles.
//
// VELOCITY-BASED CONTROL (torque accumulation) is more physically accurate and
// will be used in Phase III when the agent needs to push blocks.
//
// Arguments:
//   ecs_world:  mutable world reference
//   agent_id:   which agent to apply torque to
//   segment:    which body segment's joint to control
//   target_angle_radians: desired joint angle
//   swing_axis: for BallSocket joints, the direction of swing (normalized)
pub fn apply_torque_to_joint(
    ecs_world: &mut World,
    agent_id: u32,
    segment: BodySegment,
    target_angle_radians: f32,
    swing_axis: Vec3,
) -> bool {
    // Look up the segment entity.
    let segment_entity = {
        let registry = ecs_world.get_resource::<HumanoidRegistry>().unwrap();
        let torso_entity = match registry.torso_entity(agent_id) {
            Some(e) => e,
            None => return false,
        };
        let rig = match ecs_world.get::<HumanoidRig>(torso_entity) {
            Some(r) => r,
            None => return false,
        };
        match rig.segment_entity(segment) {
            Some(e) => e,
            None => return false,
        }
    };

    // Apply the target angle to the joint constraint on that segment.
    if let Some(mut joint_constraint) = ecs_world.get_mut::<JointConstraint>(segment_entity) {
        joint_constraint.target_angle_radians = target_angle_radians;
        joint_constraint.target_swing_axis = swing_axis.normalize_or_zero();
        true
    } else {
        false // No joint constraint on this segment (e.g., torso itself)
    }
}

// =============================================================================
// SYSTEM: update_humanoid_rigs_system
// =============================================================================
//
// LEARNING: This Bevy system runs every frame and updates the HumanoidRig's
// aggregate state (center of mass, facing direction, balance error) from
// the current positions and velocities of all segment entities.
//
// This system is SEPARATE from solve_joint_constraints_system because:
//   - It reads from segment voxels (shared read) rather than mutating them
//   - It writes only to the HumanoidRig component on the torso
//   - Running it after update_velocities_system ensures it sees committed positions
pub fn update_humanoid_rigs_system(
    mut rig_query: Query<(Entity, &mut HumanoidRig)>,
    voxel_query: Query<&Voxel>,
) {
    for (_torso_entity, mut rig) in rig_query.iter_mut() {
        if !rig.is_alive {
            continue;
        }

        let mut total_mass = 0.0f32;
        let mut weighted_position = Vec3::ZERO;
        let mut weighted_velocity = Vec3::ZERO;

        // Compute mass-weighted center of mass and velocity.
        for segment_entity_option in rig.segment_entities.iter().flatten() {
            let Ok(voxel) = voxel_query.get(*segment_entity_option) else {
                continue;
            };
            if voxel.inv_mass <= 0.0 {
                continue;
            }
            let mass = 1.0 / voxel.inv_mass;
            total_mass += mass;
            weighted_position += voxel.position * mass;
            weighted_velocity += voxel.velocity * mass;
        }

        if total_mass > 0.0 {
            rig.center_of_mass = weighted_position / total_mass;
            rig.center_of_mass_velocity = weighted_velocity / total_mass;
        }

        // Compute balance error: how far the COM is from directly above the
        // base of support (midpoint between the two shin positions).
        //
        // LEARNING: This is a simplified "support polygon" balance check.
        // A full inverted-pendulum model would compute the ZMP (Zero Moment Point).
        // For box-model walking, this approximation is sufficient for the RL reward.
        let left_shin_pos = rig
            .segment_entity(BodySegment::LeftShin)
            .and_then(|e| voxel_query.get(e).ok())
            .map(|v| v.position)
            .unwrap_or(rig.center_of_mass);

        let right_shin_pos = rig
            .segment_entity(BodySegment::RightShin)
            .and_then(|e| voxel_query.get(e).ok())
            .map(|v| v.position)
            .unwrap_or(rig.center_of_mass);

        let base_of_support_midpoint = (left_shin_pos + right_shin_pos) * 0.5;
        let horizontal_com_deviation = Vec3::new(
            rig.center_of_mass.x - base_of_support_midpoint.x,
            0.0, // only horizontal deviation matters
            rig.center_of_mass.z - base_of_support_midpoint.z,
        );

        // Normalize by a "fall threshold" distance (0.4 world units = topple threshold)
        let fall_threshold = 0.4f32;
        rig.balance_error = (horizontal_com_deviation.length() / fall_threshold).clamp(0.0, 1.0);

        // Update facing direction from the torso's velocity (if moving).
        let horizontal_velocity = Vec3::new(
            rig.center_of_mass_velocity.x,
            0.0,
            rig.center_of_mass_velocity.z,
        );
        if horizontal_velocity.length() > 0.1 {
            rig.facing_direction = horizontal_velocity.normalize();
        }
    }
}

// =============================================================================
// HELPER: collect_humanoid_observation
// =============================================================================
//
// LEARNING: Packs per-segment state into the flat float32 observation buffer.
// Called from lib.rs get_humanoid_observation_into() for the Python RL bridge.
//
// Buffer layout per segment (JOINT_OBSERVATION_STRIDE = 15 floats):
//   [0..2]  position (x, y, z)
//   [3..5]  velocity (x, y, z)
//   [6..9]  rotation quaternion (x, y, z, w)
//   [10..12] angular velocity (x, y, z)
//   [13]    joint angle error (radians, signed)
//   [14]    is_at_limit (1.0 = at limit, 0.0 = within range)
pub fn collect_humanoid_observation(
    ecs_world: &World,
    agent_id: u32,
    output_buffer: &mut [f32],
) -> usize {
    let registry = ecs_world.get_resource::<HumanoidRegistry>().unwrap();
    let torso_entity = match registry.torso_entity(agent_id) {
        Some(e) => e,
        None => return 0,
    };

    let rig = match ecs_world.get::<HumanoidRig>(torso_entity) {
        Some(r) => r,
        None => return 0,
    };

    let required_buffer_size = BodySegment::COUNT * JOINT_OBSERVATION_STRIDE;
    if output_buffer.len() < required_buffer_size {
        return 0;
    }

    let mut segments_written = 0usize;

    for (segment_index, segment_entity_option) in rig.segment_entities.iter().enumerate() {
        let write_offset = segment_index * JOINT_OBSERVATION_STRIDE;

        let Some(segment_entity) = segment_entity_option else {
            // Segment missing — write zeros for this slot.
            for slot in
                output_buffer[write_offset..write_offset + JOINT_OBSERVATION_STRIDE].iter_mut()
            {
                *slot = 0.0;
            }
            continue;
        };

        let Some(voxel) = ecs_world.get::<Voxel>(*segment_entity) else {
            for slot in
                output_buffer[write_offset..write_offset + JOINT_OBSERVATION_STRIDE].iter_mut()
            {
                *slot = 0.0;
            }
            continue;
        };

        // Position
        output_buffer[write_offset] = voxel.position.x;
        output_buffer[write_offset + 1] = voxel.position.y;
        output_buffer[write_offset + 2] = voxel.position.z;

        // Velocity
        output_buffer[write_offset + 3] = voxel.velocity.x;
        output_buffer[write_offset + 4] = voxel.velocity.y;
        output_buffer[write_offset + 5] = voxel.velocity.z;

        // Rotation quaternion
        output_buffer[write_offset + 6] = voxel.rotation.x;
        output_buffer[write_offset + 7] = voxel.rotation.y;
        output_buffer[write_offset + 8] = voxel.rotation.z;
        output_buffer[write_offset + 9] = voxel.rotation.w;

        // Angular velocity (from JointBodyProperties if present, else zero)
        let (angular_velocity, joint_angle_error, is_at_limit) =
            if let Some(joint_body) = ecs_world.get::<JointBodyProperties>(*segment_entity) {
                let joint_constraint = ecs_world.get::<JointConstraint>(*segment_entity);
                let (angle_error, at_limit) = joint_constraint
                    .map(|j| {
                        (
                            j.current_angle_error,
                            if j.is_at_limit { 1.0f32 } else { 0.0 },
                        )
                    })
                    .unwrap_or((0.0, 0.0));
                (joint_body.angular_velocity, angle_error, at_limit)
            } else {
                (Vec3::ZERO, 0.0, 0.0)
            };

        output_buffer[write_offset + 10] = angular_velocity.x;
        output_buffer[write_offset + 11] = angular_velocity.y;
        output_buffer[write_offset + 12] = angular_velocity.z;
        output_buffer[write_offset + 13] = joint_angle_error;
        output_buffer[write_offset + 14] = is_at_limit;

        segments_written += 1;
    }

    segments_written
}
