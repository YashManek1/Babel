// =============================================================================
// src/agent/sensors.rs  —  Operation Babel: Humanoid Sensory Systems
// =============================================================================
//
// LEARNING TOPIC: Sensory Systems in Embodied AI
// -----------------------------------------------
// The RL agent can only learn from what it can SENSE. We provide:
//
//   1. PROPRIOCEPTION (body-internal sensing):
//      - Joint angles and angular velocities (from JointConstraint.current_angle_error)
//      - Center of mass position and velocity (computed in humanoid.rs)
//      - Balance error (horizontal COM deviation from support polygon)
//      This mirrors biological proprioception — knowing where your limbs are
//      without visual feedback.
//
//   2. VESTIBULAR (orientation sensing):
//      - Torso orientation quaternion (from Voxel.rotation on the torso)
//      - Gravity direction relative to torso (always Vec3::NEG_Y in world space,
//        but expressed in torso-local space gives the agent "which way is up")
//      This mirrors the inner ear's vestibular system.
//
//   3. GROUND CONTACT (tactile sensing):
//      - Per-foot ground contact flag (left shin, right shin)
//      - Contact force magnitude (approximated from velocity dampening)
//      This allows the agent to detect when its feet are on the ground and
//      implement bipedal gait patterns.
//
// SPRINT 6 STUB: Lidar (64-ray distance sensing) is declared here but
// implemented in Sprint 6. The struct field and observation stride constant
// are defined now to ensure the observation space remains stable.
//
// LEARNING TOPIC: Why Separate Sensors from Joints?
// --------------------------------------------------
// Joints are ACTUATORS (they produce motion). Sensors are OBSERVERS (they read state).
// Keeping them separate follows the classic control theory separation:
//   Actuator → World → Sensor → Controller → Actuator (the control loop)
// This separation makes it easy to add new sensors without touching the joint solver.
// =============================================================================

use bevy_ecs::prelude::*;
use glam::Vec3;

// =============================================================================
// OBSERVATION CONSTANTS for sensor data
// =============================================================================

/// Number of floats contributed by sensors per humanoid agent.
/// Layout:
///   [0..2]   center_of_mass position (x, y, z)
///   [3..5]   center_of_mass velocity (x, y, z)
///   [6]      balance_error (scalar)
///   [7..9]   torso_up_vector in world space (x, y, z)
///   [10]     left_foot_contact (0.0 or 1.0)
///   [11]     right_foot_contact (0.0 or 1.0)
///   [12]     left_foot_contact_force (approximated, world units/sec)
///   [13]     right_foot_contact_force (approximated)
///   [14..17] torso_rotation quaternion (x, y, z, w)
/// Total: 18 floats.
pub const SENSOR_OBSERVATION_STRIDE: usize = 18;

/// Floor contact detection epsilon — the same value used in xpbd.rs.
/// A foot is considered in ground contact if its lowest point is within this
/// distance of the floor plane (y = -0.5).
const GROUND_CONTACT_EPSILON: f32 = 0.05;

/// Floor Y position — must match PhysicsSettings::global_floor_y in xpbd.rs.
const FLOOR_Y: f32 = -0.5;

// =============================================================================
// HumanoidSensors Component — Attached to the torso entity alongside HumanoidRig
// =============================================================================
//
// LEARNING: We attach sensors to the torso (root) entity because:
//   - The torso is always present (never despawned before other segments)
//   - Sensor data is aggregate (about the whole body), not per-segment
//   - The observation buffer for sensors has a fixed layout starting from torso
//
// Updated each frame by update_humanoid_sensors_system AFTER update_velocities_system
// commits all final positions.
#[derive(Component, Clone, Debug)]
pub struct HumanoidSensors {
    // ── Proprioception ────────────────────────────────────────────────────────
    /// World-space torso "up" vector.
    /// When upright: (0, 1, 0). When tilted: rotated accordingly.
    /// Derived from torso Voxel::rotation applied to Vec3::Y.
    pub torso_up_vector: Vec3,

    /// World-space torso "forward" vector.
    /// When facing +Z: (0, 0, 1). Derived from torso rotation applied to Vec3::Z.
    pub torso_forward_vector: Vec3,

    // ── Ground Contact (Tactile Sensing) ─────────────────────────────────────
    /// Whether the left foot (left shin) is currently in contact with the ground.
    pub left_foot_contact: bool,

    /// Whether the right foot (right shin) is currently in contact with the ground.
    pub right_foot_contact: bool,

    /// Approximate contact force magnitude for the left foot.
    /// Estimated from the Y-velocity change on the left shin (impulse proxy).
    pub left_foot_contact_force: f32,

    /// Approximate contact force magnitude for the right foot.
    pub right_foot_contact_force: f32,

    // ── Vestibular (Orientation Sensing) ─────────────────────────────────────
    /// How many degrees the torso is tilted from upright.
    /// 0.0 = perfectly upright. 90.0 = lying on side. 180.0 = upside down.
    pub tilt_angle_degrees: f32,

    // ── Sprint 6 Placeholder: Lidar ──────────────────────────────────────────
    //
    // LEARNING: We declare this field now so the HumanoidSensors component has
    // a stable memory layout that Python can depend on. In Sprint 6, this will
    // be filled with 64 ray-cast distances.
    // Currently: all zeros (no Lidar implemented yet).
    pub lidar_distances: [f32; 64],
}

impl Default for HumanoidSensors {
    fn default() -> Self {
        Self {
            torso_up_vector: Vec3::Y,
            torso_forward_vector: Vec3::Z,
            left_foot_contact: false,
            right_foot_contact: false,
            left_foot_contact_force: 0.0,
            right_foot_contact_force: 0.0,
            tilt_angle_degrees: 0.0,
            lidar_distances: [0.0; 64],
        }
    }
}

impl HumanoidSensors {
    /// Pack sensor data into a flat float32 slice.
    ///
    /// Called from lib.rs get_humanoid_observation_into() to fill the sensor
    /// portion of the observation buffer.
    ///
    /// Buffer layout matches SENSOR_OBSERVATION_STRIDE = 18 floats.
    pub fn pack_into_buffer(
        &self,
        center_of_mass: Vec3,
        center_of_mass_velocity: Vec3,
        balance_error: f32,
        torso_rotation: glam::Quat,
        output_buffer: &mut [f32],
    ) {
        debug_assert!(
            output_buffer.len() >= SENSOR_OBSERVATION_STRIDE,
            "Sensor output buffer too small: need {}, got {}",
            SENSOR_OBSERVATION_STRIDE,
            output_buffer.len()
        );

        // Center of mass position
        output_buffer[0] = center_of_mass.x;
        output_buffer[1] = center_of_mass.y;
        output_buffer[2] = center_of_mass.z;

        // Center of mass velocity
        output_buffer[3] = center_of_mass_velocity.x;
        output_buffer[4] = center_of_mass_velocity.y;
        output_buffer[5] = center_of_mass_velocity.z;

        // Balance error
        output_buffer[6] = balance_error;

        // Torso up vector
        output_buffer[7] = self.torso_up_vector.x;
        output_buffer[8] = self.torso_up_vector.y;
        output_buffer[9] = self.torso_up_vector.z;

        // Foot contacts (encoded as 0.0 / 1.0 for the neural network)
        output_buffer[10] = if self.left_foot_contact { 1.0 } else { 0.0 };
        output_buffer[11] = if self.right_foot_contact { 1.0 } else { 0.0 };

        // Contact forces
        output_buffer[12] = self.left_foot_contact_force;
        output_buffer[13] = self.right_foot_contact_force;

        // Torso rotation quaternion
        output_buffer[14] = torso_rotation.x;
        output_buffer[15] = torso_rotation.y;
        output_buffer[16] = torso_rotation.z;
        output_buffer[17] = torso_rotation.w;
    }
}

// =============================================================================
// SYSTEM: update_humanoid_sensors_system
// =============================================================================
//
// LEARNING TOPIC: Sensor Update Ordering
// ----------------------------------------
// This system must run AFTER update_velocities_system (which commits final
// positions) and AFTER update_humanoid_rigs_system (which computes COM).
//
// Schedule order in lib.rs:
//   ... → update_velocities → spatial_grid_update → compute_stress →
//   update_humanoid_rigs → update_humanoid_sensors
//
// Running sensors last ensures we observe the fully-settled simulation state,
// not intermediate solver positions. The RL agent then receives clean observations
// that reflect the true physical state.
pub fn update_humanoid_sensors_system(
    mut sensors_query: Query<(&crate::agent::humanoid::HumanoidRig, &mut HumanoidSensors)>,
    voxel_query: Query<&crate::world::voxel::Voxel>,
) {
    for (rig, mut sensors) in sensors_query.iter_mut() {
        if !rig.is_alive {
            continue;
        }

        // ── Update torso orientation sensing ──────────────────────────────────
        //
        // LEARNING: The torso's Voxel::rotation is a quaternion in world space.
        // Rotating Vec3::Y by this quaternion gives us the torso's "up" direction
        // in world space — used for balance and tilt calculations.
        let torso_up_world;
        let torso_forward_world;
        let tilt_angle;

        if let Some(torso_entity) = rig.segment_entity(crate::agent::joints::BodySegment::Torso) {
            if let Ok(torso_voxel) = voxel_query.get(torso_entity) {
                // Transform local up/forward vectors to world space via rotation.
                torso_up_world = torso_voxel.rotation * Vec3::Y;
                torso_forward_world = torso_voxel.rotation * Vec3::Z;

                // Tilt angle: angle between the agent's up vector and world up.
                // 0 degrees = perfectly upright, 180 degrees = upside down.
                let cos_tilt = torso_up_world.dot(Vec3::Y).clamp(-1.0, 1.0);
                tilt_angle = cos_tilt.acos().to_degrees();
            } else {
                torso_up_world = Vec3::Y;
                torso_forward_world = Vec3::Z;
                tilt_angle = 0.0;
            }
        } else {
            torso_up_world = Vec3::Y;
            torso_forward_world = Vec3::Z;
            tilt_angle = 0.0;
        }

        sensors.torso_up_vector = torso_up_world;
        sensors.torso_forward_vector = torso_forward_world;
        sensors.tilt_angle_degrees = tilt_angle;

        // ── Update ground contact sensing ─────────────────────────────────────
        //
        // LEARNING: Ground contact is detected by checking if the shin's lowest
        // point is within GROUND_CONTACT_EPSILON of the floor plane.
        //
        // Contact force is approximated as the magnitude of the shin's velocity
        // damped by the floor contact — a proxy for the impact impulse.
        // This is not Newton force (N = kg·m/s²), it's a normalized [0,1] value
        // suitable as an RL observation without unit conversion.

        let (left_contact, left_force) = detect_foot_ground_contact(
            &rig,
            &voxel_query,
            crate::agent::joints::BodySegment::LeftShin,
        );

        let (right_contact, right_force) = detect_foot_ground_contact(
            &rig,
            &voxel_query,
            crate::agent::joints::BodySegment::RightShin,
        );

        sensors.left_foot_contact = left_contact;
        sensors.right_foot_contact = right_contact;
        sensors.left_foot_contact_force = left_force;
        sensors.right_foot_contact_force = right_force;

        // Lidar: zeros until Sprint 6.
        sensors.lidar_distances = [0.0; 64];
    }
}

/// Detect ground contact for one foot segment (left shin or right shin).
///
/// Returns: (is_in_contact: bool, contact_force_proxy: f32)
///
/// LEARNING: The contact_force_proxy is derived from the downward velocity
/// component (velocity.y when negative = moving toward floor). This gives the
/// neural network a sense of impact magnitude without computing actual forces.
fn detect_foot_ground_contact(
    rig: &crate::agent::humanoid::HumanoidRig,
    voxel_query: &Query<&crate::world::voxel::Voxel>,
    foot_segment: crate::agent::joints::BodySegment,
) -> (bool, f32) {
    let foot_entity = match rig.segment_entity(foot_segment) {
        Some(e) => e,
        None => return (false, 0.0),
    };

    let foot_voxel = match voxel_query.get(foot_entity) {
        Ok(v) => v,
        Err(_) => return (false, 0.0),
    };

    // The foot's lowest point in world space.
    // For a cube with half-extent = 0.25 in Y (shin dimensions):
    // lowest_y = center_y - 0.25
    let foot_lowest_y = foot_voxel.position.y - 0.25;

    // Ground contact: foot is within epsilon of the floor.
    let is_in_contact = foot_lowest_y <= FLOOR_Y + GROUND_CONTACT_EPSILON;

    // Contact force proxy: magnitude of downward velocity (clamped to [0,1]).
    // A foot moving down at 1 m/s or faster gets a force reading of 1.0.
    let contact_force = if is_in_contact {
        (-foot_voxel.velocity.y).max(0.0).min(1.0)
    } else {
        0.0
    };

    (is_in_contact, contact_force)
}
