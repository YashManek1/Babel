// =============================================================================
// src/agent/joints.rs  —  Operation Babel: XPBD Joint Constraint System
// =============================================================================
//
// LEARNING TOPIC: XPBD Joints — Position-Based Rigid Body Connections
// ---------------------------------------------------------------------
// A joint constraint in XPBD works identically to a mortar bond at the
// algorithmic level, but with two key differences:
//
//   1. JOINTS ARE RIGID WITHIN A RANGE: Unlike mortar bonds that only resist
//      stretch, joints enforce both minimum AND maximum distance (forming an
//      "allowed angle range"). A knee joint can flex 0–120° but not hyperextend.
//
//   2. JOINTS TRANSMIT TORQUE: The agent applies a target angle to each joint.
//      The XPBD solver pushes the joint bodies toward that target angle each
//      step. The angular correction IS the torque — it's implicit in the
//      position update, exactly as linear correction IS the force in XPBD.
//
// JOINT ARCHITECTURE (8 joints, 11 body segments):
//
//   Torso (root, static during balance test)
//     ├─ Left Hip   ──► Left Thigh  ──► Left Knee ──► Left Shin
//     ├─ Right Hip  ──► Right Thigh ──► Right Knee ──► Right Shin
//     ├─ Left Shoulder ──► Left Upper Arm ──► Left Elbow ──► Left Forearm
//     └─ Right Shoulder ──► Right Upper Arm ──► Right Elbow ──► Right Forearm
//
// LEARNING TOPIC: Why XPBD for Joints (not Featherstone / Articulated Bodies)?
// -----------------------------------------------------------------------------
// Featherstone's Articulated Body Algorithm (used in MuJoCo, Bullet) is more
// physically accurate for rigid articulated chains. But for our use case:
//   - XPBD joints integrate seamlessly with the existing voxel collision system
//   - No separate rigid-body solver needed — same pipeline as blocks
//   - Stable at low iteration counts (critical for 2000 steps/sec target)
//   - Easy to add joint limits without special-casing angular math
//
// The tradeoff: slightly less physically precise rotation transfer at high speeds.
// For a construction AI learning to walk and balance, this tradeoff is excellent.
//
// LEARNING TOPIC: Quaternion-Based Angular Constraint
// ---------------------------------------------------
// For a hinge joint (like a knee), the constraint is:
//
//   C(q_a, q_b) = θ_current - θ_target = 0
//
// where θ_current is derived from the relative rotation between the two bodies.
// We compute:
//   relative_rotation = q_b^{-1} ⊗ q_a
//   swing_angle = 2 * acos(clamp(relative_rotation.w, -1, 1))
//
// The correction is applied as a position correction on the attachment points,
// which XPBD derives into the correct angular velocity update.
// =============================================================================

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use std::collections::HashMap;

// =============================================================================
// CONSTANTS: Joint physics parameters
// =============================================================================

/// How strongly the joint resists angular deviation from its target angle.
/// Higher = stiffer joint (faster convergence to target, more force on bodies).
/// Lower = looser joint (slower, more compliant — like tendons).
///
/// LEARNING: This is the XPBD compliance parameter α, scaled by inv_mass_sum.
/// Analogous to BASE_BOND_COMPLIANCE_FACTOR in mortar.rs but tuned for joints.
pub const JOINT_ANGULAR_STIFFNESS: f32 = 0.85;

/// Maximum angular correction per solver step in radians.
/// Prevents violent snapping when the agent starts from a rest pose and
/// receives a large torque command.
pub const MAX_ANGULAR_CORRECTION_PER_STEP: f32 = 0.12;

/// Maximum linear correction for the positional anchor constraint.
/// Keeps joint attachment points locked even under large angular forces.
pub const MAX_JOINT_LINEAR_CORRECTION: f32 = 0.18;

/// Joint damping — angular velocity reduction per contact frame.
/// 0.92 means 8% of angular velocity is removed each step (XPBD damping).
pub const JOINT_ANGULAR_DAMPING: f32 = 0.92;

/// Speed below which a joint body is considered "settled" for balance purposes.
pub const JOINT_BODY_SLEEP_SPEED: f32 = 0.15;

/// Minimum absolute torque magnitude that causes meaningful angular change.
/// Below this threshold, torque is ignored (prevents solver buzz).
pub const TORQUE_DEAD_ZONE: f32 = 0.005;

// =============================================================================
// JointType — Defines the degrees of freedom and constraints for each joint
// =============================================================================
//
// LEARNING: Different body joints need different constraint types:
//
//   Ball Socket: 3 rotational DOF, 0 linear DOF (shoulder, hip).
//     The child body can rotate freely around the parent attachment point.
//     Only the attachment point position is constrained.
//
//   Hinge: 1 rotational DOF, 0 linear DOF (knee, elbow).
//     The child body can only rotate around ONE axis (the hinge axis).
//     Twist and swing perpendicular to the hinge axis are locked.
//
//   Fixed: 0 rotational DOF, 0 linear DOF (torso segments).
//     Bodies are welded together — used internally to keep box-model limbs rigid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointType {
    /// 3-DOF ball socket: shoulder, hip. Allows swing in any direction, no twist.
    BallSocket {
        /// Maximum cone half-angle for swing (radians). Prevents hyper-extension.
        cone_limit_radians: f32,
    },
    /// 1-DOF hinge: knee, elbow. Rotation only around the specified axis.
    Hinge {
        /// The axis of rotation in the PARENT body's local space.
        hinge_axis: Vec3,
        /// Minimum angle (radians, can be negative for bidirectional hinges).
        min_angle_radians: f32,
        /// Maximum angle (radians).
        max_angle_radians: f32,
    },
}

// =============================================================================
// JointConstraint Component — Attached to the CHILD body entity
// =============================================================================
//
// LEARNING: We attach the joint data to the CHILD body (the one being
// controlled). This follows the convention in robotics where each link
// in a chain stores its connection to its parent.
//
// Example for a knee joint:
//   entity = left_shin_entity
//   parent_entity = left_thigh_entity
//   joint_type = Hinge { hinge_axis: Vec3::X, min: 0°, max: 120° }
//   local_attachment_parent = Vec3::new(0.0, -0.3, 0.0)  ← bottom of thigh
//   local_attachment_child  = Vec3::new(0.0,  0.3, 0.0)  ← top of shin
#[derive(Component, Clone, Debug)]
pub struct JointConstraint {
    /// The parent body entity this joint connects to.
    pub parent_entity: Entity,

    /// Attachment point in the PARENT body's local frame (before rotation).
    /// LEARNING: We store local attachment points so the joint anchor moves
    /// correctly as the parent body rotates — no need to update each frame.
    pub local_attachment_on_parent: Vec3,

    /// Attachment point in the CHILD body's local frame (before rotation).
    pub local_attachment_on_child: Vec3,

    /// The type of joint and its limits.
    pub joint_type: JointType,

    /// Current target angle for torque-driven control (radians).
    /// The solver drives toward this angle each step.
    /// For BallSocket joints, this is the swing angle (magnitude of desired swing).
    /// For Hinge joints, this is the rotation angle around the hinge axis.
    pub target_angle_radians: f32,

    /// Target swing direction for BallSocket joints (unit vector in parent space).
    /// The joint solver uses this to know WHICH direction to swing toward.
    /// For Hinge joints, this is ignored (the hinge axis defines the direction).
    pub target_swing_axis: Vec3,

    /// Current tension in the joint (deviation from target angle, radians).
    /// Written each step by solve_joint_constraints_system.
    /// Read by the Python bridge for observation data.
    pub current_angle_error: f32,

    /// Whether this joint is currently at its angular limit.
    /// Used by the humanoid to detect collisions with joint limits.
    pub is_at_limit: bool,
}

impl JointConstraint {
    /// Create a hinge joint (knee, elbow).
    ///
    /// # Arguments
    /// * `parent_entity` — the parent limb segment entity
    /// * `local_attachment_on_parent` — attachment point in parent's local space
    /// * `local_attachment_on_child` — attachment point in child's local space
    /// * `hinge_axis` — rotation axis in parent's local space (usually Vec3::X)
    /// * `min_angle` — minimum hinge angle in radians
    /// * `max_angle` — maximum hinge angle in radians
    pub fn new_hinge(
        parent_entity: Entity,
        local_attachment_on_parent: Vec3,
        local_attachment_on_child: Vec3,
        hinge_axis: Vec3,
        min_angle: f32,
        max_angle: f32,
    ) -> Self {
        Self {
            parent_entity,
            local_attachment_on_parent,
            local_attachment_on_child,
            joint_type: JointType::Hinge {
                hinge_axis: hinge_axis.normalize_or_zero(),
                min_angle_radians: min_angle,
                max_angle_radians: max_angle,
            },
            target_angle_radians: 0.0,
            target_swing_axis: Vec3::Y,
            current_angle_error: 0.0,
            is_at_limit: false,
        }
    }

    /// Create a ball socket joint (shoulder, hip).
    ///
    /// # Arguments
    /// * `parent_entity` — the parent limb segment entity
    /// * `local_attachment_on_parent` — attachment point in parent's local space
    /// * `local_attachment_on_child` — attachment point in child's local space
    /// * `cone_limit_radians` — maximum swing angle from neutral position
    pub fn new_ball_socket(
        parent_entity: Entity,
        local_attachment_on_parent: Vec3,
        local_attachment_on_child: Vec3,
        cone_limit_radians: f32,
    ) -> Self {
        Self {
            parent_entity,
            local_attachment_on_parent,
            local_attachment_on_child,
            joint_type: JointType::BallSocket { cone_limit_radians },
            target_angle_radians: 0.0,
            target_swing_axis: Vec3::Y,
            current_angle_error: 0.0,
            is_at_limit: false,
        }
    }
}

// =============================================================================
// JointBodyProperties Component — Physics properties for humanoid body segments
// =============================================================================
//
// LEARNING: Humanoid body segments need different physics properties than voxel
// blocks. Specifically:
//   - Fixed inv_mass matching anatomical proportions
//   - Explicit angular inertia (resists angular acceleration)
//   - A "body segment" flag so systems can distinguish humanoids from voxels
//
// We keep this as a SEPARATE component from Voxel so the humanoid can share
// the existing XPBD pipeline without polluting Voxel with humanoid-specific data.
#[derive(Component, Clone, Debug)]
pub struct JointBodyProperties {
    /// Stable humanoid agent id. Segments with the same id are connected by
    /// joints, so the collision solver skips self-collisions between them.
    pub agent_id: u32,

    /// Which humanoid body segment this is. Used for observation packing.
    pub segment: BodySegment,

    /// Inverse of the moment of inertia tensor (simplified to scalar for box shapes).
    /// Higher = easier to rotate (lighter or smaller segment).
    /// inv_inertia = 1 / (mass * size^2 / 6)  for a uniform box.
    pub inv_inertia: f32,

    /// Applied torque this frame (accumulated from Python commands).
    /// Units: N·m equivalent in simulation space.
    pub applied_torque: Vec3,

    /// Angular velocity (radians per second, world-space).
    pub angular_velocity: Vec3,
}

/// Enum identifying each named body segment in the humanoid rig.
///
/// LEARNING: Using an enum instead of a string name means:
///   - Zero heap allocation per segment
///   - Pattern matching is exhaustive (compiler catches missing cases)
///   - Observation buffer indexing is deterministic (enum as array index)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BodySegment {
    Torso,
    LeftThigh,
    LeftShin,
    RightThigh,
    RightShin,
    LeftUpperArm,
    LeftForearm,
    RightUpperArm,
    RightForearm,
}

impl BodySegment {
    /// Total number of body segments in the humanoid rig.
    /// Used to pre-size observation vectors.
    pub const COUNT: usize = 9;

    /// Dense array index for this segment.
    /// Enables O(1) observation packing: obs[segment.index() * JOINT_OBS_STRIDE].
    pub fn index(self) -> usize {
        match self {
            BodySegment::Torso => 0,
            BodySegment::LeftThigh => 1,
            BodySegment::LeftShin => 2,
            BodySegment::RightThigh => 3,
            BodySegment::RightShin => 4,
            BodySegment::LeftUpperArm => 5,
            BodySegment::LeftForearm => 6,
            BodySegment::RightUpperArm => 7,
            BodySegment::RightForearm => 8,
        }
    }

    /// Human-readable label for the egui UI control panel.
    pub fn display_label(self) -> &'static str {
        match self {
            BodySegment::Torso => "Torso",
            BodySegment::LeftThigh => "L Thigh",
            BodySegment::LeftShin => "L Shin",
            BodySegment::RightThigh => "R Thigh",
            BodySegment::RightShin => "R Shin",
            BodySegment::LeftUpperArm => "L Upper Arm",
            BodySegment::LeftForearm => "L Forearm",
            BodySegment::RightUpperArm => "R Upper Arm",
            BodySegment::RightForearm => "R Forearm",
        }
    }
}

// =============================================================================
// OBSERVATION CONSTANTS for joint data
// =============================================================================

/// Floats per joint segment in the observation buffer.
/// Layout: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
///          rot_x, rot_y, rot_z, rot_w, ang_vel_x, ang_vel_y, ang_vel_z,
///          joint_angle_error, is_at_limit]
/// Total: 15 floats per segment.
pub const JOINT_OBSERVATION_STRIDE: usize = 15;

/// Fixed simulation delta used when converting pose corrections into angular velocity.
const JOINT_SIMULATION_TIME_STEP_SECONDS: f32 = 1.0 / 60.0;

// =============================================================================
// SYSTEM: solve_joint_constraints_system
// =============================================================================
//
// LEARNING TOPIC: Joint Constraint Solving in XPBD
// ------------------------------------------------
// This system runs IN PLACE OF the mortar constraint solver for humanoid
// bodies. It solves two sub-constraints simultaneously each call:
//
// SUB-CONSTRAINT 1 — POSITIONAL ANCHOR:
//   The attachment point on the parent and the attachment point on the child
//   must be at the same world-space position. This is a rigid connection.
//   This is solved FIRST with high stiffness.
//
//   Violation: delta = world_attachment_parent - world_attachment_child
//   Correction: push child toward parent's attachment point (scaled by mass).
//
// SUB-CONSTRAINT 2 — ANGULAR TARGET:
//   The relative rotation between parent and child should match the target angle.
//   This is solved AFTER the positional constraint.
//
//   For Hinge: compute the rotation around the hinge axis and drive toward target.
//   For BallSocket: compute the swing angle and drive toward target.
//
// JACOBI AVERAGING (same as mortar.rs):
//   If a body is connected to multiple joints, corrections are averaged.
//   This prevents the same body from being pulled in conflicting directions.
pub fn solve_joint_constraints_system(
    mut query_set: ParamSet<(
        Query<(
            Entity,
            &crate::world::voxel::Voxel,
            &JointBodyProperties,
            Option<&JointConstraint>,
        )>,
        Query<(
            Entity,
            &mut crate::world::voxel::Voxel,
            &mut JointBodyProperties,
            Option<&mut JointConstraint>,
        )>,
    )>,
) {
    if query_set.p0().is_empty() {
        return;
    }

    // --- Snapshot: freeze predicted positions and rotations for this step ---
    //
    // LEARNING: Same Jacobi snapshot pattern as mortar.rs. We read all current
    // positions once, compute all corrections against the frozen snapshot, then
    // apply. This prevents order-dependent drift where the first joint solved
    // contaminates the positions used by the second joint.
    let mut position_snapshots: HashMap<Entity, (Vec3, Quat, f32)> =
        HashMap::with_capacity(BodySegment::COUNT * 2);

    for (entity, voxel, _joint_body, _joint_constraint) in query_set.p0().iter() {
        position_snapshots.insert(
            entity,
            (voxel.predicted_position, voxel.rotation, voxel.inv_mass),
        );
    }

    let mut joint_snapshots: Vec<(Entity, JointConstraint)> =
        Vec::with_capacity(BodySegment::COUNT * 2);
    for (entity, _voxel, _joint_body, joint_constraint_option) in query_set.p0().iter() {
        if let Some(joint) = joint_constraint_option {
            joint_snapshots.push((entity, joint.clone()));
        }
    }

    // Accumulated position corrections (Jacobi averaging across multiple joints)
    let mut position_corrections: HashMap<Entity, Vec3> =
        HashMap::with_capacity(BodySegment::COUNT * 2);
    let mut position_correction_counts: HashMap<Entity, u32> =
        HashMap::with_capacity(BodySegment::COUNT * 2);
    let mut rotation_targets: HashMap<Entity, (Quat, Vec3)> =
        HashMap::with_capacity(BodySegment::COUNT * 2);

    // --- Solve each joint ---
    for (child_entity, joint_snapshot) in joint_snapshots.iter() {
        let child_entity = *child_entity;

        let Some(&(parent_position, parent_rotation, parent_inv_mass)) =
            position_snapshots.get(&joint_snapshot.parent_entity)
        else {
            continue; // Parent entity no longer exists (despawned)
        };
        let Some(&(child_position, child_rotation, child_inv_mass)) =
            position_snapshots.get(&child_entity)
        else {
            continue;
        };

        // ─── Sub-Constraint 1: Positional Anchor ──────────────────────────────
        //
        // LEARNING: Transform attachment points from local to world space.
        //   world_attachment = body_center + rotation * local_attachment
        //
        // This is the standard "local-to-world" transform. We use quaternion
        // rotation (not matrix multiplication) because we store rotation as Quat.
        let world_attachment_on_parent =
            parent_position + parent_rotation * joint_snapshot.local_attachment_on_parent;
        let world_attachment_on_child =
            child_position + child_rotation * joint_snapshot.local_attachment_on_child;
        let attachment_delta = world_attachment_on_parent - world_attachment_on_child;
        let inv_mass_sum = (parent_inv_mass + child_inv_mass).max(1e-5);

        accumulate_position_correction(
            &mut position_corrections,
            &mut position_correction_counts,
            child_entity,
            (attachment_delta * (child_inv_mass / inv_mass_sum) * JOINT_ANGULAR_STIFFNESS)
                .clamp_length_max(MAX_JOINT_LINEAR_CORRECTION),
        );
        accumulate_position_correction(
            &mut position_corrections,
            &mut position_correction_counts,
            joint_snapshot.parent_entity,
            (-attachment_delta * (parent_inv_mass / inv_mass_sum) * JOINT_ANGULAR_STIFFNESS)
                .clamp_length_max(MAX_JOINT_LINEAR_CORRECTION),
        );

        // ANGULAR CONSTRAINT: Apply torque by updating target angle.
        //
        // LEARNING: The torque from Python arrives as target_angle_radians on each joint.
        // We clamp it to the joint's limits and store it. The Jacobi correction pass
        // then drives the bodies toward this angle each frame.
        let (target_rotation, current_angle_error, is_at_limit) = match &joint_snapshot.joint_type {
            JointType::Hinge {
                hinge_axis,
                min_angle_radians,
                max_angle_radians,
            } => {
                let clamped_target = joint_snapshot
                    .target_angle_radians
                    .clamp(*min_angle_radians, *max_angle_radians);
                let world_hinge_axis =
                    (parent_rotation * hinge_axis.normalize_or_zero()).normalize_or_zero();
                let relative_rotation = parent_rotation.inverse() * child_rotation;
                let (relative_axis, relative_angle) = relative_rotation.to_axis_angle();
                let signed_relative_angle = relative_angle
                    * relative_axis
                        .normalize_or_zero()
                        .dot(hinge_axis.normalize_or_zero())
                        .signum();
                let angle_error = clamped_target - signed_relative_angle;
                let limited_angle_error = angle_error.clamp(
                    -MAX_ANGULAR_CORRECTION_PER_STEP,
                    MAX_ANGULAR_CORRECTION_PER_STEP,
                ) * JOINT_ANGULAR_STIFFNESS;
                let target_rotation =
                    Quat::from_axis_angle(world_hinge_axis, limited_angle_error) * child_rotation;
                let is_at_limit =
                    (clamped_target - joint_snapshot.target_angle_radians).abs() > 0.001;
                (target_rotation.normalize(), angle_error, is_at_limit)
            }
            JointType::BallSocket { cone_limit_radians } => {
                let clamped_target = joint_snapshot
                    .target_angle_radians
                    .clamp(-*cone_limit_radians, *cone_limit_radians);
                let parent_up = (parent_rotation * Vec3::Y).normalize_or_zero();
                let desired_swing_direction_world = (parent_rotation
                    * joint_snapshot.target_swing_axis.normalize_or_zero())
                .normalize_or_zero();
                let desired_child_up_world =
                    if desired_swing_direction_world.length_squared() <= 1e-6 {
                        parent_up
                    } else {
                        let blend_factor =
                            (clamped_target.abs() / cone_limit_radians.max(1e-5)).clamp(0.0, 1.0);
                        parent_up
                            .lerp(desired_swing_direction_world, blend_factor)
                            .normalize_or_zero()
                    };
                let current_child_up_world = (child_rotation * Vec3::Y).normalize_or_zero();
                let angle_error = current_child_up_world.angle_between(desired_child_up_world);
                let correction_axis = current_child_up_world
                    .cross(desired_child_up_world)
                    .normalize_or_zero();
                let limited_angle_error = angle_error.clamp(
                    -MAX_ANGULAR_CORRECTION_PER_STEP,
                    MAX_ANGULAR_CORRECTION_PER_STEP,
                ) * JOINT_ANGULAR_STIFFNESS;
                let target_rotation = if correction_axis.length_squared() <= 1e-6 {
                    child_rotation
                } else {
                    Quat::from_axis_angle(correction_axis, limited_angle_error) * child_rotation
                };
                let is_at_limit =
                    (clamped_target - joint_snapshot.target_angle_radians).abs() > 0.001;
                (target_rotation.normalize(), angle_error, is_at_limit)
            }
        };

        let new_world_attachment_on_child =
            child_position + target_rotation * joint_snapshot.local_attachment_on_child;
        let post_rotation_anchor_delta = world_attachment_on_parent - new_world_attachment_on_child;
        accumulate_position_correction(
            &mut position_corrections,
            &mut position_correction_counts,
            child_entity,
            (post_rotation_anchor_delta * JOINT_ANGULAR_STIFFNESS)
                .clamp_length_max(MAX_JOINT_LINEAR_CORRECTION),
        );
        rotation_targets.insert(
            child_entity,
            (
                target_rotation,
                derive_angular_velocity(child_rotation, target_rotation),
            ),
        );

        if let Ok((_entity, _voxel, _joint_body, Some(mut live_joint))) =
            query_set.p1().get_mut(child_entity)
        {
            live_joint.current_angle_error = current_angle_error;
            live_joint.is_at_limit = is_at_limit;
            live_joint.target_angle_radians = match live_joint.joint_type {
                JointType::Hinge {
                    min_angle_radians,
                    max_angle_radians,
                    ..
                } => live_joint
                    .target_angle_radians
                    .clamp(min_angle_radians, max_angle_radians),
                JointType::BallSocket { cone_limit_radians } => live_joint
                    .target_angle_radians
                    .clamp(-cone_limit_radians, cone_limit_radians),
            };
        }
    }

    // Apply accumulated position corrections (Jacobi averaged).
    // LEARNING: Same pattern as mortar.rs apply step — divide by count before applying.
    for (entity, mut voxel, mut joint_body, _joint_constraint_option) in query_set.p1().iter_mut() {
        if let Some(&correction) = position_corrections.get(&entity) {
            let correction_count = position_correction_counts
                .get(&entity)
                .copied()
                .unwrap_or(1)
                .max(1);
            let averaged_correction = (correction / correction_count as f32)
                .clamp_length_max(MAX_JOINT_LINEAR_CORRECTION);
            voxel.predicted_position += averaged_correction;
        }

        if let Some((target_rotation, derived_angular_velocity)) = rotation_targets.get(&entity) {
            voxel.rotation = *target_rotation;
            voxel.angular_velocity = *derived_angular_velocity;
            joint_body.angular_velocity = *derived_angular_velocity;
            joint_body.applied_torque = *derived_angular_velocity;
        } else {
            voxel.angular_velocity *= JOINT_ANGULAR_DAMPING;
            joint_body.angular_velocity *= JOINT_ANGULAR_DAMPING;
            joint_body.applied_torque = Vec3::ZERO;
        }
    }
}

// =============================================================================
// SYSTEM: apply_joint_torques_system
// =============================================================================
//
// LEARNING TOPIC: Torque as Target Angle in XPBD
// -----------------------------------------------
// In XPBD, we don't directly apply force/torque vectors. Instead:
//
//   1. The Python controller sets joint.target_angle_radians
//   2. This system computes the ANGULAR CORRECTION needed to reach that target
//   3. The correction is applied as a position displacement on the attachment points
//   4. update_velocities_system derives angular velocity from the position change
//
// This is fundamentally different from impulse-based engines (Bullet, PhysX) where
// you apply a torque vector and integrate angular momentum. In XPBD:
//   TORQUE → TARGET ANGLE → POSITION CORRECTION → VELOCITY DERIVATION
//
// The result is identical physics behavior but with dramatically better stability.
// A joint being overloaded (target unreachable) gracefully degrades: it just
// doesn't quite reach the target rather than exploding with infinite force.
pub fn apply_joint_torques_system(
    mut body_query: Query<(&mut crate::world::voxel::Voxel, &mut JointBodyProperties)>,
    joint_query: Query<(Entity, &JointConstraint)>,
) {
    // LEARNING: For each joint, compute the correction that drives the child body
    // toward the target angle. The correction is a small angular rotation applied
    // as a position displacement on the child's predicted_position.
    //
    // This is the "angular XPBD" step described in Macklin et al. (2020):
    //   "Detailed Rigid Body Simulation with Extended Position Based Dynamics"
    //
    // We use a simplified version that works well for the box-model humanoid:
    // rather than computing full angular correction via quaternion differentiation,
    // we compute the angular error as a scalar, scale it by the joint stiffness,
    // and apply the resulting rotation to the child body's rotation quaternion.

    for (entity, joint) in joint_query.iter() {
        if joint.current_angle_error.abs() < TORQUE_DEAD_ZONE {
            continue; // No meaningful torque needed — save solver work
        }

        if let Ok((mut voxel, mut joint_body)) = body_query.get_mut(entity) {
            let torque_direction = match joint.joint_type {
                JointType::Hinge { hinge_axis, .. } => hinge_axis.normalize_or_zero(),
                JointType::BallSocket { .. } => joint.target_swing_axis.normalize_or_zero(),
            };
            let world_torque_direction = if torque_direction.length_squared() <= 1e-6 {
                Vec3::Y
            } else {
                torque_direction
            };
            let angular_impulse = world_torque_direction
                * joint.current_angle_error.clamp(
                    -MAX_ANGULAR_CORRECTION_PER_STEP,
                    MAX_ANGULAR_CORRECTION_PER_STEP,
                )
                * JOINT_ANGULAR_STIFFNESS
                * joint_body.inv_inertia.max(0.05);
            joint_body.applied_torque = angular_impulse;
            joint_body.angular_velocity =
                (joint_body.angular_velocity + angular_impulse) * JOINT_ANGULAR_DAMPING;
            voxel.angular_velocity = joint_body.angular_velocity;
        }
    }
}

fn accumulate_position_correction(
    position_corrections: &mut HashMap<Entity, Vec3>,
    position_correction_counts: &mut HashMap<Entity, u32>,
    entity: Entity,
    correction: Vec3,
) {
    if correction.length_squared() <= 1e-8 {
        return;
    }

    position_corrections
        .entry(entity)
        .and_modify(|existing_correction| *existing_correction += correction)
        .or_insert(correction);
    position_correction_counts
        .entry(entity)
        .and_modify(|count| *count += 1)
        .or_insert(1);
}

fn derive_angular_velocity(previous_rotation: Quat, next_rotation: Quat) -> Vec3 {
    let relative_rotation = next_rotation * previous_rotation.inverse();
    let (rotation_axis, rotation_angle) = relative_rotation.to_axis_angle();
    if rotation_axis.length_squared() <= 1e-6 || rotation_angle.abs() <= 1e-6 {
        Vec3::ZERO
    } else {
        rotation_axis.normalize_or_zero() * (rotation_angle / JOINT_SIMULATION_TIME_STEP_SECONDS)
    }
}
