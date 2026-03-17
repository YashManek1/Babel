// =============================================================================
// src/render/camera.rs  —  Orbital "Freecam" for the Babel Engine
// =============================================================================
//
// THEORY: Spherical Coordinates
// ──────────────────────────────────────────────────────────────────────────────
// We store the camera as THREE numbers instead of a raw position:
//
//   yaw      — rotation LEFT / RIGHT around the world's Y axis (radians)
//   pitch    — tilt UP / DOWN (clamped so we never flip upside-down)
//   distance — how far the camera sits from the TARGET point (zoom)
//
// From those three, we can always derive the camera's world position:
//
//   x = target.x + distance × cos(pitch) × sin(yaw)
//   y = target.y + distance × sin(pitch)
//   z = target.z + distance × cos(pitch) × cos(yaw)
//
// Then glam's Mat4::look_at_rh(position, target, Vec3::Y) turns that into the
// View matrix.  Combine with the Projection and you have the new MVP.
//
// WHY SPHERICAL?
//   • Orbiting around a point feels natural for a simulation viewer.
//   • We never get "gimbal lock" because pitch is simply clamped, not stored
//     as a quaternion composition.
//   • Adding pan (WASD) is just translating `target` in camera-local space.
//
// INPUT MAPPING
// ──────────────────────────────────────────────────────────────────────────────
//   RMB drag  → mouse_delta(dx, dy)  → yaw / pitch
//   Scroll    → scroll(delta)        → distance (zoom in/out)
//   WASD      → key_pressed/released → moves `target` in the XZ plane
//   Q / E     → moves `target` up / down
// =============================================================================

use glam::{Mat4, Vec3};
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// =============================================================================
// LEARNING: Why store booleans for keys instead of reacting immediately?
// =============================================================================
// Winit fires ONE event per key press/release.  But we need CONTINUOUS motion
// every frame while a key is HELD.  The solution is a "held keys" bitmask:
// on KeyDown → set flag = true, on KeyUp → set flag = false.
// Then in `update()` (called every frame) we integrate velocity while the flag
// is true.  This is the standard game-engine input pattern.
#[derive(Default, Clone, Copy)]
pub struct HeldKeys {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub q: bool, // pan down
    pub e: bool, // pan up
}

// =============================================================================
// FreecamState: the single source of truth for camera position/orientation
// =============================================================================
pub struct FreecamState {
    // ── Spherical angles (radians) ───────────────────────────────────────────
    /// Horizontal rotation around the world Y axis.
    /// 0 = looking toward +Z, π/2 = looking toward -X.
    pub yaw: f32,

    /// Vertical tilt.  Positive = looking down, negative = looking up.
    /// Clamped to ±89° so the camera never flips.
    pub pitch: f32,

    /// How far the camera eye sits from `target` (metres).
    pub distance: f32,

    // ── Orbit target ─────────────────────────────────────────────────────────
    /// The world-space point the camera always looks at.
    /// WASD/QE pan moves this point, not the camera directly.
    pub target: Vec3,

    // ── Input sensitivity tuning ─────────────────────────────────────────────
    /// Radians of yaw/pitch change per pixel of mouse drag.
    pub mouse_sensitivity: f32,

    /// Zoom multiplier per scroll "tick" (>1 = faster zoom).
    pub scroll_sensitivity: f32,

    /// World-units per second the target point moves with WASD.
    pub pan_speed: f32,

    // ── Key state ────────────────────────────────────────────────────────────
    pub keys: HeldKeys,

    // ── Right-mouse-button drag guard ────────────────────────────────────────
    // We only orbit when RMB is held, so the mouse doesn't fight with egui.
    pub rmb_held: bool,
}

impl Default for FreecamState {
    fn default() -> Self {
        Self {
            // Starting angle: slightly above the horizon, looking from the front
            yaw: std::f32::consts::FRAC_PI_4, // 45°
            pitch: 0.5,                       // ~28° down toward the scene
            distance: 18.0,                   // comfortable overview distance
            target: Vec3::new(0.0, 1.0, 0.0), // look at slightly above ground

            mouse_sensitivity: 0.005, // radians/pixel — feels natural
            scroll_sensitivity: 1.1,  // 10% zoom per scroll notch
            pan_speed: 8.0,           // units/second

            keys: HeldKeys::default(),
            rmb_held: false,
        }
    }
}

impl FreecamState {
    // =========================================================================
    // LEARNING: Deriving camera position from spherical coordinates
    // =========================================================================
    // This is the core math.  Given yaw, pitch, distance, and target, we derive
    // the camera's eye position in world space.
    //
    // Think of it as latitude/longitude on a sphere:
    //   pitch = latitude  (0 = equator, +90° = north pole)
    //   yaw   = longitude (0 = one meridian, 360° wraps around)
    //
    // The `right-handed` coordinate system means +Y is up, +Z is "toward us".
    fn eye_position(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();

        // Decompose the spherical offset into X/Y/Z components
        let offset = Vec3::new(
            self.distance * cos_pitch * sin_yaw, // left/right arc
            self.distance * sin_pitch,           // height arc
            self.distance * cos_pitch * cos_yaw, // forward/back arc
        );
        self.target + offset
    }

    // =========================================================================
    // LEARNING: Building the MVP matrix
    // =========================================================================
    // Called every frame from render_frame().  Returns the packed [[f32;4];4]
    // that gets written directly into the camera_buf uniform on the GPU.
    //
    //   View       = look_at_rh(eye, target, up)  — "right-handed" matches WGPU
    //   Projection = perspective_rh(fov, aspect, near, far)
    //   MVP        = Projection × View  (no Model needed — world space = model space)
    pub fn build_mvp(&self, aspect_ratio: f32) -> [[f32; 4]; 4] {
        // Perspective: 45° FOV, near=0.1, far=500 (much larger than before to
        // support flying far from the scene)
        let projection = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4, // 45° field of view
            aspect_ratio,
            0.1,   // near clip — don't go below this or you clip through blocks
            500.0, // far clip  — large enough to see the whole ground plane
        );

        // View: place the eye at the computed spherical position, look at target
        let view = Mat4::look_at_rh(self.eye_position(), self.target, Vec3::Y);

        // Combined MVP (no Model transform — all geometry is in world space)
        (projection * view).to_cols_array_2d()
    }

    // =========================================================================
    // Input handler: mouse drag (call this only when RMB is held)
    // =========================================================================
    // dx > 0 = mouse moved right  → yaw increases  → camera rotates right
    // dy > 0 = mouse moved down   → pitch increases → camera tilts down
    //
    // LEARNING: Why clamp pitch instead of letting it wrap?
    //   At pitch = +π/2 the camera is directly above the target. If you pass it,
    //   Vec3::Y becomes the "forward" direction and look_at_rh flips the image.
    //   Clamping to ±89° keeps everything stable.
    pub fn mouse_delta(&mut self, dx: f64, dy: f64) {
        if !self.rmb_held {
            return; // Only orbit when right mouse button is held
        }
        self.yaw += dx as f32 * self.mouse_sensitivity;
        self.pitch += dy as f32 * self.mouse_sensitivity;

        // Clamp pitch: 89° = just under the zenith / nadir
        let limit = 89.0_f32.to_radians();
        self.pitch = self.pitch.clamp(-limit, limit);

        // Yaw: let it wrap freely (no limit needed — it's a full circle)
        // Keeping it in [0, 2π] is optional but avoids float precision drift
        // over many rotations.
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);
    }

    // =========================================================================
    // Input handler: scroll wheel (zoom)
    // =========================================================================
    // LEARNING: Multiplicative zoom feels more natural than additive.
    //   Additive:      distance -= delta * 2.0  → same speed at any distance
    //   Multiplicative: distance *= factor^delta → faster when far, slower when
    //                   close — just like a camera lens.
    pub fn scroll(&mut self, delta: f32) {
        // delta > 0 = scroll up = zoom in = reduce distance
        // delta < 0 = scroll down = zoom out = increase distance
        if delta > 0.0 {
            self.distance /= self.scroll_sensitivity;
        } else {
            self.distance *= self.scroll_sensitivity;
        }
        // Prevent clipping into objects or zooming to infinity
        self.distance = self.distance.clamp(1.0, 300.0);
    }

    // =========================================================================
    // Input handler: key press / release
    // =========================================================================
    pub fn key_event(&mut self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;
        if let PhysicalKey::Code(code) = event.physical_key {
            match code {
                KeyCode::KeyW => self.keys.w = pressed,
                KeyCode::KeyA => self.keys.a = pressed,
                KeyCode::KeyS => self.keys.s = pressed,
                KeyCode::KeyD => self.keys.d = pressed,
                KeyCode::KeyQ => self.keys.q = pressed,
                KeyCode::KeyE => self.keys.e = pressed,
                _ => {}
            }
        }
    }

    // =========================================================================
    // Input handler: right mouse button state
    // =========================================================================
    pub fn set_rmb(&mut self, pressed: bool) {
        self.rmb_held = pressed;
    }

    // =========================================================================
    // Per-frame update: integrate WASD pan velocity
    // =========================================================================
    // LEARNING: dt (delta time in seconds) ensures pan speed is frame-rate
    // independent.  Without it, movement at 30fps feels half as fast as 60fps.
    //
    // We pan in CAMERA-LOCAL space so W always means "forward relative to where
    // you're looking," not "toward world +Z."  The camera's forward and right
    // vectors are derived from yaw (we ignore pitch for pan so the target stays
    // on the ground plane).
    pub fn update(&mut self, dt: f32) {
        if !self.keys.w
            && !self.keys.a
            && !self.keys.s
            && !self.keys.d
            && !self.keys.q
            && !self.keys.e
        {
            return; // Nothing held — early exit for performance
        }

        let speed = self.pan_speed * dt;

        // ── Camera-local axes (derived from yaw only — ignore pitch for pan) ─
        //
        // LEARNING: We use yaw-only for the XZ pan so that pressing W always
        // moves the target "forward on the ground."  If we included pitch,
        // W while looking down would move the target INTO the ground, which
        // feels wrong for an overhead simulation viewer.
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        // Forward = the direction the camera faces, projected onto XZ plane
        let forward = Vec3::new(sin_yaw, 0.0, cos_yaw).normalize();

        // Right = cross(forward, Y) — perpendicular to forward in the XZ plane.
        // IMPORTANT: must be cross(forward, Y) not cross(Y, forward).
        // cross(Y, forward) gives the LEFT vector in a right-handed system,
        // which makes D strafe left and A strafe right — the opposite of expected.
        let right = forward.cross(Vec3::Y).normalize();

        // ── Accumulate movement ───────────────────────────────────────────────
        let mut delta = Vec3::ZERO;

        if self.keys.w {
            delta -= forward * speed;
        } // pan away
        if self.keys.s {
            delta += forward * speed;
        } // pan toward scene center
        if self.keys.d {
            delta -= right * speed;
        } // strafe left
        if self.keys.a {
            delta += right * speed;
        } // strafe right
        if self.keys.e {
            delta.y += speed;
        } // lift target up
        if self.keys.q {
            delta.y -= speed;
        } // lower target down

        self.target += delta;
    }
}
