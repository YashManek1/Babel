// =============================================================================
// src/render/wgpu_view.rs  —  Operation Babel: WGPU Renderer
// =============================================================================
//
// SPRINT 5 ADDITIONS:
//   - Right-side "Agent Control" panel in egui for humanoid rig interaction
//   - render_frame_with_stress() now accepts an agent_panel_state reference
//   - AgentPanelState: tracks per-joint slider values and pose preset selection
//   - UiAgentCommand variants emitted when user interacts with agent panel
//   - Humanoid segment mesh rendering: box segments colored by segment type
//   - Joint visualization: thin lines drawn between attachment points
//
// SPRINT 4 ADDITIONS (unchanged):
//   - render_frame_with_stress(): stress-aware render entry point
//   - height_to_color_with_stress(): stress-aware color computation
//   - Scaffold rendering: diagonal hatch signal via color.g = -1.0
//   - SpawnScaffold UiCommand: new button in the egui toolbar
//   - despawn_scaffolding UI button: removes all scaffold in one click
//
// LEARNING TOPIC: Dual-Panel egui Layout
// ----------------------------------------
// The left panel (egui::Window "Babel Engine") contains block spawning controls.
// The NEW right panel (egui::SidePanel::right "Agent Control") contains:
//
//   Section 1 — Spawn/Despawn: one button to spawn a humanoid, one to remove all.
//   Section 2 — Pose Presets: dropdown to apply named poses (T-Stance, Standing, etc.)
//   Section 3 — Per-Joint Sliders: one angle slider per joint (8 total).
//     Each slider shows the joint name (from BodySegment::display_label()),
//     the current value in degrees, and a reset-to-zero button.
//   Section 4 — Agent Status: live balance error, COM height, foot contacts.
//
// LEARNING TOPIC: egui SidePanel vs Window
// ------------------------------------------
// egui::Window is a floating, draggable overlay (used for block spawner).
// egui::SidePanel::right is a fixed panel anchored to the right edge of the
// viewport — it never overlaps the 3D scene and its width is fixed.
// This distinction matters because:
//   - The 3D scene click-raycasting assumes the panel occupies the right portion
//     of the screen (we subtract panel_width from viewport width when casting).
//   - Windows can be minimized/moved; side panels cannot — the agent controls
//     should always be visible during debugging and training.
//
// LEARNING TOPIC: The Color Encoding Convention (unchanged from Sprint 4)
// -----------------------------------------------------------------------
// color.r < 0.0         → Ground plane (procedural grass shader)
// color.g < -0.5        → Scaffold block (diagonal hatch in shader)
// color.r ∈ [10, 20)    → NEW SPRINT 5 SIGNAL: Humanoid segment
//                          color.r = 10.0 + segment_index (0..9)
//                          Shader reads this and applies the humanoid palette.
// otherwise             → Normal block with stress-tinted color

use crate::agent::{BodySegment, HumanoidPosePreset, UiAgentCommand};
use crate::render::camera::FreecamState;
use crate::world::voxel::{MaterialType, ShapeType, Voxel};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta},
    event_loop::EventLoop,
    window::Window,
};

// =============================================================================
// LEARNING MECHANIC: The Command Pattern (unchanged)
// =============================================================================
pub enum UiCommand {
    SpawnCube {
        x: f32,
        z: f32,
        material_id: u8,
    },
    SpawnWedge {
        x: f32,
        z: f32,
        material_id: u8,
    },
    SpawnSphere {
        x: f32,
        z: f32,
        radius: f32,
        material_id: u8,
    },
    SpawnScaffold {
        x: f32,
        z: f32,
    },
    DespawnAllScaffold,
}

// =============================================================================
// AgentPanelState — Live state for the right-side agent control panel
// =============================================================================
//
// LEARNING: The panel state lives in RenderContext (not in ECS) because it is
// purely UI state — it does not represent physics truth. The panel state is the
// user's INTENT; the physics engine is the TRUTH. They are connected via
// UiAgentCommand values: when the user moves a slider, we emit a command;
// the engine processes the command next frame; the panel reflects the result.
//
// This is the same design as spawn_x / spawn_z on RenderContext for the
// block spawner panel — UI state separate from physics state.
pub struct AgentPanelState {
    /// Joint angle slider values (degrees) for the selected agent.
    /// Indexed by BodySegment::index().
    /// 9 joints total matching the humanoid rig from humanoid.rs.
    pub joint_angles_degrees: [f32; 9],

    /// Currently selected pose preset in the dropdown.
    pub selected_preset: HumanoidPosePreset,

    /// Live balance error read from the most recently updated HumanoidRig.
    /// Written by lib.rs after each physics step; read by the panel to display status.
    pub display_balance_error: f32,

    /// Live COM height.
    pub display_com_height: f32,

    /// Live left foot contact flag.
    pub display_left_foot_contact: bool,

    /// Live right foot contact flag.
    pub display_right_foot_contact: bool,

    /// How many humanoid agents are currently alive.
    pub display_agent_count: usize,
}

impl Default for AgentPanelState {
    fn default() -> Self {
        Self {
            joint_angles_degrees: [0.0; 9],
            selected_preset: HumanoidPosePreset::Standing,
            display_balance_error: 0.0,
            display_com_height: 0.0,
            display_left_foot_contact: false,
            display_right_foot_contact: false,
            display_agent_count: 0,
        }
    }
}

// =============================================================================
// LEARNING MECHANIC: Vertex Layout & bytemuck (unchanged)
// =============================================================================
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// =============================================================================
// SPRINT 5: Humanoid segment color palette
// =============================================================================
//
// LEARNING: Sprint 5 needs the Worker to read as one humanoid agent, not a
// rainbow of unrelated debug blocks. These colors stay close together while
// still making the torso, legs, and arms readable.
const HUMANOID_SEGMENT_COLORS: [[f32; 3]; BodySegment::COUNT] = [
    [0.22, 0.40, 0.62], // Torso: work jacket blue
    [0.24, 0.27, 0.30], // Left Thigh: dark work pants
    [0.20, 0.22, 0.24], // Left Shin
    [0.24, 0.27, 0.30], // Right Thigh
    [0.20, 0.22, 0.24], // Right Shin
    [0.78, 0.58, 0.40], // Left Upper Arm: skin/glove tone
    [0.70, 0.50, 0.34], // Left Forearm
    [0.78, 0.58, 0.40], // Right Upper Arm
    [0.70, 0.50, 0.34], // Right Forearm
];

const HUMANOID_HEAD_COLOR: [f32; 3] = [0.78, 0.58, 0.40];

// =============================================================================
// Mesh builders (unchanged from Sprint 4, with humanoid segment support added)
// =============================================================================

fn push_cube_mesh(
    vertex_buffer: &mut Vec<Vertex>,
    solid_index_buffer: &mut Vec<u32>,
    border_index_buffer: &mut Vec<u32>,
    voxel: &Voxel,
    render_position: Vec3,
    color: [f32; 3],
) {
    let base_index = vertex_buffer.len() as u32;
    let local_vertices = [
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
    ];

    for &local in local_vertices.iter() {
        let world = render_position + (voxel.rotation * local);
        vertex_buffer.push(Vertex {
            position: world.into(),
            color,
        });
    }

    solid_index_buffer.extend(
        [
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7,
            3, 4, 5, 1, 1, 0, 4,
        ]
        .iter()
        .map(|index| index + base_index),
    );

    border_index_buffer.extend(
        [
            0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7,
        ]
        .iter()
        .map(|index| index + base_index),
    );
}

fn push_wedge_mesh(
    vertex_buffer: &mut Vec<Vertex>,
    solid_index_buffer: &mut Vec<u32>,
    border_index_buffer: &mut Vec<u32>,
    voxel: &Voxel,
    render_position: Vec3,
    color: [f32; 3],
) {
    let base_index = vertex_buffer.len() as u32;
    let local_vertices = [
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
    ];

    for &local in local_vertices.iter() {
        let world = render_position + (voxel.rotation * local);
        vertex_buffer.push(Vertex {
            position: world.into(),
            color,
        });
    }

    solid_index_buffer.extend(
        [
            0, 1, 2, 2, 3, 0, 3, 2, 4, 4, 5, 3, 0, 3, 5, 1, 4, 2, 0, 5, 4, 4, 1, 0,
        ]
        .iter()
        .map(|index| index + base_index),
    );
    border_index_buffer.extend(
        [0, 1, 1, 2, 2, 3, 3, 0, 3, 5, 5, 4, 4, 2, 0, 5, 1, 4]
            .iter()
            .map(|index| index + base_index),
    );
}

fn push_sphere_mesh(
    vertex_buffer: &mut Vec<Vertex>,
    solid_index_buffer: &mut Vec<u32>,
    border_index_buffer: &mut Vec<u32>,
    voxel: &Voxel,
    render_position: Vec3,
    color: [f32; 3],
) {
    let base_index = vertex_buffer.len() as u32;
    let latitude_count: u32 = 12;
    let longitude_count: u32 = 18;

    for latitude_index in 0..=latitude_count {
        let theta = latitude_index as f32 * std::f32::consts::PI / latitude_count as f32;
        for longitude_index in 0..=longitude_count {
            let phi = longitude_index as f32 * 2.0 * std::f32::consts::PI / longitude_count as f32;
            let local = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ) * voxel.sphere_radius;
            let world = render_position + (voxel.rotation * local);
            vertex_buffer.push(Vertex {
                position: world.into(),
                color,
            });
        }
    }

    for latitude_index in 0..latitude_count {
        for longitude_index in 0..longitude_count {
            let row_start = latitude_index * (longitude_count + 1);
            let next_row_start = (latitude_index + 1) * (longitude_count + 1);
            solid_index_buffer.extend([
                base_index + row_start + longitude_index,
                base_index + next_row_start + longitude_index,
                base_index + row_start + longitude_index + 1,
            ]);
            solid_index_buffer.extend([
                base_index + row_start + longitude_index + 1,
                base_index + next_row_start + longitude_index,
                base_index + next_row_start + longitude_index + 1,
            ]);
            border_index_buffer.extend([
                base_index + row_start + longitude_index,
                base_index + row_start + longitude_index + 1,
            ]);
            border_index_buffer.extend([
                base_index + row_start + longitude_index,
                base_index + next_row_start + longitude_index,
            ]);
        }
    }
}

// =============================================================================
// SPRINT 5: push_humanoid_segment_mesh — scaled cube for body segments
// =============================================================================
//
// LEARNING: Humanoid segments are rendered as SCALED cubes. The scale factors
// match the segment dimensions from humanoid.rs (e.g., torso is 0.6 × 1.0 × 0.4).
//
// Rather than storing separate half-extents in Voxel (which would bloat the
// component), we pass the scale as a parameter derived from JointBodyProperties.
// For Sprint 5 (rendering all segments as cubes), the scale is hardcoded per
// segment type. Sprint 6 will read scales from the JointBodyProperties component.
fn push_humanoid_segment_mesh(
    vertex_buffer: &mut Vec<Vertex>,
    solid_index_buffer: &mut Vec<u32>,
    border_index_buffer: &mut Vec<u32>,
    voxel: &Voxel,
    render_position: Vec3,
    color: [f32; 3],
    half_extents: Vec3,
) {
    let base_index = vertex_buffer.len() as u32;

    // 8 corners of a scaled box (not unit cube).
    let local_vertices = [
        Vec3::new(-half_extents.x, -half_extents.y, half_extents.z),
        Vec3::new(half_extents.x, -half_extents.y, half_extents.z),
        Vec3::new(half_extents.x, half_extents.y, half_extents.z),
        Vec3::new(-half_extents.x, half_extents.y, half_extents.z),
        Vec3::new(-half_extents.x, -half_extents.y, -half_extents.z),
        Vec3::new(half_extents.x, -half_extents.y, -half_extents.z),
        Vec3::new(half_extents.x, half_extents.y, -half_extents.z),
        Vec3::new(-half_extents.x, half_extents.y, -half_extents.z),
    ];

    for &local in local_vertices.iter() {
        let world = render_position + (voxel.rotation * local);
        vertex_buffer.push(Vertex {
            position: world.into(),
            color,
        });
    }

    solid_index_buffer.extend(
        [
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7,
            3, 4, 5, 1, 1, 0, 4,
        ]
        .iter()
        .map(|index| index + base_index),
    );

    border_index_buffer.extend(
        [
            0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7,
        ]
        .iter()
        .map(|index| index + base_index),
    );
}

// =============================================================================
// SPRINT 4: height_to_color_with_stress() (unchanged)
// =============================================================================
fn height_to_color_with_stress(
    voxel: &Voxel,
    stress_normalized: f32,
    is_scaffold: bool,
) -> [f32; 3] {
    if is_scaffold {
        return [0.001, -2.0, 0.0];
    }

    let base = match voxel.material {
        MaterialType::Wood => [0.70, 0.52, 0.32],
        MaterialType::Steel => [0.55, 0.60, 0.68],
        MaterialType::Stone => [0.62, 0.62, 0.58],
        MaterialType::Scaffold => [0.85, 0.55, 0.15],
    };
    let height_boost = ((voxel.position.y.max(0.0) / 12.0) * 0.10).min(0.10);
    let base_tinted = [
        (base[0] + height_boost).min(1.0),
        (base[1] + height_boost).min(1.0),
        (base[2] + height_boost).min(1.0),
    ];

    let green = [0.2_f32, 0.85, 0.2];
    let yellow = [0.95_f32, 0.85, 0.1];
    let red = [0.95_f32, 0.1, 0.1];

    let stress_color = if stress_normalized <= 0.5 {
        let blend_factor = stress_normalized / 0.5;
        [
            green[0] + (yellow[0] - green[0]) * blend_factor,
            green[1] + (yellow[1] - green[1]) * blend_factor,
            green[2] + (yellow[2] - green[2]) * blend_factor,
        ]
    } else {
        let blend_factor = (stress_normalized - 0.5) / 0.5;
        [
            yellow[0] + (red[0] - yellow[0]) * blend_factor,
            yellow[1] + (red[1] - yellow[1]) * blend_factor,
            yellow[2] + (red[2] - yellow[2]) * blend_factor,
        ]
    };

    let quadratic_blend = stress_normalized * stress_normalized;
    [
        base_tinted[0] + (stress_color[0] - base_tinted[0]) * quadratic_blend,
        base_tinted[1] + (stress_color[1] - base_tinted[1]) * quadratic_blend,
        base_tinted[2] + (stress_color[2] - base_tinted[2]) * quadratic_blend,
    ]
}

// =============================================================================
// build_mesh_with_stress — builds the vertex and index buffers for ALL objects
// =============================================================================
//
// SPRINT 5: Now also accepts humanoid segment data.
// Humanoid segments are rendered as scaled cubes with segment-specific colors.
//
// Arguments:
//   voxels: (voxel_ref, stress_normalized, is_scaffold, humanoid_segment_index)
//     humanoid_segment_index: None = normal block, Some(i) = humanoid segment i
fn build_mesh_with_stress(
    voxels: &[(&Voxel, f32, bool, Option<usize>)],
    spawn_target_x: f32,
    spawn_target_z: f32,
) -> (Vec<Vertex>, Vec<u32>, Vec<u32>) {
    let mut vertex_buffer = Vec::new();
    let mut solid_index_buffer = Vec::new();
    let mut border_index_buffer = Vec::new();

    // ── Ground plane (unchanged) ──────────────────────────────────────────────
    let ground_size = 1000.0;
    let ground_y = -0.501;
    let floor_vertices = [
        Vertex {
            position: [-ground_size, ground_y, ground_size],
            color: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [ground_size, ground_y, ground_size],
            color: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [ground_size, ground_y, -ground_size],
            color: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-ground_size, ground_y, -ground_size],
            color: [-1.0, 0.0, 0.0],
        },
    ];
    vertex_buffer.extend_from_slice(&floor_vertices);
    solid_index_buffer.extend_from_slice(&[0, 1, 2, 2, 3, 0]);

    // ── Yellow spawn target cursor (unchanged) ────────────────────────────────
    let cursor_base_index = vertex_buffer.len() as u32;
    let cursor_vertices = [
        Vertex {
            position: [spawn_target_x - 0.5, -0.49, spawn_target_z - 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [spawn_target_x + 0.5, -0.49, spawn_target_z - 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [spawn_target_x + 0.5, -0.49, spawn_target_z + 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [spawn_target_x - 0.5, -0.49, spawn_target_z + 0.5],
            color: [1.0, 1.0, 0.0],
        },
    ];
    vertex_buffer.extend_from_slice(&cursor_vertices);
    border_index_buffer.extend_from_slice(&[
        cursor_base_index,
        cursor_base_index + 1,
        cursor_base_index + 1,
        cursor_base_index + 2,
        cursor_base_index + 2,
        cursor_base_index + 3,
        cursor_base_index + 3,
        cursor_base_index,
    ]);

    // ── Voxel blocks and humanoid segments ────────────────────────────────────
    for (voxel, stress_normalized, is_scaffold, humanoid_segment_index) in voxels.iter() {
        if voxel.position.is_nan() {
            continue;
        }

        let render_position = voxel.position;

        if let Some(segment_index) = humanoid_segment_index {
            // ── SPRINT 5: Humanoid segment rendering ──────────────────────────
            //
            // LEARNING: Each body segment is rendered as a proportionally-scaled box
            // using the segment-specific half-extents from humanoid.rs constants.
            // We hard-code the segment sizes here to avoid adding a dimension field
            // to the Voxel component (which would bloat the hot physics path).
            let half_extents = voxel.half_extents;
            let segment_color = HUMANOID_SEGMENT_COLORS
                .get(*segment_index)
                .copied()
                .unwrap_or([0.5, 0.5, 0.5]);

            push_humanoid_segment_mesh(
                &mut vertex_buffer,
                &mut solid_index_buffer,
                &mut border_index_buffer,
                voxel,
                render_position,
                segment_color,
                half_extents,
            );

            if *segment_index == BodySegment::Torso.index() {
                let head_center =
                    render_position + (voxel.rotation * Vec3::new(0.0, half_extents.y + 0.18, 0.0));
                push_humanoid_segment_mesh(
                    &mut vertex_buffer,
                    &mut solid_index_buffer,
                    &mut border_index_buffer,
                    voxel,
                    head_center,
                    HUMANOID_HEAD_COLOR,
                    Vec3::new(0.17, 0.17, 0.16),
                );
            }
        } else {
            // ── Normal voxel block rendering (unchanged) ──────────────────────
            let color = height_to_color_with_stress(voxel, *stress_normalized, *is_scaffold);

            match voxel.shape {
                ShapeType::Cube => push_cube_mesh(
                    &mut vertex_buffer,
                    &mut solid_index_buffer,
                    &mut border_index_buffer,
                    voxel,
                    render_position,
                    color,
                ),
                ShapeType::Wedge => push_wedge_mesh(
                    &mut vertex_buffer,
                    &mut solid_index_buffer,
                    &mut border_index_buffer,
                    voxel,
                    render_position,
                    color,
                ),
                ShapeType::Sphere => push_sphere_mesh(
                    &mut vertex_buffer,
                    &mut solid_index_buffer,
                    &mut border_index_buffer,
                    voxel,
                    render_position,
                    color,
                ),
            }
        }
    }

    (vertex_buffer, solid_index_buffer, border_index_buffer)
}

// =============================================================================
// RenderContext: owns all WGPU state + egui state + Sprint 5 agent panel state
// =============================================================================
pub struct RenderContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub window: Arc<Window>,
    pub config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    border_pipeline: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    pub camera_buf: wgpu::Buffer,
    depth_view: wgpu::TextureView,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,
    last_cursor_pixel_position: Option<(f32, f32)>,
    pub spawn_x: f32,
    pub spawn_z: f32,
    sphere_radius: f32,
    selected_material_id: u8,
    show_stress_heatmap: bool,
    pub camera: FreecamState,

    // ── SPRINT 5: Agent control panel state ──────────────────────────────────
    pub agent_panel: AgentPanelState,
}

impl RenderContext {
    pub fn new(event_loop: &EventLoop<()>) -> Option<Self> {
        pollster::block_on(Self::initialize_wgpu(event_loop))
    }

    async fn initialize_wgpu(event_loop: &EventLoop<()>) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Operation Babel: World Builder")
                        .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 768.0)),
                )
                .unwrap(),
        );
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                }],
                label: None,
            });

        let camera = FreecamState::default();
        let aspect_ratio = config.width as f32 / config.height as f32;
        let initial_mvp = camera.build_mvp(aspect_ratio);
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buf"),
            contents: bytemuck::cast_slice(&[initial_mvp]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
            label: None,
        });

        let depth_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("3D_Solid_Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let border_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("3D_Wire_Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_border"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            &window,
            None,
            None,
            None,
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        Some(Self {
            device,
            queue,
            surface,
            window,
            config,
            render_pipeline,
            border_pipeline,
            camera_bind_group,
            camera_buf,
            depth_view,
            egui_ctx,
            egui_state,
            egui_renderer,
            last_cursor_pixel_position: None,
            spawn_x: 0.0,
            spawn_z: 0.0,
            sphere_radius: 1.0,
            selected_material_id: 0,
            show_stress_heatmap: true,
            camera,
            agent_panel: AgentPanelState::default(),
        })
    }

    pub fn handle_window_event(&mut self, event: &winit::event::WindowEvent) {
        let _ = self.egui_state.on_window_event(&self.window, event);

        if let winit::event::WindowEvent::CursorMoved { position, .. } = event {
            self.last_cursor_pixel_position = Some((position.x as f32, position.y as f32));
        }

        if let winit::event::WindowEvent::MouseInput { state, button, .. } = event {
            if *button == MouseButton::Right {
                self.camera.set_rmb(*state == ElementState::Pressed);
            }
        }

        if let winit::event::WindowEvent::MouseWheel { delta, .. } = event {
            if !self.egui_ctx.wants_pointer_input() {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera.scroll(scroll);
            }
        }

        if let winit::event::WindowEvent::KeyboardInput {
            event: key_event, ..
        } = event
        {
            if !self.egui_ctx.wants_keyboard_input() {
                self.camera.key_event(key_event);
            }
        }

        if let winit::event::WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } = event
        {
            if !self.egui_ctx.wants_pointer_input() {
                if let Some((pixel_x, pixel_y)) = self.last_cursor_pixel_position {
                    let viewport_width = self.config.width as f32;
                    let viewport_height = self.config.height as f32;
                    let normalized_device_x = (pixel_x / viewport_width) * 2.0 - 1.0;
                    let normalized_device_y = 1.0 - (pixel_y / viewport_height) * 2.0;

                    let mvp = Mat4::from_cols_array_2d(
                        &self.camera.build_mvp(viewport_width / viewport_height),
                    );
                    let inverse_mvp = mvp.inverse();

                    let near_point = inverse_mvp.project_point3(Vec3::new(
                        normalized_device_x,
                        normalized_device_y,
                        0.0,
                    ));
                    let far_point = inverse_mvp.project_point3(Vec3::new(
                        normalized_device_x,
                        normalized_device_y,
                        1.0,
                    ));
                    let ray_direction = (far_point - near_point).normalize();

                    if ray_direction.y.abs() > 1e-5 {
                        let ray_parameter = -near_point.y / ray_direction.y;
                        let ground_hit = near_point + ray_direction * ray_parameter;
                        self.spawn_x = ground_hit.x.round();
                        self.spawn_z = ground_hit.z.round();
                    }
                }
            }
        }
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.camera.mouse_delta(*dx, *dy);
        }
    }

    // =========================================================================
    // SPRINT 5: render_frame_with_stress — now returns BOTH UiCommand and UiAgentCommand
    // =========================================================================
    //
    // Returns: (block_commands: Vec<UiCommand>, agent_commands: Vec<UiAgentCommand>)
    //
    // The caller (lib.rs render()) processes both vectors after this returns.
    // Block commands spawn/despawn voxel blocks (unchanged behavior).
    // Agent commands spawn/control humanoid agents (Sprint 5 new behavior).
    pub fn render_frame_with_stress(
        &mut self,
        voxels: &[(&Voxel, f32, bool, Option<usize>)],
    ) -> (Vec<UiCommand>, Vec<UiAgentCommand>) {
        let aspect_ratio = self.config.width as f32 / self.config.height as f32;
        let mvp_matrix = self.camera.build_mvp(aspect_ratio);
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&[mvp_matrix]));

        let mut block_commands: Vec<UiCommand> = Vec::new();
        let mut agent_commands: Vec<UiAgentCommand> = Vec::new();

        // ── egui UI pass ──────────────────────────────────────────────────────
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_pass(raw_input);

        // ── Left panel: Block spawner (unchanged) ─────────────────────────────
        egui::Window::new("Babel Engine")
            .resizable(false)
            .show(&self.egui_ctx, |ui| {
                ui.label(format!(
                    "Blocks: {}",
                    voxels.iter().filter(|(_, _, _, seg)| seg.is_none()).count()
                ));
                ui.separator();
                ui.label(format!("Target: [{}, {}]", self.spawn_x, self.spawn_z));
                ui.separator();
                ui.checkbox(&mut self.show_stress_heatmap, "Show Stress Heatmap");
                ui.label("Green=Safe  Yellow=Stressed  Red=Critical");
                ui.separator();
                ui.label("Material:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.selected_material_id, 0, "Wood");
                    ui.selectable_value(&mut self.selected_material_id, 1, "Steel");
                    ui.selectable_value(&mut self.selected_material_id, 2, "Stone");
                });
                if ui.button("Spawn Cube").clicked() {
                    block_commands.push(UiCommand::SpawnCube {
                        x: self.spawn_x,
                        z: self.spawn_z,
                        material_id: self.selected_material_id,
                    });
                }
                if ui.button("Spawn Wedge").clicked() {
                    block_commands.push(UiCommand::SpawnWedge {
                        x: self.spawn_x,
                        z: self.spawn_z,
                        material_id: self.selected_material_id,
                    });
                }
                ui.separator();
                ui.add(egui::Slider::new(&mut self.sphere_radius, 0.5..=4.0).text("Sphere Radius"));
                if ui.button("Spawn Sphere").clicked() {
                    block_commands.push(UiCommand::SpawnSphere {
                        x: self.spawn_x,
                        z: self.spawn_z,
                        radius: self.sphere_radius,
                        material_id: self.selected_material_id,
                    });
                }
                ui.separator();
                ui.label("Scaffold (temporary support):");
                if ui.button("Place Scaffold").clicked() {
                    block_commands.push(UiCommand::SpawnScaffold {
                        x: self.spawn_x,
                        z: self.spawn_z,
                    });
                }
                if ui.button("Remove All Scaffold").clicked() {
                    block_commands.push(UiCommand::DespawnAllScaffold);
                }
            });

        // ── SPRINT 5: Right panel: Agent control ──────────────────────────────
        //
        // LEARNING: egui::SidePanel::right anchors to the right screen edge.
        // The panel width is fixed at 240 pixels — wide enough for sliders
        // and labels, narrow enough to leave the 3D viewport usable.
        // Unlike egui::Window, this panel cannot be moved or minimized,
        // which makes it reliable for agent debugging during training.
        egui::SidePanel::right("agent_control_panel")
            .resizable(false)
            .min_width(250.0)
            .show(&self.egui_ctx, |ui| {
                // ── Section header ────────────────────────────────────────────
                ui.heading("⚙ Agent Control");
                ui.separator();

                // ── Section 1: Spawn / Despawn ────────────────────────────────
                ui.label(format!(
                    "Agents alive: {}",
                    self.agent_panel.display_agent_count
                ));
                ui.horizontal(|ui| {
                    if ui.button("Spawn Worker").clicked() {
                        agent_commands.push(UiAgentCommand::SpawnHumanoid);
                    }
                    if ui.button("Remove All").clicked() {
                        agent_commands.push(UiAgentCommand::DespawnAllHumanoids);
                    }
                });
                ui.separator();

                // ── Section 2: Agent Status ───────────────────────────────────
                //
                // LEARNING: These values are written by lib.rs after each physics step
                // into self.agent_panel. The panel reads them here for display ONLY.
                // No physics logic happens in the renderer — pure visualization.
                ui.label("Agent Status");
                let balance_display = self.agent_panel.display_balance_error;
                let balance_color = if balance_display < 0.3 {
                    egui::Color32::GREEN
                } else if balance_display < 0.7 {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::RED
                };
                ui.colored_label(
                    balance_color,
                    format!("Balance error: {:.2}", balance_display),
                );
                ui.label(format!(
                    "COM height: {:.2}m",
                    self.agent_panel.display_com_height
                ));
                ui.horizontal(|ui| {
                    let left_color = if self.agent_panel.display_left_foot_contact {
                        egui::Color32::GREEN
                    } else {
                        egui::Color32::DARK_GRAY
                    };
                    let right_color = if self.agent_panel.display_right_foot_contact {
                        egui::Color32::GREEN
                    } else {
                        egui::Color32::DARK_GRAY
                    };
                    ui.colored_label(left_color, "L-Foot");
                    ui.colored_label(right_color, "R-Foot");
                });
                ui.separator();

                ui.label("Movement");
                ui.vertical_centered(|ui| {
                    if ui.button("Forward").clicked() {
                        agent_commands
                            .push(UiAgentCommand::MoveActiveHumanoid { direction: Vec3::Z });
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Left").clicked() {
                            agent_commands.push(UiAgentCommand::MoveActiveHumanoid {
                                direction: -Vec3::X,
                            });
                        }
                        if ui.button("Backward").clicked() {
                            agent_commands.push(UiAgentCommand::MoveActiveHumanoid {
                                direction: -Vec3::Z,
                            });
                        }
                        if ui.button("Right").clicked() {
                            agent_commands
                                .push(UiAgentCommand::MoveActiveHumanoid { direction: Vec3::X });
                        }
                    });
                });
                ui.separator();

                // ── Section 3: Pose Presets ───────────────────────────────────
                //
                // LEARNING: The preset dropdown lets the user test all joints at once.
                // Each HumanoidPosePreset defines target angles for all 8 joints.
                // Clicking "Apply" emits one ApplyPosePreset command, which lib.rs
                // translates into 8 individual apply_torque_to_joint() calls.
                ui.label("Pose Presets");
                let current_preset_label = self.agent_panel.selected_preset.display_label();
                egui::ComboBox::from_id_salt("pose_preset_combo")
                    .selected_text(current_preset_label)
                    .show_ui(ui, |combo_ui| {
                        let presets = [
                            HumanoidPosePreset::TeeStance,
                            HumanoidPosePreset::Standing,
                            HumanoidPosePreset::WalkMidStride,
                            HumanoidPosePreset::Squat,
                            HumanoidPosePreset::ArmRaise,
                        ];
                        for preset in presets {
                            let is_selected = preset == self.agent_panel.selected_preset;
                            if combo_ui
                                .selectable_label(is_selected, preset.display_label())
                                .clicked()
                            {
                                self.agent_panel.selected_preset = preset;
                                // Also update the slider values to match the preset.
                                for &(segment, angle_radians) in preset.joint_angles() {
                                    let idx = segment.index();
                                    if idx < 9 {
                                        self.agent_panel.joint_angles_degrees[idx] =
                                            angle_radians.to_degrees();
                                    }
                                }
                            }
                        }
                    });

                if ui.button("Apply Pose").clicked() {
                    agent_commands.push(UiAgentCommand::ApplyPosePreset {
                        preset: self.agent_panel.selected_preset,
                    });
                }
                ui.separator();

                // ── Section 4: Per-Joint Angle Sliders ───────────────────────
                //
                // LEARNING: 8 sliders, one per joint. Each slider:
                //   - Shows the joint name (e.g. "L Thigh")
                //   - Ranges from the joint's anatomical min to max (degrees)
                //   - Emits a SetJointAngle command when the value changes
                //   - Has a "⟲" reset button to return to 0°
                //
                // The sliders operate in DEGREES for user-friendliness. Internally,
                // all joint angles are stored in RADIANS (XPBD uses radians).
                // Conversion: radians = degrees × π/180.
                ui.label("Joint Angles");
                ui.add_space(4.0);

                // Define per-joint slider ranges in degrees.
                // (segment, label, min_degrees, max_degrees)
                let joint_slider_configs: &[(BodySegment, &str, f32, f32)] = &[
                    (BodySegment::LeftThigh, "L Hip", -45.0, 90.0),
                    (BodySegment::LeftShin, "L Knee", -3.0, 120.0),
                    (BodySegment::RightThigh, "R Hip", -45.0, 90.0),
                    (BodySegment::RightShin, "R Knee", -3.0, 120.0),
                    (BodySegment::LeftUpperArm, "L Shoulder", -30.0, 90.0),
                    (BodySegment::LeftForearm, "L Elbow", 0.0, 145.0),
                    (BodySegment::RightUpperArm, "R Shoulder", -30.0, 90.0),
                    (BodySegment::RightForearm, "R Elbow", 0.0, 145.0),
                ];

                egui::ScrollArea::vertical()
                    .id_salt("joint_sliders_scroll")
                    .max_height(400.0)
                    .show(ui, |scroll_ui| {
                        for &(segment, label, min_degrees, max_degrees) in joint_slider_configs {
                            let slider_index = segment.index();
                            let current_degrees =
                                &mut self.agent_panel.joint_angles_degrees[slider_index];

                            scroll_ui.horizontal(|row_ui| {
                                // Joint label: fixed width so sliders align vertically.
                                row_ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new(egui::RichText::new(label).size(11.0)),
                                );

                                // Angle slider.
                                let slider_response = row_ui.add(
                                    egui::Slider::new(current_degrees, min_degrees..=max_degrees)
                                        .suffix("°")
                                        .fixed_decimals(0)
                                        .clamping(egui::SliderClamping::Always),
                                );

                                // Emit command when the slider value changes.
                                if slider_response.changed() {
                                    agent_commands.push(UiAgentCommand::SetJointAngle {
                                        segment,
                                        target_angle_radians: current_degrees.to_radians(),
                                    });
                                }

                                // Reset button: small "⟲" that zeroes the joint.
                                if row_ui.small_button("⟲").clicked() {
                                    *current_degrees = 0.0;
                                    agent_commands.push(UiAgentCommand::SetJointAngle {
                                        segment,
                                        target_angle_radians: 0.0,
                                    });
                                }
                            });
                        }
                    });
            });

        let full_output = self.egui_ctx.end_pass();
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let surface_texture = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                self.surface.get_current_texture().unwrap()
            }
            Err(surface_error) => panic!("Surface error: {:?}", surface_error),
        };
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        for (texture_id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *texture_id, delta);
        }

        // Apply heatmap toggle: zero stress values if disabled.
        let effective_voxels: Vec<(&Voxel, f32, bool, Option<usize>)> = if self.show_stress_heatmap
        {
            voxels.to_vec()
        } else {
            voxels
                .iter()
                .map(|(voxel, _, scaffold, segment)| (*voxel, 0.0, *scaffold, *segment))
                .collect()
        };

        let (vertex_data, solid_indices, border_indices) =
            build_mesh_with_stress(&effective_voxels, self.spawn_x, self.spawn_z);

        let vertex_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let solid_index_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&solid_indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let border_index_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&border_indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.53,
                            g: 0.81,
                            b: 0.92,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buf.slice(..));

            if !solid_indices.is_empty() {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_index_buffer(solid_index_buf.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..solid_indices.len() as u32, 0, 0..1);
            }

            if !border_indices.is_empty() {
                render_pass.set_pipeline(&self.border_pipeline);
                render_pass.set_index_buffer(border_index_buf.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..border_indices.len() as u32, 0, 0..1);
            }
        }

        {
            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.egui_ctx.pixels_per_point(),
            };
            self.egui_renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut command_encoder,
                &paint_jobs,
                &screen_descriptor,
            );
            let mut ui_pass = command_encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer
                .render(&mut ui_pass, &paint_jobs, &screen_descriptor);
            drop(ui_pass);
        }

        self.queue.submit(std::iter::once(command_encoder.finish()));
        surface_texture.present();

        for texture_id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(texture_id);
        }

        (block_commands, agent_commands)
    }
}
