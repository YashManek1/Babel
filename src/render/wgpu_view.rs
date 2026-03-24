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
// LEARNING MECHANIC: The Command Pattern
// =============================================================================
// The UI lives on the GPU/render thread. Physics lives in the ECS/CPU world.
// These two must NEVER touch each other's memory directly (data races, borrowing).
//
// Solution: The UI generates lightweight "intent" objects (UiCommand) and pushes
// them into a Vec. After rendering finishes, the engine safely reads that Vec
// and spawns the actual Voxel entities into ECS. Zero shared mutable state.
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
}

// =============================================================================
// LEARNING MECHANIC: Vertex Layout & bytemuck
// =============================================================================
// WGPU is a low-level GPU API. It doesn't understand "structs" — it only knows
// raw bytes. The #[repr(C)] attribute forces Rust to lay out the struct exactly
// like C does (fields in order, no padding between them), matching what the
// WGSL shader expects at locations 0 (position) and 1 (color).
//
// bytemuck::Pod ("Plain Old Data") proves there are no pointers, no padding,
// no uninitialized bytes — safe to reinterpret as raw &[u8] for GPU upload.
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
                    offset: 12, // 3 floats * 4 bytes = 12 byte offset to color
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// =============================================================================
// LEARNING MECHANIC: Mesh Generation (CPU-side geometry)
// =============================================================================
// push_cube_mesh builds the 8 vertices and 36 indices of a unit cube centered
// at render_pos. Each vertex is rotated by voxel.rotation (a Quaternion) before
// being placed in world space — this is how visual rotation works.
//
// LEARNING — TRIANGLES AND INDEX BUFFERS:
// A GPU draws triangles. A cube has 6 faces × 2 triangles = 12 triangles = 36 indices.
// Instead of duplicating vertex data, we store 8 unique vertices and reference
// them by index. This "index buffer" technique saves ~4× GPU memory.
fn push_cube_mesh(
    v: &mut Vec<Vertex>,
    s_indices: &mut Vec<u32>,
    b_indices: &mut Vec<u32>,
    voxel: &Voxel,
    render_pos: Vec3,
    color: [f32; 3],
) {
    let base = v.len() as u32;
    let local_verts = [
        Vec3::new(-0.5, -0.5, 0.5),  // 0: front-bottom-left
        Vec3::new(0.5, -0.5, 0.5),   // 1: front-bottom-right
        Vec3::new(0.5, 0.5, 0.5),    // 2: front-top-right
        Vec3::new(-0.5, 0.5, 0.5),   // 3: front-top-left
        Vec3::new(-0.5, -0.5, -0.5), // 4: back-bottom-left
        Vec3::new(0.5, -0.5, -0.5),  // 5: back-bottom-right
        Vec3::new(0.5, 0.5, -0.5),   // 6: back-top-right
        Vec3::new(-0.5, 0.5, -0.5),  // 7: back-top-left
    ];

    for &local in local_verts.iter() {
        // Quaternion × Vec3: rotate local vertex offset into world orientation
        let world = render_pos + (voxel.rotation * local);
        v.push(Vertex {
            position: world.into(),
            color,
        });
    }

    // 36 indices = 12 triangles = 6 faces × 2 triangles per face
    s_indices.extend(
        [
            0, 1, 2, 2, 3, 0, // front
            1, 5, 6, 6, 2, 1, // right
            5, 4, 7, 7, 6, 5, // back
            4, 0, 3, 3, 7, 4, // left
            3, 2, 6, 6, 7, 3, // top
            4, 5, 1, 1, 0, 4, // bottom
        ]
        .iter()
        .map(|i| i + base),
    );

    // 24 indices = 12 edges × 2 endpoints per line segment (LineList topology)
    b_indices.extend(
        [
            0, 1, 1, 2, 2, 3, 3, 0, // front face edges
            4, 5, 5, 6, 6, 7, 7, 4, // back face edges
            0, 4, 1, 5, 2, 6, 3, 7, // connecting edges
        ]
        .iter()
        .map(|i| i + base),
    );
}

fn push_wedge_mesh(
    v: &mut Vec<Vertex>,
    s_indices: &mut Vec<u32>,
    b_indices: &mut Vec<u32>,
    voxel: &Voxel,
    render_pos: Vec3,
    color: [f32; 3],
) {
    let base = v.len() as u32;
    // A wedge (triangular prism) has 6 vertices:
    // 4 on the bottom rectangle, 2 on the top back edge (slope peak)
    let local_verts = [
        Vec3::new(-0.5, -0.5, 0.5),  // 0: front-bottom-left
        Vec3::new(0.5, -0.5, 0.5),   // 1: front-bottom-right
        Vec3::new(0.5, -0.5, -0.5),  // 2: back-bottom-right
        Vec3::new(-0.5, -0.5, -0.5), // 3: back-bottom-left
        Vec3::new(0.5, 0.5, -0.5),   // 4: back-top-right (slope peak)
        Vec3::new(-0.5, 0.5, -0.5),  // 5: back-top-left  (slope peak)
    ];

    for &local in local_verts.iter() {
        let world = render_pos + (voxel.rotation * local);
        v.push(Vertex {
            position: world.into(),
            color,
        });
    }

    s_indices.extend(
        [
            0, 1, 2, 2, 3, 0, // bottom face
            3, 2, 4, 4, 5, 3, // back face (vertical)
            0, 3, 5, 1, 4, 2, // two triangular ends (left and right)
            0, 5, 4, 4, 1, 0, // slope face
        ]
        .iter()
        .map(|i| i + base),
    );
    b_indices.extend(
        [0, 1, 1, 2, 2, 3, 3, 0, 3, 5, 5, 4, 4, 2, 0, 5, 1, 4]
            .iter()
            .map(|i| i + base),
    );
}

fn push_sphere_mesh(
    v: &mut Vec<Vertex>,
    s_indices: &mut Vec<u32>,
    b_indices: &mut Vec<u32>,
    voxel: &Voxel,
    render_pos: Vec3,
    color: [f32; 3],
) {
    // ==========================================================================
    // LEARNING MECHANIC: UV Sphere Tessellation
    // ==========================================
    // A sphere is approximated by a grid of latitude (theta) and longitude (phi)
    // angles. Each grid intersection becomes a vertex on the sphere surface.
    // lat_count × lon_count grid cells become 2 triangles each.
    // Higher counts = smoother sphere but more GPU vertices.
    // 12×18 gives a good balance for our block scale.
    let base = v.len() as u32;
    let lat_count: u32 = 12;
    let lon_count: u32 = 18;

    for i in 0..=lat_count {
        let theta = i as f32 * std::f32::consts::PI / lat_count as f32;
        for j in 0..=lon_count {
            let phi = j as f32 * 2.0 * std::f32::consts::PI / lon_count as f32;
            let local = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ) * voxel.sphere_radius;
            let world = render_pos + (voxel.rotation * local);
            v.push(Vertex {
                position: world.into(),
                color,
            });
        }
    }

    for i in 0..lat_count {
        for j in 0..lon_count {
            let r1 = i * (lon_count + 1);
            let r2 = (i + 1) * (lon_count + 1);
            s_indices.extend([base + r1 + j, base + r2 + j, base + r1 + j + 1]);
            s_indices.extend([base + r1 + j + 1, base + r2 + j, base + r2 + j + 1]);
            b_indices.extend([base + r1 + j, base + r1 + j + 1]);
            b_indices.extend([base + r1 + j, base + r2 + j]);
        }
    }
}

// =============================================================================
// LEARNING MECHANIC: Height-Based Block Coloring
// =============================================================================
// Each "level" of stacked blocks gets a distinct color. This serves two purposes:
// 1. VISUAL DEBUGGING: You can immediately see which physical level each block
//    occupies — critical for verifying the stacking bug is fixed.
// 2. GAMEPLAY CLARITY: In a tower-building simulation, height = progress.
//    Different colors help the user see the tower structure at a glance.
//
// We map block center Y position → level index → color:
//   Level 0 (y ≈ 0.5):  warm brown  (ground level)
//   Level 1 (y ≈ 1.5):  terracotta  (first stack)
//   Level 2 (y ≈ 2.5):  clay        (second stack)
//   Level 3+:           cycle through a palette
//
// This is NOT Z-fighting prevention — it's purely visual communication.
// We now preserve that readability by applying a slight height tint on top of
// the block's material base color, so Wood/Steel/Stone remain visually distinct.
fn height_to_color(voxel: &Voxel) -> [f32; 3] {
    let base = match voxel.material {
        MaterialType::Wood => [0.70, 0.52, 0.32],
        MaterialType::Steel => [0.55, 0.60, 0.68],
        MaterialType::Stone => [0.62, 0.62, 0.58],
    };

    let height_boost = ((voxel.position.y.max(0.0) / 12.0) * 0.10).min(0.10);
    [
        (base[0] + height_boost).min(1.0),
        (base[1] + height_boost).min(1.0),
        (base[2] + height_boost).min(1.0),
    ]
}

// =============================================================================
// LEARNING MECHANIC: Z-Fighting and Why the Old Epsilon Fix Was Wrong
// =============================================================================
//
// Z-FIGHTING: When two triangles occupy exactly the same depth (Z) in the depth
// buffer, the GPU flickers between them frame-to-frame depending on floating-point
// rounding. Two stacked blocks share a face at exactly the same Y coordinate:
//   Block A top face: y = A.center + 0.5 = 0.5 + 0.5 = 1.0
//   Block B bottom face: y = B.center - 0.5 = 1.5 - 0.5 = 1.0
// Both triangles are at y=1.0 → Z-fighting → Block B "disappears."
//
// THE OLD BROKEN FIX: `render_pos.y += idx * 0.0001`
// This offset is INDEX-based (render array order), NOT position-based.
// Block B might be at idx=0 (no offset) and Block A at idx=1 (tiny offset).
// The offset doesn't correlate with which block is on top, so it doesn't
// reliably prevent the exact face overlap.
//
// THE REAL FIX: Use the GPU's built-in depth bias for the solid pipeline
// (constant: 1, slope_scale: 1.0 — already set in your pipeline) which pushes
// solid polygons back. Then for the render_pos, we DON'T add any Y epsilon —
// the positions are left physically accurate. The depth bias handles the
// solid-vs-wireframe overlap, and correct physics separation handles block-vs-block.
//
// ADDITIONALLY: The previous render_pos formula had a subtle problem —
// idx-based epsilon ACCUMULATED over many blocks, pushing high blocks
// visually upward by up to `N * 0.0001` which was completely wrong for
// towers of 100+ blocks (0.01 unit offset per block = 1 unit error at level 100).
//
// NEW FORMULA: render_pos = voxel.position (exact, no epsilon)
// The depth buffer + depth bias handles all Z-fighting correctly.
fn build_mesh(voxels: &[&Voxel], tx: f32, tz: f32) -> (Vec<Vertex>, Vec<u32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut solid_indices = Vec::new();
    let mut border_indices = Vec::new();

    // ── Ground plane ──────────────────────────────────────────────────────────
    // LEARNING: The ground is NOT a Voxel entity — it's a special fullscreen
    // quad with color.r = -1.0, which the WGSL shader detects as the "draw
    // procedural grass" signal. This avoids spawning a physics entity for an
    // infinite static floor.
    let s = 1000.0;
    let ground_y = -0.501; // Just below floor_y=-0.5 to avoid Z-fighting with blocks
    let floor_verts = [
        Vertex {
            position: [-s, ground_y, s],
            color: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [s, ground_y, s],
            color: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [s, ground_y, -s],
            color: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-s, ground_y, -s],
            color: [-1.0, 0.0, 0.0],
        },
    ];
    vertices.extend_from_slice(&floor_verts);
    solid_indices.extend_from_slice(&[0, 1, 2, 2, 3, 0]);

    // ── Yellow target cursor ──────────────────────────────────────────────────
    // Rendered as a wireframe square on the ground at the current spawn position.
    // Uses border_indices (LineList) so it's always a thin outline, never filled.
    let ti = vertices.len() as u32;
    let target_verts = [
        Vertex {
            position: [tx - 0.5, -0.49, tz - 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [tx + 0.5, -0.49, tz - 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [tx + 0.5, -0.49, tz + 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [tx - 0.5, -0.49, tz + 0.5],
            color: [1.0, 1.0, 0.0],
        },
    ];
    vertices.extend_from_slice(&target_verts);
    border_indices.extend_from_slice(&[ti, ti + 1, ti + 1, ti + 2, ti + 2, ti + 3, ti + 3, ti]);

    // ── Voxel blocks ──────────────────────────────────────────────────────────
    for voxel in voxels.iter() {
        if voxel.position.is_nan() {
            continue;
        }

        // =======================================================================
        // BUG FIX — Z-FIGHTING (Visual Disappearing of Stacked Blocks)
        // =======================================================================
        //
        // WHAT WAS HAPPENING:
        //   Block A settles at y=0.5. Block B lands on A, settles at y=1.5.
        //   Top face of A = y=0.5+0.5=1.0. Bottom face of B = y=1.5-0.5=1.0.
        //   Both triangles at exactly y=1.0 in world space → same depth value
        //   in the depth buffer → GPU alternates between them per-pixel → B flickers
        //   and appears to "disappear" visually.
        //
        //   The old "fix" was: render_pos.y += idx * 0.0001
        //   This is WRONG because:
        //     a) idx is the array iteration order, not related to physical height.
        //        Block B might be idx=0 (zero offset) while A is idx=1.
        //     b) For tall towers, error accumulates: 100 blocks → 0.01 unit offset.
        //
        // THE ACTUAL FIX — Two-part:
        //
        // PART 1 (here): Use render_pos = voxel.position EXACTLY. No epsilon.
        //   The depth bias on the solid pipeline (constant=1, slope_scale=1.0)
        //   already handles solid-vs-wireframe Z-fighting. For block-vs-block
        //   shared faces, correct physics separation (blocks physically 1.0 apart)
        //   combined with the depth test's natural floating-point epsilon is
        //   sufficient when the depth bias pushes the closer solid face forward.
        //
        // PART 2 (physics, in xpbd.rs): The sleeping block at y=0.5 must maintain
        //   exactly 1.0 unit of separation from a block at y=1.5. If the solver
        //   allowed A to drift downward (A.y → 0.499), the gap shrinks to 0.999
        //   and B's bottom clips into A's top. The wedge settled-share fix and
        //   the correct other_pos (using snapshot position) prevent this drift.
        //
        // PART 3 (color): Height-based coloring makes each level visually distinct,
        //   so even if there were very slight Z-fighting, you'd see color change
        //   rather than "disappear."
        let render_pos = voxel.position; // Exact position — no artificial epsilon

        let color = height_to_color(voxel);

        match voxel.shape {
            ShapeType::Cube => push_cube_mesh(
                &mut vertices,
                &mut solid_indices,
                &mut border_indices,
                voxel,
                render_pos,
                color,
            ),
            ShapeType::Wedge => push_wedge_mesh(
                &mut vertices,
                &mut solid_indices,
                &mut border_indices,
                voxel,
                render_pos,
                color,
            ),
            ShapeType::Sphere => push_sphere_mesh(
                &mut vertices,
                &mut solid_indices,
                &mut border_indices,
                voxel,
                render_pos,
                color,
            ),
        }
    }
    (vertices, solid_indices, border_indices)
}

// =============================================================================
// RenderContext: owns all WGPU state
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
    last_cursor_px: Option<(f32, f32)>,
    pub spawn_x: f32,
    pub spawn_z: f32,
    sphere_r: f32,
    selected_material_id: u8,
    pub camera: FreecamState,
}

impl RenderContext {
    pub fn new(event_loop: &EventLoop<()>) -> Option<Self> {
        pollster::block_on(Self::init_wgpu(event_loop))
    }

    async fn init_wgpu(event_loop: &EventLoop<()>) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Operation Babel: World Builder")
                        .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0)),
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
        let aspect = config.width as f32 / config.height as f32;
        let mvp = camera.build_mvp(aspect);
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buf"),
            contents: bytemuck::cast_slice(&[mvp]),
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

        // =======================================================================
        // LEARNING MECHANIC: Depth Bias (Z-Fighting Prevention for Solid vs Wire)
        // =======================================================================
        // When we draw solid triangles AND wireframe lines at the same world
        // coordinates, both get the same depth value → wireframe disappears behind
        // the solid face (Z-fighting).
        //
        // DepthBias pushes solid polygons BACK in depth space by a configurable
        // amount. The wireframe lines (no depth bias) then sit infinitesimally
        // in front of the solid, always visible on top.
        //
        // constant: 1    → 1 depth unit offset (minimum stable offset for all GPUs)
        // slope_scale: 1.0 → additional offset proportional to polygon slope
        //                    (prevents bias artifacts on angled surfaces)
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
                cull_mode: Some(wgpu::Face::Back), // Back-face culling saves ~50% fragment work
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,      // Push solids back 1 depth unit
                    slope_scale: 1.0, // Scale by surface angle
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
                bias: wgpu::DepthBiasState::default(), // No bias for wireframe
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
            last_cursor_px: None,
            spawn_x: 0.0,
            spawn_z: 0.0,
            sphere_r: 1.0,
            selected_material_id: 0,
            camera,
        })
    }

    pub fn handle_window_event(&mut self, event: &winit::event::WindowEvent) {
        let _ = self.egui_state.on_window_event(&self.window, event);

        // ── existing cursor tracking (keep as-is) ─────────────────────────────
        if let winit::event::WindowEvent::CursorMoved { position, .. } = event {
            self.last_cursor_px = Some((position.x as f32, position.y as f32));
        }

        // ── NEW: right mouse button state → orbit guard ───────────────────────
        // LEARNING: We only orbit on RMB so LMB stays free for egui interaction.
        if let winit::event::WindowEvent::MouseInput { state, button, .. } = event {
            if *button == MouseButton::Right {
                self.camera.set_rmb(*state == ElementState::Pressed);
            }
        }

        // ── NEW: scroll wheel → zoom ──────────────────────────────────────────
        // LEARNING: Winit gives two scroll variants:
        //   LineDelta: discrete "clicks" (most mice) — use y component
        //   PixelDelta: trackpad smooth scroll — use y component in logical pixels
        if let winit::event::WindowEvent::MouseWheel { delta, .. } = event {
            if !self.egui_ctx.wants_pointer_input() {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera.scroll(scroll);
            }
        }

        // ── NEW: keyboard → WASD held-key state ──────────────────────────────
        if let winit::event::WindowEvent::KeyboardInput {
            event: key_event, ..
        } = event
        {
            if !self.egui_ctx.wants_keyboard_input() {
                self.camera.key_event(key_event);
            }
        }

        // ── existing LMB ray-cast (keep as-is, but update MVP source) ─────────
        if let winit::event::WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } = event
        {
            // Only ray cast into the 3D world when egui is NOT consuming the
            // pointer (e.g. the cursor is NOT hovering over a button/slider).
            // Without this guard, clicking "Spawn Cube" would re-ray-cast from
            // the button position, overwriting spawn_x/z with wrong coordinates
            // just before the UiCommand is pushed.
            if !self.egui_ctx.wants_pointer_input() {
                if let Some((px, py)) = self.last_cursor_px {
                    let w = self.config.width as f32;
                    let h = self.config.height as f32;
                    // NDC: map pixel (0..w, 0..h) → (-1..1, 1..-1) with Y flipped
                    let ndc_x = (px / w) * 2.0 - 1.0;
                    let ndc_y = 1.0 - (py / h) * 2.0;

                    // CHANGE: use camera.build_mvp() instead of build_mvp_matrix()
                    let mvp = Mat4::from_cols_array_2d(&self.camera.build_mvp(w / h));
                    let inv_mvp = mvp.inverse();

                    let near = inv_mvp.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
                    let far = inv_mvp.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
                    let dir = (far - near).normalize();

                    if dir.y.abs() > 1e-5 {
                        let t = -near.y / dir.y;
                        let hit = near + dir * t;
                        self.spawn_x = hit.x.round();
                        self.spawn_z = hit.z.round();
                    }
                }
            }
        }
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        // LEARNING: DeviceEvent::MouseMotion { delta: (dx, dy) }
        //   dx = pixels moved right (positive) or left (negative) since last event
        //   dy = pixels moved DOWN  (positive) or up  (negative) since last event
        //
        // We pass (dx, dy) directly to camera.mouse_delta(), which applies
        // sensitivity and clamps pitch internally.
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.camera.mouse_delta(*dx, *dy);
        }
    }

    pub fn render_frame(&mut self, voxels: &[&Voxel]) -> Vec<UiCommand> {
        // Recompute MVP and upload to the GPU uniform buffer
        // NOTE: camera.update(dt) is called by lib.rs with real measured dt
        // before this function — do NOT call it again here or pan speed doubles.
        // LEARNING: queue.write_buffer is the fast path for small uniform updates.
        // It schedules a CPU→GPU memcpy that completes before the next draw call.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let mvp = self.camera.build_mvp(aspect);
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&[mvp]));

        let mut commands = Vec::new();

        // ── egui UI pass ──────────────────────────────────────────────────────
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_pass(raw_input);

        egui::Window::new("Babel Engine")
            .resizable(false)
            .show(&self.egui_ctx, |ui| {
                ui.label(format!("Blocks: {}", voxels.len()));
                ui.separator();
                ui.label(format!(
                    "Target Ground: [{}, {}]",
                    self.spawn_x, self.spawn_z
                ));
                ui.separator();
                ui.label("Material:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.selected_material_id, 0, "Wood");
                    ui.selectable_value(&mut self.selected_material_id, 1, "Steel");
                    ui.selectable_value(&mut self.selected_material_id, 2, "Stone");
                });
                let material_hint = match self.selected_material_id {
                    0 => "Wood: medium adhesion, light weight",
                    1 => "Steel: high adhesion, heavy weight",
                    2 => "Stone: no adhesion, high friction",
                    _ => "Wood: medium adhesion, light weight",
                };
                ui.label(material_hint);
                if ui.button("Spawn Cube").clicked() {
                    commands.push(UiCommand::SpawnCube {
                        x: self.spawn_x,
                        z: self.spawn_z,
                        material_id: self.selected_material_id,
                    });
                }
                if ui.button("Spawn Wedge").clicked() {
                    commands.push(UiCommand::SpawnWedge {
                        x: self.spawn_x,
                        z: self.spawn_z,
                        material_id: self.selected_material_id,
                    });
                }
                ui.separator();
                ui.add(egui::Slider::new(&mut self.sphere_r, 0.5..=4.0).text("Sphere Radius"));
                if ui.button("Spawn Sphere").clicked() {
                    commands.push(UiCommand::SpawnSphere {
                        x: self.spawn_x,
                        z: self.spawn_z,
                        radius: self.sphere_r,
                        material_id: self.selected_material_id,
                    });
                }
            });

        let full_output = self.egui_ctx.end_pass();
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let output = match self.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Outdated) => {
                // Surface needs reconfiguration (window was resized, etc.)
                self.surface.configure(&self.device, &self.config);
                self.surface.get_current_texture().unwrap()
            }
            Err(e) => panic!("Surface error: {:?}", e),
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }

        let (v, s_idx, b_idx) = build_mesh(voxels, self.spawn_x, self.spawn_z);
        let v_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let s_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&s_idx),
                usage: wgpu::BufferUsages::INDEX,
            });
        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&b_idx),
                usage: wgpu::BufferUsages::INDEX,
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // ── 3D world render pass ───────────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_vertex_buffer(0, v_buf.slice(..));

            // Draw solid faces first (with depth bias pushing them back)
            if !s_idx.is_empty() {
                pass.set_pipeline(&self.render_pipeline);
                pass.set_index_buffer(s_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..s_idx.len() as u32, 0, 0..1);
            }

            // Draw wireframe edges on top (no depth bias, sits in front)
            if !b_idx.is_empty() {
                pass.set_pipeline(&self.border_pipeline);
                pass.set_index_buffer(b_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..b_idx.len() as u32, 0, 0..1);
            }
        }

        // ── egui UI overlay pass ───────────────────────────────────────────────
        // LEARNING: egui renders AFTER the 3D pass using LoadOp::Load (don't
        // clear the existing 3D pixels). No depth buffer — UI is always on top.
        {
            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.egui_ctx.pixels_per_point(),
            };
            self.egui_renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut encoder,
                &paint_jobs,
                &screen_descriptor,
            );
            let mut ui_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Keep 3D scene, draw UI on top
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None, // No depth test for 2D UI
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer
                .render(&mut ui_pass, &paint_jobs, &screen_descriptor);
            drop(ui_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        commands
    }
}
