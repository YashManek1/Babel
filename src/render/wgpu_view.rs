// =============================================================================
// src/render/wgpu_view.rs  —  Operation Babel: WGPU Renderer
// =============================================================================
//
// SPRINT 4 ADDITIONS:
//   - render_frame_with_stress(): new render entry point that takes stress data
//   - height_to_color_with_stress(): stress-aware color computation
//   - Scaffold rendering: diagonal hatch signal via color.g = -1.0
//   - SpawnScaffold UiCommand: new button in the egui toolbar
//   - despawn_scaffolding UI button: removes all scaffold in one click
//
// LEARNING TOPIC: How Stress Gets Into the Render Pipeline
// --------------------------------------------------------
// The stress value for each block comes from StressMap (computed by
// compute_stress_system each frame). lib.rs collects (Voxel, stress, is_scaffold)
// tuples and passes them to render_frame_with_stress(). This function builds the
// vertex buffer with stress-tinted colors and uploads to the GPU.
//
// The path is: StressMap → CPU color computation → Vertex buffer → GPU → screen.
// All math stays on the CPU where spatial graph traversal is straightforward.
// The GPU shader just renders whatever color the CPU computed.
//
// LEARNING TOPIC: The Color Encoding Convention
// ---------------------------------------------
// We pack special rendering modes into the vertex color channels:
//   color.r < 0.0         → Ground plane (procedural grass shader)
//   color.g < -0.5        → Scaffold block (diagonal hatch in shader)
//   otherwise             → Normal block with stress-tinted color
//
// This avoids adding extra vertex attributes (which would require changing the
// Vertex struct layout, buffer creation, and pipeline attribute descriptions).
// The cost is one extra branch in the fragment shader — negligible on modern GPUs.

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
    // SPRINT 4: New UI command for scaffold placement
    SpawnScaffold {
        x: f32,
        z: f32,
    },
    // SPRINT 4: Explicit scaffold removal command (no NaN sentinel hacks)
    DespawnAllScaffold,
}

// =============================================================================
// LEARNING MECHANIC: Vertex Layout & bytemuck
// =============================================================================
// WGPU is a low-level GPU API. It doesn't understand "structs" — it only knows
// raw bytes. The #[repr(C)] attribute forces Rust to lay out the struct exactly
// like C does (fields in order, no padding), matching what the WGSL shader
// expects at locations 0 (position) and 1 (color).
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
                    offset: 12, // 3 floats × 4 bytes = 12 byte offset to color
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
    // LEARNING MECHANIC: UV Sphere Tessellation
    // A sphere is approximated by a grid of latitude (theta) and longitude (phi)
    // angles. 12×18 gives a good balance between smoothness and vertex count.
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
// SPRINT 4: height_to_color_with_stress()
// =============================================================================
//
// LEARNING TOPIC: Stress Heatmap Color Computation
// -------------------------------------------------
// This function computes the final vertex color for each block based on TWO
// inputs: the material base color and the stress level.
//
// STRESS GRADIENT:
//   stress = 0.0 → Green  [0.2, 0.8, 0.2]  — safe, no load
//   stress = 0.5 → Yellow [0.9, 0.8, 0.1]  — stressed, approaching limit
//   stress = 1.0 → Red    [0.9, 0.1, 0.1]  — critical, near failure
//
// We blend between material color and stress color based on stress level.
// LOW stress: mostly material color (you can tell what material it is)
// HIGH stress: mostly stress color (red/yellow warning dominates)
//
// This two-factor blending (material × stress) is better than pure stress
// coloring because:
//   1. You can still see material types at low stress (brown=wood, grey=stone)
//   2. The warning color is visually unmistakable at high stress
//   3. It matches real engineering practice: safe = material color, danger = red
//
// SCAFFOLD SIGNAL:
//   For scaffold blocks, we return a special signal color with color.g = -2.0.
//   The fragment shader detects color.g < -0.5 and draws the hatch pattern.
//   We don't apply stress tinting to scaffold because scaffold stress is always
//   low (ultra-light) and the hatch pattern is more informative.
fn height_to_color_with_stress(
    voxel: &Voxel,
    stress_normalized: f32,
    is_scaffold: bool,
) -> [f32; 3] {
    // SPRINT 4: Scaffold gets the hatch signal — shader handles the pattern.
    // color.r > 0.0 (not ground), color.g = -2.0 (scaffold signal to shader)
    if is_scaffold {
        return [0.001, -2.0, 0.0];
    }

    // Material base color with slight height tinting (same as before Sprint 4)
    let base = match voxel.material {
        MaterialType::Wood => [0.70, 0.52, 0.32],
        MaterialType::Steel => [0.55, 0.60, 0.68],
        MaterialType::Stone => [0.62, 0.62, 0.58],
        MaterialType::Scaffold => [0.85, 0.55, 0.15], // fallback (handled above)
    };
    let height_boost = ((voxel.position.y.max(0.0) / 12.0) * 0.10).min(0.10);
    let base_tinted = [
        (base[0] + height_boost).min(1.0),
        (base[1] + height_boost).min(1.0),
        (base[2] + height_boost).min(1.0),
    ];

    // Stress color gradient: green → yellow → red (two-segment lerp)
    // LEARNING: We evaluate the gradient at the current stress value.
    // Segment 1 [0.0, 0.5]: lerp green→yellow by (stress/0.5)
    // Segment 2 [0.5, 1.0]: lerp yellow→red by ((stress-0.5)/0.5)
    let green = [0.2_f32, 0.85, 0.2];
    let yellow = [0.95_f32, 0.85, 0.1];
    let red = [0.95_f32, 0.1, 0.1];

    let stress_color = if stress_normalized <= 0.5 {
        let t = stress_normalized / 0.5;
        [
            green[0] + (yellow[0] - green[0]) * t,
            green[1] + (yellow[1] - green[1]) * t,
            green[2] + (yellow[2] - green[2]) * t,
        ]
    } else {
        let t = (stress_normalized - 0.5) / 0.5;
        [
            yellow[0] + (red[0] - yellow[0]) * t,
            yellow[1] + (red[1] - yellow[1]) * t,
            yellow[2] + (red[2] - yellow[2]) * t,
        ]
    };

    // Blend material color and stress color.
    // At low stress: mostly material color (blend_t near 0).
    // At high stress: mostly stress color (blend_t near 1).
    //
    // LEARNING: We use a non-linear blend (stress²) so colors stay
    // near their material base for most of the safe range and only
    // sharply shift toward warning colors as stress approaches 1.0.
    // This prevents the whole structure looking slightly yellow all the time.
    let blend_t = stress_normalized * stress_normalized; // quadratic — slow to change, fast near 1.0
    [
        base_tinted[0] + (stress_color[0] - base_tinted[0]) * blend_t,
        base_tinted[1] + (stress_color[1] - base_tinted[1]) * blend_t,
        base_tinted[2] + (stress_color[2] - base_tinted[2]) * blend_t,
    ]
}

// =============================================================================
// SPRINT 4: build_mesh_with_stress()
// =============================================================================
//
// Updated mesh builder that takes (voxel, stress, is_scaffold) tuples.
// The stress value and scaffold flag are passed to height_to_color_with_stress()
// to compute the final vertex color before uploading to the GPU.
fn build_mesh_with_stress(
    voxels: &[(&Voxel, f32, bool)],
    tx: f32,
    tz: f32,
) -> (Vec<Vertex>, Vec<u32>, Vec<u32>) {
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
    // Wireframe square on the ground showing the current spawn target position.
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
    for (voxel, stress_normalized, is_scaffold) in voxels.iter() {
        if voxel.position.is_nan() {
            continue;
        }

        // Exact world position — no epsilon offset.
        // LEARNING: The depth bias on the solid pipeline (wgpu_view.rs pipeline setup)
        // already handles solid-vs-wireframe Z-fighting. For block-vs-block shared
        // faces, the physics engine ensures blocks are separated by exactly 1.0 unit,
        // so shared faces never have identical depth values.
        let render_pos = voxel.position;

        // SPRINT 4: Color now encodes both material and stress (or scaffold hatch signal)
        let color = height_to_color_with_stress(voxel, *stress_normalized, *is_scaffold);

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
    // SPRINT 4: Track whether the stress heatmap is enabled in the UI.
    // When false, blocks use their material base color only.
    // When true, the stress gradient overrides the base color at high stress.
    // This toggle lets users compare structure appearance with/without heatmap.
    show_stress_heatmap: bool,
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

        // LEARNING MECHANIC: Depth Bias (Z-Fighting Prevention for Solid vs Wire)
        // DepthBias pushes solid polygons BACK in depth space so wireframe lines
        // always sit in front of the solid face they outline.
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
                cull_mode: Some(wgpu::Face::Back), // back-face culling saves ~50% fragment work
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
            last_cursor_px: None,
            spawn_x: 0.0,
            spawn_z: 0.0,
            sphere_r: 1.0,
            selected_material_id: 0,
            show_stress_heatmap: true, // SPRINT 4: enabled by default
            camera,
        })
    }

    pub fn handle_window_event(&mut self, event: &winit::event::WindowEvent) {
        let _ = self.egui_state.on_window_event(&self.window, event);

        if let winit::event::WindowEvent::CursorMoved { position, .. } = event {
            self.last_cursor_px = Some((position.x as f32, position.y as f32));
        }

        // Right mouse button state → orbit guard
        // LEARNING: We only orbit on RMB so LMB stays free for egui interaction.
        if let winit::event::WindowEvent::MouseInput { state, button, .. } = event {
            if *button == MouseButton::Right {
                self.camera.set_rmb(*state == ElementState::Pressed);
            }
        }

        // Scroll wheel → zoom
        if let winit::event::WindowEvent::MouseWheel { delta, .. } = event {
            if !self.egui_ctx.wants_pointer_input() {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera.scroll(scroll);
            }
        }

        // Keyboard → WASD held-key state
        if let winit::event::WindowEvent::KeyboardInput {
            event: key_event, ..
        } = event
        {
            if !self.egui_ctx.wants_keyboard_input() {
                self.camera.key_event(key_event);
            }
        }

        // LMB ray-cast: find where the mouse ray hits the ground plane → spawn target
        if let winit::event::WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } = event
        {
            if !self.egui_ctx.wants_pointer_input() {
                if let Some((px, py)) = self.last_cursor_px {
                    let w = self.config.width as f32;
                    let h = self.config.height as f32;
                    let ndc_x = (px / w) * 2.0 - 1.0;
                    let ndc_y = 1.0 - (py / h) * 2.0;

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
        // LEARNING: DeviceEvent::MouseMotion gives RAW DELTA from hardware sensors.
        // Continues updating even at screen edges, no OS acceleration curves.
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.camera.mouse_delta(*dx, *dy);
        }
    }

    // =========================================================================
    // SPRINT 4: render_frame_with_stress() — Stress-aware render entry point
    // =========================================================================
    //
    // LEARNING: This replaces render_frame() to accept stress data alongside
    // voxel data. The stress values are baked into vertex colors on the CPU
    // before upload to the GPU, so the shader is unchanged in its logic —
    // it just renders whatever color the CPU computed.
    //
    // If show_stress_heatmap is false, we pass stress=0.0 for all blocks,
    // which causes height_to_color_with_stress to use pure material colors.
    pub fn render_frame_with_stress(&mut self, voxels: &[(&Voxel, f32, bool)]) -> Vec<UiCommand> {
        // Recompute MVP and upload to the GPU uniform buffer.
        // LEARNING: queue.write_buffer schedules a CPU→GPU memcpy that completes
        // before the next draw call. This is the fast path for small uniforms.
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

                // SPRINT 4: Stress heatmap toggle
                ui.checkbox(&mut self.show_stress_heatmap, "Show Stress Heatmap");
                ui.label("Green=Safe  Yellow=Stressed  Red=Critical");
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

                // SPRINT 4: Scaffold controls
                ui.separator();
                ui.label("Scaffold (temporary support):");
                if ui.button("Place Scaffold").clicked() {
                    commands.push(UiCommand::SpawnScaffold {
                        x: self.spawn_x,
                        z: self.spawn_z,
                    });
                }
                // LEARNING: The "Remove All Scaffold" button is the UI equivalent
                // of the RL agent's despawn_scaffolding() action. We emit a
                // dedicated command variant so physics code never sees NaN
                // coordinates from sentinel-based signaling.
                if ui.button("Remove All Scaffold").clicked() {
                    commands.push(UiCommand::DespawnAllScaffold);
                }
            });

        let full_output = self.egui_ctx.end_pass();
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let output = match self.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Outdated) => {
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

        // SPRINT 4: If heatmap is disabled, zero out stress values so colors
        // revert to pure material colors. The CPU does this before mesh build.
        let effective_voxels: Vec<(&Voxel, f32, bool)> = if self.show_stress_heatmap {
            voxels.to_vec()
        } else {
            voxels
                .iter()
                .map(|(v, _, scaffold)| (*v, 0.0, *scaffold))
                .collect()
        };

        let (v, s_idx, b_idx) =
            build_mesh_with_stress(&effective_voxels, self.spawn_x, self.spawn_z);

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

            if !s_idx.is_empty() {
                pass.set_pipeline(&self.render_pipeline);
                pass.set_index_buffer(s_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..s_idx.len() as u32, 0, 0..1);
            }

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

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        commands
    }
}
