use crate::world::voxel::{ShapeType, Voxel};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{event_loop::EventLoop, window::Window};

// LEARNING MECHANIC: The Command Pattern
// The UI generates these abstract commands, passing them safely back to the ECS physics thread.
pub enum UiCommand {
    SpawnCube { x: f32, z: f32 },
    SpawnWedge { x: f32, z: f32 },
    SpawnSphere { x: f32, z: f32, radius: f32 },
}

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
                    offset: 12, // Offset of color (3 * 4 bytes)
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

fn build_mvp_matrix(aspect_ratio: f32) -> [[f32; 4]; 4] {
    let projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 100.0);
    let view = Mat4::look_at_rh(Vec3::new(8.0, 8.0, 12.0), Vec3::ZERO, Vec3::Y);
    let mvp = projection * view;
    mvp.to_cols_array_2d()
}

fn push_cube_mesh(
    v: &mut Vec<Vertex>,
    s_indices: &mut Vec<u32>,
    b_indices: &mut Vec<u32>,
    voxel: &Voxel,
    render_pos: Vec3,
) {
    let base = v.len() as u32;
    let local_verts = [
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
    ];

    for &local in local_verts.iter() {
        let world = render_pos + (voxel.rotation * local);
        v.push(Vertex {
            position: world.into(),
            color: [0.70, 0.52, 0.32],
        });
    }

    s_indices.extend(
        [
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7,
            3, 4, 5, 1, 1, 0, 4,
        ]
        .iter()
        .map(|i| i + base),
    );
    b_indices.extend(
        [
            0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7,
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
) {
    let base = v.len() as u32;
    let local_verts = [
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
    ];

    for &local in local_verts.iter() {
        let world = render_pos + (voxel.rotation * local);
        v.push(Vertex {
            position: world.into(),
            color: [0.70, 0.52, 0.32],
        });
    }

    s_indices.extend(
        [
            0, 1, 2, 2, 3, 0, 3, 2, 4, 4, 5, 3, 0, 3, 5, 1, 4, 2, 0, 5, 4, 4, 1, 0,
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
) {
    let base = v.len() as u32;
    let lat_count = 12;
    let lon_count = 18;

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
                color: [0.70, 0.52, 0.32],
            });
        }
    }

    for i in 0..lat_count {
        for j in 0..lon_count {
            let r1 = i * (lon_count + 1);
            let r2 = (i + 1) * (lon_count + 1);
            s_indices.extend([base + r1 + j, base + r2 + j, base + r1 + j + 1]);
            s_indices.extend([base + r1 + j + 1, base + r2 + j, base + r2 + j + 1]);

            // Wireframe borders
            b_indices.extend([base + r1 + j, base + r1 + j + 1]);
            b_indices.extend([base + r1 + j, base + r2 + j]);
        }
    }
}

fn build_mesh(voxels: &[&Voxel], tx: f32, tz: f32) -> (Vec<Vertex>, Vec<u32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut solid_indices = Vec::new();
    let mut border_indices = Vec::new();

    // Floor rendering with procedural grass trigger
    let s = 1000.0;
    let ground_y = -0.501;
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

    // Yellow Target Cursor
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

    for (idx, voxel) in voxels.iter().enumerate() {
        if voxel.position.is_nan() {
            continue;
        }

        // FIX: Epsilon offset based on index stops blocks from visually disappearing when stacked (Z-fighting)
        let render_pos = voxel.position + Vec3::new(0.0, idx as f32 * 0.0001, 0.0);

        match voxel.shape {
            ShapeType::Cube => push_cube_mesh(
                &mut vertices,
                &mut solid_indices,
                &mut border_indices,
                voxel,
                render_pos,
            ),
            ShapeType::Wedge => push_wedge_mesh(
                &mut vertices,
                &mut solid_indices,
                &mut border_indices,
                voxel,
                render_pos,
            ),
            ShapeType::Sphere => push_sphere_mesh(
                &mut vertices,
                &mut solid_indices,
                &mut border_indices,
                voxel,
                render_pos,
            ),
        }
    }
    (vertices, solid_indices, border_indices)
}

pub struct RenderContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub window: Arc<Window>,
    pub config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    border_pipeline: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,
    last_cursor_px: Option<(f32, f32)>,
    spawn_x: f32,
    spawn_z: f32,
    sphere_r: f32,
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

        let mvp = build_mvp_matrix(config.width as f32 / config.height as f32);
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
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

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("3D_Fill_Pipeline"),
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
            egui::viewport::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            config.format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: false,
                predictable_texture_filtering: false,
            },
        );

        Some(Self {
            device,
            queue,
            surface,
            window,
            config,
            render_pipeline,
            border_pipeline,
            camera_bind_group,
            depth_view,
            egui_ctx,
            egui_state,
            egui_renderer,
            last_cursor_px: None,
            spawn_x: 0.0,
            spawn_z: 0.0,
            sphere_r: 1.0,
        })
    }

    pub fn render_frame(&mut self, voxels: &[&Voxel]) -> Vec<UiCommand> {
        let mut commands = Vec::new();
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.egui_ctx.begin_pass(raw_input);

        egui::Window::new("🏗️ Babel World Builder").show(&self.egui_ctx, |ui| {
            ui.label(format!(
                "Target Ground: [{}, {}]",
                self.spawn_x, self.spawn_z
            ));
            if ui.button("Spawn Cube").clicked() {
                commands.push(UiCommand::SpawnCube {
                    x: self.spawn_x,
                    z: self.spawn_z,
                });
            }
            if ui.button("Spawn Wedge").clicked() {
                commands.push(UiCommand::SpawnWedge {
                    x: self.spawn_x,
                    z: self.spawn_z,
                });
            }
            ui.separator();
            ui.add(egui::Slider::new(&mut self.sphere_r, 0.5..=4.0).text("Sphere Radius"));
            if ui.button("Spawn Sphere").clicked() {
                commands.push(UiCommand::SpawnSphere {
                    x: self.spawn_x,
                    z: self.spawn_z,
                    radius: self.sphere_r,
                });
            }
        });

        let full_output = self.egui_ctx.end_pass();
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let output = self.surface.get_current_texture().unwrap();
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
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        let extra_cmds = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // --- PASS 1: 3D Scene ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D_Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
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
                ..Default::default()
            });

            if !s_idx.is_empty() {
                pass.set_pipeline(&self.render_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, v_buf.slice(..));
                pass.set_index_buffer(s_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..s_idx.len() as u32, 0, 0..1);

                if !b_idx.is_empty() {
                    pass.set_pipeline(&self.border_pipeline);
                    pass.set_index_buffer(b_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..b_idx.len() as u32, 0, 0..1);
                }
            }
        }

        // --- PASS 2: Egui Overlay ---
        {
            let mut ui_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("UI_Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                })
                .forget_lifetime();

            self.egui_renderer
                .render(&mut ui_pass, &paint_jobs, &screen_descriptor);
        }

        self.queue.submit(
            extra_cmds
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        output.present();

        for x in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(x);
        }
        commands
    }

    fn cursor_to_ground_grid(&self, cursor_px: (f32, f32)) -> Option<(f32, f32)> {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(8.0, 8.0, 12.0), Vec3::ZERO, Vec3::Y);
        let inv_vp = (proj * view).inverse();

        let nx = (2.0 * cursor_px.0) / self.config.width as f32 - 1.0;
        let ny = 1.0 - (2.0 * cursor_px.1) / self.config.height as f32;

        let near = inv_vp.project_point3(Vec3::new(nx, ny, 0.0));
        let far = inv_vp.project_point3(Vec3::new(nx, ny, 1.0));
        let dir = (far - near).normalize();

        if dir.y.abs() > 1e-6 {
            let t = (-0.5 - near.y) / dir.y;
            let hit = near + dir * t;
            return Some((hit.x.round(), hit.z.round()));
        }
        None
    }

    pub fn handle_window_event(&mut self, event: &winit::event::WindowEvent) {
        let response = self.egui_state.on_window_event(&self.window, event);

        match event {
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                self.last_cursor_px = Some((position.x as f32, position.y as f32));
            }
            winit::event::WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                if !response.consumed
                    && let Some(px) = self.last_cursor_px
                    && let Some((x, z)) = self.cursor_to_ground_grid(px)
                {
                    self.spawn_x = x;
                    self.spawn_z = z;
                }
            }
            _ => {}
        }
    }
}
