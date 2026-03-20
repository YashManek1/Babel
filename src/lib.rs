use bevy_ecs::prelude::*;
use pyo3::prelude::*;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    platform::pump_events::EventLoopExtPumpEvents,
    window::WindowId,
};

mod physics;
mod render;
mod world;

use physics::xpbd::{PhysicsSettings, SolverBuffers};
use render::wgpu_view::{RenderContext, UiCommand};
use world::spatial_grid::SpatialGrid;
use world::voxel::{ShapeType, Voxel};

// Maximum allowed stack height (grid Y). Capping prevents spawns far above
// extremely tall towers which can destabilize the solver. Set to 16 for now.
const MAX_STACK_HEIGHT: i32 = 16;

#[pyclass(unsendable)]
pub struct BabelEngine {
    ecs_world: World,
    schedule: Schedule,
    event_loop: EventLoop<()>,
    renderer: Option<RenderContext>,
    last_frame: Instant,
}

struct RenderEventPump<'a> {
    renderer: &'a mut Option<RenderContext>,
}

impl ApplicationHandler for RenderEventPump<'_> {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(renderer) = self.renderer.as_mut() {
            if renderer.window.id() == window_id {
                if matches!(event, WindowEvent::CloseRequested) {
                    *self.renderer = None;
                    event_loop.exit();
                    std::process::exit(0);
                }
                renderer.handle_window_event(&event);
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // LEARNING: device_event fires for ALL input devices, not just the window.
        // This is where raw mouse delta lives — the delta that continues to
        // report even after the cursor reaches the screen edge.
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.handle_device_event(&event);
        }
    }
}

#[pymethods]
impl BabelEngine {
    #[new]
    pub fn new() -> Self {
        let mut ecs_world = World::new();

        ecs_world.insert_resource(SpatialGrid::default());
        ecs_world.insert_resource(PhysicsSettings::default());
        ecs_world.insert_resource(SolverBuffers::default());

        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                world::spatial_grid::update_spatial_grid_system,
                physics::xpbd::integrate_system,
                physics::xpbd::solve_constraints_system,
                physics::xpbd::update_velocities_system,
                world::spatial_grid::update_spatial_grid_system,
            )
                .chain(),
        );

        let event_loop = EventLoop::new().unwrap();
        let renderer = RenderContext::new(&event_loop);

        Self {
            ecs_world,
            schedule,
            event_loop,
            renderer,
            last_frame: Instant::now(),
        }
    }

    pub fn step(&mut self) -> PyResult<usize> {
        self.schedule.run(&mut self.ecs_world);
        let mut query = self.ecs_world.query::<&Voxel>();
        let count = query.iter(&self.ecs_world).count();
        Ok(count)
    }

    pub fn render(&mut self) {
        let now = Instant::now();
        let dt = self.last_frame.elapsed().as_secs_f32().min(0.1);
        self.last_frame = now;
        let mut commands_to_execute = Vec::new();
        if let Some(renderer) = &mut self.renderer {
            renderer.camera.update(dt);
        }
        {
            let mut query = self.ecs_world.query::<&Voxel>();
            let voxels: Vec<&Voxel> = query.iter(&self.ecs_world).collect();
            if let Some(renderer) = &mut self.renderer {
                commands_to_execute = renderer.render_frame(&voxels);
            }
        }

        for cmd in commands_to_execute {
            // Compute a safe drop height for the spawn column so spawned objects
            // don't intersect tall towers. We query the SpatialGrid for the
            // current highest occupied grid Y at the spawn X/Z and place the
            // new object several units above that. If the column is empty,
            // fall back to the previous default height (10.0).
            let drop_height = {
                // round spawn coords to grid coordinates
                let gx = (match &cmd {
                    UiCommand::SpawnCube { x, .. }
                    | UiCommand::SpawnWedge { x, .. }
                    | UiCommand::SpawnSphere { x, .. } => *x,
                })
                .round() as i32;
                let gz = (match &cmd {
                    UiCommand::SpawnCube { z, .. }
                    | UiCommand::SpawnWedge { z, .. }
                    | UiCommand::SpawnSphere { z, .. } => *z,
                })
                .round() as i32;

                let grid = self.ecs_world.get_resource::<SpatialGrid>().unwrap();
                match grid.column_max_y(gx, gz) {
                    Some(y) => {
                        let capped = y.min(MAX_STACK_HEIGHT);
                        capped as f32 + 5.0 // 5 units above highest block in column (capped)
                    }
                    None => 10.0,
                }
            };

            match cmd {
                UiCommand::SpawnCube { x, z } => {
                    self.ecs_world
                        .spawn(Voxel::new(x, drop_height, z, ShapeType::Cube, false));
                }
                UiCommand::SpawnWedge { x, z } => {
                    self.ecs_world
                        .spawn(Voxel::new(x, drop_height, z, ShapeType::Wedge, false));
                }
                UiCommand::SpawnSphere { x, z, radius } => {
                    self.ecs_world
                        .spawn(Voxel::new_sphere(x, drop_height, z, radius, false));
                }
            }
        }
    }

    pub fn pump_os_events(&mut self) {
        let mut app = RenderEventPump {
            renderer: &mut self.renderer,
        };
        let _ = self
            .event_loop
            .pump_app_events(Some(Duration::ZERO), &mut app);
    }

    pub fn get_debug_state(&mut self) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>, u32)> {
        let mut query = self.ecs_world.query::<&Voxel>();
        let mut result = Vec::new();
        for voxel in query.iter(&self.ecs_world) {
            if voxel.inv_mass > 0.0 {
                result.push((
                    vec![voxel.position.x, voxel.position.y, voxel.position.z],
                    vec![voxel.velocity.x, voxel.velocity.y, voxel.velocity.z],
                    vec![
                        voxel.contact_normal_accum.x,
                        voxel.contact_normal_accum.y,
                        voxel.contact_normal_accum.z,
                    ],
                    voxel.contact_count,
                ));
            }
        }
        result
    }
}

#[pymodule]
fn babel_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BabelEngine>()?;
    Ok(())
}
