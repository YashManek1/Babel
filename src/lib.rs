use bevy_ecs::prelude::*;
use pyo3::prelude::*;
use std::time::Duration;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
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

#[pyclass(unsendable)]
pub struct BabelEngine {
    ecs_world: World,
    schedule: Schedule,
    event_loop: EventLoop<()>,
    renderer: Option<RenderContext>,
}

struct RenderEventPump<'a> {
    renderer: &'a mut Option<RenderContext>,
}

impl ApplicationHandler for RenderEventPump<'_> {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(renderer) = self.renderer.as_mut() {
            if renderer.window.id() == window_id {
                renderer.handle_window_event(&event);
            }
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
        }
    }

    pub fn step(&mut self) -> PyResult<usize> {
        self.schedule.run(&mut self.ecs_world);
        let mut query = self.ecs_world.query::<&Voxel>();
        let count = query.iter(&self.ecs_world).count();
        Ok(count)
    }

    pub fn render(&mut self) {
        let mut commands_to_execute = Vec::new();

        {
            let mut query = self.ecs_world.query::<&Voxel>();
            let voxels: Vec<&Voxel> = query.iter(&self.ecs_world).collect();
            if let Some(renderer) = &mut self.renderer {
                commands_to_execute = renderer.render_frame(&voxels);
            }
        }

        for cmd in commands_to_execute {
            let drop_height = 10.0;
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
