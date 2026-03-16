use crate::world::voxel::{ShapeType, Voxel};
use bevy_ecs::prelude::*;
use glam::Vec3;
use std::collections::HashMap;

#[derive(Resource, Default)]
pub struct SpatialGrid {
    // The map now holds a Vec (list) of entities, preventing overwrites!
    pub map: HashMap<[i32; 3], Vec<(Entity, ShapeType)>>,
}

impl SpatialGrid {
    pub fn world_to_grid(position: Vec3) -> [i32; 3] {
        [
            position.x.round() as i32,
            position.y.round() as i32,
            position.z.round() as i32,
        ]
    }

    pub fn insert(&mut self, position: Vec3, entity: Entity, shape: ShapeType) {
        let grid_pos = Self::world_to_grid(position);
        // Safely appends to the list instead of overwriting the cell
        self.map.entry(grid_pos).or_default().push((entity, shape));
    }

    pub fn get_neighbors(&self, position: Vec3) -> Vec<(Entity, ShapeType, [i32; 3])> {
        let mut neighbors = Vec::new();
        let center = Self::world_to_grid(position);

        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let check_pos = [center[0] + x, center[1] + y, center[2] + z];
                    if let Some(entities) = self.map.get(&check_pos) {
                        for (entity, shape) in entities {
                            neighbors.push((*entity, shape.clone(), check_pos));
                        }
                    }
                }
            }
        }
        neighbors
    }
}

pub fn update_spatial_grid_system(mut grid: ResMut<SpatialGrid>, query: Query<(Entity, &Voxel)>) {
    grid.map.clear();
    for (entity, voxel) in query.iter() {
        grid.insert(voxel.position, entity, voxel.shape.clone());
    }
}
