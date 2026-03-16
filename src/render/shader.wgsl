struct CameraUniform {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_pos = model.position;
    out.clip_position = camera.mvp * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // If Red channel is negative, this is our signal to draw the procedural environment!
    if (in.color.r < 0.0) {
        let grid_size = 1.0;
        let line_thickness = 0.05;

        // Create a mathematical checkerboard/grid pattern
        let mod_x = in.world_pos.x - floor(in.world_pos.x / grid_size) * grid_size;
        let mod_z = in.world_pos.z - floor(in.world_pos.z / grid_size) * grid_size;
        let is_line = (mod_x < line_thickness) || (mod_z < line_thickness);

        let grass_color = vec3<f32>(0.25, 0.65, 0.25); // Vibrant Grass
        let line_color = vec3<f32>(0.15, 0.45, 0.15);  // Darker green grid lines

        var final_color = grass_color;
        if (is_line) {
            final_color = line_color;
        }

        // Distance Fog: Blend the ground into the sky smoothly at the horizon
        let dist = length(in.world_pos.xz);
        let fog_factor = clamp((dist - 10.0) / 40.0, 0.0, 1.0); // Starts fading at 10 units, gone at 50
        let sky_color = vec3<f32>(0.53, 0.81, 0.92); // Matches WGPU clear color

        final_color = mix(final_color, sky_color, fog_factor);

        return vec4<f32>(final_color, 1.0);
    }

    // Normal blocks just render their assigned color
    return vec4<f32>(in.color, 1.0);
}

@fragment
fn fs_border(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // Stark black wireframe
}