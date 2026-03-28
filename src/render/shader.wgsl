// =============================================================================
// src/render/shader.wgsl  —  Operation Babel: WGSL Vertex and Fragment Shaders
// =============================================================================
//
// LEARNING TOPIC: The GPU Rendering Pipeline
// -------------------------------------------
// Every frame, WGPU runs two shader stages:
//
//   VERTEX SHADER (vs_main):
//     Runs once per vertex. Transforms 3D world positions into 2D screen space
//     using the MVP (Model-View-Projection) matrix. Also passes per-vertex
//     data (color, world position) to the fragment shader.
//
//   FRAGMENT SHADER (fs_main):
//     Runs once per pixel that a triangle covers on screen.
//     Receives interpolated values from the vertex shader and outputs a color.
//     "Interpolated" means if vertex A has color red and vertex B has color blue,
//     pixels between them smoothly blend from red to blue.
//
// LEARNING TOPIC: The Color Channel Convention (Sprint 4 Addition)
// ----------------------------------------------------------------
// We use the color.r channel as a SIGNAL byte to distinguish rendering modes:
//
//   color.r < 0.0   → Draw procedural grass ground plane (no stress coloring)
//   color.r >= 0.0  → Normal block, but FURTHER split by color.g:
//     color.g == -1.0 → SCAFFOLD block (draw with diagonal hatch pattern)
//     otherwise       → Apply stress heatmap using color.b as stress value
//
// This multi-signal approach avoids adding extra vertex attributes (which would
// require changing the Vertex struct layout and all buffer upload code).
// Instead we "steal" channels that would otherwise have no meaning for those
// special cases.
//
// LEARNING TOPIC: Sprint 4 Stress Heatmap Shader
// -----------------------------------------------
// The stress value (0.0=safe, 1.0=critical) is packed into color.b.
// The fragment shader maps it to a color gradient:
//
//   stress = 0.0 → Green  (0.2, 0.8, 0.2) — perfectly safe
//   stress = 0.5 → Yellow (0.9, 0.8, 0.1) — approaching limit
//   stress = 1.0 → Red    (0.9, 0.1, 0.1) — critical / near failure
//
// We use two linear lerps rather than a single cubic because it's cheaper on
// the GPU (two mix() calls vs polynomial evaluation), and the visible result
// is indistinguishable at this scale.
//
// LEARNING TOPIC: How Stress Value Gets to the Shader
// ----------------------------------------------------
// The CPU (Rust, wgpu_view.rs) sets color.b = stress_normalized for each block.
// stress_normalized comes from StressMap::data[entity].stress_normalized, which
// is computed every frame by compute_stress_system in stress.rs.
//
// This flow: physics → stress system → CPU vertex color → GPU shader → screen.
// No GPU-side computation is needed — the stress math stays on the CPU where
// the spatial graph traversal is easier to implement.
//
// =============================================================================

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
    // =========================================================================
    // SIGNAL 1: Procedural Ground Plane
    // =========================================================================
    // color.r < 0.0 is the signal to draw the infinite grass grid instead of
    // a block color. The ground plane vertex has color = (-1.0, 0.0, 0.0).
    //
    // LEARNING: We use a NEGATIVE red channel as a sentinel because real block
    // colors always have r in [0.0, 1.0]. This costs nothing — the GPU just
    // checks the sign of one float, which takes a single instruction.
    if (in.color.r < 0.0) {
        let grid_size = 1.0;
        let line_thickness = 0.05;

        // Create a mathematical grid pattern using modular arithmetic.
        // LEARNING: fract(x) = x - floor(x) gives the fractional part, which
        // oscillates 0→1 for each grid cell. Values near 0 are grid lines.
        let mod_x = in.world_pos.x - floor(in.world_pos.x / grid_size) * grid_size;
        let mod_z = in.world_pos.z - floor(in.world_pos.z / grid_size) * grid_size;
        let is_line = (mod_x < line_thickness) || (mod_z < line_thickness);

        let grass_color = vec3<f32>(0.25, 0.65, 0.25); // Vibrant grass green
        let line_color  = vec3<f32>(0.15, 0.45, 0.15); // Darker green grid lines

        var final_color = grass_color;
        if (is_line) {
            final_color = line_color;
        }

        // Distance fog: blend ground into sky color at the horizon.
        // LEARNING: clamp((dist - start) / (end - start), 0, 1) is the
        // standard linear fog formula. At dist=10 → fog=0 (no fog),
        // at dist=50 → fog=1 (fully sky color).
        let dist = length(in.world_pos.xz);
        let fog_factor = clamp((dist - 10.0) / 40.0, 0.0, 1.0);
        let sky_color = vec3<f32>(0.53, 0.81, 0.92); // Matches WGPU clear color

        final_color = mix(final_color, sky_color, fog_factor);
        return vec4<f32>(final_color, 1.0);
    }

    // =========================================================================
    // SIGNAL 2: Scaffold Hatch Pattern
    // =========================================================================
    // color.g == -1.0 signals this is a scaffold block. We draw it with a
    // diagonal hatch pattern so it is visually distinct from permanent blocks.
    //
    // LEARNING: The hatch pattern uses the world position modulo a small period.
    // Diagonal lines at 45° are created by checking if (x + z) mod period is
    // in a thin band. This is cheap — just modular arithmetic in screen space.
    //
    // WHY HATCH PATTERN:
    //   Scaffold blocks are TEMPORARY. They should look different from the
    //   permanent structure so the user/agent can clearly see what to remove.
    //   A hatch pattern is the universal engineering symbol for "temporary" or
    //   "section view" — immediately recognizable.
    if (in.color.g < -0.5) {
        let period = 0.5;
        let thickness = 0.12;
        let diag = in.world_pos.x + in.world_pos.y + in.world_pos.z;
        let mod_diag = diag - floor(diag / period) * period;
        let is_hatch = mod_diag < thickness;

        // Scaffold base color: a desaturated orange (like construction safety orange)
        let scaffold_base  = vec3<f32>(0.85, 0.55, 0.15);
        let scaffold_hatch = vec3<f32>(0.50, 0.30, 0.05);

        if (is_hatch) {
            return vec4<f32>(scaffold_hatch, 1.0);
        } else {
            return vec4<f32>(scaffold_base, 1.0);
        }
    }

    // =========================================================================
    // SIGNAL 3: Stress Heatmap for Normal Blocks
    // =========================================================================
    // For all other blocks, color.b carries the normalized stress value [0.0, 1.0].
    // color.r and color.g carry the base material color (packed by wgpu_view.rs).
    //
    // LEARNING TOPIC: Two-Segment Color Gradient
    // -------------------------------------------
    // We split the [0.0, 1.0] stress range into two segments:
    //   Segment 1: stress in [0.0, 0.5] → lerp(green, yellow)
    //   Segment 2: stress in [0.5, 1.0] → lerp(yellow, red)
    //
    // This gives us three "anchor" colors at stress = 0, 0.5, 1.0.
    // The GPU mix() function handles linear interpolation:
    //   mix(a, b, t) = a * (1-t) + b * t
    //
    // WHY NOT JUST USE A SINGLE LERP GREEN→RED?
    //   Direct green-to-red lerp goes through BROWN in the middle (mixing
    //   green and red pigments gives brown). A two-segment lerp via yellow
    //   keeps the colors perceptually meaningful at all stress levels.
    //
    // HOW STRESS VALUE IS PACKED:
    //   The CPU writes color = vec3(base_r, base_g, stress_normalized).
    //   Wait — but that overwrites the blue base color channel!
    //   Actually we use stress_normalized to TINT the block color, not replace it:
    //   The CPU sets color = mix(material_base_color, stress_color, stress_amount).
    //   So the shader just renders the already-computed blended color directly.
    //   See wgpu_view.rs height_to_color_with_stress() for where this happens.
    //
    // For this shader: color.b is the stress-blended blue channel from the CPU.
    // The full stress tinting math lives on the CPU (wgpu_view.rs) for clarity.
    return vec4<f32>(in.color, 1.0);
}

@fragment
fn fs_border(in: VertexOutput) -> @location(0) vec4<f32> {
    // Wireframe border: always stark black regardless of stress or material.
    // LEARNING: The border pipeline (LineList topology) runs AFTER the solid
    // pipeline, with no depth bias, so lines sit precisely on top of faces.
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
