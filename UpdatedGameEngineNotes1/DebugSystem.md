# Debugging and Diagnostics Systems

## OpenGL Debug Output
- Enabled when the context has GL_CONTEXT_FLAG_DEBUG_BIT.
- Uses glDebugMessageCallback; logs high/medium severity messages.
- Synchronous mode to align logs with call sites.

## Visualization Modes
- Modes include: NONE, WIREFRAME, NORMALS, DEPTH_BUFFER, GBUFFER_{POSITION,NORMAL,ALBEDO,SPECULAR}, SHADOW_MAP, OVERDRAW, FRUSTUM_CULLING.
- Cycling supported via CycleVisualizationMode.

## Frame Stats & Performance
- Tracks drawCalls, triangles, vertices, textureBinds, shaderSwitches, frameTime.
- Keeps a rolling frame-time history and logs periodic summaries.
- Emits performance warnings for spikes or excessive usage.

## Tools
- Framebuffer validation and dumps (PPM output).
- Texture visualization via fullscreen quad.
- Wireframe toggling and debug line/box drawing utilities.

## Integration Points
- RenderSystem should call BeginFrame/EndFrame and update RenderStatistics each frame.
- Shaders located under assets/shaders; debug_texture shader loaded via ASSET_DIR.
