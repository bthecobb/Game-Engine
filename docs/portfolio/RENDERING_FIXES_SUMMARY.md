# Rendering System Fixes and Debug Tools

## Summary of Rendering Improvements

This document outlines the comprehensive rendering fixes and debug tools that have been implemented to address rendering issues in the CudaGame engine.

## Key Components Added

### 1. **RenderDebugSystem** - Comprehensive Rendering Debugger
Located in: `include_refactored/Rendering/RenderDebugSystem.h` and `src_refactored/Rendering/RenderDebugSystem.cpp`

#### Features:
- **Multiple Visualization Modes:**
  - Wireframe mode
  - Normal visualization
  - Depth buffer visualization
  - G-buffer component visualization (position, normal, albedo, specular)
  - Shadow map visualization
  - Overdraw visualization
  - Frustum culling visualization

- **Performance Monitoring:**
  - Real-time FPS tracking
  - Draw call counting
  - Triangle count monitoring
  - Texture bind tracking
  - Shader switch counting
  - Frame time history with min/max/average
  - Performance warnings for bottlenecks

- **OpenGL Debugging:**
  - Automatic GL error checking
  - Framebuffer validation
  - Shader program validation
  - GL state logging
  - Debug callbacks for driver messages
  - Framebuffer dump to file capability

- **Debug Drawing:**
  - Debug lines
  - Debug boxes
  - Debug spheres
  - Debug grids
  - Frustum visualization

### 2. **Debug Texture Shaders**
Located in: `assets/shaders/debug_texture.vert` and `assets/shaders/debug_texture.frag`

Simple shaders for visualizing textures and framebuffer attachments for debugging purposes.

## Integration with RenderSystem

The RenderDebugSystem has been fully integrated into the main RenderSystem with:
- Automatic initialization on startup
- Per-frame tracking and statistics
- Debug overlay rendering
- Performance monitoring integration

## How to Use the Debug System

### From Code:

```cpp
// In your game loop or debug controls:

// Cycle through visualization modes (F1 key recommended)
renderSystem->GetRenderDebugSystem()->CycleVisualizationMode();

// Set specific visualization mode
renderSystem->GetRenderDebugSystem()->SetVisualizationMode(DebugVisualizationMode::DEPTH_BUFFER);

// Check for OpenGL errors
renderSystem->GetRenderDebugSystem()->CheckGLError("After important operation");

// Validate framebuffer
renderSystem->GetRenderDebugSystem()->ValidateFramebuffer("Before rendering");

// Draw debug shapes
renderSystem->GetRenderDebugSystem()->DrawDebugBox(position, size, color);

// Dump framebuffer to file for analysis
renderSystem->GetRenderDebugSystem()->DumpFramebufferToFile(fbo, "debug_output.ppm");
```

### Visualization Modes:

1. **NONE** - Normal rendering
2. **WIREFRAME** - Show mesh wireframes
3. **NORMALS** - Visualize surface normals
4. **DEPTH_BUFFER** - Show depth buffer
5. **GBUFFER_POSITION** - View position buffer
6. **GBUFFER_NORMAL** - View normal buffer
7. **GBUFFER_ALBEDO** - View albedo/color buffer
8. **GBUFFER_SPECULAR** - View specular/metallic buffer
9. **SHADOW_MAP** - Visualize shadow map
10. **OVERDRAW** - Highlight overdraw areas
11. **FRUSTUM_CULLING** - Show frustum culling bounds

### Performance Monitoring:

The system automatically tracks and reports:
- **FPS** - Frames per second (every 60 frames)
- **Frame Time** - Min/max/average over 120 frame window
- **Draw Calls** - Number of draw calls per frame
- **Triangles** - Total triangles rendered
- **Texture Binds** - Number of texture binding operations
- **Shader Switches** - Number of shader program changes

Performance warnings are automatically logged when thresholds are exceeded:
- Draw calls > 1000
- Triangles > 10,000,000
- Frame time > 33.33ms (below 30 FPS)
- Texture binds > 500
- Shader switches > 100

## Common Rendering Issues and Solutions

### Issue 1: Black Screen
**Debug Steps:**
1. Set visualization mode to `GBUFFER_ALBEDO` to check if geometry is being rendered
2. Check framebuffer validation with `ValidateFramebuffer()`
3. Verify shader compilation in console output
4. Check GL errors with debug system

### Issue 2: Flickering/Artifacts
**Debug Steps:**
1. Enable depth buffer visualization to check Z-fighting
2. Check camera state logging in RenderSystem
3. Validate framebuffer attachments
4. Monitor frame time for spikes

### Issue 3: Poor Performance
**Debug Steps:**
1. Check performance statistics overlay
2. Look for performance warnings in console
3. Monitor draw call count
4. Use overdraw visualization to find bottlenecks
5. Check texture bind count for excessive switching

### Issue 4: Missing Objects
**Debug Steps:**
1. Enable wireframe mode to see all geometry
2. Check frustum culling visualization
3. Verify entity count in render system logs
4. Use debug drawing to visualize object bounds

## Build Integration

The debug system has been added to all relevant CMake targets:
- `CudaPhysicsDemo`
- `CudaRenderingDemo`
- `LightingIntegrationDemo`
- `EnhancedGame`
- `Full3DGame`

## Console Output

The debug system provides structured logging in JSON format for easy parsing:
```json
{
  "frame": 1234,
  "GLError": "AfterGeometryPass",
  "code": 1282,
  "name": "GL_INVALID_OPERATION"
}
```

Performance stats are logged periodically:
```
[RenderDebugSystem] Frame Statistics:
  FPS: 60.5
  Frame Time: 16.53ms (min: 15.2ms, max: 18.1ms)
  Draw Calls: 245
  Triangles: 125000
  Texture Binds: 89
  Shader Switches: 12
```

## Future Enhancements

- [ ] ImGui integration for visual debug UI
- [ ] Shader hot-reload support
- [ ] GPU timing with queries
- [ ] Memory usage tracking
- [ ] Texture memory visualization
- [ ] Draw call batching analysis
- [ ] Render graph visualization

## Troubleshooting

If the debug system doesn't initialize:
1. Check OpenGL context is created before initialization
2. Verify GLEW/GLAD is initialized
3. Check for GL_ARB_debug_output extension support
4. Review console for shader compilation errors

For performance issues with debug system:
1. Disable statistics overlay when not needed
2. Use specific visualization modes rather than cycling
3. Disable GL debug callbacks in release builds
4. Reduce frame time history buffer size if needed

---

## Quick Reference

**Key Bindings (Recommended):**
- `F1` - Cycle visualization modes
- `F2` - Toggle statistics overlay
- `F3` - Toggle wireframe
- `F4` - Dump current framebuffer
- `F5` - Toggle camera debug
- `F6` - Clear performance warnings

**Most Useful Debug Commands:**
```cpp
// Quick performance check
m_renderDebugSystem->RenderStatisticsOverlay();

// Validate rendering pipeline
m_renderDebugSystem->ValidateFramebuffer("MainPass");
m_renderDebugSystem->CheckGLError("AfterDraw");

// Visual debugging
m_renderDebugSystem->SetVisualizationMode(DebugVisualizationMode::DEPTH_BUFFER);
m_renderDebugSystem->DrawDebugBox(boundingBox.min, boundingBox.max, glm::vec3(1,0,0));
```

---

*Last Updated: 2025-08-11*
*Version: 1.0*
