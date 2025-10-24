# Asset Management and Resource Loading

## ASSET_DIR
- Defined via CMake per target as a compile definition: ASSET_DIR="<repo>/assets"
- Used by rendering systems and debug systems to load shaders and other resources.

## Shader Loading
- ShaderProgram::LoadFromFiles reads shader sources from ASSET_DIR paths.
- Core shaders expected at:
  - assets/shaders/deferred_geometry.vert, .frag
  - assets/shaders/deferred_lighting.vert, .frag
  - assets/shaders/shadow_mapping.vert, .frag
  - assets/shaders/debug_texture.vert, .frag

## Framebuffer Resources
- G-buffer uses 4 color attachments (Position RGB32F, Normal RGB16F, Albedo RGBA8, MetallicRoughness RGB8) and DepthComponent24.
- Depth format chosen to avoid blit incompatibilities with default framebuffer.

## Mesh/Texture Loading
- Mesh.cpp manages VAOs/VBOs and binds textures; ensure texture paths resolve under assets/.

## Operational Notes
- Ensure assets folder is present alongside executable or ASSET_DIR points to repository assets.
- On packaging, either copy assets folder next to binaries or adjust ASSET_DIR at configure time.
- Shader hot-reload support in debug system expects consistent shader file paths under assets/shaders.
