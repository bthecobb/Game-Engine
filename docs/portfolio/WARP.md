# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a high-performance AAA game engine featuring:
- **Architecture**: Entity-Component-System (ECS) based design for optimal performance
- **GPU Acceleration**: CUDA-powered physics and rendering systems
- **Physics**: NVIDIA PhysX integration with wall-running and character controller
- **Rendering**: Deferred rendering pipeline with PBR, shadow mapping, and post-processing
- **Combat System**: Frame-perfect rhythm-based combat with combo mechanics

## Build Commands

### Prerequisites Check
The project requires:
- CUDA Toolkit 12.x (for RTX 30 series, Compute Capability 8.6)
- Visual Studio 2019+ with MSVC compiler
- CMake 3.20+
- PhysX SDK in `vendor/PhysX/physx`

### Primary Build Commands

**Configure and build (Windows with CMake presets):**
```powershell
# Release build (recommended for performance)
cmake --preset windows-msvc-release
cmake --build --preset build-release

# Debug build (for development/debugging)
cmake --preset windows-msvc-debug
cmake --build --preset build-debug

# RelWithDebInfo (profiling/optimization)
cmake --preset windows-msvc-relwithdebinfo
cmake --build --preset build-relwithdebinfo
```

**Alternative manual build:**
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Executable Targets

After building, executables are in `build/Release/` (or selected configuration):

- **Full3DGame.exe** - Complete game with all systems (ECS, PhysX, deferred renderer, particles, combat)
- **EnhancedGame.exe** - Streamlined version without some debug features
- **CudaPhysicsDemo.exe** - CUDA physics demonstration
- **CudaRenderingDemo.exe** - CUDA rendering effects showcase
- **LightingIntegrationDemo.exe** - Deferred pipeline and lighting test
- **TestRunner.exe** - Unit tests for core systems

### Running Tests
```powershell
cd build/Release
./TestRunner.exe
```

### Building Animation System Separately
```powershell
# Use the provided batch script
./build_animation_system.bat
```

## Architecture & Code Structure

### ECS System Organization

The engine uses prioritized system execution:
- **Priority 50-100**: Physics systems (PhysX, CUDA physics)
- **Priority 100-150**: Gameplay logic (Player movement, Enemy AI, Combat)
- **Priority 150-200**: Animation systems
- **Priority 200-250**: Rendering systems (Deferred, Lighting, Particles)

### Key Source Directories

- **src_refactored/** - Modern ECS implementation
  - `Core/` - EntityManager, ComponentManager, Coordinator
  - `Physics/` - PhysicsSystem, PhysXPhysicsSystem, CudaPhysicsSystem, WallRunningSystem
  - `Rendering/` - RenderSystem, OrbitCamera, MultiLightSystem, deferred pipeline
  - `Gameplay/` - PlayerMovementSystem, CharacterControllerSystem, EnemyAISystem
  - `Particles/` - GPU-accelerated particle system
  - `Animation/` - Animation blending and state machines

- **include_refactored/** - Public headers for ECS systems

- **assets/** - Game resources
  - `shaders/` - GLSL shaders for deferred/forward rendering
  - `models/` - 3D models (accessed via ASSET_DIR macro)

### Core Components

Key components used across systems:
- `TransformComponent` - Position, rotation, scale
- `RigidbodyComponent` - Physics properties (mass, velocity, forces)
- `ColliderComponent` - Collision shapes
- `MeshComponent` - Rendering mesh data
- `AnimationComponent` - Animation state and blending
- `CombatComponent` - Combat state and combo tracking

## Development Workflow

### Adding New Systems

1. Create system in `src_refactored/[Category]/`
2. Register with Coordinator using priority value
3. Add required components to ComponentManager
4. Update CMakeLists.txt to include new source files

### Modifying Shaders

Shaders are in `assets/shaders/`:
- Deferred geometry pass: `deferred_geometry.vert/frag`
- Lighting pass: `deferred_lighting.vert/frag`
- Shadow mapping: `shadow_map.vert/frag`
- Forward pass: `forward.vert/frag`

Changes to shaders are hot-reloaded at runtime.

### PhysX Configuration

PhysX expects libraries in: `vendor/PhysX/physx/bin/win.x86_64.vc142.md/release/`

To use custom PhysX location:
```powershell
cmake --preset windows-msvc-release -DPHYSX_ROOT_DIR=C:/path/to/PhysX/physx
```

### Debugging

**G-buffer visualization** (in Full3DGame):
- F4: Cycle through G-buffer debug modes
- F5: Toggle camera frustum visualization
- PageUp/PageDown: Adjust depth scale

**Performance profiling**:
- Built-in frame timing in render loop
- CUDA profiling via nvprof/NSight

## Common Development Tasks

### Running the Full Game
```powershell
cd build/Release
./Full3DGame.exe
```

### Controls (Full3DGame)
- WASD: Movement
- Mouse: Camera control (TAB toggles capture)
- Space: Jump, Shift: Sprint
- E: Wall run
- Left/Right Click: Attack/Heavy attack
- Q: Block/Parry
- 1/2/3: Camera modes (Orbit/Free/Combat)
- K: Toggle kinematic/dynamic player
- ESC: Exit

### Testing Physics Changes
```powershell
# Run CUDA physics demo
./build/Release/CudaPhysicsDemo.exe
```

### Testing Rendering Changes
```powershell
# Run lighting demo for deferred pipeline testing
./build/Release/LightingIntegrationDemo.exe
```

### Capturing Screenshots
```powershell
# Use provided PowerShell script
cd scripts
./capture_game_screenshot.ps1 -ExecutablePath "../build/Release/Full3DGame.exe"
```

## Performance Considerations

### CUDA Optimization
- Target architecture is set to 86 (RTX 3070 Ti)
- Particle system handles 100,000+ particles on GPU
- Physics system manages 20,000+ entities at 60 FPS

### Memory Management
- ECS uses packed arrays for cache efficiency
- Component pools prevent frequent allocations
- PhysX uses release libraries (debug may have iterator conflicts)

### Rendering Pipeline
- Deferred rendering for multiple lights
- Shadow mapping with depth buffer optimization
- Forward pass for transparent objects
- G-buffer stores: position, normal, albedo, depth

## Project Roadmap References

See `AAA_Development_Pipeline/AAA_DEVELOPMENT_ROADMAP.md` for upcoming features:
- Advanced animation blend trees
- Rhythm-based combat timing
- GPU-accelerated skeletal animation
- Behavior tree AI system
- Level editor with hot-reload

## Known Issues & Workarounds

1. **PhysX Debug Build**: Debug libraries unavailable; using release libs may cause iterator debug level mismatches
2. **Verbose GL Logging**: Currently extensive; future CMake option planned for control
3. **CUDA Compatibility**: Requires `-allow-unsupported-compiler` flag for newer MSVC versions

## External Dependencies

Automatically fetched via CMake:
- GLFW (window management)
- GLAD (OpenGL loader)
- GLM (math library)
- Assimp (model loading)

Manual setup required:
- PhysX SDK (place in vendor/PhysX)
- CUDA Toolkit (system installation)
