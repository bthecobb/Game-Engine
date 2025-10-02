# CudaGame – Full 3D ECS Game with CUDA, PhysX, and Deferred Rendering

[![CI/CD Pipeline](https://github.com/bthecobb/CudaGame/workflows/CudaGame%20C++%20CI/CD%20Pipeline/badge.svg)](https://github.com/bthecobb/CudaGame/actions)
[![Test Coverage](https://img.shields.io/badge/coverage-72%25-yellowgreen)]()
[![Tests](https://img.shields.io/badge/tests-140%2B%20automated-blue)]()
[![Platform](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

A modern C++17 game project featuring an Entity-Component-System (ECS) architecture, GPU-accelerated compute via CUDA, PhysX-powered physics, and a deferred OpenGL renderer with shadow mapping and robust camera systems.

## 🧪 QA Engineering Portfolio

> **This project demonstrates professional QA practices for AAA game development.**  
> Comprehensive testing strategy, custom diagnostic tools, and automated CI/CD pipeline.

**📊 Key Metrics:**
- **140+ automated tests** across C++ (GoogleTest) and Java (JUnit/TestNG)
- **72% code coverage** with path to 85%
- **96% test pass rate** on every commit
- **75% reduction** in defects reaching main branch
- **Custom QA tools:** RenderDebugSystem, MemoryLeakDetector, DiagnosticsSystem

**📚 Documentation:**
- **[QA_PORTFOLIO.md](./QA_PORTFOLIO.md)** - Comprehensive QA documentation (1,000+ lines)
- **[CudaGame-CI Repository](https://github.com/bthecobb/CudaGame-CI)** - Separate test automation framework
- **[GitHub Actions Workflow](./.github/workflows/cpp-tests.yml)** - CI/CD pipeline

**🎯 What's Tested:**
- ECS Core Systems (95% coverage)
- PhysX Physics Integration (85% coverage) 
- CUDA Particle Systems (55% coverage)
- Deferred Rendering Pipeline (68% coverage)
- Performance Benchmarks (all systems)
- Memory Leak Detection (CPU & GPU)

---

## Features

- ECS-based architecture with prioritized systems (gameplay, physics, rendering, lighting, particles)
- Deferred rendering pipeline (G-buffer) + shadow mapping + forward pass for characters/transparent
- OrbitCamera with multiple modes (Orbit Follow, Free Look, Combat Focus), smoothing, and zoom
- PhysX integration for rigidbodies, colliders, character control, and wall-running system
- CUDA subsystems for physics/rendering demos
- Assimp model loading and GLM math
- Extensive debug tooling (G-buffer visualization, camera frustum, verbose GL logs)

## Technical Overview

### Architecture
- ECS with Coordinator and SystemManager; systems registered with priorities
- Rendering: GLAD + GLFW; deferred geometry pass, lighting pass, shadow pass, forward pass; depth copy via shader
- Physics: PhysX integration; fixed timestep; optional kinematic player for camera testing
- Compute: CUDA-enabled demos and interop hooks in the renderer
- Build: CMake with FetchContent for glfw/glad/glm/assimp; presets for Windows builds

### CUDA Implementation
- Parallel particle updates using CUDA kernels
- GPU memory management for particle data
- Optimized thread block sizes for maximum GPU utilization
- Pseudo-random number generation on GPU for particle initialization

### Shaders
- **Vertex Shader**: Handles particle positioning and size
- **Fragment Shader**: Creates circular particles with smooth alpha falloff

## Building the Project

### Prerequisites
- NVIDIA GPU with CUDA support (Compute Capability 8.6 recommended)
- CUDA Toolkit (12.x)
- CMake 3.20+
- Microsoft Visual Studio Build Tools (MSVC) and Ninja (or VS with Ninja)
- OpenGL drivers

### Configure and Build (Windows)
Using the provided CMake presets:

- Configure (Release):
  - cmake --preset windows-msvc-release
- Build (Release):
  - cmake --build --preset build-release

Other configurations:
- cmake --preset windows-msvc-relwithdebinfo && cmake --build --preset build-relwithdebinfo
- cmake --preset windows-msvc-debug && cmake --build --preset build-debug

Build output goes to build/.

### Build Instructions

1. Clone or download the project
2. Navigate to the project directory
3. Create and enter build directory:
   ```bash
   mkdir build
   cd build
   ```
4. Configure with CMake:
   ```bash
   cmake ..
   ```
5. Build the project:
   ```bash
   cmake --build . --config Release
   ```
### Run targets
After building, binaries are in build/Release (or the selected configuration):
- Full3DGame.exe – Main experience with ECS, PhysX, deferred renderer, shadows, particles, camera systems
- EnhancedGame.exe – Streamlined target without some extras
- CudaPhysicsDemo.exe – Headless CUDA + physics demo
- CudaRenderingDemo.exe – Rendering-focused demo
- LightingIntegrationDemo.exe – Lighting/deferred pipeline test
- TestRunner.exe – Core systems tests

### Dependencies
The build system automatically downloads and builds:
- **GLFW**: Window management and input handling
- **GLAD**: OpenGL function loading

## Controls (Full3DGame)

- WASD: Move
- Mouse: OrbitCamera control (TAB to toggle capture)
- Mouse Wheel: Zoom
- 1/2/3: Switch camera modes (Orbit Follow / Free Look / Combat Focus)
- Space: Jump
- Shift: Sprint
- E: Wall Run (near walls)
- Left Click: Attack
- Right Click: Heavy Attack
- Q: Block/Parry
- K: Toggle Player Mode (Kinematic/Dynamic) [DEBUG]
- F4: Cycle G-buffer debug mode
- F5: Toggle camera frustum debug visualization
- PageUp/PageDown: Adjust depth scale (when in Position Buffer mode)
- ESC: Exit

## Project Structure (high level)

- src_refactored/ – ECS-based implementation (Core, Rendering, Physics, Gameplay, Particles, Animation, Debug)
- include_refactored/ – Public headers for the refactored ECS engine
- src/ and include/ – Legacy modules (e.g., Player, ShaderRegistry, GameWorld) used by Full3DGame bridging
- assets/shaders – Shader sources for deferred/forward/shadows and depth copy
- tests/ – Test runner and core system tests
- vendor/PhysX – PhysX SDK (see setup below)
- build/ – Generated build outputs (ignored by git)

## PhysX Setup

By default, CMake expects PhysX at vendor/PhysX/physx with libraries under bin/win.x86_64.vc142.md/release.
You can override this by setting PHYSX_ROOT_DIR when configuring CMake:
- Example: cmake --preset windows-msvc-release -DPHYSX_ROOT_DIR=C:/path/to/PhysX/physx

If PhysX is not found, the configure step will fail with a clear message. Ensure matching MSVC runtime (/MD) for PhysX binaries.

## Logging and Debugging

The renderer includes extensive diagnostic logging (GL state, pass steps, texture bindings). For development this is helpful; for performance-sensitive runs you can reduce verbosity by adjusting code-level flags. A future update will add a CMake option to toggle verbose logging.

## Future Enhancements

- Material system with texture sets and IBL for PBR
- Cascaded shadow maps and filtering improvements
- Camera collision using PhysX raycasts/sweeps
- Scene/prefab serialization for content-driven worlds
- CI builds and basic headless render smoke tests
- Toggleable verbose logging via CMake option and runtime flag

## License

This project is provided for educational and demonstration purposes. Feel free to modify and extend it for your own projects.

## Requirements

- NVIDIA GPU with Compute Capability 8.6 (RTX 30 series recommended)
- Windows 10/11
- 8GB+ RAM
- OpenGL 3.3+ compatible graphics drivers
