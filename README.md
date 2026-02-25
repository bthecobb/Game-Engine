<div align="center">

# CudaGame Engine 
**A AAA-Grade, GPU-Driven C++ Game Engine**

[![Platform](https://img.shields.io/badge/platforms-Windows-lightgrey)]()
[![Graphics](https://img.shields.io/badge/API-DirectX%2012%20%7C%20CUDA-76B900)]()
[![Physics](https://img.shields.io/badge/Physics-NVIDIA%20PhysX%205-76B900)]()
[![Architecture](https://img.shields.io/badge/Architecture-ECS%20%7C%20Data--Oriented-blue)]()

*CudaGame is a bespoke, data-oriented C++17 engine built to explore modern GPU-driven rendering, procedural biomechanical animation, and seamless CUDA-to-DirectX interop.*

[Quick Start](#-building--running) • [Architecture](#-core-subsystems) • [Demos](#-included-demos) • [For Recruiters](#-portfolio--hiring-managers)

</div>

---

## 🌟 The Philosophy: GPU-Driven & Procedural

CudaGame abandons legacy rendering constructs. It is built strictly around **DirectX 12**, ditching traditional vertex pipelines for **Mesh Shaders (AS/MS)** and **Bindless Rendering**. The CPU's only job is to manage the Entity Component System (ECS) and submit massive arrays of data. The GPU handles the rest via **CUDA-accelerated frustrate/occlusion culling** feeding directly into DX12 `ExecuteIndirect` buffers.

Furthermore, CudaGame rejects pre-baked `.fbx` animation files. The entire animation system—from skeletal hierarchy to biomechanically accurate Walk/Run cycles—is **procedurally generated at startup using pure mathematics**. 

---

## 🏗️ Core Subsystems

### 1. Rendering Pipeline (DirectX 12 Ultimate)
- **Mesh Shaders**: Uses native DX12 Amplification and Mesh Shaders for high-performance geometry processing and meshlet cluster rendering.
- **Bindless Architecture**: Unbounded descriptor heaps and massive texture arrays. Materials retrieve their data entirely without CPU state changes.
- **GPU-Driven Pipeline (`ExecuteIndirect`)**: The CPU binds one command signature. Millions of instances are drawn in a single API call based on GPU-generated command buffers.
- **NVIDIA DLSS 3**: Deep Learning Super Sampling integrated natively into the RenderGraph for AI upscaling.
- **Modern FrameGraph**: Pluggable render passes managing automatic resource barriers, sync points, and transient memory.

### 2. Compute Interop (CUDA)
- **Zero-Copy DX12 Interop**: CUDA maps DirectX 12 generic buffers/textures directly into its memory space via `cudaGraphicsD3D12RegisterResource`.
- **Hardware Culling**: CUDA kernels execute extremely fast frustum and occlusion culling, writing the results directly to the DX12 Indirect Draw buffer.
- **Massive Particle Systems**: Simulation of 100,000+ particles handled entirely on CUDA SMs. 

### 3. Procedural Animation & Biomechanics
- **Math-Only Motion**: Skeletons and animations (Idle, Walk, Run) are generated algebraically.
- **Complex Biomechanics**: The procedural generator bakes complex overlapping sine waves to simulate spine counter-rotation, lateral hip sway, pelvic tracking, ankle push-off (plantar flexion), and Z-axis cross-body arm pumps.
- **Cross-Fade Blending**: A completely data-driven AAA blend tree using low-pass filtered speed parameters to execute smooth, mathematically perfect transitions between states (e.g., Idle → Walk).
- **Deterministic Event Dispatch**: Frame-perfect callbacks (e.g., `Footstep_Right`) fired reliably during blend-tree wraparounds.
- **Inverse Kinematics**: FABRIK and CCD solvers for procedural ground adaptation.

### 4. Physics Engine (NVIDIA PhysX 5)
- **Rigid Body Dynamics**: Full 6DOF physics synced to a fixed timestep, interpolated back to the variable-rate renderer.
- **Character Controller**: Kinematic CCT for precise player movement, jumping, and slope detection.

### 5. Entity Component System (ECS)
- **Data-Oriented Design**: Pure array-of-structs (AoS) managed in contiguous memory pools for maximum CPU cache localization.
- **Lock-Free Patterns**: Predictable execution order across gameplay loops (Input → AI → Physics → Animation → Render).

---

## 🛠️ Building & Running

### Prerequisites
- **Visual Studio 2022** (Desktop development with C++)
- **CMake 3.22+**
- **NVIDIA GPU** (RTX 20-series or higher required for Mesh Shaders / DLSS)
- **CUDA Toolkit 12.x**
- Windows 10/11 (Windows SDK required for DX12)

### Clone & Build
```powershell
# Clone with dependencies
git clone --recursive https://github.com/bthecobb/CudaGame.git
cd CudaGame
mkdir build
cd build

# Generate Visual Studio Solution
cmake .. -G "Visual Studio 17 2022" -A x64

# Build the Release configuration
cmake --build . --config Release --parallel 8
```

---

## 🎮 Included Demos

The repository produces several executable targets demonstrating module isolation:

### `IntegratedAnimationDemo.exe`
The crown jewel of the procedural animation system. Features:
- Orbit camera controller.
- Visualizing a character cross-fading smoothly through Idle, Walk, and Run via WASD.
- Real-time logging of precisely timed footstep events.
- DX12 hardware skeletal compute skinning.

### `DX12PipelineDemo.exe`
The rendering stress test. Features:
- An infinite procedural city generating thousands of buildings.
- Bindless material rendering across millions of polygons.
- CUDA-to-DX12 interop performing frustum culling to keep frame rates high.

---

## 🎯 Portfolio & Hiring Managers

If you are reviewing this repository for a **Graphics Programming**, **Engine Programming**, or **Technical Animation** role, this codebase demonstrates production-level proficiency in:

1. **Low-Level Graphics API**
   - Deep understanding of explicit APIs (DirectX 12), resource barriers, swap chains, and root signatures.
   - Forward-looking rendering paradigms (Mesh Shaders, ExecuteIndirect, Bindless).
2. **Compute & Parallelism**
   - Writing parallel math for the GPU (HLSL Compute / CUDA `__global__` functions).
   - Managing complex hardware synchronization between disparate compute blocks.
3. **Advanced Math & Animation**
   - Matrix/Quaternion transformation chains, dual-quaternion skinning, and inverse kinematic solvers.
   - Biomechanical analysis converted into pure C++ math curves.
4. **Systems Architecture**
   - Designing an ECS that is strictly data-oriented.
   - Large-scale C++17 architectural refactoring, minimizing dependencies, and maximizing cache lines.

### Codebase Entry Points (Where to look)
- **Rendering**: `src_refactored/Rendering/DX12RenderPipeline.cpp`
- **Mesh Shaders**: `assets/shaders/dx12/MeshShader_MS.hlsl`
- **Animation System**: `src_refactored/Animation/AnimationSystem.cpp`
- **Procedural Motion**: `src_refactored/Animation/AnimationBuilder.cpp`
- **CUDA Culling**: `src_refactored/Rendering/CullAndDraw.h`
- **Core ECS**: `src_refactored/Core/Coordinator.h`

<br>

<div align="center">
<b>Built with C++17, DirectX 12, PhysX 5, and CUDA</b><br>
<em>Brandon Cobb</em>
</div>
