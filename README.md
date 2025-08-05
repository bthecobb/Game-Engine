# CUDA Particle Game

A real-time particle simulation game built with C++, CUDA, and OpenGL that demonstrates GPU-accelerated physics computation.

## Features

- **CUDA-Accelerated Physics**: Particle updates run in parallel on the GPU using CUDA kernels
- **Real-time Rendering**: OpenGL-based rendering with custom shaders for smooth particle visualization
- **Interactive Controls**: Click and drag to spawn particles at cursor position
- **Realistic Physics**: Gravity, air resistance, and boundary collision detection
- **Visual Effects**: Color fading based on particle lifetime and smooth alpha blending

## Technical Details

### Architecture
- **Game Engine**: Custom C++ game engine with GLFW for window management
- **Graphics**: OpenGL 3.3+ with custom vertex and fragment shaders
- **Compute**: CUDA kernels for particle physics simulation
- **Build System**: CMake with automatic dependency management

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
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 12.9 or compatible)
- CMake (version 3.20+)
- C++ compiler with C++17 support
- OpenGL drivers

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
6. Run the game:
   ```bash
   ./Release/CudaGame.exe  # Windows
   ./CudaGame              # Linux/Mac
   ```

### Dependencies
The build system automatically downloads and builds:
- **GLFW**: Window management and input handling
- **GLAD**: OpenGL function loading

## Controls

- **Left Mouse Button**: Hold and drag to spawn particles
- **ESC**: Exit the game

## Project Structure

```
CudaGame/
├── src/
│   ├── main.cpp              # Entry point
│   ├── GameEngine.cpp        # Game engine implementation
│   └── ParticleSystem.cu     # CUDA particle system
├── include/
│   ├── GameEngine.h          # Game engine header
│   └── ParticleSystem.cuh    # CUDA particle system header
├── build/                    # Build directory (generated)
├── CMakeLists.txt           # CMake configuration
└── README.md               # This file
```

## Performance

The game is designed to handle up to 10,000 particles simultaneously with real-time physics simulation running entirely on the GPU. Performance will vary based on:
- GPU compute capability
- Available VRAM
- Number of active particles

## Customization

### Particle Properties
Modify the `Particle` struct in `ParticleSystem.cuh` to add new properties like:
- Mass
- Different particle types
- Additional forces

### Physics Parameters
Adjust physics constants in the CUDA kernels:
- Gravity strength
- Air resistance
- Boundary behavior
- Particle lifetime

### Visual Effects
Customize the fragment shader to change:
- Particle appearance
- Color schemes
- Blending modes
- Size variations

## Future Enhancements

Potential improvements and extensions:
- Multiple particle systems with different behaviors
- 3D particle simulation
- Particle collision detection
- Texture-based particles
- Advanced lighting effects
- Performance profiling tools
- Particle system editor
- Audio integration

## License

This project is provided for educational and demonstration purposes. Feel free to modify and extend it for your own projects.

## Requirements

- NVIDIA GPU with Compute Capability 7.5+ (RTX 20 series or newer recommended)
- Windows 10/11, Linux, or macOS
- 4GB+ RAM
- OpenGL 3.3+ compatible graphics drivers
