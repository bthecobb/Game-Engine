#pragma once

#include "Rendering/Mesh.h"
#include <cstdint>

namespace CudaGame {
namespace Rendering {

// Generate a low-poly humanoid character mesh (CPU-side data)
// Returns a mesh with ~240 triangles
// Character is 2.0 units tall, centered at origin
Mesh CreateLowPolyCharacter();

// Simple GPU upload result for the character
struct CharacterMeshGPU {
    uint32_t vao = 0;
    uint32_t vbo = 0;
    uint32_t ebo = 0;
    uint32_t indexCount = 0;
};

// Create the character VAO/VBO/EBO and return GPU handles
// Note: Requires a valid OpenGL context
CharacterMeshGPU CreateLowPolyCharacterGPU();

} // namespace Rendering
} // namespace CudaGame
