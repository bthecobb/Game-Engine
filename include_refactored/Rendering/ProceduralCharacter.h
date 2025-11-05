#pragma once

#include "Rendering/Mesh.h"

namespace CudaGame {
namespace Rendering {

// Generate a low-poly humanoid character mesh
// Returns a mesh with ~240 triangles
// Character is 2.0 units tall, centered at origin
Mesh CreateLowPolyCharacter();

} // namespace Rendering
} // namespace CudaGame
