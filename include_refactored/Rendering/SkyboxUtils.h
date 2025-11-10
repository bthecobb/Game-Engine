#pragma once

#include <glm/glm.hpp>
#include <utility>

namespace CudaGame {
namespace Rendering {

// Mapping utilities to support skybox/IBL work without requiring GL context.
// All functions operate in linear color space (no gamma applied).

struct CubemapSample {
    int faceIndex;   // 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
    glm::vec2 uv;    // [0,1] range on the selected face
};

// Map a normalized direction to equirectangular UV in [0,1].
// u = (atan2(z, x) + PI) / (2*PI), v = acos(y) / PI
glm::vec2 DirectionToEquirectUV(const glm::vec3& dir);

// Inverse of DirectionToEquirectUV.
glm::vec3 EquirectUVToDirection(const glm::vec2& uv);

// Map a normalized direction to a cubemap face and UV on that face.
CubemapSample DirectionToCubemap(const glm::vec3& dir);

// Compute mip levels for a square texture of given base size.
// Returns floor(log2(size)) + 1.
int ComputeMipLevels(int size);

// Simple gamma conversions (gamma = 2.2 by default)
glm::vec3 LinearToSRGB(const glm::vec3& linear, float gamma = 2.2f);
glm::vec3 SRGBToLinear(const glm::vec3& srgb, float gamma = 2.2f);

} // namespace Rendering
} // namespace CudaGame
