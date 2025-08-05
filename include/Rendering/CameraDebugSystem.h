#pragma once

#include "Rendering/Camera.h"
#include <glm/glm.hpp>
#include <array>

namespace CudaGame {
namespace Rendering {

// Forward declaration
class RenderSystem;

// CameraDebugSystem draws the camera frustum for debugging purposes.
class CameraDebugSystem {
public:
    CameraDebugSystem(RenderSystem* renderSystem) : m_renderSystem(renderSystem) {}

    // Draws the complete view frustum of the given camera (near plane, far plane, and connecting edges).
    void DrawFrustum(const Camera& camera);
    
    // Diagnostic methods for comprehensive camera debugging
    void LogCameraParameters(const Camera& camera);
    void LogFrustumCorners(const std::array<glm::vec3, 8>& corners);

private:
    RenderSystem* m_renderSystem;
    
    // Helper to draw a single line in world space with given color.
    void DrawLine(const glm::vec3& a, const glm::vec3& b, const glm::vec3& color);
};

} // namespace Rendering
} // namespace CudaGame

