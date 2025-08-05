#include "Rendering/CameraDebugSystem.h"
#include "Rendering/RenderSystem.h" // For low-level draw calls
#include <array>
#include <iostream>
#include <glm/gtx/string_cast.hpp> // For glm::to_string

namespace CudaGame {
namespace Rendering {

static const glm::vec3 FRUSTUM_COLOR(1.0f, 0.0f, 0.0f); // Red lines
static const glm::vec3 NEAR_PLANE_COLOR(0.0f, 1.0f, 0.0f); // Green for near plane
static const glm::vec3 FAR_PLANE_COLOR(0.0f, 0.0f, 1.0f);  // Blue for far plane

void CameraDebugSystem::DrawLine(const glm::vec3& a, const glm::vec3& b, const glm::vec3& color) {
    if (m_renderSystem) {
        m_renderSystem->DrawDebugLine(a, b, color);
    }
}

void CameraDebugSystem::DrawFrustum(const Camera& camera) {
    // Log camera parameters for console debug
    LogCameraParameters(camera);
    
    // Handle degenerate cases
    float fov = camera.GetFOV();
    float aspect = camera.GetAspectRatio();
    float nearPlane = camera.GetNearPlane();
    float farPlane = camera.GetFarPlane();
    
    if (fov <= 0.0f || aspect <= 0.0f || nearPlane <= 0.0f || farPlane <= nearPlane) {
        std::cerr << "[CameraDebugSystem] ERROR: Invalid camera parameters - FOV: " << fov 
                  << ", Aspect: " << aspect << ", Near: " << nearPlane << ", Far: " << farPlane << std::endl;
        return;
    }

    const glm::mat4 invViewProj = glm::inverse(camera.GetProjectionMatrix() * camera.GetViewMatrix());
    
    // NDC corners: z = -1 → near, z = +1 → far
    std::array<glm::vec4, 8> ndc = {{
        // Near plane (z = -1)
        {-1, -1, -1, 1}, { 1, -1, -1, 1}, { 1,  1, -1, 1}, {-1,  1, -1, 1},
        // Far plane (z = +1)
        {-1, -1,  1, 1}, { 1, -1,  1, 1}, { 1,  1,  1, 1}, {-1,  1,  1, 1}
    }};

    std::array<glm::vec3, 8> worldPts;
    for (int i = 0; i < 8; ++i) {
        // Transform into world, then divide by W
        glm::vec4 p = invViewProj * ndc[i];
        if (abs(p.w) < 1e-6f) {
            std::cerr << "[CameraDebugSystem] WARNING: Near-zero w component at corner " << i << std::endl;
            continue;
        }
        worldPts[i] = glm::vec3(p / p.w);
    }

    // Optional: log all eight points for console debug
    LogFrustumCorners(worldPts);

    // Draw the two quads (near: 0–3, far: 4–7)
    auto drawLoop = [&](int start, const glm::vec3& color, const std::string& planeName) {
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            DrawLine(worldPts[start + i], worldPts[start + j], color);
        }
        std::cout << "[CameraDebugSystem] Drew " << planeName << " plane quad" << std::endl;
    };
    
    drawLoop(0, NEAR_PLANE_COLOR, "near");  // near plane in green
    drawLoop(4, FAR_PLANE_COLOR, "far");    // far plane in blue

    // Connect corresponding corners (near to far)
    for (int i = 0; i < 4; ++i) {
        DrawLine(worldPts[i], worldPts[i + 4], FRUSTUM_COLOR); // connecting edges in red
    }
    
    std::cout << "[CameraDebugSystem] Drew 4 connecting edges" << std::endl;
}

void CameraDebugSystem::LogCameraParameters(const Camera& camera) {
    std::cout << "[CameraDebugSystem] === CAMERA DEBUG INFO ===" << std::endl;
    std::cout << "Position: " << glm::to_string(camera.GetPosition()) << std::endl;
    std::cout << "Rotation: " << glm::to_string(camera.GetRotation()) << std::endl;
    std::cout << "Forward: " << glm::to_string(camera.GetForward()) << std::endl;
    std::cout << "Right: " << glm::to_string(camera.GetRight()) << std::endl;
    std::cout << "Up: " << glm::to_string(camera.GetUp()) << std::endl;
    std::cout << "FOV: " << camera.GetFOV() << "°" << std::endl;
    std::cout << "Aspect Ratio: " << camera.GetAspectRatio() << std::endl;
    std::cout << "Near Plane: " << camera.GetNearPlane() << std::endl;
    std::cout << "Far Plane: " << camera.GetFarPlane() << std::endl;
    
    // Log matrices (first row only to avoid spam)
    glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 proj = camera.GetProjectionMatrix();
    std::cout << "View Matrix [0]: [" << view[0][0] << ", " << view[0][1] << ", " << view[0][2] << ", " << view[0][3] << "]" << std::endl;
    std::cout << "Proj Matrix [0]: [" << proj[0][0] << ", " << proj[0][1] << ", " << proj[0][2] << ", " << proj[0][3] << "]" << std::endl;
    std::cout << "========================================" << std::endl;
}

void CameraDebugSystem::LogFrustumCorners(const std::array<glm::vec3, 8>& corners) {
    static int logCount = 0;
    // Only log every 60 frames to avoid spam
    if (logCount++ % 60 == 0) {
        std::cout << "[CameraDebugSystem] Frustum corners (world space):" << std::endl;
        for (int i = 0; i < 8; ++i) {
            const char* planeNames[] = {"Near BL", "Near BR", "Near TR", "Near TL", 
                                       "Far BL", "Far BR", "Far TR", "Far TL"};
            std::cout << "  " << planeNames[i] << ": " << glm::to_string(corners[i]) << std::endl;
        }
    }
}

} // namespace Rendering
} // namespace CudaGame

