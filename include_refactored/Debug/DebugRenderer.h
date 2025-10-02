#pragma once

#include <glm/glm.hpp>
#include <string>

namespace CudaGame {
namespace Debug {

// Interface for debug rendering functionality
class IDebugRenderer {
public:
    virtual ~IDebugRenderer() = default;

    // Core drawing primitives
    virtual void DrawLine(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color) = 0;
    virtual void DrawPoint(const glm::vec3& position, const glm::vec3& color, float size = 5.0f) = 0;
    virtual void DrawSphere(const glm::vec3& center, float radius, const glm::vec3& color) = 0;
    virtual void DrawBox(const glm::vec3& center, const glm::vec3& extents, const glm::vec3& color) = 0;
    
    // Higher-level visualization
    virtual void DrawVector(const glm::vec3& origin, const glm::vec3& vector, const glm::vec3& color) = 0;
    virtual void DrawNormal(const glm::vec3& position, const glm::vec3& normal, float length = 1.0f, const glm::vec3& color = glm::vec3(0.0f, 1.0f, 0.0f)) = 0;
    virtual void DrawPath(const glm::vec3& start, const glm::vec3& direction, float length, const glm::vec3& color) = 0;
    
    // Text and labels
    virtual void DrawText3D(const glm::vec3& position, const std::string& text, const glm::vec3& color) = 0;
    virtual void DrawText2D(const glm::vec2& screenPosition, const std::string& text, const glm::vec3& color) = 0;

    // Debug state control
    virtual void SetDepthTesting(bool enable) = 0;
    virtual void SetLineWidth(float width) = 0;
    virtual void SetPointSize(float size) = 0;
    
    // Frame management
    virtual void BeginFrame() = 0;
    virtual void EndFrame() = 0;
    virtual void Clear() = 0;
};

// Predefined colors for debug visualization
namespace DebugColors {
    const glm::vec3 RED(1.0f, 0.0f, 0.0f);
    const glm::vec3 GREEN(0.0f, 1.0f, 0.0f);
    const glm::vec3 BLUE(0.0f, 0.0f, 1.0f);
    const glm::vec3 YELLOW(1.0f, 1.0f, 0.0f);
    const glm::vec3 MAGENTA(1.0f, 0.0f, 1.0f);
    const glm::vec3 CYAN(0.0f, 1.0f, 1.0f);
    const glm::vec3 WHITE(1.0f, 1.0f, 1.0f);
    const glm::vec3 BLACK(0.0f, 0.0f, 0.0f);
    const glm::vec3 ORANGE(1.0f, 0.5f, 0.0f);
    const glm::vec3 PURPLE(0.5f, 0.0f, 1.0f);
}

} // namespace Debug
} // namespace CudaGame