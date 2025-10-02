#pragma once

#include "Debug/DebugRenderer.h"
#include "Rendering/RenderDebugSystem.h"
#include <memory>
#include <vector>

namespace CudaGame {
namespace Debug {

class OpenGLDebugRenderer : public IDebugRenderer {
public:
    explicit OpenGLDebugRenderer(std::shared_ptr<Rendering::RenderDebugSystem> renderDebug);
    ~OpenGLDebugRenderer() override = default;

    // Core drawing primitives
    void DrawLine(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color) override;
    void DrawPoint(const glm::vec3& position, const glm::vec3& color, float size = 5.0f) override;
    void DrawSphere(const glm::vec3& center, float radius, const glm::vec3& color) override;
    void DrawBox(const glm::vec3& center, const glm::vec3& extents, const glm::vec3& color) override;
    
    // Higher-level visualization
    void DrawVector(const glm::vec3& origin, const glm::vec3& vector, const glm::vec3& color) override;
    void DrawNormal(const glm::vec3& position, const glm::vec3& normal, float length = 1.0f, const glm::vec3& color = glm::vec3(0.0f, 1.0f, 0.0f)) override;
    void DrawPath(const glm::vec3& start, const glm::vec3& direction, float length, const glm::vec3& color) override;
    
    // Text and labels
    void DrawText3D(const glm::vec3& position, const std::string& text, const glm::vec3& color) override;
    void DrawText2D(const glm::vec2& screenPosition, const std::string& text, const glm::vec3& color) override;
    
    // Debug state control
    void SetDepthTesting(bool enable) override;
    void SetLineWidth(float width) override;
    void SetPointSize(float size) override;
    
    // Frame management
    void BeginFrame() override;
    void EndFrame() override;
    void Clear() override;
    
    // Debug control
    void EnableDebugDrawing(bool enable) { m_debugEnabled = enable; }
    bool IsDebugDrawingEnabled() const { return m_debugEnabled; }

private:
    std::shared_ptr<Rendering::RenderDebugSystem> m_renderDebug;
    std::vector<float> m_debugLineVertices;
    bool m_debugEnabled;
};

} // namespace Debug
} // namespace CudaGame