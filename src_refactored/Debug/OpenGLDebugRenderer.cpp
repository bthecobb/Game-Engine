#include "Debug/OpenGLDebugRenderer.h"
#include "Rendering/RenderDebugSystem.h"
#include <glad/glad.h>

namespace CudaGame {
namespace Debug {

OpenGLDebugRenderer::OpenGLDebugRenderer(std::shared_ptr<Rendering::RenderDebugSystem> renderDebug)
    : m_renderDebug(std::move(renderDebug))
    , m_debugEnabled(true)
    , m_debugLineVertices() {
}

void OpenGLDebugRenderer::DrawLine(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color) {
    m_renderDebug->DrawDebugLine(start, end, color);
}

void OpenGLDebugRenderer::DrawPoint(const glm::vec3& position, const glm::vec3& color, float size) {
    glPointSize(size);
    m_renderDebug->DrawDebugLine(position, position, color); // Use point for now
}

void OpenGLDebugRenderer::DrawSphere(const glm::vec3& center, float radius, const glm::vec3& color) {
    // Draw sphere approximation using lines
    const int segments = 12;
    for (int i = 0; i < segments; i++) {
        float angle1 = (float)i / segments * 2.0f * 3.14159f;
        float angle2 = (float)(i + 1) / segments * 2.0f * 3.14159f;
        
        // XY plane
        m_renderDebug->DrawDebugLine(
            center + glm::vec3(std::cos(angle1) * radius, std::sin(angle1) * radius, 0.0f),
            center + glm::vec3(std::cos(angle2) * radius, std::sin(angle2) * radius, 0.0f),
            color
        );
        
        // XZ plane
        m_renderDebug->DrawDebugLine(
            center + glm::vec3(std::cos(angle1) * radius, 0.0f, std::sin(angle1) * radius),
            center + glm::vec3(std::cos(angle2) * radius, 0.0f, std::sin(angle2) * radius),
            color
        );
        
        // YZ plane
        m_renderDebug->DrawDebugLine(
            center + glm::vec3(0.0f, std::cos(angle1) * radius, std::sin(angle1) * radius),
            center + glm::vec3(0.0f, std::cos(angle2) * radius, std::sin(angle2) * radius),
            color
        );
    }
}

void OpenGLDebugRenderer::DrawBox(const glm::vec3& center, const glm::vec3& extents, const glm::vec3& color) {
    glm::vec3 min = center - extents;
    glm::vec3 max = center + extents;
    m_renderDebug->DrawDebugBox(min, max, color);
}

void OpenGLDebugRenderer::DrawVector(const glm::vec3& origin, const glm::vec3& vector, const glm::vec3& color) {
    const float arrowSize = 0.2f;
    glm::vec3 end = origin + vector;
    m_renderDebug->DrawDebugLine(origin, end, color);
    
    // Draw arrow head
    glm::vec3 direction = glm::normalize(vector);
    glm::vec3 perpA = glm::vec3(-direction.y, direction.x, 0.0f);
    glm::vec3 perpB = glm::cross(direction, perpA);
    
    m_renderDebug->DrawDebugLine(end, end - direction * arrowSize + perpA * arrowSize, color);
    m_renderDebug->DrawDebugLine(end, end - direction * arrowSize - perpA * arrowSize, color);
    m_renderDebug->DrawDebugLine(end, end - direction * arrowSize + perpB * arrowSize, color);
    m_renderDebug->DrawDebugLine(end, end - direction * arrowSize - perpB * arrowSize, color);
}

void OpenGLDebugRenderer::DrawNormal(const glm::vec3& position, const glm::vec3& normal, float length, const glm::vec3& color) {
    glm::vec3 end = position + normal * length;
    m_renderDebug->DrawDebugLine(position, end, color);
}

void OpenGLDebugRenderer::DrawPath(const glm::vec3& start, const glm::vec3& direction, float length, const glm::vec3& color) {
    glm::vec3 end = start + direction * length;
    m_renderDebug->DrawDebugLine(start, end, color);
}

void OpenGLDebugRenderer::DrawText3D(const glm::vec3& position, const std::string& text, const glm::vec3& color) {
    (void)position; (void)text; (void)color; // Unused parameters
    // TODO: Implement 3D text rendering
}

void OpenGLDebugRenderer::DrawText2D(const glm::vec2& screenPosition, const std::string& text, const glm::vec3& color) {
    (void)screenPosition; (void)text; (void)color; // Unused parameters
    // TODO: Implement 2D text rendering
}

void OpenGLDebugRenderer::SetDepthTesting(bool enable) {
    if (enable) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
}

void OpenGLDebugRenderer::SetLineWidth(float width) {
    glLineWidth(width);
}

void OpenGLDebugRenderer::SetPointSize(float size) {
    glPointSize(size);
}

void OpenGLDebugRenderer::BeginFrame() {
    if (!m_debugEnabled) return;
    m_renderDebug->BeginFrame();
}

void OpenGLDebugRenderer::EndFrame() {
    if (!m_debugEnabled) return;
    m_renderDebug->EndFrame();
}

void OpenGLDebugRenderer::Clear() {
    if (!m_debugEnabled) return;
    m_debugLineVertices.clear();
}

} // namespace Debug
} // namespace CudaGame