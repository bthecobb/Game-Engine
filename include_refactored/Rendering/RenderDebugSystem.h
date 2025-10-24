#pragma once

// Forward declarations for OpenGL types
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef char GLchar;
typedef unsigned int GLbitfield;

// Platform-specific calling convention
#ifdef _WIN32
    #ifndef APIENTRY
        #define APIENTRY __stdcall
    #endif
#else
    #ifndef APIENTRY
        #define APIENTRY
    #endif
#endif

#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "Core/System.h"

namespace CudaGame {
namespace Rendering {

class ShaderProgram;
class Framebuffer;

enum class DebugVisualizationMode {
    NONE = 0,
    WIREFRAME,
    NORMALS,
    DEPTH_BUFFER,
    GBUFFER_POSITION,
    GBUFFER_NORMAL,
    GBUFFER_ALBEDO,
    GBUFFER_SPECULAR,
    SHADOW_MAP,
    OVERDRAW,
    FRUSTUM_CULLING
};

struct RenderStatistics {
    int drawCalls = 0;
    int trianglesRendered = 0;
    int verticesProcessed = 0;
    int textureBinds = 0;
    int shaderSwitches = 0;
    float frameTime = 0.0f;
    float gpuTime = 0.0f;
    float cpuTime = 0.0f;
    int overdrawFactor = 0;
    int culledObjects = 0;
};

class RenderDebugSystem : public Core::System {
public:
    RenderDebugSystem();
    ~RenderDebugSystem();

    // Core system interface
    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Frame-based operations
    void BeginFrame();
    void EndFrame();

    // Debug visualization modes
    void SetVisualizationMode(DebugVisualizationMode mode);
    DebugVisualizationMode GetVisualizationMode() const { return m_currentMode; }
    void CycleVisualizationMode();

    // Render debug overlays
    void RenderDebugOverlay();
    void RenderGBufferVisualization(Framebuffer* gBuffer);
    void RenderDepthBufferVisualization();
    void RenderNormalsVisualization();
    void RenderWireframeMode(bool enable);
    void RenderOverdrawVisualization();
    void RenderShadowMapVisualization(GLuint shadowMap);

    // Statistics and profiling
    void UpdateStatistics(const RenderStatistics& stats);
    void RenderStatisticsOverlay();
    const RenderStatistics& GetStatistics() const { return m_statistics; }
    void IncrementDrawCall() { m_statistics.drawCalls++; }
    void AddTriangles(int count) { m_statistics.trianglesRendered += count; }
    void AddVertices(int count) { m_statistics.verticesProcessed += count; }
    void IncrementTextureBinds() { m_statistics.textureBinds++; }
    void IncrementShaderSwitch() { m_statistics.shaderSwitches++; }

    // OpenGL state validation and debugging
    void ValidateFramebuffer(const std::string& context);
    void CheckGLError(const std::string& context);
    void LogGLState(const std::string& context);
    void ValidateShaderProgram(GLuint program, const std::string& name);
    void DumpFramebufferToFile(GLuint fbo, const std::string& filename);

    // Frustum culling visualization
    void SetFrustumPlanes(const glm::mat4& viewProjection);
    void RenderFrustumBounds();

    // Debug markers and groups (for RenderDoc/NSight)
    void PushDebugGroup(const std::string& name);
    void PopDebugGroup();
    void InsertDebugMarker(const std::string& marker);

    // Debug draw helpers
    void DrawDebugLine(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color);
    void DrawDebugBox(const glm::vec3& min, const glm::vec3& max, const glm::vec3& color);
    void DrawDebugSphere(const glm::vec3& center, float radius, const glm::vec3& color);
    void DrawDebugGrid(float size, int divisions);

    // Performance warnings
    void CheckPerformanceIssues();
    void LogPerformanceWarning(const std::string& warning);

    // Texture debugging
    void VisualizeTexture(GLuint texture, const glm::vec2& position, const glm::vec2& size);
    void CheckTextureCompleteness(GLuint texture, const std::string& name);

    // Shader debugging
    void ReloadShaders();
    void EnableShaderHotReload(bool enable) { m_shaderHotReload = enable; }
    
    // ImGui integration (if available)
    void RenderImGuiDebugWindow();

private:
    // Helper methods
    void CreateDebugShaders();
    void CreateDebugMeshes();
    void RenderFullscreenQuad();
    void UpdateFrameTimeHistory(float frameTime);
    std::string GetVisualizationModeName(DebugVisualizationMode mode);

private:
    DebugVisualizationMode m_currentMode;
    RenderStatistics m_statistics;
    RenderStatistics m_lastFrameStats;
    
    // Debug shaders
    std::shared_ptr<ShaderProgram> m_debugTextureShader;
    std::shared_ptr<ShaderProgram> m_normalsShader;
    std::shared_ptr<ShaderProgram> m_depthShader;
    std::shared_ptr<ShaderProgram> m_wireframeShader;
    std::shared_ptr<ShaderProgram> m_overdrawShader;
    std::shared_ptr<ShaderProgram> m_debugLineShader;

    // Debug geometry
    GLuint m_fullscreenQuadVAO;
    GLuint m_fullscreenQuadVBO;
    GLuint m_debugLineVAO;
    GLuint m_debugLineVBO;
    std::vector<float> m_debugLineVertices;

    // Frame time history for graph
    static constexpr int FRAME_TIME_HISTORY_SIZE = 120;
    float m_frameTimeHistory[FRAME_TIME_HISTORY_SIZE];
    int m_frameTimeIndex;

    // Performance tracking
    float m_avgFrameTime;
    float m_minFrameTime;
    float m_maxFrameTime;
    std::vector<std::string> m_performanceWarnings;

    // Frustum planes for culling visualization
    glm::vec4 m_frustumPlanes[6];

    // Configuration
    bool m_shaderHotReload;
    bool m_showStatistics;
    bool m_showPerformanceWarnings;
    bool m_enableGLDebugOutput;

    // OpenGL debug callback
    static void APIENTRY GLDebugCallback(GLenum source, GLenum type, GLuint id, 
                                         GLenum severity, GLsizei length, 
                                         const GLchar* message, const void* userParam);
};

} // namespace Rendering
} // namespace CudaGame
