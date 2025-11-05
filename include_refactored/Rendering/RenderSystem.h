#pragma once

#include "Core/System.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/Camera.h"
#include "Rendering/LightingSystem.h"
#include "Rendering/CudaRenderingSystem.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Framebuffer.h"
#include <memory>
#include <glm/glm.hpp>
#include <cstdint>

namespace CudaGame {
namespace Rendering {

// Forward declarations for graphics API specifics
class ShaderProgram;
class Texture2D;
class Framebuffer;
class CameraDebugSystem;

class RenderSystem : public Core::System {
public:
    RenderSystem();
    ~RenderSystem();

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    void Render();
    
    // Configuration methods for integration
    void Configure() {}
    void LoadShaders() {}

    // Main camera management
    void SetMainCamera(Camera* camera) { 
        m_mainCamera = camera; 
        if (camera) {
            m_depthScale = camera->GetFarPlane() / 4.0f; // Initialize to quarter of far plane
        }
    }
    Camera* GetMainCamera() { return m_mainCamera; }

    // Lighting system integration
    void SetLightingSystem(std::shared_ptr<LightingSystem> lightingSystem) { m_lightingSystem = lightingSystem; }

    // Resource management
    bool LoadModel(const std::string& path);
    bool LoadTexture(const std::string& path);
    std::shared_ptr<ShaderProgram> CreateShader(const std::string& vertexPath, const std::string& fragmentPath);

    // CUDA-OpenGL Interop
    void InitializeCuda();
    void ShutdownCuda();
    void RegisterCudaBuffer(uint32_t bufferId);
    void UnregisterCudaBuffer(uint32_t bufferId);
    void* MapCudaBuffer(uint32_t bufferId);
    void UnmapCudaBuffer(uint32_t bufferId);

    // Render settings
    void SetClearColor(const glm::vec4& color);
    void SetViewport(int x, int y, int width, int height);
    void EnableWireframe(bool enable);
    void EnableVSync(bool enable);

    // Debugging
    void DrawDebugLine(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color);
    void DrawDebugCube(const glm::vec3& center, float size, const glm::vec3& color);
    void DrawDebugFrustum(const Camera::Frustum& frustum, const glm::vec3& color);
    void CycleDebugMode(); // Cycle through G-buffer debug modes
    void ToggleCameraDebug(); // Toggle camera frustum visualization
    float GetDepthScale() const { return m_depthScale; }
    void AdjustDepthScale(float multiplier);

private:
    Camera* m_mainCamera = nullptr;
    glm::vec4 m_clearColor = {0.53f, 0.81f, 0.92f, 1.0f}; // Sky blue background
    std::shared_ptr<LightingSystem> m_lightingSystem;
    
    // PBR Deferred Rendering Pipeline
    std::shared_ptr<Framebuffer> m_gBuffer;
    std::shared_ptr<ShaderProgram> m_geometryPassShader;
    std::shared_ptr<ShaderProgram> m_lightingPassShader;
    float m_depthScale = 50.0f; // Default depth scale for visualization control
    
    // Shadow mapping
    std::shared_ptr<Framebuffer> m_shadowMapFBO;
    std::shared_ptr<ShaderProgram> m_shadowShader;
    uint32_t m_shadowMapTexture;
    int m_shadowMapResolution = 2048;
    glm::mat4 m_lightSpaceMatrix;

    // Post-processing
    std::shared_ptr<Framebuffer> m_postProcessFBO;
    std::shared_ptr<ShaderProgram> m_postProcessShader;

    void GeometryPass();
    void LightingPass();
    void ShadowPass();
    void PostProcessingPass();
    void ForwardPass();
    
    // Render queues for transparent/opaque objects
    std::vector<Core::Entity> m_opaqueRenderQueue;
    std::vector<Core::Entity> m_transparentRenderQueue;
    void SortRenderQueues();
    
    void RenderEntity(Core::Entity entity);
    
    // Fullscreen quad for lighting pass
    uint32_t m_quadVAO = 0;
    uint32_t m_quadVBO = 0;
    void CreateFullscreenQuad();
    void RenderFullscreenQuad();
    
    // Simple cube for basic rendering
    uint32_t m_cubeVAO = 0;
    uint32_t m_cubeVBO = 0;
    void CreateSimpleCube();
    void RenderSimpleCube();
    
    // Dummy texture for unbound material maps
    uint32_t m_dummyTexture = 0;
    int m_debugMode = 0; // Debug mode for visualizing G-buffer
    
    // Camera debug system
    std::unique_ptr<CameraDebugSystem> m_cameraDebugSystem;
    bool m_cameraDebugEnabled = false;
    
    // Frame tracking for diagnostic logging
    uint64_t m_frameID = 0;
    int m_drawCallCount = 0;
    int m_triangleCount = 0;
    
    // Diagnostic helper methods
    void LogFrameStart();
    void LogPassStart(const std::string& passName, uint32_t fbo, int width, int height);
    void LogPassEnd(const std::string& passName, int drawCalls, int triangles);
    void LogDrawCall(const std::string& pass, uint32_t shader, uint32_t vao, const std::string& primitive, int count);
    void LogTextureBinding(const std::string& pass, int unit, uint32_t textureID, const std::string& format);
    void LogGLError(const std::string& location);
    void DumpGLState(const std::string& location);
    void ValidateAndLogCameraState();
};

} // namespace Rendering
} // namespace CudaGame
