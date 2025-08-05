#pragma once

#include "Core/System.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/Camera.h"
#include <glm/glm.hpp>
#include <vector>
#include <memory>

namespace CudaGame {
namespace Rendering {

// Forward declaration
class ShaderProgram;

// Manages all dynamic lighting and shadow operations
class LightingSystem : public Core::System {
public:
    LightingSystem();
    ~LightingSystem() override = default;

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Light management
    Core::Entity CreateDirectionalLight(const glm::vec3& direction, const glm::vec3& color, float intensity);
    Core::Entity CreatePointLight(const glm::vec3& position, float radius, const glm::vec3& color, float intensity);
    Core::Entity CreateSpotLight(const glm::vec3& position, const glm::vec3& direction, float innerCutoff, float outerCutoff, float radius, const glm::vec3& color, float intensity);

    // Shadow mapping
    void UpdateShadowMaps(const Camera& mainCamera, ShaderProgram& shadowShader);
    uint32_t GetDirectionalShadowMap() const { return m_directionalShadowMap; }
    uint32_t GetOmnidirectionalShadowMap() const { return m_omnidirectionalShadowMap; }
    glm::mat4 GetLightSpaceMatrix() const { return m_lightSpaceMatrix; }

    // Performance and quality settings
    void SetShadowQuality(int resolution, int cascadeCount, float cascadeDistribution);
    void SetVolumetricLighting(bool enabled, int sampleCount, float intensity);
    void SetScreenSpaceAmbientOcclusion(bool enabled, float radius, float intensity);

    // Global illumination
    void SetGlobalIllumination(bool enabled, float intensity);
    void UpdateLightProbes();
    
    // Debug visualization
    void SetDebugVisualization(bool enable) { m_debugVisualization = enable; }
    void DrawDebugLights();

private:
    // Shadow mapping resources
    uint32_t m_directionalShadowMap = 0;
    uint32_t m_directionalShadowFBO = 0;
    uint32_t m_omnidirectionalShadowMap = 0;
    uint32_t m_omnidirectionalShadowFBO = 0;
    
    // Volumetric lighting resources
    bool m_volumetricLightingEnabled = false;
    uint32_t m_volumeLightTexture = 0;

    // SSAO resources
    bool m_ssaoEnabled = false;
    uint32_t m_ssaoTexture = 0;
    uint32_t m_ssaoNoiseTexture = 0;

    // Global illumination
    bool m_globalIlluminationEnabled = false;
    
    // Debug
    bool m_debugVisualization = false;
    
    // Shadow mapping matrices
    glm::mat4 m_lightSpaceMatrix{1.0f};
    
    // Internal update methods
    void RenderDirectionalShadows(const Camera& mainCamera, ShaderProgram& shadowShader, Core::Entity lightEntity);
    void RenderOmnidirectionalShadows();
    void UpdateVolumetricLighting();
    void UpdateSSAO();
};

} // namespace Rendering
} // namespace CudaGame
