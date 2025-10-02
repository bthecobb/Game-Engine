#pragma once

#include "Core/System.h"
#include "Core/ECS_Types.h"
#include "Rendering/RenderComponents.h"  // Contains LightComponent
#include <glm/glm.hpp>
#include <vector>
#include <memory>

namespace CudaGame {
namespace Rendering {

// Forward declarations
class RenderSystem;

/**
 * MultiLightSystem - Advanced lighting system supporting multiple dynamic lights
 * Features:
 * - Multiple directional, point, and spot lights
 * - Light culling based on camera frustum
 * - Shadow mapping for multiple lights
 * - Day/night cycle
 * - Animated lights (flicker, movement)
 * - Light LOD system
 */
class MultiLightSystem : public Core::System {
public:
    // Light data structure for GPU
    struct LightData {
        glm::vec4 position;      // w = 1 for point/spot, 0 for directional
        glm::vec4 direction;     // For directional and spot lights
        glm::vec4 color;         // RGB, w = unused
        glm::vec4 attenuation;   // constant, linear, quadratic, unused
        float intensity;
        float range;             // For point and spot lights
        float innerCone;         // For spot lights (cos of angle)
        float outerCone;         // For spot lights (cos of angle)
        glm::mat4 shadowMatrix;  // For shadow mapping
        int shadowMapIndex;      // Index in shadow map array
        int type;                // 0=directional, 1=point, 2=spot
        float padding[2];        // Padding for alignment
    };
    
    // Internal light representation
    struct Light {
        Core::Entity entity;
        LightType type;
        LightData data;
        
        // Animation properties
        float flickerIntensity = 0.0f;
        float flickerSpeed = 1.0f;
        float moveRadius = 0.0f;
        float moveSpeed = 1.0f;
        glm::vec3 basePosition;
        float baseIntensity;
        
        // Shadow properties
        bool castsShadows = false;
        int shadowMapLayer = -1;
    };
    
public:
    MultiLightSystem();
    ~MultiLightSystem();
    
    bool Initialize() override;
    void Update(float deltaTime) override;
    void Shutdown() override;
    
    // Light creation
    Core::Entity CreateDirectionalLight(const glm::vec3& direction, 
                                       const glm::vec3& color = glm::vec3(1.0f),
                                       float intensity = 1.0f);
    
    Core::Entity CreatePointLight(const glm::vec3& position,
                                 const glm::vec3& color = glm::vec3(1.0f),
                                 float intensity = 1.0f,
                                 float radius = 10.0f);
    
    Core::Entity CreateSpotLight(const glm::vec3& position,
                                const glm::vec3& direction,
                                const glm::vec3& color = glm::vec3(1.0f),
                                float intensity = 1.0f,
                                float innerCone = 30.0f,
                                float outerCone = 45.0f,
                                float range = 20.0f);
    
    // Light modification
    void RemoveLight(Core::Entity entity);
    void SetLightIntensity(Core::Entity entity, float intensity);
    void SetLightColor(Core::Entity entity, const glm::vec3& color);
    void SetLightPosition(Core::Entity entity, const glm::vec3& position);
    void SetLightDirection(Core::Entity entity, const glm::vec3& direction);
    void SetLightRange(Core::Entity entity, float range);
    
    // Light animation
    void SetLightFlicker(Core::Entity entity, float intensity, float speed);
    void SetLightMovement(Core::Entity entity, float radius, float speed);
    
    // Global settings
    void SetAmbientLight(const glm::vec3& ambient) { m_ambientLight = ambient; }
    const glm::vec3& GetAmbientLight() const { return m_ambientLight; }
    
    // Day/night cycle
    void EnableDayNightCycle(float cycleDuration = 120.0f); // Duration in seconds
    void DisableDayNightCycle() { m_dayNightEnabled = false; }
    void SetTimeOfDay(float normalizedTime); // 0.0 = midnight, 0.5 = noon
    
    // Shadow mapping
    void SetShadowMapSize(int size) { m_shadowMapSize = size; }
    void SetCascadeCount(int count) { m_cascadeCount = count; }
    void RenderShadowMaps();
    
    // Rendering interface
    void BindLights(unsigned int bindingPoint);
    unsigned int GetShadowMapArray() const { return m_shadowMapArray; }
    const std::vector<const Light*>& GetActiveLights() const { return m_activeLights; }
    size_t GetActiveLightCount() const { return m_activeLights.size(); }
    
    // Configuration
    void SetMaxLights(size_t max) { m_maxLights = max; }
    size_t GetMaxLights() const { return m_maxLights; }
    
private:
    // Light management
    std::vector<Light> m_lights;
    std::vector<const Light*> m_activeLights; // Culled and sorted lights
    size_t m_maxLights;
    
    // GPU resources
    unsigned int m_lightUBO;        // Uniform buffer for light data
    unsigned int m_shadowMapArray;  // Texture array for shadow maps
    unsigned int m_shadowFBO;       // Framebuffer for shadow rendering
    int m_shadowMapSize;
    int m_cascadeCount;
    
    // Global lighting
    glm::vec3 m_ambientLight;
    
    // Day/night cycle
    bool m_dayNightEnabled = false;
    float m_dayNightDuration = 120.0f;
    float m_dayNightTime = 0.0f;
    
    // Animation
    float m_time = 0.0f;
    
    // System references
    RenderSystem* m_renderSystem;
    
    // Internal methods
    void UpdateDayNightCycle(float deltaTime);
    void CullLights();
    void UpdateLightUBO();
    void SortLightsByPriority();
};

} // namespace Rendering
} // namespace CudaGame
