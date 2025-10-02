#include "Rendering/MultiLightSystem.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderSystem.h"
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace CudaGame {
namespace Rendering {

MultiLightSystem::MultiLightSystem() 
    : m_maxLights(128)
    , m_lightUBO(0)
    , m_shadowMapArray(0)
    , m_shadowMapSize(2048)
    , m_cascadeCount(4)
    , m_renderSystem(nullptr) {
}

MultiLightSystem::~MultiLightSystem() {
    Shutdown();
}

bool MultiLightSystem::Initialize() {
    auto& coordinator = Core::Coordinator::GetInstance();
    m_renderSystem = coordinator.GetSystem<RenderSystem>().get();  // Fixed: added .get()
    
    if (!m_renderSystem) {
        std::cerr << "[MultiLightSystem] Failed to get RenderSystem!" << std::endl;
        return false;
    }
    
    // Create Uniform Buffer Object for light data
    glGenBuffers(1, &m_lightUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, m_lightUBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(LightData) * m_maxLights + sizeof(int) * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Create shadow map array texture for multiple lights
    glGenTextures(1, &m_shadowMapArray);
    glBindTexture(GL_TEXTURE_2D_ARRAY, m_shadowMapArray);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, 
                 m_shadowMapSize, m_shadowMapSize, m_maxLights,
                 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    
    float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, borderColor);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    
    // Create shadow framebuffer
    glGenFramebuffers(1, &m_shadowFBO);
    
    // Set default ambient light
    m_ambientLight = glm::vec3(0.15f, 0.15f, 0.2f);
    
    std::cout << "[MultiLightSystem] Initialized with support for " << m_maxLights << " lights" << std::endl;
    return true;
}

void MultiLightSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Update animated lights
    for (auto& light : m_lights) {
        if (light.type == LightType::POINT || light.type == LightType::SPOT) {
            // Update flickering lights
            if (light.flickerIntensity > 0.0f) {
                float flicker = 1.0f + sin(light.flickerSpeed * m_time) * light.flickerIntensity;
                light.data.intensity = light.baseIntensity * flicker;
            }
            
            // Update moving lights (e.g., torch sway)
            if (light.moveRadius > 0.0f) {
                float angle = light.moveSpeed * m_time;
                glm::vec3 offset(
                    cos(angle) * light.moveRadius,
                    sin(angle * 2.0f) * light.moveRadius * 0.5f,
                    sin(angle) * light.moveRadius
                );
                light.data.position = glm::vec4(light.basePosition + offset, 1.0f);
            }
        }
    }
    
    // Update day/night cycle if enabled
    if (m_dayNightEnabled) {
        UpdateDayNightCycle(deltaTime);
    }
    
    // Cull lights based on camera frustum
    CullLights();
    
    // Update light UBO
    UpdateLightUBO();
    
    m_time += deltaTime;
}

Core::Entity MultiLightSystem::CreateDirectionalLight(const glm::vec3& direction, 
                                                      const glm::vec3& color,
                                                      float intensity) {
    auto& coordinator = Core::Coordinator::GetInstance();
    Core::Entity entity = coordinator.CreateEntity();
    
    LightComponent lightComp;
    lightComp.type = LightType::DIRECTIONAL;
    lightComp.color = color;
    lightComp.intensity = intensity;
    lightComp.castsShadows = true;
    
    coordinator.AddComponent(entity, lightComp);
    
    // Add to internal light list
    Light light;
    light.entity = entity;
    light.type = LightType::DIRECTIONAL;
    light.data.position = glm::vec4(0.0f);
    light.data.direction = glm::vec4(glm::normalize(direction), 0.0f);  // Fixed: use direction parameter directly
    light.data.color = glm::vec4(color, 1.0f);
    light.data.intensity = intensity;
    light.baseIntensity = intensity;
    
    m_lights.push_back(light);
    
    std::cout << "[MultiLightSystem] Created directional light (Entity: " << entity << ")" << std::endl;
    return entity;
}

Core::Entity MultiLightSystem::CreatePointLight(const glm::vec3& position,
                                               const glm::vec3& color,
                                               float intensity,
                                               float radius) {
    auto& coordinator = Core::Coordinator::GetInstance();
    Core::Entity entity = coordinator.CreateEntity();
    
    LightComponent lightComp;
    lightComp.type = LightType::POINT;
    lightComp.color = color;
    lightComp.intensity = intensity;
    lightComp.radius = radius;  // Fixed: use correct field name
    lightComp.castsShadows = false; // Point light shadows are expensive
    
    coordinator.AddComponent(entity, lightComp);
    
    // Add transform for position
    TransformComponent transform;
    transform.position = position;
    coordinator.AddComponent(entity, transform);
    
    // Add to internal light list
    Light light;
    light.entity = entity;
    light.type = LightType::POINT;
    light.data.position = glm::vec4(position, 1.0f);
    light.data.direction = glm::vec4(0.0f);
    light.data.color = glm::vec4(color, 1.0f);
    light.data.intensity = intensity;
    light.data.range = radius;
    light.data.attenuation = glm::vec4(1.0f, 0.09f, 0.032f, 0.0f); // Constant, linear, quadratic
    light.baseIntensity = intensity;
    light.basePosition = position;
    
    m_lights.push_back(light);
    
    std::cout << "[MultiLightSystem] Created point light at (" 
              << position.x << ", " << position.y << ", " << position.z 
              << ") (Entity: " << entity << ")" << std::endl;
    return entity;
}

Core::Entity MultiLightSystem::CreateSpotLight(const glm::vec3& position,
                                              const glm::vec3& direction,
                                              const glm::vec3& color,
                                              float intensity,
                                              float innerCone,
                                              float outerCone,
                                              float range) {
    auto& coordinator = Core::Coordinator::GetInstance();
    Core::Entity entity = coordinator.CreateEntity();
    
    LightComponent lightComp;
    lightComp.type = LightType::SPOT;
    lightComp.color = color;
    lightComp.intensity = intensity;
    lightComp.radius = range;  // Fixed: use correct field name
    lightComp.innerCutoff = innerCone;  // Fixed: use correct field name
    lightComp.outerCutoff = outerCone;  // Fixed: use correct field name
    lightComp.castsShadows = true;
    
    coordinator.AddComponent(entity, lightComp);
    
    // Add transform
    TransformComponent transform;
    transform.position = position;
    coordinator.AddComponent(entity, transform);
    
    // Add to internal light list
    Light light;
    light.entity = entity;
    light.type = LightType::SPOT;
    light.data.position = glm::vec4(position, 1.0f);
    light.data.direction = glm::vec4(glm::normalize(direction), 0.0f);
    light.data.color = glm::vec4(color, 1.0f);
    light.data.intensity = intensity;
    light.data.range = range;
    light.data.innerCone = cos(glm::radians(innerCone));
    light.data.outerCone = cos(glm::radians(outerCone));
    light.data.attenuation = glm::vec4(1.0f, 0.09f, 0.032f, 0.0f);
    light.baseIntensity = intensity;
    light.basePosition = position;
    
    m_lights.push_back(light);
    
    std::cout << "[MultiLightSystem] Created spot light (Entity: " << entity << ")" << std::endl;
    return entity;
}

void MultiLightSystem::SetLightFlicker(Core::Entity entity, float intensity, float speed) {
    auto it = std::find_if(m_lights.begin(), m_lights.end(),
                           [entity](const Light& l) { return l.entity == entity; });
    
    if (it != m_lights.end()) {
        it->flickerIntensity = intensity;
        it->flickerSpeed = speed;
        std::cout << "[MultiLightSystem] Set flicker for light " << entity 
                  << " (intensity: " << intensity << ", speed: " << speed << ")" << std::endl;
    }
}

void MultiLightSystem::SetLightMovement(Core::Entity entity, float radius, float speed) {
    auto it = std::find_if(m_lights.begin(), m_lights.end(),
                           [entity](const Light& l) { return l.entity == entity; });
    
    if (it != m_lights.end()) {
        it->moveRadius = radius;
        it->moveSpeed = speed;
        std::cout << "[MultiLightSystem] Set movement for light " << entity 
                  << " (radius: " << radius << ", speed: " << speed << ")" << std::endl;
    }
}

void MultiLightSystem::EnableDayNightCycle(float cycleDuration) {
    m_dayNightEnabled = true;
    m_dayNightDuration = cycleDuration;
    m_dayNightTime = 0.0f;
    
    std::cout << "[MultiLightSystem] Day/night cycle enabled (duration: " 
              << cycleDuration << " seconds)" << std::endl;
}

void MultiLightSystem::UpdateDayNightCycle(float deltaTime) {
    m_dayNightTime += deltaTime;
    float cycleProgress = fmod(m_dayNightTime / m_dayNightDuration, 1.0f);
    
    // Calculate sun angle (0 = sunrise, 0.5 = sunset)
    float sunAngle = cycleProgress * 2.0f * M_PI;
    
    // Update directional light (sun)
    for (auto& light : m_lights) {
        if (light.type == LightType::DIRECTIONAL) {
            // Sun direction
            glm::vec3 sunDir(
                cos(sunAngle),
                sin(sunAngle),
                0.0f
            );
            light.data.direction = glm::vec4(sunDir, 0.0f);
            
            // Sun color and intensity based on time of day
            if (cycleProgress < 0.25f || cycleProgress > 0.75f) {
                // Night time
                light.data.color = glm::vec4(0.1f, 0.1f, 0.2f, 1.0f);
                light.data.intensity = 0.1f;
                m_ambientLight = glm::vec3(0.05f, 0.05f, 0.1f);
            } else if (cycleProgress < 0.3f || cycleProgress > 0.7f) {
                // Sunrise/sunset
                light.data.color = glm::vec4(1.0f, 0.6f, 0.3f, 1.0f);
                light.data.intensity = 0.6f;
                m_ambientLight = glm::vec3(0.3f, 0.2f, 0.15f);
            } else {
                // Day time
                light.data.color = glm::vec4(1.0f, 0.95f, 0.8f, 1.0f);
                light.data.intensity = 1.0f;
                m_ambientLight = glm::vec3(0.2f, 0.2f, 0.25f);
            }
        }
    }
}

void MultiLightSystem::CullLights() {
    // Clear active lights
    m_activeLights.clear();
    
    // For now, just add all lights (proper frustum culling can be added later)
    for (const auto& light : m_lights) {
        m_activeLights.push_back(&light);
    }
    
    // Sort by priority
    SortLightsByPriority();
    
    // Limit to max lights
    if (m_activeLights.size() > m_maxLights) {
        m_activeLights.resize(m_maxLights);
    }
}

void MultiLightSystem::SortLightsByPriority() {
    // Sort lights by type priority: Directional > Spot > Point
    std::sort(m_activeLights.begin(), m_activeLights.end(),
              [](const Light* a, const Light* b) {
                  if (a->type != b->type) {
                      return static_cast<int>(a->type) < static_cast<int>(b->type);
                  }
                  // If same type, sort by intensity
                  return a->data.intensity > b->data.intensity;
              });
}

void MultiLightSystem::UpdateLightUBO() {
    if (m_activeLights.empty()) return;
    
    // Prepare light data for GPU
    std::vector<LightData> lightData;
    lightData.reserve(m_activeLights.size());
    
    for (const auto* light : m_activeLights) {
        lightData.push_back(light->data);
    }
    
    // Update UBO
    glBindBuffer(GL_UNIFORM_BUFFER, m_lightUBO);
    
    // Upload light count
    int lightCount = static_cast<int>(lightData.size());
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(int), &lightCount);
    
    // Upload light data
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(int) * 4, 
                    sizeof(LightData) * lightData.size(), lightData.data());
    
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void MultiLightSystem::BindLights(unsigned int bindingPoint) {
    glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, m_lightUBO);
}

void MultiLightSystem::RenderShadowMaps() {
    // TODO: Implement shadow map rendering
    // This will require coordination with RenderSystem to render scene from light perspectives
}

void MultiLightSystem::RemoveLight(Core::Entity entity) {
    auto it = std::remove_if(m_lights.begin(), m_lights.end(),
                             [entity](const Light& l) { return l.entity == entity; });
    
    if (it != m_lights.end()) {
        m_lights.erase(it, m_lights.end());
        std::cout << "[MultiLightSystem] Removed light (Entity: " << entity << ")" << std::endl;
    }
}

void MultiLightSystem::SetLightIntensity(Core::Entity entity, float intensity) {
    auto it = std::find_if(m_lights.begin(), m_lights.end(),
                           [entity](const Light& l) { return l.entity == entity; });
    
    if (it != m_lights.end()) {
        it->data.intensity = intensity;
        it->baseIntensity = intensity;
    }
}

void MultiLightSystem::SetLightColor(Core::Entity entity, const glm::vec3& color) {
    auto it = std::find_if(m_lights.begin(), m_lights.end(),
                           [entity](const Light& l) { return l.entity == entity; });
    
    if (it != m_lights.end()) {
        it->data.color = glm::vec4(color, 1.0f);
    }
}

void MultiLightSystem::SetLightPosition(Core::Entity entity, const glm::vec3& position) {
    auto it = std::find_if(m_lights.begin(), m_lights.end(),
                           [entity](const Light& l) { return l.entity == entity; });
    
    if (it != m_lights.end()) {
        it->data.position = glm::vec4(position, 1.0f);
        it->basePosition = position;
    }
}

void MultiLightSystem::SetLightDirection(Core::Entity entity, const glm::vec3& direction) {
    auto it = std::find_if(m_lights.begin(), m_lights.end(),
                           [entity](const Light& l) { return l.entity == entity; });
    
    if (it != m_lights.end()) {
        it->data.direction = glm::vec4(glm::normalize(direction), 0.0f);
    }
}

void MultiLightSystem::SetLightRange(Core::Entity entity, float range) {
    auto it = std::find_if(m_lights.begin(), m_lights.end(),
                           [entity](const Light& l) { return l.entity == entity; });
    
    if (it != m_lights.end()) {
        it->data.range = range;
    }
}

void MultiLightSystem::SetTimeOfDay(float normalizedTime) {
    m_dayNightTime = normalizedTime * m_dayNightDuration;
}

void MultiLightSystem::Shutdown() {
    if (m_lightUBO) {
        glDeleteBuffers(1, &m_lightUBO);
        m_lightUBO = 0;
    }
    
    if (m_shadowMapArray) {
        glDeleteTextures(1, &m_shadowMapArray);
        m_shadowMapArray = 0;
    }
    
    if (m_shadowFBO) {
        glDeleteFramebuffers(1, &m_shadowFBO);
        m_shadowFBO = 0;
    }
    
    m_lights.clear();
    m_activeLights.clear();
    
    std::cout << "[MultiLightSystem] Shut down" << std::endl;
}

} // namespace Rendering
} // namespace CudaGame
