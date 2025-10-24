#include "Rendering/LightingSystem.h"
#include "Rendering/ShaderProgram.h"
#include "Core/Coordinator.h"
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

namespace CudaGame {
namespace Rendering {

namespace {
    // Default shadow map resolution
    const unsigned int SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;
}

LightingSystem::LightingSystem() {
    // Lighting system priority: 190
}

bool LightingSystem::Initialize() {
    std::cout << "[LightingSystem] Initializing..." << std::endl;

    // TODO: Initialize OpenGL resources when graphics API is available
    // For now, just create placeholder IDs
    m_directionalShadowFBO = 1;
    m_directionalShadowMap = 1;
    m_omnidirectionalShadowFBO = 2;
    m_omnidirectionalShadowMap = 2;

    std::cout << "[LightingSystem] Shadow mapping resources initialized." << std::endl;
    return true;
}

void LightingSystem::Shutdown() {
    std::cout << "[LightingSystem] Shutting down..." << std::endl;
    // TODO: Cleanup OpenGL resources when graphics API is available
}

void LightingSystem::Update(float deltaTime) {}

Core::Entity LightingSystem::CreateDirectionalLight(const glm::vec3& direction, const glm::vec3& color, float intensity) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    Core::Entity light = coordinator.CreateEntity();
    coordinator.AddComponent(light, LightComponent{
        LightType::DIRECTIONAL, color, intensity
    });
    coordinator.AddComponent(light, TransformComponent{
        glm::vec3(0.0f), direction, glm::vec3(1.0f)
    });
    return light;
}

Core::Entity LightingSystem::CreatePointLight(const glm::vec3& position, float radius, const glm::vec3& color, float intensity) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    Core::Entity light = coordinator.CreateEntity();
    coordinator.AddComponent(light, LightComponent{
        LightType::POINT, color, intensity, true, 1.0f, 0.005f, radius
    });
    coordinator.AddComponent(light, TransformComponent{
        position, glm::vec3(0.0f), glm::vec3(1.0f)
    });
    return light;
}

Core::Entity LightingSystem::CreateSpotLight(const glm::vec3& position, const glm::vec3& direction, float innerCutoff, float outerCutoff, float radius, const glm::vec3& color, float intensity) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    Core::Entity light = coordinator.CreateEntity();
    coordinator.AddComponent(light, LightComponent{
        LightType::SPOT, color, intensity, true, 1.0f, 0.005f, radius, 1.0f, innerCutoff, outerCutoff
    });
    coordinator.AddComponent(light, TransformComponent{
        position, direction, glm::vec3(1.0f)
    });
    return light;
}

void LightingSystem::UpdateShadowMaps(const Camera& mainCamera, ShaderProgram& shadowShader) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    // For now, we only support one directional light for shadows
    Core::Entity directionalLight = -1;
    // TODO: Iterate through all light entities
    // for (auto const& entity : lightEntities) {
    //     auto const& light = coordinator.GetComponent<LightComponent>(entity);
    //     if (light.type == LightType::DIRECTIONAL && light.castsShadows) {
    //         directionalLight = entity;
    //         break;
    //     }
    // }

    if (directionalLight != -1) {
        RenderDirectionalShadows(mainCamera, shadowShader, directionalLight);
    }
}

void LightingSystem::RenderDirectionalShadows(const Camera& mainCamera, ShaderProgram& shadowShader, Core::Entity lightEntity) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    auto const& lightTransform = coordinator.GetComponent<TransformComponent>(lightEntity);

    // Create light view-projection matrix
    float near_plane = 1.0f, far_plane = 7.5f;
    glm::mat4 lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
    glm::mat4 lightView = glm::lookAt(lightTransform.position, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
    m_lightSpaceMatrix = lightProjection * lightView;

    // Render scene from light's perspective
    shadowShader.Use();
    shadowShader.SetMat4("lightSpaceMatrix", m_lightSpaceMatrix);

    // TODO: Replace with proper rendering calls
    // glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    // glBindFramebuffer(GL_FRAMEBUFFER, m_directionalShadowFBO);
    // glClear(GL_DEPTH_BUFFER_BIT);

    // Render all shadow casters
    // TODO: Iterate through all shadow casting entities
    // for (auto const& entity : shadowCasterEntities) {
    //     if (coordinator.HasComponent<ShadowCasterComponent>(entity) && coordinator.HasComponent<MeshComponent>(entity)) {
    //         auto const& transform = coordinator.GetComponent<TransformComponent>(entity);
    //         shadowShader.SetMat4("model", transform.getMatrix());
    //         // render mesh (assuming a render function exists)
    //         // RenderMesh(coordinator.GetComponent<MeshComponent>(entity));
    //     }
    // }

    // TODO: Unbind framebuffer
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void LightingSystem::RenderOmnidirectionalShadows() {
    // Implementation for omnidirectional shadow maps (point lights) will go here
}

} // namespace Rendering
} // namespace CudaGame
