#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <string>
#include <string>

namespace CudaGame {
namespace Rendering {

// Represents a 3D mesh asset
struct MeshComponent {
    std::string modelPath;
    // OpenGL resources
    uint32_t vaoId = 0;
    uint32_t vbo = 0;
    uint32_t ebo = 0;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    bool castShadows = true;
    bool receiveShadows = true;
    bool isVisible = true;
};

// Physically-Based Rendering (PBR) material component
struct MaterialComponent {
    glm::vec3 albedo = {1.0f, 1.0f, 1.0f};
    float metallic = 0.5f;
    float roughness = 0.5f;
    float ao = 1.0f; // Ambient Occlusion
    glm::vec3 emissiveColor = {0.0f, 0.0f, 0.0f};
    float emissiveIntensity = 0.0f;
    
    // Texture map handles (IDs)
    uint32_t albedoMap = 0;
    uint32_t normalMap = 0;
    uint32_t metallicMap = 0;
    uint32_t roughnessMap = 0;
    uint32_t aoMap = 0;
    uint32_t emissiveMap = 0;
};

// Transform component for positioning objects in the world
struct TransformComponent {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f}; // Euler angles for simplicity
    glm::vec3 scale{1.0f};

    glm::mat4 getMatrix() const {
        glm::mat4 rot = glm::mat4(1.0f);
        rot = glm::rotate(rot, glm::radians(rotation.x), glm::vec3(1, 0, 0));
        rot = glm::rotate(rot, glm::radians(rotation.y), glm::vec3(0, 1, 0));
        rot = glm::rotate(rot, glm::radians(rotation.z), glm::vec3(0, 0, 1));

        return glm::translate(glm::mat4(1.0f), position) * rot * glm::scale(glm::mat4(1.0f), scale);
    }
};

// Light component for scene lighting
enum class LightType { DIRECTIONAL, POINT, SPOT };

struct LightComponent {
    LightType type = LightType::POINT;
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    bool castsShadows = true;
    float shadowStrength = 1.0f;
    float shadowBias = 0.005f;

    // For point and spot lights
    float radius = 10.0f;
    float falloff = 1.0f;

    // For spot lights
    float innerCutoff = 12.5f; // degrees
    float outerCutoff = 17.5f; // degrees
};

// Shadow caster component for entities that cast shadows
struct ShadowCasterComponent {
    bool isDynamic = true;
    float shadowResolutionScale = 1.0f;
    int cascadeCount = 4;
    float cascadeDistribution = 0.8f;
};

} // namespace Rendering
} // namespace CudaGame

