#pragma once
#ifdef _WIN32

#include <glm/glm.hpp>

namespace CudaGame {
namespace Rendering {

// Constant buffer structures must match HLSL declarations exactly
// All buffers must be aligned to 256 bytes for D3D12

// Per-frame constants (updated once per frame)
// Matches cbuffer PerFrameConstants : register(b0)
struct alignas(256) PerFrameConstants {
    glm::mat4 viewMatrix;           // 64 bytes
    glm::mat4 projMatrix;           // 64 bytes
    glm::mat4 viewProjMatrix;       // 64 bytes
    glm::mat4 prevViewProjMatrix;   // 64 bytes (for motion vectors)
    glm::vec3 cameraPosition;       // 12 bytes
    float time;                     // 4 bytes
    float deltaTime;                // 4 bytes
    glm::vec3 _padding;             // 12 bytes (alignment)
};

// Per-object constants (updated per mesh)
// Matches cbuffer PerObjectConstants : register(b1)
struct alignas(256) PerObjectConstants {
    glm::mat4 worldMatrix;          // 64 bytes
    glm::mat4 prevWorldMatrix;      // 64 bytes (for motion vectors)
    glm::mat4 normalMatrix;         // 64 bytes (transpose-inverse of world)
    uint32_t boneOffset;            // 4 bytes (Global Bone Buffer Offset)
    uint32_t isSkinned;             // 4 bytes (Bool flag)
    float _padding[14];             // 56 bytes (fill to 256)
};

// Material constants (updated per mesh)
// Matches cbuffer MaterialConstants : register(b2)
struct alignas(256) MaterialConstants {
    glm::vec4 albedoColor;          // 16 bytes
    float roughness;                // 4 bytes
    float metallic;                 // 4 bytes
    float ambientOcclusion;         // 4 bytes
    float emissiveStrength;         // 4 bytes
    glm::vec3 emissiveColor;        // 12 bytes
    float _padding;                 // 4 bytes (alignment)
    float _padding2[52];            // 208 bytes (fill to 256 total)
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
