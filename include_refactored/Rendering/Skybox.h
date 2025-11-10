#pragma once

#include <string>
#include <cstdint>
#include <glm/glm.hpp>

namespace CudaGame {
namespace Rendering {

/**
 * @brief HDR skybox with runtime loading, cubemap baking, and tone mapping.
 * 
 * Phase 1 implementation supporting:
 * - HDR equirectangular image loading via stb_image
 * - Cubemap baking via FBO rendering
 * - Skybox rendering with depth state handling
 * - Exposure and gamma controls
 * 
 * Future phases will add IBL generation (irradiance, prefiltered env, BRDF LUT).
 */
class Skybox {
public:
    Skybox();
    ~Skybox();

    // Load HDR equirectangular image and bake to cubemap
    bool LoadHDR(const std::string& hdrPath, int cubemapSize = 512);
    
    // Render skybox as background (call after deferred lighting, before forward pass)
    void Render(const glm::mat4& view, const glm::mat4& projection);
    
    // Runtime controls
    void SetExposure(float exposure) { m_exposure = exposure; }
    float GetExposure() const { return m_exposure; }
    
    void SetGamma(float gamma) { m_gamma = gamma; }
    float GetGamma() const { return m_gamma; }
    
    void SetRotation(float angleRadians) { m_rotation = angleRadians; }
    float GetRotation() const { return m_rotation; }
    
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }
    
    // Get cubemap texture ID for binding in shaders
    uint32_t GetCubemapTexture() const { return m_cubemapTexture; }
    
    // Cleanup
    void Shutdown();

private:
    // HDR loading and cubemap baking
    bool LoadHDRImage(const std::string& path, float*& data, int& width, int& height);
    bool BakeToCubemap(const float* hdrData, int hdrWidth, int hdrHeight, int cubemapSize);
    
    // Shader and geometry setup
    bool InitializeShaders();
    void CreateCubeGeometry();
    void DestroyCubeGeometry();
    void RenderCube();
    
    // OpenGL resources
    uint32_t m_cubemapTexture = 0;
    uint32_t m_cubeVAO = 0;
    uint32_t m_cubeVBO = 0;
    uint32_t m_shaderProgram = 0;
    
    // Equirectangular to cubemap conversion resources
    uint32_t m_equirectShader = 0;
    uint32_t m_captureFBO = 0;
    uint32_t m_captureRBO = 0;
    
    // Runtime state
    float m_exposure = 1.0f;
    float m_gamma = 2.2f;
    float m_rotation = 0.0f;
    bool m_enabled = true;
    bool m_initialized = false;
};

} // namespace Rendering
} // namespace CudaGame
