#pragma once

#include "Core/System.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/Camera.h"
#include "Rendering/CudaRenderingData.h"
#include <glm/glm.hpp>
#include <memory>

// Forward declarations
class ShaderProgram;

namespace CudaGame {
namespace Rendering {

// Manages GPU-accelerated rendering effects using CUDA
class CudaRenderingSystem : public Core::System {
public:
    CudaRenderingSystem();
    ~CudaRenderingSystem() override;

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Post-processing effects
    void ExecutePostProcessingPipeline(uint32_t inputTexture, uint32_t outputTexture, int width, int height);
    
    // Screen-Space Ambient Occlusion (SSAO)
    void ComputeSSAO(uint32_t gBufferPosition, uint32_t gBufferNormal, uint32_t outputTexture, 
                     const glm::mat4& projection, int width, int height);
    void SetSSAOParameters(float radius, float bias, int sampleCount);
    
    // Volumetric Lighting
    void ComputeVolumetricLighting(uint32_t depthTexture, uint32_t outputTexture, 
                                  const glm::vec3& lightPos, const glm::vec3& lightColor,
                                  const glm::mat4& lightViewProj, int width, int height);
    void SetVolumetricLightingParameters(float density, float absorption, int sampleCount);
    
    // Bloom effect
    void ComputeBloom(uint32_t inputTexture, uint32_t outputTexture, int width, int height);
    void SetBloomParameters(float threshold, float intensity, int iterations);
    
    // Tone mapping and color grading
    void ApplyToneMapping(uint32_t inputTexture, uint32_t outputTexture, int width, int height);
    void SetToneMappingParameters(float exposure, float gamma);
    
    // Advanced shadow effects
    void ComputeContactShadows(uint32_t depthTexture, uint32_t normalTexture, uint32_t outputTexture,
                              const glm::mat4& viewMatrix, const glm::mat4& projMatrix, int width, int height);
    
    // Motion blur
    void ComputeMotionBlur(uint32_t colorTexture, uint32_t velocityTexture, uint32_t outputTexture,
                          int width, int height);
    void SetMotionBlurParameters(float strength, int samples);
    
    // Temporal Anti-Aliasing (TAA)
    void ComputeTAA(uint32_t currentFrame, uint32_t previousFrame, uint32_t velocityTexture,
                   uint32_t outputTexture, int width, int height);
    
    // Debug and profiling
    void SetDebugMode(bool enabled) { m_debugMode = enabled; }
    float GetGPUFrameTime() const { return m_gpuFrameTime; }
    void PrintProfileInfo() const;

private:
    // CUDA-specific data (PIMPL pattern)
    std::unique_ptr<CudaRenderingData> m_cudaData;
    
    // Effect parameters
    struct SSAOParams {
        float radius = 0.5f;
        float bias = 0.025f;
        int sampleCount = 16;
    } m_ssaoParams;
    
    struct VolumetricParams {
        float density = 0.1f;
        float absorption = 0.1f;
        int sampleCount = 32;
    } m_volumetricParams;
    
    struct BloomParams {
        float threshold = 1.0f;
        float intensity = 1.0f;
        int iterations = 5;
    } m_bloomParams;
    
    struct ToneMappingParams {
        float exposure = 1.0f;
        float gamma = 2.2f;
    } m_toneMappingParams;
    
    struct MotionBlurParams {
        float strength = 1.0f;
        int samples = 16;
    } m_motionBlurParams;
    
    // Debug and profiling
    bool m_debugMode = false;
    float m_gpuFrameTime = 0.0f;
    
    // Internal helper methods
    void InitializeCudaResources();
    void ShutdownCudaResources();
    bool CreateCudaTexture(uint32_t& cudaTexture, int width, int height, int channels);
    void SynchronizeGPU();
};

} // namespace Rendering
} // namespace CudaGame
