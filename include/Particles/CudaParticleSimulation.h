#pragma once

#include "Particles/ParticleComponents.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <memory>

namespace CudaGame {
namespace Particles {

// CUDA-specific particle data structure (optimized for GPU memory)
struct CudaParticle {
    float3 position;
    float3 velocity;
    float3 acceleration;
    
    float4 color;
    float4 startColor;
    float4 endColor;
    
    float size;
    float startSize;
    float endSize;
    
    float lifetime;
    float age;
    float normalizedAge;
    
    float rotation;
    float angularVelocity;
    
    int isActive;
    float3 customData;
};

// GPU memory buffers for particle simulation
struct CudaParticleBuffers {
    CudaParticle* d_particles = nullptr;
    float3* d_forces = nullptr;
    int* d_activeIndices = nullptr;
    
    // Emission data
    float3* d_emissionPositions = nullptr;
    float3* d_emissionVelocities = nullptr;
    
    // Force field data
    float3* d_forceFieldPositions = nullptr;
    float4* d_forceFieldData = nullptr; // strength, radius, falloff, type
    
    int maxParticles = 0;
    int maxForceFields = 0;
    bool allocated = false;
    
    cudaError_t AllocateBuffers(int particleCount, int forceFieldCount);
    void DeallocateBuffers();
};

// CUDA kernel launch parameters
struct CudaSimulationParams {
    int numParticles;
    int numActiveParticles;
    int numForceFields;
    
    float deltaTime;
    float3 gravity;
    float globalDrag;
    
    // Emission parameters
    float3 emissionPosition;
    float emissionRate;
    int particlesToEmit;
    
    // Physics bounds
    float3 worldMin;
    float3 worldMax;
    bool enableCollisions;
    
    // Performance settings
    int threadsPerBlock;
    int numBlocks;
};

class CudaParticleSimulation {
public:
    CudaParticleSimulation();
    ~CudaParticleSimulation();

    // Initialization and cleanup
    bool Initialize(int maxParticles, int maxForceFields = 32);
    void Shutdown();
    
    // Memory management
    bool AllocateGPUMemory(int particleCount, int forceFieldCount);
    void DeallocateGPUMemory();
    
    // Data transfer between CPU and GPU
    cudaError_t UploadParticleData(const std::vector<Particle>& cpuParticles);
    cudaError_t DownloadParticleData(std::vector<Particle>& cpuParticles);
    cudaError_t UploadForceFieldData(const std::vector<ParticleForceFieldComponent>& forceFields,
                                   const std::vector<glm::vec3>& positions);
    
    // Main simulation step
    cudaError_t SimulateStep(const CudaSimulationParams& params);
    
    // Individual simulation kernels
    cudaError_t UpdateParticlePhysics(const CudaSimulationParams& params);
    cudaError_t ApplyForceFields(const CudaSimulationParams& params);
    cudaError_t UpdateParticleLifetime(const CudaSimulationParams& params);
    cudaError_t EmitNewParticles(const CudaSimulationParams& params);
    cudaError_t HandleCollisions(const CudaSimulationParams& params);
    
    // Performance monitoring
    float GetLastSimulationTime() const { return m_lastSimulationTime; }
    size_t GetGPUMemoryUsage() const { return m_gpuMemoryUsage; }
    
    // Statistics
    struct CudaStats {
        int activeParticles = 0;
        int particlesEmitted = 0;
        int particlesKilled = 0;
        float simulationTimeMs = 0.0f;
        float memoryTransferTimeMs = 0.0f;
        size_t gpuMemoryUsed = 0;
    };
    
    const CudaStats& GetStats() const { return m_stats; }
    
    // Debug and profiling
    void EnableProfiling(bool enabled) { m_profilingEnabled = enabled; }
    void PrintPerformanceReport();

private:
    // GPU memory buffers
    CudaParticleBuffers m_buffers;
    
    // CUDA streams for asynchronous operations
    cudaStream_t m_computeStream;
    cudaStream_t m_transferStream;
    
    // CUDA events for timing
    cudaEvent_t m_startEvent;
    cudaEvent_t m_stopEvent;
    
    // Performance tracking
    float m_lastSimulationTime = 0.0f;
    size_t m_gpuMemoryUsage = 0;
    bool m_profilingEnabled = false;
    
    // Statistics
    CudaStats m_stats;
    
    // Device properties
    cudaDeviceProp m_deviceProps;
    int m_deviceId = 0;
    
    // Helper methods
    bool CheckCudaDevice();
    void ConfigureKernelLaunchParams(CudaSimulationParams& params);
    cudaError_t SynchronizeGPU();
    
    // Memory management helpers
    size_t CalculateMemoryRequirements(int particleCount, int forceFieldCount);
    void UpdateMemoryUsageStats();
    
    // Error handling
    bool CheckCudaError(cudaError_t error, const char* operation);
    void HandleCudaError(cudaError_t error, const char* file, int line);
};

// CUDA kernel declarations (implemented in .cu file)
extern "C" {
    // Physics simulation kernels
    cudaError_t LaunchUpdatePhysicsKernel(CudaParticle* particles, int numParticles,
                                        float deltaTime, float3 gravity, float globalDrag,
                                        int threadsPerBlock);
    
    cudaError_t LaunchApplyForceFieldsKernel(CudaParticle* particles, int numParticles,
                                           float3* forceFieldPositions, float4* forceFieldData, int numForceFields,
                                           int threadsPerBlock);
    
    cudaError_t LaunchUpdateLifetimeKernel(CudaParticle* particles, int numParticles,
                                         float deltaTime, int* activeIndices, int* numActive,
                                         int threadsPerBlock);
    
    cudaError_t LaunchEmissionKernel(CudaParticle* particles, int maxParticles,
                                   float3* emissionPositions, float3* emissionVelocities, int numToEmit,
                                   int* activeIndices, int* numActive,
                                   int threadsPerBlock);
    
    cudaError_t LaunchCollisionKernel(CudaParticle* particles, int numParticles,
                                    float3 worldMin, float3 worldMax, float bounce,
                                    int threadsPerBlock);
    
    // Utility kernels
    cudaError_t LaunchCompactActiveParticlesKernel(CudaParticle* particles, int* activeIndices,
                                                  int numParticles, int* numActive,
                                                  int threadsPerBlock);
    
    cudaError_t LaunchResetParticleKernel(CudaParticle* particles, int numParticles,
                                        int threadsPerBlock);
}

// Utility functions for CUDA-GLM interoperability
inline float3 GlmToFloat3(const glm::vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

inline float4 GlmToFloat4(const glm::vec4& v) {
    return make_float4(v.x, v.y, v.z, v.w);
}

inline glm::vec3 Float3ToGlm(const float3& v) {
    return glm::vec3(v.x, v.y, v.z);
}

inline glm::vec4 Float4ToGlm(const float4& v) {
    return glm::vec4(v.x, v.y, v.z, v.w);
}

// Convert CPU particle to GPU particle format
CudaParticle ConvertToCudaParticle(const Particle& cpuParticle);

// Convert GPU particle back to CPU format
Particle ConvertFromCudaParticle(const CudaParticle& gpuParticle);

} // namespace Particles
} // namespace CudaGame
