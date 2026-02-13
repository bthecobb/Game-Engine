#pragma once

#include "Core/System.h"
#include <vector>
#include <glm/glm.hpp>
#include "VFX/ParticleStruct.h"

// Forward Declarations
namespace CudaGame {
    namespace Rendering {
        class DX12RenderPipeline;
    }
}
struct ID3D12Resource;
struct cudaGraphicsResource;

namespace CudaGame {
namespace VFX {

/**
 * [AAA Pattern] GPU Particle System
 * Manages particle simulation on CUDA and rendering via DX12 Indirect Execution.
 */
class ParticleSystem : public Core::System {
public:
    ParticleSystem(int maxParticles = 100000);
    ~ParticleSystem();
    
    // Lifecycle
    bool Initialize(Rendering::DX12RenderPipeline* pipeline);
    void Update(float deltaTime) override;
    void Shutdown() override;
    
    // Commands
    void SpawnBurst(const glm::vec3& position, int count, const glm::vec4& color);
    ID3D12Resource* GetBuffer() const { return m_particleBuffer; }
    
private:
    int m_maxParticles;
    int m_activeParticles;
    Rendering::DX12RenderPipeline* m_renderPipeline = nullptr;
    
    // CUDA Resources
    void* d_particles = nullptr;      // Device Pointer (Mapped)
    void* d_indirectArgs = nullptr;   // Device Pointer (Indirect Arguments)
    
    // DX12 Resources
    ID3D12Resource* m_particleBuffer = nullptr;
    cudaGraphicsResource* m_cudaResource = nullptr;
    
    // Internal CUDA helpers
    void LaunchUpdateKernel(float deltaTime);
    void LaunchSpawnKernel(const glm::vec3& position, int count, const glm::vec4& color);
};

} // namespace VFX
} // namespace CudaGame
