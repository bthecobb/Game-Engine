#include "VFX/ParticleSystem.h"
#include "VFX/ParticleKernels.h"
#include "Rendering/DX12RenderPipeline.h"
#include "Rendering/Backends/DX12RenderBackend.h"
#include "Core/CudaCore.h"
#include <iostream>
#include <d3d12.h>
#include <iostream>
#include <d3d12.h>
// // #include <d3dx12.h> // Ensure this is available or use direct structs

namespace CudaGame {
namespace VFX {

ParticleSystem::ParticleSystem(int maxParticles)
    : m_maxParticles(maxParticles), m_activeParticles(0) {
}

ParticleSystem::~ParticleSystem() {
    Shutdown();
}

bool ParticleSystem::Initialize(Rendering::DX12RenderPipeline* pipeline) {
    m_renderPipeline = pipeline;
    if (!m_renderPipeline || !m_renderPipeline->GetBackend()) return false;
    
    ID3D12Device* device = m_renderPipeline->GetBackend()->GetDevice();
    
    // 1. Create Particle Buffer (Structured Buffer)
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = m_maxParticles * sizeof(Particle);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    HRESULT hr = device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&m_particleBuffer)
    );

    if (FAILED(hr)) {
        std::cerr << "[ParticleSystem] Failed to create D3D12 generic buffer." << std::endl;
        return false;
    }
    
    // 2. Register with Cuda
    auto cuda = m_renderPipeline->GetCudaCore();
    if (cuda) {
        cuda->RegisterResource(m_particleBuffer, &m_cudaResource);
    }
    
    std::cout << "[ParticleSystem] Initialized (Shared Buffer Created)" << std::endl;
    return true;
}

void ParticleSystem::Update(float deltaTime) {
    if (!m_renderPipeline) return;
    
    auto cuda = m_renderPipeline->GetCudaCore();
    if (!cuda || !m_cudaResource) return;
    
    // Map
    size_t size;
    d_particles = cuda->MapResource(m_cudaResource, size);
    
    if (d_particles) {
        // Update
        LaunchParticleUpdate(d_particles, m_maxParticles, deltaTime);
        
        // Unmap
        cuda->UnmapResource(m_cudaResource);
        d_particles = nullptr;
    }
}

void ParticleSystem::SpawnBurst(const glm::vec3& position, int count, const glm::vec4& color) {
    if (!m_renderPipeline) return;
    auto cuda = m_renderPipeline->GetCudaCore();
    if (!cuda || !m_cudaResource) return;
    
    // Map
    size_t size;
    d_particles = cuda->MapResource(m_cudaResource, size);
    
    if (d_particles) {
        std::cout << "[ParticleSystem] Spawning " << count << " particles at " << position.x << ", " << position.y << std::endl;
        // Convert GLM to POD for safe boundary crossing
        float3_pod p = { position.x, position.y, position.z };
        float4_pod c = { color.r, color.g, color.b, color.a };
        
        LaunchParticleSpawn(d_particles, m_maxParticles, nullptr, p, count, c);
        
        cuda->UnmapResource(m_cudaResource);
        d_particles = nullptr;
    }
}

void ParticleSystem::Shutdown() {
    if (m_renderPipeline && m_renderPipeline->GetCudaCore() && m_cudaResource) {
        m_renderPipeline->GetCudaCore()->UnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    if (m_particleBuffer) {
        m_particleBuffer->Release();
        m_particleBuffer = nullptr;
    }
}

} // namespace VFX
} // namespace CudaGame
