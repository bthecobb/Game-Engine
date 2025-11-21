#pragma once
#ifdef _WIN32

#include <d3d12.h>
#include <dxgi1_6.h>
#include <vector>
#include <memory>
#include <glm/glm.hpp>

namespace CudaGame {
namespace Rendering {

// Ray tracing acceleration structure builder and manager
// Implements DXR 1.1 for hardware ray tracing
class RayTracingSystem {
public:
    RayTracingSystem();
    ~RayTracingSystem();

    // Initialize ray tracing with DXR 1.1
    bool Initialize(ID3D12Device5* device, ID3D12CommandQueue* cmdQueue);
    
    // Shutdown and cleanup
    void Shutdown();

    // Build Bottom-Level Acceleration Structure (BLAS) for geometry
    struct GeometryDesc {
        ID3D12Resource* vertexBuffer;
        uint32_t vertexCount;
        uint32_t vertexStride;
        DXGI_FORMAT vertexFormat;  // Usually DXGI_FORMAT_R32G32B32_FLOAT
        
        ID3D12Resource* indexBuffer;
        uint32_t indexCount;
        DXGI_FORMAT indexFormat;   // Usually DXGI_FORMAT_R32_UINT
        
        glm::mat4 transform;        // Local to world transform
        bool opaque;                // Is geometry opaque or transparent?
    };
    
    struct BLAS {
        ID3D12Resource* accelerationStructure = nullptr;
        ID3D12Resource* scratchBuffer = nullptr;
        uint64_t size = 0;
    };
    
    bool BuildBLAS(const std::vector<GeometryDesc>& geometries, BLAS& outBLAS);

    // Build Top-Level Acceleration Structure (TLAS) for instances
    struct InstanceDesc {
        BLAS* blas;                     // Which BLAS to instance
        glm::mat4 transform;            // World transform
        uint32_t instanceID;            // Shader identifier
        uint32_t instanceMask;          // Ray mask (default 0xFF)
        uint32_t hitGroupIndex;         // Shader binding table index
    };
    
    struct TLAS {
        ID3D12Resource* accelerationStructure = nullptr;
        ID3D12Resource* scratchBuffer = nullptr;
        ID3D12Resource* instanceBuffer = nullptr;
        uint64_t size = 0;
    };
    
    bool BuildTLAS(const std::vector<InstanceDesc>& instances, TLAS& outTLAS);
    
    // Update TLAS for dynamic objects (faster than rebuild)
    bool UpdateTLAS(const std::vector<InstanceDesc>& instances, TLAS& tlas);

    // Create ray tracing pipeline state object
    struct RayTracingPipelineDesc {
        const wchar_t* rayGenShader;        // Ray generation shader
        const wchar_t* missShader;          // Miss shader
        const wchar_t* closestHitShader;    // Closest hit shader
        uint32_t maxRecursionDepth;         // Max ray bounce count (1 for primary rays)
        uint32_t maxPayloadSize;            // Ray payload size in bytes
        uint32_t maxAttributeSize;          // Hit attribute size (usually 8 for barycentrics)
    };
    
    ID3D12StateObject* CreateRayTracingPipeline(const RayTracingPipelineDesc& desc);

    // Create shader binding table (SBT)
    struct ShaderBindingTable {
        ID3D12Resource* buffer = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS rayGenShaderTable = 0;
        D3D12_GPU_VIRTUAL_ADDRESS missShaderTable = 0;
        D3D12_GPU_VIRTUAL_ADDRESS hitGroupTable = 0;
        uint64_t rayGenSize = 0;
        uint64_t missSize = 0;
        uint64_t hitGroupSize = 0;
    };
    
    bool CreateShaderBindingTable(ID3D12StateObject* pso, ShaderBindingTable& outSBT);

    // Dispatch rays for rendering
    struct DispatchRaysDesc {
        ShaderBindingTable* sbt;
        uint32_t width;                 // Output width
        uint32_t height;                // Output height
        uint32_t depth;                 // Usually 1
        ID3D12Resource* outputBuffer;   // UAV for ray tracing output
    };
    
    void DispatchRays(ID3D12GraphicsCommandList4* cmdList, const DispatchRaysDesc& desc);

    // Check if ray tracing is supported
    bool IsRayTracingSupported() const { return m_isSupported; }
    
    // Get required feature support flags
    D3D12_RAYTRACING_TIER GetRayTracingTier() const { return m_rayTracingTier; }

private:
    // Helper: Allocate GPU buffer
    ID3D12Resource* AllocateBuffer(uint64_t size, D3D12_RESOURCE_FLAGS flags, 
                                   D3D12_RESOURCE_STATES initialState, const wchar_t* name);
    
    // Helper: Get required sizes for acceleration structures
    void GetAccelerationStructureSizes(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& inputs,
                                      uint64_t& scratchSize, uint64_t& resultSize);

    ID3D12Device5* m_device = nullptr;
    ID3D12CommandQueue* m_cmdQueue = nullptr;
    ID3D12CommandAllocator* m_cmdAllocator = nullptr;
    ID3D12GraphicsCommandList4* m_cmdList = nullptr;
    
    bool m_initialized = false;
    bool m_isSupported = false;
    D3D12_RAYTRACING_TIER m_rayTracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
    
    // Descriptor heaps for ray tracing
    ID3D12DescriptorHeap* m_srvHeap = nullptr;
    uint32_t m_srvDescriptorSize = 0;
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
