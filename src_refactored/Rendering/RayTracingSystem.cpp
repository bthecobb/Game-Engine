#ifdef _WIN32
#include "Rendering/RayTracingSystem.h"
#include <iostream>

namespace CudaGame {
namespace Rendering {

RayTracingSystem::RayTracingSystem() = default;

RayTracingSystem::~RayTracingSystem() {
    Shutdown();
}

bool RayTracingSystem::Initialize(ID3D12Device5* device, ID3D12CommandQueue* cmdQueue) {
    if (m_initialized) {
        std::cerr << "[RT] Already initialized" << std::endl;
        return true;
    }

    if (!device || !cmdQueue) {
        std::cerr << "[RT] Invalid device or command queue" << std::endl;
        return false;
    }

    m_device = device;
    m_cmdQueue = cmdQueue;

    std::cout << "[RT] Initializing DXR 1.1 Ray Tracing..." << std::endl;

    // Check ray tracing support
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    HRESULT hr = m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5));
    
    if (FAILED(hr) || options5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
        std::cerr << "[RT] Ray tracing not supported on this hardware" << std::endl;
        m_isSupported = false;
        m_initialized = true;
        return false;
    }

    m_rayTracingTier = options5.RaytracingTier;
    m_isSupported = true;

    std::cout << "[RT] Ray tracing tier: " << static_cast<int>(m_rayTracingTier) << std::endl;
    std::cout << "  Tier 1.0: Basic ray tracing" << std::endl;
    std::cout << "  Tier 1.1: Inline ray tracing, additional features" << std::endl;

    // Create command allocator and list
    hr = m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_cmdAllocator));
    if (FAILED(hr)) {
        std::cerr << "[RT] Failed to create command allocator" << std::endl;
        return false;
    }

    hr = m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmdAllocator, nullptr, IID_PPV_ARGS(&m_cmdList));
    if (FAILED(hr)) {
        std::cerr << "[RT] Failed to create command list" << std::endl;
        return false;
    }

    m_cmdList->Close();

    // Create descriptor heap for SRVs
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 256;  // Enough for ray tracing resources
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    
    hr = m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_srvHeap));
    if (FAILED(hr)) {
        std::cerr << "[RT] Failed to create descriptor heap" << std::endl;
        return false;
    }

    m_srvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    m_initialized = true;
    std::cout << "[RT] Ray tracing initialized successfully" << std::endl;
    return true;
}

void RayTracingSystem::Shutdown() {
    if (!m_initialized) return;

    if (m_srvHeap) {
        m_srvHeap->Release();
        m_srvHeap = nullptr;
    }

    if (m_cmdList) {
        m_cmdList->Release();
        m_cmdList = nullptr;
    }

    if (m_cmdAllocator) {
        m_cmdAllocator->Release();
        m_cmdAllocator = nullptr;
    }

    m_initialized = false;
    m_isSupported = false;
    std::cout << "[RT] Ray tracing shutdown complete" << std::endl;
}

bool RayTracingSystem::BuildBLAS(const std::vector<GeometryDesc>& geometries, BLAS& outBLAS) {
    if (!m_isSupported) {
        std::cerr << "[RT] Ray tracing not supported" << std::endl;
        return false;
    }

    // TODO: Implement BLAS building
    // 1. Create D3D12_RAYTRACING_GEOMETRY_DESC for each geometry
    // 2. Fill D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS
    // 3. Query required sizes with GetRaytracingAccelerationStructurePrebuildInfo
    // 4. Allocate scratch and result buffers
    // 5. Build BLAS with BuildRaytracingAccelerationStructure
    
    std::cout << "[RT] STUB: Would build BLAS for " << geometries.size() << " geometries" << std::endl;
    return false;
}

bool RayTracingSystem::BuildTLAS(const std::vector<InstanceDesc>& instances, TLAS& outTLAS) {
    if (!m_isSupported) {
        std::cerr << "[RT] Ray tracing not supported" << std::endl;
        return false;
    }

    // TODO: Implement TLAS building
    // 1. Create D3D12_RAYTRACING_INSTANCE_DESC for each instance
    // 2. Upload instance descs to GPU buffer
    // 3. Fill D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS
    // 4. Query required sizes
    // 5. Allocate scratch and result buffers
    // 6. Build TLAS with BuildRaytracingAccelerationStructure
    
    std::cout << "[RT] STUB: Would build TLAS for " << instances.size() << " instances" << std::endl;
    return false;
}

bool RayTracingSystem::UpdateTLAS(const std::vector<InstanceDesc>& instances, TLAS& tlas) {
    if (!m_isSupported) {
        std::cerr << "[RT] Ray tracing not supported" << std::endl;
        return false;
    }

    // TODO: Implement TLAS update
    // Use D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE
    // This is faster than full rebuild for dynamic objects
    
    std::cout << "[RT] STUB: Would update TLAS for " << instances.size() << " instances" << std::endl;
    return false;
}

ID3D12StateObject* RayTracingSystem::CreateRayTracingPipeline(const RayTracingPipelineDesc& desc) {
    if (!m_isSupported) {
        std::cerr << "[RT] Ray tracing not supported" << std::endl;
        return nullptr;
    }

    // TODO: Implement ray tracing pipeline creation
    // 1. Create D3D12_STATE_OBJECT_DESC with type D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE
    // 2. Add DXIL libraries for shaders
    // 3. Define hit groups
    // 4. Set shader config (payload and attribute sizes)
    // 5. Set pipeline config (max recursion depth)
    // 6. Create state object with CreateStateObject
    
    std::cout << "[RT] STUB: Would create ray tracing pipeline" << std::endl;
    std::cout << "  RayGen: " << (desc.rayGenShader ? "Yes" : "No") << std::endl;
    std::cout << "  Miss: " << (desc.missShader ? "Yes" : "No") << std::endl;
    std::cout << "  ClosestHit: " << (desc.closestHitShader ? "Yes" : "No") << std::endl;
    std::cout << "  Max recursion: " << desc.maxRecursionDepth << std::endl;
    
    return nullptr;
}

bool RayTracingSystem::CreateShaderBindingTable(ID3D12StateObject* pso, ShaderBindingTable& outSBT) {
    if (!m_isSupported || !pso) {
        std::cerr << "[RT] Invalid parameters for SBT creation" << std::endl;
        return false;
    }

    // TODO: Implement shader binding table creation
    // 1. Get shader identifiers from state object properties
    // 2. Calculate aligned sizes for each shader table section
    // 3. Allocate upload buffer
    // 4. Write shader identifiers and root arguments
    // 5. Copy to GPU buffer
    
    std::cout << "[RT] STUB: Would create shader binding table" << std::endl;
    return false;
}

void RayTracingSystem::DispatchRays(ID3D12GraphicsCommandList4* cmdList, const DispatchRaysDesc& desc) {
    if (!m_isSupported) {
        return;
    }

    // TODO: Implement ray dispatch
    // 1. Set ray tracing pipeline
    // 2. Set descriptor heaps
    // 3. Fill D3D12_DISPATCH_RAYS_DESC with SBT info
    // 4. Call DispatchRays
    
    static uint32_t frameCount = 0;
    if (frameCount % 60 == 0) {
        std::cout << "[RT] STUB: Would dispatch rays " << desc.width << "x" << desc.height << std::endl;
    }
    frameCount++;
}

ID3D12Resource* RayTracingSystem::AllocateBuffer(uint64_t size, D3D12_RESOURCE_FLAGS flags,
                                                 D3D12_RESOURCE_STATES initialState, const wchar_t* name) {
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Width = size;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = flags;

    ID3D12Resource* buffer = nullptr;
    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&buffer)
    );

    if (FAILED(hr)) {
        std::cerr << "[RT] Failed to allocate buffer" << std::endl;
        return nullptr;
    }

    if (name) {
        buffer->SetName(name);
    }

    return buffer;
}

void RayTracingSystem::GetAccelerationStructureSizes(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& inputs,
                                                     uint64_t& scratchSize, uint64_t& resultSize) {
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);
    
    scratchSize = info.ScratchDataSizeInBytes;
    resultSize = info.ResultDataMaxSizeInBytes;
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
