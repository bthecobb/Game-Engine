#pragma once

#ifdef _WIN32
#include "Rendering/ShaderManager.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace CudaGame {
namespace Rendering {

// Vertex input layout - describes vertex buffer structure
struct VertexElement {
    const char* semanticName;
    uint32_t semanticIndex;
    DXGI_FORMAT format;
    uint32_t inputSlot;
    uint32_t alignedByteOffset;
    D3D12_INPUT_CLASSIFICATION inputSlotClass;
    uint32_t instanceDataStepRate;
};

// Common vertex formats
struct VertexInputLayout {
    std::vector<VertexElement> elements;
    
    // Helpers to create common layouts
    static VertexInputLayout Position3D();
    static VertexInputLayout PositionColor();
    static VertexInputLayout PositionNormalTexcoord();
    static VertexInputLayout PositionNormalTangentTexcoord();
};

// Blend state presets
enum class BlendMode {
    Opaque,          // No blending
    AlphaBlend,      // Standard alpha blending
    Additive,        // Additive blending
    Premultiplied    // Premultiplied alpha
};

// Depth state presets
enum class DepthMode {
    None,            // No depth test/write
    ReadWrite,       // Read and write depth
    ReadOnly,        // Read depth, no write
    WriteOnly        // Write depth, no read
};

// Rasterizer state presets
enum class CullMode {
    None,
    Front,
    Back
};

enum class FillMode {
    Solid,
    Wireframe
};

// PSO Description - all the state needed to create a pipeline
struct PipelineStateDesc {
    // Shaders
    ShaderBytecode* vertexShader = nullptr;
    ShaderBytecode* pixelShader = nullptr;
    ShaderBytecode* geometryShader = nullptr;
    ShaderBytecode* hullShader = nullptr;
    ShaderBytecode* domainShader = nullptr;
    
    // Input layout
    VertexInputLayout inputLayout;
    
    // Render target formats
    std::vector<DXGI_FORMAT> rtvFormats;
    DXGI_FORMAT dsvFormat = DXGI_FORMAT_UNKNOWN;
    uint32_t sampleCount = 1;
    uint32_t sampleQuality = 0;
    
    // State presets
    BlendMode blendMode = BlendMode::Opaque;
    DepthMode depthMode = DepthMode::ReadWrite;
    CullMode cullMode = CullMode::Back;
    FillMode fillMode = FillMode::Solid;
    
    // Primitive topology
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    
    // Root signature
    ID3D12RootSignature* rootSignature = nullptr;
    
    // Debug name
    const char* debugName = nullptr;
};

// PSO Cache Entry
struct PipelineStateObject {
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
    ID3D12RootSignature* rootSignature = nullptr;
    std::string debugName;
    
    bool IsValid() const { return pso != nullptr; }
};

// AAA-grade PSO manager with caching and validation
class PipelineStateManager {
public:
    PipelineStateManager() = default;
    ~PipelineStateManager() = default;
    
    // Create PSO from description
    bool CreatePipelineState(
        ID3D12Device* device,
        const PipelineStateDesc& desc,
        PipelineStateObject& outPSO,
        std::string& outErrorMsg
    );
    
    // Get cached PSO (returns nullptr if not found)
    const PipelineStateObject* GetCachedPSO(const std::string& key) const;
    
    // Cache management
    void CachePSO(const std::string& key, const PipelineStateObject& pso);
    void ClearCache();
    size_t GetCacheSize() const { return m_psoCache.size(); }
    
    // Utility: Generate cache key from PSO description
    static std::string GenerateCacheKey(const PipelineStateDesc& desc);
    
private:
    template<typename T>
    using ComPtr = Microsoft::WRL::ComPtr<T>;
    
    // PSO cache (key = hash of desc)
    std::unordered_map<std::string, PipelineStateObject> m_psoCache;
    
    // Helper: Convert blend mode to D3D12 blend desc
    D3D12_BLEND_DESC CreateBlendDesc(BlendMode mode) const;
    
    // Helper: Convert depth mode to D3D12 depth stencil desc
    D3D12_DEPTH_STENCIL_DESC CreateDepthStencilDesc(DepthMode mode) const;
    
    // Helper: Convert rasterizer preset to D3D12 rasterizer desc
    D3D12_RASTERIZER_DESC CreateRasterizerDesc(CullMode cullMode, FillMode fillMode) const;
    
    // Helper: Convert VertexInputLayout to D3D12 input layout
    std::vector<D3D12_INPUT_ELEMENT_DESC> CreateInputLayout(const VertexInputLayout& layout) const;
};

// Root signature builder helper (AAA workflow feature)
class RootSignatureBuilder {
public:
    RootSignatureBuilder() = default;
    
    // Add root constants (inline constants, fastest)
    RootSignatureBuilder& AddConstants(uint32_t num32BitValues, uint32_t shaderRegister, uint32_t registerSpace = 0);
    
    // Add CBV (constant buffer view)
    RootSignatureBuilder& AddCBV(uint32_t shaderRegister, uint32_t registerSpace = 0);
    
    // Add SRV (shader resource view - textures, buffers)
    RootSignatureBuilder& AddSRV(uint32_t shaderRegister, uint32_t registerSpace = 0);
    
    // Add UAV (unordered access view - RW textures, buffers)
    RootSignatureBuilder& AddUAV(uint32_t shaderRegister, uint32_t registerSpace = 0);
    
    // Add descriptor table (multiple descriptors)
    RootSignatureBuilder& AddDescriptorTable(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        uint32_t numDescriptors,
        uint32_t baseShaderRegister,
        uint32_t registerSpace = 0
    );
    
    // Add static sampler
    RootSignatureBuilder& AddStaticSampler(
        uint32_t shaderRegister,
        D3D12_FILTER filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE addressMode = D3D12_TEXTURE_ADDRESS_MODE_WRAP
    );
    
    // Build root signature
    bool Build(ID3D12Device* device, Microsoft::WRL::ComPtr<ID3D12RootSignature>& outRootSig, std::string& outErrorMsg);
    
private:
    std::vector<D3D12_ROOT_PARAMETER> m_rootParams;
    std::vector<D3D12_STATIC_SAMPLER_DESC> m_staticSamplers;
    std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> m_descriptorRanges; // Storage for table ranges
};

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
