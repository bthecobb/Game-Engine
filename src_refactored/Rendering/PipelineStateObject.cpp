#ifdef _WIN32
#include "Rendering/PipelineStateObject.h"
#include <iostream>
#include <sstream>

namespace CudaGame {
namespace Rendering {

// ========== Vertex Input Layout Presets ==========

VertexInputLayout VertexInputLayout::Position3D() {
    VertexInputLayout layout;
    layout.elements = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };
    return layout;
}

VertexInputLayout VertexInputLayout::PositionColor() {
    VertexInputLayout layout;
    layout.elements = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };
    return layout;
}

VertexInputLayout VertexInputLayout::PositionNormalTexcoord() {
    VertexInputLayout layout;
    layout.elements = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };
    return layout;
}

VertexInputLayout VertexInputLayout::PositionNormalTangentTexcoord() {
    VertexInputLayout layout;
    layout.elements = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };
    return layout;
}

// ========== Pipeline State Manager ==========

bool PipelineStateManager::CreatePipelineState(
    ID3D12Device* device,
    const PipelineStateDesc& desc,
    PipelineStateObject& outPSO,
    std::string& outErrorMsg
) {
    if (!device) {
        outErrorMsg = "Device is null";
        return false;
    }

    if (!desc.vertexShader) {
        outErrorMsg = "Vertex shader is required";
        return false;
    }

    if (!desc.rootSignature) {
        outErrorMsg = "Root signature is required";
        return false;
    }

    // Check cache first
    std::string cacheKey = GenerateCacheKey(desc);
    const PipelineStateObject* cached = GetCachedPSO(cacheKey);
    if (cached) {
        outPSO = *cached;
        return true;
    }

    // Create PSO
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    
    // Root signature
    psoDesc.pRootSignature = desc.rootSignature;
    
    // Shaders
    psoDesc.VS = {desc.vertexShader->GetBufferPointer(), desc.vertexShader->GetBufferSize()};
    if (desc.pixelShader) {
        psoDesc.PS = {desc.pixelShader->GetBufferPointer(), desc.pixelShader->GetBufferSize()};
    }
    if (desc.geometryShader) {
        psoDesc.GS = {desc.geometryShader->GetBufferPointer(), desc.geometryShader->GetBufferSize()};
    }
    if (desc.hullShader) {
        psoDesc.HS = {desc.hullShader->GetBufferPointer(), desc.hullShader->GetBufferSize()};
    }
    if (desc.domainShader) {
        psoDesc.DS = {desc.domainShader->GetBufferPointer(), desc.domainShader->GetBufferSize()};
    }
    
    // Input layout
    auto inputElements = CreateInputLayout(desc.inputLayout);
    psoDesc.InputLayout = {inputElements.data(), static_cast<UINT>(inputElements.size())};
    
    // Blend state
    psoDesc.BlendState = CreateBlendDesc(desc.blendMode);
    
    // Rasterizer state
    psoDesc.RasterizerState = CreateRasterizerDesc(desc.cullMode, desc.fillMode);
    
    // Depth stencil state
    psoDesc.DepthStencilState = CreateDepthStencilDesc(desc.depthMode);
    
    // Render targets
    psoDesc.NumRenderTargets = static_cast<UINT>(desc.rtvFormats.size());
    for (size_t i = 0; i < desc.rtvFormats.size() && i < 8; ++i) {
        psoDesc.RTVFormats[i] = desc.rtvFormats[i];
    }
    psoDesc.DSVFormat = desc.dsvFormat;
    
    // Sample desc
    psoDesc.SampleDesc.Count = desc.sampleCount;
    psoDesc.SampleDesc.Quality = desc.sampleQuality;
    psoDesc.SampleMask = UINT_MAX;
    
    // Topology
    psoDesc.PrimitiveTopologyType = desc.topologyType;
    
    // Create PSO
    ComPtr<ID3D12PipelineState> pso;
    HRESULT hr = device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        outErrorMsg = "Failed to create graphics pipeline state. HRESULT: " + std::to_string(hr);
        return false;
    }
    
    // Set debug name if provided
    if (desc.debugName) {
        std::wstring wideName(desc.debugName, desc.debugName + strlen(desc.debugName));
        pso->SetName(wideName.c_str());
    }
    
    // Fill output
    outPSO.pso = pso;
    outPSO.rootSignature = desc.rootSignature;
    outPSO.debugName = desc.debugName ? desc.debugName : "";
    
    // Cache
    CachePSO(cacheKey, outPSO);
    
    return true;
}

const PipelineStateObject* PipelineStateManager::GetCachedPSO(const std::string& key) const {
    auto it = m_psoCache.find(key);
    if (it != m_psoCache.end()) {
        return &it->second;
    }
    return nullptr;
}

void PipelineStateManager::CachePSO(const std::string& key, const PipelineStateObject& pso) {
    m_psoCache[key] = pso;
}

void PipelineStateManager::ClearCache() {
    m_psoCache.clear();
}

std::string PipelineStateManager::GenerateCacheKey(const PipelineStateDesc& desc) {
    std::stringstream ss;
    ss << reinterpret_cast<uintptr_t>(desc.vertexShader) << "_";
    ss << reinterpret_cast<uintptr_t>(desc.pixelShader) << "_";
    ss << static_cast<int>(desc.blendMode) << "_";
    ss << static_cast<int>(desc.depthMode) << "_";
    ss << static_cast<int>(desc.cullMode) << "_";
    ss << static_cast<int>(desc.fillMode) << "_";
    ss << desc.rtvFormats.size();
    return ss.str();
}

D3D12_BLEND_DESC PipelineStateManager::CreateBlendDesc(BlendMode mode) const {
    D3D12_BLEND_DESC blendDesc = {};
    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.IndependentBlendEnable = FALSE;
    
    D3D12_RENDER_TARGET_BLEND_DESC rtBlend = {};
    rtBlend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    
    switch (mode) {
        case BlendMode::Opaque:
            rtBlend.BlendEnable = FALSE;
            break;
        case BlendMode::AlphaBlend:
            rtBlend.BlendEnable = TRUE;
            rtBlend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
            rtBlend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
            rtBlend.BlendOp = D3D12_BLEND_OP_ADD;
            rtBlend.SrcBlendAlpha = D3D12_BLEND_ONE;
            rtBlend.DestBlendAlpha = D3D12_BLEND_ZERO;
            rtBlend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            break;
        case BlendMode::Additive:
            rtBlend.BlendEnable = TRUE;
            rtBlend.SrcBlend = D3D12_BLEND_ONE;
            rtBlend.DestBlend = D3D12_BLEND_ONE;
            rtBlend.BlendOp = D3D12_BLEND_OP_ADD;
            rtBlend.SrcBlendAlpha = D3D12_BLEND_ONE;
            rtBlend.DestBlendAlpha = D3D12_BLEND_ZERO;
            rtBlend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            break;
        case BlendMode::Premultiplied:
            rtBlend.BlendEnable = TRUE;
            rtBlend.SrcBlend = D3D12_BLEND_ONE;
            rtBlend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
            rtBlend.BlendOp = D3D12_BLEND_OP_ADD;
            rtBlend.SrcBlendAlpha = D3D12_BLEND_ONE;
            rtBlend.DestBlendAlpha = D3D12_BLEND_ZERO;
            rtBlend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            break;
    }
    
    for (int i = 0; i < 8; ++i) {
        blendDesc.RenderTarget[i] = rtBlend;
    }
    
    return blendDesc;
}

D3D12_DEPTH_STENCIL_DESC PipelineStateManager::CreateDepthStencilDesc(DepthMode mode) const {
    D3D12_DEPTH_STENCIL_DESC depthDesc = {};
    depthDesc.StencilEnable = FALSE;
    depthDesc.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    depthDesc.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    
    switch (mode) {
        case DepthMode::None:
            depthDesc.DepthEnable = FALSE;
            depthDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
            depthDesc.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            break;
        case DepthMode::ReadWrite:
            depthDesc.DepthEnable = TRUE;
            depthDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            depthDesc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
            break;
        case DepthMode::ReadOnly:
            depthDesc.DepthEnable = TRUE;
            depthDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
            depthDesc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
            break;
        case DepthMode::WriteOnly:
            depthDesc.DepthEnable = FALSE;
            depthDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            depthDesc.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            break;
    }
    
    return depthDesc;
}

D3D12_RASTERIZER_DESC PipelineStateManager::CreateRasterizerDesc(CullMode cullMode, FillMode fillMode) const {
    D3D12_RASTERIZER_DESC rasterDesc = {};
    rasterDesc.FillMode = (fillMode == FillMode::Solid) ? D3D12_FILL_MODE_SOLID : D3D12_FILL_MODE_WIREFRAME;
    
    switch (cullMode) {
        case CullMode::None:
            rasterDesc.CullMode = D3D12_CULL_MODE_NONE;
            break;
        case CullMode::Front:
            rasterDesc.CullMode = D3D12_CULL_MODE_FRONT;
            break;
        case CullMode::Back:
            rasterDesc.CullMode = D3D12_CULL_MODE_BACK;
            break;
    }
    
    rasterDesc.FrontCounterClockwise = FALSE;
    rasterDesc.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
    rasterDesc.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    rasterDesc.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
    rasterDesc.DepthClipEnable = TRUE;
    rasterDesc.MultisampleEnable = FALSE;
    rasterDesc.AntialiasedLineEnable = FALSE;
    rasterDesc.ForcedSampleCount = 0;
    rasterDesc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    
    return rasterDesc;
}

std::vector<D3D12_INPUT_ELEMENT_DESC> PipelineStateManager::CreateInputLayout(const VertexInputLayout& layout) const {
    std::vector<D3D12_INPUT_ELEMENT_DESC> d3dElements;
    d3dElements.reserve(layout.elements.size());
    
    for (const auto& elem : layout.elements) {
        D3D12_INPUT_ELEMENT_DESC d3dElem = {};
        d3dElem.SemanticName = elem.semanticName;
        d3dElem.SemanticIndex = elem.semanticIndex;
        d3dElem.Format = elem.format;
        d3dElem.InputSlot = elem.inputSlot;
        d3dElem.AlignedByteOffset = elem.alignedByteOffset;
        d3dElem.InputSlotClass = elem.inputSlotClass;
        d3dElem.InstanceDataStepRate = elem.instanceDataStepRate;
        d3dElements.push_back(d3dElem);
    }
    
    return d3dElements;
}

// ========== Root Signature Builder ==========

RootSignatureBuilder& RootSignatureBuilder::AddConstants(uint32_t num32BitValues, uint32_t shaderRegister, uint32_t registerSpace) {
    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants.Num32BitValues = num32BitValues;
    param.Constants.ShaderRegister = shaderRegister;
    param.Constants.RegisterSpace = registerSpace;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    m_rootParams.push_back(param);
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::AddCBV(uint32_t shaderRegister, uint32_t registerSpace) {
    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    param.Descriptor.ShaderRegister = shaderRegister;
    param.Descriptor.RegisterSpace = registerSpace;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    m_rootParams.push_back(param);
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::AddSRV(uint32_t shaderRegister, uint32_t registerSpace) {
    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    param.Descriptor.ShaderRegister = shaderRegister;
    param.Descriptor.RegisterSpace = registerSpace;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    m_rootParams.push_back(param);
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::AddUAV(uint32_t shaderRegister, uint32_t registerSpace) {
    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    param.Descriptor.ShaderRegister = shaderRegister;
    param.Descriptor.RegisterSpace = registerSpace;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    m_rootParams.push_back(param);
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::AddDescriptorTable(
    D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
    uint32_t numDescriptors,
    uint32_t baseShaderRegister,
    uint32_t registerSpace
) {
    // Create descriptor range
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges(1);
    ranges[0].RangeType = rangeType;
    ranges[0].NumDescriptors = numDescriptors;
    ranges[0].BaseShaderRegister = baseShaderRegister;
    ranges[0].RegisterSpace = registerSpace;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    
    m_descriptorRanges.push_back(ranges);
    
    // Create root parameter pointing to this range
    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = 1;
    param.DescriptorTable.pDescriptorRanges = m_descriptorRanges.back().data();
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    m_rootParams.push_back(param);
    
    return *this;
}

RootSignatureBuilder& RootSignatureBuilder::AddStaticSampler(
    uint32_t shaderRegister,
    D3D12_FILTER filter,
    D3D12_TEXTURE_ADDRESS_MODE addressMode
) {
    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = filter;
    sampler.AddressU = addressMode;
    sampler.AddressV = addressMode;
    sampler.AddressW = addressMode;
    sampler.MipLODBias = 0;
    sampler.MaxAnisotropy = 16;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    sampler.MinLOD = 0.0f;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = shaderRegister;
    sampler.RegisterSpace = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    m_staticSamplers.push_back(sampler);
    return *this;
}

bool RootSignatureBuilder::Build(ID3D12Device* device, Microsoft::WRL::ComPtr<ID3D12RootSignature>& outRootSig, std::string& outErrorMsg) {
    if (!device) {
        outErrorMsg = "Device is null";
        return false;
    }
    
    D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.NumParameters = static_cast<UINT>(m_rootParams.size());
    rootSigDesc.pParameters = m_rootParams.empty() ? nullptr : m_rootParams.data();
    rootSigDesc.NumStaticSamplers = static_cast<UINT>(m_staticSamplers.size());
    rootSigDesc.pStaticSamplers = m_staticSamplers.empty() ? nullptr : m_staticSamplers.data();
    rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    
    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    
    if (FAILED(hr)) {
        if (error) {
            outErrorMsg = std::string(static_cast<const char*>(error->GetBufferPointer()), error->GetBufferSize());
        } else {
            outErrorMsg = "Failed to serialize root signature. HRESULT: " + std::to_string(hr);
        }
        return false;
    }
    
    hr = device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&outRootSig));
    if (FAILED(hr)) {
        outErrorMsg = "Failed to create root signature. HRESULT: " + std::to_string(hr);
        return false;
    }
    
    return true;
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
