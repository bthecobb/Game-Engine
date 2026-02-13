#ifdef _WIN32
#include <gtest/gtest.h>
#include "Rendering/PipelineStateObject.h"
#include "Rendering/Backends/DX12RenderBackend.h"
#include <d3d12.h>

using namespace CudaGame::Rendering;

class DX12PipelineStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize DX12 device
        m_backend = std::make_unique<DX12RenderBackend>();
        ASSERT_TRUE(m_backend->Initialize()) << "Failed to initialize DX12 backend";
        m_device = m_backend->GetDevice();
        ASSERT_NE(m_device, nullptr) << "Device is null";
    }
    
    void TearDown() override {
        m_backend.reset();
    }
    
    ID3D12Device* m_device = nullptr;
    std::unique_ptr<DX12RenderBackend> m_backend;
    PipelineStateManager m_psoManager;
};

// Test 1: Vertex input layout - Position3D
TEST_F(DX12PipelineStateTest, VertexLayoutPosition3D) {
    auto layout = VertexInputLayout::Position3D();
    
    EXPECT_EQ(layout.elements.size(), 1);
    EXPECT_STREQ(layout.elements[0].semanticName, "POSITION");
    EXPECT_EQ(layout.elements[0].semanticIndex, 0);
    EXPECT_EQ(layout.elements[0].format, DXGI_FORMAT_R32G32B32_FLOAT);
}

// Test 2: Vertex input layout - PositionColor
TEST_F(DX12PipelineStateTest, VertexLayoutPositionColor) {
    auto layout = VertexInputLayout::PositionColor();
    
    EXPECT_EQ(layout.elements.size(), 2);
    EXPECT_STREQ(layout.elements[0].semanticName, "POSITION");
    EXPECT_STREQ(layout.elements[1].semanticName, "COLOR");
    EXPECT_EQ(layout.elements[1].alignedByteOffset, 12); // After 3 floats
}

// Test 3: Vertex input layout - PositionNormalTexcoord
TEST_F(DX12PipelineStateTest, VertexLayoutPositionNormalTexcoord) {
    auto layout = VertexInputLayout::PositionNormalTexcoord();
    
    EXPECT_EQ(layout.elements.size(), 3);
    EXPECT_STREQ(layout.elements[0].semanticName, "POSITION");
    EXPECT_STREQ(layout.elements[1].semanticName, "NORMAL");
    EXPECT_STREQ(layout.elements[2].semanticName, "TEXCOORD");
    EXPECT_EQ(layout.elements[2].alignedByteOffset, 24); // After position + normal
}

// Test 4: Vertex input layout - PositionNormalTangentTexcoord
TEST_F(DX12PipelineStateTest, VertexLayoutPositionNormalTangentTexcoord) {
    auto layout = VertexInputLayout::PositionNormalTangentTexcoord();
    
    EXPECT_EQ(layout.elements.size(), 4);
    EXPECT_STREQ(layout.elements[0].semanticName, "POSITION");
    EXPECT_STREQ(layout.elements[1].semanticName, "NORMAL");
    EXPECT_STREQ(layout.elements[2].semanticName, "TANGENT");
    EXPECT_STREQ(layout.elements[3].semanticName, "TEXCOORD");
}

// Test 5: Root signature builder - Empty
TEST_F(DX12PipelineStateTest, RootSignatureBuilderEmpty) {
    RootSignatureBuilder builder;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    std::string errorMsg;
    
    bool success = builder.Build(m_device, rootSig, errorMsg);
    
    EXPECT_TRUE(success) << "Failed to build empty root signature: " << errorMsg;
    EXPECT_NE(rootSig.Get(), nullptr);
}

// Test 6: Root signature builder - Single CBV
TEST_F(DX12PipelineStateTest, RootSignatureBuilderCBV) {
    RootSignatureBuilder builder;
    builder.AddCBV(0, 0); // b0
    
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    std::string errorMsg;
    
    bool success = builder.Build(m_device, rootSig, errorMsg);
    
    EXPECT_TRUE(success) << "Failed to build root signature with CBV: " << errorMsg;
    EXPECT_NE(rootSig.Get(), nullptr);
}

// Test 7: Root signature builder - Constants + CBV
TEST_F(DX12PipelineStateTest, RootSignatureBuilderConstantsAndCBV) {
    RootSignatureBuilder builder;
    builder.AddConstants(4, 0, 0);  // 4x 32-bit constants at b0
    builder.AddCBV(1, 0);            // CBV at b1
    
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    std::string errorMsg;
    
    bool success = builder.Build(m_device, rootSig, errorMsg);
    
    EXPECT_TRUE(success) << "Failed to build complex root signature: " << errorMsg;
    EXPECT_NE(rootSig.Get(), nullptr);
}

// Test 8: Root signature builder - With sampler
TEST_F(DX12PipelineStateTest, RootSignatureBuilderWithSampler) {
    RootSignatureBuilder builder;
    builder.AddCBV(0, 0);
    builder.AddStaticSampler(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    std::string errorMsg;
    
    bool success = builder.Build(m_device, rootSig, errorMsg);
    
    EXPECT_TRUE(success) << "Failed to build root signature with sampler: " << errorMsg;
    EXPECT_NE(rootSig.Get(), nullptr);
}

// Test 9: Root signature builder - Descriptor table
TEST_F(DX12PipelineStateTest, RootSignatureBuilderDescriptorTable) {
    RootSignatureBuilder builder;
    builder.AddDescriptorTable(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0, 0); // 4 SRVs starting at t0
    
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    std::string errorMsg;
    
    bool success = builder.Build(m_device, rootSig, errorMsg);
    
    EXPECT_TRUE(success) << "Failed to build root signature with descriptor table: " << errorMsg;
    EXPECT_NE(rootSig.Get(), nullptr);
}

// Test 10: PSO cache key generation
TEST_F(DX12PipelineStateTest, PSOCacheKeyGeneration) {
    ShaderBytecode vs, ps;
    vs.data = {1, 2, 3};
    ps.data = {4, 5, 6};
    
    PipelineStateDesc desc1;
    desc1.vertexShader = &vs;
    desc1.pixelShader = &ps;
    desc1.blendMode = BlendMode::Opaque;
    desc1.depthMode = DepthMode::ReadWrite;
    
    PipelineStateDesc desc2 = desc1;
    
    PipelineStateDesc desc3 = desc1;
    desc3.blendMode = BlendMode::AlphaBlend;
    
    std::string key1 = PipelineStateManager::GenerateCacheKey(desc1);
    std::string key2 = PipelineStateManager::GenerateCacheKey(desc2);
    std::string key3 = PipelineStateManager::GenerateCacheKey(desc3);
    
    EXPECT_EQ(key1, key2) << "Identical PSO descs should generate same key";
    EXPECT_NE(key1, key3) << "Different blend mode should generate different key";
}

// Test 11: PSO cache management
TEST_F(DX12PipelineStateTest, PSOCacheManagement) {
    EXPECT_EQ(m_psoManager.GetCacheSize(), 0) << "Cache should start empty";
    
    PipelineStateObject pso;
    m_psoManager.CachePSO("test_key", pso);
    
    EXPECT_EQ(m_psoManager.GetCacheSize(), 1) << "Cache should contain 1 PSO";
    
    const PipelineStateObject* cached = m_psoManager.GetCachedPSO("test_key");
    EXPECT_NE(cached, nullptr) << "Should retrieve cached PSO";
    
    m_psoManager.ClearCache();
    EXPECT_EQ(m_psoManager.GetCacheSize(), 0) << "Cache should be empty after clear";
}

// Test 12: PSO creation validation - Missing vertex shader
TEST_F(DX12PipelineStateTest, PSOCreationMissingVertexShader) {
    PipelineStateDesc desc;
    desc.vertexShader = nullptr;
    
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    RootSignatureBuilder().Build(m_device, rootSig, std::string());
    desc.rootSignature = rootSig.Get();
    
    PipelineStateObject outPSO;
    std::string errorMsg;
    
    bool success = m_psoManager.CreatePipelineState(m_device, desc, outPSO, errorMsg);
    
    EXPECT_FALSE(success) << "Should fail without vertex shader";
    EXPECT_FALSE(errorMsg.empty()) << "Error message should be present";
}

// Test 13: PSO creation validation - Missing root signature
TEST_F(DX12PipelineStateTest, PSOCreationMissingRootSignature) {
    ShaderBytecode vs;
    vs.data = {1, 2, 3};
    
    PipelineStateDesc desc;
    desc.vertexShader = &vs;
    desc.rootSignature = nullptr;
    
    PipelineStateObject outPSO;
    std::string errorMsg;
    
    bool success = m_psoManager.CreatePipelineState(m_device, desc, outPSO, errorMsg);
    
    EXPECT_FALSE(success) << "Should fail without root signature";
    EXPECT_FALSE(errorMsg.empty()) << "Error message should be present";
}

// Test 14: Blend mode presets
TEST_F(DX12PipelineStateTest, BlendModePresets) {
    // Test that blend modes are distinct
    EXPECT_NE(static_cast<int>(BlendMode::Opaque), static_cast<int>(BlendMode::AlphaBlend));
    EXPECT_NE(static_cast<int>(BlendMode::AlphaBlend), static_cast<int>(BlendMode::Additive));
    EXPECT_NE(static_cast<int>(BlendMode::Additive), static_cast<int>(BlendMode::Premultiplied));
}

// Test 15: Depth mode presets
TEST_F(DX12PipelineStateTest, DepthModePresets) {
    // Test that depth modes are distinct
    EXPECT_NE(static_cast<int>(DepthMode::None), static_cast<int>(DepthMode::ReadWrite));
    EXPECT_NE(static_cast<int>(DepthMode::ReadWrite), static_cast<int>(DepthMode::ReadOnly));
    EXPECT_NE(static_cast<int>(DepthMode::ReadOnly), static_cast<int>(DepthMode::WriteOnly));
}

// Test 16: Cull mode presets
TEST_F(DX12PipelineStateTest, CullModePresets) {
    // Test that cull modes are distinct
    EXPECT_NE(static_cast<int>(CullMode::None), static_cast<int>(CullMode::Front));
    EXPECT_NE(static_cast<int>(CullMode::Front), static_cast<int>(CullMode::Back));
}

// Test 17: Fill mode presets
TEST_F(DX12PipelineStateTest, FillModePresets) {
    // Test that fill modes are distinct
    EXPECT_NE(static_cast<int>(FillMode::Solid), static_cast<int>(FillMode::Wireframe));
}

#endif // _WIN32
