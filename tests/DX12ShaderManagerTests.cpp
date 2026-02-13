#ifdef _WIN32
#include <gtest/gtest.h>
#include "Rendering/ShaderManager.h"
#include <fstream>
#include <filesystem>

using namespace CudaGame::Rendering;

class DX12ShaderManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test shader directory
        std::filesystem::create_directories("test_shaders");
        
        // Create a simple test shader
        CreateTestShader("test_shaders/SimpleVS.hlsl", 
            "float4 main(float3 pos : POSITION) : SV_POSITION { return float4(pos, 1.0); }");
        
        CreateTestShader("test_shaders/SimplePS.hlsl",
            "float4 main() : SV_TARGET { return float4(1, 0, 0, 1); }");
        
        // Initialize shader manager
        ASSERT_TRUE(m_shaderManager.Initialize()) << "Failed to initialize ShaderManager";
    }
    
    void TearDown() override {
        m_shaderManager.Shutdown();
        std::filesystem::remove_all("test_shaders");
    }
    
    void CreateTestShader(const std::string& path, const std::string& code) {
        std::ofstream file(path);
        file << code;
        file.close();
    }
    
    ShaderManager m_shaderManager;
};

// Test 1: Initialization
TEST_F(DX12ShaderManagerTest, Initialization) {
    ShaderManager manager;
    EXPECT_TRUE(manager.Initialize());
    manager.Shutdown();
}

// Test 2: Compile vertex shader from file
TEST_F(DX12ShaderManagerTest, CompileVertexShaderFromFile) {
    ShaderBytecode bytecode;
    std::string errorMsg;
    
    bool success = m_shaderManager.CompileFromFile(
        L"test_shaders/SimpleVS.hlsl",
        L"main",
        ShaderStage::Vertex,
        ShaderCompileFlags::None,
        bytecode,
        errorMsg
    );
    
    EXPECT_TRUE(success) << "Compilation failed: " << errorMsg;
    EXPECT_TRUE(bytecode.IsValid());
    EXPECT_GT(bytecode.GetBufferSize(), 0);
    EXPECT_EQ(bytecode.stage, ShaderStage::Vertex);
    EXPECT_EQ(bytecode.profile, L"vs_6_0");
}

// Test 3: Compile pixel shader from file
TEST_F(DX12ShaderManagerTest, CompilePixelShaderFromFile) {
    ShaderBytecode bytecode;
    std::string errorMsg;
    
    bool success = m_shaderManager.CompileFromFile(
        L"test_shaders/SimplePS.hlsl",
        L"main",
        ShaderStage::Pixel,
        ShaderCompileFlags::None,
        bytecode,
        errorMsg
    );
    
    EXPECT_TRUE(success) << "Compilation failed: " << errorMsg;
    EXPECT_TRUE(bytecode.IsValid());
    EXPECT_GT(bytecode.GetBufferSize(), 0);
    EXPECT_EQ(bytecode.stage, ShaderStage::Pixel);
    EXPECT_EQ(bytecode.profile, L"ps_6_0");
}

// Test 4: Shader caching
TEST_F(DX12ShaderManagerTest, ShaderCaching) {
    ShaderBytecode bytecode1, bytecode2;
    std::string errorMsg;
    
    // First compilation
    ASSERT_TRUE(m_shaderManager.CompileFromFile(
        L"test_shaders/SimpleVS.hlsl", L"main", ShaderStage::Vertex,
        ShaderCompileFlags::None, bytecode1, errorMsg
    ));
    
    size_t cacheSize1 = m_shaderManager.GetCacheSize();
    EXPECT_EQ(cacheSize1, 1) << "Cache should contain 1 shader";
    
    // Second compilation (should hit cache)
    ASSERT_TRUE(m_shaderManager.CompileFromFile(
        L"test_shaders/SimpleVS.hlsl", L"main", ShaderStage::Vertex,
        ShaderCompileFlags::None, bytecode2, errorMsg
    ));
    
    size_t cacheSize2 = m_shaderManager.GetCacheSize();
    EXPECT_EQ(cacheSize2, 1) << "Cache size should remain 1";
    EXPECT_EQ(bytecode1.GetBufferSize(), bytecode2.GetBufferSize()) << "Cached shader should match";
}

// Test 5: Debug vs Release compilation
TEST_F(DX12ShaderManagerTest, DebugVsReleaseCompilation) {
    ShaderBytecode debugBytecode, releaseBytecode;
    std::string errorMsg;
    
    // Compile with debug flags
    ASSERT_TRUE(m_shaderManager.CompileFromFile(
        L"test_shaders/SimpleVS.hlsl", L"main", ShaderStage::Vertex,
        ShaderCompileFlags::Debug | ShaderCompileFlags::SkipOptimization,
        debugBytecode, errorMsg
    ));
    
    // Compile with release flags
    ASSERT_TRUE(m_shaderManager.CompileFromFile(
        L"test_shaders/SimpleVS.hlsl", L"main", ShaderStage::Vertex,
        ShaderCompileFlags::None,
        releaseBytecode, errorMsg
    ));
    
    // Debug build should be larger or equal (contains debug info)
    EXPECT_GE(debugBytecode.GetBufferSize(), releaseBytecode.GetBufferSize())
        << "Debug shader should be >= release shader size";
}

// Test 6: Compile from source string
TEST_F(DX12ShaderManagerTest, CompileFromSourceString) {
    std::wstring source = L"float4 main(float3 pos : POSITION) : SV_POSITION { return float4(pos, 1.0); }";
    ShaderBytecode bytecode;
    std::string errorMsg;
    
    bool success = m_shaderManager.CompileFromSource(
        source, L"RuntimeShader", L"main",
        ShaderStage::Vertex, ShaderCompileFlags::None,
        bytecode, errorMsg
    );
    
    EXPECT_TRUE(success) << "Runtime compilation failed: " << errorMsg;
    EXPECT_TRUE(bytecode.IsValid());
}

// Test 7: Invalid shader compilation
TEST_F(DX12ShaderManagerTest, InvalidShaderCompilation) {
    CreateTestShader("test_shaders/Invalid.hlsl", "this is not valid HLSL code!");
    
    ShaderBytecode bytecode;
    std::string errorMsg;
    
    bool success = m_shaderManager.CompileFromFile(
        L"test_shaders/Invalid.hlsl", L"main", ShaderStage::Vertex,
        ShaderCompileFlags::None, bytecode, errorMsg
    );
    
    EXPECT_FALSE(success) << "Invalid shader should fail to compile";
    EXPECT_FALSE(errorMsg.empty()) << "Error message should be present";
    EXPECT_FALSE(bytecode.IsValid());
}

// Test 8: Cache key generation
TEST_F(DX12ShaderManagerTest, CacheKeyGeneration) {
    std::wstring key1 = ShaderManager::GenerateCacheKey(
        L"shader.hlsl", L"main", ShaderStage::Vertex, ShaderCompileFlags::None
    );
    
    std::wstring key2 = ShaderManager::GenerateCacheKey(
        L"shader.hlsl", L"main", ShaderStage::Vertex, ShaderCompileFlags::None
    );
    
    std::wstring key3 = ShaderManager::GenerateCacheKey(
        L"shader.hlsl", L"main", ShaderStage::Pixel, ShaderCompileFlags::None
    );
    
    EXPECT_EQ(key1, key2) << "Identical parameters should generate same key";
    EXPECT_NE(key1, key3) << "Different stage should generate different key";
}

// Test 9: Cache clearing
TEST_F(DX12ShaderManagerTest, CacheClear) {
    ShaderBytecode bytecode;
    std::string errorMsg;
    
    // Compile and cache a shader
    ASSERT_TRUE(m_shaderManager.CompileFromFile(
        L"test_shaders/SimpleVS.hlsl", L"main", ShaderStage::Vertex,
        ShaderCompileFlags::None, bytecode, errorMsg
    ));
    
    EXPECT_EQ(m_shaderManager.GetCacheSize(), 1);
    
    // Clear cache
    m_shaderManager.ClearCache();
    EXPECT_EQ(m_shaderManager.GetCacheSize(), 0) << "Cache should be empty after clear";
}

// Test 10: Multiple shader stages
TEST_F(DX12ShaderManagerTest, MultipleShaderStages) {
    ShaderBytecode vsBytecode, psBytecode;
    std::string errorMsg;
    
    // Compile vertex shader
    ASSERT_TRUE(m_shaderManager.CompileFromFile(
        L"test_shaders/SimpleVS.hlsl", L"main", ShaderStage::Vertex,
        ShaderCompileFlags::None, vsBytecode, errorMsg
    ));
    
    // Compile pixel shader
    ASSERT_TRUE(m_shaderManager.CompileFromFile(
        L"test_shaders/SimplePS.hlsl", L"main", ShaderStage::Pixel,
        ShaderCompileFlags::None, psBytecode, errorMsg
    ));
    
    EXPECT_EQ(m_shaderManager.GetCacheSize(), 2) << "Should cache both shaders";
    EXPECT_EQ(vsBytecode.stage, ShaderStage::Vertex);
    EXPECT_EQ(psBytecode.stage, ShaderStage::Pixel);
}

#endif // _WIN32
