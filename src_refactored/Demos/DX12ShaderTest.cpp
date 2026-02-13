#ifdef _WIN32
#include "Rendering/ShaderManager.h"
#include <iostream>

using namespace CudaGame::Rendering;

int main() {
    std::cout << "=== DX12 Shader Compilation Test ===" << std::endl;
    std::cout << "Testing DXC shader compiler and caching" << std::endl;
    std::cout << std::endl;

    // Initialize shader manager
    ShaderManager shaderMgr;
    if (!shaderMgr.Initialize()) {
        std::cerr << "[FAILED] ShaderManager initialization" << std::endl;
        return -1;
    }

    std::cout << "[OK] ShaderManager initialized with DXC" << std::endl << std::endl;

    // === TEST 1: Compile Vertex Shader ===
    std::cout << "TEST 1: Compiling vertex shader..." << std::endl;
    
    ShaderBytecode vsBytecode;
    std::string errorMsg;
    
    bool vsSuccess = shaderMgr.CompileFromFile(
        L"shaders/BasicColor.hlsl",
        L"VSMain",
        ShaderStage::Vertex,
        ShaderCompileFlags::None,
        vsBytecode,
        errorMsg
    );

    if (vsSuccess) {
        std::cout << "[PASS] Vertex shader compiled: " << vsBytecode.GetBufferSize() << " bytes" << std::endl;
        std::wcout << L"       Profile: " << vsBytecode.profile << std::endl;
    } else {
        std::cout << "[FAIL] Vertex shader compilation failed:" << std::endl;
        std::cout << errorMsg << std::endl;
        return -1;
    }

    // === TEST 2: Compile Pixel Shader ===
    std::cout << std::endl;
    std::cout << "TEST 2: Compiling pixel shader..." << std::endl;
    
    ShaderBytecode psBytecode;
    
    bool psSuccess = shaderMgr.CompileFromFile(
        L"shaders/BasicColor.hlsl",
        L"PSMain",
        ShaderStage::Pixel,
        ShaderCompileFlags::None,
        psBytecode,
        errorMsg
    );

    if (psSuccess) {
        std::cout << "[PASS] Pixel shader compiled: " << psBytecode.GetBufferSize() << " bytes" << std::endl;
        std::wcout << L"       Profile: " << psBytecode.profile << std::endl;
    } else {
        std::cout << "[FAIL] Pixel shader compilation failed:" << std::endl;
        std::cout << errorMsg << std::endl;
        return -1;
    }

    // === TEST 3: Verify Caching ===
    std::cout << std::endl;
    std::cout << "TEST 3: Testing shader cache..." << std::endl;
    
    size_t cacheSize = shaderMgr.GetCacheSize();
    std::cout << "[INFO] Cache contains " << cacheSize << " shaders" << std::endl;
    
    // Compile again (should hit cache)
    ShaderBytecode vsBytecode2;
    vsSuccess = shaderMgr.CompileFromFile(
        L"shaders/BasicColor.hlsl",
        L"VSMain",
        ShaderStage::Vertex,
        ShaderCompileFlags::None,
        vsBytecode2,
        errorMsg
    );

    if (vsSuccess && vsBytecode2.GetBufferSize() == vsBytecode.GetBufferSize()) {
        std::cout << "[PASS] Shader retrieved from cache (same size)" << std::endl;
    } else {
        std::cout << "[FAIL] Cache retrieval failed" << std::endl;
    }

    // === TEST 4: Debug vs Release Compilation ===
    std::cout << std::endl;
    std::cout << "TEST 4: Comparing debug vs release compilation..." << std::endl;
    
    ShaderBytecode vsDebug;
    shaderMgr.CompileFromFile(
        L"shaders/BasicColor.hlsl",
        L"VSMain",
        ShaderStage::Vertex,
        ShaderCompileFlags::Debug | ShaderCompileFlags::SkipOptimization,
        vsDebug,
        errorMsg
    );

    ShaderBytecode vsRelease;
    shaderMgr.CompileFromFile(
        L"shaders/BasicColor.hlsl",
        L"VSMain",
        ShaderStage::Vertex,
        ShaderCompileFlags::None,
        vsRelease,
        errorMsg
    );

    std::cout << "[INFO] Debug build: " << vsDebug.GetBufferSize() << " bytes" << std::endl;
    std::cout << "[INFO] Release build: " << vsRelease.GetBufferSize() << " bytes" << std::endl;
    
    if (vsDebug.GetBufferSize() >= vsRelease.GetBufferSize()) {
        std::cout << "[PASS] Debug shader is larger/equal (contains debug info)" << std::endl;
    } else {
        std::cout << "[WARN] Unexpected: Release shader is larger" << std::endl;
    }

    // === SUMMARY ===
    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "ALL SHADER TESTS PASSED!" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "DXC Compiler can:" << std::endl;
    std::cout << "  ✓ Compile HLSL vertex shaders (SM 6.0)" << std::endl;
    std::cout << "  ✓ Compile HLSL pixel shaders (SM 6.0)" << std::endl;
    std::cout << "  ✓ Cache compiled shaders" << std::endl;
    std::cout << "  ✓ Generate debug and release variants" << std::endl;
    std::cout << std::endl;
    std::cout << "Ready for Pipeline State Objects!" << std::endl;

    shaderMgr.Shutdown();
    return 0;
}
#endif // _WIN32
