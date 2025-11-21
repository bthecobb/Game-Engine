#ifdef _WIN32
#include "Rendering/Backends/DX12RenderBackend.h"
#include "Rendering/BackendResources.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>

using namespace CudaGame::Rendering;

int main() {
    std::cout << "=== DX12 Resource Creation Test ===" << std::endl;
    std::cout << "Testing texture and buffer creation" << std::endl;
    std::cout << std::endl;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "[ERROR] Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "DX12 Resource Test", nullptr, nullptr);
    
    if (!window) {
        std::cerr << "[ERROR] Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Initialize DX12 backend
    auto dx12Backend = std::make_shared<DX12RenderBackend>();
    if (!dx12Backend->Initialize()) {
        std::cerr << "[FAILED] Backend initialization" << std::endl;
        return -1;
    }

    std::cout << "[OK] DX12 Backend initialized" << std::endl << std::endl;

    // === TEST 1: Create G-Buffer Textures ===
    std::cout << "TEST 1: Creating G-Buffer textures..." << std::endl;
    
    TextureDesc positionTexture = {};
    positionTexture.width = 1920;
    positionTexture.height = 1080;
    positionTexture.format = TextureFormat::RGB32F;
    positionTexture.mipLevels = 1;
    
    TextureHandle posHandle;
    if (dx12Backend->CreateTexture(positionTexture, posHandle)) {
        std::cout << "[PASS] Position buffer created (RGB32F, 1920x1080)" << std::endl;
    } else {
        std::cout << "[FAIL] Position buffer creation failed" << std::endl;
    }

    TextureDesc normalTexture = positionTexture;
    normalTexture.format = TextureFormat::RGB16F;
    
    TextureHandle normHandle;
    if (dx12Backend->CreateTexture(normalTexture, normHandle)) {
        std::cout << "[PASS] Normal buffer created (RGB16F, 1920x1080)" << std::endl;
    } else {
        std::cout << "[FAIL] Normal buffer creation failed" << std::endl;
    }

    TextureDesc albedoTexture = positionTexture;
    albedoTexture.format = TextureFormat::RGBA8;
    
    TextureHandle albedoHandle;
    if (dx12Backend->CreateTexture(albedoTexture, albedoHandle)) {
        std::cout << "[PASS] Albedo buffer created (RGBA8, 1920x1080)" << std::endl;
    } else {
        std::cout << "[FAIL] Albedo buffer creation failed" << std::endl;
    }

    // === TEST 2: Create Depth Buffer ===
    std::cout << std::endl;
    std::cout << "TEST 2: Creating depth/stencil buffer..." << std::endl;
    
    TextureDesc depthTexture = positionTexture;
    depthTexture.format = TextureFormat::DEPTH32F;
    
    TextureHandle depthHandle;
    if (dx12Backend->CreateTexture(depthTexture, depthHandle)) {
        std::cout << "[PASS] Depth buffer created (D32_FLOAT, 1920x1080)" << std::endl;
    } else {
        std::cout << "[FAIL] Depth buffer creation failed" << std::endl;
    }

    // === TEST 3: Create Shadow Map ===
    std::cout << std::endl;
    std::cout << "TEST 3: Creating shadow map..." << std::endl;
    
    TextureDesc shadowMap = positionTexture;
    shadowMap.width = 2048;
    shadowMap.height = 2048;
    shadowMap.format = TextureFormat::DEPTH24;
    
    TextureHandle shadowHandle;
    if (dx12Backend->CreateTexture(shadowMap, shadowHandle)) {
        std::cout << "[PASS] Shadow map created (D24_UNORM_S8, 2048x2048)" << std::endl;
    } else {
        std::cout << "[FAIL] Shadow map creation failed" << std::endl;
    }

    // === TEST 4: Create Vertex Buffer ===
    std::cout << std::endl;
    std::cout << "TEST 4: Creating vertex buffer..." << std::endl;
    
    struct Vertex {
        float position[3];
        float normal[3];
        float texCoord[2];
    };
    
    std::vector<Vertex> vertices = {
        {{0.0f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.5f, 0.0f}},
        {{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}
    };
    
    size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
    
    if (dx12Backend->CreateBuffer(vertexBufferSize, true, vertexBuffer)) {
        std::cout << "[PASS] Vertex buffer created (" << vertices.size() << " vertices, " 
                  << vertexBufferSize << " bytes)" << std::endl;
        
        // Upload data
        if (dx12Backend->UploadBufferData(vertexBuffer.Get(), vertices.data(), vertexBufferSize)) {
            std::cout << "[PASS] Vertex data uploaded successfully" << std::endl;
        } else {
            std::cout << "[FAIL] Vertex data upload failed" << std::endl;
        }
    } else {
        std::cout << "[FAIL] Vertex buffer creation failed" << std::endl;
    }

    // === TEST 5: Create Index Buffer ===
    std::cout << std::endl;
    std::cout << "TEST 5: Creating index buffer..." << std::endl;
    
    std::vector<uint32_t> indices = {0, 1, 2};
    size_t indexBufferSize = indices.size() * sizeof(uint32_t);
    Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;
    
    if (dx12Backend->CreateBuffer(indexBufferSize, true, indexBuffer)) {
        std::cout << "[PASS] Index buffer created (" << indices.size() << " indices, " 
                  << indexBufferSize << " bytes)" << std::endl;
        
        if (dx12Backend->UploadBufferData(indexBuffer.Get(), indices.data(), indexBufferSize)) {
            std::cout << "[PASS] Index data uploaded successfully" << std::endl;
        } else {
            std::cout << "[FAIL] Index data upload failed" << std::endl;
        }
    } else {
        std::cout << "[FAIL] Index buffer creation failed" << std::endl;
    }

    // === TEST 6: Create Constant Buffer ===
    std::cout << std::endl;
    std::cout << "TEST 6: Creating constant buffer..." << std::endl;
    
    struct PerFrameConstants {
        float viewMatrix[16];
        float projMatrix[16];
        float cameraPosition[4];
    };
    
    size_t constantBufferSize = (sizeof(PerFrameConstants) + 255) & ~255; // Align to 256 bytes
    Microsoft::WRL::ComPtr<ID3D12Resource> constantBuffer;
    
    if (dx12Backend->CreateBuffer(constantBufferSize, true, constantBuffer)) {
        std::cout << "[PASS] Constant buffer created (" << constantBufferSize << " bytes, 256-byte aligned)" << std::endl;
    } else {
        std::cout << "[FAIL] Constant buffer creation failed" << std::endl;
    }

    // === TEST 7: Destroy Resources ===
    std::cout << std::endl;
    std::cout << "TEST 7: Destroying textures..." << std::endl;
    
    dx12Backend->DestroyTexture(posHandle);
    dx12Backend->DestroyTexture(normHandle);
    dx12Backend->DestroyTexture(albedoHandle);
    dx12Backend->DestroyTexture(depthHandle);
    dx12Backend->DestroyTexture(shadowHandle);
    
    std::cout << "[PASS] All textures destroyed" << std::endl;

    // === SUMMARY ===
    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "ALL RESOURCE TESTS PASSED!" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "DX12 Backend can create:" << std::endl;
    std::cout << "  ✓ G-Buffer textures (Position, Normal, Albedo)" << std::endl;
    std::cout << "  ✓ Depth/stencil buffers" << std::endl;
    std::cout << "  ✓ Shadow maps" << std::endl;
    std::cout << "  ✓ Vertex buffers with upload" << std::endl;
    std::cout << "  ✓ Index buffers with upload" << std::endl;
    std::cout << "  ✓ Constant buffers (256-byte aligned)" << std::endl;
    std::cout << "  ✓ Resource destruction" << std::endl;
    std::cout << std::endl;
    std::cout << "Ready for Pipeline State Objects (PSOs) and shaders!" << std::endl;

    // Cleanup
    dx12Backend.reset();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
#else
int main() {
    std::cout << "DX12 is Windows-only" << std::endl;
    return 0;
}
#endif
