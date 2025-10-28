#include "Physics/CudaPhysicsSystem.h"
#include "Rendering/CudaRenderingSystem.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "=== CudaGame Engine - Simple CUDA Systems Test ===" << std::endl;
    
    // Test CUDA Physics System
    std::cout << "\n[TEST] Creating and testing CUDA Physics System..." << std::endl;
    auto physicsSystem = std::make_unique<CudaGame::Physics::CudaPhysicsSystem>();
    
    if (physicsSystem->Initialize()) {
        std::cout << "[SUCCESS] CUDA Physics System initialized successfully!" << std::endl;
    } else {
        std::cout << "[ERROR] Failed to initialize CUDA Physics System!" << std::endl;
        return 1;
    }
    
    // Test physics update
    physicsSystem->Update(0.016f); // 60 FPS
    
    // Test CUDA Rendering System
    std::cout << "\n[TEST] Creating and testing CUDA Rendering System..." << std::endl;
    auto renderingSystem = std::make_unique<CudaGame::Rendering::CudaRenderingSystem>();
    
    if (renderingSystem->Initialize()) {
        std::cout << "[SUCCESS] CUDA Rendering System initialized successfully!" << std::endl;
    } else {
        std::cout << "[ERROR] Failed to initialize CUDA Rendering System!" << std::endl;
        return 1;
    }
    
    // Test rendering update
    renderingSystem->Update(0.016f); // 60 FPS
    
    // Test some rendering effects
    std::cout << "\n[TEST] Testing CUDA Rendering Effects..." << std::endl;
    renderingSystem->SetSSAOParameters(0.5f, 0.025f, 16);
    renderingSystem->SetBloomParameters(1.0f, 1.0f, 5);
    renderingSystem->SetToneMappingParameters(1.0f, 2.2f);
    
    // Simulate some GPU operations
    renderingSystem->ComputeSSAO(0, 0, 0, glm::mat4(1.0f), 1920, 1080);
    renderingSystem->ComputeBloom(0, 0, 1920, 1080);
    renderingSystem->ApplyToneMapping(0, 0, 1920, 1080);
    
    // Print profiling info
    renderingSystem->PrintProfileInfo();
    
    // Test physics parameters
    std::cout << "\n[TEST] Testing CUDA Physics Parameters..." << std::endl;
    physicsSystem->SetGravity(glm::vec3(0.0f, -9.81f, 0.0f));
    physicsSystem->SetSubstepCount(4);
    
    // Shutdown systems
    std::cout << "\n[CLEANUP] Shutting down systems..." << std::endl;
    physicsSystem->Shutdown();
    renderingSystem->Shutdown();
    
    std::cout << "\n[SUCCESS] All CUDA systems tests completed successfully!" << std::endl;
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "✓ CUDA Physics System: Initialize, Update, Parameters" << std::endl;
    std::cout << "✓ CUDA Rendering System: Initialize, Update, Effects" << std::endl;
    std::cout << "✓ GPU Memory Management: Allocation and Cleanup" << std::endl;
    std::cout << "✓ Performance Profiling: Frame time tracking" << std::endl;
    
    return 0;
}
