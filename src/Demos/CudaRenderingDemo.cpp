#include "Core/Coordinator.h"
#include "Rendering/CudaRenderingSystem.h"
#include "Rendering/RenderSystem.h"
#include <iostream>

// Game entry point
int main()
{
    // Initialize the ECS coordinator
    auto coordinator = std::make_shared<CudaGame::Core::Coordinator>();
    coordinator->Initialize();

    // Register systems
    auto cudaRenderingSystem = coordinator->RegisterSystem<CudaGame::Rendering::CudaRenderingSystem>();
    auto renderSystem = coordinator->RegisterSystem<CudaGame::Rendering::RenderSystem>(); // For visualization

    // Initialize systems
    cudaRenderingSystem->Initialize();
    renderSystem->Initialize();

    // Scene setup (placeholders)
    const int width = 1280;
    const int height = 720;
    uint32_t sceneTexture = 1; // Placeholder for the rendered scene texture
    uint32_t finalTexture = 2; // Placeholder for the final output texture

    std::cout << "[Demo] Starting GPU-accelerated rendering effects demo...\n" << std::endl;

    // --- Post-processing pipeline ---
    std::cout << "--- Running post-processing pipeline on GPU ---" << std::endl;

    // 1. Compute SSAO
    std::cout << "1. Computing SSAO..." << std::endl;
    cudaRenderingSystem->SetSSAOParameters(0.5f, 0.025f, 16);
    cudaRenderingSystem->ComputeSSAO(0, 0, 0, glm::mat4(1.0f), width, height); // Using placeholders

    // 2. Compute Bloom
    std::cout << "2. Computing Bloom..." << std::endl;
    cudaRenderingSystem->SetBloomParameters(1.0f, 1.0f, 5);
    cudaRenderingSystem->ComputeBloom(sceneTexture, 0, width, height); // Using placeholders

    // 3. Apply Tone Mapping and Color Grading
    std::cout << "3. Applying Tone Mapping and Color Grading..." << std::endl;
    cudaRenderingSystem->SetToneMappingParameters(1.0f, 2.2f);
    cudaRenderingSystem->ApplyToneMapping(sceneTexture, finalTexture, width, height);

    // 4. Compute Contact Shadows
    std::cout << "4. Computing Contact Shadows..." << std::endl;
    cudaRenderingSystem->ComputeContactShadows(0, 0, 0, glm::mat4(1.0f), glm::mat4(1.0f), width, height);

    // 5. Compute Motion Blur
    std::cout << "5. Computing Motion Blur..." << std::endl;
    cudaRenderingSystem->SetMotionBlurParameters(1.0f, 16);
    cudaRenderingSystem->ComputeMotionBlur(sceneTexture, 0, finalTexture, width, height);

    // 6. Compute TAA
    std::cout << "6. Computing Temporal Anti-Aliasing (TAA)..." << std::endl;
    cudaRenderingSystem->ComputeTAA(sceneTexture, 0, 0, finalTexture, width, height);

    std::cout << "\n--- Post-processing pipeline finished! ---\n" << std::endl;

    // Print profiling info
    cudaRenderingSystem->PrintProfileInfo();

    // --- Shutdown ---
    cudaRenderingSystem->Shutdown();
    renderSystem->Shutdown();

    return 0;
}
