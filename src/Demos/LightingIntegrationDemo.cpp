#include "Core/Coordinator.h"
#include "Rendering/RenderSystem.h"
#include "Rendering/LightingSystem.h"
#include <iostream>

// Game entry point
int main()
{
    // Initialize the ECS coordinator
    auto coordinator = std::make_shared<CudaGame::Core::Coordinator>();
    coordinator->Initialize();

    // Register components
    coordinator->RegisterComponent<CudaGame::Rendering::MeshComponent>();
    coordinator->RegisterComponent<CudaGame::Rendering::TransformComponent>();
    coordinator->RegisterComponent<CudaGame::Rendering::MaterialComponent>();
    coordinator->RegisterComponent<CudaGame::Rendering::LightComponent>();
    coordinator->RegisterComponent<CudaGame::Rendering::ShadowCasterComponent>();

    // Register systems
    auto renderSystem = coordinator->RegisterSystem<CudaGame::Rendering::RenderSystem>();
    auto lightingSystem = coordinator->RegisterSystem<CudaGame::Rendering::LightingSystem>();

    // Initialize systems
    renderSystem->Initialize();
    lightingSystem->Initialize();

    // Set dependencies
    renderSystem->SetLightingSystem(lightingSystem);

    // Create a directional light
    CudaGame::Core::Entity directionalLight = lightingSystem->CreateDirectionalLight(
        glm::vec3(-0.2f, -1.0f, -0.3f), 
        glm::vec3(1.0f, 1.0f, 1.0f), 
        1.0f
    );

    // Create a point light
    CudaGame::Core::Entity pointLight = lightingSystem->CreatePointLight(
        glm::vec3(0.7f, 0.2f, 2.0f), 
        10.0f,
        glm::vec3(1.0f, 0.0f, 0.0f), 
        5.0f
    );
    
    // Create a mesh entity to cast shadows
    CudaGame::Core::Entity shadowCaster = coordinator->CreateEntity();
    coordinator->AddComponent(shadowCaster, CudaGame::Rendering::MeshComponent{"path/to/your/model.obj", 0, 0, true, true});
    coordinator->AddComponent(shadowCaster, CudaGame::Rendering::TransformComponent{});
    coordinator->AddComponent(shadowCaster, CudaGame::Rendering::ShadowCasterComponent{});
    
    std::cout << "Dynamic lighting and shadow systems initialized." << std::endl;
    std::cout << "To see results, integrate with the main game loop and render a scene." << std::endl;

    return 0;
}
