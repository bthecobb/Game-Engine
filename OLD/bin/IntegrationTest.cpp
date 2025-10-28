#include "Core/Coordinator.h"
#include "Physics/CudaPhysicsSystem.h"
#include "Rendering/CudaRenderingSystem.h"
#include "Rendering/RenderSystem.h"
#include "Rendering/LightingSystem.h"
#include <iostream>

// Main integration test entry point
int main()
{
    std::cout << "==================================================" << std::endl;
    std::cout << "    AAA Game Engine - Full System Integration Test" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Initialize Core Systems
    std::cout << "\n--- Initializing Core Systems ---\n" << std::endl;
    auto& coordinator = CudaGame::Core::Coordinator::GetInstance();
    coordinator.Initialize();
    std::cout << "Coordinator initialized." << std::endl;

    // 2. Register Components
    std::cout << "\n--- Registering Components ---\n" << std::endl;
    coordinator.RegisterComponent<CudaGame::Rendering::TransformComponent>();
    coordinator.RegisterComponent<CudaGame::Physics::RigidbodyComponent>();
    coordinator.RegisterComponent<CudaGame::Physics::ColliderComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::MeshComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::LightComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::ShadowCasterComponent>();
    std::cout << "All core components registered." << std::endl;

    // 3. Register and Initialize Systems
    std::cout << "\n--- Initializing Engine Systems ---\n" << std::endl;
    auto renderSystem = coordinator.RegisterSystem<CudaGame::Rendering::RenderSystem>();
    renderSystem->Initialize();
    std::cout << "RenderSystem initialized." << std::endl;

    auto lightingSystem = coordinator.RegisterSystem<CudaGame::Rendering::LightingSystem>();
    lightingSystem->Initialize();
    std::cout << "LightingSystem initialized." << std::endl;

    auto cudaPhysicsSystem = coordinator.RegisterSystem<CudaGame::Physics::CudaPhysicsSystem>();
    cudaPhysicsSystem->Initialize();
    std::cout << "CudaPhysicsSystem initialized." << std::endl;

    auto cudaRenderingSystem = coordinator.RegisterSystem<CudaGame::Rendering::CudaRenderingSystem>();
    cudaRenderingSystem->Initialize();
    std::cout << "CudaRenderingSystem initialized." << std::endl;

    // 4. Create Game World
    std::cout << "\n--- Creating Game World ---\n" << std::endl;
    
    // Create a directional light
    CudaGame::Core::Entity light = lightingSystem->CreateDirectionalLight({-0.5f, -1.0f, -0.5f}, {1.0f, 1.0f, 1.0f}, 1.0f);
    std::cout << "Directional light created (Entity " << light << ")." << std::endl;

    // Create a dynamic physics object
    CudaGame::Core::Entity physicsObject = coordinator.CreateEntity();
    CudaGame::Rendering::TransformComponent transform;
    transform.position = {0.0f, 10.0f, 0.0f};
    coordinator.AddComponent(physicsObject, transform);
    
    CudaGame::Physics::RigidbodyComponent rb;
    rb.setMass(10.0f);
    coordinator.AddComponent(physicsObject, rb);

    CudaGame::Physics::ColliderComponent collider;
    collider.type = CudaGame::Physics::ColliderType::SPHERE;
    collider.radius = 1.0f;
    coordinator.AddComponent(physicsObject, collider);

    cudaPhysicsSystem->RegisterEntity(physicsObject, rb, collider);
    std::cout << "Dynamic physics object created and registered (Entity " << physicsObject << ")." << std::endl;
    
    // Create a static floor object
    CudaGame::Core::Entity floor = coordinator.CreateEntity();
    CudaGame::Rendering::TransformComponent floorTransform;
    floorTransform.position = {0.0f, -2.0f, 0.0f};
    floorTransform.scale = {20.0f, 0.5f, 20.0f};
    coordinator.AddComponent(floor, floorTransform);
    
    CudaGame::Physics::RigidbodyComponent floorRb;
    floorRb.setMass(0.0f); // Kinematic
    coordinator.AddComponent(floor, floorRb);
    
    CudaGame::Physics::ColliderComponent floorCollider;
    floorCollider.type = CudaGame::Physics::ColliderType::BOX;
    coordinator.AddComponent(floor, floorCollider);

    cudaPhysicsSystem->RegisterEntity(floor, floorRb, floorCollider);
    std::cout << "Static floor object created and registered (Entity " << floor << ")." << std::endl;
    
    // 5. Run Simulation
    std::cout << "\n--- Running Simulation (1 second) ---\n" << std::endl;
    float deltaTime = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        coordinator.UpdateSystems(deltaTime);
    }
    std::cout << "Simulation complete." << std::endl;

    // 6. Verify Results
    std::cout << "\n--- Verifying Results ---\n" << std::endl;
    
    // After 1 second of gravity, the object should have fallen
    const auto& finalTransform = coordinator.GetComponent<CudaGame::Rendering::TransformComponent>(physicsObject);
    std::cout << "Final position of dynamic object: (" 
              << finalTransform.position.x << ", " 
              << finalTransform.position.y << ", " 
              << finalTransform.position.z << ")" << std::endl;
    
    if (finalTransform.position.y < 9.0f) {
        std::cout << "[SUCCESS] Object fell due to gravity." << std::endl;
    } else {
        std::cout << "[FAILURE] Object did not fall as expected." << std::endl;
    }
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "  Integration Test Finished" << std::endl;
    std::cout << "======================================" << std::endl;

    return 0;
}
