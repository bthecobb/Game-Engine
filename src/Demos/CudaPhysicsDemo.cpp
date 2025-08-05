#include "Core/Coordinator.h"
#include "Physics/CudaPhysicsSystem.h"
#include "Rendering/RenderSystem.h"
#include <iostream>
#include <random>

// Game entry point
int main()
{
    // Initialize the ECS coordinator
    auto coordinator = std::make_shared<CudaGame::Core::Coordinator>();
    coordinator->Initialize();

    // Register components
    coordinator->RegisterComponent<CudaGame::Physics::RigidbodyComponent>();
    coordinator->RegisterComponent<CudaGame::Physics::ColliderComponent>();
    coordinator->RegisterComponent<CudaGame::Rendering::TransformComponent>(); // For visualization

    // Register systems
    auto cudaPhysicsSystem = coordinator->RegisterSystem<CudaGame::Physics::CudaPhysicsSystem>();
    auto renderSystem = coordinator->RegisterSystem<CudaGame::Rendering::RenderSystem>(); // For visualization

    // Initialize systems
    cudaPhysicsSystem->Initialize();
    renderSystem->Initialize();

    // Scene setup
    const int numEntities = 20000; // 20,000 physics entities!
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::cout << "[Demo] Creating " << numEntities << " physics-enabled entities..." << std::endl;

    for (int i = 0; i < numEntities; ++i) {
        CudaGame::Core::Entity entity = coordinator->CreateEntity();

        // Create rigidbody and collider
        CudaGame::Physics::RigidbodyComponent rb;
        rb.setMass(1.0f);
        rb.restitution = 0.8f;
        rb.friction = 0.1f;

        CudaGame::Physics::ColliderComponent collider;
collider.shape = CudaGame::Physics::ColliderShape::SPHERE;
        collider.radius = 0.1f;

        // Add components to the entity
        coordinator->AddComponent(entity, rb);
        coordinator->AddComponent(entity, collider);

        // Add a transform for visualization
        CudaGame::Rendering::TransformComponent transform;
        transform.position = {dist(rng), dist(rng), dist(rng)};
        coordinator->AddComponent(entity, transform);

        // Register the entity with the CUDA physics system
        cudaPhysicsSystem->RegisterEntity(entity, rb, collider);
    }

    std::cout << "[Demo] All entities created and registered." << std::endl;

    // --- Main Game Loop (simulation) ---
    float deltaTime = 1.0f / 60.0f;
    int frameCount = 0;
    int maxFrames = 300; // Run for 5 seconds

    std::cout << "\n[Demo] Starting GPU physics simulation for " << maxFrames << " frames...\n" << std::endl;

    while (frameCount < maxFrames) {
        // Update systems in the correct order
        cudaPhysicsSystem->Update(deltaTime);
        // renderSystem->Update(deltaTime); // In a real game, this would update animations, etc.

        // renderSystem->Render(); // In a real game, this would render the scene

        if ((frameCount + 1) % 60 == 0) {
            std::cout << "[Frame " << frameCount + 1 << "] Simulated one second of GPU physics." << std::endl;
        }

        frameCount++;
    }

    std::cout << "\n[Demo] Simulation finished!" << std::endl;

    // --- Shutdown ---
    cudaPhysicsSystem->Shutdown();
    renderSystem->Shutdown();

    return 0;
}
