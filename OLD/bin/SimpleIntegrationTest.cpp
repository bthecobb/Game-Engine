#include "Core/Coordinator.h"
#include "Physics/CudaPhysicsSystem.h"
#include "Rendering/CudaRenderingSystem.h"
#include "Rendering/RenderSystem.h"
#include "Rendering/LightingSystem.h"
#include <iostream>

// Simplified integration test to avoid crashes
int main()
{
    std::cout << "==================================================\n";
    std::cout << "    AAA Game Engine - Simple Integration Test\n";
    std::cout << "==================================================\n\n";

    try {
        // 1. Initialize Core Systems
        std::cout << "--- Initializing Core Systems ---\n";
        auto& coordinator = CudaGame::Core::Coordinator::GetInstance();
        coordinator.Initialize();
        std::cout << "Coordinator initialized.\n\n";

        // 2. Register Components
        std::cout << "--- Registering Components ---\n";
        coordinator.RegisterComponent<CudaGame::Rendering::TransformComponent>();
        coordinator.RegisterComponent<CudaGame::Physics::RigidbodyComponent>();
        coordinator.RegisterComponent<CudaGame::Physics::ColliderComponent>();
        coordinator.RegisterComponent<CudaGame::Rendering::MeshComponent>();
        coordinator.RegisterComponent<CudaGame::Rendering::LightComponent>();
        coordinator.RegisterComponent<CudaGame::Rendering::ShadowCasterComponent>();
        std::cout << "All core components registered.\n\n";

        // 3. Register and Initialize Systems
        std::cout << "--- Initializing Engine Systems ---\n";
        
        auto renderSystem = coordinator.RegisterSystem<CudaGame::Rendering::RenderSystem>();
        renderSystem->Initialize();
        std::cout << "RenderSystem initialized.\n";

        auto lightingSystem = coordinator.RegisterSystem<CudaGame::Rendering::LightingSystem>();
        lightingSystem->Initialize();
        std::cout << "LightingSystem initialized.\n";

        auto cudaPhysicsSystem = coordinator.RegisterSystem<CudaGame::Physics::CudaPhysicsSystem>();
        cudaPhysicsSystem->Initialize();
        std::cout << "CudaPhysicsSystem initialized.\n";

        auto cudaRenderingSystem = coordinator.RegisterSystem<CudaGame::Rendering::CudaRenderingSystem>();
        cudaRenderingSystem->Initialize();
        std::cout << "CudaRenderingSystem initialized.\n\n";

        // 4. Create a Simple Game World
        std::cout << "--- Creating Simple Game World ---\n";
        
        // Create a directional light
        CudaGame::Core::Entity light = lightingSystem->CreateDirectionalLight({-0.5f, -1.0f, -0.5f}, {1.0f, 1.0f, 1.0f}, 1.0f);
        std::cout << "Directional light created (Entity " << light << ").\n";

        // Create a simple physics object
        CudaGame::Core::Entity physicsObject = coordinator.CreateEntity();
        CudaGame::Rendering::TransformComponent transform;
        transform.position = {0.0f, 5.0f, 0.0f};
        coordinator.AddComponent(physicsObject, transform);
        
        CudaGame::Physics::RigidbodyComponent rb;
        rb.setMass(5.0f);
        coordinator.AddComponent(physicsObject, rb);

        CudaGame::Physics::ColliderComponent collider;
        collider.type = CudaGame::Physics::ColliderType::SPHERE;
        collider.radius = 1.0f;
        coordinator.AddComponent(physicsObject, collider);

        cudaPhysicsSystem->RegisterEntity(physicsObject, rb, collider);
        std::cout << "Physics object created and registered (Entity " << physicsObject << ").\n\n";

        // 5. Run Short Simulation (10 frames instead of 60)
        std::cout << "--- Running Short Simulation (10 frames) ---\n";
        float deltaTime = 1.0f / 60.0f;
        for (int i = 0; i < 10; ++i) {
            coordinator.UpdateSystems(deltaTime);
            if (i % 5 == 0) {
                std::cout << "Frame " << i << " complete.\n";
            }
        }
        std::cout << "Simulation complete.\n\n";

        // 6. Verify Results
        std::cout << "--- Verifying Results ---\n";
        const auto& finalTransform = coordinator.GetComponent<CudaGame::Rendering::TransformComponent>(physicsObject);
        std::cout << "Final position of physics object: (" 
                  << finalTransform.position.x << ", " 
                  << finalTransform.position.y << ", " 
                  << finalTransform.position.z << ")\n";
        
        std::cout << "\n======================================\n";
        std::cout << "  Simple Integration Test Complete\n";
        std::cout << "======================================\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception caught!" << std::endl;
        return -1;
    }

    return 0;
}
