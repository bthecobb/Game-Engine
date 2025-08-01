#include "Core/Coordinator.h"
#include "Physics/CudaPhysicsSystem.h"
#include "Rendering/CudaRenderingSystem.h"
#include "Rendering/RenderSystem.h"
#include "Rendering/LightingSystem.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

// Interactive demo with real physics simulation
int main()
{
    std::cout << "==================================================\n";
    std::cout << "    AAA Game Engine - Interactive Physics Demo\n";
    std::cout << "==================================================\n\n";

    try {
        // Initialize Core Systems
        std::cout << "Initializing engine systems...\n";
        auto& coordinator = CudaGame::Core::Coordinator::GetInstance();
        coordinator.Initialize();

        // Register Components
        coordinator.RegisterComponent<CudaGame::Rendering::TransformComponent>();
        coordinator.RegisterComponent<CudaGame::Physics::RigidbodyComponent>();
        coordinator.RegisterComponent<CudaGame::Physics::ColliderComponent>();
        coordinator.RegisterComponent<CudaGame::Rendering::MeshComponent>();
        coordinator.RegisterComponent<CudaGame::Rendering::LightComponent>();

        // Initialize Systems
        auto renderSystem = coordinator.RegisterSystem<CudaGame::Rendering::RenderSystem>();
        renderSystem->Initialize();
        
        auto lightingSystem = coordinator.RegisterSystem<CudaGame::Rendering::LightingSystem>();
        lightingSystem->Initialize();
        
        auto cudaPhysicsSystem = coordinator.RegisterSystem<CudaGame::Physics::CudaPhysicsSystem>();
        cudaPhysicsSystem->Initialize();

        std::cout << "Engine initialized successfully!\n\n";

        // Create multiple physics objects at different heights
        std::vector<CudaGame::Core::Entity> objects;
        std::vector<float> startHeights = {20.0f, 15.0f, 10.0f, 5.0f};
        std::vector<float> masses = {1.0f, 2.0f, 0.5f, 3.0f};

        std::cout << "Creating physics objects:\n";
        for (int i = 0; i < 4; ++i) {
            CudaGame::Core::Entity obj = coordinator.CreateEntity();
            
            CudaGame::Rendering::TransformComponent transform;
            transform.position = {(float)i * 2.0f - 3.0f, startHeights[i], 0.0f}; // Spread objects horizontally
            coordinator.AddComponent(obj, transform);
            
            CudaGame::Physics::RigidbodyComponent rb;
            rb.setMass(masses[i]);
            coordinator.AddComponent(obj, rb);

            CudaGame::Physics::ColliderComponent collider;
            collider.type = CudaGame::Physics::ColliderType::SPHERE;
            collider.radius = 0.5f;
            coordinator.AddComponent(obj, collider);

            cudaPhysicsSystem->RegisterEntity(obj, rb, collider);
            objects.push_back(obj);
            
            std::cout << "  Object " << (i+1) << ": Mass=" << masses[i] << "kg, Height=" << startHeights[i] << "m\n";
        }

        // Create ground
        CudaGame::Core::Entity ground = coordinator.CreateEntity();
        CudaGame::Rendering::TransformComponent groundTransform;
        groundTransform.position = {0.0f, 0.0f, 0.0f};
        groundTransform.scale = {10.0f, 0.1f, 10.0f};
        coordinator.AddComponent(ground, groundTransform);

        std::cout << "\nStarting real-time physics simulation...\n";
        std::cout << "Press Ctrl+C to stop\n\n";

        // Real-time simulation with visual feedback
        float totalTime = 0.0f;
        float deltaTime = 1.0f / 60.0f; // 60 FPS
        int frameCount = 0;

        while (totalTime < 10.0f) { // Run for 10 seconds
            // Update physics
            coordinator.UpdateSystems(deltaTime);
            
            // Every 10 frames (roughly 6 times per second), show positions
            if (frameCount % 10 == 0) {
                std::cout << "\r"; // Clear line
                std::cout << "Time: " << std::fixed << std::setprecision(1) << totalTime << "s | ";
                
                for (int i = 0; i < objects.size(); ++i) {
                    const auto& transform = coordinator.GetComponent<CudaGame::Rendering::TransformComponent>(objects[i]);
                    std::cout << "Obj" << (i+1) << ": " << std::setprecision(1) << transform.position.y << "m ";
                }
                std::cout << std::flush;
            }

            totalTime += deltaTime;
            frameCount++;
            
            // Sleep to maintain real-time speed
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }

        std::cout << "\n\nFinal Results:\n";
        std::cout << "==============\n";
        for (int i = 0; i < objects.size(); ++i) {
            const auto& transform = coordinator.GetComponent<CudaGame::Rendering::TransformComponent>(objects[i]);
            const auto& rb = coordinator.GetComponent<CudaGame::Physics::RigidbodyComponent>(objects[i]);
            
            std::cout << "Object " << (i+1) << " (Mass: " << rb.getMass() << "kg):\n";
            std::cout << "  Started at: " << startHeights[i] << "m\n";
            std::cout << "  Final position: (" << std::setprecision(2) 
                      << transform.position.x << ", " 
                      << transform.position.y << ", " 
                      << transform.position.z << ")\n";
            std::cout << "  Distance fallen: " << (startHeights[i] - transform.position.y) << "m\n\n";
        }

        std::cout << "Demo completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
