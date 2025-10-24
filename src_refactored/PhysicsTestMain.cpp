#include <iostream>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include "Core/Coordinator.h"
#include "Gameplay/PlayerComponents.h"
#include "Physics/PhysicsComponents.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Gameplay/PlayerMovementSystem.h"
#include "Rendering/RenderComponents.h"

int main() {
    std::cout << "=== PhysX Integration Test ===" << std::endl;
    
    // Initialize GLFW for input
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Create a simple window
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hidden window for testing
    GLFWwindow* window = glfwCreateWindow(640, 480, "Physics Test", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    // Initialize coordinator
    auto& coordinator = Core::Coordinator::GetInstance();
    coordinator.Initialize();
    
    // Register components
    coordinator.RegisterComponent<Gameplay::PlayerMovementComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerInputComponent>();
    coordinator.RegisterComponent<Physics::RigidbodyComponent>();
    coordinator.RegisterComponent<Physics::ColliderComponent>();
    coordinator.RegisterComponent<Rendering::TransformComponent>();
    
    // Register systems
    auto physicsSystem = coordinator.RegisterSystem<Physics::PhysXPhysicsSystem>();
    auto playerMovementSystem = coordinator.RegisterSystem<Gameplay::PlayerMovementSystem>();
    
    // Set system signatures
    Core::Signature physicsSignature;
    physicsSignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    physicsSignature.set(coordinator.GetComponentType<Physics::ColliderComponent>());
    physicsSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Physics::PhysXPhysicsSystem>(physicsSignature);
    
    Core::Signature playerSignature;
    playerSignature.set(coordinator.GetComponentType<Gameplay::PlayerMovementComponent>());
    playerSignature.set(coordinator.GetComponentType<Gameplay::PlayerInputComponent>());
    playerSignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    playerSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Gameplay::PlayerMovementSystem>(playerSignature);
    
    // Initialize systems
    if (!physicsSystem->Initialize()) {
        std::cerr << "Failed to initialize PhysX system" << std::endl;
        return -1;
    }
    playerMovementSystem->Initialize();
    
    // Create ground entity
    std::cout << "\nCreating ground entity..." << std::endl;
    auto ground = coordinator.CreateEntity();
    
    coordinator.AddComponent(ground, Rendering::TransformComponent{
        glm::vec3(0.0f, -1.0f, 0.0f),  // Position
        glm::vec3(0.0f),                // Rotation
        glm::vec3(100.0f, 1.0f, 100.0f) // Scale
    });
    
    Physics::RigidbodyComponent groundRB;
    groundRB.mass = 0.0f; // Static
    groundRB.isKinematic = true;
    coordinator.AddComponent(ground, groundRB);
    
    Physics::ColliderComponent groundCollider;
    groundCollider.shape = Physics::ColliderShape::BOX;
    groundCollider.halfExtents = glm::vec3(50.0f, 1.0f, 50.0f);
    coordinator.AddComponent(ground, groundCollider);
    
    // Create player entity
    std::cout << "Creating player entity..." << std::endl;
    auto player = coordinator.CreateEntity();
    
    coordinator.AddComponent(player, Rendering::TransformComponent{
        glm::vec3(0.0f, 5.0f, 0.0f),   // Start 5 units above ground
        glm::vec3(0.0f),                // Rotation
        glm::vec3(1.0f, 2.0f, 1.0f)     // Scale
    });
    
    Physics::RigidbodyComponent playerRB;
    playerRB.mass = 80.0f;
    playerRB.isKinematic = false;
    playerRB.useGravity = true;
    coordinator.AddComponent(player, playerRB);
    
    Physics::ColliderComponent playerCollider;
    playerCollider.shape = Physics::ColliderShape::BOX;
    playerCollider.halfExtents = glm::vec3(0.5f, 1.0f, 0.5f);
    coordinator.AddComponent(player, playerCollider);
    
    Gameplay::PlayerMovementComponent playerMovement;
    playerMovement.baseSpeed = 10.0f;
    playerMovement.jumpForce = 15.0f;
    coordinator.AddComponent(player, playerMovement);
    
    Gameplay::PlayerInputComponent playerInput;
    coordinator.AddComponent(player, playerInput);
    
    // Run simulation for 5 seconds
    std::cout << "\n=== Starting Physics Simulation ===" << std::endl;
    std::cout << "Player starting at Y = 5.0" << std::endl;
    std::cout << "Ground at Y = -1.0 (top at Y = 0.0)" << std::endl;
    std::cout << "Running for 5 seconds...\n" << std::endl;
    
    const float FIXED_TIMESTEP = 1.0f / 60.0f;
    float totalTime = 0.0f;
    int frameCount = 0;
    
    while (totalTime < 5.0f) {
        // Update systems
        playerMovementSystem->Update(FIXED_TIMESTEP);
        physicsSystem->Update(FIXED_TIMESTEP);
        
        // Get player position
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(player);
        auto& movement = coordinator.GetComponent<Gameplay::PlayerMovementComponent>(player);
        
        // Print status every 10 frames (6 times per second)
        if (frameCount % 10 == 0) {
            std::cout << "Time: " << totalTime << "s | "
                      << "Player Y: " << transform.position.y << " | "
                      << "Grounded: " << (movement.isGrounded ? "YES" : "NO") << " | "
                      << "Ground Distance: " << movement.groundDistance << std::endl;
        }
        
        // Check if player has fallen below ground
        if (transform.position.y < -10.0f) {
            std::cout << "\n!!! ERROR: Player fell through ground! Y = " << transform.position.y << std::endl;
            break;
        }
        
        // Check if player is properly grounded
        if (movement.isGrounded && frameCount > 60) { // After 1 second
            std::cout << "\n✓ SUCCESS: Player is grounded at Y = " << transform.position.y << std::endl;
            std::cout << "Ground distance: " << movement.groundDistance << std::endl;
            break;
        }
        
        totalTime += FIXED_TIMESTEP;
        frameCount++;
        
        // Process window events to keep it responsive
        glfwPollEvents();
    }
    
    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    physicsSystem->Shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "Test complete!" << std::endl;
    return 0;
}
