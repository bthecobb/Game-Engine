#include "Core/Coordinator.h"
#include "Gameplay/PlayerComponents.h"
#include "Gameplay/EnemyComponents.h"
#include "Gameplay/LevelComponents.h"
#include "Gameplay/PlayerMovementSystem.h"
#include "Gameplay/EnemyAISystem.h"
#include "Gameplay/LevelSystem.h"
#include "Gameplay/TargetingSystem.h"
#include "Animation/AnimationSystem.h"
#include "Particles/ParticleSystem.h"
#include "Physics/PhysicsSystem.h"
#include "Rendering/RenderSystem.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/Camera.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>

using namespace CudaGame;

// Window dimensions - Full HD for proper 3D rendering
const unsigned int WINDOW_WIDTH = 1920;
const unsigned int WINDOW_HEIGHT = 1080;

// GLFW window
GLFWwindow* window = nullptr;

bool InitializeWindow() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create window
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Enhanced CudaGame", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return false;
    }

    // Set viewport
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    std::cout << "OpenGL initialized successfully" << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    return true;
}

void CleanupWindow() {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

int main() {

    std::cout << "Starting Enhanced CudaGame with all features..." << std::endl;

    // Initialize window and OpenGL context first
    if (!InitializeWindow()) {
        std::cerr << "Failed to initialize window and OpenGL context" << std::endl;
        return -1;
    }

    // Get coordinator instance
    auto& coordinator = Core::Coordinator::GetInstance();
    coordinator.Initialize();
    
    // Register all components
    coordinator.RegisterComponent<Gameplay::PlayerMovementComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerCombatComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerInputComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerRhythmComponent>();
    coordinator.RegisterComponent<Gameplay::GrapplingHookComponent>();
    coordinator.RegisterComponent<Gameplay::EnemyAIComponent>();
    coordinator.RegisterComponent<Gameplay::EnemyCombatComponent>();
    coordinator.RegisterComponent<Gameplay::EnemyMovementComponent>();
    coordinator.RegisterComponent<Gameplay::TargetingComponent>();
    coordinator.RegisterComponent<Gameplay::DimensionalVisibilityComponent>();
    coordinator.RegisterComponent<Gameplay::WorldRotationComponent>();
    coordinator.RegisterComponent<Gameplay::PlatformComponent>();
    coordinator.RegisterComponent<Gameplay::WallComponent>();
    coordinator.RegisterComponent<Gameplay::CollectibleComponent>();
    coordinator.RegisterComponent<Gameplay::InteractableComponent>();
    coordinator.RegisterComponent<Physics::RigidbodyComponent>();
    coordinator.RegisterComponent<Physics::ColliderComponent>();
    coordinator.RegisterComponent<Rendering::TransformComponent>();
    coordinator.RegisterComponent<Rendering::MeshComponent>();
    coordinator.RegisterComponent<Rendering::MaterialComponent>();
    
    // Register and initialize systems
    auto playerMovementSystem = coordinator.RegisterSystem<Gameplay::PlayerMovementSystem>();
    auto enemyAISystem = coordinator.RegisterSystem<Gameplay::EnemyAISystem>();
    auto levelSystem = coordinator.RegisterSystem<Gameplay::LevelSystem>();
    auto targetingSystem = coordinator.RegisterSystem<Gameplay::TargetingSystem>();
    auto physicsSystem = coordinator.RegisterSystem<Physics::PhysicsSystem>();
    auto renderSystem = coordinator.RegisterSystem<Rendering::RenderSystem>();
    auto particleSystem = coordinator.RegisterSystem<Particles::ParticleSystem>();
    
    // Set system signatures
    Core::Signature playerMovementSignature;
    coordinator.SetSystemSignature<Gameplay::PlayerMovementSystem>(playerMovementSignature);

    Core::Signature enemyAISignature;
    coordinator.SetSystemSignature<Gameplay::EnemyAISystem>(enemyAISignature);

    Core::Signature levelSignature;
    coordinator.SetSystemSignature<Gameplay::LevelSystem>(levelSignature);

    Core::Signature targetingSignature;
    coordinator.SetSystemSignature<Gameplay::TargetingSystem>(targetingSignature);

    Core::Signature renderSignature;
    renderSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    renderSignature.set(coordinator.GetComponentType<Rendering::MeshComponent>());
    coordinator.SetSystemSignature<Rendering::RenderSystem>(renderSignature);

    Core::Signature physicsSignature;
    coordinator.SetSystemSignature<Physics::PhysicsSystem>(physicsSignature);

    Core::Signature particleSignature;
    coordinator.SetSystemSignature<Particles::ParticleSystem>(particleSignature);
    
    // Initialize all systems
    playerMovementSystem->Initialize();
    enemyAISystem->Initialize();
    levelSystem->Initialize();
    targetingSystem->Initialize();
    physicsSystem->Initialize();
    renderSystem->Initialize();
    particleSystem->Initialize();
    
    // Create and setup camera
    std::cout << "Creating camera..." << std::endl;
    auto camera = std::make_unique<Rendering::Camera>(Rendering::ProjectionType::PERSPECTIVE);
    camera->SetPosition(glm::vec3(0.0f, 15.0f, 30.0f));
    camera->LookAt(glm::vec3(0.0f, 0.0f, 0.0f));
    camera->SetPerspective(60.0f, (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 200.0f);
    camera->UpdateMatrices();
    
    // Set the camera in the render system
    renderSystem->SetMainCamera(camera.get());
    std::cout << "Camera set up successfully!" << std::endl;
    
    // Create the player entity
    Core::Entity player = coordinator.CreateEntity();
    
    // Add player components
    Gameplay::PlayerMovementComponent playerMovement;
    coordinator.AddComponent(player, playerMovement);
    
    Gameplay::PlayerCombatComponent playerCombat;
    playerCombat.inventory.push_back(Gameplay::WeaponType::NONE);
    playerCombat.inventory.push_back(Gameplay::WeaponType::SWORD);
    coordinator.AddComponent(player, playerCombat);
    
    Gameplay::PlayerInputComponent playerInput;
    coordinator.AddComponent(player, playerInput);
    
    Gameplay::PlayerRhythmComponent playerRhythm;
    coordinator.AddComponent(player, playerRhythm);
    
    Physics::RigidbodyComponent playerRigidbody;
    playerRigidbody.mass = 80.0f;
    coordinator.AddComponent(player, playerRigidbody);
    
    Physics::ColliderComponent playerCollider;
    playerCollider.shape = Physics::ColliderShape::BOX;
    playerCollider.size = glm::vec3(0.8f, 1.8f, 0.8f);
    coordinator.AddComponent(player, playerCollider);
    
    Rendering::TransformComponent playerTransform;
    playerTransform.position = glm::vec3(0.0f, 1.0f, 0.0f);
    playerTransform.scale = glm::vec3(0.8f, 1.8f, 0.8f);
    coordinator.AddComponent(player, playerTransform);
    
    Rendering::MeshComponent playerMesh;
    playerMesh.modelPath = "player_cube";
    coordinator.AddComponent(player, playerMesh);
    
    Rendering::MaterialComponent playerMaterial;
    playerMaterial.albedo = glm::vec3(0.0f, 0.0f, 1.0f); // Blue player
    coordinator.AddComponent(player, playerMaterial);
    
    std::cout << "Player created with all components!" << std::endl;
    
    // Create ground plane
    auto ground = coordinator.CreateEntity();
    coordinator.AddComponent(ground, Rendering::TransformComponent{
        glm::vec3(0.0f, -2.0f, 0.0f),
        glm::vec3(0.0f),
        glm::vec3(50.0f, 1.0f, 50.0f)
    });
    coordinator.AddComponent(ground, Rendering::MeshComponent{"player_cube"});
    coordinator.AddComponent(ground, Rendering::MaterialComponent{
        glm::vec3(0.3f, 0.3f, 0.3f), // Dark gray
        0.0f, 0.8f, 1.0f
    });
    
    // Create some buildings/pillars
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (i == 2 && j == 2) continue; // Skip center for player
            
            auto pillar = coordinator.CreateEntity();
            float x = (i - 2) * 10.0f;
            float z = (j - 2) * 10.0f;
            float height = 5.0f + (i + j) * 0.5f;
            
            coordinator.AddComponent(pillar, Rendering::TransformComponent{
                glm::vec3(x, height/2.0f - 1.0f, z),
                glm::vec3(0.0f),
                glm::vec3(2.0f, height, 2.0f)
            });
            coordinator.AddComponent(pillar, Rendering::MeshComponent{"player_cube"});
            coordinator.AddComponent(pillar, Rendering::MaterialComponent{
                glm::vec3(0.6f, 0.6f, 0.7f), // Light gray
                0.2f, 0.6f, 1.0f
            });
        }
    }
    
    // Create enemies
    for (int i = 0; i < 5; ++i) {
        auto enemy = coordinator.CreateEntity();
        
        float angle = i * 72.0f * 3.14159f / 180.0f;
        float x = cos(angle) * 15.0f;
        float z = sin(angle) * 15.0f;
        
        coordinator.AddComponent(enemy, Rendering::TransformComponent{
            glm::vec3(x, 1.0f, z),
            glm::vec3(0.0f),
            glm::vec3(1.2f, 2.0f, 1.2f)
        });
        coordinator.AddComponent(enemy, Rendering::MeshComponent{"player_cube"});
        coordinator.AddComponent(enemy, Rendering::MaterialComponent{
            glm::vec3(1.0f, 0.0f, 0.0f), // Red enemies
            0.0f, 0.5f, 1.0f
        });
    }
    
    std::cout << "3D environment created!" << std::endl;
    std::cout << "Wallrunning, targeting, and combat systems active!" << std::endl;

    // Main Game Loop
    std::cout << "Starting main game loop..." << std::endl;
    float deltaTime = 0.016f; // 60 FPS
    int frameCount = 0;
    
    while (!glfwWindowShouldClose(window)) {
        // Poll for and process events
        glfwPollEvents();
        
        // Update all systems
        playerMovementSystem->Update(deltaTime);
        enemyAISystem->Update(deltaTime);
        levelSystem->Update(deltaTime);
        targetingSystem->Update(deltaTime);
        physicsSystem->Update(deltaTime);
        renderSystem->Update(deltaTime);
    particleSystem->Update(deltaTime);

        // Swap front and back buffers
        glfwSwapBuffers(window);
        
        frameCount++;
        
        if (frameCount % 60 == 0) {
            std::cout << "Game running... Frame " << frameCount << "/1200" << std::endl;
        }
        
        // Exit on ESC key
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }
        
        // F4 to cycle debug modes
        static bool f4Pressed = false;
        if (glfwGetKey(window, GLFW_KEY_F4) == GLFW_PRESS) {
            if (!f4Pressed) {
                renderSystem->CycleDebugMode();
                f4Pressed = true;
            }
        } else {
            f4Pressed = false;
        }
    }

    std::cout << "Enhanced CudaGame demo completed!" << std::endl;
    std::cout << "All original features restored in ECS architecture!" << std::endl;
    
    // Cleanup
    CleanupWindow();
    
    return 0;
}
