
#include <iostream>
#include <memory>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Core ECS
#include "Core/Coordinator.h"
#include "Core/System.h"

// Systems
#include "Animation/AnimationSystem.h"
#include "Animation/IKSystem.h"
#include "Combat/CombatSystem.h"
#include "Combat/ComboManager.h"
#include "GameFeel/GameFeelSystem.h"
#include "Input/InputSystem.h"
#include "Particles/ParticleSystem.h"
#include "Physics/PhysicsSystem.h"
#include "Physics/CudaPhysicsSystem.h"
#include "Physics/WallRunningSystem.h"
#include "Rendering/RenderSystem.h"
#include "Rendering/LightingSystem.h"
#include "Rhythm/RhythmSystem.h"
#include "Audio/AudioSystem.h"

// Components
#include "Physics/PhysicsComponents.h"
#include "Rendering/RenderComponents.h"
#include "Animation/AnimationSystem.h"
#include "Combat/CombatSystem.h"


// --- GLOBALS ---
GLFWwindow* g_window = nullptr;
CudaGame::Core::Coordinator g_coordinator;
std::shared_ptr<CudaGame::Rendering::RenderSystem> g_renderSystem;
std::shared_ptr<CudaGame::Rendering::Camera> g_camera;
bool g_keys[1024];

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS)
            g_keys[key] = true;
        else if (action == GLFW_RELEASE)
            g_keys[key] = false;
    }
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void init_window() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    g_window = glfwCreateWindow(1280, 720, "Integrated Demo", NULL, NULL);
    glfwMakeContextCurrent(g_window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glViewport(0, 0, 1280, 720);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glfwSetKeyCallback(g_window, key_callback);
}

void register_components_and_systems() {
    // Register Components
    g_coordinator.RegisterComponent<CudaGame::Physics::RigidbodyComponent>();
    g_coordinator.RegisterComponent<CudaGame::Physics::ColliderComponent>();
    g_coordinator.RegisterComponent<CudaGame::Rendering::TransformComponent>();
    g_coordinator.RegisterComponent<CudaGame::Rendering::MeshComponent>();
    g_coordinator.RegisterComponent<CudaGame::Rendering::MaterialComponent>();
    g_coordinator.RegisterComponent<CudaGame::Combat::CombatComponent>();
    g_coordinator.RegisterComponent<CudaGame::Animation::AnimationComponent>();

    // Register Systems
    g_renderSystem = g_coordinator.RegisterSystem<CudaGame::Rendering::RenderSystem>();
    auto physicsSystem = g_coordinator.RegisterSystem<CudaGame::Physics::PhysicsSystem>();
    auto combatSystem = g_coordinator.RegisterSystem<CudaGame::Combat::CombatSystem>();
    auto animationSystem = g_coordinator.RegisterSystem<CudaGame::Animation::AnimationSystem>();
    
    // Set System Signatures (what components an entity needs to be processed by a system)
    CudaGame::Core::Signature renderSignature;
    renderSignature.set(g_coordinator.GetComponentType<CudaGame::Rendering::TransformComponent>());
    renderSignature.set(g_coordinator.GetComponentType<CudaGame::Rendering::MeshComponent>());
    renderSignature.set(g_coordinator.GetComponentType<CudaGame::Rendering::MaterialComponent>());
    g_coordinator.SetSystemSignature<CudaGame::Rendering::RenderSystem>(renderSignature);

    CudaGame::Core::Signature physicsSignature;
    physicsSignature.set(g_coordinator.GetComponentType<CudaGame::Rendering::TransformComponent>());
    physicsSignature.set(g_coordinator.GetComponentType<CudaGame::Physics::RigidbodyComponent>());
    physicsSignature.set(g_coordinator.GetComponentType<CudaGame::Physics::ColliderComponent>());
    g_coordinator.SetSystemSignature<CudaGame::Physics::PhysicsSystem>(physicsSignature);

    // ... signatures for other systems
    
    // Initialize systems
    g_renderSystem->Initialize();
    physicsSystem->Initialize();
    combatSystem->Initialize();
    animationSystem->Initialize();
}

void create_entities() {
    // Create a player entity
    CudaGame::Core::Entity player = g_coordinator.CreateEntity();

    CudaGame::Rendering::TransformComponent playerTransform;
    playerTransform.position = glm::vec3(0.0f, 0.0f, 0.0f);
    playerTransform.scale = glm::vec3(1.0f, 1.0f, 1.0f);
    
    CudaGame::Rendering::MeshComponent playerMesh;
    CudaGame::Rendering::MaterialComponent playerMaterial;
    playerMaterial.albedo = glm::vec3(1.0f, 0.0f, 0.0f); // Red

    g_coordinator.AddComponent(player, playerTransform);
    g_coordinator.AddComponent(player, playerMesh);
    g_coordinator.AddComponent(player, playerMaterial);
    g_coordinator.AddComponent(player, CudaGame::Physics::RigidbodyComponent{});
    g_coordinator.AddComponent(player, CudaGame::Physics::ColliderComponent{CudaGame::Physics::ColliderShape::CAPSULE});
    g_coordinator.AddComponent(player, CudaGame::Combat::CombatComponent{});
    g_coordinator.AddComponent(player, CudaGame::Animation::AnimationComponent{});

    // Create an enemy entity
    CudaGame::Core::Entity enemy = g_coordinator.CreateEntity();
    
    CudaGame::Rendering::TransformComponent enemyTransform;
    enemyTransform.position = glm::vec3(3.0f, 0.0f, 0.0f);
    enemyTransform.scale = glm::vec3(1.0f, 1.0f, 1.0f);
    
    CudaGame::Rendering::MeshComponent enemyMesh;
    CudaGame::Rendering::MaterialComponent enemyMaterial;
    enemyMaterial.albedo = glm::vec3(0.0f, 0.0f, 1.0f); // Blue
    
    g_coordinator.AddComponent(enemy, enemyTransform);
    g_coordinator.AddComponent(enemy, enemyMesh);
    g_coordinator.AddComponent(enemy, enemyMaterial);
    g_coordinator.AddComponent(enemy, CudaGame::Physics::RigidbodyComponent{});
    g_coordinator.AddComponent(enemy, CudaGame::Physics::ColliderComponent{CudaGame::Physics::ColliderShape::CAPSULE});
    g_coordinator.AddComponent(enemy, CudaGame::Combat::CombatComponent{});
    g_coordinator.AddComponent(enemy, CudaGame::Animation::AnimationComponent{});
}


int main() {
    init_window();
    register_components_and_systems();
    create_entities();

    g_camera = std::make_shared<CudaGame::Rendering::Camera>();
    g_camera->SetPerspective(45.0f, 1280.0f/720.0f, 0.1f, 100.0f);
    g_camera->SetPosition({0.0f, 2.0f, 10.0f});
    g_renderSystem->SetMainCamera(g_camera.get());

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    while (!glfwWindowShouldClose(g_window)) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // --- Input ---
        if (g_keys[GLFW_KEY_W]) {
            // Move player forward
        }
        if (g_keys[GLFW_KEY_SPACE]) {
            // Player attack
        }

        // --- Update ---
        g_coordinator.UpdateSystems(deltaTime);
        
        // --- Render ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        g_renderSystem->Render(); // This will render all entities with Transform and Mesh
        
        glfwSwapBuffers(g_window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

