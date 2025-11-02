#include "Core/Coordinator.h"
#include "Gameplay/PlayerComponents.h"
#include "Gameplay/EnemyComponents.h"
#include "Gameplay/LevelComponents.h"
#include "Gameplay/PlayerMovementSystem.h"
#include "Gameplay/CharacterControllerSystem.h"
#include "Gameplay/EnemyAISystem.h"
#include "Gameplay/LevelSystem.h"
#include "Gameplay/TargetingSystem.h"
#include "Animation/AnimationSystem.h"
#include "Particles/ParticleSystem.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Physics/WallRunningSystem.h"
#include "Rendering/RenderSystem.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/Camera.h"
#include "Rendering/OrbitCamera.h"
#include "Rendering/Debug.h"
#include "Debug/OpenGLDebugRenderer.h"
#include "Rendering/RenderDebugSystem.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <chrono>
#include <random>

using namespace CudaGame;

// Window dimensions - 1920x1080 for proper 3D rendering
const unsigned int WINDOW_WIDTH = 1920;
const unsigned int WINDOW_HEIGHT = 1080;

// GLFW window
GLFWwindow* window = nullptr;

// Camera controls
float lastX = WINDOW_WIDTH / 2.0f;
float lastY = WINDOW_HEIGHT / 2.0f;
bool firstMouse = true;
bool mouseCaptured = false;

// Game state
bool keys[1024] = {false};
bool keysPressed[1024] = {false}; // Track single key presses
bool mouseButtons[8] = {false};
Rendering::OrbitCamera* mainCamera = nullptr;

// Scroll callback for zoom
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (mainCamera) {
        mainCamera->ApplyZoom(static_cast<float>(yoffset));
    }
}

// Input callback functions
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) {
            keys[key] = true;
            keysPressed[key] = true; // Mark as just pressed
        }
        else if (action == GLFW_RELEASE) {
            keys[key] = false;
        }
    }
    
    // Toggle mouse capture with TAB
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        mouseCaptured = !mouseCaptured;
        if (mouseCaptured) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!mouseCaptured || !mainCamera) return;
    
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    // Use OrbitCamera's mouse delta method
    mainCamera->ApplyMouseDelta(xoffset, yoffset);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button >= 0 && button < 8) {
        if (action == GLFW_PRESS)
            mouseButtons[button] = true;
        else if (action == GLFW_RELEASE)
            mouseButtons[button] = false;
    }
}

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
    
    // Explicitly request depth and stencil buffers
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);

    // Create window
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CudaGame - Full 3D Experience", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Set callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return false;
    }

    // Set viewport
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    
    // Enable face culling for better performance
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    std::cout << "OpenGL initialized successfully" << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
    // PHASE 1: Fail-fast startup checks
#ifdef DEBUG_RENDERER
    {
        GLint major = 0, minor = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);
        std::cout << "{ \"startupGL\": { \"major\":" << major << ",\"minor\":" << minor << " } }" << std::endl;

        GLint depthBits = 0, stencilBits = 0;
        glGetIntegerv(GL_DEPTH_BITS, &depthBits);
        glGetIntegerv(GL_STENCIL_BITS, &stencilBits);
        std::cout << "{ \"defaultFBO\": { \"depthBits\":" << depthBits << ",\"stencilBits\":" << stencilBits << " } }" << std::endl;

        if (depthBits <= 0) {
            std::cerr << "[FATAL] Default framebuffer has no depth buffer. Cannot proceed." << std::endl;
            std::exit(1);
        }
        
        if (major < 3 || (major == 3 && minor < 3)) {
            std::cerr << "[FATAL] OpenGL 3.3+ required. Current version: " << major << "." << minor << std::endl;
            std::exit(1);
        }
        
        std::cout << "{ \"startupValidation\": \"PASSED\" }" << std::endl;
    }
#endif
    
    return true;
}

void CreateGameEnvironment(Core::Coordinator& coordinator) {
    std::cout << "Creating 3D game environment..." << std::endl;
    
    // Create ground plane (larger for 3D world)
    auto ground = coordinator.CreateEntity();
    coordinator.AddComponent(ground, Rendering::TransformComponent{
        glm::vec3(0.0f, -1.0f, 0.0f),
        glm::vec3(0.0f),
        glm::vec3(300.0f, 1.0f, 300.0f)
    });
    coordinator.AddComponent(ground, Rendering::MeshComponent{"player_cube"});
    coordinator.AddComponent(ground, Rendering::MaterialComponent{
        glm::vec3(0.3f, 0.3f, 0.3f), // Dark gray ground
        0.0f, 0.8f, 1.0f
    });
    
    // Add physics components for collision
    coordinator.AddComponent(ground, Physics::ColliderComponent{
        Physics::ColliderShape::BOX,
        glm::vec3(150.0f, 0.5f, 150.0f)  // Half extents matching the scale
    });
    // Ground is static so we don't need RigidbodyComponent (PhysX will create static actor)
    std::cout << "[DEBUG] Ground created at y=-1.0, scale.y=1.0, top surface at y=-0.5" << std::endl;
    
    // Create buildings/walls for wall-running
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-120.0, 120.0);  // Expanded to cover more of the 300x300 ground
    std::uniform_real_distribution<> height_dis(8.0, 25.0);  // Taller buildings
    
    for (int i = 0; i < 30; ++i) {  // More buildings for better coverage
        auto building = coordinator.CreateEntity();
        float x = dis(gen);
        float z = dis(gen);
        float height = height_dis(gen);
        
        // Vary building sizes for visual interest
        float buildingWidth = 6.0f + (i % 4) * 2.0f;  // 6, 8, 10, 12
        float buildingDepth = 6.0f + ((i + 1) % 4) * 2.0f;
        
        coordinator.AddComponent(building, Rendering::TransformComponent{
            glm::vec3(x, height/2.0f, z),
            glm::vec3(0.0f),
            glm::vec3(buildingWidth, height, buildingDepth)
        });
        coordinator.AddComponent(building, Rendering::MeshComponent{"player_cube"});
        
        // Vary building colors for visual detail
        float colorVariation = 0.5f + (i % 5) * 0.1f;
        glm::vec3 buildingColor = glm::vec3(colorVariation * 0.6f, colorVariation * 0.65f, colorVariation * 0.75f);
        coordinator.AddComponent(building, Rendering::MaterialComponent{
            buildingColor,
            0.4f + (i % 3) * 0.2f,  // Varying metallic
            0.5f + (i % 4) * 0.1f,  // Varying roughness
            1.0f
        });
        Gameplay::WallComponent wallComp;
        wallComp.canWallRun = true;
        coordinator.AddComponent(building, wallComp);
        
        // Collider half-extents must match visual scale
        coordinator.AddComponent(building, Physics::ColliderComponent{
            Physics::ColliderShape::BOX,
            glm::vec3(buildingWidth/2.0f, height/2.0f, buildingDepth/2.0f)
        });
    }
    
    // Create enemies throughout the map
    std::uniform_real_distribution<> enemy_pos(-30.0, 30.0);
    for (int i = 0; i < 10; ++i) {
        auto enemy = coordinator.CreateEntity();
        
        // Enemy AI component
        Gameplay::EnemyAIComponent enemyAI;
        enemyAI.detectionRange = 20.0f;
        enemyAI.attackRange = 3.0f;
        enemyAI.visionAngle = 90.0f;
        enemyAI.facingDirection = glm::vec3(1.0f, 0.0f, 0.0f);
        coordinator.AddComponent(enemy, enemyAI);
        
        // Enemy combat component
        Gameplay::EnemyCombatComponent enemyCombat;
        enemyCombat.damage = 10.0f;
        enemyCombat.attackCooldown = 2.0f;
        coordinator.AddComponent(enemy, enemyCombat);
        
        // Enemy movement
        coordinator.AddComponent(enemy, Gameplay::EnemyMovementComponent{});
        
        // Enemy physics
        Physics::RigidbodyComponent enemyRB;
        enemyRB.mass = 60.0f;
        coordinator.AddComponent(enemy, enemyRB);
        
        coordinator.AddComponent(enemy, Physics::ColliderComponent{
            Physics::ColliderShape::BOX,
            glm::vec3(0.8f, 1.8f, 0.8f)
        });
        
        // Enemy visual
        coordinator.AddComponent(enemy, Rendering::TransformComponent{
            glm::vec3(enemy_pos(gen), 1.0f, enemy_pos(gen)),
            glm::vec3(0.0f),
            glm::vec3(1.2f, 2.0f, 1.2f)
        });
        coordinator.AddComponent(enemy, Rendering::MeshComponent{"player_cube"});
        coordinator.AddComponent(enemy, Rendering::MaterialComponent{
            glm::vec3(1.0f, 0.0f, 0.0f), // Red enemies
            0.0f, 0.5f, 1.0f
        });
        
        // Targeting component
        coordinator.AddComponent(enemy, Gameplay::TargetingComponent{});
    }
    
    // Create invisible boundary walls to prevent falling off
    std::cout << "Creating boundary walls..." << std::endl;
    const float boundaryHeight = 5.0f;
    const float boundaryThickness = 1.0f;
    const float platformEdge = 145.0f;  // Slightly inside the 150 half-extent
    
    // North wall (positive Z)
    auto northWall = coordinator.CreateEntity();
    coordinator.AddComponent(northWall, Rendering::TransformComponent{
        glm::vec3(0.0f, boundaryHeight/2.0f, platformEdge),
        glm::vec3(0.0f),
        glm::vec3(300.0f, boundaryHeight, boundaryThickness)
    });
    coordinator.AddComponent(northWall, Physics::ColliderComponent{
        Physics::ColliderShape::BOX,
        glm::vec3(150.0f, boundaryHeight/2.0f, boundaryThickness/2.0f)
    });
    
    // South wall (negative Z)
    auto southWall = coordinator.CreateEntity();
    coordinator.AddComponent(southWall, Rendering::TransformComponent{
        glm::vec3(0.0f, boundaryHeight/2.0f, -platformEdge),
        glm::vec3(0.0f),
        glm::vec3(300.0f, boundaryHeight, boundaryThickness)
    });
    coordinator.AddComponent(southWall, Physics::ColliderComponent{
        Physics::ColliderShape::BOX,
        glm::vec3(150.0f, boundaryHeight/2.0f, boundaryThickness/2.0f)
    });
    
    // East wall (positive X)
    auto eastWall = coordinator.CreateEntity();
    coordinator.AddComponent(eastWall, Rendering::TransformComponent{
        glm::vec3(platformEdge, boundaryHeight/2.0f, 0.0f),
        glm::vec3(0.0f),
        glm::vec3(boundaryThickness, boundaryHeight, 300.0f)
    });
    coordinator.AddComponent(eastWall, Physics::ColliderComponent{
        Physics::ColliderShape::BOX,
        glm::vec3(boundaryThickness/2.0f, boundaryHeight/2.0f, 150.0f)
    });
    
    // West wall (negative X)
    auto westWall = coordinator.CreateEntity();
    coordinator.AddComponent(westWall, Rendering::TransformComponent{
        glm::vec3(-platformEdge, boundaryHeight/2.0f, 0.0f),
        glm::vec3(0.0f),
        glm::vec3(boundaryThickness, boundaryHeight, 300.0f)
    });
    coordinator.AddComponent(westWall, Physics::ColliderComponent{
        Physics::ColliderShape::BOX,
        glm::vec3(boundaryThickness/2.0f, boundaryHeight/2.0f, 150.0f)
    });
    
    std::cout << "Boundary walls created (invisible collision barriers)" << std::endl;
    
    // Create collectibles
    for (int i = 0; i < 20; ++i) {
        auto collectible = coordinator.CreateEntity();
        
        Gameplay::CollectibleComponent collectibleComp;
        collectibleComp.type = Gameplay::CollectibleComponent::CollectibleType::HEALTH_PACK;
        collectibleComp.value = 25.0f;
        coordinator.AddComponent(collectible, collectibleComp);
        
        coordinator.AddComponent(collectible, Rendering::TransformComponent{
            glm::vec3(dis(gen), 2.0f, dis(gen)),
            glm::vec3(0.0f),
            glm::vec3(0.5f, 0.5f, 0.5f)
        });
        coordinator.AddComponent(collectible, Rendering::MeshComponent{"player_cube"});
        coordinator.AddComponent(collectible, Rendering::MaterialComponent{
            glm::vec3(0.0f, 1.0f, 0.0f), // Green health pickups
            0.0f, 0.2f, 1.0f
        });
    }
    
    std::cout << "3D environment created with buildings, enemies, and collectibles!" << std::endl;
}

void CleanupWindow() {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

int main() {
    std::cout << "Starting CudaGame - Full 3D Experience..." << std::endl;
    
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
    coordinator.RegisterComponent<Physics::CharacterControllerComponent>();
    coordinator.RegisterComponent<Rendering::TransformComponent>();
    coordinator.RegisterComponent<Rendering::MeshComponent>();
    coordinator.RegisterComponent<Rendering::MaterialComponent>();
    
    // Register and initialize systems
    auto playerMovementSystem = coordinator.RegisterSystem<Gameplay::PlayerMovementSystem>();
    auto characterControllerSystem = coordinator.RegisterSystem<Gameplay::CharacterControllerSystem>();
    auto enemyAISystem = coordinator.RegisterSystem<Gameplay::EnemyAISystem>();
    auto levelSystem = coordinator.RegisterSystem<Gameplay::LevelSystem>();
    auto targetingSystem = coordinator.RegisterSystem<Gameplay::TargetingSystem>();
    auto physicsSystem = coordinator.RegisterSystem<Physics::PhysXPhysicsSystem>();
    auto wallRunSystem = coordinator.RegisterSystem<Physics::WallRunningSystem>();
    auto renderSystem = coordinator.RegisterSystem<Rendering::RenderSystem>();
    auto particleSystem = coordinator.RegisterSystem<Particles::ParticleSystem>();
    
    // Set system signatures
    Core::Signature playerMovementSignature;
    playerMovementSignature.set(coordinator.GetComponentType<Gameplay::PlayerMovementComponent>());
    playerMovementSignature.set(coordinator.GetComponentType<Gameplay::PlayerInputComponent>());
    playerMovementSignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    playerMovementSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Gameplay::PlayerMovementSystem>(playerMovementSignature);

    Core::Signature characterControllerSignature;
    characterControllerSignature.set(coordinator.GetComponentType<Physics::CharacterControllerComponent>());
    characterControllerSignature.set(coordinator.GetComponentType<Gameplay::PlayerInputComponent>());
    characterControllerSignature.set(coordinator.GetComponentType<Gameplay::PlayerMovementComponent>());
    characterControllerSignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    characterControllerSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Gameplay::CharacterControllerSystem>(characterControllerSignature);

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

    // PhysX system processes all entities with ColliderComponent
    // RigidbodyComponent is optional - entities without it become static actors
    Core::Signature physicsSignature;
    physicsSignature.set(coordinator.GetComponentType<Physics::ColliderComponent>());
    coordinator.SetSystemSignature<Physics::PhysXPhysicsSystem>(physicsSignature);
    
    Core::Signature wallRunSignature;
    wallRunSignature.set(coordinator.GetComponentType<Physics::CharacterControllerComponent>());
    wallRunSignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    wallRunSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Physics::WallRunningSystem>(wallRunSignature);

    Core::Signature particleSignature;
    coordinator.SetSystemSignature<Particles::ParticleSystem>(particleSignature);
    
    // Create and initialize RenderDebugSystem
    auto renderDebugSystem = coordinator.RegisterSystem<Rendering::RenderDebugSystem>();
    renderDebugSystem->Initialize();

    // Create OpenGL Debug Renderer adapter
    auto debugRenderer = std::make_shared<Debug::OpenGLDebugRenderer>(renderDebugSystem);
    debugRenderer->EnableDebugDrawing(true);

    // Initialize all systems
    playerMovementSystem->Initialize();
    characterControllerSystem->Initialize();
    enemyAISystem->Initialize();
    levelSystem->Initialize();
    targetingSystem->Initialize();
    physicsSystem->Initialize();
    wallRunSystem->Initialize();
    renderSystem->Initialize();
    particleSystem->Initialize();
    
    // Create and setup OrbitCamera with proper 3D positioning
    std::cout << "Creating 3D OrbitCamera..." << std::endl;
    auto camera = std::make_unique<Rendering::OrbitCamera>(Rendering::ProjectionType::PERSPECTIVE);
    
    // Configure orbit camera settings
    Rendering::OrbitCamera::OrbitSettings orbitSettings;
    orbitSettings.distance = 15.0f;
    orbitSettings.heightOffset = 2.0f;
    orbitSettings.mouseSensitivity = 0.05f;  // Reduced sensitivity for smoother control
    orbitSettings.smoothSpeed = 6.0f;
    camera->SetOrbitSettings(orbitSettings);
    
    camera->SetPerspective(60.0f, (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 200.0f);
    camera->SetCameraMode(Rendering::OrbitCamera::CameraMode::ORBIT_FOLLOW);
    
    // Initialize camera with a default target position for proper initial setup
    camera->SetTarget(glm::vec3(0.0f, 2.0f, 0.0f)); // Set to player's expected position
    camera->Update(0.016f, glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.0f)); // Force initial update
    camera->UpdateMatrices();
    mainCamera = camera.get();
    
    // Set the camera in the render system
    renderSystem->SetMainCamera(camera.get());
    // Set the camera in the character controller system for camera-relative movement
    characterControllerSystem->SetCamera(camera.get());
    // Set the camera in the player movement system for camera-relative movement
    playerMovementSystem->SetCamera(camera.get());
    std::cout << "3D OrbitCamera configured!" << std::endl;
    
    // Create the player entity with all systems
    Core::Entity player = coordinator.CreateEntity();
    
    // Player movement component - uses tuned defaults from PlayerComponents.h
    Gameplay::PlayerMovementComponent playerMovement;
    // baseSpeed=15.0, maxSpeed=50.0, sprintMultiplier=2.0, jumpForce=20.0 from struct defaults
    coordinator.AddComponent(player, playerMovement);
    
    // Player combat component
    Gameplay::PlayerCombatComponent playerCombat;
    playerCombat.health = 100.0f;
    playerCombat.maxHealth = 100.0f;
    playerCombat.currentWeapon = Gameplay::WeaponType::SWORD;
    playerCombat.inventory.push_back(Gameplay::WeaponType::NONE);
    playerCombat.inventory.push_back(Gameplay::WeaponType::SWORD);
    playerCombat.inventory.push_back(Gameplay::WeaponType::STAFF);
    coordinator.AddComponent(player, playerCombat);
    
    // Player input component
    Gameplay::PlayerInputComponent playerInput;
    coordinator.AddComponent(player, playerInput);
    
    // Player rhythm component
    Gameplay::PlayerRhythmComponent playerRhythm;
    coordinator.AddComponent(player, playerRhythm);
    
    // Character controller for wall-running
    Physics::CharacterControllerComponent charController;
    coordinator.AddComponent(player, charController);
    
    // Player physics
    Physics::RigidbodyComponent playerRigidbody;
    playerRigidbody.mass = 80.0f;
    playerRigidbody.isKinematic = false; // Start in dynamic mode for immediate control
    playerRigidbody.velocity = glm::vec3(0.0f, 0.0f, 0.0f); // Explicitly zero initial velocity
    playerRigidbody.acceleration = glm::vec3(0.0f, 0.0f, 0.0f); // Zero acceleration
    playerRigidbody.forceAccumulator = glm::vec3(0.0f, 0.0f, 0.0f); // Zero forces
    coordinator.AddComponent(player, playerRigidbody);
    
    Physics::ColliderComponent playerCollider;
    playerCollider.shape = Physics::ColliderShape::BOX;
    playerCollider.size = glm::vec3(0.8f, 1.8f, 0.8f);
    coordinator.AddComponent(player, playerCollider);
    
    // Player visual representation
    Rendering::TransformComponent playerTransform;
    playerTransform.position = glm::vec3(0.0f, 2.0f, 0.0f);  // Start closer to ground
    playerTransform.scale = glm::vec3(0.8f, 1.8f, 0.8f);
    coordinator.AddComponent(player, playerTransform);
    std::cout << "[DEBUG] Player spawned at y=2.0, collider halfHeight=0.9, should rest at y=0.4" << std::endl;
    
    Rendering::MeshComponent playerMesh;
    playerMesh.modelPath = "player_cube";
    coordinator.AddComponent(player, playerMesh);
    
    Rendering::MaterialComponent playerMaterial;
    playerMaterial.albedo = glm::vec3(0.0f, 0.5f, 1.0f); // Blue player
    playerMaterial.metallic = 0.2f;
    playerMaterial.roughness = 0.6f;
    coordinator.AddComponent(player, playerMaterial);
    
    std::cout << "Player created with full component set!" << std::endl;
    
    // Create the game environment
    CreateGameEnvironment(coordinator);
    
    // Main Game Loop
    std::cout << "Starting main game loop..." << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "TAB - Toggle mouse capture (REQUIRED for camera control)" << std::endl;
    std::cout << "WASD - Move player" << std::endl;
    std::cout << "Mouse - Rotate camera (when TAB is pressed)" << std::endl;
    std::cout << "Mouse Wheel - Zoom in/out" << std::endl;
    std::cout << "1 - Orbit Follow Camera (default)" << std::endl;
    std::cout << "2 - Free Look Camera" << std::endl;
    std::cout << "3 - Combat Focus Camera" << std::endl;
    std::cout << "Space - Jump (Double jump in air!)" << std::endl;
    std::cout << "Shift - Sprint" << std::endl;
    std::cout << "E - Wall Run (hold when near walls)" << std::endl;
    std::cout << "Left Control - Dash" << std::endl;
    std::cout << "Left Click - Attack" << std::endl;
    std::cout << "Right Click - Heavy Attack" << std::endl;
    std::cout << "Q - Block/Parry" << std::endl;
    std::cout << "K - Toggle Physics Mode (Dynamic/Kinematic) [DEBUG]" << std::endl;
    std::cout << "F4 - Cycle G-buffer Debug Mode" << std::endl;
    std::cout << "F5 - Toggle Camera Frustum Debug" << std::endl;
    std::cout << "PageUp/PageDown - Adjust Depth Scale" << std::endl;
    std::cout << "ESC - Exit" << std::endl;
    std::cout << "\n*** Press TAB first to enable mouse control! ***\n" << std::endl;
    
const float FIXED_TIMESTEP = 1.0f / 60.0f; // Fixed timestep for physics simulation
    float accumulator = 0.0f;
    float deltaTime = 0.016f; // Initialize with 60 FPS default
    auto lastFrame = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    
    // FPS tracking
    auto fpsLastTime = lastFrame;
    int fpsFrameCount = 0;
    float currentFPS = 0.0f;
    
    // Store previous and current physics states for interpolation
    glm::vec3 playerPrevPos = glm::vec3(0.0f, 2.0f, 0.0f);
    glm::vec3 playerCurrentPos = glm::vec3(0.0f, 2.0f, 0.0f);
    
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        auto currentFrame = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(currentFrame - lastFrame).count();
        lastFrame = currentFrame;
        
        // FPS calculation (update every second)
        fpsFrameCount++;
        auto fpsDuration = std::chrono::duration<float>(currentFrame - fpsLastTime).count();
        if (fpsDuration >= 1.0f) {
            currentFPS = fpsFrameCount / fpsDuration;
            fpsFrameCount = 0;
            fpsLastTime = currentFrame;
            std::cout << "FPS: " << static_cast<int>(currentFPS) << " | Frame Time: " << (deltaTime * 1000.0f) << "ms" << std::endl;
        }
        
        // Poll for and process events
        glfwPollEvents();
        
        // Update player input from keyboard/mouse
        auto& playerInputComp = coordinator.GetComponent<Gameplay::PlayerInputComponent>(player);
        
        // Movement input
        glm::vec2 moveInput(0.0f);
        if (keys[GLFW_KEY_W]) moveInput.y = 1.0f;
        if (keys[GLFW_KEY_S]) moveInput.y = -1.0f;
        if (keys[GLFW_KEY_A]) moveInput.x = -1.0f;
        if (keys[GLFW_KEY_D]) moveInput.x = 1.0f;

        // Normalize diagonal movement
        if (glm::length(moveInput) > 0.0f) {
            moveInput = glm::normalize(moveInput);
        }

        // Update keyboard state in player input component
        playerInputComp.keys[GLFW_KEY_W] = keys[GLFW_KEY_W];
        playerInputComp.keys[GLFW_KEY_A] = keys[GLFW_KEY_A];
        playerInputComp.keys[GLFW_KEY_S] = keys[GLFW_KEY_S];
        playerInputComp.keys[GLFW_KEY_D] = keys[GLFW_KEY_D];
        playerInputComp.keys[GLFW_KEY_SPACE] = keys[GLFW_KEY_SPACE];
        playerInputComp.keys[GLFW_KEY_LEFT_SHIFT] = keys[GLFW_KEY_LEFT_SHIFT];
        playerInputComp.keys[GLFW_KEY_LEFT_CONTROL] = keys[GLFW_KEY_LEFT_CONTROL];
        playerInputComp.keys[GLFW_KEY_E] = keys[GLFW_KEY_E];
        playerInputComp.keys[GLFW_KEY_Q] = keys[GLFW_KEY_Q];
        playerInputComp.keys[GLFW_KEY_1] = keys[GLFW_KEY_1];
        playerInputComp.keys[GLFW_KEY_2] = keys[GLFW_KEY_2];
        playerInputComp.keys[GLFW_KEY_3] = keys[GLFW_KEY_3];
        
        // Toggle kinematic mode with K key for testing
        static bool kPressed = false;
        if (keys[GLFW_KEY_K] && !kPressed) {
            auto& playerRB = coordinator.GetComponent<Physics::RigidbodyComponent>(player);
            playerRB.isKinematic = !playerRB.isKinematic;
            std::cout << "\n=== SWITCHED PLAYER MODE ===" << std::endl;
            std::cout << "Player is now: " << (playerRB.isKinematic ? "KINEMATIC (Fixed Position)" : "DYNAMIC (Physics-Based)") << std::endl;
            std::cout << "============================\n" << std::endl;
            kPressed = true;
        } else if (!keys[GLFW_KEY_K]) {
            kPressed = false;
        }
        
        // Clear other key press states at the end of frame
        for (int i = 0; i < 1024; i++) {
            if (i != GLFW_KEY_1 && i != GLFW_KEY_2 && i != GLFW_KEY_3) {
                keysPressed[i] = false;
            }
        }
        
        // Update mouse input for targeting
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        playerInputComp.mousePos = glm::vec2(mouseX, mouseY);
        playerInputComp.mouseButtons[0] = mouseButtons[GLFW_MOUSE_BUTTON_LEFT];
        playerInputComp.mouseButtons[1] = mouseButtons[GLFW_MOUSE_BUTTON_RIGHT];
        
        // Update OrbitCamera to follow player
        auto& playerTransform = coordinator.GetComponent<Rendering::TransformComponent>(player);
        
        // Get player velocity for predictive camera movement
        auto& playerRigidbody = coordinator.GetComponent<Physics::RigidbodyComponent>(player);
        glm::vec3 playerVelocity = playerRigidbody.velocity;
        
        // Handle camera mode switching
        if (keysPressed[GLFW_KEY_1]) {
            std::cout << "Switching to Camera Mode 1: ORBIT_FOLLOW" << std::endl;
            mainCamera->SetCameraMode(Rendering::OrbitCamera::CameraMode::ORBIT_FOLLOW);
            keysPressed[GLFW_KEY_1] = false; // Reset press state
        } else if (keysPressed[GLFW_KEY_2]) {
            std::cout << "Switching to Camera Mode 2: FREE_LOOK" << std::endl;
            mainCamera->SetCameraMode(Rendering::OrbitCamera::CameraMode::FREE_LOOK);
            keysPressed[GLFW_KEY_2] = false; // Reset press state
        } else if (keysPressed[GLFW_KEY_3]) {
            std::cout << "Switching to Camera Mode 3: COMBAT_FOCUS" << std::endl;
            mainCamera->SetCameraMode(Rendering::OrbitCamera::CameraMode::COMBAT_FOCUS);
            keysPressed[GLFW_KEY_3] = false; // Reset press state
        }
        
// === INTERPOLATED CAMERA UPDATE FOR SMOOTH RENDERING ===
        // Interpolate between previous and current physics states for smooth visuals
        float alpha = accumulator / FIXED_TIMESTEP;  // How far between physics updates are we?
        glm::vec3 interpolatedPos = glm::mix(playerPrevPos, playerCurrentPos, alpha);
        
        // Use interpolated position for camera target
        glm::vec3 cameraTarget = interpolatedPos;
        
        // Debug camera target selection (only log every 300 frames to reduce spam)
        static int cameraDebugCount = 0;
        cameraDebugCount++;
        if (cameraDebugCount % 300 == 0) {
            bool playerIsKinematic = coordinator.GetComponent<Physics::RigidbodyComponent>(player).isKinematic;
            std::cout << "[CameraUpdate] Target: (" << cameraTarget.x << ", " << cameraTarget.y << ", " << cameraTarget.z << ") "
                      << "Mode: " << (playerIsKinematic ? "KINEMATIC" : "DYNAMIC") 
                      << " Velocity: (" << playerVelocity.x << ", " << playerVelocity.y << ", " << playerVelocity.z << ")" << std::endl;
        }
        
        // Single, clean camera update - no conflicting calls
        // Note: Do NOT call SetTarget() here as Update() already handles target position
        mainCamera->Update(deltaTime, cameraTarget, playerVelocity);
        
        // Update all systems
        // playerMovementSystem->Update(deltaTime);  // Disabled - using CharacterControllerSystem instead
        // CharacterControllerSystem provides more advanced movement features
        characterControllerSystem->Update(deltaTime);
        enemyAISystem->Update(deltaTime);
    levelSystem->Update(deltaTime);
    targetingSystem->Update(deltaTime);
    
    // Fixed timestep physics update with state capture
    accumulator += deltaTime;
    int physicsSteps = 0;
    
    // Save previous state before physics updates
    playerPrevPos = playerCurrentPos;
    
    while (accumulator >= FIXED_TIMESTEP) {
        if (frameCount == 0) {
            std::cout << "[DEBUG] First physics update - PhysX should create actors now" << std::endl;
        }
        physicsSystem->Update(FIXED_TIMESTEP);
        accumulator -= FIXED_TIMESTEP;
        physicsSteps++;
    }
    
    // Capture current physics state after updates
    playerCurrentPos = coordinator.GetComponent<Rendering::TransformComponent>(player).position;
    
    // Debug: Log physics steps every 120 frames (approximately 2 seconds at 60 FPS)
    frameCount++;
    if (frameCount % 120 == 0) {
        std::cout << "[DEBUG] Physics steps this frame: " << physicsSteps 
                  << ", Frame deltaTime: " << deltaTime*1000.0f << "ms" << std::endl;
    }
    
    // wallRunSystem->Update(deltaTime);  // Disabled - CharacterControllerSystem handles wall-running
        particleSystem->Update(deltaTime);
        
        // Clear screen
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
    // Debug rendering frame start
        debugRenderer->BeginFrame();

        // Render
        renderSystem->Update(deltaTime);

        // Draw any debug visuals
        {
            const auto& playerTrans = coordinator.GetComponent<Rendering::TransformComponent>(player);
            // Draw player's facing direction
            glm::vec3 forward = glm::normalize(mainCamera->GetForward());
            debugRenderer->DrawLine(
                playerTrans.position,
                playerTrans.position + forward * 3.0f,
                glm::vec3(0.0f, 1.0f, 0.0f)  // Green for forward
            );

            // Draw player's velocity vector if in debug mode
            const auto& playerRB = coordinator.GetComponent<Physics::RigidbodyComponent>(player);
            if (glm::length(playerRB.velocity) > 0.1f) {
                debugRenderer->DrawLine(
                    playerTrans.position,
                    playerTrans.position + glm::normalize(playerRB.velocity) * 2.0f,
                    glm::vec3(1.0f, 0.0f, 0.0f)  // Red for velocity
                );
            }

            // Draw player's bounding box
            const auto& collider = coordinator.GetComponent<Physics::ColliderComponent>(player);
            glm::vec3 halfExtents = collider.size * 0.5f;
            debugRenderer->DrawBox(
                playerTrans.position - halfExtents,
                playerTrans.position + halfExtents,
                glm::vec3(0.0f, 0.0f, 1.0f)  // Blue for collider
            );
        }

        debugRenderer->EndFrame();
        
        // Swap front and back buffers
        glfwSwapBuffers(window);
        
        // F4 to cycle debug modes
        static bool f4Pressed = false;
        if (keys[GLFW_KEY_F4]) {
            if (!f4Pressed) {
                renderSystem->CycleDebugMode();
                f4Pressed = true;
            }
        } else {
            f4Pressed = false;
        }

        // F3 to toggle debug drawing
        static bool f3Pressed = false;
        static bool debugDrawingEnabled = true;
        if (keys[GLFW_KEY_F3]) {
            if (!f3Pressed) {
                debugDrawingEnabled = !debugDrawingEnabled;
                debugRenderer->EnableDebugDrawing(debugDrawingEnabled);
                std::cout << "Debug drawing " << (debugDrawingEnabled ? "enabled" : "disabled") << std::endl;
                f3Pressed = true;
            }
        } else {
            f3Pressed = false;
        }
        
        // F5 to toggle camera debug
        static bool f5Pressed = false;
        if (keys[GLFW_KEY_F5]) {
            if (!f5Pressed) {
                renderSystem->ToggleCameraDebug();
                f5Pressed = true;
            }
        } else {
            f5Pressed = false;
        }
        
        // PageUp/PageDown to adjust depth scale for position buffer visualization
        static bool pageUpPressed = false;
        static bool pageDownPressed = false;
        
        if (keys[GLFW_KEY_PAGE_UP] && !pageUpPressed) {
            renderSystem->AdjustDepthScale(0.8f); // Zoom in (smaller scale)
            pageUpPressed = true;
        } else if (!keys[GLFW_KEY_PAGE_UP]) {
            pageUpPressed = false;
        }
        
        if (keys[GLFW_KEY_PAGE_DOWN] && !pageDownPressed) {
            renderSystem->AdjustDepthScale(1.25f); // Zoom out (larger scale)
            pageDownPressed = true;
        } else if (!keys[GLFW_KEY_PAGE_DOWN]) {
            pageDownPressed = false;
        }
        
        // Exit on ESC key
        if (keys[GLFW_KEY_ESCAPE]) {
            glfwSetWindowShouldClose(window, true);
        }
    }

    std::cout << "\nGame ended. Thanks for playing!" << std::endl;
    
    // Cleanup
    CleanupWindow();
    
    return 0;
}
