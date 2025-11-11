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
#include "Rendering/CudaBuildingGenerator.h"
#include "Rendering/ProceduralCharacter.h"
#include "UI/UIRenderer.h"
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
// RenderSystem pointer for debug toggles
static CudaGame::Rendering::RenderSystem* g_renderSystemPtr = nullptr;

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

// UI
std::unique_ptr<UI::UIRenderer> g_uiRenderer = nullptr;
bool g_showHUD = false; // start hidden by default

// Startup UX
bool g_hasEnabledMouse = false;   // becomes true on first TAB press
bool g_showWelcomePrompt = true;  // show center prompt until TAB

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
        // Mark welcome prompt complete on first TAB
        if (!g_hasEnabledMouse) {
            g_hasEnabledMouse = true;
            g_showWelcomePrompt = false;
        }
    }

    // Debug view toggles
    if (action == GLFW_PRESS && g_renderSystemPtr) {
        if (key == GLFW_KEY_F1) {
            g_renderSystemPtr->CycleDebugMode();
        } else if (key == GLFW_KEY_F2) {
            g_renderSystemPtr->AdjustDepthScale(1.5f);
        } else if (key == GLFW_KEY_F3) {
            g_renderSystemPtr->AdjustDepthScale(1.0f/1.5f);
        } else if (key == GLFW_KEY_F5) {
            g_renderSystemPtr->ToggleCameraDebug();
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

// Persistent storage for procedurally generated buildings
static std::unique_ptr<Rendering::CudaBuildingGenerator> g_buildingGen;
static std::vector<Rendering::BuildingMesh> g_buildingMeshes;  // Keep meshes alive for VAO lifetime
static std::vector<GLuint> g_emissiveTextures;
static Rendering::CharacterMeshGPU g_characterMesh{};

void CreateGameEnvironment(Core::Coordinator& coordinator) {
    std::cout << "Creating 3D game environment..." << std::endl;
    
    // Create ground plane (OPTIMAL - 2250x2250, ~50x larger than original)
    auto ground = coordinator.CreateEntity();
    coordinator.AddComponent(ground, Rendering::TransformComponent{
        glm::vec3(0.0f, -1.0f, 0.0f),
        glm::vec3(0.0f),
        glm::vec3(2250.0f, 1.0f, 2250.0f)  // Was 300→3000→2250 (25% smaller)
    });
    coordinator.AddComponent(ground, Rendering::MeshComponent{"player_cube"});
    coordinator.AddComponent(ground, Rendering::MaterialComponent{
        glm::vec3(0.3f, 0.3f, 0.3f), // Dark gray ground
        0.0f, 0.8f, 1.0f
    });
    
    // Add physics components for collision (use halfExtents explicitly)
    {
        Physics::ColliderComponent groundCol{};
        groundCol.shape = Physics::ColliderShape::BOX;
        groundCol.halfExtents = glm::vec3(1125.0f, 0.5f, 1125.0f);  // Half extents for 2250x2250 world
        coordinator.AddComponent(ground, groundCol);
    }
    // Ground treated as static: either omit Rigidbody or set mass to 0
    Physics::RigidbodyComponent groundRB;
    groundRB.setMass(0.0f);  // Zero mass = static (inverseMass=0)
    groundRB.isKinematic = true; // explicit non-dynamic behavior
    coordinator.AddComponent(ground, groundRB);
    std::cout << "[DEBUG] Ground created at y=-1.0, scale.y=1.0, top surface at y=-0.5 (STATIC)" << std::endl;
    
    // Initialize CUDA building generator once
    if (!g_buildingGen) {
        g_buildingGen = std::make_unique<Rendering::CudaBuildingGenerator>();
        g_buildingGen->Initialize();
    }

    // Create buildings/walls for wall-running (OPTIMAL SCALE)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-900.0, 900.0);  // 2250x2250 world
    std::uniform_real_distribution<> height_dis(12.0, 40.0);  // Varied heights
    
    std::cout << "[OPTIMAL WORLD] Generating 150 buildings across 2250x2250 units..." << std::endl;
    std::cout << "[LIMITS] Tested: 6000x6000/500 buildings (too heavy), 3000x3000/200 (still heavy)" << std::endl;
    for (int i = 0; i < 150; ++i) {  // Optimal: 150 buildings (25% reduction)
        auto building = coordinator.CreateEntity();
        float x = dis(gen);
        float z = dis(gen);
        float height = height_dis(gen);
        
        // Vary building sizes for visual interest
        float buildingWidth = 6.0f + (i % 4) * 2.0f;  // 6, 8, 10, 12
        float buildingDepth = 6.0f + ((i + 1) % 4) * 2.0f;
        
        // Procedurally generate building geometry + emissive texture
        Rendering::BuildingStyle style;
        style.baseWidth = buildingWidth;
        style.baseDepth = buildingDepth;
        style.height = height;
        style.seed = static_cast<uint32_t>(i * 1337 + 42);

        Rendering::BuildingMesh mesh = g_buildingGen->GenerateBuilding(style);
        g_buildingGen->UploadToGPU(mesh);
        
        Rendering::BuildingTexture btex = g_buildingGen->GenerateBuildingTexture(style, 512);
        // Upload emissive texture
        GLuint emissiveTex = 0;
        glGenTextures(1, &emissiveTex);
        glBindTexture(GL_TEXTURE_2D, emissiveTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, btex.width, btex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, btex.emissiveData.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        // Persist resources for lifetime
        g_emissiveTextures.push_back(emissiveTex);
        g_buildingMeshes.push_back(mesh);

        // Components
        coordinator.AddComponent(building, Rendering::TransformComponent{
            glm::vec3(x, height/2.0f, z),
            glm::vec3(0.0f),
            glm::vec3(1.0f)
        });
        Rendering::MeshComponent meshComp{};
        meshComp.modelPath = ""; // use VAO path
        meshComp.vaoId = mesh.vao; meshComp.vbo = mesh.vbo; meshComp.ebo = mesh.ebo;
        meshComp.indexCount = static_cast<uint32_t>(mesh.indices.size());
        coordinator.AddComponent(building, meshComp);
        
        Rendering::MaterialComponent mat{};
        mat.albedo = glm::vec3(0.6f); // color from vertex color will dominate
        mat.metallic = 0.3f; mat.roughness = 0.6f; mat.ao = 1.0f;
        mat.emissiveMap = emissiveTex;
        mat.emissiveIntensity = 2.5f; // stronger window glow
        coordinator.AddComponent(building, mat);

        Gameplay::WallComponent wallComp; wallComp.canWallRun = true;
        coordinator.AddComponent(building, wallComp);
        
        // Collider half-extents should match mesh bounds; approximate with style
        {
            Physics::ColliderComponent col{};
            col.shape = Physics::ColliderShape::BOX;
            col.halfExtents = glm::vec3(buildingWidth/2.0f, height/2.0f, buildingDepth/2.0f);
            coordinator.AddComponent(building, col);
        }
    }
    
// Create enemies throughout the map (ensure not spawning too close to player)
    std::uniform_real_distribution<> enemy_pos(-300.0, 300.0);
for (int i = 0; i < 10; ++i) {
        auto enemy = coordinator.CreateEntity();

        // Ensure enemy spawns at a safe distance from player origin
        float x = enemy_pos(gen);
        float z = enemy_pos(gen);
        int safetyIterations = 0;
        const float minDist = 60.0f;
        while ((x * x + z * z) < (minDist * minDist) && safetyIterations < 10) {
            x = enemy_pos(gen);
            z = enemy_pos(gen);
            safetyIterations++;
        }
        
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
            glm::vec3(x, 1.0f, z),
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
    // Cleanup generated building resources
    if (g_buildingGen) {
        for (auto& m : g_buildingMeshes) {
            g_buildingGen->CleanupGPUMesh(m);
        }
        g_buildingMeshes.clear();
        g_buildingGen->Shutdown();
        g_buildingGen.reset();
    }
    if (!g_emissiveTextures.empty()) {
        glDeleteTextures(static_cast<GLsizei>(g_emissiveTextures.size()), g_emissiveTextures.data());
        g_emissiveTextures.clear();
    }
    if (g_characterMesh.vao) {
        glDeleteVertexArrays(1, &g_characterMesh.vao);
        g_characterMesh.vao = 0;
    }
    if (g_characterMesh.vbo) {
        glDeleteBuffers(1, &g_characterMesh.vbo);
        g_characterMesh.vbo = 0;
    }
    if (g_characterMesh.ebo) {
        glDeleteBuffers(1, &g_characterMesh.ebo);
        g_characterMesh.ebo = 0;
    }
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
    g_renderSystemPtr = renderSystem.get();
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
    enemyAISignature.set(coordinator.GetComponentType<Gameplay::EnemyAIComponent>());
    enemyAISignature.set(coordinator.GetComponentType<Gameplay::EnemyCombatComponent>());
    enemyAISignature.set(coordinator.GetComponentType<Gameplay::EnemyMovementComponent>());
    enemyAISignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    enemyAISignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
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
    
    // Load HDR skybox
    std::cout << "Loading HDR skybox..." << std::endl;
    if (renderSystem->LoadSkyboxHDR("C:\\Users\\Brandon\\CudaGame\\assets\\hdri\\qwantani_noon_puresky_4k.hdr", 512)) {
        std::cout << "HDR skybox loaded successfully!" << std::endl;
    } else {
        std::cerr << "Warning: Failed to load HDR skybox" << std::endl;
    }
    
    // Create and setup OrbitCamera with proper 3D positioning
    std::cout << "Creating 3D OrbitCamera..." << std::endl;
    auto camera = std::make_unique<Rendering::OrbitCamera>(Rendering::ProjectionType::PERSPECTIVE);
    
    // Configure orbit camera settings (Zelda/Kirby-style)
    Rendering::OrbitCamera::OrbitSettings orbitSettings;
    orbitSettings.distance = 8.0f;            // closer to character
    orbitSettings.heightOffset = 3.0f;        // slightly higher
    orbitSettings.mouseSensitivity = 0.03f;   // less twitchy
    orbitSettings.smoothSpeed = 12.0f;        // faster smoothing
    orbitSettings.minDistance = 5.0f;
    orbitSettings.maxDistance = 20.0f;
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
    
    Physics::ColliderComponent playerCollider{};
    playerCollider.shape = Physics::ColliderShape::BOX;
    // Half extents for an 0.8 x 1.8 x 0.8 box
    playerCollider.halfExtents = glm::vec3(0.4f, 0.9f, 0.4f);
    coordinator.AddComponent(player, playerCollider);
    
    // Player visual representation
    Rendering::TransformComponent playerTransform;
    playerTransform.position = glm::vec3(0.0f, 2.0f, 0.0f);  // Start closer to ground
    playerTransform.scale = glm::vec3(1.0f); // natural scale; mesh is ~2.0 units tall
    coordinator.AddComponent(player, playerTransform);
    std::cout << "[DEBUG] Player spawned at y=2.0, collider halfHeight=0.9, should rest at y=0.4" << std::endl;
    
    // Create procedural character GPU mesh once
    if (g_characterMesh.vao == 0) {
        g_characterMesh = Rendering::CreateLowPolyCharacterGPU();
        std::cout << "[DEBUG] Procedural character VAO created: " << g_characterMesh.vao 
                  << ", indices: " << g_characterMesh.indexCount << std::endl;
    }
    
    Rendering::MeshComponent playerMesh;
    playerMesh.modelPath.clear();
    playerMesh.vaoId = g_characterMesh.vao;
    playerMesh.indexCount = g_characterMesh.indexCount;
    coordinator.AddComponent(player, playerMesh);
    
    Rendering::MaterialComponent playerMaterial;
    playerMaterial.albedo = glm::vec3(0.2f, 0.5f, 0.95f); // bright jacket blue
    playerMaterial.metallic = 0.1f;
    playerMaterial.roughness = 0.7f;
    coordinator.AddComponent(player, playerMaterial);
    
std::cout << "Player created with full component set!" << std::endl;
    
    // Bind player entity to systems that need it
    enemyAISystem->SetPlayerEntity(player);
    targetingSystem->SetPlayerEntity(player);
    
    // Initialize UI Renderer
    g_uiRenderer = std::make_unique<UI::UIRenderer>();
    if (g_uiRenderer->Initialize()) {
        g_uiRenderer->SetViewportSize(WINDOW_WIDTH, WINDOW_HEIGHT);
        std::cout << "UI Renderer initialized successfully!" << std::endl;
    } else {
        std::cerr << "Failed to initialize UI Renderer" << std::endl;
    }
    
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
    std::cout << "F1 - Cycle G-buffer Debug Mode (includes Emissive Color/Power)" << std::endl;
    std::cout << "F5 - Toggle Camera Frustum Debug" << std::endl;
    std::cout << "F2/F3 - Adjust Depth Scale (for Position debug)" << std::endl;
    std::cout << "\nSkybox Controls:" << std::endl;
    std::cout << "+/- - Adjust skybox exposure" << std::endl;
    std::cout << "[/] - Rotate skybox" << std::endl;
    std::cout << "B - Toggle skybox on/off" << std::endl;
    std::cout << "ESC - Exit" << std::endl;
    std::cout << "\n*** Press TAB first to enable mouse control! ***\n" << std::endl;
    
const float FIXED_TIMESTEP = 1.0f / 60.0f; // Fixed timestep for physics simulation
    float accumulator = 0.0f;
    float deltaTime = 0.016f; // Initialize with 60...(truncated)
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
        
        // Skybox controls
        // Toggle skybox with B key
        static bool bPressed = false;
        if (keys[GLFW_KEY_B] && !bPressed) {
            bool currentEnabled = renderSystem->GetSkyboxEnabled();
            renderSystem->SetSkyboxEnabled(!currentEnabled);
            std::cout << "Skybox " << (currentEnabled ? "disabled" : "enabled") << std::endl;
            bPressed = true;
        } else if (!keys[GLFW_KEY_B]) {
            bPressed = false;
        }
        
        // Adjust skybox exposure with +/- (using = and - keys)
        static bool plusPressed = false;
        static bool minusPressed = false;
        if (keys[GLFW_KEY_EQUAL] && !plusPressed) { // + key (GLFW_KEY_EQUAL is same physical key)
            renderSystem->AdjustSkyboxExposure(0.1f);
            plusPressed = true;
        } else if (!keys[GLFW_KEY_EQUAL]) {
            plusPressed = false;
        }
        
        if (keys[GLFW_KEY_MINUS] && !minusPressed) {
            renderSystem->AdjustSkyboxExposure(-0.1f);
            minusPressed = true;
        } else if (!keys[GLFW_KEY_MINUS]) {
            minusPressed = false;
        }
        
        // Rotate skybox with [ and ] keys
        static bool leftBracketPressed = false;
        static bool rightBracketPressed = false;
        if (keys[GLFW_KEY_LEFT_BRACKET] && !leftBracketPressed) {
            renderSystem->AdjustSkyboxRotation(-0.1f); // Rotate counter-clockwise
            leftBracketPressed = true;
        } else if (!keys[GLFW_KEY_LEFT_BRACKET]) {
            leftBracketPressed = false;
        }
        
        if (keys[GLFW_KEY_RIGHT_BRACKET] && !rightBracketPressed) {
            renderSystem->AdjustSkyboxRotation(0.1f); // Rotate clockwise
            rightBracketPressed = true;
        } else if (!keys[GLFW_KEY_RIGHT_BRACKET]) {
            rightBracketPressed = false;
        }
        
        // Reset player and camera with R key
        static bool rPressed = false;
        if (keys[GLFW_KEY_R] && !rPressed) {
            // Reset player position
            auto& playerTrans = coordinator.GetComponent<Rendering::TransformComponent>(player);
            playerTrans.position = glm::vec3(0.0f, 2.0f, 0.0f);
            
            // Reset player velocity
            auto& playerRB = coordinator.GetComponent<Physics::RigidbodyComponent>(player);
            playerRB.velocity = glm::vec3(0.0f);
            playerRB.acceleration = glm::vec3(0.0f);
            playerRB.forceAccumulator = glm::vec3(0.0f);
            
            // Reset camera
            mainCamera->SetTarget(glm::vec3(0.0f, 2.0f, 0.0f));
            
            std::cout << "[RESET] Player and camera reset to starting position" << std::endl;
            rPressed = true;
        } else if (!keys[GLFW_KEY_R]) {
            rPressed = false;
        }
        
        // Toggle HUD with H key
        static bool hPressed = false;
        if (keys[GLFW_KEY_H] && !hPressed) {
            g_showHUD = !g_showHUD;
            std::cout << "HUD " << (g_showHUD ? "enabled" : "disabled") << std::endl;
            hPressed = true;
        } else if (!keys[GLFW_KEY_H]) {
            hPressed = false;
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
        
        // Render UI/HUD
        if (g_uiRenderer) {
            g_uiRenderer->BeginFrame();

            // Center prompt until TAB pressed
            if (g_showWelcomePrompt && !g_hasEnabledMouse) {
                const float panelW = 520.0f;
                const float panelH = 140.0f;
                const float px = WINDOW_WIDTH * 0.5f - panelW * 0.5f;
                const float py = WINDOW_HEIGHT * 0.5f - panelH * 0.5f;
                g_uiRenderer->DrawFilledRect((int)px, (int)py, (int)panelW, (int)panelH, glm::vec4(0.0f,0.0f,0.0f,0.8f));
                g_uiRenderer->RenderText("WELCOME TO CUDAGAME", (int)(px + 20), (int)(py + 20), 1.1f, glm::vec3(0.0f,1.0f,1.0f));
                g_uiRenderer->RenderText("Press TAB to enable mouse control", (int)(px + 20), (int)(py + 60), 0.95f, glm::vec3(1.0f));
                g_uiRenderer->RenderText("Press H to show the controls panel", (int)(px + 20), (int)(py + 90), 0.9f, glm::vec3(0.8f));
            }
            else if (g_showHUD) {
                // Draw HUD background panel
                g_uiRenderer->DrawFilledRect(10, 10, 320, 280, glm::vec4(0.0f, 0.0f, 0.0f, 0.7f));
                
                float yPos = 20.0f;
                float lineHeight = 20.0f;
                
                // Title
                g_uiRenderer->RenderText("=== CUDAGAME CONTROLS ===", 20, yPos, 1.0f, glm::vec3(0.0f, 1.0f, 1.0f));
                yPos += lineHeight * 1.5f;
                
                // Movement
                g_uiRenderer->RenderText("TAB: Toggle Mouse", 20, yPos, 0.8f, glm::vec3(1.0f, 1.0f, 1.0f));
                yPos += lineHeight;
                g_uiRenderer->RenderText("WASD: Move", 20, yPos, 0.8f, glm::vec3(1.0f, 1.0f, 1.0f));
                yPos += lineHeight;
                g_uiRenderer->RenderText("Space: Jump", 20, yPos, 0.8f, glm::vec3(1.0f, 1.0f, 1.0f));
                yPos += lineHeight;
                g_uiRenderer->RenderText("Shift: Sprint", 20, yPos, 0.8f, glm::vec3(1.0f, 1.0f, 1.0f));
                yPos += lineHeight;
                
                // Debug
                yPos += lineHeight * 0.5f;
                g_uiRenderer->RenderText("F1: Debug Mode", 20, yPos, 0.8f, glm::vec3(1.0f, 1.0f, 0.0f));
                yPos += lineHeight;
                g_uiRenderer->RenderText("F5: Camera Debug", 20, yPos, 0.8f, glm::vec3(1.0f, 1.0f, 0.0f));
                yPos += lineHeight;
                
                // Skybox
                yPos += lineHeight * 0.5f;
                g_uiRenderer->RenderText("+/-: Skybox Exposure", 20, yPos, 0.8f, glm::vec3(0.5f, 1.0f, 0.5f));
                yPos += lineHeight;
                g_uiRenderer->RenderText("[/]: Rotate Skybox", 20, yPos, 0.8f, glm::vec3(0.5f, 1.0f, 0.5f));
                yPos += lineHeight;
                g_uiRenderer->RenderText("B: Toggle Skybox", 20, yPos, 0.8f, glm::vec3(0.5f, 1.0f, 0.5f));
                yPos += lineHeight;
                
                // Utility
                yPos += lineHeight * 0.5f;
                g_uiRenderer->RenderText("R: Reset Position", 20, yPos, 0.8f, glm::vec3(1.0f, 0.5f, 0.5f));
                yPos += lineHeight;
                g_uiRenderer->RenderText("H: Toggle HUD", 20, yPos, 0.8f, glm::vec3(0.8f, 0.8f, 0.8f));
                yPos += lineHeight;
                
                // Status info
                yPos += lineHeight * 0.5f;
                std::string fpsText = "FPS: " + std::to_string((int)currentFPS);
                g_uiRenderer->RenderText(fpsText, 20, yPos, 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));
            }
            
            g_uiRenderer->EndFrame();
        }
        
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
    
    // Cleanup UI
    if (g_uiRenderer) {
        g_uiRenderer->Shutdown();
        g_uiRenderer.reset();
    }
    
    // Cleanup
    CleanupWindow();
    
    return 0;
}
