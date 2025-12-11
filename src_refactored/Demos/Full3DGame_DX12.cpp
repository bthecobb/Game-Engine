#ifdef _WIN32
#include "Core/Coordinator.h"
#include "Gameplay/PlayerComponents.h"
#include "Gameplay/EnemyComponents.h"
#include "Gameplay/LevelComponents.h"
#include "Gameplay/PlayerMovementSystem.h"
#include "Combat/CombatSystem.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/Camera.h"
#include "Rendering/OrbitCamera.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/DX12RenderPipeline.h"
#include "Rendering/D3D12Mesh.h"
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <PxPhysicsAPI.h>
#include <iostream>
#include <chrono>
#include <random>
#include <memory>
#include <unordered_map>
#include <cmath>

using namespace physx;

using namespace CudaGame;
using namespace CudaGame::Rendering;

// Window dimensions
const unsigned int WINDOW_WIDTH = 1920;
const unsigned int WINDOW_HEIGHT = 1080;

// GLFW window
GLFWwindow* window = nullptr;

// Game systems
Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
std::unique_ptr<DX12RenderPipeline> renderPipeline;
std::unique_ptr<OrbitCamera> mainCamera;
std::shared_ptr<Physics::PhysXPhysicsSystem> physicsSystem;

// Entity to mesh mapping for D3D12 rendering
std::unordered_map<Core::Entity, std::unique_ptr<D3D12Mesh>> entityMeshes;

// Explicit list of gameplay entities that should be rendered in the DX12 demo
// (ground, player, and enemies created in CreateGameWorld / SpawnEnemy)
std::vector<Core::Entity> renderEntities;

// Entity to PhysX actor mapping
std::unordered_map<Core::Entity, PxRigidActor*> entityPhysicsActors;

// Player entity
Core::Entity playerEntity = 0;

// Camera controls
float lastX = WINDOW_WIDTH / 2.0f;
float lastY = WINDOW_HEIGHT / 2.0f;
bool firstMouse = true;
bool mouseCaptured = true;  // Start with mouse captured for immediate control

// Input
bool keys[1024] = {false};
bool mouseButtons[8] = {false};

// Debug visualization modes
enum class DebugMode {
    NONE = 0,
    WIREFRAME,
    GBUFFER_POSITION,
    GBUFFER_NORMAL,
    GBUFFER_ALBEDO,
    DEPTH,
    MODE_COUNT
};
DebugMode currentDebugMode = DebugMode::NONE;
bool showFPS = true;  // Show FPS by default
bool showDebugInfo = false;

// Frame timing
float frameTime = 0.0f;
float fps = 0.0f;
int frameCount = 0;
float fpsAccumulator = 0.0f;

// Scroll callback for zoom
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (mainCamera) {
        mainCamera->ApplyZoom(static_cast<float>(yoffset));
    }
}

// Input callbacks
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS)
            keys[key] = true;
        else if (action == GLFW_RELEASE)
            keys[key] = false;
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
    
    // F4 - Cycle debug visualization modes
    if (key == GLFW_KEY_F4 && action == GLFW_PRESS) {
        int mode = static_cast<int>(currentDebugMode);
        mode = (mode + 1) % static_cast<int>(DebugMode::MODE_COUNT);
        currentDebugMode = static_cast<DebugMode>(mode);
        
        const char* modeNames[] = {"None", "Wireframe", "G-Buffer Position", "G-Buffer Normal", "G-Buffer Albedo", "Depth"};
        std::cout << "[Debug] Visualization mode: " << modeNames[mode] << std::endl;
        
        // Update window title to show current debug mode
        char title[128];
        snprintf(title, sizeof(title), "CudaGame DX12 - Debug: %s", modeNames[mode]);
        glfwSetWindowTitle(window, title);
        
        // Update render pipeline debug mode
        if (renderPipeline) {
            renderPipeline->SetDebugMode(static_cast<DX12RenderPipeline::DebugMode>(mode));
        }
    }
    
    // F5 - Quick toggle wireframe
    if (key == GLFW_KEY_F5 && action == GLFW_PRESS) {
        if (currentDebugMode == DebugMode::WIREFRAME) {
            currentDebugMode = DebugMode::NONE;
            glfwSetWindowTitle(window, "CudaGame DX12 - Debug: None");
        } else {
            currentDebugMode = DebugMode::WIREFRAME;
            glfwSetWindowTitle(window, "CudaGame DX12 - Debug: Wireframe");
        }
        if (renderPipeline) {
            renderPipeline->SetDebugMode(static_cast<DX12RenderPipeline::DebugMode>(currentDebugMode));
        }
        std::cout << "[Debug] Wireframe: " << (currentDebugMode == DebugMode::WIREFRAME ? "ON" : "OFF") << std::endl;
    }
    
    // F6 - Toggle FPS display
    if (key == GLFW_KEY_F6 && action == GLFW_PRESS) {
        showFPS = !showFPS;
        std::cout << "[Debug] FPS display: " << (showFPS ? "ON" : "OFF") << std::endl;
    }
    
    // F7 - Toggle debug info
    if (key == GLFW_KEY_F7 && action == GLFW_PRESS) {
        showDebugInfo = !showDebugInfo;
        std::cout << "[Debug] Debug info: " << (showDebugInfo ? "ON" : "OFF") << std::endl;
    }
    
    // ESC - Exit game
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
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
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

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

// Update player input component from GLFW
void UpdateInputComponent(Core::Entity playerEntity) {
    if (!coordinator.HasComponent<Gameplay::PlayerInputComponent>(playerEntity)) return;
    
    auto& input = coordinator.GetComponent<Gameplay::PlayerInputComponent>(playerEntity);
    
    // Keyboard state
    for (int i = 0; i < 1024; ++i) {
        input.keys[i] = (glfwGetKey(window, i) == GLFW_PRESS);
    }
    
    // Mouse buttons
    for (int i = 0; i < 8; ++i) {
        input.mouseButtons[i] = (glfwGetMouseButton(window, i) == GLFW_PRESS);
    }
    
    // Mouse position and delta
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    glm::vec2 newPos(static_cast<float>(xpos), static_cast<float>(ypos));
    input.mouseDelta = newPos - input.mousePos;
    input.mousePos = newPos;
}

// Handle player movement with enhanced features (dash, sprint, etc.)
void HandlePlayerMovement(Core::Entity playerEntity, float deltaTime) {
    if (!coordinator.HasComponent<Gameplay::PlayerInputComponent>(playerEntity)) return;
    if (!coordinator.HasComponent<Gameplay::PlayerMovementComponent>(playerEntity)) return;
    if (entityPhysicsActors.find(playerEntity) == entityPhysicsActors.end()) return;
    
    auto& input = coordinator.GetComponent<Gameplay::PlayerInputComponent>(playerEntity);
    auto& movement = coordinator.GetComponent<Gameplay::PlayerMovementComponent>(playerEntity);
    PxRigidDynamic* playerActor = static_cast<PxRigidDynamic*>(entityPhysicsActors[playerEntity]);
    
    // Update dash cooldown timer
    if (movement.dashCooldownTimer > 0.0f) {
        movement.dashCooldownTimer -= deltaTime;
    }
    
    // Update dash duration timer
    if (movement.isDashing) {
        movement.dashTimer -= deltaTime;
        if (movement.dashTimer <= 0.0f) {
            movement.isDashing = false;
            movement.movementState = Gameplay::MovementState::IDLE;
        }
    }
    
    // Get camera-relative movement direction
    glm::vec3 forward = mainCamera->GetForward();
    glm::vec3 right = mainCamera->GetRight();
    
    // Flatten to XZ plane for ground-based movement
    forward.y = 0.0f;
    if (glm::length(forward) > 0.01f) forward = glm::normalize(forward);
    right.y = 0.0f;
    if (glm::length(right) > 0.01f) right = glm::normalize(right);
    
    // Calculate movement input direction
    glm::vec3 moveDir(0.0f);
    if (input.keys[GLFW_KEY_W]) moveDir += forward;
    if (input.keys[GLFW_KEY_S]) moveDir -= forward;
    if (input.keys[GLFW_KEY_A]) moveDir -= right;
    if (input.keys[GLFW_KEY_D]) moveDir += right;
    
    bool hasMovementInput = glm::length(moveDir) > 0.01f;
    if (hasMovementInput) {
        moveDir = glm::normalize(moveDir);
    }
    
    // Dash mechanic (SHIFT key)
    if (input.keys[GLFW_KEY_LEFT_SHIFT] && !movement.isDashing && movement.dashCooldownTimer <= 0.0f && hasMovementInput) {
        movement.isDashing = true;
        movement.dashTimer = movement.dashDuration;
        movement.dashCooldownTimer = movement.dashCooldown;
        movement.movementState = Gameplay::MovementState::DASHING;
        
        // Apply dash impulse
        PxVec3 dashImpulse(moveDir.x * movement.dashForce, 0.0f, moveDir.z * movement.dashForce);
        playerActor->addForce(dashImpulse, PxForceMode::eIMPULSE);
        
        std::cout << "[Player] DASH!" << std::endl;
    }
    // Normal movement (not dashing)
    else if (!movement.isDashing && hasMovementInput) {
        float moveForce = 500.0f;
        
        // Sprint modifier (hold SHIFT while moving)
        bool isSprinting = input.keys[GLFW_KEY_LEFT_SHIFT];
        if (isSprinting) {
            moveForce *= 1.5f; // Sprint is 50% faster
            movement.movementState = Gameplay::MovementState::SPRINTING;
        } else {
            movement.movementState = Gameplay::MovementState::WALKING;
        }
        
        playerActor->addForce(PxVec3(moveDir.x * moveForce, 0.0f, moveDir.z * moveForce));
    }
    else if (!movement.isDashing) {
        movement.movementState = Gameplay::MovementState::IDLE;
    }
    
    // Jump (SPACE)
    if (input.keys[GLFW_KEY_SPACE]) {
        PxVec3 vel = playerActor->getLinearVelocity();
        
        // Simple grounded check: vertical velocity near zero
        if (vel.y < 1.0f && vel.y > -1.0f) {
            movement.isGrounded = true;
        } else {
            movement.isGrounded = false;
        }
        
        // Ground jump
        if (movement.isGrounded) {
            playerActor->addForce(PxVec3(0.0f, movement.jumpForce * 400.0f, 0.0f), PxForceMode::eIMPULSE);
            movement.movementState = Gameplay::MovementState::JUMPING;
            movement.canDoubleJump = true; // Reset double jump
            movement.isGrounded = false;
            std::cout << "[Player] JUMP!" << std::endl;
        }
        // Double jump
        else if (movement.canDoubleJump && vel.y < 5.0f) {
            playerActor->addForce(PxVec3(0.0f, movement.jumpForce * 350.0f, 0.0f), PxForceMode::eIMPULSE);
            movement.canDoubleJump = false;
            std::cout << "[Player] DOUBLE JUMP!" << std::endl;
        }
    }
}

bool InitializeWindow() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // Tell GLFW not to create OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // Create window
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CudaGame - D3D12 Full 3D", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    // Set callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    
    // Start with mouse captured
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    std::cout << "Window created successfully (mouse captured)" << std::endl;
    return true;
}

// Spawn an enemy at a given position
Core::Entity SpawnEnemy(const glm::vec3& position, const glm::vec3& color) {
    auto enemy = coordinator.CreateEntity();
    
    // Transform and visual
    coordinator.AddComponent(enemy, Rendering::TransformComponent{
        position,
        glm::vec3(0.0f),
        glm::vec3(1.5f, 3.0f, 1.5f)  // Much bigger so they're visible
    });
    coordinator.AddComponent(enemy, Rendering::MaterialComponent{
        color,
        0.1f, 0.7f, 1.0f
    });
    
    // Gameplay components
    Gameplay::EnemyAIComponent aiComp;
    aiComp.aiState = Gameplay::AIState::PATROL;
    aiComp.detectionRange = 25.0f;
    aiComp.attackRange = 3.5f;
    aiComp.patrolSpeed = 5.0f;
    // Add some patrol points in a circle around spawn
    aiComp.patrolPoints.push_back(position + glm::vec3(5.0f, 0.0f, 0.0f));
    aiComp.patrolPoints.push_back(position + glm::vec3(0.0f, 0.0f, 5.0f));
    aiComp.patrolPoints.push_back(position + glm::vec3(-5.0f, 0.0f, 0.0f));
    aiComp.patrolPoints.push_back(position + glm::vec3(0.0f, 0.0f, -5.0f));
    coordinator.AddComponent(enemy, aiComp);
    
    Gameplay::EnemyCombatComponent combatComp;
    combatComp.health = 75.0f;
    combatComp.maxHealth = 75.0f;
    combatComp.damage = 15.0f;
    combatComp.attackCooldown = 1.5f;
    coordinator.AddComponent(enemy, combatComp);
    
    Gameplay::EnemyMovementComponent movementComp;
    movementComp.speed = 10.0f;
    movementComp.maxSpeed = 15.0f;
    coordinator.AddComponent(enemy, movementComp);
    
    Gameplay::TargetingComponent targetComp;
    targetComp.targetEntity = playerEntity;
    targetComp.lockOnRange = 30.0f;
    targetComp.loseTargetRange = 50.0f;
    coordinator.AddComponent(enemy, targetComp);
    
    // Physics body
    if (physicsSystem) {
        using namespace physx;
        PxPhysics* physics = physicsSystem->GetPhysics();
        PxScene* scene = physicsSystem->GetScene();
        
        PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.1f);
        PxRigidDynamic* enemyActor = physics->createRigidDynamic(PxTransform(PxVec3(position.x, position.y, position.z)));
        PxShape* shape = physics->createShape(PxCapsuleGeometry(0.4f, 0.6f), *material);
        enemyActor->attachShape(*shape);
        shape->release();
        
        PxRigidBodyExt::updateMassAndInertia(*enemyActor, 70.0f); // 70kg
        enemyActor->setAngularDamping(0.5f);
        enemyActor->setLinearDamping(0.2f);
        
        // Lock rotations like player
        enemyActor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X, true);
        enemyActor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z, true);
        
        scene->addActor(*enemyActor);
        entityPhysicsActors[enemy] = enemyActor;
    }
    
    return enemy;
}

void CreateGameWorld() {
    std::cout << "Creating 3D game world with ECS and PhysX..." << std::endl;
    
    // Create physics ground plane (static)
    auto ground = coordinator.CreateEntity();
    renderEntities.push_back(ground);
    coordinator.AddComponent(ground, Rendering::TransformComponent{
        glm::vec3(0.0f, -1.0f, 0.0f),
        glm::vec3(0.0f),
        glm::vec3(50.0f, 1.0f, 50.0f)  // Large ground
    });
    coordinator.AddComponent(ground, Rendering::MaterialComponent{
        glm::vec3(0.3f, 0.35f, 0.3f), // Dark green ground
        0.0f, 0.85f, 1.0f
    });
    
    // Add static box collider for ground
    if (physicsSystem) {
        using namespace physx;
        PxPhysics* physics = physicsSystem->GetPhysics();
        PxScene* scene = physicsSystem->GetScene();
        
        // Create static ground plane
        PxMaterial* groundMaterial = physics->createMaterial(0.5f, 0.5f, 0.1f); // friction, friction, restitution
        PxRigidStatic* groundActor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -1.5f, 0.0f)));
        PxShape* groundShape = physics->createShape(PxBoxGeometry(25.0f, 0.5f, 25.0f), *groundMaterial);
        groundActor->attachShape(*groundShape);
        groundShape->release();
        scene->addActor(*groundActor);
        entityPhysicsActors[ground] = groundActor;
        
        std::cout << "  - Physics ground plane created" << std::endl;
    }
    
    // Create player entity AT GROUND LEVEL
    playerEntity = coordinator.CreateEntity();
    renderEntities.push_back(playerEntity);
    coordinator.AddComponent(playerEntity, Rendering::TransformComponent{
        glm::vec3(0.0f, 1.5f, 0.0f),  // At ground level (half height above ground)
        glm::vec3(0.0f),
        glm::vec3(2.0f, 3.0f, 2.0f)  // Much bigger so it's visible
    });
    coordinator.AddComponent(playerEntity, Rendering::MaterialComponent{
        glm::vec3(0.2f, 0.6f, 0.9f), // Blue player
        0.2f, 0.4f, 1.0f
    });
    
    // Add gameplay components
    coordinator.AddComponent(playerEntity, Gameplay::PlayerMovementComponent{});
    coordinator.AddComponent(playerEntity, Gameplay::PlayerCombatComponent{});
    coordinator.AddComponent(playerEntity, Gameplay::PlayerInputComponent{});
    coordinator.AddComponent(playerEntity, Gameplay::PlayerRhythmComponent{});
    coordinator.AddComponent(playerEntity, Gameplay::GrapplingHookComponent{});
    
    // Add dynamic physics body for player
    if (physicsSystem) {
        using namespace physx;
        PxPhysics* physics = physicsSystem->GetPhysics();
        PxScene* scene = physicsSystem->GetScene();
        
        PxMaterial* playerMaterial = physics->createMaterial(0.5f, 0.5f, 0.0f);
        PxRigidDynamic* playerActor = physics->createRigidDynamic(PxTransform(PxVec3(0.0f, 5.0f, 0.0f)));
        PxShape* playerShape = physics->createShape(PxCapsuleGeometry(0.5f, 0.5f), *playerMaterial);
        playerActor->attachShape(*playerShape);
        playerShape->release();
        
        // Set mass and physics properties
        PxRigidBodyExt::updateMassAndInertia(*playerActor, 80.0f); // 80kg player
        playerActor->setAngularDamping(0.5f);
        playerActor->setLinearDamping(0.1f);
        playerActor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, false); // Dynamic
        
        // Lock rotation on X and Z axes (prevent falling over)
        playerActor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X, true);
        playerActor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z, true);
        
        scene->addActor(*playerActor);
        entityPhysicsActors[playerEntity] = playerActor;
        
        std::cout << "  - Player with physics created" << std::endl;
    }
    
    // Spawn enemies in a circle around player AT GROUND LEVEL
    std::cout << "  - Spawning enemies..." << std::endl;
    float enemyRadius = 10.0f;  // Closer to player
    int numEnemies = 4;
    for (int i = 0; i < numEnemies; ++i) {
        float angle = (i / static_cast<float>(numEnemies)) * 2.0f * 3.14159f;
        glm::vec3 spawnPos(
            enemyRadius * cos(angle),
            1.5f,  // At ground level (half height above ground)
            enemyRadius * sin(angle)
        );
        
        // Different colors for each enemy
        glm::vec3 enemyColor;
        switch(i % 4) {
            case 0: enemyColor = glm::vec3(0.9f, 0.2f, 0.2f); break; // Red
            case 1: enemyColor = glm::vec3(0.9f, 0.5f, 0.1f); break; // Orange
            case 2: enemyColor = glm::vec3(0.7f, 0.2f, 0.7f); break; // Purple
            case 3: enemyColor = glm::vec3(0.2f, 0.9f, 0.2f); break; // Green
        }
        
        Core::Entity enemyEntity = SpawnEnemy(spawnPos, enemyColor);
        renderEntities.push_back(enemyEntity);
        std::cout << "    Enemy " << (i+1) << " spawned at (" << spawnPos.x << ", " << spawnPos.y << ", " << spawnPos.z << ")" << std::endl;
    }
    
    std::cout << "World created with " << (1 + 1 + numEnemies) << " entities (1 ground, 1 player, " << numEnemies << " enemies)" << std::endl;
}

// Sync ECS entities with D3D12 rendering
void SyncEntitiesToRenderer() {
    // Clear existing meshes from pipeline
    renderPipeline->ClearMeshes();
    
    // Iterate through the explicit list of world entities we created for this demo
    // This avoids assumptions about ECS entity ID ranges and ensures we sync exactly
    // the ground, player, and enemies we spawned.
    for (Core::Entity entity : renderEntities) {
        if (!coordinator.HasComponent<Rendering::TransformComponent>(entity)) continue;
        if (!coordinator.HasComponent<Rendering::MaterialComponent>(entity)) continue;
        
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        auto& material = coordinator.GetComponent<Rendering::MaterialComponent>(entity);
        
        // Create or update D3D12 mesh for this entity
        if (entityMeshes.find(entity) == entityMeshes.end()) {
            // Create new mesh based on entity scale
            std::unique_ptr<D3D12Mesh> mesh;
            
            // Ground plane: very flat and wide
            if (transform.scale.y < 2.0f && (transform.scale.x > 10.0f || transform.scale.z > 10.0f)) {
                mesh = MeshGenerator::CreatePlane(renderPipeline->GetBackend());
                std::cout << "[Mesh] Created PLANE for entity " << entity << std::endl;
            }
            // Tall entities (player, enemies) - use cube for now (will be capsule visual later)
            else if (transform.scale.y > transform.scale.x * 1.5f) {
                mesh = MeshGenerator::CreateCube(renderPipeline->GetBackend());
                std::cout << "[Mesh] Created CUBE (tall) for entity " << entity << std::endl;
            }
            // Everything else
            else {
                mesh = MeshGenerator::CreateCube(renderPipeline->GetBackend());
                std::cout << "[Mesh] Created CUBE for entity " << entity << std::endl;
            }
            
            if (mesh) {
                entityMeshes[entity] = std::move(mesh);
            }
        }
        
        // Update mesh transform and material from ECS
        if (entityMeshes[entity]) {
            entityMeshes[entity]->transform = transform.getMatrix();
            entityMeshes[entity]->GetMaterial().albedoColor = glm::vec4(material.albedo, 1.0f);
            entityMeshes[entity]->GetMaterial().metallic = material.metallic;
            entityMeshes[entity]->GetMaterial().roughness = material.roughness;
            entityMeshes[entity]->GetMaterial().ambientOcclusion = material.ao;
            
            // Add to render pipeline
            renderPipeline->AddMesh(entityMeshes[entity].get());
        }
    }
}

int main() {
    std::cout << "=== CudaGame - D3D12 Full 3D Demo ===" << std::endl;
    std::cout << "Features: PhysX Physics, D3D12 Rendering, ECS Architecture" << std::endl;
    std::cout << std::endl;

    // Initialize window
    if (!InitializeWindow()) {
        return -1;
    }
    
    // Initialize ECS
    coordinator.Initialize();
    
    // Register rendering components
    coordinator.RegisterComponent<Rendering::TransformComponent>();
    coordinator.RegisterComponent<Rendering::MaterialComponent>();
    
    // Register gameplay components
    coordinator.RegisterComponent<Gameplay::PlayerMovementComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerCombatComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerInputComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerRhythmComponent>();
    coordinator.RegisterComponent<Gameplay::GrapplingHookComponent>();
    
    // Register enemy components
    coordinator.RegisterComponent<Gameplay::EnemyAIComponent>();
    coordinator.RegisterComponent<Gameplay::EnemyCombatComponent>();
    coordinator.RegisterComponent<Gameplay::EnemyMovementComponent>();
    coordinator.RegisterComponent<Gameplay::TargetingComponent>();
    
    std::cout << "ECS initialized with all gameplay components" << std::endl;
    
    // Initialize PhysX
    physicsSystem = std::make_shared<Physics::PhysXPhysicsSystem>();
    if (!physicsSystem->Initialize()) {
        std::cerr << "Failed to initialize PhysX" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    std::cout << "PhysX physics system initialized" << std::endl;

    // Initialize orbit camera similarly to EnhancedGameMain_Full3D so behavior matches
    // the working OpenGL renderer.
    mainCamera = std::make_unique<OrbitCamera>(ProjectionType::PERSPECTIVE);

    OrbitCamera::OrbitSettings orbitSettings;
    orbitSettings.distance         = 4.5f;
    orbitSettings.heightOffset     = 0.0f;      // player height handled via target
    orbitSettings.mouseSensitivity = 0.03f;     // less twitchy
    orbitSettings.smoothSpeed      = 12.0f;
    orbitSettings.minDistance      = 3.0f;
    orbitSettings.maxDistance      = 12.0f;
    mainCamera->SetOrbitSettings(orbitSettings);

    mainCamera->SetPerspective(
        60.0f,
        static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT),
        0.1f,
        200.0f
    );

    // Start in free-look mode around the player, like the GL demo.
    mainCamera->SetCameraMode(OrbitCamera::CameraMode::FREE_LOOK);

    glm::vec3 initialTarget(0.0f, 1.5f, 0.0f); // player height at world origin
    mainCamera->SetTarget(initialTarget);
    mainCamera->SetDistance(4.5f, true);
    mainCamera->Update(0.016f, initialTarget, glm::vec3(0.0f));

    std::cout << "Camera initialized (Orbit FREE_LOOK around player)" << std::endl;

    // Initialize D3D12 rendering pipeline
    renderPipeline = std::make_unique<DX12RenderPipeline>();
    DX12RenderPipeline::InitParams renderParams = {};
    renderParams.windowHandle = window;
    renderParams.displayWidth = WINDOW_WIDTH;
    renderParams.displayHeight = WINDOW_HEIGHT;
    renderParams.enableDLSS = false;
    renderParams.enableRayTracing = false;

    if (!renderPipeline->Initialize(renderParams)) {
        std::cerr << "Failed to initialize D3D12 pipeline" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    std::cout << "D3D12 pipeline initialized" << std::endl;

    // Create game world with ECS entities
    CreateGameWorld();
    
    // Initial sync of entities to renderer
    SyncEntitiesToRenderer();
    std::cout << "Game world synced to renderer: " << renderPipeline->GetMeshCount() << " meshes" << std::endl;
    
    // (Debug-only initial entity dump removed to keep DX12 output clean.)

    // Main game loop
    std::cout << "Entering main loop (Press ESC to exit, TAB for mouse capture)..." << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    uint32_t frameCount = 0;
    float fps = 0.0f;
    float time = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Exit on ESC
        if (keys[GLFW_KEY_ESCAPE]) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Update time
        float deltaTime = 0.016f; // ~60 FPS
        time += deltaTime;
        
        // Update input component from GLFW
        UpdateInputComponent(playerEntity);
        
        // Handle player movement (dash, sprint, jump, WASD)
        if (mouseCaptured) {
            HandlePlayerMovement(playerEntity, deltaTime);
        }
        
        // Camera mode toggles (match GL renderer behavior: 1=ORBIT_FOLLOW, 2=FREE_LOOK, 3=COMBAT_FOCUS)
        if (mainCamera) {
            static bool key1Prev = false;
            static bool key2Prev = false;
            static bool key3Prev = false;

            bool key1 = keys[GLFW_KEY_1] != 0;
            bool key2 = keys[GLFW_KEY_2] != 0;
            bool key3 = keys[GLFW_KEY_3] != 0;

            if (key1 && !key1Prev) {
                mainCamera->SetCameraMode(OrbitCamera::CameraMode::ORBIT_FOLLOW);
            }
            if (key2 && !key2Prev) {
                mainCamera->SetCameraMode(OrbitCamera::CameraMode::FREE_LOOK);
            }
            if (key3 && !key3Prev) {
                mainCamera->SetCameraMode(OrbitCamera::CameraMode::COMBAT_FOCUS);
            }

            key1Prev = key1;
            key2Prev = key2;
            key3Prev = key3;
        }
        
        // Physics update (re-enabled now that rendering is stable)
        if (physicsSystem) {
            physicsSystem->Update(deltaTime);
        }
        
        // Update orbit camera to follow the player, similar to the GL renderer.
        glm::vec3 playerPos(0.0f);
        glm::vec3 playerVel(0.0f);
        if (coordinator.HasComponent<Rendering::TransformComponent>(playerEntity)) {
            playerPos = coordinator.GetComponent<Rendering::TransformComponent>(playerEntity).position;
        }
        auto physIt = entityPhysicsActors.find(playerEntity);
        if (physIt != entityPhysicsActors.end()) {
            PxRigidActor* actor = physIt->second;
            if (actor) {
                if (PxRigidDynamic* dyn = static_cast<PxRigidDynamic*>(actor)) {
                    PxVec3 v = dyn->getLinearVelocity();
                    playerVel = glm::vec3(v.x, v.y, v.z);
                }
            }
        }
        if (mainCamera) {
            mainCamera->Update(deltaTime, playerPos, playerVel);
        }
        
        // Sync ECS entities to renderer (in full game, only do this when entities change)
        SyncEntitiesToRenderer();

        // Render frame
        renderPipeline->BeginFrame(mainCamera.get());
        renderPipeline->RenderFrame();
        renderPipeline->EndFrame();

        // Calculate FPS
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(currentTime - startTime).count();

        if (elapsed >= 1.0f) {
            fps = frameCount / elapsed;
            
            auto stats = renderPipeline->GetFrameStats();
            std::cout << "[Game] FPS: " << static_cast<int>(fps)
                      << " | Draw Calls: " << stats.drawCalls
                      << " | Triangles: " << stats.triangles
                      << " | Frame: " << stats.totalFrameMs << "ms" << std::endl;

            frameCount = 0;
            startTime = currentTime;
        }
    }

    std::cout << "Shutting down..." << std::endl;

    // Cleanup
    entityMeshes.clear();
    
    if (physicsSystem) {
        physicsSystem->Shutdown();
        physicsSystem.reset();
    }
    
    renderPipeline->Shutdown();
    renderPipeline.reset();

    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "Shutdown complete" << std::endl;
    return 0;
}

#else
#include <iostream>

int main() {
    std::cerr << "D3D12 Game is only available on Windows" << std::endl;
    return -1;
}
#endif // _WIN32
