#ifdef _WIN32
#include "Core/Coordinator.h"
#include "Gameplay/PlayerComponents.h"
#include "Gameplay/EnemyComponents.h"
#include "Gameplay/LevelComponents.h"
#include "Gameplay/PlayerMovementSystem.h"
#include "Gameplay/CharacterControllerSystem.h"
#include "Gameplay/EnemyAISystem.h"
#include "Gameplay/LevelSystem.h"
#include "Gameplay/TargetingSystem.h"
#include "Combat/CombatSystem.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Physics/WallRunningSystem.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/Camera.h"
#include "Rendering/OrbitCamera.h"
#include "Rendering/ThirdPersonCameraRig.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/DX12RenderPipeline.h"
#include "Rendering/D3D12Mesh.h"
#include "Rendering/CudaBuildingGenerator.h"
#include "World/WorldChunkManager.h"
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
#include <fstream>

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
ThirdPersonCameraRig thirdPersonRig;      // Third-person camera rig (used for ORBIT_FOLLOW/COMBAT_FOCUS modes)
std::shared_ptr<Physics::PhysXPhysicsSystem> physicsSystem;
std::shared_ptr<Gameplay::CharacterControllerSystem> characterControllerSystem;
std::shared_ptr<Gameplay::EnemyAISystem> enemyAISystem;
std::shared_ptr<Gameplay::LevelSystem> levelSystem;
std::shared_ptr<Gameplay::TargetingSystem> targetingSystem;
std::shared_ptr<Gameplay::PlayerMovementSystem> playerMovementSystem;
std::shared_ptr<Physics::WallRunningSystem> wallRunningSystem;

// Entity to mesh mapping for D3D12 rendering
std::unordered_map<Core::Entity, std::unique_ptr<D3D12Mesh>> entityMeshes;

// Procedural building generator
std::unique_ptr<CudaBuildingGenerator> buildingGenerator;
std::vector<BuildingMesh> generatedBuildingMeshes;  // Keep meshes alive
std::unordered_map<Core::Entity, size_t> buildingEntityToMeshIndex;  // Map building entities to mesh index

// World chunk streaming manager
std::unique_ptr<World::WorldChunkManager> chunkManager;
uint64_t frameNumber = 0;

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

// Camera tuning parameters (tuned to reduce jitter)
float g_cameraSmoothSpeed = 0.5f;     // Tuned: rotation smoothing
float g_cameraSensitivity = 0.05f;    // Tuned by user (was 0.01)
bool g_cameraLerp = true;             // F12 to toggle lerp vs instant
float g_cameraLag = 0.0f;             // Accumulated lag for debugging

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

// Forward declarations for character mesh
class DX12RenderBackend;

// Helper to add a box to procedural character mesh (uses Vertex struct from D3D12Mesh.h)
void AddCharacterBox(std::vector<Rendering::Vertex>& vertices, std::vector<uint32_t>& indices,
                     glm::vec3 center, glm::vec3 size) {
    uint32_t baseIdx = static_cast<uint32_t>(vertices.size());
    
    float hw = size.x * 0.5f;
    float hh = size.y * 0.5f;
    float hd = size.z * 0.5f;
    
    // 24 vertices (6 faces * 4 vertices)
    glm::vec3 positions[24] = {
        // Front face (+Z)
        {-hw, -hh, hd}, {hw, -hh, hd}, {hw, hh, hd}, {-hw, hh, hd},
        // Back face (-Z)
        {hw, -hh, -hd}, {-hw, -hh, -hd}, {-hw, hh, -hd}, {hw, hh, -hd},
        // Right face (+X)
        {hw, -hh, hd}, {hw, -hh, -hd}, {hw, hh, -hd}, {hw, hh, hd},
        // Left face (-X)
        {-hw, -hh, -hd}, {-hw, -hh, hd}, {-hw, hh, hd}, {-hw, hh, -hd},
        // Top face (+Y)
        {-hw, hh, hd}, {hw, hh, hd}, {hw, hh, -hd}, {-hw, hh, -hd},
        // Bottom face (-Y)
        {-hw, -hh, -hd}, {hw, -hh, -hd}, {hw, -hh, hd}, {-hw, -hh, hd}
    };
    
    glm::vec3 faceNormals[6] = {
        {0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}
    };
    
    // Add vertices
    for (int face = 0; face < 6; face++) {
        for (int v = 0; v < 4; v++) {
            Rendering::Vertex vert;
            vert.position = center + positions[face * 4 + v];
            vert.normal = faceNormals[face];
            vert.tangent = glm::vec3(1, 0, 0);  // Simplified tangent
            vert.texcoord = glm::vec2(0, 0);    // No UVs needed
            vertices.push_back(vert);
        }
    }
    
    // Add indices (2 triangles per face)
    for (int face = 0; face < 6; face++) {
        uint32_t base = baseIdx + face * 4;
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
}

// Create procedural low-poly character mesh for DX12
std::unique_ptr<D3D12Mesh> CreateProceduralCharacterMesh(Rendering::DX12RenderBackend* backend) {
    std::vector<Rendering::Vertex> vertices;
    std::vector<uint32_t> indices;
    
    // Body proportions (scaled for ~2.0 unit tall character)
    float headSize = 0.4f;
    float headY = 1.65f;
    
    // Build character body parts (color set via material, not vertex)
    // 1. Head
    AddCharacterBox(vertices, indices, glm::vec3(0, headY, 0), glm::vec3(headSize));
    
    // 2. Hair (flat box on top)
    AddCharacterBox(vertices, indices, glm::vec3(0, headY + headSize * 0.55f, 0),
                   glm::vec3(headSize * 1.1f, 0.15f, headSize * 1.1f));
    
    // 3. Eyes (two small cubes)
    float eyeSize = 0.08f;
    float eyeOffset = headSize * 0.3f;
    AddCharacterBox(vertices, indices, glm::vec3(-eyeOffset, headY + 0.08f, headSize * 0.52f),
                   glm::vec3(eyeSize, eyeSize, eyeSize * 0.5f));
    AddCharacterBox(vertices, indices, glm::vec3(eyeOffset, headY + 0.08f, headSize * 0.52f),
                   glm::vec3(eyeSize, eyeSize, eyeSize * 0.5f));
    
    // 4. Torso
    AddCharacterBox(vertices, indices, glm::vec3(0, 1.05f, 0), glm::vec3(0.6f, 0.8f, 0.35f));
    
    // 5. Arms
    AddCharacterBox(vertices, indices, glm::vec3(-0.45f, 1.2f, 0), glm::vec3(0.18f, 0.65f, 0.18f));
    AddCharacterBox(vertices, indices, glm::vec3(0.45f, 1.2f, 0), glm::vec3(0.18f, 0.65f, 0.18f));
    
    // 6. Legs
    AddCharacterBox(vertices, indices, glm::vec3(-0.18f, 0.425f, 0), glm::vec3(0.22f, 0.85f, 0.22f));
    AddCharacterBox(vertices, indices, glm::vec3(0.18f, 0.425f, 0), glm::vec3(0.22f, 0.85f, 0.22f));
    
    // 7. Shoes
    AddCharacterBox(vertices, indices, glm::vec3(-0.18f, 0.06f, 0.08f), glm::vec3(0.22f, 0.12f, 0.33f));
    AddCharacterBox(vertices, indices, glm::vec3(0.18f, 0.06f, 0.08f), glm::vec3(0.22f, 0.12f, 0.33f));
    
    // Create D3D12Mesh from vertex/index data
    auto mesh = std::make_unique<D3D12Mesh>();
    mesh->Create(backend, vertices, indices, "PlayerCharacter");
    
    // Set character material (blue jacket color for visibility)
    mesh->GetMaterial().albedoColor = glm::vec4(0.2f, 0.5f, 0.95f, 1.0f);
    mesh->GetMaterial().roughness = 0.6f;
    mesh->GetMaterial().metallic = 0.1f;
    
    return mesh;
}

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
    
    // ===== CAMERA TUNING CONTROLS =====
    // F8 - Decrease camera smooth speed
    if (key == GLFW_KEY_F8 && action == GLFW_PRESS) {
        g_cameraSmoothSpeed = std::max(0.5f, g_cameraSmoothSpeed - 1.0f);
        std::cout << "[Camera] SmoothSpeed: " << g_cameraSmoothSpeed << std::endl;
        if (mainCamera) mainCamera->SetSmoothSpeed(g_cameraSmoothSpeed);
    }
    // F9 - Increase camera smooth speed
    if (key == GLFW_KEY_F9 && action == GLFW_PRESS) {
        g_cameraSmoothSpeed = std::min(30.0f, g_cameraSmoothSpeed + 1.0f);
        std::cout << "[Camera] SmoothSpeed: " << g_cameraSmoothSpeed << std::endl;
        if (mainCamera) mainCamera->SetSmoothSpeed(g_cameraSmoothSpeed);
    }
    // F10 - Decrease mouse sensitivity
    if (key == GLFW_KEY_F10 && action == GLFW_PRESS) {
        g_cameraSensitivity = std::max(0.01f, g_cameraSensitivity - 0.02f);
        std::cout << "[Camera] Sensitivity: " << g_cameraSensitivity << std::endl;
        if (mainCamera) mainCamera->SetMouseSensitivity(g_cameraSensitivity);
    }
    // F11 - Increase mouse sensitivity  
    if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {
        g_cameraSensitivity = std::min(1.0f, g_cameraSensitivity + 0.02f);
        std::cout << "[Camera] Sensitivity: " << g_cameraSensitivity << std::endl;
        if (mainCamera) mainCamera->SetMouseSensitivity(g_cameraSensitivity);
    }
    // F12 - Toggle camera lerp (smooth vs instant)
    if (key == GLFW_KEY_F12 && action == GLFW_PRESS) {
        g_cameraLerp = !g_cameraLerp;
        std::cout << "[Camera] Lerp: " << (g_cameraLerp ? "ON (smooth)" : "OFF (instant)") << std::endl;
        if (mainCamera) mainCamera->SetSmoothSpeed(g_cameraLerp ? g_cameraSmoothSpeed : 1000.0f);
    }
    
    // P - Save camera settings to file
    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        std::cout << "\n========== CAMERA SETTINGS SAVED ==========" << std::endl;
        std::cout << "SmoothSpeed: " << g_cameraSmoothSpeed << std::endl;
        std::cout << "Sensitivity: " << g_cameraSensitivity << std::endl;
        std::cout << "Lerp: " << (g_cameraLerp ? "true" : "false") << std::endl;
        std::cout << "============================================\n" << std::endl;
        
        // Save to file
        std::ofstream configFile("camera_config.txt");
        if (configFile.is_open()) {
            configFile << "# Camera Settings - Copy these values to code\n";
            configFile << "g_cameraSmoothSpeed = " << g_cameraSmoothSpeed << "f;\n";
            configFile << "g_cameraSensitivity = " << g_cameraSensitivity << "f;\n";
            configFile << "g_cameraLerp = " << (g_cameraLerp ? "true" : "false") << ";\n";
            configFile.close();
            std::cout << "[Camera] Settings saved to camera_config.txt" << std::endl;
        }
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
        return; // Skip first frame to avoid large delta
    }

    float xoffset = static_cast<float>(xpos - lastX);
    float yoffset = static_cast<float>(lastY - ypos);
    lastX = xpos;
    lastY = ypos;
    
    // Mouse input smoothing to reduce jitter (exponential moving average)
    static float smoothedX = 0.0f;
    static float smoothedY = 0.0f;
    const float mouseSmooth = 0.8f;  // Higher = more smoothing (was 0.5)
    
    smoothedX = smoothedX * mouseSmooth + xoffset * (1.0f - mouseSmooth);
    smoothedY = smoothedY * mouseSmooth + yoffset * (1.0f - mouseSmooth);
    
    // Apply deadzone to filter micro-movements
    const float deadzone = 0.3f;  // Higher threshold (was 0.1)
    if (std::abs(smoothedX) < deadzone) smoothedX = 0.0f;
    if (std::abs(smoothedY) < deadzone) smoothedY = 0.0f;
    
    mainCamera->ApplyMouseDelta(smoothedX, smoothedY);
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
        // Use setLinearVelocity for immediate, responsive movement
        // (addForce was causing momentum accumulation that fights with camera direction changes)
        float moveSpeed = 8.0f;  // meters per second
        
        // Sprint modifier (hold SHIFT while moving)
        bool isSprinting = input.keys[GLFW_KEY_LEFT_SHIFT];
        if (isSprinting) {
            moveSpeed *= 1.5f; // Sprint is 50% faster
            movement.movementState = Gameplay::MovementState::SPRINTING;
        } else {
            movement.movementState = Gameplay::MovementState::WALKING;
        }
        
        // Get current velocity to preserve vertical component (gravity/jumping)
        PxVec3 currentVel = playerActor->getLinearVelocity();
        
        // Set horizontal velocity directly to match camera-relative input
        PxVec3 newVel(moveDir.x * moveSpeed, currentVel.y, moveDir.z * moveSpeed);
        playerActor->setLinearVelocity(newVel);
    }
    else if (!movement.isDashing) {
        // When no input, apply friction to stop horizontal movement
        PxVec3 currentVel = playerActor->getLinearVelocity();
        PxVec3 newVel(currentVel.x * 0.9f, currentVel.y, currentVel.z * 0.9f);  // Friction
        playerActor->setLinearVelocity(newVel);
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

// ====== CHUNK-BASED BUILDING GENERATION ======
// Generates buildings for a specific chunk using deterministic seeding
void GenerateBuildingsForChunk(World::WorldChunk& chunk) {
    using namespace World;
    
    // Deterministic random generator based on chunk seed
    std::mt19937 gen(chunk.seed);
    
    // Buildings per chunk based on LOD
    int buildingsToGenerate = 0;
    switch (chunk.lodLevel) {
        case ChunkLOD::HIGH:   buildingsToGenerate = 15; break;
        case ChunkLOD::MEDIUM: buildingsToGenerate = 8;  break;
        case ChunkLOD::LOW:    buildingsToGenerate = 3;  break;
    }
    
    // Position distribution within chunk bounds
    std::uniform_real_distribution<float> xDist(chunk.bounds.min.x + 20.0f, chunk.bounds.max.x - 20.0f);
    std::uniform_real_distribution<float> zDist(chunk.bounds.min.z + 20.0f, chunk.bounds.max.z - 20.0f);
    std::uniform_real_distribution<float> heightDist(15.0f, 70.0f);
    std::uniform_real_distribution<float> widthDist(8.0f, 16.0f);
    
    for (int i = 0; i < buildingsToGenerate; ++i) {
        float x = xDist(gen);
        float z = zDist(gen);
        float height = heightDist(gen);
        float width = widthDist(gen);
        float depth = widthDist(gen);
        
        // Configure building style
        BuildingStyle style;
        style.baseWidth = width;
        style.baseDepth = depth;
        style.height = height;
        style.seed = chunk.seed + static_cast<uint32_t>(i * 12345);
        style.baseColor = glm::vec3(0.45f + (i % 10) * 0.03f, 0.5f + (i % 7) * 0.02f, 0.55f + (i % 5) * 0.03f);
        style.accentColor = glm::vec3(0.2f, 0.25f, 0.3f);
        
        // Generate building mesh
        if (buildingGenerator) {
            Rendering::BuildingMesh mesh = buildingGenerator->GenerateBuilding(style);
            size_t meshIndex = generatedBuildingMeshes.size();
            generatedBuildingMeshes.push_back(mesh);
            
            // Create building entity
            Core::Entity building = coordinator.CreateEntity();
            renderEntities.push_back(building);
            chunk.entities.push_back(building);
            chunk.meshIndices.push_back(meshIndex);
            
            buildingEntityToMeshIndex[building] = meshIndex;
            
            coordinator.AddComponent(building, Rendering::TransformComponent{
                glm::vec3(x, -0.5f, z),
                glm::vec3(0.0f),
                glm::vec3(1.0f)
            });
            
            coordinator.AddComponent(building, Rendering::MaterialComponent{
                glm::vec3(1.0f), 0.3f, 0.6f, 1.0f
            });
            
            Gameplay::WallComponent wallComp;
            wallComp.canWallRun = true;
            coordinator.AddComponent(building, wallComp);
            
            Physics::ColliderComponent buildingCollider{};
            buildingCollider.shape = Physics::ColliderShape::BOX;
            buildingCollider.halfExtents = glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f);
            coordinator.AddComponent(building, buildingCollider);
        }
    }
    
    chunk.buildingCounts[static_cast<int>(chunk.lodLevel)] = buildingsToGenerate;
}

void CreateGameWorld() {
    std::cout << "Creating 3D game world (EXPANDED 10000x10000)..." << std::endl;
    
    // ====== GROUND (10000x10000 - EXPANDED for larger world) ======
    auto ground = coordinator.CreateEntity();
    renderEntities.push_back(ground);
    coordinator.AddComponent(ground, Rendering::TransformComponent{
        glm::vec3(0.0f, -1.0f, 0.0f),
        glm::vec3(0.0f),
        glm::vec3(10000.0f, 1.0f, 10000.0f)  // EXPANDED: 10000x10000
    });
    coordinator.AddComponent(ground, Rendering::MaterialComponent{
        glm::vec3(0.25f, 0.25f, 0.28f), // Slightly blue-gray ground
        0.0f, 0.8f, 1.0f
    });
    
    // Ground collider (half extents = 5000)
    Physics::ColliderComponent groundCollider{};
    groundCollider.shape = Physics::ColliderShape::BOX;
    groundCollider.halfExtents = glm::vec3(5000.0f, 0.5f, 5000.0f);
    coordinator.AddComponent(ground, groundCollider);
    
    std::cout << "  - Ground: 10000x10000 (EXPANDED)" << std::endl;
    
    
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
    
    // Add physics/controller components (required for CharacterControllerSystem and PhysXPhysicsSystem)
    Physics::RigidbodyComponent playerRB{};
    playerRB.mass = 80.0f;
    playerRB.inverseMass = 1.0f / 80.0f;
    playerRB.isKinematic = false;
    coordinator.AddComponent(playerEntity, playerRB);
    
    // ColliderComponent is CRITICAL - PhysXPhysicsSystem only processes entities with this component!
    Physics::ColliderComponent playerCollider{};
    playerCollider.shape = Physics::ColliderShape::CAPSULE;
    playerCollider.capsuleRadius = 0.5f;
    playerCollider.capsuleHeight = 1.0f;
    playerCollider.halfExtents = glm::vec3(0.5f, 0.5f, 0.5f);  // Used for capsule half-height
    coordinator.AddComponent(playerEntity, playerCollider);
    
    coordinator.AddComponent(playerEntity, Physics::CharacterControllerComponent{});
    
    std::cout << "  - Player created at origin" << std::endl;
    
    
    // ====== ENEMIES (25, EXPANDED: ±1500 with 100 safe distance) ======
    std::cout << "  - Spawning enemies..." << std::endl;
    std::random_device rd_enemy;
    std::mt19937 gen_enemy(123);  // Fixed seed
    std::uniform_real_distribution<> enemy_pos(-1500.0, 1500.0);
    int numEnemies = 25;  // EXPANDED: 25 enemies in larger world
    
    for (int i = 0; i < numEnemies; ++i) {
        // Random position with minimum 60 unit distance from origin (OpenGL logic)
        float x = static_cast<float>(enemy_pos(gen_enemy));
        float z = static_cast<float>(enemy_pos(gen_enemy));
        const float minDist = 60.0f;
        int safetyIterations = 0;
        while ((x * x + z * z) < (minDist * minDist) && safetyIterations < 10) {
            x = static_cast<float>(enemy_pos(gen_enemy));
            z = static_cast<float>(enemy_pos(gen_enemy));
            safetyIterations++;
        }
        
        glm::vec3 spawnPos(x, 1.0f, z);
        glm::vec3 enemyColor(1.0f, 0.0f, 0.0f); // Red enemies (OpenGL style)
        
        Core::Entity enemyEntity = SpawnEnemy(spawnPos, enemyColor);
        renderEntities.push_back(enemyEntity);
        std::cout << "    Enemy " << (i+1) << " spawned at (" << spawnPos.x << ", " << spawnPos.y << ", " << spawnPos.z << ")" << std::endl;
    }
    
    // ====== BUILDINGS (500 random, EXPANDED for larger world) ======
    std::cout << "  - Generating 500 buildings (EXPANDED)..." << std::endl;
    int buildingCount = 0;
    
    // Use random distribution EXPANDED: ±4500 range
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> pos_dis(-4500.0, 4500.0);
    std::uniform_real_distribution<> height_dis(12.0, 60.0);  // EXPANDED: 12-60 (taller buildings)
    
    // Reserve space for generated building meshes
    generatedBuildingMeshes.reserve(500);
    
    for (int i = 0; i < 500; ++i) {
        float x = static_cast<float>(pos_dis(gen));
        float z = static_cast<float>(pos_dis(gen));
        float height = static_cast<float>(height_dis(gen));
        
        // OpenGL building widths: 6, 8, 10, 12 cycle
        float buildingWidth = 6.0f + (i % 4) * 2.0f;
        float buildingDepth = 6.0f + ((i + 1) % 4) * 2.0f;
        
        // Configure building style for procedural generation
        BuildingStyle style;
        style.baseWidth = buildingWidth;
        style.baseDepth = buildingDepth;
        style.height = height;
        style.seed = static_cast<uint32_t>(i * 12345);  // Unique seed per building
        style.baseColor = glm::vec3(0.5f + (i % 10) * 0.02f, 0.55f + (i % 7) * 0.02f, 0.6f + (i % 5) * 0.02f);
        style.accentColor = glm::vec3(0.2f, 0.25f, 0.3f);
        
        // Generate building mesh (with emissive windows)
        BuildingMesh mesh = buildingGenerator->GenerateBuilding(style);
        generatedBuildingMeshes.push_back(mesh);
        
        // Create building entity
        Core::Entity building = coordinator.CreateEntity();
        renderEntities.push_back(building);
        
        // Map this entity to its mesh index
        buildingEntityToMeshIndex[building] = generatedBuildingMeshes.size() - 1;
        
        coordinator.AddComponent(building, Rendering::TransformComponent{
            glm::vec3(x, -0.5f, z),  // Base at ground level (mesh already has height)
            glm::vec3(0.0f),
            glm::vec3(1.0f)  // No scaling - mesh is already at correct size
        });
        
        // Material: WHITE albedo so vertex colors control final color
        coordinator.AddComponent(building, Rendering::MaterialComponent{
            glm::vec3(1.0f),  // White albedo - vertex colors control output
            0.3f,   // Metallic
            0.6f,   // Roughness
            1.0f    // AO
        });
        
        // Add WallComponent for wall-running (matching OpenGL)
        Gameplay::WallComponent wallComp;
        wallComp.canWallRun = true;
        coordinator.AddComponent(building, wallComp);
        
        // Collider (based on style dimensions)
        Physics::ColliderComponent buildingCollider{};
        buildingCollider.shape = Physics::ColliderShape::BOX;
        buildingCollider.halfExtents = glm::vec3(buildingWidth / 2.0f, height / 2.0f, buildingDepth / 2.0f);
        coordinator.AddComponent(building, buildingCollider);
        
        buildingCount++;
    }
    
    std::cout << "    Generated " << buildingCount << " procedural buildings with emissive windows" << std::endl;
    
    // ====== BOUNDARY WALLS (invisible, matching OpenGL) ======
    std::cout << "  - Creating boundary walls..." << std::endl;
    float groundHalfSize = 1120.0f;  // Slightly inside 2250/2=1125 (OpenGL uses 145 of 150)
    float wallHeight = 5.0f;         // OpenGL: 5 units tall
    float wallThickness = 1.0f;
    
    // Wall positions: +X, -X, +Z, -Z edges
    struct WallDef { glm::vec3 pos; glm::vec3 scale; };
    WallDef walls[4] = {
        { glm::vec3(groundHalfSize, wallHeight/2, 0), glm::vec3(wallThickness, wallHeight, groundHalfSize * 2) },   // +X
        { glm::vec3(-groundHalfSize, wallHeight/2, 0), glm::vec3(wallThickness, wallHeight, groundHalfSize * 2) },  // -X
        { glm::vec3(0, wallHeight/2, groundHalfSize), glm::vec3(groundHalfSize * 2, wallHeight, wallThickness) },   // +Z
        { glm::vec3(0, wallHeight/2, -groundHalfSize), glm::vec3(groundHalfSize * 2, wallHeight, wallThickness) }   // -Z
    };
    
    for (int w = 0; w < 4; ++w) {
        Core::Entity wall = coordinator.CreateEntity();
        // Don't add to renderEntities - keep invisible (collision only)
        
        // Collision only - no visual (matching OpenGL invisible walls)
        Physics::ColliderComponent wallCollider{};
        wallCollider.shape = Physics::ColliderShape::BOX;
        wallCollider.halfExtents = walls[w].scale * 0.5f;
        coordinator.AddComponent(wall, wallCollider);
        
        coordinator.AddComponent(wall, Rendering::TransformComponent{
            walls[w].pos,
            glm::vec3(0.0f),
            walls[w].scale
        });
    }
    std::cout << "    Created 4 invisible boundary walls at ±1120" << std::endl;
    
    // ====== COLLECTIBLES (20, matching OpenGL) ======
    std::cout << "  - Creating collectibles..." << std::endl;
    int collectibleCount = 0;
    for (int i = 0; i < 20; ++i) {
        Core::Entity collectible = coordinator.CreateEntity();
        renderEntities.push_back(collectible);
        
        // CollectibleComponent (health pack)
        Gameplay::CollectibleComponent collectibleComp;
        collectibleComp.type = Gameplay::CollectibleComponent::CollectibleType::HEALTH_PACK;
        collectibleComp.value = 25.0f;
        coordinator.AddComponent(collectible, collectibleComp);
        
        // Random position within ±900 range
        float cx = static_cast<float>(pos_dis(gen));
        float cz = static_cast<float>(pos_dis(gen));
        
        coordinator.AddComponent(collectible, Rendering::TransformComponent{
            glm::vec3(cx, 2.0f, cz),  // Floating above ground
            glm::vec3(0.0f),
            glm::vec3(0.5f, 0.5f, 0.5f)
        });
        coordinator.AddComponent(collectible, Rendering::MaterialComponent{
            glm::vec3(0.0f, 1.0f, 0.0f), // Green health pickups
            0.0f, 0.2f, 1.0f
        });
        collectibleCount++;
    }
    std::cout << "    Created " << collectibleCount << " collectibles" << std::endl;
    
    std::cout << "World created with " << (1 + 1 + numEnemies + buildingCount + 4 + collectibleCount) << " entities" << std::endl;
}

// Create D3D12Mesh from procedurally generated BuildingMesh
std::unique_ptr<D3D12Mesh> CreateD3D12MeshFromBuilding(
    Rendering::DX12RenderBackend* backend, 
    const BuildingMesh& buildingMesh) 
{
    // Convert BuildingMesh to standard Vertex format with vertex colors
    std::vector<Vertex> vertices;
    vertices.reserve(buildingMesh.positions.size());
    
    // Debug: count color variations
    int brightCount = 0, darkCount = 0, wallCount = 0;
    
    for (size_t i = 0; i < buildingMesh.positions.size(); ++i) {
        Vertex v;
        v.position = buildingMesh.positions[i];
        v.normal = buildingMesh.normals[i];
        v.tangent = glm::vec3(1, 0, 0);  // Default tangent
        v.texcoord = i < buildingMesh.uvs.size() ? buildingMesh.uvs[i] : glm::vec2(0);
        
        // Set vertex color from BuildingMesh
        // RGB = vertex color for variety, A = emissive intensity for window glow
        glm::vec3 vertColor = i < buildingMesh.colors.size() ? buildingMesh.colors[i] : glm::vec3(1.0f);
        glm::vec3 emissive = i < buildingMesh.emissive.size() ? buildingMesh.emissive[i] : glm::vec3(0);
        float emissiveIntensity = glm::length(emissive) / 10.0f;  // Normalize emissive to 0-1 range
        
        v.color = glm::vec4(vertColor, emissiveIntensity);
        vertices.push_back(v);
        
        // Debug: categorize vertex
        float brightness = (vertColor.r + vertColor.g + vertColor.b) / 3.0f;
        if (brightness > 0.6f) brightCount++;
        else if (brightness < 0.2f) darkCount++;
        else wallCount++;
    }
    
    // Debug output for first building only
    static bool debugOnce = true;
    if (debugOnce && buildingMesh.positions.size() > 0) {
        std::cout << "[BuildingMesh] Vertices: " << buildingMesh.positions.size() 
                  << ", Colors: " << buildingMesh.colors.size()
                  << " (bright:" << brightCount << " dark:" << darkCount << " wall:" << wallCount << ")" << std::endl;
        debugOnce = false;
    }
    
    auto mesh = std::make_unique<D3D12Mesh>();
    if (mesh->Create(backend, vertices, buildingMesh.indices, "ProceduralBuilding")) {
        return mesh;
    }
    return nullptr;
}


// Sync ECS entities with D3D12 rendering
void SyncEntitiesToRenderer() {
    // Clear existing meshes from pipeline (needed to re-add with updated transforms)
    renderPipeline->ClearMeshes();
    
    // Iterate through the explicit list of world entities we created for this demo
    for (Core::Entity entity : renderEntities) {
        if (!coordinator.HasComponent<Rendering::TransformComponent>(entity)) continue;
        if (!coordinator.HasComponent<Rendering::MaterialComponent>(entity)) continue;
        
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        auto& material = coordinator.GetComponent<Rendering::MaterialComponent>(entity);
        
        // Create D3D12 mesh for this entity (only once, cached)
        if (entityMeshes.find(entity) == entityMeshes.end()) {
            std::unique_ptr<D3D12Mesh> mesh;
            
            // Check if this is a procedural building
            auto buildingIt = buildingEntityToMeshIndex.find(entity);
            if (buildingIt != buildingEntityToMeshIndex.end()) {
                // Use procedural building mesh
                size_t meshIdx = buildingIt->second;
                if (meshIdx < generatedBuildingMeshes.size()) {
                    mesh = CreateD3D12MeshFromBuilding(
                        renderPipeline->GetBackend(), 
                        generatedBuildingMeshes[meshIdx]
                    );
                }
            }
            // Ground plane: very flat and wide
            else if (transform.scale.y < 2.0f && (transform.scale.x > 10.0f || transform.scale.z > 10.0f)) {
                mesh = MeshGenerator::CreatePlane(renderPipeline->GetBackend());
            }
            // Everything else (player, enemies, collectibles)
            else {
                mesh = MeshGenerator::CreateCube(renderPipeline->GetBackend());
            }
            
            if (mesh) {
                entityMeshes[entity] = std::move(mesh);
            }
        }
        
        // Update mesh transform and material from ECS (every frame)
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
    coordinator.RegisterComponent<Rendering::MeshComponent>();  // Added - required by LevelSystem
    
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
    
    // Register level components (required by LevelSystem - missing before caused crash!)
    coordinator.RegisterComponent<Gameplay::DimensionalVisibilityComponent>();
    coordinator.RegisterComponent<Gameplay::WorldRotationComponent>();
    coordinator.RegisterComponent<Gameplay::PlatformComponent>();
    coordinator.RegisterComponent<Gameplay::WallComponent>();
    coordinator.RegisterComponent<Gameplay::CollectibleComponent>();
    coordinator.RegisterComponent<Gameplay::InteractableComponent>();
    
    // Register physics components (required for CharacterControllerSystem and PhysXPhysicsSystem)
    coordinator.RegisterComponent<Physics::RigidbodyComponent>();
    coordinator.RegisterComponent<Physics::ColliderComponent>();
    coordinator.RegisterComponent<Physics::CharacterControllerComponent>();
    
    std::cout << "ECS initialized with all gameplay components (matching OpenGL)" << std::endl;
    
    // Register PhysXPhysicsSystem as ECS system (so it auto-processes entities with ColliderComponent)
    // This is CRITICAL - using RegisterSystem vs make_shared means the system gets the mEntities set
    physicsSystem = coordinator.RegisterSystem<Physics::PhysXPhysicsSystem>();
    
    // PhysX system processes all entities with ColliderComponent
    // RigidbodyComponent is optional - entities without it become static actors
    Core::Signature physicsSignature;
    physicsSignature.set(coordinator.GetComponentType<Physics::ColliderComponent>());
    coordinator.SetSystemSignature<Physics::PhysXPhysicsSystem>(physicsSignature);
    
    if (!physicsSystem->Initialize()) {
        std::cerr << "Failed to initialize PhysX" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    std::cout << "PhysX physics system initialized (as ECS system)" << std::endl;
    
    // Register and initialize CharacterControllerSystem (handles camera-relative movement)
    characterControllerSystem = coordinator.RegisterSystem<Gameplay::CharacterControllerSystem>();
    
    // Set system signature: requires all these components
    Core::Signature characterControllerSignature;
    characterControllerSignature.set(coordinator.GetComponentType<Physics::CharacterControllerComponent>());
    characterControllerSignature.set(coordinator.GetComponentType<Gameplay::PlayerInputComponent>());
    characterControllerSignature.set(coordinator.GetComponentType<Gameplay::PlayerMovementComponent>());
    characterControllerSignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    characterControllerSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Gameplay::CharacterControllerSystem>(characterControllerSignature);
    
    characterControllerSystem->Initialize();
    std::cout << "CharacterControllerSystem initialized" << std::endl;
    
    // Register EnemyAISystem (re-added with proper component registrations)
    enemyAISystem = coordinator.RegisterSystem<Gameplay::EnemyAISystem>();
    Core::Signature enemyAISignature;
    enemyAISignature.set(coordinator.GetComponentType<Gameplay::EnemyAIComponent>());
    enemyAISignature.set(coordinator.GetComponentType<Gameplay::EnemyCombatComponent>());
    enemyAISignature.set(coordinator.GetComponentType<Gameplay::EnemyMovementComponent>());
    enemyAISignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    enemyAISignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Gameplay::EnemyAISystem>(enemyAISignature);
    enemyAISystem->Initialize();
    std::cout << "EnemyAISystem initialized" << std::endl;
    
    // Register LevelSystem (manages collectibles, level logic)
    levelSystem = coordinator.RegisterSystem<Gameplay::LevelSystem>();
    Core::Signature levelSignature; // Empty signature - LevelSystem doesn't require specific components
    coordinator.SetSystemSignature<Gameplay::LevelSystem>(levelSignature);
    levelSystem->Initialize();
    std::cout << "LevelSystem initialized" << std::endl;
    
    // Register TargetingSystem (for combat lock-on)
    targetingSystem = coordinator.RegisterSystem<Gameplay::TargetingSystem>();
    Core::Signature targetingSignature;
    targetingSignature.set(coordinator.GetComponentType<Gameplay::TargetingComponent>());
    coordinator.SetSystemSignature<Gameplay::TargetingSystem>(targetingSignature);
    targetingSystem->Initialize();
    std::cout << "TargetingSystem initialized" << std::endl;
    
    // Register PlayerMovementSystem (extended player movement including wall-run input)
    playerMovementSystem = coordinator.RegisterSystem<Gameplay::PlayerMovementSystem>();
    Core::Signature playerMoveSignature;
    playerMoveSignature.set(coordinator.GetComponentType<Gameplay::PlayerInputComponent>());
    playerMoveSignature.set(coordinator.GetComponentType<Gameplay::PlayerMovementComponent>());
    playerMoveSignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    playerMoveSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Gameplay::PlayerMovementSystem>(playerMoveSignature);
    playerMovementSystem->Initialize();
    std::cout << "PlayerMovementSystem initialized" << std::endl;
    
    // Register WallRunningSystem (wall-run mechanics using PhysX raycasting)
    wallRunningSystem = coordinator.RegisterSystem<Physics::WallRunningSystem>();
    Core::Signature wallRunSignature;
    wallRunSignature.set(coordinator.GetComponentType<Physics::CharacterControllerComponent>());
    wallRunSignature.set(coordinator.GetComponentType<Physics::RigidbodyComponent>());
    wallRunSignature.set(coordinator.GetComponentType<Rendering::TransformComponent>());
    coordinator.SetSystemSignature<Physics::WallRunningSystem>(wallRunSignature);
    wallRunningSystem->Initialize();
    std::cout << "WallRunningSystem initialized" << std::endl;

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
    
    // Configure ThirdPersonCameraRig at startup (used for ORBIT_FOLLOW and COMBAT_FOCUS modes)
    // These settings match the working OpenGL version
    thirdPersonRig.SetCamera(mainCamera.get());
    ThirdPersonCameraRig::Settings rigSettings;
    rigSettings.distance = 4.5f;
    rigSettings.height = 1.6f;
    rigSettings.shoulderOffsetX = 0.35f;
    rigSettings.followSmooth = 18.0f;
    rigSettings.verticalSmooth = 8.0f;
    rigSettings.smoothSpeed = 12.0f;
    rigSettings.controlYawSmooth = 10.0f;
    rigSettings.targetSmooth = 20.0f;
    rigSettings.enableDynamicShoulder = true;
    rigSettings.shoulderSmooth = 12.0f;
    rigSettings.centerBiasGain = 1.2f;
    rigSettings.centerBiasMax = 1.0f;
    thirdPersonRig.Configure(rigSettings);
    
    // Set the camera on the CharacterControllerSystem for camera-relative movement
    // (This is the key integration that was missing from DX12!)
    characterControllerSystem->SetCamera(mainCamera.get());
    if (playerMovementSystem) {
        playerMovementSystem->SetCamera(mainCamera.get());
    }

    std::cout << "Camera initialized (Orbit FREE_LOOK around player)" << std::endl;
    std::cout << "Press 1 for Third Person, 2 for Free Look, 3 for Combat Focus" << std::endl;


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
    // Initialize procedural building generator
    buildingGenerator = std::make_unique<CudaBuildingGenerator>();
    if (!buildingGenerator->Initialize()) {
        std::cerr << "Failed to initialize building generator" << std::endl;
    } else {
        std::cout << "Building generator initialized" << std::endl;
    }
    
    CreateGameWorld();
    
    // Initialize WorldChunkManager for chunk-based streaming
    // NOTE: Building generation disabled in callbacks to avoid overwhelming DX12
    // Buildings are pre-generated in CreateGameWorld for stability
    chunkManager = std::make_unique<World::WorldChunkManager>();
    chunkManager->Initialize(
        // Generate callback (runs on worker thread)
        [](World::WorldChunk& chunk) {
            // Worker thread: just log (building generation disabled for stability)
        },
        // Loaded callback (runs on main thread when chunk is ready)
        [](World::WorldChunk& chunk) {
            // Disabled: was generating buildings here but caused DX12 Present failures
            // std::cout << "[ChunkLoad] Chunk ready: (" << chunk.coord.x << ", " << chunk.coord.y << ")" << std::endl;
        },
        // Unloaded callback (runs on main thread before chunk is freed)  
        [](World::WorldChunk& chunk) {
            // Disabled: chunk unload handling
        }
    );
    std::cout << "WorldChunkManager initialized (streaming infrastructure ready, generation disabled)" << std::endl;
    
    // Bind player entity to systems that need it
    if (enemyAISystem) {
        enemyAISystem->SetPlayerEntity(playerEntity);
    }
    if (targetingSystem) {
        targetingSystem->SetPlayerEntity(playerEntity);
    }

    // Initial sync of entities to renderer
    SyncEntitiesToRenderer();
    std::cout << "Game world synced to renderer: " << renderPipeline->GetMeshCount() << " meshes" << std::endl;
    
    // (Debug-only initial entity dump removed to keep DX12 output clean.)

    // Main game loop
    std::cout << "Entering main loop (Press ESC to exit, TAB for mouse capture)..." << std::endl;
    
    // Fixed timestep physics (matches OpenGL version)
    const float FIXED_TIMESTEP = 1.0f / 60.0f;
    float accumulator = 0.0f;
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    
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

        // Calculate actual delta time from frame timing
        auto currentFrameTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentFrameTime - lastFrameTime).count();
        lastFrameTime = currentFrameTime;
        
        // Clamp deltaTime to prevent physics explosion on frame spikes
        if (deltaTime > 0.1f) deltaTime = 0.1f;
        
        time += deltaTime;
        
        
        // Update input component from GLFW
        UpdateInputComponent(playerEntity);
        
        // Handle player movement via CharacterControllerSystem (camera-relative movement)
        // This replaces the inline HandlePlayerMovement - the system uses the camera set via SetCamera()
        if (mouseCaptured && characterControllerSystem) {
            characterControllerSystem->Update(deltaTime);
        }
        
        // Update gameplay systems (matching OpenGL game loop order)
        if (enemyAISystem) {
            enemyAISystem->Update(deltaTime);
        }
        if (levelSystem) {
            levelSystem->Update(deltaTime);
        }
        if (targetingSystem) {
            targetingSystem->Update(deltaTime);
        }
        // PlayerMovementSystem DISABLED - conflicts with CharacterControllerSystem
        // Both systems were modifying player velocity, causing microsliding
        // CharacterControllerSystem alone handles all player movement now
        if (wallRunningSystem) {
            wallRunningSystem->Update(deltaTime);
        }
        
        // Update chunk streaming based on player position
        if (chunkManager && playerEntity != 0) {
            auto& playerTransform = coordinator.GetComponent<Rendering::TransformComponent>(playerEntity);
            chunkManager->Update(playerTransform.position, frameNumber);
            chunkManager->ProcessCompletedChunks();
        }
        ++frameNumber;

        // Camera mode toggles (match GL renderer behavior: 1=ORBIT_FOLLOW, 2=FREE_LOOK, 3=COMBAT_FOCUS, 4=THIRD_PERSON)
        if (mainCamera) {
            static bool key1Prev = false;
            static bool key2Prev = false;
            static bool key3Prev = false;
            static bool key4Prev = false;

            bool key1 = keys[GLFW_KEY_1] != 0;
            bool key2 = keys[GLFW_KEY_2] != 0;
            bool key3 = keys[GLFW_KEY_3] != 0;

            if (key1 && !key1Prev) {
                mainCamera->SetCameraMode(OrbitCamera::CameraMode::ORBIT_FOLLOW);
                glfwSetWindowTitle(window, "CudaGame DX12 - Camera: Third Person (Over-the-Shoulder)");
            }
            if (key2 && !key2Prev) {
                mainCamera->SetCameraMode(OrbitCamera::CameraMode::FREE_LOOK);
                glfwSetWindowTitle(window, "CudaGame DX12 - Camera: Free Look");
            }
            if (key3 && !key3Prev) {
                mainCamera->SetCameraMode(OrbitCamera::CameraMode::COMBAT_FOCUS);
                glfwSetWindowTitle(window, "CudaGame DX12 - Camera: Combat Focus");
            }

            key1Prev = key1;
            key2Prev = key2;
            key3Prev = key3;
        }
        
        // Fixed timestep physics update (matches OpenGL version - fixes speed bursting)
        // This runs physics at consistent 60 FPS regardless of frame rate
        accumulator += deltaTime;
        while (accumulator >= FIXED_TIMESTEP) {
            if (physicsSystem) {
                physicsSystem->Update(FIXED_TIMESTEP);
            }
            accumulator -= FIXED_TIMESTEP;
        }
        
        
        // Debug: Log CharacterControllerSystem entity count and player velocity every 60 frames
        static int debugFrameCounter = 0;
        if (++debugFrameCounter % 60 == 0 && coordinator.HasComponent<Physics::RigidbodyComponent>(playerEntity)) {
            auto& playerRB = coordinator.GetComponent<Physics::RigidbodyComponent>(playerEntity);
            auto& playerTransform = coordinator.GetComponent<Rendering::TransformComponent>(playerEntity);
            std::cout << "[DX12 DEBUG] Player pos: (" << playerTransform.position.x << ", " 
                      << playerTransform.position.y << ", " << playerTransform.position.z << ") "
                      << "vel: (" << playerRB.velocity.x << ", " << playerRB.velocity.y << ", " 
                      << playerRB.velocity.z << ") "
                      << "force: (" << playerRB.forceAccumulator.x << ", " << playerRB.forceAccumulator.y 
                      << ", " << playerRB.forceAccumulator.z << ")" << std::endl;
        }
        
        // Update camera with player position/velocity from PhysX (avoids ECS-PhysX sync jitter)
        glm::vec3 playerPos(0.0f);
        glm::vec3 playerVel(0.0f);
        
        // Read position directly from PhysX actor (not ECS) to avoid sync jitter
        auto physIt = entityPhysicsActors.find(playerEntity);
        if (physIt != entityPhysicsActors.end() && physIt->second) {
            PxRigidActor* actor = physIt->second;
            PxTransform pxTrans = actor->getGlobalPose();
            playerPos = glm::vec3(pxTrans.p.x, pxTrans.p.y, pxTrans.p.z);
            
            // Get velocity if dynamic actor
            if (PxRigidDynamic* dyn = actor->is<PxRigidDynamic>()) {
                PxVec3 v = dyn->getLinearVelocity();
                playerVel = glm::vec3(v.x, v.y, v.z);
            }
        } else {
            // Fallback to ECS position if no physics actor
            if (coordinator.HasComponent<Rendering::TransformComponent>(playerEntity)) {
                playerPos = coordinator.GetComponent<Rendering::TransformComponent>(playerEntity).position;
            }
        }
        
        // ====== BOUNDED CAMERA FOLLOW SYSTEM ======
        // Based on RE Engine (damped spring-arm) + Unreal (offset clamping)
        static glm::vec3 smoothedPlayerPos = playerPos;
        
        // Calculate player horizontal speed
        float playerSpeed = glm::length(glm::vec2(playerVel.x, playerVel.z));
        
        // SEPARATE X/Z AND Y SMOOTHING to prevent jump jitter
        
        // 1. HORIZONTAL SMOOTHING (X/Z) - velocity-based
        float horizontalSmoothFactor;
        if (playerSpeed < 0.1f) {
            // Snap horizontal when idle
            smoothedPlayerPos.x = playerPos.x;
            smoothedPlayerPos.z = playerPos.z;
            horizontalSmoothFactor = 50.0f;  // Used for offset calc
        } else if (playerSpeed < 0.5f) {
            horizontalSmoothFactor = 40.0f;
        } else if (playerSpeed < 2.0f) {
            horizontalSmoothFactor = 25.0f;
        } else {
            horizontalSmoothFactor = 15.0f;
        }
        
        if (playerSpeed >= 0.1f) {
            float hBlend = glm::clamp(deltaTime * horizontalSmoothFactor, 0.0f, 1.0f);
            
            // Direction-aware lag (faster catch-up when behind)
            glm::vec3 toPlayer = playerPos - smoothedPlayerPos;
            float distToPlayer = glm::length(glm::vec2(toPlayer.x, toPlayer.z));
            if (distToPlayer > 0.5f && playerSpeed > 0.5f) {
                glm::vec3 playerMoveDir = glm::normalize(glm::vec3(playerVel.x, 0, playerVel.z));
                glm::vec3 toPlayerDir = glm::normalize(toPlayer);
                if (glm::dot(playerMoveDir, toPlayerDir) > 0.5f) {
                    hBlend = glm::min(hBlend * 2.0f, 1.0f);
                }
            }
            
            smoothedPlayerPos.x = glm::mix(smoothedPlayerPos.x, playerPos.x, hBlend);
            smoothedPlayerPos.z = glm::mix(smoothedPlayerPos.z, playerPos.z, hBlend);
        }
        
        // 2. VERTICAL SMOOTHING (Y) - gentle, constant factor for smooth jumps
        const float verticalSmoothFactor = 8.0f;  // Gentler for smooth vertical tracking
        float vBlend = glm::clamp(deltaTime * verticalSmoothFactor, 0.0f, 1.0f);
        smoothedPlayerPos.y = glm::mix(smoothedPlayerPos.y, playerPos.y, vBlend);
        
        // 3. BOUNDED OFFSET CLAMP (prevents excessive lag)
        glm::vec3 offset = smoothedPlayerPos - playerPos;
        const float maxHorizontalOffset = 1.5f;
        const float maxVerticalOffset = 2.0f;  // Allow more vertical lag for smooth jumps
        offset.x = glm::clamp(offset.x, -maxHorizontalOffset, maxHorizontalOffset);
        offset.z = glm::clamp(offset.z, -maxHorizontalOffset, maxHorizontalOffset);
        offset.y = glm::clamp(offset.y, -maxVerticalOffset, maxVerticalOffset);
        smoothedPlayerPos = playerPos + offset;
        
        // Single camera update - use smoothed player position
        if (mainCamera) {
            mainCamera->Update(deltaTime, smoothedPlayerPos, playerVel);
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
