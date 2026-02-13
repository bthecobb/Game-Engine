#ifdef _WIN32
#include "Core/Coordinator.h"
#include "Gameplay/PlayerComponents.h"
#include "Gameplay/EnemyComponents.h"
#include "Gameplay/LevelComponents.h"
#include "Gameplay/PlayerMovementSystem.h"
#include "Gameplay/CharacterControllerSystem.h"
#include "Gameplay/AnimationControllerComponent.h"
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
#include "Rendering/ProceduralHumanoidMesh.h"
#include "Animation/ProceduralAnimationGenerator.h"

#include <cmath>
#include <fstream>
#include <glm/gtx/string_cast.hpp>

using namespace physx;

using namespace CudaGame;
using namespace CudaGame;
using namespace CudaGame::Rendering;

#include "Testing/TestSystem.h" // Added for Unified Testing System

#include "Gameplay/CharacterFactory.h"
#include "Animation/AnimationSystem.h"
#include "Animation/AnimationComponent.h"
#include "AI/AIComponent.h"
#include "Gameplay/CombatComponents.h"

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
std::shared_ptr<CudaGame::Animation::AnimationSystem> animationSystem;

// Entity to mesh mapping for D3D12 rendering
std::unordered_map<Core::Entity, std::unique_ptr<D3D12Mesh>> entityMeshes;

// Procedural building generator
std::unique_ptr<CudaBuildingGenerator> buildingGenerator;
std::vector<BuildingMesh> generatedBuildingMeshes;  // Keep meshes alive
std::unordered_map<Core::Entity, size_t> buildingEntityToMeshIndex;  // Map building entities to mesh index

// World chunk streaming manager
std::unique_ptr<World::WorldChunkManager> chunkManager;
uint64_t frameNumber = 0;

// Character Factory
std::unique_ptr<Gameplay::CharacterFactory> characterFactory;

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
namespace CudaGame { namespace Rendering { class DX12RenderBackend; struct BuildingMesh; } }
class DX12RenderBackend; // Local alias or global? DX12RenderBackend is in CudaGame::Rendering namespace.
// Using full qualification to be safe
std::unique_ptr<D3D12Mesh> CreateD3D12MeshFromBuilding(
    CudaGame::Rendering::DX12RenderBackend* backend, 
    const CudaGame::Rendering::BuildingMesh& buildingMesh
);

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
    (void)mods;
    if (button >= 0 && button < 8) {
        if (action == GLFW_PRESS)
            mouseButtons[button] = true;
        else if (action == GLFW_RELEASE)
            mouseButtons[button] = false;
    }
}

// Update player input component from GLFW
void UpdateInputComponent(Core::Entity entity) {
    if (!coordinator.HasComponent<Gameplay::PlayerInputComponent>(entity)) return;
    
    auto& input = coordinator.GetComponent<Gameplay::PlayerInputComponent>(entity);
    
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

    // Debug input (spammy but needed)
    static int inputLogCounter = 0;
    if (++inputLogCounter % 60 == 0) {
        if (input.keys[GLFW_KEY_W] || input.keys[GLFW_KEY_S] || input.keys[GLFW_KEY_A] || input.keys[GLFW_KEY_D]) {
            std::cout << "[Input] WASD pressed. Mouse Delta: (" << input.mouseDelta.x << ", " << input.mouseDelta.y << ")" << std::endl;
        }
        if (glm::length(input.mouseDelta) > 1.0f) {
            std::cout << "[Input] Mouse moved. Delta: (" << input.mouseDelta.x << ", " << input.mouseDelta.y << ")" << std::endl;
        }
    }
}

// Handle player movement with enhanced features (dash, sprint, etc.)
void HandlePlayerMovement(Core::Entity entity, float deltaTime) {
    if (!coordinator.HasComponent<Gameplay::PlayerInputComponent>(entity)) return;
    if (!coordinator.HasComponent<Gameplay::PlayerMovementComponent>(entity)) return;
    if (entityPhysicsActors.find(entity) == entityPhysicsActors.end()) return;
    
    auto& input = coordinator.GetComponent<Gameplay::PlayerInputComponent>(entity);
    auto& movement = coordinator.GetComponent<Gameplay::PlayerMovementComponent>(entity);
    PxRigidDynamic* playerActor = static_cast<PxRigidDynamic*>(entityPhysicsActors[entity]);
    
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

            // Create Visual Mesh for Rendering
            if (renderPipeline && renderPipeline->GetBackend()) {
                auto buildingRenderMesh = CreateD3D12MeshFromBuilding(renderPipeline->GetBackend(), mesh);
                if (buildingRenderMesh) {
                    // Set transform
                    // buildingRenderMesh->transform is updated by system or manual logic.
                    // Here we set initial transform.
                    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(x, -0.5f, z)); // Match entity transform
                    // Note: entity transform is handled by rendering system (UpdateRenderComponents etc).
                    // But D3D12Mesh holds its own transform for per-object constants in DX12 pipeline currently.
                    // We must ensure the System syncs them.
                    // For static buildings, setting it once here is okay for now.
                    buildingRenderMesh->transform = model;

                    renderPipeline->AddMesh(buildingRenderMesh.get());
                    entityMeshes[building] = std::move(buildingRenderMesh);
                    // Add mesh entity to renderEntities for potential future use
                }
            }
        }
    }
    
    chunk.buildingCounts[static_cast<int>(chunk.lodLevel)] = buildingsToGenerate;
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
    // Debug output for building generation
    if (buildingMesh.positions.size() > 0) {
        std::cerr << "[BuildingMesh] Generated: " << buildingMesh.positions.size() << " verts, " 
                  << buildingMesh.indices.size() << " indices. "
                  << "Colors: " << buildingMesh.colors.size()
                  << " (bright:" << brightCount << " dark:" << darkCount << " wall:" << wallCount << ")" << std::endl;

        // Log first 4 vertices for debugging
        std::cerr << "[BuildingMesh] First 4 vertices:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)4, buildingMesh.positions.size()); ++i) {
            std::cerr << "  V[" << i << "] Pos: " << glm::to_string(buildingMesh.positions[i]) 
                      << " Norm: " << glm::to_string(buildingMesh.normals[i]) << std::endl;
        }
    } else {
        std::cerr << "[BuildingMesh] ERROR: Generated mesh has 0 vertices!" << std::endl;
    }
    
    auto mesh = std::make_unique<D3D12Mesh>();
    if (mesh->Create(backend, vertices, buildingMesh.indices, "ProceduralBuilding")) {
        // Generate meshlets for Mesh Shader pipeline
        mesh->GenerateMeshlets(backend, vertices, buildingMesh.indices);
        return mesh;
    }
    return nullptr;
}


// Sync ECS entities with D3D12 rendering
void SyncEntitiesToRenderer() {
    renderPipeline->ClearMeshes();
    
    for (Core::Entity entity : renderEntities) {
        if (!coordinator.HasComponent<Rendering::TransformComponent>(entity)) continue;
        if (!coordinator.HasComponent<Rendering::MaterialComponent>(entity)) continue;
        
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        auto& material = coordinator.GetComponent<Rendering::MaterialComponent>(entity);
        
        if (entityMeshes.find(entity) == entityMeshes.end()) {
            std::unique_ptr<D3D12Mesh> mesh;
            
            // 1. Check for explicit MeshComponent (Asset Loading)
            bool meshLoaded = false;
            if (coordinator.HasComponent<Rendering::MeshComponent>(entity)) {
                auto& meshComp = coordinator.GetComponent<Rendering::MeshComponent>(entity);
                
                // Check if procedural mesh requested (Internal Debug Model)
                // Check if procedural mesh requested (Internal Debug Model)
                if (meshComp.modelPath == "internal:ProceduralArm") {
                    std::cout << "[Renderer] Generating Procedural Skinned Arm..." << std::endl;
                    
                    std::shared_ptr<Animation::Skeleton> skeleton = nullptr;
                    if (coordinator.HasComponent<Animation::AnimationComponent>(entity)) {
                         auto& animComp = coordinator.GetComponent<Animation::AnimationComponent>(entity);
                         skeleton = animComp.skeleton;
                    }
                    if (!skeleton) {
                        std::cout << "[Renderer] Warning: No skeleton found for ProceduralArm, creating empty." << std::endl;
                        skeleton = std::make_shared<Animation::Skeleton>();
                    }

                    mesh = Rendering::MeshGenerator::CreateSkinnedBlockMesh(renderPipeline->GetBackend(), skeleton);
                    
                    if (mesh) {
                        meshLoaded = true;
                        // Link skeleton and animations intentionally (similar to LoadFromFile logic)
                        if (coordinator.HasComponent<Animation::AnimationComponent>(entity)) {
                             auto& animComp = coordinator.GetComponent<Animation::AnimationComponent>(entity);
                             // animComp.skeleton already set via pointer copy if it existed
                             animComp.globalTransforms.resize(skeleton->bones.size(), glm::mat4(1.0f));
                             
                             const auto& anims = mesh->GetAnimations();
                             if (!anims.empty()) {
                                 std::cout << "[Renderer] Registering " << anims.size() << " procedural animations." << std::endl;
                                 for (const auto& clip : anims) {
                                     animComp.animations[clip->name] = *clip;
                                 }
                                 // Auto-play
                                 animComp.currentAnimation = anims[0]->name;
                                 animComp.isPlaying = true;
                                 animComp.playbackSpeed = 1.0f;
                             }
                        }
                    }
                }
                // Standard File Loading
                else if (!meshComp.modelPath.empty() && meshComp.modelPath.find(".") != std::string::npos) {
                    mesh = std::make_unique<D3D12Mesh>();
                     
                    bool needsSkeleton = coordinator.HasComponent<Animation::AnimationComponent>(entity);
                    
                    std::cerr << "[Renderer] Loading mesh from file: " << meshComp.modelPath << " (Skeleton required: " << needsSkeleton << ")" << std::endl;
                    if (mesh->LoadFromFile(renderPipeline->GetBackend(), meshComp.modelPath, needsSkeleton)) {
                        meshLoaded = true;
                        std::cerr << "[Renderer] Loaded mesh SUCCESS: " << meshComp.modelPath << std::endl;
                        
                        // Link skeleton to AnimationComponent if loaded
                        if (needsSkeleton && mesh->GetSkeleton()) {
                            std::cerr << "[Renderer] Linking skeleton..." << std::endl;
                            auto& animComp = coordinator.GetComponent<Animation::AnimationComponent>(entity);
                            animComp.skeleton = mesh->GetSkeleton();
                            std::cerr << "[Renderer] Skeleton linked. Bones: " << (animComp.skeleton ? std::to_string(animComp.skeleton->bones.size()) : "NULL") << std::endl;
                            
                            animComp.globalTransforms.resize(mesh->GetSkeleton()->bones.size(), glm::mat4(1.0f));
                            std::cerr << "[Renderer] Resized globalTransforms." << std::endl;
                            
                            // Register embedded animations
                            const auto& anims = mesh->GetAnimations();
                            std::cerr << "[Renderer] Got " << anims.size() << " animations." << std::endl;
                            
                            if (!anims.empty()) {
                                std::cerr << "[Renderer] Registering " << anims.size() << " animations for entity " << entity << std::endl;
                                int animIndex = 0;
                                for (const auto& clip : anims) {
                                    if (clip) {
                                        std::cerr << "  - Clip " << animIndex << ": " << (clip->name.empty() ? "UNNAMED" : clip->name) << std::endl;
                                        animComp.animations[clip->name] = *clip; // Copy content
                                        animIndex++;
                                    } else {
                                        std::cerr << "  - Clip " << animIndex << " is NULL!" << std::endl;
                                    }
                                }
                                
                                // Auto-play first animation if none selected
                                if (animComp.currentAnimation.empty()) {
                                    animComp.currentAnimation = anims[0]->name;
                                    animComp.isPlaying = true;
                                    animComp.playbackSpeed = 1.0f;
                                    std::cerr << "[Renderer] Auto-playing animation: " << animComp.currentAnimation << std::endl;
                                }
                            }
                        }
                    } else {
                        mesh.reset(); // Failed to load
                        std::cerr << "[Renderer] Failed to load mesh: " << meshComp.modelPath << ", falling back..." << std::endl;
                    }
                }
            }

            if (meshLoaded) {
                 // Already loaded
            }
            // 2. Procedural Buildings
            else if (buildingEntityToMeshIndex.find(entity) != buildingEntityToMeshIndex.end()) {
                size_t meshIdx = buildingEntityToMeshIndex[entity];
                if (meshIdx < generatedBuildingMeshes.size()) {
                    mesh = CreateD3D12MeshFromBuilding(renderPipeline->GetBackend(), generatedBuildingMeshes[meshIdx]);
                }
            }
            // 3. Skinned Mesh (Procedural Fallback)
            else if (coordinator.HasComponent<Animation::AnimationComponent>(entity)) {
                auto& animComp = coordinator.GetComponent<Animation::AnimationComponent>(entity);
                if (animComp.skeleton && animComp.skeleton->bones.size() > 0) {
                     // Using procedural block mesh if existing skeleton is valid but not from file
                     mesh = MeshGenerator::CreateSkinnedBlockMesh(renderPipeline->GetBackend(), animComp.skeleton);
                } else {
                     // No skeleton or empty, use cube
                     mesh = MeshGenerator::CreateCube(renderPipeline->GetBackend());
                }
            }
            // 4. Transform-based procedural shapes
            else if (transform.scale.y < 2.0f && (transform.scale.x > 10.0f || transform.scale.z > 10.0f)) {
                mesh = MeshGenerator::CreatePlane(renderPipeline->GetBackend());
            }
            else {
                mesh = MeshGenerator::CreateCube(renderPipeline->GetBackend());
            }
            
            if (mesh) {
                entityMeshes[entity] = std::move(mesh);
            }
        }
        
        if (entityMeshes[entity]) {
            entityMeshes[entity]->transform = transform.getMatrix();
            entityMeshes[entity]->GetMaterial().albedoColor = glm::vec4(material.albedo, 1.0f);
            entityMeshes[entity]->GetMaterial().metallic = material.metallic;
            entityMeshes[entity]->GetMaterial().roughness = material.roughness;
            entityMeshes[entity]->GetMaterial().ambientOcclusion = material.ao;
            
            // Sync Animation Bone Matrices - handled in animation sync block below
            // NOTE: finalBoneMatrices (with inverse bind pose) are synced at line ~1619

            entityMeshes[entity]->UpdateGPUInstanceData();
            renderPipeline->AddMesh(entityMeshes[entity].get());
        }
    }
}

// ... CreateGameWorld modified ...

void CreateGameWorld() {
    std::cerr << "[Setup] Entering CreateGameWorld..." << std::endl;
    std::cerr << "Creating 3D game world (EXPANDED 10000x10000)..." << std::endl;
    
    // Ground
    auto ground = coordinator.CreateEntity();
    renderEntities.push_back(ground);
    coordinator.AddComponent(ground, Rendering::TransformComponent{
        glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f), glm::vec3(10000.0f, 1.0f, 10000.0f)
    });
    coordinator.AddComponent(ground, Rendering::MaterialComponent{
        glm::vec3(0.25f, 0.25f, 0.28f), 0.0f, 0.8f, 1.0f
    });
    Physics::ColliderComponent groundCollider{};
    groundCollider.shape = Physics::ColliderShape::BOX;
    groundCollider.halfExtents = glm::vec3(5000.0f, 0.5f, 5000.0f);
    coordinator.AddComponent(ground, groundCollider);
    
    std::cerr << "  - Ground: 10000x10000 (EXPANDED)" << std::endl;
    
    // Spawn Player using CharacterFactory (Phase 8 AAA)
    // Setup Profile first if needed, but Factory has default "Knight"
    // We override position.
    glm::vec3 spawnPos(0.0f, 1.5f, 0.0f);
    
    if (characterFactory) {
        std::cerr << "[Setup] Calling CharacterFactory::SpawnCharacter..." << std::endl;
        playerEntity = characterFactory->SpawnCharacter("Knight", spawnPos);
        std::cerr << "[Setup] CharacterFactory::SpawnCharacter RETURNED. Entity: " << playerEntity << std::endl;
        // playerEntity = characterFactory->SpawnCharacter("ProceduralTest", spawnPos);
    } else {
        std::cerr << "[Setup] Factory disabled. Spawning simple player cube." << std::endl;
        playerEntity = coordinator.CreateEntity();
        coordinator.AddComponent(playerEntity, Rendering::TransformComponent{spawnPos, glm::vec3(0), glm::vec3(1)});
        coordinator.AddComponent(playerEntity, Rendering::MeshComponent{"cube"}); // Assumes cube.obj exists or handled
        coordinator.AddComponent(playerEntity, Rendering::MaterialComponent{glm::vec3(0,1,0), 0.5f, 0.5f});
        Physics::RigidbodyComponent rb;
        rb.mass = 80.0f;
        coordinator.AddComponent(playerEntity, rb);
        Physics::CharacterControllerComponent cct;
        cct.height = 1.8f;
        cct.radius = 0.5f;
        coordinator.AddComponent(playerEntity, cct);
        coordinator.AddComponent(playerEntity, Gameplay::PlayerMovementComponent{}); // Needed for movement
    }
    
    if (playerEntity != 0) {
        renderEntities.push_back(playerEntity);
        
        // Check for Animation
        if (coordinator.HasComponent<Animation::AnimationComponent>(playerEntity)) {
             auto& anim = coordinator.GetComponent<Animation::AnimationComponent>(playerEntity);
             if (!anim.currentAnimation.empty()) {
                std::cout << "[Setup] Player playing loaded animation: " << anim.currentAnimation << std::endl;
             } else {
                std::cout << "[Setup] Player has no default animation." << std::endl;
             }
        }
        
        // Add Player Logic Components (if missing)
        if (!coordinator.HasComponent<Gameplay::PlayerInputComponent>(playerEntity)) {
             coordinator.AddComponent(playerEntity, Gameplay::PlayerInputComponent{});
        }
        if (!coordinator.HasComponent<Gameplay::PlayerRhythmComponent>(playerEntity)) {
            coordinator.AddComponent(playerEntity, Gameplay::PlayerRhythmComponent{});
        }
        if (!coordinator.HasComponent<Gameplay::GrapplingHookComponent>(playerEntity)) {
            coordinator.AddComponent(playerEntity, Gameplay::GrapplingHookComponent{});
        }
        if (!coordinator.HasComponent<Gameplay::PlayerMovementComponent>(playerEntity)) {
            coordinator.AddComponent(playerEntity, Gameplay::PlayerMovementComponent{});
        }
        if (!coordinator.HasComponent<Physics::CharacterControllerComponent>(playerEntity)) {
            coordinator.AddComponent(playerEntity, Physics::CharacterControllerComponent{});
        }
        
        // Add physics components for movement (required for PhysXPhysicsSystem)
        if (!coordinator.HasComponent<Physics::RigidbodyComponent>(playerEntity)) {
            Physics::RigidbodyComponent rb;
            rb.mass = 70.0f;  // Human mass
            rb.useGravity = true;
            rb.isKinematic = false;
            coordinator.AddComponent(playerEntity, rb);
        }
        if (!coordinator.HasComponent<Physics::ColliderComponent>(playerEntity)) {
            Physics::ColliderComponent col;
            col.shape = Physics::ColliderShape::CAPSULE;
            col.radius = 0.4f;
            col.halfExtents = glm::vec3(0.4f, 0.9f, 0.4f);  // Capsule half-height
            coordinator.AddComponent(playerEntity, col);
        }
        
        // Ensure Material is set to RED for visibility debugging -> REMOVED to allow Factory Blue
        if (coordinator.HasComponent<Rendering::MaterialComponent>(playerEntity)) {
            auto& mat = coordinator.GetComponent<Rendering::MaterialComponent>(playerEntity);
            mat.roughness = 0.5f;
            mat.metallic = 0.0f;
        }
        
        std::cout << "  - Player created via CharacterFactory" << std::endl;

        // TEST: Replace with Procedural Mesh for Phase 1 verification
        bool useProceduralMesh = true; 
        if (useProceduralMesh) {
             std::cerr << "  - [ProceduralSystem] Replacing Player Mesh with Procedural Humanoid..." << std::endl;
             if (renderPipeline) { // Update AnimationComponent to match new skeleton
                 auto procMesh = Rendering::ProceduralHumanoidMesh::Create(renderPipeline->GetBackend());
                 if (procMesh) {
                     // Get skeleton reference before moving mesh
                     auto skeleton = procMesh->GetSkeleton();
                     
                     // 1. Assign to Renderer
                     entityMeshes[playerEntity] = std::move(procMesh);
                     
                     // 2. Update Animation Component
                     if (coordinator.HasComponent<Animation::AnimationComponent>(playerEntity)) {
                         auto& animComp = coordinator.GetComponent<Animation::AnimationComponent>(playerEntity);
                         animComp.skeleton = skeleton;
                         animComp.finalBoneMatrices.resize(skeleton->bones.size(), glm::mat4(1.0f));
                         
                         // Create and Play Procedural Clip
                         auto walkClip = Animation::ProceduralAnimationGenerator::CreateWalkClip(skeleton);
                         auto idleClip = Animation::ProceduralAnimationGenerator::CreateIdleClip(skeleton);
                         
                         // Store animations in mesh so they persist
                         entityMeshes[playerEntity]->AddAnimation(walkClip);
                         entityMeshes[playerEntity]->AddAnimation(idleClip);
                         
                         animComp.currentAnimation = "Procedural_Walk"; 
                         animComp.useProceduralGeneration = true; // Enable Phase 2 runtime logic
                         animComp.isPlaying = true;
                         animComp.animationTime = 0.0f;
                         
                         std::cout << "  - [ProceduralSystem] Active. Playing: " << animComp.currentAnimation << std::endl;
                     }
                 } else {
                     std::cerr << "  - [ProceduralSystem] FAILURE: Could not create mesh!" << std::endl;
                 }
             }
        }
    } else {
        std::cerr << "FAILED TO SPAWN PLAYER!" << std::endl;
    }

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
        
        glm::vec3 enemySpawnPos(x, 1.0f, z);
        glm::vec3 enemyColor(1.0f, 0.0f, 0.0f); // Red enemies (OpenGL style)
        
        Core::Entity enemyEntity = SpawnEnemy(enemySpawnPos, enemyColor);
        renderEntities.push_back(enemyEntity);
        std::cout << "    Enemy " << (i+1) << " spawned at (" << enemySpawnPos.x << ", " << enemySpawnPos.y << ", " << enemySpawnPos.z << ")" << std::endl;
    }
    
    // ====== BUILDINGS (Dynamic City Generation) ======
    std::cout << "  - Generating Dynamic City (World Size: 10000x10000)..." << std::endl;
    int buildingCount = 0;
    
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for deterministic layout
    std::uniform_real_distribution<> offset_dis(-80.0, 80.0); // Random offset within block (250 size)
    std::uniform_real_distribution<> height_dis(20.0, 140.0);  // Taller buildings
    std::uniform_real_distribution<> width_dis(10.0, 25.0);    // Varied width
    std::uniform_real_distribution<> probability_dis(0.0, 1.0); // For density check
    
    // Dynamic World Settings
    const float worldMin = -4500.0f;
    const float worldMax = 4500.0f;
    const float blockSize = 250.0f; 
    
    // Reserve space for generated building meshes (Estimate: 40x40 blocks * 2 buildings avg)
    generatedBuildingMeshes.reserve(3200);
    
    // Iterate through world in blocks
    for (float x = worldMin; x <= worldMax; x += blockSize) {
        for (float z = worldMin; z <= worldMax; z += blockSize) {
            
            // 1. Density Check: 60% chance to have buildings in this block
            // This creates the "per 2-3 buildings" clustering effect users asked for
            // Skip center spawn area
            if (abs(x) < 200.0f && abs(z) < 200.0f) continue;
            
            // Simple hash-like check or random for density
            // Using random here since seed is fixed
            if (probability_dis(gen) > 0.6f) continue; // 40% empty blocks
            
            // 2. Spawn Cluster in this block (1-3 buildings)
            int buildingsInBlock = 1 + (static_cast<int>(probability_dis(gen) * 10) % 3);
            
            for (int b = 0; b < buildingsInBlock; ++b) {
                // Calculate position with offset
                float xPos = x + static_cast<float>(offset_dis(gen));
                float zPos = z + static_cast<float>(offset_dis(gen));
            
            // Skip center area (spawn zone)
            if (abs(xPos) < 100.0f && abs(zPos) < 100.0f) continue;

            float height = static_cast<float>(height_dis(gen));
            
            // Varied building dimensions
            float buildingWidth = static_cast<float>(width_dis(gen));
            float buildingDepth = static_cast<float>(width_dis(gen));

        
            
            // Generate seed based on position
            uint32_t seed = static_cast<uint32_t>((abs(xPos) * 1000 + abs(zPos)) * 12345);

        
        // Configure building style for procedural generation
        BuildingStyle style;
        style.baseWidth = buildingWidth;
        style.baseDepth = buildingDepth;
        style.height = height;
        style.seed = seed;  // Unique seed per building
        style.baseColor = glm::vec3(0.5f + (seed % 10) * 0.02f, 0.55f + (seed % 7) * 0.02f, 0.6f + (seed % 5) * 0.02f);
        style.accentColor = glm::vec3(0.2f, 0.25f, 0.3f);
        
        // Generate building mesh (with emissive windows)
        if (buildingGenerator) {
            Rendering::BuildingMesh mesh = buildingGenerator->GenerateBuilding(style);
            generatedBuildingMeshes.push_back(mesh);
            
            // Create building entity
            Core::Entity building = coordinator.CreateEntity();
            renderEntities.push_back(building);
            
            // Map this entity to its mesh index
            buildingEntityToMeshIndex[building] = generatedBuildingMeshes.size() - 1;
            
            coordinator.AddComponent(building, Rendering::TransformComponent{
                glm::vec3(xPos, -0.5f, zPos),  // Base at ground level (mesh already has height)
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
    } // End inner loop (b)
    } // End mid loop (z)
} // End outer loop (x)
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
        
        // Random position within ±500 range (City Area)
        std::uniform_real_distribution<> pos_dis(-500.0, 500.0);
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



int main() {
    // Force unbuffered output for debugging
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    std::cout << "=== CudaGame - D3D12 Full 3D Demo ===" << std::endl;
    std::cout << "Features: PhysX Physics, D3D12 Rendering, ECS Architecture" << std::endl;
    std::cout << std::endl;
    std::cout << "[INIT] Starting initialization..." << std::flush;
    
    // File-based checkpoint logging for crash tracing
    std::ofstream checkpoint("checkpoint.log", std::ios::trunc);
    checkpoint << "=== INITIALIZATION CHECKPOINT LOG ===" << std::endl;
    checkpoint.flush();
    
    #define CHECKPOINT(msg) do { checkpoint << "[CP] " << msg << std::endl; checkpoint.flush(); } while(0)

    // Initialize window
    if (!InitializeWindow()) {
        return -1;
    }
    CHECKPOINT("Window initialized");
    
    // Initialize ECS
    // Initialize ECS
    coordinator.Initialize();
    CHECKPOINT("ECS coordinator initialized");
    
    // Register rendering components
    coordinator.RegisterComponent<Rendering::TransformComponent>();
    coordinator.RegisterComponent<Rendering::MaterialComponent>();
    coordinator.RegisterComponent<Rendering::MeshComponent>();  // Added - required by LevelSystem
    CHECKPOINT("Rendering components registered");
    
    // Register gameplay components
    coordinator.RegisterComponent<Gameplay::PlayerMovementComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerCombatComponent>();
    coordinator.RegisterComponent<Gameplay::CombatComponent>(); // Added for CharacterFactory
    coordinator.RegisterComponent<Gameplay::PlayerInputComponent>();
    coordinator.RegisterComponent<Gameplay::PlayerRhythmComponent>();
    coordinator.RegisterComponent<Gameplay::GrapplingHookComponent>();
    coordinator.RegisterComponent<AI::AIComponent>(); // Added for CharacterFactory
    CHECKPOINT("Player components registered");
    
    // Register enemy components
    coordinator.RegisterComponent<Gameplay::EnemyAIComponent>();
    coordinator.RegisterComponent<Gameplay::EnemyCombatComponent>();
    coordinator.RegisterComponent<Gameplay::EnemyMovementComponent>();
    coordinator.RegisterComponent<Gameplay::TargetingComponent>();
    CHECKPOINT("Enemy components registered");
    
    // Register level components (required by LevelSystem - missing before caused crash!)
    coordinator.RegisterComponent<Gameplay::DimensionalVisibilityComponent>();
    coordinator.RegisterComponent<Gameplay::WorldRotationComponent>();
    coordinator.RegisterComponent<Gameplay::PlatformComponent>();
    coordinator.RegisterComponent<Gameplay::WallComponent>();
    coordinator.RegisterComponent<Gameplay::CollectibleComponent>();
    coordinator.RegisterComponent<Gameplay::InteractableComponent>();
    CHECKPOINT("Level components registered");
    
    // Register physics components (required for CharacterControllerSystem and PhysXPhysicsSystem)
    coordinator.RegisterComponent<Physics::RigidbodyComponent>();
    coordinator.RegisterComponent<Physics::ColliderComponent>();
    coordinator.RegisterComponent<Physics::CharacterControllerComponent>();
    CHECKPOINT("Physics components registered");
    
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
    
    // Register Animation Component (Required for Character Controller)
    coordinator.RegisterComponent<Animation::AnimationComponent>();
    coordinator.RegisterComponent<Gameplay::AnimationControllerComponent>();
    
    // Register AnimationSystem (Simple Clip Playback)
    animationSystem = coordinator.RegisterSystem<CudaGame::Animation::AnimationSystem>();
    Core::Signature animSignature;
    animSignature.set(coordinator.GetComponentType<Animation::AnimationComponent>());
    coordinator.SetSystemSignature<CudaGame::Animation::AnimationSystem>(animSignature);
    animationSystem->initialize();
    std::cout << "AnimationSystem initialized" << std::endl;
    
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
    CHECKPOINT("EnemyAISystem done");
    
    // Register LevelSystem (manages collectibles, level logic)
    CHECKPOINT("About to register LevelSystem");
    levelSystem = coordinator.RegisterSystem<Gameplay::LevelSystem>();
    CHECKPOINT("LevelSystem registered");
    Core::Signature levelSignature; // Empty signature - LevelSystem doesn't require specific components
    coordinator.SetSystemSignature<Gameplay::LevelSystem>(levelSignature);
    CHECKPOINT("LevelSystem signature set");
    levelSystem->Initialize();
    std::cout << "LevelSystem initialized" << std::endl;
    CHECKPOINT("LevelSystem initialized");
    
    // Register TargetingSystem (for combat lock-on)
    CHECKPOINT("About to register TargetingSystem");
    targetingSystem = coordinator.RegisterSystem<Gameplay::TargetingSystem>();
    CHECKPOINT("TargetingSystem registered");
    Core::Signature targetingSignature;
    targetingSignature.set(coordinator.GetComponentType<Gameplay::TargetingComponent>());
    CHECKPOINT("TargetingComponent type retrieved");
    coordinator.SetSystemSignature<Gameplay::TargetingSystem>(targetingSignature);
    CHECKPOINT("TargetingSystem signature set");
    targetingSystem->Initialize();
    std::cout << "TargetingSystem initialized" << std::endl;
    CHECKPOINT("TargetingSystem initialized");
    
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
        5000.0f
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


    std::cerr << "[Main] Initializing D3D12 RenderPipeline..." << std::endl;
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
    std::cerr << "[Main] D3D12 pipeline initialized SUCCESS. Next: CharacterFactory" << std::endl;
    std::cout << "D3D12 pipeline initialized" << std::endl;

    // Initialize CharacterFactory
    std::cerr << "[Main] Creating CharacterFactory..." << std::endl;
    characterFactory = std::make_unique<Gameplay::CharacterFactory>();
    
    std::cerr << "[Main] Initializing CharacterFactory..." << std::endl;
    characterFactory->Initialize();
    std::cout << "CharacterFactory initialized" << std::endl;

    // Create game world with ECS entities
    // Initialize procedural building generator
    std::cerr << "[Main] Initializing BuildingGenerator..." << std::endl;
    buildingGenerator = std::make_unique<CudaBuildingGenerator>();
    if (!buildingGenerator->Initialize()) {
        std::cerr << "Failed to initialize building generator" << std::endl;
    } else {
        std::cerr << "[Main] Building generator initialized" << std::endl;
    }
    
    // --- Manual Procedural Animation Setup for Debugging ---
    /*
    {
        auto procAnim = std::make_unique<Animation::AnimationClip>();
        procAnim->name = "ProceduralWave";
        // ... (lines omitted for brevity, block disabled) ...
        if (animationSystem) {
             animationSystem->registerAnimationClip(std::move(procAnim));
             std::cout << "[Setup] Registered 'ProceduralWave' animation." << std::endl;
        }
    }
    */
    // -----------------------------------------------------

    std::cout << "[Setup] Calling CreateGameWorld..." << std::endl;
    CreateGameWorld();
    std::cout << "[Setup] CreateGameWorld FINISHED." << std::endl;
    
    // Initialize WorldChunkManager for chunk-based streaming
    // NOTE: Building generation disabled in callbacks to avoid overwhelming DX12
    // Buildings are pre-generated in CreateGameWorld for stability
    std::cout << "[Setup] Initializing ChunkManager..." << std::endl;
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
    std::cout << "[Setup] Syncing entities to renderer..." << std::endl;
    SyncEntitiesToRenderer();
    std::cout << "[Setup] SyncEntitiesToRenderer FINISHED." << std::endl;
    std::cout << "Game world synced to renderer: " << renderPipeline->GetMeshCount() << " meshes" << std::endl;
    
    // Take initial verification screenshot (Frame 0)
    std::cerr << "[Verify] Taking Frame 0 Screenshot..." << std::endl;
    renderPipeline->RenderFrame(); // Render once to populate buffers
    renderPipeline->SaveScreenshot("frame_0_init.bmp");
    std::cerr << "[Verify] Frame 0 Saved." << std::endl;

    // === UNIFIED TESTING SYSTEM ===
    std::cerr << "[TestSystem] Initializing..." << std::endl;
    auto& testSystem = CudaGame::Testing::TestSystem::GetInstance();
    std::cerr << "[TestSystem] Instance acquired." << std::endl;
    
    // Scenario 1: Procedural Walk (Visually verify leg movement)
    std::cerr << "[TestSystem] Registering Scenario: ProceduralWalk" << std::endl;
    testSystem.RegisterScenario("ProceduralWalk", 4.0f, [&](){
        std::cout << "[TestSetup] Setting up Procedural Walk..." << std::endl;
        if (playerEntity != 0 && coordinator.HasComponent<Animation::AnimationComponent>(playerEntity)) {
             auto& anim = coordinator.GetComponent<Animation::AnimationComponent>(playerEntity);
             anim.useProceduralGeneration = true;
             anim.currentAnimation = "Procedural_Walk"; 
             anim.proceduralPhase = 0.0f; 
        }
        // Force Camera to side view
        if (mainCamera) {
            mainCamera->SetCameraMode(OrbitCamera::CameraMode::FREE_LOOK);
            mainCamera->SetViewAngles(0.0f, 20.0f); // Yaw, Pitch
            mainCamera->SetDistance(5.0f);
        }
    });

    // Check Arguments
    std::cerr << "[TestSystem] Checking arguments..." << std::endl;
    for(int i=1; i < __argc; ++i) {
        if (std::string(__argv[i]) == "--test-suite") {
             if (i+1 < __argc) {
                 std::cerr << "[TestSystem] Starting suite: " << __argv[i+1] << std::endl;
                 testSystem.StartSuite(__argv[i+1]);
             }
        }
    }
    std::cerr << "[TestSystem] Initialization done." << std::endl;
    
    // (Debug-only initial entity dump removed to keep DX12 output clean.)
    std::cerr << "[Main] About to enter main loop..." << std::endl;

    // Main game loop
    std::cerr << "[CHECKPOINT] All initialization complete. Entering main loop..." << std::endl;
    std::cout << "Press ESC to exit, TAB for mouse capture..." << std::endl;
    
    // Fixed timestep physics (matches OpenGL version)
    const float FIXED_TIMESTEP = 1.0f / 60.0f;
    float accumulator = 0.0f;
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    frameNumber = 0;
    
    std::cerr << "[Main] Starting while loop..." << std::endl;
    while (!glfwWindowShouldClose(window)) {
         static int logSafeGuard = 0;
         if (logSafeGuard < 10) std::cerr << "[Loop] Frame " << frameNumber << " Start" << std::endl;

        // Calculate delta time
        if (logSafeGuard < 10) std::cerr << "[Loop] Calculating deltaTime..." << std::endl;
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
        lastFrameTime = currentTime;
        
        // Cap delta time to avoid spiraling on lag spikes
        if (deltaTime > 0.1f) deltaTime = 0.1f;
        
        frameTime = deltaTime;
        fpsAccumulator += deltaTime;
        frameCount++;
        
        // Update FPS counter every 0.5s
        if (fpsAccumulator >= 0.5f) {
            fps = frameCount / fpsAccumulator;
            frameCount = 0;
            fpsAccumulator = 0.0f;
            
            if (showFPS) {
                // std::cout << "\rFPS: " << static_cast<int>(fps) << " | Meshes: " << renderPipeline->GetMeshCount() << std::flush;
            }
        }
        
        if (logSafeGuard < 10) std::cerr << "[Loop] Polling GLFW events..." << std::endl;
        glfwPollEvents();
        
        
        // Update input component from GLFW
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating input..." << std::endl;
        UpdateInputComponent(playerEntity);
        if (logSafeGuard < 10) std::cerr << "[Loop] Input Updated" << std::endl;

        // Handle player movement via CharacterControllerSystem (camera-relative movement)
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating CharacterController..." << std::endl;
        if (mouseCaptured && characterControllerSystem) {
            characterControllerSystem->Update(deltaTime);
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Controller Updated" << std::endl;
        
        // Update gameplay systems (matching OpenGL game loop order)
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating EnemyAI..." << std::endl;
        if (enemyAISystem) {
            enemyAISystem->Update(deltaTime);
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating LevelSystem..." << std::endl;
        if (levelSystem) {
            levelSystem->Update(deltaTime);
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating TargetingSystem..." << std::endl;
        if (targetingSystem) {
            targetingSystem->Update(deltaTime);
        }
        // PlayerMovementSystem DISABLED - conflicts with CharacterControllerSystem
        // Both systems were modifying player velocity, causing microsliding
        // CharacterControllerSystem alone handles all player movement now
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating WallRunning..." << std::endl;
        if (wallRunningSystem) {
            wallRunningSystem->Update(deltaTime);
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Gameplay Systems Updated" << std::endl;
        
        // Update Animation System
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating Animation..." << std::endl;
        if (animationSystem) {
             if (logSafeGuard < 10) std::cerr << "[Loop] Calling animationSystem->update()..." << std::endl;
             animationSystem->update(deltaTime);
             if (logSafeGuard < 10) std::cerr << "[Loop] animationSystem->update() done." << std::endl;
             
             // SYNC ANIMATION TO RENDERER
             // Copy computed bone matrices from ECS to D3D12Mesh resources so the renderer can upload them
             static int syncLogCounter = 0;
             bool shouldLog = (syncLogCounter++ % 60 == 0); // Log once per second
             
             if (logSafeGuard < 10) std::cerr << "[Loop] Syncing bone matrices to renderer..." << std::endl;
             for (Core::Entity entity : renderEntities) {
                 if (coordinator.HasComponent<Animation::AnimationComponent>(entity)) {
                      auto& animComp = coordinator.GetComponent<Animation::AnimationComponent>(entity);
                      
                      // Find corresponding mesh resource
                      auto it = entityMeshes.find(entity);
                      if (it != entityMeshes.end() && it->second) {
                          D3D12Mesh* mesh = it->second.get();
                          
                          // Only update if we have valid data
                          if (!animComp.finalBoneMatrices.empty()) {
                              mesh->boneMatrices = animComp.finalBoneMatrices;
                          }
                      }
                 }
             }
             if (logSafeGuard < 10) std::cerr << "[Loop] Bone matrices synced." << std::endl;
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Animation Updated/Synced" << std::endl;
        
        // Update chunk streaming based on player position
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating ChunkManager..." << std::endl;
        if (chunkManager && playerEntity != 0) {
            auto& playerTransform = coordinator.GetComponent<Rendering::TransformComponent>(playerEntity);
            chunkManager->Update(playerTransform.position, frameNumber);
            chunkManager->ProcessCompletedChunks();
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Chunks Updated" << std::endl;
        ++frameNumber;

        // Camera mode toggles (match GL renderer behavior: 1=ORBIT_FOLLOW, 2=FREE_LOOK, 3=COMBAT_FOCUS, 4=THIRD_PERSON)
        if (logSafeGuard < 10) std::cerr << "[Loop] Handling camera mode..." << std::endl;
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
        if (logSafeGuard < 10) std::cerr << "[Loop] Camera mode done." << std::endl;
        
        // Fixed timestep physics update (matches OpenGL version - fixes speed bursting)
        // This runs physics at consistent 60 FPS regardless of frame rate
        if (logSafeGuard < 10) std::cerr << "[Loop] Physics update..." << std::endl;
        accumulator += deltaTime;
        while (accumulator >= FIXED_TIMESTEP) {
            if (physicsSystem) {
                if (logSafeGuard < 10) std::cerr << "[Loop] Calling physicsSystem->Update()..." << std::endl;
                physicsSystem->Update(FIXED_TIMESTEP);
                if (logSafeGuard < 10) std::cerr << "[Loop] physicsSystem->Update() done." << std::endl;
            }
            accumulator -= FIXED_TIMESTEP;
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Physics loop done." << std::endl;
        
        
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
        if (logSafeGuard < 10) std::cerr << "[Loop] Getting player position from PhysX..." << std::endl;
        
        // Update camera with player position/velocity from PhysX (avoids ECS-PhysX sync jitter)
        glm::vec3 playerPos(0.0f);
        glm::vec3 playerVel(0.0f);
        
        // Read position directly from PhysX actor (not ECS) to avoid sync jitter
        if (logSafeGuard < 10) std::cerr << "[Loop] Finding playerEntity in entityPhysicsActors..." << std::endl;
        auto physIt = entityPhysicsActors.find(playerEntity);
        if (logSafeGuard < 10) std::cerr << "[Loop] Found: " << (physIt != entityPhysicsActors.end()) << std::endl;
        if (physIt != entityPhysicsActors.end() && physIt->second) {
            if (logSafeGuard < 10) std::cerr << "[Loop] Accessing PhysX actor..." << std::endl;
            PxRigidActor* actor = physIt->second;
            if (logSafeGuard < 10) std::cerr << "[Loop] Calling getGlobalPose()..." << std::endl;
            PxTransform pxTrans = actor->getGlobalPose();
            if (logSafeGuard < 10) std::cerr << "[Loop] Got transform." << std::endl;
            playerPos = glm::vec3(pxTrans.p.x, pxTrans.p.y, pxTrans.p.z);
            
            // Get velocity if dynamic actor
            if (PxRigidDynamic* dyn = actor->is<PxRigidDynamic>()) {
                PxVec3 v = dyn->getLinearVelocity();
                playerVel = glm::vec3(v.x, v.y, v.z);
            }
        } else {
            // Fallback to ECS position if no physics actor
            if (logSafeGuard < 10) std::cerr << "[Loop] No physics actor, using ECS fallback..." << std::endl;
            if (coordinator.HasComponent<Rendering::TransformComponent>(playerEntity)) {
                playerPos = coordinator.GetComponent<Rendering::TransformComponent>(playerEntity).position;
            }
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Player position retrieved." << std::endl;
        
        // ====== BOUNDED CAMERA FOLLOW SYSTEM ======
        // Based on RE Engine (damped spring-arm) + Unreal (offset clamping)
        if (logSafeGuard < 10) std::cerr << "[Loop] Camera follow system..." << std::endl;
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
        if (logSafeGuard < 10) std::cerr << "[Loop] Updating camera..." << std::endl;
        if (mainCamera) {
            mainCamera->Update(deltaTime, smoothedPlayerPos, playerVel);
        }
        if (logSafeGuard < 10) std::cerr << "[Loop] Camera updated." << std::endl;
        
        // Sync ECS entities to renderer (in full game, only do this when entities change)
        if (logSafeGuard < 10) std::cerr << "[Loop] Syncing Entities..." << std::endl;
        SyncEntitiesToRenderer();
        if (logSafeGuard < 10) std::cerr << "[Loop] Entities Synced." << std::endl;

        // Render frame
        if (logSafeGuard < 10) std::cerr << "[Loop] BeginFrame..." << std::endl;
        renderPipeline->BeginFrame(mainCamera.get());
        
        if (logSafeGuard < 10) std::cerr << "[Loop] RenderFrame..." << std::endl;
        renderPipeline->RenderFrame();
        
        if (logSafeGuard < 10) std::cerr << "[Loop] EndFrame..." << std::endl;
        renderPipeline->EndFrame();
        
        if (logSafeGuard < 10) std::cerr << "[Loop] Frame Finished." << std::endl;
        logSafeGuard++;
        
        // Auto-Screenshot after 20 frames to prove it runs
        if (frameCount == 20) {
             std::cout << "[Auto-Verify] Taking Screenshot..." << std::endl;
             renderPipeline->SaveScreenshot("verification_screenshot.bmp");
        }

        // === TEST SYSTEM UPDATE ===
        if (testSystem.IsActive()) {
            testSystem.Update(deltaTime);
            if (testSystem.ShouldCaptureFrame()) {
                 std::string filename = "TestResults/" + testSystem.GetCurrentTestName() + "/frame_" + std::to_string(testSystem.GetFrameCounter()) + ".bmp";
                 renderPipeline->SaveScreenshot(filename);
            }
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
