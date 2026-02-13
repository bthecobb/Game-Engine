#ifdef _WIN32
#include "Core/Coordinator.h"
#include "Rendering/DX12RenderPipeline.h"
#include "Rendering/D3D12Mesh.h"
#include "Rendering/RenderComponents.h"
#include "Animation/AnimationSystem.h"
#include "Animation/AnimationComponent.h"
#include "Gameplay/CharacterControllerSystem.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Physics/PhysicsComponents.h"
#include "Gameplay/PlayerComponents.h"
#include "Gameplay/CombatSystem.h"
#include "Gameplay/CombatComponents.h"
#include "Gameplay/AnimationControllerComponent.h" // Added
#include "AI/AIComponent.h" // Added
#include "Gameplay/CharacterFactory.h"
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>

using namespace CudaGame;
using namespace CudaGame::Rendering;

// Constants
const unsigned int WINDOW_WIDTH = 1280;
const unsigned int WINDOW_HEIGHT = 720;

// Globals
GLFWwindow* window = nullptr;
Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
std::unique_ptr<DX12RenderPipeline> renderPipeline;
std::unique_ptr<CudaGame::Animation::AnimationSystem> animationSystem;
std::unique_ptr<CudaGame::Physics::PhysXPhysicsSystem> physicsSystem;
std::unique_ptr<CudaGame::Gameplay::CharacterControllerSystem> charControllerSystem;

// Input State
bool keys[1024] = {false};
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) keys[key] = true;
        else if (action == GLFW_RELEASE) keys[key] = false;
    }
}

// Skeleton Factory
std::shared_ptr<CudaGame::Animation::Skeleton> CreatePrismaticSkeleton() {
    auto skeleton = std::make_shared<CudaGame::Animation::Skeleton>();
    skeleton->bones.resize(2);
    
    // Bone 0: Root (Static)
    skeleton->bones[0].name = "Root";
    skeleton->bones[0].parentIndex = -1;
    skeleton->bones[0].inverseBindPose = glm::mat4(1.0f);
    skeleton->boneNameToIndex["Root"] = 0;
    
    // Bone 1: Top (Moving)
    skeleton->bones[1].name = "Top";
    skeleton->bones[1].parentIndex = 0;
    skeleton->bones[1].inverseBindPose = glm::translate(glm::vec3(0, -0.5f, 0)); // Bind pose at local 0.5
    skeleton->boneNameToIndex["Top"] = 1;
    
    return skeleton;
}

// Procedural Animation Clip
void CreateTestClips(CudaGame::Animation::AnimationSystem* sys) {
    auto clip = std::make_unique<CudaGame::Animation::AnimationClip>();
    clip->name = "Bend";
    clip->duration = 2.0f;
    clip->isLooping = true;
    
    // Animate "Top" bone
    CudaGame::Animation::AnimationClip::Channel ch;
    ch.boneName = "Top";
    
    // Keyframes
    ch.times = {0.0f, 1.0f, 2.0f};
    ch.positions = {glm::vec3(0,0.5,0), glm::vec3(0.5, 0.5, 0), glm::vec3(0, 0.5, 0)}; // Move Right and Back
    ch.rotations = {glm::quat(1,0,0,0), glm::quat(glm::vec3(0,0, -0.5)), glm::quat(1,0,0,0)};
    ch.scales = {glm::vec3(1), glm::vec3(1), glm::vec3(1)};
    
    clip->channels.push_back(ch);
    sys->registerAnimationClip(std::move(clip));
}

namespace DemoMeshGen {
    std::unique_ptr<D3D12Mesh> CreateSkinnedPrism(DX12RenderBackend* backend) {
        std::cout << "[DemoMeshGen] Creating Skinned Prism..." << std::endl;
        auto mesh = std::make_unique<D3D12Mesh>();
        
        std::vector<Rendering::SkinnedVertex> vertices;
        std::vector<uint32_t> indices;
        
        // Tall Cube: -0.5 to 1.5 Y.
        // Bottom (0.0 bind) -> Bone 0
        // Top (1.0 bind) -> Bone 1
        
        struct RawVert { glm::vec3 p; float w1; };
        std::vector<RawVert> rawVerts;
        
        // Generate a tessellated tower
        const int Y_SEGMENTS = 10;
        for (int y = 0; y <= Y_SEGMENTS; ++y) {
            float fy = (float)y / Y_SEGMENTS; // 0..1
            float posY = fy * 2.0f - 0.5f; // -0.5 to 1.5
            
            // Bone Weight: Linear gradient
            float w1 = fy; 
            if (fy < 0.2f) w1 = 0.0f;
            if (fy > 0.8f) w1 = 1.0f;
            
            // Quad ring
            float r = 0.2f;
            vertices.push_back(Rendering::SkinnedVertex(
                Rendering::Vertex(glm::vec3(-r, posY, -r), glm::vec3(0,0,-1), glm::vec3(1,0,0), glm::vec2(0, fy)),
                glm::ivec4(0, 1, 0, 0), glm::vec4(1.0f - w1, w1, 0, 0)
            ));
            vertices.push_back(Rendering::SkinnedVertex(
                Rendering::Vertex(glm::vec3( r, posY, -r), glm::vec3(0,0,-1), glm::vec3(1,0,0), glm::vec2(1, fy)),
                glm::ivec4(0, 1, 0, 0), glm::vec4(1.0f - w1, w1, 0, 0)
            ));
            vertices.push_back(Rendering::SkinnedVertex(
                Rendering::Vertex(glm::vec3( r, posY,  r), glm::vec3(0,0, 1), glm::vec3(1,0,0), glm::vec2(1, fy)),
                glm::ivec4(0, 1, 0, 0), glm::vec4(1.0f - w1, w1, 0, 0)
            ));
            vertices.push_back(Rendering::SkinnedVertex(
                Rendering::Vertex(glm::vec3(-r, posY,  r), glm::vec3(0,0, 1), glm::vec3(1,0,0), glm::vec2(0, fy)),
                glm::ivec4(0, 1, 0, 0), glm::vec4(1.0f - w1, w1, 0, 0)
            ));
            
            if (y < Y_SEGMENTS) {
                int base = y * 4;
                // 4 faces
                const int ring = 4;
                for (int k=0; k<4; ++k) {
                    int n = (k+1)%4;
                    indices.push_back(base + k);
                    indices.push_back(base + k + ring);
                    indices.push_back(base + n);
                    
                    indices.push_back(base + n);
                    indices.push_back(base + k + ring);
                    indices.push_back(base + n + ring);
                }
            }
        }

        mesh->CreateSkinned(backend, vertices, indices, "SkinnedPrism");
        
        // Cyan Material
        mesh->GetMaterial().albedoColor = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
        mesh->GetMaterial().roughness = 0.4f;
        mesh->GetMaterial().metallic = 0.1f;
        
        return mesh;
    }
}

int main() {
    std::cout << std::unitbuf;
    std::cout << "[IntegratedAnimationDemo] Starting..." << std::endl;

    // 1. Init Window
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Integrated Animation Demo", nullptr, nullptr);
    if (!window) return -1;
    glfwSetKeyCallback(window, key_callback);

    // 2. Init ECS & Components
    coordinator.Initialize();
    coordinator.RegisterComponent<CudaGame::Animation::AnimationComponent>();
    coordinator.RegisterComponent<CudaGame::Physics::RigidbodyComponent>();
    coordinator.RegisterComponent<CudaGame::Physics::CharacterControllerComponent>();
    coordinator.RegisterComponent<CudaGame::Gameplay::PlayerInputComponent>();
    coordinator.RegisterComponent<CudaGame::Gameplay::PlayerMovementComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::TransformComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::MaterialComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::MeshComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::Gameplay::AnimationControllerComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::Gameplay::CombatComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::AI::AIComponent>(); // Added

    // 3. Init Systems
    // Physics
    physicsSystem = std::make_unique<CudaGame::Physics::PhysXPhysicsSystem>();
    physicsSystem->Initialize();
    
    // Animation
    animationSystem = std::make_unique<CudaGame::Animation::AnimationSystem>();
    animationSystem->Initialize();
    CreateTestClips(animationSystem.get());
    
    // Character Controller
    charControllerSystem = std::make_unique<CudaGame::Gameplay::CharacterControllerSystem>();
    charControllerSystem->Initialize();
    
    // Character Factory (Phase 5)
    auto characterFactory = std::make_unique<CudaGame::Gameplay::CharacterFactory>();
    characterFactory->Initialize();
    
    // Combat System (Phase 6)
    auto combatSystem = std::make_unique<CudaGame::Gameplay::CombatSystem>();
    combatSystem->Initialize();

    // 4. Init Rendering
    renderPipeline = std::make_unique<DX12RenderPipeline>();
    DX12RenderPipeline::InitParams params = {};
    params.windowHandle = window;
    params.displayWidth = WINDOW_WIDTH;
    params.displayHeight = WINDOW_HEIGHT;
    params.enableDLSS = false; 
    params.enableRayTracing = false;
    renderPipeline->Initialize(params);
    
    // Link Pipeline to Factory (for mesh creation if implemented, currently mocked)
    characterFactory->SetRenderPipeline(renderPipeline.get());

    // 5. Setup Assets & Profiles
    auto sharedSkeleton = CreatePrismaticSkeleton();
    characterFactory->RegisterSkeleton("PrismSkeleton", sharedSkeleton);
    
    // Weapon Definition
    CudaGame::Gameplay::WeaponDefinition swordDef;
    swordDef.name = "IronSword";
    swordDef.damage = 25.0f;
    swordDef.maxAmmo = 0; // Infinite
    characterFactory->RegisterWeaponDefinition("IronSword", swordDef);
    
    CudaGame::Gameplay::CharacterProfile profile;
    profile.profileName = "PrismGuard";
    profile.skeletonID = "PrismSkeleton";
    profile.animSetID = "WarriorSet"; // Default set registered in Factory::Initialize
    profile.startingWeaponID = "IronSword";
    profile.runSpeed = 8.0f;
    profile.colliderRadius = 0.5f;
    profile.colliderHeight = 2.0f;
    characterFactory->RegisterProfile("PrismGuard", profile);

    // 6. Spawn Characters
    std::vector<Core::Entity> entities;
    
    // Player
    Core::Entity playerID = characterFactory->SpawnCharacter("PrismGuard", glm::vec3(0, 0, 0));
    entities.push_back(playerID);
    std::cout << "[Demo] Player Spawned. Adding Input..." << std::endl;
    // Add Input Component manually (Factory doesn't add Input by default, only AI/Movement)
    coordinator.AddComponent(playerID, CudaGame::Gameplay::PlayerInputComponent{});
    std::cout << "[Demo] Player Input Added" << std::endl;
    
    // NPC (Dummy Target)
    Core::Entity npcID = characterFactory->SpawnCharacter("PrismGuard", glm::vec3(1.5, 0, 0));
    // Give NPC some velocity or logic? For now just static.
    entities.push_back(npcID);

    // 7. Create Meshes (Bridging ECS -> Renderer)
    std::unordered_map<Core::Entity, D3D12Mesh*> entityToMesh;
    std::vector<std::unique_ptr<D3D12Mesh>> meshes; // Owner
    
    // Helper to add mesh for entity (Needs to run every frame for new projectiles? Or just static list?)
    // For this demo, we assume static list of characters.
    // BUT we spawned weapons! 
    // We need to iterate ALL entities with Transform + Material to create meshes.
    
    // Let's optimize: Loop all entities ID 0 to 1000.
    // If has Transform + Material + NO Mesh yet -> Create Mesh.
    // This is a mini-RenderSystem.
    
    auto RefreshMeshes = [&]() {
        std::cout << "[Demo] RefreshMeshes Start" << std::endl;
        for (Core::Entity e = 0; e < 1000; ++e) {
            if (coordinator.HasComponent<Rendering::TransformComponent>(e) && 
                coordinator.HasComponent<Rendering::MaterialComponent>(e)) {
                
                if (entityToMesh.find(e) == entityToMesh.end()) {
                    // New Entity needs mesh
                    std::unique_ptr<D3D12Mesh> mesh;
                    if (coordinator.HasComponent<CudaGame::Gameplay::WeaponComponent>(e)) {
                        // Weapon -> Red Box
                        mesh = DemoMeshGen::CreateSkinnedPrism(renderPipeline->GetBackend()); 
                        mesh->GetMaterial().albedoColor = glm::vec4(1, 0, 0, 1);
                        mesh->transform = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 1.0f, 0.2f)); // Thin blade
                    } else {
                        // Character -> Prism
                        mesh = DemoMeshGen::CreateSkinnedPrism(renderPipeline->GetBackend());
                        if (coordinator.HasComponent<CudaGame::Animation::AnimationComponent>(e)) {
                            auto& animComp = coordinator.GetComponent<CudaGame::Animation::AnimationComponent>(e);
                            mesh->SetSkeleton(animComp.skeleton);
                        }
                    }

                    
                    entityToMesh[e] = mesh.get();
                    renderPipeline->AddMesh(mesh.get());
                    meshes.push_back(std::move(mesh));
                }
            }
        }
        std::cout << "[Demo] RefreshMeshes End" << std::endl;
    };
    
    RefreshMeshes();

    std::cout << "[Demo] Loop Start. Player ID: " << playerID << " NPC ID: " << npcID << std::endl;

    // Particle System (Phase 7)
    // auto particleSystem = std::make_unique<CudaGame::VFX::ParticleSystem>(100000); // 100k particles
    // if (!particleSystem->Initialize(renderPipeline.get())) {
    //     std::cerr << "[Demo] Failed to initialize Particle System" << std::endl;
    // }
    
    // ...
    
    // 7. Main Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (keys[GLFW_KEY_ESCAPE]) glfwSetWindowShouldClose(window, true);

        // --- Logic ---
        float dt = 0.016f;
        
        // Setup Player Input
        if (coordinator.HasComponent<CudaGame::Gameplay::PlayerInputComponent>(playerID)) {
            // ... (Existing input logic)
            auto input = coordinator.GetComponent<CudaGame::Gameplay::PlayerInputComponent>(playerID);
            // ... keys ...
            int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
            if (state == GLFW_PRESS) {
                input.mouseButtons[0] = true;
                // Verify VFX: Spawn particles on click
                // Get player pos
                auto& t = coordinator.GetComponent<CudaGame::Rendering::TransformComponent>(playerID);
                // particleSystem->SpawnBurst(t.position + glm::vec3(0,1,0), 10, glm::vec4(1, 0.5, 0, 1)); // Fire!
            } else {
                input.mouseButtons[0] = false;
            }
            coordinator.GetComponent<CudaGame::Gameplay::PlayerInputComponent>(playerID) = input;
        }
        
        // --- Update Systems ---
        physicsSystem->Update(dt);
        // ... systems ...
        charControllerSystem->Update(dt);
        combatSystem->Update(dt);
        animationSystem->Update(dt);
        
        // Update VFX
        // particleSystem->Update(dt);
        
        // Ensure Meshes ... 
        RefreshMeshes();
        
        // Sync Transforms ... (Existing loop)
        for (auto const& [e, mesh] : entityToMesh) {
             // ...
             if (coordinator.HasComponent<CudaGame::Rendering::TransformComponent>(e)) {
                auto& t = coordinator.GetComponent<CudaGame::Rendering::TransformComponent>(e);
                if (coordinator.HasComponent<CudaGame::Gameplay::WeaponComponent>(e)) {
                     mesh->transform = glm::translate(glm::mat4(1.0f), t.position) * glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 1.0f, 0.2f));
                } else {
                     mesh->transform = glm::translate(glm::mat4(1.0f), t.position);
                }
            }
        }

        // --- Render ---
        // Scene meshes are already added via RefreshMeshes -> AddMesh

        // Camera
        static Camera camera(ProjectionType::PERSPECTIVE);
        camera.SetPosition(glm::vec3(0, 2, 8));
        camera.LookAt(glm::vec3(1.5, 0, 0));
        
        std::cout << "[Demo] Frame Start" << std::endl;
        renderPipeline->BeginFrame(&camera);
        std::cout << "[Demo] BeginFrame Done" << std::endl;
        renderPipeline->RenderFrame();
        std::cout << "[Demo] RenderFrame Done" << std::endl;
        renderPipeline->EndFrame();
        std::cout << "[Demo] EndFrame Done" << std::endl;
    }

    renderPipeline->Shutdown();
    glfwTerminate();
    return 0;
}
#endif
