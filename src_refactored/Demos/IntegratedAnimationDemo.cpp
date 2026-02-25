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
#include "Gameplay/AnimationControllerComponent.h"
#include "Gameplay/AnimationControllerSystem.h" // Added
#include "Animation/IKSystem.h" // Added IKSystem
#include "AI/AIComponent.h"
#include "Gameplay/LevelComponents.h" // Added for WallComponent
#include "Gameplay/CharacterFactory.h"
#include "Animation/AnimationBuilder.h"
#include "Animation/BlendTree.h"
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
std::shared_ptr<CudaGame::Animation::AnimationSystem> animationSystem;
std::shared_ptr<CudaGame::Physics::PhysXPhysicsSystem> physicsSystem;
std::shared_ptr<CudaGame::Gameplay::CharacterControllerSystem> charControllerSystem;
std::shared_ptr<CudaGame::Gameplay::AnimationControllerSystem> animControllerSystem; // Added
std::shared_ptr<CudaGame::Animation::IKSystem> ikSystem; // Added IKSystem

// Input State
bool keys[1024] = {false};
void key_callback(GLFWwindow* /*window*/, int key, int scancode, int action, int mode) {
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
        
        std::vector<Rendering::Vertex> vertices;
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
            vertices.push_back(Rendering::Vertex(
                glm::vec3(-r, posY, -r), glm::vec3(0,0,-1), glm::vec3(1,0,0), glm::vec2(0, fy), glm::vec4(1.0f),
                glm::ivec4(0, 1, 0, 0), glm::vec4(1.0f - w1, w1, 0, 0)
            ));
            vertices.push_back(Rendering::Vertex(
                glm::vec3( r, posY, -r), glm::vec3(0,0,-1), glm::vec3(1,0,0), glm::vec2(1, fy), glm::vec4(1.0f),
                glm::ivec4(0, 1, 0, 0), glm::vec4(1.0f - w1, w1, 0, 0)
            ));
            vertices.push_back(Rendering::Vertex(
                glm::vec3( r, posY,  r), glm::vec3(0,0, 1), glm::vec3(1,0,0), glm::vec2(1, fy), glm::vec4(1.0f),
                glm::ivec4(0, 1, 0, 0), glm::vec4(1.0f - w1, w1, 0, 0)
            ));
            vertices.push_back(Rendering::Vertex(
                glm::vec3(-r, posY,  r), glm::vec3(0,0, 1), glm::vec3(1,0,0), glm::vec2(0, fy), glm::vec4(1.0f),
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

    std::unique_ptr<D3D12Mesh> CreateProceduralHumanoidMesh(DX12RenderBackend* backend, const CudaGame::Animation::Skeleton& skeleton) {
        std::cout << "[DemoMeshGen] Creating Procedural Humanoid..." << std::endl;
        auto mesh = std::make_unique<D3D12Mesh>();
        
        std::vector<Rendering::Vertex> vertices;
        std::vector<uint32_t> indices;
        
        int vOffset = 0;
        
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            const auto& bone = skeleton.bones[i];
            glm::mat4 bindFromInv = glm::inverse(bone.inverseBindPose);
            
            // Determine box size based on name
            glm::vec3 boxSize(0.1f); // default joint size
            if (bone.name == "Hips") boxSize = glm::vec3(0.3f, 0.15f, 0.2f);
            else if (bone.name == "Spine" || bone.name == "Chest") boxSize = glm::vec3(0.25f, 0.2f, 0.15f);
            else if (bone.name == "Head") boxSize = glm::vec3(0.2f, 0.25f, 0.2f);
            else if (bone.name.find("Leg") != std::string::npos) boxSize = glm::vec3(0.12f, 0.4f, 0.12f);
            else if (bone.name.find("Arm") != std::string::npos) boxSize = glm::vec3(0.3f, 0.1f, 0.1f); 
            
            // Helper to add a box
            glm::vec3 half = boxSize * 0.5f;
            
            // 8 corners
            glm::vec3 corners[8] = {
                {-half.x, -half.y, -half.z}, { half.x, -half.y, -half.z},
                { half.x,  half.y, -half.z}, {-half.x,  half.y, -half.z},
                {-half.x, -half.y,  half.z}, { half.x, -half.y,  half.z},
                { half.x,  half.y,  half.z}, {-half.x,  half.y,  half.z}
            };
            
            // Center in Model Space (for normals)
            glm::vec3 center = glm::vec3(bindFromInv * glm::vec4(0,0,0,1));
            
            // Transform corners by Bind Pose
            for (auto& p : corners) {
                p = glm::vec3(bindFromInv * glm::vec4(p, 1.0f));
            }
            
            // Push vertices
            for (int k = 0; k < 8; ++k) {
                glm::vec3 n = glm::normalize(corners[k] - center); // Sphere-like normal
                vertices.push_back(Rendering::Vertex(
                    corners[k], n, glm::vec3(1,0,0), glm::vec2(0,0), glm::vec4(1.0f),
                    glm::ivec4((int)i, 0, 0, 0), glm::vec4(1.0f, 0, 0, 0)
                ));
            }
            
            // Indices (Inverted Winding for Front Facing)
            // Original: 0,1,2 ... caused Back Facing.
            // New: 0,2,1 ...
            uint32_t cubeIndices[] = {
                // Front (+Z, Index 4-7) -> Was 4,5,6 (Out). Keep? 
                // Wait, previous analysis: 4,5,6 was +Z (Out). 
                // So Front face was visible?
                // Left face was Out.
                // Back/Right/Top/Bottom were In.
                // Let's create Double Sided geometry to be absolutely sure.
                // It doubles index count but guarantees visibility.
                
                // Set 1 (CW)
                0,1,2, 2,3,0, 4,5,6, 6,7,4, 0,4,7, 7,3,0, 1,5,6, 6,2,1, 3,2,6, 6,7,3, 0,1,5, 5,4,0,
                // Set 2 (CCW/Inverted)
                2,1,0, 0,3,2, 6,5,4, 4,7,6, 7,4,0, 0,3,7, 6,5,1, 1,2,6, 6,2,3, 3,7,6, 5,1,0, 0,4,5
            };
            
            for (int idx : cubeIndices) {
                indices.push_back(vOffset + idx);
            }
            vOffset += 8;
        }

        mesh->CreateSkinned(backend, vertices, indices, "ProceduralHumanoid");
        
        // Gray Material
        mesh->GetMaterial().albedoColor = glm::vec4(0.7f, 0.7f, 0.7f, 1.0f);
        mesh->GetMaterial().roughness = 0.5f;
        mesh->GetMaterial().metallic = 0.0f;
        
        return mesh;
    }

    std::unique_ptr<D3D12Mesh> CreateGroundPlane(DX12RenderBackend* backend) {
        std::cout << "[DemoMeshGen] Creating Ground Plane..." << std::endl;
        auto mesh = std::make_unique<D3D12Mesh>();
        
        std::vector<Rendering::Vertex> vertices;
        std::vector<uint32_t> indices;
        
        // Large Quad: -50 to +50 on XZ
        float size = 50.0f;
        float y = 0.0f;
        float uvScale = 20.0f; // Tile texture 20 times
        
        // 4 corners
        vertices.push_back(Rendering::Vertex(glm::vec3(-size, y, -size), glm::vec3(0,1,0), glm::vec3(1,0,0), glm::vec2(0, 0)));
        vertices.push_back(Rendering::Vertex(glm::vec3( size, y, -size), glm::vec3(0,1,0), glm::vec3(1,0,0), glm::vec2(uvScale, 0)));
        vertices.push_back(Rendering::Vertex(glm::vec3( size, y,  size), glm::vec3(0,1,0), glm::vec3(1,0,0), glm::vec2(uvScale, uvScale)));
        vertices.push_back(Rendering::Vertex(glm::vec3(-size, y,  size), glm::vec3(0,1,0), glm::vec3(1,0,0), glm::vec2(0, uvScale)));
        
        // 2 Triangles (CCW)
        indices = {0, 2, 1, 0, 3, 2};
        
        mesh->Create(backend, vertices, indices, "GroundPlane");
        
        // Dark Gray Material
        mesh->GetMaterial().albedoColor = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);
        mesh->GetMaterial().roughness = 0.8f;
        mesh->GetMaterial().metallic = 0.0f;
        
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
    std::cout << "[Demo] Coordinator Address: " << &coordinator << std::endl;
    coordinator.RegisterComponent<CudaGame::Animation::AnimationComponent>();
    coordinator.RegisterComponent<CudaGame::Physics::RigidbodyComponent>();
    coordinator.RegisterComponent<CudaGame::Physics::CharacterControllerComponent>();
    coordinator.RegisterComponent<CudaGame::Gameplay::PlayerInputComponent>();
    coordinator.RegisterComponent<CudaGame::Gameplay::PlayerMovementComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::TransformComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::MaterialComponent>();
    std::cout << "MeshComponent Name: " << typeid(CudaGame::Rendering::MeshComponent).name() << " Size: " << sizeof(CudaGame::Rendering::MeshComponent) << std::endl;
    coordinator.RegisterComponent<CudaGame::Rendering::MeshComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::Gameplay::AnimationControllerComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::Gameplay::CombatComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::Gameplay::WeaponComponent>(); // Added - CRITICAL FIX
    coordinator.RegisterComponent<CudaGame::AI::AIComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::Physics::ColliderComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::Gameplay::WallComponent>(); // Added
    coordinator.RegisterComponent<CudaGame::Animation::IKComponent>(); // Added - CRITICAL FIX

    // 3. Init Systems
    // Physics - MUST be first and registered
    physicsSystem = coordinator.RegisterSystem<CudaGame::Physics::PhysXPhysicsSystem>();
    physicsSystem->Initialize();
    
    // Animation
    animationSystem = coordinator.RegisterSystem<CudaGame::Animation::AnimationSystem>();
    animationSystem->Initialize();
    CreateTestClips(animationSystem.get());
    
    // IK System
    ikSystem = coordinator.RegisterSystem<CudaGame::Animation::IKSystem>();
    {
        CudaGame::Core::Signature signature;
        signature.set(coordinator.GetComponentType<CudaGame::Rendering::TransformComponent>());
        signature.set(coordinator.GetComponentType<CudaGame::Animation::AnimationComponent>());
        signature.set(coordinator.GetComponentType<CudaGame::Animation::IKComponent>());
        coordinator.SetSystemSignature<CudaGame::Animation::IKSystem>(signature);
    }
    ikSystem->Initialize();

    // Character Controller
    charControllerSystem = coordinator.RegisterSystem<CudaGame::Gameplay::CharacterControllerSystem>();
    charControllerSystem->Initialize();

    // Animation Controller
    animControllerSystem = coordinator.RegisterSystem<CudaGame::Gameplay::AnimationControllerSystem>();
    animControllerSystem->Initialize();
    
    // Set System Signatures
    {
        CudaGame::Core::Signature sig;
        sig.set(coordinator.GetComponentType<CudaGame::Physics::RigidbodyComponent>());
        sig.set(coordinator.GetComponentType<CudaGame::Physics::ColliderComponent>());
        coordinator.SetSystemSignature<CudaGame::Physics::PhysXPhysicsSystem>(sig);
    }
    {
        CudaGame::Core::Signature sig;
        sig.set(coordinator.GetComponentType<CudaGame::Animation::AnimationComponent>());
        coordinator.SetSystemSignature<CudaGame::Animation::AnimationSystem>(sig);
    }
    {
        CudaGame::Core::Signature sig;
        sig.set(coordinator.GetComponentType<CudaGame::Physics::CharacterControllerComponent>());
        sig.set(coordinator.GetComponentType<CudaGame::Physics::RigidbodyComponent>());
        sig.set(coordinator.GetComponentType<CudaGame::Rendering::TransformComponent>());
        sig.set(coordinator.GetComponentType<CudaGame::Gameplay::PlayerMovementComponent>());
        coordinator.SetSystemSignature<CudaGame::Gameplay::CharacterControllerSystem>(sig);
    }
    {
        CudaGame::Core::Signature sig;
        sig.set(coordinator.GetComponentType<CudaGame::Gameplay::AnimationControllerComponent>());
        coordinator.SetSystemSignature<CudaGame::Gameplay::AnimationControllerSystem>(sig);
    }
    
    {
        CudaGame::Core::Signature sig;
        sig.set(coordinator.GetComponentType<CudaGame::Animation::AnimationComponent>());
        sig.set(coordinator.GetComponentType<CudaGame::Rendering::TransformComponent>());
        sig.set(coordinator.GetComponentType<CudaGame::Animation::IKComponent>());
        coordinator.SetSystemSignature<CudaGame::Animation::IKSystem>(sig);
    }
    
    // Character Factory (Phase 5)
    auto characterFactory = std::make_unique<CudaGame::Gameplay::CharacterFactory>();
    characterFactory->Initialize();
    
    // --- Create Ground & Ramp for IK Verification ---
    Core::Entity groundEntity = coordinator.CreateEntity();
    CudaGame::Rendering::TransformComponent groundTransform{};
    groundTransform.position = glm::vec3(0.0f, -0.5f, 0.0f);
    coordinator.AddComponent(groundEntity, groundTransform);
    CudaGame::Physics::ColliderComponent groundCollider{};
    groundCollider.shape = CudaGame::Physics::ColliderShape::BOX;
    groundCollider.halfExtents = glm::vec3(50.0f, 0.5f, 50.0f);
    coordinator.AddComponent(groundEntity, groundCollider);
    std::cerr << "[Demo] Ground Plane Created." << std::endl;
    
    Core::Entity rampEntity = coordinator.CreateEntity();
    CudaGame::Rendering::TransformComponent rampTransform{};
    // Place a 10m long, 2m high ramp starting 5 meters in front of the character
    rampTransform.position = glm::vec3(0.0f, 1.0f, 10.0f); 
    rampTransform.rotation = glm::vec3(11.0f, 0.0f, 0.0f); // ~11 degree slope (Pitch 11)
    coordinator.AddComponent(rampEntity, rampTransform);
    CudaGame::Physics::ColliderComponent rampCollider{};
    rampCollider.shape = CudaGame::Physics::ColliderShape::BOX;
    rampCollider.halfExtents = glm::vec3(5.0f, 0.5f, 5.0f); // 10x1x10 rotated
    coordinator.AddComponent(rampEntity, rampCollider);
    std::cerr << "[Demo] Sloped Ramp Created." << std::endl;
    // ----------------------------------------------
    
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
    // Use AnimationBuilder to create procedural humanoid
    auto sharedSkeleton = CudaGame::Animation::AnimationBuilder::CreateHumanoidSkeleton();
    characterFactory->RegisterSkeleton("HumanoidSkeleton", sharedSkeleton);
    
    // Generate Procedural Clips
    animationSystem->registerAnimationClip(CudaGame::Animation::AnimationBuilder::CreateIdleClip(*sharedSkeleton));
    animationSystem->registerAnimationClip(CudaGame::Animation::AnimationBuilder::CreateWalkClip(*sharedSkeleton));
    animationSystem->registerAnimationClip(CudaGame::Animation::AnimationBuilder::CreateRunClip(*sharedSkeleton));

    // Create State Machine Logic (shared for all Prisms for now... or we can build it per entity in a setup phase)
    // Ideally CharacterFactory might handle this, or we do it manual key injection for now.

    // Weapon Definition
    CudaGame::Gameplay::WeaponDefinition swordDef;
    swordDef.name = "IronSword";
    swordDef.damage = 25.0f;
    swordDef.maxAmmo = 0; // Infinite
    characterFactory->RegisterWeaponDefinition("IronSword", swordDef);
    
    CudaGame::Gameplay::CharacterProfile profile;
    profile.profileName = "PrismGuard";
    profile.skeletonID = "HumanoidSkeleton";
    profile.animSetID = "ProceduralHumanoid";
    profile.startingWeaponID = "IronSword";
    profile.runSpeed = 8.0f;
    profile.colliderRadius = 0.5f;
    profile.colliderHeight = 2.0f;
    profile.meshID = "SkinnedPrism"; // Ensure non-empty
    std::cout << "Registering PrismGuard with MeshID: " << profile.meshID << " Len: " << profile.meshID.length() << std::endl;
    characterFactory->RegisterProfile("PrismGuard", profile);

    // 6. Spawn Characters
    std::unordered_map<Core::Entity, D3D12Mesh*> entityToMesh;
    std::vector<std::unique_ptr<D3D12Mesh>> meshes;
    std::vector<Core::Entity> entities;
    
    // Player
    Core::Entity playerID = characterFactory->SpawnCharacter("PrismGuard", glm::vec3(0, 0, 0));
    entities.push_back(playerID);
    std::cout << "[Demo] Player Spawned. Adding Input..." << std::endl;
    // Add Input Component manually (Factory doesn't add Input by default, only AI/Movement)
    coordinator.AddComponent(playerID, CudaGame::Gameplay::PlayerInputComponent{});
    std::cout << "[Demo] Player Input Added" << std::endl;
    
    // Add IK Component for Feet
    CudaGame::Animation::IKComponent ikComp;
    
    // Left Leg Chain
    CudaGame::Animation::IKChain leftLeg;
    leftLeg.name = "LeftLeg";
    leftLeg.startJointIndex = sharedSkeleton->GetBoneIndex("LeftUpLeg");
    leftLeg.endJointIndex = sharedSkeleton->GetBoneIndex("LeftFoot");
    leftLeg.iterationCount = 15;
    // Naive chain population: UpLeg -> Leg -> Foot
    int lUp = sharedSkeleton->GetBoneIndex("LeftUpLeg");
    int lKnee = sharedSkeleton->GetBoneIndex("LeftLeg");
    int lFoot = sharedSkeleton->GetBoneIndex("LeftFoot");
    if (lUp != -1) leftLeg.jointIndices.push_back(lUp);
    if (lKnee != -1) leftLeg.jointIndices.push_back(lKnee);
    if (lFoot != -1) leftLeg.jointIndices.push_back(lFoot);
    
    ikComp.AddChain(leftLeg);
    
    // Right Leg Chain
    CudaGame::Animation::IKChain rightLeg;
    rightLeg.name = "RightLeg";
    rightLeg.startJointIndex = sharedSkeleton->GetBoneIndex("RightUpLeg");
    rightLeg.endJointIndex = sharedSkeleton->GetBoneIndex("RightFoot");
    rightLeg.iterationCount = 15;
    int rUp = sharedSkeleton->GetBoneIndex("RightUpLeg");
    int rKnee = sharedSkeleton->GetBoneIndex("RightLeg");
    int rFoot = sharedSkeleton->GetBoneIndex("RightFoot");
    if (rUp != -1) rightLeg.jointIndices.push_back(rUp);
    if (rKnee != -1) rightLeg.jointIndices.push_back(rKnee);
    if (rFoot != -1) rightLeg.jointIndices.push_back(rFoot);
    
    ikComp.AddChain(rightLeg);
    
    coordinator.AddComponent(playerID, ikComp);

    // NPC (Dummy Target)
    Core::Entity npcID = characterFactory->SpawnCharacter("PrismGuard", glm::vec3(1.5, 0, 0));
    // Give NPC some velocity or logic? For now just static.
    entities.push_back(npcID);
    
    // --- Ground Plane ---
    {
        Core::Entity groundID = coordinator.CreateEntity();
        coordinator.AddComponent(groundID, CudaGame::Rendering::TransformComponent{glm::vec3(0, -0.05f, 0), glm::vec3(0), glm::vec3(1.0f)});
        coordinator.AddComponent(groundID, CudaGame::Rendering::MaterialComponent{glm::vec4(0.2f, 0.2f, 0.2f, 1.0f)});
        
        // Physics (Static/Kinematic)
        CudaGame::Physics::RigidbodyComponent rb;
        rb.mass = 0.0f; // Infinite
        rb.isKinematic = true;
        rb.useGravity = false;
        coordinator.AddComponent(groundID, rb);
        
        CudaGame::Physics::ColliderComponent col;
        col.shape = CudaGame::Physics::ColliderShape::BOX;
        col.halfExtents = glm::vec3(50.0f, 0.1f, 50.0f); // Large box
        coordinator.AddComponent(groundID, col);
        
        std::cout << "[Demo] Created Ground Entity: " << groundID << std::endl;
        
        // Create Mesh Manually (Skip RefreshMeshes logic)
        // RefreshMeshes loops 0-1000. It will check groundID too.
        // It checks if (entityToMesh.find(e) == end()).
        // So if we add it here, RefreshMeshes skips it. Perfect.
        auto groundMesh = DemoMeshGen::CreateGroundPlane(renderPipeline->GetBackend());
        // Set Transform
        groundMesh->transform = glm::translate(glm::mat4(1.0f), glm::vec3(0, -0.05f, 0));
        
        entityToMesh[groundID] = groundMesh.get();
        renderPipeline->AddMesh(groundMesh.get());
        meshes.push_back(std::move(groundMesh));
    }

    // 7. Create Meshes (Bridging ECS -> Renderer)
    
    // Helper to add mesh for entity (Needs to run every frame for new projectiles? Or just static list?)
    // For this demo, we assume static list of characters.
    // BUT we spawned weapons! 
    // We need to iterate ALL entities with Transform + Material to create meshes.
    
    // Let's optimize: Loop all entities ID 0 to 1000.
    // If has Transform + Material + NO Mesh yet -> Create Mesh.
    // This is a mini-RenderSystem.
    
    auto RefreshMeshes = [&]() {
        // std::cout << "[Demo] RefreshMeshes Start" << std::endl;
        int newMeshes = 0;
        for (Core::Entity e = 0; e < 1000; ++e) {
            if (coordinator.HasComponent<Rendering::TransformComponent>(e)) {
                 if (coordinator.HasComponent<Rendering::MaterialComponent>(e)) {
                
                    if (entityToMesh.find(e) == entityToMesh.end()) {
                        // New Entity needs mesh
                        std::unique_ptr<D3D12Mesh> mesh;
                        if (coordinator.HasComponent<CudaGame::Gameplay::WeaponComponent>(e)) {
                            // Weapon -> Red Box
                            mesh = DemoMeshGen::CreateSkinnedPrism(renderPipeline->GetBackend()); 
                            mesh->GetMaterial().albedoColor = glm::vec4(1, 0, 0, 1);
                            mesh->transform = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 1.0f, 0.2f)); // Thin blade
                        } else {
                            // Character -> Procedural Humanoid
                            if (coordinator.HasComponent<CudaGame::Animation::AnimationComponent>(e)) {
                                auto& animComp = coordinator.GetComponent<CudaGame::Animation::AnimationComponent>(e);
                                if (animComp.skeleton) { 
                                    mesh = DemoMeshGen::CreateProceduralHumanoidMesh(renderPipeline->GetBackend(), *animComp.skeleton);
                                    mesh->SetSkeleton(animComp.skeleton);
                                } else {
                                    mesh = DemoMeshGen::CreateSkinnedPrism(renderPipeline->GetBackend());
                                }
                            } else {
                                 mesh = DemoMeshGen::CreateSkinnedPrism(renderPipeline->GetBackend());
                            }
                        }

                        
                        entityToMesh[e] = mesh.get();
                        renderPipeline->AddMesh(mesh.get());
                        meshes.push_back(std::move(mesh));
                        newMeshes++;
                        std::cerr << "[DemoDebug] Created Mesh for Entity " << e << std::endl;
                    }
                 }
            }
        }
        if (newMeshes > 0) std::cout << "[Demo] RefreshMeshes Added " << newMeshes << " meshes." << std::endl;
    };
    
    RefreshMeshes();

    std::cerr << "[Demo] Loop Start. Player ID: " << playerID << " NPC ID: " << npcID << std::endl;

    // --- State Machine Setup for Player ---
    // Create Blend Tree Nodes
    auto idleClip = animationSystem->getAnimationClip("Idle");
    auto walkClip = animationSystem->getAnimationClip("Walk");
    auto runClip  = animationSystem->getAnimationClip("Run");

    std::cerr << "[Demo] SM Setup: idleClip=" << (idleClip ? "OK" : "NULL")
              << " walkClip=" << (walkClip ? "OK" : "NULL")
              << " runClip=" << (runClip ? "OK" : "NULL") << std::endl;

    if (idleClip && walkClip && runClip) {
        using namespace CudaGame::Animation;
        // Idle
        auto idleNode = std::make_shared<ClipNode>(idleClip);
        
        // Walk (Blend 1D based on Speed?) 
        // Or better: Full Locomotion BlendSpace
        // Idle (0) -> Walk (1) -> Run (2)
        auto locoMix = std::make_shared<BlendNode1D>();
        locoMix->SetBlendInput("Speed");
        locoMix->AddChild(std::make_shared<ClipNode>(idleClip), 0.0f);
        locoMix->AddChild(std::make_shared<ClipNode>(walkClip), 4.0f); // Walk at speed 4
        locoMix->AddChild(std::make_shared<ClipNode>(runClip), 8.0f);  // Run at speed 8
        
        // State Machine
        auto stateMachine = std::make_shared<AnimationStateMachine>();
        
        auto locoState = std::make_shared<AnimationGraphState>("Locomotion");
        locoState->SetRootNode(locoMix);
        stateMachine->AddState(locoState);
        
        stateMachine->SetStartState("Locomotion");
        
        // Assign to Player
        auto& animComp = coordinator.GetComponent<AnimationComponent>(playerID);
        animComp.stateMachine = stateMachine;
        
        // Enable Root Motion to prevent mesh drifting from capsule
        animationSystem->enableRootMotion(playerID, true);
        
        // --- Animation Events: Footsteps ---
        // Register named callbacks (the event system fires these by name)
        // Set up IK System dependencies
        ikSystem->SetPhysicsSystem(physicsSystem.get());

        // Wire Footstep Events to IK
        auto* rawIkSys = ikSystem.get();
        animationSystem->registerAnimationEvent("Footstep_Left", [rawIkSys, playerID]() {
            static int footstepCount = 0;
            std::cerr << "[Event] Footstep_Left fired! (frame " << footstepCount++ << ")" << std::endl;
            rawIkSys->TriggerFootstep(playerID, "LeftLeg");
        });
        animationSystem->registerAnimationEvent("Footstep_Right", [rawIkSys, playerID]() {
            static int footstepCount = 0;
            std::cerr << "[Event] Footstep_Right fired! (frame " << footstepCount++ << ")" << std::endl;
            rawIkSys->TriggerFootstep(playerID, "RightLeg");
        });
        
        // Patch the RunClip's stub events with the global callbacks so the clip fires them
        // Use a local raw pointer for lambda capture (cannot capture global shared_ptr in MSVC lambdas)
        CudaGame::Animation::AnimationSystem* animSysPtr = animationSystem.get();
        auto* runClip2 = animSysPtr->getAnimationClip("Run");
        if (runClip2) {
            for (auto& ev : runClip2->events) {
                auto evName = ev.name; // capture by value
                ev.callback = [animSysPtr, evName]() {
                    animSysPtr->triggerAnimationEvent(evName);
                };
            }
            std::cerr << "[Demo] Patched " << runClip2->events.size() << " events on Run clip." << std::endl;
        }
        
        std::cerr << "[Demo] Animation State Machine assigned to Player." << std::endl;
    }

    // Particle System (Phase 7)
    // auto particleSystem = std::make_unique<CudaGame::VFX::ParticleSystem>(100000); // 100k particles
    // if (!particleSystem->Initialize(renderPipeline.get())) {
    //     std::cerr << "[Demo] Failed to initialize Particle System" << std::endl;
    // }
    
    // ...
    
    // 7. Main Loop
    int frameCount = 0;
    while (!glfwWindowShouldClose(window)) {
        frameCount++;
        if (frameCount > 200) break; // Auto-exit for testing

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
                // ...
            } else {
                input.mouseButtons[0] = false;
            }
            coordinator.GetComponent<CudaGame::Gameplay::PlayerInputComponent>(playerID) = input;
            
            // --- Update Animation Inputs ---
            // Map Input/Velocity -> Animation Speed
            // Get Velocity
            if (coordinator.HasComponent<CudaGame::Physics::RigidbodyComponent>(playerID)) {
                auto& rb = coordinator.GetComponent<CudaGame::Physics::RigidbodyComponent>(playerID);
                float speed = glm::length(rb.velocity);
                if (speed < 0.1f) speed = 0.0f;
                
                // Hack: If we don't have physics velocity yet (kinematic?), use input
                if (speed == 0.0f) {
                     if (input.keys[GLFW_KEY_W]) speed = 4.0f;
                     if (input.keys[GLFW_KEY_LEFT_SHIFT]) speed = 8.0f;
                }
                
                // Update Animation Component properties which usually feed into SM
                if (coordinator.HasComponent<CudaGame::Animation::AnimationComponent>(playerID)) {
                    auto& ac = coordinator.GetComponent<CudaGame::Animation::AnimationComponent>(playerID);
                    ac.movementSpeed = speed; // "Speed" input in BlendTree
                    


                    // Also can set Direction
                }
            }
        }
        
        // --- Update Systems ---
        physicsSystem->Update(dt);
        // ... systems ...
        charControllerSystem->Update(dt);
        animControllerSystem->Update(dt); // Added
        combatSystem->Update(dt);
        // --- 3-Phase Speed Ramp: exercises full Idle→Walk→Run cross-fade ---
        // Phase 1 (frames   1-50) : Idle  — confirm smoothedSpeed settles from 0
        // Phase 2 (frames  51-120): Walk  — confirm blend cross-fades (≈0.12s at 8/s)
        // Phase 3 (frames 121-200): Run   — confirm footstep events fire
        if (coordinator.HasComponent<CudaGame::Animation::AnimationComponent>(playerID)) {
            auto& ac = coordinator.GetComponent<CudaGame::Animation::AnimationComponent>(playerID);
            if      (frameCount <= 50)  ac.movementSpeed = 0.0f;
            else if (frameCount <= 120) ac.movementSpeed = 4.0f;
            else                        ac.movementSpeed = 8.0f;

            // Log smoothedSpeed every 20 frames so the cross-fade ramp is visible
            if (frameCount % 20 == 0) {
                std::cerr << "[CrossFade] Frame " << frameCount
                          << "  target=" << ac.movementSpeed
                          << "  smoothed=" << ac.smoothedSpeed << std::endl;
            }
        }
        animationSystem->Update(dt);
        ikSystem->Update(dt); // Added IK Update
        
        // Update VFX
        // particleSystem->Update(dt);
        
        // Ensure Meshes ... 
        RefreshMeshes();
        
        // Sync Transforms & Animation Data
        for (auto const& [e, mesh] : entityToMesh) {
             if (coordinator.HasComponent<CudaGame::Rendering::TransformComponent>(e)) {
                auto& t = coordinator.GetComponent<CudaGame::Rendering::TransformComponent>(e);
                if (coordinator.HasComponent<CudaGame::Gameplay::WeaponComponent>(e)) {
                     mesh->transform = glm::translate(glm::mat4(1.0f), t.position) * glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 1.0f, 0.2f));
                } else {
                     mesh->transform = glm::translate(glm::mat4(1.0f), t.position);
                }
            }
            
            // Sync Animation Matrices
            if (coordinator.HasComponent<CudaGame::Animation::AnimationComponent>(e)) {
                auto& anim = coordinator.GetComponent<CudaGame::Animation::AnimationComponent>(e);
                
                if (!anim.finalBoneMatrices.empty()) {
                    // Resize mesh matrices if needed
                    if (mesh->boneMatrices.size() != anim.finalBoneMatrices.size()) {
                        mesh->boneMatrices.resize(anim.finalBoneMatrices.size());
                    }
                    // Copy
                    std::copy(anim.finalBoneMatrices.begin(), anim.finalBoneMatrices.end(), mesh->boneMatrices.begin());
                    
                    static int demoLogCounter = 0;
                    demoLogCounter++;
                    if (demoLogCounter < 120 && demoLogCounter % 60 == 0) {
                         std::cerr << "[DemoSync] Entity " << e << " Copied " << anim.finalBoneMatrices.size() << " matrices to Mesh." << std::endl;
                    }
                } else {
                     static int demoLogCounter2 = 0;
                     if (demoLogCounter2 < 120 && demoLogCounter2 % 60 == 0) {
                         std::cerr << "[DemoSync] Entity " << e << " FinalBoneMatrices EMPTY!" << std::endl;
                     }
                     demoLogCounter2++;
                }
            }
        }

        // Camera
        static Camera camera(ProjectionType::PERSPECTIVE);
        camera.SetPosition(glm::vec3(0, 2, 8));
        camera.LookAt(glm::vec3(1.5, 0, 0));

        std::cout << "[Demo] Frame " << frameCount << " Render..." << std::endl;
        renderPipeline->BeginFrame(&camera);
        // std::cout << "[Demo] BeginFrame Done" << std::endl;
        renderPipeline->RenderFrame();
        // std::cout << "[Demo] RenderFrame Done" << std::endl;
        renderPipeline->EndFrame();
        // std::cout << "[Demo] EndFrame Done" << std::endl;
    }

    renderPipeline->Shutdown();
    glfwTerminate();
    return 0;
}
#endif
