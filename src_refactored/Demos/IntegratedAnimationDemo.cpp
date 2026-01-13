#ifdef _WIN32
#include "Core/Coordinator.h"
#include "Rendering/DX12RenderPipeline.h"
#include "Rendering/D3D12Mesh.h"
#include "Rendering/RenderComponents.h"
#include "Animation/AnimationSystem.h"
#include "Animation/AnimationComponent.h"
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
const unsigned int WINDOW_WIDTH = 1920;
const unsigned int WINDOW_HEIGHT = 1080;

// Globals
GLFWwindow* window = nullptr;
Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
std::unique_ptr<DX12RenderPipeline> renderPipeline;
std::unique_ptr<CudaGame::Animation::AnimationSystem> animationSystem;
ID3D12Resource* boneBuffer = nullptr;
void* boneBufferMapped = nullptr;

// Input State
bool keys[1024] = {false};
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) keys[key] = true;
        else if (action == GLFW_RELEASE) keys[key] = false;
    }
}

// Skeleton Factory matching our Procedural Baker
std::shared_ptr<CudaGame::Animation::Skeleton> CreateHumanoidSkeleton() {
    auto skeleton = std::make_shared<CudaGame::Animation::Skeleton>();
    skeleton->bones.resize(11);
    
    // Helper
    auto addBone = [&](int id, const std::string& name, int parent, glm::vec3 pos) {
        skeleton->bones[id].name = name;
        skeleton->bones[id].parentIndex = parent;
        skeleton->bones[id].bindPosition = pos;
        skeleton->bones[id].bindRotation = glm::quat(1,0,0,0);
        skeleton->bones[id].bindScale = glm::vec3(1);
        
        // Simple Inverse Bind (assuming identity bind pose for this procedural test)
        // In a real app, we'd compute this from the bind pose world matrices
        skeleton->bones[id].inverseBindMatrix = glm::mat4(1.0f); // Simplification for demo
    };

    addBone(0, "Root", -1, glm::vec3(0, 0, 0));
    addBone(1, "Spine", 0, glm::vec3(0, 0.5f, 0));
    addBone(2, "Head", 1, glm::vec3(0, 0.5f, 0));
    
    // Legs
    addBone(3, "L_Hip", 0, glm::vec3(0.2f, 0, 0));
    addBone(4, "L_Knee", 3, glm::vec3(0, -0.5f, 0));
    addBone(5, "R_Hip", 0, glm::vec3(-0.2f, 0, 0));
    addBone(6, "R_Knee", 5, glm::vec3(0, -0.5f, 0));
    
    // Arms
    addBone(7, "L_Shoulder", 1, glm::vec3(0.3f, 0.3f, 0));
    addBone(8, "L_Elbow", 7, glm::vec3(0.3f, 0, 0));
    addBone(9, "R_Shoulder", 1, glm::vec3(-0.3f, 0.3f, 0));
    addBone(10, "R_Elbow", 9, glm::vec3(-0.3f, 0, 0));

    return skeleton;
}

namespace DemoMeshGen {
    std::unique_ptr<D3D12Mesh> CreateSkinnedCube(DX12RenderBackend* backend) {
        auto mesh = std::make_unique<D3D12Mesh>();
        
        // Simple Cube with bone weights
        // 8 vertices
        std::vector<Rendering::Vertex> vertices;
        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0, // Front
            1, 5, 6, 6, 2, 1, // Right
            5, 4, 7, 7, 6, 5, // Back
            4, 0, 3, 3, 7, 4, // Left
            3, 2, 6, 6, 7, 3, // Top
            4, 5, 1, 1, 0, 4  // Bottom
        };

        // Positions (+- 0.5)
        glm::vec3 pos[] = {
            {-0.5, -0.5,  0.5}, { 0.5, -0.5,  0.5}, { 0.5,  0.5,  0.5}, {-0.5,  0.5,  0.5}, // Front
            {-0.5, -0.5, -0.5}, { 0.5, -0.5, -0.5}, { 0.5,  0.5, -0.5}, {-0.5,  0.5, -0.5}  // Back
        };

        // Normals (simplified, usually per face for hard edges)
        // For skinning demo, smooth shading is fine or just identity normals
        glm::vec3 norm(0, 1, 0);

        for(int i=0; i<8; ++i) {
            Rendering::Vertex v;
            v.position = pos[i];
            v.normal = glm::normalize(pos[i]); // Sphere-like normals
            v.texcoord = glm::vec2(0);
            
            // Skinning:
            // Bottom vertices (y < 0) -> Bone 0 (Root)
            // Top vertices (y > 0) -> Bone 1 (Spine)
            if (pos[i].y < 0) {
                v.boneIndices = glm::ivec4(0, 0, 0, 0);
                v.boneWeights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
            } else {
                v.boneIndices = glm::ivec4(1, 0, 0, 0);
                v.boneWeights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
            }
            vertices.push_back(v);
        }

        // Adjust indices because we shared vertices (for hard edges we need 24 verts)
        // For this demo, let's just use shared vertices for simplicity
        
        mesh->Create(backend, vertices, indices, "SkinnedCube");
        
        // Set material properties for proper lighting
        mesh->GetMaterial().albedoColor = glm::vec4(0.2f, 0.5f, 1.0f, 1.0f); // Blue
        mesh->GetMaterial().roughness = 0.6f;
        mesh->GetMaterial().metallic = 0.0f;
        
        return mesh;
    }
}

int main() {
    std::cout << "[IntegratedAnimationDemo] Starting..." << std::endl;

    // 1. Init Window
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Integrated Animation Demo", nullptr, nullptr);
    if (!window) return -1;
    glfwSetKeyCallback(window, key_callback);

    // 2. Init ECS & Components
    coordinator.Initialize();
    coordinator.RegisterComponent<Rendering::TransformComponent>();
    coordinator.RegisterComponent<Rendering::MaterialComponent>();
    coordinator.RegisterComponent<CudaGame::Animation::AnimationComponent>();

    // 3. Init Animation System
    animationSystem = std::make_unique<CudaGame::Animation::AnimationSystem>();
    animationSystem->Initialize(); // This calls createDefaultAnimations()!

    // 4. Init Rendering
    renderPipeline = std::make_unique<DX12RenderPipeline>();
    DX12RenderPipeline::InitParams params = {};
    params.windowHandle = window;
    params.displayWidth = WINDOW_WIDTH;
    params.displayHeight = WINDOW_HEIGHT;
    renderPipeline->Initialize(params);

    // 5. Create Bone Buffer (GPU Resource)
    {
        auto device = renderPipeline->GetBackend()->GetDevice();
        D3D12_HEAP_PROPERTIES heapProps = {D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = sizeof(glm::mat4) * 100; // Enough for 11 bones
        desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
        desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        
        device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&boneBuffer));
        boneBuffer->Map(0, nullptr, &boneBufferMapped);
    }

    // 6. Create Player Entity
    auto player = coordinator.CreateEntity();
    coordinator.AddComponent(player, Rendering::TransformComponent{glm::vec3(0,0,0), glm::vec3(0), glm::vec3(1)});
    coordinator.AddComponent(player, Rendering::MaterialComponent{glm::vec3(0.2f, 0.8f, 1.0f)}); // Blue

    // Attach Skeleton & Animation
    auto skeleton = CreateHumanoidSkeleton();
    animationSystem->AddComponent(player, skeleton);
    
    // Start with Idle
    animationSystem->playAnimation(player, "Idle", 0.0f);

    // 7. Create Skinned Mesh
    auto mesh = DemoMeshGen::CreateSkinnedCube(renderPipeline->GetBackend());
    mesh->SetBoneBuffer(boneBuffer);
    mesh->transform = glm::mat4(1.0f); // Identity

    std::cout << "[Demo] Setup Loop Complete. Controls: W(Run), Space(Jump), Shift(Dash), E(WallRun)" << std::endl;

    // 8. Main Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (keys[GLFW_KEY_ESCAPE]) glfwSetWindowShouldClose(window, true);

        // --- Logic ---
        float dt = 0.016f;
        static std::string currentState = "Idle";
        std::string targetState = "Idle";

        if (keys[GLFW_KEY_W]) targetState = "Run";
        if (keys[GLFW_KEY_SPACE]) targetState = "Jump";
        if (keys[GLFW_KEY_LEFT_SHIFT]) targetState = "Dash";
        if (keys[GLFW_KEY_E]) targetState = "WallRun";

        // Simple state machine triggering
        if (targetState != currentState) {
            animationSystem->playAnimation(player, targetState, 0.2f); // 0.2s blend
            currentState = targetState;
            std::cout << "State: " << currentState << std::endl;
        }

        // --- Update Systems ---
        animationSystem->Update(dt);

        // --- Data Transfer (CPU -> GPU) ---
        auto* animComp = animationSystem->GetComponent(player);
        if (animComp && !animComp->globalBoneMatrices.empty()) {
            memcpy(boneBufferMapped, animComp->globalBoneMatrices.data(), animComp->globalBoneMatrices.size() * sizeof(glm::mat4));
        }

        // --- Render ---
        DX12RenderPipeline::SceneData scene;
        scene.meshes.push_back(mesh.get()); // Add our player mesh
        
        // Add light
        DX12RenderPipeline::DirectionalLight light;
        light.direction = glm::normalize(glm::vec3(-1,-1,-1));
        scene.directionalLights.push_back(light);

        // Camera (Static View)
        static Camera camera(ProjectionType::PERSPECTIVE);
        camera.SetPosition(glm::vec3(0, 2, 5));
        camera.LookAt(glm::vec3(0, 1, 0));
        
        renderPipeline->BeginFrame(&camera);
        renderPipeline->RenderFrame(scene);
        renderPipeline->EndFrame();
    }

    renderPipeline->Shutdown();
    glfwTerminate();
    return 0;
}
#endif
