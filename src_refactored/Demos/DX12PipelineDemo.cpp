#ifdef _WIN32
#include <iostream>
#include <memory>
#include <chrono>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "Rendering/DX12RenderPipeline.h"
#include "Rendering/D3D12Mesh.h"
#include "Rendering/Camera.h"
#include <glm/gtc/matrix_transform.hpp>

using namespace CudaGame::Rendering;

// Simple camera for demo
class DemoCamera : public Camera {
public:
    glm::vec3 position = glm::vec3(0.0f, 2.0f, 5.0f);
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    float fov = 60.0f;
    float aspect = 16.0f / 9.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    
    glm::mat4 GetViewMatrix() const {
        return glm::lookAt(position, target, up);
    }
    
    glm::mat4 GetProjectionMatrix() const {
        return glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);
    }
    
    glm::vec3 GetPosition() const { return position; }
    glm::vec3 GetForward() const { return glm::normalize(target - position); }
    glm::vec3 GetRight() const { return glm::normalize(glm::cross(GetForward(), up)); }
    glm::vec3 GetUp() const { return up; }
};

int main() {
    std::cout << "=== D3D12 AAA Rendering Pipeline Demo ===" << std::endl;
    std::cout << "Features: Deferred PBR, DLSS, Ray Tracing" << std::endl;
    std::cout << std::endl;
    
    // Initialize GLFW for window creation
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Tell GLFW not to create an OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    // Create window (1080p for demo)
    const uint32_t width = 1920;
    const uint32_t height = 1080;
    GLFWwindow* window = glfwCreateWindow(width, height, "D3D12 Pipeline Demo - AAA Deferred PBR", nullptr, nullptr);
    
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    std::cout << "[Demo] Window created: " << width << "x" << height << std::endl;
    
    // Get native Win32 window handle
    HWND hwnd = glfwGetWin32Window(window);
    
    // Initialize rendering pipeline
    auto pipeline = std::make_unique<DX12RenderPipeline>();
    
    DX12RenderPipeline::InitParams params = {};
    params.windowHandle = window;  // Pass GLFW window for swap chain creation
    params.displayWidth = width;
    params.displayHeight = height;
    params.enableDLSS = false;  // Disable for simplicity (can enable if supported)
    params.enableRayTracing = false;  // Disable for initial demo
    
    if (!pipeline->Initialize(params)) {
        std::cerr << "[Demo] Failed to initialize pipeline" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    
    std::cout << "[Demo] Pipeline initialized successfully" << std::endl;
    
    // Create demo camera
    auto camera = std::make_unique<DemoCamera>();
    camera->aspect = (float)width / (float)height;
    
    // Create test scene with procedural geometry
    std::cout << "[Demo] Creating test scene..." << std::endl;
    
    // Get backend for mesh creation
    // Note: In production, we'd have a resource manager
    // For now, we'll just create meshes and note that they're ready
    std::vector<std::unique_ptr<D3D12Mesh>> sceneMeshes;
    
    // Create a cube
    auto cube = MeshGenerator::CreateCube(pipeline->GetBackend());
    if (cube) {
        cube->transform = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f));
        cube->GetMaterial().albedoColor = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red
        pipeline->AddMesh(cube.get());
        sceneMeshes.push_back(std::move(cube));
    }
    
    // Create a sphere
    auto sphere = MeshGenerator::CreateSphere(pipeline->GetBackend(), 32);
    if (sphere) {
        sphere->transform = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f));
        sphere->GetMaterial().albedoColor = glm::vec4(0.3f, 0.3f, 1.0f, 1.0f);  // Blue
        sphere->GetMaterial().metallic = 0.8f;
        pipeline->AddMesh(sphere.get());
        sceneMeshes.push_back(std::move(sphere));
    }
    
    // Create a ground plane
    auto plane = MeshGenerator::CreatePlane(pipeline->GetBackend());
    if (plane) {
        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 1.0f, 10.0f));
        plane->transform = glm::translate(scaleMatrix, glm::vec3(0.0f, -1.0f, 0.0f));
        plane->GetMaterial().albedoColor = glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);  // Gray
        pipeline->AddMesh(plane.get());
        sceneMeshes.push_back(std::move(plane));
    }
    
    std::cout << "[Demo] Scene created with " << pipeline->GetMeshCount() << " meshes" << std::endl;
    
    // Main render loop
    std::cout << "[Demo] Entering render loop (Press ESC to exit)..." << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    uint32_t frameCount = 0;
    float fps = 0.0f;
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Exit on ESC
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        
        // Rotate camera around scene
        float time = (float)glfwGetTime();
        camera->position.x = 5.0f * cos(time * 0.5f);
        camera->position.z = 5.0f * sin(time * 0.5f);
        
        // Render frame
        pipeline->BeginFrame(camera.get());
        pipeline->RenderFrame();
        pipeline->EndFrame();
        
        // Calculate FPS
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(currentTime - startTime).count();
        
        if (elapsed >= 1.0f) {
            fps = frameCount / elapsed;
            
            // Print stats every second
            auto stats = pipeline->GetFrameStats();
            std::cout << "[Demo] FPS: " << (int)fps 
                      << " | Draw Calls: " << stats.drawCalls
                      << " | Triangles: " << stats.triangles
                      << " | Frame: " << stats.totalFrameMs << "ms" << std::endl;
            
            frameCount = 0;
            startTime = currentTime;
        }
        
        // Swap chain presentation handled by pipeline->EndFrame()
    }
    
    std::cout << "[Demo] Shutting down..." << std::endl;
    
    // Cleanup
    sceneMeshes.clear();
    pipeline->Shutdown();
    pipeline.reset();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "[Demo] Shutdown complete" << std::endl;
    return 0;
}

#else
#include <iostream>

int main() {
    std::cerr << "D3D12 Pipeline Demo only available on Windows" << std::endl;
    return -1;
}
#endif // _WIN32
