#include "Testing/TestFramework.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderSystem.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Framebuffer.h"
#include "Rendering/LightingSystem.h"
#include "Rendering/RenderDebugSystem.h"
#include <memory>
#include <vector>
#include <chrono>

using namespace CudaGame::Testing;
using namespace CudaGame::Core;
using namespace CudaGame::Rendering;

// Mock OpenGL context for headless testing
class GLContextMock {
private:
    struct GLResource {
        GLenum type;
        GLuint id;
        bool isDeleted;
    };
    
    std::vector<GLResource> resources;
    bool isValid;
    
public:
    GLContextMock() : isValid(true) {}
    
    GLuint CreateResource(GLenum type) {
        static GLuint nextId = 1;
        resources.push_back({type, nextId, false});
        return nextId++;
    }
    
    bool IsResourceValid(GLuint id) const {
        auto it = std::find_if(resources.begin(), resources.end(),
            [id](const GLResource& r) { return r.id == id && !r.isDeleted; });
        return it != resources.end();
    }
    
    void DeleteResource(GLuint id) {
        auto it = std::find_if(resources.begin(), resources.end(),
            [id](GLResource& r) { return r.id == id; });
        if (it != resources.end()) {
            it->isDeleted = true;
        }
    }
    
    size_t GetActiveResourceCount() const {
        return std::count_if(resources.begin(), resources.end(),
            [](const GLResource& r) { return !r.isDeleted; });
    }
    
    bool IsValid() const { return isValid; }
    void Invalidate() { isValid = false; }
};

// Performance monitoring utilities
class PerformanceMonitor {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::milli>;
    
    TimePoint startTime;
    std::vector<Duration> measurements;
    const size_t maxSamples;
    const double targetFrameTime;  // in milliseconds
    
public:
    PerformanceMonitor(size_t maxSamples = 100, double targetFPS = 60.0)
        : maxSamples(maxSamples)
        , targetFrameTime(1000.0 / targetFPS) {}
    
    void StartMeasurement() {
        startTime = Clock::now();
    }
    
    void EndMeasurement() {
        auto endTime = Clock::now();
        measurements.push_back(Duration(endTime - startTime));
        if (measurements.size() > maxSamples) {
            measurements.erase(measurements.begin());
        }
    }
    
    double GetAverageTime() const {
        if (measurements.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& m : measurements) {
            sum += m.count();
        }
        return sum / measurements.size();
    }
    
    double GetWorstTime() const {
        if (measurements.empty()) return 0.0;
        return std::max_element(measurements.begin(), measurements.end())->count();
    }
    
    bool MeetsPerformanceTarget() const {
        return GetWorstTime() <= targetFrameTime;
    }
};

class RenderingSystemTestSuite {
private:
    std::shared_ptr<Coordinator> coordinator;
    std::unique_ptr<RenderSystem> renderSystem;
    std::unique_ptr<LightingSystem> lightingSystem;
    std::unique_ptr<RenderDebugSystem> debugSystem;
    std::unique_ptr<GLContextMock> glContext;
    std::unique_ptr<PerformanceMonitor> perfMonitor;
    
    const float EPSILON = 0.001f;
    const int WINDOW_WIDTH = 1920;
    const int WINDOW_HEIGHT = 1080;
    const glm::vec4 CLEAR_COLOR = glm::vec4(0.1f, 0.1f, 0.15f, 1.0f);

public:
    void SetUp() {
        coordinator = std::make_shared<Coordinator>();
        coordinator->Initialize();
        
        // Register rendering components
        coordinator->RegisterComponent<TransformComponent>();
        coordinator->RegisterComponent<MeshComponent>();
        coordinator->RegisterComponent<MaterialComponent>();
        coordinator->RegisterComponent<LightComponent>();
        
        // Initialize GL context mock
        glContext = std::make_unique<GLContextMock>();
        
        // Initialize systems
        renderSystem = std::make_unique<RenderSystem>();
        lightingSystem = std::make_unique<LightingSystem>();
        debugSystem = std::make_unique<RenderDebugSystem>();
        
        renderSystem->Initialize();
        lightingSystem->Initialize();
        debugSystem->Initialize();
        
        // Initialize performance monitor
        perfMonitor = std::make_unique<PerformanceMonitor>();
    }

    void TearDown() {
        // Verify all GL resources are properly cleaned up
        ASSERT_EQ(glContext->GetActiveResourceCount(), 0)
            << "Memory leak detected: Not all GL resources were properly deleted";
        
        debugSystem.reset();
        lightingSystem.reset();
        renderSystem.reset();
        glContext.reset();
        coordinator.reset();
    }

    // G-buffer setup and validation tests
    void TestGBufferInitialization() {
        // Verify G-buffer creation
        renderSystem->InitializeGBuffer(WINDOW_WIDTH, WINDOW_HEIGHT);
        
        auto& gBuffer = renderSystem->GetGBuffer();
        ASSERT_TRUE(gBuffer.IsValid());
        ASSERT_EQ(gBuffer.GetWidth(), WINDOW_WIDTH);
        ASSERT_EQ(gBuffer.GetHeight(), WINDOW_HEIGHT);
        
        // Verify all required textures are created
        ASSERT_TRUE(glContext->IsResourceValid(gBuffer.GetPositionTexture()));
        ASSERT_TRUE(glContext->IsResourceValid(gBuffer.GetNormalTexture()));
        ASSERT_TRUE(glContext->IsResourceValid(gBuffer.GetAlbedoTexture()));
        ASSERT_TRUE(glContext->IsResourceValid(gBuffer.GetMetallicRoughnessTexture()));
        ASSERT_TRUE(glContext->IsResourceValid(gBuffer.GetDepthTexture()));
    }

    void TestGBufferResolution() {
        // Test different resolutions
        const std::vector<std::pair<int, int>> resolutions = {
            {1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}
        };
        
        for (const auto& [width, height] : resolutions) {
            renderSystem->InitializeGBuffer(width, height);
            auto& gBuffer = renderSystem->GetGBuffer();
            
            ASSERT_EQ(gBuffer.GetWidth(), width);
            ASSERT_EQ(gBuffer.GetHeight(), height);
            ASSERT_TRUE(gBuffer.IsValid());
        }
    }

    // Deferred pipeline tests
    void TestGeometryPass() {
        // Create test scene
        Entity cube = CreateTestCube();
        Entity light = CreateTestLight();
        
        perfMonitor->StartMeasurement();
        
        // Execute geometry pass
        renderSystem->BeginGeometryPass();
        renderSystem->RenderEntity(cube);
        renderSystem->EndGeometryPass();
        
        perfMonitor->EndMeasurement();
        
        // Verify performance
        ASSERT_TRUE(perfMonitor->MeetsPerformanceTarget())
            << "Geometry pass exceeded performance budget";
        
        // Verify G-buffer contents
        auto& gBuffer = renderSystem->GetGBuffer();
        VerifyGBufferContents(gBuffer);
    }

    void TestLightingPass() {
        // Create test scene with multiple lights
        std::vector<Entity> lights;
        for (int i = 0; i < 4; i++) {
            lights.push_back(CreateTestLight());
        }
        
        perfMonitor->StartMeasurement();
        
        // Execute lighting pass
        renderSystem->BeginLightingPass();
        for (const auto& light : lights) {
            lightingSystem->RenderLight(light);
        }
        renderSystem->EndLightingPass();
        
        perfMonitor->EndMeasurement();
        
        // Verify performance
        ASSERT_TRUE(perfMonitor->MeetsPerformanceTarget())
            << "Lighting pass exceeded performance budget";
    }

    // Resource management tests
    void TestShaderCompilation() {
        // Test geometry pass shader
        ShaderProgram geometryShader;
        ASSERT_TRUE(geometryShader.CompileFromFile("geometry_pass"));
        ASSERT_TRUE(geometryShader.IsValid());
        
        // Test lighting pass shader
        ShaderProgram lightingShader;
        ASSERT_TRUE(lightingShader.CompileFromFile("lighting_pass"));
        ASSERT_TRUE(lightingShader.IsValid());
        
        // Verify uniform locations
        ASSERT_NE(geometryShader.GetUniformLocation("modelMatrix"), -1);
        ASSERT_NE(lightingShader.GetUniformLocation("lightPosition"), -1);
    }

    void TestTextureManagement() {
        const int maxTextures = 1000;
        std::vector<GLuint> textureIds;
        
        // Stress test texture creation
        for (int i = 0; i < maxTextures; i++) {
            GLuint texId = glContext->CreateResource(GL_TEXTURE_2D);
            textureIds.push_back(texId);
            ASSERT_TRUE(glContext->IsResourceValid(texId));
        }
        
        // Verify cleanup
        for (GLuint id : textureIds) {
            glContext->DeleteResource(id);
            ASSERT_FALSE(glContext->IsResourceValid(id));
        }
    }

    // Debug visualization tests
    void TestDebugVisualization() {
        // Test each debug visualization mode
        const std::vector<DebugVisualizationMode> modes = {
            DebugVisualizationMode::WIREFRAME,
            DebugVisualizationMode::NORMALS,
            DebugVisualizationMode::DEPTH,
            DebugVisualizationMode::GBUFFER_ALBEDO,
            DebugVisualizationMode::GBUFFER_NORMAL,
            DebugVisualizationMode::GBUFFER_POSITION
        };
        
        Entity testObject = CreateTestCube();
        
        for (auto mode : modes) {
            debugSystem->SetVisualizationMode(mode);
            debugSystem->RenderDebugView(testObject);
            ASSERT_TRUE(debugSystem->IsVisualizationValid());
        }
    }

    // Performance tests
    void TestRenderingPerformance() {
        const int FRAME_COUNT = 1000;
        const float TARGET_FRAME_TIME = 16.67f;  // ~60 FPS
        
        // Create complex scene
        std::vector<Entity> entities;
        for (int i = 0; i < 100; i++) {
            entities.push_back(CreateTestCube());
        }
        
        double totalTime = 0.0;
        double worstFrameTime = 0.0;
        
        // Render multiple frames
        for (int frame = 0; frame < FRAME_COUNT; frame++) {
            perfMonitor->StartMeasurement();
            
            renderSystem->BeginGeometryPass();
            for (const auto& entity : entities) {
                renderSystem->RenderEntity(entity);
            }
            renderSystem->EndGeometryPass();
            
            renderSystem->BeginLightingPass();
            lightingSystem->RenderLights();
            renderSystem->EndLightingPass();
            
            perfMonitor->EndMeasurement();
            
            double frameTime = perfMonitor->GetWorstTime();
            worstFrameTime = std::max(worstFrameTime, frameTime);
            totalTime += frameTime;
        }
        
        double averageFrameTime = totalTime / FRAME_COUNT;
        
        ASSERT_LT(averageFrameTime, TARGET_FRAME_TIME)
            << "Average frame time exceeds target";
        ASSERT_LT(worstFrameTime, TARGET_FRAME_TIME * 1.5)
            << "Worst frame time significantly exceeds target";
    }

private:
    // Utility functions for test setup
    Entity CreateTestCube() {
        Entity entity = coordinator->CreateEntity();
        
        // Transform
        TransformComponent transform;
        transform.position = glm::vec3(0.0f, 0.0f, 0.0f);
        transform.scale = glm::vec3(1.0f);
        coordinator->AddComponent(entity, transform);
        
        // Mesh
        MeshComponent mesh;
        mesh.modelPath = "cube.obj";
        coordinator->AddComponent(entity, mesh);
        
        // Material
        MaterialComponent material;
        material.albedo = glm::vec3(0.8f);
        material.metallic = 0.0f;
        material.roughness = 0.5f;
        coordinator->AddComponent(entity, material);
        
        return entity;
    }

    Entity CreateTestLight() {
        Entity entity = coordinator->CreateEntity();
        
        // Transform
        TransformComponent transform;
        transform.position = glm::vec3(5.0f, 5.0f, 5.0f);
        coordinator->AddComponent(entity, transform);
        
        // Light
        LightComponent light;
        light.color = glm::vec3(1.0f);
        light.intensity = 1.0f;
        light.radius = 10.0f;
        coordinator->AddComponent(entity, light);
        
        return entity;
    }

    void VerifyGBufferContents(const Framebuffer& gBuffer) {
        // Verify position buffer
        {
            GLuint positionTex = gBuffer.GetPositionTexture();
            ASSERT_TRUE(glContext->IsResourceValid(positionTex));
            // Add more specific position data validation
        }
        
        // Verify normal buffer
        {
            GLuint normalTex = gBuffer.GetNormalTexture();
            ASSERT_TRUE(glContext->IsResourceValid(normalTex));
            // Add more specific normal data validation
        }
        
        // Verify depth buffer
        {
            GLuint depthTex = gBuffer.GetDepthTexture();
            ASSERT_TRUE(glContext->IsResourceValid(depthTex));
            // Verify depth values are in correct range
        }
    }
};

// Function to create and register the test suite
std::shared_ptr<TestSuite> CreateRenderingSystemTestSuite() {
    auto suite = std::make_shared<TestSuite>("Rendering System");
    auto fixture = std::make_shared<RenderingSystemTestSuite>();
    
    // G-buffer tests
    suite->AddTest("G-Buffer Initialization", [fixture]() {
        fixture->SetUp();
        fixture->TestGBufferInitialization();
        fixture->TearDown();
    });
    
    suite->AddTest("G-Buffer Resolution", [fixture]() {
        fixture->SetUp();
        fixture->TestGBufferResolution();
        fixture->TearDown();
    });
    
    // Pipeline tests
    suite->AddTest("Geometry Pass", [fixture]() {
        fixture->SetUp();
        fixture->TestGeometryPass();
        fixture->TearDown();
    });
    
    suite->AddTest("Lighting Pass", [fixture]() {
        fixture->SetUp();
        fixture->TestLightingPass();
        fixture->TearDown();
    });
    
    // Resource management tests
    suite->AddTest("Shader Compilation", [fixture]() {
        fixture->SetUp();
        fixture->TestShaderCompilation();
        fixture->TearDown();
    });
    
    suite->AddTest("Texture Management", [fixture]() {
        fixture->SetUp();
        fixture->TestTextureManagement();
        fixture->TearDown();
    });
    
    // Debug visualization tests
    suite->AddTest("Debug Visualization", [fixture]() {
        fixture->SetUp();
        fixture->TestDebugVisualization();
        fixture->TearDown();
    });
    
    // Performance tests
    suite->AddTest("Rendering Performance", [fixture]() {
        fixture->SetUp();
        fixture->TestRenderingPerformance();
        fixture->TearDown();
    });
    
    return suite;
}