#ifdef _WIN32
#include <gtest/gtest.h>
#include "Rendering/DX12RenderPipeline.h"
#include <memory>

using namespace CudaGame::Rendering;

// Simple camera mock for testing
struct TestCamera {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float viewMatrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float projMatrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

class DX12RenderPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        pipeline = std::make_unique<DX12RenderPipeline>();
    }

    void TearDown() override {
        if (pipeline) {
            pipeline->Shutdown();
            pipeline.reset();
        }
    }

    std::unique_ptr<DX12RenderPipeline> pipeline;
};

// Test 1: Basic initialization
TEST_F(DX12RenderPipelineTest, BasicInitialization) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 1920;
    params.displayHeight = 1080;
    params.enableDLSS = false;
    params.enableRayTracing = false;
    
    bool result = pipeline->Initialize(params);
    EXPECT_TRUE(result);
}

// Test 2: Initialize with DLSS
TEST_F(DX12RenderPipelineTest, InitializeWithDLSS) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 3840;  // 4K
    params.displayHeight = 2160;
    params.enableDLSS = true;
    params.dlssMode = DLSSQualityMode::Quality;
    params.enableRayTracing = false;
    
    bool result = pipeline->Initialize(params);
    EXPECT_TRUE(result);
    
    // DLSS Quality mode should render at lower resolution
    // With Quality (1.5x), 4K (3840x2160) -> ~2560x1440
    // Exact resolution depends on DLSS, but should be less than display
}

// Test 3: Initialize with ray tracing
TEST_F(DX12RenderPipelineTest, InitializeWithRayTracing) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 1920;
    params.displayHeight = 1080;
    params.enableDLSS = false;
    params.enableRayTracing = true;
    
    bool result = pipeline->Initialize(params);
    EXPECT_TRUE(result);
}

// Test 4: Initialize with all features
TEST_F(DX12RenderPipelineTest, InitializeWithAllFeatures) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 3840;
    params.displayHeight = 2160;
    params.enableDLSS = true;
    params.dlssMode = DLSSQualityMode::Performance;
    params.enableRayTracing = true;
    
    bool result = pipeline->Initialize(params);
    EXPECT_TRUE(result);
}

// Test 5: Prevent double initialization
TEST_F(DX12RenderPipelineTest, PreventDoubleInitialization) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 1920;
    params.displayHeight = 1080;
    
    bool result1 = pipeline->Initialize(params);
    EXPECT_TRUE(result1);
    
    // Second init should succeed but not re-initialize
    bool result2 = pipeline->Initialize(params);
    EXPECT_TRUE(result2);
}

// Test 6: Shutdown before initialization
TEST_F(DX12RenderPipelineTest, ShutdownBeforeInit) {
    // Should not crash
    EXPECT_NO_THROW(pipeline->Shutdown());
}

// Test 7: Render frame without initialization
TEST_F(DX12RenderPipelineTest, RenderWithoutInit) {
    TestCamera camera;
    
    // Should not crash
    EXPECT_NO_THROW({
        pipeline->BeginFrame(reinterpret_cast<Camera*>(&camera));
        pipeline->RenderFrame();
        pipeline->EndFrame();
    });
}

// Test 8: Complete frame cycle
TEST_F(DX12RenderPipelineTest, CompleteFrameCycle) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 1920;
    params.displayHeight = 1080;
    
    ASSERT_TRUE(pipeline->Initialize(params));
    
    TestCamera camera;
    
    // Render multiple frames
    for (int i = 0; i < 5; i++) {
        EXPECT_NO_THROW({
            pipeline->BeginFrame(reinterpret_cast<Camera*>(&camera));
            pipeline->RenderFrame();
            pipeline->EndFrame();
        });
        
        auto stats = pipeline->GetFrameStats();
        EXPECT_GE(stats.totalFrameMs, 0.0f);
    }
}

// Test 9: DLSS quality mode switching
TEST_F(DX12RenderPipelineTest, DLSSQualityModeSwitching) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 3840;
    params.displayHeight = 2160;
    params.enableDLSS = true;
    params.dlssMode = DLSSQualityMode::Quality;
    
    ASSERT_TRUE(pipeline->Initialize(params));
    
    // Switch quality modes
    EXPECT_NO_THROW({
        pipeline->SetDLSSQualityMode(DLSSQualityMode::Performance);
        pipeline->SetDLSSQualityMode(DLSSQualityMode::Balanced);
        pipeline->SetDLSSQualityMode(DLSSQualityMode::UltraPerformance);
        pipeline->SetDLSSQualityMode(DLSSQualityMode::Quality);
    });
}

// Test 10: Performance stats tracking
TEST_F(DX12RenderPipelineTest, PerformanceStatsTracking) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 1920;
    params.displayHeight = 1080;
    params.enableDLSS = true;
    params.enableRayTracing = true;
    
    ASSERT_TRUE(pipeline->Initialize(params));
    
    TestCamera camera;
    pipeline->BeginFrame(reinterpret_cast<Camera*>(&camera));
    pipeline->RenderFrame();
    pipeline->EndFrame();
    
    auto stats = pipeline->GetFrameStats();
    
    // Check that stats are being tracked
    EXPECT_GT(stats.totalFrameMs, 0.0f);
    EXPECT_GE(stats.geometryPassMs, 0.0f);
    EXPECT_GE(stats.lightingPassMs, 0.0f);
    EXPECT_GE(stats.rayTracingPassMs, 0.0f);
    EXPECT_GE(stats.dlssPassMs, 0.0f);
    EXPECT_GE(stats.drawCalls, 0);
    EXPECT_GE(stats.triangles, 0);
}

// Test 11: Multiple resolutions
TEST_F(DX12RenderPipelineTest, MultipleResolutions) {
    struct TestCase {
        uint32_t width;
        uint32_t height;
        const char* name;
    };
    
    TestCase cases[] = {
        {1280, 720, "720p"},
        {1920, 1080, "1080p"},
        {2560, 1440, "1440p"},
        {3840, 2160, "4K"},
    };
    
    for (const auto& testCase : cases) {
        auto testPipeline = std::make_unique<DX12RenderPipeline>();
        
        DX12RenderPipeline::InitParams params = {};
        params.displayWidth = testCase.width;
        params.displayHeight = testCase.height;
        
        bool result = testPipeline->Initialize(params);
        EXPECT_TRUE(result) << "Failed to initialize at " << testCase.name;
        
        testPipeline->Shutdown();
    }
}

// Test 12: DLSS with different quality modes
TEST_F(DX12RenderPipelineTest, DLSSQualityModes) {
    DLSSQualityMode modes[] = {
        DLSSQualityMode::UltraPerformance,
        DLSSQualityMode::Performance,
        DLSSQualityMode::Balanced,
        DLSSQualityMode::Quality,
        DLSSQualityMode::UltraQuality,
        DLSSQualityMode::DLAA,
    };
    
    for (auto mode : modes) {
        auto testPipeline = std::make_unique<DX12RenderPipeline>();
        
        DX12RenderPipeline::InitParams params = {};
        params.displayWidth = 3840;
        params.displayHeight = 2160;
        params.enableDLSS = true;
        params.dlssMode = mode;
        
        bool result = testPipeline->Initialize(params);
        EXPECT_TRUE(result);
        
        testPipeline->Shutdown();
    }
}

// Test 13: Shutdown and reinitialize
TEST_F(DX12RenderPipelineTest, ShutdownAndReinitialize) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 1920;
    params.displayHeight = 1080;
    
    // First initialization
    ASSERT_TRUE(pipeline->Initialize(params));
    pipeline->Shutdown();
    
    // Second initialization should work
    ASSERT_TRUE(pipeline->Initialize(params));
    pipeline->Shutdown();
}

// Test 14: Render without camera
TEST_F(DX12RenderPipelineTest, RenderWithoutCamera) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 1920;
    params.displayHeight = 1080;
    
    ASSERT_TRUE(pipeline->Initialize(params));
    
    // Should not crash, but also should not render
    EXPECT_NO_THROW({
        pipeline->BeginFrame(nullptr);
        pipeline->RenderFrame();
        pipeline->EndFrame();
    });
}

// Test 15: Mesh count tracking
TEST_F(DX12RenderPipelineTest, MeshCountTracking) {
    DX12RenderPipeline::InitParams params = {};
    params.displayWidth = 1920;
    params.displayHeight = 1080;
    params.enableDLSS = false;
    params.enableRayTracing = false;
    
    ASSERT_TRUE(pipeline->Initialize(params));
    
    // Initially no meshes
    EXPECT_EQ(pipeline->GetMeshCount(), 0u);
    
    // Add nullptr meshes (should be handled gracefully)
    pipeline->AddMesh(nullptr);
    EXPECT_EQ(pipeline->GetMeshCount(), 1u);  // Added but will be skipped in rendering
    
    // Clear meshes
    pipeline->ClearMeshes();
    EXPECT_EQ(pipeline->GetMeshCount(), 0u);
    
    // Render with no meshes should produce 0 draw calls
    TestCamera camera;
    pipeline->BeginFrame(reinterpret_cast<Camera*>(&camera));
    pipeline->RenderFrame();
    pipeline->EndFrame();
    
    auto stats = pipeline->GetFrameStats();
    EXPECT_EQ(stats.drawCalls, 0u);
    EXPECT_EQ(stats.triangles, 0u);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#else
// Non-Windows platform - provide empty tests
#include <gtest/gtest.h>

TEST(DX12RenderPipelineTest, NotAvailableOnThisPlatform) {
    GTEST_SKIP() << "DX12 rendering pipeline only available on Windows";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif // _WIN32
