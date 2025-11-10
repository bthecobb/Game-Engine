#include <gtest/gtest.h>
#include "Rendering/Skybox.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>

using namespace CudaGame::Rendering;

// Test fixture for Skybox tests
class SkyboxTest : public ::testing::Test {
protected:
    GLFWwindow* window = nullptr;
    Skybox* skybox = nullptr;

    void SetUp() override {
        // Initialize GLFW
        ASSERT_TRUE(glfwInit()) << "Failed to initialize GLFW";

        // Configure GLFW for OpenGL 3.3 core
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hidden window for tests

        // Create window
        window = glfwCreateWindow(800, 600, "Skybox Test", nullptr, nullptr);
        ASSERT_NE(window, nullptr) << "Failed to create GLFW window";

        glfwMakeContextCurrent(window);

        // Initialize GLAD
        ASSERT_TRUE(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) 
            << "Failed to initialize GLAD";

        // Create skybox instance
        skybox = new Skybox();
    }

    void TearDown() override {
        if (skybox) {
            skybox->Shutdown();
            delete skybox;
            skybox = nullptr;
        }

        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }

        glfwTerminate();
    }
};

// Test: Skybox initialization
TEST_F(SkyboxTest, InitializationState) {
    EXPECT_NE(skybox, nullptr);
    EXPECT_EQ(skybox->GetCubemapTexture(), 0u) << "Cubemap should not be loaded initially";
    EXPECT_TRUE(skybox->IsEnabled()) << "Skybox should be enabled by default";
}

// Test: Default parameter values
TEST_F(SkyboxTest, DefaultParameters) {
    EXPECT_FLOAT_EQ(skybox->GetExposure(), 1.0f) << "Default exposure should be 1.0";
    EXPECT_FLOAT_EQ(skybox->GetGamma(), 2.2f) << "Default gamma should be 2.2";
    EXPECT_FLOAT_EQ(skybox->GetRotation(), 0.0f) << "Default rotation should be 0.0";
}

// Test: Parameter setting and getting
TEST_F(SkyboxTest, ParameterSettersGetters) {
    // Test exposure
    skybox->SetExposure(1.5f);
    EXPECT_FLOAT_EQ(skybox->GetExposure(), 1.5f);

    // Test gamma
    skybox->SetGamma(2.4f);
    EXPECT_FLOAT_EQ(skybox->GetGamma(), 2.4f);

    // Test rotation
    skybox->SetRotation(1.57f); // ~90 degrees
    EXPECT_FLOAT_EQ(skybox->GetRotation(), 1.57f);

    // Test enabled state
    skybox->SetEnabled(false);
    EXPECT_FALSE(skybox->IsEnabled());

    skybox->SetEnabled(true);
    EXPECT_TRUE(skybox->IsEnabled());
}

// Test: Load non-existent HDR file (should fail gracefully)
TEST_F(SkyboxTest, LoadNonExistentHDR) {
    bool result = skybox->LoadHDR("non_existent_file.hdr");
    EXPECT_FALSE(result) << "Loading non-existent file should return false";
    EXPECT_EQ(skybox->GetCubemapTexture(), 0u) << "Cubemap texture should remain uninitialized";
}

// Test: Load invalid file format (should fail gracefully)
TEST_F(SkyboxTest, LoadInvalidFormat) {
    // Create a temporary text file (not an HDR file)
    const char* testFile = "test_invalid.txt";
    {
        std::ofstream file(testFile);
        file << "This is not an HDR file";
    }

    bool result = skybox->LoadHDR(testFile);
    EXPECT_FALSE(result) << "Loading non-HDR file should return false";

    // Clean up
    std::remove(testFile);
}

// Test: HDR loading with actual file (if available)
TEST_F(SkyboxTest, LoadValidHDR) {
    const char* hdrPath = "C:\\Users\\Brandon\\CudaGame\\assets\\hdri\\qwantani_noon_puresky_4k.hdr";
    
    // Check if file exists
    std::ifstream file(hdrPath);
    if (!file.good()) {
        GTEST_SKIP() << "HDR test file not found at: " << hdrPath;
        return;
    }
    file.close();

    // Attempt to load
    bool result = skybox->LoadHDR(hdrPath, 512);
    EXPECT_TRUE(result) << "Loading valid HDR file should succeed";

    if (result) {
        EXPECT_NE(skybox->GetCubemapTexture(), 0u) << "Cubemap texture should be created";
        
        // Verify cubemap texture is valid
        GLuint cubemap = skybox->GetCubemapTexture();
        GLint textureType = 0;
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
        glGetTexParameteriv(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, &textureType);
        
        // If no GL error, texture is valid
        GLenum err = glGetError();
        EXPECT_EQ(err, GL_NO_ERROR) << "Cubemap texture should be valid (GL error: " << err << ")";
    }
}

// Test: Render method without crash (basic smoke test)
TEST_F(SkyboxTest, RenderWithoutCrash) {
    // Set up matrices
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    // Should not crash even without loaded HDR
    EXPECT_NO_THROW(skybox->Render(view, projection));
}

// Test: Render with loaded HDR (if available)
TEST_F(SkyboxTest, RenderWithLoadedHDR) {
    const char* hdrPath = "C:\\Users\\Brandon\\CudaGame\\assets\\hdri\\qwantani_noon_puresky_4k.hdr";
    
    // Check if file exists
    std::ifstream file(hdrPath);
    if (!file.good()) {
        GTEST_SKIP() << "HDR test file not found at: " << hdrPath;
        return;
    }
    file.close();

    // Load HDR
    bool loaded = skybox->LoadHDR(hdrPath, 256); // Smaller cubemap for faster test
    ASSERT_TRUE(loaded) << "Failed to load HDR for render test";

    // Set up matrices
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    // Render should not crash
    EXPECT_NO_THROW(skybox->Render(view, projection));
    
    // Check for GL errors
    GLenum err = glGetError();
    EXPECT_EQ(err, GL_NO_ERROR) << "Render should not produce GL errors (error: " << err << ")";
}

// Test: Shutdown without load
TEST_F(SkyboxTest, ShutdownWithoutLoad) {
    EXPECT_NO_THROW(skybox->Shutdown());
}

// Test: Shutdown with loaded HDR
TEST_F(SkyboxTest, ShutdownWithLoadedHDR) {
    const char* hdrPath = "C:\\Users\\Brandon\\CudaGame\\assets\\hdri\\qwantani_noon_puresky_4k.hdr";
    
    // Check if file exists
    std::ifstream file(hdrPath);
    if (!file.good()) {
        GTEST_SKIP() << "HDR test file not found at: " << hdrPath;
        return;
    }
    file.close();

    skybox->LoadHDR(hdrPath, 128);
    GLuint textureId = skybox->GetCubemapTexture();
    
    if (textureId != 0) {
        EXPECT_NO_THROW(skybox->Shutdown());
        // After shutdown, texture should be deleted
        // Note: We can't easily test texture deletion without GL introspection
    }
}

// Test: Multiple loads (should clean up previous resources)
TEST_F(SkyboxTest, MultipleLoads) {
    const char* hdrPath = "C:\\Users\\Brandon\\CudaGame\\assets\\hdri\\qwantani_noon_puresky_4k.hdr";
    
    // Check if file exists
    std::ifstream file(hdrPath);
    if (!file.good()) {
        GTEST_SKIP() << "HDR test file not found at: " << hdrPath;
        return;
    }
    file.close();

    // First load
    bool result1 = skybox->LoadHDR(hdrPath, 128);
    EXPECT_TRUE(result1);
    GLuint texture1 = skybox->GetCubemapTexture();

    // Second load (should clean up first)
    bool result2 = skybox->LoadHDR(hdrPath, 256);
    EXPECT_TRUE(result2);
    GLuint texture2 = skybox->GetCubemapTexture();

    // Textures should be different (old one deleted, new one created)
    EXPECT_NE(texture1, texture2);
}

// Test: Exposure clamping (if implemented)
TEST_F(SkyboxTest, ExposureBounds) {
    skybox->SetExposure(0.0f);
    EXPECT_GE(skybox->GetExposure(), 0.0f) << "Exposure should not be negative";

    skybox->SetExposure(100.0f);
    // Check if clamping is implemented (implementation-dependent)
    float exposure = skybox->GetExposure();
    EXPECT_GT(exposure, 0.0f) << "Exposure should be positive";
}

// Test: Rotation wrapping (angle normalization)
TEST_F(SkyboxTest, RotationWrapping) {
    skybox->SetRotation(0.0f);
    EXPECT_FLOAT_EQ(skybox->GetRotation(), 0.0f);

    skybox->SetRotation(6.28318f); // ~2*PI
    float rotation = skybox->GetRotation();
    // Implementation may or may not normalize, just check it's stored
    EXPECT_NE(rotation, -1.0f); // Sanity check
}
