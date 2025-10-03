#include "Testing/TestFramework.h"
#include "Rendering/OrbitCamera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/epsilon.hpp>
#include <memory>
#include <string>
#include <sstream>

using namespace CudaGame::Testing;
using namespace CudaGame::Core;
using namespace CudaGame::Rendering;

class OrbitCameraTestSuite {
private:
    std::unique_ptr<OrbitCamera> camera;
    glm::vec3 targetPosition;
    const float EPSILON = 0.001f;

    // Helper to check if two vectors are approximately equal
    bool VecEquals(const glm::vec3& a, const glm::vec3& b) {
        return glm::all(glm::epsilonEqual(a, b, EPSILON));
    }

public:
    void SetUp() {
        camera = std::make_unique<OrbitCamera>(ProjectionType::PERSPECTIVE);
        
        // Configure camera for testing
        OrbitCamera::OrbitSettings settings;
        settings.distance = 15.0f;
        settings.heightOffset = 2.0f;
        settings.mouseSensitivity = 0.05f;
        settings.smoothSpeed = 6.0f;
        camera->SetOrbitSettings(settings);
        
        // Set standard perspective parameters
        camera->SetPerspective(60.0f, 16.0f/9.0f, 0.1f, 200.0f);
    }

    void TearDown() {
        camera.reset();
    }

    // Test camera initialization
void TestCameraInitialization() {
        ASSERT_TRUE(camera != nullptr);
        ASSERT_EQ(camera->GetProjectionType(), ProjectionType::PERSPECTIVE);
        ASSERT_EQ(camera->GetCameraMode(), OrbitCamera::CameraMode::ORBIT_FOLLOW);
        
        // Verify default mode
        ASSERT_EQ(camera->GetCameraMode(), OrbitCamera::CameraMode::ORBIT_FOLLOW);
        
        // Verify perspective parameters
        float fov, aspect, nearPlane, farPlane;
        camera->GetPerspectiveParams(fov, aspect, nearPlane, farPlane);
        ASSERT_NEAR(fov, 60.0f, EPSILON);
        ASSERT_NEAR(aspect, 16.0f/9.0f, EPSILON);
        ASSERT_NEAR(nearPlane, 0.1f, EPSILON);
        ASSERT_NEAR(farPlane, 200.0f, EPSILON);
    }

    // Test camera modes
    void TestCameraModeTransitions() {
        // Test ORBIT_FOLLOW mode
        camera->SetCameraMode(OrbitCamera::CameraMode::ORBIT_FOLLOW);
        ASSERT_EQ(camera->GetCameraMode(), OrbitCamera::CameraMode::ORBIT_FOLLOW);
        
        // Test FREE_LOOK mode
        camera->SetCameraMode(OrbitCamera::CameraMode::FREE_LOOK);
        ASSERT_EQ(camera->GetCameraMode(), OrbitCamera::CameraMode::FREE_LOOK);
        
        // Test COMBAT_FOCUS mode
        camera->SetCameraMode(OrbitCamera::CameraMode::COMBAT_FOCUS);
        ASSERT_EQ(camera->GetCameraMode(), OrbitCamera::CameraMode::COMBAT_FOCUS);
    }

    // Test orbit settings
    void TestOrbitSettings() {
        OrbitCamera::OrbitSettings settings;
        settings.distance = 20.0f;
        settings.heightOffset = 3.0f;
        settings.mouseSensitivity = 0.1f;
        settings.smoothSpeed = 8.0f;
        
        camera->SetOrbitSettings(settings);
        
        auto& currentSettings = camera->GetOrbitSettings();
        ASSERT_NEAR(currentSettings.distance, 20.0f, EPSILON);
        ASSERT_NEAR(currentSettings.heightOffset, 3.0f, EPSILON);
        ASSERT_NEAR(currentSettings.mouseSensitivity, 0.1f, EPSILON);
        ASSERT_NEAR(currentSettings.smoothSpeed, 8.0f, EPSILON);
    }

    // Test camera movement
    void TestCameraMovement() {
        glm::vec3 targetPos(10.0f, 2.0f, 5.0f);
        glm::vec3 velocity(1.0f, 0.0f, 0.0f);
        float deltaTime = 0.016f;  // ~60 FPS
        
        // Update camera for several frames
        for (int i = 0; i < 5; i++) {
            camera->Update(deltaTime, targetPos, velocity);
        }
        
        // Verify camera follows target
        glm::vec3 cameraPos = camera->GetPosition();
        ASSERT_GT(glm::distance(cameraPos, targetPos), camera->GetOrbitSettings().distance - 1.0f);
        ASSERT_LT(glm::distance(cameraPos, targetPos), camera->GetOrbitSettings().distance + 1.0f);
    }

    // Test zoom functionality
    void TestCameraZoom() {
        float initialDistance = camera->GetOrbitSettings().distance;
        
        // Zoom in
        camera->ApplyZoom(1.0f);
        ASSERT_LT(camera->GetOrbitSettings().distance, initialDistance);
        
        // Zoom out
        camera->ApplyZoom(-1.0f);
        ASSERT_NEAR(camera->GetOrbitSettings().distance, initialDistance, EPSILON);
    }

    // Test mouse input
    void TestMouseInput() {
        // Store initial orientation
        glm::vec3 initialForward = camera->GetForward();
        
        // Apply mouse movement
        camera->ApplyMouseDelta(10.0f, 5.0f);
        
        // Verify camera rotated
        glm::vec3 newForward = camera->GetForward();
        ASSERT_NE(initialForward, newForward);
    }

    // Test matrix generation
    void TestViewProjectionMatrix() {
        camera->UpdateMatrices();
        
        glm::mat4 viewMatrix = camera->GetViewMatrix();
        glm::mat4 projMatrix = camera->GetProjectionMatrix();
        
        // Verify matrices are not identity
        glm::mat4 identity(1.0f);
        ASSERT_NE(viewMatrix, identity);
        ASSERT_NE(projMatrix, identity);
        
        // Verify projection matrix maintains aspect ratio
        float fov, aspect, nearPlane, farPlane;
        camera->GetPerspectiveParams(fov, aspect, nearPlane, farPlane);
        ASSERT_NEAR(projMatrix[1][1] / projMatrix[0][0], 1.0f/aspect, EPSILON);
    }
};

// Function to create and register the test suite
std::shared_ptr<TestSuite> CreateOrbitCameraTestSuite() {
    auto suite = std::make_shared<TestSuite>("Orbit Camera System");
    auto fixture = std::make_shared<OrbitCameraTestSuite>();
    
    // Camera initialization
    suite->AddTest("Camera Initialization", [fixture]() {
        fixture->SetUp();
        fixture->TestCameraInitialization();
        fixture->TearDown();
    });
    
    // Camera modes
    suite->AddTest("Camera Mode Transitions", [fixture]() {
        fixture->SetUp();
        fixture->TestCameraModeTransitions();
        fixture->TearDown();
    });
    
    // Orbit settings
    suite->AddTest("Orbit Settings", [fixture]() {
        fixture->SetUp();
        fixture->TestOrbitSettings();
        fixture->TearDown();
    });
    
    // Camera movement
    suite->AddTest("Camera Movement", [fixture]() {
        fixture->SetUp();
        fixture->TestCameraMovement();
        fixture->TearDown();
    });
    
    // Zoom functionality
    suite->AddTest("Camera Zoom", [fixture]() {
        fixture->SetUp();
        fixture->TestCameraZoom();
        fixture->TearDown();
    });
    
    // Mouse input
    suite->AddTest("Mouse Input", [fixture]() {
        fixture->SetUp();
        fixture->TestMouseInput();
        fixture->TearDown();
    });
    
    // Matrix generation
    suite->AddTest("View Projection Matrix", [fixture]() {
        fixture->SetUp();
        fixture->TestViewProjectionMatrix();
        fixture->TearDown();
    });
    
    return suite;
}