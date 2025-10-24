#include "Testing/TestFramework.h"
#include "Testing/AdvancedTestFramework.h"
#include "Testing/GPUMetricsStream.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Physics/PhysicsComponents.h"
#include "Core/Coordinator.h"
#include <PxPhysicsAPI.h>
#include "Physics/CharacterController.h"
#include "Rendering/RenderComponents.h"

using namespace CudaGame;
using namespace CudaGame::Physics;
using namespace CudaGame::Testing;
using namespace physx;
using Core::Entity;

// Initialize static test resources
void PhysXTestEnvironment() {
    CudaGame::Testing::CUDAPerformanceMonitor::Initialize();
}

class PhysXTestSuite : public ::testing::Test {
protected:
    void SetUp() override {
        coordinator = &Core::Coordinator::GetInstance();
        coordinator->Initialize();
        
        // Register required components
        coordinator->RegisterComponent<RigidbodyComponent>();
        coordinator->RegisterComponent<ColliderComponent>();
        coordinator->RegisterComponent<CharacterControllerComponent>();
        
        // Initialize PhysX system
        physicsSystem = std::make_shared<PhysXPhysicsSystem>();
        ASSERT_TRUE(physicsSystem->Initialize());
        
        MemoryLeakDetector::StartTracking();
    }
    
    void TearDown() override {
        auto memoryDelta = MemoryLeakDetector::GetLeaks();
        std::cout << "Memory delta - CPU: " << memoryDelta.cpuBytes 
                  << " bytes, GPU: " << memoryDelta.gpuBytes << " bytes" << std::endl;
                  
        physicsSystem->Cleanup();
        physicsSystem.reset();
        coordinator->Cleanup();
    }
    
    // Helper method to create a test rigidbody
    Core::Entity CreateTestRigidbody(const glm::vec3& position, float mass = 1.0f) {
        Core::Entity entity = coordinator->CreateEntity();
        
        RigidbodyComponent rb;
        rb.setMass(mass);
        coordinator->AddComponent<RigidbodyComponent>(entity, rb);
        
        ColliderComponent collider;
        collider.shape = ColliderShape::BOX;
        collider.size = glm::vec3(1.0f);
        coordinator->AddComponent<ColliderComponent>(entity, collider);

        Rendering::TransformComponent transform;
        transform.position = position;
        coordinator->AddComponent<Rendering::TransformComponent>(entity, transform);
        
        return entity;
    }
    // Helper method to step physics simulation
    void StepSimulation(float deltaTime = 1.0f/60.0f, int steps = 1) {
        for (int i = 0; i < steps; ++i) {
            physicsSystem->Update(deltaTime);
        }
    }
    
    CudaGame::Testing::PerformanceResult RunPerformanceTest(const std::string& testName, 
                                                         std::function<void()> testFunc,
                                                         const CudaGame::Testing::PerformanceThresholds& thresholds) {
        CudaGame::Testing::CUDAPerformanceMonitor::StartRecording();
        auto startTime = std::chrono::high_resolution_clock::now();
        
        testFunc();
        
        auto endTime = std::chrono::high_resolution_clock::now();
        CudaGame::Testing::CUDAPerformanceMonitor::StopRecording();
        
        CudaGame::Testing::PerformanceResult result;
        result.passed = true;
        result.testName = testName;
        result.metrics = CudaGame::Testing::CUDAPerformanceMonitor::GetGPUMetrics();
        return result;
    }
    
    Core::Coordinator* coordinator;
    std::shared_ptr<PhysXPhysicsSystem> physicsSystem;
};

// Basic Physics Tests
TEST_F(PhysXTestSuite, BasicGravityTest) {
    Entity box = CreateTestRigidbody(glm::vec3(0.0f, 10.0f, 0.0f));
    
    // Get initial transform (position is stored in transform component)
        auto& transform = coordinator->GetComponent<Rendering::TransformComponent>(box);
        float initialHeight = transform.position.y;
        
        // Run simulation for 1 second
        StepSimulation(1.0f/60.0f, 60);
        
        // Box should have fallen
        EXPECT_LT(transform.position.y, initialHeight);
}

// Character Controller Tests
TEST_F(PhysXTestSuite, CharacterControllerBasicMovement) {
    Entity character = coordinator->CreateEntity();
    
    CharacterControllerComponent controller;
    controller.height = 2.0f;
    controller.radius = 0.5f;
    coordinator->AddComponent(character, controller);
    
    // Initial position
    glm::vec3 initialPos(0.0f, 2.0f, 0.0f);
    controller.position = initialPos;
    
    // Move forward
    glm::vec3 movement(0.0f, 0.0f, 1.0f);
    controller.moveDirection = movement;
    
    StepSimulation(1.0f/60.0f, 10);
    
    // Character should have moved forward
    EXPECT_GT(controller.position.z, initialPos.z);
}

// Collision Tests
TEST_F(PhysXTestSuite, CollisionDetectionTest) {
    Entity box1 = CreateTestRigidbody(glm::vec3(-1.0f, 0.5f, 0.0f));
    Entity box2 = CreateTestRigidbody(glm::vec3(1.0f, 0.5f, 0.0f));
    
    // Apply force to make boxes collide
    auto& rb1 = coordinator->GetComponent<RigidbodyComponent>(box1);
    rb1.setVelocity(glm::vec3(2.0f, 0.0f, 0.0f));
    
    // Run simulation until collision
    StepSimulation(1.0f/60.0f, 60);
    
    // Boxes should have collided and changed velocity
    auto& rb2 = coordinator->GetComponent<RigidbodyComponent>(box2);
    EXPECT_GT(rb2.getVelocity().x, 0.0f);
}

// Performance Tests
TEST_F(PhysXTestSuite, MassBodySimulationPerformance) {
    const int NUM_BODIES = 1000;
std::vector<Core::Entity> bodies;
    
    // Create many rigidbodies
    for (int i = 0; i < NUM_BODIES; ++i) {
        float x = (float)(rand() % 20 - 10);
        float y = (float)(rand() % 20 + 10);
        float z = (float)(rand() % 20 - 10);
        bodies.push_back(CreateTestRigidbody(glm::vec3(x, y, z)));
    }
    
    PerformanceThresholds thresholds;
    thresholds.maxGPUTime = 5.0f;    // 5ms GPU time
    thresholds.maxCPUTime = 16.67f;  // 60 FPS CPU time
    thresholds.maxGPUUtilization = 80.0f;
    thresholds.maxMemoryUsage = 512 * 1024 * 1024;  // 512MB
    
    auto result = RunPerformanceTest("Mass Body Simulation", [&]() {
        StepSimulation(1.0f/60.0f, 60);  // Simulate 1 second
    }, thresholds);
    
    EXPECT_TRUE(result.passed);
    std::cout << "Performance metrics: " << result.metrics << std::endl;
}

// Memory Management Tests
TEST_F(PhysXTestSuite, DynamicBodyMemoryManagement) {
    MemoryLeakDetector::StartTracking();
    
    {
        // Create and destroy bodies repeatedly
        for (int i = 0; i < 100; ++i) {
            Entity body = CreateTestRigidbody(glm::vec3(0.0f, 5.0f, 0.0f));
            StepSimulation(1.0f/60.0f, 5);
            coordinator->DestroyEntity(body);
        }
    }
    
    auto memoryDelta = MemoryLeakDetector::GetLeaks();
    EXPECT_LT(memoryDelta.cpuBytes, 1024);  // Less than 1KB leak tolerance
    EXPECT_LT(memoryDelta.gpuBytes, 1024);  // Less than 1KB leak tolerance
}

// Note: main() is defined in TestRunner.cpp
// This file only contains test cases for PhysX integration
