#include "Testing/TestFramework.h"
#include "Core/Coordinator.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Physics/WallRunningSystem.h"
#include "Gameplay/CharacterControllerSystem.h"
#include "Rendering/RenderComponents.h"
#include <memory>

using namespace CudaGame::Testing;
using namespace CudaGame::Core;
using namespace CudaGame::Physics;
using namespace CudaGame::Gameplay;
using namespace CudaGame::Rendering;

class CharacterControllerTestSuite {
private:
    std::shared_ptr<Coordinator> coordinator;
    std::shared_ptr<PhysXPhysicsSystem> physicsSystem;
    std::shared_ptr<CharacterControllerSystem> characterSystem;
    std::shared_ptr<WallRunningSystem> wallRunSystem;
    Entity player;
    const float EPSILON = 0.001f;
    const float FIXED_TIMESTEP = 1.0f / 60.0f;

public:
    void SetUp() {
        coordinator = std::make_shared<Coordinator>();
        coordinator->Initialize();
        
        // Register all required components
        coordinator->RegisterComponent<TransformComponent>();
        coordinator->RegisterComponent<RigidbodyComponent>();
        coordinator->RegisterComponent<ColliderComponent>();
        coordinator->RegisterComponent<CharacterControllerComponent>();
        coordinator->RegisterComponent<WallComponent>();
        coordinator->RegisterComponent<PlayerInputComponent>();
        coordinator->RegisterComponent<PlayerMovementComponent>();
        
        // Create and initialize systems
        physicsSystem = std::make_shared<PhysXPhysicsSystem>();
        characterSystem = std::make_shared<CharacterControllerSystem>();
        wallRunSystem = std::make_shared<WallRunningSystem>();
        
        physicsSystem->Initialize();
        characterSystem->Initialize();
        wallRunSystem->Initialize();
        
        // Create player entity
        player = coordinator->CreateEntity();
        SetupPlayerComponents(player);
    }

    void TearDown() {
        coordinator.reset();
        physicsSystem.reset();
        characterSystem.reset();
        wallRunSystem.reset();
    }

private:
    void SetupPlayerComponents(Entity entity) {
        // Transform
        TransformComponent transform;
        transform.position = glm::vec3(0.0f, 2.0f, 0.0f);
        transform.scale = glm::vec3(0.8f, 1.8f, 0.8f);
        coordinator->AddComponent(entity, transform);
        
        // Character controller
        CharacterControllerComponent controller;
        coordinator->AddComponent(entity, controller);
        
        // Physics
        RigidbodyComponent rb;
        rb.mass = 80.0f;
        coordinator->AddComponent(entity, rb);
        
        ColliderComponent collider;
        collider.shape = ColliderShape::BOX;
        collider.size = glm::vec3(0.8f, 1.8f, 0.8f);
        coordinator->AddComponent(entity, collider);
        
        // Movement
        PlayerMovementComponent movement;
        movement.baseSpeed = 10.0f;
        movement.maxSpeed = 20.0f;
        movement.jumpForce = 15.0f;
        coordinator->AddComponent(entity, movement);
        
        // Input (empty by default)
        coordinator->AddComponent(entity, PlayerInputComponent{});
    }

public:
    // Basic character controller tests
    void TestCharacterInitialization() {
        ASSERT_TRUE(coordinator->HasComponent<CharacterControllerComponent>(player));
        ASSERT_TRUE(coordinator->HasComponent<RigidbodyComponent>(player));
        ASSERT_TRUE(coordinator->HasComponent<ColliderComponent>(player));
        
        auto& transform = coordinator->GetComponent<TransformComponent>(player);
        ASSERT_NEAR(transform.position.y, 2.0f, EPSILON);
    }

    void TestBasicMovement() {
        // Simulate forward movement
        auto& input = coordinator->GetComponent<PlayerInputComponent>(player);
        input.keys[GLFW_KEY_W] = true;
        
        // Update for a few frames
        for (int i = 0; i < 5; i++) {
            characterSystem->Update(FIXED_TIMESTEP);
            physicsSystem->Update(FIXED_TIMESTEP);
        }
        
        auto& transform = coordinator->GetComponent<TransformComponent>(player);
        auto& rb = coordinator->GetComponent<RigidbodyComponent>(player);
        
        // Should have moved forward
        ASSERT_GT(glm::length(rb.velocity), 0.0f);
        ASSERT_GT(transform.position.z, 0.0f);
    }

    void TestJumping() {
        auto& input = coordinator->GetComponent<PlayerInputComponent>(player);
        float initialHeight = coordinator->GetComponent<TransformComponent>(player).position.y;
        
        // Trigger jump
        input.keys[GLFW_KEY_SPACE] = true;
        
        // Update for a few frames
        for (int i = 0; i < 5; i++) {
            characterSystem->Update(FIXED_TIMESTEP);
            physicsSystem->Update(FIXED_TIMESTEP);
        }
        
        float newHeight = coordinator->GetComponent<TransformComponent>(player).position.y;
        ASSERT_GT(newHeight, initialHeight);
    }

    void TestDoubleJump() {
        auto& input = coordinator->GetComponent<PlayerInputComponent>(player);
        float initialHeight = coordinator->GetComponent<TransformComponent>(player).position.y;
        
        // First jump
        input.keys[GLFW_KEY_SPACE] = true;
        characterSystem->Update(FIXED_TIMESTEP);
        physicsSystem->Update(FIXED_TIMESTEP);
        
        // Reset space key
        input.keys[GLFW_KEY_SPACE] = false;
        characterSystem->Update(FIXED_TIMESTEP);
        
        // Second jump
        input.keys[GLFW_KEY_SPACE] = true;
        
        // Update for a few frames
        for (int i = 0; i < 5; i++) {
            characterSystem->Update(FIXED_TIMESTEP);
            physicsSystem->Update(FIXED_TIMESTEP);
        }
        
        float finalHeight = coordinator->GetComponent<TransformComponent>(player).position.y;
        ASSERT_GT(finalHeight, initialHeight + 2.0f);  // Should gain significant height from double jump
    }

    void TestSprinting() {
        auto& input = coordinator->GetComponent<PlayerInputComponent>(player);
        auto& movement = coordinator->GetComponent<PlayerMovementComponent>(player);
        
        // Normal movement first
        input.keys[GLFW_KEY_W] = true;
        characterSystem->Update(FIXED_TIMESTEP);
        physicsSystem->Update(FIXED_TIMESTEP);
        
        float normalSpeed = glm::length(coordinator->GetComponent<RigidbodyComponent>(player).velocity);
        
        // Now sprint
        input.keys[GLFW_KEY_LEFT_SHIFT] = true;
        
        for (int i = 0; i < 5; i++) {
            characterSystem->Update(FIXED_TIMESTEP);
            physicsSystem->Update(FIXED_TIMESTEP);
        }
        
        float sprintSpeed = glm::length(coordinator->GetComponent<RigidbodyComponent>(player).velocity);
        ASSERT_GT(sprintSpeed, normalSpeed);
        ASSERT_LE(sprintSpeed, movement.maxSpeed);
    }

    // Wall running tests
    void TestWallRunningDetection() {
        // Create a wall
        Entity wall = coordinator->CreateEntity();
        
        TransformComponent wallTransform;
        wallTransform.position = glm::vec3(2.0f, 5.0f, 0.0f);
        wallTransform.scale = glm::vec3(1.0f, 10.0f, 10.0f);
        coordinator->AddComponent(wall, wallTransform);
        
        ColliderComponent wallCollider;
        wallCollider.shape = ColliderShape::BOX;
        wallCollider.size = wallTransform.scale;
        coordinator->AddComponent(wall, wallCollider);
        
        WallComponent wallComp;
        wallComp.canWallRun = true;
        coordinator->AddComponent(wall, wallComp);
        
        // Move player next to wall
        auto& playerTransform = coordinator->GetComponent<TransformComponent>(player);
        playerTransform.position = glm::vec3(1.5f, 5.0f, 0.0f);
        
        // Simulate wall run input
        auto& input = coordinator->GetComponent<PlayerInputComponent>(player);
        input.keys[GLFW_KEY_E] = true;  // Wall run key
        input.keys[GLFW_KEY_W] = true;  // Forward movement
        
        // Update systems
        for (int i = 0; i < 10; i++) {
            wallRunSystem->Update(FIXED_TIMESTEP);
            characterSystem->Update(FIXED_TIMESTEP);
            physicsSystem->Update(FIXED_TIMESTEP);
        }
        
        // Check if we're wall running
        auto& controller = coordinator->GetComponent<CharacterControllerComponent>(player);
        ASSERT_TRUE(controller.isWallRunning);
        
        // Verify wall run position is maintained
        auto& finalTransform = coordinator->GetComponent<TransformComponent>(player);
        ASSERT_NEAR(finalTransform.position.x, 1.5f, 0.5f);  // Should stay close to wall
        ASSERT_GT(finalTransform.position.y, 5.0f);  // Should maintain or gain height
    }

    void TestWallRunningGravity() {
        // Similar setup to wall detection test
        TestWallRunningDetection();
        
        // Get initial height
        float initialHeight = coordinator->GetComponent<TransformComponent>(player).position.y;
        
        // Update for several frames
        for (int i = 0; i < 30; i++) {
            wallRunSystem->Update(FIXED_TIMESTEP);
            characterSystem->Update(FIXED_TIMESTEP);
            physicsSystem->Update(FIXED_TIMESTEP);
        }
        
        // Height should not have significantly decreased
        float finalHeight = coordinator->GetComponent<TransformComponent>(player).position.y;
        ASSERT_NEAR(finalHeight, initialHeight, 1.0f);  // Allow small variation
    }

    void TestWallRunningJump() {
        // Setup wall running
        TestWallRunningDetection();
        
        // Get position before jump
        auto initialPos = coordinator->GetComponent<TransformComponent>(player).position;
        
        // Trigger wall jump
        auto& input = coordinator->GetComponent<PlayerInputComponent>(player);
        input.keys[GLFW_KEY_SPACE] = true;
        
        // Update for several frames
        for (int i = 0; i < 10; i++) {
            wallRunSystem->Update(FIXED_TIMESTEP);
            characterSystem->Update(FIXED_TIMESTEP);
            physicsSystem->Update(FIXED_TIMESTEP);
        }
        
        // Should have pushed away from wall and gained height
        auto& finalPos = coordinator->GetComponent<TransformComponent>(player).position;
        ASSERT_GT(glm::distance(finalPos, initialPos), 2.0f);  // Significant movement
        ASSERT_GT(finalPos.y, initialPos.y);  // Gained height
        ASSERT_LT(finalPos.x, initialPos.x);  // Pushed away from wall
    }
};

// Function to create and register the test suite
std::shared_ptr<TestSuite> CreateCharacterControllerTestSuite() {
    auto suite = std::make_shared<TestSuite>("Character Controller System");
    auto fixture = std::make_shared<CharacterControllerTestSuite>();
    
    // Basic movement tests
    suite->AddTest("Character Initialization", [fixture]() {
        fixture->SetUp();
        fixture->TestCharacterInitialization();
        fixture->TearDown();
    });
    
    suite->AddTest("Basic Movement", [fixture]() {
        fixture->SetUp();
        fixture->TestBasicMovement();
        fixture->TearDown();
    });
    
    suite->AddTest("Jumping", [fixture]() {
        fixture->SetUp();
        fixture->TestJumping();
        fixture->TearDown();
    });
    
    suite->AddTest("Double Jump", [fixture]() {
        fixture->SetUp();
        fixture->TestDoubleJump();
        fixture->TearDown();
    });
    
    suite->AddTest("Sprinting", [fixture]() {
        fixture->SetUp();
        fixture->TestSprinting();
        fixture->TearDown();
    });
    
    // Wall running tests
    suite->AddTest("Wall Running Detection", [fixture]() {
        fixture->SetUp();
        fixture->TestWallRunningDetection();
        fixture->TearDown();
    });
    
    suite->AddTest("Wall Running Gravity", [fixture]() {
        fixture->SetUp();
        fixture->TestWallRunningGravity();
        fixture->TearDown();
    });
    
    suite->AddTest("Wall Running Jump", [fixture]() {
        fixture->SetUp();
        fixture->TestWallRunningJump();
        fixture->TearDown();
    });
    
    return suite;
}