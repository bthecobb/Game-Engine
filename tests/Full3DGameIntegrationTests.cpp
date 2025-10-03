#include "Testing/TestFramework.h"
#include "Core/Coordinator.h"
#include "Gameplay/PlayerMovementSystem.h"
#include "Gameplay/CharacterControllerSystem.h"
#include "Gameplay/EnemyAISystem.h"
#include "Gameplay/LevelSystem.h"
#include "Gameplay/TargetingSystem.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Physics/WallRunningSystem.h"
#include "Rendering/RenderSystem.h"
#include "Rendering/OrbitCamera.h"
#include "Rendering/RenderDebugSystem.h"
#include "Animation/AnimationSystem.h"
#include <memory>
#include <chrono>
#include <random>
#include <sstream>

using namespace CudaGame::Testing;
using namespace CudaGame::Core;
using namespace CudaGame::Gameplay;
using namespace CudaGame::Physics;
using namespace CudaGame::Rendering;
using namespace CudaGame::Animation;

// Advanced performance monitoring for integration tests
class GamePerformanceMonitor {
private:
    struct FrameMetrics {
        double frameTime;
        double physicsTime;
        double renderTime;
        double aiTime;
        double memoryUsage;
        std::vector<std::string> warnings;
    };
    
    std::vector<FrameMetrics> metrics;
    std::chrono::high_resolution_clock::time_point testStart;
    const size_t maxFrames;
    
public:
    explicit GamePerformanceMonitor(size_t maxFrames = 1000) 
        : maxFrames(maxFrames) {
        metrics.reserve(maxFrames);
        testStart = std::chrono::high_resolution_clock::now();
    }
    
    void RecordFrame(const FrameMetrics& frame) {
        metrics.push_back(frame);
        if (metrics.size() > maxFrames) {
            metrics.erase(metrics.begin());
        }
        
        // Log warnings for performance issues
        if (!frame.warnings.empty()) {
            std::stringstream ss;
            ss << "Frame " << metrics.size() << " warnings:\n";
            for (const auto& warning : frame.warnings) {
                ss << "  - " << warning << "\n";
            }
            std::cout << ss.str();
        }
    }
    
    double GetAverageFrameTime() const {
        if (metrics.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& frame : metrics) {
            sum += frame.frameTime;
        }
        return sum / metrics.size();
    }
    
    double GetWorstFrameTime() const {
        if (metrics.empty()) return 0.0;
        return std::max_element(metrics.begin(), metrics.end(),
            [](const FrameMetrics& a, const FrameMetrics& b) {
                return a.frameTime < b.frameTime;
            })->frameTime;
    }
    
    double GetAverageMemoryUsage() const {
        if (metrics.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& frame : metrics) {
            sum += frame.memoryUsage;
        }
        return sum / metrics.size();
    }
    
    std::string GenerateReport() const {
        std::stringstream ss;
        ss << "\n=== Performance Report ===\n";
        ss << "Test duration: " << GetTestDuration() << " seconds\n";
        ss << "Total frames: " << metrics.size() << "\n";
        ss << "Average frame time: " << GetAverageFrameTime() << "ms\n";
        ss << "Worst frame time: " << GetWorstFrameTime() << "ms\n";
        ss << "Average memory usage: " << GetAverageMemoryUsage() << "MB\n";
        ss << "Frame time distribution:\n";
        ss << "  Physics: " << GetAveragePhysicsPercentage() << "%\n";
        ss << "  Render: " << GetAverageRenderPercentage() << "%\n";
        ss << "  AI: " << GetAverageAIPercentage() << "%\n";
        return ss.str();
    }
    
private:
    double GetTestDuration() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - testStart).count();
    }
    
    double GetAveragePhysicsPercentage() const {
        if (metrics.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& frame : metrics) {
            sum += (frame.physicsTime / frame.frameTime) * 100.0;
        }
        return sum / metrics.size();
    }
    
    double GetAverageRenderPercentage() const {
        if (metrics.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& frame : metrics) {
            sum += (frame.renderTime / frame.frameTime) * 100.0;
        }
        return sum / metrics.size();
    }
    
    double GetAverageAIPercentage() const {
        if (metrics.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& frame : metrics) {
            sum += (frame.aiTime / frame.frameTime) * 100.0;
        }
        return sum / metrics.size();
    }
};

// Game state validation utilities
class GameStateValidator {
public:
    struct ValidationResult {
        bool success;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    
    static ValidationResult ValidatePlayerState(const Entity player,
                                             const std::shared_ptr<Coordinator>& coordinator) {
        ValidationResult result{true};
        
        // Check required components
        if (!coordinator->HasComponent<PlayerMovementComponent>(player)) {
            result.success = false;
            result.errors.push_back("Player missing movement component");
        }
        
        if (!coordinator->HasComponent<CharacterControllerComponent>(player)) {
            result.success = false;
            result.errors.push_back("Player missing character controller");
        }
        
        if (!coordinator->HasComponent<RigidbodyComponent>(player)) {
            result.success = false;
            result.errors.push_back("Player missing rigidbody");
        }
        
        // Validate transform
        if (coordinator->HasComponent<TransformComponent>(player)) {
            const auto& transform = coordinator->GetComponent<TransformComponent>(player);
            if (glm::length(transform.scale) < 0.1f) {
                result.warnings.push_back("Player scale unusually small");
            }
            if (transform.position.y < -100.0f) {
                result.errors.push_back("Player below world bounds");
            }
        }
        
        // Validate physics state
        if (coordinator->HasComponent<RigidbodyComponent>(player)) {
            const auto& rb = coordinator->GetComponent<RigidbodyComponent>(player);
            if (glm::length(rb.velocity) > 100.0f) {
                result.warnings.push_back("Player velocity unusually high");
            }
        }
        
        return result;
    }
    
    static ValidationResult ValidateWorldState(const std::shared_ptr<Coordinator>& coordinator) {
        ValidationResult result{true};
        
        // Verify essential systems
        if (!coordinator->GetSystem<PhysXPhysicsSystem>()) {
            result.errors.push_back("Physics system not initialized");
        }
        if (!coordinator->GetSystem<RenderSystem>()) {
            result.errors.push_back("Render system not initialized");
        }
        
        // Check entity limits
        size_t entityCount = coordinator->GetEntityCount();
        if (entityCount > 10000) {
            result.warnings.push_back("High entity count: " + std::to_string(entityCount));
        }
        
        return result;
    }
};

class Full3DGameTestSuite {
private:
    std::shared_ptr<Coordinator> coordinator;
    std::shared_ptr<PlayerMovementSystem> movementSystem;
    std::shared_ptr<CharacterControllerSystem> characterSystem;
    std::shared_ptr<EnemyAISystem> aiSystem;
    std::shared_ptr<LevelSystem> levelSystem;
    std::shared_ptr<TargetingSystem> targetingSystem;
    std::shared_ptr<PhysXPhysicsSystem> physicsSystem;
    std::shared_ptr<WallRunningSystem> wallRunSystem;
    std::shared_ptr<RenderSystem> renderSystem;
    std::shared_ptr<RenderDebugSystem> debugSystem;
    std::shared_ptr<AnimationSystem> animationSystem;
    
    std::unique_ptr<GamePerformanceMonitor> perfMonitor;
    Entity player;
    
    const float FIXED_TIMESTEP = 1.0f / 60.0f;
    const float EPSILON = 0.001f;

public:
    void SetUp() {
        coordinator = std::make_shared<Coordinator>();
        coordinator->Initialize();
        
        // Register all components
        RegisterComponents();
        
        // Create and initialize all systems
        InitializeSystems();
        
        // Create player and world
        player = CreatePlayer();
        CreateTestWorld();
        
        // Initialize performance monitoring
        perfMonitor = std::make_unique<GamePerformanceMonitor>();
    }

    void TearDown() {
        // Log final performance report
        std::cout << perfMonitor->GenerateReport();
        
        // Validate final state
        auto finalState = GameStateValidator::ValidateWorldState(coordinator);
        ASSERT_TRUE(finalState.success) << "World state invalid at teardown";
        
        // Cleanup systems in correct order
        aiSystem.reset();
        wallRunSystem.reset();
        physicsSystem.reset();
        renderSystem.reset();
        debugSystem.reset();
        coordinator.reset();
    }

    // Integration Test: Full Game Loop
    void TestFullGameLoop() {
        const int FRAME_COUNT = 1000;
        
        for (int frame = 0; frame < FRAME_COUNT; frame++) {
            auto frameStart = std::chrono::high_resolution_clock::now();
            GamePerformanceMonitor::FrameMetrics metrics;
            
            // Update game systems
            UpdateGameSystems(metrics);
            
            // Validate frame
            ValidateFrameState(frame);
            
            // Record performance
            auto frameEnd = std::chrono::high_resolution_clock::now();
            metrics.frameTime = std::chrono::duration<double, std::milli>(
                frameEnd - frameStart).count();
            perfMonitor->RecordFrame(metrics);
            
            // Early exit if performance is severely degraded
            if (metrics.frameTime > 100.0) {  // >100ms is unacceptable
                FAIL() << "Performance degradation detected. Frame time: "
                      << metrics.frameTime << "ms";
            }
        }
    }

    // Integration Test: Combat System
    void TestCombatSystem() {
        // Create enemy for combat test
        Entity enemy = CreateTestEnemy();
        
        // Setup combat scenario
        SetupCombatScenario(enemy);
        
        // Run combat sequence
        const int COMBAT_FRAMES = 300;
        for (int frame = 0; frame < COMBAT_FRAMES; frame++) {
            // Simulate player attack input
            if (frame == 60) {
                TriggerPlayerAttack();
            }
            
            // Update systems
            UpdateCombatSystems();
            
            // Validate combat state
            ValidateCombatState(enemy);
        }
    }

    // Integration Test: Movement and Physics
    void TestMovementAndPhysics() {
        // Test different movement scenarios
        TestBasicMovement();
        TestWallRunning();
        TestDoubleJump();
        TestCombatMovement();
    }

    // Integration Test: Stress Test
    void TestStressScenario() {
        // Create busy scene
        CreateStressTestScene();
        
        // Run stress test
        const int STRESS_FRAMES = 600;
        for (int frame = 0; frame < STRESS_FRAMES; frame++) {
            UpdateGameSystems(GamePerformanceMonitor::FrameMetrics{});
            
            // Add random events every 60 frames
            if (frame % 60 == 0) {
                AddRandomStressEvents();
            }
            
            // Validate system stability
            ValidateSystemStability();
        }
    }

private:
    void RegisterComponents() {
        coordinator->RegisterComponent<PlayerMovementComponent>();
        coordinator->RegisterComponent<CharacterControllerComponent>();
        coordinator->RegisterComponent<EnemyAIComponent>();
        coordinator->RegisterComponent<TargetingComponent>();
        coordinator->RegisterComponent<RigidbodyComponent>();
        coordinator->RegisterComponent<ColliderComponent>();
        coordinator->RegisterComponent<TransformComponent>();
        coordinator->RegisterComponent<MeshComponent>();
        coordinator->RegisterComponent<MaterialComponent>();
        coordinator->RegisterComponent<AnimationComponent>();
    }

    void InitializeSystems() {
        movementSystem = coordinator->RegisterSystem<PlayerMovementSystem>();
        characterSystem = coordinator->RegisterSystem<CharacterControllerSystem>();
        aiSystem = coordinator->RegisterSystem<EnemyAISystem>();
        levelSystem = coordinator->RegisterSystem<LevelSystem>();
        targetingSystem = coordinator->RegisterSystem<TargetingSystem>();
        physicsSystem = coordinator->RegisterSystem<PhysXPhysicsSystem>();
        wallRunSystem = coordinator->RegisterSystem<WallRunningSystem>();
        renderSystem = coordinator->RegisterSystem<RenderSystem>();
        debugSystem = coordinator->RegisterSystem<RenderDebugSystem>();
        animationSystem = coordinator->RegisterSystem<AnimationSystem>();
        
        // Initialize all systems
        movementSystem->Initialize();
        characterSystem->Initialize();
        aiSystem->Initialize();
        levelSystem->Initialize();
        targetingSystem->Initialize();
        physicsSystem->Initialize();
        wallRunSystem->Initialize();
        renderSystem->Initialize();
        debugSystem->Initialize();
        animationSystem->Initialize();
    }

    Entity CreatePlayer() {
        Entity entity = coordinator->CreateEntity();
        
        // Add components with AAA-game standard configuration
        AddPlayerComponents(entity);
        
        return entity;
    }

    void CreateTestWorld() {
        // Create ground
        Entity ground = coordinator->CreateEntity();
        AddGroundComponents(ground);
        
        // Create walls for parkour
        CreateParkourEnvironment();
        
        // Create combat areas
        CreateCombatAreas();
        
        // Add atmospheric elements
        CreateAtmosphericElements();
    }

    void UpdateGameSystems(GamePerformanceMonitor::FrameMetrics& metrics) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // AI Update
        auto aiStart = std::chrono::high_resolution_clock::now();
        aiSystem->Update(FIXED_TIMESTEP);
        metrics.aiTime = GetDurationMs(aiStart);
        
        // Physics Update
        auto physicsStart = std::chrono::high_resolution_clock::now();
        physicsSystem->Update(FIXED_TIMESTEP);
        wallRunSystem->Update(FIXED_TIMESTEP);
        characterSystem->Update(FIXED_TIMESTEP);
        metrics.physicsTime = GetDurationMs(physicsStart);
        
        // Rendering Update
        auto renderStart = std::chrono::high_resolution_clock::now();
        renderSystem->Update(FIXED_TIMESTEP);
        debugSystem->Update(FIXED_TIMESTEP);
        metrics.renderTime = GetDurationMs(renderStart);
        
        // Animation Update
        animationSystem->Update(FIXED_TIMESTEP);
        
        metrics.frameTime = GetDurationMs(start);
    }

    void ValidateFrameState(int frameNumber) {
        // Validate player state
        auto playerState = GameStateValidator::ValidatePlayerState(player, coordinator);
        ASSERT_TRUE(playerState.success) << "Player state validation failed at frame "
                                       << frameNumber;
        
        // Validate world state
        auto worldState = GameStateValidator::ValidateWorldState(coordinator);
        ASSERT_TRUE(worldState.success) << "World state validation failed at frame "
                                      << frameNumber;
        
        // Log any warnings
        for (const auto& warning : playerState.warnings) {
            std::cout << "Frame " << frameNumber << " warning: " << warning << "\n";
        }
    }

    double GetDurationMs(const std::chrono::high_resolution_clock::time_point& start) {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start).count();
    }

    // Additional private utility methods...
    void AddPlayerComponents(Entity entity) {
        // Transform
        TransformComponent transform;
        transform.position = glm::vec3(0.0f, 2.0f, 0.0f);
        transform.scale = glm::vec3(0.8f, 1.8f, 0.8f);
        coordinator->AddComponent(entity, transform);
        
        // Physics
        RigidbodyComponent rb;
        rb.mass = 80.0f;
        rb.drag = 0.1f;
        coordinator->AddComponent(entity, rb);
        
        ColliderComponent collider;
        collider.shape = ColliderShape::CAPSULE;
        collider.radius = 0.4f;
        collider.height = 1.8f;
        coordinator->AddComponent(entity, collider);
        
        // Movement
        PlayerMovementComponent movement;
        movement.baseSpeed = 10.0f;
        movement.maxSpeed = 20.0f;
        movement.jumpForce = 15.0f;
        coordinator->AddComponent(entity, movement);
        
        // Character Controller
        CharacterControllerComponent controller;
        coordinator->AddComponent(entity, controller);
        
        // Animation
        AnimationComponent anim;
        anim.currentState = "idle";
        coordinator->AddComponent(entity, anim);
    }

    void CreateParkourEnvironment() {
        const int WALL_COUNT = 10;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> pos(-50.0, 50.0);
        std::uniform_real_distribution<> height(10.0, 20.0);
        
        for (int i = 0; i < WALL_COUNT; i++) {
            Entity wall = coordinator->CreateEntity();
            
            TransformComponent transform;
            transform.position = glm::vec3(pos(gen), height(gen)/2.0f, pos(gen));
            transform.scale = glm::vec3(1.0f, height(gen), 10.0f);
            coordinator->AddComponent(wall, transform);
            
            ColliderComponent collider;
            collider.shape = ColliderShape::BOX;
            collider.size = transform.scale;
            coordinator->AddComponent(wall, collider);
            
            // Make wall runnable
            coordinator->AddComponent(wall, WallComponent{true});
        }
    }
};

// Function to create and register the test suite
std::shared_ptr<TestSuite> CreateFull3DGameIntegrationTestSuite() {
    auto suite = std::make_shared<TestSuite>("Full3DGame Integration");
    auto fixture = std::make_shared<Full3DGameTestSuite>();
    
    // Full game loop test
    suite->AddTest("Full Game Loop", [fixture]() {
        fixture->SetUp();
        fixture->TestFullGameLoop();
        fixture->TearDown();
    });
    
    // Combat system test
    suite->AddTest("Combat System", [fixture]() {
        fixture->SetUp();
        fixture->TestCombatSystem();
        fixture->TearDown();
    });
    
    // Movement and physics test
    suite->AddTest("Movement and Physics", [fixture]() {
        fixture->SetUp();
        fixture->TestMovementAndPhysics();
        fixture->TearDown();
    });
    
    // Stress test
    suite->AddTest("Stress Test", [fixture]() {
        fixture->SetUp();
        fixture->TestStressScenario();
        fixture->TearDown();
    });
    
    return suite;
}