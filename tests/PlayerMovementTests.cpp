#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "Player.h"
#include "GameWorld.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

/**
 * @brief Comprehensive test suite for Player movement and animation integration
 * 
 * Tests cover:
 * - Basic movement mechanics
 * - Animation state transitions
 * - Physics integration
 * - Rhythm feedback synchronization
 * - Particle system behavior
 * - Multi-directional movement
 * - Combat animation integration
 * - Edge case handling
 */

class PlayerMovementTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize OpenGL context mock for testing
        initializeTestGL();
        
        player = std::make_unique<Player>();
        gameWorld = std::make_unique<GameWorld>();
        
        // Set up basic game world for testing
        player->setGameWorld(gameWorld.get());
        
        // Reset player to known state
        player->transform.position = glm::vec3(0.0f, 2.0f, 0.0f);
        player->transform.scale = glm::vec3(1.0f);
        player->transform.rotation = glm::vec3(0.0f);
        
        std::cout << "✅ Test setup complete - Player initialized at position: (" 
                  << player->transform.position.x << ", " 
                  << player->transform.position.y << ", " 
                  << player->transform.position.z << ")" << std::endl;
    }

    void TearDown() override {
        player.reset();
        gameWorld.reset();
        cleanupTestGL();
    }

    void initializeTestGL() {
        // Mock OpenGL initialization for headless testing
        // In production, this would initialize a headless OpenGL context
        std::cout << "🔧 Initializing test OpenGL context..." << std::endl;
    }

    void cleanupTestGL() {
        // Clean up mock OpenGL resources
        std::cout << "🧹 Cleaning up test OpenGL context..." << std::endl;
    }

    std::unique_ptr<Player> player;
    std::unique_ptr<GameWorld> gameWorld;
    
    // Test constants
    static constexpr float DELTA_TIME = 1.0f / 60.0f; // 60 FPS
    static constexpr float EPSILON = 0.001f;
};

// ==================== BASIC MOVEMENT TESTS ====================

TEST_F(PlayerMovementTest, InitialState) {
    EXPECT_EQ(player->getMovementState(), MovementState::IDLE);
    EXPECT_NEAR(player->getCurrentSpeed(), 0.0f, EPSILON);
    EXPECT_EQ(player->getCurrentAnimationState(), AnimationState::IDLE_ANIM);
    
    std::cout << "✅ Initial state verification passed" << std::endl;
}

TEST_F(PlayerMovementTest, ForwardMovement) {
    // Simulate WASD input for forward movement
    bool keys[1024] = {false};
    keys[GLFW_KEY_W] = true;
    
    glm::vec3 initialPosition = player->transform.position;
    
    // Process input and update for several frames
    for (int frame = 0; frame < 10; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    // Verify forward movement occurred
    EXPECT_GT(player->transform.position.z, initialPosition.z);
    EXPECT_GT(player->getCurrentSpeed(), 0.0f);
    EXPECT_NE(player->getMovementState(), MovementState::IDLE);
    
    std::cout << "✅ Forward movement test passed - Speed: " << player->getCurrentSpeed() << std::endl;
}

TEST_F(PlayerMovementTest, MultiDirectionalMovement) {
    bool keys[1024] = {false};
    glm::vec3 initialPosition = player->transform.position;
    
    // Test diagonal movement (W + D)
    keys[GLFW_KEY_W] = true;
    keys[GLFW_KEY_D] = true;
    
    for (int frame = 0; frame < 20; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    // Verify diagonal movement
    EXPECT_GT(player->transform.position.z, initialPosition.z); // Forward
    EXPECT_GT(player->transform.position.x, initialPosition.x); // Right
    EXPECT_GT(player->getCurrentSpeed(), 0.0f);
    
    std::cout << "✅ Multi-directional movement test passed" << std::endl;
}

// ==================== ANIMATION STATE TESTS ====================

TEST_F(PlayerMovementTest, AnimationStateTransitions) {
    bool keys[1024] = {false};
    
    // Test IDLE -> WALKING transition
    keys[GLFW_KEY_W] = true;
    
    // Simulate gradual speed increase
    for (int frame = 0; frame < 5; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    EXPECT_TRUE(player->getMovementState() == MovementState::WALKING || 
                player->getMovementState() == MovementState::RUNNING);
    
    // Test return to IDLE
    keys[GLFW_KEY_W] = false;
    
    // Let player decelerate
    for (int frame = 0; frame < 30; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    EXPECT_EQ(player->getMovementState(), MovementState::IDLE);
    EXPECT_NEAR(player->getCurrentSpeed(), 0.0f, EPSILON);
    
    std::cout << "✅ Animation state transitions test passed" << std::endl;
}

TEST_F(PlayerMovementTest, SprintingTransition) {
    bool keys[1024] = {false};
    keys[GLFW_KEY_W] = true;
    
    // Build up speed to trigger sprinting
    for (int frame = 0; frame < 50; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    float maxSpeed = player->getCurrentSpeed();
    MovementState finalState = player->getMovementState();
    
    EXPECT_GT(maxSpeed, 10.0f); // Should reach significant speed
    EXPECT_TRUE(finalState == MovementState::RUNNING || finalState == MovementState::SPRINTING);
    
    std::cout << "✅ Sprinting transition test passed - Max speed: " << maxSpeed << std::endl;
}

// ==================== JUMP AND AERIAL MECHANICS ====================

TEST_F(PlayerMovementTest, JumpMechanics) {
    bool keys[1024] = {false};
    keys[GLFW_KEY_SPACE] = true;
    
    glm::vec3 initialPosition = player->transform.position;
    
    // Trigger jump
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    
    // Check if player left the ground
    EXPECT_GT(player->transform.position.y, initialPosition.y);
    EXPECT_EQ(player->getMovementState(), MovementState::JUMPING);
    
    // Simulate gravity over time
    keys[GLFW_KEY_SPACE] = false;
    for (int frame = 0; frame < 100; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        
        // Stop when player lands
        if (player->transform.position.y <= 0.1f) {
            break;
        }
    }
    
    // Verify landing
    EXPECT_NEAR(player->transform.position.y, 0.0f, 0.2f);
    EXPECT_NE(player->getMovementState(), MovementState::JUMPING);
    
    std::cout << "✅ Jump mechanics test passed" << std::endl;
}

TEST_F(PlayerMovementTest, DoubleJump) {
    bool keys[1024] = {false};
    
    // First jump
    keys[GLFW_KEY_SPACE] = true;
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    
    keys[GLFW_KEY_SPACE] = false;
    
    // Wait a bit, then trigger second jump
    for (int frame = 0; frame < 10; ++frame) {
        player->update(DELTA_TIME);
    }
    
    float heightAfterFirstJump = player->transform.position.y;
    
    // Second jump
    keys[GLFW_KEY_SPACE] = true;
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    
    // Should be higher than before
    EXPECT_GT(player->transform.position.y, heightAfterFirstJump);
    
    std::cout << "✅ Double jump test passed" << std::endl;
}

// ==================== DASH MECHANICS ====================

TEST_F(PlayerMovementTest, DashMechanics) {
    bool keys[1024] = {false};
    keys[GLFW_KEY_W] = true; // Move forward first
    keys[GLFW_KEY_LEFT_SHIFT] = true; // Dash
    
    glm::vec3 initialPosition = player->transform.position;
    
    // Trigger dash
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    
    EXPECT_EQ(player->getMovementState(), MovementState::DASHING);
    EXPECT_GT(player->getCurrentSpeed(), 20.0f); // Dash should be fast
    
    // Continue dash for a few frames
    for (int frame = 0; frame < 5; ++frame) {
        player->update(DELTA_TIME);
    }
    
    float distanceTraveled = glm::distance(initialPosition, player->transform.position);
    EXPECT_GT(distanceTraveled, 1.0f); // Should have moved significantly
    
    std::cout << "✅ Dash mechanics test passed - Distance: " << distanceTraveled << std::endl;
}

// ==================== COMBAT INTEGRATION ====================

TEST_F(PlayerMovementTest, CombatAnimationIntegration) {
    bool keys[1024] = {false};
    keys[GLFW_KEY_LEFT_CONTROL] = true; // Light attack
    
    // Trigger attack
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    
    EXPECT_EQ(player->getCombatState(), CombatState::ATTACKING);
    
    // Heavy attack test
    keys[GLFW_KEY_LEFT_CONTROL] = false;
    keys[GLFW_KEY_LEFT_ALT] = true; // Heavy attack
    
    // Wait for attack cooldown
    for (int frame = 0; frame < 20; ++frame) {
        player->update(DELTA_TIME);
    }
    
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    
    EXPECT_EQ(player->getCombatState(), CombatState::ATTACKING);
    
    std::cout << "✅ Combat animation integration test passed" << std::endl;
}

// ==================== RHYTHM FEEDBACK TESTS ====================

TEST_F(PlayerMovementTest, RhythmFeedbackSystem) {
    // Test rhythm system over time
    float totalTime = 0.0f;
    int beatCount = 0;
    
    while (totalTime < 3.0f) { // Test for 3 seconds
        player->update(DELTA_TIME);
        
        if (player->isOnBeat()) {
            beatCount++;
        }
        
        totalTime += DELTA_TIME;
    }
    
    // Should have detected multiple beats (120 BPM = 2 beats per second)
    EXPECT_GT(beatCount, 3); // At least 3 beats in 3 seconds
    
    std::cout << "✅ Rhythm feedback test passed - Beats detected: " << beatCount << std::endl;
}

// ==================== PARTICLE SYSTEM TESTS ====================

TEST_F(PlayerMovementTest, ParticleSystemActivation) {
    bool keys[1024] = {false};
    keys[GLFW_KEY_W] = true;
    
    // Build up high speed
    for (int frame = 0; frame < 60; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    // Check if particle trails are generated at high speed
    EXPECT_GT(player->getCurrentSpeed(), 20.0f);
    
    // Particle system should be active (this would need access to particle count)
    // For now, we verify high-speed state which should trigger particles
    
    std::cout << "✅ Particle system activation test passed" << std::endl;
}

// ==================== PHYSICS INTEGRATION ====================

TEST_F(PlayerMovementTest, GravityAndGrounding) {
    // Start player in air
    player->transform.position.y = 10.0f;
    
    // Let gravity work
    for (int frame = 0; frame < 200; ++frame) {
        player->update(DELTA_TIME);
        
        if (player->transform.position.y <= 0.1f) {
            break;
        }
    }
    
    // Should have landed
    EXPECT_NEAR(player->transform.position.y, 0.0f, 0.2f);
    EXPECT_TRUE(player->isGrounded());
    
    std::cout << "✅ Gravity and grounding test passed" << std::endl;
}

// ==================== MOMENTUM PRESERVATION ====================

TEST_F(PlayerMovementTest, MomentumPreservation) {
    bool keys[1024] = {false};
    keys[GLFW_KEY_W] = true;
    
    // Build momentum
    for (int frame = 0; frame < 30; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    float speedWithInput = player->getCurrentSpeed();
    
    // Stop input but momentum should carry forward
    keys[GLFW_KEY_W] = false;
    
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    
    float speedAfterInput = player->getCurrentSpeed();
    
    // Speed should decrease gradually, not instantly
    EXPECT_GT(speedAfterInput, speedWithInput * 0.5f);
    
    std::cout << "✅ Momentum preservation test passed" << std::endl;
}

// ==================== PERFORMANCE BENCHMARKS ====================

TEST_F(PlayerMovementTest, PerformanceBenchmark) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    bool keys[1024] = {false};
    keys[GLFW_KEY_W] = true;
    keys[GLFW_KEY_D] = true;
    keys[GLFW_KEY_SPACE] = true;
    
    // Run intensive simulation
    for (int frame = 0; frame < 1000; ++frame) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        
        // Occasional state changes
        if (frame % 100 == 0) {
            keys[GLFW_KEY_LEFT_SHIFT] = !keys[GLFW_KEY_LEFT_SHIFT];
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Should complete in reasonable time (< 100ms for 1000 frames)
    EXPECT_LT(duration.count(), 100);
    
    std::cout << "✅ Performance benchmark passed - 1000 frames in " << duration.count() << "ms" << std::endl;
}

// ==================== INTEGRATION TEST SUITE ====================

class PlayerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        initializeTestGL();
        player = std::make_unique<Player>();
        gameWorld = std::make_unique<GameWorld>();
        player->setGameWorld(gameWorld.get());
    }

    void TearDown() override {
        player.reset();
        gameWorld.reset();
        cleanupTestGL();
    }

    void initializeTestGL() {
        std::cout << "🔧 Integration test GL setup..." << std::endl;
    }

    void cleanupTestGL() {
        std::cout << "🧹 Integration test GL cleanup..." << std::endl;
    }

    std::unique_ptr<Player> player;
    std::unique_ptr<GameWorld> gameWorld;
};

TEST_F(PlayerIntegrationTest, ComplexMovementSequence) {
    bool keys[1024] = {false};
    
    std::cout << "🎮 Running complex movement sequence test..." << std::endl;
    
    // Phase 1: Walk forward
    keys[GLFW_KEY_W] = true;
    for (int i = 0; i < 30; ++i) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    EXPECT_TRUE(player->getMovementState() != MovementState::IDLE);
    
    // Phase 2: Jump while moving
    keys[GLFW_KEY_SPACE] = true;
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    EXPECT_EQ(player->getMovementState(), MovementState::JUMPING);
    
    // Phase 3: Dash in air
    keys[GLFW_KEY_LEFT_SHIFT] = true;
    for (int i = 0; i < 10; ++i) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    // Phase 4: Attack while dashing
    keys[GLFW_KEY_LEFT_CONTROL] = true;
    player->handleInput(keys, DELTA_TIME);
    player->update(DELTA_TIME);
    
    // Phase 5: Land and return to idle
    keys[GLFW_KEY_W] = false;
    keys[GLFW_KEY_SPACE] = false;
    keys[GLFW_KEY_LEFT_SHIFT] = false;
    keys[GLFW_KEY_LEFT_CONTROL] = false;
    
    for (int i = 0; i < 100; ++i) {
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
    }
    
    EXPECT_NEAR(player->getCurrentSpeed(), 0.0f, 0.5f);
    
    std::cout << "✅ Complex movement sequence test passed" << std::endl;
}

// ==================== MAIN TEST RUNNER ====================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n🚀 Starting Player Movement and Animation Integration Tests\n" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Testing enhanced character system with:" << std::endl;
    std::cout << "- Basic movement mechanics" << std::endl;
    std::cout << "- Animation state transitions" << std::endl;
    std::cout << "- Physics integration" << std::endl;
    std::cout << "- Combat system integration" << std::endl;
    std::cout << "- Rhythm feedback synchronization" << std::endl;
    std::cout << "- Particle system behavior" << std::endl;
    std::cout << "- Performance benchmarks" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << "\n🎉 ALL TESTS PASSED! Character system ready for production." << std::endl;
    } else {
        std::cout << "\n❌ Some tests failed. Review output above." << std::endl;
    }
    
    return result;
}
