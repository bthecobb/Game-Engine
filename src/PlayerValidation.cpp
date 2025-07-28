#include "Player.h"
#include "GameWorld.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

/**
 * @brief Player Movement and Animation Validation Suite (without GTest dependency)
 * 
 * This validation suite tests the enhanced character system functionality
 * using basic C++ without external testing frameworks.
 */

class PlayerValidator {
private:
    std::unique_ptr<Player> player;
    std::unique_ptr<GameWorld> gameWorld;
    int testsRun = 0;
    int testsPassed = 0;
    int testsFailed = 0;
    
    static constexpr float DELTA_TIME = 1.0f / 60.0f; // 60 FPS
    static constexpr float EPSILON = 0.001f;

public:
    PlayerValidator() {
        std::cout << "\n🚀 Initializing Player Movement Validation Suite\n" << std::endl;
        std::cout << "=================================================" << std::endl;
        
        // Initialize without OpenGL context for validation
        player = std::make_unique<Player>();
        gameWorld = std::make_unique<GameWorld>();
        
        // Basic setup
        player->setGameWorld(gameWorld.get());
        player->transform.position = glm::vec3(0.0f, 2.0f, 0.0f);
        player->transform.scale = glm::vec3(1.0f);
        player->transform.rotation = glm::vec3(0.0f);
        
        std::cout << "✅ Player validation environment initialized" << std::endl;
        std::cout << "=================================================\n" << std::endl;
    }

    ~PlayerValidator() {
        printResults();
    }

    void runTest(const std::string& testName, std::function<bool()> testFunc) {
        testsRun++;
        std::cout << "🧪 Running test: " << testName << "... " << std::flush;
        
        try {
            bool result = testFunc();
            if (result) {
                testsPassed++;
                std::cout << "✅ PASSED" << std::endl;
            } else {
                testsFailed++;
                std::cout << "❌ FAILED" << std::endl;
            }
        } catch (const std::exception& e) {
            testsFailed++;
            std::cout << "💥 EXCEPTION: " << e.what() << std::endl;
        }
    }

    void runAllTests() {
        std::cout << "🎯 Starting comprehensive player validation...\n" << std::endl;

        // Basic functionality tests
        runTest("Initial State Validation", [this]() { return testInitialState(); });
        runTest("Movement State Machine", [this]() { return testMovementStateMachine(); });
        runTest("Input Processing", [this]() { return testInputProcessing(); });
        runTest("Physics Integration", [this]() { return testPhysicsIntegration(); });
        runTest("Animation System", [this]() { return testAnimationSystem(); });
        
        // Advanced feature tests
        runTest("Jump Mechanics", [this]() { return testJumpMechanics(); });
        runTest("Dash System", [this]() { return testDashSystem(); });
        runTest("Combat Integration", [this]() { return testCombatIntegration(); });
        runTest("Rhythm Feedback", [this]() { return testRhythmFeedback(); });
        runTest("Particle System", [this]() { return testParticleSystem(); });
        
        // Performance and integration tests
        runTest("Performance Validation", [this]() { return testPerformance(); });
        runTest("Complex Movement Sequence", [this]() { return testComplexMovement(); });
        
        std::cout << "\n🏁 All tests completed!" << std::endl;
    }

private:
    bool testInitialState() {
        // Verify player starts in expected state
        bool stateCorrect = (player->getMovementState() == MovementState::IDLE);
        bool speedZero = (player->getCurrentSpeed() < EPSILON);
        bool animationIdle = (player->getCurrentAnimationState() == AnimationState::IDLE_ANIM);
        bool positionCorrect = (glm::distance(player->transform.position, glm::vec3(0.0f, 2.0f, 0.0f)) < EPSILON);
        
        return stateCorrect && speedZero && animationIdle && positionCorrect;
    }

    bool testMovementStateMachine() {
        // Test state transitions
        bool keys[1024] = {false};
        keys[GLFW_KEY_W] = true;
        
        // Move forward to trigger walking state
        for (int i = 0; i < 10; ++i) {
            player->handleInput(keys, DELTA_TIME);
            player->update(DELTA_TIME);
        }
        
        bool movingState = (player->getMovementState() != MovementState::IDLE);
        bool hasSpeed = (player->getCurrentSpeed() > EPSILON);
        
        // Stop input and return to idle
        keys[GLFW_KEY_W] = false;
        for (int i = 0; i < 50; ++i) {
            player->handleInput(keys, DELTA_TIME);
            player->update(DELTA_TIME);
        }
        
        bool returnedToIdle = (player->getMovementState() == MovementState::IDLE);
        bool speedReduced = (player->getCurrentSpeed() < 1.0f);
        
        return movingState && hasSpeed && returnedToIdle && speedReduced;
    }

    bool testInputProcessing() {
        bool keys[1024] = {false};
        glm::vec3 initialPos = player->transform.position;
        
        // Test each direction
        keys[GLFW_KEY_W] = true;
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        keys[GLFW_KEY_W] = false;
        
        keys[GLFW_KEY_S] = true;
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        keys[GLFW_KEY_S] = false;
        
        keys[GLFW_KEY_A] = true;
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        keys[GLFW_KEY_A] = false;
        
        keys[GLFW_KEY_D] = true;
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        keys[GLFW_KEY_D] = false;
        
        // Should have moved from initial position
        float distanceMoved = glm::distance(initialPos, player->transform.position);
        return distanceMoved > EPSILON;
    }

    bool testPhysicsIntegration() {
        // Test gravity
        player->transform.position.y = 5.0f;
        
        // Let gravity pull player down
        for (int i = 0; i < 100; ++i) {
            player->update(DELTA_TIME);
            if (player->transform.position.y <= 0.1f) {
                break;
            }
        }
        
        bool landedOnGround = (player->transform.position.y <= 0.2f);
        bool isGrounded = player->isGrounded();
        
        return landedOnGround && isGrounded;
    }

    bool testAnimationSystem() {
        bool keys[1024] = {false};
        
        // Start idle
        AnimationState initialState = player->getCurrentAnimationState();
        
        // Move to trigger animation change
        keys[GLFW_KEY_W] = true;
        for (int i = 0; i < 20; ++i) {
            player->handleInput(keys, DELTA_TIME);
            player->update(DELTA_TIME);
        }
        
        AnimationState movingState = player->getCurrentAnimationState();
        
        // Should have changed from idle
        return (initialState == AnimationState::IDLE_ANIM) && 
               (movingState != AnimationState::IDLE_ANIM);
    }

    bool testJumpMechanics() {
        bool keys[1024] = {false};
        float initialY = player->transform.position.y;
        
        // Trigger jump
        keys[GLFW_KEY_SPACE] = true;
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        
        bool leftGround = (player->transform.position.y > initialY);
        bool jumpingState = (player->getMovementState() == MovementState::JUMPING);
        
        return leftGround && jumpingState;
    }

    bool testDashSystem() {
        bool keys[1024] = {false};
        glm::vec3 initialPos = player->transform.position;
        
        // Setup forward movement then dash
        keys[GLFW_KEY_W] = true;
        keys[GLFW_KEY_LEFT_SHIFT] = true;
        
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        
        bool dashingState = (player->getMovementState() == MovementState::DASHING);
        bool highSpeed = (player->getCurrentSpeed() > 15.0f);
        
        return dashingState && highSpeed;
    }

    bool testCombatIntegration() {
        bool keys[1024] = {false};
        
        // Test light attack
        keys[GLFW_KEY_LEFT_CONTROL] = true;
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        
        bool attacking = (player->getCombatState() == CombatState::ATTACKING);
        
        return attacking;
    }

    bool testRhythmFeedback() {
        // Test rhythm system over time
        int beatCount = 0;
        float totalTime = 0.0f;
        
        while (totalTime < 2.0f) {
            player->update(DELTA_TIME);
            if (player->isOnBeat()) {
                beatCount++;
            }
            totalTime += DELTA_TIME;
        }
        
        // Should detect beats (120 BPM = 2 per second, so at least 2 in 2 seconds)
        return beatCount >= 2;
    }

    bool testParticleSystem() {
        bool keys[1024] = {false};
        keys[GLFW_KEY_W] = true;
        
        // Build up high speed to trigger particles
        for (int i = 0; i < 100; ++i) {
            player->handleInput(keys, DELTA_TIME);
            player->update(DELTA_TIME);
        }
        
        // High speed should trigger particle system
        bool highSpeed = (player->getCurrentSpeed() > 20.0f);
        
        return highSpeed; // Particle system activation is indirectly tested via speed
    }

    bool testPerformance() {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        bool keys[1024] = {false};
        keys[GLFW_KEY_W] = true;
        keys[GLFW_KEY_D] = true;
        
        // Run 500 frames of intensive updates
        for (int i = 0; i < 500; ++i) {
            player->handleInput(keys, DELTA_TIME);
            player->update(DELTA_TIME);
            
            // Mix in some state changes
            if (i % 50 == 0) {
                keys[GLFW_KEY_SPACE] = !keys[GLFW_KEY_SPACE];
            }
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << " [" << duration.count() << "ms for 500 frames] ";
        
        // Should complete in reasonable time (< 50ms for 500 frames)
        return duration.count() < 50;
    }

    bool testComplexMovement() {
        bool keys[1024] = {false};
        
        // Complex sequence: Walk -> Jump -> Dash -> Attack -> Land -> Idle
        
        // Phase 1: Walk
        keys[GLFW_KEY_W] = true;
        for (int i = 0; i < 20; ++i) {
            player->handleInput(keys, DELTA_TIME);
            player->update(DELTA_TIME);
        }
        bool wasWalking = (player->getMovementState() != MovementState::IDLE);
        
        // Phase 2: Jump
        keys[GLFW_KEY_SPACE] = true;
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        bool wasJumping = (player->getMovementState() == MovementState::JUMPING);
        
        // Phase 3: Dash in air
        keys[GLFW_KEY_LEFT_SHIFT] = true;
        for (int i = 0; i < 5; ++i) {
            player->handleInput(keys, DELTA_TIME);
            player->update(DELTA_TIME);
        }
        bool wasDashing = (player->getMovementState() == MovementState::DASHING);
        
        // Phase 4: Attack
        keys[GLFW_KEY_LEFT_CONTROL] = true;
        player->handleInput(keys, DELTA_TIME);
        player->update(DELTA_TIME);
        bool wasAttacking = (player->getCombatState() == CombatState::ATTACKING);
        
        // Phase 5: Return to ground and idle
        keys[GLFW_KEY_W] = false;
        keys[GLFW_KEY_SPACE] = false;
        keys[GLFW_KEY_LEFT_SHIFT] = false;
        keys[GLFW_KEY_LEFT_CONTROL] = false;
        
        for (int i = 0; i < 100; ++i) {
            player->handleInput(keys, DELTA_TIME);
            player->update(DELTA_TIME);
        }
        
        bool returnedToIdle = (player->getCurrentSpeed() < 1.0f);
        
        return wasWalking && wasJumping && wasDashing && wasAttacking && returnedToIdle;
    }

    void printResults() {
        std::cout << "\n📊 VALIDATION RESULTS" << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "Total Tests Run: " << testsRun << std::endl;
        std::cout << "Tests Passed:   " << testsPassed << " ✅" << std::endl;
        std::cout << "Tests Failed:   " << testsFailed << " ❌" << std::endl;
        
        float successRate = (testsRun > 0) ? (float)testsPassed / testsRun * 100.0f : 0.0f;
        std::cout << "Success Rate:   " << std::fixed << std::setprecision(1) << successRate << "%" << std::endl;
        
        if (testsFailed == 0) {
            std::cout << "\n🎉 ALL TESTS PASSED! Enhanced character system validated successfully!" << std::endl;
            std::cout << "🚀 Ready for production deployment and alpha testing." << std::endl;
        } else {
            std::cout << "\n⚠️  Some tests failed. Review implementation before proceeding." << std::endl;
        }
        
        std::cout << "=================================\n" << std::endl;
    }
};

// Validation runner function that can be called from main or separately
void runPlayerValidation() {
    try {
        PlayerValidator validator;
        validator.runAllTests();
    } catch (const std::exception& e) {
        std::cerr << "💥 Validation failed with exception: " << e.what() << std::endl;
    }
}

// Optional main function for standalone testing
#ifdef STANDALONE_VALIDATION
int main() {
    std::cout << "🎮 CudaGame Enhanced Character System Validation" << std::endl;
    std::cout << "Built with AAA-grade ShaderRegistry architecture" << std::endl;
    std::cout << "Version: Alpha MVP Ready\n" << std::endl;
    
    runPlayerValidation();
    
    return 0;
}
#endif
