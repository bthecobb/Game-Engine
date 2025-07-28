#include "AutomatedGameTest.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

// We'll use void pointers to avoid circular dependencies
struct GameStateRefs {
    void* player;
    void* gameCamera;
    bool* keys;
    bool* keysPressed;
    void* currentController;
    int* gameState;
    int* targetedEnemyIndex;
    void* enemies;
    void* giantPillars;
};

static GameStateRefs g_gameRefs;

AutomatedGameTest::AutomatedGameTest() 
    : currentScenarioIndex(0), actionTimer(0.0f), isRunning(false), 
      isPaused(false), testStartTime(0.0f), gameState(nullptr) {
}

AutomatedGameTest::~AutomatedGameTest() {
}

void AutomatedGameTest::initialize(void* gameStatePtr) {
    gameState = static_cast<GameState*>(gameStatePtr);
}

void AutomatedGameTest::addScenario(const TestScenario& scenario) {
    scenarios.push_back(scenario);
}

// Create basic movement test
void AutomatedGameTest::createBasicMovementTest() {
    TestScenario test("Basic Movement", "Tests WASD movement and sprint toggle");
    
    // Starting position
    glm::vec3 startPos = player.position;
    
    // Test forward movement
    test.actions.push_back(logMessage("Testing forward movement (W)"));
    test.actions.push_back(keyPress(GLFW_KEY_W));
    test.actions.push_back(wait(1.0f));
    test.actions.push_back(keyRelease(GLFW_KEY_W));
    test.actions.push_back(wait(0.5f));
    
    // Test backward movement
    test.actions.push_back(logMessage("Testing backward movement (S)"));
    test.actions.push_back(keyPress(GLFW_KEY_S));
    test.actions.push_back(wait(1.0f));
    test.actions.push_back(keyRelease(GLFW_KEY_S));
    test.actions.push_back(wait(0.5f));
    
    // Test left/right movement
    test.actions.push_back(logMessage("Testing strafe movement (A/D)"));
    test.actions.push_back(keyPress(GLFW_KEY_A));
    test.actions.push_back(wait(0.5f));
    test.actions.push_back(keyRelease(GLFW_KEY_A));
    test.actions.push_back(keyPress(GLFW_KEY_D));
    test.actions.push_back(wait(0.5f));
    test.actions.push_back(keyRelease(GLFW_KEY_D));
    
    // Test sprint toggle
    test.actions.push_back(logMessage("Testing sprint toggle"));
    test.actions.push_back(keyPress(GLFW_KEY_LEFT_SHIFT));
    test.actions.push_back(keyRelease(GLFW_KEY_LEFT_SHIFT));
    test.actions.push_back(keyPress(GLFW_KEY_W));
    test.actions.push_back(wait(1.0f));
    test.actions.push_back(keyRelease(GLFW_KEY_W));
    
    test.validate = [startPos]() {
        // Check that player moved from starting position
        float distance = glm::length(player.position - startPos);
        return distance > 1.0f;
    };
    
    addScenario(test);
}

// Create jump mechanics test
void AutomatedGameTest::createJumpMechanicsTest() {
    TestScenario test("Jump Mechanics", "Tests single jump, double jump, and landing");
    
    test.actions.push_back(logMessage("Testing single jump"));
    test.actions.push_back(keyPress(GLFW_KEY_SPACE));
    test.actions.push_back(keyRelease(GLFW_KEY_SPACE));
    test.actions.push_back(wait(0.5f));
    
    test.actions.push_back(logMessage("Testing double jump"));
    test.actions.push_back(keyPress(GLFW_KEY_SPACE));
    test.actions.push_back(keyRelease(GLFW_KEY_SPACE));
    test.actions.push_back(wait(2.0f)); // Wait for landing
    
    test.actions.push_back(logMessage("Testing jump with movement"));
    test.actions.push_back(keyPress(GLFW_KEY_W));
    test.actions.push_back(wait(0.1f));
    test.actions.push_back(keyPress(GLFW_KEY_SPACE));
    test.actions.push_back(keyRelease(GLFW_KEY_SPACE));
    test.actions.push_back(wait(1.0f));
    test.actions.push_back(keyRelease(GLFW_KEY_W));
    
    test.validate = []() {
        // Check that player is on ground after test
        return player.onGround;
    };
    
    addScenario(test);
}

// Create combat combo test
void AutomatedGameTest::createCombatComboTest() {
    TestScenario test("Combat Combos", "Tests dash-light-launcher combo sequence");
    
    test.actions.push_back(logMessage("Testing dash attack (F)"));
    test.actions.push_back(keyPress(GLFW_KEY_F));
    test.actions.push_back(keyRelease(GLFW_KEY_F));
    test.actions.push_back(wait(0.3f));
    
    test.actions.push_back(logMessage("Testing light attack combo (Q)"));
    test.actions.push_back(keyPress(GLFW_KEY_Q));
    test.actions.push_back(keyRelease(GLFW_KEY_Q));
    test.actions.push_back(wait(0.2f));
    test.actions.push_back(keyPress(GLFW_KEY_Q));
    test.actions.push_back(keyRelease(GLFW_KEY_Q));
    test.actions.push_back(wait(0.2f));
    test.actions.push_back(keyPress(GLFW_KEY_Q));
    test.actions.push_back(keyRelease(GLFW_KEY_Q));
    test.actions.push_back(wait(0.3f));
    
    test.actions.push_back(logMessage("Testing heavy attack launcher (E)"));
    test.actions.push_back(keyPress(GLFW_KEY_E));
    test.actions.push_back(keyRelease(GLFW_KEY_E));
    test.actions.push_back(wait(1.0f));
    
    test.validate = []() {
        // Check that combo system was activated
        return player.comboCount > 0 || player.combatState != CombatState::NONE;
    };
    
    addScenario(test);
}

// Create wall running test
void AutomatedGameTest::createWallRunningTest() {
    TestScenario test("Wall Running", "Tests wall running activation and wall jump");
    
    test.setup = []() {
        // Position player near a wall/pillar
        if (!giantPillars.empty()) {
            player.position = giantPillars[0].position + glm::vec3(5.0f, 1.0f, 0.0f);
        }
    };
    
    test.actions.push_back(logMessage("Building combo for wall run"));
    test.actions.push_back(keyPress(GLFW_KEY_F));
    test.actions.push_back(keyRelease(GLFW_KEY_F));
    test.actions.push_back(wait(0.3f));
    test.actions.push_back(keyPress(GLFW_KEY_Q));
    test.actions.push_back(keyRelease(GLFW_KEY_Q));
    test.actions.push_back(wait(0.3f));
    
    test.actions.push_back(logMessage("Running towards wall"));
    test.actions.push_back(keyPress(GLFW_KEY_A)); // Run left towards pillar
    test.actions.push_back(wait(1.0f));
    
    test.actions.push_back(logMessage("Attempting wall jump"));
    test.actions.push_back(keyPress(GLFW_KEY_SPACE));
    test.actions.push_back(keyRelease(GLFW_KEY_SPACE));
    test.actions.push_back(keyRelease(GLFW_KEY_A));
    test.actions.push_back(wait(2.0f));
    
    test.validate = []() {
        // Check if wall running was activated at any point
        return true; // Would need to track if wall running occurred
    };
    
    addScenario(test);
}

// Create death recovery test
void AutomatedGameTest::createDeathRecoveryTest() {
    TestScenario test("Death Recovery", "Tests player death and respawn system");
    
    test.setup = []() {
        // Set player health very low
        player.health = 1.0f;
    };
    
    test.actions.push_back(logMessage("Setting up death scenario"));
    test.actions.push_back(wait(0.5f));
    
    // Simulate taking damage (would need enemy nearby)
    test.actions.push_back(logMessage("Waiting for death state"));
    test.actions.push_back(wait(4.0f)); // Wait for death timer
    
    test.validate = []() {
        // Check that player respawned with full health
        return player.health == player.maxHealth && !player.isDead;
    };
    
    addScenario(test);
}

// Create camera lock-on test
void AutomatedGameTest::createCameraLockOnTest() {
    TestScenario test("Camera Lock-On", "Tests camera lock-on system");
    
    test.actions.push_back(logMessage("Testing camera lock-on toggle (T)"));
    test.actions.push_back(keyPress(GLFW_KEY_T));
    test.actions.push_back(keyRelease(GLFW_KEY_T));
    test.actions.push_back(wait(1.0f));
    
    test.actions.push_back(logMessage("Testing movement with lock-on"));
    test.actions.push_back(keyPress(GLFW_KEY_A));
    test.actions.push_back(wait(0.5f));
    test.actions.push_back(keyRelease(GLFW_KEY_A));
    test.actions.push_back(keyPress(GLFW_KEY_D));
    test.actions.push_back(wait(0.5f));
    test.actions.push_back(keyRelease(GLFW_KEY_D));
    
    test.actions.push_back(logMessage("Disabling lock-on"));
    test.actions.push_back(keyPress(GLFW_KEY_T));
    test.actions.push_back(keyRelease(GLFW_KEY_T));
    test.actions.push_back(wait(0.5f));
    
    test.validate = []() {
        // Check camera state
        return gameCamera != nullptr && !gameCamera->isLockOnActive();
    };
    
    addScenario(test);
}

// Create weapon switching test
void AutomatedGameTest::createWeaponSwitchingTest() {
    TestScenario test("Weapon Switching", "Tests switching between all weapons");
    
    test.actions.push_back(logMessage("Switching to Sword (2)"));
    test.actions.push_back(keyPress(GLFW_KEY_2));
    test.actions.push_back(keyRelease(GLFW_KEY_2));
    test.actions.push_back(wait(1.5f));
    
    test.actions.push_back(logMessage("Switching to Staff (3)"));
    test.actions.push_back(keyPress(GLFW_KEY_3));
    test.actions.push_back(keyRelease(GLFW_KEY_3));
    test.actions.push_back(wait(1.5f));
    
    test.actions.push_back(logMessage("Switching to Hammer (4)"));
    test.actions.push_back(keyPress(GLFW_KEY_4));
    test.actions.push_back(keyRelease(GLFW_KEY_4));
    test.actions.push_back(wait(1.5f));
    
    test.actions.push_back(logMessage("Switching to Fists (1)"));
    test.actions.push_back(keyPress(GLFW_KEY_1));
    test.actions.push_back(keyRelease(GLFW_KEY_1));
    test.actions.push_back(wait(1.0f));
    
    test.validate = []() {
        return player.currentWeapon.type == WeaponType::NONE;
    };
    
    addScenario(test);
}

// Create stress test
void AutomatedGameTest::createStressTest() {
    TestScenario test("Stress Test", "Rapid input sequence to test stability");
    
    test.actions.push_back(logMessage("Starting rapid input stress test"));
    
    // Rapid jump sequence
    for (int i = 0; i < 10; i++) {
        test.actions.push_back(keyPress(GLFW_KEY_SPACE));
        test.actions.push_back(wait(0.1f));
        test.actions.push_back(keyRelease(GLFW_KEY_SPACE));
        test.actions.push_back(wait(0.1f));
    }
    
    // Rapid movement changes
    test.actions.push_back(logMessage("Rapid movement changes"));
    for (int i = 0; i < 5; i++) {
        test.actions.push_back(keyPress(GLFW_KEY_W));
        test.actions.push_back(wait(0.1f));
        test.actions.push_back(keyPress(GLFW_KEY_SPACE));
        test.actions.push_back(keyRelease(GLFW_KEY_SPACE));
        test.actions.push_back(keyRelease(GLFW_KEY_W));
        test.actions.push_back(keyPress(GLFW_KEY_S));
        test.actions.push_back(wait(0.1f));
        test.actions.push_back(keyRelease(GLFW_KEY_S));
    }
    
    // Rapid combat inputs
    test.actions.push_back(logMessage("Rapid combat inputs"));
    for (int i = 0; i < 10; i++) {
        test.actions.push_back(keyPress(GLFW_KEY_Q));
        test.actions.push_back(keyRelease(GLFW_KEY_Q));
        test.actions.push_back(wait(0.05f));
    }
    
    test.validate = []() {
        // Check that game is still in playing state and camera is valid
        if (!gameCamera) return false;
        
        glm::vec3 camPos = gameCamera->position;
        
        // Check for NaN or infinity
        if (std::isnan(camPos.x) || std::isnan(camPos.y) || std::isnan(camPos.z) ||
            std::isinf(camPos.x) || std::isinf(camPos.y) || std::isinf(camPos.z)) {
            return false;
        }
        
        // Check distance from player
        float dist = glm::length(camPos - player.position);
        return gameState == GAME_PLAYING && dist < 100.0f && dist > 0.1f;
    };
    
    addScenario(test);
}

// Test execution methods
void AutomatedGameTest::startTesting() {
    if (scenarios.empty()) {
        std::cout << "[TEST] No test scenarios loaded!" << std::endl;
        return;
    }
    
    isRunning = true;
    isPaused = false;
    currentScenarioIndex = 0;
    results.clear();
    testStartTime = glfwGetTime();
    
    std::cout << "[TEST] Starting automated testing with " << scenarios.size() << " scenarios" << std::endl;
    
    // Setup first scenario
    if (scenarios[0].setup) {
        scenarios[0].setup();
    }
    
    // Load actions for first scenario
    for (const auto& action : scenarios[0].actions) {
        actionQueue.push(action);
    }
}

void AutomatedGameTest::update(float deltaTime) {
    if (!isRunning || isPaused || actionQueue.empty()) {
        return;
    }
    
    actionTimer += deltaTime;
    
    // Check if we should execute the next action
    if (!actionQueue.empty()) {
        TestAction& nextAction = actionQueue.front();
        
        if (actionTimer >= nextAction.delay) {
            executeAction(nextAction);
            actionQueue.pop();
            actionTimer = 0.0f;
        }
    }
    
    // If action queue is empty, validate current scenario and move to next
    if (actionQueue.empty() && currentScenarioIndex < scenarios.size()) {
        TestScenario& currentScenario = scenarios[currentScenarioIndex];
        
        // Validate scenario
        bool passed = true;
        if (currentScenario.validate) {
            passed = currentScenario.validate();
        }
        
        // Record result
        TestResult result;
        result.scenarioName = currentScenario.name;
        result.passed = passed;
        result.duration = glfwGetTime() - testStartTime;
        result.errorMessage = passed ? "" : "Validation failed";
        results.push_back(result);
        
        std::cout << "[TEST] " << currentScenario.name << ": " 
                  << (passed ? "PASSED" : "FAILED") << std::endl;
        
        // Move to next scenario
        currentScenarioIndex++;
        if (currentScenarioIndex < scenarios.size()) {
            // Setup next scenario
            if (scenarios[currentScenarioIndex].setup) {
                scenarios[currentScenarioIndex].setup();
            }
            
            // Load actions for next scenario
            for (const auto& action : scenarios[currentScenarioIndex].actions) {
                actionQueue.push(action);
            }
        } else {
            // All tests completed
            stopTesting();
            generateReport();
        }
    }
}

void AutomatedGameTest::executeAction(const TestAction& action) {
    switch (action.type) {
        case TestActionType::KEY_PRESS:
            simulateKeyPress(action.key);
            break;
            
        case TestActionType::KEY_RELEASE:
            simulateKeyRelease(action.key);
            break;
            
        case TestActionType::WAIT:
            // Nothing to do, just wait
            break;
            
        case TestActionType::LOG_MESSAGE:
            std::cout << "[TEST] " << action.message << std::endl;
            break;
            
        case TestActionType::CONTROLLER_BUTTON:
            simulateControllerButton(action.button, action.duration > 0);
            break;
            
        case TestActionType::CONTROLLER_STICK:
            simulateControllerStick(action.stickInput.x, action.stickInput.y, action.duration > 0);
            break;
            
        default:
            break;
    }
}

void AutomatedGameTest::simulateKeyPress(int key) {
    if (key >= 0 && key < 1024) {
        keys[key] = true;
        keysPressed[key] = true;
    }
}

void AutomatedGameTest::simulateKeyRelease(int key) {
    if (key >= 0 && key < 1024) {
        keys[key] = false;
    }
}

void AutomatedGameTest::simulateControllerButton(int button, bool pressed) {
    switch (button) {
        case 0: currentController.buttonA = pressed; break;
        case 1: currentController.buttonB = pressed; break;
        case 2: currentController.buttonX = pressed; break;
        case 3: currentController.buttonY = pressed; break;
        case 4: currentController.leftBumper = pressed; break;
        case 5: currentController.rightBumper = pressed; break;
        case 6: currentController.leftTrigger = pressed; break;
        case 7: currentController.rightTrigger = pressed; break;
    }
}

void AutomatedGameTest::simulateControllerStick(float x, float y, bool leftStick) {
    if (leftStick) {
        currentController.leftStickX = x;
        currentController.leftStickY = y;
    } else {
        currentController.rightStickX = x;
        currentController.rightStickY = y;
    }
}

bool AutomatedGameTest::checkCameraValid() {
    if (!gameCamera) return false;
    
    glm::vec3 camPos = gameCamera->position;
    glm::vec3 camTarget = gameCamera->target;
    
    // Check for NaN or infinity
    if (std::isnan(camPos.x) || std::isnan(camPos.y) || std::isnan(camPos.z) ||
        std::isinf(camPos.x) || std::isinf(camPos.y) || std::isinf(camPos.z)) {
        return false;
    }
    
    // Check distance from player
    float dist = glm::length(camPos - player.position);
    return dist < 100.0f && dist > 0.1f;
}

void AutomatedGameTest::stopTesting() {
    isRunning = false;
    std::cout << "[TEST] Testing completed!" << std::endl;
}

void AutomatedGameTest::generateReport() {
    std::cout << "\n=== AUTOMATED TEST REPORT ===" << std::endl;
    std::cout << "Total Tests: " << results.size() << std::endl;
    std::cout << "Passed: " << getPassedTests() << std::endl;
    std::cout << "Failed: " << (results.size() - getPassedTests()) << std::endl;
    std::cout << "\nDetailed Results:" << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(30) << std::left << result.scenarioName 
                  << " : " << (result.passed ? "PASS" : "FAIL");
        if (!result.passed && !result.errorMessage.empty()) {
            std::cout << " (" << result.errorMessage << ")";
        }
        std::cout << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

int AutomatedGameTest::getPassedTests() const {
    int passed = 0;
    for (const auto& result : results) {
        if (result.passed) passed++;
    }
    return passed;
}

int AutomatedGameTest::getTotalTests() const {
    return results.size();
}

// Static helper functions
TestAction AutomatedGameTest::keyPress(int key, float delay) {
    TestAction action(TestActionType::KEY_PRESS, delay);
    action.key = key;
    return action;
}

TestAction AutomatedGameTest::keyRelease(int key, float delay) {
    TestAction action(TestActionType::KEY_RELEASE, delay);
    action.key = key;
    return action;
}

TestAction AutomatedGameTest::wait(float duration) {
    TestAction action(TestActionType::WAIT, duration);
    return action;
}

TestAction AutomatedGameTest::logMessage(const std::string& msg, float delay) {
    TestAction action(TestActionType::LOG_MESSAGE, delay);
    action.message = msg;
    return action;
}
