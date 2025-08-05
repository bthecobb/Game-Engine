#include "AutomatedGameTest.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

// Simple implementation that compiles without external dependencies
// The actual game integration will be done in RhythmArenaDemo.cpp

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
    
    addScenario(test);
}

// Create wall running test
void AutomatedGameTest::createWallRunningTest() {
    TestScenario test("Wall Running", "Tests wall running activation and wall jump");
    
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
    
    addScenario(test);
}

// Create death recovery test
void AutomatedGameTest::createDeathRecoveryTest() {
    TestScenario test("Death Recovery", "Tests player death and respawn system");
    
    test.actions.push_back(logMessage("Setting up death scenario"));
    test.actions.push_back(wait(0.5f));
    
    // Simulate taking damage (would need enemy nearby)
    test.actions.push_back(logMessage("Waiting for death state"));
    test.actions.push_back(wait(4.0f)); // Wait for death timer
    
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

// Stub implementations - will be linked with actual game
void AutomatedGameTest::simulateKeyPress(int key) {
    // Will be implemented in RhythmArenaDemo.cpp
}

void AutomatedGameTest::simulateKeyRelease(int key) {
    // Will be implemented in RhythmArenaDemo.cpp
}

void AutomatedGameTest::simulateControllerButton(int button, bool pressed) {
    // Will be implemented in RhythmArenaDemo.cpp
}

void AutomatedGameTest::simulateControllerStick(float x, float y, bool leftStick) {
    // Will be implemented in RhythmArenaDemo.cpp
}

bool AutomatedGameTest::checkCameraValid() {
    return true; // Will be implemented in RhythmArenaDemo.cpp
}

void AutomatedGameTest::stopTesting() {
    isRunning = false;
    std::cout << "[TEST] Testing completed!" << std::endl;
}

void AutomatedGameTest::pauseTesting() {
    isPaused = true;
}

void AutomatedGameTest::resumeTesting() {
    isPaused = false;
}

bool AutomatedGameTest::checkPlayerPosition(const glm::vec3& expectedPos, float tolerance) {
    return true; // Will be implemented in RhythmArenaDemo.cpp
}

bool AutomatedGameTest::checkPlayerHealth(float expectedHealth, float tolerance) {
    return true; // Will be implemented in RhythmArenaDemo.cpp
}

bool AutomatedGameTest::checkPlayerState(const std::string& stateName) {
    return true; // Will be implemented in RhythmArenaDemo.cpp
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

void AutomatedGameTest::saveReportToFile(const std::string& filename) {
    // TODO: Implement file saving
}

std::string AutomatedGameTest::getLastError() const {
    return "";
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

TestAction AutomatedGameTest::keyHold(int key, float duration, float delay) {
    TestAction action(TestActionType::KEY_HOLD, delay);
    action.key = key;
    action.duration = duration;
    return action;
}

TestAction AutomatedGameTest::checkPosition(const glm::vec3& pos, float delay) {
    TestAction action(TestActionType::CHECK_POSITION, delay);
    action.position = pos;
    return action;
}

TestAction AutomatedGameTest::controllerButton(int button, bool pressed, float delay) {
    TestAction action(TestActionType::CONTROLLER_BUTTON, delay);
    action.button = button;
    action.duration = pressed ? 1.0f : 0.0f;
    return action;
}

TestAction AutomatedGameTest::controllerStick(float x, float y, bool leftStick, float delay) {
    TestAction action(TestActionType::CONTROLLER_STICK, delay);
    action.stickInput = glm::vec2(x, y);
    action.duration = leftStick ? 1.0f : 0.0f;
    return action;
}

// GameSessionRecorder implementation
GameSessionRecorder::GameSessionRecorder() 
    : isRecording(false), isReplaying(false), recordStartTime(0.0f), 
      replayStartTime(0.0f), replayIndex(0) {
}

void GameSessionRecorder::startRecording() {
    isRecording = true;
    isReplaying = false;
    recordedEvents.clear();
    recordStartTime = glfwGetTime();
}

void GameSessionRecorder::stopRecording() {
    isRecording = false;
}

void GameSessionRecorder::startReplay() {
    if (recordedEvents.empty()) return;
    
    isReplaying = true;
    isRecording = false;
    replayIndex = 0;
    replayStartTime = glfwGetTime();
}

void GameSessionRecorder::stopReplay() {
    isReplaying = false;
}

void GameSessionRecorder::update(float currentTime) {
    // TODO: Implement replay logic
}

void GameSessionRecorder::recordKeyEvent(int key, bool pressed) {
    if (!isRecording) return;
    
    InputEvent event;
    event.timestamp = glfwGetTime() - recordStartTime;
    event.type = pressed ? TestActionType::KEY_PRESS : TestActionType::KEY_RELEASE;
    event.key = key;
    event.pressed = pressed;
    recordedEvents.push_back(event);
}

void GameSessionRecorder::recordControllerButton(int button, bool pressed) {
    if (!isRecording) return;
    
    InputEvent event;
    event.timestamp = glfwGetTime() - recordStartTime;
    event.type = TestActionType::CONTROLLER_BUTTON;
    event.button = button;
    event.pressed = pressed;
    recordedEvents.push_back(event);
}

void GameSessionRecorder::recordControllerStick(float x, float y, bool leftStick) {
    if (!isRecording) return;
    
    InputEvent event;
    event.timestamp = glfwGetTime() - recordStartTime;
    event.type = TestActionType::CONTROLLER_STICK;
    event.stickInput = glm::vec2(x, y);
    event.pressed = leftStick;
    recordedEvents.push_back(event);
}

void GameSessionRecorder::saveSession(const std::string& filename) {
    // TODO: Implement session saving
}

void GameSessionRecorder::loadSession(const std::string& filename) {
    // TODO: Implement session loading
}
