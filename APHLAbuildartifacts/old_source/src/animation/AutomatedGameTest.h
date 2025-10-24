#pragma once

#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <queue>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

// Test action types
enum class TestActionType {
    KEY_PRESS,
    KEY_RELEASE,
    KEY_HOLD,
    WAIT,
    CHECK_POSITION,
    CHECK_STATE,
    CHECK_HEALTH,
    MOVE_TO_POSITION,
    CONTROLLER_BUTTON,
    CONTROLLER_STICK,
    LOG_MESSAGE,
    SCREENSHOT
};

// Test action structure
struct TestAction {
    TestActionType type;
    float delay;  // Time to wait before executing this action
    int key;      // For keyboard actions
    int button;   // For controller actions
    float duration; // For hold actions
    glm::vec3 position; // For position checks/moves
    glm::vec2 stickInput; // For controller stick input
    std::string message; // For logging
    std::function<bool()> condition; // For state checks
    
    TestAction(TestActionType t, float d = 0.0f) : type(t), delay(d), key(0), button(0), duration(0.0f) {}
};

// Test scenario structure
struct TestScenario {
    std::string name;
    std::string description;
    std::vector<TestAction> actions;
    std::function<void()> setup;
    std::function<bool()> validate;
    bool expectSuccess;
    
    TestScenario(const std::string& n, const std::string& desc) 
        : name(n), description(desc), expectSuccess(true) {}
};

// Automated test runner class
class AutomatedGameTest {
private:
    std::vector<TestScenario> scenarios;
    std::queue<TestAction> actionQueue;
    int currentScenarioIndex;
    float actionTimer;
    bool isRunning;
    bool isPaused;
    float testStartTime;
    
    // Test results
    struct TestResult {
        std::string scenarioName;
        bool passed;
        float duration;
        std::string errorMessage;
    };
    std::vector<TestResult> results;
    
    // Reference to game state (will be set externally)
    struct GameState* gameState;
    
public:
    AutomatedGameTest();
    ~AutomatedGameTest();
    
    // Initialize with game state reference
    void initialize(void* gameStatePtr);
    
    // Add test scenarios
    void addScenario(const TestScenario& scenario);
    
    // Create predefined test scenarios
    void createBasicMovementTest();
    void createJumpMechanicsTest();
    void createCombatComboTest();
    void createWallRunningTest();
    void createDeathRecoveryTest();
    void createCameraLockOnTest();
    void createWeaponSwitchingTest();
    void createStressTest();
    
    // Test execution
    void startTesting();
    void stopTesting();
    void pauseTesting();
    void resumeTesting();
    void update(float deltaTime);
    
    // Action execution
    void executeAction(const TestAction& action);
    void simulateKeyPress(int key);
    void simulateKeyRelease(int key);
    void simulateControllerButton(int button, bool pressed);
    void simulateControllerStick(float x, float y, bool leftStick);
    
    // Test validation
    bool checkPlayerPosition(const glm::vec3& expectedPos, float tolerance = 1.0f);
    bool checkPlayerHealth(float expectedHealth, float tolerance = 0.1f);
    bool checkPlayerState(const std::string& stateName);
    bool checkCameraValid();
    
    // Results and reporting
    void generateReport();
    void saveReportToFile(const std::string& filename);
    std::string getLastError() const;
    int getPassedTests() const;
    int getTotalTests() const;
    
    // Utility functions
    static TestAction keyPress(int key, float delay = 0.0f);
    static TestAction keyRelease(int key, float delay = 0.0f);
    static TestAction keyHold(int key, float duration, float delay = 0.0f);
    static TestAction wait(float duration);
    static TestAction checkPosition(const glm::vec3& pos, float delay = 0.0f);
    static TestAction logMessage(const std::string& msg, float delay = 0.0f);
    static TestAction controllerButton(int button, bool pressed, float delay = 0.0f);
    static TestAction controllerStick(float x, float y, bool leftStick, float delay = 0.0f);
};

// Helper class for recording and replaying game sessions
class GameSessionRecorder {
private:
    struct InputEvent {
        float timestamp;
        TestActionType type;
        int key;
        int button;
        glm::vec2 stickInput;
        bool pressed;
    };
    
    std::vector<InputEvent> recordedEvents;
    bool isRecording;
    bool isReplaying;
    float recordStartTime;
    float replayStartTime;
    size_t replayIndex;
    
public:
    GameSessionRecorder();
    
    void startRecording();
    void stopRecording();
    void startReplay();
    void stopReplay();
    void update(float currentTime);
    
    void recordKeyEvent(int key, bool pressed);
    void recordControllerButton(int button, bool pressed);
    void recordControllerStick(float x, float y, bool leftStick);
    
    void saveSession(const std::string& filename);
    void loadSession(const std::string& filename);
    
    bool getIsRecording() const { return isRecording; }
    bool getIsReplaying() const { return isReplaying; }
};
