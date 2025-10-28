#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>

// Simple vector3 implementation for demo
struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator*(float scale) const { return Vec3(x * scale, y * scale, z * scale); }
};

struct Vec2 {
    float x, y;
    Vec2(float x = 0, float y = 0) : x(x), y(y) {}
    float length() const { return sqrt(x*x + y*y); }
};

// Simple animation states
enum class SimpleAnimationType {
    IDLE,
    WALK,
    RUN,
    SPRINT,
    JUMP
};

// Simple animation frame
struct SimpleAnimationFrame {
    Vec3 headPos{0, 1.5f, 0};
    Vec3 torsoPos{0, 0.5f, 0};
    Vec3 leftArmPos{-0.6f, 0.8f, 0};
    Vec3 rightArmPos{0.6f, 0.8f, 0};
    Vec3 leftLegPos{-0.2f, -0.8f, 0};
    Vec3 rightLegPos{0.2f, -0.8f, 0};
    
    float energyLevel = 0.5f;
    float animationTime = 0.0f;
};

class SimpleAnimationController {
private:
    SimpleAnimationType m_currentType = SimpleAnimationType::IDLE;
    float m_animationTime = 0.0f;
    float m_speed = 0.0f;
    Vec2 m_direction{0, 0};
    
public:
    void update(float deltaTime, const Vec2& inputDirection, float speed) {
        m_animationTime += deltaTime;
        m_direction = inputDirection;
        m_speed = speed;
        
        // Simple state selection
        if (speed < 0.5f) {
            m_currentType = SimpleAnimationType::IDLE;
        } else if (speed < 5.0f) {
            m_currentType = SimpleAnimationType::WALK;
        } else if (speed < 15.0f) {
            m_currentType = SimpleAnimationType::RUN;
        } else {
            m_currentType = SimpleAnimationType::SPRINT;
        }
    }
    
    SimpleAnimationFrame getCurrentFrame() const {
        SimpleAnimationFrame frame;
        frame.animationTime = m_animationTime;
        
        switch (m_currentType) {
            case SimpleAnimationType::IDLE:
                generateIdleFrame(frame);
                break;
            case SimpleAnimationType::WALK:
                generateWalkFrame(frame);
                break;
            case SimpleAnimationType::RUN:
                generateRunFrame(frame);
                break;
            case SimpleAnimationType::SPRINT:
                generateSprintFrame(frame);
                break;
            case SimpleAnimationType::JUMP:
                generateJumpFrame(frame);
                break;
        }
        
        return frame;
    }
    
    std::string getCurrentAnimationName() const {
        switch (m_currentType) {
            case SimpleAnimationType::IDLE: return "Idle";
            case SimpleAnimationType::WALK: return "Walk";
            case SimpleAnimationType::RUN: return "Run";
            case SimpleAnimationType::SPRINT: return "Sprint";
            case SimpleAnimationType::JUMP: return "Jump";
            default: return "Unknown";
        }
    }
    
private:
    void generateIdleFrame(SimpleAnimationFrame& frame) const {
        // Subtle breathing and sway
        float breathPhase = m_animationTime * 3.0f;
        float swayPhase = m_animationTime * 1.5f;
        
        float breathOffset = sin(breathPhase) * 0.01f;
        float swayOffset = sin(swayPhase) * 0.005f;
        
        frame.headPos = Vec3(swayOffset, 1.5f + breathOffset * 0.5f, 0);
        frame.torsoPos = Vec3(swayOffset, 0.5f + breathOffset, 0);
        frame.leftArmPos = Vec3(-0.6f + swayOffset, 0.8f, 0);
        frame.rightArmPos = Vec3(0.6f + swayOffset, 0.8f, 0);
        frame.leftLegPos = Vec3(-0.2f + swayOffset, -0.8f, 0);
        frame.rightLegPos = Vec3(0.2f + swayOffset, -0.8f, 0);
        
        frame.energyLevel = 0.2f;
    }
    
    void generateWalkFrame(SimpleAnimationFrame& frame) const {
        float cycleTime = fmod(m_animationTime, 1.0f);
        float cyclePhase = cycleTime * 2.0f * M_PI;
        
        float verticalBob = sin(cyclePhase * 2.0f) * 0.05f;
        float legSwing = sin(cyclePhase) * 0.3f;
        float armSwing = sin(cyclePhase + M_PI) * 0.2f;
        
        frame.headPos = Vec3(0, 1.5f + verticalBob * 0.5f, 0);
        frame.torsoPos = Vec3(0, 0.5f + verticalBob, 0);
        
        frame.leftLegPos = Vec3(-0.2f, -0.8f, legSwing);
        frame.rightLegPos = Vec3(0.2f, -0.8f, -legSwing);
        
        frame.leftArmPos = Vec3(-0.6f, 0.8f, armSwing);
        frame.rightArmPos = Vec3(0.6f, 0.8f, -armSwing);
        
        frame.energyLevel = 0.6f;
    }
    
    void generateRunFrame(SimpleAnimationFrame& frame) const {
        float cycleTime = fmod(m_animationTime, 0.6f);
        float cyclePhase = cycleTime * 2.0f * M_PI / 0.6f;
        
        float verticalBounce = sin(cyclePhase * 2.0f) * 0.1f;
        float legStride = sin(cyclePhase) * 0.5f;
        float armPump = sin(cyclePhase + M_PI) * 0.4f;
        
        // Forward lean
        frame.headPos = Vec3(0, 1.5f + verticalBounce * 0.7f, 0.1f);
        frame.torsoPos = Vec3(0, 0.5f + verticalBounce, 0.05f);
        
        frame.leftLegPos = Vec3(-0.2f, -0.8f, legStride);
        frame.rightLegPos = Vec3(0.2f, -0.8f, -legStride);
        
        frame.leftArmPos = Vec3(-0.6f, 0.8f, armPump * 0.2f);
        frame.rightArmPos = Vec3(0.6f, 0.8f, -armPump * 0.2f);
        
        frame.energyLevel = 0.8f;
    }
    
    void generateSprintFrame(SimpleAnimationFrame& frame) const {
        float cycleTime = fmod(m_animationTime, 0.4f);
        float cyclePhase = cycleTime * 2.0f * M_PI / 0.4f;
        
        float verticalBound = sin(cyclePhase * 2.0f) * 0.15f;
        float legDrive = sin(cyclePhase) * 0.7f;
        float armDrive = sin(cyclePhase + M_PI) * 0.6f;
        
        // Aggressive forward lean
        frame.headPos = Vec3(0, 1.5f + verticalBound * 0.8f, 0.2f);
        frame.torsoPos = Vec3(0, 0.5f + verticalBound, 0.15f);
        
        frame.leftLegPos = Vec3(-0.2f, -0.8f, legDrive);
        frame.rightLegPos = Vec3(0.2f, -0.8f, -legDrive);
        
        frame.leftArmPos = Vec3(-0.6f, 0.8f, armDrive * 0.3f);
        frame.rightArmPos = Vec3(0.6f, 0.8f, -armDrive * 0.3f);
        
        frame.energyLevel = 1.0f;
    }
    
    void generateJumpFrame(SimpleAnimationFrame& frame) const {
        // Extended pose for jumping
        frame.headPos = Vec3(0, 1.7f, 0);
        frame.torsoPos = Vec3(0, 0.8f, 0);
        frame.leftArmPos = Vec3(-0.6f, 1.2f, 0);
        frame.rightArmPos = Vec3(0.6f, 1.2f, 0);
        frame.leftLegPos = Vec3(-0.2f, -0.4f, 0);
        frame.rightLegPos = Vec3(0.2f, -0.4f, 0);
        
        frame.energyLevel = 1.0f;
    }
};

class SimpleAnimationDemo {
private:
    SimpleAnimationController m_controller;
    bool m_isRunning = false;
    float m_demoTime = 0.0f;
    
public:
    void run() {
        std::cout << "🎮 Simple Animation System Demo" << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << "Demonstrating basic character animation cycles" << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        std::cout << "================================" << std::endl;
        
        m_isRunning = true;
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        
        while (m_isRunning && frameCount < 500) { // Run for ~8.3 seconds at 60 FPS
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;
            
            deltaTime = std::min(deltaTime, 1.0f / 30.0f); // Cap at 30 FPS minimum
            
            update(deltaTime);
            frameCount++;
            
            // Sleep to maintain ~60 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            
            // Print status every 60 frames (1 second)
            if (frameCount % 60 == 0) {
                printStatus();
            }
            
            // Change scenario every 3 seconds
            if (frameCount % 180 == 0) {
                changeScenario(frameCount / 180);
            }
        }
        
        std::cout << "\n🎬 Demo completed!" << std::endl;
        printSummary();
    }
    
private:
    void update(float deltaTime) {
        m_demoTime += deltaTime;
        
        // Simulate different movement patterns
        Vec2 input = simulateInput();
        float speed = calculateSpeed(input);
        
        m_controller.update(deltaTime, input, speed);
    }
    
    Vec2 simulateInput() {
        float cycleTime = fmod(m_demoTime, 10.0f); // 10-second cycles
        
        if (cycleTime < 2.0f) {
            // Idle
            return Vec2(0, 0);
        } else if (cycleTime < 4.0f) {
            // Walking forward
            return Vec2(0, 1);
        } else if (cycleTime < 6.0f) {
            // Running in circles
            float angle = cycleTime * 2.0f * M_PI;
            return Vec2(sin(angle), cos(angle));
        } else if (cycleTime < 8.0f) {
            // Sprinting forward
            return Vec2(0, 1);
        } else {
            // Return to idle
            return Vec2(0, 0);
        }
    }
    
    float calculateSpeed(const Vec2& input) {
        float cycleTime = fmod(m_demoTime, 10.0f);
        
        if (cycleTime < 2.0f) {
            return 0.0f; // Idle
        } else if (cycleTime < 4.0f) {
            return 2.0f; // Walk speed
        } else if (cycleTime < 6.0f) {
            return 8.0f; // Run speed
        } else if (cycleTime < 8.0f) {
            return 20.0f; // Sprint speed
        } else {
            return 0.0f; // Return to idle
        }
    }
    
    void changeScenario(int scenarioIndex) {
        switch (scenarioIndex % 4) {
            case 0:
                std::cout << "\n🚶 Scenario: Basic Movement Showcase" << std::endl;
                break;
            case 1:
                std::cout << "\n🏃 Scenario: Progressive Speed Demo" << std::endl;
                break;
            case 2:
                std::cout << "\n⚡ Scenario: High Energy Sprint" << std::endl;
                break;
            case 3:
                std::cout << "\n🧘 Scenario: Return to Calm" << std::endl;
                break;
        }
    }
    
    void printStatus() {
        SimpleAnimationFrame frame = m_controller.getCurrentFrame();
        
        std::cout << "\n📊 Status (t=" << std::fixed << std::setprecision(1) << m_demoTime << "s)" << std::endl;
        std::cout << "   Animation: " << m_controller.getCurrentAnimationName() << std::endl;
        std::cout << "   Energy: " << std::setprecision(2) << frame.energyLevel << std::endl;
        std::cout << "   Head Y: " << frame.headPos.y << "m" << std::endl;
        std::cout << "   Torso Y: " << frame.torsoPos.y << "m" << std::endl;
    }
    
    void printSummary() {
        std::cout << "\n📈 Demo Summary" << std::endl;
        std::cout << "===============" << std::endl;
        std::cout << "Total runtime: " << m_demoTime << " seconds" << std::endl;
        std::cout << "Animations demonstrated:" << std::endl;
        std::cout << "  ✅ Idle breathing and sway" << std::endl;
        std::cout << "  ✅ Walking gait cycle" << std::endl;
        std::cout << "  ✅ Running with bob and lean" << std::endl;
        std::cout << "  ✅ Sprinting with aggressive lean" << std::endl;
        std::cout << "  ✅ Smooth state transitions" << std::endl;
        std::cout << "  ✅ Procedural movement generation" << std::endl;
    }
};

int main() {
    std::cout << "🎮 Simple Animation System Demo" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "A lightweight demonstration of character animation cycles" << std::endl;
    std::cout << "Features: Idle, Walk, Run, Sprint animations with procedural enhancement" << std::endl;
    std::cout << "===============================" << std::endl;
    
    SimpleAnimationDemo demo;
    
    try {
        demo.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 Thank you for watching the Simple Animation Demo!" << std::endl;
    return 0;
}
