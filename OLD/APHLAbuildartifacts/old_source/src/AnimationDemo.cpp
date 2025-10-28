#include "AnimationSystem.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <random>

using namespace CudaGame::Animation;

class AnimationSystemDemo {
private:
    AnimationController m_animationController;
    RhythmFeedbackSystem m_rhythmFeedback;
    AnimationEventSystem m_eventSystem;
    
    // Demo parameters
    bool m_isRunning;
    float m_demoTime;
    std::mt19937 m_rng;
    
    // Simulated audio data
    std::vector<float> m_simulatedAudio;
    float m_audioPhase;
    float m_demoBPM;
    
    // Input simulation
    glm::vec2 m_currentInput;
    float m_currentSpeed;
    
public:
    AnimationSystemDemo() 
        : m_isRunning(false), m_demoTime(0.0f), m_rng(std::random_device{}()),
          m_audioPhase(0.0f), m_demoBPM(128.0f), m_currentInput(0.0f), m_currentSpeed(0.0f) {
        
        m_simulatedAudio.resize(512, 0.0f); // Small audio buffer for demo
    }
    
    void initialize() {
        std::cout << "🎮 Initializing Animation System Demo..." << std::endl;
        std::cout << "===========================================" << std::endl;
        
        // Initialize animation controller
        m_animationController.initialize();
        
        // Initialize rhythm feedback system
        m_rhythmFeedback.initialize(&m_animationController);
        m_rhythmFeedback.getRhythmAnalyzer().calibrate(m_demoBPM);
        
        // Setup event listeners
        setupEventListeners();
        
        std::cout << "✅ Demo initialized successfully!" << std::endl;
        std::cout << "===========================================" << std::endl;
    }
    
    void setupEventListeners() {
        // Footstep events
        m_eventSystem.addListener(AnimationEventType::FOOTSTEP, 
            [](const AnimationEventData& data) {
                std::cout << "👟 Footstep! Foot " << data.footIndex 
                          << " at intensity " << data.intensity << std::endl;
            });
        
        // Attack events
        m_eventSystem.addListener(AnimationEventType::ATTACK_IMPACT,
            [](const AnimationEventData& data) {
                std::cout << "👊 Attack impact! Intensity: " << data.intensity 
                          << ", Velocity: (" << data.velocity.x << ", " << data.velocity.y << ", " << data.velocity.z << ")" << std::endl;
            });
        
        // Rhythm beat events
        m_eventSystem.addListener(AnimationEventType::RHYTHM_BEAT,
            [](const AnimationEventData& data) {
                std::cout << "🎵 Rhythm beat! Phase: " << data.rhythmPhase 
                          << ", Intensity: " << data.intensity << std::endl;
            });
    }
    
    void run() {
        std::cout << "\n🎬 Starting Animation System Demo..." << std::endl;
        std::cout << "Press Ctrl+C to stop the demo" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        m_isRunning = true;
        
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        
        while (m_isRunning && frameCount < 1000) { // Run for ~16.7 seconds at 60 FPS
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;
            
            // Cap delta time to prevent huge jumps
            deltaTime = std::min(deltaTime, 1.0f / 30.0f); // Max 30 FPS minimum
            
            update(deltaTime);
            
            frameCount++;
            
            // Sleep to maintain ~60 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            
            // Change demo scenarios periodically
            if (frameCount % 180 == 0) { // Every 3 seconds
                changeScenario(frameCount / 180);
            }
        }
        
        std::cout << "\n🎬 Demo completed!" << std::endl;
        printSummary();
    }
    
private:
    void update(float deltaTime) {
        m_demoTime += deltaTime;
        
        // Generate simulated audio data with beat
        generateSimulatedAudio(deltaTime);
        
        // Simulate input based on current scenario
        simulateInput();
        
        // Update animation controller
        m_animationController.updateMovementAnimation(m_currentInput, m_currentSpeed);
        m_animationController.update(deltaTime);
        
        // Update rhythm feedback system
        m_rhythmFeedback.update(m_simulatedAudio, deltaTime);
        
        // Get current animation frame and check for events
        AnimationKeyframe currentFrame = m_animationController.getCurrentFrame();
        static AnimationKeyframe previousFrame = currentFrame;
        
        m_eventSystem.update(currentFrame, previousFrame);
        previousFrame = currentFrame;
        
        // Process rhythm events
        auto rhythmEvents = m_rhythmFeedback.consumeRhythmEvents();
        for (const auto& event : rhythmEvents) {
            processRhythmEvent(event);
        }
        
        // Print status every 60 frames (1 second)
        static int statusCounter = 0;
        if (++statusCounter >= 60) {
            printStatus();
            statusCounter = 0;
        }
    }
    
    void generateSimulatedAudio(float deltaTime) {
        // Update audio phase based on BPM
        float beatsPerSecond = m_demoBPM / 60.0f;
        m_audioPhase += beatsPerSecond * deltaTime;
        
        // Wrap phase
        while (m_audioPhase >= 1.0f) {
            m_audioPhase -= 1.0f;
        }
        
        // Generate audio buffer with beat pattern
        for (size_t i = 0; i < m_simulatedAudio.size(); ++i) {
            float t = (float)i / m_simulatedAudio.size();
            
            // Bass drum on beat (kick)
            float kick = (m_audioPhase < 0.1f) ? 0.8f * exp(-10.0f * m_audioPhase) : 0.0f;
            
            // Hi-hat on off-beats
            float hihat = ((m_audioPhase > 0.4f && m_audioPhase < 0.6f) || 
                          (m_audioPhase > 0.9f || m_audioPhase < 0.1f)) ? 0.3f : 0.0f;
            
            // Add some bass frequency content
            float bass = kick * sin(t * 2.0f * M_PI * 60.0f); // 60 Hz bass
            
            // Add mid frequency content
            float mid = hihat * sin(t * 2.0f * M_PI * 1000.0f); // 1kHz hi-hat
            
            // Combine and add some noise
            std::uniform_real_distribution<float> noise(-0.05f, 0.05f);
            m_simulatedAudio[i] = bass + mid + noise(m_rng);
        }
    }
    
    void simulateInput() {
        // Different input patterns based on demo time
        float cycleTime = fmod(m_demoTime, 12.0f); // 12-second cycles
        
        if (cycleTime < 2.0f) {
            // Idle
            m_currentInput = glm::vec2(0.0f);
            m_currentSpeed = 0.0f;
        } else if (cycleTime < 4.0f) {
            // Walking forward
            m_currentInput = glm::vec2(0.0f, 1.0f);
            m_currentSpeed = 2.0f;
        } else if (cycleTime < 6.0f) {
            // Running in circles
            float angle = cycleTime * 2.0f * M_PI;
            m_currentInput = glm::vec2(sin(angle), cos(angle));
            m_currentSpeed = 8.0f;
        } else if (cycleTime < 8.0f) {
            // Sprinting forward
            m_currentInput = glm::vec2(0.0f, 1.0f);
            m_currentSpeed = 20.0f;
        } else if (cycleTime < 10.0f) {
            // Side-stepping (dancing to the beat)
            RhythmParameters rhythm = m_rhythmFeedback.getRhythmParameters();
            float sideStep = sin(rhythm.phase * 2.0f * M_PI) * rhythm.intensity;
            m_currentInput = glm::vec2(sideStep, 0.0f);
            m_currentSpeed = 3.0f;
        } else {
            // Return to idle
            m_currentInput = glm::vec2(0.0f);
            m_currentSpeed = 0.0f;
        }
    }
    
    void changeScenario(int scenarioIndex) {
        switch (scenarioIndex % 5) {
            case 0:
                std::cout << "\n🎪 Scenario: Basic Movement Showcase" << std::endl;
                m_demoBPM = 120.0f;
                break;
            case 1:
                std::cout << "\n🎵 Scenario: Rhythm Dancing" << std::endl;
                m_demoBPM = 128.0f;
                m_rhythmFeedback.setSyncStrength(1.5f);
                break;
            case 2:
                std::cout << "\n⚡ Scenario: High Energy Sprint" << std::endl;
                m_demoBPM = 140.0f;
                m_rhythmFeedback.setFeedbackIntensity(1.0f);
                break;
            case 3:
                std::cout << "\n🧘 Scenario: Calm Breathing" << std::endl;
                m_demoBPM = 80.0f;
                m_rhythmFeedback.setSyncStrength(0.5f);
                break;
            case 4:
                std::cout << "\n🎭 Scenario: Combat Simulation" << std::endl;
                m_demoBPM = 160.0f;
                simulateCombat();
                break;
        }
        
        // Recalibrate rhythm system
        m_rhythmFeedback.getRhythmAnalyzer().calibrate(m_demoBPM);
        
        std::cout << "Set BPM to " << m_demoBPM << std::endl;
    }
    
    void simulateCombat() {
        // Trigger some attack animations
        if (m_demoTime > 1.0f) {
            m_animationController.setMovementState(MovementAnimationType::ATTACK_LIGHT_1);
        }
    }
    
    void processRhythmEvent(const RhythmEvent& event) {
        switch (event.type) {
            case RhythmEventType::BEAT_HIT:
                // Could trigger particle effects, screen shake, etc.
                break;
            case RhythmEventType::PHASE_CHANGE:
                // Could trigger color changes, lighting effects, etc.
                break;
        }
    }
    
    void printStatus() {
        // Get current animation state
        AnimationState animState = m_animationController.getAnimationState();
        RhythmParameters rhythmParams = m_rhythmFeedback.getRhythmParameters();
        MovementModifiers movementMods = m_rhythmFeedback.getMovementModifiers();
        VisualEffects visualFX = m_rhythmFeedback.getVisualEffects();
        
        std::cout << "\n📊 Status Update (t=" << std::fixed << std::setprecision(1) << m_demoTime << "s)" << std::endl;
        std::cout << "   Animation: " << animationTypeToString(animState.currentType) << std::endl;
        std::cout << "   Input: (" << std::setprecision(2) << m_currentInput.x << ", " << m_currentInput.y << ") Speed: " << m_currentSpeed << std::endl;
        std::cout << "   Rhythm: Phase=" << rhythmParams.phase << ", OnBeat=" << (rhythmParams.isOnBeat ? "Yes" : "No") << ", Intensity=" << rhythmParams.intensity << std::endl;
        std::cout << "   Movement Boost: " << movementMods.speedMultiplier << "x" << std::endl;
        std::cout << "   Visual Pulse: " << visualFX.pulseAmplitude << std::endl;
    }
    
    void printSummary() {
        std::cout << "\n📈 Demo Summary" << std::endl;
        std::cout << "=================" << std::endl;
        std::cout << "Total runtime: " << m_demoTime << " seconds" << std::endl;
        std::cout << "Final BPM: " << m_demoBPM << std::endl;
        
        RhythmParameters finalRhythm = m_rhythmFeedback.getRhythmParameters();
        std::cout << "Final rhythm confidence: " << finalRhythm.intensity << std::endl;
        
        std::cout << "\n✨ Features demonstrated:" << std::endl;
        std::cout << "  ✅ Full movement animation cycles (idle, walk, run, sprint)" << std::endl;
        std::cout << "  ✅ Rhythm beat detection and analysis" << std::endl;
        std::cout << "  ✅ Animation synchronization with audio beats" << std::endl;
        std::cout << "  ✅ Procedural animation enhancements" << std::endl;
        std::cout << "  ✅ Event-driven animation system" << std::endl;
        std::cout << "  ✅ Adaptive BPM detection" << std::endl;
        std::cout << "  ✅ Multi-layered feedback system" << std::endl;
        std::cout << "  ✅ Visual and movement modulation" << std::endl;
    }
    
    std::string animationTypeToString(MovementAnimationType type) {
        switch (type) {
            case MovementAnimationType::IDLE: return "Idle";
            case MovementAnimationType::WALK_FORWARD: return "Walk Forward";
            case MovementAnimationType::WALK_BACKWARD: return "Walk Backward";
            case MovementAnimationType::WALK_LEFT: return "Walk Left";
            case MovementAnimationType::WALK_RIGHT: return "Walk Right";
            case MovementAnimationType::RUN_FORWARD: return "Run Forward";
            case MovementAnimationType::RUN_BACKWARD: return "Run Backward";
            case MovementAnimationType::RUN_LEFT: return "Run Left";
            case MovementAnimationType::RUN_RIGHT: return "Run Right";
            case MovementAnimationType::SPRINT_FORWARD: return "Sprint Forward";
            case MovementAnimationType::SPRINT_BACKWARD: return "Sprint Backward";
            case MovementAnimationType::SPRINT_LEFT: return "Sprint Left";
            case MovementAnimationType::SPRINT_RIGHT: return "Sprint Right";
            case MovementAnimationType::JUMP_START: return "Jump Start";
            case MovementAnimationType::JUMP_APEX: return "Jump Apex";
            case MovementAnimationType::JUMP_FALL: return "Jump Fall";
            case MovementAnimationType::JUMP_LAND: return "Jump Land";
            case MovementAnimationType::DASH_HORIZONTAL: return "Dash";
            case MovementAnimationType::WALL_RUN_LEFT: return "Wall Run Left";
            case MovementAnimationType::WALL_RUN_RIGHT: return "Wall Run Right";
            case MovementAnimationType::ATTACK_LIGHT_1: return "Light Attack 1";
            case MovementAnimationType::ATTACK_LIGHT_2: return "Light Attack 2";
            case MovementAnimationType::ATTACK_LIGHT_3: return "Light Attack 3";
            case MovementAnimationType::ATTACK_HEAVY: return "Heavy Attack";
            default: return "Unknown";
        }
    }
};

// Demo entry point
int main() {
    std::cout << "🎮 Enhanced Animation System with Rhythm Feedback Demo" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "This demo showcases:" << std::endl;
    std::cout << "• Complete movement animation cycles" << std::endl;
    std::cout << "• Real-time rhythm analysis and beat detection" << std::endl;
    std::cout << "• Animation synchronization with audio beats" << std::endl;
    std::cout << "• Procedural animation enhancements" << std::endl;
    std::cout << "• Event-driven feedback systems" << std::endl;
    std::cout << "• Adaptive movement and visual modulation" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    AnimationSystemDemo demo;
    
    try {
        demo.initialize();
        demo.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 Thank you for watching the Animation System Demo!" << std::endl;
    return 0;
}

// Additional utility for external testing
extern "C" {
    // C interface for testing from other languages
    void* create_animation_demo() {
        return new AnimationSystemDemo();
    }
    
    void initialize_demo(void* demo) {
        static_cast<AnimationSystemDemo*>(demo)->initialize();
    }
    
    void run_demo(void* demo) {
        static_cast<AnimationSystemDemo*>(demo)->run();
    }
    
    void destroy_demo(void* demo) {
        delete static_cast<AnimationSystemDemo*>(demo);
    }
}
