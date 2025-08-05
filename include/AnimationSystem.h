#pragma once

#define _USE_MATH_DEFINES
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <glm/glm.hpp>

namespace CudaGame {
namespace Animation {

/**
 * @brief Enhanced Animation System for AAA-grade character movement
 * 
 * This system provides sophisticated animation blending, procedural movement,
 * and state-driven animation transitions for realistic character locomotion.
 */

enum class MovementAnimationType {
    IDLE,
    WALK_FORWARD,
    WALK_BACKWARD,
    WALK_LEFT,
    WALK_RIGHT,
    RUN_FORWARD,
    RUN_BACKWARD,
    RUN_LEFT,
    RUN_RIGHT,
    SPRINT_FORWARD,
    SPRINT_BACKWARD,
    SPRINT_LEFT,
    SPRINT_RIGHT,
    JUMP_START,
    JUMP_APEX,
    JUMP_FALL,
    JUMP_LAND,
    DASH_HORIZONTAL,
    DASH_VERTICAL,
    WALL_RUN_LEFT,
    WALL_RUN_RIGHT,
    ATTACK_LIGHT_1,
    ATTACK_LIGHT_2,
    ATTACK_LIGHT_3,
    ATTACK_HEAVY,
    BLOCK_IDLE,
    BLOCK_HIT,
    TRANSITION // Special state for blending between animations
};

// Enhanced keyframe with interpolation metadata
struct AnimationKeyframe {
    float timeStamp;
    std::vector<glm::vec3> bodyPartPositions;
    std::vector<glm::vec3> bodyPartRotations;
    std::vector<glm::vec3> bodyPartScales;
    
    // Advanced features
    glm::vec3 rootMotion{0.0f}; // Root motion displacement
    float movementSpeed = 1.0f; // Speed multiplier for this frame
    float rhythmIntensity = 0.0f; // Rhythm-based intensity modifier
    
    // Procedural animation parameters
    float breathingAmplitude = 0.02f;
    float idleSway = 0.01f;
    float energyLevel = 1.0f; // Affects animation vigor
};

// Animation clip with advanced properties
struct AnimationClip {
    MovementAnimationType type;
    std::vector<AnimationKeyframe> keyframes;
    float duration;
    bool isLooping = true;
    bool allowBlending = true;
    float blendInTime = 0.2f;
    float blendOutTime = 0.2f;
    
    // Movement characteristics
    float minSpeed = 0.0f;
    float maxSpeed = 100.0f;
    glm::vec2 speedRange{0.0f, 10.0f}; // Min/max speeds for this animation
    
    // Procedural modifiers
    bool hasRootMotion = false;
    bool affectedByRhythm = true;
    float rhythmSensitivity = 0.5f;
    
    // Animation priority (higher = more important)
    int priority = 0;
};

// Animation state with blending information
struct AnimationState {
    MovementAnimationType currentType = MovementAnimationType::IDLE;
    MovementAnimationType targetType = MovementAnimationType::IDLE;
    float currentTime = 0.0f;
    float blendWeight = 1.0f;
    float transitionProgress = 0.0f;
    bool isTransitioning = false;
    
    // Locomotion parameters
    glm::vec2 movementDirection{0.0f};
    float movementSpeed = 0.0f;
    float turnSpeed = 0.0f;
    
    // Rhythm integration
    float rhythmPhase = 0.0f;
    bool isOnBeat = false;
    float beatIntensity = 0.0f;
};

class AnimationController {
public:
    AnimationController();
    ~AnimationController() = default;
    
    // Core animation control
    void initialize();
    void update(float deltaTime);
    void setMovementState(MovementAnimationType type, float blendTime = 0.2f);
    
    // Movement-based animation selection
    void updateMovementAnimation(const glm::vec2& inputDirection, float speed);
    void updateMovementAnimation(const glm::vec3& velocity, bool isGrounded);
    
    // Advanced animation features
    void setRhythmParameters(float phase, bool onBeat, float intensity);
    void setEnergyLevel(float energy); // Affects animation vigor
    void addProceduralLayer(const std::string& layerName, float weight);
    
    // Animation blending
    void blendAnimations(MovementAnimationType from, MovementAnimationType to, float factor);
    AnimationKeyframe getCurrentFrame() const;
    
    // Getters
    MovementAnimationType getCurrentAnimationType() const { return m_currentState.currentType; }
    const AnimationState& getAnimationState() const { return m_currentState; }
    bool isTransitioning() const { return m_currentState.isTransitioning; }
    float getAnimationProgress() const;
    
    // Body part access
    glm::vec3 getBodyPartPosition(int partIndex) const;
    glm::vec3 getBodyPartRotation(int partIndex) const;
    glm::vec3 getBodyPartScale(int partIndex) const;
    
private:
    AnimationState m_currentState;
    std::unordered_map<MovementAnimationType, std::unique_ptr<AnimationClip>> m_animationClips;
    
    // Procedural animation layers
    std::unordered_map<std::string, float> m_proceduralLayers;
    float m_globalEnergyLevel = 1.0f;
    
    // Rhythm integration
    float m_rhythmPhase = 0.0f;
    bool m_isOnBeat = false;
    float m_beatIntensity = 0.0f;
    
    // Current frame data (cached)
    mutable AnimationKeyframe m_currentFrame;
    mutable bool m_frameNeedsUpdate = true;
    
    // Internal methods
    void loadAnimationClips();
    void createIdleAnimation();
    void createWalkAnimations();
    void createRunAnimations();
    void createSprintAnimations();
    void createJumpAnimations();
    void createDashAnimations();
    void createWallRunAnimations();
    void createCombatAnimations();
    
    // Animation utilities
    AnimationKeyframe interpolateKeyframes(const AnimationKeyframe& a, const AnimationKeyframe& b, float t) const;
    MovementAnimationType selectMovementAnimation(const glm::vec2& direction, float speed) const;
    void updateTransition(float deltaTime);
    void applyProceduralAnimation(AnimationKeyframe& frame, float deltaTime) const;
    void applyRhythmModulation(AnimationKeyframe& frame) const;
    
    // Advanced blending
    float calculateBlendWeight(MovementAnimationType type, float speed) const;
    void smoothTransition(MovementAnimationType newType, float blendTime);
    
    // Helper methods for animation creation
    void createDirectionalVariation(MovementAnimationType newType, MovementAnimationType baseType, const glm::vec3& direction);
    void createJumpPhase(MovementAnimationType type, float duration);
    void createMirroredAnimation(MovementAnimationType newType, MovementAnimationType baseType);
    void createAttackVariation(MovementAnimationType newType, MovementAnimationType baseType, float intensity);
};

// Animation utilities for procedural generation
class ProceduralAnimationGenerator {
public:
    static void addBreathingAnimation(AnimationKeyframe& frame, float time, float amplitude = 0.02f);
    static void addIdleSway(AnimationKeyframe& frame, float time, float amplitude = 0.01f);
    static void addWalkBob(AnimationKeyframe& frame, float cycleTime, float speed);
    static void addRunBounce(AnimationKeyframe& frame, float cycleTime, float speed);
    static void addSprintLean(AnimationKeyframe& frame, const glm::vec2& direction, float speed);
    static void addRhythmPulse(AnimationKeyframe& frame, float phase, float intensity);
    
    // Advanced procedural effects
    static void addWindEffect(AnimationKeyframe& frame, const glm::vec3& windDirection, float strength);
    static void addMomentumLean(AnimationKeyframe& frame, const glm::vec3& velocity);
    static void addTurnAnticipation(AnimationKeyframe& frame, float turnSpeed);
};

// Animation event system for triggering effects
struct AnimationEvent {
    float triggerTime;
    std::string eventType;
    std::unordered_map<std::string, float> parameters;
};

class AnimationEventManager {
public:
    void registerEvent(MovementAnimationType animType, const AnimationEvent& event);
    void checkEvents(MovementAnimationType animType, float currentTime, float deltaTime);
    void setEventCallback(const std::string& eventType, std::function<void(const std::unordered_map<std::string, float>&)> callback);
    
private:
    std::unordered_map<MovementAnimationType, std::vector<AnimationEvent>> m_animationEvents;
    std::unordered_map<std::string, std::function<void(const std::unordered_map<std::string, float>&)>> m_eventCallbacks;
};

} // namespace Animation
} // namespace CudaGame
