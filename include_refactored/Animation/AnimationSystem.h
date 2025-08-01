#pragma once

#include "Core/System.h"
#include "Animation/BoneTransform.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>

namespace CudaGame {
namespace Animation {

// Forward declarations
class AnimationClip;
class Skeleton;
class BlendTree;

// Animation states for AAA-quality character animation
enum class AnimationState {
    IDLE,
    IDLE_BORED,
    WALKING,
    RUNNING,
    SPRINTING,
    JUMPING,
    AIRBORNE,
    FALLING,
    LANDING,
    DIVING,
    WALL_RUNNING,
    SLIDING,
    COMBAT_IDLE,
    ATTACKING,
    PARRYING,
    GRABBING,
    STUNNED,
    DEATH,
    // Weapon-specific animations
    SWORD_IDLE,
    SWORD_ATTACK_1,
    SWORD_ATTACK_2,
    SWORD_COMBO_FINISHER,
    STAFF_CAST,
    STAFF_SPIN,
    HAMMER_CHARGE,
    HAMMER_SLAM
};

// Animation blend modes for smooth transitions
enum class BlendMode {
    REPLACE,     // Replace current animation
    ADDITIVE,    // Add to current animation
    MULTIPLY,    // Multiply with current animation
    OVERLAY      // Overlay on top of current animation
};

// Animation keyframe
struct Keyframe {
    float time;
    std::vector<BoneTransform> boneTransforms;
};

// AnimationClip is forward declared above and defined in AnimationResources.h

// Animation component for entities
struct AnimationComponent {
    AnimationState currentState = AnimationState::IDLE;
    AnimationState previousState = AnimationState::IDLE;
    float animationTime = 0.0f;
    float blendWeight = 1.0f;
    float playbackSpeed = 1.0f;
    bool isPlaying = true;
    bool hasTransitioned = false;
    
    // Blend tree parameters
    float movementSpeed = 0.0f;
    float combatIntensity = 0.0f;
    float rhythmSync = 0.0f;
    glm::vec2 movementDirection{0.0f};
};

// Animation state machine
class AnimationStateMachine {
public:
    AnimationStateMachine();
    ~AnimationStateMachine();
    
    void update(float deltaTime);
    void transitionTo(AnimationState newState, float transitionTime = 0.2f);
    bool canTransitionTo(AnimationState targetState) const;
    
    AnimationState getCurrentState() const { return m_currentState; }
    float getStateTime() const { return m_stateTime; }
    bool isTransitioning() const { return m_isTransitioning; }
    
private:
    AnimationState m_currentState;
    AnimationState m_previousState;
    float m_stateTime;
    float m_transitionTime;
    float m_transitionDuration;
    bool m_isTransitioning;
    
    std::unordered_map<AnimationState, std::vector<AnimationState>> m_transitions;
    void initializeTransitions();
};

// Main animation system
class AnimationSystem : public Core::System {
public:
    AnimationSystem();
    ~AnimationSystem();
    
    // Override Core::System methods
    bool Initialize() override { return initialize(); }
    void Shutdown() override { shutdown(); }
    void Update(float deltaTime) override { update(deltaTime); }
    
    // Configuration methods for integration
    void Configure() {}
    void LoadResources() {}
    
    // AnimationSystem specific methods
    bool initialize();
    void shutdown();
    void update(float deltaTime);
    
    // Animation clip management
    bool loadAnimationClip(const std::string& filePath);
    AnimationClip* getAnimationClip(const std::string& name);
    void registerAnimationClip(std::unique_ptr<AnimationClip> clip);
    
    // Entity animation control
    void playAnimation(uint32_t entityId, AnimationState state, float transitionTime = 0.2f);
    void setAnimationSpeed(uint32_t entityId, float speed);
    void pauseAnimation(uint32_t entityId);
    void resumeAnimation(uint32_t entityId);
    
    // Blend tree parameters
    void setBlendParameter(uint32_t entityId, const std::string& paramName, float value);
    void setBlendParameter(uint32_t entityId, const std::string& paramName, const glm::vec2& value);
    
    // Advanced features
    void enableRootMotion(uint32_t entityId, bool enable);
    void setLayerWeight(uint32_t entityId, int layerIndex, float weight);
    void crossFadeAnimation(uint32_t entityId, AnimationState targetState, float duration);
    
    // Rhythm integration
    void syncToRhythm(uint32_t entityId, float beatTiming, float bpm);
    void setRhythmMultiplier(float multiplier);
    
    // IK (Inverse Kinematics) support
    void enableIK(uint32_t entityId, bool enable);
    void setIKTarget(uint32_t entityId, const std::string& chainName, const glm::vec3& target);
    
    // Animation events
    void registerAnimationEvent(const std::string& eventName, std::function<void()> callback);
    void triggerAnimationEvent(const std::string& eventName);
    
    // Debug and visualization
    void setDebugVisualization(bool enable);
    void drawAnimationDebug(uint32_t entityId);
    
private:
    bool m_initialized;
    std::unordered_map<std::string, std::unique_ptr<AnimationClip>> m_animationClips;
    std::unordered_map<uint32_t, AnimationComponent> m_animationComponents;
    std::unordered_map<uint32_t, std::unique_ptr<AnimationStateMachine>> m_stateMachines;
    
    // Rhythm system integration
    float m_currentBPM;
    float m_beatTimer;
    float m_rhythmMultiplier;
    
    // Debug
    bool m_debugVisualization;
    
    // Internal methods
    void updateEntityAnimation(uint32_t entityId, AnimationComponent& component, float deltaTime);
    void applyBoneTransforms(uint32_t entityId, const std::vector<BoneTransform>& transforms);
    BoneTransform blendBoneTransforms(const BoneTransform& a, const BoneTransform& b, float weight);
    
    // Default animations creation
    void createDefaultAnimations();
    std::unique_ptr<AnimationClip> createIdleAnimation();
    std::unique_ptr<AnimationClip> createWalkAnimation();
    std::unique_ptr<AnimationClip> createRunAnimation();
    std::unique_ptr<AnimationClip> createAttackAnimation();
};

// Animation events
struct AnimationEvent {
    std::string name;
    float triggerTime;
    std::function<void()> callback;
};

// Helper functions for animation math
namespace AnimationMath {
    glm::quat slerp(const glm::quat& a, const glm::quat& b, float t);
    glm::vec3 lerp(const glm::vec3& a, const glm::vec3& b, float t);
    float smoothStep(float t);
    float easeInOut(float t);
}

} // namespace Animation
} // namespace CudaGame
