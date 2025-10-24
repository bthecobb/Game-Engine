#pragma once

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

// Forward declarations
struct AnimationClip;
struct BoneTransform;
struct AnimationCurve;

// Enhanced animation curve types for smooth interpolation
enum class CurveType {
    LINEAR,
    EASE_IN,
    EASE_OUT,
    EASE_IN_OUT,
    SPRING,
    BOUNCE,
    ELASTIC,
    CUSTOM_BEZIER
};

// Animation interpolation modes
enum class InterpolationMode {
    LINEAR,
    SPHERICAL_LINEAR, // For rotations
    CUBIC_SPLINE,
    HERMITE,
    CATMULL_ROM
};

// Animation layer blending modes
enum class BlendMode {
    OVERRIDE,
    ADDITIVE,
    MULTIPLY,
    LAYER_ADDITIVE,
    MASKED
};

// Body part identification for detailed character animation
enum class BodyPart {
    // Core body
    PELVIS,
    SPINE_LOWER,
    SPINE_MIDDLE, 
    SPINE_UPPER,
    NECK,
    HEAD,
    
    // Left arm
    LEFT_SHOULDER,
    LEFT_UPPER_ARM,
    LEFT_FOREARM,
    LEFT_HAND,
    LEFT_FINGERS,
    
    // Right arm
    RIGHT_SHOULDER,
    RIGHT_UPPER_ARM,
    RIGHT_FOREARM,
    RIGHT_HAND,
    RIGHT_FINGERS,
    
    // Left leg
    LEFT_THIGH,
    LEFT_SHIN,
    LEFT_FOOT,
    LEFT_TOES,
    
    // Right leg
    RIGHT_THIGH,
    RIGHT_SHIN,
    RIGHT_FOOT,
    RIGHT_TOES,
    
    // Weapon attachment points
    WEAPON_MAIN_HAND,
    WEAPON_OFF_HAND,
    
    COUNT
};

// Animation event system for scripted actions
struct AnimationEvent {
    float timeStamp;
    std::string eventName;
    std::map<std::string, float> parameters;
    std::function<void(const std::map<std::string, float>&)> callback;
    
    AnimationEvent(float time, const std::string& name) 
        : timeStamp(time), eventName(name) {}
};

// Keyframe data for individual bone transformations
struct BoneKeyframe {
    float time;
    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;
    CurveType easingType;
    glm::vec4 bezierControlPoints; // For custom bezier curves
    
    BoneKeyframe(float t = 0.0f) 
        : time(t), position(0.0f), rotation(1.0f, 0.0f, 0.0f, 0.0f), 
          scale(1.0f), easingType(CurveType::LINEAR) {}
};

// Animation curve for smooth value interpolation
struct AnimationCurve {
    std::vector<BoneKeyframe> keyframes;
    InterpolationMode interpolation;
    bool loop;
    
    AnimationCurve() : interpolation(InterpolationMode::LINEAR), loop(false) {}
    
    // Evaluate the curve at a specific time
    BoneKeyframe evaluate(float time) const;
    
    // Add keyframe with automatic sorting
    void addKeyframe(const BoneKeyframe& keyframe);
    
    // Get the total duration of the curve
    float getDuration() const;
};

// Complete bone transformation data
struct BoneTransform {
    BodyPart bodyPart;
    AnimationCurve positionCurve;
    AnimationCurve rotationCurve;
    AnimationCurve scaleCurve;
    
    // Additional properties for physics-based animation
    float mass;
    float drag;
    bool usePhysics;
    glm::vec3 physicsVelocity;
    
    BoneTransform(BodyPart part) 
        : bodyPart(part), mass(1.0f), drag(0.1f), usePhysics(false), physicsVelocity(0.0f) {}
        
    // Get the final transform at a specific time
    glm::mat4 getTransformMatrix(float time) const;
    
    // Apply physics simulation
    void updatePhysics(float deltaTime, const glm::vec3& gravity = glm::vec3(0, -9.81f, 0));
};

// Animation clip containing all bone animations for a specific action
struct AnimationClip {
    std::string name;
    float duration;
    bool loop;
    float speed;
    std::map<BodyPart, BoneTransform> boneTransforms;
    std::vector<AnimationEvent> events;
    
    // Motion properties
    glm::vec3 rootMotionOffset;
    bool hasRootMotion;
    
    // Blend tree properties
    std::vector<std::string> blendParameters;
    
    AnimationClip(const std::string& clipName) 
        : name(clipName), duration(1.0f), loop(false), speed(1.0f), 
          rootMotionOffset(0.0f), hasRootMotion(false) {}
          
    // Get bone transform at specific time
    glm::mat4 getBoneTransform(BodyPart bodyPart, float time) const;
    
    // Add animation event
    void addEvent(float time, const std::string& eventName, 
                  std::function<void(const std::map<std::string, float>&)> callback);
    
    // Process events at current time
    void processEvents(float currentTime, float deltaTime);
};

// Animation state for state machine
struct AnimationState {
    std::string name;
    std::shared_ptr<AnimationClip> clip;
    std::vector<std::string> transitions;
    std::map<std::string, float> conditions;
    BlendMode blendMode;
    float weight;
    
    AnimationState(const std::string& stateName) 
        : name(stateName), blendMode(BlendMode::OVERRIDE), weight(1.0f) {}
};

// Animation layer for complex blending
struct AnimationLayer {
    std::string name;
    std::map<std::string, AnimationState> states;
    std::string currentState;
    float weight;
    BlendMode blendMode;
    std::vector<BodyPart> affectedBones;
    
    AnimationLayer(const std::string& layerName) 
        : name(layerName), weight(1.0f), blendMode(BlendMode::OVERRIDE) {}
        
    // Transition to a new state
    bool transitionTo(const std::string& newState, const std::map<std::string, float>& parameters);
    
    // Update the current state
    void update(float deltaTime);
    
    // Get the final transform for a body part
    glm::mat4 getLayerTransform(BodyPart bodyPart, float time) const;
};

// Main fluid animation controller
class FluidAnimationController {
private:
    std::map<std::string, std::shared_ptr<AnimationClip>> animationClips;
    std::vector<AnimationLayer> layers;
    std::map<std::string, float> parameters;
    
    // Blending state
    std::string currentAnimation;
    std::string targetAnimation;
    float blendTime;
    float blendDuration;
    bool isBlending;
    
    // Root motion
    glm::vec3 rootMotionVelocity;
    glm::vec3 totalRootMotion;
    
    // Physics integration
    bool usePhysicsBlending;
    float physicsBlendWeight;
    
public:
    FluidAnimationController();
    ~FluidAnimationController();
    
    // Clip management
    void loadAnimationClip(const std::string& name, std::shared_ptr<AnimationClip> clip);
    void removeAnimationClip(const std::string& name);
    std::shared_ptr<AnimationClip> getAnimationClip(const std::string& name);
    
    // Layer management
    void addLayer(const std::string& name, float weight = 1.0f, BlendMode blendMode = BlendMode::OVERRIDE);
    void removeLayer(const std::string& name);
    void setLayerWeight(const std::string& name, float weight);
    AnimationLayer* getLayer(const std::string& name);
    
    // State management
    void addState(const std::string& layerName, const std::string& stateName, 
                  const std::string& clipName);
    void addTransition(const std::string& layerName, const std::string& fromState, 
                       const std::string& toState, const std::map<std::string, float>& conditions);
    
    // Parameter control
    void setParameter(const std::string& name, float value);
    float getParameter(const std::string& name) const;
    void setBool(const std::string& name, bool value);
    bool getBool(const std::string& name) const;
    
    // Animation control
    void playAnimation(const std::string& name, float blendTime = 0.3f);
    void crossFade(const std::string& fromAnim, const std::string& toAnim, float duration);
    void stopAnimation(const std::string& name);
    void pauseAnimation(const std::string& name);
    void resumeAnimation(const std::string& name);
    
    // Update and evaluation
    void update(float deltaTime);
    glm::mat4 getBoneTransform(BodyPart bodyPart) const;
    std::map<BodyPart, glm::mat4> getAllBoneTransforms() const;
    
    // Root motion
    glm::vec3 getRootMotionVelocity() const { return rootMotionVelocity; }
    glm::vec3 consumeRootMotion();
    void enableRootMotion(bool enable);
    
    // Physics integration
    void enablePhysicsBlending(bool enable, float blendWeight = 0.5f);
    void applyPhysicsImpulse(BodyPart bodyPart, const glm::vec3& impulse);
    
    // Advanced features
    void setTimeScale(float scale);
    void addIKTarget(BodyPart bodyPart, const glm::vec3& target);
    void removeIKTarget(BodyPart bodyPart);
    void setLookAtTarget(const glm::vec3& target);
    
    // Scripting interface
    void executeScript(const std::string& script);
    void registerScriptFunction(const std::string& name, std::function<void()> func);
};

// Animation builder for easy clip creation
class AnimationBuilder {
private:
    std::shared_ptr<AnimationClip> currentClip;
    
public:
    AnimationBuilder(const std::string& clipName);
    
    // Clip properties
    AnimationBuilder& setDuration(float duration);
    AnimationBuilder& setLoop(bool loop);
    AnimationBuilder& setSpeed(float speed);
    AnimationBuilder& enableRootMotion(const glm::vec3& offset);
    
    // Keyframe creation
    AnimationBuilder& addKeyframe(BodyPart bodyPart, float time, 
                                  const glm::vec3& position, 
                                  const glm::quat& rotation = glm::quat(1,0,0,0),
                                  const glm::vec3& scale = glm::vec3(1.0f));
    
    // Curve manipulation
    AnimationBuilder& setCurveType(BodyPart bodyPart, CurveType type);
    AnimationBuilder& setInterpolation(BodyPart bodyPart, InterpolationMode mode);
    
    // Events
    AnimationBuilder& addEvent(float time, const std::string& eventName,
                               std::function<void(const std::map<std::string, float>&)> callback);
    
    // Physics
    AnimationBuilder& enablePhysics(BodyPart bodyPart, float mass = 1.0f, float drag = 0.1f);
    
    // Build the final clip
    std::shared_ptr<AnimationClip> build();
};

// Predefined animation presets for common movements
namespace AnimationPresets {
    // Basic movement animations
    std::shared_ptr<AnimationClip> createIdleAnimation();
    std::shared_ptr<AnimationClip> createWalkAnimation(float speed = 1.0f);
    std::shared_ptr<AnimationClip> createRunAnimation(float speed = 1.5f);
    std::shared_ptr<AnimationClip> createSprintAnimation(float speed = 2.0f);
    
    // Jump animations
    std::shared_ptr<AnimationClip> createJumpStartAnimation();
    std::shared_ptr<AnimationClip> createJumpAirAnimation();
    std::shared_ptr<AnimationClip> createJumpLandAnimation();
    std::shared_ptr<AnimationClip> createDoubleJumpAnimation();
    
    // Combat animations
    std::shared_ptr<AnimationClip> createPunchComboAnimation();
    std::shared_ptr<AnimationClip> createKickAnimation();
    std::shared_ptr<AnimationClip> createBlockAnimation();
    std::shared_ptr<AnimationClip> createDodgeAnimation();
    
    // Weapon animations
    std::shared_ptr<AnimationClip> createSwordSlashAnimation();
    std::shared_ptr<AnimationClip> createStaffCastAnimation();
    std::shared_ptr<AnimationClip> createHammerSlamAnimation();
    
    // Advanced movements
    std::shared_ptr<AnimationClip> createWallRunAnimation();
    std::shared_ptr<AnimationClip> createSlideAnimation();
    std::shared_ptr<AnimationClip> createDashAnimation();
    
    // Reaction animations
    std::shared_ptr<AnimationClip> createHitReactionAnimation();
    std::shared_ptr<AnimationClip> createDeathAnimation();
    std::shared_ptr<AnimationClip> createStunAnimation();
    
    // Enemy-specific animations
    std::shared_ptr<AnimationClip> createEnemyPatrolAnimation();
    std::shared_ptr<AnimationClip> createEnemyAlertAnimation();
    std::shared_ptr<AnimationClip> createEnemyAttackAnimation();
    std::shared_ptr<AnimationClip> createEnemyChaseAnimation();
}

// Utility functions for animation manipulation
namespace AnimationUtils {
    // Interpolation helpers
    float easeIn(float t);
    float easeOut(float t);
    float easeInOut(float t);
    float bounce(float t);
    float elastic(float t);
    float spring(float t, float damping = 0.5f);
    
    // Curve evaluation
    glm::vec3 evaluateBezier(const glm::vec3& p0, const glm::vec3& p1, 
                             const glm::vec3& p2, const glm::vec3& p3, float t);
    glm::quat slerpQuaternions(const glm::quat& a, const glm::quat& b, float t);
    
    // Bone hierarchy helpers
    std::vector<BodyPart> getChildBones(BodyPart parent);
    BodyPart getParentBone(BodyPart child);
    bool isChildOf(BodyPart child, BodyPart parent);
    
    // Animation blending
    glm::mat4 blendTransforms(const glm::mat4& a, const glm::mat4& b, float weight, BlendMode mode);
    std::map<BodyPart, glm::mat4> blendAnimations(
        const std::map<BodyPart, glm::mat4>& animA,
        const std::map<BodyPart, glm::mat4>& animB,
        float weight, BlendMode mode);
}
