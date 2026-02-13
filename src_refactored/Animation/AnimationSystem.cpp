#include "Animation/AnimationSystem.h"
#include "Animation/AnimationResources.h"
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp> // Required for slerp
#include "Core/Coordinator.h" // Required for ECS integration
#include "Physics/PhysicsComponents.h"
#include <cmath>
#include <algorithm>

namespace CudaGame {
namespace Animation {

// AnimationStateMachine implementation
SimpleAnimationStateMachine::SimpleAnimationStateMachine()
    : m_currentState(AnimationState::IDLE), m_previousState(AnimationState::IDLE),
      m_stateTime(0.0f), m_transitionTime(0.0f), m_transitionDuration(0.0f),
      m_isTransitioning(false) {
    initializeTransitions();
}

// Improved helper that takes the Channel directly
static BoneTransform InterpolateChannel(const AnimationClip::Channel& channel, float time) {
    BoneTransform result;
    
    // Position
    if (!channel.positions.empty()) {
        if (channel.positions.size() == 1 || channel.times.size() < 2) {
            result.position = channel.positions[0];
        } else {
            // Find keyframe
            size_t p0 = 0, p1 = 0;
            // Ensure loop bound is valid
            size_t maxFrame = channel.times.size() - 1;
            for (size_t i = 0; i < maxFrame; i++) {
                if (time < channel.times[i+1]) {
                    p0 = i;
                    p1 = i + 1;
                    break;
                }
            }
            // Safety against p1 out of bounds (if time > duration)
            if (p1 >= channel.positions.size()) {
                 p0 = channel.positions.size() - 1;
                 p1 = p0;
            }

            float t0 = channel.times[p0];
            float t1 = channel.times[p1];
            float factor = (t1 - t0 > 0.0001f) ? (time - t0) / (t1 - t0) : 0.0f;
            result.position = glm::mix(channel.positions[p0], channel.positions[p1], factor);
        }
    }
    
    // Rotation
    if (!channel.rotations.empty()) {
        if (channel.rotations.size() == 1 || channel.times.size() < 2) {
            result.rotation = channel.rotations[0];
        } else {
             size_t p0 = 0, p1 = 0;
             size_t maxFrame = channel.times.size() - 1;
            for (size_t i = 0; i < maxFrame; i++) {
                if (time < channel.times[i+1]) {
                    p0 = i;
                    p1 = i + 1;
                    break;
                }
            }
            if (p1 >= channel.rotations.size()) {
                 p0 = channel.rotations.size() - 1;
                 p1 = p0;
            }

            float t0 = channel.times[p0];
            float t1 = channel.times[p1];
            float factor = (t1 - t0 > 0.0001f) ? (time - t0) / (t1 - t0) : 0.0f;
            result.rotation = glm::slerp(channel.rotations[p0], channel.rotations[p1], factor);
        }
    }
    
    // Scale
    if (!channel.scales.empty()) {
        if (channel.scales.size() == 1 || channel.times.size() < 2) {
            result.scale = channel.scales[0];
        } else {
             size_t p0 = 0, p1 = 0;
             size_t maxFrame = channel.times.size() - 1;
            for (size_t i = 0; i < maxFrame; i++) {
                if (time < channel.times[i+1]) {
                    p0 = i;
                    p1 = i + 1;
                    break;
                }
            }
            if (p1 >= channel.scales.size()) {
                 p0 = channel.scales.size() - 1;
                 p1 = p0;
            }
            
            float t0 = channel.times[p0];
            float t1 = channel.times[p1];
            float factor = (t1 - t0 > 0.0001f) ? (time - t0) / (t1 - t0) : 0.0f;
            result.scale = glm::mix(channel.scales[p0], channel.scales[p1], factor);
        }
    }
    return result;
}

SimpleAnimationStateMachine::~SimpleAnimationStateMachine() {}

void SimpleAnimationStateMachine::update(float deltaTime) {
    // Placeholder for updating the animation logic
    m_stateTime += deltaTime;
    if (m_isTransitioning) {
        m_transitionTime += deltaTime;
        if (m_transitionTime >= m_transitionDuration) {
            m_isTransitioning = false;
            m_transitionTime = 0.0f;
            m_stateTime = 0.0f;
        }
    }
}

void SimpleAnimationStateMachine::transitionTo(AnimationState newState, float transitionTime) {
    if (canTransitionTo(newState)) {
        m_previousState = m_currentState;
        m_currentState = newState;
        m_transitionDuration = transitionTime;
        m_isTransitioning = true;
        m_transitionTime = 0.0f;
    }
}

bool SimpleAnimationStateMachine::canTransitionTo(AnimationState targetState) const {
    (void)targetState;
    // Placeholder for transition validity logic
    return true;
}

void SimpleAnimationStateMachine::initializeTransitions() {
    // Define valid state transitions (placeholder)
}

// AnimationSystem implementation
AnimationSystem::AnimationSystem() : m_initialized(false), m_currentBPM(120.0f),
                                     m_beatTimer(0.0f), m_rhythmMultiplier(1.0f),
                                     m_debugVisualization(false) {}

AnimationSystem::~AnimationSystem() {}

bool AnimationSystem::initialize() {
    m_initialized = true;
    return m_initialized;
}

void AnimationSystem::shutdown() {
    m_initialized = false;
    m_animationClips.clear();
    m_animationComponents.clear();
    m_stateMachines.clear();
}

void AnimationSystem::update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    static int frameCount = 0;
    bool debugLog = (frameCount++ < 10);

    if (debugLog) std::cout << "[AnimSys] Update Start. Entities: " << mEntities.size() << std::endl; 
    
    // Iterate over all entities registered with this system (Signature: AnimationComponent)
    int count = 0;
    for (auto const& entity : mEntities) {
        if (debugLog) std::cout << "[AnimSys] Processing Entity " << entity << std::endl;
        auto& animComp = coordinator.GetComponent<AnimationComponent>(entity);
        updateEntityAnimation(entity, animComp, deltaTime);
        count++;
    }
    if (debugLog) std::cout << "[AnimSys] Update End. Processed: " << count << std::endl;
    
    // Update rhythm integration
    m_beatTimer += deltaTime * (m_currentBPM / 60.0f) * m_rhythmMultiplier;
}

void AnimationSystem::updateEntityAnimation(uint32_t entityId, AnimationComponent& component, float deltaTime) {
    (void)entityId;
    if (!component.skeleton) {
        // std::cout << "No skeleton for entity " << entityId << std::endl;
        return;
    }
    
    // Update time
    // Update time
    component.animationTime += deltaTime * component.playbackSpeed;
    
    // Procedural Override
    if (component.useProceduralGeneration) {
        GenerateProceduralPose(entityId, component, deltaTime);
        return; // Skip clip sampling
    }
    
    std::string clipName = "Idle"; // Default
    if (component.currentState == AnimationState::WALKING) clipName = "Walk";
    else if (component.currentState == AnimationState::RUNNING) clipName = "Run";
    else if (component.currentState == AnimationState::ATTACKING) clipName = "Attack";
    
    // Explicit override
    if (!component.currentAnimation.empty()) {
        clipName = component.currentAnimation;
    }
    
    try {
        // Check local component animations first
        AnimationClip* clip = nullptr;
        if (component.animations.count(clipName)) {
            clip = &component.animations[clipName];
        } else {
            clip = getAnimationClip(clipName);
        }
        
        // Fallback if clip mapping failed
        if (!clip && !m_animationClips.empty()) {
            clip = m_animationClips.begin()->second.get();
        }
        
        if (!clip) return;
        
        
        std::cout << "  Using Clip: " << clip->name << " Time: " << component.animationTime << std::endl;

        if (clip->isLooping) {
            component.animationTime = fmod(component.animationTime, clip->duration);
        } else if (component.animationTime > clip->duration) {
            component.animationTime = clip->duration;
            component.isPlaying = false;
        }
        
        size_t boneCount = component.skeleton->bones.size();
        if (component.globalTransforms.size() != boneCount || component.finalBoneMatrices.size() != boneCount) {
            component.globalTransforms.resize(boneCount);
            component.finalBoneMatrices.resize(boneCount);
        }
        
        std::cout << "  Bone Count: " << boneCount << std::endl;
        
        for (size_t i = 0; i < boneCount; ++i) {
            const auto& bone = component.skeleton->bones[i];
            
            const AnimationClip::Channel* channel = nullptr;
            if (!clip->channels.empty()) {
                for (size_t k = 0; k < clip->channels.size(); ++k) {
                    // std::cout << "      Checking Chan " << k << std::endl;
                    const auto& ch = clip->channels[k];
                    if (ch.boneName == bone.name) {
                        channel = &ch;
                        break;
                    }
                }
            }

            glm::mat4 localMat;
            if (channel) {
                BoneTransform localAnim = InterpolateChannel(*channel, component.animationTime);
                localMat = localAnim.ToMatrix();
            } else {
                localMat = glm::mat4(1.0f);
            }
            
            if (bone.parentIndex == -1) {
                component.globalTransforms[i] = localMat;
            } else {
                // Safety check for parent index
                if (bone.parentIndex >= 0 && bone.parentIndex < static_cast<int>(component.globalTransforms.size())) {
                     component.globalTransforms[i] = component.globalTransforms[bone.parentIndex] * localMat;
                } else {
                     component.globalTransforms[i] = localMat;
                }
            }
            
            component.finalBoneMatrices[i] = component.globalTransforms[i] * bone.inverseBindPose;
        }
    } catch (...) {
        std::cout << "[AnimSys] CRASH SUPPRESSED in updateEntityAnimation" << std::endl;
    }
}

void AnimationSystem::AddComponent(uint32_t entityId, const AnimationComponent& component) {
    m_animationComponents[entityId] = component;
}

AnimationComponent* AnimationSystem::GetComponent(uint32_t entityId) {
    auto it = m_animationComponents.find(entityId);
    if (it != m_animationComponents.end()) {
        return &it->second;
    }
    return nullptr;
}


bool AnimationSystem::loadAnimationClip(const std::string& filePath) {
    (void)filePath;
    // Placeholder for loading animation clips
    return true;
}

AnimationClip* AnimationSystem::getAnimationClip(const std::string& name) {
    auto it = m_animationClips.find(name);
    if (it != m_animationClips.end()) {
        return it->second.get();
    }
    return nullptr;
}

void AnimationSystem::registerAnimationClip(std::unique_ptr<AnimationClip> clip) {
    if (clip) {
        std::string name = clip->name;
        m_animationClips[name] = std::move(clip);
    }
}

void AnimationSystem::playAnimation(uint32_t entityId, AnimationState state, float transitionTime) {
    (void)entityId; (void)state; (void)transitionTime;
}

void AnimationSystem::setAnimationSpeed(uint32_t entityId, float speed) {
    (void)entityId; (void)speed;
}

void AnimationSystem::pauseAnimation(uint32_t entityId) {
    (void)entityId;
}

void AnimationSystem::resumeAnimation(uint32_t entityId) {
    (void)entityId;
}

void AnimationSystem::setBlendParameter(uint32_t entityId, const std::string& paramName, float value) {
    (void)entityId; (void)paramName; (void)value;
}

void AnimationSystem::setBlendParameter(uint32_t entityId, const std::string& paramName, const glm::vec2& value) {
    (void)entityId; (void)paramName; (void)value;
}

void AnimationSystem::enableRootMotion(uint32_t entityId, bool enable) {
    (void)entityId; (void)enable;
}

void AnimationSystem::setLayerWeight(uint32_t entityId, int layerIndex, float weight) {
    (void)entityId; (void)layerIndex; (void)weight;
}

void AnimationSystem::crossFadeAnimation(uint32_t entityId, AnimationState targetState, float duration) {
    (void)entityId; (void)targetState; (void)duration;
}

void AnimationSystem::syncToRhythm(uint32_t entityId, float beatTiming, float bpm) {
    (void)entityId; (void)beatTiming; (void)bpm;
}

void AnimationSystem::setRhythmMultiplier(float multiplier) {
    m_rhythmMultiplier = multiplier;
}

void AnimationSystem::enableIK(uint32_t entityId, bool enable) {
    (void)entityId; (void)enable;
}

void AnimationSystem::setIKTarget(uint32_t entityId, const std::string& chainName, const glm::vec3& target) {
    (void)entityId; (void)chainName; (void)target;
}

void AnimationSystem::registerAnimationEvent(const std::string& eventName, std::function<void()> callback) {
    (void)eventName; (void)callback;
}

void AnimationSystem::triggerAnimationEvent(const std::string& eventName) {
    (void)eventName;
}

void AnimationSystem::setDebugVisualization(bool enable) {
    m_debugVisualization = enable;
}

void AnimationSystem::drawAnimationDebug(uint32_t entityId) {
    (void)entityId;
}

BoneTransform AnimationSystem::blendBoneTransforms(const BoneTransform& a, const BoneTransform& b, float weight) {
    (void)b; (void)weight;
    return a;
}

std::unique_ptr<AnimationClip> AnimationSystem::createIdleAnimation() {
    // Placeholder for creating default idle animation
    auto clip = std::make_unique<AnimationClip>();
    if (clip) {
        clip->name = "Idle";
        clip->duration = 1.0f;
        clip->isLooping = true;
    }
    return clip;
}




void AnimationSystem::GenerateProceduralPose(uint32_t entityId, AnimationComponent& component, float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // 1. Get Movement Speed
    float speed = 0.0f;
    using namespace Physics;
    if (coordinator.HasComponent<RigidbodyComponent>(entityId)) {
        auto& rb = coordinator.GetComponent<RigidbodyComponent>(entityId);
        speed = glm::length(glm::vec2(rb.velocity.x, rb.velocity.z));
    }
    
    // 2. Update Phases
    // Locomotion Phase (Run Cycle): Tied to speed
    float runFreq = 1.0f + (speed * 0.1f); // Faster steps at higher speed
    component.proceduralPhase += deltaTime * runFreq * (speed > 0.1f ? 1.0f : 0.0f);
    component.proceduralPhase = fmod(component.proceduralPhase, 1.0f);
    
    // Breathing Phase (Life Cycle): Constant rhythm
    component.breathingPhase += deltaTime * component.breathingFrequency;
    component.breathingPhase = fmod(component.breathingPhase, 1.0f);
    
    if (!component.skeleton) return;
    
    // 3. Resize Matrices if needed
    size_t boneCount = component.skeleton->bones.size();
    if (component.finalBoneMatrices.size() != boneCount) {
        component.finalBoneMatrices.resize(boneCount, glm::mat4(1.0f));
    }
    if (component.globalTransforms.size() != boneCount) {
        component.globalTransforms.resize(boneCount, glm::mat4(1.0f));
    }
    
    // 4. Compute Transforms Hierarchy
    // We pass the component itself to give the compute function access to all phases
    for (int i = 0; i < boneCount; ++i) {
        glm::mat4 localTransform = ComputeProceduralBoneTransform(i, component, speed);
        
        int parentId = component.skeleton->bones[i].parentIndex;
        glm::mat4 globalTransform;
        
        if (parentId == -1) {
            globalTransform = localTransform;
        } else {
            // Assumes bones are sorted parent-first (standard for skeletons)
            if (parentId >= 0 && parentId < i) {
                 globalTransform = component.globalTransforms[parentId] * localTransform;
            } else {
                 globalTransform = localTransform;
            }
        }
        
        component.globalTransforms[i] = globalTransform;
        component.finalBoneMatrices[i] = globalTransform * component.skeleton->bones[i].inverseBindPose;
    }
}

glm::mat4 AnimationSystem::ComputeProceduralBoneTransform(int boneId, const AnimationComponent& anim, float speed) {
    float PI = 3.14159265f;
    float phase = anim.proceduralPhase;
    float walkBlend = glm::min(speed / 4.0f, 1.0f); // 0.0 at rest, 1.0 at full run(4m/s)
    
    // Default Transform
    float angleX = 0.0f;
    float angleY = 0.0f;
    float angleZ = 0.0f;
    glm::vec3 offset(0.0f);
    
    // ---------------------------------------------------------
    // PROCEDURAL ANIMATION LAYERS
    // ---------------------------------------------------------
    
    switch (boneId) {
        // --- CORE BODY ---
        case 0: // HIPS (Root)
        {
            offset = glm::vec3(0, 1.0f, 0); // Base height
            
            // Layer 1: Walk Bounce (Vertical Bob)
            // Bounces twice per cycle (once per step) -> sin(PI*phase)
            float bounceAmp = 0.08f * walkBlend;
            offset.y += bounceAmp * std::abs(sin(2.0f * PI * phase));
            
            // Layer 2: Lateral Sway
            // Sways left/right once per cycle -> sin(2PI*phase)
            float swayAmp = 0.05f * walkBlend;
            offset.x += swayAmp * sin(2.0f * PI * phase);
            
            // Layer 3: Breathing (Vertical heaving in idle)
            // Only active when standing still
            float idleBreath = (1.0f - walkBlend) * 0.02f * sin(2.0f * PI * anim.breathingPhase);
            offset.y += idleBreath;
        }
        break;

        case 1: // SPINE
        {
            offset = glm::vec3(0, 0.2f, 0);
            
            // Layer 1: Counter-Sway (Balance)
            // Rotate opposite to Hips to keep center of mass stable
            float swayRot = 0.05f * walkBlend;
            angleZ = -swayRot * sin(2.0f * PI * phase); 
            
            // Layer 2: Acceleration Lean (Forward Tilt)
            float leanAmt = 0.1f * walkBlend;
            angleX = leanAmt; 
            
            // Layer 3: Breathing
            // Chest expantion/rotation
            angleX += anim.breathingAmplitude * sin(2.0f * PI * anim.breathingPhase);
        }
        break;

        case 2: // NECK
            offset = glm::vec3(0, 0.3f, 0);
            angleX = 0.05f; // Slight natural forward tilt
            break;

        case 3: // HEAD
        {
            offset = glm::vec3(0, 0.15f, 0);
            
            // Layer 1: Stabilization (Look Forward)
            // Determine global spine rotation (approximate) and counter-rotate
            // Spine rotates Z by `angleZ` (from case 1). We counter it.
            // Note: Exact counter-rotation requires global context, this is local approx.
            float spineCounter = -(0.05f * walkBlend) * sin(2.0f * PI * phase); 
            angleZ = -spineCounter * 0.8f; // Dampened counter-rotation
            
            // Layer 2: Constant "Search" or "Focus" (Micro-movements)
            // (Optional: Noise can be added here)
        }
        break;

        // --- LEGS (Locomotion) ---
        case 4: // LeftUpLeg
            offset = glm::vec3(-0.15f, -0.1f, 0);
            // Sine wave drive
            angleX = (0.6f * walkBlend) * sin(2.0f * PI * phase);
            break;
            
        case 7: // RightUpLeg
            offset = glm::vec3(0.15f, -0.1f, 0);
            // Sine wave drive (Opposite Phase)
            angleX = (0.6f * walkBlend) * sin(2.0f * PI * phase + PI);
            break;
            
        case 5: // LeftLeg (Knee)
            offset = glm::vec3(0, -0.4f, 0);
            // Bend logic: Only bend when lifting (forward swing) or planting
            // Simplified: Rectified sine wave offset
            angleX = (0.8f * walkBlend) * std::max(0.0f, sin(2.0f * PI * phase - 0.5f)); 
            break;
            
        case 8: // RightLeg (Knee)
            offset = glm::vec3(0, -0.4f, 0);
            angleX = (0.8f * walkBlend) * std::max(0.0f, sin(2.0f * PI * phase + PI - 0.5f));
            break;
            
        case 6: // LeftFoot
        case 9: // RightFoot
            offset = glm::vec3(0, -0.4f, 0);
            // Slight ankle compensation
            angleX = -0.2f * walkBlend;
            break;

        // --- ARMS (Secondary Action) ---
        case 10: // LeftArm
            offset = glm::vec3(-0.2f, 0.15f, 0);
            // Counter-swing to Right Leg (Phase + PI)
            angleX = (0.4f * walkBlend) * sin(2.0f * PI * phase + PI);
            angleZ = -0.1f; // T-Pose relax
            break;
            
        case 12: // RightArm
            offset = glm::vec3(0.2f, 0.15f, 0);
            // Counter-swing to Left Leg
            angleX = (0.4f * walkBlend) * sin(2.0f * PI * phase);
            angleZ = 0.1f;
            break;
            
        case 11: // LeftForeArm
            offset = glm::vec3(-0.3f, 0, 0);
            angleX = 0.4f * walkBlend; // Always slightly bent when running
            break;
            
        case 13: // RightForeArm
            offset = glm::vec3(0.3f, 0, 0);
            angleX = 0.4f * walkBlend;
            break;
    }

    // ---------------------------------------------------------
    // COMPOSE MATRIX
    // ---------------------------------------------------------
    glm::mat4 m = glm::translate(glm::mat4(1.0f), offset);
    // Rotate in Z-X-Y order (Roll-Pitch-Yaw) or however the skeleton expects.
    // Standard Identity -> Translate -> Rotate
    m = glm::rotate(m, angleY, glm::vec3(0, 1, 0)); // Yaw
    m = glm::rotate(m, angleX, glm::vec3(1, 0, 0)); // Pitch
    m = glm::rotate(m, angleZ, glm::vec3(0, 0, 1)); // Roll
    
    return m;
}

} // namespace Animation
} // namespace CudaGame

