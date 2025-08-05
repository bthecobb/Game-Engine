#include "AnimationSystem.h"
#include <iostream>
#include <algorithm>

namespace CudaGame {
namespace Animation {

// Add missing AnimationController method implementations
void AnimationController::updateTransition(float deltaTime) {
    m_currentState.transitionProgress += deltaTime / 0.3f; // Default 0.3s transition
    
    if (m_currentState.transitionProgress >= 1.0f) {
        // Transition complete
        m_currentState.currentType = m_currentState.targetType;
        m_currentState.currentTime = 0.0f;
        m_currentState.isTransitioning = false;
        m_currentState.transitionProgress = 0.0f;
    }
    
    m_frameNeedsUpdate = true;
}

AnimationState AnimationController::getAnimationState() const {
    return m_currentState;
}

// Animation Event System Implementation
AnimationEventSystem::AnimationEventSystem() {}

void AnimationEventSystem::addListener(AnimationEventType eventType, const AnimationEventCallback& callback) {
    m_eventListeners[eventType].push_back(callback);
}

void AnimationEventSystem::removeListener(AnimationEventType eventType, const AnimationEventCallback& callback) {
    auto& listeners = m_eventListeners[eventType];
    listeners.erase(
        std::remove_if(listeners.begin(), listeners.end(),
            [&callback](const auto& listener) {
                // This is a simplified comparison - in real implementation you'd need proper function comparison
                return false; // Placeholder - proper implementation would compare function targets
            }),
        listeners.end()
    );
}

void AnimationEventSystem::triggerEvent(AnimationEventType eventType, const AnimationEventData& eventData) {
    auto it = m_eventListeners.find(eventType);
    if (it != m_eventListeners.end()) {
        for (const auto& callback : it->second) {
            callback(eventData);
        }
    }
}

void AnimationEventSystem::update(const AnimationKeyframe& currentFrame, const AnimationKeyframe& previousFrame) {
    // Check for footstep events
    checkFootstepEvents(currentFrame, previousFrame);
    
    // Check for attack events
    checkAttackEvents(currentFrame, previousFrame);
    
    // Check for rhythm events
    checkRhythmEvents(currentFrame, previousFrame);
}

void AnimationEventSystem::checkFootstepEvents(const AnimationKeyframe& current, const AnimationKeyframe& previous) {
    if (current.bodyPartPositions.size() >= 6 && previous.bodyPartPositions.size() >= 6) {
        // Check left foot contact
        float leftFootHeight = current.bodyPartPositions[4].y;
        float prevLeftFootHeight = previous.bodyPartPositions[4].y;
        
        if (leftFootHeight <= -0.75f && prevLeftFootHeight > -0.75f) {
            AnimationEventData eventData;
            eventData.position = current.bodyPartPositions[4];
            eventData.intensity = current.energyLevel;
            eventData.footIndex = 0; // Left foot
            triggerEvent(AnimationEventType::FOOTSTEP, eventData);
        }
        
        // Check right foot contact
        float rightFootHeight = current.bodyPartPositions[5].y;
        float prevRightFootHeight = previous.bodyPartPositions[5].y;
        
        if (rightFootHeight <= -0.75f && prevRightFootHeight > -0.75f) {
            AnimationEventData eventData;
            eventData.position = current.bodyPartPositions[5];
            eventData.intensity = current.energyLevel;
            eventData.footIndex = 1; // Right foot
            triggerEvent(AnimationEventType::FOOTSTEP, eventData);
        }
    }
}

void AnimationEventSystem::checkAttackEvents(const AnimationKeyframe& current, const AnimationKeyframe& previous) {
    if (current.bodyPartPositions.size() >= 4 && previous.bodyPartPositions.size() >= 4) {
        // Check for punch/strike peak
        glm::vec3 rightHandPos = current.bodyPartPositions[3];
        glm::vec3 prevRightHandPos = previous.bodyPartPositions[3];
        
        // If hand moves forward significantly and is at strike position
        if (rightHandPos.z > 0.4f && prevRightHandPos.z <= 0.4f) {
            AnimationEventData eventData;
            eventData.position = rightHandPos;
            eventData.intensity = current.energyLevel;
            eventData.velocity = rightHandPos - prevRightHandPos;
            triggerEvent(AnimationEventType::ATTACK_IMPACT, eventData);
        }
    }
}

void AnimationEventSystem::checkRhythmEvents(const AnimationKeyframe& current, const AnimationKeyframe& previous) {
    // Check for rhythm beat alignment
    if (current.isOnBeat && !previous.isOnBeat) {
        AnimationEventData eventData;
        eventData.intensity = current.beatIntensity;
        eventData.rhythmPhase = current.rhythmPhase;
        triggerEvent(AnimationEventType::RHYTHM_BEAT, eventData);
    }
}

// Animation Blending Utilities
namespace AnimationBlending {

AnimationKeyframe blendFrames(const AnimationKeyframe& a, const AnimationKeyframe& b, float t, BlendMode mode) {
    switch (mode) {
        case BlendMode::LINEAR:
            return linearBlend(a, b, t);
        case BlendMode::SMOOTH:
            return smoothBlend(a, b, t);
        case BlendMode::CUBIC:
            return cubicBlend(a, b, t);
        case BlendMode::ADDITIVE:
            return additiveBlend(a, b, t);
        default:
            return linearBlend(a, b, t);
    }
}

AnimationKeyframe linearBlend(const AnimationKeyframe& a, const AnimationKeyframe& b, float t) {
    AnimationKeyframe result;
    
    // Basic interpolation
    result.timeStamp = glm::mix(a.timeStamp, b.timeStamp, t);
    result.movementSpeed = glm::mix(a.movementSpeed, b.movementSpeed, t);
    result.rhythmIntensity = glm::mix(a.rhythmIntensity, b.rhythmIntensity, t);
    result.breathingAmplitude = glm::mix(a.breathingAmplitude, b.breathingAmplitude, t);
    result.idleSway = glm::mix(a.idleSway, b.idleSway, t);
    result.energyLevel = glm::mix(a.energyLevel, b.energyLevel, t);
    result.rootMotion = glm::mix(a.rootMotion, b.rootMotion, t);
    
    // Interpolate body parts
    size_t partCount = std::min(a.bodyPartPositions.size(), b.bodyPartPositions.size());
    result.bodyPartPositions.resize(partCount);
    result.bodyPartRotations.resize(partCount);
    result.bodyPartScales.resize(partCount);
    
    for (size_t i = 0; i < partCount; ++i) {
        result.bodyPartPositions[i] = glm::mix(a.bodyPartPositions[i], b.bodyPartPositions[i], t);
        result.bodyPartRotations[i] = glm::mix(a.bodyPartRotations[i], b.bodyPartRotations[i], t);
        result.bodyPartScales[i] = glm::mix(a.bodyPartScales[i], b.bodyPartScales[i], t);
    }
    
    return result;
}

AnimationKeyframe smoothBlend(const AnimationKeyframe& a, const AnimationKeyframe& b, float t) {
    // Apply smoothstep function: 3t² - 2t³
    float smoothT = t * t * (3.0f - 2.0f * t);
    return linearBlend(a, b, smoothT);
}

AnimationKeyframe cubicBlend(const AnimationKeyframe& a, const AnimationKeyframe& b, float t) {
    // Apply cubic easing: t³
    float cubicT = t * t * t;
    return linearBlend(a, b, cubicT);
}

AnimationKeyframe additiveBlend(const AnimationKeyframe& a, const AnimationKeyframe& b, float t) {
    AnimationKeyframe result = a;
    
    // Add weighted B to A
    result.movementSpeed += b.movementSpeed * t;
    result.rhythmIntensity += b.rhythmIntensity * t;
    result.breathingAmplitude += b.breathingAmplitude * t;
    result.idleSway += b.idleSway * t;
    result.energyLevel = std::min(1.0f, result.energyLevel + b.energyLevel * t);
    result.rootMotion += b.rootMotion * t;
    
    size_t partCount = std::min(a.bodyPartPositions.size(), b.bodyPartPositions.size());
    for (size_t i = 0; i < partCount && i < result.bodyPartPositions.size(); ++i) {
        result.bodyPartPositions[i] += b.bodyPartPositions[i] * t;
        result.bodyPartRotations[i] += b.bodyPartRotations[i] * t;
        result.bodyPartScales[i] += (b.bodyPartScales[i] - glm::vec3(1.0f)) * t;
    }
    
    return result;
}

float applyEasing(float t, EasingType easing) {
    switch (easing) {
        case EasingType::LINEAR:
            return t;
        case EasingType::EASE_IN:
            return t * t;
        case EasingType::EASE_OUT:
            return 1 - (1 - t) * (1 - t);
        case EasingType::EASE_IN_OUT:
            return t < 0.5f ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
        case EasingType::BOUNCE:
            return applyBounceEasing(t);
        case EasingType::ELASTIC:
            return applyElasticEasing(t);
        default:
            return t;
    }
}

float applyBounceEasing(float t) {
    if (t < 1/2.75f) {
        return 7.5625f * t * t;
    } else if (t < 2/2.75f) {
        t -= 1.5f/2.75f;
        return 7.5625f * t * t + 0.75f;
    } else if (t < 2.5f/2.75f) {
        t -= 2.25f/2.75f;
        return 7.5625f * t * t + 0.9375f;
    } else {
        t -= 2.625f/2.75f;
        return 7.5625f * t * t + 0.984375f;
    }
}

float applyElasticEasing(float t) {
    if (t == 0 || t == 1) return t;
    
    float p = 0.3f;
    float s = p / 4;
    return pow(2, -10 * t) * sin((t - s) * (2 * M_PI) / p) + 1;
}

} // namespace AnimationBlending

// Animation Layer System
AnimationLayer::AnimationLayer(const std::string& name, float weight, LayerBlendMode blendMode)
    : m_name(name), m_weight(weight), m_blendMode(blendMode), m_enabled(true) {}

void AnimationLayer::setWeight(float weight) {
    m_weight = std::clamp(weight, 0.0f, 1.0f);
}

void AnimationLayer::setBlendMode(LayerBlendMode mode) {
    m_blendMode = mode;
}

void AnimationLayer::setEnabled(bool enabled) {
    m_enabled = enabled;
}

void AnimationLayerSystem::addLayer(const std::string& name, float weight, LayerBlendMode blendMode) {
    m_layers.emplace_back(name, weight, blendMode);
}

void AnimationLayerSystem::removeLayer(const std::string& name) {
    m_layers.erase(
        std::remove_if(m_layers.begin(), m_layers.end(),
            [&name](const AnimationLayer& layer) {
                return layer.getName() == name;
            }),
        m_layers.end()
    );
}

AnimationLayer* AnimationLayerSystem::getLayer(const std::string& name) {
    auto it = std::find_if(m_layers.begin(), m_layers.end(),
        [&name](const AnimationLayer& layer) {
            return layer.getName() == name;
        });
    
    return (it != m_layers.end()) ? &(*it) : nullptr;
}

AnimationKeyframe AnimationLayerSystem::blendLayers(const std::vector<AnimationKeyframe>& layerFrames) {
    if (layerFrames.empty() || m_layers.empty()) {
        return AnimationKeyframe{};
    }
    
    if (layerFrames.size() != m_layers.size()) {
        std::cerr << "Warning: Layer count mismatch in AnimationLayerSystem::blendLayers" << std::endl;
        return layerFrames[0];
    }
    
    AnimationKeyframe result = layerFrames[0];
    
    for (size_t i = 1; i < layerFrames.size() && i < m_layers.size(); ++i) {
        if (!m_layers[i].isEnabled()) continue;
        
        float weight = m_layers[i].getWeight();
        
        switch (m_layers[i].getBlendMode()) {
            case LayerBlendMode::OVERRIDE:
                result = AnimationBlending::linearBlend(result, layerFrames[i], weight);
                break;
            case LayerBlendMode::ADDITIVE:
                result = AnimationBlending::additiveBlend(result, layerFrames[i], weight);
                break;
            case LayerBlendMode::MULTIPLY:
                // Multiply blend mode implementation
                for (size_t j = 0; j < result.bodyPartScales.size() && j < layerFrames[i].bodyPartScales.size(); ++j) {
                    result.bodyPartScales[j] *= glm::mix(glm::vec3(1.0f), layerFrames[i].bodyPartScales[j], weight);
                }
                break;
        }
    }
    
    return result;
}

// Animation State Machine Utilities
namespace AnimationStateMachine {

bool canTransition(MovementAnimationType from, MovementAnimationType to) {
    // Define transition rules
    switch (from) {
        case MovementAnimationType::IDLE:
            // Can transition to any movement
            return true;
            
        case MovementAnimationType::WALK_FORWARD:
        case MovementAnimationType::WALK_BACKWARD:
        case MovementAnimationType::WALK_LEFT:
        case MovementAnimationType::WALK_RIGHT:
            // Walk can transition to any other movement or combat
            return true;
            
        case MovementAnimationType::RUN_FORWARD:
        case MovementAnimationType::RUN_BACKWARD:
        case MovementAnimationType::RUN_LEFT:
        case MovementAnimationType::RUN_RIGHT:
            // Run can transition to any movement or combat
            return true;
            
        case MovementAnimationType::SPRINT_FORWARD:
        case MovementAnimationType::SPRINT_BACKWARD:
        case MovementAnimationType::SPRINT_LEFT:
        case MovementAnimationType::SPRINT_RIGHT:
            // Sprint can transition to movement or dash
            return to != MovementAnimationType::JUMP_START; // Can't jump directly from sprint
            
        case MovementAnimationType::JUMP_START:
            // Jump start can only transition to jump apex
            return to == MovementAnimationType::JUMP_APEX;
            
        case MovementAnimationType::JUMP_APEX:
            // Jump apex can only transition to fall
            return to == MovementAnimationType::JUMP_FALL;
            
        case MovementAnimationType::JUMP_FALL:
            // Jump fall can only transition to land
            return to == MovementAnimationType::JUMP_LAND;
            
        case MovementAnimationType::JUMP_LAND:
            // Jump land can transition to any ground movement
            return isGroundMovement(to);
            
        case MovementAnimationType::DASH_HORIZONTAL:
            // Dash can transition to any movement after completion
            return true;
            
        case MovementAnimationType::WALL_RUN_LEFT:
        case MovementAnimationType::WALL_RUN_RIGHT:
            // Wall run can transition to jump or fall
            return to == MovementAnimationType::JUMP_START || to == MovementAnimationType::JUMP_FALL;
            
        case MovementAnimationType::ATTACK_LIGHT_1:
        case MovementAnimationType::ATTACK_LIGHT_2:
        case MovementAnimationType::ATTACK_LIGHT_3:
            // Light attacks can chain or return to movement
            return isAttack(to) || isGroundMovement(to);
            
        case MovementAnimationType::ATTACK_HEAVY:
            // Heavy attack has longer recovery
            return isGroundMovement(to);
            
        default:
            return true;
    }
}

bool isGroundMovement(MovementAnimationType type) {
    return type == MovementAnimationType::IDLE ||
           type == MovementAnimationType::WALK_FORWARD ||
           type == MovementAnimationType::WALK_BACKWARD ||
           type == MovementAnimationType::WALK_LEFT ||
           type == MovementAnimationType::WALK_RIGHT ||
           type == MovementAnimationType::RUN_FORWARD ||
           type == MovementAnimationType::RUN_BACKWARD ||
           type == MovementAnimationType::RUN_LEFT ||
           type == MovementAnimationType::RUN_RIGHT ||
           type == MovementAnimationType::SPRINT_FORWARD ||
           type == MovementAnimationType::SPRINT_BACKWARD ||
           type == MovementAnimationType::SPRINT_LEFT ||
           type == MovementAnimationType::SPRINT_RIGHT;
}

bool isAirMovement(MovementAnimationType type) {
    return type == MovementAnimationType::JUMP_START ||
           type == MovementAnimationType::JUMP_APEX ||
           type == MovementAnimationType::JUMP_FALL ||
           type == MovementAnimationType::JUMP_LAND ||
           type == MovementAnimationType::DASH_HORIZONTAL ||
           type == MovementAnimationType::WALL_RUN_LEFT ||
           type == MovementAnimationType::WALL_RUN_RIGHT;
}

bool isAttack(MovementAnimationType type) {
    return type == MovementAnimationType::ATTACK_LIGHT_1 ||
           type == MovementAnimationType::ATTACK_LIGHT_2 ||
           type == MovementAnimationType::ATTACK_LIGHT_3 ||
           type == MovementAnimationType::ATTACK_HEAVY;
}

float getTransitionTime(MovementAnimationType from, MovementAnimationType to) {
    // Quick transitions within same category
    if (isGroundMovement(from) && isGroundMovement(to)) {
        return 0.2f;
    }
    
    // Slower transitions between categories
    if (isGroundMovement(from) && isAirMovement(to)) {
        return 0.1f; // Quick for responsiveness
    }
    
    if (isAirMovement(from) && isGroundMovement(to)) {
        return 0.3f; // Longer for landing
    }
    
    // Attack transitions
    if (isAttack(from) || isAttack(to)) {
        return 0.15f;
    }
    
    return 0.25f; // Default transition time
}

MovementAnimationType getDefaultTransition(MovementAnimationType from) {
    if (isAirMovement(from)) {
        return MovementAnimationType::JUMP_LAND;
    }
    
    if (isAttack(from)) {
        return MovementAnimationType::IDLE;
    }
    
    return MovementAnimationType::IDLE;
}

} // namespace AnimationStateMachine

} // namespace Animation
} // namespace CudaGame
