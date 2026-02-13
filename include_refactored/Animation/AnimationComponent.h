#pragma once

#include <string>
#include <unordered_map>
#include "Animation/AnimationResources.h"
#include <glm/glm.hpp>
#include <vector>
#include <memory>

namespace CudaGame {
namespace Animation {

// Animation component for entities
struct AnimationComponent {
    AnimationState currentState = AnimationState::IDLE;
    AnimationState previousState = AnimationState::IDLE;
    float animationTime = 0.0f;
    float blendWeight = 1.0f;
    float playbackSpeed = 1.0f;
    bool isPlaying = true;
    bool hasTransitioned = false;
    
    // Procedural Animation State
    bool useProceduralGeneration = false;
    float proceduralPhase = 0.0f;
    float breathingPhase = 0.0f;
    float breathingFrequency = 0.5f; // Hz
    float breathingAmplitude = 0.1f; // Radians
    
    // Playback state
    std::string currentAnimation;
    
    // Blend tree parameters
    float movementSpeed = 0.0f;
    float combatIntensity = 0.0f;
    glm::vec2 movementDirection{0.0f};
    
    // Runtime data
    std::shared_ptr<Skeleton> skeleton;
    std::unordered_map<std::string, AnimationClip> animations; // Loaded clips
    std::unordered_map<AnimationState, std::string> stateMap;  // Map State -> Clip Name
    std::vector<glm::mat4> globalTransforms; // Global space transforms
    std::vector<glm::mat4> finalBoneMatrices; // Skinning matrices (Global * InverseBind)
};

} // namespace Animation
} // namespace CudaGame
