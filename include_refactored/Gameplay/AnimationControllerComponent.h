#pragma once

#include "Gameplay/CharacterResources.h"
#include "Animation/BlendTree.h" // Required for BlendInput
#include <string>
#include <unordered_map>
#include <vector>

#include "Animation/AnimationStateMachine.h"

namespace CudaGame {
namespace Gameplay {

/**
 * [AAA Pattern] Animation Controller Component
 * Acts as the interface between Gameplay Logic (Physics/AI) and the Animation System.
 * Stores parameters (Speed, IsGrounded) that drive the State Machine.
 */
struct AnimationControllerComponent {
    // Reference to the Animation Set (Static Data)
    // The AnimationSystem uses this to resolve State -> Clip
    ResourceID animationSetID;
    
    // Legacy generic parameters (keep for compatibility)
    // Gameplay systems write to these; Animation Graph reads them.
    std::unordered_map<std::string, float> floatParams;
    std::unordered_map<std::string, bool> boolParams;
    // Removed: std::unordered_map<std::string, int> intParams;
    
    // AAA Animation System (Phase 8)
    std::shared_ptr<CudaGame::Animation::AnimationStateMachine> stateMachine;
    
    // Inputs for the current frame (e.g., "Speed", "IsGrounded")
    // These are fed into the State Machine Update()
    std::vector<Animation::BlendInput> nextFrameInputs;
    
    void SetInput(const std::string& name, float value) {
        for (auto& input : nextFrameInputs) {
            if (input.name == name) {
                input.value = value;
                return;
            }
        }
        Animation::BlendInput input;
        input.name = name;
        input.value = value;
        nextFrameInputs.push_back(input);
    }
    
    // Helper to set/get common params easily
    void SetSpeed(float speed) { floatParams["Speed"] = speed; }
    void SetGrounded(bool grounded) { boolParams["IsGrounded"] = grounded; }
    void SetVerticalSpeed(float speed) { floatParams["VerticalSpeed"] = speed; }
    void Trigger(const std::string& triggerName) { boolParams[triggerName] = true; }
};

} // namespace Gameplay
} // namespace CudaGame
