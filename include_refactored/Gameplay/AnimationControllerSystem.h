#pragma once

#include "Core/System.h"
#include "Gameplay/AnimationControllerComponent.h"
#include "Animation/AnimationSystem.h"

namespace CudaGame {
namespace Gameplay {

/**
 * [AAA Pattern] Animation Controller System
 * Evaluates animation logic/state machine based on parameters set by gameplay systems.
 * Writes the resulting state to the AnimationComponent for the AnimationSystem to render.
 */
class AnimationControllerSystem : public Core::System {
public:
    AnimationControllerSystem();
    ~AnimationControllerSystem() = default;
    
    bool Initialize() override;
    void Update(float deltaTime) override;
    void Shutdown() override;
    
private:
    // Internal Logic
    void UpdateStateMachine(Core::Entity entity, 
                           AnimationControllerComponent& controller, 
                           Animation::AnimationComponent& anim,
                           float deltaTime);
                           
    // Helper to resolve clip IDs based on state (mock implementation for now)
    // In full version, this would query the AnimationSet resource.
};

} // namespace Gameplay
} // namespace CudaGame
