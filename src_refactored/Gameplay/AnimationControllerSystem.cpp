#include "Gameplay/AnimationControllerSystem.h"
#include "Core/Coordinator.h"
#include "Animation/IKSolver.h"
#include <iostream>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include "Animation/BoneTransform.h"
#include "Animation/AnimationStateMachine.h"
#include "Physics/PhysicsComponents.h"



namespace CudaGame {
namespace Gameplay {



AnimationControllerSystem::AnimationControllerSystem() {
}

bool AnimationControllerSystem::Initialize() {
    std::cout << "[AnimationControllerSystem] Initialized" << std::endl;
    return true;
}

void AnimationControllerSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    for (auto const& entity : mEntities) {
        if (!coordinator.HasComponent<AnimationControllerComponent>(entity) || 
            !coordinator.HasComponent<Animation::AnimationComponent>(entity)) {
            continue;
        }
        
        auto& controller = coordinator.GetComponent<AnimationControllerComponent>(entity);
        auto& anim = coordinator.GetComponent<Animation::AnimationComponent>(entity);
        
        UpdateStateMachine(entity, controller, anim, deltaTime);
    }
}

void AnimationControllerSystem::UpdateStateMachine(Core::Entity entity, 
                                                  AnimationControllerComponent& controller, 
                                                  Animation::AnimationComponent& anim,
                                                  float deltaTime) {
    using namespace Animation;
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // --- Phase 1: Input Gathering ---
    float speed = 0.0f;
    float vSpeed = 0.0f;
    bool isGrounded = true; // Assume true for now if no physics
    
    if (coordinator.HasComponent<Physics::RigidbodyComponent>(entity)) {
        auto& rb = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
        
        // Horizontal Speed (XZ)
        glm::vec3 velocity = rb.velocity;
        speed = glm::length(glm::vec2(velocity.x, velocity.z));
        vSpeed = velocity.y;
        
        // Update Controller Params for Debug/Graph
        controller.floatParams["Speed"] = speed;
        controller.floatParams["VerticalSpeed"] = vSpeed;
    }
    
    // Read controller overrides (e.g. from Input System)
    bool isWallRunning = controller.boolParams["IsWallRunning"];
    bool isDashing = controller.boolParams["IsDashing"];
    
    // --- Phase 2: State Logic ---
    AnimationState targetState = AnimationState::IDLE;
    
    if (isWallRunning) {
        targetState = AnimationState::WALL_RUNNING;
    } else if (isDashing) {
        targetState = AnimationState::RUNNING; 
    } else {
        // Ground movement logic
        if (speed > 0.1f) {
            if (speed > 6.0f) {
                targetState = AnimationState::RUNNING;
            } else {
                targetState = AnimationState::WALKING;
            }
        } else {
            targetState = AnimationState::IDLE;
        }
    }
    
    // --- Phase 3: Transition Handling ---
    if (targetState != anim.currentState) {
        // State Change Detected
        anim.previousState = anim.currentState;
        anim.currentState = targetState;
        anim.animationTime = 0.0f;
        anim.hasTransitioned = true;
        
        // Resolve Clip Name
        // LEGACY: This logic is now handled by the AnimationStateMachine.
        // We only keep it for entities that lack a State Machine (fallback).
        if (!controller.stateMachine) {
            std::string clipName = "";
            if (anim.stateMap.count(targetState)) {
                clipName = anim.stateMap[targetState];
            }
            
            if (!clipName.empty()) {
                anim.currentAnimation = clipName;
                anim.isPlaying = true; // Restart playback
                std::cout << "[Anim] Entity " << entity << " Transition: " << (int)anim.previousState << " -> " << (int)anim.currentState << " (" << clipName << ")" << std::endl;
            }
        }
    }
    
    // Pass blend params (for Blend Trees)
    anim.movementSpeed = speed;
    
    // --- PHASE 8: Graph Evaluation ---
    if (controller.stateMachine) {
        // 1. Sync legacy params to Graph Inputs (Migration helper)
        controller.SetInput("Speed", speed);
        controller.SetInput("VerticalSpeed", vSpeed);
        // ... others
        
        // 2. Update Graph - REMOVED
        // The AnimationSystem is responsible for updating the state machine.
        // We only populate inputs here.
        /*
        if (anim.skeleton) {
            controller.stateMachine->Update(deltaTime, controller.nextFrameInputs, *anim.skeleton);
        }
        */
        controller.nextFrameInputs.clear(); // Consume inputs
    }
}

void AnimationControllerSystem::Shutdown() {
    std::cout << "[AnimationControllerSystem] Shut down" << std::endl;
}

} // namespace Gameplay
} // namespace CudaGame
