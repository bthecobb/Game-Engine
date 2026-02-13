#include <iostream>
#include <cassert>
#include "Core/Coordinator.h"
#include "Gameplay/AnimationControllerSystem.h"
#include "Physics/PhysicsComponents.h"
#include "Animation/AnimationComponent.h"
#include "Animation/AnimationStateMachine.h"

using namespace CudaGame;
using namespace CudaGame::Gameplay;
using namespace CudaGame::Animation;
using namespace CudaGame::Physics;

// Mock for Test Runner
void RunTest(const std::string& name, std::function<bool()> test) {
    std::cout << "Running " << name << "... ";
    if (test()) {
        std::cout << "[PASS]" << std::endl;
    } else {
        std::cout << "[FAIL]" << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "[AnimationSystemTests] Starting..." << std::endl;
    
    // 1. Test Idle to Running Transition
    RunTest("StateTransition_IdleToMove", []() {
        auto& coordinator = Core::Coordinator::GetInstance();
        coordinator.Initialize();
        
        // Register Components
        coordinator.RegisterComponent<Animation::AnimationComponent>();
        coordinator.RegisterComponent<Gameplay::AnimationControllerComponent>();
        coordinator.RegisterComponent<Physics::RigidbodyComponent>();
        
        // Create System
        auto animSystem = coordinator.RegisterSystem<AnimationControllerSystem>();
        {
            Core::Signature signature;
            signature.set(coordinator.GetComponentType<Animation::AnimationComponent>());
            signature.set(coordinator.GetComponentType<Gameplay::AnimationControllerComponent>());
            coordinator.SetSystemSignature<AnimationControllerSystem>(signature);
        }
        
        // Create Entity
        auto entity = coordinator.CreateEntity();
        
        Animation::AnimationComponent animComp;
        animComp.currentState = AnimationState::IDLE;
        animComp.stateMap[AnimationState::IDLE] = "Idle";
        animComp.stateMap[AnimationState::WALKING] = "Walk";
        animComp.stateMap[AnimationState::RUNNING] = "Run";
        coordinator.AddComponent(entity, animComp);
        
        Gameplay::AnimationControllerComponent ctrlComp;
        coordinator.AddComponent(entity, ctrlComp);
        
        Physics::RigidbodyComponent rb;
        rb.velocity = glm::vec3(0,0,0);
        coordinator.AddComponent(entity, rb);
        
        // Initial Update (Velocity 0) -> Should be IDLE
        animSystem->Update(0.016f);
        
        if (coordinator.GetComponent<Animation::AnimationComponent>(entity).currentState != AnimationState::IDLE) {
            std::cout << "Expected IDLE, got " << (int)coordinator.GetComponent<Animation::AnimationComponent>(entity).currentState << std::endl;
            return false;
        }
        
        // Move (Velocity 7.0) -> Should be RUNNING
        coordinator.GetComponent<Physics::RigidbodyComponent>(entity).velocity = glm::vec3(7.0f, 0, 0);
        
        animSystem->Update(0.016f);
        
        auto& updatedAnim = coordinator.GetComponent<Animation::AnimationComponent>(entity);
        if (updatedAnim.currentState != AnimationState::RUNNING) {
            std::cout << "Expected RUNNING, got " << (int)updatedAnim.currentState << std::endl;
            return false;
        }
        
        // Check Clip Name Update
        if (updatedAnim.currentAnimation != "Run") {
             std::cout << "Expected clip 'Run', got '" << updatedAnim.currentAnimation << "'" << std::endl;
             return false;
        }
        
        return true;
    });

    std::cout << "[AnimationSystemTests] All Tests Passed." << std::endl;
    return 0;
}
