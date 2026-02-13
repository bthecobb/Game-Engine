#pragma once

#include "Core/System.h"
#include "Physics/CharacterController.h"
#include "Physics/PhysicsComponents.h"
#include "Gameplay/PlayerComponents.h"
#include "Rendering/RenderComponents.h"
#include <glm/glm.hpp>

namespace CudaGame {

// Forward declarations
namespace Physics {
    class PhysXPhysicsSystem;
}

namespace Rendering {
    class OrbitCamera;
}

namespace Animation {
    struct AnimationComponent;
}

namespace Gameplay {

struct AnimationControllerComponent;

/**
 * Advanced Character Controller System
 * Handles player movement with camera-relative controls, wall-running,
 * dashing, double jumping, coyote time, and jump buffering
 */
class CharacterControllerSystem : public Core::System {
public:
    CharacterControllerSystem();
    ~CharacterControllerSystem() = default;
    
    bool Initialize() override;
    void Update(float deltaTime) override;
    void Shutdown() override;
    
    // Set the camera for camera-relative movement
    void SetCamera(Rendering::OrbitCamera* camera);
    
    // Configuration
    void SetCoyoteTime(float time) { m_coyoteTime = time; }
    void SetJumpBufferTime(float time) { m_jumpBufferTime = time; }
    
private:
    // System references
    Physics::PhysXPhysicsSystem* m_physicsSystem;
    Rendering::OrbitCamera* m_camera;
    
    // Configuration
    float m_coyoteTime;      // Time after leaving ground where jump is still allowed
    float m_jumpBufferTime;  // Time to buffer jump input before landing
    
    // Timer tracking (per entity would be better, but simplified for now)
    float m_lastGroundedTime = 0.0f;
    float m_jumpBufferTimer = 0.0f;
    
    // Core update methods
    void UpdateTimers(Physics::CharacterControllerComponent& controller, float deltaTime);
    void CheckGrounding(Core::Entity entity, 
                       Physics::CharacterControllerComponent& controller,
                       const Rendering::TransformComponent& transform,
                       const Physics::RigidbodyComponent& rigidbody);
    
    // Movement methods
    glm::vec3 GetCameraRelativeMovement(const PlayerInputComponent& input,
                                        const PlayerMovementComponent& movement);
    void ApplyMovement(Physics::CharacterControllerComponent& controller,
                      const PlayerMovementComponent& movement,
                      Physics::RigidbodyComponent& rigidbody,
                      const glm::vec3& moveDirection,
                      float deltaTime);
    
    // Jump methods
    void HandleJump(Physics::CharacterControllerComponent& controller,
                   const PlayerMovementComponent& movement,
                   Physics::RigidbodyComponent& rigidbody,
                   const PlayerInputComponent& input);
    void PerformJump(Physics::CharacterControllerComponent& controller,
                    const PlayerMovementComponent& movement,
                    Physics::RigidbodyComponent& rigidbody);
    void PerformAirJump(Physics::CharacterControllerComponent& controller,
                       const PlayerMovementComponent& movement,
                       Physics::RigidbodyComponent& rigidbody);
    void PerformWallJump(Physics::CharacterControllerComponent& controller,
                        Physics::RigidbodyComponent& rigidbody);
    
    // Wall running methods
    void CheckWallRunning(Core::Entity entity,
                         Physics::CharacterControllerComponent& controller,
                         const Rendering::TransformComponent& transform,
                         Physics::RigidbodyComponent& rigidbody,
                         const PlayerInputComponent& input);
    void ApplyWallRunMovement(Physics::CharacterControllerComponent& controller,
                             Physics::RigidbodyComponent& rigidbody,
                             float deltaTime);
    void ExitWallRun(Physics::CharacterControllerComponent& controller);
    
    // Dash methods
    void HandleDashing(Physics::CharacterControllerComponent& controller,
                      const PlayerMovementComponent& movement,
                      Physics::RigidbodyComponent& rigidbody,
                      const PlayerInputComponent& input,
                      const glm::vec3& moveDirection,
                      float deltaTime);
    
    // Event methods
    void OnLanding(Physics::CharacterControllerComponent& controller);
    
    // Animation integration
    void UpdateAnimationState(const Physics::CharacterControllerComponent& controller,
                             const Physics::RigidbodyComponent& rb,
                             const PlayerMovementComponent& movement,
                             Animation::AnimationComponent& anim);
                             
    // Decoupled Animation integration
    void UpdateAnimationController(const Physics::CharacterControllerComponent& controller,
                                  const Physics::RigidbodyComponent& rb,
                                  const PlayerMovementComponent& movement,
                                  AnimationControllerComponent& animCtrl);
};

} // namespace Gameplay
} // namespace CudaGame
