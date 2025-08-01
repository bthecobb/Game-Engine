#pragma once

#include "Core/System.h"
#include "Core/Coordinator.h"
#include "PlayerComponents.h"
#include "LevelComponents.h"
#include "Physics/PhysicsComponents.h" 
#include "Rendering/RenderComponents.h"

namespace CudaGame {
namespace Gameplay {

class PlayerMovementSystem : public Core::System {
public:
    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

private:
    void HandleInput(Core::Entity entity, PlayerInputComponent& input, PlayerMovementComponent& movement, float deltaTime);
    void UpdateMovement(Core::Entity entity, PlayerMovementComponent& movement, Physics::RigidbodyComponent& rigidbody, float deltaTime);
    void UpdateWallRunning(Core::Entity entity, PlayerMovementComponent& movement, Physics::RigidbodyComponent& rigidbody, float deltaTime);
    void UpdateDashing(Core::Entity entity, PlayerMovementComponent& movement, Physics::RigidbodyComponent& rigidbody, float deltaTime);
    void ApplyGravity(PlayerMovementComponent& movement, Physics::RigidbodyComponent& rigidbody, float deltaTime);
    void CheckGrounding(Core::Entity entity, PlayerMovementComponent& movement);
    bool CheckWallCollision(Core::Entity entity, PlayerMovementComponent& movement, glm::vec3& wallNormal);
    
    glm::vec2 GetMovementInput(const PlayerInputComponent& input);
    void BuildMomentum(PlayerMovementComponent& movement, glm::vec2 inputDirection, float deltaTime, float targetSpeed);
};

} // namespace Gameplay
} // namespace CudaGame
