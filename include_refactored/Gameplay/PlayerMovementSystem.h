#pragma once

#include "Core/System.h"
#include "Core/Coordinator.h"
#include "PlayerComponents.h"
#include "LevelComponents.h"
#include "Physics/PhysicsComponents.h" 
#include "Rendering/RenderComponents.h"

namespace CudaGame {
namespace Rendering {
    class Camera;  // Forward declaration
}  // namespace Rendering
}

namespace CudaGame {
namespace Gameplay {

class PlayerMovementSystem : public Core::System {
public:
    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    void SetCamera(Rendering::Camera* camera) { m_camera = camera; }

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

private:
    Rendering::Camera* m_camera = nullptr;
};

} // namespace Gameplay
} // namespace CudaGame
