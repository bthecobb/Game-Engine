#pragma once

#include "Core/System.h"
#include "Core/Coordinator.h"
#include "EnemyComponents.h"
#include "PlayerComponents.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/RenderComponents.h"

namespace CudaGame {
namespace Gameplay {

class EnemyAISystem : public Core::System {
public:
    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Bind the player entity explicitly to avoid fragile lookups
    void SetPlayerEntity(Core::Entity player) { m_playerEntity = player; }

private:
    void UpdateAI(Core::Entity entity, EnemyAIComponent& ai, EnemyCombatComponent& combat, 
                  EnemyMovementComponent& movement, Physics::RigidbodyComponent& rigidbody, 
                  Rendering::TransformComponent& transform, float deltaTime);
    
    void UpdatePatrol(Core::Entity entity, EnemyAIComponent& ai, EnemyMovementComponent& movement, 
                     Physics::RigidbodyComponent& rigidbody, Rendering::TransformComponent& transform, float deltaTime);
    
    void UpdateChase(Core::Entity entity, EnemyAIComponent& ai, EnemyMovementComponent& movement,
                    Physics::RigidbodyComponent& rigidbody, Rendering::TransformComponent& transform, 
                    const glm::vec3& playerPos, float deltaTime);
    
    void UpdateAttack(Core::Entity entity, EnemyAIComponent& ai, EnemyCombatComponent& combat,
                     const glm::vec3& playerPos, float deltaTime);
    
    bool CanSeePlayer(const EnemyAIComponent& ai, const Rendering::TransformComponent& enemyTransform, 
                     const glm::vec3& playerPos);
    
    Core::Entity FindPlayerEntity();
    glm::vec3 GetPlayerPosition();
    
    void MoveTowardsTarget(EnemyMovementComponent& movement, Physics::RigidbodyComponent& rigidbody,
                          const glm::vec3& currentPos, const glm::vec3& targetPos, float deltaTime);

    Core::Entity m_playerEntity = 0;
};

} // namespace Gameplay
} // namespace CudaGame
