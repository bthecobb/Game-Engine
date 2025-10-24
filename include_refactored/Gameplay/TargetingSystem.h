#pragma once

#include "Core/System.h"
#include "Core/Coordinator.h"
#include "EnemyComponents.h"
#include "PlayerComponents.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/RenderComponents.h"

namespace CudaGame {
namespace Gameplay {

class TargetingSystem : public Core::System {
public:
    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

private:
    void UpdateTargeting(Core::Entity entity, TargetingComponent& targeting, 
                        Rendering::TransformComponent& transform);
    
    void UpdatePlayerTargeting(Core::Entity entity, PlayerCombatComponent& combat,
                              PlayerInputComponent& input, Rendering::TransformComponent& transform);
    
    Core::Entity FindNearestEnemy(const glm::vec3& position, float maxRange);
    Core::Entity FindPlayerEntity();
    
    void DrawTargetingIndicator(Core::Entity targetEntity);
    void DrawCrosshair(const glm::vec3& targetPosition);
    
    // Targeting parameters
    float autoTargetRange = 15.0f;
    float manualTargetRange = 25.0f;
    bool isAutoTargetingEnabled = true;
};

} // namespace Gameplay
} // namespace CudaGame
