#include "Gameplay/TargetingSystem.h"
#include "Core/Coordinator.h"
#include <iostream>

namespace CudaGame {
namespace Gameplay {

bool TargetingSystem::Initialize() {
    std::cout << "[TargetingSystem] Initialized. Managing " << mEntities.size() << " entities." << std::endl;
    return true;
}

void TargetingSystem::Shutdown() {
    std::cout << "TargetingSystem shut down" << std::endl;
}

void TargetingSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Update enemy targeting
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<TargetingComponent>(entity)) {
            auto& targeting = coordinator.GetComponent<TargetingComponent>(entity);
            auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
            
            UpdateTargeting(entity, targeting, transform);
        }
        
        // Update player targeting
        if (coordinator.HasComponent<PlayerCombatComponent>(entity) && 
            coordinator.HasComponent<PlayerInputComponent>(entity)) {
            auto& combat = coordinator.GetComponent<PlayerCombatComponent>(entity);
            auto& input = coordinator.GetComponent<PlayerInputComponent>(entity);
            auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
            
            UpdatePlayerTargeting(entity, combat, input, transform);
        }
    }
}

void TargetingSystem::UpdateTargeting(Core::Entity entity, TargetingComponent& targeting, 
                                     Rendering::TransformComponent& transform) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Find target if we don't have one
    if (!targeting.hasTarget || targeting.targetEntity == 0) {
        targeting.targetEntity = FindPlayerEntity();
        
        if (targeting.targetEntity != 0) {
            targeting.hasTarget = true;
        }
    }
    
    // Update target information if we have a target
    if (targeting.hasTarget && targeting.targetEntity != 0) {
        if (coordinator.HasComponent<Rendering::TransformComponent>(targeting.targetEntity)) {
            auto& targetTransform = coordinator.GetComponent<Rendering::TransformComponent>(targeting.targetEntity);
            
            targeting.targetDirection = targetTransform.position - transform.position;
            targeting.targetDistance = glm::length(targeting.targetDirection);
            
            if (targeting.targetDistance > 0.0f) {
                targeting.targetDirection = glm::normalize(targeting.targetDirection);
            }
            
            // Lose target if too far away
            if (targeting.targetDistance > targeting.loseTargetRange) {
                targeting.hasTarget = false;
                targeting.targetEntity = 0;
            }
        }
    }
}

void TargetingSystem::UpdatePlayerTargeting(Core::Entity entity, PlayerCombatComponent& combat,
                                           PlayerInputComponent& input, Rendering::TransformComponent& transform) {
    // Handle manual targeting with right mouse button
    if (input.mouseButtons[1]) { // Right mouse button
        Core::Entity nearestEnemy = FindNearestEnemy(transform.position, manualTargetRange);
        
        if (nearestEnemy != 0) {
            // Player is now targeting this enemy
            DrawTargetingIndicator(nearestEnemy);
        }
    }
    
    // Auto-targeting when attacking
    if (isAutoTargetingEnabled && input.mouseButtons[0]) { // Left mouse button (attack)
        Core::Entity nearestEnemy = FindNearestEnemy(transform.position, autoTargetRange);
        
        if (nearestEnemy != 0) {
            DrawTargetingIndicator(nearestEnemy);
            
            auto& coordinator = Core::Coordinator::GetInstance();
            if (coordinator.HasComponent<Rendering::TransformComponent>(nearestEnemy)) {
                auto& enemyTransform = coordinator.GetComponent<Rendering::TransformComponent>(nearestEnemy);
                DrawCrosshair(enemyTransform.position);
            }
        }
    }
}

Core::Entity TargetingSystem::FindNearestEnemy(const glm::vec3& position, float maxRange) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    Core::Entity nearestEnemy = 0;
    float nearestDistance = maxRange + 1.0f;
    
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<EnemyAIComponent>(entity) && 
            coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
            
            auto& enemyTransform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
            float distance = glm::length(enemyTransform.position - position);
            
            if (distance <= maxRange && distance < nearestDistance) {
                nearestDistance = distance;
                nearestEnemy = entity;
            }
        }
    }
    
    return nearestEnemy;
}

Core::Entity TargetingSystem::FindPlayerEntity() {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<PlayerMovementComponent>(entity)) {
            return entity;
        }
    }
    
    return 0;
}

void TargetingSystem::DrawTargetingIndicator(Core::Entity targetEntity) {
    // In a real implementation, this would draw UI elements or visual indicators
    // For now, just log the targeting
    std::cout << "Targeting enemy entity: " << targetEntity << std::endl;
}

void TargetingSystem::DrawCrosshair(const glm::vec3& targetPosition) {
    // In a real implementation, this would render a crosshair at the target position
    // For now, just log the crosshair position
    std::cout << "Crosshair at position: (" << targetPosition.x << ", " 
              << targetPosition.y << ", " << targetPosition.z << ")" << std::endl;
}

} // namespace Gameplay
} // namespace CudaGame
