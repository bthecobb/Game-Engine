#include "Gameplay/EnemyAISystem.h"
#include "Core/Coordinator.h"
#include <iostream>
#include <algorithm>

namespace CudaGame {
namespace Gameplay {

bool EnemyAISystem::Initialize() {
    std::cout << "[EnemyAISystem] Initialized. Managing " << mEntities.size() << " entities." << std::endl;
    return true;
}

void EnemyAISystem::Shutdown() {
    std::cout << "EnemyAISystem shut down" << std::endl;
}

void EnemyAISystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();

    // Iterate over filtered enemy entities only
    for (auto const& entity : mEntities) {
        if (!coordinator.HasComponent<EnemyAIComponent>(entity) ||
            !coordinator.HasComponent<EnemyCombatComponent>(entity) ||
            !coordinator.HasComponent<EnemyMovementComponent>(entity) ||
            !coordinator.HasComponent<Physics::RigidbodyComponent>(entity) ||
            !coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
            continue;
        }

        auto& ai = coordinator.GetComponent<EnemyAIComponent>(entity);
        auto& combat = coordinator.GetComponent<EnemyCombatComponent>(entity);
        auto& movement = coordinator.GetComponent<EnemyMovementComponent>(entity);
        auto& rigidbody = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);

        UpdateAI(entity, ai, combat, movement, rigidbody, transform, deltaTime);
    }
}

void EnemyAISystem::UpdateAI(Core::Entity entity, EnemyAIComponent& ai, EnemyCombatComponent& combat, 
                  EnemyMovementComponent& movement, Physics::RigidbodyComponent& rigidbody, 
                  Rendering::TransformComponent& transform, float deltaTime) {
    
    // If player not bound yet, remain in patrol and skip targeting
    if (m_playerEntity == 0) {
        ai.aiState = AIState::PATROL;
        return;
    }

    glm::vec3 playerPos = GetPlayerPosition();
    
    // Check if can see player for state transitions
    if (ai.aiState == AIState::PATROL && CanSeePlayer(ai, transform, playerPos)) {
        ai.aiState = AIState::CHASE;
        ai.lastKnownPlayerPos = playerPos;
        ai.alertLevel = 1.0f;
    }
    
    switch (ai.aiState) {
        case AIState::PATROL:
            UpdatePatrol(entity, ai, movement, rigidbody, transform, deltaTime);
            break;
        case AIState::CHASE:
            UpdateChase(entity, ai, movement, rigidbody, transform, playerPos, deltaTime);
            break;
        case AIState::ATTACK:
            UpdateAttack(entity, ai, combat, playerPos, deltaTime);
            break;
        case AIState::STUNNED:
            // Handle stunned state
            break;
        case AIState::RETREATING:
            // Handle retreating state
            break;
    }
}

void EnemyAISystem::UpdatePatrol(Core::Entity entity, EnemyAIComponent& ai, EnemyMovementComponent& movement, 
                     Physics::RigidbodyComponent& rigidbody, Rendering::TransformComponent& transform, float deltaTime) {
    
    if (ai.patrolPoints.empty()) {
        // Create simple back and forth patrol if no points set
        ai.patrolPoints.push_back(transform.position + glm::vec3(5.0f, 0.0f, 0.0f));
        ai.patrolPoints.push_back(transform.position - glm::vec3(5.0f, 0.0f, 0.0f));
        ai.currentPatrolIndex = 0;
        return;
    }
    
    const glm::vec3& targetPos = ai.patrolPoints[ai.currentPatrolIndex];
    float distanceToTarget = glm::length(targetPos - transform.position);
    
    if (distanceToTarget < 1.0f) {
        // Reached patrol point, wait or move to next
        if (ai.waitTimer <= 0.0f) {
            ai.currentPatrolIndex = (ai.currentPatrolIndex + 1) % ai.patrolPoints.size();
            ai.waitTimer = ai.waitTime;
        } else {
            ai.waitTimer -= deltaTime;
        }
    } else {
        // Move towards patrol point
        MoveTowardsTarget(movement, rigidbody, transform.position, targetPos, deltaTime);
        ai.waitTimer = 0.0f;
    }
}

void EnemyAISystem::UpdateChase(Core::Entity entity, EnemyAIComponent& ai, EnemyMovementComponent& movement,
                    Physics::RigidbodyComponent& rigidbody, Rendering::TransformComponent& transform, 
                    const glm::vec3& playerPos, float deltaTime) {
    
    float distanceToPlayer = glm::length(playerPos - transform.position);
    
    if (distanceToPlayer <= ai.attackRange) {
        ai.aiState = AIState::ATTACK;
    } else if (distanceToPlayer > ai.detectionRange * 1.5f) {
        // Lost player, return to patrol
        ai.aiState = AIState::PATROL;
        ai.alertLevel = 0.0f;
    } else {
        // Chase the player
        MoveTowardsTarget(movement, rigidbody, transform.position, playerPos, deltaTime);
        ai.lastKnownPlayerPos = playerPos;
    }
}

void EnemyAISystem::UpdateAttack(Core::Entity entity, EnemyAIComponent& ai, EnemyCombatComponent& combat,
                     const glm::vec3& playerPos, float deltaTime) {
    
    auto& coordinator = Core::Coordinator::GetInstance();
    auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
    
    float distanceToPlayer = glm::length(playerPos - transform.position);
    
    if (distanceToPlayer > ai.attackRange * 1.2f) {
        // Player moved out of attack range, return to chase
        ai.aiState = AIState::CHASE;
        return;
    }
    
    // Attack logic
    if (combat.attackTimer <= 0.0f) {
        combat.attackTimer = combat.attackCooldown;
        std::cout << "Enemy attacked the player!" << std::endl;
        // Here you would apply damage to player, trigger animations, etc.
    } else {
        combat.attackTimer -= deltaTime;
    }
}

bool EnemyAISystem::CanSeePlayer(const EnemyAIComponent& ai, const Rendering::TransformComponent& enemyTransform, 
                     const glm::vec3& playerPos) {
    glm::vec3 toPlayer = playerPos - enemyTransform.position;
    float distance = glm::length(toPlayer);
    
    if (distance > ai.detectionRange) return false;
    
    // Check if player is within vision cone
    glm::vec3 toPlayerNorm = glm::normalize(toPlayer);
    float dotProduct = glm::dot(ai.facingDirection, toPlayerNorm);
    float angleToPlayer = glm::degrees(acos(glm::clamp(dotProduct, -1.0f, 1.0f)));
    
    return angleToPlayer <= ai.visionAngle * 0.5f;
}

Core::Entity EnemyAISystem::FindPlayerEntity() {
    if (m_playerEntity != 0) return m_playerEntity;

    auto& coordinator = Core::Coordinator::GetInstance();
    // Fallback (should be avoided once SetPlayerEntity is called)
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<PlayerMovementComponent>(entity)) {
            return entity;
        }
    }
    return 0;
}

glm::vec3 EnemyAISystem::GetPlayerPosition() {
    auto& coordinator = Core::Coordinator::GetInstance();
    Core::Entity playerEntity = (m_playerEntity != 0) ? m_playerEntity : FindPlayerEntity();
    
    if (playerEntity != 0 && coordinator.HasComponent<Rendering::TransformComponent>(playerEntity)) {
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(playerEntity);
        return transform.position;
    }
    
    // Should not be used if player not found; caller should guard using m_playerEntity check
    return glm::vec3(0.0f, 0.0f, 0.0f);
}

void EnemyAISystem::MoveTowardsTarget(EnemyMovementComponent& movement, Physics::RigidbodyComponent& rigidbody,
                          const glm::vec3& currentPos, const glm::vec3& targetPos, float deltaTime) {
    glm::vec3 direction = targetPos - currentPos;
    float distance = glm::length(direction);
    
    if (distance > 0.1f) {
        direction = glm::normalize(direction);
        movement.velocity = direction * movement.speed;
        rigidbody.velocity = movement.velocity;
    } else {
        movement.velocity = glm::vec3(0.0f);
        rigidbody.velocity = glm::vec3(0.0f);
    }
}

} // namespace Gameplay
} // namespace CudaGame
