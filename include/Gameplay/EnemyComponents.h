#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>

namespace CudaGame {
namespace Gameplay {

enum class AIState {
    PATROL,
    CHASE, 
    ATTACK,
    STUNNED,
    RETREATING
};

// Enemy AI component
struct EnemyAIComponent {
    AIState aiState = AIState::PATROL;
    glm::vec3 lastKnownPlayerPos{0.0f};
    float alertLevel = 0.0f;
    float retreatTimer = 0.0f;
    
    // Detection parameters
    float detectionRange = 25.0f;
    float attackRange = 3.5f;
    float visionAngle = 60.0f; // degrees
    glm::vec3 facingDirection{1.0f, 0.0f, 0.0f};
    
    // Patrol parameters
    std::vector<glm::vec3> patrolPoints;
    int currentPatrolIndex = 0;
    float patrolSpeed = 5.0f;
    float waitTime = 2.0f;
    float waitTimer = 0.0f;
};

// Enemy combat component
struct EnemyCombatComponent {
    float health = 75.0f;
    float maxHealth = 75.0f;
    float damage = 15.0f;
    float attackCooldown = 1.5f;
    float attackTimer = 0.0f;
    bool isDead = false;
    bool isAggressive = false;
};

// Enemy movement component
struct EnemyMovementComponent {
    glm::vec3 velocity{0.0f};
    float speed = 10.0f;
    float maxSpeed = 15.0f;
    bool isGrounded = true;
    float jumpForce = 15.0f;
    float gravity = 40.0f;
};

// Targeting component for enemies that target player
struct TargetingComponent {
    Core::Entity targetEntity = 0;
    float targetDistance = 0.0f;
    glm::vec3 targetDirection{0.0f};
    bool hasTarget = false;
    float lockOnRange = 30.0f;
    float loseTargetRange = 50.0f;
};

} // namespace Gameplay  
} // namespace CudaGame
