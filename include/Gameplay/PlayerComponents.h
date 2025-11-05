#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>
#include <vector>

namespace CudaGame {
namespace Gameplay {

enum class MovementState {
    IDLE,
    WALKING,
    RUNNING,
    SPRINTING,
    JUMPING,
    WALL_RUNNING,
    DASHING
};

enum class CombatState {
    NEUTRAL,
    ATTACKING,
    BLOCKING,  
    STUNNED
};

enum class WeaponType {
    NONE, SWORD, STAFF, HAMMER
};

// Player movement component
struct PlayerMovementComponent {
    glm::vec3 velocity{0.0f};
    MovementState movementState = MovementState::IDLE;
    
    // Speed parameters - tuned for smooth, responsive control
    float baseSpeed = 5.0f;        // Walking speed (was 10.0 - way too fast)
    float maxSpeed = 10.0f;        // Sprint speed (was 50.0 - insanely fast)
    float acceleration = 12.0f;    // Gradual acceleration (was 30.0 - too instant)
    float deceleration = 15.0f;    // Slightly faster decel for tight control
    float airAcceleration = 8.0f;  // Reduced air control (was 15.0)
    
    // Jump parameters
    float jumpForce = 20.0f;
    float gravity = 40.0f;
    bool isGrounded = false;
    bool canDoubleJump = true;
    
    // Dash parameters
    float dashForce = 80.0f;
    float dashDuration = 0.2f;
    float dashCooldown = 1.0f;
    float dashTimer = 0.0f;
    float dashCooldownTimer = 0.0f;
    bool isDashing = false;
    
    // Wall running
    float wallRunSpeed = 25.0f;
    float wallRunDuration = 2.0f;
    float wallRunTimer = 0.0f;
    bool isWallRunning = false;
    glm::vec3 wallNormal{0.0f};
};

// Player combat component
struct PlayerCombatComponent {
    CombatState combatState = CombatState::NEUTRAL;
    WeaponType currentWeapon = WeaponType::NONE;
    std::vector<WeaponType> inventory;
    
    float health = 100.0f;
    float maxHealth = 100.0f;
    float attackCooldown = 0.5f;
    float attackTimer = 0.0f;
    float attackRange = 4.0f;
    
    bool isBlocking = false;
    bool isParrying = false;
    float parryTimer = 0.0f;
    float parryWindow = 0.3f;
    
    int comboCount = 0;
    float comboWindow = 0.8f;
};

// Player input component
struct PlayerInputComponent {
    bool keys[1024] = {false};
    glm::vec2 mousePos{0.0f};
    glm::vec2 mouseDelta{0.0f};
    bool mouseButtons[8] = {false};
};

// Player rhythm feedback component  
struct PlayerRhythmComponent {
    float beatTimer = 0.0f;
    bool isOnBeat = false;
    float rhythmMultiplier = 1.0f;
    float perfectTimingWindow = 0.1f;
};

// Grappling hook component
struct GrapplingHookComponent {
    bool isActive = false;
    bool isAttached = false;
    glm::vec3 hookPosition{0.0f};
    glm::vec3 targetPosition{0.0f};
    float maxRange = 50.0f;
    float swingSpeed = 20.0f;
    Core::Entity attachedEntity = 0;
};

} // namespace Gameplay
} // namespace CudaGame
