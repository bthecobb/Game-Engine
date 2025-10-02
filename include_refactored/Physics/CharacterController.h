#pragma once

#include "Physics/PhysicsComponents.h"
#include "Core/ECS_Types.h"
#include <glm/glm.hpp>

namespace CudaGame {
namespace Physics {

// Forward declarations
class CharacterControllerSystem;

// Constants for character controller physics
namespace CharacterControllerConstants {
    constexpr float DEFAULT_JUMP_FORCE = 15.0f;
    constexpr float DEFAULT_DASH_SPEED = 30.0f;
    constexpr float DEFAULT_DASH_COOLDOWN = 1.0f;
    constexpr float DEFAULT_WALL_RUN_SPEED = 10.0f;
    constexpr float DEFAULT_MAX_WALL_RUN_TIME = 2.0f;
    constexpr float DEFAULT_MAX_DASH_TIME = 0.2f;
    constexpr int DEFAULT_MAX_AIR_JUMPS = 1;
    constexpr float DEFAULT_COYOTE_TIME = 0.15f;
    constexpr float DEFAULT_JUMP_BUFFER_TIME = 0.1f;
}

// Character controller component definition
struct CharacterControllerComponent {
    // Core properties
    float height = 2.0f;
    float radius = 0.5f;
    glm::vec3 position{0.0f};
    glm::vec3 moveDirection{0.0f};
    glm::vec3 velocity{0.0f};
    
    // Movement states
    bool isGrounded = false;
    bool isWallRunning = false;
    bool isDashing = false;
    bool isJumping = false;
    bool canDoubleJump = false;

    // Wall-running properties
    glm::vec3 wallNormal{0.0f};
    float wallRunTimer = 0.0f;
    float maxWallRunTime = CharacterControllerConstants::DEFAULT_MAX_WALL_RUN_TIME;
    float wallRunSpeed = CharacterControllerConstants::DEFAULT_WALL_RUN_SPEED;
    
    // Dashing properties
    glm::vec3 dashDirection{0.0f};
    float dashTimer = 0.0f;
    float maxDashTime = CharacterControllerConstants::DEFAULT_MAX_DASH_TIME;
    float dashSpeed = CharacterControllerConstants::DEFAULT_DASH_SPEED;
    float dashCooldown = CharacterControllerConstants::DEFAULT_DASH_COOLDOWN;
    float lastDashTime = 0.0f;

    // Jump properties
    float jumpForce = CharacterControllerConstants::DEFAULT_JUMP_FORCE;
    int airJumps = 0;
    int maxAirJumps = CharacterControllerConstants::DEFAULT_MAX_AIR_JUMPS;
    float lastGroundedTime = 0.0f;  // For coyote time
    float jumpBufferTimer = 0.0f;   // For jump buffering
    
    // Momentum preservation
    glm::vec3 preservedMomentum{0.0f};
    bool shouldPreserveMomentum = false;
};

} // namespace Physics
} // namespace CudaGame

