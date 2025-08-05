#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>

namespace CudaGame {
namespace Physics {

// Character controller for advanced movement mechanics
struct CharacterControllerComponent {
    // Movement states
    bool isGrounded = false;
    bool isWallRunning = false;
    bool isDashing = false;
    bool isJumping = false;
    bool canDoubleJump = false;

    // Wall-running properties
    glm::vec3 wallNormal{0.0f};
    float wallRunTimer = 0.0f;
    float maxWallRunTime = 2.0f;
    float wallRunSpeed = 10.0f;
    
    // Dashing properties
    glm::vec3 dashDirection{0.0f};
    float dashTimer = 0.0f;
    float maxDashTime = 0.2f;
    float dashSpeed = 30.0f;
    float dashCooldown = 1.0f;
    float lastDashTime = 0.0f;

    // Jump properties
    float jumpForce = 15.0f;
    int airJumps = 0;
    int maxAirJumps = 1;
    
    // Momentum preservation
    glm::vec3 preservedMomentum{0.0f};
    bool shouldPreserveMomentum = false;
};

} // namespace Physics
} // namespace CudaGame

