#include "Gameplay/PlayerMovementSystem.h"
#include "Core/Coordinator.h"
#include "Rendering/Camera.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>

namespace CudaGame {
namespace Gameplay {

bool PlayerMovementSystem::Initialize() {
    std::cout << "[PlayerMovementSystem] Initialized. Managing " << mEntities.size() << " entities." << std::endl;
    return true;
}

void PlayerMovementSystem::Shutdown() {
    std::cout << "PlayerMovementSystem shut down" << std::endl;
}

void PlayerMovementSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Use the ECS system's entities set - only process entities that match our signature
    for (auto const& entity : mEntities) {
        // All entities in mEntities are guaranteed to have the required components
        auto& input = coordinator.GetComponent<PlayerInputComponent>(entity);
        auto& movement = coordinator.GetComponent<PlayerMovementComponent>(entity);
        auto& rigidbody = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        
        // Store previous position for debugging
        glm::vec3 previousPosition = transform.position;
        
        // Sync vertical velocity from rigidbody (which gets it from PhysX collision response)
        // This preserves PhysX's collision handling for Y while game logic controls X/Z
        movement.velocity.y = rigidbody.velocity.y;
        
        // Only process input and physics if not kinematic (kinematic = scripted/external control)
        if (!rigidbody.isKinematic) {
            HandleInput(entity, input, movement, deltaTime);
            UpdateMovement(entity, movement, rigidbody, deltaTime);
            UpdateWallRunning(entity, movement, rigidbody, deltaTime);
            UpdateDashing(entity, movement, rigidbody, deltaTime);
            ApplyGravity(movement, rigidbody, deltaTime);
            CheckGrounding(entity, movement);
            
            // PhysX handles position updates via velocity integration
            // DO NOT manually update position here or it causes double-update explosion
            
            // Clamp position to prevent runaway (safety check)
            const float MAX_POSITION = 10000.0f;
            if (glm::length(transform.position) > MAX_POSITION) {
                transform.position = glm::normalize(transform.position) * MAX_POSITION;
                movement.velocity = glm::vec3(0.0f);
                rigidbody.velocity = glm::vec3(0.0f);
                std::cout << "[PlayerMovement] WARNING: Position clamped to prevent explosion!" << std::endl;
            }
            
            // Removed per-frame console spam - use F3 debug overlay instead
        } else {
            // Kinematic mode: position is controlled externally (by scripts/cutscenes/debug)
            // Clear velocity to prevent conflicting movement calculations
            movement.velocity = glm::vec3(0.0f);
            rigidbody.velocity = glm::vec3(0.0f);
            
            // Removed per-frame console spam
        }
        
        // Debug overlay available via F3 key - console logging disabled for performance
    }
}

void PlayerMovementSystem::HandleInput(Core::Entity entity, PlayerInputComponent& input, 
                                     PlayerMovementComponent& movement, float deltaTime) {
    glm::vec2 moveInput = GetMovementInput(input);
    
    // Input debug available via F3 overlay - console spam removed
    
    // Jump input - handles both ground jump and double jump
    static bool jumpKeyWasPressed = false;
    bool jumpKeyPressed = input.keys[GLFW_KEY_SPACE];
    
    // Detect new jump press (not held)
    if (jumpKeyPressed && !jumpKeyWasPressed) {
        if (movement.isGrounded) {
            // Ground jump
            movement.velocity.y = movement.jumpForce;
            movement.isGrounded = false;
            movement.canDoubleJump = true;  // Reset double jump when leaving ground
            movement.movementState = MovementState::JUMPING;
            // Jump triggered
        } else if (movement.canDoubleJump) {
            // Double jump
            movement.velocity.y = movement.jumpForce;
            movement.canDoubleJump = false;  // Consume double jump
            // Double jump triggered
        }
    }
    jumpKeyWasPressed = jumpKeyPressed;
    
    // Sprint input
    bool isSprinting = input.keys[GLFW_KEY_LEFT_SHIFT];
    float targetSpeed = isSprinting ? movement.maxSpeed : movement.baseSpeed;
    
    // Dash input
    if (input.keys[GLFW_KEY_LEFT_CONTROL] && movement.dashCooldownTimer <= 0.0f) {
        movement.isDashing = true;
        movement.dashTimer = movement.dashDuration;
        movement.dashCooldownTimer = movement.dashCooldown;
        movement.movementState = MovementState::DASHING;
        // Dash triggered
    }
    
    // Update dash cooldown
    if (movement.dashCooldownTimer > 0.0f) {
        movement.dashCooldownTimer -= deltaTime;
    }
    
    BuildMomentum(movement, moveInput, deltaTime, targetSpeed);
}

void PlayerMovementSystem::UpdateMovement(Core::Entity entity, PlayerMovementComponent& movement, 
                                        Physics::RigidbodyComponent& rigidbody, float deltaTime) {
    // Update movement state based on velocity
    float horizontalSpeed = glm::length(glm::vec2(movement.velocity.x, movement.velocity.z));
    
    if (movement.isGrounded) {
        if (horizontalSpeed < 0.1f) {
            movement.movementState = MovementState::IDLE;
        } else if (horizontalSpeed < movement.baseSpeed) {
            movement.movementState = MovementState::WALKING;
        } else if (horizontalSpeed < movement.baseSpeed * 1.5f) {
            movement.movementState = MovementState::RUNNING;
        } else {
            movement.movementState = MovementState::SPRINTING;
        }
    }
    
    // Apply velocity to rigidbody
    // Only set horizontal velocity - let PhysX handle vertical velocity from collisions
    // When grounded, don't override if velocity is zero to avoid feedback loop with collision response
    if (!movement.isGrounded || glm::length(glm::vec2(movement.velocity.x, movement.velocity.z)) > 0.01f) {
        rigidbody.velocity.x = movement.velocity.x;
        rigidbody.velocity.z = movement.velocity.z;
    }
    // Only set vertical velocity when actively changing it (jumping, etc)
    if (!movement.isGrounded) {
        rigidbody.velocity.y = movement.velocity.y;
    }
}

void PlayerMovementSystem::UpdateWallRunning(Core::Entity entity, PlayerMovementComponent& movement, 
                                           Physics::RigidbodyComponent& rigidbody, float deltaTime) {
    if (movement.isWallRunning) {
        movement.wallRunTimer += deltaTime;
        
        if (movement.wallRunTimer >= movement.wallRunDuration) {
            movement.isWallRunning = false;
            movement.wallRunTimer = 0.0f;
            movement.movementState = MovementState::JUMPING;
        } else {
            // Apply wall run velocity
            glm::vec3 wallRunDirection = glm::cross(movement.wallNormal, glm::vec3(0.0f, 1.0f, 0.0f));
            movement.velocity = wallRunDirection * movement.wallRunSpeed;
            movement.velocity.y = 0.0f; // No gravity while wall running
        }
    } else {
        // Check for wall collision
        glm::vec3 wallNormal;
        if (!movement.isGrounded && CheckWallCollision(entity, movement, wallNormal)) {
            movement.isWallRunning = true;
            movement.wallNormal = wallNormal;
            movement.wallRunTimer = 0.0f;
            movement.movementState = MovementState::WALL_RUNNING;
        }
    }
}

void PlayerMovementSystem::UpdateDashing(Core::Entity entity, PlayerMovementComponent& movement, 
                                        Physics::RigidbodyComponent& rigidbody, float deltaTime) {
    if (movement.isDashing) {
        movement.dashTimer -= deltaTime;
        
        if (movement.dashTimer <= 0.0f) {
            movement.isDashing = false;
        } else {
            // Apply dash velocity (maintain current direction but boost speed)
            glm::vec3 dashDirection = glm::normalize(glm::vec3(movement.velocity.x, 0.0f, movement.velocity.z));
            if (glm::length(dashDirection) > 0.1f) {
                movement.velocity = dashDirection * movement.dashForce;
                movement.velocity.y = 0.0f; // No vertical component during dash
            }
        }
    }
}

void PlayerMovementSystem::ApplyGravity(PlayerMovementComponent& movement, Physics::RigidbodyComponent& rigidbody, float deltaTime) {
    // Only apply gravity when in air - let PhysX handle ground collision
    if (!movement.isGrounded && !movement.isWallRunning) {
        movement.velocity.y -= movement.gravity * deltaTime;
    }
    // When grounded, don't modify Y velocity - let PhysX collision response handle it
    // Rigidbody.velocity.y will be synced from PhysX each frame
}

void PlayerMovementSystem::CheckGrounding(Core::Entity entity, PlayerMovementComponent& movement) {
    // Simplified grounding check - in a real implementation this would raycast down
    auto& coordinator = Core::Coordinator::GetInstance();
    auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
    
    // Ground calculations:
    // - Ground center: y = -1.0, scale.y = 1.0, halfExtent = 0.5
    // - Ground top surface: y = -1.0 + 0.5 = -0.5
    // - Player collider: size = (0.8, 1.8, 0.8), halfExtent.y = 0.9
    // - Player center when standing on ground: y = -0.5 + 0.9 = 0.4
    const float GROUND_LEVEL = 0.4f;  // Player center position when standing on ground
    const float GROUND_THRESHOLD = 0.3f;  // Tolerance for grounding
    
    // Check if player is on or near ground and falling/stopped
    // Let PhysX handle ALL collision response including velocity - we just track grounded state
    if (transform.position.y <= GROUND_LEVEL + GROUND_THRESHOLD && movement.velocity.y <= 0.1f) {
        movement.isGrounded = true;
        // DON'T manually set velocity - PhysX collision will handle stopping
        // DON'T manually set position - PhysX collision will handle positioning
    } else {
        movement.isGrounded = false;
    }
}

bool PlayerMovementSystem::CheckWallCollision(Core::Entity entity, PlayerMovementComponent& movement, glm::vec3& wallNormal) {
    // Simplified wall collision check - in a real implementation this would use proper collision detection
    // For now, return false to disable wall running until proper collision system is in place
    return false;
}

glm::vec2 PlayerMovementSystem::GetMovementInput(const PlayerInputComponent& input) {
    glm::vec2 moveInput(0.0f);
    
    // WASD movement
    if (input.keys[GLFW_KEY_W]) moveInput.y += 1.0f;
    if (input.keys[GLFW_KEY_S]) moveInput.y -= 1.0f;
    if (input.keys[GLFW_KEY_A]) moveInput.x -= 1.0f;  
    if (input.keys[GLFW_KEY_D]) moveInput.x += 1.0f;
    
    // Normalize diagonal movement
    if (glm::length(moveInput) > 1.0f) {
        moveInput = glm::normalize(moveInput);
    }
    
    return moveInput;
}

void PlayerMovementSystem::BuildMomentum(PlayerMovementComponent& movement, glm::vec2 inputDirection, float deltaTime, float targetSpeed) {
    static int buildMomentumDebugCounter = 0;
    
    if (glm::length(inputDirection) > 0.1f) {
        // Apply acceleration
        float accel = movement.isGrounded ? movement.acceleration : movement.airAcceleration;
        
        // Convert input to camera-relative movement
        glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);  // Default forward
        glm::vec3 right = glm::vec3(1.0f, 0.0f, 0.0f);     // Default right
        
        if (m_camera) {
            // Get camera forward and right vectors (projected onto XZ plane)
            forward = m_camera->GetForward();
            forward.y = 0.0f;
            forward = glm::normalize(forward);
            
            right = m_camera->GetRight();
            right.y = 0.0f;
            right = glm::normalize(right);
        } else {
            if (buildMomentumDebugCounter++ % 60 == 0) {
                std::cout << "[WARNING] Camera not set for PlayerMovementSystem!" << std::endl;
            }
        }
        
        // Calculate movement direction relative to camera
        glm::vec3 moveDirection = forward * inputDirection.y + right * inputDirection.x;
        glm::vec3 targetVelocity = moveDirection * targetSpeed;
        
        // Debug log every 60 frames
        if (buildMomentumDebugCounter++ % 60 == 0) {
            std::cout << "[DEBUG BuildMomentum] Input: (" << inputDirection.x << ", " << inputDirection.y << ")" << std::endl;
            std::cout << "[DEBUG BuildMomentum] Camera Forward: (" << forward.x << ", " << forward.y << ", " << forward.z << ")" << std::endl;
            std::cout << "[DEBUG BuildMomentum] Camera Right: (" << right.x << ", " << right.y << ", " << right.z << ")" << std::endl;
            std::cout << "[DEBUG BuildMomentum] Move Direction: (" << moveDirection.x << ", " << moveDirection.y << ", " << moveDirection.z << ")" << std::endl;
            std::cout << "[DEBUG BuildMomentum] Target Speed: " << targetSpeed << ", Accel: " << accel << std::endl;
            std::cout << "[DEBUG BuildMomentum] Target Velocity: (" << targetVelocity.x << ", " << targetVelocity.y << ", " << targetVelocity.z << ")" << std::endl;
        }
        
        glm::vec3 currentHorizontal = glm::vec3(movement.velocity.x, 0.0f, movement.velocity.z);
        glm::vec3 velocityDiff = targetVelocity - currentHorizontal;
        
        if (glm::length(velocityDiff) > 0.1f) {
            glm::vec3 acceleration = glm::normalize(velocityDiff) * accel * deltaTime;
            if (glm::length(acceleration) > glm::length(velocityDiff)) {
                acceleration = velocityDiff;
            }
            
            movement.velocity.x += acceleration.x;
            movement.velocity.z += acceleration.z;
            
            if (buildMomentumDebugCounter % 60 == 0) {
                std::cout << "[DEBUG BuildMomentum] Applied acceleration: (" << acceleration.x << ", " << acceleration.z << ")" << std::endl;
                std::cout << "[DEBUG BuildMomentum] New velocity: (" << movement.velocity.x << ", " << movement.velocity.y << ", " << movement.velocity.z << ")" << std::endl;
            }
        }
    } else {
        // Apply deceleration
        glm::vec3 currentHorizontal = glm::vec3(movement.velocity.x, 0.0f, movement.velocity.z);
        if (glm::length(currentHorizontal) > 0.1f) {
            glm::vec3 deceleration = glm::normalize(currentHorizontal) * movement.deceleration * deltaTime;
            if (glm::length(deceleration) > glm::length(currentHorizontal)) {
                movement.velocity.x = 0.0f;
                movement.velocity.z = 0.0f;
            } else {
                movement.velocity.x -= deceleration.x;
                movement.velocity.z -= deceleration.z;
            }
        }
    }
}

} // namespace Gameplay
} // namespace CudaGame
