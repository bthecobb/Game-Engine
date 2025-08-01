#include "Gameplay/PlayerMovementSystem.h"
#include "Core/Coordinator.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>

namespace CudaGame {
namespace Gameplay {

bool PlayerMovementSystem::Initialize() {
    std::cout << "PlayerMovementSystem initialized" << std::endl;
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
        
        HandleInput(entity, input, movement, deltaTime);
        UpdateMovement(entity, movement, rigidbody, deltaTime);
        UpdateWallRunning(entity, movement, rigidbody, deltaTime);
        UpdateDashing(entity, movement, rigidbody, deltaTime);
        ApplyGravity(movement, rigidbody, deltaTime);
        CheckGrounding(entity, movement);
        
        // Apply the movement velocity to transform position
        transform.position += movement.velocity * deltaTime;
        
        // Debug output to verify movement is working
        static int frameCounter = 0;
        if (frameCounter++ % 60 == 0) { // Every second at 60 FPS
            std::cout << "[PlayerMovement] Entity " << entity 
                      << " Pos: (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")"
                      << " Vel: (" << movement.velocity.x << ", " << movement.velocity.y << ", " << movement.velocity.z << ")"
                      << " Grounded: " << movement.isGrounded << std::endl;
        }
    }
}

void PlayerMovementSystem::HandleInput(Core::Entity entity, PlayerInputComponent& input, 
                                     PlayerMovementComponent& movement, float deltaTime) {
    glm::vec2 moveInput = GetMovementInput(input);
    
    // Jump input
    if (input.keys[GLFW_KEY_SPACE] && movement.isGrounded) {
        movement.velocity.y = movement.jumpForce;
        movement.isGrounded = false;
        movement.movementState = MovementState::JUMPING;
        std::cout << "[PlayerMovement] Jump!" << std::endl;
    }
    
    // Sprint input
    bool isSprinting = input.keys[GLFW_KEY_LEFT_SHIFT];
    float targetSpeed = isSprinting ? movement.maxSpeed : movement.baseSpeed;
    
    // Dash input
    if (input.keys[GLFW_KEY_LEFT_CONTROL] && movement.dashCooldownTimer <= 0.0f) {
        movement.isDashing = true;
        movement.dashTimer = movement.dashDuration;
        movement.dashCooldownTimer = movement.dashCooldown;
        movement.movementState = MovementState::DASHING;
        std::cout << "[PlayerMovement] Dash!" << std::endl;
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
    rigidbody.velocity = movement.velocity;
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
    if (!movement.isGrounded && !movement.isWallRunning) {
        movement.velocity.y -= movement.gravity * deltaTime;
    }
}

void PlayerMovementSystem::CheckGrounding(Core::Entity entity, PlayerMovementComponent& movement) {
    // Simplified grounding check - in a real implementation this would raycast down
    auto& coordinator = Core::Coordinator::GetInstance();
    auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
    
    // Check if player is close to ground level (simplified)
    if (transform.position.y <= 0.1f && movement.velocity.y <= 0.0f) {
        movement.isGrounded = true;
        movement.velocity.y = 0.0f;
        transform.position.y = 0.0f;
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
    if (glm::length(inputDirection) > 0.1f) {
        // Apply acceleration
        float accel = movement.isGrounded ? movement.acceleration : movement.airAcceleration;
        glm::vec3 targetVelocity = glm::vec3(inputDirection.x, 0.0f, inputDirection.y) * targetSpeed;
        
        glm::vec3 currentHorizontal = glm::vec3(movement.velocity.x, 0.0f, movement.velocity.z);
        glm::vec3 velocityDiff = targetVelocity - currentHorizontal;
        
        if (glm::length(velocityDiff) > 0.1f) {
            glm::vec3 acceleration = glm::normalize(velocityDiff) * accel * deltaTime;
            if (glm::length(acceleration) > glm::length(velocityDiff)) {
                acceleration = velocityDiff;
            }
            
            movement.velocity.x += acceleration.x;
            movement.velocity.z += acceleration.z;
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
