#include "Gameplay/PlayerMovementSystem.h"
#include "Core/Coordinator.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>

namespace CudaGame {
namespace Gameplay {

bool PlayerMovementSystem::Initialize() {
    std::cout << "[PlayerMovementSystem] Initialized with PhysX integration" << std::endl;
    return true;
}

void PlayerMovementSystem::Shutdown() {
    std::cout << "[PlayerMovementSystem] Shut down" << std::endl;
}

void PlayerMovementSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    for (auto const& entity : mEntities) {
        auto& input = coordinator.GetComponent<PlayerInputComponent>(entity);
        auto& movement = coordinator.GetComponent<PlayerMovementComponent>(entity);
        auto& rigidbody = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        
        // IMPORTANT: For non-kinematic bodies, PhysX controls the position
        // We should only modify velocity, not position directly
        if (!rigidbody.isKinematic) {
            HandleInput(entity, input, movement, deltaTime);
            UpdateMovement(entity, movement, rigidbody, deltaTime);
            UpdateWallRunning(entity, movement, rigidbody, deltaTime);
            UpdateDashing(entity, movement, rigidbody, deltaTime);
            
            // Check grounding based on current PhysX position
            CheckGrounding(entity, movement, transform);
            
            // Apply forces through the rigidbody, not direct position manipulation
            ApplyMovementForces(movement, rigidbody, deltaTime);
            
            // Debug output every 60 frames
            static int frameCounter = 0;
            if (entity == 0 && frameCounter++ % 60 == 0) {
                std::cout << "\n=== PLAYER PHYSICS DEBUG ===" << std::endl;
                std::cout << "Position: (" << transform.position.x << ", " 
                          << transform.position.y << ", " << transform.position.z << ")" << std::endl;
                std::cout << "Velocity: (" << rigidbody.velocity.x << ", " 
                          << rigidbody.velocity.y << ", " << rigidbody.velocity.z << ")" << std::endl;
                std::cout << "Grounded: " << (movement.isGrounded ? "Yes" : "No") << std::endl;
                std::cout << "Ground Distance: " << movement.groundDistance << std::endl;
                std::cout << "===========================\n" << std::endl;
            }
        } else {
            // Kinematic mode: clear velocities
            movement.velocity = glm::vec3(0.0f);
            rigidbody.velocity = glm::vec3(0.0f);
        }
    }
}

void PlayerMovementSystem::HandleInput(Core::Entity entity, PlayerInputComponent& input, 
                                     PlayerMovementComponent& movement, float deltaTime) {
    glm::vec2 moveInput = GetMovementInput(input);
    
    // Jump input - only if grounded
    if (input.keys[GLFW_KEY_SPACE] && movement.isGrounded && !movement.jumpQueued) {
        movement.jumpQueued = true;
        std::cout << "[PlayerMovement] Jump queued!" << std::endl;
    }
    
    // Sprint input
    movement.isSprinting = input.keys[GLFW_KEY_LEFT_SHIFT];
    
    // Dash input
    if (input.keys[GLFW_KEY_LEFT_CONTROL] && movement.dashCooldownTimer <= 0.0f) {
        movement.isDashing = true;
        movement.dashTimer = movement.dashDuration;
        movement.dashCooldownTimer = movement.dashCooldown;
        movement.movementState = MovementState::DASHING;
    }
    
    // Update dash cooldown
    if (movement.dashCooldownTimer > 0.0f) {
        movement.dashCooldownTimer -= deltaTime;
    }
    
    // Store movement input for force application
    movement.inputDirection = moveInput;
}

void PlayerMovementSystem::UpdateMovement(Core::Entity entity, PlayerMovementComponent& movement, 
                                        Physics::RigidbodyComponent& rigidbody, float deltaTime) {
    // Update movement state based on velocity
    float horizontalSpeed = glm::length(glm::vec2(rigidbody.velocity.x, rigidbody.velocity.z));
    
    if (movement.isGrounded) {
        if (horizontalSpeed < 0.1f) {
            movement.movementState = MovementState::IDLE;
        } else if (horizontalSpeed < movement.baseSpeed * 0.5f) {
            movement.movementState = MovementState::WALKING;
        } else if (horizontalSpeed < movement.baseSpeed) {
            movement.movementState = MovementState::RUNNING;
        } else {
            movement.movementState = MovementState::SPRINTING;
        }
    } else if (rigidbody.velocity.y > 0) {
        movement.movementState = MovementState::JUMPING;
    } else {
        movement.movementState = MovementState::FALLING;
    }
}

void PlayerMovementSystem::UpdateWallRunning(Core::Entity entity, PlayerMovementComponent& movement, 
                                           Physics::RigidbodyComponent& rigidbody, float deltaTime) {
    if (movement.isWallRunning) {
        movement.wallRunTimer += deltaTime;
        
        if (movement.wallRunTimer >= movement.wallRunDuration) {
            movement.isWallRunning = false;
            movement.wallRunTimer = 0.0f;
            movement.movementState = MovementState::FALLING;
        }
    }
}

void PlayerMovementSystem::UpdateDashing(Core::Entity entity, PlayerMovementComponent& movement, 
                                        Physics::RigidbodyComponent& rigidbody, float deltaTime) {
    if (movement.isDashing) {
        movement.dashTimer -= deltaTime;
        
        if (movement.dashTimer <= 0.0f) {
            movement.isDashing = false;
            movement.movementState = movement.isGrounded ? MovementState::IDLE : MovementState::FALLING;
        }
    }
}

void PlayerMovementSystem::CheckGrounding(Core::Entity entity, PlayerMovementComponent& movement,
                                         const Rendering::TransformComponent& transform) {
    // Ground check settings
    const float PLAYER_HEIGHT = 2.0f; // Player is 2 units tall
    const float PLAYER_HALF_HEIGHT = PLAYER_HEIGHT * 0.5f;
    const float GROUND_CHECK_THRESHOLD = 0.2f; // How close to ground to be considered grounded
    const float GROUND_LEVEL = 0.0f; // Top of ground collider
    
    // Calculate distance from player's feet to ground
    float playerFeetY = transform.position.y - PLAYER_HALF_HEIGHT;
    movement.groundDistance = playerFeetY - GROUND_LEVEL;
    
    // Check if grounded
    bool wasGrounded = movement.isGrounded;
    movement.isGrounded = (movement.groundDistance <= GROUND_CHECK_THRESHOLD) && 
                         (movement.velocity.y <= 0.1f); // Not moving up
    
    // Landing detection
    if (!wasGrounded && movement.isGrounded) {
        std::cout << "[PlayerMovement] Landed! Ground distance: " << movement.groundDistance << std::endl;
        movement.jumpQueued = false; // Clear any queued jumps
    }
}

void PlayerMovementSystem::ApplyMovementForces(PlayerMovementComponent& movement, 
                                              Physics::RigidbodyComponent& rigidbody, 
                                              float deltaTime) {
    // Calculate target speed based on state
    float targetSpeed = movement.isSprinting ? movement.maxSpeed : movement.baseSpeed;
    if (movement.isDashing) {
        targetSpeed = movement.dashForce;
    }
    
    // Movement forces
    if (glm::length(movement.inputDirection) > 0.1f) {
        glm::vec3 targetVelocity(movement.inputDirection.x * targetSpeed, 0.0f, 
                                movement.inputDirection.y * targetSpeed);
        glm::vec3 currentHorizontal(rigidbody.velocity.x, 0.0f, rigidbody.velocity.z);
        glm::vec3 velocityDiff = targetVelocity - currentHorizontal;
        
        // Apply acceleration force
        float accel = movement.isGrounded ? movement.acceleration : movement.airAcceleration;
        glm::vec3 force = velocityDiff * rigidbody.mass * accel;
        
        // Limit force to prevent instant acceleration
        float maxForce = rigidbody.mass * movement.acceleration * 2.0f;
        if (glm::length(force) > maxForce) {
            force = glm::normalize(force) * maxForce;
        }
        
        rigidbody.addForce(force);
    } else if (movement.isGrounded) {
        // Apply friction when no input and grounded
        glm::vec3 frictionForce = -glm::vec3(rigidbody.velocity.x, 0.0f, rigidbody.velocity.z) 
                                 * rigidbody.mass * movement.deceleration;
        rigidbody.addForce(frictionForce);
    }
    
    // Jump force
    if (movement.jumpQueued && movement.isGrounded) {
        float jumpImpulse = movement.jumpForce * rigidbody.mass;
        rigidbody.addForce(glm::vec3(0.0f, jumpImpulse, 0.0f));
        movement.jumpQueued = false;
        movement.isGrounded = false;
        movement.movementState = MovementState::JUMPING;
        std::cout << "[PlayerMovement] Jump executed! Force: " << jumpImpulse << std::endl;
    }
    
    // Wall run forces
    if (movement.isWallRunning) {
        // Counteract gravity while wall running
        rigidbody.addForce(glm::vec3(0.0f, 9.81f * rigidbody.mass, 0.0f));
        
        // Apply horizontal wall run force
        glm::vec3 wallRunDir = glm::cross(movement.wallNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        rigidbody.addForce(wallRunDir * movement.wallRunSpeed * rigidbody.mass);
    }
    
    // Clear forces at end of frame (PhysX will handle this, but good practice)
    rigidbody.clearAccumulator();
}

bool PlayerMovementSystem::CheckWallCollision(Core::Entity entity, PlayerMovementComponent& movement, 
                                             glm::vec3& wallNormal) {
    // TODO: Implement proper wall collision detection with PhysX raycasts
    return false;
}

glm::vec2 PlayerMovementSystem::GetMovementInput(const PlayerInputComponent& input) {
    glm::vec2 moveInput(0.0f);
    
    if (input.keys[GLFW_KEY_W]) moveInput.y += 1.0f;
    if (input.keys[GLFW_KEY_S]) moveInput.y -= 1.0f;
    if (input.keys[GLFW_KEY_A]) moveInput.x -= 1.0f;  
    if (input.keys[GLFW_KEY_D]) moveInput.x += 1.0f;
    
    if (glm::length(moveInput) > 1.0f) {
        moveInput = glm::normalize(moveInput);
    }
    
    return moveInput;
}

void PlayerMovementSystem::BuildMomentum(PlayerMovementComponent& movement, glm::vec2 inputDirection, 
                                        float deltaTime, float targetSpeed) {
    // This function is deprecated in favor of ApplyMovementForces
    // which properly integrates with PhysX
}

} // namespace Gameplay
} // namespace CudaGame
