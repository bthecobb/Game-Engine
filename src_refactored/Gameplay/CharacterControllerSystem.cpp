#include "Gameplay/CharacterControllerSystem.h"
#include "Core/Coordinator.h"
#include "Physics/PhysXPhysicsSystem.h"
#include "Rendering/OrbitCamera.h"
#include "Gameplay/LevelComponents.h"  // For WallComponent
#include "Animation/AnimationSystem.h"
#include "Gameplay/AnimationControllerComponent.h"
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <fstream>

// GLFW key constants (to avoid linking GLFW when not needed)
#define GLFW_KEY_W 87
#define GLFW_KEY_A 65
#define GLFW_KEY_S 83
#define GLFW_KEY_D 68
#define GLFW_KEY_E 69
#define GLFW_KEY_SPACE 32
#define GLFW_KEY_LEFT_SHIFT 340
#define GLFW_KEY_LEFT_CONTROL 341

namespace CudaGame {
namespace Gameplay {

CharacterControllerSystem::CharacterControllerSystem() 
    : m_physicsSystem(nullptr)
    , m_camera(nullptr)
    , m_coyoteTime(0.15f)
    , m_jumpBufferTime(0.1f) {
}

bool CharacterControllerSystem::Initialize() {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Get physics system reference
    m_physicsSystem = Core::Coordinator::GetInstance().GetSystem<Physics::PhysXPhysicsSystem>().get();
    if (!m_physicsSystem) {
        std::cerr << "[CharacterControllerSystem] Failed to get PhysX system!" << std::endl;
        return false;
    }
    
    std::cout << "[CharacterControllerSystem] Initialized" << std::endl;
    return true;
}

void CharacterControllerSystem::SetCamera(Rendering::OrbitCamera* camera) {
    m_camera = camera;
}

void CharacterControllerSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // DEBUG: Log entity count and status every 60 frames TO FILE (avoids shader compiler interleaving)
    static int debugCounter = 0;
    static std::ofstream ccsLog("ccs_debug.txt", std::ios::trunc);
    if (++debugCounter % 60 == 0 && ccsLog.is_open()) {
        ccsLog << "[CCS DEBUG] Frame " << debugCounter 
               << " | Entities: " << mEntities.size() 
               << " | Camera: " << (m_camera ? "SET" : "NULL")
               << " | PhysX: " << (m_physicsSystem ? "SET" : "NULL") << std::endl;
        ccsLog.flush();
    }
    
    for (auto const& entity : mEntities) {
        // Get all required components
        auto& charController = coordinator.GetComponent<Physics::CharacterControllerComponent>(entity);
        auto& rigidbody = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        auto& movement = coordinator.GetComponent<PlayerMovementComponent>(entity);
        auto& input = coordinator.GetComponent<PlayerInputComponent>(entity);
        
        // Update timers
        UpdateTimers(charController, deltaTime);
        
        // Perform ground check
        CheckGrounding(entity, charController, transform, rigidbody);
        
        // Handle input with camera-relative movement
        glm::vec3 moveDirection = GetCameraRelativeMovement(input, movement);
        
        // Handle jump with coyote time and jump buffering
        HandleJump(charController, movement, rigidbody, input);
        
        // Apply movement forces
        ApplyMovement(charController, movement, rigidbody, moveDirection, deltaTime);
        
        // Check for wall running opportunities
        CheckWallRunning(entity, charController, transform, rigidbody, input);
        
        // Handle dashing
        HandleDashing(charController, movement, rigidbody, input, moveDirection, deltaTime);
        
        // Update Animation State (DECOUPLED via AnimationControllerComponent)
        if (coordinator.HasComponent<AnimationControllerComponent>(entity)) {
            auto& animCtrl = coordinator.GetComponent<AnimationControllerComponent>(entity);
            UpdateAnimationController(charController, rigidbody, movement, animCtrl);
        } else if (coordinator.HasComponent<Animation::AnimationComponent>(entity)) {
            auto& animComp = coordinator.GetComponent<Animation::AnimationComponent>(entity);
            UpdateAnimationState(charController, rigidbody, movement, animComp);
        }
    }
}

void CharacterControllerSystem::UpdateAnimationController(
    const Physics::CharacterControllerComponent& controller,
    const Physics::RigidbodyComponent& rb,
    const Gameplay::PlayerMovementComponent& movement,
    Gameplay::AnimationControllerComponent& animCtrl)
{
    // Write Physics State to Blackboard
    float horizontalSpeed = glm::length(glm::vec3(rb.velocity.x, 0, rb.velocity.z));
    
    animCtrl.SetSpeed(horizontalSpeed);
    animCtrl.SetVerticalSpeed(rb.velocity.y);
    animCtrl.SetGrounded(controller.isGrounded);
    
    animCtrl.boolParams["IsWallRunning"] = controller.isWallRunning;
    animCtrl.boolParams["IsDashing"] = controller.isDashing;
    
    // We leave the State determination to AnimationControllerSystem
}

// Legacy Direct Update (to be deprecated or used as fallback)
void CharacterControllerSystem::UpdateAnimationState(
    const Physics::CharacterControllerComponent& controller,
    const Physics::RigidbodyComponent& rb,
    const PlayerMovementComponent& movement,
    Animation::AnimationComponent& anim) 
{
    using namespace Animation;
    
    AnimationState targetState = AnimationState::IDLE;
    
    // Determine target state based on physics
    if (controller.isWallRunning) {
        targetState = AnimationState::WALL_RUNNING;
    } else if (controller.isDashing) {
        // Dashing usually overrides everything
        // We don't have a DASH state in enum? Let's check.
        // Step 11463 View: Yes, we do? No, not explicit. 
        // We have RUNNING, SPRINTING.
        // Let's use RUNNING for now or add DASH later.
        targetState = AnimationState::RUNNING; 
    } else if (!controller.isGrounded) {
        // Airborne
        if (rb.velocity.y > 0.5f) {
            targetState = AnimationState::JUMPING;
        } else {
            targetState = AnimationState::FALLING;
        }
    } else {
        // Grounded
        float horizontalSpeed = glm::length(glm::vec3(rb.velocity.x, 0, rb.velocity.z));
        if (horizontalSpeed > 0.1f) {
            if (horizontalSpeed > movement.baseSpeed * 1.1f) {
                targetState = AnimationState::SPRINTING; // Use Running clip if Sprinting not available
            } else {
                targetState = AnimationState::WALKING; // Or RUNNING depending on speed
            }
            
            // Map strictly to available clips for now (Idle, Walk, Run, Attack)
            // If speed > 5, Run. Else Walk.
            if (horizontalSpeed > 5.0f || movement.movementState == MovementState::RUNNING) {
                targetState = AnimationState::RUNNING;
            } else {
                targetState = AnimationState::WALKING;
            }
        } else {
            targetState = AnimationState::IDLE;
        }
    }
    
    // Attack override (if we had combat component here, we'd check it)
    // For now, simple movement state mapping.
    
    // State Transition Logic
    if (targetState != anim.currentState) {
        anim.previousState = anim.currentState;
        anim.currentState = targetState;
        anim.animationTime = 0.0f; // Reset time on state change
        // In a real system, we'd crossfade here.
        // AnimationSystem::updateEntityAnimation handles the clip selection based on currentState.
    }
    
    // Update Blend Parameters
    anim.movementSpeed = glm::length(glm::vec3(rb.velocity.x, 0, rb.velocity.z));
}

void CharacterControllerSystem::UpdateTimers(Physics::CharacterControllerComponent& controller, float deltaTime) {
    // Update coyote time
    if (!controller.isGrounded) {
        controller.lastGroundedTime += deltaTime;
    } else {
        controller.lastGroundedTime = 0.0f;
    }
    
    // Update jump buffer
    if (controller.jumpBufferTimer > 0) {
        controller.jumpBufferTimer -= deltaTime;
    }
    
    // Update wall run timer
    if (controller.isWallRunning) {
        controller.wallRunTimer += deltaTime;
        if (controller.wallRunTimer >= controller.maxWallRunTime) {
            ExitWallRun(controller);
        }
    }
    
    // Update dash timer
    if (controller.isDashing) {
        controller.dashTimer -= deltaTime;
        if (controller.dashTimer <= 0) {
            controller.isDashing = false;
        }
    }
    
    // Update dash cooldown
    if (controller.lastDashTime > 0) {
        controller.lastDashTime -= deltaTime;
    }
}

void CharacterControllerSystem::CheckGrounding(Core::Entity entity, 
                                              Physics::CharacterControllerComponent& controller,
                                              const Rendering::TransformComponent& transform,
                                              const Physics::RigidbodyComponent& rigidbody) {
    // Perform raycast down from character position
    const float GROUND_CHECK_DISTANCE = 0.2f;
    const float CHARACTER_HEIGHT = 1.8f;
    
    glm::vec3 rayStart = transform.position;
    glm::vec3 rayEnd = rayStart - glm::vec3(0, CHARACTER_HEIGHT * 0.5f + GROUND_CHECK_DISTANCE, 0);
    
    // Simple ground check - in a real implementation, use PhysX raycast
    bool wasGrounded = controller.isGrounded;
    // Ground height matches top surface of ground cube at y = -0.5
    float groundY = -0.5f;
    float feetY = transform.position.y - CHARACTER_HEIGHT * 0.5f;
    
    controller.isGrounded = (feetY <= groundY + GROUND_CHECK_DISTANCE) && (rigidbody.velocity.y <= 0.1f);
    
    // Landing detection
    if (!wasGrounded && controller.isGrounded) {
        OnLanding(controller);
        
        // Check if we have a buffered jump
        if (controller.jumpBufferTimer > 0) {
            controller.isJumping = true;
            controller.jumpBufferTimer = 0;
        }
    }
    
    // Just left ground
    if (wasGrounded && !controller.isGrounded && !controller.isJumping) {
        // Started falling (not jumping)
        controller.lastGroundedTime = 0;
    }
}

glm::vec3 CharacterControllerSystem::GetCameraRelativeMovement(const PlayerInputComponent& input,
                                                               const PlayerMovementComponent& movement) {
    // Get input direction
    glm::vec2 inputDir(0.0f);
    if (input.keys[GLFW_KEY_W]) inputDir.y += 1.0f;
    if (input.keys[GLFW_KEY_S]) inputDir.y -= 1.0f;
    if (input.keys[GLFW_KEY_A]) inputDir.x -= 1.0f;
    if (input.keys[GLFW_KEY_D]) inputDir.x += 1.0f;
    
    // DEBUG: Log input detection every 60 frames
    static int inputDebugCounter = 0;
    if (++inputDebugCounter % 60 == 0 && glm::length(inputDir) > 0.01f) {
        std::cout << "[CCS INPUT] inputDir: (" << inputDir.x << ", " << inputDir.y << ")" << std::endl;
    }
    
    // Normalize diagonal movement
    if (glm::length(inputDir) > 1.0f) {
        inputDir = glm::normalize(inputDir);
    }
    
    // If no camera, return world-space movement
    if (!m_camera) {
        return glm::vec3(inputDir.x, 0, inputDir.y);
    }
    
    // Get camera forward and right vectors (projected on XZ plane)
    glm::vec3 camForward = m_camera->GetForward();
    camForward.y = 0;
    camForward = glm::normalize(camForward);
    
    glm::vec3 camRight = m_camera->GetRight();
    camRight.y = 0;
    camRight = glm::normalize(camRight);
    
    // Calculate camera-relative movement
    glm::vec3 moveDirection = camForward * inputDir.y + camRight * inputDir.x;
    
    // Apply movement speed (sprint multiplies base speed, doesn't use maxSpeed)
    float speed = input.keys[GLFW_KEY_LEFT_SHIFT] ? (movement.baseSpeed * movement.sprintMultiplier) : movement.baseSpeed;
    return moveDirection * speed;
}

void CharacterControllerSystem::HandleJump(Physics::CharacterControllerComponent& controller,
                                          const PlayerMovementComponent& movement,
                                          Physics::RigidbodyComponent& rigidbody,
                                          const PlayerInputComponent& input) {
    // Jump input buffering
    static bool jumpPressed = false;
    bool jumpThisFrame = input.keys[GLFW_KEY_SPACE] && !jumpPressed;
    jumpPressed = input.keys[GLFW_KEY_SPACE];
    
    if (jumpThisFrame) {
        controller.jumpBufferTimer = m_jumpBufferTime;
    }
    
    // Can we jump? (grounded or within coyote time)
    bool canJump = controller.isGrounded || 
                   (controller.lastGroundedTime < m_coyoteTime && controller.airJumps == 0);
    
    // Wall jump
    if (controller.isWallRunning && jumpThisFrame) {
        PerformWallJump(controller, rigidbody);
        return;
    }
    
    // Regular jump or double jump
    if (controller.jumpBufferTimer > 0) {
        if (canJump) {
            // Ground jump
            PerformJump(controller, movement, rigidbody);
            controller.jumpBufferTimer = 0;
            controller.airJumps = 0;
        } else if (controller.canDoubleJump && controller.airJumps < controller.maxAirJumps) {
            // Air jump
            PerformAirJump(controller, movement, rigidbody);
            controller.jumpBufferTimer = 0;
            controller.airJumps++;
        }
    }
}

void CharacterControllerSystem::PerformJump(Physics::CharacterControllerComponent& controller,
                                           const PlayerMovementComponent& movement,
                                           Physics::RigidbodyComponent& rigidbody) {
    // Clear vertical velocity and apply jump impulse
    rigidbody.velocity.y = 0;
    float jumpImpulse = movement.jumpForce * rigidbody.mass;
    rigidbody.addForce(glm::vec3(0, jumpImpulse, 0));
    
    controller.isJumping = true;
    controller.isGrounded = false;
    
    std::cout << "[CharacterController] Jump performed! Force: " << jumpImpulse << std::endl;
}

void CharacterControllerSystem::PerformAirJump(Physics::CharacterControllerComponent& controller,
                                              const PlayerMovementComponent& movement,
                                              Physics::RigidbodyComponent& rigidbody) {
    // Air jump with full force for double jump
    rigidbody.velocity.y = 0;
    float jumpImpulse = movement.jumpForce * rigidbody.mass * 1.1f; // 110% of normal jump for extra height
    rigidbody.addForce(glm::vec3(0, jumpImpulse, 0));
    
    std::cout << "[CharacterController] Air jump performed! Jumps used: " 
              << (controller.airJumps + 1) << "/" << controller.maxAirJumps << std::endl;
}

void CharacterControllerSystem::PerformWallJump(Physics::CharacterControllerComponent& controller,
                                               Physics::RigidbodyComponent& rigidbody) {
    // Jump away from wall
    glm::vec3 jumpDirection = controller.wallNormal + glm::vec3(0, 1, 0);
    jumpDirection = glm::normalize(jumpDirection);
    
    float wallJumpForce = controller.jumpForce * rigidbody.mass * 1.2f;
    rigidbody.velocity = jumpDirection * wallJumpForce;
    
    ExitWallRun(controller);
    controller.isJumping = true;
    
    std::cout << "[CharacterController] Wall jump performed!" << std::endl;
}

void CharacterControllerSystem::ApplyMovement(Physics::CharacterControllerComponent& controller,
                                             const PlayerMovementComponent& movement,
                                             Physics::RigidbodyComponent& rigidbody,
                                             const glm::vec3& moveDirection,
                                             float deltaTime) {
    // Don't apply movement if dashing
    if (controller.isDashing) {
        // Apply dash velocity
        rigidbody.velocity = controller.dashDirection * controller.dashSpeed;
        return;
    }
    
    // Wall running movement
    if (controller.isWallRunning) {
        ApplyWallRunMovement(controller, rigidbody, deltaTime);
        return;
    }
    
    // Regular movement
    if (glm::length(moveDirection) > 0.01f) {
        glm::vec3 targetVelocity = moveDirection;
        targetVelocity.y = rigidbody.velocity.y; // Preserve vertical velocity
        
        // Apply acceleration
        float accel = controller.isGrounded ? movement.acceleration : movement.airAcceleration;
        glm::vec3 velocityDiff = targetVelocity - rigidbody.velocity;
        velocityDiff.y = 0; // Don't affect vertical
        
        glm::vec3 force = velocityDiff * rigidbody.mass * accel;
        rigidbody.addForce(force);
        
        static int movementDebugCounter = 0;
        if (movementDebugCounter++ % 60 == 0) {
            std::cout << "[CharacterController] Movement - Force: (" << force.x << ", " << force.y << ", " << force.z << ")" 
                      << " Velocity: (" << rigidbody.velocity.x << ", " << rigidbody.velocity.y << ", " << rigidbody.velocity.z << ")" 
                      << " Grounded: " << (controller.isGrounded ? "YES" : "NO") << std::endl;
        }
        
        // Clamp horizontal velocity to maxSpeed
        glm::vec3 horizontalVel = glm::vec3(rigidbody.velocity.x, 0, rigidbody.velocity.z);
        float horizontalSpeed = glm::length(horizontalVel);
        if (horizontalSpeed > movement.maxSpeed) {
            horizontalVel = glm::normalize(horizontalVel) * movement.maxSpeed;
            rigidbody.velocity.x = horizontalVel.x;
            rigidbody.velocity.z = horizontalVel.z;
        }
        
        // Momentum preservation
        if (controller.shouldPreserveMomentum) {
            rigidbody.velocity += controller.preservedMomentum * deltaTime;
            controller.preservedMomentum *= 0.95f; // Decay
            
            if (glm::length(controller.preservedMomentum) < 0.1f) {
                controller.shouldPreserveMomentum = false;
            }
        }
    } else if (controller.isGrounded) {
        // Apply friction
        glm::vec3 friction = -glm::vec3(rigidbody.velocity.x, 0, rigidbody.velocity.z);
        friction *= movement.deceleration * rigidbody.mass;
        rigidbody.addForce(friction);
    }
}

void CharacterControllerSystem::ApplyWallRunMovement(Physics::CharacterControllerComponent& controller,
                                                    Physics::RigidbodyComponent& rigidbody,
                                                    float deltaTime) {
    // Calculate wall run direction (perpendicular to wall normal)
    glm::vec3 wallRunDir = glm::cross(controller.wallNormal, glm::vec3(0, 1, 0));
    
    // Determine direction based on player velocity
    if (glm::dot(wallRunDir, rigidbody.velocity) < 0) {
        wallRunDir = -wallRunDir;
    }
    
    // Apply wall run velocity
    rigidbody.velocity = wallRunDir * controller.wallRunSpeed;
    
    // Slight upward force to counteract gravity
    rigidbody.velocity.y = 2.0f;
    
    // Stick to wall
    glm::vec3 stickForce = -controller.wallNormal * 500.0f;
    rigidbody.addForce(stickForce);
}

void CharacterControllerSystem::CheckWallRunning(Core::Entity entity,
                                                Physics::CharacterControllerComponent& controller,
                                                const Rendering::TransformComponent& transform,
                                                Physics::RigidbodyComponent& rigidbody,
                                                const PlayerInputComponent& input) {
    // Only check if pressing wall run key and not grounded
    if (!input.keys[GLFW_KEY_E] || controller.isGrounded) {
        if (controller.isWallRunning) {
            ExitWallRun(controller);
        }
        return;
    }
    
    // Already wall running
    if (controller.isWallRunning) {
        return;
    }
    
    // Check for walls using raycasts
    const float WALL_CHECK_DISTANCE = 1.5f;
    glm::vec3 checkDirections[] = {
        glm::vec3(1, 0, 0),   // Right
        glm::vec3(-1, 0, 0),  // Left
        glm::vec3(0, 0, 1),   // Forward
        glm::vec3(0, 0, -1)   // Back
    };
    
    for (const auto& dir : checkDirections) {
        glm::vec3 rayStart = transform.position;
        glm::vec3 rayEnd = rayStart + dir * WALL_CHECK_DISTANCE;
        
        bool nearWall = false;
        glm::vec3 wallNormal = glm::vec3(0.0f);
        
        // Check against world bounds as "walls"
        if ((transform.position.x > 19.0f && dir.x > 0) ||
            (transform.position.x < -19.0f && dir.x < 0) ||
            (transform.position.z > 19.0f && dir.z > 0) ||
            (transform.position.z < -19.0f && dir.z < 0)) {
            nearWall = true;
            wallNormal = -dir;
        }
        
        // Check for wall entities (for testing)
        auto& coordinator = Core::Coordinator::GetInstance();
        for (Core::Entity otherEntity = 0; otherEntity < 1000; ++otherEntity) {
            if (otherEntity == entity) continue;
            
            // Check if entity exists and has wall component
            if (coordinator.HasComponent<Rendering::TransformComponent>(otherEntity) &&
                coordinator.HasComponent<WallComponent>(otherEntity)) {
                
                auto& wallTransform = coordinator.GetComponent<Rendering::TransformComponent>(otherEntity);
                float distance = glm::distance(transform.position, wallTransform.position);
                
                // If close enough to wall
                if (distance < WALL_CHECK_DISTANCE + 2.0f) {
                    // Calculate direction to wall
                    glm::vec3 toWall = glm::normalize(wallTransform.position - transform.position);
                    
                    // Check if this direction matches our check direction
                    if (glm::dot(toWall, dir) > 0.7f) {  // Roughly same direction
                        nearWall = true;
                        wallNormal = -toWall;
                        break;
                    }
                }
            }
        }
        
        if (nearWall) {
            // Start wall running
            controller.isWallRunning = true;
            controller.wallNormal = wallNormal; // Use calculated wall normal
            controller.wallRunTimer = 0;
            controller.canDoubleJump = true; // Enable double jump after wall run
            
            // Preserve momentum
            controller.preservedMomentum = rigidbody.velocity * 0.5f;
            controller.shouldPreserveMomentum = true;
            
            std::cout << "[CharacterController] Started wall running!" << std::endl;
            break;
        }
    }
}

void CharacterControllerSystem::ExitWallRun(Physics::CharacterControllerComponent& controller) {
    controller.isWallRunning = false;
    controller.wallRunTimer = 0;
    controller.wallNormal = glm::vec3(0);
    
    std::cout << "[CharacterController] Exited wall run" << std::endl;
}

void CharacterControllerSystem::HandleDashing(Physics::CharacterControllerComponent& controller,
                                             const PlayerMovementComponent& movement,
                                             Physics::RigidbodyComponent& rigidbody,
                                             const PlayerInputComponent& input,
                                             const glm::vec3& moveDirection,
                                             float deltaTime) {
    // Check for dash input
    static bool dashPressed = false;
    bool dashThisFrame = input.keys[GLFW_KEY_LEFT_CONTROL] && !dashPressed;
    dashPressed = input.keys[GLFW_KEY_LEFT_CONTROL];
    
    if (dashThisFrame && controller.lastDashTime <= 0) {
        // Start dash
        controller.isDashing = true;
        controller.dashTimer = controller.maxDashTime;
        controller.lastDashTime = controller.dashCooldown;
        
        // Set dash direction (use movement direction or forward if no input)
        if (glm::length(moveDirection) > 0.01f) {
            controller.dashDirection = glm::normalize(moveDirection);
        } else if (m_camera) {
            controller.dashDirection = glm::normalize(m_camera->GetForward());
            controller.dashDirection.y = 0;
        } else {
            controller.dashDirection = glm::vec3(0, 0, 1); // Default forward
        }
        
        // Preserve current momentum
        controller.preservedMomentum = rigidbody.velocity;
        controller.shouldPreserveMomentum = true;
        
        std::cout << "[CharacterController] Dash initiated!" << std::endl;
    }
}

void CharacterControllerSystem::OnLanding(Physics::CharacterControllerComponent& controller) {
    controller.isJumping = false;
    controller.airJumps = 0;
    controller.canDoubleJump = false;
    
    std::cout << "[CharacterController] Landed!" << std::endl;
}

void CharacterControllerSystem::Shutdown() {
    std::cout << "[CharacterControllerSystem] Shut down" << std::endl;
}

} // namespace Gameplay
} // namespace CudaGame
