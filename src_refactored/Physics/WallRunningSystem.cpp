#include "Physics/WallRunningSystem.h"
#include "Core/Coordinator.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <iostream>
#include <algorithm>

namespace CudaGame {
namespace Physics {

WallRunningSystem::WallRunningSystem() {
    // Wall-running should update after general physics
}

WallRunningSystem::~WallRunningSystem() {}

bool WallRunningSystem::Initialize() {
    std::cout << "[WallRunningSystem] Initializing wall-running physics..." << std::endl;
    return true;
}

void WallRunningSystem::Shutdown() {
    m_wallSurfaces.clear();
    m_wallRunStartCallbacks.clear();
    m_wallRunEndCallbacks.clear();
    std::cout << "[WallRunningSystem] Shutting down wall-running physics." << std::endl;
}

void WallRunningSystem::Update(float deltaTime) {
    UpdateWallRunning(deltaTime);
    
    if (m_debugVisualization) {
        DrawDebugInfo();
    }
}

void WallRunningSystem::UpdateWallRunning(float deltaTime) {
// Get coordinator to access entity components
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // TODO: Implement entity queries when ECS system is ready
    // For now, placeholder implementation
}

void WallRunningSystem::UpdateCharacterController(Core::Entity entity, CharacterControllerComponent& controller, 
                                                RigidbodyComponent& rigidbody, const Rendering::TransformComponent& transform, 
                                                float deltaTime) {
    // Update timers
    if (controller.isWallRunning) {
        controller.wallRunTimer += deltaTime;
        
        // Check if wall-run time has expired
        if (controller.wallRunTimer >= controller.maxWallRunTime) {
            EndWallRun(controller, rigidbody);
            return;
        }
        
        UpdateWallRunPhysics(controller, rigidbody, deltaTime);
    } else {
        // Check for wall-running opportunities
        WallSurface detectedWall;
        if (DetectWall(transform.position, rigidbody.velocity, detectedWall)) {
            if (CanStartWallRun(controller, rigidbody.velocity, detectedWall)) {
                StartWallRun(controller, rigidbody, detectedWall);
                
                // Trigger callbacks
                for (auto& callback : m_wallRunStartCallbacks) {
                    callback(entity, detectedWall.normal);
                }
            }
        }
    }
    
    // Update dash timer
    if (controller.isDashing) {
        controller.dashTimer += deltaTime;
        if (controller.dashTimer >= controller.maxDashTime) {
            controller.isDashing = false;
            controller.dashTimer = 0.0f;
        }
    }
}

bool WallRunningSystem::DetectWall(const glm::vec3& position, const glm::vec3& velocity, WallSurface& outWall) {
    // Cast rays in multiple directions to detect walls
    std::vector<glm::vec3> directions = {
        glm::vec3(1.0f, 0.0f, 0.0f),   // Right
        glm::vec3(-1.0f, 0.0f, 0.0f),  // Left
        glm::vec3(0.0f, 0.0f, 1.0f),   // Forward
        glm::vec3(0.0f, 0.0f, -1.0f)   // Backward
    };
    
    for (const auto& dir : directions) {
        if (RaycastForWall(position, dir, m_wallDetectionRange, outWall)) {
            // Check if this wall is suitable for running
            if (IsValidWallForRunning(outWall, velocity)) {
                return true;
            }
        }
    }
    
    return false;
}

bool WallRunningSystem::CanStartWallRun(const CharacterControllerComponent& controller, const glm::vec3& velocity, const WallSurface& wall) {
    // Can't start wall-running if already wall-running or on ground
    if (controller.isWallRunning || controller.isGrounded) {
        return false;
    }
    
    // Must have minimum horizontal speed
    glm::vec3 horizontalVelocity = glm::vec3(velocity.x, 0.0f, velocity.z);
    if (glm::length(horizontalVelocity) < m_wallRunMinSpeed) {
        return false;
    }
    
    // Wall must be within acceptable angle range
    float wallAngle = GetWallAngle(wall.normal);
    return wallAngle <= m_maxWallRunAngle;
}

void WallRunningSystem::StartWallRun(CharacterControllerComponent& controller, RigidbodyComponent& rigidbody, const WallSurface& wall) {
    controller.isWallRunning = true;
    controller.wallNormal = wall.normal;
    controller.wallRunTimer = 0.0f;
    
    // Preserve current momentum but modify for wall-running
    controller.preservedMomentum = ConserveMomentumOnTransition(rigidbody.velocity, wall.normal, m_momentumConservationFactor);
    controller.shouldPreserveMomentum = true;
    
    // Reduce gravity effect during wall-running
    rigidbody.velocity.y = std::max(rigidbody.velocity.y, 0.0f); // Remove negative Y velocity
}

void WallRunningSystem::UpdateWallRunPhysics(CharacterControllerComponent& controller, RigidbodyComponent& rigidbody, float deltaTime) {
    // Apply reduced gravity during wall-running
    glm::vec3 gravityForce = glm::vec3(0.0f, -9.81f * m_wallRunGravityScale, 0.0f);
    rigidbody.addForce(gravityForce * rigidbody.mass);
    
    // Calculate wall-running direction (parallel to wall)
    glm::vec3 wallRunDirection = CalculateWallRunDirection(controller.wallNormal, glm::vec3(0.0f, 0.0f, 1.0f), rigidbody.velocity);
    
    // Apply wall-running velocity
    glm::vec3 targetVelocity = wallRunDirection * controller.wallRunSpeed;
    targetVelocity.y = rigidbody.velocity.y; // Preserve vertical component
    
    // Smoothly interpolate to target velocity
    rigidbody.velocity = glm::mix(rigidbody.velocity, targetVelocity, deltaTime * 5.0f);
    
    // Apply a small force toward the wall to maintain contact
    glm::vec3 wallForce = -controller.wallNormal * 2.0f;
    rigidbody.addForce(wallForce);
}

void WallRunningSystem::EndWallRun(CharacterControllerComponent& controller, RigidbodyComponent& rigidbody, bool jumped) {
    controller.isWallRunning = false;
    controller.wallRunTimer = 0.0f;
    
    // Apply preserved momentum if available
    if (controller.shouldPreserveMomentum) {
        rigidbody.velocity += controller.preservedMomentum * 0.5f; // Reduced momentum application
        controller.shouldPreserveMomentum = false;
    }
    
    // If jumped off wall, add some outward velocity
    if (jumped) {
        glm::vec3 wallKickVelocity = controller.wallNormal * 5.0f; // Push away from wall
        wallKickVelocity.y = 8.0f; // Add upward velocity
        rigidbody.velocity += wallKickVelocity;
    }
}

glm::vec3 WallRunningSystem::CalculateWallRunDirection(const glm::vec3& wallNormal, const glm::vec3& inputDirection, const glm::vec3& currentVelocity) {
    // Project current velocity onto the plane defined by the wall normal
    glm::vec3 projectedVelocity = ProjectVectorOntoPlane(currentVelocity, wallNormal);
    
    // Get the direction along the wall (perpendicular to wall normal)
    glm::vec3 wallRight = glm::cross(wallNormal, glm::vec3(0.0f, 1.0f, 0.0f));
    if (glm::length(wallRight) < 0.1f) {
        // If wall is horizontal, use forward direction
        wallRight = glm::cross(wallNormal, glm::vec3(0.0f, 0.0f, 1.0f));
    }
    wallRight = glm::normalize(wallRight);
    
    // Determine direction based on current movement
    float rightDot = glm::dot(projectedVelocity, wallRight);
    return rightDot >= 0.0f ? wallRight : -wallRight;
}

glm::vec3 WallRunningSystem::ConserveMomentumOnTransition(const glm::vec3& currentVelocity, const glm::vec3& wallNormal, float conservationFactor) {
    // Project velocity onto wall plane to conserve tangential momentum
    glm::vec3 tangentialVelocity = ProjectVectorOntoPlane(currentVelocity, wallNormal);
    return tangentialVelocity * conservationFactor;
}

bool WallRunningSystem::RaycastForWall(const glm::vec3& origin, const glm::vec3& direction, float maxDistance, WallSurface& outWall) {
    // This is a simplified raycast - in a real implementation, this would use the physics world
    // For now, we'll check against registered wall surfaces
    
    float closestDistance = maxDistance;
    bool foundWall = false;
    
    for (const auto& [entity, wallSurface] : m_wallSurfaces) {
        // Simple distance check to wall position
        float distance = glm::distance(origin, wallSurface.position);
        if (distance < closestDistance && distance < maxDistance) {
            outWall = wallSurface;
            closestDistance = distance;
            foundWall = true;
        }
    }
    
    return foundWall;
}

bool WallRunningSystem::IsGrounded(const glm::vec3& position, const ColliderComponent& collider) {
    // Simple ground check - cast ray downward
    // In a full implementation, this would use proper collision detection
    WallSurface dummy;
    return RaycastForWall(position, glm::vec3(0.0f, -1.0f, 0.0f), 1.1f, dummy);
}

float WallRunningSystem::GetWallAngle(const glm::vec3& wallNormal) {
    // Calculate angle from vertical (how steep the wall is)
    return glm::acos(glm::abs(glm::dot(wallNormal, glm::vec3(0.0f, 1.0f, 0.0f))));
}

bool WallRunningSystem::IsValidWallForRunning(const WallSurface& wall, const glm::vec3& playerVelocity) {
    if (!wall.canWallRun) return false;
    
    // Check wall angle
    if (GetWallAngle(wall.normal) > m_maxWallRunAngle) return false;
    
    // Check if player is moving toward or along the wall
    float velocityDot = glm::dot(glm::normalize(playerVelocity), -wall.normal);
    return velocityDot > 0.1f; // Moving toward wall
}

glm::vec3 WallRunningSystem::ProjectVectorOntoPlane(const glm::vec3& vector, const glm::vec3& planeNormal) {
    return vector - glm::dot(vector, planeNormal) * planeNormal;
}

// Wall surface management
void WallRunningSystem::RegisterWallSurface(Core::Entity entity, const WallSurface& surface) {
    m_wallSurfaces[entity] = surface;
}

void WallRunningSystem::UnregisterWallSurface(Core::Entity entity) {
    m_wallSurfaces.erase(entity);
}

// Callback registration
void WallRunningSystem::RegisterWallRunStartCallback(WallRunStartCallback callback) {
    m_wallRunStartCallbacks.push_back(callback);
}

void WallRunningSystem::RegisterWallRunEndCallback(WallRunEndCallback callback) {
    m_wallRunEndCallbacks.push_back(callback);
}

// Debug methods
void WallRunningSystem::DrawDebugInfo() {
    // Debug visualization would be implemented here
    // This would draw wall normals, velocity vectors, etc.
}

void WallRunningSystem::DrawWallNormal(const glm::vec3& position, const glm::vec3& normal) {
    // Debug line drawing - implementation depends on rendering system
}

void WallRunningSystem::DrawVelocityVector(const glm::vec3& position, const glm::vec3& velocity) {
    // Debug line drawing - implementation depends on rendering system
}

void WallRunningSystem::DrawWallRunPath(const glm::vec3& position, const glm::vec3& direction) {
    // Debug line drawing - implementation depends on rendering system
}

} // namespace Physics
} // namespace CudaGame
