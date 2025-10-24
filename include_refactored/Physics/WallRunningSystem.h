#pragma once

#include "Core/System.h"
#include "Physics/CharacterController.h"
#include "Physics/PhysicsComponents.h"
#include "../Debug/DebugRenderer.h"
#include "Physics/CollisionDetection.h"
#include "Rendering/RenderComponents.h"
#include <glm/glm.hpp>
#include <PxPhysicsAPI.h>
#include <glm/gtc/matrix_transform.hpp>
#include <functional>
#include <unordered_map>
#include <vector>

namespace CudaGame {
namespace Physics {

// Wall surface data for wall-running mechanics
struct WallSurface {
    glm::vec3 normal;
    glm::vec3 position;
    float friction = 0.1f;
    bool canWallRun = true;
};

class WallRunningSystem : public Core::System {
    friend class PhysXPhysicsSystem; // Allow PhysX system to set scene pointer
public:
    WallRunningSystem();
    ~WallRunningSystem();

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Wall-running configuration
    void SetWallRunGravityScale(float scale) { m_wallRunGravityScale = scale; }
    void SetWallRunMinSpeed(float speed) { m_wallRunMinSpeed = speed; }
    void SetWallRunMaxTime(float time) { m_wallRunMaxTime = time; }
    void SetWallDetectionRange(float range) { m_wallDetectionRange = range; }

    // Momentum conservation settings
    void SetMomentumConservation(float factor) { m_momentumConservationFactor = factor; }
    void SetMaxWallRunAngle(float degrees) { m_maxWallRunAngle = glm::radians(degrees); }

    // Wall surface management
    void RegisterWallSurface(Core::Entity entity, const WallSurface& surface);
    void UnregisterWallSurface(Core::Entity entity);

    // Debug settings
    void SetDebugVisualization(bool enable) { m_debugVisualization = enable; }
    void DrawDebugInfo();
    void SetDebugRenderer(Debug::IDebugRenderer* renderer) { m_debugRenderer = renderer; }

    // Callbacks for wall-running events
    using WallRunStartCallback = std::function<void(Core::Entity entity, const glm::vec3& normal)>;
    using WallRunEndCallback = std::function<void(Core::Entity entity, bool jumped)>;
    
    void RegisterWallRunStartCallback(WallRunStartCallback callback);
    void RegisterWallRunEndCallback(WallRunEndCallback callback);

private:
    // Configuration
    float m_wallRunGravityScale = 0.3f;
    float m_wallRunMinSpeed = 5.0f;
    float m_wallRunMaxTime = 3.0f;
    float m_wallDetectionRange = 1.5f;
    float m_momentumConservationFactor = 0.8f;
    float m_maxWallRunAngle = glm::radians(45.0f); // Maximum angle from vertical for wall-running

    // PhysX references
    physx::PxScene* m_physicsScene = nullptr;
    float m_minWallFriction = 0.8f;  // Minimum friction for wall-runnable surfaces
    
    // Debug
    bool m_debugVisualization = false;
    Debug::IDebugRenderer* m_debugRenderer = nullptr;
    WallSurface m_lastWallHit;
    glm::vec3 m_lastHitPoint = glm::vec3(0.0f);

    // Debug drawing helpers
    void DrawDebugLine(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color);

    // Wall surfaces
    std::unordered_map<Core::Entity, WallSurface> m_wallSurfaces;

    // Callbacks
    std::vector<WallRunStartCallback> m_wallRunStartCallbacks;
    std::vector<WallRunEndCallback> m_wallRunEndCallbacks;

    // Internal methods
    void UpdateWallRunning(float deltaTime);
    void UpdateCharacterController(Core::Entity entity, CharacterControllerComponent& controller, 
                                 RigidbodyComponent& rigidbody, const Rendering::TransformComponent& transform, 
                                 float deltaTime);

    // Wall detection
    bool DetectWall(const glm::vec3& position, const glm::vec3& velocity, WallSurface& outWall);
    bool CanStartWallRun(const CharacterControllerComponent& controller, const glm::vec3& velocity, const WallSurface& wall);
    
    // Wall-running physics
    void StartWallRun(CharacterControllerComponent& controller, RigidbodyComponent& rigidbody, const WallSurface& wall);
    void UpdateWallRunPhysics(CharacterControllerComponent& controller, RigidbodyComponent& rigidbody, float deltaTime);
    void EndWallRun(CharacterControllerComponent& controller, RigidbodyComponent& rigidbody, bool jumped = false);

    // Momentum calculations
    glm::vec3 CalculateWallRunDirection(const glm::vec3& wallNormal, const glm::vec3& inputDirection, const glm::vec3& currentVelocity);
    glm::vec3 ConserveMomentumOnTransition(const glm::vec3& currentVelocity, const glm::vec3& wallNormal, float conservationFactor);
    
    // Raycasting for wall detection
    bool RaycastForWall(const glm::vec3& origin, const glm::vec3& direction, float maxDistance, WallSurface& outWall);
    
    // Wall check for raycasts
    bool CheckIfWall(const physx::PxShape* shape);
    
    // Ground detection for character controller
    bool IsGrounded(const glm::vec3& position, const ColliderComponent& collider);
    
    // Utility functions
    float GetWallAngle(const glm::vec3& wallNormal);
    bool IsValidWallForRunning(const WallSurface& wall, const glm::vec3& playerVelocity);
    glm::vec3 ProjectVectorOntoPlane(const glm::vec3& vector, const glm::vec3& planeNormal);
    
    // Debug drawing helpers
    void DrawWallNormal(const glm::vec3& position, const glm::vec3& normal);
    void DrawVelocityVector(const glm::vec3& position, const glm::vec3& velocity);
    void DrawWallRunPath(const glm::vec3& position, const glm::vec3& direction);
};

} // namespace Physics
} // namespace CudaGame
