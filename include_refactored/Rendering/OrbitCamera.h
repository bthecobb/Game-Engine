#pragma once

#include "Rendering/Camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <functional>

namespace CudaGame {
namespace Rendering {

class OrbitCamera : public Camera {
public:
    enum class CameraMode {
        ORBIT_FOLLOW,   // Standard third-person follow
        FREE_LOOK,      // Free camera movement
        COMBAT_FOCUS    // Enhanced combat positioning
    };

    struct OrbitSettings {
        float distance = 15.0f;          // Distance from target
        float heightOffset = 2.0f;       // Height above target
        float minDistance = 3.0f;        // Minimum zoom distance
        float maxDistance = 50.0f;       // Maximum zoom distance
        float minPitch = -80.0f;         // Minimum vertical angle (degrees)
        float maxPitch = 80.0f;          // Maximum vertical angle (degrees)
        float mouseSensitivity = 0.1f;   // Mouse sensitivity
        float smoothSpeed = 8.0f;        // Camera smoothing speed
        float zoomSpeed = 2.0f;          // Zoom sensitivity
        bool invertY = false;            // Invert Y-axis
    };

    struct CollisionSettings {
        bool enableCollision = true;     // Enable collision detection
        float collisionRadius = 0.5f;    // Camera collision sphere radius
        float raycastDistance = 1.0f;    // Additional raycast distance
        float minCollisionDistance = 1.0f; // Minimum distance when colliding
    };

public:
    OrbitCamera(ProjectionType type = ProjectionType::PERSPECTIVE);
    ~OrbitCamera() = default;

    // Core orbit camera methods
    void Update(float deltaTime, const glm::vec3& targetPosition, const glm::vec3& targetVelocity = glm::vec3(0.0f));
    void ApplyMouseDelta(float xDelta, float yDelta);
    void ApplyZoom(float zoomDelta);
    
    // Mode management
    void SetCameraMode(CameraMode mode);
    CameraMode GetCameraMode() const { return m_currentMode; }
    
    // Configuration
    void SetOrbitSettings(const OrbitSettings& settings) { m_orbitSettings = settings; }
    const OrbitSettings& GetOrbitSettings() const { return m_orbitSettings; }
    
    void SetCollisionSettings(const CollisionSettings& settings) { m_collisionSettings = settings; }
    const CollisionSettings& GetCollisionSettings() const { return m_collisionSettings; }
    
    // Target management
    void SetTarget(const glm::vec3& target) { m_targetPosition = target; }
    const glm::vec3& GetTarget() const { return m_targetPosition; }
    
    // Distance control
    void SetDistance(float distance, bool instant = false);
    
    // Spherical coordinates access
    float GetYaw() const { return m_yaw; }
    float GetPitch() const { return m_pitch; }
    float GetDistance() const { return m_currentDistance; }
    
    // Direction vectors (already inherited from Camera base class)
    // GetForward() and GetRight() are available from Camera
    
    // Collision detection (to be implemented with physics system)
    void SetCollisionCheckCallback(std::function<bool(const glm::vec3&, const glm::vec3&, float)> callback);
    
    // Debug visualization
    void EnableDebugVisualization(bool enable) { m_debugVisualization = enable; }
    bool IsDebugVisualizationEnabled() const { return m_debugVisualization; }

private:
    // Core orbit state
    CameraMode m_currentMode;
    glm::vec3 m_targetPosition;
    glm::vec3 m_desiredPosition;
    glm::vec3 m_currentPosition;
    
    // Spherical coordinates
    float m_yaw;           // Horizontal rotation (degrees)
    float m_pitch;         // Vertical rotation (degrees)
    float m_targetDistance; // Desired distance from target
    float m_currentDistance; // Current distance (for collision handling)
    
    // Smoothing and interpolation
    glm::vec3 m_velocity;
    float m_yawVelocity;
    float m_pitchVelocity;
    float m_distanceVelocity;
    
    // Settings
    OrbitSettings m_orbitSettings;
    CollisionSettings m_collisionSettings;
    
    // Collision detection
    std::function<bool(const glm::vec3&, const glm::vec3&, float)> m_collisionCallback;
    
    // Debug
    bool m_debugVisualization;
    
    // Internal methods
    void UpdateOrbitFollow(float deltaTime, const glm::vec3& targetPosition, const glm::vec3& targetVelocity);
    void UpdateFreeLook(float deltaTime);
    void UpdateCombatFocus(float deltaTime, const glm::vec3& targetPosition, const glm::vec3& targetVelocity);
    
    glm::vec3 CalculateDesiredPosition(const glm::vec3& target) const;
    glm::vec3 HandleCollision(const glm::vec3& desiredPos, const glm::vec3& targetPos);
    void ApplySmoothing(float deltaTime);
    void ClampAngles();
    void UpdateCameraVectorsFromPosition();
    
    // Spherical coordinate conversion
    glm::vec3 SphericalToCartesian(float yaw, float pitch, float distance) const;
    void CartesianToSpherical(const glm::vec3& position, const glm::vec3& target, float& outYaw, float& outPitch, float& outDistance) const;
    
    // State validation and initialization
    bool ValidateCameraState();
    void InitializeCameraMode(CameraMode mode);
};

} // namespace Rendering
} // namespace CudaGame
