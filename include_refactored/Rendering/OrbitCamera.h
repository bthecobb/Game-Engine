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
        float minPitch = -70.0f;         // Tightened to prevent gimbal lock
        float maxPitch = 70.0f;          // Tightened to prevent gimbal lock
        float mouseSensitivity = 0.05f;  // Mouse input sensitivity
        float smoothSpeed = 8.0f;        // Rotation smoothing (higher = faster response)
        float positionFollowSpeed = 12.0f; // Position following speed
        float zoomSpeed = 2.0f;          // Zoom sensitivity
        bool invertY = false;            // Invert Y-axis
    };

    struct CollisionSettings {
        bool enableCollision = true;     // Enable collision detection
        float collisionRadius = 0.5f;    // Camera collision sphere radius
        float raycastDistance = 1.0f;    // Additional raycast distance
        float minCollisionDistance = 1.0f; // Minimum distance when colliding
    };
    
    // AAA-standard anti-jitter settings for smooth camera behavior
    struct AntiJitterSettings {
        bool enabled = true;             // Master toggle
        float deadZone = 0.5f;           // Ignore movements smaller than this (pixels)
        float microMovementThreshold = 2.0f; // Filter micro-movements below this speed
        float temporalSmoothingFactor = 0.85f; // Blend factor for temporal filter (0-1)
        float velocityDamping = 0.92f;   // Damping applied to velocity each frame
        float accelerationSmoothing = 0.7f; // Smooth acceleration changes
        bool useAdaptiveSmoothing = true;  // Adjust smoothing based on velocity
        float adaptiveMinSmooth = 4.0f;  // Min smoothing when moving fast
        float adaptiveMaxSmooth = 15.0f; // Max smoothing when stationary
    };
    
    // Camera tuning presets for different gameplay scenarios
    enum class TuningPreset {
        RESPONSIVE,   // Fast, twitchy - good for action
        CINEMATIC,    // Smooth, filmic - good for exploration
        COMBAT,       // Balanced with tight tracking
        CUSTOM        // User-defined settings
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
    
    // Runtime tuning setters
    void SetSmoothSpeed(float speed) { m_orbitSettings.smoothSpeed = speed; }
    void SetMouseSensitivity(float sensitivity) { m_orbitSettings.mouseSensitivity = sensitivity; }
    void SetPositionFollowSpeed(float speed) { m_orbitSettings.positionFollowSpeed = speed; }
    float GetSmoothSpeed() const { return m_orbitSettings.smoothSpeed; }
    float GetMouseSensitivity() const { return m_orbitSettings.mouseSensitivity; }
    float GetPositionFollowSpeed() const { return m_orbitSettings.positionFollowSpeed; }
    
    // Anti-jitter configuration
    void SetAntiJitterSettings(const AntiJitterSettings& settings) { m_antiJitterSettings = settings; }
    const AntiJitterSettings& GetAntiJitterSettings() const { return m_antiJitterSettings; }
    void SetAntiJitterEnabled(bool enabled) { m_antiJitterSettings.enabled = enabled; }
    bool IsAntiJitterEnabled() const { return m_antiJitterSettings.enabled; }
    
    // Tuning presets
    void ApplyTuningPreset(TuningPreset preset);
    TuningPreset GetCurrentPreset() const { return m_currentPreset; }


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
    AntiJitterSettings m_antiJitterSettings;
    TuningPreset m_currentPreset = TuningPreset::RESPONSIVE;
    
    // Anti-jitter temporal state
    glm::vec3 m_previousTargetPosition{0.0f};
    glm::vec3 m_smoothedTargetPosition{0.0f};
    glm::vec2 m_previousMouseDelta{0.0f};
    glm::vec2 m_smoothedMouseDelta{0.0f};
    float m_previousYaw = 0.0f;
    float m_previousPitch = 0.0f;
    
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
    
    // Anti-jitter internal methods
    float CalculateAdaptiveSmoothing(float targetSpeed) const;
    glm::vec2 FilterMouseInput(float xDelta, float yDelta);
    glm::vec3 FilterTargetPosition(const glm::vec3& rawTarget, float deltaTime);
};

} // namespace Rendering
} // namespace CudaGame
