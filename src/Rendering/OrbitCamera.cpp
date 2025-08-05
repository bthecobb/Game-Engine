#include "Rendering/OrbitCamera.h"
#include <glm/gtx/euler_angles.hpp>
#include <cmath>
#include <iostream>
#include <chrono>

namespace CudaGame {
namespace Rendering {

OrbitCamera::OrbitCamera(ProjectionType type) : Camera(type),
    m_currentMode(CameraMode::ORBIT_FOLLOW),
    m_yaw(0.0f),
    m_pitch(-20.0f),  // Start with a slight downward angle
    m_targetDistance(15.0f),  // Use explicit default instead of uninitialized m_orbitSettings
    m_currentDistance(15.0f),
    m_yawVelocity(0.0f),
    m_pitchVelocity(0.0f),
    m_distanceVelocity(0.0f),
    m_debugVisualization(false) {

    // Initialize orbit settings with defaults
    m_orbitSettings = OrbitSettings();
    
    // Initialize positions
    m_targetPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    m_currentPosition = glm::vec3(0.0f, m_orbitSettings.heightOffset, m_orbitSettings.distance);
    m_desiredPosition = m_currentPosition;
    
    // Initialize camera vectors
    UpdateCameraVectorsFromPosition();
}

void OrbitCamera::Update(float deltaTime, const glm::vec3& targetPosition, const glm::vec3& targetVelocity) {
    switch (m_currentMode) {
        case CameraMode::ORBIT_FOLLOW:
            UpdateOrbitFollow(deltaTime, targetPosition, targetVelocity);
            break;
        case CameraMode::FREE_LOOK:
            UpdateFreeLook(deltaTime);
            break;
        case CameraMode::COMBAT_FOCUS:
            UpdateCombatFocus(deltaTime, targetPosition, targetVelocity);
            break;
    }
    
    UpdateMatrices();
}

void OrbitCamera::ApplyMouseDelta(float xDelta, float yDelta) {
    float inverted = m_orbitSettings.invertY ? -1.0f : 1.0f;
    m_yaw += xDelta * m_orbitSettings.mouseSensitivity;
    m_pitch += yDelta * m_orbitSettings.mouseSensitivity * inverted;
    ClampAngles();
}

void OrbitCamera::ApplyZoom(float zoomDelta) {
    m_targetDistance -= zoomDelta * m_orbitSettings.zoomSpeed;
    m_targetDistance = glm::clamp(m_targetDistance, m_orbitSettings.minDistance, m_orbitSettings.maxDistance);
}

void OrbitCamera::SetCameraMode(CameraMode mode) {
    // Get current timestamp for debugging
    auto now = std::chrono::steady_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    const char* oldModeName = "UNKNOWN";
    const char* newModeName = "UNKNOWN";
    
    // Convert old mode to string
    switch (m_currentMode) {
        case CameraMode::ORBIT_FOLLOW: oldModeName = "ORBIT_FOLLOW"; break;
        case CameraMode::FREE_LOOK: oldModeName = "FREE_LOOK"; break;
        case CameraMode::COMBAT_FOCUS: oldModeName = "COMBAT_FOCUS"; break;
    }
    
    // Convert new mode to string
    switch (mode) {
        case CameraMode::ORBIT_FOLLOW: newModeName = "ORBIT_FOLLOW"; break;
        case CameraMode::FREE_LOOK: newModeName = "FREE_LOOK"; break;
        case CameraMode::COMBAT_FOCUS: newModeName = "COMBAT_FOCUS"; break;
    }
    
    // Only log and validate if mode actually changes
    if (m_currentMode != mode) {
        std::cout << "{\n";
        std::cout << "  \"event\": \"camera_mode_switch\",\n";
        std::cout << "  \"timestamp_ms\": " << timestamp << ",\n";
        std::cout << "  \"previous_mode\": \"" << oldModeName << "\",\n";
        std::cout << "  \"new_mode\": \"" << newModeName << "\",\n";
        std::cout << "  \"camera_state_before\": {\n";
        std::cout << "    \"position\": [" << m_currentPosition.x << ", " << m_currentPosition.y << ", " << m_currentPosition.z << "],\n";
        std::cout << "    \"target\": [" << m_targetPosition.x << ", " << m_targetPosition.y << ", " << m_targetPosition.z << "],\n";
        std::cout << "    \"yaw\": " << m_yaw << ",\n";
        std::cout << "    \"pitch\": " << m_pitch << ",\n";
        std::cout << "    \"distance\": " << m_currentDistance << "\n";
        std::cout << "  },\n";
        
        // Validate current camera state before switching
        bool stateValid = ValidateCameraState();
        std::cout << "  \"state_valid_before_switch\": " << (stateValid ? "true" : "false") << ",\n";
        
        // Set the new mode
        m_currentMode = mode;
        
        // Initialize state for the new mode
        InitializeCameraMode(mode);
        
        // Validate state after mode switch
        bool stateValidAfter = ValidateCameraState();
        std::cout << "  \"state_valid_after_switch\": " << (stateValidAfter ? "true" : "false") << "\n";
        std::cout << "}" << std::endl;
        
        // Force matrix update after mode change
        UpdateMatrices();
    }
}

void OrbitCamera::SetCollisionCheckCallback(std::function<bool(const glm::vec3&, const glm::vec3&, float)> callback) {
    m_collisionCallback = callback;
}

void OrbitCamera::UpdateOrbitFollow(float deltaTime, const glm::vec3& targetPosition, const glm::vec3& targetVelocity) {
    m_targetPosition = targetPosition + glm::vec3(0.0f, m_orbitSettings.heightOffset, 0.0f);
    
    // Smoothly update current distance towards target distance
    m_currentDistance = glm::mix(m_currentDistance, m_targetDistance, deltaTime * m_orbitSettings.smoothSpeed);
    
    m_desiredPosition = CalculateDesiredPosition(m_targetPosition);
    m_currentPosition = glm::mix(m_currentPosition, m_desiredPosition, deltaTime * m_orbitSettings.smoothSpeed);

    if (m_collisionSettings.enableCollision && m_collisionCallback) {
        m_currentPosition = HandleCollision(m_currentPosition, m_targetPosition);
    }

    UpdateCameraVectorsFromPosition();
}

void OrbitCamera::UpdateFreeLook(float deltaTime) {
    // Free look mode: maintain current position relative to target, but allow independent movement
    // Keep the camera at its current spherical coordinates but don't force following the target
    
    // Update target position if it has changed (for reference)
    // But don't force the camera to follow it smoothly like in orbit mode
    
    // Calculate desired position based on current angles and distance
    m_desiredPosition = CalculateDesiredPosition(m_targetPosition);
    
    // In free look, we can either:
    // 1. Keep camera fixed in space while target moves
    // 2. Smoothly track target position changes but allow independent rotation
    // For now, let's use option 2 with reduced smoothing
    float freeLookSmoothSpeed = m_orbitSettings.smoothSpeed * 0.3f; // Much slower following
    m_currentPosition = glm::mix(m_currentPosition, m_desiredPosition, deltaTime * freeLookSmoothSpeed);
    
    // Handle collision if enabled
    if (m_collisionSettings.enableCollision && m_collisionCallback) {
        m_currentPosition = HandleCollision(m_currentPosition, m_targetPosition);
    }
    
    UpdateCameraVectorsFromPosition();
}

void OrbitCamera::UpdateCombatFocus(float deltaTime, const glm::vec3& targetPosition, const glm::vec3& targetVelocity) {
    // Combat focus mode: Enhanced following with predictive positioning and closer tracking
    m_targetPosition = targetPosition + glm::vec3(0.0f, m_orbitSettings.heightOffset, 0.0f);
    
    // Add velocity prediction for smoother combat camera
    glm::vec3 predictedTarget = m_targetPosition + (targetVelocity * 0.2f); // Look ahead 0.2 seconds
    
    // Use tighter distance control for combat
    float combatDistance = m_orbitSettings.distance * 0.7f; // Closer for combat
    m_currentDistance = glm::mix(m_currentDistance, combatDistance, deltaTime * m_orbitSettings.smoothSpeed * 1.5f);
    
    // Calculate desired position with prediction
    m_desiredPosition = CalculateDesiredPosition(predictedTarget);
    
    // Enhanced smoothing for combat responsiveness
    float combatSmoothSpeed = m_orbitSettings.smoothSpeed * 1.3f;
    m_currentPosition = glm::mix(m_currentPosition, m_desiredPosition, deltaTime * combatSmoothSpeed);
    
    // Handle collision if enabled
    if (m_collisionSettings.enableCollision && m_collisionCallback) {
        m_currentPosition = HandleCollision(m_currentPosition, predictedTarget);
    }
    
    UpdateCameraVectorsFromPosition();
}

glm::vec3 OrbitCamera::CalculateDesiredPosition(const glm::vec3& target) const {
    return target + SphericalToCartesian(m_yaw, m_pitch, m_currentDistance);
}

glm::vec3 OrbitCamera::HandleCollision(const glm::vec3& desiredPos, const glm::vec3& targetPos) {
    if (m_collisionCallback(desiredPos, targetPos, m_collisionSettings.collisionRadius)) {
        float collisionDist = m_targetDistance;
        glm::vec3 collisionPoint = desiredPos; // Replace with actual collision handling logic
        return glm::normalize(collisionPoint - targetPos) * collisionDist + targetPos;
    }
    return desiredPos;
}

void OrbitCamera::ApplySmoothing(float deltaTime) {
    // Logic here if needed
}

void OrbitCamera::ClampAngles() {
    // Clamp pitch to prevent gimbal lock and camera flipping
    m_pitch = glm::clamp(m_pitch, m_orbitSettings.minPitch, m_orbitSettings.maxPitch);
    
    // Normalize yaw to [-180, 180] to prevent angle accumulation
    while (m_yaw > 180.0f) m_yaw -= 360.0f;
    while (m_yaw < -180.0f) m_yaw += 360.0f;
    
    // Debug logging every 60 frames
    static int debugFrameCount = 0;
    debugFrameCount++;
    if (debugFrameCount % 60 == 0) {
        std::cout << "[DEBUG] Camera angles - Yaw: " << m_yaw << ", Pitch: " << m_pitch 
                  << ", Distance: " << m_currentDistance << std::endl;
    }
}

void OrbitCamera::UpdateCameraVectorsFromPosition() {
    // Set camera position and update vectors
    SetPosition(m_currentPosition);
    
    // Calculate camera vectors from current position to target
    glm::vec3 direction = glm::normalize(m_targetPosition - m_currentPosition);
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    
    // Re-orthonormalize to prevent drift
    glm::vec3 right = glm::normalize(glm::cross(direction, worldUp));
    glm::vec3 up = glm::normalize(glm::cross(right, direction));
    
    // Store the vectors
    m_forward = direction;
    m_right = right;
    m_up = up;
    
    // Update view matrix using lookAt
    LookAt(m_targetPosition, up);
    
    // Debug camera position every 60 frames
    static int debugPosCount = 0;
    debugPosCount++;
    if (debugPosCount % 60 == 0) {
        std::cout << "[DEBUG] Camera pos: (" << m_currentPosition.x << ", " 
                  << m_currentPosition.y << ", " << m_currentPosition.z << ")\n";
        std::cout << "[DEBUG] Target pos: (" << m_targetPosition.x << ", " 
                  << m_targetPosition.y << ", " << m_targetPosition.z << ")\n";
        std::cout << "[DEBUG] Forward: (" << m_forward.x << ", " 
                  << m_forward.y << ", " << m_forward.z << ")" << std::endl;
    }
}

glm::vec3 OrbitCamera::SphericalToCartesian(float yaw, float pitch, float distance) const {
    float yawRad = glm::radians(yaw);
    float pitchRad = glm::radians(pitch);
    
    // Fix coordinate system: Camera should orbit around target
    // Negative Z forward, Y up, X right (OpenGL convention)
    return glm::vec3(
        distance * cos(pitchRad) * sin(yawRad),  // X: left/right
        distance * sin(pitchRad),                // Y: up/down  
        distance * cos(pitchRad) * cos(yawRad)   // Z: forward/back
    );
}

void OrbitCamera::CartesianToSpherical(const glm::vec3& position, const glm::vec3& target, float& outYaw, float& outPitch, float& outDistance) const {
    glm::vec3 offset = position - target;
    outDistance = glm::length(offset);
    outYaw = glm::degrees(atan2(offset.x, offset.z));
    outPitch = glm::degrees(asin(offset.y / outDistance));
}

bool OrbitCamera::ValidateCameraState() {
    bool isValid = true;
    
    // Check for NaN or infinite values in position
    if (!std::isfinite(m_currentPosition.x) || !std::isfinite(m_currentPosition.y) || !std::isfinite(m_currentPosition.z)) {
        std::cout << "[ERROR] Invalid camera position detected: NaN or Inf" << std::endl;
        isValid = false;
    }
    
    // Check for NaN or infinite values in target
    if (!std::isfinite(m_targetPosition.x) || !std::isfinite(m_targetPosition.y) || !std::isfinite(m_targetPosition.z)) {
        std::cout << "[ERROR] Invalid target position detected: NaN or Inf" << std::endl;
        isValid = false;
    }
    
    // Check angles
    if (!std::isfinite(m_yaw) || !std::isfinite(m_pitch)) {
        std::cout << "[ERROR] Invalid camera angles: yaw=" << m_yaw << ", pitch=" << m_pitch << std::endl;
        isValid = false;
    }
    
    // Check distance
    if (!std::isfinite(m_currentDistance) || m_currentDistance <= 0.0f) {
        std::cout << "[ERROR] Invalid camera distance: " << m_currentDistance << std::endl;
        isValid = false;
    }
    
    // Check camera vectors
    if (!std::isfinite(m_forward.x) || !std::isfinite(m_forward.y) || !std::isfinite(m_forward.z)) {
        std::cout << "[ERROR] Invalid forward vector detected" << std::endl;
        isValid = false;
    }
    
    return isValid;
}

void OrbitCamera::InitializeCameraMode(CameraMode mode) {
    // Reset any problematic state when switching modes
    switch (mode) {
        case CameraMode::ORBIT_FOLLOW:
            // Ensure safe defaults for orbit follow
            if (m_currentDistance <= 0.0f) {
                m_currentDistance = m_orbitSettings.distance;
                m_targetDistance = m_orbitSettings.distance;
            }
            // Reset velocities to prevent jerky movement
            m_yawVelocity = 0.0f;
            m_pitchVelocity = 0.0f;
            m_distanceVelocity = 0.0f;
            break;
            
        case CameraMode::FREE_LOOK:
            // For free look, preserve current position but reset target following
            // No specific initialization needed yet
            break;
            
        case CameraMode::COMBAT_FOCUS:
            // Combat focus might need tighter following parameters
            if (m_currentDistance <= 0.0f) {
                m_currentDistance = m_orbitSettings.distance * 0.8f; // Closer for combat
                m_targetDistance = m_currentDistance;
            }
            break;
    }
    
    // Clamp angles after mode initialization
    ClampAngles();
}

} // namespace Rendering
} // namespace CudaGame

