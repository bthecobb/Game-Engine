#include "Rendering/OrbitCamera.h"
#include <glm/gtx/euler_angles.hpp>
#include <cmath>
#include <iostream>

namespace CudaGame {
namespace Rendering {

OrbitCamera::OrbitCamera(ProjectionType type) : Camera(type),
    m_currentMode(CameraMode::ORBIT_FOLLOW),
    m_yaw(0.0f),
    m_pitch(0.0f),
    m_targetDistance(m_orbitSettings.distance),
    m_currentDistance(m_orbitSettings.distance),
    m_yawVelocity(0.0f),
    m_pitchVelocity(0.0f),
    m_distanceVelocity(0.0f),
    m_debugVisualization(false) {

    m_targetPosition = glm::vec3(0.0f);
    m_desiredPosition = glm::vec3(0.0f);
    m_currentPosition = glm::vec3(0.0f);
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
    m_currentMode = mode;
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
    // Additional logic for free look if needed
}

void OrbitCamera::UpdateCombatFocus(float deltaTime, const glm::vec3& targetPosition, const glm::vec3& targetVelocity) {
    // Additional logic for combat focus mode
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

} // namespace Rendering
} // namespace CudaGame

