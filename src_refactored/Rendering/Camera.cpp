#include "Rendering/Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <cmath>

namespace CudaGame {
namespace Rendering {

Camera::Camera(ProjectionType type) : m_projectionType(type) {
    UpdateMatrices();
}

void Camera::SetPosition(const glm::vec3& position) {
    m_position = position;
    UpdateViewMatrix();
    UpdateFrustum();
}

void Camera::SetRotation(const glm::vec3& rotation) {
    m_rotation = rotation;
    UpdateCameraVectors();
    UpdateViewMatrix();
    UpdateFrustum();
}

void Camera::LookAt(const glm::vec3& target, const glm::vec3& up) {
    m_viewMatrix = glm::lookAt(m_position, target, up);
    
    // Calculate rotation from view matrix
    glm::vec3 forward = glm::normalize(target - m_position);
    m_forward = forward;
    m_right = glm::normalize(glm::cross(forward, up));
    m_up = glm::normalize(glm::cross(m_right, forward));
    
    // Extract Euler angles from forward vector
    m_rotation.y = atan2(forward.x, forward.z); // Yaw
    m_rotation.x = asin(-forward.y); // Pitch
    m_rotation.z = 0.0f; // Roll (we don't calculate roll from look-at)
    
    UpdateFrustum();
}

void Camera::SetPerspective(float fov, float aspectRatio, float nearPlane, float farPlane) {
    m_projectionType = ProjectionType::PERSPECTIVE;
    m_fov = fov;
    m_aspectRatio = aspectRatio;
    m_nearPlane = nearPlane;
    m_farPlane = farPlane;
    
    UpdateProjectionMatrix();
    UpdateFrustum();
}

void Camera::SetOrthographic(float left, float right, float bottom, float top, float nearPlane, float farPlane) {
    m_projectionType = ProjectionType::ORTHOGRAPHIC;
    m_orthoLeft = left;
    m_orthoRight = right;
    m_orthoBottom = bottom;
    m_orthoTop = top;
    m_nearPlane = nearPlane;
    m_farPlane = farPlane;
    UpdateProjectionMatrix();
    UpdateFrustum();
}

void Camera::MoveForward(float distance) {
    m_position += m_forward * distance;
    UpdateViewMatrix();
    UpdateFrustum();
}

void Camera::MoveRight(float distance) {
    m_position += m_right * distance;
    UpdateViewMatrix();
    UpdateFrustum();
}

void Camera::MoveUp(float distance) {
    m_position += m_up * distance;
    UpdateViewMatrix();
    UpdateFrustum();
}

void Camera::Rotate(float yaw, float pitch, float roll) {
    m_rotation.x += pitch;
    m_rotation.y += yaw;
    m_rotation.z += roll;
    
    // Constrain pitch to avoid gimbal lock
    const float maxPitch = glm::radians(89.0f);
    m_rotation.x = glm::clamp(m_rotation.x, -maxPitch, maxPitch);
    
    UpdateCameraVectors();
    UpdateViewMatrix();
    UpdateFrustum();
}

void Camera::UpdateMatrices() {
    UpdateCameraVectors();
    UpdateViewMatrix();
    UpdateProjectionMatrix();
    UpdateFrustum();
}

void Camera::UpdateCameraVectors() {
    // Calculate the new front vector
    glm::vec3 front;
    front.x = cos(m_rotation.y) * cos(m_rotation.x);
    front.y = sin(m_rotation.x);
    front.z = sin(m_rotation.y) * cos(m_rotation.x);
    m_forward = glm::normalize(front);
    
    // Re-calculate the right and up vector
    m_right = glm::normalize(glm::cross(m_forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    m_up = glm::normalize(glm::cross(m_right, m_forward));
}

void Camera::UpdateViewMatrix() {
    m_viewMatrix = glm::lookAt(m_position, m_position + m_forward, m_up);
}

void Camera::UpdateProjectionMatrix() {
    if (m_projectionType == ProjectionType::PERSPECTIVE) {
        m_projectionMatrix = glm::perspective(glm::radians(m_fov), m_aspectRatio, m_nearPlane, m_farPlane);
    } else {
        m_projectionMatrix = glm::ortho(m_orthoLeft, m_orthoRight, m_orthoBottom, m_orthoTop, m_nearPlane, m_farPlane);
    }
}

void Camera::UpdateFrustum() {
    // Calculate frustum planes for culling
    glm::mat4 viewProj = m_projectionMatrix * m_viewMatrix;
    
    // Extract frustum planes from view-projection matrix
    // Left plane
    m_frustum.planes[0] = glm::vec4(
        viewProj[0][3] + viewProj[0][0],
        viewProj[1][3] + viewProj[1][0],
        viewProj[2][3] + viewProj[2][0],
        viewProj[3][3] + viewProj[3][0]
    );
    
    // Right plane
    m_frustum.planes[1] = glm::vec4(
        viewProj[0][3] - viewProj[0][0],
        viewProj[1][3] - viewProj[1][0],
        viewProj[2][3] - viewProj[2][0],
        viewProj[3][3] - viewProj[3][0]
    );
    
    // Bottom plane
    m_frustum.planes[2] = glm::vec4(
        viewProj[0][3] + viewProj[0][1],
        viewProj[1][3] + viewProj[1][1],
        viewProj[2][3] + viewProj[2][1],
        viewProj[3][3] + viewProj[3][1]
    );
    
    // Top plane
    m_frustum.planes[3] = glm::vec4(
        viewProj[0][3] - viewProj[0][1],
        viewProj[1][3] - viewProj[1][1],
        viewProj[2][3] - viewProj[2][1],
        viewProj[3][3] - viewProj[3][1]
    );
    
    // Near plane
    m_frustum.planes[4] = glm::vec4(
        viewProj[0][3] + viewProj[0][2],
        viewProj[1][3] + viewProj[1][2],
        viewProj[2][3] + viewProj[2][2],
        viewProj[3][3] + viewProj[3][2]
    );
    
    // Far plane
    m_frustum.planes[5] = glm::vec4(
        viewProj[0][3] - viewProj[0][2],
        viewProj[1][3] - viewProj[1][2],
        viewProj[2][3] - viewProj[2][2],
        viewProj[3][3] - viewProj[3][2]
    );
    
    // Normalize planes
    for (int i = 0; i < 6; i++) {
        float length = glm::length(glm::vec3(m_frustum.planes[i]));
        m_frustum.planes[i] /= length;
    }
    
}

} // namespace Rendering
} // namespace CudaGame
