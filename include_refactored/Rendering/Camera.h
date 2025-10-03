#pragma once

namespace CudaGame {
namespace Rendering {

enum class ProjectionType {
    PERSPECTIVE,
    ORTHOGRAPHIC
};

struct PerspectiveParams {
    float fov;
    float aspectRatio;
    float nearPlane;
    float farPlane;
};

class Camera {
public:
    virtual ~Camera() = default;
    ProjectionType GetProjectionType() const { return m_projectionType; }
    const PerspectiveParams& GetPerspectiveParams() const { return m_perspParams; }

protected:
    ProjectionType m_projectionType;
    PerspectiveParams m_perspParams;
};

} // namespace Rendering
} // namespace CudaGame

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace CudaGame {
namespace Rendering {

enum class ProjectionType {
    PERSPECTIVE,
    ORTHOGRAPHIC
};

class Camera {
public:
    Camera(ProjectionType type = ProjectionType::PERSPECTIVE);
    ~Camera() = default;

    // Camera transformation methods
    void SetPosition(const glm::vec3& position);
    void SetRotation(const glm::vec3& rotation); // Euler angles
    void LookAt(const glm::vec3& target, const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f));

    // Projection settings
    void SetPerspective(float fov, float aspectRatio, float nearPlane, float farPlane);
    void SetOrthographic(float left, float right, float bottom, float top, float nearPlane, float farPlane);

    // Movement methods (for first-person cameras)
    void MoveForward(float distance);
    void MoveRight(float distance);
    void MoveUp(float distance);
    void Rotate(float yaw, float pitch, float roll = 0.0f);

    // Getters
    const glm::vec3& GetPosition() const { return m_position; }
    const glm::vec3& GetRotation() const { return m_rotation; }
    const glm::vec3& GetForward() const { return m_forward; }
    const glm::vec3& GetRight() const { return m_right; }
    const glm::vec3& GetUp() const { return m_up; }

    const glm::mat4& GetViewMatrix() const { return m_viewMatrix; }
    const glm::mat4& GetProjectionMatrix() const { return m_projectionMatrix; }
    glm::mat4 GetViewProjectionMatrix() const { return m_projectionMatrix * m_viewMatrix; }

    // Projection parameters
    float GetFOV() const { return m_fov; }
    float GetAspectRatio() const { return m_aspectRatio; }
    float GetNearPlane() const { return m_nearPlane; }
    float GetFarPlane() const { return m_farPlane; }

    // Frustum culling support
    struct Frustum {
        glm::vec4 planes[6]; // left, right, bottom, top, near, far
    };
    const Frustum& GetFrustum() const { return m_frustum; }

    // Update camera matrices (call after any transformation)
    void UpdateMatrices();

protected:
    // Camera properties
    glm::vec3 m_position{0.0f, 0.0f, 0.0f};
    glm::vec3 m_rotation{0.0f, 0.0f, 0.0f}; // Pitch, Yaw, Roll

    // Camera vectors
    glm::vec3 m_forward{0.0f, 0.0f, -1.0f};
    glm::vec3 m_right{1.0f, 0.0f, 0.0f};
    glm::vec3 m_up{0.0f, 1.0f, 0.0f};

    // Matrices
    glm::mat4 m_viewMatrix{1.0f};
    glm::mat4 m_projectionMatrix{1.0f};

private:

    // Projection settings
    ProjectionType m_projectionType;
    float m_fov = 45.0f;
    float m_aspectRatio = 16.0f / 9.0f;
    float m_nearPlane = 0.1f;
    float m_farPlane = 1000.0f;

    // Orthographic settings
    float m_orthoLeft = -10.0f;
    float m_orthoRight = 10.0f;
    float m_orthoBottom = -10.0f;
    float m_orthoTop = 10.0f;

    // Frustum for culling
    Frustum m_frustum;

    // Helper methods
    void UpdateCameraVectors();
    void UpdateViewMatrix();
    void UpdateProjectionMatrix();
    void UpdateFrustum();
};

} // namespace Rendering
} // namespace CudaGame
