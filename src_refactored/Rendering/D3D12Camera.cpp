#ifdef _WIN32
#include "Rendering/Camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace CudaGame {
namespace Rendering {

// Override UpdateProjectionMatrix to use D3D12-compatible projection
class D3D12Camera : public Camera {
public:
    D3D12Camera() : Camera(ProjectionType::PERSPECTIVE) {}
    
    void UpdateProjectionMatrix() {
        if (GetFOV() > 0) {
            // Use D3D12 left-handed, zero-to-one depth range projection
            m_projectionMatrix = glm::perspectiveLH_ZO(
                glm::radians(GetFOV()), 
                GetAspectRatio(), 
                GetNearPlane(), 
                GetFarPlane()
            );
        }
    }
    
    void SetPerspective(float fov, float aspectRatio, float nearPlane, float farPlane) {
        Camera::SetPerspective(fov, aspectRatio, nearPlane, farPlane);
        UpdateProjectionMatrix();  // Override with D3D12 projection
    }
};

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
