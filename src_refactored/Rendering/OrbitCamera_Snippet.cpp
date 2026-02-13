#include "Rendering/OrbitCamera.h"
#include <algorithm>
#include <iostream>

namespace CudaGame {
namespace Rendering {

// ... (Existing code) ...

void OrbitCamera::SetViewAngles(float yaw, float pitch) {
    m_yaw = yaw;
    m_pitch = std::clamp(pitch, m_orbitSettings.minPitch, m_orbitSettings.maxPitch);
    m_previousYaw = m_yaw;
    m_previousPitch = m_pitch;
    UpdateCameraVectorsFromPosition();
}
// Note: This is partial content, I should use replace_file_content to insert it into existing file.
