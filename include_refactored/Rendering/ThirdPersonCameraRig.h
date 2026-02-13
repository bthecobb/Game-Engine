#pragma once

#include "Rendering/OrbitCamera.h"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <cmath>

namespace CudaGame {
namespace Rendering {

class ThirdPersonCameraRig {
public:
    struct Settings {
        float distance = 4.5f;        // behind the player
        float height = 1.6f;          // above the player
        float shoulderOffsetX = 0.6f; // to the right (negative for left)
        float softZoneWidth = 0.6f;   // meters in camera-right direction (unused by default)
        float softZoneForward = 0.9f; // meters in camera-forward direction (unused by default)
        float softZoneHeight = 0.4f;  // meters in camera-up direction (unused by default)
        bool  enableLateralSoftZone = false; // off by default
        bool  enableVerticalSoftZone = false; // off by default
        float followSmooth = 10.0f;   // how fast anchor XZ follows player (1/s)
        float verticalSmooth = 8.0f;  // how fast camera height follows player (1/s)
        float smoothSpeed = 12.0f;    // handed to OrbitCamera
        // Control yaw smoothing (critically damped), in Hz
        float controlYawSmooth = 12.0f;
        // Center bias when idle: how quickly bias engages (1/s) and max reduction fraction (0..1)
        float centerBiasGain = 1.2f;
        float centerBiasMax = 0.6f;
        // Additional smoothing to reduce jitter
        float shoulderSmooth = 10.0f; // 1/s smoothing of dynamic shoulder scale
        float targetSmooth = 18.0f;   // 1/s 2nd-order smoothing toward final target
        bool  enableDynamicShoulder = true; // allow lateral reduction while strafing
        // Idle detection thresholds
        float idleSpeedThreshold = 0.05f;   // m/s
        float yawNoiseDegThreshold = 0.05f; // deg/frame considered as no yaw input
        // Safety: cap the per-frame movement of final target to avoid spikes on rapid input flips (meters per frame; <=0 disables)
        float maxTargetStep = 1.5f;
    };

    ThirdPersonCameraRig() = default;

    void SetCamera(OrbitCamera* cam) { m_camera = cam; }
    void Configure(const Settings& s) { m_settings = s; }

    // Call once per frame, after physics, before rendering
    void Update(float dt, const glm::vec3& playerWorldPos, const glm::vec3& playerVelocity) {
        if (!m_camera) return;

        auto isFinite3 = [](const glm::vec3& v){ return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); };

        if (!m_hasAnchor || !isFinite3(m_anchor)) {
            m_anchor = playerWorldPos;
            m_anchorVelXZ = glm::vec2(0.0f);
            m_hasAnchor = true;
        }

        // Get camera axes (we'll replace right with control-yaw-smoothed right later)
        glm::vec3 camRightInstant = m_camera->GetRight();
        if (!isFinite3(camRightInstant) || glm::dot(camRightInstant, camRightInstant) < 1e-6f) camRightInstant = glm::vec3(1,0,0);
        const glm::vec3 camUp          = m_camera->GetUp();

        // Smooth-follow anchor in XZ using critically damped 2nd-order dynamics
        float dtClamped = glm::clamp(dt, 0.0f, 0.05f);
        // Precompute natural frequencies (rad/s) for stability-aware substepping
        float wFollow = 2.0f * glm::pi<float>() * glm::max(0.1f, m_settings.followSmooth);
        float wYaw    = 2.0f * glm::pi<float>() * glm::max(0.1f, m_settings.controlYawSmooth);
        float wTarget = 2.0f * glm::pi<float>() * glm::max(0.1f, m_settings.targetSmooth);
        float wMax    = glm::max(wFollow, glm::max(wYaw, wTarget));
        // Choose substeps so that wMax * step <= ~0.5 for better numerical stability on stiff settings
        int substeps = std::max(1, std::min((int)std::ceil((wMax * dtClamped) / 0.5f), 6));
        float step   = (substeps > 0) ? (dtClamped / substeps) : dtClamped;

        glm::vec2 xz(m_anchor.x, m_anchor.z);
        glm::vec2 targetXZ(playerWorldPos.x, playerWorldPos.z);
        for (int i = 0; i < substeps; ++i) {
            glm::vec2 a = (wFollow*wFollow) * (targetXZ - xz) - 2.0f * wFollow * m_anchorVelXZ;
            m_anchorVelXZ += a * step;
            xz += m_anchorVelXZ * step;
        }
        m_anchor.x = xz.x;
        m_anchor.z = xz.y;

        // Vertical anchor smoothing based on feet height to avoid handheld bobbing
        const float kHalfHeight = 0.9f; // matches player collider
        float desiredFeetY = playerWorldPos.y - kHalfHeight;
        if (!m_hasAnchorY) { m_anchorY = desiredFeetY; m_hasAnchorY = true; }
        float k = 1.0f - std::exp(-m_settings.verticalSmooth * dt);
        k = glm::clamp(k, 0.0f, 1.0f);
        m_anchorY = glm::mix(m_anchorY, desiredFeetY, k);

        // Control yaw smoothing (critically damped)
        auto shortestAngleDelta = [](float aDeg, float bDeg) {
            float d = bDeg - aDeg;
            while (d > 180.0f) d -= 360.0f;
            while (d < -180.0f) d += 360.0f;
            return d;
        };
        float rawYaw = m_camera->GetYaw();
        if (!m_hasControlYaw) {
            m_controlYawDeg = rawYaw;
            m_controlYawVel = 0.0f;
            m_hasControlYaw = true;
            m_prevCameraYawDeg = rawYaw;
        }
        for (int i = 0; i < substeps; ++i) {
            float err = shortestAngleDelta(m_controlYawDeg, rawYaw);
            float a = (wYaw*wYaw) * err - 2.0f * wYaw * m_controlYawVel;
            m_controlYawVel += a * step;
            m_controlYawDeg += m_controlYawVel * step;
            // Normalize to [-180,180]
            if (m_controlYawDeg > 180.0f) m_controlYawDeg -= 360.0f;
            if (m_controlYawDeg < -180.0f) m_controlYawDeg += 360.0f;
        }
        float yawRad = m_controlYawDeg * (glm::pi<float>()/180.0f);
        glm::vec3 controlRight = glm::normalize(glm::vec3(std::sin(yawRad + glm::half_pi<float>()), 0.0f, std::cos(yawRad + glm::half_pi<float>())));
        if (!isFinite3(controlRight) || glm::dot(controlRight, controlRight) < 1e-6f) controlRight = camRightInstant;

        // Gradually blend in control-right to avoid initial snap
        m_controlBlend = glm::clamp(m_controlBlend + dt * 5.0f, 0.0f, 1.0f); // ~200ms fade-in
        glm::vec3 blendedRight = glm::normalize(glm::mix(camRightInstant, controlRight, m_controlBlend));
        if (!isFinite3(blendedRight) || glm::dot(blendedRight, blendedRight) < 1e-6f) blendedRight = camRightInstant;

        // Dynamic shoulder scale: reduce offset when strafing sideways to avoid lateral tugging
        glm::vec2 vXZ(playerVelocity.x, playerVelocity.z);
        float vLen = glm::length(vXZ);
        float shoulderScaleRaw = 1.0f;
        if (m_settings.enableDynamicShoulder && vLen > 0.001f) {
            glm::vec2 vDir = vXZ / glm::max(vLen, 1e-6f);
            glm::vec2 rightXZ = glm::normalize(glm::vec2(controlRight.x, controlRight.z));
            float side = std::abs(glm::dot(vDir, rightXZ)); // 0=forward, 1=full strafe
            shoulderScaleRaw = 1.0f - 0.7f * side; // keep 30% at full strafe
        }
        // Smooth shoulder scale to avoid lateral jitter
        float sShoulder = 1.0f - std::exp(-m_settings.shoulderSmooth * dt);
        sShoulder = glm::clamp(sShoulder, 0.0f, 1.0f);
        m_shoulderScaleSmoothed = glm::mix(m_shoulderScaleSmoothed, shoulderScaleRaw, sShoulder);

        // Center bias when idle (no movement and no yaw input for a while)
        float yawDeltaNow = std::abs(shortestAngleDelta(m_prevCameraYawDeg, rawYaw));
        bool idle = (vLen < m_settings.idleSpeedThreshold) && (yawDeltaNow < m_settings.yawNoiseDegThreshold);
        if (idle) {
            m_idleTimer += dt;
        } else {
            m_idleTimer = 0.0f;
        }
        float biasFactor = 1.0f - std::exp(-m_settings.centerBiasGain * m_idleTimer); // 0..1
        float biasReduction = glm::clamp(m_settings.centerBiasMax * biasFactor, 0.0f, m_settings.centerBiasMax);

        float shoulder = m_settings.shoulderOffsetX * m_shoulderScaleSmoothed * (1.0f - biasReduction);

        // Build desired target
        glm::vec3 targetDesired = glm::vec3(m_anchor.x, m_anchorY + m_settings.height, m_anchor.z) + blendedRight * shoulder;
        if (!isFinite3(targetDesired)) {
            // Robust fallback
            targetDesired = playerWorldPos + glm::vec3(0.0f, m_settings.height, 0.0f);
        }
        // Smooth final target with critically damped dynamics
        if (!m_hasTargetPos) { m_targetPos = targetDesired; m_targetVel = glm::vec3(0.0f); m_hasTargetPos = true; }
        for (int i = 0; i < substeps; ++i) {
            glm::vec3 a = (wTarget*wTarget) * (targetDesired - m_targetPos) - 2.0f * wTarget * m_targetVel;
            m_targetVel += a * step;
            m_targetPos += m_targetVel * step;
        }
        // Clamp per-frame movement of the final target to avoid oscillation spikes during rapid flips
        if (m_settings.maxTargetStep > 0.0f && m_hasLastTarget) {
            glm::vec3 delta = m_targetPos - m_lastTarget;
            float len = glm::length(delta);
            if (len > m_settings.maxTargetStep) {
                m_targetPos = m_lastTarget + (delta * (m_settings.maxTargetStep / glm::max(len, 1e-6f)));
            }
        }
        m_camera->SetTarget(m_targetPos);

        // Ensure camera uses requested distance and smoothing
        auto orbit = m_camera->GetOrbitSettings();
        orbit.smoothSpeed = m_settings.smoothSpeed;
        orbit.heightOffset = 0.0f; // height handled by rig
        orbit.minDistance = std::min(orbit.minDistance, m_settings.distance * 0.5f);
        orbit.maxDistance = std::max(orbit.maxDistance, m_settings.distance * 2.0f);
        m_camera->SetOrbitSettings(orbit);
        m_camera->SetDistance(m_settings.distance, false);

        m_camera->Update(dt, m_targetPos, playerVelocity);
        m_lastTarget = m_targetPos;
        m_hasLastTarget = true;
        m_prevCameraYawDeg = rawYaw;
    }

private:
    OrbitCamera* m_camera = nullptr;
    Settings m_settings{};

    bool m_hasAnchor = false;
    glm::vec3 m_anchor{0.0f};
    bool m_hasAnchorY = false;
    float m_anchorY = 0.0f;

    // 2nd-order smoothing state for XZ
    glm::vec2 m_anchorVelXZ{0.0f, 0.0f};

    bool m_hasLastTarget = false;
    glm::vec3 m_lastTarget{0.0f};

    // Control yaw smoothing state
    bool  m_hasControlYaw = false;
    float m_controlYawDeg = 0.0f;
    float m_controlYawVel = 0.0f;
    float m_prevCameraYawDeg = 0.0f;

    // Center bias state
    float m_idleTimer = 0.0f;

    // Fade-in for control-right to avoid initial snap
    float m_controlBlend = 0.0f;

    // Shoulder smoothing state
    float m_shoulderScaleSmoothed = 1.0f;

    // Target smoothing state
    bool  m_hasTargetPos = false;
    glm::vec3 m_targetPos{0.0f};
    glm::vec3 m_targetVel{0.0f};
};

} // namespace Rendering
} // namespace CudaGame
