#pragma once

#include "Core/System.h"
#include "Rendering/Camera.h"
#include "GameFeel/ScreenShakeEffect.h"
#include <glm/glm.hpp>

namespace CudaGame {
namespace GameFeel {

// Manages hit-stop, screen shake, and other game feel effects
class GameFeelSystem : public Core::System {
public:
    GameFeelSystem();
    ~GameFeelSystem() = default;

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;
    
    // Configuration method for integration
    void Configure() {}

    // Hit-stop effects
    void ApplyHitStop(float duration);
    bool IsInHitStop() const { return m_hitstopTimer > 0.0f; }
    float GetHitStopMultiplier() const { return IsInHitStop() ? 0.0f : 1.0f; }

    // Screen shake effects
    void TriggerScreenShake(float intensity, float duration, float frequency = 20.0f);
    void TriggerPunchShake(float intensity, float duration = 0.2f);
    void TriggerTraumaShake(float intensity, float duration, float decay = 2.0f);
    void TriggerDirectionalShake(const glm::vec2& direction, float intensity, float duration);
    void TriggerExplosionShake(float intensity, float duration);
    
    glm::mat4 GetScreenShakeMatrix() const;
    bool IsShaking() const;

    // Time dilation effects (slow-motion)
    void SetTimeScale(float scale, float duration = 0.0f);
    float GetTimeScale() const { return m_timeScale; }
    bool IsTimeDilated() const { return m_timeScale < 1.0f; }
    
    // Global settings
    void SetGlobalHitStopMultiplier(float multiplier) { m_globalHitstopMultiplier = multiplier; }
    void SetGlobalShakeMultiplier(float multiplier) { m_globalShakeMultiplier = multiplier; }

    // Debugging
    void SetDebugVisualization(bool enable);

private:
    // Hit-stop
    float m_hitstopTimer = 0.0f;
    float m_globalHitstopMultiplier = 1.0f;

    // Screen shake
    ScreenShakeManager m_shakeManager;
    float m_globalShakeMultiplier = 1.0f;
    
    // Time dilation
    float m_timeScale = 1.0f;
    float m_timeScaleDuration = 0.0f;
    float m_timeScaleTimer = 0.0f;
    
    // Debugging
    bool m_debugVisualization = false;
    
    // Private update methods
    void UpdateHitStop(float deltaTime);
    void UpdateScreenShake(float deltaTime);
    void UpdateTimeDilation(float deltaTime);
};

} // namespace GameFeel
} // namespace CudaGame

