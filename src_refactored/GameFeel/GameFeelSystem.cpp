#include "GameFeel/GameFeelSystem.h"
#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>
#include <iostream>

namespace CudaGame {
namespace GameFeel {

GameFeelSystem::GameFeelSystem() {
    // Game feel should update after physics but before rendering
}

bool GameFeelSystem::Initialize() {
    std::cout << "[GameFeelSystem] Initializing game feel effects..." << std::endl;
    return true;
}

void GameFeelSystem::Shutdown() {
    std::cout << "[GameFeelSystem] Shutting down game feel effects." << std::endl;
}

void GameFeelSystem::Update(float deltaTime) {
    float scaledDeltaTime = deltaTime * m_timeScale;
    
    UpdateHitStop(scaledDeltaTime);
    UpdateScreenShake(scaledDeltaTime);
    UpdateTimeDilation(scaledDeltaTime);
}

// Hit-stop effects
void GameFeelSystem::ApplyHitStop(float duration) {
    m_hitstopTimer = std::max(m_hitstopTimer, duration * m_globalHitstopMultiplier);
}

void GameFeelSystem::UpdateHitStop(float deltaTime) {
    if (m_hitstopTimer > 0.0f) {
        m_hitstopTimer -= deltaTime;
    }
}

// Screen shake effects
void GameFeelSystem::TriggerScreenShake(float intensity, float duration, float frequency) {
    m_shakeManager.AddRandomShake(intensity * m_globalShakeMultiplier, duration, frequency);
}

void GameFeelSystem::TriggerPunchShake(float intensity, float duration) {
    m_shakeManager.AddPunchShake(intensity * m_globalShakeMultiplier, duration);
}

void GameFeelSystem::TriggerTraumaShake(float intensity, float duration, float decay) {
    m_shakeManager.AddTraumaShake(intensity * m_globalShakeMultiplier, duration, decay);
}

void GameFeelSystem::TriggerDirectionalShake(const glm::vec2& direction, float intensity, float duration) {
    m_shakeManager.AddDirectionalShake(direction, intensity * m_globalShakeMultiplier, duration);
}

void GameFeelSystem::TriggerExplosionShake(float intensity, float duration) {
    m_shakeManager.AddExplosionShake(intensity * m_globalShakeMultiplier, duration);
}

glm::mat4 GameFeelSystem::GetScreenShakeMatrix() const {
    return m_shakeManager.GetShakeMatrix();
}

bool GameFeelSystem::IsShaking() const {
    return m_shakeManager.HasActiveShakes();
}

void GameFeelSystem::UpdateScreenShake(float deltaTime) {
    m_shakeManager.Update(deltaTime);
}

// Time dilation effects
void GameFeelSystem::SetTimeScale(float scale, float duration) {
    m_timeScale = scale;
    m_timeScaleDuration = duration;
    m_timeScaleTimer = 0.0f;
}

void GameFeelSystem::UpdateTimeDilation(float deltaTime) {
    if (m_timeScaleDuration > 0.0f) {
        m_timeScaleTimer += deltaTime;
        if (m_timeScaleTimer >= m_timeScaleDuration) {
            m_timeScale = 1.0f;
            m_timeScaleDuration = 0.0f;
        }
    }
}

void GameFeelSystem::SetDebugVisualization(bool enable) {
    m_debugVisualization = enable;
}

} // namespace GameFeel
} // namespace CudaGame
