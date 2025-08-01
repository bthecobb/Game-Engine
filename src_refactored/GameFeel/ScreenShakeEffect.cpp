#include "GameFeel/ScreenShakeEffect.h"
#include <cmath>
#include <algorithm>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <glm/gtc/matrix_transform.hpp>

namespace CudaGame {
namespace GameFeel {

// Helper functions to replace missing glm functions
static float linearRand(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

static float perlin(float x) {
    // Simple pseudo-perlin noise implementation
    float intPart = std::floor(x);
    float fracPart = x - intPart;
    
    // Smooth interpolation
    float fade = fracPart * fracPart * fracPart * (fracPart * (fracPart * 6.0f - 15.0f) + 10.0f);
    
    // Hash function for pseudo-random values
    auto hash = [](int x) -> float {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return (x & 0x7fffffff) / float(0x7fffffff) * 2.0f - 1.0f;
    };
    
    float a = hash(static_cast<int>(intPart));
    float b = hash(static_cast<int>(intPart) + 1);
    
    return a + fade * (b - a);
}

static float perlin(const glm::vec2& v) {
    return perlin(v.x + v.y * 57.0f);
}

static float perlin(const glm::vec1& v) {
    return perlin(v.x);
}

// ScreenShakeEffect implementation
ScreenShakeEffect::ScreenShakeEffect(ShakeType type, float intensity, float duration, float frequency)
    : m_type(type), m_intensity(intensity), m_duration(duration), m_frequency(frequency) {
    
    // Initialize random seeds for noise-based shakes
    m_xSeed = linearRand(0.0f, 1000.0f);
    m_ySeed = linearRand(0.0f, 1000.0f);
}

void ScreenShakeEffect::Update(float deltaTime) {
    m_timer += deltaTime;
}

glm::vec2 ScreenShakeEffect::GetOffset() const {
    if (IsFinished()) {
        return glm::vec2(0.0f);
    }
    
    switch (m_type) {
        case ShakeType::Random:
            return GetRandomOffset();
        case ShakeType::Directional:
            return GetDirectionalOffset();
        case ShakeType::Punch:
            return GetPunchOffset();
        case ShakeType::Trauma:
            return GetTraumaOffset();
        case ShakeType::Explosion:
            return GetExplosionOffset();
        default:
            return GetRandomOffset();
    }
}

void ScreenShakeEffect::SetNoiseSettings(float xSeed, float ySeed) {
    m_xSeed = xSeed;
    m_ySeed = ySeed;
}

glm::vec2 ScreenShakeEffect::GetRandomOffset() const {
    float decay = GetDecayFactor();
    float currentIntensity = m_intensity * decay;
    
    float x = perlin(glm::vec2(m_xSeed, m_timer * m_frequency)) * currentIntensity;
    float y = perlin(glm::vec2(m_ySeed, m_timer * m_frequency)) * currentIntensity;
    
    return glm::vec2(x, y);
}

glm::vec2 ScreenShakeEffect::GetDirectionalOffset() const {
    float decay = GetDecayFactor();
    float currentIntensity = m_intensity * decay;
    
    // Oscillate along the direction vector
    float oscillation = std::sin(m_timer * m_frequency * 2.0f * M_PI);
    return m_direction * currentIntensity * oscillation;
}

glm::vec2 ScreenShakeEffect::GetPunchOffset() const {
    // Sharp, quick decay for punch effects
    float progress = m_timer / m_duration;
    float decay = std::pow(1.0f - progress, 4.0f); // Sharp falloff
    
    float x = perlin(glm::vec2(m_xSeed, m_timer * m_frequency * 2.0f)) * m_intensity * decay;
    float y = perlin(glm::vec2(m_ySeed, m_timer * m_frequency * 2.0f)) * m_intensity * decay;
    
    return glm::vec2(x, y);
}

glm::vec2 ScreenShakeEffect::GetTraumaOffset() const {
    float progress = m_timer / m_duration;
    float decay = std::pow(1.0f - progress, m_decayExponent);
    float currentIntensity = m_intensity * decay;
    
    float x = perlin(glm::vec2(m_xSeed, m_timer * m_frequency)) * currentIntensity;
    float y = perlin(glm::vec2(m_ySeed, m_timer * m_frequency)) * currentIntensity;
    
    return glm::vec2(x, y);
}

glm::vec2 ScreenShakeEffect::GetExplosionOffset() const {
    float decay = GetDecayFactor();
    float currentIntensity = m_intensity * decay;
    
    // Radial shake pattern
    float angle = m_timer * m_frequency;
    float radius = perlin(glm::vec1(m_timer * m_frequency * 0.5f)) * currentIntensity;
    
    return glm::vec2(
        std::cos(angle) * radius,
        std::sin(angle) * radius
    );
}

float ScreenShakeEffect::GetDecayFactor() const {
    float progress = m_timer / m_duration;
    return std::max(0.0f, 1.0f - progress);
}

// ScreenShakeManager implementation
void ScreenShakeManager::Update(float deltaTime) {
    // Update all active shakes
    for (auto& shake : m_activeShakes) {
        shake.Update(deltaTime);
    }
    
    // Remove finished shakes
    RemoveFinishedShakes();
}

void ScreenShakeManager::AddShake(const ScreenShakeEffect& shake) {
    m_activeShakes.push_back(shake);
}

void ScreenShakeManager::ClearAllShakes() {
    m_activeShakes.clear();
}

glm::vec2 ScreenShakeManager::GetTotalOffset() const {
    glm::vec2 totalOffset(0.0f);
    
    for (const auto& shake : m_activeShakes) {
        totalOffset += shake.GetOffset();
    }
    
    return totalOffset;
}

glm::mat4 ScreenShakeManager::GetShakeMatrix() const {
    glm::vec2 offset = GetTotalOffset();
    return glm::translate(glm::mat4(1.0f), glm::vec3(offset, 0.0f));
}

bool ScreenShakeManager::HasActiveShakes() const {
    return !m_activeShakes.empty();
}

void ScreenShakeManager::AddRandomShake(float intensity, float duration, float frequency) {
    ScreenShakeEffect shake(ShakeType::Random, intensity, duration, frequency);
    AddShake(shake);
}

void ScreenShakeManager::AddDirectionalShake(const glm::vec2& direction, float intensity, float duration) {
    ScreenShakeEffect shake(ShakeType::Directional, intensity, duration);
    shake.SetDirection(glm::normalize(direction));
    AddShake(shake);
}

void ScreenShakeManager::AddPunchShake(float intensity, float duration) {
    ScreenShakeEffect shake(ShakeType::Punch, intensity, duration, 30.0f); // Higher frequency for punch
    AddShake(shake);
}

void ScreenShakeManager::AddTraumaShake(float intensity, float duration, float decay) {
    ScreenShakeEffect shake(ShakeType::Trauma, intensity, duration);
    shake.SetDecayFunction(decay);
    AddShake(shake);
}

void ScreenShakeManager::AddExplosionShake(float intensity, float duration) {
    ScreenShakeEffect shake(ShakeType::Explosion, intensity, duration, 15.0f);
    AddShake(shake);
}

void ScreenShakeManager::RemoveFinishedShakes() {
    m_activeShakes.erase(
        std::remove_if(m_activeShakes.begin(), m_activeShakes.end(),
            [](const ScreenShakeEffect& shake) { return shake.IsFinished(); }),
        m_activeShakes.end()
    );
}

} // namespace GameFeel
} // namespace CudaGame
