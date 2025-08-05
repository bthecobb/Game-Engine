#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace CudaGame {
namespace GameFeel {

// Different types of screen shake patterns
enum class ShakeType {
    Random,     // Random noise-based shake
    Directional, // Shake in a specific direction
    Punch,      // Quick, sharp shake
    Trauma,     // Exponential decay shake
    Explosion   // Radial shake pattern
};

// Individual screen shake effect
class ScreenShakeEffect {
public:
    ScreenShakeEffect(ShakeType type, float intensity, float duration, float frequency = 20.0f);
    ~ScreenShakeEffect() = default;

    void Update(float deltaTime);
    glm::vec2 GetOffset() const;
    bool IsFinished() const { return m_timer >= m_duration; }

    // Setters for different shake types
    void SetDirection(const glm::vec2& direction) { m_direction = direction; }
    void SetDecayFunction(float exponent) { m_decayExponent = exponent; }
    void SetNoiseSettings(float xSeed, float ySeed);

private:
    ShakeType m_type;
    float m_intensity;
    float m_duration;
    float m_frequency;
    float m_timer = 0.0f;
    float m_decayExponent = 2.0f;
    
    // Direction for directional shakes
    glm::vec2 m_direction{0.0f, 1.0f};
    
    // Noise seeds for random shakes
    float m_xSeed = 0.0f;
    float m_ySeed = 100.0f;
    
    // Helper methods
    glm::vec2 GetRandomOffset() const;
    glm::vec2 GetDirectionalOffset() const;
    glm::vec2 GetPunchOffset() const;
    glm::vec2 GetTraumaOffset() const;
    glm::vec2 GetExplosionOffset() const;
    float GetDecayFactor() const;
};

// Manages multiple screen shake effects
class ScreenShakeManager {
public:
    ScreenShakeManager() = default;
    ~ScreenShakeManager() = default;

    void Update(float deltaTime);
    void AddShake(const ScreenShakeEffect& shake);
    void ClearAllShakes();
    
    glm::vec2 GetTotalOffset() const;
    glm::mat4 GetShakeMatrix() const;
    
    bool HasActiveShakes() const;
    int GetActiveShakeCount() const { return static_cast<int>(m_activeShakes.size()); }

    // Convenience methods for common shake types
    void AddRandomShake(float intensity, float duration, float frequency = 20.0f);
    void AddDirectionalShake(const glm::vec2& direction, float intensity, float duration);
    void AddPunchShake(float intensity, float duration = 0.2f);
    void AddTraumaShake(float intensity, float duration, float decay = 2.0f);
    void AddExplosionShake(float intensity, float duration);

private:
    std::vector<ScreenShakeEffect> m_activeShakes;
    
    void RemoveFinishedShakes();
};

} // namespace GameFeel
} // namespace CudaGame
