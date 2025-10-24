#pragma once

#include <chrono>
#include <vector>
#include <functional>

enum class RhythmTiming {
    PERFECT,    // ±50ms
    GOOD,      // ±100ms  
    OKAY,      // ±200ms
    MISS       // Outside timing window
};

struct RhythmAction {
    std::chrono::high_resolution_clock::time_point timestamp;
    RhythmTiming timing;
    float multiplier;
};

class RhythmSystem {
public:
    RhythmSystem(float bpm = 140.0f);
    ~RhythmSystem() = default;
    
    // Core functionality
    void initialize();
    void update(float deltaTime);
    
    // Timing analysis
    RhythmTiming analyzeActionTiming() const;
    float getTimingMultiplier(RhythmTiming timing) const;
    bool isOnBeat(float toleranceMs = 100.0f) const;
    
    // Beat information
    float getBeatProgress() const; // 0.0 to 1.0 within current beat
    int getCurrentBeat() const;
    float getTimeToNextBeat() const;
    
    // Visual/Audio feedback
    float getBeatIntensity() const; // For visual pulsing effects
    bool shouldTriggerBeatEffect() const;
    
    // Combat integration
    float getAttackMultiplier() const;
    float getSpeedMultiplier() const;
    void registerAction(RhythmTiming timing);
    
    // Combo system
    int getComboCount() const { return m_comboCount; }
    float getComboMultiplier() const;
    void resetCombo();
    
    // Settings
    void setBPM(float bpm);
    void setVolume(float volume) { m_volume = volume; }
    void setEnabled(bool enabled) { m_enabled = enabled; }

private:
    // Timing
    float m_bpm;
    float m_beatDuration; // Time between beats in seconds
    std::chrono::high_resolution_clock::time_point m_startTime;
    std::chrono::high_resolution_clock::time_point m_lastBeatTime;
    
    // State
    bool m_enabled = true;
    float m_volume = 0.3f; // Subtle background presence
    int m_currentBeat = 0;
    
    // Timing windows (in milliseconds)
    float m_perfectWindow = 50.0f;
    float m_goodWindow = 100.0f;
    float m_okayWindow = 200.0f;
    
    // Multipliers
    float m_perfectMultiplier = 1.25f;
    float m_goodMultiplier = 1.10f;
    float m_okayMultiplier = 1.0f;
    float m_missMultiplier = 0.8f;
    
    // Speed bonuses
    float m_perfectSpeedBonus = 0.2f;
    float m_goodSpeedBonus = 0.1f;
    
    // Combo system
    int m_comboCount = 0;
    int m_maxCombo = 0;
    float m_comboDecayTime = 2.0f; // Time before combo starts decaying
    float m_timeSinceLastAction = 0.0f;
    std::vector<RhythmAction> m_recentActions;
    
    // Visual effects
    bool m_beatJustTriggered = false;
    float m_beatIntensity = 0.0f;
    
    // Internal methods
    void updateBeatTracking(float deltaTime);
    void updateComboSystem(float deltaTime);
    void updateVisualEffects(float deltaTime);
    float calculateCurrentTime() const;
    float getTimeToBeat(float currentTime) const;
    bool isBeatTime(float currentTime, float tolerance) const;
    void triggerBeatEffect();
    
    // Combo calculations
    float calculateComboMultiplier(int combo) const;
    void decayCombo(float deltaTime);
};
