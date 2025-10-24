#include "RhythmSystem.h"
#include <iostream>
#include <algorithm>
#include <cmath>

RhythmSystem::RhythmSystem(float bpm) : m_bpm(bpm) {
    m_beatDuration = 60.0f / m_bpm; // Convert BPM to seconds per beat
    m_startTime = std::chrono::high_resolution_clock::now();
    m_lastBeatTime = m_startTime;
}

void RhythmSystem::initialize() {
    m_startTime = std::chrono::high_resolution_clock::now();
    m_lastBeatTime = m_startTime;
    m_currentBeat = 0;
    m_comboCount = 0;
    m_maxCombo = 0;
    
    std::cout << "RhythmSystem initialized - BPM: " << m_bpm 
              << ", Beat Duration: " << m_beatDuration << "s" << std::endl;
}

void RhythmSystem::update(float deltaTime) {
    if (!m_enabled) return;
    
    updateBeatTracking(deltaTime);
    updateComboSystem(deltaTime);
    updateVisualEffects(deltaTime);
}

RhythmTiming RhythmSystem::analyzeActionTiming() const {
    if (!m_enabled) return RhythmTiming::OKAY;
    
    float currentTime = calculateCurrentTime();
    float timeToBeat = getTimeToBeat(currentTime);
    float timeToBeatMs = std::abs(timeToBeat) * 1000.0f;
    
    if (timeToBeatMs <= m_perfectWindow) {
        return RhythmTiming::PERFECT;
    } else if (timeToBeatMs <= m_goodWindow) {
        return RhythmTiming::GOOD;
    } else if (timeToBeatMs <= m_okayWindow) {
        return RhythmTiming::OKAY;
    } else {
        return RhythmTiming::MISS;
    }
}

float RhythmSystem::getTimingMultiplier(RhythmTiming timing) const {
    switch (timing) {
        case RhythmTiming::PERFECT: return m_perfectMultiplier;
        case RhythmTiming::GOOD:    return m_goodMultiplier;
        case RhythmTiming::OKAY:    return m_okayMultiplier;
        case RhythmTiming::MISS:    return m_missMultiplier;
        default: return 1.0f;
    }
}

bool RhythmSystem::isOnBeat(float toleranceMs) const {
    if (!m_enabled) return true;
    
    float currentTime = calculateCurrentTime();
    return isBeatTime(currentTime, toleranceMs / 1000.0f);
}

float RhythmSystem::getBeatProgress() const {
    float currentTime = calculateCurrentTime();
    float timeInBeat = std::fmod(currentTime, m_beatDuration);
    return timeInBeat / m_beatDuration;
}

int RhythmSystem::getCurrentBeat() const {
    return m_currentBeat;
}

float RhythmSystem::getTimeToNextBeat() const {
    float currentTime = calculateCurrentTime();
    float timeInBeat = std::fmod(currentTime, m_beatDuration);
    return m_beatDuration - timeInBeat;
}

float RhythmSystem::getBeatIntensity() const {
    return m_beatIntensity;
}

bool RhythmSystem::shouldTriggerBeatEffect() const {
    return m_beatJustTriggered;
}

float RhythmSystem::getAttackMultiplier() const {
    return getTimingMultiplier(analyzeActionTiming()) * getComboMultiplier();
}

float RhythmSystem::getSpeedMultiplier() const {
    RhythmTiming timing = analyzeActionTiming();
    float baseMultiplier = 1.0f;
    
    switch (timing) {
        case RhythmTiming::PERFECT:
            baseMultiplier = 1.0f + m_perfectSpeedBonus;
            break;
        case RhythmTiming::GOOD:
            baseMultiplier = 1.0f + m_goodSpeedBonus;
            break;
        default:
            baseMultiplier = 1.0f;
            break;
    }
    
    return baseMultiplier * (1.0f + getComboMultiplier() * 0.1f);
}

void RhythmSystem::registerAction(RhythmTiming timing) {
    RhythmAction action;
    action.timestamp = std::chrono::high_resolution_clock::now();
    action.timing = timing;
    action.multiplier = getTimingMultiplier(timing);
    
    m_recentActions.push_back(action);
    m_timeSinceLastAction = 0.0f;
    
    // Update combo
    if (timing == RhythmTiming::PERFECT || timing == RhythmTiming::GOOD) {
        m_comboCount++;
        m_maxCombo = std::max(m_maxCombo, m_comboCount);
        
        // Debug output for rhythm feedback
        std::cout << "Rhythm Hit! Timing: " << (int)timing 
                  << ", Combo: " << m_comboCount 
                  << ", Multiplier: " << action.multiplier << std::endl;
    } else {
        m_comboCount = 0; // Break combo on miss or okay timing
    }
    
    // Keep only recent actions (last 10)
    if (m_recentActions.size() > 10) {
        m_recentActions.erase(m_recentActions.begin());
    }
}

float RhythmSystem::getComboMultiplier() const {
    return calculateComboMultiplier(m_comboCount);
}

void RhythmSystem::resetCombo() {
    m_comboCount = 0;
}

void RhythmSystem::setBPM(float bpm) {
    m_bpm = bpm;
    m_beatDuration = 60.0f / m_bpm;
    std::cout << "BPM changed to: " << m_bpm << std::endl;
}

// Private methods
void RhythmSystem::updateBeatTracking(float deltaTime) {
    float currentTime = calculateCurrentTime();
    
    // Check if we've hit a new beat
    int newBeat = static_cast<int>(currentTime / m_beatDuration);
    if (newBeat > m_currentBeat) {
        m_currentBeat = newBeat;
        m_lastBeatTime = std::chrono::high_resolution_clock::now();
        triggerBeatEffect();
        
        // Debug output for beat tracking
        if (m_currentBeat % 4 == 0) { // Every 4 beats (measure)
            std::cout << "Beat: " << m_currentBeat << " (Measure)" << std::endl;
        }
    }
}

void RhythmSystem::updateComboSystem(float deltaTime) {
    m_timeSinceLastAction += deltaTime;
    
    // Decay combo if no action for too long
    if (m_timeSinceLastAction > m_comboDecayTime && m_comboCount > 0) {
        decayCombo(deltaTime);
    }
}

void RhythmSystem::updateVisualEffects(float deltaTime) {
    // Reset beat trigger flag
    m_beatJustTriggered = false;
    
    // Update beat intensity (pulsing effect)
    float beatProgress = getBeatProgress();
    
    // Create a pulse that peaks at the beat and fades
    if (beatProgress < 0.1f) {
        m_beatIntensity = 1.0f - (beatProgress / 0.1f);
    } else {
        m_beatIntensity = 0.0f;
    }
    
    // Apply volume scaling
    m_beatIntensity *= m_volume;
}

float RhythmSystem::calculateCurrentTime() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - m_startTime);
    return duration.count() / 1000000.0f;
}

float RhythmSystem::getTimeToBeat(float currentTime) const {
    float timeInBeat = std::fmod(currentTime, m_beatDuration);
    
    // Calculate time to nearest beat (past or future)
    float timeToNextBeat = m_beatDuration - timeInBeat;
    float timeToPrevBeat = -timeInBeat;
    
    // Return whichever is closer
    return (std::abs(timeToNextBeat) < std::abs(timeToPrevBeat)) ? timeToNextBeat : timeToPrevBeat;
}

bool RhythmSystem::isBeatTime(float currentTime, float tolerance) const {
    float timeToBeat = getTimeToBeat(currentTime);
    return std::abs(timeToBeat) <= tolerance;
}

void RhythmSystem::triggerBeatEffect() {
    m_beatJustTriggered = true;
    
    // Trigger visual/audio effects here
    // For now, just set the flag for external systems to check
}

float RhythmSystem::calculateComboMultiplier(int combo) const {
    if (combo <= 0) return 0.0f;
    
    // Logarithmic scaling: significant boost early, diminishing returns
    float multiplier = std::log(combo + 1) * 0.3f;
    return std::min(multiplier, 2.0f); // Cap at 2x multiplier
}

void RhythmSystem::decayCombo(float deltaTime) {
    // Gradual combo decay
    float decayRate = 1.0f; // Lose 1 combo per second after decay time
    float comboLoss = decayRate * deltaTime;
    
    m_comboCount = std::max(0, static_cast<int>(m_comboCount - comboLoss));
    
    if (m_comboCount == 0) {
        std::cout << "Combo broken due to inactivity" << std::endl;
    }
}
