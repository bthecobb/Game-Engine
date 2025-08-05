#include "Rhythm/RhythmSystem.h"
#include <iostream>
#include <cmath>

namespace CudaGame {
namespace Rhythm {

RhythmSystem::RhythmSystem() {
    // Priority can be managed externally if needed
}

bool RhythmSystem::Initialize() {
    std::cout << "[RhythmSystem] Initializing rhythm-based combat system..." << std::endl;
    
    // Calculate initial timing based on default BPM
    m_secondsPerBeat = 60.0f / m_bpm;
    
    return true;
}

void RhythmSystem::Shutdown() {
    std::cout << "[RhythmSystem] Shutting down rhythm system." << std::endl;
    m_beatCallbacks.clear();
}

void RhythmSystem::Update(float deltaTime) {
    UpdateBeat(deltaTime);
}

void RhythmSystem::SetAudioSystem(Audio::AudioSystem* audioSystem) {
    m_audioSystem = audioSystem;
}

void RhythmSystem::StartMusicTrack(const std::string& trackName, float bpm) {
    if (m_audioSystem) {
        // In a real implementation, this would start playing the track
        std::cout << "[RhythmSystem] Starting track: " << trackName << " at " << bpm << " BPM" << std::endl;
    }
    SetBPM(bpm);
}

void RhythmSystem::SetBPM(float bpm) {
    m_bpm = bpm;
    m_secondsPerBeat = 60.0f / m_bpm;
    std::cout << "[RhythmSystem] BPM set to " << m_bpm << " (beat every " << m_secondsPerBeat << "s)" << std::endl;
}

float RhythmSystem::GetCurrentBeat() const {
    return m_beatTimer / m_secondsPerBeat;
}

float RhythmSystem::GetNextBeatTime() const {
    float currentBeat = GetCurrentBeat();
    float nextBeat = std::floor(currentBeat) + 1.0f;
    return nextBeat * m_secondsPerBeat;
}

float RhythmSystem::GetTimeSinceLastBeat() const {
    float currentBeat = GetCurrentBeat();
    float lastBeat = std::floor(currentBeat);
    return (currentBeat - lastBeat) * m_secondsPerBeat;
}

bool RhythmSystem::IsOnBeat(float window) const {
    float timeSinceLastBeat = GetTimeSinceLastBeat();
    float timeToNextBeat = m_secondsPerBeat - timeSinceLastBeat;
    
    return (timeSinceLastBeat <= window) || (timeToNextBeat <= window);
}

RhythmAccuracy RhythmSystem::GetTimingAccuracy(float inputTime) const {
    float timeSinceLastBeat = GetTimeSinceLastBeat();
    float timeToNextBeat = m_secondsPerBeat - timeSinceLastBeat;
    float distanceFromBeat = std::min(timeSinceLastBeat, timeToNextBeat);
    
    if (distanceFromBeat <= m_timingWindows.perfectWindow) {
        return RhythmAccuracy::Perfect;
    } else if (distanceFromBeat <= m_timingWindows.goodWindow) {
        return RhythmAccuracy::Good;
    } else if (distanceFromBeat <= m_timingWindows.okWindow) {
        return RhythmAccuracy::Ok;
    } else {
        return RhythmAccuracy::Miss;
    }
}

void RhythmSystem::RegisterBeatCallback(BeatCallback callback) {
    m_beatCallbacks.push_back(callback);
}

float RhythmSystem::GetRhythmBasedDamageMultiplier(RhythmAccuracy accuracy) const {
    switch (accuracy) {
        case RhythmAccuracy::Perfect: return 1.5f;
        case RhythmAccuracy::Good:    return 1.25f;
        case RhythmAccuracy::Ok:      return 1.1f;
        case RhythmAccuracy::Miss:    return 0.9f;
        default: return 1.0f;
    }
}

void RhythmSystem::OnPlayerAction(uint32_t entityId, float inputTime) {
    RhythmAccuracy accuracy = GetTimingAccuracy(inputTime);
    
    if (m_debugVisualization) {
        const char* accuracyStr = "";
        switch (accuracy) {
            case RhythmAccuracy::Perfect: accuracyStr = "PERFECT"; break;
            case RhythmAccuracy::Good:    accuracyStr = "GOOD"; break;
            case RhythmAccuracy::Ok:      accuracyStr = "OK"; break;
            case RhythmAccuracy::Miss:    accuracyStr = "MISS"; break;
        }
        std::cout << "[RhythmSystem] Entity " << entityId << " action: " << accuracyStr 
                  << " (multiplier: " << GetRhythmBasedDamageMultiplier(accuracy) << ")" << std::endl;
    }
}

void RhythmSystem::SetDebugVisualization(bool enable) {
    m_debugVisualization = enable;
    if (enable) {
        std::cout << "[RhythmSystem] Debug visualization enabled" << std::endl;
    }
}

void RhythmSystem::UpdateBeat(float deltaTime) {
    m_beatTimer += deltaTime;
    
    // Check if we've crossed a beat boundary
    int newBeat = static_cast<int>(m_beatTimer / m_secondsPerBeat);
    if (newBeat > m_currentBeat) {
        m_currentBeat = newBeat;
        DispatchBeatEvent();
    }
}

void RhythmSystem::DispatchBeatEvent() {
    BeatEvent event;
    event.beatNumber = m_currentBeat;
    event.bpm = m_bpm;
    event.timeSignature = 4.0f; // Assume 4/4 time for now
    
    if (m_debugVisualization) {
        std::cout << "[RhythmSystem] Beat " << event.beatNumber << " at " << event.bpm << " BPM" << std::endl;
    }
    
    for (const auto& callback : m_beatCallbacks) {
        callback(event);
    }
}

} // namespace Rhythm
} // namespace CudaGame
