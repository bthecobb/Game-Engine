#pragma once

#include "Core/System.h"
#include "Audio/AudioSystem.h"
#include <vector>
#include <functional>

namespace CudaGame {
namespace Rhythm {

// Defines the timing windows for rhythm actions
struct RhythmTiming {
    float perfectWindow = 0.05f; // +/- seconds for a perfect hit
    float goodWindow = 0.1f;     // +/- seconds for a good hit
    float okWindow = 0.15f;      // +/- seconds for an OK hit
};

// Defines different timing accuracy levels
enum class RhythmAccuracy {
    Miss,
    Ok,
    Good,
    Perfect
};

// Event dispatched when a beat occurs
struct BeatEvent {
    int beatNumber;
    float bpm;
    float timeSignature;
};

// Component for entities that interact with the rhythm system
struct RhythmComponent {
    bool isSynced = true;
    float rhythmMultiplier = 1.0f;
    int combo = 0;
    RhythmAccuracy lastAccuracy = RhythmAccuracy::Miss;
};

// Manages all rhythm-based mechanics in the game
class RhythmSystem : public Core::System {
public:
    RhythmSystem();
    ~RhythmSystem() override = default;

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Music and BPM management
    void SetAudioSystem(Audio::AudioSystem* audioSystem);
    void StartMusicTrack(const std::string& trackName, float bpm);
    void SetBPM(float bpm);
    float GetBPM() const { return m_bpm; }

    // Beat tracking and timing
    float GetCurrentBeat() const;
    float GetNextBeatTime() const;
    float GetTimeSinceLastBeat() const;
    bool IsOnBeat(float window) const;
    RhythmAccuracy GetTimingAccuracy(float inputTime) const;

    // Callbacks and events
    using BeatCallback = std::function<void(const BeatEvent&)>;
    void RegisterBeatCallback(BeatCallback callback);

    // Gameplay integration
    float GetRhythmBasedDamageMultiplier(RhythmAccuracy accuracy) const;
    void OnPlayerAction(uint32_t entityId, float inputTime);

    // Debugging
    void SetDebugVisualization(bool enable);

private:
    Audio::AudioSystem* m_audioSystem = nullptr;
    float m_bpm = 120.0f;
    float m_secondsPerBeat = 0.5f;
    float m_beatTimer = 0.0f;
    int m_currentBeat = 0;
    
    RhythmTiming m_timingWindows;
    
    std::vector<BeatCallback> m_beatCallbacks;
    
    bool m_debugVisualization = false;

    void UpdateBeat(float deltaTime);
    void DispatchBeatEvent();
};

} // namespace Rhythm
} // namespace CudaGame
