#pragma once

#include "Core/System.h"
#include "Audio/AudioComponents.h"
#include "Audio/AudioResources.h"
#include "Physics/CollisionDetection.h" // For AABB
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace CudaGame {
namespace Audio {

// Main audio system for managing all audio-related tasks
class AudioSystem : public Core::System {
public:
    AudioSystem();
    ~AudioSystem();

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Audio clip management
    bool LoadAudioClip(const std::string& name, const std::string& filePath);
    AudioClip* GetAudioClip(const std::string& name);
    void UnloadAudioClip(const std::string& name);

    // Audio source control
    void Play(Core::Entity entity);
    void Stop(Core::Entity entity);
    void Pause(Core::Entity entity);
    void SetVolume(Core::Entity entity, float volume);
    void SetPitch(Core::Entity entity, float pitch);
    void SetLooping(Core::Entity entity, bool looping);
    
    // Music control
    void PlayMusic(const std::string& trackName, float fadeInTime = 1.0f);
    void StopMusic(float fadeOutTime = 1.0f);
    void SetMusicVolume(float volume);
    void CrossfadeToTrack(const std::string& newTrack, float crossfadeTime = 2.0f);
    void SetMusicLayerVolume(const std::string& layerName, float volume, float fadeTime = 1.0f);
    void ActivateMusicLayer(const std::string& layerName);
    void DeactivateMusicLayer(const std::string& layerName);

    // 3D audio settings
    void UpdateListener(const glm::vec3& position, const glm::vec3& velocity, 
                        const glm::vec3& forward, const glm::vec3& up);
    
    // Reverb zones
    void AddReverbZone(const Physics::AABB& zone, const ReverbPreset& preset);
    void UpdateReverbZones(const glm::vec3& listenerPosition);

    // Rhythm integration
    void SetBPM(float bpm);
    float GetCurrentBeat();
    bool IsOnBeat(float window = 0.1f);
    void RegisterBeatCallback(std::function<void(int beat)> callback);

    // Audio buses/groups for mixing
    void CreateAudioBus(const std::string& name, float volume = 1.0f);
    void SetBusVolume(const std::string& name, float volume);
    void AssignToBus(Core::Entity entity, const std::string& busName);

    // Debugging
    void SetDebugVisualization(bool enable);
    void DrawDebugAudioSources();

private:
    // Audio backend (e.g., OpenAL, FMOD, Wwise)
    // For now, this will be a placeholder for the architecture
    // In a real engine, this would be an abstraction layer for the audio API
    class AudioBackend* m_backend = nullptr;

    // Resource management
    std::unordered_map<std::string, std::unique_ptr<AudioClip>> m_audioClips;
    std::unordered_map<std::string, std::unique_ptr<ReverbPreset>> m_reverbPresets;

    // Music system
    std::unique_ptr<MusicComponent> m_musicComponent;

    // Rhythm system
    float m_bpm = 120.0f;
    float m_beatTimer = 0.0f;
    int m_lastBeat = 0;
    std::vector<std::function<void(int beat)>> m_beatCallbacks;

    // Debug visualization
    bool m_debugVisualization = false;
    
    // Internal update methods
    void UpdateAudioSources(float deltaTime);
    void UpdateMusic(float deltaTime);
    void UpdateRhythm(float deltaTime);
    
    // Helper methods
    void InitializeDefaultReverbPresets();
};

} // namespace Audio
} // namespace CudaGame
