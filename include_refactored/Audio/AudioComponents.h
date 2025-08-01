#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace CudaGame {
namespace Audio {

// Audio source component for 3D positional audio
struct AudioSourceComponent {
    std::string audioClipName;
    bool isPlaying = false;
    bool isLooping = false;
    float volume = 1.0f;
    float pitch = 1.0f;
    float spatialBlend = 1.0f; // 0 = 2D, 1 = 3D
    
    // 3D audio properties
    float minDistance = 1.0f;
    float maxDistance = 500.0f;
    float dopplerLevel = 1.0f;
    bool bypassEffects = false;
    bool bypassListenerEffects = false;
    bool bypassReverbZones = false;
    
    // Playback state
    float currentTime = 0.0f;
    bool mute = false;
    int priority = 128; // 0 = highest priority, 256 = lowest
    
    // Effects
    bool enableLowPassFilter = false;
    float lowPassCutoff = 5000.0f;
    bool enableHighPassFilter = false;
    float highPassCutoff = 10.0f;
    bool enableReverb = false;
    float reverbLevel = 0.0f;
};

// Audio listener component (typically attached to the camera/player)
struct AudioListenerComponent {
    float volume = 1.0f;
    bool pauseOnFocusLoss = true;
    
    // Velocity for doppler effect calculation
    glm::vec3 velocity{0.0f};
    
    // Global audio effects
    bool enableGlobalReverb = false;
    float globalReverbLevel = 0.0f;
    std::string reverbPreset = "Default";
};

// Music component for background music and dynamic soundtracks
struct MusicComponent {
    std::string currentTrack;
    std::string nextTrack;
    bool isPlaying = false;
    bool isLooping = true;
    float volume = 1.0f;
    float fadeInTime = 1.0f;
    float fadeOutTime = 1.0f;
    
    // Rhythm integration
    float bpm = 120.0f;
    float currentBeat = 0.0f;
    float beatsPerMeasure = 4.0f;
    bool syncToBeat = false;
    
    // Dynamic music layers
    struct MusicLayer {
        std::string trackName;
        float volume = 1.0f;
        bool isActive = false;
        float crossfadeTime = 2.0f;
    };
    
    std::vector<MusicLayer> layers;
    int activeLayers = 0;
};

// Audio trigger component for ambient sounds and audio zones
struct AudioTriggerComponent {
    enum class TriggerType {
        ON_ENTER,
        ON_EXIT,
        ON_STAY,
        ON_COLLISION,
        ON_INTERACTION
    };
    
    TriggerType triggerType = TriggerType::ON_ENTER;
    std::string audioClipName;
    float volume = 1.0f;
    float pitch = 1.0f;
    bool playOnce = true;
    bool hasTriggered = false;
    
    // Zone properties
    float triggerRadius = 5.0f;
    glm::vec3 triggerSize{5.0f, 5.0f, 5.0f}; // For box triggers
    bool useBoxTrigger = false; // false = sphere, true = box
    
    // Audio properties
    float fadeInTime = 0.5f;
    float fadeOutTime = 0.5f;
    bool spatialize = true;
    
    // Cooldown
    float cooldownTime = 0.0f;
    float lastTriggerTime = 0.0f;
};

// Audio emitter component for environmental audio effects
struct AudioEmitterComponent {
    enum class EmitterType {
        POINT,
        DIRECTIONAL,
        AMBIENT_ZONE
    };
    
    EmitterType type = EmitterType::POINT;
    std::string audioClipName;
    bool isActive = true;
    bool autoPlay = true;
    
    // Volume and spatial properties
    float volume = 1.0f;
    float innerRadius = 5.0f;
    float outerRadius = 50.0f;
    
    // For directional emitters
    glm::vec3 direction{0.0f, 0.0f, 1.0f};
    float coneAngle = 60.0f; // degrees
    float coneOuterAngle = 90.0f; // degrees
    float coneOuterGain = 0.25f;
    
    // Randomization
    float pitchRandomization = 0.0f;
    float volumeRandomization = 0.0f;
    float playbackRandomDelay = 0.0f;
    
    // Environmental effects
    bool enableOcclusion = true;
    bool enableObstruction = true;
    float occlusionStrength = 1.0f;
    float obstructionStrength = 1.0f;
};

} // namespace Audio
} // namespace CudaGame
