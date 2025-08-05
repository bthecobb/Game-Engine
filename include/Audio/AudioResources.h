#pragma once

#include <string>
#include <vector>
#include <memory>

namespace CudaGame {
namespace Audio {

// Audio format enumeration
enum class AudioFormat {
    MONO_8,
    MONO_16,
    STEREO_8,
    STEREO_16,
    MONO_FLOAT32,
    STEREO_FLOAT32
};

// Audio clip resource
class AudioClip {
public:
    AudioClip(const std::string& name);
    ~AudioClip();

    // Resource loading
    bool LoadFromFile(const std::string& filePath);
    bool LoadFromMemory(const void* data, size_t size, AudioFormat format, int sampleRate);
    void Unload();

    // Getters
    const std::string& GetName() const { return m_name; }
    float GetLength() const { return m_length; }
    int GetSampleRate() const { return m_sampleRate; }
    int GetChannels() const { return m_channels; }
    AudioFormat GetFormat() const { return m_format; }
    bool IsLoaded() const { return m_isLoaded; }
    bool IsStreaming() const { return m_isStreaming; }
    
    // Audio data access
    const void* GetData() const { return m_data.data(); }
    size_t GetDataSize() const { return m_data.size(); }
    
    // Properties
    void SetLoadType(bool streaming) { m_isStreaming = streaming; }
    void SetCompressionFormat(const std::string& format) { m_compressionFormat = format; }

private:
    std::string m_name;
    std::vector<uint8_t> m_data;
    float m_length = 0.0f;
    int m_sampleRate = 44100;
    int m_channels = 2;
    AudioFormat m_format = AudioFormat::STEREO_16;
    bool m_isLoaded = false;
    bool m_isStreaming = false;
    std::string m_compressionFormat = "PCM";
    
    // Internal loading helpers
    bool LoadWAV(const std::string& filePath);
    bool LoadOGG(const std::string& filePath);
    bool LoadMP3(const std::string& filePath);
};

// Audio buffer for OpenAL/FMOD integration
class AudioBuffer {
public:
    AudioBuffer();
    ~AudioBuffer();

    bool Initialize(const AudioClip* clip);
    void Cleanup();
    
    uint32_t GetBufferID() const { return m_bufferID; }
    bool IsValid() const { return m_bufferID != 0; }

private:
    uint32_t m_bufferID = 0;
    bool m_isInitialized = false;
};

// Streaming audio buffer for large audio files
class StreamingAudioBuffer {
public:
    StreamingAudioBuffer();
    ~StreamingAudioBuffer();

    bool Initialize(const std::string& filePath, size_t bufferSize = 65536);
    void Cleanup();
    
    bool FillBuffer(uint32_t bufferID);
    bool IsEndOfStream() const { return m_endOfStream; }
    void Reset();
    
    float GetProgress() const { return m_progress; }

private:
    std::string m_filePath;
    size_t m_bufferSize;
    size_t m_currentPosition = 0;
    size_t m_totalSize = 0;
    float m_progress = 0.0f;
    bool m_endOfStream = false;
    bool m_isInitialized = false;
    
    // File handle (implementation specific)
    void* m_fileHandle = nullptr;
};

// Audio reverb preset
struct ReverbPreset {
    std::string name;
    float roomSize = 0.5f;
    float damping = 0.5f;
    float wetLevel = 0.3f;
    float dryLevel = 0.7f;
    float width = 1.0f;
    float predelay = 0.0f;
    
    // Advanced reverb parameters
    float density = 1.0f;
    float diffusion = 1.0f;
    float gain = 0.32f;
    float gainHF = 0.89f;
    float gainLF = 1.0f;
    float decayTime = 1.49f;
    float decayHFRatio = 0.83f;
    float decayLFRatio = 1.0f;
    float reflectionsGain = 0.05f;
    float reflectionsDelay = 0.007f;
    float lateReverbGain = 1.26f;
    float lateReverbDelay = 0.011f;
    float airAbsorptionGainHF = 0.994f;
    float roomRolloffFactor = 0.0f;
};

// Audio effects processor
class AudioEffects {
public:
    // Low-pass filter
    struct LowPassFilter {
        float cutoffFrequency = 5000.0f;
        float resonance = 1.0f;
        bool enabled = false;
    };
    
    // High-pass filter
    struct HighPassFilter {
        float cutoffFrequency = 10.0f;
        float resonance = 1.0f;
        bool enabled = false;
    };
    
    // Echo effect
    struct EchoEffect {
        float delay = 0.5f;
        float wetDryMix = 0.5f;
        float feedback = 0.5f;
        bool enabled = false;
    };
    
    // Chorus effect
    struct ChorusEffect {
        float wetDryMix = 0.5f;
        float depth = 1.0f;
        float rate = 0.8f;
        bool enabled = false;
    };
    
    // Distortion effect
    struct DistortionEffect {
        float level = 0.5f;
        bool enabled = false;
    };
    
    // Parametric EQ
    struct ParametricEQ {
        float centerFreq = 8000.0f;
        float octaveRange = 1.0f;
        float frequencyGain = 1.0f;
        bool enabled = false;
    };

private:
    LowPassFilter m_lowPass;
    HighPassFilter m_highPass;
    EchoEffect m_echo;
    ChorusEffect m_chorus;
    DistortionEffect m_distortion;
    ParametricEQ m_eq;
};

} // namespace Audio
} // namespace CudaGame
