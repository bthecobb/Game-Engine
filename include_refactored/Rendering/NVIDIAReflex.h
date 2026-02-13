#pragma once

#ifdef _WIN32
#include <cstdint>
#include <string>

namespace CudaGame {
namespace Rendering {

/**
 * NVIDIA Reflex Low Latency SDK Integration
 * 
 * Reduces system latency by optimizing CPU-GPU synchronization and
 * providing low-latency present modes.
 * 
 * Expected latency reduction: 20-40% (typical 30-50ms → 18-30ms)
 */
class NVIDIAReflex {
public:
    // Reflex marker types (Phase Classification Labels)
    enum class Marker {
        SIMULATION_START,   // Beginning of game logic/physics
        SIMULATION_END,     // End of game logic/physics
        RENDERSUBMIT_START, // Start recording render commands
        RENDERSUBMIT_END,   // End recording render commands (before Present)
        PRESENT_START,      // About to present frame
        PRESENT_END,        // Present completed
        INPUT_SAMPLE,       // Input sampled from user
        TRIGGER_FLASH,      // Optional: Flash trigger for analysis
        PC_LATENCY_PING,    // Optional: Ping for latency measurement
    };

    // Reflex modes
    enum class Mode {
        OFF,                // Reflex disabled
        ENABLED,            // Reflex enabled, standard low latency
        ENABLED_BOOST,      // Reflex enabled + GPU boost for lowest latency
    };

    // Statistics
    struct Stats {
        float gameToRenderLatencyMs = 0.0f;  // Time from input to render
        float renderPresentLatencyMs = 0.0f; // Time from render to present
        float totalLatencyMs = 0.0f;         // Total system latency
        uint64_t frameID = 0;                // Current frame ID
        bool reflexSupported = false;        // Is Reflex available on this GPU?
    };

    NVIDIAReflex() = default;
    ~NVIDIAReflex();

    /**
     * Initialize Reflex SDK
     * Call once at startup after D3D12 device creation
     */
    bool Initialize(void* d3d12Device);

    /**
     * Shutdown Reflex SDK
     */
    void Shutdown();

    /**
     * Set Reflex mode (OFF, ENABLED, ENABLED_BOOST)
     */
    void SetMode(Mode mode);

    /**
     * Get current mode
     */
    Mode GetMode() const { return m_mode; }

    /**
     * Mark a specific point in the frame (PCL markers)
     * Call this at key points to track latency
     */
    void SetMarker(Marker marker, uint64_t frameID);

    /**
     * Get latest latency statistics
     * Call once per frame to update stats
     */
    Stats GetStats();

    /**
     * Check if Reflex is supported on this GPU
     */
    bool IsSupported() const { return m_supported; }

    /**
     * Enable/disable sleep mode for low latency
     * When enabled, reduces pre-rendered frames to 1
     */
    void SetSleepMode(bool enabled);

private:
    bool m_initialized = false;
    bool m_supported = false;
    Mode m_mode = Mode::OFF;
    Stats m_latestStats;
    
    // Reflex SDK handle (opaque pointer to avoid including SDK headers)
    void* m_reflexHandle = nullptr;
    
    // Frame tracking
    uint64_t m_currentFrameID = 0;
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
