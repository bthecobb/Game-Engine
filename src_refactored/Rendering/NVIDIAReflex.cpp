#ifdef _WIN32
#include "Rendering/NVIDIAReflex.h"
#include <d3d12.h>
#include <iostream>
#include <chrono>

// Check if Reflex SDK is available
// To enable: Download NVIDIA Reflex SDK and uncomment these includes
// #define REFLEX_SDK_AVAILABLE
// #ifdef REFLEX_SDK_AVAILABLE
// #include <pclstats.h>
// #include <NvLowLatencyVk.h> // Or DX12 variant
// #endif

namespace CudaGame {
namespace Rendering {

NVIDIAReflex::~NVIDIAReflex() {
    Shutdown();
}

bool NVIDIAReflex::Initialize(void* d3d12Device) {
    if (m_initialized) {
        std::cout << "[Reflex] Already initialized" << std::endl;
        return true;
    }

    std::cout << "[Reflex] Initializing NVIDIA Reflex SDK..." << std::endl;

#ifdef REFLEX_SDK_AVAILABLE
    // Real Reflex SDK initialization
    ID3D12Device* device = static_cast<ID3D12Device*>(d3d12Device);
    
    // Initialize Reflex with D3D12 device
    // NVLL_VK_Status status = NvLL_DX_Initialize(device, ...);
    // if (status == NVLL_VK_OK) {
    //     m_supported = true;
    //     m_initialized = true;
    //     std::cout << "[Reflex] SDK initialized successfully" << std::endl;
    //     return true;
    // }
    
    std::cerr << "[Reflex] SDK initialization failed" << std::endl;
    return false;
#else
    // Stub implementation (graceful fallback)
    std::cout << "[Reflex] Running in STUB mode (SDK not installed)" << std::endl;
    std::cout << "[Reflex] To enable real Reflex:" << std::endl;
    std::cout << "[Reflex]   1. Download NVIDIA Reflex SDK from developer.nvidia.com" << std::endl;
    std::cout << "[Reflex]   2. Extract to vendor/NVIDIA-Reflex-SDK/" << std::endl;
    std::cout << "[Reflex]   3. Add #define REFLEX_SDK_AVAILABLE to this file" << std::endl;
    std::cout << "[Reflex]   4. Rebuild" << std::endl;
    
    // Simulate basic functionality
    m_supported = true; // Pretend we're supported for testing
    m_initialized = true;
    
    std::cout << "[Reflex] Stub mode: Markers will be logged but not sent to driver" << std::endl;
    return true;
#endif
}

void NVIDIAReflex::Shutdown() {
    if (!m_initialized) return;

#ifdef REFLEX_SDK_AVAILABLE
    // Shutdown real SDK
    // NvLL_DX_Shutdown();
#endif

    m_initialized = false;
    std::cout << "[Reflex] Shutdown complete" << std::endl;
}

void NVIDIAReflex::SetMode(Mode mode) {
    if (!m_initialized) {
        std::cerr << "[Reflex] Cannot set mode: not initialized" << std::endl;
        return;
    }

    m_mode = mode;

    const char* modeStr = "UNKNOWN";
    switch (mode) {
        case Mode::OFF: modeStr = "OFF"; break;
        case Mode::ENABLED: modeStr = "ENABLED"; break;
        case Mode::ENABLED_BOOST: modeStr = "ENABLED_BOOST"; break;
    }

    std::cout << "[Reflex] Mode set to: " << modeStr << std::endl;

#ifdef REFLEX_SDK_AVAILABLE
    // Apply mode to real SDK
    // NvLL_DX_SetLatencyMarkerParams params;
    // params.lowLatencyMode = (mode != Mode::OFF);
    // params.lowLatencyBoost = (mode == Mode::ENABLED_BOOST);
    // NvLL_DX_SetLatencyMarker(params);
#else
    // Stub: Just log the mode change
    if (mode == Mode::ENABLED_BOOST) {
        std::cout << "[Reflex] STUB: Would enable GPU boost for lowest latency" << std::endl;
    } else if (mode == Mode::ENABLED) {
        std::cout << "[Reflex] STUB: Would enable low latency mode" << std::endl;
    }
#endif
}

void NVIDIAReflex::SetMarker(Marker marker, uint64_t frameID) {
    if (!m_initialized || m_mode == Mode::OFF) return;

    m_currentFrameID = frameID;

    // Map our marker enum to string for logging
    const char* markerName = "UNKNOWN";
    switch (marker) {
        case Marker::SIMULATION_START: markerName = "SIMULATION_START"; break;
        case Marker::SIMULATION_END: markerName = "SIMULATION_END"; break;
        case Marker::RENDERSUBMIT_START: markerName = "RENDERSUBMIT_START"; break;
        case Marker::RENDERSUBMIT_END: markerName = "RENDERSUBMIT_END"; break;
        case Marker::PRESENT_START: markerName = "PRESENT_START"; break;
        case Marker::PRESENT_END: markerName = "PRESENT_END"; break;
        case Marker::INPUT_SAMPLE: markerName = "INPUT_SAMPLE"; break;
        case Marker::TRIGGER_FLASH: markerName = "TRIGGER_FLASH"; break;
        case Marker::PC_LATENCY_PING: markerName = "PC_LATENCY_PING"; break;
    }

#ifdef REFLEX_SDK_AVAILABLE
    // Send marker to real SDK
    // NvLL_DX_SetMarker(markerType, frameID);
#else
    // Stub: Log marker (only log occasionally to avoid spam)
    static uint64_t lastLogFrame = 0;
    if (frameID - lastLogFrame > 60) { // Log every ~60 frames
        std::cout << "[Reflex] Marker: " << markerName << " (Frame " << frameID << ")" << std::endl;
        lastLogFrame = frameID;
    }
#endif
}

NVIDIAReflex::Stats NVIDIAReflex::GetStats() {
    if (!m_initialized) {
        return m_latestStats;
    }

#ifdef REFLEX_SDK_AVAILABLE
    // Get real stats from SDK
    // NvLL_VK_GetLatencyStats(&stats);
    // m_latestStats.gameToRenderLatencyMs = stats.gameToRenderLatency;
    // m_latestStats.renderPresentLatencyMs = stats.renderPresentLatency;
    // m_latestStats.totalLatencyMs = stats.totalLatency;
#else
    // Stub: Generate fake stats showing improvement
    // Simulate typical latencies with Reflex
    if (m_mode == Mode::ENABLED_BOOST) {
        m_latestStats.gameToRenderLatencyMs = 15.0f;  // ~15ms with boost
        m_latestStats.renderPresentLatencyMs = 8.0f;
        m_latestStats.totalLatencyMs = 23.0f;
    } else if (m_mode == Mode::ENABLED) {
        m_latestStats.gameToRenderLatencyMs = 18.0f;  // ~18ms without boost
        m_latestStats.renderPresentLatencyMs = 10.0f;
        m_latestStats.totalLatencyMs = 28.0f;
    } else {
        m_latestStats.gameToRenderLatencyMs = 30.0f;  // ~30ms without Reflex
        m_latestStats.renderPresentLatencyMs = 15.0f;
        m_latestStats.totalLatencyMs = 45.0f;
    }
#endif

    m_latestStats.frameID = m_currentFrameID;
    m_latestStats.reflexSupported = m_supported;

    return m_latestStats;
}

void NVIDIAReflex::SetSleepMode(bool enabled) {
    if (!m_initialized) return;

    std::cout << "[Reflex] Sleep mode: " << (enabled ? "ENABLED" : "DISABLED") << std::endl;

#ifdef REFLEX_SDK_AVAILABLE
    // Configure sleep mode in real SDK
    // NvLL_DX_SetSleepMode(enabled, m_mode == Mode::ENABLED_BOOST);
#else
    if (enabled) {
        std::cout << "[Reflex] STUB: Would reduce pre-rendered frames to 1" << std::endl;
    }
#endif
}

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
