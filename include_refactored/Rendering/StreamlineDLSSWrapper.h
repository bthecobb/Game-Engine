#pragma once
#ifdef _WIN32

#include <d3d12.h>
#include <cstdint>
#include <glm/glm.hpp>

namespace CudaGame {
namespace Rendering {

// Quality modes for DLSS upscaling
enum class DLSSQualityMode {
    UltraPerformance = 0,  // 1080p -> 4K (9x pixels, highest FPS)
    Performance = 1,        // ~1440p -> 4K (4x pixels, high FPS)
    Balanced = 2,           // ~1662p -> 4K (2.3x pixels)
    Quality = 3,            // 1800p -> 4K (1.5x pixels, best quality/perf balance)
    UltraQuality = 4,       // 2880p -> 4K (1.3x pixels)
    DLAA = 5                // 4K -> 4K (anti-aliasing only, best quality)
};

// Streamline-based DLSS wrapper for D3D12
// Provides NVIDIA DLSS Super Resolution via Streamline SDK
class StreamlineDLSSWrapper {
public:
    StreamlineDLSSWrapper();
    ~StreamlineDLSSWrapper();

    // Initialize DLSS with output resolution
    bool Initialize(ID3D12Device* device, ID3D12CommandQueue* cmdQueue, uint32_t outputWidth, uint32_t outputHeight);
    
    // Shutdown and cleanup
    void Shutdown();

    // Set quality mode (call before Execute)
    void SetQualityMode(DLSSQualityMode mode);

    // Get render resolution for current quality mode
    void GetRenderResolution(uint32_t& outWidth, uint32_t& outHeight) const;

    // Get jitter offset for current frame (for TAA/DLSS)
    glm::vec2 GetJitterOffset(uint32_t frameIndex) const;

    // Execute DLSS upscaling (call per frame)
    struct DLSSInputs {
        ID3D12Resource* colorBuffer;        // Input: Rendered color (render resolution)
        ID3D12Resource* depthBuffer;        // Input: Depth buffer (render resolution)
        ID3D12Resource* motionVectors;      // Input: Motion vectors (screen space, pixels/frame)
        ID3D12Resource* outputBuffer;       // Output: Upscaled color (output resolution)
        ID3D12Resource* exposureTexture;    // Optional: Exposure (1x1 or full res)
        float jitterOffsetX;                // Jitter offset X (pixels)
        float jitterOffsetY;                // Jitter offset Y (pixels)
        float sharpness;                    // Sharpness [0-1] (deprecated but kept for compatibility)
        bool reset;                         // Reset accumulation (scene cut, teleport, etc)
        float preExposure;                  // Pre-exposure multiplier
    };
    void Execute(ID3D12GraphicsCommandList* cmdList, const DLSSInputs& inputs);

    // Check if DLSS is available on current hardware
    bool IsAvailable() const { return m_isAvailable; }

    // Get upscale factor
    float GetUpscaleFactor() const { return m_upscaleFactor; }

private:
    // Halton sequence for jitter pattern (low-discrepancy sequence)
    float HaltonSequence(uint32_t index, uint32_t base) const;

    // Calculate render resolution based on quality mode
    void CalculateRenderResolution();

    ID3D12Device* m_device = nullptr;
    ID3D12CommandQueue* m_cmdQueue = nullptr;
    
    DLSSQualityMode m_qualityMode = DLSSQualityMode::Quality;
    uint32_t m_outputWidth = 3840;
    uint32_t m_outputHeight = 2160;
    uint32_t m_renderWidth = 2560;
    uint32_t m_renderHeight = 1440;
    float m_upscaleFactor = 1.5f;
    
    bool m_initialized = false;
    bool m_isAvailable = false;
    
    // Streamline feature handle
    void* m_slContext = nullptr;
    uint32_t m_viewportId = 0;
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
