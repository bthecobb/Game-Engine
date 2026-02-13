#pragma once

#ifdef _WIN32
#include <d3d12.h>
#include <wrl/client.h>
#include <glm/glm.hpp>
#include <cstdint>

// Forward declare NVSDK types (will be available when SDK is installed)
// For now, provide stub interface
#ifdef DLSS_SDK_AVAILABLE
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#else
// Stub types when SDK not installed
typedef void* NVSDK_NGX_Handle;
typedef void* NVSDK_NGX_Parameter;
#endif

namespace CudaGame {
namespace Rendering {

// DLSS Quality Modes
enum class DLSSQualityMode {
    UltraPerformance, // 1080p -> 4K (9x upscale, highest FPS)
    Performance,       // 1440p -> 4K (4x upscale, balanced)
    Balanced,          // 1662p -> 4K (2.3x upscale)
    Quality,           // 1800p -> 4K (1.5x upscale, best quality)
    UltraQuality,      // 2880p -> 4K (1.3x upscale)
    DLAA               // 4K -> 4K (AA only, no upscaling)
};

// DLSS input resources for a frame
struct DLSSInputs {
    ID3D12Resource* colorBuffer;         // Rendered color (input resolution)
    ID3D12Resource* depthBuffer;         // Depth buffer
    ID3D12Resource* motionVectors;       // 2D motion vectors (screen space)
    ID3D12Resource* outputBuffer;        // Upscaled output (target resolution)
    ID3D12Resource* exposureTexture;     // Optional: exposure for HDR
    
    float jitterOffsetX;                 // Camera jitter X [-0.5, 0.5]
    float jitterOffsetY;                 // Camera jitter Y [-0.5, 0.5]
    float sharpness;                     // 0.0 = soft, 1.0 = sharp
    float preExposure;                   // Pre-exposure value (1.0 default)
    bool reset;                          // Reset accumulation (camera cut, etc.)
};

// AAA-grade DLSS wrapper for super resolution
class NVIDIADLSSWrapper {
public:
    NVIDIADLSSWrapper();
    ~NVIDIADLSSWrapper();
    
    // Initialize DLSS with target output resolution
    bool Initialize(ID3D12Device* device, uint32_t outputWidth, uint32_t outputHeight);
    void Shutdown();
    
    // Set quality mode (changes internal render resolution)
    void SetQualityMode(DLSSQualityMode mode);
    DLSSQualityMode GetQualityMode() const { return m_qualityMode; }
    
    // Get recommended render resolution for current quality mode
    void GetRenderResolution(uint32_t& outWidth, uint32_t& outHeight) const;
    
    // Get jitter offset for current frame (for TAA/DLSS)
    glm::vec2 GetJitterOffset(uint32_t frameIndex) const;
    
    // Execute DLSS upscaling
    void Execute(ID3D12GraphicsCommandList* cmdList, const DLSSInputs& inputs);
    
    // Query capabilities
    bool IsAvailable() const { return m_isAvailable; }
    bool IsInitialized() const { return m_initialized; }
    
    // Performance stats
    float GetUpscaleFactor() const { return m_upscaleFactor; }
    uint32_t GetOutputWidth() const { return m_outputWidth; }
    uint32_t GetOutputHeight() const { return m_outputHeight; }
    
private:
    template<typename T>
    using ComPtr = Microsoft::WRL::ComPtr<T>;
    
    ID3D12Device* m_device = nullptr;
    
    // DLSS SDK handles
#ifdef DLSS_SDK_AVAILABLE
    NVSDK_NGX_Handle* m_dlssFeature = nullptr;
    NVSDK_NGX_Parameter* m_dlssParams = nullptr;
#else
    void* m_dlssFeature = nullptr;
    void* m_dlssParams = nullptr;
#endif
    
    // Configuration
    DLSSQualityMode m_qualityMode = DLSSQualityMode::Quality;
    uint32_t m_outputWidth = 0;
    uint32_t m_outputHeight = 0;
    uint32_t m_renderWidth = 0;
    uint32_t m_renderHeight = 0;
    float m_upscaleFactor = 1.5f;
    
    bool m_isAvailable = false;
    bool m_initialized = false;
    
    // Halton sequence for jitter pattern (AAA standard)
    float HaltonSequence(uint32_t index, uint32_t base) const;
    
    // Calculate render resolution from quality mode
    void CalculateRenderResolution();
};

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
