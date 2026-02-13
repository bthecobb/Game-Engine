#ifdef _WIN32
#include "Rendering/NVIDIADLSSWrapper.h"
#include <iostream>
#include <cmath>

// Include DLSS SDK headers
#ifdef DLSS_SDK_AVAILABLE
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_params_dlssd.h>
#endif

namespace CudaGame {
namespace Rendering {

NVIDIADLSSWrapper::NVIDIADLSSWrapper() = default;

NVIDIADLSSWrapper::~NVIDIADLSSWrapper() {
    Shutdown();
}

bool NVIDIADLSSWrapper::Initialize(ID3D12Device* device, uint32_t outputWidth, uint32_t outputHeight) {
    if (m_initialized) {
        std::cerr << "[DLSS] Already initialized" << std::endl;
        return true;
    }

    if (!device) {
        std::cerr << "[DLSS] Invalid device" << std::endl;
        return false;
    }

    m_device = device;
    m_outputWidth = outputWidth;
    m_outputHeight = outputHeight;

#ifdef DLSS_SDK_AVAILABLE
    std::cout << "[DLSS] Initializing NVIDIA DLSS SDK 3.10..." << std::endl;

    // Initialize NGX
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init(
        1,                          // Application ID (get from NVIDIA developer portal)
        L".",                       // Working directory
        device,
        nullptr,                    // Feature info
        NVSDK_NGX_Version_API
    );

    if (NVSDK_NGX_FAILED(result)) {
        std::cerr << "[DLSS] Failed to initialize NGX SDK: " << std::hex << result << std::endl;
        m_isAvailable = false;
        m_initialized = false;
        return false;
    }

    // Calculate render resolution for current quality mode
    CalculateRenderResolution();

    // Get DLSS capabilities
    NVSDK_NGX_Parameter* params = nullptr;
    result = NVSDK_NGX_D3D12_GetCapabilityParameters(&params);
    if (NVSDK_NGX_FAILED(result)) {
        std::cerr << "[DLSS] Failed to get capability parameters" << std::endl;
        NVSDK_NGX_D3D12_Shutdown();
        return false;
    }

    // Check if DLSS is supported
    int dlssAvailable = 0;
    result = params->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);
    if (NVSDK_NGX_FAILED(result) || !dlssAvailable) {
        std::cerr << "[DLSS] DLSS not available on this hardware" << std::endl;
        NVSDK_NGX_D3D12_Shutdown();
        return false;
    }

    // Get optimal settings
    unsigned int optimalRenderWidth, optimalRenderHeight;
    unsigned int minRenderWidth, minRenderHeight;
    unsigned int maxRenderWidth, maxRenderHeight;
    float sharpness;

    NVSDK_NGX_PerfQuality_Value perfQuality;
    switch (m_qualityMode) {
        case DLSSQualityMode::UltraPerformance:
            perfQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance;
            break;
        case DLSSQualityMode::Performance:
            perfQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf;
            break;
        case DLSSQualityMode::Balanced:
            perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
            break;
        case DLSSQualityMode::Quality:
            perfQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
            break;
        case DLSSQualityMode::UltraQuality:
            perfQuality = NVSDK_NGX_PerfQuality_Value_UltraQuality;
            break;
        case DLSSQualityMode::DLAA:
            perfQuality = NVSDK_NGX_PerfQuality_Value_DLAA;
            break;
        default:
            perfQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    }

    result = NGX_DLSS_GET_OPTIMAL_SETTINGS(
        params,
        m_outputWidth, m_outputHeight,
        perfQuality,
        &optimalRenderWidth, &optimalRenderHeight,
        &maxRenderWidth, &maxRenderHeight,
        &minRenderWidth, &minRenderHeight,
        &sharpness
    );

    if (NVSDK_NGX_SUCCEEDED(result)) {
        m_renderWidth = optimalRenderWidth;
        m_renderHeight = optimalRenderHeight;
        m_upscaleFactor = static_cast<float>(m_outputWidth) / static_cast<float>(m_renderWidth);

        std::cout << "[DLSS] Optimal settings:" << std::endl;
        std::cout << "  Output: " << m_outputWidth << "x" << m_outputHeight << std::endl;
        std::cout << "  Render: " << m_renderWidth << "x" << m_renderHeight << std::endl;
        std::cout << "  Upscale Factor: " << m_upscaleFactor << "x" << std::endl;
        std::cout << "  Sharpness: " << sharpness << std::endl;
    }

    m_dlssParams = params;
    m_isAvailable = true;
    m_initialized = true;

    std::cout << "[DLSS] Initialized successfully" << std::endl;
    return true;

#else
    // Stub mode when SDK not available
    std::cout << "[DLSS] Running in STUB mode (SDK not compiled in)" << std::endl;
    std::cout << "[DLSS] To enable DLSS:" << std::endl;
    std::cout << "  1. Ensure DLSS SDK is in vendor/NVIDIA-DLSS-SDK/" << std::endl;
    std::cout << "  2. Add -DDLSS_SDK_AVAILABLE to CMake configuration" << std::endl;
    std::cout << "  3. Rebuild" << std::endl;

    // Calculate render resolution for stub mode
    CalculateRenderResolution();

    m_isAvailable = false;
    m_initialized = true;
    return true;
#endif
}

void NVIDIADLSSWrapper::Shutdown() {
    if (!m_initialized) return;

#ifdef DLSS_SDK_AVAILABLE
    if (m_dlssFeature) {
        NVSDK_NGX_D3D12_ReleaseFeature(m_dlssFeature);
        m_dlssFeature = nullptr;
    }

    if (m_isAvailable) {
        NVSDK_NGX_D3D12_Shutdown();
    }
#endif

    m_initialized = false;
    m_isAvailable = false;
    std::cout << "[DLSS] Shutdown complete" << std::endl;
}

void NVIDIADLSSWrapper::SetQualityMode(DLSSQualityMode mode) {
    m_qualityMode = mode;
    CalculateRenderResolution();

    // If already initialized, would need to recreate DLSS feature
    // For now, recommend re-initialization
    if (m_initialized) {
        std::cout << "[DLSS] Quality mode changed, recommend re-initialization" << std::endl;
    }
}

void NVIDIADLSSWrapper::GetRenderResolution(uint32_t& outWidth, uint32_t& outHeight) const {
    outWidth = m_renderWidth;
    outHeight = m_renderHeight;
}

glm::vec2 NVIDIADLSSWrapper::GetJitterOffset(uint32_t frameIndex) const {
    // Halton sequence (2, 3) for temporal jitter pattern
    // This is the industry-standard approach for TAA/DLSS
    float x = HaltonSequence(frameIndex + 1, 2) - 0.5f;
    float y = HaltonSequence(frameIndex + 1, 3) - 0.5f;
    return glm::vec2(x, y);
}

void NVIDIADLSSWrapper::Execute(ID3D12GraphicsCommandList* cmdList, const DLSSInputs& inputs) {
    if (!m_initialized) {
        std::cerr << "[DLSS] Not initialized" << std::endl;
        return;
    }

#ifdef DLSS_SDK_AVAILABLE
    if (!m_isAvailable || !m_dlssFeature) {
        // Create DLSS feature on first execute
        if (m_isAvailable && !m_dlssFeature) {
            NVSDK_NGX_DLSS_Create_Params createParams = {};
            createParams.Feature.InWidth = m_renderWidth;
            createParams.Feature.InHeight = m_renderHeight;
            createParams.Feature.InTargetWidth = m_outputWidth;
            createParams.Feature.InTargetHeight = m_outputHeight;
            createParams.Feature.InPerfQualityValue = static_cast<NVSDK_NGX_PerfQuality_Value>(
                static_cast<int>(m_qualityMode)
            );
            createParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
                                                 NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                                 NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

            NVSDK_NGX_Result result = NGX_D3D12_CREATE_DLSS_EXT(
                cmdList,
                1,  // Feature slot
                m_dlssParams,
                &m_dlssFeature,
                &createParams
            );

            if (NVSDK_NGX_FAILED(result)) {
                std::cerr << "[DLSS] Failed to create DLSS feature" << std::endl;
                return;
            }
        } else {
            return;
        }
    }

    // Execute DLSS upscaling
    NVSDK_NGX_D3D12_DLSS_Eval_Params evalParams = {};
    evalParams.Feature.pInColor = inputs.colorBuffer;
    evalParams.Feature.pInOutput = inputs.outputBuffer;
    evalParams.pInDepth = inputs.depthBuffer;
    evalParams.pInMotionVectors = inputs.motionVectors;
    evalParams.pInExposureTexture = inputs.exposureTexture;
    evalParams.InJitterOffsetX = inputs.jitterOffsetX;
    evalParams.InJitterOffsetY = inputs.jitterOffsetY;
    evalParams.Feature.InSharpness = inputs.sharpness;
    evalParams.InReset = inputs.reset;
    evalParams.InMVScaleX = 1.0f;
    evalParams.InMVScaleY = 1.0f;
    evalParams.InPreExposure = inputs.preExposure;

    NVSDK_NGX_Result result = NGX_D3D12_EVALUATE_DLSS_EXT(
        cmdList,
        m_dlssFeature,
        m_dlssParams,
        &evalParams
    );

    if (NVSDK_NGX_FAILED(result)) {
        std::cerr << "[DLSS] Evaluation failed: " << std::hex << result << std::endl;
    }
#else
    // Stub mode - just log that DLSS would execute
    static uint32_t frameCount = 0;
    if (frameCount % 60 == 0) {
        std::cout << "[DLSS] STUB: Would upscale " << m_renderWidth << "x" << m_renderHeight 
                  << " -> " << m_outputWidth << "x" << m_outputHeight << std::endl;
    }
    frameCount++;
#endif
}

float NVIDIADLSSWrapper::HaltonSequence(uint32_t index, uint32_t base) const {
    float result = 0.0f;
    float f = 1.0f;
    uint32_t i = index;

    while (i > 0) {
        f = f / static_cast<float>(base);
        result += f * static_cast<float>(i % base);
        i = static_cast<uint32_t>(std::floor(static_cast<float>(i) / static_cast<float>(base)));
    }

    return result;
}

void NVIDIADLSSWrapper::CalculateRenderResolution() {
    // Calculate render resolution based on quality mode and output resolution
    switch (m_qualityMode) {
        case DLSSQualityMode::UltraPerformance:
            m_upscaleFactor = 3.0f;  // 1080p -> 4K
            break;
        case DLSSQualityMode::Performance:
            m_upscaleFactor = 2.0f;  // ~1440p -> 4K
            break;
        case DLSSQualityMode::Balanced:
            m_upscaleFactor = 1.7f;  // ~1662p -> 4K
            break;
        case DLSSQualityMode::Quality:
            m_upscaleFactor = 1.5f;  // 1800p -> 4K
            break;
        case DLSSQualityMode::UltraQuality:
            m_upscaleFactor = 1.3f;  // 2880p -> 4K
            break;
        case DLSSQualityMode::DLAA:
            m_upscaleFactor = 1.0f;  // 4K -> 4K (AA only)
            break;
    }

    m_renderWidth = static_cast<uint32_t>(static_cast<float>(m_outputWidth) / m_upscaleFactor);
    m_renderHeight = static_cast<uint32_t>(static_cast<float>(m_outputHeight) / m_upscaleFactor);

    // Ensure even dimensions
    m_renderWidth = (m_renderWidth + 1) & ~1;
    m_renderHeight = (m_renderHeight + 1) & ~1;
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
