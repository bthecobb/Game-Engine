#ifdef _WIN32
#include "Rendering/StreamlineDLSSWrapper.h"
#include <iostream>
#include <cmath>
#include <dxgi1_6.h>
#include <wrl/client.h>

// Include Streamline SDK headers
#ifdef STREAMLINE_SDK_AVAILABLE
#include <sl.h>
#include <sl_dlss.h>
#include <sl_consts.h>
#endif

namespace CudaGame {
namespace Rendering {

StreamlineDLSSWrapper::StreamlineDLSSWrapper() = default;

StreamlineDLSSWrapper::~StreamlineDLSSWrapper() {
    Shutdown();
}

bool StreamlineDLSSWrapper::Initialize(ID3D12Device* device, ID3D12CommandQueue* cmdQueue, uint32_t outputWidth, uint32_t outputHeight) {
    if (m_initialized) {
        std::cerr << "[DLSS] Already initialized" << std::endl;
        return true;
    }

    if (!device || !cmdQueue) {
        std::cerr << "[DLSS] Invalid device or command queue" << std::endl;
        return false;
    }

    m_device = device;
    m_cmdQueue = cmdQueue;
    m_outputWidth = outputWidth;
    m_outputHeight = outputHeight;

#ifdef STREAMLINE_SDK_AVAILABLE
    std::cout << "[DLSS] Initializing NVIDIA Streamline SDK 2.9.0..." << std::endl;

    // Step 1: Initialize Streamline preferences
    sl::Preferences prefs = {};
    prefs.showConsole = false;  // Set to true for debugging
    prefs.logLevel = sl::LogLevel::eDefault;
    prefs.pathsToPlugins = nullptr;  // Plugins in same directory as executable
    prefs.numPathsToPlugins = 0;
    prefs.pathToLogsAndData = nullptr;  // No file logging
    prefs.logMessageCallback = nullptr;  // Could add custom logging
    prefs.applicationId = 0;  // Get from NVIDIA developer portal for production
    prefs.engine = sl::EngineType::eCustom;
    prefs.engineVersion = "1.0";
    prefs.projectId = "CudaGame";
    prefs.renderAPI = sl::RenderAPI::eD3D12;
    
    // Enable DLSS feature
    sl::Feature features[] = { sl::kFeatureDLSS };
    prefs.featuresToLoad = features;
    prefs.numFeaturesToLoad = 1;
    prefs.flags = sl::PreferenceFlags::eDisableCLStateTracking;  // We'll restore CL state manually

    // Initialize Streamline
    sl::Result result = slInit(prefs, sl::kSDKVersion);
    if (result != sl::Result::eOk) {
        std::cerr << "[DLSS] Failed to initialize Streamline: " << static_cast<int>(result) << std::endl;
        if (result == sl::Result::eErrorDriverOutOfDate) {
            std::cerr << "[DLSS] Driver is out of date - please update NVIDIA drivers" << std::endl;
        }
        m_isAvailable = false;
        m_initialized = false;
        return false;
    }

    // Step 2: Set D3D12 device
    result = slSetD3DDevice(device);
    if (result != sl::Result::eOk) {
        std::cerr << "[DLSS] Failed to set D3D12 device: " << static_cast<int>(result) << std::endl;
        slShutdown();
        return false;
    }

    // Step 3: Check if DLSS is supported on this adapter
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (SUCCEEDED(factory->EnumAdapters1(0, &adapter))) {
            DXGI_ADAPTER_DESC1 desc = {};
            if (SUCCEEDED(adapter->GetDesc1(&desc))) {
                sl::AdapterInfo adapterInfo = {};
                adapterInfo.deviceLUID = (uint8_t*)&desc.AdapterLuid;
                adapterInfo.deviceLUIDSizeInBytes = sizeof(LUID);
                
                result = slIsFeatureSupported(sl::kFeatureDLSS, adapterInfo);
                if (result != sl::Result::eOk) {
                    std::cerr << "[DLSS] DLSS not supported on this adapter" << std::endl;
                    if (result == sl::Result::eErrorNoSupportedAdapterFound) {
                        std::cerr << "[DLSS] GPU does not support DLSS (RTX GPU required)" << std::endl;
                    }
                    slShutdown();
                    return false;
                }
            }
        }
    }

    // Step 4: Query optimal settings
    CalculateRenderResolution();
    
    sl::DLSSOptions options = {};
    options.mode = static_cast<sl::DLSSMode>(static_cast<int>(m_qualityMode) + 1);  // Our enum is 0-based, SL is 1-based
    options.outputWidth = m_outputWidth;
    options.outputHeight = m_outputHeight;
    
    sl::DLSSOptimalSettings optimalSettings = {};
    result = slDLSSGetOptimalSettings(options, optimalSettings);
    if (result == sl::Result::eOk) {
        m_renderWidth = optimalSettings.optimalRenderWidth;
        m_renderHeight = optimalSettings.optimalRenderHeight;
        m_upscaleFactor = static_cast<float>(m_outputWidth) / static_cast<float>(m_renderWidth);

        std::cout << "[DLSS] Optimal settings from Streamline:" << std::endl;
        std::cout << "  Output: " << m_outputWidth << "x" << m_outputHeight << std::endl;
        std::cout << "  Render: " << m_renderWidth << "x" << m_renderHeight << std::endl;
        std::cout << "  Upscale Factor: " << m_upscaleFactor << "x" << std::endl;
        std::cout << "  Min Render: " << optimalSettings.renderWidthMin << "x" << optimalSettings.renderHeightMin << std::endl;
        std::cout << "  Max Render: " << optimalSettings.renderWidthMax << "x" << optimalSettings.renderHeightMax << std::endl;
    }

    m_isAvailable = true;
    m_initialized = true;

    std::cout << "[DLSS] Initialized successfully via Streamline" << std::endl;
    return true;

#else
    // Stub mode when Streamline SDK not available
    std::cout << "[DLSS] Running in STUB mode (Streamline SDK not compiled in)" << std::endl;
    std::cout << "[DLSS] To enable DLSS:" << std::endl;
    std::cout << "  1. Ensure Streamline SDK is in vendor/streamline-sdk/" << std::endl;
    std::cout << "  2. Add -DSTREAMLINE_SDK_AVAILABLE to CMake configuration" << std::endl;
    std::cout << "  3. Rebuild" << std::endl;

    // Calculate render resolution for stub mode
    CalculateRenderResolution();

    m_isAvailable = false;
    m_initialized = true;
    return true;
#endif
}

void StreamlineDLSSWrapper::Shutdown() {
    if (!m_initialized) return;

#ifdef STREAMLINE_SDK_AVAILABLE
    if (m_isAvailable) {
        slShutdown();
    }
#endif

    m_initialized = false;
    m_isAvailable = false;
    std::cout << "[DLSS] Shutdown complete" << std::endl;
}

void StreamlineDLSSWrapper::SetQualityMode(DLSSQualityMode mode) {
    m_qualityMode = mode;
    CalculateRenderResolution();
    
#ifdef STREAMLINE_SDK_AVAILABLE
    if (m_initialized && m_isAvailable) {
        // Update DLSS options with new quality mode
        sl::DLSSOptions options = {};
        options.mode = static_cast<sl::DLSSMode>(static_cast<int>(m_qualityMode) + 1);
        options.outputWidth = m_outputWidth;
        options.outputHeight = m_outputHeight;
        options.colorBuffersHDR = sl::Boolean::eTrue;
        
        sl::ViewportHandle viewport(m_viewportId);
        sl::Result result = slDLSSSetOptions(viewport, options);
        if (result == sl::Result::eOk) {
            std::cout << "[DLSS] Quality mode updated to " << static_cast<int>(mode) << std::endl;
        } else {
            std::cerr << "[DLSS] Failed to set quality mode: " << static_cast<int>(result) << std::endl;
        }
    }
#else
    if (m_initialized) {
        std::cout << "[DLSS] Quality mode changed to " << static_cast<int>(mode) << " (stub mode)" << std::endl;
    }
#endif
}

void StreamlineDLSSWrapper::GetRenderResolution(uint32_t& outWidth, uint32_t& outHeight) const {
    outWidth = m_renderWidth;
    outHeight = m_renderHeight;
}

glm::vec2 StreamlineDLSSWrapper::GetJitterOffset(uint32_t frameIndex) const {
    // Halton sequence (2, 3) for temporal jitter pattern
    // This is the industry-standard approach for TAA/DLSS
    float x = HaltonSequence(frameIndex + 1, 2) - 0.5f;
    float y = HaltonSequence(frameIndex + 1, 3) - 0.5f;
    return glm::vec2(x, y);
}

void StreamlineDLSSWrapper::Execute(ID3D12GraphicsCommandList* cmdList, const DLSSInputs& inputs) {
    if (!m_initialized) {
        std::cerr << "[DLSS] Not initialized" << std::endl;
        return;
    }

#ifdef STREAMLINE_SDK_AVAILABLE
    if (!m_isAvailable) {
        return;
    }

    // Create Streamline resources for tagging
    sl::Resource colorInRes(sl::ResourceType::eTex2d, inputs.colorBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET);
    sl::Resource depthRes(sl::ResourceType::eTex2d, inputs.depthBuffer, D3D12_RESOURCE_STATE_DEPTH_READ);
    sl::Resource mvecRes(sl::ResourceType::eTex2d, inputs.motionVectors, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    sl::Resource colorOutRes(sl::ResourceType::eTex2d, inputs.outputBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    
    // Create extents for render and output resolutions
    sl::Extent renderExtent = { 0, 0, m_renderWidth, m_renderHeight };
    sl::Extent outputExtent = { 0, 0, m_outputWidth, m_outputHeight };
    
    // Tag resources with their types
    sl::ResourceTag colorInTag(&colorInRes, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eOnlyValidNow, &renderExtent);
    sl::ResourceTag depthTag(&depthRes, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &renderExtent);
    sl::ResourceTag mvecTag(&mvecRes, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow, &renderExtent);
    sl::ResourceTag colorOutTag(&colorOutRes, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eOnlyValidNow, &outputExtent);
    
    // Optionally tag exposure if provided
    sl::Resource exposureRes;
    sl::ResourceTag exposureTag;
    bool hasExposure = false;
    if (inputs.exposureTexture) {
        exposureRes = sl::Resource(sl::ResourceType::eTex2d, inputs.exposureTexture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        sl::Extent exposureExtent = { 0, 0, 1, 1 };  // Exposure is typically 1x1
        exposureTag = sl::ResourceTag(&exposureRes, sl::kBufferTypeExposure, sl::ResourceLifecycle::eOnlyValidNow, &exposureExtent);
        hasExposure = true;
    }
    
    // Create array of tags
    sl::ResourceTag tags[5];
    uint32_t tagCount = 0;
    tags[tagCount++] = colorInTag;
    tags[tagCount++] = depthTag;
    tags[tagCount++] = mvecTag;
    tags[tagCount++] = colorOutTag;
    if (hasExposure) {
        tags[tagCount++] = exposureTag;
    }
    
    // Get frame token (for now, use a simple counter - in production, use slGetNewFrameToken)
    static uint32_t frameIndex = 0;
    sl::FrameToken* frameToken = reinterpret_cast<sl::FrameToken*>(&frameIndex);
    frameIndex++;
    
    // Set tags for this frame
    sl::ViewportHandle viewport(m_viewportId);
    sl::Result result = slSetTag(viewport, tags, tagCount, cmdList);
    if (result != sl::Result::eOk) {
        std::cerr << "[DLSS] Failed to set tags: " << static_cast<int>(result) << std::endl;
        return;
    }
    
    // Set DLSS options
    sl::DLSSOptions options = {};
    options.mode = static_cast<sl::DLSSMode>(static_cast<int>(m_qualityMode) + 1);
    options.outputWidth = m_outputWidth;
    options.outputHeight = m_outputHeight;
    options.colorBuffersHDR = sl::Boolean::eTrue;
    options.preExposure = inputs.preExposure;
    options.sharpness = inputs.sharpness;
    options.useAutoExposure = hasExposure ? sl::Boolean::eFalse : sl::Boolean::eTrue;
    
    result = slDLSSSetOptions(viewport, options);
    if (result != sl::Result::eOk) {
        std::cerr << "[DLSS] Failed to set options: " << static_cast<int>(result) << std::endl;
        return;
    }
    
    // Evaluate DLSS - Streamline will inject upscaling into the command list
    const sl::BaseStructure* evalInputs[] = { &viewport };
    result = slEvaluateFeature(sl::kFeatureDLSS, *frameToken, evalInputs, 1, cmdList);
    if (result != sl::Result::eOk) {
        std::cerr << "[DLSS] Evaluation failed: " << static_cast<int>(result) << std::endl;
    }
    
    // IMPORTANT: Host application must restore command list state after slEvaluateFeature
    // This includes descriptor heaps, root signature, PSO, viewports, etc.

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

float StreamlineDLSSWrapper::HaltonSequence(uint32_t index, uint32_t base) const {
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

void StreamlineDLSSWrapper::CalculateRenderResolution() {
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
