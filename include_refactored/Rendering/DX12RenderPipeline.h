#pragma once
#ifdef _WIN32

#include <d3d12.h>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "Rendering/Backends/DX12RenderBackend.h"
#include "Rendering/StreamlineDLSSWrapper.h"
#include "Rendering/RayTracingSystem.h"
#include "Rendering/Camera.h"
#include "Rendering/D3D12Mesh.h"
#include "Rendering/D3D12Constants.h"
#include <wrl/client.h>

namespace CudaGame {
namespace Rendering {

// AAA-quality deferred rendering pipeline for D3D12
// Features: G-Buffer, PBR lighting, DLSS upscaling, ray tracing
class DX12RenderPipeline {
public:
    DX12RenderPipeline();
    ~DX12RenderPipeline();

    // Initialization
    struct InitParams {
        void* windowHandle = nullptr;   // GLFW window handle
        uint32_t displayWidth = 3840;   // 4K display resolution
        uint32_t displayHeight = 2160;
        bool enableDLSS = true;
        bool enableRayTracing = true;
        DLSSQualityMode dlssMode = DLSSQualityMode::Quality;
    };
    
    bool Initialize(const InitParams& params);
    void Shutdown();

    // Frame rendering (main entry point)
    void BeginFrame(Camera* camera);
    void RenderFrame();
    void EndFrame();
    
    // Scene management
    void AddMesh(D3D12Mesh* mesh) { m_meshes.push_back(mesh); }
    void ClearMeshes() { m_meshes.clear(); }
    size_t GetMeshCount() const { return m_meshes.size(); }

    // Configuration
    void SetDLSSEnabled(bool enabled) { m_dlssEnabled = enabled; }
    void SetDLSSQualityMode(DLSSQualityMode mode);
    void SetRayTracingEnabled(bool enabled) { m_rayTracingEnabled = enabled; }
    
    // Debug visualization modes
    enum class DebugMode {
        NONE = 0,
        WIREFRAME,
        GBUFFER_POSITION,
        GBUFFER_NORMAL,
        GBUFFER_ALBEDO,
        DEPTH
    };
    void SetDebugMode(DebugMode mode) { m_debugMode = mode; }
    DebugMode GetDebugMode() const { return m_debugMode; }
    
    // Getters
    uint32_t GetRenderWidth() const { return m_renderWidth; }
    uint32_t GetRenderHeight() const { return m_renderHeight; }
    uint32_t GetDisplayWidth() const { return m_displayWidth; }
    uint32_t GetDisplayHeight() const { return m_displayHeight; }
    bool IsDLSSEnabled() const { return m_dlssEnabled; }
    bool IsRayTracingEnabled() const { return m_rayTracingEnabled; }
    DX12RenderBackend* GetBackend() const { return m_backend.get(); }
    
    // Performance stats
    struct FrameStats {
        float totalFrameMs = 0.0f;
        float geometryPassMs = 0.0f;
        float lightingPassMs = 0.0f;
        float rayTracingPassMs = 0.0f;
        float dlssPassMs = 0.0f;
        uint32_t drawCalls = 0;
        uint32_t triangles = 0;
    };
    FrameStats GetFrameStats() const { return m_stats; }

private:
    // Maximum number of meshes/materials we support per frame for constant buffers
    static constexpr uint32_t MAX_MESHES_PER_FRAME = 1024;
    // === G-Buffer Layout (AAA standard) ===
    struct GBuffer {
        ID3D12Resource* albedoRoughness = nullptr;     // RGB: Albedo, A: Roughness
        ID3D12Resource* normalMetallic = nullptr;       // RGB: Normal, A: Metallic
        ID3D12Resource* emissiveAO = nullptr;          // RGB: Emissive, A: AO
        ID3D12Resource* velocity = nullptr;            // RG: Motion vectors (screen space)
        ID3D12Resource* depth = nullptr;               // R32_FLOAT depth
        
        // Views
        D3D12_CPU_DESCRIPTOR_HANDLE albedoRTV;
        D3D12_CPU_DESCRIPTOR_HANDLE normalRTV;
        D3D12_CPU_DESCRIPTOR_HANDLE emissiveRTV;
        D3D12_CPU_DESCRIPTOR_HANDLE velocityRTV;
        D3D12_CPU_DESCRIPTOR_HANDLE depthDSV;
        
        D3D12_CPU_DESCRIPTOR_HANDLE albedoSRV;
        D3D12_CPU_DESCRIPTOR_HANDLE normalSRV;
        D3D12_CPU_DESCRIPTOR_HANDLE emissiveSRV;
        D3D12_CPU_DESCRIPTOR_HANDLE velocitySRV;
        D3D12_CPU_DESCRIPTOR_HANDLE depthSRV;
    };

    // === Lighting Buffers ===
    struct LightingBuffers {
        ID3D12Resource* litColor = nullptr;            // HDR lighting result
        ID3D12Resource* reflections = nullptr;         // RT reflections
        ID3D12Resource* shadows = nullptr;             // RT shadows
        ID3D12Resource* ambientOcclusion = nullptr;    // RT AO
        
        D3D12_CPU_DESCRIPTOR_HANDLE litColorRTV;
        D3D12_CPU_DESCRIPTOR_HANDLE reflectionsUAV;
        D3D12_CPU_DESCRIPTOR_HANDLE shadowsUAV;
        D3D12_CPU_DESCRIPTOR_HANDLE aoUAV;
    };

    // === Output Buffers ===
    struct OutputBuffers {
        ID3D12Resource* finalColor = nullptr;          // DLSS output (display res)
        ID3D12Resource* postProcess = nullptr;         // Post-process buffer
        
        D3D12_CPU_DESCRIPTOR_HANDLE finalColorUAV;
        D3D12_CPU_DESCRIPTOR_HANDLE postProcessRTV;
    };

    // === Render Passes (AAA pipeline) ===
    void GBufferPass();       // Populate G-Buffer MRTs at render resolution
    void GeometryPass();      // Forward pass to swapchain (temporary debug path)
    void ShadowPass();        // Generate shadow maps
    void LightingPass();      // Deferred lighting from G-Buffer
    void RayTracingPass();    // RT reflections, shadows, AO
    void DLSSPass();          // Upscale render res → display res
    void PostProcessPass();   // Bloom, tone mapping, etc.
    void UIPass();            // Render UI at display res

    // === Resource Management ===
    bool CreateGBuffer();
    bool CreateLightingBuffers();
    bool CreateOutputBuffers();
    void DestroyRenderTargets();
    
    // === Shader & PSO Management ===
    bool CompileShaders();
    bool CreateRootSignature();
    bool CreateGBufferPassPSO();
    bool CreateGeometryPassPSO();
    
    // === Constant Buffer Management ===
    bool CreateConstantBuffers();
    
    // Helpers
    ID3D12Resource* CreateTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format, 
                                    D3D12_RESOURCE_FLAGS flags, const wchar_t* name);
    void TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, 
                           D3D12_RESOURCE_STATES after);

    // === Core Systems ===
    std::unique_ptr<DX12RenderBackend> m_backend;
    std::unique_ptr<StreamlineDLSSWrapper> m_dlss;
    std::unique_ptr<RayTracingSystem> m_rayTracing;
    
    Camera* m_camera = nullptr;
    std::vector<D3D12Mesh*> m_meshes;  // Scene meshes to render
    
    // === Resolutions ===
    uint32_t m_displayWidth = 3840;
    uint32_t m_displayHeight = 2160;
    uint32_t m_renderWidth = 2560;   // DLSS Quality: 1800p → 4K
    uint32_t m_renderHeight = 1440;
    
    // === Render Targets ===
    GBuffer m_gBuffer;
    LightingBuffers m_lightingBuffers;
    OutputBuffers m_outputBuffers;
    
    // === Shaders & Pipeline State ===
    Microsoft::WRL::ComPtr<ID3DBlob> m_geometryVS;
    Microsoft::WRL::ComPtr<ID3DBlob> m_gbufferPS;     // Geometry pass → G-Buffer MRTs
    Microsoft::WRL::ComPtr<ID3DBlob> m_geometryPS;    // Forward debug pass → swapchain
    Microsoft::WRL::ComPtr<ID3DBlob> m_lightingCS;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_gbufferPassPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_geometryPassPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_wireframePSO;  // Debug wireframe mode
    
    // === Constant Buffers ===
    Microsoft::WRL::ComPtr<ID3D12Resource> m_perFrameCB;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_perObjectCB;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_materialCB;
    PerFrameConstants* m_perFrameData = nullptr;
    PerObjectConstants* m_perObjectData = nullptr;
    MaterialConstants* m_materialData = nullptr;
    
    // === Feature Flags ===
    bool m_initialized = false;
    bool m_dlssEnabled = true;
    bool m_rayTracingEnabled = true;
    DebugMode m_debugMode = DebugMode::NONE;
    
    // === Frame State ===
    uint64_t m_frameIndex = 0;
    glm::mat4 m_prevViewProj = glm::mat4(1.0f);
    
    // === Performance Tracking ===
    FrameStats m_stats;
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
