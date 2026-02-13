#pragma once
#ifdef _WIN32

#include <cuda_runtime.h> // Required for cudaExternalMemory_t
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
#include "Rendering/CullAndDraw.h"

namespace CudaGame {
    namespace Core {
        class CudaCore;
    }
}
// struct cudaGraphicsResource; // Removed - using cuda_runtime.h

namespace CudaGame {
namespace Rendering {

// AAA-quality deferred rendering pipeline for D3D12
// Features: G-Buffer, PBR lighting, DLSS upscaling, ray tracing
class DX12RenderPipeline {
public:
    DX12RenderPipeline();
    ~DX12RenderPipeline();


// ... (skipping unchanged parts for brevity in tool call if possible, but I must match Target) ...
// Actually, I can't skip parts in ReplacementContent easily without separate chunks.
// I will target the include block first, then the member block. 
// Using Multi-Edit is better or just 2 replace calls.
// The tool supports multiple chunks? No, this is replace_file_content (single block).
// I will just use two calls or one big one if contiguous.
// They are far apart (Lines 4 and 299). Two calls.


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
    CudaGame::Core::CudaCore* GetCudaCore() const { return m_cudaCore.get(); }
    
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
    
    // === Visual Verification (Testing) ===
    // Saves the current back buffer to a BMP file.
    // Ensure this is called at the end of a frame (Engine Loop).
    void SaveScreenshot(const std::string& filename);

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
    void SkyboxPass();        // Procedural atmospheric sky
    void ShadowPass();        // Generate shadow maps
    void LightingPass();      // Deferred lighting from G-Buffer
    void RayTracingPass();    // RT reflections, shadows, AO
    void DLSSPass();          // Upscale render res → display res
    
    void PostProcessPass();   // Bloom, tone mapping, etc.
    void UIPass();            // Render UI at display res
    void RenderMeshesWithMeshShader(ID3D12GraphicsCommandList* cmdList); // DX12 Ultimate path
    void ExecuteRenderKernel(void* pCmdList);           // GPU-driven kernel dispatch
    void EnsureIndirectBufferExists();                  // GPU path: allocate buffer only (no data copy)

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
    bool CreateSkyboxPSO();
    bool CreateMeshShaderPSO();           // DX12 Ultimate mesh shader PSO
    bool CheckMeshShaderSupport();        // Query D3D12 for mesh shader support
    
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
    
    // Screenshot Readback
    Microsoft::WRL::ComPtr<ID3D12Resource> m_readbackBuffer;
    uint32_t m_readbackBufferSize = 0;
    
    // === Shaders & Pipeline State ===
    Microsoft::WRL::ComPtr<ID3DBlob> m_geometryVS;
    Microsoft::WRL::ComPtr<ID3DBlob> m_gbufferPS;     // Geometry pass → G-Buffer MRTs
    Microsoft::WRL::ComPtr<ID3DBlob> m_geometryPS;    // Forward debug pass → swapchain
    Microsoft::WRL::ComPtr<ID3DBlob> m_lightingCS;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_gbufferPassPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_geometryPassPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_wireframePSO;  // Debug wireframe mode
    
    // Animation Support
    Microsoft::WRL::ComPtr<ID3DBlob> m_skinningVS;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_skinnedGeometryPassPSO;
    bool CreateSkinnedGeometryPassPSO();
    
    // Skybox rendering
    Microsoft::WRL::ComPtr<ID3DBlob> m_skyboxVS;
    Microsoft::WRL::ComPtr<ID3DBlob> m_skyboxPS;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_skyboxPSO;
    bool m_skyboxEnabled = true;
    
    // Mesh Shader Pipeline (DX12 Ultimate - AAA tier)
    Microsoft::WRL::ComPtr<ID3DBlob> m_amplificationShader;   // Per-meshlet culling
    Microsoft::WRL::ComPtr<ID3DBlob> m_meshShader;            // Vertex generation
    Microsoft::WRL::ComPtr<ID3DBlob> m_meshPixelShader;       // PBR shading
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_meshShaderPSO;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_meshShaderRootSig;
    bool m_meshShadersSupported = false; // Set during init
    bool m_meshShadersEnabled = false;   // Runtime toggle
    
    // Particle Rendering (Phase 7)
    Microsoft::WRL::ComPtr<ID3DBlob> m_particleMeshShader;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_particlePSO;
    bool CreateParticlePSO();
    
    struct ParticleBatch {
        ID3D12Resource* buffer;
        int count;
    };
    std::vector<ParticleBatch> m_particleBatches;
    
public:
    void RenderParticles(ID3D12Resource* particleBuffer, int count); // Internal use mostly
    void SubmitParticles(ID3D12Resource* particleBuffer, int count) {
        m_particleBatches.push_back({particleBuffer, count});
    }
    
private:    
    // === Constant Buffers ===
    Microsoft::WRL::ComPtr<ID3D12Resource> m_perFrameCB;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_perObjectCB;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_materialCB;
    PerFrameConstants* m_perFrameData = nullptr;
    PerObjectConstants* m_perObjectData = nullptr;
    MaterialConstants* m_materialData = nullptr;
    
    // === Indirect Execution (CudaCore Invention 1) ===
    enum class RenderPath {
        VertexShader_Fallback,      // Legacy/Safe loop
        Indirect_GPU_Driven         // High-performance ExecuteIndirect
    };

    struct IndirectCommand {
        D3D12_GPU_VIRTUAL_ADDRESS cbv;         // Per-object constant buffer (8 bytes)
        D3D12_GPU_VIRTUAL_ADDRESS materialCbv; // Material constant buffer (8 bytes)
        D3D12_VERTEX_BUFFER_VIEW vbv;          // Vertex Buffer View (16 bytes)
        D3D12_INDEX_BUFFER_VIEW ibv;           // Index Buffer View (16 bytes)
        D3D12_DRAW_INDEXED_ARGUMENTS drawArguments; // Draw command (20 bytes)
        uint32_t padding;                      // Align: 8+8+16+16+20 = 68. +4 = 72 bytes.
    };

    // === Kernel Execution Methods ===
    // void ExecuteRenderKernel_Fix(void* cmdList);
    void ExecuteVertexShaderPacket(ID3D12GraphicsCommandList* cmdList);
    void ExecuteIndirectPacket(ID3D12GraphicsCommandList* cmdList);
    
    // === Indirect Resources ===
    bool CreateCommandSignature();
    void UploadIndirectCommands(); // CPU Generation (Phase 2)
    
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_commandSignature;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_indirectCommandBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_indirectCommandUploadBuffer;
    uint32_t m_indirectCommandMaxCount = 0;

    RenderPath m_activePath = RenderPath::VertexShader_Fallback;

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

    // === CUDA Interop (Phase 3) ===
    // === CUDA Interop (Phase 3) ===
    std::unique_ptr<CudaGame::Core::CudaCore> m_cudaCore;
    
    // External Memory Handles (Modern)
    cudaExternalMemory_t m_extMemIndirectBuffer = nullptr;
    cudaExternalMemory_t m_extMemObjectCulling = nullptr;
    cudaExternalMemory_t m_extMemDrawCounter = nullptr;
    
    // Buffer sizes for CUDA mapping
    size_t m_objectCullingBufferSize = 0;
    size_t m_indirectBufferSize = 0;
    size_t m_drawCounterSize = 0;
    
    // Synchronization (Fences)
    Microsoft::WRL::ComPtr<ID3D12Fence> m_cullingFence;
    cudaExternalSemaphore_t m_cudaCullingFenceSem = nullptr;
    uint64_t m_cullingFenceValue = 0;
    
    // Mapped Device Pointers (for Kernel)
    void* m_devIndirectBuffer = nullptr;
    void* m_devObjectCulling = nullptr;
    void* m_devDrawCounter = nullptr;
    
    Microsoft::WRL::ComPtr<ID3D12Resource> m_objectCullingDataBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_objectCullingUploadBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_drawCounterBuffer;
    
    // Debug Readback (Phase 3 Verification)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_readbackDrawCounter;
    void ReadbackDrawCounter();

    void UploadObjectCullingData();
    
    // === Animation System (Phase 4) ===
    Microsoft::WRL::ComPtr<ID3D12Resource> m_globalBoneBuffer;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_boneSrvHeap;
    static const uint32_t MAX_BONES_GLOBAL = 10000; // 100 characters * 100 bones
    void UploadBoneMatrices();
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
