#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#ifdef _WIN32
#include "Rendering/DX12RenderPipeline.h"
// Helper for BMP writing
#include <fstream>
#include <vector>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t signature = 0x4D42; // "BM"
    uint32_t fileSize = 0;
    uint32_t reserved = 0;
    uint32_t dataOffset = 54; // 14 + 40
};
struct BMPInfoHeader {
    uint32_t headerSize = 40;
    int32_t width = 0;
    int32_t height = 0;
    uint16_t planes = 1;
    uint16_t bpp = 32; // RGBA
    uint32_t compression = 0;
    uint32_t imageSize = 0;
    int32_t xPixelsPerMeter = 0;
    int32_t yPixelsPerMeter = 0;
    uint32_t colorsUsed = 0;
    uint32_t colorsImportant = 0;
};
#pragma pack(pop)
#include "Rendering/ShaderCompiler.h"
#include "Core/CudaCore.h"
#include <iostream>
#include <codecvt>
#include <locale>
#include "Rendering/StreamHelper.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp> // Added for debug logs

#ifndef ASSET_DIR
#define ASSET_DIR "Assets/"
#endif

namespace CudaGame {
namespace Rendering {

DX12RenderPipeline::DX12RenderPipeline() = default;

DX12RenderPipeline::~DX12RenderPipeline() {
    Shutdown();
}

bool DX12RenderPipeline::Initialize(const InitParams& params) {
    if (m_initialized) {
        std::cerr << "[Pipeline] Already initialized" << std::endl;
        return true;
    }

    std::cout << "[Pipeline] Initializing AAA rendering pipeline..." << std::endl;
    std::cout << "[Pipeline] Display: " << params.displayWidth << "x" << params.displayHeight << std::endl;
    std::cout << "[Pipeline] DLSS: " << (params.enableDLSS ? "Enabled" : "Disabled") << std::endl;
    std::cout << "[Pipeline] Ray Tracing: " << (params.enableRayTracing ? "Enabled" : "Disabled") << std::endl;

    m_displayWidth = params.displayWidth;
    m_displayHeight = params.displayHeight;
    m_dlssEnabled = params.enableDLSS;
    m_rayTracingEnabled = params.enableRayTracing;
    
    m_activePath = RenderPath::VertexShader_Fallback; // CPU Fallback (GPU interop not yet initialized)

    // Step 0: Initialize CUDA (Phase 3)
    if (!m_cudaCore) {
        m_cudaCore = std::make_unique<Core::CudaCore>();
    }
    
    // Step 1: Initialize DX12 backend
    m_backend = std::make_unique<DX12RenderBackend>();
    if (!m_backend->Initialize()) {
        std::cerr << "[Pipeline] Failed to initialize DX12 backend" << std::endl;
        return false;
    }
    
    // Step 1b: Create swap chain for window presentation
    if (params.windowHandle) {
        if (!m_backend->CreateSwapchain(params.windowHandle, params.displayWidth, params.displayHeight)) {
            std::cerr << "[Pipeline] Failed to create swap chain" << std::endl;
            return false;
        }
    }

    // Initialize CUDA Context (needs D3D device)
    if (!m_cudaCore->Initialize(m_backend->GetDevice())) {
        std::cerr << "[Pipeline] Failed to initialize CUDA Core" << std::endl;
        return false;
    }

    // Phase 3: Enable GPU Path
    m_activePath = RenderPath::Indirect_GPU_Driven;


    // Step 2: Initialize DLSS if enabled
    if (m_dlssEnabled) {
        m_dlss = std::make_unique<StreamlineDLSSWrapper>();
        m_dlss->SetQualityMode(params.dlssMode);
        
        if (!m_dlss->Initialize(m_backend->GetDevice(), m_backend->GetCommandQueue(), 
                               m_displayWidth, m_displayHeight)) {
            std::cerr << "[Pipeline] DLSS initialization failed, disabling" << std::endl;
            m_dlssEnabled = false;
        } else {
            // Get optimal render resolution from DLSS
            m_dlss->GetRenderResolution(m_renderWidth, m_renderHeight);
            std::cout << "[Pipeline] Render resolution: " << m_renderWidth << "x" << m_renderHeight << std::endl;
        }
    }

    // If DLSS disabled, render at display resolution
    if (!m_dlssEnabled) {
        m_renderWidth = m_displayWidth;
        m_renderHeight = m_displayHeight;
    }

    // Step 3: Initialize ray tracing if enabled
    if (m_rayTracingEnabled) {
        m_rayTracing = std::make_unique<RayTracingSystem>();
        
        // Query ID3D12Device5 for ray tracing
        ID3D12Device5* device5 = nullptr;
        HRESULT hr = m_backend->GetDevice()->QueryInterface(IID_PPV_ARGS(&device5));
        if (FAILED(hr)) {
            std::cerr << "[Pipeline] Ray tracing requires ID3D12Device5, disabling" << std::endl;
            m_rayTracingEnabled = false;
        } else {
            if (!m_rayTracing->Initialize(device5, m_backend->GetCommandQueue())) {
                std::cerr << "[Pipeline] Ray tracing initialization failed, disabling" << std::endl;
                m_rayTracingEnabled = false;
            }
            device5->Release();
        }
    }

    // Step 4: Create render targets
    if (!CreateGBuffer()) {
        std::cerr << "[Pipeline] Failed to create G-Buffer" << std::endl;
        return false;
    }

    if (!CreateLightingBuffers()) {
        std::cerr << "[Pipeline] Failed to create lighting buffers" << std::endl;
        return false;
    }

    if (!CreateOutputBuffers()) {
        std::cerr << "[Pipeline] Failed to create output buffers" << std::endl;
        return false;
    }
    
    // Step 5: Compile shaders
    if (!CompileShaders()) {
        std::cerr << "[Pipeline] Failed to compile shaders" << std::endl;
        return false;
    }
    
    // Step 6: Create root signature and PSOs
    if (!CreateRootSignature()) {
        std::cerr << "[Pipeline] Failed to create root signature" << std::endl;
        return false;
    }


    
    // Create Indirect Command Signature (Phase 2)
    if (!CreateCommandSignature()) {
        std::cerr << "[Pipeline] Failed to create command signature" << std::endl;
        return false;
    }

    // First, PSO for writing into the G-Buffer MRTs.
    if (!CreateGBufferPassPSO()) {
        std::cerr << "[Pipeline] Failed to create G-Buffer geometry pass PSO" << std::endl;
        return false;
    }

    // Second, PSO for the forward debug pass that renders directly to the swapchain.
    if (!CreateGeometryPassPSO()) {
        std::cerr << "[Pipeline] Failed to create geometry pass PSO" << std::endl;
        return false;
    }
    
    // Third, Skinned PSO for animated characters (G-Buffer)
    if (!CreateSkinnedGeometryPassPSO()) {
        std::cerr << "[Pipeline] Failed to create Skinned PSO" << std::endl;
        return false;
    }
    
    // Skybox PSO (optional - continue without if shaders unavailable)
    CreateSkyboxPSO();
    
    // Mesh Shader PSO (DX12 Ultimate)
    if (CheckMeshShaderSupport()) {
        m_meshShadersEnabled = true;
        if (!CreateMeshShaderPSO()) {
            std::cerr << "[Pipeline] Failed to create mesh shader PSO, falling back to vertex shader" << std::endl;
            m_meshShadersEnabled = false;
        }
    } else {
        m_meshShadersEnabled = false;
    }
    
    // Create Particle PSO (Phase 7)
    if (m_meshShadersSupported && !CreateParticlePSO()) {
        std::cerr << "[Pipeline] Failed to create Particle PSO" << std::endl;
    }
    
    if (!CreateConstantBuffers()) {
        std::cerr << "[Pipeline] Failed to create constant buffers" << std::endl;
        return false;
    }

    m_initialized = true;
    std::cout << "[Pipeline] Initialized successfully" << std::endl;
    std::cout << "[Pipeline] G-Buffer: " << m_renderWidth << "x" << m_renderHeight << std::endl;
    std::cout << "[Pipeline] Output: " << m_displayWidth << "x" << m_displayHeight << std::endl;
    std::cout << "[Pipeline] Mesh Shaders Enabled: " << (m_meshShadersEnabled ? "TRUE" : "FALSE") << std::endl;
    
    return true;
}

void DX12RenderPipeline::Shutdown() {
    if (!m_initialized) return;

    // WAIT FOR GPU: Ensure all commands are finished before destroying resources
    if (m_backend) {
        std::cout << "[Pipeline] Waiting for GPU to finish..." << std::endl;
        m_backend->WaitForGPU();
    }

    DestroyRenderTargets();

    if (m_rayTracing) {
        m_rayTracing->Shutdown();
    }

    if (m_dlss) {
        m_dlss->Shutdown();
    }

    m_backend.reset();
    
    m_initialized = false;
    std::cout << "[Pipeline] Shutdown complete" << std::endl;
}

void DX12RenderPipeline::SetDLSSQualityMode(DLSSQualityMode mode) {
    if (m_dlss) {
        m_dlss->SetQualityMode(mode);
        
        // Update render resolution
        uint32_t newWidth, newHeight;
        m_dlss->GetRenderResolution(newWidth, newHeight);
        
        if (newWidth != m_renderWidth || newHeight != m_renderHeight) {
            std::cout << "[Pipeline] Render resolution changed: " 
                      << newWidth << "x" << newHeight << std::endl;
            
            // Would need to recreate G-Buffer here
            // For now, just log the change
            m_renderWidth = newWidth;
            m_renderHeight = newHeight;
        }
    }
}

// === Frame Rendering ===

void DX12RenderPipeline::BeginFrame(Camera* camera) {
    if (!m_initialized) return;
    m_camera = camera;
    m_frameIndex++;
    m_stats = FrameStats();  // Reset stats
    
    // Update per-frame constants
    // For DX12, reuse the same view/projection matrices the OpenGL path uses.
    // Camera already computes a right-handed view matrix with glm::lookAt and a
    // projection matrix via glm::perspective (with GLM_FORCE_DEPTH_ZERO_TO_ONE),
    // which gives us a 0..1 depth range compatible with D3D12. We disable
    // back-face culling in the PSO, so handedness differences do not matter.
    if (m_perFrameData && m_camera) {
        const glm::mat4 view = m_camera->GetViewMatrix();
        const glm::mat4 proj = m_camera->GetProjectionMatrix();

        // Compute view-projection in GLM's column-major space (proj * view),
        // exactly as in the OpenGL renderer.
        glm::mat4 viewProj = proj * view;

        // Write matrices directly; HLSL uses column_major to match GLM layout.
        m_perFrameData->viewMatrix       = view;
        m_perFrameData->projMatrix       = proj;
        m_perFrameData->viewProjMatrix   = viewProj;
        m_perFrameData->prevViewProjMatrix = m_prevViewProj;
        m_perFrameData->cameraPosition   = m_camera->GetPosition();
        m_perFrameData->time             = static_cast<float>(m_frameIndex) * 0.016f;  // Assuming 60 FPS
        m_perFrameData->deltaTime        = 0.016f;

        // Store for next frame (column-major)
        m_prevViewProj = viewProj;
    }
    
    // Begin backend frame with sky-colored clear
    // Use a procedural sky gradient (zenith blue to horizon light blue)
    // For now, use a static pleasant sky blue as the background
    float skyR = 0.4f;  // 102/255
    float skyG = 0.6f;  // 178/255  
    float skyB = 0.9f;  // 230/255 - nice sky blue
    glm::vec4 clearColor = glm::vec4(skyR, skyG, skyB, 1.0f);
    m_backend->BeginFrame(clearColor, m_displayWidth, m_displayHeight);
    m_backend->GetCommandList()->SetName(L"DX12 Render Loop");
}

void DX12RenderPipeline::RenderFrame() {
    if (!m_initialized || !m_camera) {
        return;
    }

    static uint64_t frameCount = 0;
    frameCount++;
    if (frameCount < 120 || frameCount % 60 == 0) std::cerr << "[Frame " << frameCount << "] Start" << std::endl;

    // AAA Rendering Pipeline:
    // 1. Shadow Pass
    // 2. G-Buffer Pass (Opaque)
    // 3. Lighting Pass (Deferred)
    // 4. Ray Tracing (Reflections/Shadows)
    // 5. DLSS (Upscale)
    // 6. Post Process (Bloom/ToneMap)
    
    // Phase 4: Upload Animation Matrices
    if (frameCount < 10) std::cerr << "[Frame " << frameCount << "] UploadBoneMatrices..." << std::endl;
    UploadBoneMatrices();
    if (frameCount < 10) std::cerr << "[Frame " << frameCount << "] UploadBoneMatrices done." << std::endl;
    
    // RESTORED: Re-enable passes to fix freeze
    // RESTORED: Re-enable passes to fix freeze
    if (frameCount < 10) std::cerr << "[Frame " << frameCount << "] ShadowPass..." << std::endl;
    ShadowPass();
    // GBufferPass(); // GPU Path disabled while debugging CPU Fallback
    
    // CPU Fallback G-Buffer Fill (Invention 1 repurposed)
    if (frameCount < 10) std::cerr << "[Frame " << frameCount << "] GeometryPass..." << std::endl;
    GeometryPass(); 
    if (frameCount < 10) std::cerr << "[Frame Trace] GeometryPass Done." << std::endl;
    
    if (frameCount < 10) std::cerr << "[Frame " << frameCount << "] SkyboxPass..." << std::endl;
    SkyboxPass();     
    if (frameCount < 10) std::cerr << "[Frame Trace] SkyboxPass Done." << std::endl;

    if (frameCount < 10) std::cerr << "[Frame " << frameCount << "] LightingPass..." << std::endl;
    LightingPass();
    if (frameCount < 10) std::cerr << "[Frame Trace] LightingPass Done." << std::endl;

    // === Pixel Readback Debug ===
    // === Pixel Readback Debug ===
    if (!m_readbackBuffer) {
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = 512; // Sufficient for row pitch
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        m_backend->GetDevice()->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readbackBuffer));
        m_readbackBuffer->SetName(L"DebugReadbackBuffer");
    }
    
    // Copy center pixel
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource = m_gBuffer.albedoRoughness;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = 0;
    
    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource = m_readbackBuffer.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint.Offset = 0;
    dstLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dstLoc.PlacedFootprint.Footprint.Width = 1;
    dstLoc.PlacedFootprint.Footprint.Height = 1;
    dstLoc.PlacedFootprint.Footprint.Depth = 1;
    dstLoc.PlacedFootprint.Footprint.RowPitch = 256; // Aligned
    
    // Source Box (Center)
    D3D12_BOX box = {};
    box.left = m_renderWidth / 2;
    box.right = box.left + 1;
    box.top = m_renderHeight / 2;
    box.bottom = box.top + 1;
    box.front = 0;
    box.back = 1;
    
    TransitionResource(m_gBuffer.albedoRoughness, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE); // Ensure
    m_backend->GetCommandList()->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &box);
    
    // Note: We need to WAIT for this copy before reading. 
    // Since we can't easily stall here without m_fence access inside Frame, 
    // we will read the *previous* frame's value or accept latency.
    // For simplicity: We will rely on EndFrame's WaitForGPU to flush this command list, 
    // then read it at the START of the next frame.
    
    static int readbackCounter = 0;
    readbackCounter++;
    if (readbackCounter < 300 || readbackCounter % 60 == 0) {
        std::cout << "[PIXEL DEBUG] Attempting readback..." << std::endl;
        
        // Force sync for debug
        m_backend->WaitForGPU(); 
        
        void* pData;
        if (SUCCEEDED(m_readbackBuffer->Map(0, nullptr, &pData))) {
             uint8_t* rgba = (uint8_t*)pData;
             std::cout << "[PIXEL DEBUG] Center Pixel (" << m_renderWidth/2 << "," << m_renderHeight/2 << "): " 
                       << (int)rgba[0] << ", " << (int)rgba[1] << ", " << (int)rgba[2] << ", " << (int)rgba[3] 
                       << std::endl;
             m_readbackBuffer->Unmap(0, nullptr);
        } else {
             std::cerr << "[PIXEL DEBUG] Failed to Map readback buffer!" << std::endl;
        }
    }
    
    if (m_dlssEnabled) {
        DLSSPass();
    }
    
    PostProcessPass();
    
    /* 
    // NEW KERNEL ARCHITECTURE (Invention 1)
    // Execute AFTER lighting/post-process to ensure it's visible (overlay)
    GeometryPass(); 
    */
    
    // Particle Pass (Phase 7)
    if (!m_particleBatches.empty()) {
        for (const auto& batch : m_particleBatches) {
            RenderParticles(batch.buffer, batch.count);
        }
        m_particleBatches.clear();
    }
    
    UIPass();
    
    // Calculate total frame time
    // ...
    if (frameCount < 120 || frameCount % 60 == 0) std::cout << "[Frame " << frameCount << "] End" << std::endl;
}



void DX12RenderPipeline::EndFrame() {
    if (!m_initialized) return;

    // Transition backbuffer to present state
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_backend->GetCurrentBackBuffer();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = 0; 
    
    m_backend->GetCommandList()->ResourceBarrier(1, &barrier);

    // Execute command list
    m_backend->GetCommandList()->Close();
    ID3D12CommandList* ppCommandLists[] = { m_backend->GetCommandList() };
    m_backend->GetCommandQueue()->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present
    m_backend->Present();
    
    // Core Logic: Wait for GPU here if needed, or rely on Backend frame fencing.
    // For now, WaitForGPU ensures stability until Fence logic is verified robust.
    m_backend->WaitForGPU(); 
}

// === Render Passes ===

void DX12RenderPipeline::GBufferPass() {
    if (m_meshes.empty()) {
        return;
    }

    ID3D12GraphicsCommandList* cmdList = m_backend->GetCommandList();

    // Bind root signature and G-Buffer PSO
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    cmdList->SetPipelineState(m_gbufferPassPSO.Get());
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Viewport/scissor at render resolution (DLSS render res)
    D3D12_VIEWPORT viewport = {};
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width  = (FLOAT)m_renderWidth;
    viewport.Height = (FLOAT)m_renderHeight;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    cmdList->RSSetViewports(1, &viewport);

    D3D12_RECT scissor = {};
    scissor.left   = 0;
    scissor.top    = 0;
    scissor.right  = (LONG)m_renderWidth;
    scissor.bottom = (LONG)m_renderHeight;
    cmdList->RSSetScissorRects(1, &scissor);

    // Bind G-Buffer render targets + depth
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[4] = {
        m_gBuffer.albedoRTV,
        m_gBuffer.normalRTV,
        m_gBuffer.emissiveRTV,
        m_gBuffer.velocityRTV
    };
    cmdList->OMSetRenderTargets(4, rtvs, FALSE, &m_gBuffer.depthDSV);

    // Clear G-Buffer attachments
    const float clearBlack[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        cmdList->ClearRenderTargetView(rtvs[i], clearBlack, 0, nullptr);
    }
    cmdList->ClearDepthStencilView(m_gBuffer.depthDSV, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    // Bind per-frame constants
    D3D12_GPU_VIRTUAL_ADDRESS perFrameAddress = m_perFrameCB->GetGPUVirtualAddress();
    cmdList->SetGraphicsRootConstantBufferView(0, perFrameAddress);
    
    // Bind Global Bone Buffer (Root Param 3) - Only if it exists
    if (m_globalBoneBuffer) {
        cmdList->SetGraphicsRootShaderResourceView(3, m_globalBoneBuffer->GetGPUVirtualAddress());
    }

    // Constant buffer slot sizes (256-byte aligned)
    const uint32_t perObjectSlotSize = 256;
    const uint32_t materialSlotSize  = 256;

    uint32_t meshIndex = 0;
    // Track current PSO to minimize switching
    ID3D12PipelineState* currentPSO = m_gbufferPassPSO.Get();
    
    for (D3D12Mesh* mesh : m_meshes) {
        meshIndex++;
        if (!mesh) continue;
        
        bool isSkinned = (mesh->GetSkeleton() != nullptr);
        ID3D12PipelineState* requiredPSO = isSkinned ? m_skinnedGeometryPassPSO.Get() : m_gbufferPassPSO.Get();
        
        if (requiredPSO != currentPSO) {
            cmdList->SetPipelineState(requiredPSO);
            currentPSO = requiredPSO;
        }

        // Per-object constants
        if (m_perObjectData) {
            glm::mat4 world = mesh->transform;
            uint32_t meshSlot = meshIndex - 1;
            if (meshSlot >= MAX_MESHES_PER_FRAME) break;

            PerObjectConstants* objCB = reinterpret_cast<PerObjectConstants*>(
                reinterpret_cast<uint8_t*>(m_perObjectData) + meshSlot * perObjectSlotSize);
            objCB->worldMatrix = world;
            objCB->prevWorldMatrix = world;
            objCB->normalMatrix = glm::transpose(glm::inverse(world));
            
            // Animation Data
            if (isSkinned) {
                objCB->boneOffset = mesh->globalBoneOffset;
                objCB->isSkinned = 1;
            } else {
                objCB->boneOffset = 0;
                objCB->isSkinned = 0;
            }
        }

        // Material constants
        if (m_materialData) {
            uint32_t meshSlot = meshIndex - 1;
            if (meshSlot >= MAX_MESHES_PER_FRAME) break;

            MaterialConstants* matCB = reinterpret_cast<MaterialConstants*>(
                reinterpret_cast<uint8_t*>(m_materialData) + meshSlot * materialSlotSize);

            const Material& mat = mesh->GetMaterial();
            matCB->albedoColor      = mat.albedoColor;
            matCB->roughness        = mat.roughness;
            matCB->metallic         = mat.metallic;
            matCB->ambientOcclusion = mat.ambientOcclusion;
            matCB->emissiveStrength = mat.emissiveStrength;
            matCB->emissiveColor    = mat.emissiveColor;
        }

        // Bind CBVs for this mesh
        uint32_t meshSlot = meshIndex - 1;
        D3D12_GPU_VIRTUAL_ADDRESS perObjectAddress = m_perObjectCB->GetGPUVirtualAddress() + meshSlot * perObjectSlotSize;
        D3D12_GPU_VIRTUAL_ADDRESS materialAddress  = m_materialCB->GetGPUVirtualAddress()  + meshSlot * materialSlotSize;
        cmdList->SetGraphicsRootConstantBufferView(1, perObjectAddress);
        cmdList->SetGraphicsRootConstantBufferView(2, materialAddress);

        // Geometry
        const D3D12_VERTEX_BUFFER_VIEW& vbv = mesh->GetVertexBufferView();
        const D3D12_INDEX_BUFFER_VIEW& ibv = mesh->GetIndexBufferView();
        cmdList->IASetVertexBuffers(0, 1, &vbv);
        cmdList->IASetIndexBuffer(&ibv);

        cmdList->DrawIndexedInstanced(mesh->GetIndexCount(), 1, 0, 0, 0);
    }
}

void DX12RenderPipeline::GeometryPass() {
    std::cerr << "[GeometryPass] Entered." << std::endl;
    // KERNEL ARCHITECTURE (Invention 1): Redirect to new dispatcher
    if (m_meshes.empty()) {
        static bool warned = false;
        if (!warned) {
             std::cerr << "[Pipeline] WARNING: No meshes to render!" << std::endl;
             warned = true;
        }
        return;
    }
    std::cerr << "[GeometryPass] Meshes: " << m_meshes.size() << std::endl;

    // DEBUG: Log render path status
    static int logCounter = 0;
    logCounter++;
    // Log first 300 frames continuously, then every 60
    if (logCounter < 300 || logCounter % 60 == 0) {
        std::string pathName = (m_activePath == RenderPath::Indirect_GPU_Driven) ? "Indirect (GPU)" : "Fallback (CPU)";
        std::cerr << "[Pipeline DEBUG] Path: " << pathName 
                  << " | Meshes: " << m_meshes.size() 
                  << " | CamPos: " << (m_camera ? glm::to_string(m_camera->GetPosition()) : "NULL") 
                  << std::endl;
    }

    if (m_activePath == RenderPath::Indirect_GPU_Driven) {
        // GPU-Driven Rendering Path using CUDA CullAndDraw kernel
        std::cerr << "[GeometryPass] Executing GPU-Driven path via ExecuteRenderKernel..." << std::endl;
        ExecuteRenderKernel(static_cast<void*>(m_backend->GetCommandList()));
        std::cerr << "[GeometryPass] ExecuteRenderKernel returned." << std::endl;
        return;
    }
    // CPU Fallback path - generates actual draw commands

    // LEGACY / FALLBACK PATH (Draw Loop)
    ID3D12GraphicsCommandList* cmdList = m_backend->GetCommandList();
    
    // Bind root signature and PSO
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    
    // Validate matrices
    if (m_perFrameData) {
        // Sanity check projection
        if (m_perFrameData->projMatrix[3][3] != 0.0f) { // Perspective usually has 0 there
             // Just a quick heuristic check
        }
    }

    // Explicitly choose PSO
    if (m_debugMode == DebugMode::WIREFRAME && m_wireframePSO) {
        cmdList->SetPipelineState(m_wireframePSO.Get());
    } else {
        if (!m_gbufferPassPSO) {
            std::cerr << "[Pipeline CRITICAL] m_gbufferPassPSO is NULL!" << std::endl;
            return;
        }
        cmdList->SetPipelineState(m_gbufferPassPSO.Get());
    }
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // Set viewport and scissor to RENDER resolution (for G-Buffer)
    D3D12_VIEWPORT viewport = {};
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = (FLOAT)m_renderWidth;
    viewport.Height = (FLOAT)m_renderHeight;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    cmdList->RSSetViewports(1, &viewport);
    
    D3D12_RECT scissor = {};
    scissor.left = 0;
    scissor.top = 0;
    scissor.right = (LONG)m_renderWidth;
    scissor.bottom = (LONG)m_renderHeight;
    cmdList->RSSetScissorRects(1, &scissor);
    
    // Bind per-frame constants
    D3D12_GPU_VIRTUAL_ADDRESS perFrameAddress = m_perFrameCB->GetGPUVirtualAddress();
    cmdList->SetGraphicsRootConstantBufferView(0, perFrameAddress);
    
    // DEFERRED RENDERING: Bind G-Buffer MRTs
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[4] = {
        m_gBuffer.albedoRTV,
        m_gBuffer.normalRTV,
        m_gBuffer.emissiveRTV,
        m_gBuffer.velocityRTV
    };
    cmdList->OMSetRenderTargets(4, rtvs, FALSE, &m_gBuffer.depthDSV);

    // Clear G-Buffer for fallback path (since GBufferPass isn't called)
    const float clearBlack[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        cmdList->ClearRenderTargetView(rtvs[i], clearBlack, 0, nullptr);
    }
    cmdList->ClearDepthStencilView(m_gBuffer.depthDSV, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    
    uint32_t drawCallCount = 0;
    uint32_t triangleCount = 0;

    // Constant buffer slot sizes (256-byte aligned)
    const uint32_t perObjectSlotSize = 256;
    const uint32_t materialSlotSize  = 256;
    
    uint32_t meshIndex = 0;
    for (D3D12Mesh* mesh : m_meshes) {
        meshIndex++;
        if (!mesh) continue;
        
        // Debug first few meshes
        if ((logCounter < 300 || logCounter % 60 == 0) && meshIndex <= 3) {
            std::cout << "  [Draw " << meshIndex << "] Pos: " << glm::to_string(mesh->transform[3]) 
                      << " Indices: " << mesh->GetIndexCount() 
                      << " Skinned: " << (mesh->GetSkeleton() ? "YES" : "NO") << std::endl;
        }

        // Update per-object constants
        // ... (rest of update logic identical to previous) ...
        if (m_perObjectData) {
            glm::mat4 world = mesh->transform;
            uint32_t meshSlot = meshIndex - 1;
            if (meshSlot >= MAX_MESHES_PER_FRAME) break;

            PerObjectConstants* objCB = reinterpret_cast<PerObjectConstants*>(
                reinterpret_cast<uint8_t*>(m_perObjectData) + meshSlot * perObjectSlotSize);

            objCB->worldMatrix = world;
            objCB->prevWorldMatrix = world;
            objCB->normalMatrix = glm::transpose(glm::inverse(world));
            
            bool isSkinned = (mesh->GetSkeleton() != nullptr);
            objCB->isSkinned = isSkinned ? 1 : 0;
             if (isSkinned) {
                objCB->boneOffset = mesh->globalBoneOffset;
            } else {
                objCB->boneOffset = 0;
            }
        }
        
        // Update material constants
        if (m_materialData) {
            uint32_t meshSlot = meshIndex - 1;
            if (meshSlot >= MAX_MESHES_PER_FRAME) break;

            MaterialConstants* matCB = reinterpret_cast<MaterialConstants*>(
                reinterpret_cast<uint8_t*>(m_materialData) + meshSlot * materialSlotSize);
            
            // Force Alpha to 1.0 just in case
            Material mat = mesh->GetMaterial();
            mat.albedoColor.a = 1.0f; 

            matCB->albedoColor       = mat.albedoColor;
            matCB->roughness         = mat.roughness;
            matCB->metallic          = mat.metallic;
            matCB->ambientOcclusion  = mat.ambientOcclusion;
            matCB->emissiveStrength  = mat.emissiveStrength;
            matCB->emissiveColor     = mat.emissiveColor;
        }
        
        // Bind CBVs
        uint32_t meshSlot = meshIndex - 1;
        D3D12_GPU_VIRTUAL_ADDRESS perObjectAddress = m_perObjectCB->GetGPUVirtualAddress() + meshSlot * perObjectSlotSize;
        D3D12_GPU_VIRTUAL_ADDRESS materialAddress  = m_materialCB->GetGPUVirtualAddress()  + meshSlot * materialSlotSize;
        cmdList->SetGraphicsRootConstantBufferView(1, perObjectAddress);
        cmdList->SetGraphicsRootConstantBufferView(2, materialAddress);
        
        // Force Standard PSO for debugging (ignore skinning for a moment if it's broken)
        // cmdList->SetPipelineState(m_geometryPassPSO.Get()); 
        
        // Proper PSO selection
        if (mesh->GetSkeleton() && m_globalBoneBuffer) {
             cmdList->SetGraphicsRootShaderResourceView(3, m_globalBoneBuffer->GetGPUVirtualAddress());
             cmdList->SetPipelineState(m_skinnedGeometryPassPSO.Get());
        } else {
             cmdList->SetPipelineState(m_geometryPassPSO.Get());
        }

        const D3D12_VERTEX_BUFFER_VIEW& vbv = mesh->GetVertexBufferView();
        const D3D12_INDEX_BUFFER_VIEW& ibv = mesh->GetIndexBufferView();
        
        cmdList->IASetVertexBuffers(0, 1, &vbv);
        cmdList->IASetIndexBuffer(&ibv);
        
        cmdList->DrawIndexedInstanced(mesh->GetIndexCount(), 1, 0, 0, 0);
        
        drawCallCount++;
        triangleCount += mesh->GetIndexCount() / 3;
    }
    
    m_stats.geometryPassMs = 1.0f;
    m_stats.drawCalls = drawCallCount;
    m_stats.triangles = triangleCount;
}

void DX12RenderPipeline::ShadowPass() {
    // TODO: Implement shadow map generation
    // - Render from light perspective
    // - Write depth only
    
    m_stats.geometryPassMs += 0.5f;  // Placeholder
}

void DX12RenderPipeline::LightingPass() {
    // DEBUG: Clear LitColor to RED to verify chain
    ID3D12GraphicsCommandList* cmdList = m_backend->GetCommandList();
    
    // Transition LitColor to RTV
    TransitionResource(m_lightingBuffers.litColor, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    
    // Clear to Red
    float clearColor[] = {1.0f, 0.0f, 0.0f, 1.0f}; 
    cmdList->ClearRenderTargetView(m_lightingBuffers.litColorRTV, clearColor, 0, nullptr);
    
    // Transition back for DLSS
    TransitionResource(m_lightingBuffers.litColor, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    
    m_stats.lightingPassMs = 0.1f;
}

void DX12RenderPipeline::SkyboxPass() {
    if (!m_skyboxEnabled || !m_skyboxPSO) {
        return;
    }
    
    ID3D12GraphicsCommandList* cmdList = m_backend->GetCommandList();
    
    // Set skybox PSO
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    cmdList->SetPipelineState(m_skyboxPSO.Get());
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // Set viewport and scissor
    D3D12_VIEWPORT viewport = {};
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = (FLOAT)m_displayWidth;
    viewport.Height = (FLOAT)m_displayHeight;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    cmdList->RSSetViewports(1, &viewport);
    
    D3D12_RECT scissor = {};
    scissor.left = 0;
    scissor.top = 0;
    scissor.right = (LONG)m_displayWidth;
    scissor.bottom = (LONG)m_displayHeight;
    cmdList->RSSetScissorRects(1, &scissor);
    
    // Bind per-frame constants (camera matrices needed for view direction)
    D3D12_GPU_VIRTUAL_ADDRESS perFrameAddress = m_perFrameCB->GetGPUVirtualAddress();
    cmdList->SetGraphicsRootConstantBufferView(0, perFrameAddress);
    
    // Bind swap chain render target (skybox renders behind everything)
    D3D12_CPU_DESCRIPTOR_HANDLE swapChainRTV = m_backend->GetCurrentRenderTargetView();
    D3D12_CPU_DESCRIPTOR_HANDLE mainDSV = m_backend->GetDepthStencilView();
    cmdList->OMSetRenderTargets(1, &swapChainRTV, FALSE, &mainDSV);
    
    // Draw fullscreen triangle (3 vertices, no vertex buffer)
    cmdList->DrawInstanced(3, 1, 0, 0);
}

void DX12RenderPipeline::RayTracingPass() {
    // TODO: Implement ray tracing
    // - Trace reflection rays (1 ray/pixel)
    // - Trace shadow rays (1 ray/pixel per light)
    // - Trace AO rays (4-8 rays/pixel)
    // - Write to RT buffers
    
    m_stats.rayTracingPassMs = 5.0f;  // Placeholder
}

void DX12RenderPipeline::DLSSPass() {
    if (!m_dlss || !m_dlssEnabled) {
        return;
    }

    // Call DLSS to upscale render resolution → display resolution
    StreamlineDLSSWrapper::DLSSInputs inputs = {};
    inputs.colorBuffer = m_lightingBuffers.litColor;     // Input: lit color at render res
    inputs.depthBuffer = m_gBuffer.depth;                // Depth buffer
    inputs.motionVectors = m_gBuffer.velocity;           // Motion vectors
    inputs.outputBuffer = m_outputBuffers.finalColor;    // Output: upscaled at display res
    inputs.exposureTexture = nullptr;                     // Optional
    inputs.jitterOffsetX = 0.0f;  // TODO: Calculate from frame index
    inputs.jitterOffsetY = 0.0f;
    inputs.sharpness = 0.0f;      // Let DLSS decide
    inputs.reset = false;          // Set true on scene cuts
    inputs.preExposure = 1.0f;
    
    m_dlss->Execute(m_backend->GetCommandList(), inputs);
    
    m_stats.dlssPassMs = 1.5f;  // Placeholder
}

void DX12RenderPipeline::PostProcessPass() {
    // Copy Albedo (G-Buffer) to SwapChain for DEBUG
    ID3D12GraphicsCommandList* cmdList = m_backend->GetCommandList();
    ID3D12Resource* backBuffer = m_backend->GetCurrentBackBuffer();
    
    // Use Albedo as source (Validation of Draw)
    ID3D12Resource* source = m_gBuffer.albedoRoughness; 
    
    if (!source || !backBuffer) return;

    // Transition Source to Copy Source
    TransitionResource(source, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    // Transition BackBuffer to Copy Dest
    TransitionResource(backBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
    
    // Copy
    cmdList->CopyResource(backBuffer, source);
    
    // Transition BackBuffer Back to Render Target (for EndFrame)
    TransitionResource(backBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
    // Transition Source back
    TransitionResource(source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
}

void DX12RenderPipeline::UIPass() {
    // TODO: Implement UI rendering
    // - Render UI elements at display resolution
    // - Overlay debug text
}

// === Resource Creation ===

bool DX12RenderPipeline::CreateGBuffer() {
    std::cout << "[Pipeline] Creating G-Buffer: " << m_renderWidth << "x" << m_renderHeight << std::endl;

    // Create albedo/roughness buffer (RGBA8)
    m_gBuffer.albedoRoughness = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        L"GBuffer_AlbedoRoughness"
    );
    if (!m_gBuffer.albedoRoughness) {
        std::cerr << "[Pipeline] Failed to create albedo/roughness buffer" << std::endl;
        return false;
    }
    std::cout << "  - Albedo/Roughness: RGBA8_UNORM [OK]" << std::endl;
    
    // Create normal/metallic buffer (RGBA16F)
    m_gBuffer.normalMetallic = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        L"GBuffer_NormalMetallic"
    );
    if (!m_gBuffer.normalMetallic) {
        std::cerr << "[Pipeline] Failed to create normal/metallic buffer" << std::endl;
        return false;
    }
    std::cout << "  - Normal/Metallic: RGBA16_FLOAT [OK]" << std::endl;
    
    // Create emissive/AO buffer (RGBA16F)
    m_gBuffer.emissiveAO = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        L"GBuffer_EmissiveAO"
    );
    if (!m_gBuffer.emissiveAO) {
        std::cerr << "[Pipeline] Failed to create emissive/AO buffer" << std::endl;
        return false;
    }
    std::cout << "  - Emissive/AO: RGBA16_FLOAT [OK]" << std::endl;
    
    // Create velocity buffer (RG16F)
    m_gBuffer.velocity = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_R16G16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        L"GBuffer_Velocity"
    );
    if (!m_gBuffer.velocity) {
        std::cerr << "[Pipeline] Failed to create velocity buffer" << std::endl;
        return false;
    }
    std::cout << "  - Velocity: RG16_FLOAT [OK]" << std::endl;
    
    // Create depth buffer (D32F)
    m_gBuffer.depth = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_D32_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        L"GBuffer_Depth"
    );
    if (!m_gBuffer.depth) {
        std::cerr << "[Pipeline] Failed to create depth buffer" << std::endl;
        return false;
    }
    std::cout << "  - Depth: D32_FLOAT [OK]" << std::endl;
    
    // Create descriptor views for G-Buffer
    ID3D12DescriptorHeap* rtvHeap = m_backend->GetRTVHeap();
    ID3D12DescriptorHeap* dsvHeap = m_backend->GetDSVHeap();
    UINT rtvDescriptorSize = m_backend->GetRTVDescriptorSize();
    UINT dsvDescriptorSize = m_backend->GetDSVDescriptorSize();
    
    // Get base RTV handle (skip first 3 for swap chain backbuffers)
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += 3 * rtvDescriptorSize;  // Skip swapchain RTVs
    
    // Create RTVs for G-Buffer
    m_backend->GetDevice()->CreateRenderTargetView(m_gBuffer.albedoRoughness, nullptr, rtvHandle);
    m_gBuffer.albedoRTV = rtvHandle;
    rtvHandle.ptr += rtvDescriptorSize;
    
    m_backend->GetDevice()->CreateRenderTargetView(m_gBuffer.normalMetallic, nullptr, rtvHandle);
    m_gBuffer.normalRTV = rtvHandle;
    rtvHandle.ptr += rtvDescriptorSize;
    
    m_backend->GetDevice()->CreateRenderTargetView(m_gBuffer.emissiveAO, nullptr, rtvHandle);
    m_gBuffer.emissiveRTV = rtvHandle;
    rtvHandle.ptr += rtvDescriptorSize;
    
    m_backend->GetDevice()->CreateRenderTargetView(m_gBuffer.velocity, nullptr, rtvHandle);
    m_gBuffer.velocityRTV = rtvHandle;
    
    // Create DSV for G-Buffer depth
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    dsvHandle.ptr += dsvDescriptorSize;  // Skip main depth buffer (index 0)
    
    m_backend->GetDevice()->CreateDepthStencilView(m_gBuffer.depth, nullptr, dsvHandle);
    m_gBuffer.depthDSV = dsvHandle;
    
    std::cout << "[Pipeline] G-Buffer descriptor views created" << std::endl;
    
    return true;
}

bool DX12RenderPipeline::CreateLightingBuffers() {
    std::cout << "[Pipeline] Creating lighting buffers: " << m_renderWidth << "x" << m_renderHeight << std::endl;
    
    // Create HDR lit color buffer (RGBA16F)
    m_lightingBuffers.litColor = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        L"Lighting_LitColor"
    );
    if (!m_lightingBuffers.litColor) {
        std::cerr << "[Pipeline] Failed to create lit color buffer" << std::endl;
        return false;
    }
    std::cout << "  - Lit Color: RGBA16_FLOAT (HDR) [OK]" << std::endl;
    
    // Create RT reflections buffer (RGBA16F)
    m_lightingBuffers.reflections = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        L"Lighting_Reflections"
    );
    if (!m_lightingBuffers.reflections) {
        std::cerr << "[Pipeline] Failed to create reflections buffer" << std::endl;
        return false;
    }
    std::cout << "  - Reflections: RGBA16_FLOAT [OK]" << std::endl;
    
    // Create RT shadows buffer (R8)
    m_lightingBuffers.shadows = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_R8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        L"Lighting_Shadows"
    );
    if (!m_lightingBuffers.shadows) {
        std::cerr << "[Pipeline] Failed to create shadows buffer" << std::endl;
        return false;
    }
    std::cout << "  - Shadows: R8_UNORM [OK]" << std::endl;
    
    // Create RT AO buffer (R8)
    m_lightingBuffers.ambientOcclusion = CreateTexture2D(
        m_renderWidth, m_renderHeight,
        DXGI_FORMAT_R8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        L"Lighting_AO"
    );
    if (!m_lightingBuffers.ambientOcclusion) {
        std::cerr << "[Pipeline] Failed to create AO buffer" << std::endl;
        return false;
    }
    std::cout << "  - AO: R8_UNORM [OK]" << std::endl;

    // Create RTV for LitColor
    // Offset: 3 (Swapchain) + 4 (G-Buffer) = 7
    ID3D12DescriptorHeap* rtvHeap = m_backend->GetRTVHeap();
    UINT rtvDescriptorSize = m_backend->GetRTVDescriptorSize();
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += (3 + 4) * rtvDescriptorSize;
    
    m_backend->GetDevice()->CreateRenderTargetView(m_lightingBuffers.litColor, nullptr, rtvHandle);
    m_lightingBuffers.litColorRTV = rtvHandle;
    std::cout << "  - LitColor RTV Created at Index 7" << std::endl;
    
    return true;
}

bool DX12RenderPipeline::CreateOutputBuffers() {
    std::cout << "[Pipeline] Creating output buffers: " << m_displayWidth << "x" << m_displayHeight << std::endl;
    
    // Create final color buffer at display resolution (DLSS output)
    m_outputBuffers.finalColor = CreateTexture2D(
        m_displayWidth, m_displayHeight,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        L"Output_FinalColor"
    );
    if (!m_outputBuffers.finalColor) {
        std::cerr << "[Pipeline] Failed to create final color buffer" << std::endl;
        return false;
    }
    std::cout << "  - Final Color: RGBA16_FLOAT (DLSS output) [OK]" << std::endl;
    
    // Create post-process buffer
    m_outputBuffers.postProcess = CreateTexture2D(
        m_displayWidth, m_displayHeight,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        L"Output_PostProcess"
    );
    if (!m_outputBuffers.postProcess) {
        std::cerr << "[Pipeline] Failed to create post-process buffer" << std::endl;
        return false;
    }
    std::cout << "  - Post-Process: RGBA8_UNORM [OK]" << std::endl;
    
    return true;
}

void DX12RenderPipeline::DestroyRenderTargets() {
    // TODO: Release all ID3D12Resource objects
    
    // G-Buffer
    if (m_gBuffer.albedoRoughness) m_gBuffer.albedoRoughness->Release();
    if (m_gBuffer.normalMetallic) m_gBuffer.normalMetallic->Release();
    if (m_gBuffer.emissiveAO) m_gBuffer.emissiveAO->Release();
    if (m_gBuffer.velocity) m_gBuffer.velocity->Release();
    if (m_gBuffer.depth) m_gBuffer.depth->Release();
    
    // Lighting buffers
    if (m_lightingBuffers.litColor) m_lightingBuffers.litColor->Release();
    if (m_lightingBuffers.reflections) m_lightingBuffers.reflections->Release();
    if (m_lightingBuffers.shadows) m_lightingBuffers.shadows->Release();
    if (m_lightingBuffers.ambientOcclusion) m_lightingBuffers.ambientOcclusion->Release();
    
    // Output buffers
    if (m_outputBuffers.finalColor) m_outputBuffers.finalColor->Release();
    if (m_outputBuffers.postProcess) m_outputBuffers.postProcess->Release();
}

ID3D12Resource* DX12RenderPipeline::CreateTexture2D(uint32_t width, uint32_t height, 
                                                    DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                                                    const wchar_t* name) {
    // Determine heap type based on flags
    D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
    
    // Determine initial state from flags
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
        initialState = D3D12_RESOURCE_STATE_RENDER_TARGET;
    } else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) {
        initialState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    } else if (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
        initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    
    // Describe the texture
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    
    // Heap properties for default heap (GPU memory)
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = heapType;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;
    
    // Clear value for render targets/depth
    D3D12_CLEAR_VALUE clearValue = {};
    D3D12_CLEAR_VALUE* pClearValue = nullptr;
    
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
        clearValue.Format = format;
        clearValue.Color[0] = 0.0f;
        clearValue.Color[1] = 0.0f;
        clearValue.Color[2] = 0.0f;
        clearValue.Color[3] = 0.0f;
        pClearValue = &clearValue;
    } else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) {
        clearValue.Format = format;
        clearValue.DepthStencil.Depth = 1.0f;
        clearValue.DepthStencil.Stencil = 0;
        pClearValue = &clearValue;
    }
    
    // Create the resource
    ID3D12Resource* resource = nullptr;
    HRESULT hr = m_backend->GetDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_SHARED,
        &desc,
        initialState,
        pClearValue,
        IID_PPV_ARGS(&resource)
    );
    
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create texture: " << std::hex << hr << std::endl;
        return nullptr;
    }
    
    // Set debug name
    if (name && resource) {
        resource->SetName(name);
    }
    
    return resource;
}

void DX12RenderPipeline::TransitionResource(ID3D12Resource* resource, 
                                           D3D12_RESOURCE_STATES before,
                                           D3D12_RESOURCE_STATES after) {
    if (!resource || before == after) {
        return;
    }
    
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    
    m_backend->GetCommandList()->ResourceBarrier(1, &barrier);
}

// === Shader & PSO Management ===

bool DX12RenderPipeline::CompileShaders() {
    std::cout << "[Pipeline] Compiling shaders..." << std::endl;
    
    // Get shader path (use ASSET_DIR for absolute path)
    std::string assetDirStr = ASSET_DIR;
    std::wstring assetDir(assetDirStr.begin(), assetDirStr.end());
    std::wstring shaderPath = assetDir + L"/shaders/dx12/";
    
    // Compile geometry pass vertex shader
    m_geometryVS = ShaderCompiler::CompileFromFile(
        shaderPath + L"GeometryPass_VS.hlsl",
        "main",  // Entry point in HLSL
        ShaderCompiler::ShaderType::Vertex,
        true  // Enable debug for now
    );
    if (!m_geometryVS) {
        std::cerr << "[Pipeline] Failed to compile geometry VS" << std::endl;
        return false;
    }
    
    // Compile G-Buffer pixel shader (writes MRTs)
    m_gbufferPS = ShaderCompiler::CompileFromFile(
        shaderPath + L"GeometryPass_PS.hlsl",
        "main",  // Entry point in HLSL
        ShaderCompiler::ShaderType::Pixel,
        true
    );
    if (!m_gbufferPS) {
        std::cerr << "[Pipeline] Failed to compile G-Buffer PS" << std::endl;
        return false;
    }
    
    // Compile forward pass pixel shader (single render target)
    m_geometryPS = ShaderCompiler::CompileFromFile(
        shaderPath + L"ForwardPass_PS.hlsl",
        "main",  // Entry point in HLSL
        ShaderCompiler::ShaderType::Pixel,
        true
    );
    if (!m_geometryPS) {
        std::cerr << "[Pipeline] Failed to compile geometry PS" << std::endl;
        return false;
    }
    
    // Compile deferred lighting compute shader
    m_lightingCS = ShaderCompiler::CompileFromFile(
        shaderPath + L"DeferredLighting_CS.hlsl",
        "main",  // Entry point in HLSL
        ShaderCompiler::ShaderType::Compute,
        true
    );
    if (!m_lightingCS) {
        std::cerr << "[Pipeline] Failed to compile lighting CS" << std::endl;
        return false;
    }
    
    // Compile skybox shaders (optional - continue without if they fail)
    m_skyboxVS = ShaderCompiler::CompileFromFile(
        shaderPath + L"Skybox_VS.hlsl",
        "main",
        ShaderCompiler::ShaderType::Vertex,
        true
    );
    
    // Compile Skinning VS (Phase 4)
    m_skinningVS = ShaderCompiler::CompileFromFile(
        shaderPath + L"SkinningShader.hlsl",
        "main",
        ShaderCompiler::ShaderType::Vertex,
        true
    );
    if (!m_skinningVS) {
        std::cerr << "[Pipeline] Failed to compile Skinning VS" << std::endl;
        return false;
    }
    m_skyboxPS = ShaderCompiler::CompileFromFile(
        shaderPath + L"Skybox_PS.hlsl",
        "main",
        ShaderCompiler::ShaderType::Pixel,
        true
    );
    if (m_skyboxVS && m_skyboxPS) {
        std::cout << "[Pipeline] Skybox shaders compiled successfully" << std::endl;
    } else {
        std::cerr << "[Pipeline] Skybox shaders not available (non-critical)" << std::endl;
        m_skyboxEnabled = false;
    }
    
    std::cout << "[Pipeline] All shaders compiled successfully" << std::endl;
    std::cout << "[Pipeline] All shaders compiled successfully" << std::endl;
    m_particleMeshShader = ShaderCompiler::CompileFromFile(shaderPath + L"ParticleMeshShader.hlsl", "main", ShaderCompiler::ShaderType::Mesh);
    if (!m_particleMeshShader) {
        // Not fatal if particle shader optional
        std::cerr << "[Pipeline] Optional: Failed to compile ParticleMeshShader" << std::endl;
    }

    return true;
}

bool DX12RenderPipeline::CreateRootSignature() {
    std::cout << "[Pipeline] Creating root signature..." << std::endl;
    
    // Root parameters for geometry pass
    // - 1x CBV for per-frame constants (camera matrices, time)
    // - 1x CBV for per-object constants (world matrix)
    // Root signature desc
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    
    // Add SRV for Bone Buffer (t0, space1)
    // Parameter 3: SRV t0, space1 (Global Bone Matrices)
    // Use Root SRV for raw buffer binding (SetGraphicsRootShaderResourceView)
    D3D12_ROOT_PARAMETER1 rootParams[4] = {}; 
    
    // ... [Copy previous 0,1,2 params] -> No, I should redefine the array or just append the new one.
    // I need to replace the whole block or carefully insert.
    // For safety, I will rewrite the parameter setup to include the 4th.
    
    // 0: Per-frame (b0)
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].Descriptor.RegisterSpace = 0;
    rootParams[0].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    // 1: Per-object (b1)
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[1].Descriptor.ShaderRegister = 1;
    rootParams[1].Descriptor.RegisterSpace = 0;
    rootParams[1].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    // 2: Material (b2)
    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[2].Descriptor.ShaderRegister = 2;
    rootParams[2].Descriptor.RegisterSpace = 0;
    rootParams[2].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
    rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    
    // 3: Bone Buffer (t0, space1) - SRV
    rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rootParams[3].Descriptor.ShaderRegister = 0;
    rootParams[3].Descriptor.RegisterSpace = 1; 
    rootParams[3].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE; 
    rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX; // Only needed in VS
    
    rootSigDesc.Desc_1_1.NumParameters = 4;
    rootSigDesc.Desc_1_1.pParameters = rootParams;
    rootSigDesc.Desc_1_1.NumStaticSamplers = 0;
    rootSigDesc.Desc_1_1.pStaticSamplers = nullptr;
    rootSigDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    
    // Serialize root signature
    Microsoft::WRL::ComPtr<ID3DBlob> signatureBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
    
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &signatureBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "[Pipeline] Root signature serialization failed:\n";
            std::cerr << static_cast<const char*>(errorBlob->GetBufferPointer()) << std::endl;
        }
        return false;
    }
    
    // Create root signature
    hr = m_backend->GetDevice()->CreateRootSignature(
        0,
        signatureBlob->GetBufferPointer(),
        signatureBlob->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSignature)
    );
    
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create root signature" << std::endl;
        return false;
    }
    
    std::cout << "[Pipeline] Root signature created" << std::endl;
    return true;
}

bool DX12RenderPipeline::CreateGeometryPassPSO() {
    std::cout << "[Pipeline] Creating geometry pass PSO (forward to swapchain)..." << std::endl;
    
    // Input layout (matches Vertex struct with vertex color)
    D3D12_INPUT_ELEMENT_DESC inputElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 44, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},  // Vertex color + emissive
        {"BLENDINDICES", 0, DXGI_FORMAT_R32G32B32A32_SINT, 0, 60, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"BLENDWEIGHT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 76, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };
    
    // PSO description
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = {m_geometryVS->GetBufferPointer(), m_geometryVS->GetBufferSize()};
    psoDesc.PS = {m_geometryPS->GetBufferPointer(), m_geometryPS->GetBufferSize()};
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;  // NO CULLING
    psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    // ENABLE DEPTH TESTING NOW THAT TRANSFORMS ARE VERIFIED
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    psoDesc.InputLayout = {inputElements, _countof(inputElements)};
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    // Forward rendering: single swap chain target
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;  // Swap chain format
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleDesc.Count = 1;
    
    HRESULT hr = m_backend->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_geometryPassPSO));
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create geometry pass PSO" << std::endl;
        return false;
    }
    
    std::cout << "[Pipeline] Geometry pass PSO created" << std::endl;
    
    // Create wireframe PSO (same as geometry pass but with wireframe fill mode)
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    hr = m_backend->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_wireframePSO));
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create wireframe PSO (non-critical)" << std::endl;
        // Non-critical - continue without wireframe support
    } else {
        std::cout << "[Pipeline] Wireframe PSO created" << std::endl;
    }
    
    return true;
}

bool DX12RenderPipeline::CreateSkyboxPSO() {
    if (!m_skyboxVS || !m_skyboxPS) {
        std::cout << "[Pipeline] Skybox shaders not available, skipping PSO creation" << std::endl;
        m_skyboxEnabled = false;
        return false;
    }
    
    std::cout << "[Pipeline] Creating Skybox PSO..." << std::endl;
    
    // Skybox uses a fullscreen triangle - no input layout needed
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = {m_skyboxVS->GetBufferPointer(), m_skyboxVS->GetBufferSize()};
    psoDesc.PS = {m_skyboxPS->GetBufferPointer(), m_skyboxPS->GetBufferSize()};
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
    psoDesc.RasterizerState.DepthClipEnable = FALSE;  // Skybox is always at max depth
    // Disable depth writing but test against existing geometry
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;  // Don't write depth
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;  // Only draw where nothing else was drawn
    psoDesc.InputLayout = {nullptr, 0};  // No vertex input - fullscreen triangle is generated in VS
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleDesc.Count = 1;
    
    HRESULT hr = m_backend->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_skyboxPSO));
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create Skybox PSO: 0x" << std::hex << hr << std::dec << std::endl;
        m_skyboxEnabled = false;
        return false;
    }
    
    std::cout << "[Pipeline] Skybox PSO created successfully" << std::endl;
    return true;
}
bool DX12RenderPipeline::CreateGBufferPassPSO() {
    std::cout << "[Pipeline] Creating G-Buffer geometry pass PSO..." << std::endl;

    // Input layout (matches Vertex struct with vertex color)
    D3D12_INPUT_ELEMENT_DESC inputElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 44, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},  // Vertex color + emissive
        {"BLENDINDICES", 0, DXGI_FORMAT_R32G32B32A32_SINT, 0, 60, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"BLENDWEIGHT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 76, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = {m_geometryVS->GetBufferPointer(), m_geometryVS->GetBufferSize()};
    psoDesc.PS = {m_gbufferPS->GetBufferPointer(), m_gbufferPS->GetBufferSize()};
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.BlendState.RenderTarget[1].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.BlendState.RenderTarget[2].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.BlendState.RenderTarget[3].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;  // No culling for now
    psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    psoDesc.InputLayout = {inputElements, _countof(inputElements)};
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

    // G-Buffer MRT formats
    psoDesc.NumRenderTargets = 4;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;          // Albedo+roughness
    psoDesc.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;      // Normal+metallic
    psoDesc.RTVFormats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;      // Emissive+AO
    psoDesc.RTVFormats[3] = DXGI_FORMAT_R16G16_FLOAT;            // Velocity (RG)
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleDesc.Count = 1;

    HRESULT hr = m_backend->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_gbufferPassPSO));
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create G-Buffer geometry pass PSO" << std::endl;
        return false;
    }

    std::cout << "[Pipeline] G-Buffer geometry pass PSO created" << std::endl;
    return true;
}

bool DX12RenderPipeline::CreateSkinnedGeometryPassPSO() {
    std::cout << "[Pipeline] Creating Skinned Geometry Pass PSO..." << std::endl;
    
    if (!m_skinningVS) {
        std::cerr << "[Pipeline] Skinned VS missing" << std::endl;
        return false;
    }

    // Input layout (Matches SkinnedVertex)
    // Vertex (60 bytes) + Indices (16) + Weights (16) = 92 bytes
    D3D12_INPUT_ELEMENT_DESC inputElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        // Bone Data (Offsets likely 64 and 80 due to alignment padding)
        {"BLENDINDICES", 0, DXGI_FORMAT_R32G32B32A32_SINT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"BLENDWEIGHT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = {m_skinningVS->GetBufferPointer(), m_skinningVS->GetBufferSize()};
    psoDesc.PS = {m_gbufferPS->GetBufferPointer(), m_gbufferPS->GetBufferSize()};
    
    // Initialize Blend State (Opaque)
    psoDesc.BlendState.AlphaToCoverageEnable = FALSE;
    psoDesc.BlendState.IndependentBlendEnable = FALSE;
    for (int i = 0; i < 8; ++i) {
        psoDesc.BlendState.RenderTarget[i].BlendEnable = FALSE;
        psoDesc.BlendState.RenderTarget[i].LogicOpEnable = FALSE;
        psoDesc.BlendState.RenderTarget[i].SrcBlend = D3D12_BLEND_ONE;
        psoDesc.BlendState.RenderTarget[i].DestBlend = D3D12_BLEND_ZERO;
        psoDesc.BlendState.RenderTarget[i].BlendOp = D3D12_BLEND_OP_ADD;
        psoDesc.BlendState.RenderTarget[i].SrcBlendAlpha = D3D12_BLEND_ONE;
        psoDesc.BlendState.RenderTarget[i].DestBlendAlpha = D3D12_BLEND_ZERO;
        psoDesc.BlendState.RenderTarget[i].BlendOpAlpha = D3D12_BLEND_OP_ADD;
        psoDesc.BlendState.RenderTarget[i].LogicOp = D3D12_LOGIC_OP_NOOP;
        psoDesc.BlendState.RenderTarget[i].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE; // Can enable backface culling later
    psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    psoDesc.InputLayout = {inputElements, _countof(inputElements)};
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

    // G-Buffer MRT formats (Matches CreateGBufferPassPSO)
    psoDesc.NumRenderTargets = 4;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;          
    psoDesc.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;      
    psoDesc.RTVFormats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;      
    psoDesc.RTVFormats[3] = DXGI_FORMAT_R16G16_FLOAT;            
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleDesc.Count = 1;

    HRESULT hr = m_backend->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_skinnedGeometryPassPSO));
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create Skinned Geometry Pass PSO" << std::endl;
        
        // Retrieve and print detailed D3D12 validation errors
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
        if (SUCCEEDED(m_backend->GetDevice()->QueryInterface(IID_PPV_ARGS(&infoQueue)))) {
            UINT64 numMessages = infoQueue->GetNumStoredMessages();
            if (numMessages > 0) std::cerr << "[D3D12 Debug Info]:" << std::endl;
            
            for (UINT64 i = 0; i < numMessages; i++) {
                SIZE_T messageLength = 0;
                infoQueue->GetMessage(i, nullptr, &messageLength);
                
                if (messageLength > 0) {
                    std::vector<byte> messageBuffer(messageLength);
                    D3D12_MESSAGE* message = reinterpret_cast<D3D12_MESSAGE*>(messageBuffer.data());
                    
                    if (SUCCEEDED(infoQueue->GetMessage(i, message, &messageLength))) {
                         if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
                             std::cerr << "  [" << message->ID << "]: " << message->pDescription << std::endl;
                         }
                    }
                }
            }
            infoQueue->ClearStoredMessages();
        }
        return false;
    }

    std::cout << "[Pipeline] Skinned Geometry Pass PSO created" << std::endl;
    return true;
}

// === Mesh Shader Support (DX12 Ultimate) ===
bool DX12RenderPipeline::CheckMeshShaderSupport() {
    if (!m_backend || !m_backend->GetDevice()) {
        return false;
    }
    
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7 = {};
    HRESULT hr = m_backend->GetDevice()->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS7,
        &options7,
        sizeof(options7)
    );
    
    if (SUCCEEDED(hr) && options7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED) {
        std::cout << "[Pipeline] Mesh shaders SUPPORTED (Tier " 
                  << static_cast<int>(options7.MeshShaderTier) << ")" << std::endl;
        m_meshShadersSupported = true;
        return true;
    }
    
    std::cout << "[Pipeline] Mesh shaders NOT SUPPORTED on this device" << std::endl;
    m_meshShadersSupported = false;
    return false;
}

bool DX12RenderPipeline::CreateMeshShaderPSO() {

    if (!m_meshShadersSupported) {
        std::cout << "[Pipeline] Skipping mesh shader PSO - not supported" << std::endl;
        return false;
    }
    
    std::cout << "[Pipeline] Creating mesh shader PSO..." << std::endl;
    
    // Compile Amplification Shader (SM 6.5)
    std::wstring asPath = std::wstring(L"") + 
        std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(ASSET_DIR) +
        L"/shaders/dx12/AmplificationShader.hlsl";
    m_amplificationShader = ShaderCompiler::CompileFromFile(
        asPath, "main", ShaderCompiler::ShaderType::Amplification
    );
    if (!m_amplificationShader) {
        std::cerr << "[Pipeline] Failed to compile amplification shader" << std::endl;
        // Non-fatal: mesh shaders optional
        m_meshShadersEnabled = false;
        return false;
    }
    
    // Compile Mesh Shader (SM 6.5)
    std::wstring msPath = std::wstring(L"") + 
        std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(ASSET_DIR) +
        L"/shaders/dx12/MeshShader.hlsl";
    m_meshShader = ShaderCompiler::CompileFromFile(
        msPath, "main", ShaderCompiler::ShaderType::Mesh
    );
    if (!m_meshShader) {
        std::cerr << "[Pipeline] Failed to compile mesh shader" << std::endl;
        m_meshShadersEnabled = false;
        return false;
    }
    // Compile Pixel Shader (Standard PS 5.1)
    std::wstring psPath = std::wstring(L"") + 
        std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(ASSET_DIR) +
        L"/shaders/dx12/MeshShader_PS.hlsl";
    m_meshPixelShader = ShaderCompiler::CompileFromFile(
        psPath, "main", ShaderCompiler::ShaderType::Pixel_6_5
    );
    if (!m_meshPixelShader) {
        std::cerr << "[Pipeline] Failed to compile mesh shader PS" << std::endl;
        m_meshShadersEnabled = false;
        return false;
    }
    
    // Create mesh shader root signature (bindless pattern)
    // This uses descriptor tables for meshlet buffers
    // Create mesh shader root signature (bindless pattern)
    // This uses descriptor tables for meshlet buffers
    // 0: CBV b0 (Camera)
    // 1: Constants b1 (Instance)
    // 2: CBV b2 (Material)
    // 3-10: SRV t0-t7 (Meshlet data)
    D3D12_ROOT_PARAMETER1 rootParams[12] = {};
    
    // 0: Camera Constants (CBV b0)
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].Descriptor.RegisterSpace = 0;
    rootParams[0].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    // 1: Root Constants (b1) - instanceId etc.
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParams[1].Constants.ShaderRegister = 1;
    rootParams[1].Constants.RegisterSpace = 0;
    rootParams[1].Constants.Num32BitValues = 4;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    // 2: Material Constants (b2)
    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[2].Descriptor.ShaderRegister = 2;
    rootParams[2].Descriptor.RegisterSpace = 0;
    rootParams[2].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC;
    rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    
    // 3-10: SRVs (t0-t7) - Direct buffer bindings
    for(int i=0; i<8; i++) {
        rootParams[3+i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rootParams[3+i].Descriptor.ShaderRegister = i;
        rootParams[3+i].Descriptor.RegisterSpace = 0;
        rootParams[3+i].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC;
        rootParams[3+i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    // 11: Meshlet Bounds (t8) - Required for AS culling
    rootParams[11].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rootParams[11].Descriptor.ShaderRegister = 8; // Register t8
    rootParams[11].Descriptor.RegisterSpace = 0; // Space 0
    rootParams[11].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC;
    rootParams[11].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    // Static sampler
    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_ANISOTROPIC;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.MaxAnisotropy = 16;
    sampler.ShaderRegister = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rootSigDesc.Desc_1_1.NumParameters = 12; // Updated count
    rootSigDesc.Desc_1_1.pParameters = rootParams;
    rootSigDesc.Desc_1_1.NumStaticSamplers = 1;
    rootSigDesc.Desc_1_1.pStaticSamplers = &sampler;
    rootSigDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
    
    Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &signature, &error);
    if (FAILED(hr)) {
        if (error) std::cerr << "[Pipeline] Root sig error: " << (char*)error->GetBufferPointer() << std::endl;
        return false;
    }
    
    hr = m_backend->GetDevice()->CreateRootSignature(
        0,
        signature->GetBufferPointer(),
        signature->GetBufferSize(),
        IID_PPV_ARGS(&m_meshShaderRootSig)
    );
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create mesh shader root sig" << std::endl;
        return false;
    }
    
    // Create Mesh Shader PSO
    // Use aligned subobject wrapper to ensure correct stream layout (defined in StreamHelper.h)
    
    struct alignas(void*) MeshShaderStream {
        StreamSubobject<ID3D12RootSignature*, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE> RootSig;
        StreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS> AS;
        StreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS> MS;
        StreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS> PS;
        StreamSubobject<D3D12_RASTERIZER_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER> Rasterizer;
        StreamSubobject<D3D12_DEPTH_STENCIL_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL> DepthStencil;
        StreamSubobject<D3D12_BLEND_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND> Blend;
        StreamSubobject<D3D12_PRIMITIVE_TOPOLOGY_TYPE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY> Topology;
        StreamSubobject<D3D12_RT_FORMAT_ARRAY, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS> RTVFormats;
        StreamSubobject<DXGI_FORMAT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT> DSVFormat;
        StreamSubobject<DXGI_SAMPLE_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC> SampleDesc;
        StreamSubobject<UINT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK> SampleMask;
        StreamSubobject<D3D12_PIPELINE_STATE_FLAGS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS> Flags;
    } stream;
    
    // Zero-initialize the stream to avoid padding garbage
    memset(&stream, 0, sizeof(stream));
    
    // Re-set types since memset cleared them
    stream.RootSig.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
    stream.AS.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS;
    stream.MS.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
    stream.PS.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
    stream.Rasterizer.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER;
    stream.DepthStencil.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
    stream.Blend.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND;
    stream.Topology.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY;
    stream.RTVFormats.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
    stream.DSVFormat.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT;
    stream.SampleDesc.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC;
    stream.SampleMask.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK;
    stream.Flags.Type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS;

    stream.RootSig.Desc = m_meshShaderRootSig.Get();
    stream.AS.Desc = { m_amplificationShader->GetBufferPointer(), m_amplificationShader->GetBufferSize() };
    stream.MS.Desc = { m_meshShader->GetBufferPointer(), m_meshShader->GetBufferSize() };
    stream.PS.Desc = { m_meshPixelShader->GetBufferPointer(), m_meshPixelShader->GetBufferSize() };
    
    // Rasterizer State
    stream.Rasterizer.Desc = {};
    stream.Rasterizer.Desc.FillMode = D3D12_FILL_MODE_SOLID;
    stream.Rasterizer.Desc.CullMode = D3D12_CULL_MODE_BACK;
    stream.Rasterizer.Desc.FrontCounterClockwise = FALSE;
    stream.Rasterizer.Desc.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
    stream.Rasterizer.Desc.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    stream.Rasterizer.Desc.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
    stream.Rasterizer.Desc.DepthClipEnable = TRUE;
    stream.Rasterizer.Desc.MultisampleEnable = FALSE;
    stream.Rasterizer.Desc.AntialiasedLineEnable = FALSE;
    stream.Rasterizer.Desc.ForcedSampleCount = 0;
    stream.Rasterizer.Desc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    
    // Depth Stencil State
    stream.DepthStencil.Desc = {};
    stream.DepthStencil.Desc.DepthEnable = TRUE;
    stream.DepthStencil.Desc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    stream.DepthStencil.Desc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    stream.DepthStencil.Desc.StencilEnable = FALSE;
    
    // Blend State
    stream.Blend.Desc = {};
    stream.Blend.Desc.AlphaToCoverageEnable = FALSE;
    stream.Blend.Desc.IndependentBlendEnable = FALSE;
    stream.Blend.Desc.RenderTarget[0].BlendEnable = FALSE;
    stream.Blend.Desc.RenderTarget[0].LogicOpEnable = FALSE;
    stream.Blend.Desc.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
    stream.Blend.Desc.RenderTarget[0].DestBlend = D3D12_BLEND_ZERO;
    stream.Blend.Desc.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    stream.Blend.Desc.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    stream.Blend.Desc.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    stream.Blend.Desc.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    stream.Blend.Desc.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_NOOP;
    stream.Blend.Desc.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    stream.Topology.Desc = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    
    stream.RTVFormats.Desc.NumRenderTargets = 4;
    stream.RTVFormats.Desc.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;       // Albedo + Roughness
    stream.RTVFormats.Desc.RTFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;   // Normal + Metallic
    stream.RTVFormats.Desc.RTFormats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;   // Emissive + AO
    stream.RTVFormats.Desc.RTFormats[3] = DXGI_FORMAT_R16G16_FLOAT;         // Velocity
    
    stream.DSVFormat.Desc = DXGI_FORMAT_D32_FLOAT;
    
    stream.SampleDesc.Desc.Count = 1;
    stream.SampleDesc.Desc.Quality = 0;
    
    stream.SampleMask.Desc = UINT_MAX;
    
    stream.Flags.Desc = D3D12_PIPELINE_STATE_FLAG_NONE;

    D3D12_PIPELINE_STATE_STREAM_DESC psoDesc = {};
    psoDesc.SizeInBytes = sizeof(MeshShaderStream);
    psoDesc.pPipelineStateSubobjectStream = &stream;
    
    Microsoft::WRL::ComPtr<ID3D12Device2> device2;
    if (SUCCEEDED(m_backend->GetDevice()->QueryInterface(IID_PPV_ARGS(&device2)))) {
        hr = device2->CreatePipelineState(&psoDesc, IID_PPV_ARGS(&m_meshShaderPSO));
    } else {
        hr = E_NOINTERFACE;
    }
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create mesh shader PSO (HRESULT: 0x" << std::hex << hr << ")" << std::endl;
        
        // Retrieve and print detailed D3D12 validation errors
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
        if (SUCCEEDED(m_backend->GetDevice()->QueryInterface(IID_PPV_ARGS(&infoQueue)))) {
            UINT64 numMessages = infoQueue->GetNumStoredMessages();
            if (numMessages > 0) std::cerr << "[D3D12 Debug Info]:" << std::endl;
            
            for (UINT64 i = 0; i < numMessages; i++) {
                SIZE_T messageLength = 0;
                infoQueue->GetMessage(i, nullptr, &messageLength);
                
                if (messageLength > 0) {
                    std::vector<byte> messageBuffer(messageLength);
                    D3D12_MESSAGE* message = reinterpret_cast<D3D12_MESSAGE*>(messageBuffer.data());
                    
                    if (SUCCEEDED(infoQueue->GetMessage(i, message, &messageLength))) {
                         // Only print errors/cautions
                         if (message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || 
                             message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
                             message->Severity == D3D12_MESSAGE_SEVERITY_WARNING) {
                             std::cerr << "  [" << message->ID << "]: " << message->pDescription << std::endl;
                         }
                    }
                }
            }
            // Clear message queue to avoid re-printing old errors
            infoQueue->ClearStoredMessages();
        }
        
        m_meshShadersEnabled = false;
        return false;
    }
    
    std::cout << "[Pipeline] Mesh shader PSO created successfully" << std::endl;
    
    // Debug: Write success marker file
    std::ofstream marker("pso_success.txt");
    if (marker) {
        marker << "Mesh Shader PSO Created Successfully!" << std::endl;
        marker.close();
    }
    
    return true;
}

void DX12RenderPipeline::RenderMeshesWithMeshShader(ID3D12GraphicsCommandList* cmdList) {
    if (!m_meshShadersSupported || !m_meshShadersEnabled || !m_meshShaderPSO) return;
    
    // Mesh shaders require CommandList6
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList6> cmdList6;
    if (FAILED(cmdList->QueryInterface(IID_PPV_ARGS(&cmdList6)))) {
        std::cerr << "[Pipeline] Failed to query ID3D12GraphicsCommandList6" << std::endl;
        return;
    }
    
    cmdList6->SetGraphicsRootSignature(m_meshShaderRootSig.Get());
    cmdList6->SetPipelineState(m_meshShaderPSO.Get());
    
    // Bind Camera Buffer (b0)
    cmdList6->SetGraphicsRootConstantBufferView(0, m_perFrameCB->GetGPUVirtualAddress());

    // Bind Material Buffer (b2) (Using correct member m_materialCB)
    if (m_materialCB) {
        cmdList6->SetGraphicsRootConstantBufferView(2, m_materialCB->GetGPUVirtualAddress());
    }

    struct Constants {
        uint32_t instanceId;
        uint32_t meshletOffset;  // Added to match HLSL RootConstants layout
        uint32_t meshletCount;
        uint32_t _pad;
    } constants = { 0, 0, 0, 0 };

    for (D3D12Mesh* mesh : m_meshes) {
        if (!mesh) continue;
        const auto& buffers = mesh->GetMeshletBuffers();
        if (buffers.meshletCount == 0 || !buffers.meshlets) continue;

        constants.meshletCount = buffers.meshletCount;
        // meshletOffset is 0 for single-mesh per buffer
        
        // Bind meshlet buffers to root parameters 3-10
        if(buffers.positions) cmdList6->SetGraphicsRootShaderResourceView(3, buffers.positions->GetGPUVirtualAddress());
        if(buffers.normals) cmdList6->SetGraphicsRootShaderResourceView(4, buffers.normals->GetGPUVirtualAddress());
        if(buffers.uvs) cmdList6->SetGraphicsRootShaderResourceView(5, buffers.uvs->GetGPUVirtualAddress());
        if(buffers.colors) cmdList6->SetGraphicsRootShaderResourceView(6, buffers.colors->GetGPUVirtualAddress());
        
        if(buffers.vertexIndices) cmdList6->SetGraphicsRootShaderResourceView(7, buffers.vertexIndices->GetGPUVirtualAddress());
        if(buffers.primitives) cmdList6->SetGraphicsRootShaderResourceView(8, buffers.primitives->GetGPUVirtualAddress());
        if(buffers.meshlets) cmdList6->SetGraphicsRootShaderResourceView(9, buffers.meshlets->GetGPUVirtualAddress());
        
        // Ensure Instances buffer (t7) is ALWAYS bound to prevent TDR
        if (buffers.instances) {
             cmdList6->SetGraphicsRootShaderResourceView(10, buffers.instances->GetGPUVirtualAddress());
        } else {
             // Fallback/Safety...
        }

        // Bind Meshlet Bounds to restart parameter 11 (t8)
        if (buffers.bounds) {
             cmdList6->SetGraphicsRootShaderResourceView(11, buffers.bounds->GetGPUVirtualAddress());
        }

        // Bind 4 constants (instanceId, meshletOffset, meshletCount, padding)
        cmdList6->SetGraphicsRoot32BitConstants(1, 4, &constants, 0);

        UINT groupCount = (buffers.meshletCount + 31) / 32;
        cmdList6->DispatchMesh(groupCount, 1, 1);
    }
}

bool DX12RenderPipeline::CreateConstantBuffers() {
    std::cerr << "[Pipeline] Creating constant buffers..." << std::endl;
    
    // Create upload heaps for constant buffers (CPU writable, GPU readable)
    // D3D12 requires constant buffers to be 256-byte aligned and offsets must be
    // multiples of 256 bytes.

    const size_t perFrameSize      = sizeof(PerFrameConstants);   // Single instance
    const size_t perObjectSlotSize = 256; // One 256-byte slot per mesh
    const size_t materialSlotSize  = 256; // One 256-byte slot per mesh

    // Per-frame constants (updated once per frame)
    if (!m_backend->CreateBuffer(perFrameSize, true, m_perFrameCB)) {
        std::cerr << "[Pipeline] Failed to create per-frame constant buffer" << std::endl;
        return false;
    }
    m_perFrameCB->SetName(L"PerFrameConstantBuffer");
    
    // Map the buffer for persistent CPU writes
    D3D12_RANGE readRange = {0, 0};  // We won't read from this buffer on CPU
    HRESULT hr = m_perFrameCB->Map(0, &readRange, reinterpret_cast<void**>(&m_perFrameData));
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to map per-frame constant buffer" << std::endl;
        return false;
    }
    std::cerr << "[Pipeline] PerFrame CB created" << std::endl;
    
    // Per-object constants (updated per mesh) - allocate a slot per mesh we can render
    const size_t perObjectBufferSize = perObjectSlotSize * MAX_MESHES_PER_FRAME;
    if (!m_backend->CreateBuffer(perObjectBufferSize, true, m_perObjectCB)) {
        std::cerr << "[Pipeline] Failed to create per-object constant buffer" << std::endl;
        return false;
    }
    m_perObjectCB->SetName(L"PerObjectConstantBuffer");
    
    hr = m_perObjectCB->Map(0, &readRange, reinterpret_cast<void**>(&m_perObjectData));
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to map per-object constant buffer" << std::endl;
        return false;
    }
    std::cerr << "[Pipeline] PerObject CB created" << std::endl;
    
    // Material constants
    std::cerr << "[Pipeline] Creating Material CB..." << std::endl;
    const size_t materialBufferSize = materialSlotSize * MAX_MESHES_PER_FRAME;
    if (!m_backend->CreateBuffer(materialBufferSize, true, m_materialCB)) {
        std::cerr << "[Pipeline] Failed to create material constant buffer" << std::endl;
        return false;
    }
    m_materialCB->SetName(L"MaterialConstantBuffer");
    
    hr = m_materialCB->Map(0, &readRange, reinterpret_cast<void**>(&m_materialData));
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to map material constant buffer" << std::endl;
        return false;
    }
    std::cerr << "[Pipeline] Material CB created" << std::endl;
    
    std::cerr << "[Pipeline] Creating Culling/Counter Buffers..." << std::endl;

    // Phase 3: Object Culling Data Buffer (256-byte aligned max size)
    const size_t cullingSize = MAX_MESHES_PER_FRAME * sizeof(ObjectCullingData);
    
    // Upload Heap for frequent CPU updates
    D3D12_HEAP_PROPERTIES uploadHeap = {};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    uploadHeap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    uploadHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    uploadHeap.CreationNodeMask = 1;
    uploadHeap.VisibleNodeMask = 1;
    
    D3D12_RESOURCE_DESC cullingDesc = {};
    cullingDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    cullingDesc.Width = cullingSize;
    cullingDesc.Height = 1;
    cullingDesc.DepthOrArraySize = 1;
    cullingDesc.MipLevels = 1;
    cullingDesc.Format = DXGI_FORMAT_UNKNOWN;
    cullingDesc.SampleDesc.Count = 1;
    cullingDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    cullingDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr2 = m_backend->GetDevice()->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &cullingDesc, 
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_objectCullingUploadBuffer));
    m_objectCullingUploadBuffer->SetName(L"ObjectCullingUploadBuffer");

    D3D12_HEAP_PROPERTIES defaultHeapCull = {};
    defaultHeapCull.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC cullingDescDefault = cullingDesc;
    cullingDescDefault.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    HRESULT hr2b = m_backend->GetDevice()->CreateCommittedResource(
         &defaultHeapCull, D3D12_HEAP_FLAG_SHARED, &cullingDescDefault,
         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_objectCullingDataBuffer));
        
    if (FAILED(hr2)) {
         std::cout << "[Pipeline] Failed to create Culling Buffer" << std::endl;  
         return false;
    }
        
    // Phase 3: Draw Counter Buffer (Default Heap + UAV for atomic)
    D3D12_HEAP_PROPERTIES defaultHeap = {};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    
    D3D12_RESOURCE_DESC countDesc = cullingDesc; 
    countDesc.Width = 256; 
    countDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; 
    
    HRESULT hr3 = m_backend->GetDevice()->CreateCommittedResource(
        &defaultHeap, D3D12_HEAP_FLAG_SHARED, &countDesc, 
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_drawCounterBuffer));
        
    if (FAILED(hr3)) {
         std::cout << "[Pipeline] Failed to create Counter Buffer" << std::endl;  
         return false;
    }
    
    // Create Readback Buffer for Counter
    D3D12_HEAP_PROPERTIES readbackHeap = {};
    readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;
    readbackHeap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    readbackHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    readbackHeap.CreationNodeMask = 1;
    readbackHeap.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC readbackDesc = countDesc;
    readbackDesc.Flags = D3D12_RESOURCE_FLAG_NONE; // Readback buffers cannot have UAV flag

    HRESULT hr3b = m_backend->GetDevice()->CreateCommittedResource(
        &readbackHeap, D3D12_HEAP_FLAG_NONE, &readbackDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readbackDrawCounter));
        
    if (FAILED(hr3b)) {
        std::cerr << "[Pipeline] Failed to create Readback Draw Counter! HR=" << hr3b << std::endl;
        return false;
    }
    m_readbackDrawCounter->SetName(L"ReadbackDrawCounter");

    std::cerr << "[Pipeline] Culling/Counter Buffers created. Registering with CUDA..." << std::endl;

    // Register with CudaCore (Modern)
    if (m_cudaCore) {
        if (m_extMemObjectCulling) m_cudaCore->FreeImportedResource(m_extMemObjectCulling);
        if (m_extMemDrawCounter) m_cudaCore->FreeImportedResource(m_extMemDrawCounter);
        
        std::cerr << "[Pipeline] Importing Culling Buffer (size=" << cullingSize << ")..." << std::endl;
        m_extMemObjectCulling = m_cudaCore->ImportD3D12Resource(m_objectCullingDataBuffer.Get(), cullingSize);
        m_objectCullingBufferSize = cullingSize;  // Store for mapping
        
        std::cerr << "[Pipeline] Importing Draw Counter (size=" << countDesc.Width << ")..." << std::endl;
        m_extMemDrawCounter = m_cudaCore->ImportD3D12Resource(m_drawCounterBuffer.Get(), countDesc.Width);
        m_drawCounterSize = countDesc.Width;  // Store for mapping
    }
    std::cerr << "[Pipeline] CUDA Registration Complete." << std::endl;

    
    // Phase 4: Animation Bone Matrices Buffer
    std::cerr << "[Pipeline] Creating Global Bone Buffer..." << std::endl;
    const size_t boneBufferSize = MAX_BONES_GLOBAL * sizeof(glm::mat4); // 640KB
    
    // Upload Heap for dynamic updates per frame
    D3D12_HEAP_PROPERTIES boneHeap = {};
    boneHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    boneHeap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    boneHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    boneHeap.CreationNodeMask = 1;
    boneHeap.VisibleNodeMask = 1;
    
    D3D12_RESOURCE_DESC boneDesc = {};
    boneDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    boneDesc.Width = boneBufferSize;
    boneDesc.Height = 1;
    boneDesc.DepthOrArraySize = 1;
    boneDesc.MipLevels = 1;
    boneDesc.Format = DXGI_FORMAT_UNKNOWN;
    boneDesc.SampleDesc.Count = 1;
    boneDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    boneDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr4 = m_backend->GetDevice()->CreateCommittedResource(
        &boneHeap, D3D12_HEAP_FLAG_NONE, &boneDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_globalBoneBuffer));
        
    if (FAILED(hr4)) {
         std::cerr << "[Pipeline] Failed to create Global Bone Buffer" << std::endl;  
         return false;
    }
    m_globalBoneBuffer->SetName(L"GlobalBoneBuffer");
    std::cerr << "[Pipeline] Global Bone Buffer created (" << (boneBufferSize/1024) << " KB)" << std::endl;

    return true;
}

void DX12RenderPipeline::UploadObjectCullingData() {
    if (!m_objectCullingDataBuffer || m_meshes.empty()) return;
    
    ObjectCullingData* data = nullptr;
    D3D12_RANGE readRange = {0, 0};
    if (SUCCEEDED(m_objectCullingUploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&data)))) {
        const uint32_t perObjectSlotSize = 256;
        const uint32_t materialSlotSize = 256;
        
        for (size_t i = 0; i < m_meshes.size(); ++i) {
            D3D12Mesh* mesh = m_meshes[i];
            
            // Sphere (Use bounding box center/radius approximation)
            glm::vec3 center(0.0f);
            float radius = 1.0f;
            if (mesh) { 
                // mesh->GetBounds(center, radius); // Assuming this exists or using default
            }
            
            data[i].sphereCenter = center; 
            data[i].sphereRadius = radius;
            
            // Views
            data[i].vbv_loc = mesh->GetVertexBufferView().BufferLocation;
            data[i].vbv_size = mesh->GetVertexBufferView().SizeInBytes;
            data[i].vbv_stride = mesh->GetVertexBufferView().StrideInBytes;
            
            data[i].ibv_loc = mesh->GetIndexBufferView().BufferLocation;
            data[i].ibv_size = mesh->GetIndexBufferView().SizeInBytes;
            data[i].ibv_format = mesh->GetIndexBufferView().Format;
            
            // Constants
            data[i].cbv = m_perObjectCB->GetGPUVirtualAddress() + i * perObjectSlotSize;
            data[i].materialCbv = m_materialCB->GetGPUVirtualAddress() + i * materialSlotSize;
            
            // Matrix
            data[i].worldMatrix = mesh->transform;
            
            // Draw
            data[i].indexCount = mesh->GetIndexCount();
            
            // FIX: Update Rendering Constant Buffers (m_perObjectCB / m_materialCB)
            // The GPU-path skips the CPU GeometryPass loop, so we must update them here!
            uint32_t meshSlot = (uint32_t)i;
            if (meshSlot < MAX_MESHES_PER_FRAME) {
                // Update Per-Object CB
                if (m_perObjectData) {
                    PerObjectConstants* objCB = reinterpret_cast<PerObjectConstants*>(
                        reinterpret_cast<uint8_t*>(m_perObjectData) + meshSlot * perObjectSlotSize);

                    objCB->worldMatrix = mesh->transform;
                    objCB->prevWorldMatrix = mesh->transform; // No motion vectors for now
                    objCB->normalMatrix = glm::transpose(glm::inverse(mesh->transform));
                    
                    bool isSkinned = (mesh->GetSkeleton() != nullptr);
                    objCB->isSkinned = isSkinned ? 1 : 0;
                    objCB->boneOffset = isSkinned ? mesh->globalBoneOffset : 0;
                }

                // Update Material CB
                if (m_materialData) {
                    MaterialConstants* matCB = reinterpret_cast<MaterialConstants*>(
                        reinterpret_cast<uint8_t*>(m_materialData) + meshSlot * materialSlotSize);
                    
                    Material mat = mesh->GetMaterial();
                    mat.albedoColor.a = 1.0f; // Force opaque

                    matCB->albedoColor       = mat.albedoColor;
                    matCB->roughness         = mat.roughness;
                    matCB->metallic          = mat.metallic;
                    matCB->ambientOcclusion  = mat.ambientOcclusion;
                    matCB->emissiveStrength  = mat.emissiveStrength;
                    matCB->emissiveColor     = mat.emissiveColor;
                }
            }
        }
        m_objectCullingUploadBuffer->Unmap(0, nullptr);

        // Copy from Upload to Default
        TransitionResource(m_objectCullingDataBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        m_backend->GetCommandList()->CopyResource(m_objectCullingDataBuffer.Get(), m_objectCullingUploadBuffer.Get());
        TransitionResource(m_objectCullingDataBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }
}


// =================================================================================================
// KERNEL ARCHITECTURE IMPLEMENTATION (Invention 1)
// =================================================================================================

extern "C" void LaunchCullAndDrawKernel(
    const void* objects,
    void* commands,
    unsigned int* drawCounter,
    int objectCount,
    const float* frustumPlanes,
    cudaStream_t stream
);

void DX12RenderPipeline::ExecuteRenderKernel(void* pCmdList) {
    ID3D12GraphicsCommandList* cmdList = static_cast<ID3D12GraphicsCommandList*>(pCmdList);
    std::cerr << "[ExecuteRenderKernel] Entered. Path: " << (int)m_activePath << std::endl;
    switch (m_activePath) {
        case RenderPath::VertexShader_Fallback:
            std::cerr << "[ExecuteRenderKernel] Calling ExecuteVertexShaderPacket..." << std::endl;
            ExecuteVertexShaderPacket(cmdList);
            std::cerr << "[ExecuteRenderKernel] ExecuteVertexShaderPacket done." << std::endl;
            break;
        case RenderPath::Indirect_GPU_Driven:
            // Phase 3: GPU Generation
            std::cerr << "[ExecuteRenderKernel] GPU-Driven path. Uploading culling data..." << std::endl;
            UploadObjectCullingData();
            std::cerr << "[ExecuteRenderKernel] UploadObjectCullingData done." << std::endl;
            // GPU Path: Only ensure buffer exists - DO NOT copy CPU data (it would overwrite CUDA kernel output)
            EnsureIndirectBufferExists();
            // UploadIndirectCommands(); // DISABLED: Isolation Test
            std::cerr << "[ExecuteRenderKernel] Buffer checked." << std::endl;
            
            std::cerr << "[ExecuteRenderKernel] Checking CUDA resources: cudaCore=" << (m_cudaCore ? "valid" : "NULL")
                      << " extMemObjectCulling=" << (m_extMemObjectCulling ? "valid" : "NULL")
                      << " extMemIndirectBuffer=" << (m_extMemIndirectBuffer ? "valid" : "NULL")
                      << " extMemDrawCounter=" << (m_extMemDrawCounter ? "valid" : "NULL") << std::endl;
            
            if (m_cudaCore && m_extMemObjectCulling && m_extMemIndirectBuffer && m_extMemDrawCounter) { 
                
                // 1. SIGNAL from DX12: "Buffers are ready for Compute"
                // We use the Fence to tell CUDA it's safe to read/write these buffers.
                std::cerr << "[ExecuteRenderKernel] Signaling DX12 fence..." << std::endl;
                m_cullingFenceValue++;
                
                // CRITICAL FIX: Close and execute command list to actually signal the fence
                // Otherwise CUDA waits forever for a signal that hasn't been issued
                std::cerr << "[ExecuteRenderKernel] Flushing command list to execute signal..." << std::endl;
                HRESULT hr = cmdList->Close();
                if (SUCCEEDED(hr)) {
                    ID3D12CommandList* ppCommandLists[] = { cmdList };
                    m_backend->GetCommandQueue()->ExecuteCommandLists(1, ppCommandLists);
                    m_backend->GetCommandQueue()->Signal(m_cullingFence.Get(), m_cullingFenceValue);
                    
                    // Wait for GPU to complete before CUDA can use resources
                    std::cerr << "[ExecuteRenderKernel] Waiting for DX12 GPU to complete..." << std::endl;
                    m_backend->WaitForGPU();
                    
                    // Reset command list for subsequent rendering
                    std::cerr << "[ExecuteRenderKernel] Resetting command list..." << std::endl;
                    m_backend->ResetCommandList();
                } else {
                    std::cerr << "[ExecuteRenderKernel] ERROR: Failed to close command list!" << std::endl;
                    return;
                }
                
                // 2. WAIT in CUDA for DX12 (now fence is actually signaled)
                // NOTE: Since WaitForGPU() above fully synchronizes CPU and GPU,
                // the CUDA external semaphore wait is redundant and its import may be broken.
                // Skipping this call since the DX12 resources are fully ready.
                std::cerr << "[ExecuteRenderKernel] Skipping CUDA semaphore wait (WaitForGPU already synced)..." << std::endl;
                // m_cudaCore->WaitSemaphore(m_cudaCullingFenceSem, m_cullingFenceValue); // DISABLED
                
                // 3. Map (Fastest layout for Import)
                std::cerr << "[ExecuteRenderKernel] Mapping imported resources..." << std::endl;
                // Use stored sizes from import time
                void* d_objects = m_cudaCore->MapImportedResource(m_extMemObjectCulling, 0, m_objectCullingBufferSize);
                void* d_commands = m_cudaCore->MapImportedResource(m_extMemIndirectBuffer, 0, m_indirectBufferSize);
                void* d_counter = m_cudaCore->MapImportedResource(m_extMemDrawCounter, 0, m_drawCounterSize);
                std::cerr << "[ExecuteRenderKernel] Mapped: objects=" << d_objects 
                          << " commands=" << d_commands << " counter=" << d_counter << std::endl;
                
                if (d_objects && d_commands && d_counter) {
                    /* ISOLATION TEST: Disable CUDA kernel
                    std::cerr << "[ExecuteRenderKernel] Clearing buffers..." << std::endl;
                    cudaMemset(d_counter, 0, sizeof(uint32_t));
                    // Clear command buffer so unused slots have indexCount=0 (DrawIndexed draws nothing)
                    cudaMemset(d_commands, 0, m_indirectBufferSize);
                    
                    // ... Frustum ...

                    std::cerr << "[ExecuteRenderKernel] Launching CullAndDraw kernel for " << m_meshes.size() << " meshes..." << std::endl;
                    LaunchCullAndDrawKernel(d_objects, d_commands, (unsigned int*)d_counter, (int)m_meshes.size(), planes, 0);
                    std::cerr << "[ExecuteRenderKernel] CullAndDraw kernel launched." << std::endl;
                    */
                    std::cerr << "[ExecuteRenderKernel] ISOLATION TEST: SKIPPED (Re-enabling below)." << std::endl;
                    
                    // RE-ENABLE CUDA KERNEL
                    cudaMemset(d_counter, 0, sizeof(uint32_t));
                    cudaMemset(d_commands, 0, m_indirectBufferSize);
                    
                    // Frustum Extraction
                    glm::mat4 viewProj = m_camera->GetProjectionMatrix() * m_camera->GetViewMatrix();
                    float planes[24]; 
                    const float* m = glm::value_ptr(viewProj);
                    // Left
                    planes[0] = m[3] + m[0]; planes[1] = m[7] + m[4]; planes[2] = m[11] + m[8]; planes[3] = m[15] + m[12];
                    // Right
                    planes[4] = m[3] - m[0]; planes[5] = m[7] - m[4]; planes[6] = m[11] - m[8]; planes[7] = m[15] - m[12];
                    // Bottom
                    planes[8] = m[3] + m[1]; planes[9] = m[7] + m[5]; planes[10] = m[11] + m[9]; planes[11] = m[15] + m[13];
                    // Top
                    planes[12] = m[3] - m[1]; planes[13] = m[7] - m[5]; planes[14] = m[11] - m[9]; planes[15] = m[15] - m[13];
                    // Near
                    planes[16] = m[3] + m[2]; planes[17] = m[7] + m[6]; planes[18] = m[11] + m[10]; planes[19] = m[15] + m[14];
                    // Far
                    planes[20] = m[3] - m[2]; planes[21] = m[7] - m[6]; planes[22] = m[11] - m[10]; planes[23] = m[15] - m[14];

                    for (int i = 0; i < 6; ++i) {
                        float len = sqrt(planes[i*4]*planes[i*4] + planes[i*4+1]*planes[i*4+1] + planes[i*4+2]*planes[i*4+2]);
                        planes[i*4] /= len; planes[i*4+1] /= len; planes[i*4+2] /= len; planes[i*4+3] /= len;
                    }

                    LaunchCullAndDrawKernel(d_objects, d_commands, (unsigned int*)d_counter, (int)m_meshes.size(), planes, 0);
                    std::cerr << "[ExecuteRenderKernel] CullAndDraw kernel launched." << std::endl;
                }
                
                // 4. Unmap (Release virtual address range)
                if (d_objects) m_cudaCore->UnmapImportedResource(d_objects);
                if (d_commands) m_cudaCore->UnmapImportedResource(d_commands);
                if (d_counter) m_cudaCore->UnmapImportedResource(d_counter);
                std::cerr << "[ExecuteRenderKernel] Unmapped resources." << std::endl;
                
                // 5. SIGNAL from CUDA: "Compute Done"
                // DISABLED: CUDA external semaphore import is broken, WaitForGPU provides sync
                // m_cullingFenceValue++;
                // m_cudaCore->SignalSemaphore(m_cudaCullingFenceSem, m_cullingFenceValue);
                
                // 6. WAIT in DX12 for Compute
                // DISABLED: Using cudaDeviceSynchronize instead for now
                // m_backend->GetCommandQueue()->Wait(m_cullingFence.Get(), m_cullingFenceValue);
                
                // Sync CUDA to ensure kernel is complete before DX12 continues
                std::cerr << "[ExecuteRenderKernel] Syncing CUDA..." << std::endl;
                cudaDeviceSynchronize();
                std::cerr << "[ExecuteRenderKernel] CUDA synced." << std::endl;
                
                // 7. Copy Draw Counter to Readback Buffer (Debug)
                // Need to transition Default buffer to Copy Source
                {
                    D3D12_RESOURCE_BARRIER barriers[1];
                    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                    barriers[0].Transition.pResource = m_drawCounterBuffer.Get();
                    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON; // Implicit decay from UAV? Check.
                    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    cmdList->ResourceBarrier(1, barriers);

                    cmdList->CopyResource(m_readbackDrawCounter.Get(), m_drawCounterBuffer.Get());

                    // Transition back to Common logic handled by next frame decay or explicit
                    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                    cmdList->ResourceBarrier(1, barriers);
                }
            }
            
            ExecuteIndirectPacket(cmdList);
            
            // Readback on CPU (Note: This is delayed by frame latency provided Execute is fully flushed)
            // But ReadbackDrawCounter() handles map/unmap. Calls naturally after Execute.
            // Actually, we need to wait for GPU to finish this copy before reading.
            // Since we wait for GPU at end of frame (in simple loop) or via Fence, calling it at start of next frame is best.
            // For verification, we'll force flush in ReadbackDrawCounter.
            ReadbackDrawCounter();
            break;
    }
}

void DX12RenderPipeline::ReadbackDrawCounter() {
    if (!m_readbackDrawCounter) return;

    // Direct Map - Expects data from previous frame or initial state.
    // No explicit wait here to avoid stalling command recording.
    
    D3D12_RANGE readRange = {0, 256};
    void* pData = nullptr;
    if (SUCCEEDED(m_readbackDrawCounter->Map(0, &readRange, &pData))) {
        uint32_t count = *reinterpret_cast<uint32_t*>(pData);
        
        static int logCounter = 0;
        // Log first 60 frames then every 60
        if (logCounter < 60 || logCounter % 60 == 0) {
             std::cerr << "[GPU Culling] Visible Objects: " << count << " / " << m_meshes.size() << std::endl;
        }
        logCounter++;
        m_readbackDrawCounter->Unmap(0, nullptr);
    }
}

void DX12RenderPipeline::ExecuteVertexShaderPacket(ID3D12GraphicsCommandList* cmdList) {
    // START SAFE KERNEL
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    
    if (m_debugMode == DebugMode::WIREFRAME && m_wireframePSO) {
        cmdList->SetPipelineState(m_wireframePSO.Get());
    } else {
        cmdList->SetPipelineState(m_geometryPassPSO.Get());
    }
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    uint32_t drawCallCount = 0;
    uint32_t triangleCount = 0;

    for (size_t i = 0; i < m_meshes.size(); ++i) {
        D3D12Mesh* mesh = m_meshes[i];
        if (!mesh) continue;

        // Per-object constants
        if (m_perObjectData) {
            m_perObjectData[i].worldMatrix = mesh->transform;
            m_perObjectData[i].prevWorldMatrix = mesh->transform;
            
            // Explicitly calculate inverse first to help compiler
            glm::mat4 invWorld = glm::inverse(mesh->transform);
            m_perObjectData[i].normalMatrix = glm::transpose(invWorld);
            
            // Calculate GPU address
            D3D12_GPU_VIRTUAL_ADDRESS objCBAddress = m_perObjectCB->GetGPUVirtualAddress() + (i * 256); // 256 byte alignment
            cmdList->SetGraphicsRootConstantBufferView(1, objCBAddress);
        }

        // Material constants
        if (m_materialData) {
            // TODO: Real material system lookup
            m_materialData[i].albedoColor = glm::vec4(1.0f); 
            m_materialData[i].roughness = 0.5f;
            m_materialData[i].metallic = 0.0f;
            
            D3D12_GPU_VIRTUAL_ADDRESS matCBAddress = m_materialCB->GetGPUVirtualAddress() + (i * 256);
            cmdList->SetGraphicsRootConstantBufferView(2, matCBAddress);
        }

        const D3D12_VERTEX_BUFFER_VIEW& vbv = mesh->GetVertexBufferView();
        const D3D12_INDEX_BUFFER_VIEW& ibv = mesh->GetIndexBufferView();
        
        cmdList->IASetVertexBuffers(0, 1, &vbv);
        cmdList->IASetIndexBuffer(&ibv);
        
        cmdList->DrawIndexedInstanced(mesh->GetIndexCount(), 1, 0, 0, 0);
        
        drawCallCount++;
        triangleCount += mesh->GetIndexCount() / 3;
    }
    
    // Update stats only if taking this path
    m_stats.drawCalls = drawCallCount;
    m_stats.triangles = triangleCount;
}




bool DX12RenderPipeline::CreateCommandSignature() {
    // AAA SAFETY: Ensure C++ struct matches GPU stride
    static_assert(sizeof(IndirectCommand) == 72, "IndirectCommand size mismatch! Must be 72 bytes for API agreement.");
    std::cout << "[Pipeline] IndirectCommand Size Verified: " << sizeof(IndirectCommand) << " bytes" << std::endl;

    // Define arguments: CBV, CBV, VBV, IBV, DRAW
    D3D12_INDIRECT_ARGUMENT_DESC args[5] = {};
    
    // Arg 0: Per-Object CBV (Root Parameter 1)
    args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
    args[0].ConstantBufferView.RootParameterIndex = 1;
    
    // Arg 1: Material CBV (Root Parameter 2)
    args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
    args[1].ConstantBufferView.RootParameterIndex = 2;
    
    // Arg 2: Vertex Buffer View
    args[2].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
    args[2].VertexBuffer.Slot = 0; // Bind to slot 0

    // Arg 3: Index Buffer View
    args[3].Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;

    // Arg 4: Draw Indexed Arguments
    args[4].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    
    D3D12_COMMAND_SIGNATURE_DESC desc = {};
    desc.ByteStride = sizeof(IndirectCommand);
    desc.NumArgumentDescs = _countof(args);
    desc.pArgumentDescs = args;
    desc.NodeMask = 0;
    
    // Root signature must be specified if we change root arguments
    HRESULT hr = m_backend->GetDevice()->CreateCommandSignature(
        &desc,
        m_rootSignature.Get(),
        IID_PPV_ARGS(&m_commandSignature)
    );
    
    if (FAILED(hr)) {
        std::cerr << "[Pipeline] Failed to create Indirect Command Signature" << std::endl;
        return false;
    }
    
    std::cout << "[Pipeline] Indirect Command Signature created successfully (Stride: " << desc.ByteStride << ")" << std::endl;

    // === Phase 4: Create Synchronization Fence ===
    HRESULT hrFence = m_backend->GetDevice()->CreateFence(
        0, 
        D3D12_FENCE_FLAG_SHARED, // Important for Interop!
        IID_PPV_ARGS(&m_cullingFence)
    );

    if (FAILED(hrFence)) {
        std::cerr << "[Pipeline] Failed to create shared synchronization fence" << std::endl;
        return false;
    }

    if (m_cudaCore) {
        m_cudaCullingFenceSem = m_cudaCore->ImportD3D12Fence(m_cullingFence.Get());
        std::cout << "[Pipeline] Shared Fence Imported to CUDA" << std::endl;
    }

    return true;
}

void DX12RenderPipeline::EnsureIndirectBufferExists() {
    // GPU Path: Only allocate the buffer - DO NOT fill/copy data (CUDA kernel will fill it)
    size_t meshCount = m_meshes.size();
    if (meshCount == 0) return;
    
    size_t bufferSize = meshCount * sizeof(IndirectCommand);
    
    if (!m_indirectCommandBuffer || m_indirectCommandMaxCount < meshCount) {
        // Release old
        m_indirectCommandBuffer.Reset();
        
        // Create new (Growth strategy: 1.5x)
        m_indirectCommandMaxCount = std::max((uint32_t)meshCount, (uint32_t)(m_indirectCommandMaxCount * 1.5));
        if (m_indirectCommandMaxCount < 128) m_indirectCommandMaxCount = 128;
        
        bufferSize = m_indirectCommandMaxCount * sizeof(IndirectCommand);
        
        D3D12_HEAP_PROPERTIES defaultHeapInd = {};
        defaultHeapInd.Type = D3D12_HEAP_TYPE_DEFAULT;
        
        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Alignment = 0;
        resDesc.Width = bufferSize;
        resDesc.Height = 1;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels = 1;
        resDesc.Format = DXGI_FORMAT_UNKNOWN;
        resDesc.SampleDesc.Count = 1;
        resDesc.SampleDesc.Quality = 0;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr = m_backend->GetDevice()->CreateCommittedResource(
            &defaultHeapInd,
            D3D12_HEAP_FLAG_SHARED,
            &resDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_indirectCommandBuffer)
        );
        m_indirectCommandBuffer->SetName(L"IndirectCommandBuffer_GPU");
        
        if (FAILED(hr)) {
            std::cerr << "[Pipeline] Failed to allocate GPU Indirect Command Buffer" << std::endl;
            return;
        }

        // Import to CUDA
        if (m_cudaCore) {
            if (m_extMemIndirectBuffer) m_cudaCore->FreeImportedResource(m_extMemIndirectBuffer);
            std::cerr << "[Pipeline] Importing GPU Indirect Buffer (size=" << bufferSize << ")..." << std::endl;
            m_extMemIndirectBuffer = m_cudaCore->ImportD3D12Resource(m_indirectCommandBuffer.Get(), bufferSize);
            m_indirectBufferSize = bufferSize;
        }
    }
}

void DX12RenderPipeline::UploadIndirectCommands() {
    // Phase 2: CPU Generation of Indirect Buffer
    size_t meshCount = m_meshes.size();
    if (meshCount == 0) return;
    
    // 1. Resize/Create Buffer if needed
    // Using Upload Heap for simple CPU->GPU streaming in Phase 2
    static const size_t ALIGNMENT = 256; 
    size_t bufferSize = meshCount * sizeof(IndirectCommand);
    
    if (!m_indirectCommandBuffer || m_indirectCommandMaxCount < meshCount) {
        // Release old
        m_indirectCommandBuffer.Reset();
        
        // Create new (Growth strategy: 1.5x)
        m_indirectCommandMaxCount = std::max((uint32_t)meshCount, (uint32_t)(m_indirectCommandMaxCount * 1.5));
        // Ensure minimum size
        if (m_indirectCommandMaxCount < 128) m_indirectCommandMaxCount = 128;
        
        bufferSize = m_indirectCommandMaxCount * sizeof(IndirectCommand);
        
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Alignment = 0;
        resDesc.Width = bufferSize;
        resDesc.Height = 1;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels = 1;
        resDesc.Format = DXGI_FORMAT_UNKNOWN;
        resDesc.SampleDesc.Count = 1;
        resDesc.SampleDesc.Quality = 0;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        HRESULT hr = m_backend->GetDevice()->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &resDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_indirectCommandUploadBuffer)
        );
        m_indirectCommandUploadBuffer->SetName(L"IndirectCommandUploadBuffer");


        D3D12_HEAP_PROPERTIES defaultHeapInd = {};
        defaultHeapInd.Type = D3D12_HEAP_TYPE_DEFAULT;
        
        D3D12_RESOURCE_DESC resDescDefault = resDesc;
        resDescDefault.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr2 = m_backend->GetDevice()->CreateCommittedResource(
            &defaultHeapInd,
            D3D12_HEAP_FLAG_SHARED,
            &resDescDefault,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_indirectCommandBuffer)
        );
        
        if (FAILED(hr)) {
            std::cerr << "[Pipeline] Failed to allocate Indirect Command Buffer" << std::endl;
            return;
        }

        // Phase 3: Register with CUDA (Modern)
        if (m_cudaCore) {
            // Free old if exists
            if (m_extMemIndirectBuffer) m_cudaCore->FreeImportedResource(m_extMemIndirectBuffer);
            
            // Import new
            std::cerr << "[Pipeline] Importing Indirect Buffer (size=" << bufferSize << ")..." << std::endl;
            m_extMemIndirectBuffer = m_cudaCore->ImportD3D12Resource(m_indirectCommandBuffer.Get(), bufferSize);
            m_indirectBufferSize = bufferSize;  // Store for mapping
        }
    }
    
    // 2. Map and Generate Commands
    IndirectCommand* pCommands = nullptr;
    D3D12_RANGE readRange = {0, 0}; // We do not intend to read
    
    HRESULT hr = m_indirectCommandUploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pCommands));
    if (SUCCEEDED(hr)) {
        const uint32_t perObjectSlotSize = 256;
        const uint32_t materialSlotSize = 256;
        
        for (size_t i = 0; i < meshCount; ++i) {
            D3D12Mesh* mesh = m_meshes[i];
            
            // Calculate CBV addresses
            D3D12_GPU_VIRTUAL_ADDRESS perObjectAddr = m_perObjectCB->GetGPUVirtualAddress() + i * perObjectSlotSize;
            D3D12_GPU_VIRTUAL_ADDRESS materialAddr = m_materialCB->GetGPUVirtualAddress() + i * materialSlotSize;
            
            pCommands[i].cbv = perObjectAddr;
            pCommands[i].materialCbv = materialAddr;
            pCommands[i].vbv = mesh->GetVertexBufferView(); // Bind specific mesh geometry
            pCommands[i].ibv = mesh->GetIndexBufferView();
            
            pCommands[i].drawArguments.IndexCountPerInstance = mesh->GetIndexCount();
            pCommands[i].drawArguments.InstanceCount = 1;
            pCommands[i].drawArguments.StartIndexLocation = 0;
            pCommands[i].drawArguments.BaseVertexLocation = 0;
            pCommands[i].drawArguments.StartInstanceLocation = 0;
        }
        
        // DEBUG: Print first command data
        if (meshCount > 0) {
            std::cerr << "[TEST] Uploaded Command[0]:" << std::endl;
            std::cerr << "  CBV: " << pCommands[0].cbv << std::endl;
            std::cerr << "  MatCBV: " << pCommands[0].materialCbv << std::endl;
            std::cerr << "  VBV: Loc=" << pCommands[0].vbv.BufferLocation << " Size=" << pCommands[0].vbv.SizeInBytes << " Stride=" << pCommands[0].vbv.StrideInBytes << std::endl;
            std::cerr << "  IBV: Loc=" << pCommands[0].ibv.BufferLocation << " Size=" << pCommands[0].ibv.SizeInBytes << " Format=" << pCommands[0].ibv.Format << std::endl;
            std::cerr << "  Draw: Idx=" << pCommands[0].drawArguments.IndexCountPerInstance << " Inst=" << pCommands[0].drawArguments.InstanceCount << std::endl;
        }
        
        m_indirectCommandUploadBuffer->Unmap(0, nullptr);

        // Copy from Upload to Default
        TransitionResource(m_indirectCommandBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        m_backend->GetCommandList()->CopyResource(m_indirectCommandBuffer.Get(), m_indirectCommandUploadBuffer.Get());
        TransitionResource(m_indirectCommandBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }
}

void DX12RenderPipeline::ExecuteIndirectPacket(ID3D12GraphicsCommandList* cmdList) {
    std::cerr << "[ExecuteIndirectPacket] Entered. Buffer=" << (m_indirectCommandBuffer ? "valid" : "NULL") 
              << " Signature=" << (m_commandSignature ? "valid" : "NULL") << std::endl;
    if (!m_indirectCommandBuffer || !m_commandSignature) {
        std::cerr << "[ExecuteIndirectPacket] EARLY EXIT - missing buffer or signature!" << std::endl;
        return;
    }
    static int logCount = 0;
    if (logCount++ < 10) { 
        std::cerr << "[Kernel] ExecuteIndirectPacket: Executing " << m_meshes.size() << " max commands" << std::endl; 
    }

    // ===================================================================
    // CRITICAL: After ResetCommandList() in ExecuteRenderKernel, ALL DX12  
    // state is lost. We must rebind EVERYTHING before ExecuteIndirect.
    // This mirrors the state setup in the CPU fallback path (GeometryPass).
    // ===================================================================
    
    // 1. Root signature
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    
    // 2. PSO - use m_skinnedGeometryPassPSO (Unified Shader)
    if (m_debugMode == DebugMode::WIREFRAME && m_wireframePSO) {
        cmdList->SetPipelineState(m_wireframePSO.Get());
    } else {
        // UNIFIED PATH: Always use the Skinned PSO (which is now the Unified PSO)
        if (m_skinnedGeometryPassPSO) {
            cmdList->SetPipelineState(m_skinnedGeometryPassPSO.Get());
        } else {
             cmdList->SetPipelineState(m_gbufferPassPSO.Get());
        }
    }
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // 3. Viewport (render resolution for G-Buffer)
    D3D12_VIEWPORT viewport = {};
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = (FLOAT)m_renderWidth;
    viewport.Height = (FLOAT)m_renderHeight;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    cmdList->RSSetViewports(1, &viewport);

    // 4. Scissor rect
    D3D12_RECT scissor = {};
    scissor.left = 0;
    scissor.top = 0;
    scissor.right = (LONG)m_renderWidth;
    scissor.bottom = (LONG)m_renderHeight;
    cmdList->RSSetScissorRects(1, &scissor);

    // 5. Per-frame constants (camera, matrices, etc.)
    D3D12_GPU_VIRTUAL_ADDRESS perFrameAddress = m_perFrameCB->GetGPUVirtualAddress();
    cmdList->SetGraphicsRootConstantBufferView(0, perFrameAddress);

    // 5b. BIND BONE BUFFER (Unified Path Requirement)
    if (m_globalBoneBuffer) {
        cmdList->SetGraphicsRootShaderResourceView(3, m_globalBoneBuffer->GetGPUVirtualAddress());
    }

    // 6. Bind G-Buffer MRTs as render targets
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[4] = {
        m_gBuffer.albedoRTV,
        m_gBuffer.normalRTV,
        m_gBuffer.emissiveRTV,
        m_gBuffer.velocityRTV
    };
    cmdList->OMSetRenderTargets(4, rtvs, FALSE, &m_gBuffer.depthDSV);

    // 7. Clear G-Buffer (since GBufferPass isn't called in this path)
    const float clearBlack[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Clear Albedo to Black
    cmdList->ClearRenderTargetView(rtvs[0], clearBlack, 0, nullptr);
    cmdList->ClearRenderTargetView(rtvs[1], clearBlack, 0, nullptr);
    cmdList->ClearRenderTargetView(rtvs[2], clearBlack, 0, nullptr);
    cmdList->ClearRenderTargetView(rtvs[3], clearBlack, 0, nullptr);
    cmdList->ClearDepthStencilView(m_gBuffer.depthDSV, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    // =================================================================================
    // BARRIER: Transition Indirect Command Buffer from COMMON -> INDIRECT_ARGUMENT
    // CUDA writes to it in COMMON state (interop requirement).
    // DX12 reads it as INDIRECT_ARGUMENT.
    // =================================================================================
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_indirectCommandBuffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    // Execute Indirect
    // Command structure: [PerObjectCBV, MaterialCBV, VBV, IBV, DrawArgs]
    // Stride: sizeof(IndirectCommand) = 72
    cmdList->ExecuteIndirect(
        m_commandSignature.Get(),
        (UINT)m_meshes.size(),          // MaxCommandCount: execute all command slots
        m_indirectCommandBuffer.Get(),  // ArgumentBuffer: contains the actual commands
        0,                              // ArgumentBufferOffset
        nullptr,                        // CountBuffer: using CPU mesh count for isolation test
        0                               // CountBufferOffset
    );
    
    // =================================================================================
    // BARRIER: Transition BACK to COMMON for CUDA
    // =================================================================================
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    cmdList->ResourceBarrier(1, &barrier);
    
    // Update stats
    m_stats.drawCalls++;
}

void DX12RenderPipeline::UploadBoneMatrices() {
    if (!m_globalBoneBuffer) return;
    
    // Map Buffer
    glm::mat4* mappedData = nullptr;
    D3D12_RANGE readRange = {0, 0};
    if (SUCCEEDED(m_globalBoneBuffer->Map(0, &readRange, reinterpret_cast<void**>(&mappedData)))) {
        
        uint32_t currentBoneOffset = 0;
        
        for (auto* mesh : m_meshes) {
            if (!mesh) continue;
            
            size_t boneCount = mesh->boneMatrices.size();
            if (boneCount > 0) {
                // Check capability
                if (currentBoneOffset + boneCount > MAX_BONES_GLOBAL) {
                    std::cerr << "[Pipeline] WARNING: Bone Buffer Overflow! Increase MAX_BONES_GLOBAL." << std::endl;
                    break; 
                }
                
                // Copy data
                memcpy(mappedData + currentBoneOffset, mesh->boneMatrices.data(), boneCount * sizeof(glm::mat4));
                
                // Update Offset for Shader
                mesh->globalBoneOffset = currentBoneOffset;
                
                // Advance
                currentBoneOffset += (uint32_t)boneCount;
            }
        }
        
        m_globalBoneBuffer->Unmap(0, nullptr);
    }
}

// === Particle Rendering (Phase 7) ===

bool DX12RenderPipeline::CreateParticlePSO() {
    // Phase 7: Particle System using Mesh Shaders
    // Deferring this until Phase 7 to prioritize CudaCore culling (Phase 3).
    // std::cout << "[Pipeline] Particle PSO deferred." << std::endl;
    return false;
}


void DX12RenderPipeline::RenderParticles(ID3D12Resource* particleBuffer, int count) {
    if (!m_particlePSO || !particleBuffer) {
        if (!m_particlePSO) std::cout << "[Pipeline] SKIP: Particle PSO missing" << std::endl;
        return;
    }
    
    // throttle logs
    static int logCount = 0;
    if (logCount++ % 60 == 0) std::cout << "[Pipeline] RenderParticles: " << count << std::endl;
    
    ID3D12GraphicsCommandList6* cmdList6 = nullptr;
    m_backend->GetCommandList()->QueryInterface(IID_PPV_ARGS(&cmdList6));
    if(!cmdList6) return;

    cmdList6->SetPipelineState(m_particlePSO.Get());
    cmdList6->SetGraphicsRootSignature(m_meshShaderRootSig.Get());
    
    // Bind Constants
    cmdList6->SetGraphicsRootConstantBufferView(0, m_perFrameCB->GetGPUVirtualAddress());
    
    // Slot 3: Particle Buffer (StructuredBuffer<Particle>) SRV
    // Note: Root Param 3 expects an SRV.
    cmdList6->SetGraphicsRootShaderResourceView(3, particleBuffer->GetGPUVirtualAddress());
    
    // Dispatch
    int dispatchX = (count + 31) / 32;
    cmdList6->DispatchMesh(dispatchX, 1, 1);
    
    cmdList6->Release();
}


void DX12RenderPipeline::SaveScreenshot(const std::string& filename) {
    if (!m_initialized) return;

    auto device = m_backend->GetDevice();
    auto cmdList = m_backend->GetCommandList();
    auto cmdQueue = m_backend->GetCommandQueue();

    // Source: Output Final Color (Present Buffer)
    // Actually, let's use the current back buffer from backend to be safe it's what's visible
    ID3D12Resource* srcResource = m_backend->GetCurrentBackBuffer();
    if (!srcResource) return; 

    D3D12_RESOURCE_DESC desc = srcResource->GetDesc();
    uint64_t width = desc.Width;
    uint32_t height = desc.Height;
    uint32_t rowPitch = (width * 4 + 255) & ~255;
    uint32_t bufferSize = height * rowPitch;

    if (!m_readbackBuffer || m_readbackBufferSize < bufferSize) {
        m_readbackBufferSize = bufferSize;
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = bufferSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        
        device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_readbackBuffer)
        );
        m_readbackBuffer->SetName(L"ScreenshotReadback");
    }

    // Wait for previous work
    m_backend->WaitForGPU(); 
    m_backend->ResetCommandList();

    // Transition Source
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = srcResource;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT; 
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cmdList->ResourceBarrier(1, &barrier);

    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource = srcResource;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource = m_readbackBuffer.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint.Offset = 0;
    dstLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dstLoc.PlacedFootprint.Footprint.Width = (uint32_t)width;
    dstLoc.PlacedFootprint.Footprint.Height = height;
    dstLoc.PlacedFootprint.Footprint.Depth = 1;
    dstLoc.PlacedFootprint.Footprint.RowPitch = rowPitch;

    cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

    // Transition back
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    cmdList->ResourceBarrier(1, &barrier);

    cmdList->Close();
    ID3D12CommandList* lists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, lists);
    m_backend->WaitForGPU();

    // Write BMP
    void* pData;
    m_readbackBuffer->Map(0, nullptr, &pData);

    BMPHeader header;
    BMPInfoHeader infoHeader;
    infoHeader.width = (int32_t)width;
    infoHeader.height = -(int32_t)height;
    infoHeader.imageSize = width * height * 4;
    header.fileSize = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + infoHeader.imageSize;

    std::ofstream file(filename, std::ios::binary);
    if (file) {
        file.write((char*)&header, sizeof(header));
        file.write((char*)&infoHeader, sizeof(infoHeader));
        uint8_t* rowData = (uint8_t*)pData;
        for (uint32_t y = 0; y < height; ++y) {
             file.write((char*)rowData, width * 4);
             rowData += rowPitch;
        }
    }
    m_readbackBuffer->Unmap(0, nullptr);
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
 
