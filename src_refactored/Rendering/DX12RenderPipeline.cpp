#ifdef _WIN32
#include "Rendering/DX12RenderPipeline.h"
#include "Rendering/ShaderCompiler.h"
#include <iostream>

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
    
    // Step 7: Create constant buffers
    if (!CreateConstantBuffers()) {
        std::cerr << "[Pipeline] Failed to create constant buffers" << std::endl;
        return false;
    }

    m_initialized = true;
    std::cout << "[Pipeline] Initialized successfully" << std::endl;
    std::cout << "[Pipeline] G-Buffer: " << m_renderWidth << "x" << m_renderHeight << std::endl;
    std::cout << "[Pipeline] Output: " << m_displayWidth << "x" << m_displayHeight << std::endl;
    
    return true;
}

void DX12RenderPipeline::Shutdown() {
    if (!m_initialized) return;

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
    
    // Begin backend frame (clears swap chain render target)
    glm::vec4 clearColor = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);  // BRIGHT CYAN for debugging
    m_backend->BeginFrame(clearColor, m_displayWidth, m_displayHeight);
}

void DX12RenderPipeline::RenderFrame() {
    if (!m_initialized || !m_camera) {
        return;
    }

    // AAA Rendering Pipeline:
    // 1. Shadow pass (light perspective)
    // 2. G-Buffer pass (geometry → MRTs at render res)
    // 3. Geometry pass (temporary forward debug pass → swapchain)
    // 4. Lighting pass (deferred PBR)
    // 5. Ray tracing pass (reflections, shadows, AO)
    // 6. DLSS upscaling (render res → display res)
    // 7. Post-processing (bloom, tone mapping)
    // 8. UI rendering (display res)

    ShadowPass();
    GBufferPass();
    GeometryPass();
    LightingPass();
    
    if (m_rayTracingEnabled) {
        RayTracingPass();
    }
    
    if (m_dlssEnabled) {
        DLSSPass();
    }
    
    PostProcessPass();
    UIPass();
    
    // Calculate total frame time
    m_stats.totalFrameMs = m_stats.geometryPassMs + m_stats.lightingPassMs + 
                           m_stats.rayTracingPassMs + m_stats.dlssPassMs;
}

void DX12RenderPipeline::EndFrame() {
    // Present to swap chain
    m_backend->Present();
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

    // Constant buffer slot sizes (256-byte aligned)
    const uint32_t perObjectSlotSize = 256;
    const uint32_t materialSlotSize  = 256;

    uint32_t meshIndex = 0;
    for (D3D12Mesh* mesh : m_meshes) {
        meshIndex++;
        if (!mesh) continue;

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
    if (m_meshes.empty()) {
        return;
    }
    
    ID3D12GraphicsCommandList* cmdList = m_backend->GetCommandList();
    
    // Bind root signature and PSO (wireframe or solid based on debug mode)
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    if (m_debugMode == DebugMode::WIREFRAME && m_wireframePSO) {
        cmdList->SetPipelineState(m_wireframePSO.Get());
    } else {
        cmdList->SetPipelineState(m_geometryPassPSO.Get());
    }
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // Set viewport and scissor to DISPLAY resolution (since we're rendering directly to swap chain)
    D3D12_VIEWPORT viewport = {};
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = (FLOAT)m_displayWidth;  // Use display resolution, not render resolution
    viewport.Height = (FLOAT)m_displayHeight;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    cmdList->RSSetViewports(1, &viewport);
    
    D3D12_RECT scissor = {};
    scissor.left = 0;
    scissor.top = 0;
    scissor.right = (LONG)m_displayWidth;  // Use display resolution
    scissor.bottom = (LONG)m_displayHeight;
    cmdList->RSSetScissorRects(1, &scissor);
    
    // Bind per-frame constants (same for all meshes)
    D3D12_GPU_VIRTUAL_ADDRESS perFrameAddress = m_perFrameCB->GetGPUVirtualAddress();
    cmdList->SetGraphicsRootConstantBufferView(0, perFrameAddress);
    
    // FORWARD RENDERING: Render directly to swap chain (like OpenGL version)
    // Bind swap chain render target + main depth buffer (BeginFrame already cleared these)
    D3D12_CPU_DESCRIPTOR_HANDLE swapChainRTV = m_backend->GetCurrentRenderTargetView();
    D3D12_CPU_DESCRIPTOR_HANDLE mainDSV = m_backend->GetDepthStencilView();
    cmdList->OMSetRenderTargets(1, &swapChainRTV, FALSE, &mainDSV);
    
    // DEBUG: Print camera matrices once per second
    static uint32_t debugFrameCounter = 0;
    if (debugFrameCounter++ % 60 == 0 && m_perFrameData) {
        std::cout << "[DEBUG] Camera pos: (" << m_perFrameData->cameraPosition.x << ", " 
                  << m_perFrameData->cameraPosition.y << ", " << m_perFrameData->cameraPosition.z << ")" << std::endl;
        std::cout << "[DEBUG] ViewProj[0]: (" << m_perFrameData->viewProjMatrix[0][0] << ", " 
                  << m_perFrameData->viewProjMatrix[0][1] << ", " << m_perFrameData->viewProjMatrix[0][2] << ", "
                  << m_perFrameData->viewProjMatrix[0][3] << ")" << std::endl;
    }
    
    // Render all meshes
    uint32_t drawCallCount = 0;
    uint32_t triangleCount = 0;

    // Constant buffer slot sizes (256-byte aligned)
    const uint32_t perObjectSlotSize = 256;
    const uint32_t materialSlotSize  = 256;
    
    uint32_t meshIndex = 0;
    for (D3D12Mesh* mesh : m_meshes) {
        meshIndex++;
        if (!mesh) continue;
        
        // Update per-object constants (GLM column-major, matches HLSL column_major)
        if (m_perObjectData) {
            // World transform comes directly from ECS (TransformComponent::getMatrix)
            glm::mat4 world = mesh->transform;

            // Each mesh gets its own 256-byte slot in the per-object constant buffer.
            uint32_t meshSlot = meshIndex - 1; // 0-based index
            if (meshSlot >= MAX_MESHES_PER_FRAME) {
                break; // Should never happen in this demo
            }
            PerObjectConstants* objCB = reinterpret_cast<PerObjectConstants*>(
                reinterpret_cast<uint8_t*>(m_perObjectData) + meshSlot * perObjectSlotSize);

            objCB->worldMatrix = world;
            objCB->prevWorldMatrix = world;  // TODO: Store prev transform
            // Normal matrix is transpose(inverse(world))
            objCB->normalMatrix = glm::transpose(glm::inverse(world));
        }
        
        // Update material constants (each mesh has its own slot)
        if (m_materialData) {
            uint32_t meshSlot = meshIndex - 1;
            if (meshSlot >= MAX_MESHES_PER_FRAME) {
                break;
            }

            MaterialConstants* matCB = reinterpret_cast<MaterialConstants*>(
                reinterpret_cast<uint8_t*>(m_materialData) + meshSlot * materialSlotSize);

            const Material& mat = mesh->GetMaterial();
            matCB->albedoColor       = mat.albedoColor;
            matCB->roughness         = mat.roughness;
            matCB->metallic          = mat.metallic;
            matCB->ambientOcclusion  = mat.ambientOcclusion;
            matCB->emissiveStrength  = mat.emissiveStrength;
            matCB->emissiveColor     = mat.emissiveColor;
        }
        
        // Bind per-object and material constant buffers for this mesh (using correct slot offsets)
        uint32_t meshSlot = meshIndex - 1;
        D3D12_GPU_VIRTUAL_ADDRESS perObjectAddress = m_perObjectCB->GetGPUVirtualAddress() + meshSlot * perObjectSlotSize;
        D3D12_GPU_VIRTUAL_ADDRESS materialAddress  = m_materialCB->GetGPUVirtualAddress()  + meshSlot * materialSlotSize;
        cmdList->SetGraphicsRootConstantBufferView(1, perObjectAddress);
        cmdList->SetGraphicsRootConstantBufferView(2, materialAddress);
        
        // Bind vertex and index buffers
        const D3D12_VERTEX_BUFFER_VIEW& vbv = mesh->GetVertexBufferView();
        const D3D12_INDEX_BUFFER_VIEW& ibv = mesh->GetIndexBufferView();
        
        cmdList->IASetVertexBuffers(0, 1, &vbv);
        cmdList->IASetIndexBuffer(&ibv);
        
        // Draw mesh
        cmdList->DrawIndexedInstanced(mesh->GetIndexCount(), 1, 0, 0, 0);
        
        drawCallCount++;
        triangleCount += mesh->GetIndexCount() / 3;
    }
    
    m_stats.geometryPassMs = 1.0f;  // TODO: Real timing
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
    // TODO: Implement deferred lighting
    // - Read G-Buffer
    // - Apply PBR lighting (directional, point, spot lights)
    // - Write to litColor buffer
    
    m_stats.lightingPassMs = 2.0f;  // Placeholder
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
    // TODO: Implement post-processing
    // - Bloom
    // - Tone mapping
    // - Color grading
    // - Vignette, chromatic aberration, etc.
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
        D3D12_HEAP_FLAG_NONE,
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
    
    std::cout << "[Pipeline] All shaders compiled successfully" << std::endl;
    return true;
}

bool DX12RenderPipeline::CreateRootSignature() {
    std::cout << "[Pipeline] Creating root signature..." << std::endl;
    
    // Root parameters for geometry pass
    // - 1x CBV for per-frame constants (camera matrices, time)
    // - 1x CBV for per-object constants (world matrix)
    // - 1x CBV for material constants (albedo, roughness, metallic)
    D3D12_ROOT_PARAMETER1 rootParams[3] = {};
    
    // Per-frame constants (b0)
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;  // b0
    rootParams[0].Descriptor.RegisterSpace = 0;
    rootParams[0].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    // Per-object constants (b1)
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[1].Descriptor.ShaderRegister = 1;  // b1
    rootParams[1].Descriptor.RegisterSpace = 0;
    rootParams[1].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    // Material constants (b2)
    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[2].Descriptor.ShaderRegister = 2;  // b2
    rootParams[2].Descriptor.RegisterSpace = 0;
    rootParams[2].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
    rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    
    // Root signature desc
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rootSigDesc.Desc_1_1.NumParameters = 3;
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
    
    // Input layout (matches Vertex struct)
    D3D12_INPUT_ELEMENT_DESC inputElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
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

bool DX12RenderPipeline::CreateGBufferPassPSO() {
    std::cout << "[Pipeline] Creating G-Buffer geometry pass PSO..." << std::endl;

    // Input layout (matches Vertex struct)
    D3D12_INPUT_ELEMENT_DESC inputElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
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

bool DX12RenderPipeline::CreateConstantBuffers() {
    std::cout << "[Pipeline] Creating constant buffers..." << std::endl;
    
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
    
    // Material constants (updated per mesh) - also allocate a slot per mesh
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
    
    std::cout << "[Pipeline] Constant buffers created and mapped" << std::endl;
    return true;
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
