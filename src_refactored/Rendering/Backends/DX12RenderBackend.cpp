#ifdef _WIN32
#include "Rendering/Backends/DX12RenderBackend.h"
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <stdexcept>
#include <iostream>

namespace CudaGame {
namespace Rendering {

DX12RenderBackend::~DX12RenderBackend() {
    if (m_initialized) {
        WaitForGPU();
        
        if (m_fenceEvent) {
            CloseHandle(m_fenceEvent);
            m_fenceEvent = nullptr;
        }
    }
}

bool DX12RenderBackend::Initialize() {
    if (m_initialized) {
        std::cout << "[DX12] Backend already initialized." << std::endl;
        return true;
    }

    HRESULT hr;
    
    // Enable debug layer in debug builds
    // Enable debug layer (during dev, even in Release to catch PSO errors)
    // #ifdef _DEBUG
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
        debugController->EnableDebugLayer();
        std::cout << "[DX12] Debug layer enabled" << std::endl;
    }
    // #endif

    // Create DXGI factory
    UINT factoryFlags = 0;
#ifdef _DEBUG
    factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif
    
    hr = CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_factory));
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create DXGI factory" << std::endl;
        return false;
    }

    // Enumerate and select hardware adapter (prefer NVIDIA for RTX features)
    ComPtr<IDXGIAdapter1> adapter;
    SIZE_T maxDedicatedVideoMemory = 0;
    ComPtr<IDXGIAdapter1> selectedAdapter;
    
    for (UINT adapterIndex = 0; m_factory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapterIndex) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        
        // Skip software adapters
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        
        // Test D3D12 support
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr))) {
            // Select adapter with most VRAM (typically the discrete GPU)
            if (desc.DedicatedVideoMemory > maxDedicatedVideoMemory) {
                maxDedicatedVideoMemory = desc.DedicatedVideoMemory;
                selectedAdapter = adapter;
                
                // Log adapter info
                std::wcout << L"[DX12] Found adapter: " << desc.Description 
                          << L" (" << (desc.DedicatedVideoMemory / 1024 / 1024) << L" MB)" << std::endl;
            }
        }
    }

    if (!selectedAdapter) {
        std::cerr << "[DX12] No suitable D3D12 adapter found" << std::endl;
        return false;
    }

    // Create D3D12 device
    hr = D3D12CreateDevice(selectedAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device));
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create D3D12 device" << std::endl;
        return false;
    }
    
    std::cout << "[DX12] Device created successfully" << std::endl;

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;
    
    hr = m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_graphicsQueue));
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create command queue" << std::endl;
        return false;
    }

    // Create descriptor heaps
    if (!CreateDescriptorHeaps()) {
        std::cerr << "[DX12] Failed to create descriptor heaps" << std::endl;
        return false;
    }

    // Initialize frame resources vector (swapchain will populate them later)
    m_frameResources.resize(FRAME_COUNT);

    // Create initial command allocator and command list
    ComPtr<ID3D12CommandAllocator> tempAllocator;
    hr = m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&tempAllocator));
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create command allocator" << std::endl;
        return false;
    }
    
    hr = m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, 
                                      tempAllocator.Get(), nullptr, IID_PPV_ARGS(&m_cmdList));
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create command list" << std::endl;
        return false;
    }
    
    // Close command list initially
    m_cmdList->Close();

    // Create synchronization objects
    hr = m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create fence" << std::endl;
        return false;
    }
    
    m_fenceValue = 1;
    
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) {
        std::cerr << "[DX12] Failed to create fence event" << std::endl;
        return false;
    }

    // Initialize NVIDIA Reflex for latency reduction
    m_reflex = std::make_unique<NVIDIAReflex>();
    /* 
    // DISABLE REFLEX FOR DEBUGGING STABILITY
    if (m_reflex->Initialize(m_device.Get())) {
        // Enable Reflex with boost for lowest latency
        m_reflex->SetMode(NVIDIAReflex::Mode::ENABLED_BOOST);
        m_reflex->SetSleepMode(true);
        std::cout << "[DX12] NVIDIA Reflex enabled" << std::endl;
    } else {
        std::cerr << "[DX12] Reflex initialization failed (non-critical)" << std::endl;
    }
    */
    std::cout << "[DX12] Reflex DISABLED for debugging" << std::endl;

    m_initialized = true;
    std::cout << "[DX12] Backend initialized successfully" << std::endl;
    // std::cout << "[DX12] Backend initialized successfully" << std::endl; // Duplicate in original?
    return true;
}

void DX12RenderBackend::ResetCommandList() {
    FrameResources& frame = m_frameResources[m_currentFrameIndex];
    if (FAILED(frame.commandAllocator->Reset())) {
        std::cerr << "[DX12] Failed to reset command allocator" << std::endl;
        return;
    }
    if (FAILED(m_cmdList->Reset(frame.commandAllocator.Get(), nullptr))) {
        std::cerr << "[DX12] Failed to reset command list" << std::endl;
    }
}

bool DX12RenderBackend::CreateDescriptorHeaps() {
    HRESULT hr;
    
    // RTV descriptor heap (3 swapchain + 4 G-Buffer + 1 LitColor = 8 total)
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = FRAME_COUNT + 5; // Swapchain + G-Buffer RTVs + LitColor
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    
    hr = m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap));
    if (FAILED(hr)) return false;
    
    m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // DSV descriptor heap (1 swapchain + 1 G-Buffer = 2 total)
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 2;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    
    hr = m_device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_dsvHeap));
    if (FAILED(hr)) return false;
    
    m_dsvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    // CBV/SRV/UAV descriptor heap (shader-visible)
    D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavHeapDesc = {};
    cbvSrvUavHeapDesc.NumDescriptors = MAX_DESCRIPTORS_PER_HEAP;
    cbvSrvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvSrvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    
    hr = m_device->CreateDescriptorHeap(&cbvSrvUavHeapDesc, IID_PPV_ARGS(&m_cbvSrvUavHeap));
    if (FAILED(hr)) return false;
    
    m_cbvSrvUavDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    
    std::cout << "[DX12] Descriptor heaps created successfully" << std::endl;
    return true;
}

bool DX12RenderBackend::CreateFrameResources() {
    HRESULT hr;
    
    // Frame resources vector should already be resized in Initialize()
    if (m_frameResources.size() != FRAME_COUNT) {
        m_frameResources.resize(FRAME_COUNT);
    }
    
    // Get RTV handle for first descriptor
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    
    // Create render target view for each frame
    for (uint32_t i = 0; i < FRAME_COUNT; ++i) {
        // Get backbuffer from swapchain
        hr = m_swapchain->GetBuffer(i, IID_PPV_ARGS(&m_frameResources[i].renderTarget));
        if (FAILED(hr)) {
            std::cerr << "[DX12] Failed to get swapchain buffer " << i << std::endl;
            return false;
        }
        
        // Create RTV
        m_device->CreateRenderTargetView(m_frameResources[i].renderTarget.Get(), nullptr, rtvHandle);
        m_frameResources[i].rtvHandle = rtvHandle;
        
        // Move to next descriptor
        rtvHandle.ptr += m_rtvDescriptorSize;
        
        // Create command allocator per frame
        hr = m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, 
                                              IID_PPV_ARGS(&m_frameResources[i].commandAllocator));
        if (FAILED(hr)) {
            std::cerr << "[DX12] Failed to create command allocator for frame " << i << std::endl;
            return false;
        }
    }
    
    std::cout << "[DX12] Frame resources created for " << FRAME_COUNT << " frames" << std::endl;
    return true;
}

bool DX12RenderBackend::CreateSwapchain(void* windowHandle, int width, int height) {
    if (!m_device || !m_graphicsQueue) {
        std::cerr << "[DX12] Cannot create swapchain: device/queue not initialized" << std::endl;
        return false;
    }
    
    m_width = width;
    m_height = height;
    
    // Get native window handle from GLFW
    HWND hwnd = glfwGetWin32Window(static_cast<GLFWwindow*>(windowHandle));
    
    // Describe swapchain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.Stereo = FALSE;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = FRAME_COUNT;
    swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING; // For variable refresh rate
    
    // Create swapchain
    ComPtr<IDXGISwapChain1> swapChain1;
    HRESULT hr = m_factory->CreateSwapChainForHwnd(
        m_graphicsQueue.Get(),
        hwnd,
        &swapChainDesc,
        nullptr,
        nullptr,
        &swapChain1
    );
    
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create swapchain" << std::endl;
        return false;
    }
    
    // Disable Alt+Enter fullscreen toggle (handle it ourselves)
    m_factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    
    // Get SwapChain3 interface
    hr = swapChain1.As(&m_swapchain);
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to get SwapChain3 interface" << std::endl;
        return false;
    }
    
    m_backBufferIndex = m_swapchain->GetCurrentBackBufferIndex();
    
    // Create frame resources (RTVs, command allocators)
    if (!CreateFrameResources()) {
        std::cerr << "[DX12] Failed to create frame resources" << std::endl;
        return false;
    }
    
    // Create depth/stencil buffer
    if (!CreateDepthStencilBuffer(width, height)) {
        std::cerr << "[DX12] Failed to create depth/stencil buffer" << std::endl;
        return false;
    }
    
    std::cout << "[DX12] Swapchain created: " << width << "x" << height << " (" << FRAME_COUNT << " buffers)" << std::endl;
    return true;
}

bool DX12RenderBackend::CreateDepthStencilBuffer(int width, int height) {
    // Describe depth/stencil buffer
    D3D12_RESOURCE_DESC depthStencilDesc = {};
    depthStencilDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthStencilDesc.Alignment = 0;
    depthStencilDesc.Width = width;
    depthStencilDesc.Height = height;
    depthStencilDesc.DepthOrArraySize = 1;
    depthStencilDesc.MipLevels = 1;
    depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.SampleDesc.Count = 1;
    depthStencilDesc.SampleDesc.Quality = 0;
    depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    
    D3D12_CLEAR_VALUE clearValue = {};
    clearValue.Format = DXGI_FORMAT_D32_FLOAT;
    clearValue.DepthStencil.Depth = 1.0f;
    clearValue.DepthStencil.Stencil = 0;
    
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    
    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &depthStencilDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &clearValue,
        IID_PPV_ARGS(&m_depthStencilBuffer)
    );
    
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create depth/stencil buffer" << std::endl;
        return false;
    }
    
    // Create depth/stencil view
    m_dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    m_device->CreateDepthStencilView(m_depthStencilBuffer.Get(), nullptr, m_dsvHandle);
    
    std::cout << "[DX12] Depth/stencil buffer created" << std::endl;
    return true;
}

void DX12RenderBackend::BeginFrame(const glm::vec4& clearColor, int width, int height) {
    if (!m_swapchain) {
        std::cerr << "[DX12] BeginFrame called without swapchain" << std::endl;
        return;
    }
    // std::cout << "[DX12] BeginFrame..." << std::endl;
    
    m_frameCounter++;
    
    // REFLEX MARKER: Start render submit
    if (m_reflex && m_reflex->IsSupported()) {
        m_reflex->SetMarker(NVIDIAReflex::Marker::RENDERSUBMIT_START, m_frameCounter);
    }
    
    FrameResources& frame = m_frameResources[m_currentFrameIndex];
    
    // Reset command allocator
    frame.commandAllocator->Reset();
    
    // Reset command list
    m_cmdList->Reset(frame.commandAllocator.Get(), nullptr);
    
    // Transition render target to render target state
    // Transition render target to render target state
    TransitionResource(frame.renderTarget.Get(), 
                      D3D12_RESOURCE_STATE_PRESENT, 
                      D3D12_RESOURCE_STATE_RENDER_TARGET);
    // std::cout << "SKIP BARRIER (BeginFrame)" << std::endl;
    
    // Set render targets
    m_cmdList->OMSetRenderTargets(1, &frame.rtvHandle, FALSE, &m_dsvHandle);
    
    // Clear render target
    const float clearColorArray[4] = { clearColor.r, clearColor.g, clearColor.b, clearColor.a };
    m_cmdList->ClearRenderTargetView(frame.rtvHandle, clearColorArray, 0, nullptr);
    
    // Clear depth/stencil
    m_cmdList->ClearDepthStencilView(m_dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    
    // Set viewport and scissor
    D3D12_VIEWPORT viewport = {};
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    m_cmdList->RSSetViewports(1, &viewport);
    
    D3D12_RECT scissorRect = {};
    scissorRect.left = 0;
    scissorRect.top = 0;
    scissorRect.right = width;
    scissorRect.bottom = height;
    m_cmdList->RSSetScissorRects(1, &scissorRect);
}

void DX12RenderBackend::Present() {
    if (!m_swapchain) return;
    
    FrameResources& frame = m_frameResources[m_currentFrameIndex];
    
    // Transition render target back to present state
    // Transition render target back to present state
    TransitionResource(frame.renderTarget.Get(), 
                      D3D12_RESOURCE_STATE_RENDER_TARGET, 
                      D3D12_RESOURCE_STATE_PRESENT);
    // std::cout << "SKIP BARRIER (Present)" << std::endl;
    
    // REFLEX MARKER: End render submit (commands recorded, about to execute)
    if (m_reflex && m_reflex->IsSupported()) {
        m_reflex->SetMarker(NVIDIAReflex::Marker::RENDERSUBMIT_END, m_frameCounter);
    }
    
    // Execute command list
    m_cmdList->Close();
    ID3D12CommandList* cmdLists[] = { m_cmdList.Get() };
    m_graphicsQueue->ExecuteCommandLists(1, cmdLists);
    
    // REFLEX MARKER: About to present
    if (m_reflex && m_reflex->IsSupported()) {
        m_reflex->SetMarker(NVIDIAReflex::Marker::PRESENT_START, m_frameCounter);
    }
    
    // Present (vsync off with tearing for low latency)
    // std::cout << "[DX12] Presenting..." << std::endl;
    HRESULT hr = m_swapchain->Present(0, DXGI_PRESENT_ALLOW_TEARING);
    if (FAILED(hr)) {
        std::cerr << "[DX12] Present failed with HRESULT: " << std::hex << hr << std::dec << std::endl;
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
            HRESULT reason = m_device->GetDeviceRemovedReason();
            std::cerr << "[DX12] Device Removed Reason: " << std::hex << reason << std::dec << std::endl;
        }
    }
    
    // REFLEX MARKER: Present completed
    if (m_reflex && m_reflex->IsSupported()) {
        m_reflex->SetMarker(NVIDIAReflex::Marker::PRESENT_END, m_frameCounter);
    }
    
    // Move to next frame
    MoveToNextFrame();
}

void DX12RenderBackend::MoveToNextFrame() {
    // Schedule signal command
    const uint64_t currentFenceValue = m_fenceValue;
    m_graphicsQueue->Signal(m_fence.Get(), currentFenceValue);
    m_fenceValue++;
    
    // Update frame index
    m_backBufferIndex = m_swapchain->GetCurrentBackBufferIndex();
    m_currentFrameIndex = (m_currentFrameIndex + 1) % FRAME_COUNT;
    
    // Wait if GPU hasn't finished with this frame's resources
    FrameResources& nextFrame = m_frameResources[m_currentFrameIndex];
    if (m_fence->GetCompletedValue() < nextFrame.fenceValue) {
        std::cout << "[DX12] Waiting for Fence " << nextFrame.fenceValue << std::endl;
        m_fence->SetEventOnCompletion(nextFrame.fenceValue, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
        std::cout << "[DX12] Wait Complete" << std::endl;
    }
    
    nextFrame.fenceValue = currentFenceValue;
}

void DX12RenderBackend::WaitForGPU() {
    if (!m_graphicsQueue || !m_fence) return;
    
    // Signal and wait
    m_graphicsQueue->Signal(m_fence.Get(), m_fenceValue);
    m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent);
    WaitForSingleObject(m_fenceEvent, INFINITE);
    m_fenceValue++;
}

void DX12RenderBackend::TransitionResource(ID3D12Resource* resource, 
                                           D3D12_RESOURCE_STATES before, 
                                           D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    
    m_cmdList->ResourceBarrier(1, &barrier);
}

void DX12RenderBackend::BindFramebuffer(Framebuffer* /*fb*/) {
    // TODO: support offscreen targets with descriptor tables
}

void DX12RenderBackend::UnbindFramebuffer() {
    // Bind backbuffer again
    FrameResources& frame = m_frameResources[m_currentFrameIndex];
    m_cmdList->OMSetRenderTargets(1, &frame.rtvHandle, FALSE, &m_dsvHandle);
}

void DX12RenderBackend::BlitDepth(Framebuffer* /*fb*/, int /*width*/, int /*height*/) {
    // TODO: implement depth copy with compute shader or PSO
}

bool DX12RenderBackend::CreateTexture(const TextureDesc& desc, TextureHandle& outHandle) {
    if (!m_device) {
        std::cerr << "[DX12] Cannot create texture: device not initialized" << std::endl;
        return false;
    }

    // Map TextureFormat to DXGI_FORMAT
    DXGI_FORMAT dxgiFormat = DXGI_FORMAT_UNKNOWN;
    bool isDepthFormat = false;
    
    switch (desc.format) {
        case TextureFormat::RGBA8:
            dxgiFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
            break;
        case TextureFormat::RGB8:
            dxgiFormat = DXGI_FORMAT_R8G8B8A8_UNORM; // DX12 doesn't support RGB8, use RGBA8
            break;
        case TextureFormat::RG16F:
            dxgiFormat = DXGI_FORMAT_R16G16_FLOAT;
            break;
        case TextureFormat::RGB16F:
            dxgiFormat = DXGI_FORMAT_R16G16B16A16_FLOAT; // No RGB16F in DX12
            break;
        case TextureFormat::RGB32F:
            dxgiFormat = DXGI_FORMAT_R32G32B32A32_FLOAT; // No RGB32F in DX12
            break;
        case TextureFormat::DEPTH24:
            dxgiFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
            isDepthFormat = true;
            break;
        case TextureFormat::DEPTH32F:
            dxgiFormat = DXGI_FORMAT_D32_FLOAT;
            isDepthFormat = true;
            break;
        default:
            std::cerr << "[DX12] Unsupported texture format" << std::endl;
            return false;
    }

    // Create resource description
    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = desc.width;
    resourceDesc.Height = desc.height;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = desc.mipLevels;
    resourceDesc.Format = dxgiFormat;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // Add appropriate flags
    if (isDepthFormat) {
        resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    } else {
        resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }

    // Heap properties (default heap for GPU-only resources)
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    // Clear value (for render targets/depth)
    D3D12_CLEAR_VALUE clearValue = {};
    clearValue.Format = dxgiFormat;
    D3D12_CLEAR_VALUE* pClearValue = nullptr;
    
    if (isDepthFormat) {
        clearValue.DepthStencil.Depth = 1.0f;
        clearValue.DepthStencil.Stencil = 0;
        pClearValue = &clearValue;
    } else if (resourceDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
        clearValue.Color[0] = 0.0f;
        clearValue.Color[1] = 0.0f;
        clearValue.Color[2] = 0.0f;
        clearValue.Color[3] = 1.0f;
        pClearValue = &clearValue;
    }

    // Initial state
    D3D12_RESOURCE_STATES initialState = isDepthFormat ? 
        D3D12_RESOURCE_STATE_DEPTH_WRITE : 
        D3D12_RESOURCE_STATE_RENDER_TARGET;

    // Create committed resource
    ComPtr<ID3D12Resource> resource;
    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        initialState,
        pClearValue,
        IID_PPV_ARGS(&resource)
    );

    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create texture resource" << std::endl;
        return false;
    }

    // Create descriptor (SRV for shader access)
    // For now, store the resource pointer directly in the handle
    // In production, we'd allocate from descriptor heap and store descriptor index
    outHandle.value = reinterpret_cast<uint64_t>(resource.Detach());

    std::cout << "[DX12] Created texture: " << desc.width << "x" << desc.height 
              << " (Format: " << static_cast<int>(desc.format) << ")" << std::endl;

    return true;
}

void DX12RenderBackend::DestroyTexture(TextureHandle& handle) {
    if (handle.value == 0) return;

    // Retrieve resource pointer from handle
    ID3D12Resource* resource = reinterpret_cast<ID3D12Resource*>(handle.value);
    if (resource) {
        resource->Release();
        std::cout << "[DX12] Texture destroyed" << std::endl;
    }

    handle.value = 0;
}

void DX12RenderBackend::SetReflexMode(NVIDIAReflex::Mode mode) {
    if (m_reflex) {
        m_reflex->SetMode(mode);
    }
}

NVIDIAReflex::Stats DX12RenderBackend::GetReflexStats() {
    if (m_reflex) {
        return m_reflex->GetStats();
    }
    return NVIDIAReflex::Stats{};
}

bool DX12RenderBackend::CreateBuffer(size_t sizeInBytes, bool isUploadHeap, ComPtr<ID3D12Resource>& outBuffer) {
    if (!m_device) {
        std::cerr << "[DX12] Cannot create buffer: device not initialized" << std::endl;
        return false;
    }

    // Heap properties
    D3D12_HEAP_PROPERTIES heapProps = {};
    if (isUploadHeap) {
        // Upload heap: CPU can write, GPU can read (for staging/dynamic data)
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    } else {
        // Default heap: GPU-only (best performance for static data)
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    }
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    // Resource description
    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = sizeInBytes;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // Initial state
    D3D12_RESOURCE_STATES initialState = isUploadHeap ? 
        D3D12_RESOURCE_STATE_GENERIC_READ : 
        D3D12_RESOURCE_STATE_COMMON;

    // Create buffer
    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&outBuffer)
    );

    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to create buffer (size: " << sizeInBytes << " bytes)" << std::endl;
        return false;
    }

    std::cout << "[DX12] Created buffer: " << sizeInBytes << " bytes "
              << (isUploadHeap ? "(Upload Heap)" : "(Default Heap)") << std::endl;

    return true;
}

bool DX12RenderBackend::UploadBufferData(ID3D12Resource* buffer, const void* data, size_t sizeInBytes) {
    if (!buffer || !data) {
        std::cerr << "[DX12] Invalid buffer or data pointer" << std::endl;
        return false;
    }

    // Map buffer memory (only works for upload heaps)
    void* mappedData = nullptr;
    D3D12_RANGE readRange = {0, 0}; // We won't read from this buffer
    
    HRESULT hr = buffer->Map(0, &readRange, &mappedData);
    if (FAILED(hr)) {
        std::cerr << "[DX12] Failed to map buffer for upload" << std::endl;
        return false;
    }

    // Copy data
    memcpy(mappedData, data, sizeInBytes);

    // Unmap
    D3D12_RANGE writtenRange = {0, sizeInBytes};
    buffer->Unmap(0, &writtenRange);

    std::cout << "[DX12] Uploaded " << sizeInBytes << " bytes to buffer" << std::endl;
    return true;
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
