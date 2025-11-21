#pragma once

#ifdef _WIN32
#include "Rendering/RenderBackend.h"
#include "Rendering/NVIDIAReflex.h"
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdint>
#include <vector>
#include <memory>

namespace CudaGame {
namespace Rendering {

// DX12 Configuration
static constexpr uint32_t FRAME_COUNT = 3; // Triple buffering
static constexpr uint32_t MAX_DESCRIPTORS_PER_HEAP = 256;

// Per-frame resources for triple buffering
struct FrameResources {
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator;
    Microsoft::WRL::ComPtr<ID3D12Resource> renderTarget;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    uint64_t fenceValue = 0;
};

class DX12RenderBackend : public IRenderBackend {
public:
    DX12RenderBackend() = default;
    ~DX12RenderBackend() override;

    // IRenderBackend implementation
    bool Initialize() override;
    void BeginFrame(const glm::vec4& clearColor, int width, int height) override;
    void BindFramebuffer(Framebuffer* fb) override;
    void UnbindFramebuffer() override;
    void BlitDepth(Framebuffer* fb, int width, int height) override;

    bool CreateTexture(const TextureDesc& desc, TextureHandle& outHandle) override;
    void DestroyTexture(TextureHandle& handle) override;

    // DX12-specific methods
    bool CreateSwapchain(void* windowHandle, int width, int height);
    void Present();
    void WaitForGPU();
    void MoveToNextFrame();

    // Getters for advanced features
    ID3D12Device* GetDevice() const { return m_device.Get(); }
    ID3D12GraphicsCommandList* GetCommandList() const { return m_cmdList.Get(); }
    ID3D12CommandQueue* GetCommandQueue() const { return m_graphicsQueue.Get(); }
    
    // Get current render target and depth stencil views
    D3D12_CPU_DESCRIPTOR_HANDLE GetCurrentRenderTargetView() const {
        return m_frameResources[m_currentFrameIndex].rtvHandle;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE GetDepthStencilView() const {
        return m_dsvHandle;
    }
    ID3D12Resource* GetCurrentBackBuffer() const {
        return m_frameResources[m_currentFrameIndex].renderTarget.Get();
    }
    
    // Get descriptor heaps and sizes
    ID3D12DescriptorHeap* GetRTVHeap() const { return m_rtvHeap.Get(); }
    ID3D12DescriptorHeap* GetDSVHeap() const { return m_dsvHeap.Get(); }
    UINT GetRTVDescriptorSize() const { return m_rtvDescriptorSize; }
    UINT GetDSVDescriptorSize() const { return m_dsvDescriptorSize; }

    // NVIDIA Reflex integration
    NVIDIAReflex* GetReflex() { return m_reflex.get(); }
    void SetReflexMode(NVIDIAReflex::Mode mode);
    NVIDIAReflex::Stats GetReflexStats();

private:
    template<typename T>
    using ComPtr = Microsoft::WRL::ComPtr<T>;

public:
    // Buffer creation (vertex/index/constant buffers)
    bool CreateBuffer(size_t sizeInBytes, bool isUploadHeap, ComPtr<ID3D12Resource>& outBuffer);
    bool UploadBufferData(ID3D12Resource* buffer, const void* data, size_t sizeInBytes);

    // Helper methods
    bool CreateDescriptorHeaps();
    bool CreateFrameResources();
    bool CreateDepthStencilBuffer(int width, int height);
    void TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);

    // Core D3D12 objects
    ComPtr<ID3D12Device>           m_device;
    ComPtr<IDXGIFactory6>          m_factory;
    ComPtr<ID3D12CommandQueue>     m_graphicsQueue;
    ComPtr<ID3D12GraphicsCommandList> m_cmdList;

    // Swapchain and backbuffers
    ComPtr<IDXGISwapChain3>        m_swapchain;
    std::vector<FrameResources>    m_frameResources;
    uint32_t                       m_currentFrameIndex = 0;
    uint32_t                       m_backBufferIndex = 0;

    // Descriptor heaps
    ComPtr<ID3D12DescriptorHeap>   m_rtvHeap;           // Render Target Views
    ComPtr<ID3D12DescriptorHeap>   m_dsvHeap;           // Depth Stencil Views
    ComPtr<ID3D12DescriptorHeap>   m_cbvSrvUavHeap;     // CBV/SRV/UAV for textures
    uint32_t                       m_rtvDescriptorSize = 0;
    uint32_t                       m_dsvDescriptorSize = 0;
    uint32_t                       m_cbvSrvUavDescriptorSize = 0;

    // Depth/Stencil buffer
    ComPtr<ID3D12Resource>         m_depthStencilBuffer;
    D3D12_CPU_DESCRIPTOR_HANDLE    m_dsvHandle = {};

    // Synchronization
    ComPtr<ID3D12Fence>            m_fence;
    HANDLE                         m_fenceEvent = nullptr;
    uint64_t                       m_fenceValue = 0;

    // Dimensions
    int m_width = 0;
    int m_height = 0;

    // State
    bool m_initialized = false;

    // NVIDIA Reflex for latency reduction
    std::unique_ptr<NVIDIAReflex> m_reflex;
    uint64_t m_frameCounter = 0;
};

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
