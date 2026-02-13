#include "Core/CudaCore.h"
#include <string>
#include <iostream>

// Windows/DX12 includes
#include <d3d12.h>
#include <dxgi1_4.h>

// CUDA includes
// #include <cuda_d3d12_interop.h> // Usually needed, but cuda_runtime.h might cover some. 
// We likely need to ensure the build system links against cudart.lib

namespace CudaGame {
namespace Core {

// Helper for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CudaCore] CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)

CudaCore::CudaCore() {}

CudaCore::~CudaCore() {
    Shutdown();
}

bool CudaCore::Initialize(ID3D12Device* d3dDevice) {
    if (m_initialized) return true;
    m_d3dDevice = d3dDevice;

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CudaCore] No CUDA devices found!" << std::endl;
        return false;
    }

    // Pick first device
    m_cudaDeviceID = 0;
    CUDA_CHECK(cudaSetDevice(m_cudaDeviceID));
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, m_cudaDeviceID);
    std::cout << "[CudaCore] Initialized Device: " << props.name << std::endl;

    m_initialized = true;
    return true;
}

void CudaCore::Shutdown() {
    if (!m_initialized) return;
    cudaDeviceSynchronize();
    m_initialized = false;
}

// ---------------------------------------------------------
// Modern Interop: External Memory (DX12 -> CUDA)
// ---------------------------------------------------------

cudaExternalMemory_t CudaCore::ImportD3D12Resource(ID3D12Resource* d3dResource, size_t size) {
    if (!d3dResource) return nullptr;

    cudaExternalMemoryHandleDesc desc = {};
    desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.handle = d3dResource; // For NT handles, we'd need CreateSharedHandle. 
                                            // But for D3D12 Resource, we pass the pointer IF the driver supports it?
                                            // Actually, dedicated resources usually require a shared handle.
                                            // However, cudaExternalMemoryHandleTypeD3D12Resource allows passing the ID3D12Resource* directly
                                            // IF the flags D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS are set?
                                            // NOTE: Standard practice is D3D12Resource* directly in modern CUDA drivers.

    desc.handle.win32.handle = d3dResource; 
    
    // Use the allocation size from D3D12
    D3D12_RESOURCE_ALLOCATION_INFO allocInfo = m_d3dDevice->GetResourceAllocationInfo(0, 1, &d3dResource->GetDesc());
    desc.size = allocInfo.SizeInBytes; // Must match allocation exactly
    
    HANDLE sharedHandle = nullptr;
    HRESULT hr = m_d3dDevice->CreateSharedHandle(
        d3dResource,
        nullptr,
        GENERIC_ALL,
        nullptr,
        &sharedHandle
    );
    
    if (FAILED(hr)) {
        std::cerr << "[CudaCore] Failed to create shared handle: " << std::hex << hr << std::endl;
        return nullptr;
    }

    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = sharedHandle;
    desc.flags = cudaExternalMemoryDedicated;

    cudaExternalMemory_t extMem = nullptr;
    cudaError_t err = cudaImportExternalMemory(&extMem, &desc);
    
    // Close the NT handle (CUDA has its own reference now if successful, or we failed)
    CloseHandle(sharedHandle);
    
    if (err != cudaSuccess) {
        std::cerr << "[CudaCore] Failed to import external memory: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return extMem;
}

void CudaCore::FreeImportedResource(cudaExternalMemory_t extMem) {
    if (extMem) {
        CUDA_CHECK(cudaDestroyExternalMemory(extMem));
    }
}

void* CudaCore::MapImportedResource(cudaExternalMemory_t extMem, size_t offset, size_t size) {
    if (!extMem) {
        std::cerr << "[CudaCore] MapImportedResource: extMem is NULL!" << std::endl;
        return nullptr;
    }
    
    // Size of 0 is invalid for cudaExternalMemoryGetMappedBuffer
    if (size == 0) {
        std::cerr << "[CudaCore] MapImportedResource: size=0 is invalid! Caller must provide buffer size." << std::endl;
        return nullptr;
    }
    
    std::cerr << "[CudaCore] MapImportedResource: offset=" << offset << " size=" << size << std::endl;

    void* devPtr = nullptr;
    cudaExternalMemoryBufferDesc desc = {};
    desc.offset = offset;
    desc.size = size;
    desc.flags = 0;

    cudaError_t err = cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &desc);
    if (err != cudaSuccess) {
        std::cerr << "[CudaCore] Failed to map buffer: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    std::cerr << "[CudaCore] MapImportedResource: Success, devPtr=" << devPtr << std::endl;
    return devPtr;
}

void CudaCore::UnmapImportedResource(void* devPtr) {
    // IMPORTANT: External memory mapped pointers via cudaExternalMemoryGetMappedBuffer
    // CANNOT be freed with cudaFree! They are valid as long as the external memory handle exists.
    // This function is now a no-op - just clears the caller's copy of the pointer.
    // The memory is freed when the external memory handle is destroyed via FreeImportedResource.
    (void)devPtr; // Suppress unused warning
    // std::cerr << "[CudaCore] UnmapImportedResource: no-op (external memory)" << std::endl;
}

// ---------------------------------------------------------
// Modern Interop: External Semaphores (Fences)
// ---------------------------------------------------------

cudaExternalSemaphore_t CudaCore::ImportD3D12Fence(ID3D12Fence* d3dFence) {
    if (!d3dFence) return nullptr;

    cudaExternalSemaphoreHandleDesc desc = {};
    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.handle = d3dFence; // Pass pointer directly
    desc.flags = 0;

    cudaExternalSemaphore_t extSem = nullptr;
    cudaError_t err = cudaImportExternalSemaphore(&extSem, &desc);
    if (err != cudaSuccess) {
        std::cerr << "[CudaCore] Failed to import fence: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return extSem;
}

void CudaCore::FreeImportedFence(cudaExternalSemaphore_t extSem) {
    if (extSem) {
        CUDA_CHECK(cudaDestroyExternalSemaphore(extSem));
    }
}

void CudaCore::SignalSemaphore(cudaExternalSemaphore_t extSem, uint64_t value, cudaStream_t stream) {
    if (!extSem) return;
    cudaExternalSemaphoreSignalParams params = {};
    params.params.fence.value = value;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream));
}

void CudaCore::WaitSemaphore(cudaExternalSemaphore_t extSem, uint64_t value, cudaStream_t stream) {
    if (!extSem) return;
    cudaExternalSemaphoreWaitParams params = {};
    params.params.fence.value = value;
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream));
}

// ---------------------------------------------------------
// Legacy (Stubbed/Disabled)
// ---------------------------------------------------------

void CudaCore::RegisterResource(ID3D12Resource* d3dResource, cudaGraphicsResource_t* cudaResource, cudaGraphicsRegisterFlags flags) {
    std::cerr << "[CudaCore] DEPRECATED RegisterResource called. Use ImportD3D12Resource instead." << std::endl;
}

void CudaCore::UnregisterResource(cudaGraphicsResource_t cudaResource) {
    // No-op
}

void* CudaCore::MapResource(cudaGraphicsResource_t cudaResource, size_t& outSize, cudaStream_t stream) {
    std::cerr << "[CudaCore] DEPRECATED MapResource called." << std::endl;
    return nullptr;
}

void CudaCore::UnmapResource(cudaGraphicsResource_t cudaResource, cudaStream_t stream) {
    // No-op
}

void CudaCore::CheckCudaError(cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        std::cerr << "[CudaCore] Error in " << context << ": " << cudaGetErrorString(error) << std::endl;
    }
}

} // namespace Core
} // namespace CudaGame
