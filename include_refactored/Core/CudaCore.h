#pragma once

#include <d3d12.h>
#include <cuda_runtime.h>
//#include <cuda_d3d12_interop.h>
#include <vector>
#include <iostream>

namespace CudaGame {
namespace Core {

class CudaCore {
public:
    CudaCore();
    ~CudaCore();

    bool Initialize(ID3D12Device* d3dDevice);
    void Shutdown();

    // Resource Registration (DX12 -> CUDA)
    void RegisterResource(ID3D12Resource* d3dResource, cudaGraphicsResource_t* cudaResource, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    void UnregisterResource(cudaGraphicsResource_t cudaResource);

    // Mapping for Write
    void* MapResource(cudaGraphicsResource_t cudaResource, size_t& outSize, cudaStream_t stream = 0);
    void UnmapResource(cudaGraphicsResource_t cudaResource, cudaStream_t stream = 0);

    // Helper
    static void CheckCudaError(cudaError_t error, const char* context);

private:
    bool m_initialized = false;
    int m_cudaDeviceID = -1;
    
    // We might need to store the D3D device if we do complex interop setup
    ID3D12Device* m_d3dDevice = nullptr;
};

} // namespace Core
} // namespace CudaGame
