#include "Core/CudaCore.h"
#include <string>

namespace CudaGame {
namespace Core {

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

    // Pick first device for now
    m_cudaDeviceID = 0;
    CheckCudaError(cudaSetDevice(m_cudaDeviceID), "cudaSetDevice");
    
    // Print device info
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, m_cudaDeviceID);
    std::cout << "[CudaCore] Initialized Device: " << props.name << " (CC " << props.major << "." << props.minor << ")" << std::endl;

    m_initialized = true;
    return true;
}

void CudaCore::Shutdown() {
    if (!m_initialized) return;
    
    // Ensure all work is done
    cudaDeviceSynchronize();
    
    // Explicit CUDA reset is usually not needed if we want to keep context, 
    // but good for full shutdown.
    // cudaDeviceReset();
    
    m_initialized = false;
    std::cout << "[CudaCore] Shutdown complete." << std::endl;
}

void CudaCore::RegisterResource(ID3D12Resource* d3dResource, cudaGraphicsResource_t* cudaResource, cudaGraphicsRegisterFlags flags) {
    if (!m_initialized) return;
    
    //cudaError_t err = cudaGraphicsD3D12RegisterResource(cudaResource, d3dResource, flags);
    //CheckCudaError(err, "cudaGraphicsD3D12RegisterResource");
    
    //if (err == cudaSuccess) {
    //    std::cout << "[CudaCore] Registered DX12 Resource with CUDA." << std::endl;
    //}
    std::cout << "[CudaCore] Resource registration DISABLED (Header missing)" << std::endl;
}

void CudaCore::UnregisterResource(cudaGraphicsResource_t cudaResource) {
    if (!cudaResource) return;
    CheckCudaError(cudaGraphicsUnregisterResource(cudaResource), "cudaGraphicsUnregisterResource");
}

void* CudaCore::MapResource(cudaGraphicsResource_t cudaResource, size_t& outSize, cudaStream_t stream) {
    if (!cudaResource) return nullptr;

    // 1. Map
    CheckCudaError(cudaGraphicsMapResources(1, &cudaResource, stream), "cudaGraphicsMapResources");
    
    // 2. Get Pointer
    void* devPtr = nullptr;
    size_t size = 0;
    
    cudaError_t err = cudaGraphicsResourceGetMappedPointer(&devPtr, &size, cudaResource);
    CheckCudaError(err, "cudaGraphicsResourceGetMappedPointer");
    
    outSize = size;
    return devPtr;
}

void CudaCore::UnmapResource(cudaGraphicsResource_t cudaResource, cudaStream_t stream) {
    if (!cudaResource) return;
    CheckCudaError(cudaGraphicsUnmapResources(1, &cudaResource, stream), "cudaGraphicsUnmapResources");
}

void CudaCore::CheckCudaError(cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        std::cerr << "[CudaCore] Error in " << context << ": " 
                  << cudaGetErrorString(error) << " (" << error << ")" << std::endl;
    }
}

} // namespace Core
} // namespace CudaGame
