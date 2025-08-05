#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel() {
    printf("Hello from CUDA kernel!\n");
}

int main() {
    std::cout << "Testing CUDA..." << std::endl;
    
    // Check CUDA device
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        
        // Launch simple kernel
        simpleKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
    
    std::cout << "CUDA test completed!" << std::endl;
    return 0;
}
