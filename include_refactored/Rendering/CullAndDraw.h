#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <driver_types.h> // For cudaStream_t

namespace CudaGame {
namespace Rendering {

// Layout must exactly match CullAndDraw.cu
struct ObjectCullingData {
    // Bounding Sphere (Local Space)
    glm::vec3 sphereCenter;
    float sphereRadius;
    
    // Geometry (VBV)
    uint64_t vbv_loc;    // GPU Address
    uint32_t vbv_size;
    uint32_t vbv_stride;
    
    // Geometry (IBV)
    uint64_t ibv_loc;    // GPU Address
    uint32_t ibv_size;
    uint32_t ibv_format;
    
    // Constants
    uint64_t cbv;
    uint64_t materialCbv;
    
    // Transform
    glm::mat4 worldMatrix; // Added for self-contained culling data
    
    // Draw Info
    uint32_t indexCount;
};

} // namespace Rendering
} // namespace CudaGame

// C-Linkage for CUDA Kernel Wrapper
extern "C" void LaunchCullAndDrawKernel(
    const void* objects,         // ObjectCullingData*
    void* commands,              // IndirectCommand*
    unsigned int* drawCounter,   // uint32_t*
    int objectCount,
    const float* frustumPlanes,  // 6 * 4 floats
    cudaStream_t stream
);
