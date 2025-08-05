#pragma once

#include <cstdint>

namespace CudaGame {
namespace Rendering {

// CUDA-specific data structure (PIMPL pattern)
struct CudaRenderingData {
    // CUDA textures and resources for various effects
    uint32_t ssaoNoiseTexture = 0;
    uint32_t bloomTextures[2] = {0, 0};
    
    // CUDA events for profiling
    void* startEvent = nullptr;
    void* stopEvent = nullptr;
    
    bool isInitialized = false;
};

} // namespace Rendering
} // namespace CudaGame

