#include "Rendering/CullAndDraw.h"
#include <iostream>

extern "C" void LaunchCullAndDrawKernel(
    const void* objects,
    void* commands,
    unsigned int* drawCounter,
    int objectCount,
    const float* frustumPlanes,
    cudaStream_t stream
) {
    static bool printed = false;
    if (!printed) {
        std::cout << "[CullAndDraw] WARNING: Running Stubbed Kernel (CUDA compilation disabled)." << std::endl;
        printed = true;
    }
    // No-op: Output commands will be empty.
    // If we wanted to fallback, we'd need to map the buffer on CPU, which might not be possible if it's default heap.
}
