#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

// Kernel to compute Screen-Space Ambient Occlusion (SSAO)
__global__ void computeSSAO(cudaTextureObject_t positionTexture, cudaTextureObject_t normalTexture, 
                          cudaSurfaceObject_t outputSurface, 
                          int width, int height, const float* sampleKernel, 
                          float radius, float bias, int sampleCount) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Get position and normal from G-buffer
    float4 position = tex2D<float4>(positionTexture, x, y);
    float3 normal = make_float3(tex2D<float4>(normalTexture, x, y));

    float occlusion = 0.0f;
    for (int i = 0; i < sampleCount; ++i) {
        // Get sample direction
        float3 sampleDir = make_float3(sampleKernel[i * 3], sampleKernel[i * 3 + 1], sampleKernel[i * 3 + 2]);
        
        // Create sample point in tangent space
        float3 samplePoint = make_float3(position.x + sampleDir.x * radius, 
                                       position.y + sampleDir.y * radius, 
                                       position.z + sampleDir.z * radius);
        
        // Project sample point onto screen
        // TODO: This would require projection matrix and viewport transform
        // For now, assume a simple orthographic projection
        float2 screenPos = make_float2(samplePoint.x, samplePoint.y);
        
        // Get depth of sample point from depth buffer
        float sampleDepth = tex2D<float4>(positionTexture, screenPos.x, screenPos.y).z;
        
        // Check if sample point is occluded
        if (sampleDepth < samplePoint.z - bias) {
            occlusion += 1.0f;
        }
    }

    occlusion = 1.0f - (occlusion / sampleCount);

    // Write occlusion value to output texture
    surf2Dwrite(make_float4(occlusion, occlusion, occlusion, 1.0f), outputSurface, x * sizeof(float4), y);
}

// Wrapper function to launch the SSAO kernel
extern "C" void launch_computeSSAO(cudaTextureObject_t positionTexture, cudaTextureObject_t normalTexture,
                                   cudaSurfaceObject_t outputSurface, 
                                   int width, int height, const float* sampleKernel, 
                                   float radius, float bias, int sampleCount) {
    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    computeSSAO<<<numBlocks, blockSize>>>(positionTexture, normalTexture, outputSurface, 
                                         width, height, sampleKernel, radius, bias, sampleCount);
}
