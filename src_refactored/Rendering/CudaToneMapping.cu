#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

// Kernel for tone mapping and color grading
__global__ void toneMappingAndColorGrading(cudaTextureObject_t inputTexture, cudaSurfaceObject_t outputSurface,
                                       int width, int height, float exposure, float gamma, 
                                       float saturation, float contrast, float brightness) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Read HDR color from input texture
    float4 hdrColor = tex2D<float4>(inputTexture, x, y);

    // Apply exposure
    hdrColor.x *= exposure;
    hdrColor.y *= exposure;
    hdrColor.z *= exposure;

    // Tone mapping (Reinhard operator)
    float3 mappedColor = {
        hdrColor.x / (hdrColor.x + 1.0f),
        hdrColor.y / (hdrColor.y + 1.0f),
        hdrColor.z / (hdrColor.z + 1.0f)
    };

    // Color grading
    // Saturation
    float luma = dot(make_float3(mappedColor.x, mappedColor.y, mappedColor.z), make_float3(0.2126f, 0.7152f, 0.0722f));
    mappedColor.x = luma + saturation * (mappedColor.x - luma);
    mappedColor.y = luma + saturation * (mappedColor.y - luma);
    mappedColor.z = luma + saturation * (mappedColor.z - luma);
    
    // Contrast
    mappedColor.x = (mappedColor.x - 0.5f) * contrast + 0.5f;
    mappedColor.y = (mappedColor.y - 0.5f) * contrast + 0.5f;
    mappedColor.z = (mappedColor.z - 0.5f) * contrast + 0.5f;
    
    // Brightness
    mappedColor.x += brightness;
    mappedColor.y += brightness;
    mappedColor.z += brightness;
    
    // Gamma correction
    float invGamma = 1.0f / gamma;
    mappedColor.x = powf(mappedColor.x, invGamma);
    mappedColor.y = powf(mappedColor.y, invGamma);
    mappedColor.z = powf(mappedColor.z, invGamma);

    // Write final LDR color to output surface
    float4 ldrColor = make_float4(mappedColor.x, mappedColor.y, mappedColor.z, 1.0f);
    surf2Dwrite(ldrColor, outputSurface, x * sizeof(float4), y);
}

// Wrapper function for launching the tone mapping kernel
extern "C" void launch_toneMappingAndColorGrading(cudaTextureObject_t inputTexture, cudaSurfaceObject_t outputSurface,
                                                int width, int height, float exposure, float gamma,
                                                float saturation, float contrast, float brightness) {
    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    toneMappingAndColorGrading<<<numBlocks, blockSize>>>(inputTexture, outputSurface, width, height, exposure, gamma, 
                                                      saturation, contrast, brightness);
}
