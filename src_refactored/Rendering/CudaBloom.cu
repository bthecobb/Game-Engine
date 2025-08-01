#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

// Kernel to extract bright areas from an image (thresholding)
__global__ void extractBrightAreas(cudaTextureObject_t inputTexture, cudaSurfaceObject_t outputSurface,
                                 int width, int height, float threshold) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float4 pixel = tex2D<float4>(inputTexture, x, y);
    float brightness = dot(make_float3(pixel.x, pixel.y, pixel.z), make_float3(0.2126, 0.7152, 0.0722));

    if (brightness > threshold) {
        surf2Dwrite(pixel, outputSurface, x * sizeof(float4), y);
    } else {
        surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, 1.0f), outputSurface, x * sizeof(float4), y);
    }
}

// Separable Gaussian blur kernel (horizontal pass)
__global__ void gaussianBlurHorizontal(cudaTextureObject_t inputTexture, cudaSurfaceObject_t outputSurface,
                                     int width, int height, const float* blurKernel, int kernelSize) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int kernelRadius = kernelSize / 2;

    for (int i = 0; i < kernelSize; ++i) {
        int sampleX = x + i - kernelRadius;
        if (sampleX >= 0 && sampleX < width) {
            float4 pixel = tex2D<float4>(inputTexture, sampleX, y);
            float weight = blurKernel[i];
            sum.x += pixel.x * weight;
            sum.y += pixel.y * weight;
            sum.z += pixel.z * weight;
            sum.w += pixel.w * weight;
        }
    }

    surf2Dwrite(sum, outputSurface, x * sizeof(float4), y);
}

// Separable Gaussian blur kernel (vertical pass)
__global__ void gaussianBlurVertical(cudaTextureObject_t inputTexture, cudaSurfaceObject_t outputSurface,
                                   int width, int height, const float* blurKernel, int kernelSize) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int kernelRadius = kernelSize / 2;

    for (int i = 0; i < kernelSize; ++i) {
        int sampleY = y + i - kernelRadius;
        if (sampleY >= 0 && sampleY < height) {
            float4 pixel = tex2D<float4>(inputTexture, x, sampleY);
            float weight = blurKernel[i];
            sum.x += pixel.x * weight;
            sum.y += pixel.y * weight;
            sum.z += pixel.z * weight;
            sum.w += pixel.w * weight;
        }
    }

    surf2Dwrite(sum, outputSurface, x * sizeof(float4), y);
}

// Kernel to blend two textures together (additive blending)
__global__ void blendTextures(cudaTextureObject_t textureA, cudaTextureObject_t textureB,
                          cudaSurfaceObject_t outputSurface, int width, int height, float intensity) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float4 pixelA = tex2D<float4>(textureA, x, y);
    float4 pixelB = tex2D<float4>(textureB, x, y);
    
    pixelA.x += pixelB.x * intensity;
    pixelA.y += pixelB.y * intensity;
    pixelA.z += pixelB.z * intensity;
    
    surf2Dwrite(pixelA, outputSurface, x * sizeof(float4), y);
}

// Wrapper functions for launching bloom kernels
extern "C" void launch_extractBrightAreas(cudaTextureObject_t inputTexture, cudaSurfaceObject_t outputSurface,
                                        int width, int height, float threshold) {
    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    extractBrightAreas<<<numBlocks, blockSize>>>(inputTexture, outputSurface, width, height, threshold);
}

extern "C" void launch_gaussianBlur(cudaTextureObject_t inputTexture, cudaSurfaceObject_t tempSurface, 
                                  cudaSurfaceObject_t outputSurface, int width, int height, 
                                  const float* blurKernel, int kernelSize) {
    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Horizontal pass
    gaussianBlurHorizontal<<<numBlocks, blockSize>>>(inputTexture, tempSurface, width, height, blurKernel, kernelSize);
    
    // Vertical pass
    gaussianBlurVertical<<<numBlocks, blockSize>>>(inputTexture, outputSurface, width, height, blurKernel, kernelSize);
}

extern "C" void launch_blendTextures(cudaTextureObject_t textureA, cudaTextureObject_t textureB,
                                   cudaSurfaceObject_t outputSurface, int width, int height, float intensity) {
    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    blendTextures<<<numBlocks, blockSize>>>(textureA, textureB, outputSurface, width, height, intensity);
}
