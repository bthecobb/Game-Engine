#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// SSAO parameters
struct SSAOParameters {
    int sampleCount;
    float radius;
    float bias;
    float power;
};

// Sample kernel for hemisphere sampling
__constant__ float3 d_sampleKernel[32];

// Simple pseudo-random rotation
__device__ inline float3 getRandomVector(int x, int y) {
    uint32_t hash = (x * 73856093) ^ (y * 19349663);
    float angle = float(hash % 628) * 0.01f;  // 0 to 2*PI
    return make_float3(cosf(angle), sinf(angle), 0.0f);
}

__device__ inline float3 normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.001f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0, 1, 0);
}

__device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Simple SSAO kernel
__global__ void ComputeSSAOKernel(
    const float* gBufferPosition,   // XYZ world position
    const float* gBufferNormal,     // XYZ world normal
    const float* gBufferDepth,      // Z depth
    float* ssaoOutput,              // Output: occlusion factor (0-1)
    SSAOParameters params,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    int pixelIdx3 = pixelIdx * 3;
    
    // Read G-buffer data
    float3 fragPos = make_float3(
        gBufferPosition[pixelIdx3 + 0],
        gBufferPosition[pixelIdx3 + 1],
        gBufferPosition[pixelIdx3 + 2]
    );
    
    float3 normal = make_float3(
        gBufferNormal[pixelIdx3 + 0],
        gBufferNormal[pixelIdx3 + 1],
        gBufferNormal[pixelIdx3 + 2]
    );
    
    // Skip background pixels
    if (length(normal) < 0.1f) {
        ssaoOutput[pixelIdx] = 1.0f;  // No occlusion
        return;
    }
    
    normal = normalize(normal);
    
    // Create TBN matrix for sample kernel rotation
    float3 randomVec = getRandomVector(x, y);
    float3 tangent = normalize(cross(randomVec, normal));
    float3 bitangent = cross(normal, tangent);
    
    // Sample hemisphere around point
    float occlusion = 0.0f;
    
    for (int i = 0; i < params.sampleCount; i++) {
        // Get sample direction in tangent space
        float3 sampleDir = d_sampleKernel[i];
        
        // Transform to world space
        float3 samplePos;
        samplePos.x = fragPos.x + (tangent.x * sampleDir.x + bitangent.x * sampleDir.y + normal.x * sampleDir.z) * params.radius;
        samplePos.y = fragPos.y + (tangent.y * sampleDir.x + bitangent.y * sampleDir.y + normal.y * sampleDir.z) * params.radius;
        samplePos.z = fragPos.z + (tangent.z * sampleDir.x + bitangent.z * sampleDir.y + normal.z * sampleDir.z) * params.radius;
        
        // Simple occlusion test: if sample is below surface, it's occluded
        // (Simplified - in production, project to screen space and sample depth)
        float sampleDepth = samplePos.y;
        float surfaceDepth = fragPos.y;
        
        if (sampleDepth < surfaceDepth - params.bias) {
            occlusion += 1.0f;
        }
    }
    
    // Calculate final occlusion
    occlusion = 1.0f - (occlusion / float(params.sampleCount));
    occlusion = powf(fmaxf(occlusion, 0.0f), params.power);
    
    ssaoOutput[pixelIdx] = occlusion;
}

// Blur SSAO to reduce noise
__global__ void BlurSSAOKernel(
    const float* ssaoInput,
    float* ssaoOutput,
    int width,
    int height,
    int blurRadius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int dy = -blurRadius; dy <= blurRadius; dy++) {
        for (int dx = -blurRadius; dx <= blurRadius; dx++) {
            int sx = x + dx;
            int sy = y + dy;
            
            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                sum += ssaoInput[sy * width + sx];
                count++;
            }
        }
    }
    
    ssaoOutput[y * width + x] = sum / float(count);
}

// C-linkage wrapper functions
extern "C" {

void LaunchSSAOKernel(
    const float* gBufferPosition,
    const float* gBufferNormal,
    const float* gBufferDepth,
    float* ssaoOutput,
    const SSAOParameters& params,
    int width,
    int height
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    ComputeSSAOKernel<<<gridSize, blockSize>>>(
        gBufferPosition,
        gBufferNormal,
        gBufferDepth,
        ssaoOutput,
        params,
        width,
        height
    );
    
    cudaDeviceSynchronize();
}

void LaunchSSAOBlurKernel(
    const float* ssaoInput,
    float* ssaoOutput,
    int width,
    int height,
    int blurRadius
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    BlurSSAOKernel<<<gridSize, blockSize>>>(
        ssaoInput,
        ssaoOutput,
        width,
        height,
        blurRadius
    );
    
    cudaDeviceSynchronize();
}

void InitializeSSAOSampleKernel(const float* samples, int count) {
    cudaMemcpyToSymbol(d_sampleKernel, samples, count * sizeof(float3));
}

} // extern "C"
