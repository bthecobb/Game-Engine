#include "VFX/ParticleKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h> // For randomness

namespace CudaGame {
namespace VFX {

// Device helper for random
__device__ float rand_float(unsigned int& seed) {
    seed = (seed * 1664525u + 1013904223u);
    return (float)(seed & 0xFFFFFF) / 16777216.0f;
}

// Reuse Particle struct definition from header (ensure alignment matches C++)
// We include "ParticleSystem.h" in header so it's visible.

__global__ void UpdateParticlesKernel(Particle* particles, int count, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle& p = particles[idx];
    
    if (p.life > 0.0f) {
        // Simple Physics
        p.velocity.y -= 9.8f * deltaTime; // Gravity
        
        p.position.x += p.velocity.x * deltaTime;
        p.position.y += p.velocity.y * deltaTime;
        p.position.z += p.velocity.z * deltaTime;
        
        p.life -= deltaTime;
        
        // Floor interaction
        if (p.position.y < 0.0f) {
            p.position.y = 0.0f;
            p.velocity.y *= -0.5f; // Bounce
            p.velocity.x *= 0.9f;  // Friction
            p.velocity.z *= 0.9f;
        }
        
        // Fade alpha
        p.color.w = fminf(1.0f, p.life * 2.0f); // w is alpha
    } else {
        // Reset or just stay dead
        p.size = 0.0f;
    }
}

void LaunchParticleUpdate(void* d_particles, int count, float deltaTime) {
    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;
    
    UpdateParticlesKernel<<<numBlocks, blockSize>>>((Particle*)d_particles, count, deltaTime);
}

__global__ void SpawnParticlesKernel(Particle* particles, int maxCount, int startIndex, float3 pos, int count, float4 color) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int globalIdx = (startIndex + idx) % maxCount; // Ring buffer style
    
    Particle& p = particles[globalIdx];
    p.life = 2.0f + 0.5f * (float)(idx % 10) / 10.0f; // Varied life
    
    p.position.x = pos.x;
    p.position.y = pos.y;
    p.position.z = pos.z;
    
    p.size = 0.1f;
    p.color.x = color.x;
    p.color.y = color.y;
    p.color.z = color.z;
    p.color.w = color.w;
    
    // Random velocity cone
    unsigned int seed = idx + 12345;
    float rx = rand_float(seed) * 2.0f - 1.0f;
    float rz = rand_float(seed) * 2.0f - 1.0f;
    float ry = rand_float(seed) * 1.5f + 2.0f; // Upward
    
    p.velocity.x = rx * 2.0f;
    p.velocity.y = ry * 2.0f; 
    p.velocity.z = rz * 2.0f;
}

void LaunchParticleSpawn(void* d_particles, int maxCount, int* activeCountPtr, float3_pod pos, int spawnCount, float4_pod color) {
    // Cast POD to CUDA float3
    float3 p = make_float3(pos.x, pos.y, pos.z);
    float4 c = make_float4(color.x, color.y, color.z, color.w);
    
    int blockSize = 256;
    int numBlocks = (spawnCount + blockSize - 1) / blockSize;
    
    SpawnParticlesKernel<<<numBlocks, blockSize>>>((Particle*)d_particles, maxCount, 0, p, spawnCount, c);
}

} // namespace VFX
} // namespace CudaGame
