#pragma once

#pragma once

#include "VFX/ParticleStruct.h" // Safe for CUDA

namespace CudaGame {
namespace VFX {

// Kernel Wrappers
void LaunchParticleUpdate(void* d_particles, int count, float deltaTime);
// Use POD types to ensure no GLM in header that NVCC sees
void LaunchParticleSpawn(void* d_particles, int maxCount, int* activeCountPtr, float3_pod pos, int spawnCount, float4_pod color);

} // namespace VFX
} // namespace CudaGame
