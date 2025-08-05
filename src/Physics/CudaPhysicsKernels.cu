#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA-equivalent of glm::vec3 for device code
struct float3 {
    float x, y, z;
};

// Kernel to apply forces and update velocities (Euler integration)
__global__ void integrateForces(float3* velocities, const float3* forces, const float* masses, int numEntities, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numEntities) {
        // Apply gravity (assuming constant gravity for now)
        float3 gravity = {0.0f, -9.81f, 0.0f};
        
        // F = ma -> a = F/m
        float mass = masses[idx];
        if (mass > 0.0f) {
            float3 acceleration = {
                forces[idx].x / mass + gravity.x,
                forces[idx].y / mass + gravity.y,
                forces[idx].z / mass + gravity.z
            };
            
            // v = v0 + at
            velocities[idx].x += acceleration.x * dt;
            velocities[idx].y += acceleration.y * dt;
            velocities[idx].z += acceleration.z * dt;
        }
    }
}

// Kernel to update positions based on velocities
__global__ void updatePositions(float3* positions, const float3* velocities, int numEntities, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numEntities) {
        // p = p0 + vt
        positions[idx].x += velocities[idx].x * dt;
        positions[idx].y += velocities[idx].y * dt;
        positions[idx].z += velocities[idx].z * dt;
    }
}

// Wrapper functions to be called from the CudaPhysicsSystem
extern "C" void launch_integrateForces(float3* velocities, const float3* forces, const float* masses, int numEntities, float dt) {
    int blockSize = 256;
    int numBlocks = (numEntities + blockSize - 1) / blockSize;
    integrateForces<<<numBlocks, blockSize>>>(velocities, forces, masses, numEntities, dt);
}

extern "C" void launch_updatePositions(float3* positions, const float3* velocities, int numEntities, float dt) {
    int blockSize = 256;
    int numBlocks = (numEntities + blockSize - 1) / blockSize;
    updatePositions<<<numBlocks, blockSize>>>(positions, velocities, numEntities, dt);
}
