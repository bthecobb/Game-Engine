#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA-equivalent data structures
struct float3 {
    float x, y, z;
};

enum ColliderType { BOX = 0, SPHERE = 1, CAPSULE = 2 };

struct CudaCollider {
    ColliderType type;
    float3 center;
    float3 halfExtents; // For box
    float radius;       // For sphere/capsule
    float height;       // For capsule
};

struct CollisionInfo {
    int entityA;
    int entityB;
    float3 normal;
    float penetration;
    bool hasCollision;
};

// Device functions for collision detection
__device__ bool sphereVsSphere(const float3& posA, float radiusA, const float3& posB, float radiusB, CollisionInfo& info) {
    float3 diff = {posB.x - posA.x, posB.y - posA.y, posB.z - posA.z};
    float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    float radiusSum = radiusA + radiusB;
    
    if (distSq < radiusSum * radiusSum) {
        float dist = sqrtf(distSq);
        info.penetration = radiusSum - dist;
        
        if (dist > 0.0f) {
            info.normal.x = diff.x / dist;
            info.normal.y = diff.y / dist;
            info.normal.z = diff.z / dist;
        } else {
            info.normal = {1.0f, 0.0f, 0.0f}; // Default normal
        }
        
        info.hasCollision = true;
        return true;
    }
    
    info.hasCollision = false;
    return false;
}

__device__ bool boxVsBox(const float3& posA, const float3& halfExtentsA, const float3& posB, const float3& halfExtentsB, CollisionInfo& info) {
    float3 diff = {fabsf(posB.x - posA.x), fabsf(posB.y - posA.y), fabsf(posB.z - posA.z)};
    float3 overlap = {
        halfExtentsA.x + halfExtentsB.x - diff.x,
        halfExtentsA.y + halfExtentsB.y - diff.y,
        halfExtentsA.z + halfExtentsB.z - diff.z
    };
    
    if (overlap.x > 0.0f && overlap.y > 0.0f && overlap.z > 0.0f) {
        // Find minimum overlap axis
        if (overlap.x <= overlap.y && overlap.x <= overlap.z) {
            info.normal = {(posB.x > posA.x) ? 1.0f : -1.0f, 0.0f, 0.0f};
            info.penetration = overlap.x;
        } else if (overlap.y <= overlap.z) {
            info.normal = {0.0f, (posB.y > posA.y) ? 1.0f : -1.0f, 0.0f};
            info.penetration = overlap.y;
        } else {
            info.normal = {0.0f, 0.0f, (posB.z > posA.z) ? 1.0f : -1.0f};
            info.penetration = overlap.z;
        }
        
        info.hasCollision = true;
        return true;
    }
    
    info.hasCollision = false;
    return false;
}

// Broad-phase collision detection kernel (using spatial grid)
__global__ void broadPhaseCollisionDetection(const float3* positions, const CudaCollider* colliders, 
                                            CollisionInfo* collisions, int numEntities, int maxCollisions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numEntities) return;
    
    int collisionCount = 0;
    
    // Test against all other entities (O(n²) - would use spatial partitioning in production)
    for (int other = idx + 1; other < numEntities && collisionCount < maxCollisions; ++other) {
        CollisionInfo info;
        info.entityA = idx;
        info.entityB = other;
        info.hasCollision = false;
        
        const float3& posA = positions[idx];
        const float3& posB = positions[other];
        const CudaCollider& colliderA = colliders[idx];
        const CudaCollider& colliderB = colliders[other];
        
        // Perform collision detection based on collider types
        if (colliderA.type == SPHERE && colliderB.type == SPHERE) {
            sphereVsSphere(posA, colliderA.radius, posB, colliderB.radius, info);
        } else if (colliderA.type == BOX && colliderB.type == BOX) {
            boxVsBox(posA, colliderA.halfExtents, posB, colliderB.halfExtents, info);
        }
        // Add more collision type combinations as needed
        
        if (info.hasCollision) {
            // Store collision in global memory (use atomic operations for thread safety)
            int collisionIndex = atomicAdd(&collisionCount, 1);
            if (collisionIndex < maxCollisions) {
                collisions[idx * maxCollisions + collisionIndex] = info;
            }
        }
    }
}

// Collision response kernel
__global__ void resolveCollisions(float3* positions, float3* velocities, const float* masses,
                                 const CollisionInfo* collisions, int numEntities, int maxCollisions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numEntities) return;
    
    // Process collisions for this entity
    for (int i = 0; i < maxCollisions; ++i) {
        const CollisionInfo& collision = collisions[idx * maxCollisions + i];
        
        if (!collision.hasCollision) break;
        
        int otherIdx = (collision.entityA == idx) ? collision.entityB : collision.entityA;
        
        // Simple position correction (separate objects)
        float massA = masses[idx];
        float massB = masses[otherIdx];
        float totalMass = massA + massB;
        
        if (totalMass > 0.0f) {
            float correctionA = massB / totalMass;
            float correctionB = massA / totalMass;
            
            float3 correction = {
                collision.normal.x * collision.penetration * 0.5f,
                collision.normal.y * collision.penetration * 0.5f,
                collision.normal.z * collision.penetration * 0.5f
            };
            
            if (collision.entityA == idx) {
                positions[idx].x -= correction.x * correctionA;
                positions[idx].y -= correction.y * correctionA;
                positions[idx].z -= correction.z * correctionA;
            } else {
                positions[idx].x += correction.x * correctionB;
                positions[idx].y += correction.y * correctionB;
                positions[idx].z += correction.z * correctionB;
            }
        }
    }
}

// Wrapper functions for launching kernels from C++
extern "C" void launch_broadPhaseCollisionDetection(const float3* positions, const CudaCollider* colliders,
                                                   CollisionInfo* collisions, int numEntities, int maxCollisions) {
    int blockSize = 256;
    int numBlocks = (numEntities + blockSize - 1) / blockSize;
    broadPhaseCollisionDetection<<<numBlocks, blockSize>>>(positions, colliders, collisions, numEntities, maxCollisions);
}

extern "C" void launch_resolveCollisions(float3* positions, float3* velocities, const float* masses,
                                        const CollisionInfo* collisions, int numEntities, int maxCollisions) {
    int blockSize = 256;
    int numBlocks = (numEntities + blockSize - 1) / blockSize;
    resolveCollisions<<<numBlocks, blockSize>>>(positions, velocities, masses, collisions, numEntities, maxCollisions);
}
