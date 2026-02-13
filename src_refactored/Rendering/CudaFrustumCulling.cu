// CudaFrustumCulling.cu - GPU-accelerated frustum culling kernel
// Tests object AABBs against camera frustum planes in parallel

#include <cuda_runtime.h>
#include <cstdint>

// AABB structure for GPU
struct CudaAABB {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};

// Frustum plane (ax + by + cz + d = 0)
struct CudaPlane {
    float a, b, c, d;
};

// Frustum with 6 planes (left, right, bottom, top, near, far)
struct CudaFrustum {
    CudaPlane planes[6];
};

// Test if AABB intersects frustum plane
__device__ inline bool TestPlane(const CudaPlane& plane, const CudaAABB& aabb) {
    // Find the point on AABB most aligned with plane normal
    float px = (plane.a > 0) ? aabb.maxX : aabb.minX;
    float py = (plane.b > 0) ? aabb.maxY : aabb.minY;
    float pz = (plane.c > 0) ? aabb.maxZ : aabb.minZ;
    
    // Test if this point is on positive side of plane
    float dist = plane.a * px + plane.b * py + plane.c * pz + plane.d;
    return dist >= 0.0f;
}

// Test if AABB is visible (intersects frustum)
__device__ inline bool IsVisible(const CudaFrustum& frustum, const CudaAABB& aabb) {
    // Test against all 6 frustum planes
    for (int i = 0; i < 6; ++i) {
        if (!TestPlane(frustum.planes[i], aabb)) {
            return false;  // AABB is completely outside this plane
        }
    }
    return true;  // AABB intersects or is inside frustum
}

// ====== MAIN CULLING KERNEL ======
// Each thread processes one object
// Input: objectBounds - array of AABBs for all objects
// Input: frustum - camera frustum (6 planes)
// Output: visibilityBits - 1 bit per object (1 = visible, 0 = culled)
__global__ void FrustumCullKernel(
    const CudaAABB* __restrict__ objectBounds,
    const CudaFrustum* __restrict__ frustum,
    uint32_t* __restrict__ visibilityBits,
    int numObjects
) {
    int objectId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (objectId >= numObjects) return;
    
    // Test visibility
    bool visible = IsVisible(*frustum, objectBounds[objectId]);
    
    // Write to visibility buffer (atomic to handle bit packing)
    int wordIndex = objectId / 32;
    int bitIndex = objectId % 32;
    
    if (visible) {
        atomicOr(&visibilityBits[wordIndex], 1u << bitIndex);
    }
}

// ====== COMPACT VISIBLE INDICES KERNEL ======
// Generates list of visible object indices for draw calls
__global__ void CompactVisibleKernel(
    const uint32_t* __restrict__ visibilityBits,
    int* __restrict__ visibleIndices,
    int* __restrict__ visibleCount,
    int numObjects
) {
    int objectId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (objectId >= numObjects) return;
    
    int wordIndex = objectId / 32;
    int bitIndex = objectId % 32;
    
    if (visibilityBits[wordIndex] & (1u << bitIndex)) {
        int outputIndex = atomicAdd(visibleCount, 1);
        visibleIndices[outputIndex] = objectId;
    }
}

// ====== HOST WRAPPER FUNCTIONS ======
extern "C" {

// Allocate GPU buffers for culling
cudaError_t AllocateCullingBuffers(
    void** d_objectBounds,
    void** d_frustum,
    void** d_visibilityBits,
    void** d_visibleIndices,
    void** d_visibleCount,
    int maxObjects
) {
    cudaError_t err;
    
    err = cudaMalloc(d_objectBounds, maxObjects * sizeof(CudaAABB));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(d_frustum, sizeof(CudaFrustum));
    if (err != cudaSuccess) return err;
    
    int numWords = (maxObjects + 31) / 32;
    err = cudaMalloc(d_visibilityBits, numWords * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(d_visibleIndices, maxObjects * sizeof(int));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(d_visibleCount, sizeof(int));
    if (err != cudaSuccess) return err;
    
    return cudaSuccess;
}

// Free GPU buffers
void FreeCullingBuffers(
    void* d_objectBounds,
    void* d_frustum,
    void* d_visibilityBits,
    void* d_visibleIndices,
    void* d_visibleCount
) {
    cudaFree(d_objectBounds);
    cudaFree(d_frustum);
    cudaFree(d_visibilityBits);
    cudaFree(d_visibleIndices);
    cudaFree(d_visibleCount);
}

// Upload object bounds to GPU
cudaError_t UploadObjectBounds(void* d_objectBounds, const void* h_bounds, int numObjects) {
    return cudaMemcpy(d_objectBounds, h_bounds, numObjects * sizeof(CudaAABB), cudaMemcpyHostToDevice);
}

// Upload frustum planes to GPU
cudaError_t UploadFrustum(void* d_frustum, const void* h_frustum) {
    return cudaMemcpy(d_frustum, h_frustum, sizeof(CudaFrustum), cudaMemcpyHostToDevice);
}

// Execute frustum culling
cudaError_t ExecuteFrustumCulling(
    void* d_objectBounds,
    void* d_frustum,
    void* d_visibilityBits,
    int numObjects
) {
    // Clear visibility bits
    int numWords = (numObjects + 31) / 32;
    cudaMemset(d_visibilityBits, 0, numWords * sizeof(uint32_t));
    
    // Launch kernel (256 threads per block)
    int threadsPerBlock = 256;
    int numBlocks = (numObjects + threadsPerBlock - 1) / threadsPerBlock;
    
    FrustumCullKernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<const CudaAABB*>(d_objectBounds),
        reinterpret_cast<const CudaFrustum*>(d_frustum),
        reinterpret_cast<uint32_t*>(d_visibilityBits),
        numObjects
    );
    
    return cudaGetLastError();
}

// Get visible indices
cudaError_t GetVisibleIndices(
    void* d_visibilityBits,
    void* d_visibleIndices,
    void* d_visibleCount,
    int* h_visibleIndices,
    int* h_visibleCount,
    int numObjects
) {
    // Clear count
    cudaMemset(d_visibleCount, 0, sizeof(int));
    
    // Launch compact kernel
    int threadsPerBlock = 256;
    int numBlocks = (numObjects + threadsPerBlock - 1) / threadsPerBlock;
    
    CompactVisibleKernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<const uint32_t*>(d_visibilityBits),
        reinterpret_cast<int*>(d_visibleIndices),
        reinterpret_cast<int*>(d_visibleCount),
        numObjects
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    // Copy results back to host
    err = cudaMemcpy(h_visibleCount, d_visibleCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_visibleIndices, d_visibleIndices, (*h_visibleCount) * sizeof(int), cudaMemcpyDeviceToHost);
    return err;
}

} // extern "C"
