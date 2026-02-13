#pragma once

// CUDA Frustum Culling - GPU-accelerated visibility testing
// Tests object AABBs against camera frustum planes in parallel

#include <cstdint>
#include <glm/glm.hpp>

namespace CudaGame {
namespace Rendering {

// AABB for GPU culling (matches CudaAABB in .cu)
struct CullingAABB {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    
    // Construct from glm vectors
    static CullingAABB FromMinMax(const glm::vec3& min, const glm::vec3& max) {
        return { min.x, min.y, min.z, max.x, max.y, max.z };
    }
    
    // Construct from center and half-extents
    static CullingAABB FromCenterExtents(const glm::vec3& center, const glm::vec3& halfExtents) {
        return {
            center.x - halfExtents.x, center.y - halfExtents.y, center.z - halfExtents.z,
            center.x + halfExtents.x, center.y + halfExtents.y, center.z + halfExtents.z
        };
    }
};

// Frustum plane (ax + by + cz + d = 0)
struct CullingPlane {
    float a, b, c, d;
};

// Frustum with 6 planes
struct CullingFrustum {
    CullingPlane planes[6];  // left, right, bottom, top, near, far
    
    // Extract frustum from view-projection matrix
    static CullingFrustum FromViewProjection(const glm::mat4& vp);
};

// GPU Culling System
class CudaFrustumCuller {
public:
    CudaFrustumCuller();
    ~CudaFrustumCuller();
    
    // Initialize with max object count
    bool Initialize(int maxObjects);
    
    // Shutdown and free resources
    void Shutdown();
    
    // Upload object bounds to GPU
    void UpdateObjectBounds(const CullingAABB* bounds, int numObjects);
    
    // Execute culling with current frustum
    void Cull(const CullingFrustum& frustum);
    
    // Get results (returns count of visible objects, fills indices)
    int GetVisibleObjects(int* visibleIndices, int maxIndices);
    
    // Check if culling is available (CUDA device found)
    bool IsAvailable() const { return m_available; }
    
    // Stats
    int GetLastCulledCount() const { return m_lastCulledCount; }
    int GetLastVisibleCount() const { return m_lastVisibleCount; }

private:
    bool m_available = false;
    bool m_initialized = false;
    int m_maxObjects = 0;
    int m_currentObjectCount = 0;
    
    int m_lastCulledCount = 0;
    int m_lastVisibleCount = 0;
    
    // GPU buffers (void* to avoid CUDA header dependency)
    void* m_d_objectBounds = nullptr;
    void* m_d_frustum = nullptr;
    void* m_d_visibilityBits = nullptr;
    void* m_d_visibleIndices = nullptr;
    void* m_d_visibleCount = nullptr;
};

} // namespace Rendering
} // namespace CudaGame
