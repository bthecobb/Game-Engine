// CudaFrustumCulling.cpp - C++ wrapper for CUDA frustum culling
// Wraps the CUDA kernel with a clean C++ interface

#include "Rendering/CudaFrustumCulling.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cmath>

// Forward declarations for CUDA functions (defined in .cu file)
extern "C" {
    cudaError_t AllocateCullingBuffers(void** d_objectBounds, void** d_frustum, 
        void** d_visibilityBits, void** d_visibleIndices, void** d_visibleCount, int maxObjects);
    void FreeCullingBuffers(void* d_objectBounds, void* d_frustum, 
        void* d_visibilityBits, void* d_visibleIndices, void* d_visibleCount);
    cudaError_t UploadObjectBounds(void* d_objectBounds, const void* h_bounds, int numObjects);
    cudaError_t UploadFrustum(void* d_frustum, const void* h_frustum);
    cudaError_t ExecuteFrustumCulling(void* d_objectBounds, void* d_frustum, 
        void* d_visibilityBits, int numObjects);
    cudaError_t GetVisibleIndices(void* d_visibilityBits, void* d_visibleIndices, 
        void* d_visibleCount, int* h_visibleIndices, int* h_visibleCount, int numObjects);
}

namespace CudaGame {
namespace Rendering {

// Extract frustum planes from view-projection matrix
CullingFrustum CullingFrustum::FromViewProjection(const glm::mat4& vp) {
    CullingFrustum frustum;
    
    // Left plane: row3 + row0
    frustum.planes[0] = { vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], 
                          vp[2][3] + vp[2][0], vp[3][3] + vp[3][0] };
    // Right plane: row3 - row0  
    frustum.planes[1] = { vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], 
                          vp[2][3] - vp[2][0], vp[3][3] - vp[3][0] };
    // Bottom plane: row3 + row1
    frustum.planes[2] = { vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], 
                          vp[2][3] + vp[2][1], vp[3][3] + vp[3][1] };
    // Top plane: row3 - row1
    frustum.planes[3] = { vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], 
                          vp[2][3] - vp[2][1], vp[3][3] - vp[3][1] };
    // Near plane: row3 + row2
    frustum.planes[4] = { vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], 
                          vp[2][3] + vp[2][2], vp[3][3] + vp[3][2] };
    // Far plane: row3 - row2
    frustum.planes[5] = { vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], 
                          vp[2][3] - vp[2][2], vp[3][3] - vp[3][2] };
    
    // Normalize planes
    for (int i = 0; i < 6; ++i) {
        float len = std::sqrt(frustum.planes[i].a * frustum.planes[i].a + 
                              frustum.planes[i].b * frustum.planes[i].b + 
                              frustum.planes[i].c * frustum.planes[i].c);
        if (len > 0.0001f) {
            frustum.planes[i].a /= len;
            frustum.planes[i].b /= len;
            frustum.planes[i].c /= len;
            frustum.planes[i].d /= len;
        }
    }
    
    return frustum;
}

CudaFrustumCuller::CudaFrustumCuller() {}

CudaFrustumCuller::~CudaFrustumCuller() {
    Shutdown();
}

bool CudaFrustumCuller::Initialize(int maxObjects) {
    if (m_initialized) {
        Shutdown();
    }
    
    m_maxObjects = maxObjects;
    
    // Try to allocate CUDA buffers
    auto err = AllocateCullingBuffers(
        &m_d_objectBounds, &m_d_frustum, &m_d_visibilityBits,
        &m_d_visibleIndices, &m_d_visibleCount, maxObjects
    );
    
    if (err != 0) {  // cudaSuccess = 0
        std::cerr << "[CudaFrustumCuller] Failed to allocate CUDA buffers" << std::endl;
        m_available = false;
        return false;
    }
    
    m_available = true;
    m_initialized = true;
    std::cout << "[CudaFrustumCuller] Initialized for " << maxObjects << " objects" << std::endl;
    return true;
}

void CudaFrustumCuller::Shutdown() {
    if (!m_initialized) return;
    
    FreeCullingBuffers(m_d_objectBounds, m_d_frustum, m_d_visibilityBits,
                       m_d_visibleIndices, m_d_visibleCount);
    
    m_d_objectBounds = nullptr;
    m_d_frustum = nullptr;
    m_d_visibilityBits = nullptr;
    m_d_visibleIndices = nullptr;
    m_d_visibleCount = nullptr;
    
    m_initialized = false;
    m_available = false;
}

void CudaFrustumCuller::UpdateObjectBounds(const CullingAABB* bounds, int numObjects) {
    if (!m_initialized || !m_available) return;
    if (numObjects > m_maxObjects) {
        std::cerr << "[CudaFrustumCuller] Too many objects: " << numObjects << " > " << m_maxObjects << std::endl;
        return;
    }
    
    m_currentObjectCount = numObjects;
    UploadObjectBounds(m_d_objectBounds, bounds, numObjects);
}

void CudaFrustumCuller::Cull(const CullingFrustum& frustum) {
    if (!m_initialized || !m_available || m_currentObjectCount == 0) return;
    
    UploadFrustum(m_d_frustum, &frustum);
    ExecuteFrustumCulling(m_d_objectBounds, m_d_frustum, m_d_visibilityBits, m_currentObjectCount);
}

int CudaFrustumCuller::GetVisibleObjects(int* visibleIndices, int maxIndices) {
    if (!m_initialized || !m_available || m_currentObjectCount == 0) {
        return 0;
    }
    
    int visibleCount = 0;
    GetVisibleIndices(m_d_visibilityBits, m_d_visibleIndices, m_d_visibleCount,
                      visibleIndices, &visibleCount, m_currentObjectCount);
    
    m_lastVisibleCount = visibleCount;
    m_lastCulledCount = m_currentObjectCount - visibleCount;
    
    return (visibleCount <= maxIndices) ? visibleCount : maxIndices;
}

} // namespace Rendering
} // namespace CudaGame
