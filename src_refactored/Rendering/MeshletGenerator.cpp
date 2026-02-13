// MeshletGenerator.cpp - Meshlet generation from indexed meshes
// Implements optimal meshlet partitioning for mesh shader pipeline

#include "Rendering/Meshlet.h"
#include <iostream>
#include <glm/gtc/epsilon.hpp>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

namespace CudaGame {
namespace Rendering {

size_t MeshletMesh::GetTotalTriangles() const {
    size_t total = 0;
    for (const auto& m : meshlets) {
        total += m.primitiveCount;
    }
    return total;
}

glm::vec4 MeshletGenerator::ComputeBoundingSphere(
    const Meshlet& meshlet,
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& vertexIndices
) {
    if (meshlet.vertexCount == 0) {
        return glm::vec4(0.0f);
    }
    
    // Compute centroid
    glm::vec3 center(0.0f);
    for (uint32_t i = 0; i < meshlet.vertexCount; ++i) {
        uint32_t idx = vertexIndices[meshlet.vertexOffset + i];
        center += positions[idx];
    }
    center /= static_cast<float>(meshlet.vertexCount);
    
    // Find max radius
    float maxRadiusSq = 0.0f;
    for (uint32_t i = 0; i < meshlet.vertexCount; ++i) {
        uint32_t idx = vertexIndices[meshlet.vertexOffset + i];
        float distSq = glm::dot(positions[idx] - center, positions[idx] - center);
        maxRadiusSq = std::max(maxRadiusSq, distSq);
    }
    
    return glm::vec4(center, std::sqrt(maxRadiusSq));
}

glm::vec4 MeshletGenerator::ComputeNormalCone(
    const Meshlet& meshlet,
    const std::vector<glm::vec3>& normals,
    const std::vector<uint32_t>& vertexIndices
) {
    if (meshlet.vertexCount == 0) {
        return glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);  // Default up, can't cull
    }
    
    // Compute average normal
    glm::vec3 avgNormal(0.0f);
    for (uint32_t i = 0; i < meshlet.vertexCount; ++i) {
        uint32_t idx = vertexIndices[meshlet.vertexOffset + i];
        avgNormal += normals[idx];
    }
    avgNormal = glm::normalize(avgNormal);
    
    // Find min dot product (max angle deviation)
    float minDot = 1.0f;
    for (uint32_t i = 0; i < meshlet.vertexCount; ++i) {
        uint32_t idx = vertexIndices[meshlet.vertexOffset + i];
        float d = glm::dot(avgNormal, glm::normalize(normals[idx]));
        minDot = std::min(minDot, d);
    }
    
    // Add some tolerance for numerical stability
    minDot = std::max(-1.0f, minDot - 0.01f);
    
    return glm::vec4(avgNormal, minDot);
}

MeshletMesh MeshletGenerator::Generate(
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& normals,
    const std::vector<uint32_t>& indices
) {
    MeshletMesh result;
    result.positions = positions;
    result.normals = normals;
    result.uvs.resize(positions.size(), glm::vec2(0.0f));
    result.colors.resize(positions.size(), glm::vec4(1.0f));
    
    // Compute AABB
    result.aabbMin = glm::vec3(FLT_MAX);
    result.aabbMax = glm::vec3(-FLT_MAX);
    for (const auto& p : positions) {
        result.aabbMin = glm::min(result.aabbMin, p);
        result.aabbMax = glm::max(result.aabbMax, p);
    }
    
    // Simple greedy meshlet generation
    // Production: Use meshoptimizer or DirectXMesh for optimal clustering
    
    size_t numTriangles = indices.size() / 3;
    std::vector<bool> triangleUsed(numTriangles, false);
    
    size_t currentTriangle = 0;
    
    while (currentTriangle < numTriangles) {
        Meshlet meshlet = {};
        meshlet.vertexOffset = static_cast<uint32_t>(result.vertexIndices.size());
        meshlet.primitiveOffset = static_cast<uint32_t>(result.primitiveIndices.size());
        
        std::unordered_map<uint32_t, uint32_t> localVertexMap;  // global -> local index
        
        // Fill meshlet with triangles
        for (size_t t = currentTriangle; t < numTriangles && 
             meshlet.primitiveCount < MESHLET_MAX_PRIMITIVES; ++t) {
            
            if (triangleUsed[t]) continue;
            
            uint32_t i0 = indices[t * 3 + 0];
            uint32_t i1 = indices[t * 3 + 1];
            uint32_t i2 = indices[t * 3 + 2];
            
            // Count new vertices this triangle would add
            uint32_t newVerts = 0;
            if (localVertexMap.find(i0) == localVertexMap.end()) newVerts++;
            if (localVertexMap.find(i1) == localVertexMap.end()) newVerts++;
            if (localVertexMap.find(i2) == localVertexMap.end()) newVerts++;
            
            // Check if adding this triangle would exceed vertex limit
            if (meshlet.vertexCount + newVerts > MESHLET_MAX_VERTICES) {
                continue;  // Skip this triangle, try next
            }
            
            // Add vertices
            auto addVertex = [&](uint32_t globalIdx) -> uint32_t {
                auto it = localVertexMap.find(globalIdx);
                if (it != localVertexMap.end()) {
                    return it->second;
                }
                uint32_t localIdx = meshlet.vertexCount++;
                localVertexMap[globalIdx] = localIdx;
                result.vertexIndices.push_back(globalIdx);
                return localIdx;
            };
            
            uint32_t l0 = addVertex(i0);
            uint32_t l1 = addVertex(i1);
            uint32_t l2 = addVertex(i2);
            
            // Add primitive (packed as bytes)
            result.primitiveIndices.push_back(static_cast<uint8_t>(l0));
            result.primitiveIndices.push_back(static_cast<uint8_t>(l1));
            result.primitiveIndices.push_back(static_cast<uint8_t>(l2));
            meshlet.primitiveCount++;
            
            triangleUsed[t] = true;
        }
        
        // Compute bounds
        glm::vec4 sphere = ComputeBoundingSphere(meshlet, positions, result.vertexIndices);
        meshlet.boundingSphereCenter = glm::vec3(sphere);
        meshlet.boundingSphereRadius = sphere.w;
        
        glm::vec4 cone = ComputeNormalCone(meshlet, normals, result.vertexIndices);
        meshlet.coneAxis = glm::vec3(cone);
        meshlet.coneAngle = std::acos(std::clamp(cone.w, -1.0f, 1.0f));
        
        result.meshlets.push_back(meshlet);
        
        // Find next unused triangle
        while (currentTriangle < numTriangles && triangleUsed[currentTriangle]) {
            ++currentTriangle;
        }
    }
    
    return result;
}

void MeshletGenerator::ValidateMeshlets(MeshletMesh& mesh) {
    if (mesh.meshlets.empty()) return;
    
    std::vector<Meshlet> validMeshlets;
    validMeshlets.reserve(mesh.meshlets.size());
    
    for (const auto& m : mesh.meshlets) {
        // 1. Degenerate Count Check
        if (m.vertexCount == 0 || m.primitiveCount == 0) {
            continue; // Skip empty
        }
        
        // 2. Hardware Limit Check (NV Mesh Shader)
        if (m.vertexCount > MESHLET_MAX_VERTICES || m.primitiveCount > MESHLET_MAX_PRIMITIVES) {
            // This suggests logic error in Split, but for safety we cull it
            continue; 
        }
        
        // 3. NaN Check (Bounding Sphere)
        if (std::isnan(m.boundingSphereRadius) || std::isinf(m.boundingSphereRadius)) {
             continue;
        }

        validMeshlets.push_back(m);
    }
    
    if (validMeshlets.size() != mesh.meshlets.size()) {
        std::cerr << "[MeshletGenerator] Removed " << (mesh.meshlets.size() - validMeshlets.size()) 
                  << " invalid/degenerate meshlets." << std::endl;
        mesh.meshlets = validMeshlets;
    }
}

} // namespace Rendering
} // namespace CudaGame
