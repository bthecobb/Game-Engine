#pragma once
// Meshlet.h - Data structures for mesh shader pipeline
// AAA-tier geometry processing using DX12 Ultimate mesh shaders

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

namespace CudaGame {
namespace Rendering {

// Maximum vertices/primitives per meshlet (GPU wave size optimal)
constexpr uint32_t MESHLET_MAX_VERTICES = 64;
constexpr uint32_t MESHLET_MAX_PRIMITIVES = 126;  // 126 triangles max

/**
 * @brief Single meshlet - a small cluster of triangles
 * 
 * Meshlets enable per-cluster culling on the GPU, dramatically
 * reducing overdraw and improving performance with dense geometry.
 */
struct Meshlet {
    uint32_t vertexOffset;      // Offset into vertex index buffer
    uint32_t vertexCount;       // Number of unique vertices (max 64)
    uint32_t primitiveOffset;   // Offset into primitive buffer
    uint32_t primitiveCount;    // Number of triangles (max 126)
    
    // Bounding sphere for frustum culling
    glm::vec3 boundingSphereCenter;
    float boundingSphereRadius;
    
    // Cone for backface culling (apex at center, axis = normal)
    glm::vec3 coneAxis;         // Average normal direction
    float coneAngle;            // Cone half-angle (radians)
};

/**
 * @brief GPU-friendly meshlet bounds for culling shader
 */
struct MeshletBounds {
    glm::vec4 sphere;           // xyz = center, w = radius
    glm::vec4 cone;             // xyz = axis, w = cos(angle)
};

/**
 * @brief Complete meshlet data for a mesh
 */
struct MeshletMesh {
    std::vector<Meshlet> meshlets;
    
    // Vertex indices: meshlets reference into this
    std::vector<uint32_t> vertexIndices;
    
    // Packed triangle indices (3 bytes per triangle)
    std::vector<uint8_t> primitiveIndices;
    
    // Original mesh vertices (position, normal, UV, etc.)
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> colors;
    
    // Mesh-level bounds
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    
    size_t GetMeshletCount() const { return meshlets.size(); }
    size_t GetTotalTriangles() const;
    size_t GetTotalVertices() const { return positions.size(); }
};

/**
 * @brief Generates meshlets from standard indexed mesh
 */
class MeshletGenerator {
public:
    /**
     * @brief Generate meshlets from indexed triangle mesh
     * @param positions Vertex positions
     * @param indices Triangle indices (3 per triangle)
     * @return MeshletMesh with all meshlet data
     */
    static MeshletMesh Generate(
        const std::vector<glm::vec3>& positions,
        const std::vector<glm::vec3>& normals,
        const std::vector<uint32_t>& indices
    );
    
    /**
     * @brief Compute bounding sphere for a meshlet
     */
    static glm::vec4 ComputeBoundingSphere(
        const Meshlet& meshlet,
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& vertexIndices
    );
    
    /**
     * @brief Compute normal cone for backface culling
     */
    static glm::vec4 ComputeNormalCone(
        const Meshlet& meshlet,
        const std::vector<glm::vec3>& normals,
        const std::vector<uint32_t>& vertexIndices
    );

    /**
     * @brief Validate and prune degenerate meshlets
     */
    static void ValidateMeshlets(MeshletMesh& mesh);
};

} // namespace Rendering
} // namespace CudaGame
