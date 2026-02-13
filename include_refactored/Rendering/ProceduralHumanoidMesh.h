#pragma once

#include "Rendering/D3D12Mesh.h"
#include <memory>
#include <vector>

namespace CudaGame {
namespace Rendering {

// Generates a high-quality procedural mesh (capsules/spheres) matching the 14-bone skeleton
class ProceduralHumanoidMesh {
public:
    static std::unique_ptr<D3D12Mesh> Create(DX12RenderBackend* backend);

private:
    // Helper to add a capsule segment
    static void AddCapsule(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                          const glm::vec3 start, const glm::vec3 end, float radius, 
                          int boneIndex, int boneIndex2 = -1);

    // Helper to add a sphere (Head/Joints)
    static void AddSphere(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                         const glm::vec3 center, float radius, int boneIndex);
                         
    // Helper to add a box (Torso/Hips)
    static void AddBox(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                      const glm::vec3 center, const glm::vec3 halfExtents, int boneIndex);
};

} // namespace Rendering
} // namespace CudaGame
