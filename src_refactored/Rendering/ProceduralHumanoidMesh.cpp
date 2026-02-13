#include "Rendering/ProceduralHumanoidMesh.h"
#include "Animation/ProceduralAnimationGenerator.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtx/transform.hpp>

namespace CudaGame {
namespace Rendering {

std::unique_ptr<D3D12Mesh> ProceduralHumanoidMesh::Create(DX12RenderBackend* backend) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Dimensions
    const float torsoWidth = 0.35f;
    const float torsoThick = 0.2f;
    const float headSize = 0.12f;
    const float limbRadius = 0.06f;
    
    // Skeleton Offsets (Must match ProceduralAnimationGenerator)
    // 0: Hips
    AddBox(vertices, indices, glm::vec3(0, 1.0f, 0), glm::vec3(torsoWidth/2, 0.1f, torsoThick/2), 0);
    
    // 1: Spine
    AddBox(vertices, indices, glm::vec3(0, 1.25f, 0), glm::vec3(torsoWidth/2 * 0.9f, 0.15f, torsoThick/2 * 0.9f), 1);
    
    // 2: Neck (Capsule)
    AddCapsule(vertices, indices, glm::vec3(0, 1.4f, 0), glm::vec3(0, 1.5f, 0), 0.05f, 2);
    
    // 3: Head (Sphere)
    AddSphere(vertices, indices, glm::vec3(0, 1.62f, 0), headSize, 3);
    
    // Legs (Capsules)
    // 4: LeftUpLeg (-0.15, -0.1 relative to hips -> 1.0) -> Start: (-0.15, 0.9, 0)
    // Length 0.4 down
    AddCapsule(vertices, indices, glm::vec3(-0.15f, 0.9f, 0), glm::vec3(-0.15f, 0.5f, 0), limbRadius, 4);
    // 5: LeftLeg
    AddCapsule(vertices, indices, glm::vec3(-0.15f, 0.5f, 0), glm::vec3(-0.15f, 0.1f, 0), limbRadius * 0.8f, 5);
    // 6: LeftFoot (Box)
    AddBox(vertices, indices, glm::vec3(-0.15f, 0.05f, 0.1f), glm::vec3(0.06f, 0.05f, 0.15f), 6);
    
    // 7: RightUpLeg
    AddCapsule(vertices, indices, glm::vec3(0.15f, 0.9f, 0), glm::vec3(0.15f, 0.5f, 0), limbRadius, 7);
    // 8: RightLeg
    AddCapsule(vertices, indices, glm::vec3(0.15f, 0.5f, 0), glm::vec3(0.15f, 0.1f, 0), limbRadius * 0.8f, 8);
    // 9: RightFoot
    AddBox(vertices, indices, glm::vec3(0.15f, 0.05f, 0.1f), glm::vec3(0.06f, 0.05f, 0.15f), 9);
    
    // Arms
    // 10: LeftArm (-0.2, 0.15 rel to Spine -> 1.2) -> Start: (-0.2, 1.35, 0)
    AddCapsule(vertices, indices, glm::vec3(-0.25f, 1.35f, 0), glm::vec3(-0.55f, 1.35f, 0), limbRadius * 0.7f, 10);
    // 11: LeftForeArm
    AddCapsule(vertices, indices, glm::vec3(-0.55f, 1.35f, 0), glm::vec3(-0.85f, 1.35f, 0), limbRadius * 0.6f, 11);
    
    // 12: RightArm
    AddCapsule(vertices, indices, glm::vec3(0.25f, 1.35f, 0), glm::vec3(0.55f, 1.35f, 0), limbRadius * 0.7f, 12);
    // 13: RightForeArm
    AddCapsule(vertices, indices, glm::vec3(0.55f, 1.35f, 0), glm::vec3(0.85f, 1.35f, 0), limbRadius * 0.6f, 13);

    // Create Mesh
    auto mesh = std::make_unique<D3D12Mesh>();
    
    // We need the skeleton for the mesh to recognize it needs skinning
    // Note: D3D12Mesh::CreateSkinned doesn't strictly need m_skeleton set if we provide weights manually,
    // but the renderer might check it.
    auto skeleton = Animation::ProceduralAnimationGenerator::CreateHumanoidSkeleton();
    mesh->SetSkeleton(skeleton);
    
    if (!mesh->CreateSkinned(backend, vertices, indices, "ProceduralHumanoid")) {
        return nullptr;
    }
    
    // Important: Generate meshlets for GPU rendering
    // Note: Use a simplified cast/conversion helper if GenerateMeshlets expects standard Vertex
    // For now D3D12Mesh::GenerateMeshlets takes Vertex, but the skinned data is separate.
    // The D3D12Mesh::CreateSkinned handles the skinned vertex buffer upload.
    // However, the mesh shader pipeline expects meshlet data.
    // We need to implement GenerateMeshletsSkinned or cast.
    // For simplicity in Phase 1, we rely on the GeometryPass_Fallback (VertexShader) which uses the VB/IB directly.
    // The Mesh Shader path requires `Meshlet` generation which is separate. 
    // We'll skip meshlet generation for the skinned mesh for now as we are likely using the fallback path or VS path for skinned meshes.
    // (Checked: DX12RenderPipeline uses VertexShader fallback if meshlets are missing OR checks m_activePath)
    
    return mesh;
}

void ProceduralHumanoidMesh::AddBox(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                                    const glm::vec3 center, const glm::vec3 halfExtents, int boneIndex) {
    uint32_t baseIndex = (uint32_t)vertices.size();
    
    glm::vec3 corners[8] = {
        {-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1},
        {-1,-1,1}, {1,-1,1}, {1,1,1}, {-1,1,1}
    };
    
    for (auto& c : corners) c = center + c * halfExtents;

    const uint32_t boxIndices[36] = {
        0,1,2, 0,2,3, // Back
        4,5,6, 4,6,7, // Front
        0,3,7, 0,7,4, // Left
        1,2,6, 1,6,5, // Right
        3,2,6, 3,6,7, // Top
        0,1,5, 0,5,4  // Bottom
    };
    
    const glm::vec3 normals[6] = {
        {0,0,-1}, {0,0,1}, {-1,0,0}, {1,0,0}, {0,1,0}, {0,-1,0}
    };

    // Add unique vertices for flat shading (6 faces * 4 vertices = 24)
    // Mapping 8 corners to 24 verts
    int faceVerts[6][4] = {
        {0,1,2,3}, {5,4,7,6}, {4,0,3,7}, {1,5,6,2}, {3,2,6,7}, {4,5,1,0}
    };

    for (int f = 0; f < 6; f++) {
        for (int i = 0; i < 4; i++) {
            int cIdx = faceVerts[f][i];
            
            // Replaced SkinnedVertex with Vertex unified constructor
            // Vertex(pos, norm, tan, uv, col, bIndices, bWeights)
            Vertex v(
                corners[cIdx],
                normals[f],
                glm::vec3(0.0f), // Tangent placeholder
                glm::vec2(i<2?0:1, i%2?1:0), // UV
                glm::vec4(0.8f, 0.8f, 0.9f, 1.0f), // Color
                glm::ivec4(boneIndex, 0, 0, 0), // Bone Indices
                glm::vec4(1.0f, 0, 0, 0) // Bone Weights
            );
            vertices.push_back(v);
        }
        
        uint32_t base = baseIndex + f*4;
        indices.push_back(base + 0); indices.push_back(base + 1); indices.push_back(base + 2);
        indices.push_back(base + 0); indices.push_back(base + 2); indices.push_back(base + 3);
    }
}

void ProceduralHumanoidMesh::AddCapsule(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                                        const glm::vec3 start, const glm::vec3 end, float radius, 
                                        int boneIndex, int boneIndex2) {
    // Determine cylinder basis
    glm::vec3 axis = end - start;
    float len = glm::length(axis);
    axis = glm::normalize(axis);
    
    // Choose arbitrary perpendicular
    glm::vec3 temp = (abs(axis.y) < 0.9f) ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
    glm::vec3 right = glm::normalize(glm::cross(axis, temp));
    glm::vec3 up = glm::cross(right, axis);
    
    uint32_t rings = 8;
    uint32_t sectors = 12;
    uint32_t baseIndex = (uint32_t)vertices.size();

    // Body (Cylinder)
    for (uint32_t r = 0; r <= rings; r++) {
        float t = (float)r / rings;
        glm::vec3 center = start + axis * len * t;
        
        // Weight blending (linear along length)
        float w2 = t; 
        float w1 = 1.0f - w2;
        if (boneIndex2 == -1) { w1 = 1.0f; w2 = 0.0f; }
        
        for (uint32_t s = 0; s <= sectors; s++) {
            float phi = (float)s / sectors * glm::two_pi<float>();
            float sinP = sin(phi);
            float cosP = cos(phi);
            
            glm::vec3 offset = right * cosP * radius + up * sinP * radius;
            glm::vec3 pos = center + offset;
            glm::vec3 norm = glm::normalize(offset); // Normal points out from axis
            
            Vertex v(
                pos,
                norm,
                glm::vec3(0.0f), // Tangent
                glm::vec2(t, (float)s/sectors), // UV
                glm::vec4(1.0f), // Color
                glm::ivec4(boneIndex, boneIndex2==-1?0:boneIndex2, 0, 0), // Bone Indices
                glm::vec4(w1, w2, 0, 0) // Bone Weights
            );
            vertices.push_back(v);
        }
    }
    
    // Cylinder Indices
    for (uint32_t r = 0; r < rings; r++) {
        for (uint32_t s = 0; s < sectors; s++) {
            uint32_t i0 = baseIndex + r * (sectors + 1) + s;
            uint32_t i1 = i0 + 1;
            uint32_t i2 = i0 + (sectors + 1);
            uint32_t i3 = i2 + 1;
            
            indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
            indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
        }
    }
    
    // End Caps (Semi-spheres) - simplified to just flat caps for Phase 1 robustness
    // Ideally we add spheres at endpoints, but sticking to cylinder is safer for first compile.
}

void ProceduralHumanoidMesh::AddSphere(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                                       const glm::vec3 center, float radius, int boneIndex) {
    uint32_t rings = 12;
    uint32_t sectors = 16;
    uint32_t baseIndex = (uint32_t)vertices.size();
    
    for (uint32_t r = 0; r <= rings; r++) {
        float theta = (float)r / rings * glm::pi<float>();
        float sinT = sin(theta);
        float cosT = cos(theta);
        
        for (uint32_t s = 0; s <= sectors; s++) {
            float phi = (float)s / sectors * glm::two_pi<float>();
            glm::vec3 dir(sinT * cos(phi), cosT, sinT * sin(phi));
            
            Vertex v(
                center + dir * radius,
                dir,
                glm::vec3(0.0f), // Tangent
                glm::vec2((float)s/sectors, (float)r/rings), // UV
                glm::vec4(1.0f, 0.8f, 0.8f, 1.0f), // Face Color
                glm::ivec4(boneIndex, 0, 0, 0), // Bone Indices
                glm::vec4(1.0f, 0, 0, 0) // Bone Weights
            );
            vertices.push_back(v);
        }
    }
    
    for (uint32_t r = 0; r < rings; r++) {
        for (uint32_t s = 0; s < sectors; s++) {
            uint32_t i0 = baseIndex + r * (sectors + 1) + s;
            uint32_t i1 = i0 + 1;
            uint32_t i2 = i0 + (sectors + 1);
            uint32_t i3 = i2 + 1;
            
            indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
            indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
        }
    }
}

} // namespace Rendering
} // namespace CudaGame
