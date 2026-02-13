#pragma once
#ifdef _WIN32

#include <d3d12.h>
#include <vector>
#include <memory>
#include <string>
#include <glm/glm.hpp>
#include "Rendering/Meshlet.h"
#include "Animation/AnimationResources.h"

namespace CudaGame {
namespace Rendering {

// Forward declarations
class DX12RenderBackend;

// Vertex format for geometry pass (matches HLSL)
// AAA Standard: position, normal, tangent, texcoord, color
// Vertex format for geometry pass (Unified for Static and Skinned)
// AAA Standard: position, normal, tangent, texcoord, color, boneIndices, boneWeights
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec2 texcoord;
    glm::vec4 color;     // rgb = vertex color multiplier, a = emissive intensity
    
    // Animation Data (Unified)
    glm::ivec4 boneIndices; // Up to 4 bones (Default: 0,0,0,0)
    glm::vec4 boneWeights;  // Weights sum to 1.0 (Default: 0,0,0,0 for static)

    Vertex() 
        : color(1.0f, 1.0f, 1.0f, 0.0f), 
          boneIndices(0), 
          boneWeights(0.0f) {}  

    Vertex(const glm::vec3& pos, const glm::vec3& norm, const glm::vec3& tan, const glm::vec2& uv)
        : position(pos), normal(norm), tangent(tan), texcoord(uv), 
          color(1.0f, 1.0f, 1.0f, 0.0f),
          boneIndices(0),
          boneWeights(0.0f) {}

    Vertex(const glm::vec3& pos, const glm::vec3& norm, const glm::vec3& tan, const glm::vec2& uv, const glm::vec4& col)
        : position(pos), normal(norm), tangent(tan), texcoord(uv), 
          color(col),
          boneIndices(0),
          boneWeights(0.0f) {}
          
    // Skinned Constructor
    Vertex(const glm::vec3& pos, const glm::vec3& norm, const glm::vec3& tan, const glm::vec2& uv, const glm::vec4& col, 
           const glm::ivec4& bIndices, const glm::vec4& bWeights)
        : position(pos), normal(norm), tangent(tan), texcoord(uv), 
          color(col),
          boneIndices(bIndices),
          boneWeights(bWeights) {}
};

// Extended vertex for procedural buildings with emissive windows
struct BuildingVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texcoord;
    glm::vec3 color;      // Vertex color for variety
    glm::vec3 emissive;   // Emissive glow for lit windows
    
    BuildingVertex() = default;
    BuildingVertex(const glm::vec3& pos, const glm::vec3& norm, const glm::vec2& uv,
                   const glm::vec3& col, const glm::vec3& emit)
        : position(pos), normal(norm), texcoord(uv), color(col), emissive(emit) {}
};

// Material properties for rendering
struct Material {
    glm::vec4 albedoColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float ambientOcclusion = 1.0f;
    float emissiveStrength = 0.0f;
    glm::vec3 emissiveColor = glm::vec3(0.0f);
    float _padding = 0.0f;
};

// GPU buffer wrapper
struct GPUBuffer {
    ID3D12Resource* resource = nullptr;
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView = {};
    D3D12_INDEX_BUFFER_VIEW indexBufferView = {};
    uint32_t elementCount = 0;
    uint32_t stride = 0;
    
    ~GPUBuffer() {
        if (resource) {
            resource->Release();
            resource = nullptr;
        }
    }
};

// Mesh for D3D12 rendering
class D3D12Mesh {
public:
    D3D12Mesh() = default;
    ~D3D12Mesh();
    
    // Create mesh from vertex/index data
    bool Create(DX12RenderBackend* backend,
                const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices,
                const std::string& name);

    // Load mesh from file (OBJ/FBX/X/GLTF)
    bool LoadFromFile(DX12RenderBackend* backend, const std::string& filename, bool loadSkeleton = true);
    
    // Getters
    const D3D12_VERTEX_BUFFER_VIEW& GetVertexBufferView() const { return m_vertexBuffer.vertexBufferView; }
    const D3D12_INDEX_BUFFER_VIEW& GetIndexBufferView() const { return m_indexBuffer.indexBufferView; }
    uint32_t GetIndexCount() const { return m_indexBuffer.elementCount; }
    uint32_t GetVertexCount() const { return m_vertexBuffer.elementCount; }
    
    Material& GetMaterial() { return m_material; }
    const Material& GetMaterial() const { return m_material; }
    
    // Mesh Shader Support
    struct MeshletBuffers {
        ID3D12Resource* meshlets = nullptr;
        ID3D12Resource* vertexIndices = nullptr;
        ID3D12Resource* primitives = nullptr;
        ID3D12Resource* bounds = nullptr;
        // Attribute buffers (SoA for mesh shader)
        ID3D12Resource* positions = nullptr;
        ID3D12Resource* normals = nullptr;
        ID3D12Resource* uvs = nullptr;
        ID3D12Resource* colors = nullptr;
        ID3D12Resource* instances = nullptr;
        
        uint32_t meshletCount = 0;
        
        void Release() {
            if (meshlets) { meshlets->Release(); meshlets = nullptr; }
            if (vertexIndices) { vertexIndices->Release(); vertexIndices = nullptr; }
            if (primitives) { primitives->Release(); primitives = nullptr; }
            if (bounds) { bounds->Release(); bounds = nullptr; }
            if (positions) { positions->Release(); positions = nullptr; }
            if (normals) { normals->Release(); normals = nullptr; }
            if (uvs) { uvs->Release(); uvs = nullptr; }
            if (colors) { colors->Release(); colors = nullptr; }
            if (instances) { instances->Release(); instances = nullptr; }
        }
    };
    
    // Generate and upload meshlets for this mesh
    bool GenerateMeshlets(DX12RenderBackend* backend, 
                         const std::vector<Vertex>& vertices,
                         const std::vector<uint32_t>& indices);
                         
    const MeshletBuffers& GetMeshletBuffers() const { return m_meshlets; }
    
    // Transform
    glm::mat4 transform = glm::mat4(1.0f);
    
    // Bounds (AABB)
    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);

    // Update GPU instance buffer with current transform
    void UpdateGPUInstanceData();
    
    // Animation State (Per-Instance)
    std::vector<glm::mat4> boneMatrices;
    uint32_t globalBoneOffset = 0; // Offset in the global structured buffer
    
    // Load animation from file (Assimp)
    bool LoadAnimation(const std::string& filePath, Animation::Skeleton& outSkeleton, std::unique_ptr<Animation::AnimationClip>& outClip);
    
    // Get loaded animations from file
    const std::vector<std::shared_ptr<Animation::AnimationClip>>& GetAnimations() const { return m_animations; }
    
    // Add manual animation (for procedural meshes)
    void AddAnimation(std::shared_ptr<Animation::AnimationClip> clip) {
        m_animations.push_back(clip);
    }

    // Set skeleton for skinning
    void SetSkeleton(std::shared_ptr<Animation::Skeleton> skeleton) { m_skeleton = skeleton; }
    std::shared_ptr<Animation::Skeleton> GetSkeleton() const { return m_skeleton; }
    
    // Create skinned mesh (Unified)
    bool CreateSkinned(DX12RenderBackend* backend,
                      const std::vector<Vertex>& vertices,
                      const std::vector<uint32_t>& indices,
                      const std::string& name);

    void SetBoneBuffer(ID3D12Resource* buffer) { m_globalBoneBuffer = buffer; }

private:
    ID3D12Resource* m_globalBoneBuffer = nullptr;
    GPUBuffer m_vertexBuffer;
    GPUBuffer m_indexBuffer;
    MeshletBuffers m_meshlets;
    Material m_material;
    std::string m_name;
    std::shared_ptr<Animation::Skeleton> m_skeleton;
    std::vector<std::shared_ptr<Animation::AnimationClip>> m_animations;
};

// Procedural mesh generators
class MeshGenerator {
public:
    // Generate cube mesh (1x1x1 centered at origin)
    static std::unique_ptr<D3D12Mesh> CreateCube(DX12RenderBackend* backend);
    
    // Generate sphere mesh (radius 1, centered at origin)
    static std::unique_ptr<D3D12Mesh> CreateSphere(DX12RenderBackend* backend, uint32_t segments = 32);
    
    // Generate plane mesh (1x1 on XZ plane)
    static std::unique_ptr<D3D12Mesh> CreatePlane(DX12RenderBackend* backend);
    
    // Generate a multi-box mesh (Skinned) based on the provided Skeleton
    static std::unique_ptr<D3D12Mesh> CreateSkinnedBlockMesh(DX12RenderBackend* backend, std::shared_ptr<Animation::Skeleton> skeleton);
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
