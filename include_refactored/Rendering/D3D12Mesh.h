#pragma once
#ifdef _WIN32

#include <d3d12.h>
#include <vector>
#include <memory>
#include <string>
#include <glm/glm.hpp>

namespace CudaGame {
namespace Rendering {

// Forward declarations
class DX12RenderBackend;

// Vertex format for geometry pass (matches HLSL)
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec2 texcoord;
    
    Vertex() = default;
    Vertex(const glm::vec3& pos, const glm::vec3& norm, const glm::vec3& tan, const glm::vec2& uv)
        : position(pos), normal(norm), tangent(tan), texcoord(uv) {}
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
    
    // Getters
    const D3D12_VERTEX_BUFFER_VIEW& GetVertexBufferView() const { return m_vertexBuffer.vertexBufferView; }
    const D3D12_INDEX_BUFFER_VIEW& GetIndexBufferView() const { return m_indexBuffer.indexBufferView; }
    uint32_t GetIndexCount() const { return m_indexBuffer.elementCount; }
    uint32_t GetVertexCount() const { return m_vertexBuffer.elementCount; }
    
    Material& GetMaterial() { return m_material; }
    const Material& GetMaterial() const { return m_material; }
    
    // Transform
    glm::mat4 transform = glm::mat4(1.0f);
    
private:
    GPUBuffer m_vertexBuffer;
    GPUBuffer m_indexBuffer;
    Material m_material;
    std::string m_name;
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
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
