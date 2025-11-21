#ifdef _WIN32
#include "Rendering/D3D12Mesh.h"
#include "Rendering/Backends/DX12RenderBackend.h"
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <cstring>

namespace CudaGame {
namespace Rendering {

D3D12Mesh::~D3D12Mesh() {
    // GPUBuffer destructors will release resources
}

bool D3D12Mesh::Create(DX12RenderBackend* backend,
                       const std::vector<Vertex>& vertices,
                       const std::vector<uint32_t>& indices,
                       const std::string& name) {
    m_name = name;
    
    ID3D12Device* device = backend->GetDevice();
    
    // Create vertex buffer
    {
        const UINT64 vertexBufferSize = vertices.size() * sizeof(Vertex);
        
        // Create upload heap for CPU->GPU transfer
        D3D12_HEAP_PROPERTIES uploadHeapProps = {};
        uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        uploadHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        uploadHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        uploadHeapProps.CreationNodeMask = 1;
        uploadHeapProps.VisibleNodeMask = 1;
        
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Alignment = 0;
        bufferDesc.Width = vertexBufferSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.SampleDesc.Quality = 0;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        
        HRESULT hr = device->CreateCommittedResource(
            &uploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_vertexBuffer.resource)
        );
        
        if (FAILED(hr)) {
            std::cerr << "[Mesh] Failed to create vertex buffer: " << name << std::endl;
            return false;
        }
        
        // Copy vertex data to upload heap
        void* pVertexDataBegin;
        D3D12_RANGE readRange = {0, 0};  // We don't intend to read from this resource on CPU
        hr = m_vertexBuffer.resource->Map(0, &readRange, &pVertexDataBegin);
        if (SUCCEEDED(hr)) {
            memcpy(pVertexDataBegin, vertices.data(), vertexBufferSize);
            m_vertexBuffer.resource->Unmap(0, nullptr);
        } else {
            std::cerr << "[Mesh] Failed to map vertex buffer" << std::endl;
            return false;
        }
        
        // Initialize vertex buffer view
        m_vertexBuffer.vertexBufferView.BufferLocation = m_vertexBuffer.resource->GetGPUVirtualAddress();
        m_vertexBuffer.vertexBufferView.StrideInBytes = sizeof(Vertex);
        m_vertexBuffer.vertexBufferView.SizeInBytes = (UINT)vertexBufferSize;
        m_vertexBuffer.elementCount = (uint32_t)vertices.size();
        m_vertexBuffer.stride = sizeof(Vertex);
        
        // Set debug name
        std::wstring wname = std::wstring(name.begin(), name.end()) + L"_VB";
        m_vertexBuffer.resource->SetName(wname.c_str());
    }
    
    // Create index buffer
    {
        const UINT64 indexBufferSize = indices.size() * sizeof(uint32_t);
        
        D3D12_HEAP_PROPERTIES uploadHeapProps = {};
        uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        uploadHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        uploadHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        uploadHeapProps.CreationNodeMask = 1;
        uploadHeapProps.VisibleNodeMask = 1;
        
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Alignment = 0;
        bufferDesc.Width = indexBufferSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.SampleDesc.Quality = 0;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        
        HRESULT hr = device->CreateCommittedResource(
            &uploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_indexBuffer.resource)
        );
        
        if (FAILED(hr)) {
            std::cerr << "[Mesh] Failed to create index buffer: " << name << std::endl;
            return false;
        }
        
        // Copy index data to upload heap
        void* pIndexDataBegin;
        D3D12_RANGE readRange = {0, 0};
        hr = m_indexBuffer.resource->Map(0, &readRange, &pIndexDataBegin);
        if (SUCCEEDED(hr)) {
            memcpy(pIndexDataBegin, indices.data(), indexBufferSize);
            m_indexBuffer.resource->Unmap(0, nullptr);
        } else {
            std::cerr << "[Mesh] Failed to map index buffer" << std::endl;
            return false;
        }
        
        // Initialize index buffer view
        m_indexBuffer.indexBufferView.BufferLocation = m_indexBuffer.resource->GetGPUVirtualAddress();
        m_indexBuffer.indexBufferView.Format = DXGI_FORMAT_R32_UINT;
        m_indexBuffer.indexBufferView.SizeInBytes = (UINT)indexBufferSize;
        m_indexBuffer.elementCount = (uint32_t)indices.size();
        m_indexBuffer.stride = sizeof(uint32_t);
        
        // Set debug name
        std::wstring wname = std::wstring(name.begin(), name.end()) + L"_IB";
        m_indexBuffer.resource->SetName(wname.c_str());
    }
    
    std::cout << "[Mesh] Created: " << name 
              << " (" << vertices.size() << " verts, " << indices.size() << " indices)" << std::endl;
    
    return true;
}

// ========== Procedural Mesh Generators ==========

std::unique_ptr<D3D12Mesh> MeshGenerator::CreateCube(DX12RenderBackend* backend) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    // Cube vertices (8 corners, 24 vertices for proper normals per face)
    const glm::vec3 positions[8] = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},  // Back
        {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}      // Front
    };
    
    // Define 6 faces with proper normals and tangents
    struct Face {
        uint32_t v[4];     // Vertex indices
        glm::vec3 normal;
        glm::vec3 tangent;
    };
    
    Face faces[6] = {
        {{0, 1, 2, 3}, {0, 0, -1}, {1, 0, 0}},  // Back
        {{5, 4, 7, 6}, {0, 0, 1}, {-1, 0, 0}},  // Front
        {{4, 0, 3, 7}, {-1, 0, 0}, {0, 0, 1}},  // Left
        {{1, 5, 6, 2}, {1, 0, 0}, {0, 0, -1}},  // Right
        {{3, 2, 6, 7}, {0, 1, 0}, {1, 0, 0}},   // Top
        {{4, 5, 1, 0}, {0, -1, 0}, {1, 0, 0}}   // Bottom
    };
    
    const glm::vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    
    // Build vertices for each face
    for (const auto& face : faces) {
        for (int i = 0; i < 4; i++) {
            vertices.emplace_back(
                positions[face.v[i]],
                face.normal,
                face.tangent,
                uvs[i]
            );
        }
    }
    
    // Build indices (2 triangles per face)
    for (uint32_t f = 0; f < 6; f++) {
        uint32_t base = f * 4;
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
    
    auto mesh = std::make_unique<D3D12Mesh>();
    if (!mesh->Create(backend, vertices, indices, "Cube")) {
        return nullptr;
    }
    
    return mesh;
}

std::unique_ptr<D3D12Mesh> MeshGenerator::CreateSphere(DX12RenderBackend* backend, uint32_t segments) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    const float radius = 1.0f;
    const uint32_t rings = segments;
    const uint32_t sectors = segments * 2;
    
    // Generate vertices
    for (uint32_t r = 0; r <= rings; r++) {
        float theta = (float)r / (float)rings * glm::pi<float>();
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);
        
        for (uint32_t s = 0; s <= sectors; s++) {
            float phi = (float)s / (float)sectors * 2.0f * glm::pi<float>();
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);
            
            glm::vec3 pos(
                radius * sinTheta * cosPhi,
                radius * cosTheta,
                radius * sinTheta * sinPhi
            );
            
            glm::vec3 normal = glm::normalize(pos);
            glm::vec3 tangent(-sinPhi, 0, cosPhi);
            glm::vec2 uv((float)s / (float)sectors, (float)r / (float)rings);
            
            vertices.emplace_back(pos, normal, tangent, uv);
        }
    }
    
    // Generate indices
    for (uint32_t r = 0; r < rings; r++) {
        for (uint32_t s = 0; s < sectors; s++) {
            uint32_t first = r * (sectors + 1) + s;
            uint32_t second = first + sectors + 1;
            
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);
            
            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
    
    auto mesh = std::make_unique<D3D12Mesh>();
    if (!mesh->Create(backend, vertices, indices, "Sphere")) {
        return nullptr;
    }
    
    return mesh;
}

std::unique_ptr<D3D12Mesh> MeshGenerator::CreatePlane(DX12RenderBackend* backend) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    // Simple quad on XZ plane
    vertices.emplace_back(glm::vec3(-0.5f, 0, -0.5f), glm::vec3(0, 1, 0), glm::vec3(1, 0, 0), glm::vec2(0, 0));
    vertices.emplace_back(glm::vec3(0.5f, 0, -0.5f), glm::vec3(0, 1, 0), glm::vec3(1, 0, 0), glm::vec2(1, 0));
    vertices.emplace_back(glm::vec3(0.5f, 0, 0.5f), glm::vec3(0, 1, 0), glm::vec3(1, 0, 0), glm::vec2(1, 1));
    vertices.emplace_back(glm::vec3(-0.5f, 0, 0.5f), glm::vec3(0, 1, 0), glm::vec3(1, 0, 0), glm::vec2(0, 1));
    
    indices = {0, 1, 2, 0, 2, 3};
    
    auto mesh = std::make_unique<D3D12Mesh>();
    if (!mesh->Create(backend, vertices, indices, "Plane")) {
        return nullptr;
    }
    
    return mesh;
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
