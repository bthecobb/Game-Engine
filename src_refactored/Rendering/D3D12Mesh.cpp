#ifdef _WIN32
#include "Rendering/D3D12Mesh.h"
#include "Rendering/Backends/DX12RenderBackend.h"
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <cstring>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace CudaGame {
namespace Rendering {

D3D12Mesh::~D3D12Mesh() {
    m_meshlets.Release();
    // GPUBuffer destructors will release resources
}

bool D3D12Mesh::Create(DX12RenderBackend* backend,
                       const std::vector<Vertex>& vertices,
                       const std::vector<uint32_t>& indices,
                       const std::string& name) {
    m_name = name;
    
    ID3D12Device* device = backend->GetDevice();
    
    // Compute Bounds
    if (!vertices.empty()) {
        boundsMin = vertices[0].position;
        boundsMax = vertices[0].position;
        for (const auto& v : vertices) {
            boundsMin = glm::min(boundsMin, v.position);
            boundsMax = glm::max(boundsMax, v.position);
        }
    }
    
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

bool D3D12Mesh::GenerateMeshlets(DX12RenderBackend* backend, 
                                const std::vector<Vertex>& vertices,
                                const std::vector<uint32_t>& indices) {
    if (vertices.empty() || indices.empty()) return false;
    
    std::cout << "[Mesh] Generating meshlets for " << m_name << "..." << std::endl;
    std::cout << "[Mesh] sizeof(Meshlet) = " << sizeof(Meshlet) << " bytes" << std::endl;
    
    // 1. Prepare data for generator
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> colors;
    
    positions.reserve(vertices.size());
    normals.reserve(vertices.size());
    uvs.reserve(vertices.size());
    colors.reserve(vertices.size());
    
    for (const auto& v : vertices) {
        positions.push_back(v.position);
        normals.push_back(v.normal);
        uvs.push_back(v.texcoord);
        colors.push_back(v.color);
    }
    
    // 2. Generate meshlets
    MeshletMesh meshInfo = MeshletGenerator::Generate(positions, normals, indices);
    m_meshlets.meshletCount = (uint32_t)meshInfo.meshlets.size();
    
    std::cout << "[Mesh] Generated " << m_meshlets.meshletCount << " meshlets for " << m_name << std::endl;
    
    ID3D12Device* device = backend->GetDevice();
    
    // Helper to create and upload buffer
    auto CreateUploadBuffer = [&](const void* data, size_t size, const std::wstring& name) -> ID3D12Resource* {
        if (size == 0) return nullptr;
        
        ID3D12Resource* buffer = nullptr;
        D3D12_HEAP_PROPERTIES uploadHeapProps = {};
        uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        uploadHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        uploadHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        uploadHeapProps.CreationNodeMask = 1;
        uploadHeapProps.VisibleNodeMask = 1;
        
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Alignment = 0;
        bufferDesc.Width = size;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        
        HRESULT hr = device->CreateCommittedResource(
            &uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buffer)
        );
        
        if (FAILED(hr)) {
            std::cerr << "[Mesh] Failed to create meshlet buffer: " << std::string(name.begin(), name.end()) << std::endl;
            return nullptr;
        }
        
        buffer->SetName(name.c_str());
        
        void* mapped = nullptr;
        D3D12_RANGE readRange = {0, 0};
        if (SUCCEEDED(buffer->Map(0, &readRange, &mapped))) {
            memcpy(mapped, data, size);
            buffer->Unmap(0, nullptr);
        }
        
        return buffer;
    };
    
    std::wstring wname = std::wstring(m_name.begin(), m_name.end());
    
    // 3. Create buffers
    // Meshlets
    m_meshlets.meshlets = CreateUploadBuffer(meshInfo.meshlets.data(), 
        meshInfo.meshlets.size() * sizeof(Meshlet), L"Meshlets_" + wname);
        
    // Unique Vertex Indices (indices into the attribute arrays)
    m_meshlets.vertexIndices = CreateUploadBuffer(meshInfo.vertexIndices.data(), 
        meshInfo.vertexIndices.size() * sizeof(uint32_t), L"MeshletVertices_" + wname);
        
    // Primitive Indices (packed)
    m_meshlets.primitives = CreateUploadBuffer(meshInfo.primitiveIndices.data(), 
        meshInfo.primitiveIndices.size() * sizeof(uint8_t), L"MeshletPrims_" + wname);
    
    // Meshlet Bounds
    std::vector<MeshletBounds> bounds;
    bounds.reserve(meshInfo.meshlets.size());
    for (const auto& m : meshInfo.meshlets) {
        MeshletBounds b;
        b.sphere = glm::vec4(m.boundingSphereCenter, m.boundingSphereRadius);
        b.cone = glm::vec4(m.coneAxis, cos(m.coneAngle));
        bounds.push_back(b);
    }
    m_meshlets.bounds = CreateUploadBuffer(bounds.data(), 
        bounds.size() * sizeof(MeshletBounds), L"MeshletBounds_" + wname);
        
    // Attribute Buffers
    m_meshlets.positions = CreateUploadBuffer(positions.data(), positions.size() * sizeof(glm::vec3), L"Positions_" + wname);
    m_meshlets.normals = CreateUploadBuffer(normals.data(), normals.size() * sizeof(glm::vec3), L"Normals_" + wname);
    m_meshlets.uvs = CreateUploadBuffer(uvs.data(), uvs.size() * sizeof(glm::vec2), L"UVs_" + wname);
    m_meshlets.colors = CreateUploadBuffer(colors.data(), colors.size() * sizeof(glm::vec4), L"Colors_" + wname);
    
    // Instance Data Buffer (Single instance, updateable)
    struct InstanceData {
        glm::mat4 world;
        glm::mat4 normal;
        uint32_t meshletOffset;
        uint32_t meshletCount;
        uint32_t pad[2];
    };
    
    InstanceData inst;
    inst.world = transform;
    inst.normal = glm::transpose(glm::inverse(transform));
    inst.meshletOffset = 0;
    inst.meshletCount = m_meshlets.meshletCount;
    inst.pad[0] = 0; inst.pad[1] = 0;
    
    m_meshlets.instances = CreateUploadBuffer(&inst, sizeof(InstanceData), L"Instance_" + wname);
    
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
    
    // Generate meshlets for mesh shader pipeline
    mesh->GenerateMeshlets(backend, vertices, indices);
    
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
    
    mesh->GenerateMeshlets(backend, vertices, indices);
    
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
    
    mesh->GenerateMeshlets(backend, vertices, indices);
    
    return mesh;
}

std::unique_ptr<D3D12Mesh> MeshGenerator::CreateSkinnedBlockMesh(DX12RenderBackend* backend, std::shared_ptr<Animation::Skeleton> skeleton) {
    if (!skeleton) return nullptr;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    // Base Cube Vertices (Bone Local Space)
    const glm::vec3 rawPos[8] = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}
    };
    const glm::vec3 faceNormals[6] = {
        {0,0,-1}, {0,0,1}, {-1,0,0}, {1,0,0}, {0,1,0}, {0,-1,0}
    };
    const uint32_t faceIndices[6][4] = {
        {0,1,2,3}, {5,4,7,6}, {4,0,3,7}, {1,5,6,2}, {3,2,6,7}, {4,5,1,0}
    };
    const glm::vec2 uvs[4] = {{0,0}, {1,0}, {1,1}, {0,1}};

    for (int i = 0; i < skeleton->bones.size(); ++i) {
        const auto& bone = skeleton->bones[i];
        glm::mat4 bindMatrix = glm::inverse(bone.inverseBindPose);
        
        glm::vec3 boxScale(0.1f);
        if (bone.name.find("Hips") != std::string::npos) boxScale = glm::vec3(0.3f, 0.2f, 0.2f);
        else if (bone.name.find("Spine") != std::string::npos) boxScale = glm::vec3(0.25f, 0.4f, 0.2f);
        else if (bone.name.find("Head") != std::string::npos) boxScale = glm::vec3(0.2f, 0.25f, 0.2f);
        else if (bone.name.find("Leg") != std::string::npos) boxScale = glm::vec3(0.12f, 0.4f, 0.12f);
        else if (bone.name.find("Arm") != std::string::npos) boxScale = glm::vec3(0.1f, 0.35f, 0.1f);
        else if (bone.name.find("Foot") != std::string::npos) boxScale = glm::vec3(0.1f, 0.1f, 0.25f);
        
        for (int f = 0; f < 6; ++f) {
            for (int k = 0; k < 4; ++k) {
                uint32_t rawIdx = faceIndices[f][k];
                glm::vec3 localPos = rawPos[rawIdx] * boxScale;
                glm::vec4 modelPos4 = bindMatrix * glm::vec4(localPos, 1.0f);
                glm::vec3 modelPos = glm::vec3(modelPos4);
                
                glm::vec3 normal = glm::mat3(bindMatrix) * faceNormals[f];
                
                Vertex v;
                v.position = modelPos;
                v.normal = glm::normalize(normal);
                v.texcoord = uvs[k];
                
                v.boneIndices = glm::ivec4(i, 0, 0, 0);
                v.boneWeights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                vertices.push_back(v);
            }
            uint32_t base = (uint32_t)vertices.size() - 4;
            indices.push_back(base + 0); indices.push_back(base + 1); indices.push_back(base + 2);
            indices.push_back(base + 0); indices.push_back(base + 2); indices.push_back(base + 3);
        }
    }
    
    auto mesh = std::make_unique<D3D12Mesh>();
    mesh->SetSkeleton(skeleton);
    if (!mesh->CreateSkinned(backend, vertices, indices, "ProceduralCharacter")) {
        return nullptr;
    }
    return mesh;
}

void D3D12Mesh::UpdateGPUInstanceData() {
    if (!m_meshlets.instances) return;
    
    // Instance Data Buffer (Single instance)
    struct InstanceData {
        glm::mat4 world;
        glm::mat4 normal;
        uint32_t meshletOffset;
        uint32_t meshletCount;
        uint32_t pad[2];
    };
    
    InstanceData inst;
    inst.world = transform;
    inst.normal = glm::transpose(glm::inverse(transform));
    inst.meshletOffset = 0;
    inst.meshletCount = m_meshlets.meshletCount;
    inst.pad[0] = 0; inst.pad[1] = 0;
    
    // Map and Copy (assuming upload heap)
    void* mapped = nullptr;
    D3D12_RANGE readRange = {0, 0};
    if (SUCCEEDED(m_meshlets.instances->Map(0, &readRange, &mapped))) {
        memcpy(mapped, &inst, sizeof(InstanceData));
        m_meshlets.instances->Unmap(0, nullptr);
    }
}


// Convert Assimp Matrix to GLM
inline glm::mat4 AssimpToGLM(const aiMatrix4x4& from) {
    glm::mat4 to;
    to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
    to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
    to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
    to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
    return to;
}

// Helper to recursively build skeleton
void ProcessNode(aiNode* node, int parentIndex, Animation::Skeleton& skeleton) {
    std::string nodeName = node->mName.C_Str();
    
    int myIndex = -1;
    if (skeleton.boneNameToIndex.find(nodeName) == skeleton.boneNameToIndex.end()) {
        Animation::Skeleton::Bone newBone;
        newBone.name = nodeName;
        newBone.parentIndex = parentIndex;
        newBone.localTransform = AssimpToGLM(node->mTransformation);
        newBone.inverseBindPose = glm::mat4(1.0f); 
        
        myIndex = (int)skeleton.bones.size();
        skeleton.bones.push_back(newBone);
        skeleton.boneNameToIndex[nodeName] = myIndex;
    } else {
        myIndex = skeleton.boneNameToIndex[nodeName];
    }
    
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        ProcessNode(node->mChildren[i], myIndex, skeleton);
    }
}

bool D3D12Mesh::CreateSkinned(DX12RenderBackend* backend,
                              const std::vector<Vertex>& vertices,
                              const std::vector<uint32_t>& indices,
                              const std::string& name) {
    m_name = name;
    ID3D12Device* device = backend->GetDevice();
    
    // Create Vertex Buffer (Skinned - Unified Format)
    {
        // Using unified Vertex struct now
        const UINT64 vertexBufferSize = vertices.size() * sizeof(Vertex);
        
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
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        
        HRESULT hr = device->CreateCommittedResource(
            &uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_vertexBuffer.resource));
            
        if (FAILED(hr)) return false;
        
        void* pData;
        m_vertexBuffer.resource->Map(0, nullptr, &pData);
        memcpy(pData, vertices.data(), vertexBufferSize);
        m_vertexBuffer.resource->Unmap(0, nullptr);
        
        m_vertexBuffer.vertexBufferView.BufferLocation = m_vertexBuffer.resource->GetGPUVirtualAddress();
        m_vertexBuffer.vertexBufferView.StrideInBytes = sizeof(Vertex); 
        m_vertexBuffer.vertexBufferView.SizeInBytes = (UINT)vertexBufferSize;
        m_vertexBuffer.elementCount = (uint32_t)vertices.size();
        m_vertexBuffer.stride = sizeof(Vertex);
        
        std::wstring wname = std::wstring(name.begin(), name.end()) + L"_SkinnedVB";
        m_vertexBuffer.resource->SetName(wname.c_str());
    }
    
    // Create Index Buffer (Reuse logic from Create)
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
        bufferDesc.Width = indexBufferSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        
        HRESULT hr = device->CreateCommittedResource(
            &uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_indexBuffer.resource));
            
        if (FAILED(hr)) return false;
        
        void* pData;
        m_indexBuffer.resource->Map(0, nullptr, &pData);
        memcpy(pData, indices.data(), indexBufferSize);
        m_indexBuffer.resource->Unmap(0, nullptr);
        
        m_indexBuffer.indexBufferView.BufferLocation = m_indexBuffer.resource->GetGPUVirtualAddress();
        m_indexBuffer.indexBufferView.Format = DXGI_FORMAT_R32_UINT;
        m_indexBuffer.indexBufferView.SizeInBytes = (UINT)indexBufferSize;
        m_indexBuffer.elementCount = (uint32_t)indices.size();
        m_indexBuffer.stride = sizeof(uint32_t);
        
        std::wstring wname = std::wstring(name.begin(), name.end()) + L"_IB";
        m_indexBuffer.resource->SetName(wname.c_str());
    }
    
    return true;
}

bool D3D12Mesh::LoadFromFile(DX12RenderBackend* backend, const std::string& filename, bool loadSkeleton) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filename, 
        aiProcess_Triangulate | 
        aiProcess_GenSmoothNormals | 
        aiProcess_CalcTangentSpace |
        aiProcess_ConvertToLeftHanded | 
        aiProcess_LimitBoneWeights);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "[D3D12Mesh] Assimp Error: " << importer.GetErrorString() << std::endl;
        return false;
    }

    // Load Skeleton if requested
    if (loadSkeleton && !m_skeleton) {
        m_skeleton = std::make_shared<Animation::Skeleton>();
        ProcessNode(scene->mRootNode, -1, *m_skeleton);
        std::cout << "[D3D12Mesh] Loaded skeleton with " << m_skeleton->bones.size() << " bones" << std::endl;
    }

    // Extract Animations
    if (scene->mNumAnimations > 0) {
        m_animations.clear();
        std::cout << "[D3D12Mesh] Found " << scene->mNumAnimations << " embedded animations." << std::endl;
        
        for (unsigned int a = 0; a < scene->mNumAnimations; a++) {
            aiAnimation* anim = scene->mAnimations[a];
            auto clip = std::make_shared<Animation::AnimationClip>();
            clip->name = anim->mName.C_Str();
            if (clip->name.empty()) clip->name = "anim_" + std::to_string(a);
            
            if (anim->mTicksPerSecond != 0)
                 clip->duration = (float)(anim->mDuration / anim->mTicksPerSecond);
            else
                 clip->duration = (float)anim->mDuration;

            clip->isLooping = true; 
            
            for (unsigned int i = 0; i < anim->mNumChannels; i++) {
                aiNodeAnim* channel = anim->mChannels[i];
                Animation::AnimationClip::Channel myChannel;
                myChannel.boneName = channel->mNodeName.C_Str();
                
                // Positions
                for (unsigned int k = 0; k < channel->mNumPositionKeys; k++) {
                    myChannel.times.push_back((float)(channel->mPositionKeys[k].mTime / (anim->mTicksPerSecond ? anim->mTicksPerSecond : 24.0)));
                    aiVector3D p = channel->mPositionKeys[k].mValue;
                    myChannel.positions.push_back(glm::vec3(p.x, p.y, p.z));
                }
                
                // Rotations
                for (unsigned int k = 0; k < channel->mNumRotationKeys; k++) {
                    aiQuaternion q = channel->mRotationKeys[k].mValue;
                    myChannel.rotations.push_back(glm::quat(q.w, q.x, q.y, q.z));
                }
                
                // Scales
                for (unsigned int k = 0; k < channel->mNumScalingKeys; k++) {
                    aiVector3D s = channel->mScalingKeys[k].mValue;
                    myChannel.scales.push_back(glm::vec3(s.x, s.y, s.z));
                }
                
                clip->channels.push_back(myChannel);
            }
            m_animations.push_back(clip);
            std::cout << "  - Loaded Animation: " << clip->name << " (" << clip->duration << "s)" << std::endl;
        }
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    bool hasBones = false;

    // Process all meshes in the scene and merge them (simplified for single-mesh characters)
    for (unsigned int m = 0; m < scene->mNumMeshes; m++) {
        aiMesh* mesh = scene->mMeshes[m];
        uint32_t vertexOffset = (uint32_t)vertices.size();

        if (mesh->HasBones()) {
            hasBones = true;
        }

        size_t startVert = vertices.size();
        vertices.resize(startVert + mesh->mNumVertices);

        // Extract Vertices
        // Extract Vertices
        // Extract Vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            Vertex& v = vertices[startVert + i];
            
            v.position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
            
            if (mesh->HasNormals())
                v.normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
            else v.normal = glm::vec3(0, 1, 0);
            
            if (mesh->HasTangentsAndBitangents())
                v.tangent = glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);
            else v.tangent = glm::vec3(1, 0, 0);
            
            if (mesh->mTextureCoords[0])
                v.texcoord = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            else
                v.texcoord = glm::vec2(0.0f, 0.0f);

            // Default color white/emissive 0
            v.color = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
            
            // Init Unified Bone Data
            v.boneIndices = glm::ivec4(0);
            v.boneWeights = glm::vec4(0.0f);
        }

        // Extract Bone Weights
        if (mesh->HasBones() && m_skeleton) {
            for (unsigned int i = 0; i < mesh->mNumBones; i++) {
                aiBone* bone = mesh->mBones[i];
                std::string boneName = bone->mName.C_Str();
                
                if (m_skeleton->boneNameToIndex.find(boneName) == m_skeleton->boneNameToIndex.end()) continue;
                
                int boneID = m_skeleton->boneNameToIndex[boneName];
                
                // Store inverse bind pose
                m_skeleton->bones[boneID].inverseBindPose = AssimpToGLM(bone->mOffsetMatrix);

                for (unsigned int w = 0; w < bone->mNumWeights; w++) {
                    uint32_t vertexID = startVert + bone->mWeights[w].mVertexId;
                    float weight = bone->mWeights[w].mWeight;
                    
                    Vertex& v = vertices[vertexID];
                    
                    if (v.boneWeights.x == 0.0f) {
                        v.boneIndices.x = boneID;
                        v.boneWeights.x = weight;
                    } else if (v.boneWeights.y == 0.0f) {
                        v.boneIndices.y = boneID;
                        v.boneWeights.y = weight;
                    } else if (v.boneWeights.z == 0.0f) {
                        v.boneIndices.z = boneID;
                        v.boneWeights.z = weight;
                    } else if (v.boneWeights.w == 0.0f) {
                        v.boneIndices.w = boneID;
                        v.boneWeights.w = weight;
                    }
                }
            }
        }
        
        // Indices
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(vertexOffset + face.mIndices[j]);
        }
    }
    
    // Normalize weights
    if (hasBones) {
        for (auto& v : vertices) {
            float total = v.boneWeights.x + v.boneWeights.y + v.boneWeights.z + v.boneWeights.w;
            if (total > 0.0f) {
                v.boneWeights /= total;
            }
        }
        return CreateSkinned(backend, vertices, indices, filename);
    } else {
        return Create(backend, vertices, indices, filename);
    }
}

bool D3D12Mesh::LoadAnimation(const std::string& filePath, Animation::Skeleton& outSkeleton, std::unique_ptr<Animation::AnimationClip>& outClip) {
    Assimp::Importer importer;
    // Basic flags for animation
    const aiScene* scene = importer.ReadFile(filePath, aiProcess_LimitBoneWeights | aiProcess_ConvertToLeftHanded);
    
    if (!scene || !scene->mRootNode) {
        std::cerr << "[D3D12Mesh] Assimp Error: " << importer.GetErrorString() << std::endl;
        return false;
    }
    
    // Build Skeleton from Hierarchy
    outSkeleton.bones.clear();
    outSkeleton.boneNameToIndex.clear();
    ProcessNode(scene->mRootNode, -1, outSkeleton);
    
    // Load Animations
    if (scene->mNumAnimations > 0) {
        aiAnimation* anim = scene->mAnimations[0];
        outClip = std::make_unique<Animation::AnimationClip>();
        outClip->name = anim->mName.C_Str();
        
        if (anim->mTicksPerSecond != 0)
             outClip->duration = (float)(anim->mDuration / anim->mTicksPerSecond);
        else
             outClip->duration = (float)anim->mDuration;

        outClip->isLooping = true; 
        
        for (unsigned int i = 0; i < anim->mNumChannels; i++) {
            aiNodeAnim* channel = anim->mChannels[i];
            Animation::AnimationClip::Channel myChannel;
            myChannel.boneName = channel->mNodeName.C_Str();
            
            // Positions
            for (unsigned int k = 0; k < channel->mNumPositionKeys; k++) {
                myChannel.times.push_back((float)(channel->mPositionKeys[k].mTime / (anim->mTicksPerSecond ? anim->mTicksPerSecond : 24.0)));
                aiVector3D p = channel->mPositionKeys[k].mValue;
                myChannel.positions.push_back(glm::vec3(p.x, p.y, p.z));
            }
            
            // Rotations
            for (unsigned int k = 0; k < channel->mNumRotationKeys; k++) {
                aiQuaternion q = channel->mRotationKeys[k].mValue;
                myChannel.rotations.push_back(glm::quat(q.w, q.x, q.y, q.z));
            }
            
            // Scales
            for (unsigned int k = 0; k < channel->mNumScalingKeys; k++) {
                aiVector3D s = channel->mScalingKeys[k].mValue;
                myChannel.scales.push_back(glm::vec3(s.x, s.y, s.z));
            }
            
            outClip->channels.push_back(myChannel);
        }
    }
    
    return true; 
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
