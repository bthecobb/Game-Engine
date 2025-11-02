#include "Rendering/CudaBuildingGenerator.h"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <algorithm>

// External C functions from .cu file
extern "C" {
    void LaunchBuildingGeometryKernel(void* vertices, void* indices, const void* styleData, int* vertexCount, int* indexCount);
    void LaunchBuildingTextureKernel(void* albedo, void* normal, void* material, int width, int height, const void* styleData);
    void LaunchMeshSimplificationKernel(const void* srcVerts, const void* srcIdxs, void* dstVerts, void* dstIdxs, 
                                       int srcVertCount, int srcIdxCount, float reduction, int* outCounts);
}

namespace CudaGame {
namespace Rendering {

// Helper structure matching CUDA kernel expectations
struct BuildingStyleGPU {
    int type;
    float baseWidth;
    float baseDepth;
    float height;
    float taperFactor;
    int windowRowsPerFloor;
    int windowsPerRow;
    float windowSize;
    float windowInset;
    bool hasRoof;
    float roofHeight;
    bool flatRoof;
    struct { float x, y, z; } baseColor;
    struct { float x, y, z; } accentColor;
    float metallic;
    float roughness;
    uint32_t seed;
};

// Helper to convert BuildingStyle to GPU format
BuildingStyleGPU ConvertToGPUStyle(const BuildingStyle& style) {
    BuildingStyleGPU gpuStyle;
    gpuStyle.type = static_cast<int>(style.type);
    gpuStyle.baseWidth = style.baseWidth;
    gpuStyle.baseDepth = style.baseDepth;
    gpuStyle.height = style.height;
    gpuStyle.taperFactor = style.taperFactor;
    gpuStyle.windowRowsPerFloor = style.windowRowsPerFloor;
    gpuStyle.windowsPerRow = style.windowsPerRow;
    gpuStyle.windowSize = style.windowSize;
    gpuStyle.windowInset = style.windowInset;
    gpuStyle.hasRoof = style.hasRoof;
    gpuStyle.roofHeight = style.roofHeight;
    gpuStyle.flatRoof = style.flatRoof;
    gpuStyle.baseColor = {style.baseColor.x, style.baseColor.y, style.baseColor.z};
    gpuStyle.accentColor = {style.accentColor.x, style.accentColor.y, style.accentColor.z};
    gpuStyle.metallic = style.metallic;
    gpuStyle.roughness = style.roughness;
    gpuStyle.seed = style.seed;
    return gpuStyle;
}

CudaBuildingGenerator::CudaBuildingGenerator() {
}

CudaBuildingGenerator::~CudaBuildingGenerator() {
    if (m_initialized) {
        Shutdown();
    }
}

bool CudaBuildingGenerator::Initialize() {
    if (m_initialized) {
        return true;
    }
    
    // Initialize CUDA device
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CudaBuildingGenerator] No CUDA devices found or CUDA error: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Select device 0
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "[CudaBuildingGenerator] Failed to set CUDA device: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[CudaBuildingGenerator] Using CUDA device: " << prop.name << std::endl;
    
    // Allocate memory pools
    m_vertexPoolSize = sizeof(float) * 12 * 1000;  // ~1000 vertices worth
    m_indexPoolSize = sizeof(uint32_t) * 6000;      // ~6000 indices
    m_texturePoolSize = 512 * 512 * 4 * 3;         // 512x512 RGBA * 3 textures
    
    cudaMalloc(&m_deviceVertexPool, m_vertexPoolSize);
    cudaMalloc(&m_deviceIndexPool, m_indexPoolSize);
    cudaMalloc(&m_deviceTexturePool, m_texturePoolSize);
    
    m_initialized = true;
    std::cout << "[CudaBuildingGenerator] Initialized successfully" << std::endl;
    
    return true;
}

void CudaBuildingGenerator::Shutdown() {
    if (!m_initialized) {
        return;
    }
    
    if (m_deviceVertexPool) cudaFree(m_deviceVertexPool);
    if (m_deviceIndexPool) cudaFree(m_deviceIndexPool);
    if (m_deviceTexturePool) cudaFree(m_deviceTexturePool);
    
    m_deviceVertexPool = nullptr;
    m_deviceIndexPool = nullptr;
    m_deviceTexturePool = nullptr;
    
    cudaDeviceReset();
    m_initialized = false;
    
    std::cout << "[CudaBuildingGenerator] Shutdown complete" << std::endl;
}

BuildingMesh CudaBuildingGenerator::GenerateBuilding(const BuildingStyle& style) {
    if (!m_initialized) {
        std::cerr << "[CudaBuildingGenerator] Not initialized!" << std::endl;
        return BuildingMesh();
    }
    
    BuildingMesh mesh;
    GenerateBaseGeometry(mesh, style);
    
    return mesh;
}

std::vector<BuildingMesh> CudaBuildingGenerator::GenerateBuildingBatch(const std::vector<BuildingStyle>& styles) {
    std::vector<BuildingMesh> meshes;
    meshes.reserve(styles.size());
    
    for (const auto& style : styles) {
        meshes.push_back(GenerateBuilding(style));
    }
    
    return meshes;
}

BuildingTexture CudaBuildingGenerator::GenerateBuildingTexture(const BuildingStyle& style, int resolution) {
    BuildingTexture texture;
    texture.width = resolution;
    texture.height = resolution;
    
    GenerateFacadeTexture(texture, style);
    
    return texture;
}

void CudaBuildingGenerator::GenerateLODs(BuildingMesh& mesh, const std::vector<float>& lodDistances) {
    // For now, create simple LODs by reducing vertex count
    mesh.lodLevels.clear();
    mesh.lodIndices.clear();
    
    for (size_t i = 0; i < lodDistances.size(); ++i) {
        float reduction = (i + 1) * 0.3f;  // 30%, 60%, 90% reduction
        if (reduction >= 1.0f) reduction = 0.9f;
        
        BuildingMesh lodMesh;
        SimplifyMesh(mesh, lodMesh, reduction);
        
        LODLevel level;
        level.vertexCount = lodMesh.positions.size();
        level.indexCount = lodMesh.indices.size();
        level.distance = lodDistances[i];
        
        mesh.lodLevels.push_back(level);
        mesh.lodIndices.push_back(lodMesh.indices);
    }
}

void CudaBuildingGenerator::UploadToGPU(BuildingMesh& mesh) {
    // Create VAO
    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);
    
    // Create VBO
    glGenBuffers(1, &mesh.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    
    // Interleave vertex data
    std::vector<float> vertexData;
    vertexData.reserve(mesh.positions.size() * 11);  // pos(3) + normal(3) + uv(2) + color(3)
    
    for (size_t i = 0; i < mesh.positions.size(); ++i) {
        vertexData.push_back(mesh.positions[i].x);
        vertexData.push_back(mesh.positions[i].y);
        vertexData.push_back(mesh.positions[i].z);
        
        vertexData.push_back(mesh.normals[i].x);
        vertexData.push_back(mesh.normals[i].y);
        vertexData.push_back(mesh.normals[i].z);
        
        vertexData.push_back(mesh.uvs[i].x);
        vertexData.push_back(mesh.uvs[i].y);
        
        vertexData.push_back(mesh.colors[i].x);
        vertexData.push_back(mesh.colors[i].y);
        vertexData.push_back(mesh.colors[i].z);
    }
    
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    
    // Normal attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    
    // UV attribute
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(6 * sizeof(float)));
    
    // Color attribute
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(8 * sizeof(float)));
    
    // Create EBO
    glGenBuffers(1, &mesh.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(uint32_t), mesh.indices.data(), GL_STATIC_DRAW);
    
    glBindVertexArray(0);
}

void CudaBuildingGenerator::CleanupGPUMesh(BuildingMesh& mesh) {
    if (mesh.vao) glDeleteVertexArrays(1, &mesh.vao);
    if (mesh.vbo) glDeleteBuffers(1, &mesh.vbo);
    if (mesh.ebo) glDeleteBuffers(1, &mesh.ebo);
    
    mesh.vao = 0;
    mesh.vbo = 0;
    mesh.ebo = 0;
}

void CudaBuildingGenerator::GenerateBaseGeometry(BuildingMesh& mesh, const BuildingStyle& style) {
    // Convert style to GPU format
    BuildingStyleGPU gpuStyle = ConvertToGPUStyle(style);
    
    // Allocate device memory for geometry
    void* d_vertices = nullptr;
    void* d_indices = nullptr;
    int* d_vertexCount = nullptr;
    int* d_indexCount = nullptr;
    
    cudaMalloc(&d_vertices, 24 * 11 * sizeof(float));  // 24 verts max * 11 floats per vert
    cudaMalloc(&d_indices, 36 * sizeof(uint32_t));      // 36 indices max
    cudaMalloc(&d_vertexCount, sizeof(int));
    cudaMalloc(&d_indexCount, sizeof(int));
    
    // Launch CUDA kernel
    LaunchBuildingGeometryKernel(d_vertices, d_indices, &gpuStyle, d_vertexCount, d_indexCount);
    
    // Copy results back
    int vertexCount, indexCount;
    cudaMemcpy(&vertexCount, d_vertexCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&indexCount, d_indexCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Allocate host memory and copy vertex data
    std::vector<float> vertexData(vertexCount * 11);
    cudaMemcpy(vertexData.data(), d_vertices, vertexCount * 11 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Deinterleave into separate arrays
    mesh.positions.resize(vertexCount);
    mesh.normals.resize(vertexCount);
    mesh.uvs.resize(vertexCount);
    mesh.colors.resize(vertexCount);
    
    for (int i = 0; i < vertexCount; ++i) {
        int offset = i * 11;
        mesh.positions[i] = glm::vec3(vertexData[offset+0], vertexData[offset+1], vertexData[offset+2]);
        mesh.normals[i] = glm::vec3(vertexData[offset+3], vertexData[offset+4], vertexData[offset+5]);
        mesh.uvs[i] = glm::vec2(vertexData[offset+6], vertexData[offset+7]);
        mesh.colors[i] = glm::vec3(vertexData[offset+8], vertexData[offset+9], vertexData[offset+10]);
    }
    
    // Copy index data
    mesh.indices.resize(indexCount);
    cudaMemcpy(mesh.indices.data(), d_indices, indexCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Calculate bounding box
    mesh.boundsMin = glm::vec3(FLT_MAX);
    mesh.boundsMax = glm::vec3(-FLT_MAX);
    for (const auto& pos : mesh.positions) {
        mesh.boundsMin = glm::min(mesh.boundsMin, pos);
        mesh.boundsMax = glm::max(mesh.boundsMax, pos);
    }
    
    // Cleanup
    cudaFree(d_vertices);
    cudaFree(d_indices);
    cudaFree(d_vertexCount);
    cudaFree(d_indexCount);
}

void CudaBuildingGenerator::GenerateWindows(BuildingMesh&, const BuildingStyle&) {
    // Windows are generated procedurally in texture, not geometry for low-poly aesthetic
}

void CudaBuildingGenerator::GenerateRoof(BuildingMesh&, const BuildingStyle&) {
    // Roof is part of base geometry for simplicity
}

void CudaBuildingGenerator::GenerateDetails(BuildingMesh&, const BuildingStyle&) {
    // Minimal details for low-poly aesthetic
}

void CudaBuildingGenerator::GenerateFacadeTexture(BuildingTexture& texture, const BuildingStyle& style) {
    BuildingStyleGPU gpuStyle = ConvertToGPUStyle(style);
    
    int textureSize = texture.width * texture.height * 4;
    
    // Allocate device memory
    uint8_t* d_albedo = nullptr;
    uint8_t* d_normal = nullptr;
    uint8_t* d_material = nullptr;
    
    cudaMalloc(&d_albedo, textureSize);
    cudaMalloc(&d_normal, textureSize);
    cudaMalloc(&d_material, textureSize);
    
    // Launch texture generation kernel
    LaunchBuildingTextureKernel(d_albedo, d_normal, d_material, texture.width, texture.height, &gpuStyle);
    
    // Copy results back
    texture.albedoData.resize(textureSize);
    texture.normalData.resize(textureSize);
    texture.metallicRoughnessAO.resize(textureSize);
    
    cudaMemcpy(texture.albedoData.data(), d_albedo, textureSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(texture.normalData.data(), d_normal, textureSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(texture.metallicRoughnessAO.data(), d_material, textureSize, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_albedo);
    cudaFree(d_normal);
    cudaFree(d_material);
}

void CudaBuildingGenerator::GenerateWindowTexture(BuildingTexture&, const BuildingStyle&) {
    // Combined with facade texture
}

void CudaBuildingGenerator::GenerateMaterialMaps(BuildingTexture&, const BuildingStyle&) {
    // Combined with facade texture
}

void CudaBuildingGenerator::SimplifyMesh(const BuildingMesh& source, BuildingMesh& target, float targetReduction) {
    // Simple CPU-based simplification for now
    // In production, use proper edge collapse on GPU
    int stride = std::max(1, int(1.0f / (1.0f - targetReduction)));
    
    target.positions.clear();
    target.normals.clear();
    target.uvs.clear();
    target.colors.clear();
    target.indices.clear();
    
    for (size_t i = 0; i < source.positions.size(); i += stride) {
        target.positions.push_back(source.positions[i]);
        target.normals.push_back(source.normals[i]);
        target.uvs.push_back(source.uvs[i]);
        target.colors.push_back(source.colors[i]);
    }
    
    // Rebuild indices
    for (size_t i = 0; i < source.indices.size(); i += 3) {
        uint32_t i0 = source.indices[i] / stride;
        uint32_t i1 = source.indices[i+1] / stride;
        uint32_t i2 = source.indices[i+2] / stride;
        
        if (i0 < target.positions.size() && i1 < target.positions.size() && i2 < target.positions.size()) {
            target.indices.push_back(i0);
            target.indices.push_back(i1);
            target.indices.push_back(i2);
        }
    }
}

} // namespace Rendering
} // namespace CudaGame
