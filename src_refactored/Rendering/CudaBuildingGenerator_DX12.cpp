#include "Rendering/CudaBuildingGenerator.h"
#include "Rendering/D3D12Mesh.h"
#include "Rendering/Backends/DX12RenderBackend.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "Core/CudaCore.h"

// DX12 implementation of CudaBuildingGenerator
// This file replaces OpenGL upload with D3D12 resource creation
// AND implements the CUDA generation logic (previously in CudaBuildingGenerator.cpp)
// but adapted for the DX12 build target.

// External C functions from .cu file
extern "C" {
    void LaunchBuildingGeometryKernel(void* vertices, void* indices, const void* styleData, int* vertexCount, int* indexCount);
    void LaunchBuildingTextureKernel(void* albedo, void* normal, void* material, int width, int height, const void* styleData);
    void LaunchEmissiveTextureKernel(void* emissiveData, int width, int height, const void* styleData);
    void LaunchMeshSimplificationKernel(const void* srcVerts, const void* srcIdxs, void* dstVerts, void* dstIdxs, 
                                       int srcVertCount, int srcIndexCount, float reduction, int* outCounts);
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
static BuildingStyleGPU ConvertToGPUStyle(const BuildingStyle& style) {
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
    std::cout << "[CudaBuildingGenerator] DX12 version constructed" << std::endl;
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
    // Check if CudaCore has already initialized the device?
    // In this architecture, CudaCore usually handles device selection.
    // We'll trust that CudaCore::Initialize() was called by the render pipeline.
    // But we should verify we have a context or can create one.
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CudaBuildingGenerator] No CUDA devices found or CUDA error: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Attempt to set device 0. If CudaCore already set it, this is a no-op or harmless switch.
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "[CudaBuildingGenerator] Failed to set CUDA device: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[CudaBuildingGenerator] Using CUDA device for generation: " << prop.name << std::endl;
    
    // Allocate memory pools
    // These sizes are estimates; purely CPU-GPU logic here (no GL Interop)
    m_vertexPoolSize = sizeof(float) * 14 * 25000;  // Support larger batches
    m_indexPoolSize = sizeof(uint32_t) * 40000;
    m_texturePoolSize = 1024 * 1024 * 4 * 4;         // 1024x1024 RGBA * 4 textures
    
    cudaMalloc(&m_deviceVertexPool, m_vertexPoolSize);
    cudaMalloc(&m_deviceIndexPool, m_indexPoolSize);
    cudaMalloc(&m_deviceTexturePool, m_texturePoolSize);
    
    m_initialized = true;
    std::cout << "[CudaBuildingGenerator] Initialized successfully (DX12/CUDA Mode)" << std::endl;
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
    
    // Do NOT reset device here as it might be used by CudaCore/DX12 backend
    // cudaDeviceReset(); 
    
    m_initialized = false;
    std::cout << "[CudaBuildingGenerator] DX12 shutdown complete" << std::endl;
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
    mesh.lodLevels.clear();
    mesh.lodIndices.clear();
    
    for (size_t i = 0; i < lodDistances.size(); ++i) {
        float reduction = (i + 1) * 0.3f;
        if (reduction >= 1.0f) reduction = 0.9f;
        
        BuildingMesh lodMesh;
        SimplifyMesh(mesh, lodMesh, reduction);
        
        LODLevel level;
        level.vertexCount = static_cast<int>(lodMesh.positions.size());
        level.indexCount = static_cast<int>(lodMesh.indices.size());
        level.distance = lodDistances[i];
        
        mesh.lodLevels.push_back(level);
        mesh.lodIndices.push_back(lodMesh.indices);
    }
}

// DX12 version - does nothing (mesh creation handled separately)
void CudaBuildingGenerator::UploadToGPU(BuildingMesh& mesh) {
    // For DX12, we use CreateD3D12Mesh instead
    // This function is kept for API compatibility but does nothing
    (void)mesh;
}

void CudaBuildingGenerator::CleanupGPUMesh(BuildingMesh& mesh) {
    // OpenGL IDs are not used in DX12 version
    mesh.vao = 0;
    mesh.vbo = 0;
    mesh.ebo = 0;
}

// REAL CUDA IMPLEMENTATION (Ported from CudaBuildingGenerator.cpp)
void CudaBuildingGenerator::GenerateBaseGeometry(BuildingMesh& mesh, const BuildingStyle& style) {
    // Convert style to GPU format
    BuildingStyleGPU gpuStyle = ConvertToGPUStyle(style);
    
    // Allocate device memory for geometry
    // Note: In a real batch scenario, we would use pre-allocated pools. 
    // Here we alloc/free per building for simplicity and safety.
    void* d_vertices = nullptr;
    void* d_indices = nullptr;
    int* d_vertexCount = nullptr;
    int* d_indexCount = nullptr;
    
    cudaMalloc(&d_vertices, 24 * 14 * sizeof(float));  // 24 verts max * 14 floats per vert
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
    std::vector<float> vertexData(vertexCount * 14);
    cudaMemcpy(vertexData.data(), d_vertices, vertexCount * 14 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Deinterleave into separate arrays
    mesh.positions.resize(vertexCount);
    mesh.normals.resize(vertexCount);
    mesh.uvs.resize(vertexCount);
    mesh.colors.resize(vertexCount);
    mesh.emissive.resize(vertexCount);
    
    for (int i = 0; i < vertexCount; ++i) {
        int offset = i * 14;
        mesh.positions[i] = glm::vec3(vertexData[offset+0], vertexData[offset+1], vertexData[offset+2]);
        mesh.normals[i] = glm::vec3(vertexData[offset+3], vertexData[offset+4], vertexData[offset+5]);
        mesh.uvs[i] = glm::vec2(vertexData[offset+6], vertexData[offset+7]);
        mesh.colors[i] = glm::vec3(vertexData[offset+8], vertexData[offset+9], vertexData[offset+10]);
        mesh.emissive[i] = glm::vec3(vertexData[offset+11], vertexData[offset+12], vertexData[offset+13]);
    }
    
    // Copy index data
    mesh.indices.resize(indexCount);
    cudaMemcpy(mesh.indices.data(), d_indices, indexCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Post-process: enforce correct roof vertex normals and winding for top faces
    // (Ported from CudaBuildingGenerator.cpp)
    {
        // Determine top plane from generated mesh to avoid style/precision mismatch
        float maxY = -FLT_MAX;
        for (const auto& p : mesh.positions) maxY = std::max(maxY, p.y);
        const float epsMax = 0.02f;
        const float epsStyle = 0.01f;
        
        std::vector<uint8_t> isTop(mesh.positions.size(), 0);
        for (size_t vi = 0; vi < mesh.positions.size(); ++vi) {
            float y = mesh.positions[vi].y;
            if (std::abs(y - maxY) < epsMax || std::abs(y - style.height) < epsStyle) {
                isTop[vi] = 1;
            }
        }
        
        // Fix winding
        for (size_t ii = 0; ii + 2 < mesh.indices.size(); ii += 3) {
            uint32_t i0 = mesh.indices[ii + 0];
            uint32_t i1 = mesh.indices[ii + 1];
            uint32_t i2 = mesh.indices[ii + 2];
            if (i0 < mesh.positions.size() && i1 < mesh.positions.size() && i2 < mesh.positions.size()) {
                if (isTop[i0] && isTop[i1] && isTop[i2]) {
                    const glm::vec3& v0 = mesh.positions[i0];
                    const glm::vec3& v1 = mesh.positions[i1];
                    const glm::vec3& v2 = mesh.positions[i2];
                    glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                    if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z) || glm::length(n) < 1e-6f) {
                        // Degenerate
                    } else if (n.y < 0.0f) {
                        std::swap(mesh.indices[ii + 1], mesh.indices[ii + 2]);
                    }
                }
            }
        }
        
        // Fix normals logic removed: It incorrectly overwrites wall top normals because they share the same Y height as the roof.
        // CUDA kernel already generates correct face normals for duplicated vertices.

    }

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
    // Windows are now generated in GenerateBaseGeometry via vertex colors
}

void CudaBuildingGenerator::GenerateRoof(BuildingMesh&, const BuildingStyle&) {
    // Part of base geometry
}

void CudaBuildingGenerator::GenerateDetails(BuildingMesh&, const BuildingStyle&) {
    // Part of base geometry
}

// REAL CUDA IMPLEMENTATION (Ported)
void CudaBuildingGenerator::GenerateFacadeTexture(BuildingTexture& texture, const BuildingStyle& style) {
    BuildingStyleGPU gpuStyle = ConvertToGPUStyle(style);
    
    int textureSize = texture.width * texture.height * 4;
    
    // Allocate device memory
    uint8_t* d_albedo = nullptr;
    uint8_t* d_normal = nullptr;
    uint8_t* d_material = nullptr;
    uint8_t* d_emissive = nullptr;
    
    cudaMalloc(&d_albedo, textureSize);
    cudaMalloc(&d_normal, textureSize);
    cudaMalloc(&d_material, textureSize);
    cudaMalloc(&d_emissive, textureSize);
    
    // Launch texture generation kernels
    LaunchBuildingTextureKernel(d_albedo, d_normal, d_material, texture.width, texture.height, &gpuStyle);
    LaunchEmissiveTextureKernel(d_emissive, texture.width, texture.height, &gpuStyle);
    
    // Copy results back
    texture.albedoData.resize(textureSize);
    texture.normalData.resize(textureSize);
    texture.metallicRoughnessAO.resize(textureSize);
    texture.emissiveData.resize(textureSize);
    
    cudaMemcpy(texture.albedoData.data(), d_albedo, textureSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(texture.normalData.data(), d_normal, textureSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(texture.metallicRoughnessAO.data(), d_material, textureSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(texture.emissiveData.data(), d_emissive, textureSize, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_albedo);
    cudaFree(d_normal);
    cudaFree(d_material);
    cudaFree(d_emissive);
}

void CudaBuildingGenerator::GenerateWindowTexture(BuildingTexture&, const BuildingStyle&) {
    // Handled in GenerateFacadeTexture
}

void CudaBuildingGenerator::GenerateMaterialMaps(BuildingTexture&, const BuildingStyle&) {
    // Handled in GenerateFacadeTexture
}

// REAL CUDA IMPLEMENTATION (Ported - though simpler CPU version might suffice, we'll keep it simple here)
void CudaBuildingGenerator::SimplifyMesh(const BuildingMesh& source, BuildingMesh& target, float targetReduction) {
    // Use the simple CPU implementation for now to avoid complexity of moving data back and forth just for decimation
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
