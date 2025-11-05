#include "Rendering/CudaBuildingGenerator.h"
#include <glad/glad.h>
#include <iostream>
#include <algorithm>
#include <cmath>

// CPU fallback implementation when CUDA is not available
// This file is compiled when ENABLE_CUDA is OFF

namespace CudaGame {
namespace Rendering {

CudaBuildingGenerator::CudaBuildingGenerator() {
    std::cout << "[CudaBuildingGenerator] Using CPU fallback (CUDA not available)" << std::endl;
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
    
    std::cout << "[CudaBuildingGenerator] Initialized with CPU fallback" << std::endl;
    m_initialized = true;
    return true;
}

void CudaBuildingGenerator::Shutdown() {
    if (!m_initialized) {
        return;
    }
    
    m_initialized = false;
    std::cout << "[CudaBuildingGenerator] CPU fallback shutdown complete" << std::endl;
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
    vertexData.reserve(mesh.positions.size() * 11);
    
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

// CPU-based simple box generation
void CudaBuildingGenerator::GenerateBaseGeometry(BuildingMesh& mesh, const BuildingStyle& style) {
    const float hw = style.baseWidth * 0.5f;
    const float hd = style.baseDepth * 0.5f;
    const float h = style.height;
    
    // Simple hash for color variation
    auto hash = [](uint32_t x) -> float {
        x ^= x >> 16;
        x *= 0x85ebca6b;
        x ^= x >> 13;
        x *= 0xc2b2ae35;
        x ^= x >> 16;
        return float(x) / float(0xFFFFFFFF);
    };
    
    // Generate 6 faces (front, back, left, right, top, bottom)
    struct Face {
        glm::vec3 verts[4];
        glm::vec3 normal;
    };
    
    Face faces[6] = {
        // Front (+Z)
        {{ glm::vec3(-hw, 0, hd), glm::vec3(hw, 0, hd), glm::vec3(hw, h, hd), glm::vec3(-hw, h, hd) }, glm::vec3(0, 0, 1) },
        // Back (-Z)
        {{ glm::vec3(hw, 0, -hd), glm::vec3(-hw, 0, -hd), glm::vec3(-hw, h, -hd), glm::vec3(hw, h, -hd) }, glm::vec3(0, 0, -1) },
        // Right (+X)
        {{ glm::vec3(hw, 0, hd), glm::vec3(hw, 0, -hd), glm::vec3(hw, h, -hd), glm::vec3(hw, h, hd) }, glm::vec3(1, 0, 0) },
        // Left (-X)
        {{ glm::vec3(-hw, 0, -hd), glm::vec3(-hw, 0, hd), glm::vec3(-hw, h, hd), glm::vec3(-hw, h, -hd) }, glm::vec3(-1, 0, 0) },
        // Top (+Y)
        {{ glm::vec3(-hw, h, -hd), glm::vec3(hw, h, -hd), glm::vec3(hw, h, hd), glm::vec3(-hw, h, hd) }, glm::vec3(0, 1, 0) },
        // Bottom (-Y)
        {{ glm::vec3(-hw, 0, hd), glm::vec3(hw, 0, hd), glm::vec3(hw, 0, -hd), glm::vec3(-hw, 0, -hd) }, glm::vec3(0, -1, 0) }
    };
    
    glm::vec2 uvs[4] = { glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1) };
    
    for (int faceIdx = 0; faceIdx < 6; ++faceIdx) {
        float colorVar = hash(style.seed + faceIdx);
        glm::vec3 vertColor = style.baseColor + glm::vec3((colorVar - 0.5f) * 0.1f);
        
        for (int i = 0; i < 4; ++i) {
            mesh.positions.push_back(faces[faceIdx].verts[i]);
            mesh.normals.push_back(faces[faceIdx].normal);
            mesh.uvs.push_back(uvs[i]);
            mesh.colors.push_back(vertColor);
        }
        
        // Two triangles per face
        int baseVert = faceIdx * 4;
        mesh.indices.push_back(baseVert + 0);
        mesh.indices.push_back(baseVert + 1);
        mesh.indices.push_back(baseVert + 2);
        mesh.indices.push_back(baseVert + 0);
        mesh.indices.push_back(baseVert + 2);
        mesh.indices.push_back(baseVert + 3);
    }
    
    // Calculate bounding box
    mesh.boundsMin = glm::vec3(FLT_MAX);
    mesh.boundsMax = glm::vec3(-FLT_MAX);
    for (const auto& pos : mesh.positions) {
        mesh.boundsMin = glm::min(mesh.boundsMin, pos);
        mesh.boundsMax = glm::max(mesh.boundsMax, pos);
    }
}

void CudaBuildingGenerator::GenerateWindows(BuildingMesh&, const BuildingStyle&) {
    // Windows in texture for low-poly
}

void CudaBuildingGenerator::GenerateRoof(BuildingMesh&, const BuildingStyle&) {
    // Part of base geometry
}

void CudaBuildingGenerator::GenerateDetails(BuildingMesh&, const BuildingStyle&) {
    // Minimal for low-poly
}

void CudaBuildingGenerator::GenerateFacadeTexture(BuildingTexture& texture, const BuildingStyle& style) {
    int textureSize = texture.width * texture.height * 4;
    texture.albedoData.resize(textureSize);
    texture.normalData.resize(textureSize);
    texture.metallicRoughnessAO.resize(textureSize);
    texture.emissiveData.resize(textureSize);
    
    auto hash = [](uint32_t x) -> float {
        x ^= x >> 16;
        x *= 0x85ebca6b;
        x ^= x >> 13;
        x *= 0xc2b2ae35;
        x ^= x >> 16;
        return float(x) / float(0xFFFFFFFF);
    };
    
    float floors = style.height / 3.0f;
    int floorCount = int(floors);
    
    for (int y = 0; y < texture.height; ++y) {
        for (int x = 0; x < texture.width; ++x) {
            int pixelIdx = (y * texture.width + x) * 4;
            
            float u = float(x) / float(texture.width);
            float v = float(y) / float(texture.height);
            
            int windowRow = int(v * floorCount * style.windowRowsPerFloor);
            int windowCol = int(u * style.windowsPerRow);
            
            float windowCenterU = (windowCol + 0.5f) / style.windowsPerRow;
            float windowCenterV = (windowRow + 0.5f) / (floorCount * style.windowRowsPerFloor);
            
            float distU = std::abs(u - windowCenterU) * style.windowsPerRow;
            float distV = std::abs(v - windowCenterV) * (floorCount * style.windowRowsPerFloor);
            
            bool isWindow = (distU < style.windowSize * 0.5f) && (distV < style.windowSize * 0.5f);
            
            uint32_t seed = style.seed + windowRow * 1000 + windowCol;
            float rnd = hash(seed);
            
            if (isWindow) {
                float brightness = 0.3f + rnd * 0.4f;
                texture.albedoData[pixelIdx + 0] = uint8_t(brightness * style.accentColor.x * 255);
                texture.albedoData[pixelIdx + 1] = uint8_t(brightness * style.accentColor.y * 255);
                texture.albedoData[pixelIdx + 2] = uint8_t(brightness * style.accentColor.z * 255);
                texture.albedoData[pixelIdx + 3] = 255;
                
                texture.metallicRoughnessAO[pixelIdx + 0] = 200;
                texture.metallicRoughnessAO[pixelIdx + 1] = 50;
                texture.metallicRoughnessAO[pixelIdx + 2] = 255;
                
                // Emissive: RGB color of window light, A = intensity (0..255)
                texture.emissiveData[pixelIdx + 0] = uint8_t(style.accentColor.x * 255);
                texture.emissiveData[pixelIdx + 1] = uint8_t(style.accentColor.y * 255);
                texture.emissiveData[pixelIdx + 2] = uint8_t(style.accentColor.z * 255);
                // vary intensity a bit per-window
                texture.emissiveData[pixelIdx + 3] = uint8_t(200 + rnd * 55);
            } else {
                texture.albedoData[pixelIdx + 0] = uint8_t((style.baseColor.x + (rnd - 0.5f) * 0.05f) * 255);
                texture.albedoData[pixelIdx + 1] = uint8_t((style.baseColor.y + (rnd - 0.5f) * 0.05f) * 255);
                texture.albedoData[pixelIdx + 2] = uint8_t((style.baseColor.z + (rnd - 0.5f) * 0.05f) * 255);
                texture.albedoData[pixelIdx + 3] = 255;
                
                texture.metallicRoughnessAO[pixelIdx + 0] = uint8_t(style.metallic * 255);
                texture.metallicRoughnessAO[pixelIdx + 1] = uint8_t(style.roughness * 255);
                texture.metallicRoughnessAO[pixelIdx + 2] = 255;
                
                // Non-window: no emissive
                texture.emissiveData[pixelIdx + 0] = 0;
                texture.emissiveData[pixelIdx + 1] = 0;
                texture.emissiveData[pixelIdx + 2] = 0;
                texture.emissiveData[pixelIdx + 3] = 0;
            }
            
            texture.normalData[pixelIdx + 0] = 128;
            texture.normalData[pixelIdx + 1] = 128;
            texture.normalData[pixelIdx + 2] = 255;
            texture.normalData[pixelIdx + 3] = 255;
        }
    }
}

void CudaBuildingGenerator::GenerateWindowTexture(BuildingTexture&, const BuildingStyle&) {
    // Combined with facade
}

void CudaBuildingGenerator::GenerateMaterialMaps(BuildingTexture&, const BuildingStyle&) {
    // Combined with facade
}

void CudaBuildingGenerator::SimplifyMesh(const BuildingMesh& source, BuildingMesh& target, float targetReduction) {
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
