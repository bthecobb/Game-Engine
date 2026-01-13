#include "Rendering/CudaBuildingGenerator.h"
#include "Rendering/D3D12Mesh.h"
#include "Rendering/Backends/DX12RenderBackend.h"
#include <iostream>
#include <algorithm>
#include <cmath>

// DX12 implementation of CudaBuildingGenerator
// This file replaces OpenGL upload with D3D12 resource creation

namespace CudaGame {
namespace Rendering {

// Simple hash for procedural variation
static float hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return float(x) / float(0xFFFFFFFF);
}

CudaBuildingGenerator::CudaBuildingGenerator() {
    std::cout << "[CudaBuildingGenerator] DX12 version initialized" << std::endl;
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
    
    std::cout << "[CudaBuildingGenerator] Initialized for DX12" << std::endl;
    m_initialized = true;
    return true;
}

void CudaBuildingGenerator::Shutdown() {
    if (!m_initialized) {
        return;
    }
    
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

// CPU-based building geometry generation with subdivided walls for window patterns
void CudaBuildingGenerator::GenerateBaseGeometry(BuildingMesh& mesh, const BuildingStyle& style) {
    const float hw = style.baseWidth * 0.5f;
    const float hd = style.baseDepth * 0.5f;
    const float h = style.height;
    
    // Window grid parameters
    const int windowsPerRow = 6;
    const int floorsPerBuilding = std::max(1, static_cast<int>(h / 3.0f));  // ~3 units per floor
    const float windowMargin = 0.15f;
    
    // Helper to add a subdivided wall face with NON-SHARED vertices per cell
    // Each cell gets its own 4 unique vertices to avoid color interpolation
    auto addSubdividedWall = [&](const glm::vec3& corner0, const glm::vec3& corner1, 
                                  const glm::vec3& corner2, const glm::vec3& corner3,
                                  const glm::vec3& normal, int faceIdx) {
        int gridX = windowsPerRow;
        int gridY = floorsPerBuilding;
        
        // Generate each cell as a separate quad with unique vertices
        for (int cellY = 0; cellY < gridY; ++cellY) {
            for (int cellX = 0; cellX < gridX; ++cellX) {
                // Calculate corner UVs for this cell
                float u0 = static_cast<float>(cellX) / static_cast<float>(gridX);
                float u1 = static_cast<float>(cellX + 1) / static_cast<float>(gridX);
                float v0 = static_cast<float>(cellY) / static_cast<float>(gridY);
                float v1 = static_cast<float>(cellY + 1) / static_cast<float>(gridY);
                
                // Bilinear interpolation for 4 corner positions
                glm::vec3 pos00 = glm::mix(glm::mix(corner0, corner1, u0), glm::mix(corner3, corner2, u0), v0);
                glm::vec3 pos10 = glm::mix(glm::mix(corner0, corner1, u1), glm::mix(corner3, corner2, u1), v0);
                glm::vec3 pos11 = glm::mix(glm::mix(corner0, corner1, u1), glm::mix(corner3, corner2, u1), v1);
                glm::vec3 pos01 = glm::mix(glm::mix(corner0, corner1, u0), glm::mix(corner3, corner2, u0), v1);
                
                // ALL cells are windows (lit or dark) - no edge wall border
                glm::vec3 cellColor;
                glm::vec3 emissiveColor(0.0f);
                
                // Every cell is a window - determine if lit or dark
                uint32_t windowHash = style.seed + faceIdx * 1000 + cellY * 100 + cellX;
                float isLit = hash(windowHash);
                
                if (isLit > 0.45f) {
                    // Lit window (55% are lit)
                    float colorVar = hash(windowHash + 12345);
                    if (colorVar < 0.6f) {
                        cellColor = glm::vec3(1.0f, 0.95f, 0.75f);   // Warm yellow
                        emissiveColor = glm::vec3(1.0f, 0.9f, 0.6f) * 3.0f;
                    } else if (colorVar < 0.8f) {
                        cellColor = glm::vec3(0.75f, 0.88f, 1.0f);   // Cool blue
                        emissiveColor = glm::vec3(0.6f, 0.8f, 1.0f) * 2.5f;
                    } else {
                        cellColor = glm::vec3(0.75f, 1.0f, 0.8f);    // Green tint
                        emissiveColor = glm::vec3(0.6f, 1.0f, 0.7f) * 2.8f;
                    }
                } else {
                    // Dark window (45% are dark)
                    cellColor = glm::vec3(0.08f, 0.1f, 0.15f);
                }
                
                // Add 4 unique vertices for this cell (same color for all 4)
                int baseVertex = static_cast<int>(mesh.positions.size());
                
                mesh.positions.push_back(pos00);
                mesh.normals.push_back(normal);
                mesh.uvs.push_back(glm::vec2(u0, v0));
                mesh.colors.push_back(cellColor);
                mesh.emissive.push_back(emissiveColor);
                
                mesh.positions.push_back(pos10);
                mesh.normals.push_back(normal);
                mesh.uvs.push_back(glm::vec2(u1, v0));
                mesh.colors.push_back(cellColor);
                mesh.emissive.push_back(emissiveColor);
                
                mesh.positions.push_back(pos11);
                mesh.normals.push_back(normal);
                mesh.uvs.push_back(glm::vec2(u1, v1));
                mesh.colors.push_back(cellColor);
                mesh.emissive.push_back(emissiveColor);
                
                mesh.positions.push_back(pos01);
                mesh.normals.push_back(normal);
                mesh.uvs.push_back(glm::vec2(u0, v1));
                mesh.colors.push_back(cellColor);
                mesh.emissive.push_back(emissiveColor);
                
                // Two triangles for the quad
                mesh.indices.push_back(baseVertex + 0);
                mesh.indices.push_back(baseVertex + 1);
                mesh.indices.push_back(baseVertex + 2);
                mesh.indices.push_back(baseVertex + 0);
                mesh.indices.push_back(baseVertex + 2);
                mesh.indices.push_back(baseVertex + 3);
            }
        }
    };
    
    // Helper for simple 4-vertex faces (top/bottom - no windows)
    auto addSimpleFace = [&](const glm::vec3 verts[4], const glm::vec3& normal, bool flipWinding) {
        int baseVertex = static_cast<int>(mesh.positions.size());
        glm::vec2 uvs[4] = { {0,0}, {1,0}, {1,1}, {0,1} };
        
        for (int i = 0; i < 4; ++i) {
            mesh.positions.push_back(verts[i]);
            mesh.normals.push_back(normal);
            mesh.uvs.push_back(uvs[i]);
            mesh.colors.push_back(style.baseColor);
            mesh.emissive.push_back(glm::vec3(0.0f));
        }
        
        if (flipWinding) {
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 1);
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 3);
            mesh.indices.push_back(baseVertex + 2);
        } else {
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 1);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 3);
        }
    };
    
    const float edgeDrop = 0.02f;
    
    // Wall faces - subdivided for window patterns
    // Front (+Z)
    addSubdividedWall(
        glm::vec3(-hw, 0, hd), glm::vec3(hw, 0, hd),
        glm::vec3(hw, h - edgeDrop, hd), glm::vec3(-hw, h - edgeDrop, hd),
        glm::vec3(0, 0, 1), 0);
    
    // Back (-Z)
    addSubdividedWall(
        glm::vec3(hw, 0, -hd), glm::vec3(-hw, 0, -hd),
        glm::vec3(-hw, h - edgeDrop, -hd), glm::vec3(hw, h - edgeDrop, -hd),
        glm::vec3(0, 0, -1), 1);
    
    // Right (+X)
    addSubdividedWall(
        glm::vec3(hw, 0, hd), glm::vec3(hw, 0, -hd),
        glm::vec3(hw, h - edgeDrop, -hd), glm::vec3(hw, h - edgeDrop, hd),
        glm::vec3(1, 0, 0), 2);
    
    // Left (-X)
    addSubdividedWall(
        glm::vec3(-hw, 0, -hd), glm::vec3(-hw, 0, hd),
        glm::vec3(-hw, h - edgeDrop, hd), glm::vec3(-hw, h - edgeDrop, -hd),
        glm::vec3(-1, 0, 0), 3);
    
    // Top (+Y) - simple, no windows
    glm::vec3 topVerts[4] = {
        glm::vec3(-hw, h, -hd), glm::vec3(hw, h, -hd),
        glm::vec3(hw, h, hd), glm::vec3(-hw, h, hd)
    };
    addSimpleFace(topVerts, glm::vec3(0, 1, 0), true);
    
    // Bottom (-Y) - simple, no windows
    glm::vec3 botVerts[4] = {
        glm::vec3(-hw, 0, hd), glm::vec3(hw, 0, hd),
        glm::vec3(hw, 0, -hd), glm::vec3(-hw, 0, -hd)
    };
    addSimpleFace(botVerts, glm::vec3(0, -1, 0), false);
    
    // Calculate bounding box
    mesh.boundsMin = glm::vec3(FLT_MAX);
    mesh.boundsMax = glm::vec3(-FLT_MAX);
    for (const auto& pos : mesh.positions) {
        mesh.boundsMin = glm::min(mesh.boundsMin, pos);
        mesh.boundsMax = glm::max(mesh.boundsMax, pos);
    }
}

void CudaBuildingGenerator::GenerateWindows(BuildingMesh&, const BuildingStyle&) {
    // Windows are now generated in GenerateBaseGeometry via vertex colors
}

void CudaBuildingGenerator::GenerateRoof(BuildingMesh&, const BuildingStyle&) {
    // TODO: Add detailed roof geometry
}

void CudaBuildingGenerator::GenerateDetails(BuildingMesh&, const BuildingStyle&) {
    // TODO: Add architectural details (ledges, cornices, etc.)
}

void CudaBuildingGenerator::GenerateFacadeTexture(BuildingTexture& texture, const BuildingStyle& style) {
    // Generate procedural facade texture
    texture.albedoData.resize(texture.width * texture.height * 4);
    texture.normalData.resize(texture.width * texture.height * 4);
    texture.metallicRoughnessAO.resize(texture.width * texture.height * 4);
    texture.emissiveData.resize(texture.width * texture.height * 4);
    
    // Fill with facade pattern
    for (int y = 0; y < texture.height; ++y) {
        for (int x = 0; x < texture.width; ++x) {
            int idx = (y * texture.width + x) * 4;
            
            float u = float(x) / float(texture.width);
            float v = float(y) / float(texture.height);
            
            // Window grid
            int windowX = static_cast<int>(u * 6.0f);
            int windowY = static_cast<int>(v * 10.0f);
            float localU = fmodf(u * 6.0f, 1.0f);
            float localV = fmodf(v * 10.0f, 1.0f);
            
            bool isWindow = (localU > 0.15f && localU < 0.85f) && 
                           (localV > 0.15f && localV < 0.85f);
            
            uint32_t windowHash = style.seed + windowY * 100 + windowX;
            bool isLit = hash(windowHash) > 0.55f;
            
            if (isWindow) {
                if (isLit) {
                    // Lit window
                    texture.albedoData[idx + 0] = 255;
                    texture.albedoData[idx + 1] = 230;
                    texture.albedoData[idx + 2] = 150;
                    texture.albedoData[idx + 3] = 255;
                    
                    texture.emissiveData[idx + 0] = 255;
                    texture.emissiveData[idx + 1] = 230;
                    texture.emissiveData[idx + 2] = 150;
                    texture.emissiveData[idx + 3] = 200;
                } else {
                    // Dark window
                    texture.albedoData[idx + 0] = 20;
                    texture.albedoData[idx + 1] = 30;
                    texture.albedoData[idx + 2] = 40;
                    texture.albedoData[idx + 3] = 255;
                    
                    texture.emissiveData[idx + 0] = 0;
                    texture.emissiveData[idx + 1] = 0;
                    texture.emissiveData[idx + 2] = 0;
                    texture.emissiveData[idx + 3] = 0;
                }
                
                // Window material: reflective
                texture.metallicRoughnessAO[idx + 0] = 180;  // Metallic
                texture.metallicRoughnessAO[idx + 1] = 50;   // Roughness (smooth)
                texture.metallicRoughnessAO[idx + 2] = 255;  // AO
                texture.metallicRoughnessAO[idx + 3] = 255;
            } else {
                // Wall
                float variation = hash(style.seed + x + y * texture.width);
                uint8_t wallColor = static_cast<uint8_t>(150 + variation * 20);
                
                texture.albedoData[idx + 0] = wallColor;
                texture.albedoData[idx + 1] = static_cast<uint8_t>(wallColor * 0.95f);
                texture.albedoData[idx + 2] = static_cast<uint8_t>(wallColor * 0.9f);
                texture.albedoData[idx + 3] = 255;
                
                texture.emissiveData[idx + 0] = 0;
                texture.emissiveData[idx + 1] = 0;
                texture.emissiveData[idx + 2] = 0;
                texture.emissiveData[idx + 3] = 0;
                
                // Wall material: rough concrete
                texture.metallicRoughnessAO[idx + 0] = 10;   // Metallic
                texture.metallicRoughnessAO[idx + 1] = 200;  // Roughness
                texture.metallicRoughnessAO[idx + 2] = 255;  // AO
                texture.metallicRoughnessAO[idx + 3] = 255;
            }
            
            // Normal map (flat for now)
            texture.normalData[idx + 0] = 128;
            texture.normalData[idx + 1] = 128;
            texture.normalData[idx + 2] = 255;
            texture.normalData[idx + 3] = 255;
        }
    }
}

void CudaBuildingGenerator::GenerateWindowTexture(BuildingTexture&, const BuildingStyle&) {
    // Handled in GenerateFacadeTexture
}

void CudaBuildingGenerator::GenerateMaterialMaps(BuildingTexture&, const BuildingStyle&) {
    // Handled in GenerateFacadeTexture
}

void CudaBuildingGenerator::SimplifyMesh(const BuildingMesh& source, BuildingMesh& target, float targetReduction) {
    // Simple LOD: skip every N vertices
    int skipRate = static_cast<int>(1.0f / (1.0f - targetReduction));
    if (skipRate < 2) skipRate = 2;
    
    target.positions.clear();
    target.normals.clear();
    target.uvs.clear();
    target.colors.clear();
    target.emissive.clear();
    target.indices.clear();
    
    std::vector<int> indexMap(source.positions.size(), -1);
    int newIndex = 0;
    
    for (size_t i = 0; i < source.positions.size(); i += skipRate) {
        target.positions.push_back(source.positions[i]);
        target.normals.push_back(source.normals[i]);
        target.uvs.push_back(source.uvs[i]);
        if (i < source.colors.size()) target.colors.push_back(source.colors[i]);
        if (i < source.emissive.size()) target.emissive.push_back(source.emissive[i]);
        indexMap[i] = newIndex++;
    }
    
    // Rebuild indices
    for (size_t i = 0; i < source.indices.size(); i += 3) {
        int i0 = source.indices[i];
        int i1 = source.indices[i + 1];
        int i2 = source.indices[i + 2];
        
        // Find nearest valid vertex for each
        while (indexMap[i0] < 0 && i0 > 0) i0--;
        while (indexMap[i1] < 0 && i1 > 0) i1--;
        while (indexMap[i2] < 0 && i2 > 0) i2--;
        
        if (indexMap[i0] >= 0 && indexMap[i1] >= 0 && indexMap[i2] >= 0) {
            if (indexMap[i0] != indexMap[i1] && indexMap[i1] != indexMap[i2] && indexMap[i0] != indexMap[i2]) {
                target.indices.push_back(indexMap[i0]);
                target.indices.push_back(indexMap[i1]);
                target.indices.push_back(indexMap[i2]);
            }
        }
    }
}

} // namespace Rendering
} // namespace CudaGame
