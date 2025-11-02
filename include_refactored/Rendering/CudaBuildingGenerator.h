#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <memory>

namespace CudaGame {
namespace Rendering {

// Building style parameters for procedural generation
struct BuildingStyle {
    enum class Type {
        MODERN,           // Clean lines, large windows
        INDUSTRIAL,       // Heavy, mechanical features
        RESIDENTIAL,      // Smaller, cozy features
        OFFICE,           // Grid-like, uniform windows
        SKYSCRAPER        // Tall, tapered design
    };
    
    Type type = Type::MODERN;
    
    // Geometric parameters
    float baseWidth = 8.0f;
    float baseDepth = 8.0f;
    float height = 15.0f;
    float taperFactor = 0.0f;  // 0 = no taper, 1 = pyramid
    
    // Window parameters
    int windowRowsPerFloor = 2;
    int windowsPerRow = 4;
    float windowSize = 0.6f;
    float windowInset = 0.1f;
    
    // Roof parameters
    bool hasRoof = true;
    float roofHeight = 2.0f;
    bool flatRoof = false;
    
    // Material/Color
    glm::vec3 baseColor = glm::vec3(0.6f, 0.65f, 0.7f);
    glm::vec3 accentColor = glm::vec3(0.2f, 0.3f, 0.4f);
    float metallic = 0.4f;
    float roughness = 0.6f;
    
    // Seed for procedural variation
    uint32_t seed = 0;
};

// LOD level specification
struct LODLevel {
    int vertexCount;
    int indexCount;
    float distance;  // Distance threshold for this LOD
};

// Generated building mesh data
struct BuildingMesh {
    // Vertex data
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> colors;  // Vertex colors for variety
    
    // Index data
    std::vector<uint32_t> indices;
    
    // LOD meshes (optional)
    std::vector<LODLevel> lodLevels;
    std::vector<std::vector<uint32_t>> lodIndices;  // Index buffers for each LOD
    
    // OpenGL buffer IDs (after upload)
    uint32_t vao = 0;
    uint32_t vbo = 0;
    uint32_t ebo = 0;
    
    // Bounding box for culling
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
};

// Procedural texture data
struct BuildingTexture {
    int width = 512;
    int height = 512;
    std::vector<uint8_t> albedoData;      // RGBA
    std::vector<uint8_t> normalData;      // RGB (tangent space normals)
    std::vector<uint8_t> metallicRoughnessAO;  // R=metallic, G=roughness, B=AO
    
    uint32_t albedoTexture = 0;
    uint32_t normalTexture = 0;
    uint32_t metallicRoughnessTexture = 0;
};

// Main CUDA building generator class
class CudaBuildingGenerator {
public:
    CudaBuildingGenerator();
    ~CudaBuildingGenerator();
    
    // Initialize CUDA context and resources
    bool Initialize();
    void Shutdown();
    
    // Generate a single building
    BuildingMesh GenerateBuilding(const BuildingStyle& style);
    
    // Generate multiple buildings in parallel
    std::vector<BuildingMesh> GenerateBuildingBatch(const std::vector<BuildingStyle>& styles);
    
    // Generate procedural textures for a building
    BuildingTexture GenerateBuildingTexture(const BuildingStyle& style, int resolution = 512);
    
    // Generate LOD levels for an existing mesh
    void GenerateLODs(BuildingMesh& mesh, const std::vector<float>& lodDistances);
    
    // Upload mesh data to OpenGL
    void UploadToGPU(BuildingMesh& mesh);
    
    // Cleanup OpenGL resources
    void CleanupGPUMesh(BuildingMesh& mesh);
    
    // Performance settings
    void SetMaxConcurrentBuildings(int count) { m_maxConcurrentBuildings = count; }
    void EnableCUDAOpenGLInterop(bool enable) { m_useInterop = enable; }
    
private:
    // CUDA device management
    bool m_initialized = false;
    int m_cudaDevice = 0;
    bool m_useInterop = true;
    int m_maxConcurrentBuildings = 30;
    
    // CUDA memory pools
    void* m_deviceVertexPool = nullptr;
    void* m_deviceIndexPool = nullptr;
    void* m_deviceTexturePool = nullptr;
    size_t m_vertexPoolSize = 0;
    size_t m_indexPoolSize = 0;
    size_t m_texturePoolSize = 0;
    
    // Internal generation methods
    void GenerateBaseGeometry(BuildingMesh& mesh, const BuildingStyle& style);
    void GenerateWindows(BuildingMesh& mesh, const BuildingStyle& style);
    void GenerateRoof(BuildingMesh& mesh, const BuildingStyle& style);
    void GenerateDetails(BuildingMesh& mesh, const BuildingStyle& style);
    
    void GenerateFacadeTexture(BuildingTexture& texture, const BuildingStyle& style);
    void GenerateWindowTexture(BuildingTexture& texture, const BuildingStyle& style);
    void GenerateMaterialMaps(BuildingTexture& texture, const BuildingStyle& style);
    
    void SimplifyMesh(const BuildingMesh& source, BuildingMesh& target, float targetReduction);
    
    // CUDA kernel wrappers (implemented in .cu file)
    void LaunchGeometryKernel(void* vertices, void* indices, const BuildingStyle& style, int* outVertexCount, int* outIndexCount);
    void LaunchTextureKernel(void* textureData, int width, int height, const BuildingStyle& style);
    void LaunchSimplificationKernel(void* srcVertices, void* srcIndices, void* dstVertices, void* dstIndices, 
                                    int srcVertexCount, int srcIndexCount, float reduction, int* outCounts);
};

} // namespace Rendering
} // namespace CudaGame
