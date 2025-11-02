#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

// Simple vector types for CUDA
struct float3 {
    float x, y, z;
    __device__ __host__ float3(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
};

struct float2 {
    float x, y;
    __device__ __host__ float2(float _x = 0, float _y = 0) : x(_x), y(_y) {}
};

// Building vertex structure
struct BuildingVertex {
    float3 position;
    float3 normal;
    float2 uv;
    float3 color;
};

// Building style parameters (matches C++ struct)
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
    float3 baseColor;
    float3 accentColor;
    float metallic;
    float roughness;
    uint32_t seed;
};

// Pseudo-random number generator for procedural variation
__device__ inline float hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return float(x) / float(0xFFFFFFFF);
}

__device__ inline float3 normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0f) {
        return float3(v.x / len, v.y / len, v.z / len);
    }
    return float3(0, 1, 0);
}

__device__ inline float3 cross(float3 a, float3 b) {
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Generate a low-poly building box with windows
__global__ void GenerateBuildingGeometryKernel(
    BuildingVertex* vertices,
    uint32_t* indices,
    const BuildingStyleGPU style,
    int* vertexCount,
    int* indexCount
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread generates one face of the building
    if (tid >= 6) return;  // 6 faces: 4 walls + top + bottom
    
    const float hw = style.baseWidth * 0.5f;
    const float hd = style.baseDepth * 0.5f;
    const float h = style.height;
    
    // Base vertex index for this face
    int baseVert = tid * 4;
    int baseIdx = tid * 6;
    
    // Generate vertices for each face
    float3 faceVerts[4];
    float3 faceNormal;
    float2 faceUVs[4] = {
        float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 1)
    };
    
    switch(tid) {
        case 0: // Front face (+Z)
            faceVerts[0] = float3(-hw, 0, hd);
            faceVerts[1] = float3(hw, 0, hd);
            faceVerts[2] = float3(hw, h, hd);
            faceVerts[3] = float3(-hw, h, hd);
            faceNormal = float3(0, 0, 1);
            break;
        case 1: // Back face (-Z)
            faceVerts[0] = float3(hw, 0, -hd);
            faceVerts[1] = float3(-hw, 0, -hd);
            faceVerts[2] = float3(-hw, h, -hd);
            faceVerts[3] = float3(hw, h, -hd);
            faceNormal = float3(0, 0, -1);
            break;
        case 2: // Right face (+X)
            faceVerts[0] = float3(hw, 0, hd);
            faceVerts[1] = float3(hw, 0, -hd);
            faceVerts[2] = float3(hw, h, -hd);
            faceVerts[3] = float3(hw, h, hd);
            faceNormal = float3(1, 0, 0);
            break;
        case 3: // Left face (-X)
            faceVerts[0] = float3(-hw, 0, -hd);
            faceVerts[1] = float3(-hw, 0, hd);
            faceVerts[2] = float3(-hw, h, hd);
            faceVerts[3] = float3(-hw, h, -hd);
            faceNormal = float3(-1, 0, 0);
            break;
        case 4: // Top face (+Y)
            faceVerts[0] = float3(-hw, h, -hd);
            faceVerts[1] = float3(hw, h, -hd);
            faceVerts[2] = float3(hw, h, hd);
            faceVerts[3] = float3(-hw, h, hd);
            faceNormal = float3(0, 1, 0);
            break;
        case 5: // Bottom face (-Y)
            faceVerts[0] = float3(-hw, 0, hd);
            faceVerts[1] = float3(hw, 0, hd);
            faceVerts[2] = float3(hw, 0, -hd);
            faceVerts[3] = float3(-hw, 0, -hd);
            faceNormal = float3(0, -1, 0);
            break;
    }
    
    // Add some procedural color variation
    float colorVar = hash(style.seed + tid);
    float3 vertColor = float3(
        style.baseColor.x + (colorVar - 0.5f) * 0.1f,
        style.baseColor.y + (colorVar - 0.5f) * 0.1f,
        style.baseColor.z + (colorVar - 0.5f) * 0.1f
    );
    
    // Write vertices
    for (int i = 0; i < 4; i++) {
        vertices[baseVert + i].position = faceVerts[i];
        vertices[baseVert + i].normal = faceNormal;
        vertices[baseVert + i].uv = faceUVs[i];
        vertices[baseVert + i].color = vertColor;
    }
    
    // Write indices (two triangles per face)
    indices[baseIdx + 0] = baseVert + 0;
    indices[baseIdx + 1] = baseVert + 1;
    indices[baseIdx + 2] = baseVert + 2;
    indices[baseIdx + 3] = baseVert + 0;
    indices[baseIdx + 4] = baseVert + 2;
    indices[baseIdx + 5] = baseVert + 3;
    
    // Update counts (only first thread)
    if (tid == 0) {
        *vertexCount = 24;  // 6 faces * 4 vertices
        *indexCount = 36;   // 6 faces * 6 indices
    }
}

// Generate procedural facade texture with windows
__global__ void GenerateBuildingTextureKernel(
    uint8_t* albedoData,
    uint8_t* normalData,
    uint8_t* materialData,
    int width,
    int height,
    const BuildingStyleGPU style
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = (y * width + x) * 4;  // RGBA
    
    // Calculate which floor and window we're in
    float floors = style.height / 3.0f;  // Assume 3m per floor
    int floorCount = int(floors);
    
    float u = float(x) / float(width);
    float v = float(y) / float(height);
    
    int windowRow = int(v * floorCount * style.windowRowsPerFloor);
    int windowCol = int(u * style.windowsPerRow);
    
    // Determine if we're in a window area
    float windowCenterU = (windowCol + 0.5f) / style.windowsPerRow;
    float windowCenterV = (windowRow + 0.5f) / (floorCount * style.windowRowsPerFloor);
    
    float distU = fabsf(u - windowCenterU) * style.windowsPerRow;
    float distV = fabsf(v - windowCenterV) * (floorCount * style.windowRowsPerFloor);
    
    bool isWindow = (distU < style.windowSize * 0.5f) && (distV < style.windowSize * 0.5f);
    
    // Procedural variation
    uint32_t seed = style.seed + windowRow * 1000 + windowCol;
    float rnd = hash(seed);
    
    if (isWindow) {
        // Window - darker, more reflective
        float brightness = 0.3f + rnd * 0.4f;
        albedoData[pixelIdx + 0] = uint8_t(brightness * style.accentColor.x * 255);
        albedoData[pixelIdx + 1] = uint8_t(brightness * style.accentColor.y * 255);
        albedoData[pixelIdx + 2] = uint8_t(brightness * style.accentColor.z * 255);
        albedoData[pixelIdx + 3] = 255;
        
        // Material: metallic window
        materialData[pixelIdx + 0] = 200;  // Metallic
        materialData[pixelIdx + 1] = 50;   // Low roughness (shiny)
        materialData[pixelIdx + 2] = 255;  // Full AO
    } else {
        // Wall - base color with slight variation
        albedoData[pixelIdx + 0] = uint8_t((style.baseColor.x + (rnd - 0.5f) * 0.05f) * 255);
        albedoData[pixelIdx + 1] = uint8_t((style.baseColor.y + (rnd - 0.5f) * 0.05f) * 255);
        albedoData[pixelIdx + 2] = uint8_t((style.baseColor.z + (rnd - 0.5f) * 0.05f) * 255);
        albedoData[pixelIdx + 3] = 255;
        
        // Material: non-metallic wall
        materialData[pixelIdx + 0] = uint8_t(style.metallic * 255);
        materialData[pixelIdx + 1] = uint8_t(style.roughness * 255);
        materialData[pixelIdx + 2] = 255;  // Full AO
    }
    
    // Normal map (flat for now, could add detail)
    normalData[pixelIdx + 0] = 128;  // X: 0.5 (neutral)
    normalData[pixelIdx + 1] = 128;  // Y: 0.5 (neutral)
    normalData[pixelIdx + 2] = 255;  // Z: 1.0 (pointing out)
    normalData[pixelIdx + 3] = 255;
}

// Simple mesh simplification (edge collapse)
__global__ void SimplifyMeshKernel(
    const BuildingVertex* srcVertices,
    const uint32_t* srcIndices,
    BuildingVertex* dstVertices,
    uint32_t* dstIndices,
    int srcVertexCount,
    int srcIndexCount,
    float reductionFactor,
    int* outVertexCount,
    int* outIndexCount
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simple implementation: keep every Nth vertex based on reduction factor
    // In production, use proper edge collapse algorithms
    int stride = int(1.0f / (1.0f - reductionFactor));
    if (stride < 1) stride = 1;
    
    if (tid < srcVertexCount && (tid % stride) == 0) {
        int dstIdx = tid / stride;
        dstVertices[dstIdx] = srcVertices[tid];
    }
    
    // Update indices accordingly
    // (Simplified - in production, rebuild topology)
    if (tid < srcIndexCount) {
        int newIdx = (srcIndices[tid] / stride);
        dstIndices[tid] = newIdx;
    }
    
    if (tid == 0) {
        *outVertexCount = (srcVertexCount + stride - 1) / stride;
        *outIndexCount = srcIndexCount;  // Simplified
    }
}

// C-linkage wrapper functions for calling from C++
extern "C" {

void LaunchBuildingGeometryKernel(
    void* vertices,
    void* indices,
    const void* styleData,
    int* vertexCount,
    int* indexCount
) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    
    const BuildingStyleGPU* style = static_cast<const BuildingStyleGPU*>(styleData);
    
    GenerateBuildingGeometryKernel<<<gridSize, blockSize>>>(
        static_cast<BuildingVertex*>(vertices),
        static_cast<uint32_t*>(indices),
        *style,
        vertexCount,
        indexCount
    );
    
    cudaDeviceSynchronize();
}

void LaunchBuildingTextureKernel(
    void* albedo,
    void* normal,
    void* material,
    int width,
    int height,
    const void* styleData
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    const BuildingStyleGPU* style = static_cast<const BuildingStyleGPU*>(styleData);
    
    GenerateBuildingTextureKernel<<<gridSize, blockSize>>>(
        static_cast<uint8_t*>(albedo),
        static_cast<uint8_t*>(normal),
        static_cast<uint8_t*>(material),
        width,
        height,
        *style
    );
    
    cudaDeviceSynchronize();
}

void LaunchMeshSimplificationKernel(
    const void* srcVerts,
    const void* srcIdxs,
    void* dstVerts,
    void* dstIdxs,
    int srcVertCount,
    int srcIdxCount,
    float reduction,
    int* outCounts
) {
    dim3 blockSize(256);
    dim3 gridSize((srcVertCount + 255) / 256);
    
    SimplifyMeshKernel<<<gridSize, blockSize>>>(
        static_cast<const BuildingVertex*>(srcVerts),
        static_cast<const uint32_t*>(srcIdxs),
        static_cast<BuildingVertex*>(dstVerts),
        static_cast<uint32_t*>(dstIdxs),
        srcVertCount,
        srcIdxCount,
        reduction,
        &outCounts[0],
        &outCounts[1]
    );
    
    cudaDeviceSynchronize();
}

} // extern "C"
