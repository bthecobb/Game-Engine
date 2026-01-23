#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include "Rendering/CullAndDraw.h" 
// Note: We need to define the kernel logic again or include it. 
// Since CullAndDraw.h only has the struct, we need the full implementation here.
// I will rewrite the full implementation from Step 10321 but updated to use the header.

// Because CullAndDraw.h has the struct ObjectCullingData, we don't need to redefine it unless it's C++ only.
// Assuming CullAndDraw.h is compatible with CUDA (it uses standard types and glm which works in CUDA if handled right).
// However, glm vectors in CUDA kernels need care. 
// For safety/speed, I'll redefine the struct LOCALLY with float3 to be 100% sure of layout/alignment in CUDA.
// C++ side uses GLM, CUDA side uses built-in vector types.

struct IndirectCommand {
    unsigned long long cbv;
    unsigned long long materialCbv;
    unsigned long long vbv_loc;
    unsigned int      vbv_size;
    unsigned int      vbv_stride;
    unsigned long long ibv_loc;
    unsigned int      ibv_size;
    unsigned int      ibv_format;
    unsigned int      indexCount;
    unsigned int      instanceCount;
    unsigned int      startIndex;
    int               baseVertex;
    unsigned int      startInstance;
    unsigned int      padding;
};

// Internal Kernel Representation (Must match C++ layout)
struct ObjectCullingDataGPU {
    float3 sphereCenter;
    float sphereRadius;
    unsigned long long vbv_loc;
    unsigned int      vbv_size;
    unsigned int      vbv_stride;
    unsigned long long ibv_loc;
    unsigned int      ibv_size;
    unsigned int      ibv_format;
    unsigned long long cbv;
    unsigned long long materialCbv;
    Matrix4x4 worldMatrix; // Added
    unsigned int      indexCount;
};

struct Matrix4x4 {
    float m[4][4];
};

struct FrustumPlane {
    float x, y, z, w;
};

__constant__ FrustumPlane c_frustumPlanes[6];

__device__ float Dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ float3 TransformPoint(float3 p, Matrix4x4 m) {
    float3 r;
    r.x = m.m[0][0] * p.x + m.m[1][0] * p.y + m.m[2][0] * p.z + m.m[3][0];
    r.y = m.m[0][1] * p.x + m.m[1][1] * p.y + m.m[2][1] * p.z + m.m[3][1];
    r.z = m.m[0][2] * p.x + m.m[1][2] * p.y + m.m[2][2] * p.z + m.m[3][2];
    return r;
}

__device__ bool IsVisible(float3 center, float radius) {
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        float dist = c_frustumPlanes[i].x * center.x + c_frustumPlanes[i].y * center.y + c_frustumPlanes[i].z * center.z + c_frustumPlanes[i].w;
        if (dist < -radius) return false;
    }
    return true;
}

__global__ void CullAndDrawKernel(
    const ObjectCullingDataGPU* objects,
    IndirectCommand* commands,
    unsigned int* drawCounter,
    int objectCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= objectCount) return;

    ObjectCullingDataGPU obj = objects[idx];
    Matrix4x4 world = obj.worldMatrix; // Use embedded matrix

    float3 worldCenter = TransformPoint(obj.sphereCenter, world);
    
    // Scale estimation
    float scaleX = sqrtf(world.m[0][0]*world.m[0][0] + world.m[0][1]*world.m[0][1] + world.m[0][2]*world.m[0][2]);
    float scaleY = sqrtf(world.m[1][0]*world.m[1][0] + world.m[1][1]*world.m[1][1] + world.m[1][2]*world.m[1][2]);
    float scaleZ = sqrtf(world.m[2][0]*world.m[2][0] + world.m[2][1]*world.m[2][1] + world.m[2][2]*world.m[2][2]);
    float maxScale = fmaxf(scaleX, fmaxf(scaleY, scaleZ));
    float worldRadius = obj.sphereRadius * maxScale;

    if (IsVisible(worldCenter, worldRadius)) {
        unsigned int cmdIdx = atomicAdd(drawCounter, 1);
        IndirectCommand cmd;
        cmd.cbv = obj.cbv;
        cmd.materialCbv = obj.materialCbv;
        cmd.vbv_loc = obj.vbv_loc;
        cmd.vbv_size = obj.vbv_size;
        cmd.vbv_stride = obj.vbv_stride;
        cmd.ibv_loc = obj.ibv_loc;
        cmd.ibv_size = obj.ibv_size;
        cmd.ibv_format = obj.ibv_format;
        cmd.indexCount = obj.indexCount;
        cmd.instanceCount = 1;
        cmd.startIndex = 0;
        cmd.baseVertex = 0;
        cmd.startInstance = 0;
        cmd.padding = 0;
        commands[cmdIdx] = cmd;
    }
}

extern "C" void LaunchCullAndDrawKernel(
    const void* objects,
    void* commands,
    unsigned int* drawCounter,
    int objectCount,
    const float* frustumPlanes,
    cudaStream_t stream
) {
    cudaMemcpyToSymbolAsync(c_frustumPlanes, frustumPlanes, sizeof(FrustumPlane) * 6, 0, cudaMemcpyHostToDevice, stream);
    int blockSize = 128;
    int numBlocks = (objectCount + blockSize - 1) / blockSize;
    CullAndDrawKernel<<<numBlocks, blockSize, 0, stream>>>(
        (const ObjectCullingDataGPU*)objects,
        (IndirectCommand*)commands,
        drawCounter,
        objectCount
    );
}
