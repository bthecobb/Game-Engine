#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Define CUDA_HOST_DEVICE if not already defined
#ifndef CUDA_HOST_DEVICE
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif
#endif

namespace CudaGame {
namespace Animation {

// POD structure for Bone Transform, suitable for GPU buffers
struct CudaBoneTransform {
    glm::vec3 position;
    float padding1; // Alignment
    glm::quat rotation;
    glm::vec3 scale;
    float padding2; // Alignment

    CUDA_HOST_DEVICE CudaBoneTransform() 
        : position(0.0f), rotation(1.0f, 0.0f, 0.0f, 0.0f), scale(1.0f), padding1(0), padding2(0) {}
};

// Flattened skeleton structure for linear memory access (SoA largely preferred for coalesce, but AoS is easiest for initial port)
// We'll stick to AoS for transforms for now, or SoA if we want strict optimization.
// Let's use a flat array structure concept.

struct CudaSkeletonData {
    int boneCount;
    // Pointers to GPU memory would be stored in a separate manager. 
    // This struct defines the data layout for a single instance in a kernel.
    int* parentIndices;      // [boneCount]
    glm::mat4* bindPoses;    // [boneCount]
};

// Represents a procedural animation curve parameters
// e.g., A * sin(frequency * t + phase) + offset
struct ProceduralCurve {
    float amplitude;
    float frequency;
    float phase;
    float offset;
    int axis; // 0=X, 1=Y, 2=Z, 3=W (Rot)
    int targetType; // 0=Pos, 1=Rot, 2=Scale
};

// Compute-ready clip data
struct CudaProceduralClip {
    int curveCount;
    ProceduralCurve* curves; // Pointer to array of curves
    float duration;
};

} // namespace Animation
} // namespace CudaGame
