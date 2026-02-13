#pragma once

namespace CudaGame {
namespace VFX {

struct float3_pod { float x, y, z; };
struct float4_pod { float x, y, z, w; };

struct Particle {
    float3_pod position;
    float life;
    float3_pod velocity;
    float size;
    float4_pod color;
};

} // namespace VFX
} // namespace CudaGame
