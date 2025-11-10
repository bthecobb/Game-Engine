#include "Rendering/SkyboxUtils.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>
#include <cmath>

namespace CudaGame {
namespace Rendering {

static inline glm::vec3 SafeNormalize(const glm::vec3& v) {
    float len = glm::length(v);
    if (len > 0.0f) return v / len;
    return glm::vec3(0.0f, 1.0f, 0.0f);
}

glm::vec2 DirectionToEquirectUV(const glm::vec3& dIn) {
    glm::vec3 d = SafeNormalize(dIn);
    float theta = std::atan2(d.z, d.x); // [-pi, pi]
    float phi   = std::acos(glm::clamp(d.y, -1.0f, 1.0f)); // [0, pi]
    float u = (theta + glm::pi<float>()) / (2.0f * glm::pi<float>());
    float v = phi / glm::pi<float>();
    return glm::vec2(u, v);
}

glm::vec3 EquirectUVToDirection(const glm::vec2& uv) {
    float u = glm::clamp(uv.x, 0.0f, 1.0f);
    float v = glm::clamp(uv.y, 0.0f, 1.0f);
    float theta = u * 2.0f * glm::pi<float>() - glm::pi<float>();
    float phi   = v * glm::pi<float>();
    float sinPhi = std::sin(phi);
    glm::vec3 d;
    d.x = std::cos(theta) * sinPhi;
    d.y = std::cos(phi);
    d.z = std::sin(theta) * sinPhi;
    return glm::normalize(d);
}

CubemapSample DirectionToCubemap(const glm::vec3& dIn) {
    glm::vec3 d = SafeNormalize(dIn);
    glm::vec3 ad = glm::abs(d);
    int face = 0;
    float uc = 0.0f, vc = 0.0f; // in [-1,1]

    if (ad.x >= ad.y && ad.x >= ad.z) {
        if (d.x > 0.0f) {
            face = 0; // +X
            uc = -d.z / ad.x;
            vc = -d.y / ad.x;
        } else {
            face = 1; // -X
            uc =  d.z / ad.x;
            vc = -d.y / ad.x;
        }
    } else if (ad.y >= ad.x && ad.y >= ad.z) {
        if (d.y > 0.0f) {
            face = 2; // +Y
            uc =  d.x / ad.y;
            vc =  d.z / ad.y;
        } else {
            face = 3; // -Y
            uc =  d.x / ad.y;
            vc = -d.z / ad.y;
        }
    } else { // ad.z is largest
        if (d.z > 0.0f) {
            face = 4; // +Z
            uc =  d.x / ad.z;
            vc = -d.y / ad.z;
        } else {
            face = 5; // -Z
            uc = -d.x / ad.z;
            vc = -d.y / ad.z;
        }
    }

    glm::vec2 uv = 0.5f * (glm::vec2(uc, vc) + glm::vec2(1.0f));
    return CubemapSample{ face, glm::clamp(uv, glm::vec2(0.0f), glm::vec2(1.0f)) };
}

int ComputeMipLevels(int size) {
    if (size <= 0) return 0;
    int levels = 1;
    int v = size;
    while (v > 1) {
        v >>= 1;
        ++levels;
    }
    return levels;
}

glm::vec3 LinearToSRGB(const glm::vec3& linear, float gamma) {
    glm::vec3 c = glm::clamp(linear, glm::vec3(0.0f), glm::vec3(1000.0f));
    float inv = 1.0f / glm::max(gamma, 1e-6f);
    return glm::pow(c, glm::vec3(inv));
}

glm::vec3 SRGBToLinear(const glm::vec3& srgb, float gamma) {
    glm::vec3 c = glm::clamp(srgb, glm::vec3(0.0f), glm::vec3(1000.0f));
    return glm::pow(c, glm::vec3(glm::max(gamma, 1e-6f)));
}

} // namespace Rendering
} // namespace CudaGame
