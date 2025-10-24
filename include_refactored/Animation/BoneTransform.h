#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

namespace CudaGame {
namespace Animation {

struct BoneTransform {
    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;

    BoneTransform() : position(0.0f), scale(1.0f) {}
    BoneTransform(const glm::vec3& pos, const glm::quat& rot, const glm::vec3& scl)
        : position(pos), rotation(rot), scale(scl) {}

    glm::mat4 ToMatrix() const {
        return glm::translate(position) * glm::toMat4(rotation) * glm::scale(scale);
    }
};

} // namespace Animation
} // namespace CudaGame
