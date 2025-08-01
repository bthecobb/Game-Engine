#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace CudaGame {
namespace Physics {

struct Transform {
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f, 1.0f, 1.0f};
    
    glm::mat4 GetMatrix() const {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 r = glm::mat4_cast(rotation);
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }
};

} // namespace Physics
} // namespace CudaGame
