#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>
#include <unordered_map>

namespace CudaGame {
namespace Animation {

// Forward declaration
struct BoneTransform;

// Animation clip data structure
struct AnimationClip {
    std::string name;
    float duration;
    bool isLooping;
    
    struct Channel {
        std::string boneName;
        std::vector<float> times;
        std::vector<glm::vec3> positions;
        std::vector<glm::quat> rotations;
        std::vector<glm::vec3> scales;
    };
    
    std::vector<Channel> channels;
    
    // Helper methods
    float getDuration() const { return duration; }
    
    // Simplified bone transform interpolation
    BoneTransform interpolateBoneTransform(size_t boneIndex, float time) const;
};

// Animation skeleton structure
struct Skeleton {
    struct Bone {
        std::string name;
        int parentIndex;
        glm::mat4 inverseBindPose;
        glm::mat4 localTransform;
    };
    
    std::vector<Bone> bones;
    std::unordered_map<std::string, int> boneNameToIndex;
};

} // namespace Animation
} // namespace CudaGame
