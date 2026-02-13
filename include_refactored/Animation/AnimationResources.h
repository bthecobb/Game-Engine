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
class AnimationClip {
public:
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
class Skeleton {
public:
    struct Bone {
        std::string name;
        int parentIndex;
        glm::mat4 inverseBindPose;
        glm::mat4 localTransform;
    };
    
    std::vector<Bone> bones;
    std::unordered_map<std::string, int> boneNameToIndex;
};

// Animation states for AAA-quality character animation
enum class AnimationState {
    IDLE,
    IDLE_BORED,
    WALKING,
    RUNNING,
    SPRINTING,
    JUMPING,
    AIRBORNE,
    FALLING,
    LANDING,
    DIVING,
    WALL_RUNNING,
    SLIDING,
    COMBAT_IDLE,
    ATTACKING,
    PARRYING,
    GRABBING,
    STUNNED,
    DEATH,
    // Weapon-specific animations
    SWORD_IDLE,
    SWORD_ATTACK_1,
    SWORD_ATTACK_2,
    SWORD_COMBO_FINISHER,
    STAFF_CAST,
    STAFF_SPIN,
    HAMMER_CHARGE,
    HAMMER_SLAM
};

// Animation blend modes for smooth transitions
enum class BlendMode {
    REPLACE,     // Replace current animation
    ADDITIVE,    // Add to current animation
    MULTIPLY,    // Multiply with current animation
    OVERLAY      // Overlay on top of current animation
};

} // namespace Animation
} // namespace CudaGame
