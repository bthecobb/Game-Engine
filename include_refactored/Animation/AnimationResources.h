#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>

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
    
    // --- Animation Events ---
    struct AnimationEvent {
        float time;                      // Trigger time within the clip (seconds)
        std::string name;                // Event identifier e.g. "Footstep_Left"
        std::function<void()> callback;  // Callback to fire
    };
    
    std::vector<AnimationEvent> events;
    
    // Add and sort an event by time (keeps events sorted ascending for linear scan)
    void AddEvent(float time, const std::string& name, std::function<void()> callback) {
        events.push_back({ time, name, std::move(callback) });
        std::sort(events.begin(), events.end(),
            [](const AnimationEvent& a, const AnimationEvent& b) { return a.time < b.time; });
    }
    
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
    
    int GetBoneIndex(const std::string& name) const {
        auto it = boneNameToIndex.find(name);
        return (it != boneNameToIndex.end()) ? it->second : -1;
    }
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

// Mask defining per-bone weights (0.0 = Base, 1.0 = Overlay)
struct BoneMask {
    std::vector<float> weights;
};

} // namespace Animation
} // namespace CudaGame
