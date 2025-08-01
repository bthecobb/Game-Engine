#include "Animation/AnimationResources.h"
#include "Animation/BoneTransform.h"
#include <algorithm>
#include <glm/gtx/quaternion.hpp>

namespace CudaGame {
namespace Animation {

BoneTransform AnimationClip::interpolateBoneTransform(size_t boneIndex, float time) const {
    BoneTransform result;
    
    // Default transform
    result.position = glm::vec3(0.0f);
    result.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    result.scale = glm::vec3(1.0f);
    
    if (boneIndex >= channels.size()) {
        return result;
    }
    
    const Channel& channel = channels[boneIndex];
    
    // Handle empty channel
    if (channel.times.empty()) {
        return result;
    }
    
    // Find the two keyframes to interpolate between
    auto it = std::lower_bound(channel.times.begin(), channel.times.end(), time);
    
    if (it == channel.times.begin()) {
        // Before first keyframe
        if (!channel.positions.empty()) result.position = channel.positions[0];
        if (!channel.rotations.empty()) result.rotation = channel.rotations[0];
        if (!channel.scales.empty()) result.scale = channel.scales[0];
    }
    else if (it == channel.times.end()) {
        // After last keyframe
        if (!channel.positions.empty()) result.position = channel.positions.back();
        if (!channel.rotations.empty()) result.rotation = channel.rotations.back();
        if (!channel.scales.empty()) result.scale = channel.scales.back();
    }
    else {
        // Between two keyframes - interpolate
        size_t idx1 = std::distance(channel.times.begin(), it) - 1;
        size_t idx2 = idx1 + 1;
        
        float t1 = channel.times[idx1];
        float t2 = channel.times[idx2];
        float alpha = (time - t1) / (t2 - t1);
        
        // Interpolate position
        if (idx2 < channel.positions.size()) {
            result.position = glm::mix(channel.positions[idx1], channel.positions[idx2], alpha);
        }
        
        // Interpolate rotation
        if (idx2 < channel.rotations.size()) {
            result.rotation = glm::slerp(channel.rotations[idx1], channel.rotations[idx2], alpha);
        }
        
        // Interpolate scale
        if (idx2 < channel.scales.size()) {
            result.scale = glm::mix(channel.scales[idx1], channel.scales[idx2], alpha);
        }
    }
    
    return result;
}

} // namespace Animation
} // namespace CudaGame
