#pragma once

#include "Animation/AnimationResources.h"
#include <memory>
#include <vector>

namespace CudaGame {
namespace Animation {

class AnimationBuilder {
public:
    // Create a standard Humanoid Skeleton (Hips, Spine, Head, Arms, Legs)
    static std::shared_ptr<Skeleton> CreateHumanoidSkeleton();

    // Procedural Clip Generators
    // Generates a math-based breathing animation
    static std::unique_ptr<AnimationClip> CreateIdleClip(const Skeleton& skeleton);

    // Generates a math-based walk cycle
    static std::unique_ptr<AnimationClip> CreateWalkClip(const Skeleton& skeleton, float speed = 1.0f);

    // Generates a math-based run cycle
    static std::unique_ptr<AnimationClip> CreateRunClip(const Skeleton& skeleton, float speed = 1.0f);

private:
    // Helper to add a bone to the list and map
    static void AddBone(Skeleton& skeleton, const std::string& name, const std::string& parentName, 
                        const glm::vec3& localPos);
    
    // Helper to bake a curve into keyframes
    static void BakeCurveToChannel(AnimationClip::Channel& channel, float duration, 
                                   int targetType, int axis, 
                                   float amp, float freq, float phase, float offset);

    // Helper to bake a constant (fixed) rotation or position offset into all keyframes.
    // Useful for lean angles, rest poses, and elbow bends that don't animate.
    static void BakeConstantToChannel(AnimationClip::Channel& channel, float duration,
                                      int targetType, int axis, float value, int numKeys = 2);
};

} // namespace Animation
} // namespace CudaGame
