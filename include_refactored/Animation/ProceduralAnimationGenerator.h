#pragma once

#include "Animation/AnimationResources.h"
#include <memory>
#include <map>
#include <vector>
#include <string>

namespace CudaGame {
namespace Animation {

class ProceduralAnimationGenerator {
public:
    // Generates a basic humanoid hierarchy (Hips -> Spine -> Head, Hips -> Legs, Spines -> Arms)
    static std::shared_ptr<Skeleton> CreateHumanoidSkeleton();
    
    // Generates a static idle pose animation
    static std::shared_ptr<AnimationClip> CreateIdleClip(std::shared_ptr<Skeleton> skeleton);
    
    // Generates a walking loop animation (sine wave legs/arms)
    static std::shared_ptr<AnimationClip> CreateWalkClip(std::shared_ptr<Skeleton> skeleton);
    
    // Generates a running loop animation (faster, wider amplitude)
    static std::shared_ptr<AnimationClip> CreateRunClip(std::shared_ptr<Skeleton> skeleton);
    
    // Generates a simple "Wave" animation for the Right Arm
    static std::shared_ptr<AnimationClip> CreateWaveClip(std::shared_ptr<Skeleton> skeleton);
    
    // Generates a BoneMask for Upper Body (Spine, Arms, Head)
    static std::shared_ptr<struct BoneMask> CreateUpperBodyMask(std::shared_ptr<Skeleton> skeleton);

private:
    static void AddBone(std::vector<Skeleton::Bone>& bones, const std::string& name, int parentIdx, 
                       const glm::vec3& offset, const glm::vec3& scale = glm::vec3(1.0f));
};

} // namespace Animation
} // namespace CudaGame
