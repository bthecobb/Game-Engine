#include "Animation/ProceduralAnimationGenerator.h"
#include "Animation/BlendTree.h"
#include <glm/gtx/quaternion.hpp>
#include <cmath>
#include <algorithm> // For min/max if needed

namespace CudaGame {
namespace Animation {

void ProceduralAnimationGenerator::AddBone(std::vector<Skeleton::Bone>& bones, const std::string& name, int parentIdx, 
                                          const glm::vec3& offset, const glm::vec3& scale) {
    Skeleton::Bone bone;
    bone.name = name;
    // bone.id derived from index
    bone.parentIndex = parentIdx;
    bone.inverseBindPose = glm::translate(glm::mat4(1.0f), -offset); // Inverse bind pose logic simplified
    // Note: True inverse bind pose requires accumulated world transform inversion. 
    // For procedural, we just need the hierarchy ID structure mostly.
    
    // Transform Component (Rest Pose)
    // We don't store rest pose in Bone struct usually, it's in the Skeleton or BindPose array.
    // But AnimationClip needs to know target bone indices.
    bones.push_back(bone);
}

std::shared_ptr<Skeleton> ProceduralAnimationGenerator::CreateHumanoidSkeleton() {
    auto skeleton = std::make_shared<Skeleton>();
    std::vector<Skeleton::Bone>& bones = skeleton->bones;
    
    // 0: Hips (Root)
    AddBone(bones, "Hips", -1, glm::vec3(0.0f, 1.0f, 0.0f));
    
    // 1: Spine
    AddBone(bones, "Spine", 0, glm::vec3(0.0f, 0.2f, 0.0f));
    // 2: Neck
    AddBone(bones, "Neck", 1, glm::vec3(0.0f, 0.3f, 0.0f));
    // 3: Head
    AddBone(bones, "Head", 2, glm::vec3(0.0f, 0.15f, 0.0f));
    
    // Left Leg
    // 4: LeftUpLeg
    AddBone(bones, "LeftUpLeg", 0, glm::vec3(-0.15f, -0.1f, 0.0f));
    // 5: LeftLeg (Knee)
    AddBone(bones, "LeftLeg", 4, glm::vec3(0.0f, -0.4f, 0.0f));
    // 6: LeftFoot
    AddBone(bones, "LeftFoot", 5, glm::vec3(0.0f, -0.4f, 0.0f));
    
    // Right Leg
    // 7: RightUpLeg
    AddBone(bones, "RightUpLeg", 0, glm::vec3(0.15f, -0.1f, 0.0f));
    // 8: RightLeg
    AddBone(bones, "RightLeg", 7, glm::vec3(0.0f, -0.4f, 0.0f));
    // 9: RightFoot
    AddBone(bones, "RightFoot", 8, glm::vec3(0.0f, -0.4f, 0.0f));
    
    // Left Arm
    // 10: LeftArm
    AddBone(bones, "LeftArm", 1, glm::vec3(-0.2f, 0.15f, 0.0f));
    // 11: LeftForeArm
    AddBone(bones, "LeftForeArm", 10, glm::vec3(-0.3f, 0.0f, 0.0f));
    
    // Right Arm
    // 12: RightArm
    AddBone(bones, "RightArm", 1, glm::vec3(0.2f, 0.15f, 0.0f));
    // 13: RightForeArm
    AddBone(bones, "RightForeArm", 12, glm::vec3(0.3f, 0.0f, 0.0f));
    
    // Just stick to simple hierarchy for demo
    
    return skeleton;
}

std::shared_ptr<AnimationClip> ProceduralAnimationGenerator::CreateIdleClip(std::shared_ptr<Skeleton> skeleton) {
    auto clip = std::make_shared<AnimationClip>();
    clip->name = "Procedural_Idle";
    clip->duration = 1.0f;
    // clip->ticksPerSecond = 30.0f; // Removed
    
    // Create trivial keyframes for all bones (Identity)
    for (const auto& bone : skeleton->bones) {
        AnimationClip::Channel channel;
        channel.boneName = bone.name;
        
        // Determine frames (Spine needs 3 for breathing, others 1)
        std::vector<float> keyTimes = {0.0f};
        if (bone.name == "Spine") {
            keyTimes = {0.0f, 0.5f, 1.0f};
        }
        
        // Base Position (Bind Pose approximation)
        glm::vec3 basePos = glm::vec3(0.0f);
        if (bone.name == "Hips") basePos = glm::vec3(0.0f, 1.0f, 0.0f);
        else if (bone.name == "Spine") basePos = glm::vec3(0.0f, 0.2f, 0.0f);
        else if (bone.name == "LeftUpLeg") basePos = glm::vec3(-0.15f, -0.1f, 0.0f);
        else if (bone.name == "LeftLeg") basePos = glm::vec3(0.0f, -0.4f, 0.0f);
        else if (bone.name == "RightUpLeg") basePos = glm::vec3(0.15f, -0.1f, 0.0f);
        else if (bone.name == "RightLeg") basePos = glm::vec3(0.0f, -0.4f, 0.0f);
        
        for (float t : keyTimes) {
            channel.times.push_back(t);
            channel.positions.push_back(basePos);
            channel.scales.push_back(glm::vec3(1.0f));
            
            // Rotation
            glm::quat rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            if (bone.name == "Spine") {
                if (t > 0.4f && t < 0.6f) { // approx 0.5
                    rot = glm::quat(glm::vec3(0.05f, 0.0f, 0.0f));
                }
            }
            channel.rotations.push_back(rot);
        }
        
        clip->channels.push_back(channel);
    }
    
    return clip;
}

std::shared_ptr<AnimationClip> ProceduralAnimationGenerator::CreateWalkClip(std::shared_ptr<Skeleton> skeleton) {
    auto clip = std::make_shared<AnimationClip>();
    clip->name = "Procedural_Walk";
    clip->duration = 1.0f;
    // clip->ticksPerSecond = 30.0f; // Removed
    
    int frames = 30;
    
    for (const auto& bone : skeleton->bones) {
        AnimationClip::Channel channel;
        channel.boneName = bone.name;
        
        // Bind Pose Position (Simplified - assuming matches Skeleton)
        glm::vec3 bindPos(0.0f);
        if (bone.name == "Hips") bindPos = glm::vec3(0.0f, 1.0f, 0.0f);
        if (bone.name == "LeftUpLeg") bindPos = glm::vec3(-0.15f, -0.1f, 0.0f);
        if (bone.name == "RightUpLeg") bindPos = glm::vec3(0.15f, -0.1f, 0.0f);
        if (bone.name == "LeftLeg" || bone.name == "RightLeg") bindPos = glm::vec3(0.0f, -0.4f, 0.0f);
        // ... others
        
        for (int i = 0; i <= frames; ++i) {
            float time = (float)i / 30.0f; // 0 to 1
            float angle = time * 6.28318f; // 2*PI
            
            // Rotation
            glm::quat rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            
            if (bone.name == "LeftUpLeg") {
                // Sine wave X-axis
                rot = glm::angleAxis(sin(angle) * 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
            } else if (bone.name == "RightUpLeg") {
                // Cosine wave (phase shift)
                rot = glm::angleAxis(sin(angle + 3.14f) * 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
            } else if (bone.name == "LeftLeg") {
                // Knee bends only positive
                rot = glm::angleAxis(abs(sin(angle)) * 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
            } else if (bone.name == "RightLeg") {
                // Knee bends only positive (match RightUpLeg phase)
                rot = glm::angleAxis(abs(sin(angle + 3.14f)) * 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
            }
            
            channel.times.push_back(time);
            channel.positions.push_back(bindPos);
            channel.rotations.push_back(rot);
            channel.scales.push_back(glm::vec3(1.0f));
        }
        clip->channels.push_back(channel);
    }
    
    return clip;
}

std::shared_ptr<AnimationClip> ProceduralAnimationGenerator::CreateRunClip(std::shared_ptr<Skeleton> skeleton) {
     auto clip = CreateWalkClip(skeleton);
     clip->name = "Procedural_Run";
     // Increase Amplitude by modifying keys? 
     // For simplicity, just return walk for now, but name it run so graph treats it diff
     // In real implementation we'd scale the rotation angles.
     return clip;
}

std::shared_ptr<AnimationClip> ProceduralAnimationGenerator::CreateWaveClip(std::shared_ptr<Skeleton> skeleton) {
    auto clip = std::make_shared<AnimationClip>();
    clip->name = "Procedural_Wave";
    clip->duration = 2.0f;
    // clip->ticksPerSecond = 30.0f; // Removed
    int frames = 60;
    
    for (const auto& bone : skeleton->bones) {
        AnimationClip::Channel channel;
        channel.boneName = bone.name;
        // boneId removed
        
        // Identity / Bind Pose logic
        glm::vec3 bindPos(0.0f); // Simplification from prev
        
        for (int i = 0; i <= frames; ++i) {
            float time = (float)i / 30.0f;
            float angle = time * 3.14159f * 4.0f; // Fast wave
            
            glm::quat rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            
            if (bone.name == "RightArm") {
                // Raise arm forward/up
                rot = glm::angleAxis(-1.5f, glm::vec3(0.0f, 0.0f, 1.0f)); 
            } else if (bone.name == "RightForeArm") {
                // Wave back and forth Z-axis
                rot = glm::angleAxis(sin(angle) * 0.5f, glm::vec3(0.0f, 0.0f, 1.0f)); 
            }
            
            channel.times.push_back(time);
            channel.positions.push_back(bindPos);
            channel.rotations.push_back(rot);
            channel.scales.push_back(glm::vec3(1.0f));
        }
        clip->channels.push_back(channel);
    }
    return clip;
}

std::shared_ptr<BoneMask> ProceduralAnimationGenerator::CreateUpperBodyMask(std::shared_ptr<Skeleton> skeleton) {
    auto mask = std::make_shared<BoneMask>();
    mask->weights.resize(skeleton->bones.size(), 0.0f);
    
    for (int i = 0; i < skeleton->bones.size(); ++i) {
        const auto& bone = skeleton->bones[i];
        // Simple string matching for demo
        bool isUpper = false;
        if (bone.name.find("Spine") != std::string::npos) isUpper = true;
        if (bone.name.find("Head") != std::string::npos) isUpper = true;
        if (bone.name.find("Neck") != std::string::npos) isUpper = true;
        if (bone.name.find("Arm") != std::string::npos) isUpper = true;
        
        if (isUpper) {
            mask->weights[i] = 1.0f;
        }
    }
    return mask;
}

} // namespace Animation
} // namespace CudaGame
