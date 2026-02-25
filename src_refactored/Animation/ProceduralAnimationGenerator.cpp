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
    clip->duration = 2.0f;
    clip->isLooping = true;
    
    const float D = clip->duration;
    const float f = 0.5f; // breathing Hz
    const int   K = 32;   // keyframes
    const float dt = D / (K - 1);

    for (const auto& bone : skeleton->bones) {
        AnimationClip::Channel channel;
        channel.boneName = bone.name;
        
        // Initialize channel with K keyframes at bind-pose
        glm::vec3 basePos(0.0f);
        if (bone.name == "Hips")       basePos = glm::vec3(0.0f, 1.0f, 0.0f);
        else if (bone.name == "Spine") basePos = glm::vec3(0.0f, 0.2f, 0.0f);
        else if (bone.name == "LeftUpLeg")  basePos = glm::vec3(-0.15f, -0.1f, 0.0f);
        else if (bone.name == "RightUpLeg") basePos = glm::vec3( 0.15f, -0.1f, 0.0f);
        else if (bone.name == "LeftLeg" || bone.name == "RightLeg") basePos = glm::vec3(0.0f, -0.4f, 0.0f);

        for (int i = 0; i < K; ++i) {
            float t = i * dt;
            float s = std::sin(t * f * 6.28318f);

            channel.times.push_back(t);
            glm::vec3 pos = basePos;
            glm::quat rot(1.0f, 0.0f, 0.0f, 0.0f);

            if (bone.name == "Spine") {
                rot = glm::quat(glm::vec3(0.04f * s, 0.0f, 0.02f * s)); // breathe + micro sway
            } else if (bone.name == "Hips") {
                pos.z += 0.012f * s;    // lateral sway
                pos.y += 0.008f * s;    // micro bob
            } else if (bone.name == "Head") {
                rot = glm::quat(glm::vec3(0.025f * s, 0.0f, 0.0f)); // slow nod
            } else if (bone.name == "LeftForeArm" || bone.name == "RightForeArm") {
                rot = glm::quat(glm::vec3(0.30f, 0.0f, 0.0f)); // hang angle
            }

            channel.positions.push_back(pos);
            channel.rotations.push_back(rot);
            channel.scales.push_back(glm::vec3(1.0f));
        }
        clip->channels.push_back(channel);
    }
    return clip;
}

std::shared_ptr<AnimationClip> ProceduralAnimationGenerator::CreateWalkClip(std::shared_ptr<Skeleton> skeleton) {
    auto clip = std::make_shared<AnimationClip>();
    clip->name = "Procedural_Walk";
    clip->duration = 1.2f;
    clip->isLooping = true;

    const int   K  = 32;
    const float D  = clip->duration;
    const float dt = D / (K - 1);
    const float f  = 1.0f / D; // cycle Hz

    for (const auto& bone : skeleton->bones) {
        AnimationClip::Channel channel;
        channel.boneName = bone.name;

        glm::vec3 bindPos(0.0f);
        if (bone.name == "Hips")       bindPos = glm::vec3(0.0f, 1.0f, 0.0f);
        if (bone.name == "LeftUpLeg")  bindPos = glm::vec3(-0.15f, -0.1f, 0.0f);
        if (bone.name == "RightUpLeg") bindPos = glm::vec3( 0.15f, -0.1f, 0.0f);
        if (bone.name == "LeftLeg" || bone.name == "RightLeg") bindPos = glm::vec3(0.0f, -0.4f, 0.0f);

        for (int i = 0; i <= K; ++i) {
            float t = (float)i * dt;
            float a = t * 6.28318f * f; // angle in radians

            glm::vec3 pos = bindPos;
            glm::quat rot(1.0f, 0.0f, 0.0f, 0.0f);

            if (bone.name == "Hips") {
                pos.y += 0.05f * std::sin(a * 2.0f); // bob 2x per step
                pos.z += 0.04f * std::sin(a);         // lateral sway
            } else if (bone.name == "Spine") {
                rot = glm::quat(glm::vec3(0.0f, 0.10f * std::sin(a + 3.14f), 0.0f)); // counter-rotation
            } else if (bone.name == "LeftUpLeg") {
                rot = glm::angleAxis( std::sin(a)         * 0.50f, glm::vec3(1,0,0));
            } else if (bone.name == "RightUpLeg") {
                rot = glm::angleAxis( std::sin(a + 3.14f) * 0.50f, glm::vec3(1,0,0));
            } else if (bone.name == "LeftLeg") {
                rot = glm::angleAxis(std::abs(std::sin(a))         * 0.50f + 0.25f, glm::vec3(1,0,0));
            } else if (bone.name == "RightLeg") {
                rot = glm::angleAxis(std::abs(std::sin(a + 3.14f)) * 0.50f + 0.25f, glm::vec3(1,0,0));
            } else if (bone.name == "LeftFoot") {
                rot = glm::angleAxis( std::sin(a + 3.14f) * 0.25f, glm::vec3(1,0,0)); // ankle push-off
            } else if (bone.name == "RightFoot") {
                rot = glm::angleAxis( std::sin(a)         * 0.25f, glm::vec3(1,0,0));
            } else if (bone.name == "LeftArm") {
                rot = glm::angleAxis( std::sin(a + 3.14f) * 0.30f, glm::vec3(1,0,0)); // opposite legs
            } else if (bone.name == "RightArm") {
                rot = glm::angleAxis( std::sin(a)         * 0.30f, glm::vec3(1,0,0));
            } else if (bone.name == "Head") {
                pos.y += 0.015f * std::sin(a * 2.0f); // head bob
            }

            channel.times.push_back(t);
            channel.positions.push_back(pos);
            channel.rotations.push_back(rot);
            channel.scales.push_back(glm::vec3(1.0f));
        }
        clip->channels.push_back(channel);
    }
    return clip;
}

std::shared_ptr<AnimationClip> ProceduralAnimationGenerator::CreateRunClip(std::shared_ptr<Skeleton> skeleton) {
    auto clip = std::make_shared<AnimationClip>();
    clip->name = "Procedural_Run";
    clip->duration = 0.8f;
    clip->isLooping = true;

    const int   K  = 32;
    const float D  = clip->duration;
    const float dt = D / (K - 1);
    const float f  = 1.0f / D;

    for (const auto& bone : skeleton->bones) {
        AnimationClip::Channel channel;
        channel.boneName = bone.name;

        glm::vec3 bindPos(0.0f);
        if (bone.name == "Hips")       bindPos = glm::vec3(0.0f, 0.95f, 0.0f); // slight crouch
        if (bone.name == "LeftUpLeg")  bindPos = glm::vec3(-0.15f, -0.1f, 0.0f);
        if (bone.name == "RightUpLeg") bindPos = glm::vec3( 0.15f, -0.1f, 0.0f);
        if (bone.name == "LeftLeg" || bone.name == "RightLeg") bindPos = glm::vec3(0.0f, -0.4f, 0.0f);

        for (int i = 0; i <= K; ++i) {
            float t = (float)i * dt;
            float a = t * 6.28318f * f;

            glm::vec3 pos = bindPos;
            glm::quat rot(1.0f, 0.0f, 0.0f, 0.0f);

            if (bone.name == "Hips") {
                pos.y += 0.10f * std::sin(a * 2.0f);
                pos.z += 0.05f * std::sin(a);
            } else if (bone.name == "Spine") {
                float lean = 0.15f; // constant forward lean
                rot = glm::quat(glm::vec3(lean, 0.15f * std::sin(a + 3.14f), 0.0f));
            } else if (bone.name == "LeftUpLeg") {
                rot = glm::angleAxis( std::sin(a)         * 0.80f + 0.20f, glm::vec3(1,0,0));
            } else if (bone.name == "RightUpLeg") {
                rot = glm::angleAxis( std::sin(a + 3.14f) * 0.80f + 0.20f, glm::vec3(1,0,0));
            } else if (bone.name == "LeftLeg") {
                rot = glm::angleAxis(std::abs(std::sin(a))         * 0.80f + 0.60f, glm::vec3(1,0,0));
            } else if (bone.name == "RightLeg") {
                rot = glm::angleAxis(std::abs(std::sin(a + 3.14f)) * 0.80f + 0.60f, glm::vec3(1,0,0));
            } else if (bone.name == "LeftFoot") {
                rot = glm::angleAxis( std::sin(a + 3.14f) * 0.55f + 0.05f, glm::vec3(1,0,0));
            } else if (bone.name == "RightFoot") {
                rot = glm::angleAxis( std::sin(a)         * 0.55f + 0.05f, glm::vec3(1,0,0));
            } else if (bone.name == "LeftArm") {
                rot = glm::angleAxis( std::sin(a + 3.14f) * 0.60f + 0.20f, glm::vec3(1,0,0));
            } else if (bone.name == "RightArm") {
                rot = glm::angleAxis( std::sin(a)         * 0.60f + 0.20f, glm::vec3(1,0,0));
            } else if (bone.name == "LeftForeArm" || bone.name == "RightForeArm") {
                rot = glm::quat(glm::vec3(1.50f, 0.0f, 0.0f)); // bent elbow pump
            } else if (bone.name == "Head") {
                pos.y += 0.04f * std::sin(a * 2.0f);
            }

            channel.times.push_back(t);
            channel.positions.push_back(pos);
            channel.rotations.push_back(rot);
            channel.scales.push_back(glm::vec3(1.0f));
        }
        clip->channels.push_back(channel);
    }

    // Footstep events (for parity with AnimationBuilder path)
    clip->AddEvent(0.0f,   "Footstep_Left",  nullptr);
    clip->AddEvent(D * 0.5f, "Footstep_Right", nullptr);

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
