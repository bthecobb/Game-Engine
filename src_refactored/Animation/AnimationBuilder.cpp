#include "Animation/AnimationBuilder.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace CudaGame {
namespace Animation {

void AnimationBuilder::AddBone(Skeleton& skeleton, const std::string& name, const std::string& parentName, const glm::vec3& localPos) {
    Skeleton::Bone bone;
    bone.name = name;
    bone.localTransform = glm::translate(glm::mat4(1.0f), localPos);
    
    // Calculate parent index
    if (parentName.empty()) {
        bone.parentIndex = -1;
    } else {
        if (skeleton.boneNameToIndex.find(parentName) != skeleton.boneNameToIndex.end()) {
            bone.parentIndex = skeleton.boneNameToIndex[parentName];
        } else {
            bone.parentIndex = -1; // Fallback
        }
    }
    
    glm::mat4 worldBind = bone.localTransform;
    if (bone.parentIndex != -1) {
        glm::mat4 parentWorldBind = glm::inverse(skeleton.bones[bone.parentIndex].inverseBindPose);
        worldBind = parentWorldBind * bone.localTransform;
    } else {
        worldBind = bone.localTransform;
    }
    
    bone.inverseBindPose = glm::inverse(worldBind);
    
    int index = (int)skeleton.bones.size();
    skeleton.bones.push_back(bone);
    skeleton.boneNameToIndex[name] = index;
}

std::shared_ptr<Skeleton> AnimationBuilder::CreateHumanoidSkeleton() {
    auto skeleton = std::make_shared<Skeleton>();
    
    // Hips (Root) - at 1.0m height
    AddBone(*skeleton, "Hips", "", glm::vec3(0.0f, 1.0f, 0.0f));
    
    // Spine Chain
    AddBone(*skeleton, "Spine", "Hips", glm::vec3(0.0f, 0.15f, 0.0f));
    AddBone(*skeleton, "Chest", "Spine", glm::vec3(0.0f, 0.2f, 0.0f));
    AddBone(*skeleton, "Neck", "Chest", glm::vec3(0.0f, 0.2f, 0.0f));
    AddBone(*skeleton, "Head", "Neck", glm::vec3(0.0f, 0.1f, 0.0f));
    
    // Left Leg
    AddBone(*skeleton, "LeftUpLeg",  "Hips",      glm::vec3(-0.1f, -0.05f, 0.0f));
    AddBone(*skeleton, "LeftLeg",    "LeftUpLeg", glm::vec3(0.0f, -0.4f, 0.0f));
    AddBone(*skeleton, "LeftFoot",   "LeftLeg",   glm::vec3(0.0f, -0.4f, 0.0f));
    
    // Right Leg
    AddBone(*skeleton, "RightUpLeg",  "Hips",       glm::vec3(0.1f, -0.05f, 0.0f));
    AddBone(*skeleton, "RightLeg",    "RightUpLeg", glm::vec3(0.0f, -0.4f, 0.0f));
    AddBone(*skeleton, "RightFoot",   "RightLeg",   glm::vec3(0.0f, -0.4f, 0.0f));
    
    // Left Arm
    AddBone(*skeleton, "LeftArm",    "Chest", glm::vec3(-0.2f, 0.1f, 0.0f));
    AddBone(*skeleton, "LeftForeArm","LeftArm", glm::vec3(-0.25f, 0.0f, 0.0f));
    
    // Right Arm
    AddBone(*skeleton, "RightArm",    "Chest", glm::vec3(0.2f, 0.1f, 0.0f));
    AddBone(*skeleton, "RightForeArm","RightArm", glm::vec3(0.25f, 0.0f, 0.0f));
    
    return skeleton;
}

// ----------------------------------------------------------------
// BakeCurveToChannel
// Bakes a sine curve (amp*sin(freq*2π*t + phase) + offset) into
// numKeys keyframes on the given axis of position (type=0) or rotation (type=1).
// ----------------------------------------------------------------
void AnimationBuilder::BakeCurveToChannel(AnimationClip::Channel& channel, float duration, 
    int targetType, int axis, float amp, float freq, float phase, float offset)
{
    const int numKeys = 32;
    float dt = duration / (numKeys - 1);
    
    // Ensure vectors are initialized to right size
    if (channel.times.empty()) {
        channel.times.resize(numKeys);
        channel.positions.resize(numKeys, glm::vec3(0.0f));
        channel.rotations.resize(numKeys, glm::quat(1,0,0,0));
        channel.scales.resize(numKeys, glm::vec3(1.0f));
        for (int i = 0; i < numKeys; ++i) channel.times[i] = i * dt;
    }
    
    for (int i = 0; i < numKeys && i < (int)channel.times.size(); ++i) {
        float t = channel.times[i];
        float val = amp * std::sin(t * freq * 6.28318f + phase) + offset;
        
        if (targetType == 0) { // Position
            channel.positions[i][axis] = val;
        } else if (targetType == 1) { // Rotation (Euler per axis -> quat)
            glm::vec3 euler(0.0f);
            euler[axis] = val;
            // Compose onto existing rotation (preserves multi-axis baking)
            channel.rotations[i] = glm::quat(euler) * channel.rotations[i];
        }
    }
}

// ----------------------------------------------------------------
// BakeConstantToChannel
// Bakes a fixed constant value (e.g. forward lean, elbow hang angle)
// into numKeys evenly-spaced keyframes. Composes onto existing rotation.
// ----------------------------------------------------------------
void AnimationBuilder::BakeConstantToChannel(AnimationClip::Channel& channel, float duration,
    int targetType, int axis, float value, int numKeys)
{
    if (channel.times.empty()) {
        channel.times.resize(numKeys);
        channel.positions.resize(numKeys, glm::vec3(0.0f));
        channel.rotations.resize(numKeys, glm::quat(1,0,0,0));
        channel.scales.resize(numKeys, glm::vec3(1.0f));
        float dt = duration / (numKeys - 1);
        for (int i = 0; i < numKeys; ++i) channel.times[i] = i * dt;
    }

    for (int i = 0; i < (int)channel.times.size(); ++i) {
        if (targetType == 0) {
            channel.positions[i][axis] = value;
        } else if (targetType == 1) {
            glm::vec3 euler(0.0f);
            euler[axis] = value;
            channel.rotations[i] = glm::quat(euler) * channel.rotations[i];
        }
    }
}

// ----------------------------------------------------------------
// CreateIdleClip  (2.0s loop)
// Breathing spine, micro lateral hip sway, slow head nod, forearm hang.
// ----------------------------------------------------------------
std::unique_ptr<AnimationClip> AnimationBuilder::CreateIdleClip(const Skeleton& skeleton) {
    auto clip = std::make_unique<AnimationClip>();
    clip->name = "Idle";
    clip->duration = 2.0f;
    clip->isLooping = true;
    
    const float D = clip->duration;
    const float f = 0.5f; // 0.5 Hz breathing cycle

    // Spine: gentle inhale pitch (X) + micro lateral sway (Z)
    {
        AnimationClip::Channel ch; ch.boneName = "Spine";
        BakeCurveToChannel(ch, D, 1, 0, 0.04f, f, 0.0f, 0.0f);  // X pitch: breath
        BakeCurveToChannel(ch, D, 1, 2, 0.02f, f, 0.0f, 0.0f);  // Z roll: sway
        clip->channels.push_back(ch);
    }

    // Chest: secondary breathe amplitude (slightly more than spine)
    {
        AnimationClip::Channel ch; ch.boneName = "Chest";
        BakeCurveToChannel(ch, D, 1, 0, 0.03f, f, 0.0f, 0.0f);
        clip->channels.push_back(ch);
    }

    // Hips: micro lateral Z-translate sway (weight shift even at rest)
    {
        AnimationClip::Channel ch; ch.boneName = "Hips";
        BakeCurveToChannel(ch, D, 0, 2, 0.012f, f, 0.0f, 0.0f);  // Z lateral
        BakeCurveToChannel(ch, D, 0, 1, 0.008f, f, 0.0f, 1.0f);  // Y subtle bob + base height
        clip->channels.push_back(ch);
    }

    // Head: slow nod (pitch) — breath-rate
    {
        AnimationClip::Channel ch; ch.boneName = "Head";
        BakeCurveToChannel(ch, D, 1, 0, 0.025f, f, 0.0f, 0.0f);  // slow X nod
        clip->channels.push_back(ch);
    }

    // Left arm: sway at breath rate, forearm hangs slightly bent
    {
        AnimationClip::Channel lArm; lArm.boneName = "LeftArm";
        BakeCurveToChannel(lArm, D, 1, 2, 0.04f, f, 1.0f, -0.15f); // Z: slight abduction
        clip->channels.push_back(lArm);

        AnimationClip::Channel lFore; lFore.boneName = "LeftForeArm";
        BakeConstantToChannel(lFore, D, 1, 0, 0.30f);  // X: forearm hang angle ~17°
        clip->channels.push_back(lFore);
    }

    // Right arm: mirror
    {
        AnimationClip::Channel rArm; rArm.boneName = "RightArm";
        BakeCurveToChannel(rArm, D, 1, 2, -0.04f, f, 1.0f, 0.15f);
        clip->channels.push_back(rArm);

        AnimationClip::Channel rFore; rFore.boneName = "RightForeArm";
        BakeConstantToChannel(rFore, D, 1, 0, 0.30f);
        clip->channels.push_back(rFore);
    }
    
    return clip;
}

// ----------------------------------------------------------------
// CreateWalkClip  (1.2s/speed loop)
// Legs, knee bend, spine counter-rotation, hip sway, head bob,
// ankle plantar-flexion, arm swing with Z cross-body component.
// ----------------------------------------------------------------
std::unique_ptr<AnimationClip> AnimationBuilder::CreateWalkClip(const Skeleton& skeleton, float speed) {
    auto clip = std::make_unique<AnimationClip>();
    clip->name = "Walk";
    clip->duration = 1.2f / speed;
    clip->isLooping = true;
    
    const float D = clip->duration;
    const float f = 1.0f / D; // cycle freq (Hz)

    // Hips: vertical bob (2x per step) + lateral Z sway (2x per step)
    {
        AnimationClip::Channel ch; ch.boneName = "Hips";
        BakeCurveToChannel(ch, D, 0, 1, 0.05f, f * 2.0f, 0.0f,  1.0f); // Y bob
        BakeCurveToChannel(ch, D, 0, 2, 0.04f, f,        0.0f,  0.0f); // Z lateral sway
        clip->channels.push_back(ch);
    }

    // Spine: axial Y-rotation counter to hips (gives organic twist)
    {
        AnimationClip::Channel ch; ch.boneName = "Spine";
        BakeCurveToChannel(ch, D, 1, 1, 0.10f, f, 3.14f, 0.0f); // Y: counter-rotate
        clip->channels.push_back(ch);
    }

    // Left Leg: swing (X), knee bend (X offset)
    {
        AnimationClip::Channel ul; ul.boneName = "LeftUpLeg";
        BakeCurveToChannel(ul, D, 1, 0, 0.50f, f, 0.0f, 0.0f);
        clip->channels.push_back(ul);

        AnimationClip::Channel kn; kn.boneName = "LeftLeg";
        BakeCurveToChannel(kn, D, 1, 0, 0.40f, f, 1.5f, 0.40f); // knee bends trailing
        clip->channels.push_back(kn);

        AnimationClip::Channel ft; ft.boneName = "LeftFoot";
        // Ankle plantar-flexion: push-off at stance end (phase ~PI offset from leg)
        BakeCurveToChannel(ft, D, 1, 0, 0.25f, f, 3.14f, 0.0f);
        clip->channels.push_back(ft);
    }

    // Right Leg: 180° phase from left
    {
        AnimationClip::Channel ur; ur.boneName = "RightUpLeg";
        BakeCurveToChannel(ur, D, 1, 0, 0.50f, f, 3.14f, 0.0f);
        clip->channels.push_back(ur);

        AnimationClip::Channel kn; kn.boneName = "RightLeg";
        BakeCurveToChannel(kn, D, 1, 0, 0.40f, f, 3.14f + 1.5f, 0.40f);
        clip->channels.push_back(kn);

        AnimationClip::Channel ft; ft.boneName = "RightFoot";
        BakeCurveToChannel(ft, D, 1, 0, 0.25f, f, 0.0f, 0.0f);
        clip->channels.push_back(ft);
    }

    // Left Arm: forward-back swing (X, opposite to legs) + Z cross-body
    {
        AnimationClip::Channel la; la.boneName = "LeftArm";
        BakeCurveToChannel(la, D, 1, 0,  0.30f, f, 3.14f, 0.0f); // X: swing opposite right leg
        BakeCurveToChannel(la, D, 1, 2,  0.05f, f, 3.14f, 0.0f); // Z: small cross-body
        clip->channels.push_back(la);

        AnimationClip::Channel lf; lf.boneName = "LeftForeArm";
        BakeConstantToChannel(lf, D, 1, 0, 0.25f); // slight hang
        clip->channels.push_back(lf);
    }

    // Right Arm: opposite to left
    {
        AnimationClip::Channel ra; ra.boneName = "RightArm";
        BakeCurveToChannel(ra, D, 1, 0,  0.30f, f, 0.0f, 0.0f);
        BakeCurveToChannel(ra, D, 1, 2, -0.05f, f, 0.0f, 0.0f);
        clip->channels.push_back(ra);

        AnimationClip::Channel rf; rf.boneName = "RightForeArm";
        BakeConstantToChannel(rf, D, 1, 0, 0.25f);
        clip->channels.push_back(rf);
    }

    // Head: vertical Y-translate bob (2x per step, small)
    {
        AnimationClip::Channel hd; hd.boneName = "Head";
        BakeCurveToChannel(hd, D, 0, 1, 0.015f, f * 2.0f, 0.0f, 0.0f); // Y bob
        clip->channels.push_back(hd);
    }
    
    return clip;
}

// ----------------------------------------------------------------
// CreateRunClip  (0.8s/speed loop)
// Higher amplitude legs, forward spine lean, aggressive ankle push-off,
// pronounced head bob, pumping arms with Z cross-body, footstep events.
// ----------------------------------------------------------------
std::unique_ptr<AnimationClip> AnimationBuilder::CreateRunClip(const Skeleton& skeleton, float speed) {
    auto clip = std::make_unique<AnimationClip>();
    clip->name = "Run";
    clip->duration = 0.8f / speed;
    clip->isLooping = true;
    
    const float D = clip->duration;
    const float f = 1.0f / D;

    // Hips: more vertical drive + lateral sway
    {
        AnimationClip::Channel ch; ch.boneName = "Hips";
        BakeCurveToChannel(ch, D, 0, 1, 0.10f, f * 2.0f, 0.0f, 0.95f); // Y: lower+higher bob
        BakeCurveToChannel(ch, D, 0, 2, 0.05f, f,        0.0f, 0.0f);  // Z sway
        clip->channels.push_back(ch);
    }

    // Spine: constant forward lean + aggressive axial counter-rotation
    {
        AnimationClip::Channel ch; ch.boneName = "Spine";
        BakeConstantToChannel(ch, D, 1, 0, 0.15f); // X: forward lean ~8.6°
        BakeCurveToChannel(ch,  D, 1, 1, 0.15f, f, 3.14f, 0.0f); // Y: twist
        clip->channels.push_back(ch);
    }

    // Chest: additional lean
    {
        AnimationClip::Channel ch; ch.boneName = "Chest";
        BakeConstantToChannel(ch, D, 1, 0, 0.08f); // X: lean continuation up the chain
        clip->channels.push_back(ch);
    }

    // Left Leg: high knee drive + deep knee bend
    {
        AnimationClip::Channel ul; ul.boneName = "LeftUpLeg";
        BakeCurveToChannel(ul, D, 1, 0, 0.80f, f, 0.0f, 0.20f); // forward lean offset
        clip->channels.push_back(ul);

        AnimationClip::Channel kn; kn.boneName = "LeftLeg";
        BakeCurveToChannel(kn, D, 1, 0, 0.80f, f, 1.0f, 0.80f); // deep knee bend
        clip->channels.push_back(kn);

        AnimationClip::Channel ft; ft.boneName = "LeftFoot";
        // Strong push-off at stance -> toe-off (high amplitude, phase locked)
        BakeCurveToChannel(ft, D, 1, 0, 0.55f, f, 3.14f, 0.05f);
        clip->channels.push_back(ft);
    }

    // Right Leg: 180° offset
    {
        AnimationClip::Channel ur; ur.boneName = "RightUpLeg";
        BakeCurveToChannel(ur, D, 1, 0, 0.80f, f, 3.14f, 0.20f);
        clip->channels.push_back(ur);

        AnimationClip::Channel kn; kn.boneName = "RightLeg";
        BakeCurveToChannel(kn, D, 1, 0, 0.80f, f, 3.14f + 1.0f, 0.80f);
        clip->channels.push_back(kn);

        AnimationClip::Channel ft; ft.boneName = "RightFoot";
        BakeCurveToChannel(ft, D, 1, 0, 0.55f, f, 0.0f, 0.05f);
        clip->channels.push_back(ft);
    }

    // Arms: bent elbows, strong forward-back swing + Z cross-body drive
    {
        AnimationClip::Channel la; la.boneName = "LeftArm";
        BakeCurveToChannel(la, D, 1, 0,  0.60f, f, 3.14f, 0.20f); // X swing
        BakeCurveToChannel(la, D, 1, 2,  0.10f, f, 3.14f, 0.0f);  // Z cross-body
        clip->channels.push_back(la);

        AnimationClip::Channel lf; lf.boneName = "LeftForeArm";
        BakeConstantToChannel(lf, D, 1, 0, 1.50f); // ~85° bent elbow when pumping
        clip->channels.push_back(lf);
    }

    {
        AnimationClip::Channel ra; ra.boneName = "RightArm";
        BakeCurveToChannel(ra, D, 1, 0,  0.60f, f, 0.0f,  0.20f);
        BakeCurveToChannel(ra, D, 1, 2, -0.10f, f, 0.0f,  0.0f);
        clip->channels.push_back(ra);

        AnimationClip::Channel rf; rf.boneName = "RightForeArm";
        BakeConstantToChannel(rf, D, 1, 0, 1.50f);
        clip->channels.push_back(rf);
    }

    // Head: pronounced bob (2x per step)
    {
        AnimationClip::Channel hd; hd.boneName = "Head";
        BakeCurveToChannel(hd, D, 0, 1, 0.04f, f * 2.0f, 0.0f, 0.0f); // Y bob
        // Slight stabilisation pitch counter to spine lean
        BakeCurveToChannel(hd, D, 1, 0,-0.05f, f, 0.0f, 0.0f);
        clip->channels.push_back(hd);
    }

    // --- Animation Events: Footsteps ---
    // Left heel-strike at cycle start, right at halfway
    clip->AddEvent(0.0f,          "Footstep_Left",  nullptr); // callback wired at runtime
    clip->AddEvent(D * 0.5f,      "Footstep_Right", nullptr);
    
    return clip;
}

} // namespace Animation
} // namespace CudaGame
