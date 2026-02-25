#include "Animation/BlendTree.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <iostream>

namespace CudaGame {
namespace Animation {

// Helper: Spherically interpolate rotations and linearly interpolate positions/scales
static BoneTransform InterpolateChannel(const AnimationClip::Channel& channel, float time) {
    BoneTransform result;
    
    // Position
    if (!channel.positions.empty()) {
        if (channel.positions.size() == 1 || channel.times.size() < 2) {
            result.position = channel.positions[0];
        } else {
            size_t p0 = 0, p1 = 0;
            // Optimization: Binary search or cached index would be better, linear for now
            size_t maxFrame = channel.times.size() - 1;
            for (size_t i = 0; i < maxFrame; i++) {
                if (time < channel.times[i+1]) {
                    p0 = i;
                    p1 = i + 1;
                    break;
                }
            }
            if (p1 >= channel.positions.size()) { p0 = channel.positions.size() - 1; p1 = p0; }

            float t0 = channel.times[p0];
            float t1 = channel.times[p1];
            float factor = (t1 - t0 > 0.0001f) ? (time - t0) / (t1 - t0) : 0.0f;
            result.position = glm::mix(channel.positions[p0], channel.positions[p1], factor);
        }
    }
    
    // Rotation
    if (!channel.rotations.empty()) {
        if (channel.rotations.size() == 1 || channel.times.size() < 2) {
            result.rotation = channel.rotations[0];
        } else {
             size_t p0 = 0, p1 = 0;
             size_t maxFrame = channel.times.size() - 1;
            for (size_t i = 0; i < maxFrame; i++) {
                if (time < channel.times[i+1]) {
                    p0 = i;
                    p1 = i + 1;
                    break;
                }
            }
            if (p1 >= channel.rotations.size()) { p0 = channel.rotations.size() - 1; p1 = p0; }

            float t0 = channel.times[p0];
            float t1 = channel.times[p1];
            float factor = (t1 - t0 > 0.0001f) ? (time - t0) / (t1 - t0) : 0.0f;
            result.rotation = glm::slerp(channel.rotations[p0], channel.rotations[p1], factor);
        }
    }
    
    // Scale
    if (!channel.scales.empty()) {
        if (channel.scales.size() == 1 || channel.times.size() < 2) {
            result.scale = channel.scales[0];
        } else {
             size_t p0 = 0, p1 = 0;
             size_t maxFrame = channel.times.size() - 1;
            for (size_t i = 0; i < maxFrame; i++) {
                if (time < channel.times[i+1]) {
                    p0 = i;
                    p1 = i + 1;
                    break;
                }
            }
            if (p1 >= channel.scales.size()) { p0 = channel.scales.size() - 1; p1 = p0; }
            
            float t0 = channel.times[p0];
            float t1 = channel.times[p1];
            float factor = (t1 - t0 > 0.0001f) ? (time - t0) / (t1 - t0) : 0.0f;
            result.scale = glm::mix(channel.scales[p0], channel.scales[p1], factor);
        }
    }
    return result;
}

// ----------------------------------------------------------------------------------
// Helpers for Blending
// ----------------------------------------------------------------------------------
static void BlendPoses(const std::vector<BoneTransform>& poseA, const std::vector<BoneTransform>& poseB, float weight, std::vector<BoneTransform>& outPose) {
    if (poseA.size() != poseB.size()) {
        if (!poseA.empty()) outPose = poseA; // Fallback
        return;
    }
    
    outPose.resize(poseA.size());
    for (size_t i = 0; i < poseA.size(); ++i) {
        // Interpolate A -> B with weight
        outPose[i].position = glm::mix(poseA[i].position, poseB[i].position, weight);
        outPose[i].rotation = glm::slerp(poseA[i].rotation, poseB[i].rotation, weight);
        outPose[i].scale = glm::mix(poseA[i].scale, poseB[i].scale, weight);
    }
}

// ----------------------------------------------------------------------------------
// ClipNode
// ----------------------------------------------------------------------------------
void ClipNode::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs, const Skeleton& skeleton) {
    // 1. Initialize outPose to Bind Pose
    outPose.resize(skeleton.bones.size());
    for (size_t i = 0; i < skeleton.bones.size(); ++i) {
         // Create BoneTransform from localTransform (Bind Pose)
         // Note: localTransform is mat4. BoneTransform is pos/rot/scale.
         // Ideally Skeleton::Bone stores decomposed transform or BoneTransform.
         // We'll extract or just assume Identity if we want to be purely additive?
         // No, 'outPose' is local space transforms.
         // Let's assume we want Identity if no channel exists? 
         // Most engines: If no key, keep bind pose.
         // We don't have easy decomposer here (glm::decompose exists but is slow).
         // Minimal impl: Identity.
         outPose[i] = BoneTransform(); 
    }
    
    if (!m_clip) return;
    
    float sampleTime = time * m_playbackSpeed;
    if (m_clip->isLooping) {
        sampleTime = fmod(sampleTime, m_clip->duration);
    } else {
        sampleTime = glm::clamp(sampleTime, 0.0f, m_clip->duration);
    }
    
    for (const auto& channel : m_clip->channels) {
        auto it = skeleton.boneNameToIndex.find(channel.boneName);
        if (it != skeleton.boneNameToIndex.end()) {
            int boneIndex = it->second;
            if (boneIndex >= 0 && boneIndex < (int)outPose.size()) {
                outPose[boneIndex] = InterpolateChannel(channel, sampleTime);
            }
        }
    }
}

// ----------------------------------------------------------------------------------
// LinearBlendNode
// ----------------------------------------------------------------------------------
void LinearBlendNode::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs, const Skeleton& skeleton) {
    if (!m_inputA || !m_inputB) return;
    
    float weight = 0.5f;
    for (const auto& input : inputs) {
        if (input.name == m_blendInputName) {
            weight = glm::clamp(input.value, 0.0f, 1.0f);
            break;
        }
    }
    
    std::vector<BoneTransform> poseA;
    std::vector<BoneTransform> poseB;
    
    m_inputA->Evaluate(time, poseA, inputs, skeleton);
    m_inputB->Evaluate(time, poseB, inputs, skeleton);
    
    BlendPoses(poseA, poseB, weight, outPose);
}

// ----------------------------------------------------------------------------------
// BlendNode1D
// ----------------------------------------------------------------------------------
void BlendNode1D::AddChild(std::shared_ptr<BlendNode> node, float threshold) {
    m_children.push_back({node, threshold});
    // Sort by threshold
    std::sort(m_children.begin(), m_children.end(), 
        [](const ChildNode& a, const ChildNode& b) { return a.threshold < b.threshold; });
}

void BlendNode1D::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs, const Skeleton& skeleton) {
    if (m_children.empty()) return;
    if (m_children.size() == 1) {
        m_children[0].node->Evaluate(time, outPose, inputs, skeleton);
        return;
    }
    
    float param = 0.0f;
    for (const auto& input : inputs) {
        if (input.name == m_blendInputName) {
            param = input.value;
            break;
        }
    }
    
    // Find neighbors
    // Children sorted by threshold
    // e.g. [0.0 (Idle), 0.5 (Walk), 1.0 (Run)]
    // Param 0.25 -> blend Idl/Walk (0.25/0.50 = 0.5 weight)
    
    size_t idxA = 0;
    size_t idxB = 0;
    
    // Find first node > param
    auto it = std::upper_bound(m_children.begin(), m_children.end(), param,
        [](float val, const ChildNode& node) { return val < node.threshold; });
        
    if (it == m_children.begin()) {
        idxA = idxB = 0; // Below lowest
    } else if (it == m_children.end()) {
        idxA = idxB = m_children.size() - 1; // Above highest
    } else {
        idxB = std::distance(m_children.begin(), it);
        idxA = idxB - 1;
    }
    
    if (idxA == idxB) {
        m_children[idxA].node->Evaluate(time, outPose, inputs, skeleton);
    } else {
        float tA = m_children[idxA].threshold;
        float tB = m_children[idxB].threshold;
        float weight = (param - tA) / (tB - tA);
        weight = glm::clamp(weight, 0.0f, 1.0f);
        
        std::vector<BoneTransform> poseA, poseB;
        m_children[idxA].node->Evaluate(time, poseA, inputs, skeleton);
        m_children[idxB].node->Evaluate(time, poseB, inputs, skeleton);
        
        BlendPoses(poseA, poseB, weight, outPose);
    }
}

// ----------------------------------------------------------------------------------
// BlendNode2D
// ----------------------------------------------------------------------------------
void BlendNode2D::SetChildren(std::shared_ptr<BlendNode> topLeft, std::shared_ptr<BlendNode> topRight,
                              std::shared_ptr<BlendNode> bottomLeft, std::shared_ptr<BlendNode> bottomRight) {
    m_children[0] = topLeft;
    m_children[1] = topRight;
    m_children[2] = bottomLeft;
    m_children[3] = bottomRight;
}

void BlendNode2D::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs, const Skeleton& skeleton) {
    // Bilinear interpolation
    float x = 0.0f;
    float y = 0.0f;
    
    for (const auto& input : inputs) {
        if (input.name == m_blendInputXName) x = input.value; // e.g. -1 to 1
        else if (input.name == m_blendInputYName) y = input.value;
    }
    
    // Normalize -1..1 to 0..1 for lerp
    float u = (x + 1.0f) * 0.5f;
    float v = (y + 1.0f) * 0.5f;
    
    u = glm::clamp(u, 0.0f, 1.0f);
    v = glm::clamp(v, 0.0f, 1.0f);
    
    std::vector<BoneTransform> poseTL, poseTR, poseBL, poseBR;
    
    if(m_children[0]) m_children[0]->Evaluate(time, poseTL, inputs, skeleton);
    if(m_children[1]) m_children[1]->Evaluate(time, poseTR, inputs, skeleton);
    if(m_children[2]) m_children[2]->Evaluate(time, poseBL, inputs, skeleton);
    if(m_children[3]) m_children[3]->Evaluate(time, poseBR, inputs, skeleton);

    // If any pose missing, handle gracefully? Assumed valid for now.
    if (poseTL.empty()) poseTL = poseBL; // fallback
    // ... robustness checks elided for brevity
    
    std::vector<BoneTransform> topRow, bottomRow;
    BlendPoses(poseTL, poseTR, u, topRow);
    BlendPoses(poseBL, poseBR, u, bottomRow);
    BlendPoses(topRow, bottomRow, v, outPose);
}

// ----------------------------------------------------------------------------------
// LayeredBlendNode
// ----------------------------------------------------------------------------------
void LayeredBlendNode::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs, const Skeleton& skeleton) {
    if (!m_baseNode || !m_overlayNode) return;
    
    m_baseNode->Evaluate(time, outPose, inputs, skeleton);
    
    std::vector<BoneTransform> overlayPose;
    m_overlayNode->Evaluate(time, overlayPose, inputs, skeleton);
    
    if (overlayPose.size() != outPose.size()) return;
    
    // Apply overlay based on mask
    // If mask weight > 0, we blend overlay on top (or replace)
    // Default mode: Replace based on weight
    
    float masterAlpha = 1.0f;
    if (!m_alphaInputName.empty()) {
        for(const auto& input : inputs) {
            if (input.name == m_alphaInputName) {
                masterAlpha = input.value;
                break;
            }
        }
    }
    
    if (m_mask && m_mask->weights.size() == outPose.size()) {
        for (size_t i = 0; i < outPose.size(); ++i) {
            float w = m_mask->weights[i] * masterAlpha;
            if (w > 0.001f) {
                outPose[i].position = glm::mix(outPose[i].position, overlayPose[i].position, w);
                outPose[i].rotation = glm::slerp(outPose[i].rotation, overlayPose[i].rotation, w);
                outPose[i].scale = glm::mix(outPose[i].scale, overlayPose[i].scale, w);
            }
        }
    }
}

// ----------------------------------------------------------------------------------
// BlendTree
// ----------------------------------------------------------------------------------
void BlendTree::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs, const Skeleton& skeleton) {
    if (m_rootNode) {
        m_rootNode->Evaluate(time, outPose, inputs, skeleton);
    }
}

} // namespace Animation
} // namespace CudaGame
