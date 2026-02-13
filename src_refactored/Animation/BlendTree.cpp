#include "Animation/BlendTree.h"
#include "Animation/AnimationResources.h"
#include <algorithm>
#include <cmath>
#include <glm/gtx/quaternion.hpp>

namespace CudaGame {
namespace Animation {

// ClipNode implementation
void ClipNode::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) {
    if (!m_clip) return;
    
    float evalTime = fmod(time * m_playbackSpeed, m_clip->getDuration());
    
    // This is a simplified evaluation. A real implementation would need to handle bone hierarchies.
    for (size_t i = 0; i < outPose.size(); ++i) {
        outPose[i] = m_clip->interpolateBoneTransform(i, evalTime);
    }
}

// LinearBlendNode implementation
void LinearBlendNode::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) {
    float blendWeight = 0.0f;
    for (const auto& input : inputs) {
        if (input.name == m_blendInputName) {
            blendWeight = glm::clamp(input.value, 0.0f, 1.0f);
            break;
        }
    }

    std::vector<BoneTransform> poseA(outPose.size()), poseB(outPose.size());
    m_inputA->Evaluate(time, poseA, inputs);
    m_inputB->Evaluate(time, poseB, inputs);

    for (size_t i = 0; i < outPose.size(); ++i) {
        outPose[i].position = glm::mix(poseA[i].position, poseB[i].position, blendWeight);
        outPose[i].rotation = glm::slerp(poseA[i].rotation, poseB[i].rotation, blendWeight);
        outPose[i].scale = glm::mix(poseA[i].scale, poseB[i].scale, blendWeight);
    }
}

// BlendNode1D implementation
void BlendNode1D::AddChild(std::shared_ptr<BlendNode> node, float threshold) {
    m_children.push_back({node, threshold});
    std::sort(m_children.begin(), m_children.end(), [](const ChildNode& a, const ChildNode& b) {
        return a.threshold < b.threshold;
    });
}

void BlendNode1D::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) {
    float blendValue = 0.0f;
    for (const auto& input : inputs) {
        if (input.name == m_blendInputName) {
            blendValue = input.value;
            break;
        }
    }
    
    if (m_children.empty()) return;
    if (m_children.size() == 1) {
        m_children[0].node->Evaluate(time, outPose, inputs);
        return;
    }

    // Find the two children to blend between
    size_t rightIdx = 0;
    while (rightIdx < m_children.size() && m_children[rightIdx].threshold < blendValue) {
        rightIdx++;
    }
    
    if (rightIdx == 0) {
        m_children[0].node->Evaluate(time, outPose, inputs);
        return;
    }
    if (rightIdx == m_children.size()) {
        m_children.back().node->Evaluate(time, outPose, inputs);
        return;
    }

    size_t leftIdx = rightIdx - 1;
    float leftThreshold = m_children[leftIdx].threshold;
    float rightThreshold = m_children[rightIdx].threshold;
    float blendWeight = (blendValue - leftThreshold) / (rightThreshold - leftThreshold);

    std::vector<BoneTransform> poseA(outPose.size()), poseB(outPose.size());
    m_children[leftIdx].node->Evaluate(time, poseA, inputs);
    m_children[rightIdx].node->Evaluate(time, poseB, inputs);
    
    for (size_t i = 0; i < outPose.size(); ++i) {
        outPose[i].position = glm::mix(poseA[i].position, poseB[i].position, blendWeight);
        outPose[i].rotation = glm::slerp(poseA[i].rotation, poseB[i].rotation, blendWeight);
        outPose[i].scale = glm::mix(poseA[i].scale, poseB[i].scale, blendWeight);
    }
}

// BlendNode2D implementation
void BlendNode2D::SetChildren(std::shared_ptr<BlendNode> topLeft, std::shared_ptr<BlendNode> topRight,
                              std::shared_ptr<BlendNode> bottomLeft, std::shared_ptr<BlendNode> bottomRight) {
    m_children[0] = topLeft;
    m_children[1] = topRight;
    m_children[2] = bottomLeft;
    m_children[3] = bottomRight;
}

void BlendNode2D::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) {
    float blendX = 0.0f, blendY = 0.0f;
    for (const auto& input : inputs) {
        if (input.name == m_blendInputXName) blendX = glm::clamp(input.value, -1.0f, 1.0f);
        if (input.name == m_blendInputYName) blendY = glm::clamp(input.value, -1.0f, 1.0f);
    }
    
    float wX = (blendX + 1.0f) * 0.5f;
    float wY = (blendY + 1.0f) * 0.5f;
    
    std::vector<BoneTransform> poseTL(outPose.size()), poseTR(outPose.size());
    std::vector<BoneTransform> poseBL(outPose.size()), poseBR(outPose.size());
    
    m_children[0]->Evaluate(time, poseTL, inputs);
    m_children[1]->Evaluate(time, poseTR, inputs);
    m_children[2]->Evaluate(time, poseBL, inputs);
    m_children[3]->Evaluate(time, poseBR, inputs);
    
    for (size_t i = 0; i < outPose.size(); ++i) {
        // Bilinear interpolation
        glm::vec3 posTop = glm::mix(poseTL[i].position, poseTR[i].position, wX);
        glm::vec3 posBottom = glm::mix(poseBL[i].position, poseBR[i].position, wX);
        outPose[i].position = glm::mix(posBottom, posTop, wY);
        
        glm::quat rotTop = glm::slerp(poseTL[i].rotation, poseTR[i].rotation, wX);
        glm::quat rotBottom = glm::slerp(poseBL[i].rotation, poseBR[i].rotation, wX);
        outPose[i].rotation = glm::slerp(rotBottom, rotTop, wY);
        
        glm::vec3 scaleTop = glm::mix(poseTL[i].scale, poseTR[i].scale, wX);
        glm::vec3 scaleBottom = glm::mix(poseBL[i].scale, poseBR[i].scale, wX);
        outPose[i].scale = glm::mix(scaleBottom, scaleTop, wY);
    }
}

// BlendTree implementation
void BlendTree::Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) {
    if (m_rootNode) {
        m_rootNode->Evaluate(time, outPose, inputs);
    }
}

} // namespace Animation
} // namespace CudaGame
