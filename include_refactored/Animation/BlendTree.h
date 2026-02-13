#pragma once

#include "Animation/AnimationResources.h"
#include "Animation/BoneTransform.h"
#include <string>
#include <vector>
#include <memory>

namespace CudaGame {
namespace Animation {

class BlendNode;

// Input source for blend parameters
struct BlendInput {
    std::string name;
    float value = 0.0f;
};

// Base class for all nodes in a blend tree
class BlendNode {
public:
    virtual ~BlendNode() = default;
    virtual void Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) = 0;
    
    std::string GetName() const { return m_name; }
    void SetName(const std::string& name) { m_name = name; }

protected:
    std::string m_name;
};

// A leaf node that represents a single animation clip
class ClipNode : public BlendNode {
public:
    ClipNode(AnimationClip* clip, float playbackSpeed = 1.0f) 
        : m_clip(clip), m_playbackSpeed(playbackSpeed) {}

    void Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) override;

private:
    AnimationClip* m_clip;
    float m_playbackSpeed;
};

// A node that performs a linear interpolation between two child nodes
class LinearBlendNode : public BlendNode {
public:
    LinearBlendNode(std::shared_ptr<BlendNode> inputA, std::shared_ptr<BlendNode> inputB, const std::string& blendInputName)
        : m_inputA(inputA), m_inputB(inputB), m_blendInputName(blendInputName) {}
    
    void Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) override;

private:
    std::shared_ptr<BlendNode> m_inputA;
    std::shared_ptr<BlendNode> m_inputB;
    std::string m_blendInputName;
};

// A 1D blend node that blends between multiple children based on a single parameter
class BlendNode1D : public BlendNode {
public:
    void AddChild(std::shared_ptr<BlendNode> node, float threshold);
    void SetBlendInput(const std::string& inputName) { m_blendInputName = inputName; }

    void Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) override;

private:
    struct ChildNode {
        std::shared_ptr<BlendNode> node;
        float threshold;
    };
    std::vector<ChildNode> m_children;
    std::string m_blendInputName;
};

// A 2D blend node (e.g., for directional movement)
class BlendNode2D : public BlendNode {
public:
    // Blends between four children based on two parameters (e.g., x and y movement)
    void SetChildren(std::shared_ptr<BlendNode> topLeft, std::shared_ptr<BlendNode> topRight,
                     std::shared_ptr<BlendNode> bottomLeft, std::shared_ptr<BlendNode> bottomRight);
                     
    void SetBlendInputs(const std::string& inputXName, const std::string& inputYName) {
        m_blendInputXName = inputXName;
        m_blendInputYName = inputYName;
    }

    void Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) override;

private:
    std::shared_ptr<BlendNode> m_children[4]; // TL, TR, BL, BR
    std::string m_blendInputXName;
    std::string m_blendInputYName;
};

// Mask defining per-bone weights (0.0 = Base, 1.0 = Overlay)
struct BoneMask {
    std::vector<float> weights;
};

// A node that layers an overlay animation on top of a base animation using a mask
class LayeredBlendNode : public BlendNode {
public:
    LayeredBlendNode(std::shared_ptr<BlendNode> base, std::shared_ptr<BlendNode> overlay, 
                     std::shared_ptr<BoneMask> mask, const std::string& alphaInputName = "")
        : m_baseNode(base), m_overlayNode(overlay), m_mask(mask), m_alphaInputName(alphaInputName) {}
        
    void Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs) override;

private:
    std::shared_ptr<BlendNode> m_baseNode;
    std::shared_ptr<BlendNode> m_overlayNode;
    std::shared_ptr<BoneMask> m_mask;
    std::string m_alphaInputName; // Optional master alpha for the layer
};

// The root of the animation blend tree for an entity
class BlendTree {
public:
    BlendTree() = default;
    
    void SetRoot(std::shared_ptr<BlendNode> root) { m_rootNode = root; }
    std::shared_ptr<BlendNode> GetRoot() { return m_rootNode; }
    
    void Evaluate(float time, std::vector<BoneTransform>& outPose, const std::vector<BlendInput>& inputs);

private:
    std::shared_ptr<BlendNode> m_rootNode;
};

} // namespace Animation
} // namespace CudaGame
