#pragma once

#include "Animation/BlendTree.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace CudaGame {
namespace Animation {

// Forward declarations
class AnimationStateMachine;

// A single state in the state machine (e.g., "Locomotion", "Jump", "Attack")
class AnimationGraphState {
public:
    AnimationGraphState(const std::string& name);
    
    // Set the root blend node for this state (e.g., a BlendSpace1D for Walk/Run)
    void SetRootNode(std::shared_ptr<BlendNode> node) { m_rootNode = node; }
    std::shared_ptr<BlendNode> GetRootNode() { return m_rootNode; }
    
    const std::string& GetName() const { return m_name; }

    // Logic to run when entering/exiting this state
    void OnEnter();
    void OnExit();

private:
    std::string m_name;
    std::shared_ptr<BlendNode> m_rootNode;
};

// A transition rule between two states
struct AnimationTransition {
    std::string fromState;
    std::string toState;
    float blendDuration = 0.2f; // Time to cross-fade
    
    // Condition function: returns true if transition should occur
    std::function<bool(const std::vector<BlendInput>&)> condition;
};

// The brain of the animation system
class AnimationStateMachine {
public:
    AnimationStateMachine();
    ~AnimationStateMachine() = default;

    // Build the graph
    void AddState(std::shared_ptr<AnimationGraphState> state);
    void AddTransition(const AnimationTransition& transition);
    
    // Runtime
    void SetStartState(const std::string& stateName);
    void Update(float deltaTime, const std::vector<BlendInput>& inputs);
    
    // Output
    const std::vector<BoneTransform>& GetGlobalPose() const { return m_finalPose; }
    
    // Debug
    const std::string& GetCurrentStateName() const;

private:
    // Helper to evaluate a specific state's graph
    void EvaluateState(std::shared_ptr<AnimationGraphState> state, float time, 
                      std::vector<BoneTransform>& outPose, 
                      const std::vector<BlendInput>& inputs);

    // Blends two full poses together
    void BlendPoses(const std::vector<BoneTransform>& source, 
                   const std::vector<BoneTransform>& target, 
                   float weight, 
                   std::vector<BoneTransform>& outResult);

    std::map<std::string, std::shared_ptr<AnimationGraphState>> m_states;
    std::vector<AnimationTransition> m_transitions;
    
    std::shared_ptr<AnimationGraphState> m_currentState;
    std::shared_ptr<AnimationGraphState> m_targetState; // For cross-fading
    
    float m_currentTime = 0.0f;       // Time in current state
    float m_transitionTime = 0.0f;    // Time elapsed in current transition
    float m_currentTransitionDuration = 0.0f;
    bool m_isTransitioning = false;
    
    std::vector<BoneTransform> m_finalPose;
    std::vector<BoneTransform> m_sourcePose; // Temp buffer for blending
    std::vector<BoneTransform> m_targetPose; // Temp buffer for blending
};

} // namespace Animation
} // namespace CudaGame
