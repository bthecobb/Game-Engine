#include "Animation/AnimationStateMachine.h"
#include "Animation/AnimationResources.h"
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace CudaGame {
namespace Animation {

// ----------------------------------------------------------------------------------
// AnimationGraphState
// ----------------------------------------------------------------------------------
AnimationGraphState::AnimationGraphState(const std::string& name) : m_name(name) {}

void AnimationGraphState::OnEnter() {
    // Optional: Reset internal state of nodes?
    // For now, nothing to do.
}

void AnimationGraphState::OnExit() {
    // Optional: Cleanup or notify
}

// ----------------------------------------------------------------------------------
// AnimationStateMachine
// ----------------------------------------------------------------------------------
AnimationStateMachine::AnimationStateMachine() {}

void AnimationStateMachine::AddState(std::shared_ptr<AnimationGraphState> state) {
    if (state) {
        m_states[state->GetName()] = state;
    }
}

void AnimationStateMachine::AddTransition(const AnimationTransition& transition) {
    m_transitions.push_back(transition);
}

void AnimationStateMachine::SetStartState(const std::string& stateName) {
    auto it = m_states.find(stateName);
    if (it != m_states.end()) {
        m_currentState = it->second;
        m_currentState->OnEnter();
        m_currentTime = 0.0f;
    }
}

const std::string& AnimationStateMachine::GetCurrentStateName() const {
    static std::string empty = "";
    return m_currentState ? m_currentState->GetName() : empty;
}

void AnimationStateMachine::Update(float deltaTime, const std::vector<BlendInput>& inputs, const Skeleton& skeleton) {
    // 1. Check Transitions
    if (m_currentState && !m_isTransitioning) {
        for (const auto& trans : m_transitions) {
            if (trans.fromState == m_currentState->GetName()) {
                if (trans.condition && trans.condition(inputs)) {
                    // Start Transition
                    auto it = m_states.find(trans.toState);
                    if (it != m_states.end()) {
                        m_targetState = it->second;
                        m_targetState->OnEnter();
                        
                        m_currentTransitionDuration = trans.blendDuration;
                        m_transitionTime = 0.0f;
                        m_isTransitioning = true;
                        
                        // Break after finding first valid transition
                        // (Priority based on order of addition)
                        break;
                    }
                }
            }
        }
    }
    
    // 2. Update Time
    m_previousTime = m_currentTime;  // Snapshot for event detection
    m_currentTime += deltaTime;
    
    // 3. Evaluate States
    if (m_currentState) {
        EvaluateState(m_currentState, m_currentTime, m_finalPose, inputs, skeleton);
    }
    
    // 4. Handle Transition Blending
    if (m_isTransitioning && m_targetState) {
        m_transitionTime += deltaTime;
        float t = glm::clamp(m_transitionTime / m_currentTransitionDuration, 0.0f, 1.0f);
        
        // Evaluate Target State (Time starts at 0 for new state?)
        // Usually yes, new state starts at 0 unless synced.
        // We'll assume start at 0 + transitionTime.
        std::vector<BoneTransform> targetPose;
        EvaluateState(m_targetState, m_transitionTime, targetPose, inputs, skeleton);
        
        // Blend Current -> Target
        // We need to blend m_finalPose (which currently holds Current State result) with TargetPose
        BlendPoses(m_finalPose, targetPose, t, m_finalPose);
        
        // Finish Transition
        if (m_transitionTime >= m_currentTransitionDuration) {
            m_currentState->OnExit();
            m_currentState = m_targetState;
            m_currentTime = m_transitionTime; // Continue time from transition
            m_targetState = nullptr;
            m_isTransitioning = false;
        }
    }
}

void AnimationStateMachine::EvaluateState(std::shared_ptr<AnimationGraphState> state, float time, 
                                         std::vector<BoneTransform>& outPose, 
                                         const std::vector<BlendInput>& inputs,
                                         const Skeleton& skeleton) {
    if (state && state->GetRootNode()) {
        state->GetRootNode()->Evaluate(time, outPose, inputs, skeleton);
    }
}


void AnimationStateMachine::BlendPoses(const std::vector<BoneTransform>& source, 
                                      const std::vector<BoneTransform>& target, 
                                      float weight, 
                                      std::vector<BoneTransform>& outResult) {
    if (source.size() != target.size()) {
        if (!target.empty()) outResult = target;
        else outResult = source;
        return;
    }
    
    outResult.resize(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        outResult[i].position = glm::mix(source[i].position, target[i].position, weight);
        outResult[i].rotation = glm::slerp(source[i].rotation, target[i].rotation, weight);
        outResult[i].scale = glm::mix(source[i].scale, target[i].scale, weight);
    }
}

} // namespace Animation
} // namespace CudaGame
