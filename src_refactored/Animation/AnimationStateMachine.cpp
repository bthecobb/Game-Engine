#include "Animation/AnimationStateMachine.h"
#include <iostream>
#include <glm/gtx/compatibility.hpp> // For lerp/slerp helpers if needed

namespace CudaGame {
namespace Animation {

// === AnimationGraphState ===

AnimationGraphState::AnimationGraphState(const std::string& name) 
    : m_name(name) {
}

void AnimationGraphState::OnEnter() {
    // Optional: Reset internal state time or trigger events
}

void AnimationGraphState::OnExit() {
    // Optional: Cleanup
}

// === AnimationStateMachine ===

AnimationStateMachine::AnimationStateMachine() {
    // Reserve logical size, though resize happens on evaluate
}

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
        if (m_currentState) {
            m_currentState->OnExit();
        }
        m_currentState = it->second;
        m_currentState->OnEnter();
        m_currentTime = 0.0f;
        m_isTransitioning = false;
    } else {
        std::cerr << "[AnimationStateMachine] Error: Start state '" << stateName << "' not found." << std::endl;
    }
}

const std::string& AnimationStateMachine::GetCurrentStateName() const {
    static std::string empty = "None";
    if (m_targetState) {
        // If transitioning, maybe return "Transitioning: A->B"? 
        // For simple logic, return the source state driving the logic mainly, or target?
        // Let's return current authoritative state.
        return m_currentState ? m_currentState->GetName() : empty;
    }
    return m_currentState ? m_currentState->GetName() : empty;
}

void AnimationStateMachine::Update(float deltaTime, const std::vector<BlendInput>& inputs) {
    if (!m_currentState) return;

    // 1. Check Transitions (only if not already transitioning)
    // Note: A robust system allows interrupting transitions, but we keep it simple (finish transition first).
    if (!m_isTransitioning) {
        for (const auto& trans : m_transitions) {
            if (trans.fromState == m_currentState->GetName()) {
                if (trans.condition && trans.condition(inputs)) {
                    auto it = m_states.find(trans.toState);
                    if (it != m_states.end()) {
                        // Start Transition
                        m_targetState = it->second;
                        m_currentTransitionDuration = trans.blendDuration;
                        m_transitionTime = 0.0f;
                        m_isTransitioning = true;
                        
                        m_targetState->OnEnter();
                        // std::cout << "[AnimSM] Transition: " << m_currentState->GetName() << " -> " << m_targetState->GetName() << std::endl;
                        break; 
                    }
                }
            }
            // Allow "Any" state transitions? (fromState == "Any") -> Implementation detail for later
        }
    }

    // 2. Advance Time
    m_currentTime += deltaTime;
    
    // 3. Evaluate Main Configuration
    // We need to know the skeleton size. Usually retrieved from the root node or a bind pose.
    // For now, assume the nodes will resize the vector if needed (BlendTree impl does this).
    // Specifically, BlendTree::Evaluate expects we pass a sized vector or it resizes?
    // Looking at BlendTree.cpp: `Evaluate` takes `std::vector<BoneTransform>& outPose`.
    // It assumes outPose is sized correctly or the clip resizes it. 
    // Wait, ClipNode iterates `for (size_t i = 0; i < outPose.size(); ++i)`.
    // This implies `outPose` MUST be pre-sized to the skeleton size before calling Evaluate.
    
    // FIX: We need to ensure m_finalPose is sized.
    // The safest way is to assume a standard size (e.g. 100 bones) or let the first evaluation set it.
    if (m_finalPose.empty()) m_finalPose.resize(100); // Default safety, should be set by Character
    if (m_sourcePose.size() != m_finalPose.size()) m_sourcePose.resize(m_finalPose.size());
    if (m_targetPose.size() != m_finalPose.size()) m_targetPose.resize(m_finalPose.size());

    if (m_isTransitioning) {
        // Evaluate Source
        EvaluateState(m_currentState, m_currentTime, m_sourcePose, inputs);
        
        // Evaluate Target (Start at time 0? Or sync? Standard is time 0 for new state)
        EvaluateState(m_targetState, m_transitionTime, m_targetPose, inputs);
        
        m_transitionTime += deltaTime;
        float progress = glm::clamp(m_transitionTime / m_currentTransitionDuration, 0.0f, 1.0f);
        
        // Blend result into final pose
        BlendPoses(m_sourcePose, m_targetPose, progress, m_finalPose);
        
        // End Transition
        if (progress >= 1.0f) {
            m_currentState->OnExit();
            m_currentState = m_targetState;
            m_targetState = nullptr;
            m_isTransitioning = false;
            m_currentTime = m_transitionTime; // Continue time from the transition
        }
    } else {
        // Just evaluate current
        EvaluateState(m_currentState, m_currentTime, m_finalPose, inputs);
    }
}

void AnimationStateMachine::EvaluateState(std::shared_ptr<AnimationGraphState> state, float time, 
                                         std::vector<BoneTransform>& outPose, 
                                         const std::vector<BlendInput>& inputs) {
    if (state && state->GetRootNode()) {
        state->GetRootNode()->Evaluate(time, outPose, inputs);
    } else {
        // Identity pose if no root
        for (auto& bone : outPose) {
            bone.position = glm::vec3(0.0f);
            bone.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            bone.scale = glm::vec3(1.0f);
        }
    }
}

void AnimationStateMachine::BlendPoses(const std::vector<BoneTransform>& source, 
                                      const std::vector<BoneTransform>& target, 
                                      float weight, 
                                      std::vector<BoneTransform>& outResult) {
    size_t count = std::min(source.size(), std::min(target.size(), outResult.size()));
    
    for (size_t i = 0; i < count; ++i) {
        outResult[i].position = glm::mix(source[i].position, target[i].position, weight);
        outResult[i].rotation = glm::slerp(source[i].rotation, target[i].rotation, weight);
        outResult[i].scale    = glm::mix(source[i].scale, target[i].scale, weight);
    }
}

} // namespace Animation
} // namespace CudaGame
