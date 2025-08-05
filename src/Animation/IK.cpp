#include "Animation/IK.h"
#include <glm/gtx/quaternion.hpp>
#include <iostream>
#include <algorithm>

namespace CudaGame {
namespace Animation {

// Skeleton implementation
void Skeleton::AddJoint(const SkeletonJoint& joint) {
    m_joints.push_back(joint);
    m_jointMap[joint.name] = m_joints.size() - 1;
}

int Skeleton::GetJointIndex(const std::string& name) const {
    auto it = m_jointMap.find(name);
    return (it != m_jointMap.end()) ? it->second : -1;
}

const SkeletonJoint* Skeleton::GetJoint(int index) const {
    return (index >= 0 && index < static_cast<int>(m_joints.size())) ? &m_joints[index] : nullptr;
}

const SkeletonJoint* Skeleton::GetJoint(const std::string& name) const {
    int index = GetJointIndex(name);
    return (index != -1) ? GetJoint(index) : nullptr;
}

// IKComponent implementation
void IKComponent::AddChain(const IKChain& chain) {
    chains.push_back(chain);
}

void IKComponent::SetTarget(const std::string& chainName, const glm::vec3& target) {
    ikTargets[chainName] = target;
}

bool IKComponent::HasChain(const std::string& chainName) const {
    return std::any_of(chains.begin(), chains.end(), 
        [&](const IKChain& chain) { return chain.name == chainName; });
}

// IKSolver implementations (simplified placeholders)
namespace IKSolver {

void SolveCCD(Skeleton& skeleton, const IKChain& chain, const glm::vec3& target) {
    std::cout << "[IK] Solving CCD for chain: " << chain.name << std::endl;
    // In a real implementation, this would iterate from the end effector backwards,
    // rotating each joint to point towards the target.
}

void SolveFABRIK(Skeleton& skeleton, const IKChain& chain, const glm::vec3& target) {
    std::cout << "[IK] Solving FABRIK for chain: " << chain.name << std::endl;
    // In a real implementation, this would involve forward and backward passes
    // to reposition the joints along the line to the target.
}

void SolveJacobianTranspose(Skeleton& skeleton, const IKChain& chain, const glm::vec3& target) {
    std::cout << "[IK] Solving Jacobian Transpose for chain: " << chain.name << std::endl;
    // This is a more advanced technique that uses the Jacobian matrix
    // to approximate the relationship between joint angles and end effector position.
}

} // namespace IKSolver

} // namespace Animation
} // namespace CudaGame
