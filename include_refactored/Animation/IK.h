#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <unordered_map>

namespace CudaGame {
namespace Animation {

#include "Animation/AnimationResources.h"

namespace CudaGame {
namespace Animation {

// Defines an IK chain (e.g., a character's leg or arm)
struct IKChain {
    std::string name;
    int startJointIndex = -1;
    int endJointIndex = -1; // End effector
    std::vector<int> jointIndices; // All joints in chain, ordered parent to child
    int iterationCount = 10;
    float tolerance = 0.001f;
    bool isEnabled = true;
};

// IK component to be attached to an entity
struct IKComponent {
    std::vector<IKChain> chains;
    std::unordered_map<std::string, glm::vec3> ikTargets;
    
    void AddChain(const IKChain& chain) { chains.push_back(chain); }
    void SetTarget(const std::string& chainName, const glm::vec3& target) { ikTargets[chainName] = target; }
    bool HasChain(const std::string& chainName) const { return ikTargets.find(chainName) != ikTargets.end(); }
};

// IK solver algorithms
namespace IKSolver {
    // Cyclic Coordinate Descent (CCD)
    void SolveCCD(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target);
    
    // Forward And Backward Reaching Inverse Kinematics (FABRIK)
    void SolveFABRIK(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target);
    
    // Jacobian Transpose (Placeholder)
    void SolveJacobianTranspose(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target);
}

} // namespace Animation
} // namespace CudaGame
