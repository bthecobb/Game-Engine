#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>

#include "Animation/AnimationResources.h"
#include <vector>
#include <string>
#include <unordered_map>

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


class IKSolver {
public:
    /**
     * Solves Two-Bone IK (e.g., Leg: Hip->Knee->Foot, Arm: Shoulder->Elbow->Hand).
     * 
     * @param rootPos Global position of the start bone (Hip).
     * @param jointPos Global position of the middle bone (Knee).
     * @param endPos Global position of the end bone (Ankle).
     * @param targetPos Desired global position for the end bone.
     * @param poleVector Direction to bend the joint towards (e.g. forward for knees).
     * @param outRootRot Output: New global rotation correction for the root bone.
     * @param outJointRot Output: New global rotation correction for the joint bone.
     * @return True if solution found.
     */
    static bool SolveTwoBoneIK(
        const glm::vec3& rootPos, 
        const glm::vec3& jointPos, 
        const glm::vec3& endPos, 
        const glm::vec3& targetPos, 
        const glm::vec3& poleVector,
        glm::quat& outRootRot, 
        glm::quat& outJointRot
    );

    // Cyclic Coordinate Descent (CCD)
    static void SolveCCD(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target);
    
    // Forward And Backward Reaching Inverse Kinematics (FABRIK)
    static void SolveFABRIK(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target);
};

} // namespace Animation
} // namespace CudaGame
