#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>

namespace CudaGame {
namespace Animation {

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
};

} // namespace Animation
} // namespace CudaGame
