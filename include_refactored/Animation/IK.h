#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <unordered_map>

namespace CudaGame {
namespace Animation {

// Defines a single joint in a skeleton
struct SkeletonJoint {
    std::string name;
    int parentIndex = -1;
    glm::mat4 inverseBindPose;
    glm::mat4 currentPose;
};

// Represents a character's skeleton
class Skeleton {
public:
    Skeleton() = default;
    ~Skeleton() = default;

    void AddJoint(const SkeletonJoint& joint);
    const std::vector<SkeletonJoint>& GetJoints() const { return m_joints; }
    int GetJointIndex(const std::string& name) const;
    const SkeletonJoint* GetJoint(int index) const;
    const SkeletonJoint* GetJoint(const std::string& name) const;

    size_t GetJointCount() const { return m_joints.size(); }

private:
    std::vector<SkeletonJoint> m_joints;
    std::unordered_map<std::string, int> m_jointMap;
};

// Defines an IK chain (e.g., a character's leg or arm)
struct IKChain {
    std::string name;
    int startJointIndex = -1;
    int endJointIndex = -1;
    std::vector<int> jointIndices;
    int iterationCount = 10;
    float tolerance = 0.001f;
    bool isEnabled = true;
};

// IK component to be attached to an entity
struct IKComponent {
    std::vector<IKChain> chains;
    std::unordered_map<std::string, glm::vec3> ikTargets;
    
    void AddChain(const IKChain& chain);
    void SetTarget(const std::string& chainName, const glm::vec3& target);
    bool HasChain(const std::string& chainName) const;
};

// IK solver algorithms
namespace IKSolver {
    // Cyclic Coordinate Descent (CCD)
    void SolveCCD(Skeleton& skeleton, const IKChain& chain, const glm::vec3& target);
    
    // Forward And Backward Reaching Inverse Kinematics (FABRIK)
    void SolveFABRIK(Skeleton& skeleton, const IKChain& chain, const glm::vec3& target);
    
    // Jacobian Transpose
    void SolveJacobianTranspose(Skeleton& skeleton, const IKChain& chain, const glm::vec3& target);
}

} // namespace Animation
} // namespace CudaGame
