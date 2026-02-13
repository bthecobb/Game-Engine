#include "Animation/IKSolver.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <iostream>

namespace CudaGame {
namespace Animation {

bool IKSolver::SolveTwoBoneIK(
    const glm::vec3& rootPos, 
    const glm::vec3& jointPos, 
    const glm::vec3& endPos, 
    const glm::vec3& targetPos, 
    const glm::vec3& poleVector,
    glm::quat& outRootRot, 
    glm::quat& outJointRot
) {
    // 1. Calculate Lengths
    float a = glm::length(jointPos - rootPos);     // Upper Leg
    float b = glm::length(endPos - jointPos);      // Lower Leg
    float c = glm::length(targetPos - rootPos);    // Distance to Target
    
    // Safety: prevent division by zero or unreachable targets
    if (a < 0.001f || b < 0.001f) return false;
    
    // Clamp reach
    float maxReach = a + b;
    float targetDist = c;
    if (targetDist > maxReach * 0.999f) {
        targetDist = maxReach * 0.999f; // Slightly less to avoid straight line singularity
        c = targetDist;
    }
    
    // 2. Law of Cosines
    // Alpha: Angle at Root
    // Beta: Angle at Joint
    float cosAlpha = (a*a + c*c - b*b) / (2.0f * a * c);
    float cosBeta = (a*a + b*b - c*c) / (2.0f * a * b);
    
    // Clamp values for acos
    cosAlpha = glm::clamp(cosAlpha, -1.0f, 1.0f);
    cosBeta = glm::clamp(cosBeta, -1.0f, 1.0f);
    
    float alpha = acos(cosAlpha);
    float beta = acos(cosBeta); // Internal angle (usually < 180)
    // The actual bend deviation from straight line involves (PI - beta) usually?
    // Let's think: if beta is small, leg is folded. if beta is 180 (PI), leg is straight.
    // joint rotation usually starts at 0 (straight) or 0 (folded).
    // For a knee, 0 usually means straight. 
    // Wait, Bind Pose dictates this. 
    // Let's assume we output GLOBAL rotation corrections.
    
    // 3. Solving Rotation
    
    // Current vector directions
    glm::vec3 rootToTarget = glm::normalize(targetPos - rootPos);
    
    // We construct a coordinate system for the limb.
    // Plane defined by (Root, Target, Pole).
    glm::vec3 planeNormal = glm::cross(rootToTarget, poleVector);
    if (glm::length2(planeNormal) < 0.001f) {
        // Pole vector aligns with target vector - undefined plane. Pick Up vector.
        planeNormal = glm::cross(rootToTarget, glm::vec3(0,1,0));
    }
    planeNormal = glm::normalize(planeNormal);
    
    // Compute local Up vector for the root joint (perpendicular to target vector in the plane)
    glm::vec3 rootUp = glm::cross(planeNormal, rootToTarget);
    
    // Angle Alpha is how much we rotate AWAY from the target vector, INTO the plane (towards pole).
    // Rotation Axis is 'planeNormal'.
    // If pole vector is "Forward" (Knee), we rotate 'alpha' around planeNormal.
    
    // ROTATION 1: Pivot Root to look at Target
    // Simple LookAt from Root to Target, with Up as planeNormal? 
    // Ideally we want Root->Joint vector to be rotated by Alpha from the Root->Target line.
    
    // Let's compute the desired direction of the Upper Leg (Root->Joint).
    // It lies in the plane, rotated by Alpha from Root->Target.
    glm::quat alphaRot = glm::angleAxis(alpha, planeNormal);
    glm::vec3 desiredUpperDir = alphaRot * rootToTarget;
    
    // Now we need the rotation that takes the INITIAL bind Upper Dir to this DESIRED dir.
    // BUT usually IK outputs absolute rotations or overrides.
    // Since we are outputting "New Global Rotation", let's construct it.
    // We need to know the bone's local forward axis.
    // Assumption: Bones point along Y axis? Or X? 
    // In our `ProceduralAnimation`, legs point down (-Y).
    glm::vec3 boneForward = glm::vec3(0.0f, -1.0f, 0.0f); 
    
    // Create Root Rotation
    // Align BoneForward (-Y) to DesiredUpperDir
    outRootRot = glm::rotation(boneForward, desiredUpperDir);
    // But this leaves Twist undefined. We need to align the "Knee Axis" too.
    // The Knee Axis (rotation axis) is the Plane Normal.
    // The bone's local rotation axis (e.g. X axis) should align with Plane Normal.
    glm::vec3 boneAxis = glm::vec3(1.0f, 0.0f, 0.0f); // X axis is knee bend axis
    glm::vec3 currentAxis = outRootRot * boneAxis;
    glm::quat twist = glm::rotation(currentAxis, planeNormal);
    outRootRot = twist * outRootRot;
    
    // Create Joint Rotation (Knee)
    // The Lower Leg direction is rotated from Upper Leg by (PI - Beta).
    // Or simpler: We know where the Joint IS (Root + DesiredUpper * a).
    // We know where End IS (Target).
    // LowerDir = normalize(Target - Joint).
    // Joint Rotation should align BoneForward to LowerDir.
    // And ensure its axis aligns with plane normal.
    glm::vec3 desiredLowerDir = glm::normalize(targetPos - (rootPos + desiredUpperDir * a));
    outJointRot = glm::rotation(boneForward, desiredLowerDir);
    
    // Twist correction for joint
    currentAxis = outJointRot * boneAxis;
    twist = glm::rotation(currentAxis, planeNormal);
    outJointRot = twist * outJointRot;
    
    return true;
}

} // namespace Animation
} // namespace CudaGame
