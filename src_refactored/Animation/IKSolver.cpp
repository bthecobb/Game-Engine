#include "Animation/IKSolver.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <algorithm>
#include <iostream>

namespace CudaGame {
namespace Animation {

// Helper: Get position from matrix
static glm::vec3 GetPosition(const glm::mat4& mat) {
    return glm::vec3(mat[3]);
}

// Helper: Set position in matrix
static void SetPosition(glm::mat4& mat, const glm::vec3& pos) {
    mat[3] = glm::vec4(pos, 1.0f);
}

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
    // float beta = acos(cosBeta); // Unused variable
    
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
    
    // Angle Alpha is how much we rotate AWAY from the target vector, INTO the plane (towards pole).
    // Rotation Axis is 'planeNormal'.
    
    // Let's compute the desired direction of the Upper Leg (Root->Joint).
    // It lies in the plane, rotated by Alpha from Root->Target.
    glm::quat alphaRot = glm::angleAxis(alpha, planeNormal);
    glm::vec3 desiredUpperDir = alphaRot * rootToTarget;
    
    // Now we need the rotation that takes the INITIAL bind Upper Dir to this DESIRED dir.
    // Assumption: Bones point along -Y axis in our procedural skeleton.
    glm::vec3 boneForward = glm::vec3(0.0f, -1.0f, 0.0f); 
    
    // Create Root Rotation
    // Align BoneForward (-Y) to DesiredUpperDir
    outRootRot = glm::rotation(boneForward, desiredUpperDir);
    
    // Twist correction: Align the "Knee Axis" (X) with Plane Normal
    glm::vec3 boneAxis = glm::vec3(1.0f, 0.0f, 0.0f); // X axis is knee bend axis
    glm::vec3 currentAxis = outRootRot * boneAxis;
    glm::quat twist = glm::rotation(currentAxis, planeNormal);
    outRootRot = twist * outRootRot;
    
    // Create Joint Rotation (Knee)
    // The Lower Leg direction is derived from Joint -> Target
    glm::vec3 desiredLowerDir = glm::normalize(targetPos - (rootPos + desiredUpperDir * a));
    outJointRot = glm::rotation(boneForward, desiredLowerDir);
    
    // Twist correction for joint
    currentAxis = outJointRot * boneAxis;
    twist = glm::rotation(currentAxis, planeNormal);
    outJointRot = twist * outJointRot;
    
    return true;
}

// FABRIK Solver
void IKSolver::SolveFABRIK(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target) {
    if (chain.jointIndices.empty()) return;
    
    // 1. Extract chain positions
    std::vector<glm::vec3> chainPositions;
    std::vector<float> boneLengths;
    std::vector<int> indices = chain.jointIndices; // Ordered Parent -> Child
    
    for (int idx : indices) {
        if (idx >= 0 && idx < (int)globalTransforms.size()) {
            chainPositions.push_back(GetPosition(globalTransforms[idx]));
        } else {
            return; // Invalid index
        }
    }
    
    // Calculate lengths
    for (size_t i = 0; i < chainPositions.size() - 1; ++i) {
        boneLengths.push_back(glm::length(chainPositions[i+1] - chainPositions[i]));
    }
    
    // Root position (fixed)
    glm::vec3 rootPos = chainPositions[0];
    
    // Check reachability
    float totalLength = 0.0f;
    for (float l : boneLengths) totalLength += l;
    
    float distToTarget = glm::distance(rootPos, target);
    
    if (distToTarget > totalLength) {
        // Target unreachable - stretch
        for (size_t i = 0; i < boneLengths.size(); ++i) {
            float r = glm::distance(target, rootPos);
            float lambda = boneLengths[i] / r;
            chainPositions[i+1] = (1.0f - lambda) * chainPositions[i] + lambda * target;
        }
    } else {
        // Target reachable - iterate
        int n = (int)chainPositions.size();
        for (int iter = 0; iter < chain.iterationCount; ++iter) {
            // Backward: Set end effector to target
            chainPositions[n-1] = target;
            for (int i = n - 2; i >= 0; --i) {
                float r = glm::distance(chainPositions[i+1], chainPositions[i]);
                float lambda = boneLengths[i] / r;
                chainPositions[i] = (1.0f - lambda) * chainPositions[i+1] + lambda * chainPositions[i];
            }
            
            // Forward: Set root to original
            chainPositions[0] = rootPos;
            for (int i = 0; i < n - 1; ++i) {
                float r = glm::distance(chainPositions[i+1], chainPositions[i]);
                float lambda = boneLengths[i] / r;
                chainPositions[i+1] = (1.0f - lambda) * chainPositions[i] + lambda * chainPositions[i+1];
            }
            
            // Check tolerance
            if (glm::distance(chainPositions[n-1], target) < chain.tolerance) {
                break;
            }
        }
    }
    
    // Apply new positions to rotations
    for (size_t i = 0; i < indices.size() - 1; ++i) {
        int currentIndex = indices[i];
        int nextIndex = indices[i+1];
        
        glm::vec3 currentPos = GetPosition(globalTransforms[currentIndex]);
        glm::vec3 nextPos = GetPosition(globalTransforms[nextIndex]); // Old next pos (from matrix)
        
        glm::vec3 desiredNextPos = chainPositions[i+1];
        
        glm::vec3 currentDir = glm::normalize(nextPos - currentPos);
        glm::vec3 desiredDir = glm::normalize(desiredNextPos - currentPos);
        
        // Compute rotation from currentDir to desiredDir
        if (glm::dot(currentDir, desiredDir) < 0.999f) {
           glm::quat rot = glm::rotation(currentDir, desiredDir);
           glm::mat4 rotMat = glm::toMat4(rot);
           
           // Apply to current bone: Rotate orientation
           glm::vec3 pos = GetPosition(globalTransforms[currentIndex]);
           globalTransforms[currentIndex][3] = glm::vec4(0,0,0,1); // zero pos
           globalTransforms[currentIndex] = rotMat * globalTransforms[currentIndex];
           globalTransforms[currentIndex][3] = glm::vec4(pos, 1.0f); // restore pos
        }
        
        // Set exact position
        SetPosition(globalTransforms[indices[i+1]], chainPositions[i+1]);
    }
}

// CCD Solver
void IKSolver::SolveCCD(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target) {
    if (chain.jointIndices.empty()) return;
    int endEffectorIdx = chain.jointIndices.back();
    
    for (int iter = 0; iter < chain.iterationCount; ++iter) {
         // Check distance
         glm::vec3 currentEffectorPos = GetPosition(globalTransforms[endEffectorIdx]);
         if (glm::distance(currentEffectorPos, target) < chain.tolerance) break;
         
         // Iterate backwards from second-to-last joint
         for (int i = (int)chain.jointIndices.size() - 2; i >= 0; --i) {
             int jointIdx = chain.jointIndices[i];
             glm::mat4& jointMat = globalTransforms[jointIdx];
             glm::vec3 jointPos = GetPosition(jointMat);
             
             glm::vec3 toEffector = glm::normalize(GetPosition(globalTransforms[endEffectorIdx]) - jointPos);
             glm::vec3 toTarget = glm::normalize(target - jointPos);
             
             // Rotate joint to align toEffector with toTarget
             float cosTheta = glm::clamp(glm::dot(toEffector, toTarget), -1.0f, 1.0f);
             if (cosTheta < 0.9999f) {
                 float angle = glm::acos(cosTheta);
                 glm::vec3 axis = glm::normalize(glm::cross(toEffector, toTarget));
                 
                 // Apply rotation around jointPos
                 glm::mat4 rot = glm::rotate(angle, axis);
                 
                 // Apply to this joint's rotation
                 glm::vec3 pos = GetPosition(jointMat);
                 jointMat[3] = glm::vec4(0,0,0,1);
                 jointMat = rot * jointMat;
                 jointMat[3] = glm::vec4(pos, 1.0f);
                 
                 // Propagate change to children
                 for (size_t j = i + 1; j < chain.jointIndices.size(); ++j) {
                     int childIdx = chain.jointIndices[j];
                     glm::mat4& childMat = globalTransforms[childIdx];
                     
                     // childNew = rot * (childOld - pivot) + pivot
                     // Simplified translation logic
                     glm::vec3 childPos = GetPosition(childMat);
                     glm::vec3 relPos = childPos - jointPos;
                     // Rotate Position
                     // glm::vec3 newRelPos = glm::rotate(rot, relPos); // glm::rotate takes mat4 or quat
                     glm::vec3 newRelPos = glm::vec3(rot * glm::vec4(relPos, 1.0f));
                     
                     // Rotate Orientation (simple approximation)
                     // childMat = rot * childMat; // This rotates around global origin? No.
                     
                     // Correct way: Apply 'rot' to childMat
                     childMat[3] = glm::vec4(0,0,0,1);
                     childMat = rot * childMat;
                     childMat[3] = glm::vec4(jointPos + newRelPos, 1.0f);
                 }
             }
         }
    }
}

} // namespace Animation
} // namespace CudaGame
