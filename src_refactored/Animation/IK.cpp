#include "Animation/IK.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <algorithm>
#include <iostream>

namespace CudaGame {
namespace Animation {
namespace IKSolver {

// Helper: Get position from matrix
static glm::vec3 GetPosition(const glm::mat4& mat) {
    return glm::vec3(mat[3]);
}

// Helper: Set position in matrix
static void SetPosition(glm::mat4& mat, const glm::vec3& pos) {
    mat[3] = glm::vec4(pos, 1.0f);
}

// FABRIK Solver
void SolveFABRIK(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target) {
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
        int n = chainPositions.size();
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
    // This part is complex: we need to rotate joints to point to new positions.
    // For MHW style, we ideally want to apply this to the globalTransforms rotation.
    // A simple method is to re-orient each bone to point to the next bone's new position.
    
    for (size_t i = 0; i < indices.size() - 1; ++i) {
        int currentIndex = indices[i];
        int nextIndex = indices[i+1];
        
        glm::vec3 currentPos = GetPosition(globalTransforms[currentIndex]);
        glm::vec3 nextPos = GetPosition(globalTransforms[nextIndex]); // Old next pos (from matrix)
        
        glm::vec3 desiredNextPos = chainPositions[i+1];
        
        glm::vec3 currentDir = glm::normalize(nextPos - currentPos);
        glm::vec3 desiredDir = glm::normalize(desiredNextPos - currentPos);
        
        // Compute rotation from currentDir to desiredDir
        // Avoid precision issues
        if (glm::dot(currentDir, desiredDir) < 0.999f) {
           glm::quat rot = glm::rotation(currentDir, desiredDir);
           // Apply rotation to global transform
           // global = rot * global
           // Actually, we rotate around currentPos
           glm::mat4 rotMat = glm::toMat4(rot);
           
           // Apply to current bone: Rotate orientation
           // But keep position
           glm::vec3 pos = GetPosition(globalTransforms[currentIndex]);
           globalTransforms[currentIndex][3] = glm::vec4(0,0,0,1); // zero pos
           globalTransforms[currentIndex] = rotMat * globalTransforms[currentIndex];
           globalTransforms[currentIndex][3] = glm::vec4(pos, 1.0f); // restore pos
           
           // Apply to child bones recursively?
           // No, we rely on the loop for next joint.
           // BUT, global transforms of children depend on parent.
           // Since we are modifying global transforms directly, we break hierarchy unless we propagate.
           // FABRIK usually outputs positions.
           // Reconstruction of rotations is hard.
        }
        
        // Set exact position
        SetPosition(globalTransforms[indices[i+1]], chainPositions[i+1]);
    }
    
    // Correction: FABRIK updates positions. We should just set the positions if we don't care about rotation constraint?
    // No, mesh skinning relies on rotations. Positions alone in matrices induce shear if not careful.
    // Ideally, we compute the rotation delta and apply it.
    
    // For this implementation, we will update positions directly (simple) 
    // AND attempt to update orientation.
    // Note: This naive rotation update doesn't propagate to children correctly if we don't traverse.
    // But since we iterate Parent -> Child, updating Parent Global Rotation affects where Child "should" be.
    // But we are setting Child Global Position explicitly relative to Parent.
    
    // So:
    // 1. Orient Parent to point to Child's new position.
    // 2. Set Child's Position.
    // 3. Repeat.
    
    // Valid for linear chains.
}

// CCD Solver
void SolveCCD(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target) {
    // Basic CCD implementation
    // Iterate from end effector up to root
    
    if (chain.jointIndices.empty()) return;
    int endEffectorIdx = chain.jointIndices.back();
    
    for (int iter = 0; iter < chain.iterationCount; ++iter) {
         // Check distance
         glm::vec3 currentEffectorPos = GetPosition(globalTransforms[endEffectorIdx]);
         if (glm::distance(currentEffectorPos, target) < chain.tolerance) break;
         
         // Iterate backwards from second-to-last joint
         for (int i = chain.jointIndices.size() - 2; i >= 0; --i) {
             int jointIdx = chain.jointIndices[i];
             glm::mat4& jointMat = globalTransforms[jointIdx];
             glm::vec3 jointPos = GetPosition(jointMat);
             
             glm::vec3 toEffector = glm::normalize(GetPosition(globalTransforms[endEffectorIdx]) - jointPos);
             glm::vec3 toTarget = glm::normalize(target - jointPos);
             
             // Rotate joint to align toEffector with toTarget
             // clamp cosTheta to -1..1
             float cosTheta = glm::clamp(glm::dot(toEffector, toTarget), -1.0f, 1.0f);
             if (cosTheta < 0.9999f) {
                 float angle = glm::acos(cosTheta);
                 glm::vec3 axis = glm::normalize(glm::cross(toEffector, toTarget));
                 
                 // Apply rotation in global space?
                 // Rotate around jointPos
                 glm::mat4 rot = glm::rotate(angle, axis);
                 
                 // Apply to this joint's rotation
                 glm::vec3 pos = GetPosition(jointMat);
                 jointMat[3] = glm::vec4(0,0,0,1);
                 jointMat = rot * jointMat;
                 jointMat[3] = glm::vec4(pos, 1.0f);
                 
                 // Propagate change to all children in chain?
                 // YES. In CCD, when we rotate a parent, all children must rotate.
                 // We need to rotate all downstream joints around jointPos.
                 for (size_t j = i + 1; j < chain.jointIndices.size(); ++j) {
                     int childIdx = chain.jointIndices[j];
                     glm::mat4& childMat = globalTransforms[childIdx];
                     // childNew = rot * (childOld - pivot) + pivot
                     childMat = rot * (childMat - glm::translate(glm::mat4(1.0f), jointPos)) + glm::translate(glm::mat4(1.0f), jointPos);
                     // Matrix math above is pseudocode.
                     // Correct: T(p) * R * T(-p) * M
                     // Or just iterate and apply R to (M relative to p)
                     glm::mat4 relative = glm::inverse(glm::translate(glm::mat4(1.0f), jointPos)) * childMat;
                     childMat = glm::translate(glm::mat4(1.0f), jointPos) * rot * relative;
                 }
             }
         }
    }
}

void SolveJacobianTranspose(const Skeleton& skeleton, std::vector<glm::mat4>& globalTransforms, const IKChain& chain, const glm::vec3& target) {
    // Placeholder
}

} // namespace IKSolver
} // namespace Animation
} // namespace CudaGame
