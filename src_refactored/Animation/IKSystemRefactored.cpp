#pragma message("COMPILING IKSystem.cpp")
#include "Animation/IKSystem.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderComponents.h"
#include "Animation/AnimationComponent.h"
#include "Animation/IKSolver.h"
#include "Physics/PhysXPhysicsSystem.h" // Added
#include <iostream>

namespace CudaGame {
namespace Animation {

IKSystem::IKSystem() {
    m_coordinator = &Core::Coordinator::GetInstance();
    // IK should update after animation but before rendering
}

bool IKSystem::Initialize() {
    std::cout << "[IKSystem] Initializing procedural animation system..." << std::endl;
    return true;
}

void IKSystem::Shutdown() {
    std::cout << "[IKSystem] Shutting down procedural animation system." << std::endl;
}

void IKSystem::Update(float deltaTime) {
    for (auto const& entityId : mEntities) {
        // System Signature guarantees we have IKComponent, AnimationComponent, TransformComponent
        // But we should verify or just use them.
        
        // Procedural Placement Logic
        if (m_footPlacementSettings.count(entityId)) {
            UpdateFootPlacement(entityId, deltaTime);
        }
        
        if (m_handPlacementSettings.count(entityId)) {
            UpdateHandPlacement(entityId, deltaTime);
        }
        
        if (m_lookAtTargets.count(entityId)) {
            UpdateLookAt(entityId, deltaTime);
        }
        
        // Solve IK Chains
        // Only if we have IKComponent (guaranteed by signature, usually)
        if (m_coordinator->HasComponent<IKComponent>(entityId)) {
            auto& ikComp = m_coordinator->GetComponent<IKComponent>(entityId);
            for (const auto& chain : ikComp.chains) {
                if (chain.isEnabled) {
                    SolveIKChain(entityId, chain);
                }
            }
        }
    }
}

// Placeholder implementations for IK system methods...

void IKSystem::TriggerFootstep(Core::Entity entity, const std::string& chainName) {
    if (!m_coordinator->HasComponent<IKComponent>(entity) || !m_physicsSystem) return;
    auto& ikComp = m_coordinator->GetComponent<IKComponent>(entity);
    
    // Find the chain to enable ground tracking
    for (auto& chain : ikComp.chains) {
        if (chain.name == chainName) {
            chain.isEnabled = true; // Actively adapt to ground
            
            // We use a simple custom timer stored in ikTargets to track IK fade-out,
            // or we just leave it governed by the procedural generation. 
            // For now, enable it so Update() will solve it each frame.
            return;
        }
    }
}

void IKSystem::UpdateFootPlacement(uint32_t entityId, float /*deltaTime*/) {
    if (!m_physicsSystem || !m_coordinator->HasComponent<Rendering::TransformComponent>(entityId) ||
        !m_coordinator->HasComponent<IKComponent>(entityId) ||
        !m_coordinator->HasComponent<AnimationComponent>(entityId)) {
        return;
    }
    
    auto& transform = m_coordinator->GetComponent<Rendering::TransformComponent>(entityId);
    auto& ikComp = m_coordinator->GetComponent<IKComponent>(entityId);
    auto& anim = m_coordinator->GetComponent<AnimationComponent>(entityId);
    
    // Only adapt chains that are currently flagged as planted (e.g. by Footstep event)
    glm::mat4 modelMatrix = transform.getMatrix();
    
    for (auto& chain : ikComp.chains) {
        if (!chain.isEnabled || chain.endJointIndex < 0 || chain.endJointIndex >= (int)anim.globalTransforms.size()) {
            continue;
        }
        
        // 1. Where does the animation *want* the foot to be in World Space?
        glm::mat4 localFootMat = anim.globalTransforms[chain.endJointIndex];
        glm::vec3 worldFootPos = glm::vec3(modelMatrix * localFootMat[3]);
        
        // 2. Cast a ray down from above the foot
        glm::vec3 origin = worldFootPos + glm::vec3(0.0f, 0.5f, 0.0f); // Start 0.5m above foot
        glm::vec3 dir(0.0f, -1.0f, 0.0f); // Straight down
        glm::vec3 hitPos;
        
        if (m_physicsSystem->Raycast(origin, dir, 1.0f, hitPos)) {
            // 3. Set the target to the hit position (+ ankle offset)
            float ankleHeight = 0.1f; // Adjust based on character geometry
            ikComp.SetTarget(chain.name, hitPos + glm::vec3(0.0f, ankleHeight, 0.0f));
        } else {
            // No ground found beneath foot, disable IK for this leg
            chain.isEnabled = false;
        }
    }
}

void IKSystem::UpdateHandPlacement(uint32_t entityId, float /*deltaTime*/) {
    // Check if entity has required components
    if (!m_coordinator->HasComponent<Rendering::TransformComponent>(entityId) ||
        !m_coordinator->HasComponent<IKComponent>(entityId)) {
        return;
    }
    
    auto& transform = m_coordinator->GetComponent<Rendering::TransformComponent>(entityId);
    auto& ikData = m_coordinator->GetComponent<IKComponent>(entityId);
    
    // Placeholder implementation for hand placement
    // Would solve hand positions for interactions
}

void IKSystem::UpdateLookAt(uint32_t entityId, float /*deltaTime*/) {
    // Check if entity has required components
    if (!m_coordinator->HasComponent<Rendering::TransformComponent>(entityId) ||
        !m_coordinator->HasComponent<IKComponent>(entityId)) {
        return;
    }
    
    auto& transform = m_coordinator->GetComponent<Rendering::TransformComponent>(entityId);
    auto& ikData = m_coordinator->GetComponent<IKComponent>(entityId);
    
    // Placeholder implementation for look-at IK
    // Would orient head/eyes toward target
}

void IKSystem::SolveIKChain(uint32_t entityId, const IKChain& chain) {
    // Check if entity has required components
    if (!m_coordinator->HasComponent<Rendering::TransformComponent>(entityId) ||
        !m_coordinator->HasComponent<AnimationComponent>(entityId)) {
        return;
    }
    
    auto& transform = m_coordinator->GetComponent<Rendering::TransformComponent>(entityId);
    auto& anim = m_coordinator->GetComponent<AnimationComponent>(entityId);
    
    if (!anim.skeleton) return;
    
    // Get target position
    glm::vec3 targetPos = glm::vec3(0.0f);
    
    // Check if target is set in IKComponent
    if (!m_coordinator->HasComponent<IKComponent>(entityId)) return;
    auto& ikComp = m_coordinator->GetComponent<IKComponent>(entityId);
    
    auto it = ikComp.ikTargets.find(chain.name);
    if (it != ikComp.ikTargets.end()) {
        targetPos = it->second;
    } else {
        // Fallback or dynamic target logic
        // For now, skip if no target
        return;
    }
    
    // Convert target to local space if needed?
    // FABRIK usually works in Model Space (which globalTransforms are often in, relative to root) 
    // OR World Space.
    // 'globalTransforms' in AnimationComponent are usually Model Space (relative to Entity Transform).
    // If 'targetPos' is World Space, we need to transform it to Model Space.
    
    // Assume targetPos is World Space. We need it in Model Space.
    glm::mat4 modelMatrix = transform.getMatrix();
    glm::mat4 invModel = glm::inverse(modelMatrix);
    glm::vec3 modelTarget = glm::vec3(invModel * glm::vec4(targetPos, 1.0f));
    
    // Solve
    // We modify globalTransforms directly
    IKSolver::SolveFABRIK(*anim.skeleton, anim.globalTransforms, chain, modelTarget);
    
    // Note: After modification, we might need to re-compute finalBoneMatrices
    // But AnimationSystem usually does that. If IK runs *after* AnimationSystem but *before* Render,
    // we need to update finalBoneMatrices here too.
    for (int idx : chain.jointIndices) {
        if (idx >= 0 && idx < (int)anim.finalBoneMatrices.size()) {
            const auto& bone = anim.skeleton->bones[idx];
            anim.finalBoneMatrices[idx] = anim.globalTransforms[idx] * bone.inverseBindPose;
        }
    }
}

} // namespace Animation
} // namespace CudaGame
