#include "Gameplay/AnimationControllerSystem.h"
#include "Core/Coordinator.h"
#include "Animation/IKSolver.h"
#include <iostream>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include "Animation/BoneTransform.h"
#include "Animation/AnimationStateMachine.h"
#include "Physics/PhysicsComponents.h"

// Helper to extract translation from matrix
static glm::vec3 GetPosition(const glm::mat4& m) { return glm::vec3(m[3]); }

namespace CudaGame {
namespace Gameplay {

// Helper to apply IK for a single leg
void ApplyFootIK(Animation::AnimationComponent& anim, const std::string& side) {
    auto& bones = anim.skeleton->bones;
    
    // Find Chain Indices (Naive string search - optimization: cache indices)
    // Find Chain Indices (Naive string search - optimization: cache indices)
    int hipIdx = -1, kneeIdx = -1, footIdx = -1;
    for (int i = 0; i < (int)bones.size(); ++i) {
        const auto& b = bones[i];
        if (b.name == side + "UpLeg") hipIdx = i;
        if (b.name == side + "Leg") kneeIdx = i;
        if (b.name == side + "Foot") footIdx = i;
    }
    
    if (hipIdx == -1 || kneeIdx == -1 || footIdx == -1) return;
    
    // positions from Forward Kinematics
    glm::vec3 rootPos = GetPosition(anim.globalTransforms[hipIdx]);
    glm::vec3 jointPos = GetPosition(anim.globalTransforms[kneeIdx]);
    glm::vec3 endPos = GetPosition(anim.globalTransforms[footIdx]);
    
    // Ground Check (Y = 0)
    // If foot is penetrating ground (y < 0), we want to lift it.
    // Or better: We want to plant the foot at Y=0 if it's close.
    // Simple logic: Target is EndPos projected to Y=0.
    // But we only clamp if it's below.
    glm::vec3 targetPos = endPos;
    if (targetPos.y < 0.05f) { // epsilon
        targetPos.y = 0.0f;
    } else {
        // If above ground, IK not needed? 
        // For 'Planting', we might want to pull it DOWN if it's close?
        // Let's just fix floor penetration for phase 8.
        return; 
    }
    
    // Pole Vector: Knee points forward (Z+) relative to character?
    // Assume Z+ is global forward for this test. 
    glm::vec3 pole = glm::vec3(0,0,1); 
    
    glm::quat newRootRot, newJointRot;
    if (Animation::IKSolver::SolveTwoBoneIK(rootPos, jointPos, endPos, targetPos, pole, newRootRot, newJointRot)) {
        // Apply changes to Global Transforms
        // Note: This overrides scale/translation, strictly applying rotation + positional constraint
        
        // Root: Pos stays same, Rot changes
        anim.globalTransforms[hipIdx] = glm::translate(glm::mat4(1.0f), rootPos) * glm::mat4_cast(newRootRot);
        
        // Joint: Pos is derived from New Root! IKSolver assumes connected.
        // We recalculate Joint Pos based on Root
        // Length 1
        float l1 = glm::length(jointPos - rootPos);
        glm::vec3 newJointPos = rootPos + (newRootRot * glm::vec3(0,-1,0)) * l1; // Assuming BoneForward is -Y
        // Wait, Procedural Skeleton has specific offsets.
        // IKSolver logic assumed BoneForward is -Y.
        // Let's verify Procedural Generator offsets for legs.
        // LeftUpLeg offset was (-0.15, -0.1, 0) relative to Hip. 
        // LeftLeg offset was (0, -0.4, 0) relative to UpLeg. -> This is -Y axis!
        // So yes, BoneForward is -Y.
        
        anim.globalTransforms[kneeIdx] = glm::translate(glm::mat4(1.0f), newJointPos) * glm::mat4_cast(newJointRot);
        
        // Foot: Pos derived from New Joint
        float l2 = glm::length(endPos - jointPos);
        glm::vec3 newFootPos = newJointPos + (newJointRot * glm::vec3(0,-1,0)) * l2;
        
        // Foot rotation? Usually aligns with ground.
        // For now, keep original rotation or identity?
        // Just Update position
        anim.globalTransforms[footIdx] = glm::translate(glm::mat4(1.0f), newFootPos) * glm::mat4_cast(glm::quat(1,0,0,0)); 
    }
}

AnimationControllerSystem::AnimationControllerSystem() {
}

bool AnimationControllerSystem::Initialize() {
    std::cout << "[AnimationControllerSystem] Initialized" << std::endl;
    return true;
}

void AnimationControllerSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    for (auto const& entity : mEntities) {
        if (!coordinator.HasComponent<AnimationControllerComponent>(entity) || 
            !coordinator.HasComponent<Animation::AnimationComponent>(entity)) {
            continue;
        }
        
        auto& controller = coordinator.GetComponent<AnimationControllerComponent>(entity);
        auto& anim = coordinator.GetComponent<Animation::AnimationComponent>(entity);
        
        UpdateStateMachine(entity, controller, anim, deltaTime);
    }
}

void AnimationControllerSystem::UpdateStateMachine(Core::Entity entity, 
                                                  AnimationControllerComponent& controller, 
                                                  Animation::AnimationComponent& anim,
                                                  float deltaTime) {
    using namespace Animation;
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // --- Phase 1: Input Gathering ---
    float speed = 0.0f;
    float vSpeed = 0.0f;
    bool isGrounded = true; // Assume true for now if no physics
    
    if (coordinator.HasComponent<Physics::RigidbodyComponent>(entity)) {
        auto& rb = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
        
        // Horizontal Speed (XZ)
        glm::vec3 velocity = rb.velocity;
        speed = glm::length(glm::vec2(velocity.x, velocity.z));
        vSpeed = velocity.y;
        
        // Update Controller Params for Debug/Graph
        controller.floatParams["Speed"] = speed;
        controller.floatParams["VerticalSpeed"] = vSpeed;
    }
    
    // Read controller overrides (e.g. from Input System)
    bool isWallRunning = controller.boolParams["IsWallRunning"];
    bool isDashing = controller.boolParams["IsDashing"];
    
    // --- Phase 2: State Logic ---
    AnimationState targetState = AnimationState::IDLE;
    
    if (isWallRunning) {
        targetState = AnimationState::WALL_RUNNING;
    } else if (isDashing) {
        targetState = AnimationState::RUNNING; 
    } else {
        // Ground movement logic
        if (speed > 0.1f) {
            if (speed > 6.0f) {
                targetState = AnimationState::RUNNING;
            } else {
                targetState = AnimationState::WALKING;
            }
        } else {
            targetState = AnimationState::IDLE;
        }
    }
    
    // --- Phase 3: Transition Handling ---
    if (targetState != anim.currentState) {
        // State Change Detected
        anim.previousState = anim.currentState;
        anim.currentState = targetState;
        anim.animationTime = 0.0f;
        anim.hasTransitioned = true;
        
        // Resolve Clip Name
        std::string clipName = "";
        if (anim.stateMap.count(targetState)) {
            clipName = anim.stateMap[targetState];
        }
        
        if (!clipName.empty()) {
            anim.currentAnimation = clipName;
            anim.isPlaying = true; // Restart playback
            std::cout << "[Anim] Entity " << entity << " Transition: " << (int)anim.previousState << " -> " << (int)anim.currentState << " (" << clipName << ")" << std::endl;
        } else {
             // Fallback or maintain current?
             // std::cout << "[Anim] Warning: No clip for state " << (int)targetState << std::endl;
        }
    }
    
    // Pass blend params (for Blend Trees)
    anim.movementSpeed = speed;
    
    // --- PHASE 8: Graph Evaluation ---
    if (controller.stateMachine) {
        // 1. Sync legacy params to Graph Inputs (Migration helper)
        controller.SetInput("Speed", speed);
        controller.SetInput("VerticalSpeed", vSpeed);
        // ... others
        
        // 2. Update Graph
        controller.stateMachine->Update(deltaTime, controller.nextFrameInputs);
        controller.nextFrameInputs.clear(); // Consume inputs
        
        // 3. Apply Output Pose to AnimationComponent
        const std::vector<Animation::BoneTransform>& localPose = controller.stateMachine->GetGlobalPose(); // Name is GlobalPose but it returns local bone transforms usually?
        // Wait, generic BlendTrees return Local Bone Transforms relative to parent.
        // We need to compute Global Matrices for rendering.
        
        // Resize globals if needed
        if (anim.globalTransforms.size() != localPose.size()) {
            anim.globalTransforms.resize(localPose.size());
        }
        
        // Forward Kinematics (Root -> Children)
        // We need the skeleton hierarchy for this.
        if (anim.skeleton) {
            for (size_t i = 0; i < anim.skeleton->bones.size(); ++i) {
                if (i >= localPose.size()) continue;
                
                const auto& bone = anim.skeleton->bones[i];
                
                glm::mat4 localMat = localPose[i].ToMatrix();
                                     
                if (bone.parentIndex == -1) {
                    anim.globalTransforms[i] = localMat;
                } else {
                    anim.globalTransforms[i] = anim.globalTransforms[bone.parentIndex] * localMat;
                }
            }
            
            // 4. IK Pass (Phase 8 Step 4)
            ApplyFootIK(anim, "Left");
            ApplyFootIK(anim, "Right");
        }
    }
}

void AnimationControllerSystem::Shutdown() {
    std::cout << "[AnimationControllerSystem] Shut down" << std::endl;
}

} // namespace Gameplay
} // namespace CudaGame
