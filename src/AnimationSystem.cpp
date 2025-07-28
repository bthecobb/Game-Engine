#include "AnimationSystem.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace CudaGame {
namespace Animation {

AnimationController::AnimationController() {
    // Initialize with default state
    m_currentState.currentType = MovementAnimationType::IDLE;
    m_currentState.targetType = MovementAnimationType::IDLE;
}

void AnimationController::initialize() {
    std::cout << "🎭 Initializing Enhanced Animation System..." << std::endl;
    
    loadAnimationClips();
    
    // Initialize procedural layers
    m_proceduralLayers["breathing"] = 1.0f;
    m_proceduralLayers["idle_sway"] = 0.8f;
    m_proceduralLayers["rhythm"] = 1.0f;
    
    std::cout << "✅ Animation system initialized with " << m_animationClips.size() << " animation clips" << std::endl;
}

void AnimationController::loadAnimationClips() {
    createIdleAnimation();
    createWalkAnimations();
    createRunAnimations();
    createSprintAnimations();
    createJumpAnimations();
    createDashAnimations();
    createWallRunAnimations();
    createCombatAnimations();
}

void AnimationController::createIdleAnimation() {
    auto clip = std::make_unique<AnimationClip>();
    clip->type = MovementAnimationType::IDLE;
    clip->duration = 4.0f; // Long idle cycle for breathing
    clip->isLooping = true;
    clip->speedRange = {0.0f, 0.5f};
    clip->priority = 1;
    
    // Create keyframes for subtle idle animation
    for (int i = 0; i < 9; ++i) {
        AnimationKeyframe frame;
        frame.timeStamp = (float)i * 0.5f;
        
        // Initialize body part positions (6 parts: head, torso, left arm, right arm, left leg, right leg)
        frame.bodyPartPositions.resize(6);
        frame.bodyPartRotations.resize(6);
        frame.bodyPartScales.resize(6, glm::vec3(1.0f));
        
        // Base positions
        frame.bodyPartPositions[0] = glm::vec3(0.0f, 1.5f, 0.0f);  // Head
        frame.bodyPartPositions[1] = glm::vec3(0.0f, 0.5f, 0.0f);  // Torso
        frame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.8f, 0.0f); // Left arm
        frame.bodyPartPositions[3] = glm::vec3(0.6f, 0.8f, 0.0f);  // Right arm
        frame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.8f, 0.0f); // Left leg
        frame.bodyPartPositions[5] = glm::vec3(0.2f, -0.8f, 0.0f);  // Right leg
        
        // Subtle breathing and sway
        float breathPhase = (float)i / 8.0f * 2.0f * M_PI;
        float swayPhase = (float)i / 8.0f * M_PI;
        
        frame.bodyPartPositions[1].y += sin(breathPhase) * 0.01f; // Torso breathing
        frame.bodyPartPositions[0].y += sin(breathPhase) * 0.005f; // Head breathing
        
        // Subtle sway
        for (int j = 0; j < 6; ++j) {
            frame.bodyPartPositions[j].x += sin(swayPhase) * 0.005f;
            frame.bodyPartRotations[j] = glm::vec3(0.0f, 0.0f, sin(swayPhase) * 0.02f);
        }
        
        frame.breathingAmplitude = 0.02f;
        frame.idleSway = 0.01f;
        frame.energyLevel = 0.2f; // Low energy for idle
        
        clip->keyframes.push_back(frame);
    }
    
    m_animationClips[MovementAnimationType::IDLE] = std::move(clip);
}

void AnimationController::createWalkAnimations() {
    // Walk Forward
    auto walkForward = std::make_unique<AnimationClip>();
    walkForward->type = MovementAnimationType::WALK_FORWARD;
    walkForward->duration = 1.0f;
    walkForward->isLooping = true;
    walkForward->speedRange = {0.5f, 5.0f};
    walkForward->hasRootMotion = true;
    walkForward->priority = 2;
    
    // Create walking cycle keyframes
    for (int i = 0; i < 9; ++i) {
        AnimationKeyframe frame;
        frame.timeStamp = (float)i / 8.0f;
        float cyclePhase = frame.timeStamp * 2.0f * M_PI;
        
        frame.bodyPartPositions.resize(6);
        frame.bodyPartRotations.resize(6);
        frame.bodyPartScales.resize(6, glm::vec3(1.0f));
        
        // Base positions with walk cycle
        frame.bodyPartPositions[0] = glm::vec3(0.0f, 1.5f, 0.0f);  // Head
        frame.bodyPartPositions[1] = glm::vec3(0.0f, 0.5f, 0.0f);  // Torso
        frame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.8f, 0.0f); // Left arm
        frame.bodyPartPositions[3] = glm::vec3(0.6f, 0.8f, 0.0f);  // Right arm
        frame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.8f, 0.0f); // Left leg
        frame.bodyPartPositions[5] = glm::vec3(0.2f, -0.8f, 0.0f);  // Right leg
        
        // Walking motion
        float verticalBob = sin(cyclePhase * 2.0f) * 0.05f; // Double frequency for bob
        float legSwing = sin(cyclePhase) * 0.3f;
        float armSwing = sin(cyclePhase + M_PI) * 0.2f; // Arms opposite to legs
        
        // Apply bob to torso and head
        frame.bodyPartPositions[0].y += verticalBob * 0.5f;
        frame.bodyPartPositions[1].y += verticalBob;
        
        // Leg movement (alternating)
        frame.bodyPartPositions[4].z += legSwing;  // Left leg forward/back
        frame.bodyPartPositions[5].z -= legSwing;  // Right leg opposite
        frame.bodyPartRotations[4].x = legSwing * 0.5f;
        frame.bodyPartRotations[5].x = -legSwing * 0.5f;
        
        // Arm swing (opposite to legs)
        frame.bodyPartRotations[2].x = armSwing;  // Left arm
        frame.bodyPartRotations[3].x = -armSwing; // Right arm
        
        // Root motion
        frame.rootMotion = glm::vec3(0.0f, 0.0f, 0.8f * (1.0f / 8.0f)); // Forward movement per frame
        frame.movementSpeed = 1.0f;
        frame.energyLevel = 0.6f;
        
        walkForward->keyframes.push_back(frame);
    }
    
    m_animationClips[MovementAnimationType::WALK_FORWARD] = std::move(walkForward);
    
    // Create variations for other directions (simplified)
    createDirectionalVariation(MovementAnimationType::WALK_BACKWARD, MovementAnimationType::WALK_FORWARD, glm::vec3(0, 0, -1));
    createDirectionalVariation(MovementAnimationType::WALK_LEFT, MovementAnimationType::WALK_FORWARD, glm::vec3(-1, 0, 0));
    createDirectionalVariation(MovementAnimationType::WALK_RIGHT, MovementAnimationType::WALK_FORWARD, glm::vec3(1, 0, 0));
}

void AnimationController::createRunAnimations() {
    auto runForward = std::make_unique<AnimationClip>();
    runForward->type = MovementAnimationType::RUN_FORWARD;
    runForward->duration = 0.6f; // Faster cycle
    runForward->isLooping = true;
    runForward->speedRange = {5.0f, 15.0f};
    runForward->hasRootMotion = true;
    runForward->priority = 3;
    
    for (int i = 0; i < 7; ++i) {
        AnimationKeyframe frame;
        frame.timeStamp = (float)i / 6.0f;
        float cyclePhase = frame.timeStamp * 2.0f * M_PI;
        
        frame.bodyPartPositions.resize(6);
        frame.bodyPartRotations.resize(6);
        frame.bodyPartScales.resize(6, glm::vec3(1.0f));
        
        // Base positions
        frame.bodyPartPositions[0] = glm::vec3(0.0f, 1.5f, 0.0f);
        frame.bodyPartPositions[1] = glm::vec3(0.0f, 0.5f, 0.0f);
        frame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.8f, 0.0f);
        frame.bodyPartPositions[3] = glm::vec3(0.6f, 0.8f, 0.0f);
        frame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.8f, 0.0f);
        frame.bodyPartPositions[5] = glm::vec3(0.2f, -0.8f, 0.0f);
        
        // Running motion (more pronounced)
        float verticalBounce = sin(cyclePhase * 2.0f) * 0.1f;
        float legStride = sin(cyclePhase) * 0.5f;
        float armPump = sin(cyclePhase + M_PI) * 0.4f;
        
        // Forward lean
        frame.bodyPartPositions[0].z += 0.1f; // Head lean forward
        frame.bodyPartPositions[1].z += 0.05f; // Torso lean
        frame.bodyPartRotations[1].x = 0.1f; // Torso forward tilt
        
        // Bounce
        frame.bodyPartPositions[0].y += verticalBounce * 0.7f;
        frame.bodyPartPositions[1].y += verticalBounce;
        
        // Leg stride
        frame.bodyPartPositions[4].z += legStride;
        frame.bodyPartPositions[5].z -= legStride;
        frame.bodyPartRotations[4].x = legStride * 0.8f;
        frame.bodyPartRotations[5].x = -legStride * 0.8f;
        
        // Arm pumping
        frame.bodyPartRotations[2].x = armPump;
        frame.bodyPartRotations[3].x = -armPump;
        frame.bodyPartPositions[2].z += armPump * 0.2f;
        frame.bodyPartPositions[3].z -= armPump * 0.2f;
        
        frame.rootMotion = glm::vec3(0.0f, 0.0f, 1.8f * (1.0f / 6.0f));
        frame.movementSpeed = 2.0f;
        frame.energyLevel = 0.8f;
        
        runForward->keyframes.push_back(frame);
    }
    
    m_animationClips[MovementAnimationType::RUN_FORWARD] = std::move(runForward);
    
    // Create directional variations
    createDirectionalVariation(MovementAnimationType::RUN_BACKWARD, MovementAnimationType::RUN_FORWARD, glm::vec3(0, 0, -1));
    createDirectionalVariation(MovementAnimationType::RUN_LEFT, MovementAnimationType::RUN_FORWARD, glm::vec3(-1, 0, 0));
    createDirectionalVariation(MovementAnimationType::RUN_RIGHT, MovementAnimationType::RUN_FORWARD, glm::vec3(1, 0, 0));
}

void AnimationController::createSprintAnimations() {
    auto sprintForward = std::make_unique<AnimationClip>();
    sprintForward->type = MovementAnimationType::SPRINT_FORWARD;
    sprintForward->duration = 0.4f; // Very fast cycle
    sprintForward->isLooping = true;
    sprintForward->speedRange = {15.0f, 50.0f};
    sprintForward->hasRootMotion = true;
    sprintForward->priority = 4;
    
    for (int i = 0; i < 5; ++i) {
        AnimationKeyframe frame;
        frame.timeStamp = (float)i / 4.0f;
        float cyclePhase = frame.timeStamp * 2.0f * M_PI;
        
        frame.bodyPartPositions.resize(6);
        frame.bodyPartRotations.resize(6);
        frame.bodyPartScales.resize(6, glm::vec3(1.0f));
        
        // Base positions
        frame.bodyPartPositions[0] = glm::vec3(0.0f, 1.5f, 0.0f);
        frame.bodyPartPositions[1] = glm::vec3(0.0f, 0.5f, 0.0f);
        frame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.8f, 0.0f);
        frame.bodyPartPositions[3] = glm::vec3(0.6f, 0.8f, 0.0f);
        frame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.8f, 0.0f);
        frame.bodyPartPositions[5] = glm::vec3(0.2f, -0.8f, 0.0f);
        
        // Sprinting motion (extreme)
        float verticalBound = sin(cyclePhase * 2.0f) * 0.15f;
        float legDrive = sin(cyclePhase) * 0.7f;
        float armDrive = sin(cyclePhase + M_PI) * 0.6f;
        
        // Aggressive forward lean
        frame.bodyPartPositions[0].z += 0.2f; // Head far forward
        frame.bodyPartPositions[1].z += 0.15f; // Torso forward
        frame.bodyPartRotations[1].x = 0.2f; // Strong forward tilt
        
        // Powerful bounding
        frame.bodyPartPositions[0].y += verticalBound * 0.8f;
        frame.bodyPartPositions[1].y += verticalBound;
        
        // Powerful leg drive
        frame.bodyPartPositions[4].z += legDrive;
        frame.bodyPartPositions[5].z -= legDrive;
        frame.bodyPartRotations[4].x = legDrive * 1.2f;
        frame.bodyPartRotations[5].x = -legDrive * 1.2f;
        
        // Intense arm drive
        frame.bodyPartRotations[2].x = armDrive;
        frame.bodyPartRotations[3].x = -armDrive;
        frame.bodyPartPositions[2].z += armDrive * 0.3f;
        frame.bodyPartPositions[3].z -= armDrive * 0.3f;
        
        frame.rootMotion = glm::vec3(0.0f, 0.0f, 3.0f * (1.0f / 4.0f));
        frame.movementSpeed = 4.0f;
        frame.energyLevel = 1.0f; // Maximum energy
        
        sprintForward->keyframes.push_back(frame);
    }
    
    m_animationClips[MovementAnimationType::SPRINT_FORWARD] = std::move(sprintForward);
    
    // Create directional variations
    createDirectionalVariation(MovementAnimationType::SPRINT_BACKWARD, MovementAnimationType::SPRINT_FORWARD, glm::vec3(0, 0, -1));
    createDirectionalVariation(MovementAnimationType::SPRINT_LEFT, MovementAnimationType::SPRINT_FORWARD, glm::vec3(-1, 0, 0));
    createDirectionalVariation(MovementAnimationType::SPRINT_RIGHT, MovementAnimationType::SPRINT_FORWARD, glm::vec3(1, 0, 0));
}

void AnimationController::createJumpAnimations() {
    // Jump Start
    auto jumpStart = std::make_unique<AnimationClip>();
    jumpStart->type = MovementAnimationType::JUMP_START;
    jumpStart->duration = 0.2f;
    jumpStart->isLooping = false;
    jumpStart->priority = 5;
    
    AnimationKeyframe prepFrame, launchFrame;
    
    // Prep frame (crouched)
    prepFrame.timeStamp = 0.0f;
    prepFrame.bodyPartPositions.resize(6);
    prepFrame.bodyPartRotations.resize(6);
    prepFrame.bodyPartScales.resize(6, glm::vec3(1.0f));
    
    prepFrame.bodyPartPositions[0] = glm::vec3(0.0f, 1.3f, 0.0f);  // Head lower
    prepFrame.bodyPartPositions[1] = glm::vec3(0.0f, 0.3f, 0.0f);  // Torso crouched
    prepFrame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.6f, 0.0f); // Arms ready
    prepFrame.bodyPartPositions[3] = glm::vec3(0.6f, 0.6f, 0.0f);
    prepFrame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.6f, 0.0f); // Legs bent
    prepFrame.bodyPartPositions[5] = glm::vec3(0.2f, -0.6f, 0.0f);
    
    // Launch frame (extended)
    launchFrame.timeStamp = 1.0f;
    launchFrame.bodyPartPositions.resize(6);
    launchFrame.bodyPartRotations.resize(6);
    launchFrame.bodyPartScales.resize(6, glm::vec3(1.0f));
    
    launchFrame.bodyPartPositions[0] = glm::vec3(0.0f, 1.7f, 0.0f);  // Head up
    launchFrame.bodyPartPositions[1] = glm::vec3(0.0f, 0.8f, 0.0f);  // Torso extended
    launchFrame.bodyPartPositions[2] = glm::vec3(-0.6f, 1.2f, 0.0f); // Arms up
    launchFrame.bodyPartPositions[3] = glm::vec3(0.6f, 1.2f, 0.0f);
    launchFrame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.4f, 0.0f); // Legs extended
    launchFrame.bodyPartPositions[5] = glm::vec3(0.2f, -0.4f, 0.0f);
    
    launchFrame.energyLevel = 1.0f;
    
    jumpStart->keyframes = {prepFrame, launchFrame};
    m_animationClips[MovementAnimationType::JUMP_START] = std::move(jumpStart);
    
    // Create other jump phases (simplified)
    createJumpPhase(MovementAnimationType::JUMP_APEX, 0.5f);
    createJumpPhase(MovementAnimationType::JUMP_FALL, 0.3f);
    createJumpPhase(MovementAnimationType::JUMP_LAND, 0.2f);
}

void AnimationController::createDashAnimations() {
    auto dashHorizontal = std::make_unique<AnimationClip>();
    dashHorizontal->type = MovementAnimationType::DASH_HORIZONTAL;
    dashHorizontal->duration = 0.3f;
    dashHorizontal->isLooping = false;
    dashHorizontal->priority = 6;
    
    // Create dash keyframes with extreme lean
    for (int i = 0; i < 4; ++i) {
        AnimationKeyframe frame;
        frame.timeStamp = (float)i / 3.0f;
        
        frame.bodyPartPositions.resize(6);
        frame.bodyPartRotations.resize(6);
        frame.bodyPartScales.resize(6, glm::vec3(1.0f));
        
        float dashLean = 0.5f * sin(frame.timeStamp * M_PI); // Peak lean in middle
        
        // Extreme forward lean for dash
        frame.bodyPartPositions[0] = glm::vec3(0.0f, 1.4f, dashLean * 0.3f);
        frame.bodyPartPositions[1] = glm::vec3(0.0f, 0.4f, dashLean * 0.2f);
        frame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.7f, dashLean * 0.1f);
        frame.bodyPartPositions[3] = glm::vec3(0.6f, 0.7f, dashLean * 0.1f);
        frame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.8f, 0.0f);
        frame.bodyPartPositions[5] = glm::vec3(0.2f, -0.8f, 0.0f);
        
        // Rotation for lean
        frame.bodyPartRotations[1].x = dashLean * 0.8f;
        frame.bodyPartRotations[0].x = dashLean * 0.4f;
        
        frame.rootMotion = glm::vec3(0.0f, 0.0f, 2.0f * (1.0f / 3.0f));
        frame.energyLevel = 1.0f;
        
        dashHorizontal->keyframes.push_back(frame);
    }
    
    m_animationClips[MovementAnimationType::DASH_HORIZONTAL] = std::move(dashHorizontal);
}

void AnimationController::createWallRunAnimations() {
    // Wall run left
    auto wallRunLeft = std::make_unique<AnimationClip>();
    wallRunLeft->type = MovementAnimationType::WALL_RUN_LEFT;
    wallRunLeft->duration = 0.8f;
    wallRunLeft->isLooping = true;
    wallRunLeft->priority = 5;
    
    for (int i = 0; i < 5; ++i) {
        AnimationKeyframe frame;
        frame.timeStamp = (float)i / 4.0f;
        
        frame.bodyPartPositions.resize(6);
        frame.bodyPartRotations.resize(6);
        frame.bodyPartScales.resize(6, glm::vec3(1.0f));
        
        // Leaning left against wall
        frame.bodyPartPositions[0] = glm::vec3(-0.3f, 1.5f, 0.0f);  // Head tilted left
        frame.bodyPartPositions[1] = glm::vec3(-0.2f, 0.5f, 0.0f);  // Torso leaning
        frame.bodyPartPositions[2] = glm::vec3(-0.8f, 0.9f, 0.0f);  // Left arm against wall
        frame.bodyPartPositions[3] = glm::vec3(0.4f, 0.7f, 0.0f);   // Right arm balanced
        frame.bodyPartPositions[4] = glm::vec3(-0.3f, -0.8f, 0.0f); // Left leg
        frame.bodyPartPositions[5] = glm::vec3(0.1f, -0.8f, 0.0f);  // Right leg
        
        // Wall contact rotation
        frame.bodyPartRotations[1].z = -0.3f; // Torso tilted toward wall
        frame.bodyPartRotations[0].z = -0.2f; // Head tilt
        
        frame.energyLevel = 0.9f;
        
        wallRunLeft->keyframes.push_back(frame);
    }
    
    m_animationClips[MovementAnimationType::WALL_RUN_LEFT] = std::move(wallRunLeft);
    
    // Create wall run right as mirror
    createMirroredAnimation(MovementAnimationType::WALL_RUN_RIGHT, MovementAnimationType::WALL_RUN_LEFT);
}

void AnimationController::createCombatAnimations() {
    // Light Attack 1
    auto lightAttack1 = std::make_unique<AnimationClip>();
    lightAttack1->type = MovementAnimationType::ATTACK_LIGHT_1;
    lightAttack1->duration = 0.4f;
    lightAttack1->isLooping = false;
    lightAttack1->priority = 7;
    
    AnimationKeyframe windupFrame, strikeFrame, recoveryFrame;
    
    // Windup
    windupFrame.timeStamp = 0.0f;
    windupFrame.bodyPartPositions.resize(6);
    windupFrame.bodyPartRotations.resize(6);
    windupFrame.bodyPartScales.resize(6, glm::vec3(1.0f));
    
    windupFrame.bodyPartPositions[0] = glm::vec3(0.0f, 1.5f, 0.0f);
    windupFrame.bodyPartPositions[1] = glm::vec3(0.0f, 0.5f, 0.0f);
    windupFrame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.8f, 0.0f);
    windupFrame.bodyPartPositions[3] = glm::vec3(0.8f, 1.0f, -0.3f); // Right arm back
    windupFrame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.8f, 0.0f);
    windupFrame.bodyPartPositions[5] = glm::vec3(0.2f, -0.8f, 0.0f);
    
    windupFrame.bodyPartRotations[3].x = -0.5f; // Right arm cocked back
    
    // Strike
    strikeFrame.timeStamp = 0.3f;
    strikeFrame.bodyPartPositions.resize(6);
    strikeFrame.bodyPartRotations.resize(6);
    strikeFrame.bodyPartScales.resize(6, glm::vec3(1.0f));
    
    strikeFrame.bodyPartPositions[0] = glm::vec3(0.0f, 1.5f, 0.0f);
    strikeFrame.bodyPartPositions[1] = glm::vec3(0.0f, 0.5f, 0.0f);
    strikeFrame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.8f, 0.0f);
    strikeFrame.bodyPartPositions[3] = glm::vec3(0.6f, 0.8f, 0.5f); // Right arm extended
    strikeFrame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.8f, 0.0f);
    strikeFrame.bodyPartPositions[5] = glm::vec3(0.2f, -0.8f, 0.0f);
    
    strikeFrame.bodyPartRotations[3].x = 0.3f; // Right arm forward
    strikeFrame.energyLevel = 1.0f;
    
    // Recovery
    recoveryFrame.timeStamp = 1.0f;
    recoveryFrame.bodyPartPositions.resize(6);
    recoveryFrame.bodyPartRotations.resize(6);
    recoveryFrame.bodyPartScales.resize(6, glm::vec3(1.0f));
    
    recoveryFrame.bodyPartPositions[0] = glm::vec3(0.0f, 1.5f, 0.0f);
    recoveryFrame.bodyPartPositions[1] = glm::vec3(0.0f, 0.5f, 0.0f);
    recoveryFrame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.8f, 0.0f);
    recoveryFrame.bodyPartPositions[3] = glm::vec3(0.6f, 0.8f, 0.0f); // Back to neutral
    recoveryFrame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.8f, 0.0f);
    recoveryFrame.bodyPartPositions[5] = glm::vec3(0.2f, -0.8f, 0.0f);
    
    lightAttack1->keyframes = {windupFrame, strikeFrame, recoveryFrame};
    m_animationClips[MovementAnimationType::ATTACK_LIGHT_1] = std::move(lightAttack1);
    
    // Create other attack variations (simplified)
    createAttackVariation(MovementAnimationType::ATTACK_LIGHT_2, MovementAnimationType::ATTACK_LIGHT_1, 0.5f);
    createAttackVariation(MovementAnimationType::ATTACK_LIGHT_3, MovementAnimationType::ATTACK_LIGHT_1, 0.7f);
    createAttackVariation(MovementAnimationType::ATTACK_HEAVY, MovementAnimationType::ATTACK_LIGHT_1, 1.5f);
}

// Helper methods for creating animation variations
void AnimationController::createDirectionalVariation(MovementAnimationType newType, MovementAnimationType baseType, const glm::vec3& direction) {
    if (m_animationClips.find(baseType) == m_animationClips.end()) return;
    
    auto newClip = std::make_unique<AnimationClip>(*m_animationClips[baseType]);
    newClip->type = newType;
    
    // Modify keyframes for direction
    for (auto& frame : newClip->keyframes) {
        frame.rootMotion = direction * glm::length(frame.rootMotion);
        
        // Adjust body orientation for movement direction
        if (direction.x != 0) {
            // Side movement - add lean
            for (auto& pos : frame.bodyPartPositions) {
                pos.x += direction.x * 0.1f;
            }
            for (auto& rot : frame.bodyPartRotations) {
                rot.z += direction.x * 0.2f;
            }
        }
        if (direction.z < 0) {
            // Backward movement - different arm swing
            std::swap(frame.bodyPartRotations[2].x, frame.bodyPartRotations[3].x);
        }
    }
    
    m_animationClips[newType] = std::move(newClip);
}

void AnimationController::createJumpPhase(MovementAnimationType type, float duration) {
    auto clip = std::make_unique<AnimationClip>();
    clip->type = type;
    clip->duration = duration;
    clip->isLooping = false;
    clip->priority = 5;
    
    AnimationKeyframe frame;
    frame.timeStamp = 0.0f;
    frame.bodyPartPositions.resize(6);
    frame.bodyPartRotations.resize(6);
    frame.bodyPartScales.resize(6, glm::vec3(1.0f));
    
    // Different poses for different jump phases
    switch (type) {
        case MovementAnimationType::JUMP_APEX:
            // Arms spread, legs tucked
            frame.bodyPartPositions[0] = glm::vec3(0.0f, 1.6f, 0.0f);
            frame.bodyPartPositions[1] = glm::vec3(0.0f, 0.6f, 0.0f);
            frame.bodyPartPositions[2] = glm::vec3(-0.8f, 1.0f, 0.0f);
            frame.bodyPartPositions[3] = glm::vec3(0.8f, 1.0f, 0.0f);
            frame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.4f, 0.0f);
            frame.bodyPartPositions[5] = glm::vec3(0.2f, -0.4f, 0.0f);
            break;
            
        case MovementAnimationType::JUMP_FALL:
            // Preparing for landing
            frame.bodyPartPositions[0] = glm::vec3(0.0f, 1.4f, 0.0f);
            frame.bodyPartPositions[1] = glm::vec3(0.0f, 0.4f, 0.0f);
            frame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.7f, 0.0f);
            frame.bodyPartPositions[3] = glm::vec3(0.6f, 0.7f, 0.0f);
            frame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.6f, 0.0f);
            frame.bodyPartPositions[5] = glm::vec3(0.2f, -0.6f, 0.0f);
            break;
            
        case MovementAnimationType::JUMP_LAND:
            // Crouched landing
            frame.bodyPartPositions[0] = glm::vec3(0.0f, 1.2f, 0.0f);
            frame.bodyPartPositions[1] = glm::vec3(0.0f, 0.2f, 0.0f);
            frame.bodyPartPositions[2] = glm::vec3(-0.6f, 0.5f, 0.0f);
            frame.bodyPartPositions[3] = glm::vec3(0.6f, 0.5f, 0.0f);
            frame.bodyPartPositions[4] = glm::vec3(-0.2f, -0.5f, 0.0f);
            frame.bodyPartPositions[5] = glm::vec3(0.2f, -0.5f, 0.0f);
            break;
        default:
            break;
    }
    
    clip->keyframes.push_back(frame);
    m_animationClips[type] = std::move(clip);
}

void AnimationController::createMirroredAnimation(MovementAnimationType newType, MovementAnimationType baseType) {
    if (m_animationClips.find(baseType) == m_animationClips.end()) return;
    
    auto newClip = std::make_unique<AnimationClip>(*m_animationClips[baseType]);
    newClip->type = newType;
    
    // Mirror keyframes across X axis
    for (auto& frame : newClip->keyframes) {
        for (auto& pos : frame.bodyPartPositions) {
            pos.x = -pos.x;
        }
        for (auto& rot : frame.bodyPartRotations) {
            rot.z = -rot.z;
        }
    }
    
    m_animationClips[newType] = std::move(newClip);
}

void AnimationController::createAttackVariation(MovementAnimationType newType, MovementAnimationType baseType, float intensity) {
    if (m_animationClips.find(baseType) == m_animationClips.end()) return;
    
    auto newClip = std::make_unique<AnimationClip>(*m_animationClips[baseType]);
    newClip->type = newType;
    newClip->duration *= intensity; // Heavier attacks take longer
    
    // Modify keyframes for intensity
    for (auto& frame : newClip->keyframes) {
        frame.energyLevel = std::min(1.0f, frame.energyLevel * intensity);
        
        // Scale movement ranges
        for (int i = 0; i < frame.bodyPartPositions.size(); ++i) {
            glm::vec3 offset = frame.bodyPartPositions[i] - glm::vec3(0.0f, 0.5f, 0.0f);
            frame.bodyPartPositions[i] = glm::vec3(0.0f, 0.5f, 0.0f) + offset * intensity;
        }
    }
    
    m_animationClips[newType] = std::move(newClip);
}

void AnimationController::update(float deltaTime) {
    m_frameNeedsUpdate = true;
    
    // Update current animation time
    if (m_currentState.isTransitioning) {
        updateTransition(deltaTime);
    } else {
        m_currentState.currentTime += deltaTime;
        
        // Handle looping
        auto currentClip = m_animationClips.find(m_currentState.currentType);
        if (currentClip != m_animationClips.end()) {
            if (currentClip->second->isLooping) {
                if (m_currentState.currentTime >= currentClip->second->duration) {
                    m_currentState.currentTime = fmod(m_currentState.currentTime, currentClip->second->duration);
                }
            } else {
                m_currentState.currentTime = std::min(m_currentState.currentTime, currentClip->second->duration);
            }
        }
    }
    
    // Update rhythm parameters
    m_currentState.rhythmPhase = m_rhythmPhase;
    m_currentState.isOnBeat = m_isOnBeat;
    m_currentState.beatIntensity = m_beatIntensity;
}

void AnimationController::updateMovementAnimation(const glm::vec2& inputDirection, float speed) {
    MovementAnimationType newType = selectMovementAnimation(inputDirection, speed);
    
    if (newType != m_currentState.currentType && !m_currentState.isTransitioning) {
        setMovementState(newType);
    }
    
    m_currentState.movementDirection = inputDirection;
    m_currentState.movementSpeed = speed;
}

MovementAnimationType AnimationController::selectMovementAnimation(const glm::vec2& direction, float speed) const {
    // Speed-based selection first
    if (speed < 0.5f) {
        return MovementAnimationType::IDLE;
    } else if (speed < 5.0f) {
        // Walk animations
        if (abs(direction.x) > abs(direction.y)) {
            return direction.x > 0 ? MovementAnimationType::WALK_RIGHT : MovementAnimationType::WALK_LEFT;
        } else {
            return direction.y > 0 ? MovementAnimationType::WALK_FORWARD : MovementAnimationType::WALK_BACKWARD;
        }
    } else if (speed < 15.0f) {
        // Run animations
        if (abs(direction.x) > abs(direction.y)) {
            return direction.x > 0 ? MovementAnimationType::RUN_RIGHT : MovementAnimationType::RUN_LEFT;
        } else {
            return direction.y > 0 ? MovementAnimationType::RUN_FORWARD : MovementAnimationType::RUN_BACKWARD;
        }
    } else {
        // Sprint animations
        if (abs(direction.x) > abs(direction.y)) {
            return direction.x > 0 ? MovementAnimationType::SPRINT_RIGHT : MovementAnimationType::SPRINT_LEFT;
        } else {
            return direction.y > 0 ? MovementAnimationType::SPRINT_FORWARD : MovementAnimationType::SPRINT_BACKWARD;
        }
    }
}

void AnimationController::setMovementState(MovementAnimationType type, float blendTime) {
    if (type == m_currentState.currentType) return;
    
    m_currentState.targetType = type;
    m_currentState.isTransitioning = true;
    m_currentState.transitionProgress = 0.0f;
    m_frameNeedsUpdate = true;
}

void AnimationController::setRhythmParameters(float phase, bool onBeat, float intensity) {
    m_rhythmPhase = phase;
    m_isOnBeat = onBeat;
    m_beatIntensity = intensity;
}

AnimationKeyframe AnimationController::getCurrentFrame() const {
    if (!m_frameNeedsUpdate) {
        return m_currentFrame;
    }
    
    auto currentClip = m_animationClips.find(m_currentState.currentType);
    if (currentClip == m_animationClips.end() || currentClip->second->keyframes.empty()) {
        // Return default frame
        AnimationKeyframe defaultFrame;
        defaultFrame.bodyPartPositions.resize(6, glm::vec3(0.0f));
        defaultFrame.bodyPartRotations.resize(6, glm::vec3(0.0f));
        defaultFrame.bodyPartScales.resize(6, glm::vec3(1.0f));
        return defaultFrame;
    }
    
    const auto& keyframes = currentClip->second->keyframes;
    float normalizedTime = m_currentState.currentTime / currentClip->second->duration;
    
    // Find keyframes to interpolate between
    int currentIndex = 0;
    int nextIndex = 1;
    
    for (int i = 0; i < keyframes.size() - 1; ++i) {
        if (normalizedTime >= keyframes[i].timeStamp && normalizedTime <= keyframes[i + 1].timeStamp) {
            currentIndex = i;
            nextIndex = i + 1;
            break;
        }
    }
    
    if (nextIndex >= keyframes.size()) {
        nextIndex = currentClip->second->isLooping ? 0 : keyframes.size() - 1;
    }
    
    // Interpolate between keyframes
    float localTime = (normalizedTime - keyframes[currentIndex].timeStamp) / 
                     (keyframes[nextIndex].timeStamp - keyframes[currentIndex].timeStamp);
    localTime = std::clamp(localTime, 0.0f, 1.0f);
    
    AnimationKeyframe interpolatedFrame = interpolateKeyframes(keyframes[currentIndex], keyframes[nextIndex], localTime);
    
    // Apply procedural modifications
    applyProceduralAnimation(interpolatedFrame, m_currentState.currentTime);
    applyRhythmModulation(interpolatedFrame);
    
    m_currentFrame = interpolatedFrame;
    m_frameNeedsUpdate = false;
    
    return m_currentFrame;
}

AnimationKeyframe AnimationController::interpolateKeyframes(const AnimationKeyframe& a, const AnimationKeyframe& b, float t) const {
    AnimationKeyframe result;
    
    result.timeStamp = glm::mix(a.timeStamp, b.timeStamp, t);
    result.movementSpeed = glm::mix(a.movementSpeed, b.movementSpeed, t);
    result.rhythmIntensity = glm::mix(a.rhythmIntensity, b.rhythmIntensity, t);
    result.breathingAmplitude = glm::mix(a.breathingAmplitude, b.breathingAmplitude, t);
    result.idleSway = glm::mix(a.idleSway, b.idleSway, t);
    result.energyLevel = glm::mix(a.energyLevel, b.energyLevel, t);
    result.rootMotion = glm::mix(a.rootMotion, b.rootMotion, t);
    
    // Interpolate body part transforms
    size_t partCount = std::min(a.bodyPartPositions.size(), b.bodyPartPositions.size());
    result.bodyPartPositions.resize(partCount);
    result.bodyPartRotations.resize(partCount);
    result.bodyPartScales.resize(partCount);
    
    for (size_t i = 0; i < partCount; ++i) {
        result.bodyPartPositions[i] = glm::mix(a.bodyPartPositions[i], b.bodyPartPositions[i], t);
        result.bodyPartRotations[i] = glm::mix(a.bodyPartRotations[i], b.bodyPartRotations[i], t);
        result.bodyPartScales[i] = glm::mix(a.bodyPartScales[i], b.bodyPartScales[i], t);
    }
    
    return result;
}

void AnimationController::applyProceduralAnimation(AnimationKeyframe& frame, float deltaTime) const {
    // Apply breathing
    if (m_proceduralLayers.find("breathing") != m_proceduralLayers.end()) {
        ProceduralAnimationGenerator::addBreathingAnimation(frame, deltaTime, 
            frame.breathingAmplitude * m_proceduralLayers.at("breathing"));
    }
    
    // Apply idle sway
    if (m_proceduralLayers.find("idle_sway") != m_proceduralLayers.end()) {
        ProceduralAnimationGenerator::addIdleSway(frame, deltaTime, 
            frame.idleSway * m_proceduralLayers.at("idle_sway"));
    }
}

void AnimationController::applyRhythmModulation(AnimationKeyframe& frame) const {
    if (m_proceduralLayers.find("rhythm") != m_proceduralLayers.end()) {
        ProceduralAnimationGenerator::addRhythmPulse(frame, m_rhythmPhase, 
            m_beatIntensity * m_proceduralLayers.at("rhythm"));
    }
}

// Procedural Animation Generator implementations
void ProceduralAnimationGenerator::addBreathingAnimation(AnimationKeyframe& frame, float time, float amplitude) {
    float breathCycle = sin(time * 3.0f) * amplitude;
    
    if (frame.bodyPartPositions.size() > 1) {
        frame.bodyPartPositions[1].y += breathCycle; // Torso breathing
    }
    if (frame.bodyPartPositions.size() > 0) {
        frame.bodyPartPositions[0].y += breathCycle * 0.5f; // Head follows
    }
}

void ProceduralAnimationGenerator::addIdleSway(AnimationKeyframe& frame, float time, float amplitude) {
    float swayCycle = sin(time * 1.5f) * amplitude;
    
    for (auto& pos : frame.bodyPartPositions) {
        pos.x += swayCycle;
    }
    for (auto& rot : frame.bodyPartRotations) {
        rot.z += swayCycle * 2.0f;
    }
}

void ProceduralAnimationGenerator::addRhythmPulse(AnimationKeyframe& frame, float phase, float intensity) {
    float pulse = sin(phase * 2.0f * M_PI) * intensity * 0.05f;
    
    for (auto& scale : frame.bodyPartScales) {
        scale *= (1.0f + pulse);
    }
}

void ProceduralAnimationGenerator::addWalkBob(AnimationKeyframe& frame, float cycleTime, float speed) {
    float bob = sin(cycleTime * 4.0f * M_PI) * 0.03f * speed;
    
    for (auto& pos : frame.bodyPartPositions) {
        pos.y += bob;
    }
}

void ProceduralAnimationGenerator::addRunBounce(AnimationKeyframe& frame, float cycleTime, float speed) {
    float bounce = sin(cycleTime * 6.0f * M_PI) * 0.08f * speed;
    
    for (auto& pos : frame.bodyPartPositions) {
        pos.y += bounce;
    }
}

void ProceduralAnimationGenerator::addSprintLean(AnimationKeyframe& frame, const glm::vec2& direction, float speed) {
    float leanAmount = speed * 0.02f;
    glm::vec3 lean(direction.x * leanAmount, 0, direction.y * leanAmount);
    
    if (frame.bodyPartPositions.size() > 1) {
        frame.bodyPartPositions[1] += lean; // Torso lean
        frame.bodyPartRotations[1].x += lean.z;
        frame.bodyPartRotations[1].z += lean.x;
    }
}

void ProceduralAnimationGenerator::addWindEffect(AnimationKeyframe& frame, const glm::vec3& windDirection, float strength) {
    glm::vec3 windOffset = windDirection * strength * 0.01f;
    
    for (auto& pos : frame.bodyPartPositions) {
        pos += windOffset;
    }
}

void ProceduralAnimationGenerator::addMomentumLean(AnimationKeyframe& frame, const glm::vec3& velocity) {
    float speed = glm::length(velocity);
    if (speed > 0.1f) {
        glm::vec3 direction = glm::normalize(velocity);
        float leanAmount = std::min(speed * 0.005f, 0.1f);
        
        if (frame.bodyPartPositions.size() > 1) {
            frame.bodyPartPositions[1] += direction * leanAmount;
            frame.bodyPartRotations[1].x += direction.z * leanAmount * 10.0f;
        }
    }
}

void ProceduralAnimationGenerator::addTurnAnticipation(AnimationKeyframe& frame, float turnSpeed) {
    float anticipation = turnSpeed * 0.1f;
    
    if (frame.bodyPartPositions.size() > 1) {
        frame.bodyPartRotations[1].y += anticipation; // Torso pre-turn
    }
}

} // namespace Animation
} // namespace CudaGame
