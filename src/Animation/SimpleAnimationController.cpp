#include "SimpleAnimationController.h"
#include <algorithm>

SimpleAnimationController::SimpleAnimationController() 
    : currentState("idle"), animationTime(0.0f), energyLevel(0.2f),
      walkSpeed(2.0f), runSpeed(4.0f), sprintSpeed(6.0f), breathingRate(1.0f) {
    
    // Initialize body parts to default positions
    currentBodyParts.head = {0.0f, 1.5f, 0.0f};
    currentBodyParts.torso = {0.0f, 0.5f, 0.0f};
    currentBodyParts.leftArm = {-0.6f, 0.8f, 0.0f};
    currentBodyParts.rightArm = {0.6f, 0.8f, 0.0f};
    currentBodyParts.leftLeg = {-0.3f, -0.5f, 0.0f};
    currentBodyParts.rightLeg = {0.3f, -0.5f, 0.0f};
}

SimpleAnimationController::~SimpleAnimationController() {
}

void SimpleAnimationController::setState(const std::string& state) {
    if (currentState != state) {
        currentState = state;
        
        // Update energy level based on state
        if (state == "idle") {
            energyLevel = 0.2f;
        } else if (state == "walk") {
            energyLevel = 0.6f;
        } else if (state == "run") {
            energyLevel = 0.8f;
        } else if (state == "sprint") {
            energyLevel = 1.0f;
        } else if (state == "jump") {
            energyLevel = 0.9f;
        }
    }
}

void SimpleAnimationController::update(float deltaTime) {
    animationTime += deltaTime;
    updateBodyParts();
}

void SimpleAnimationController::updateBodyParts() {
    // Reset to base positions
    currentBodyParts.head = {0.0f, 1.5f, 0.0f};
    currentBodyParts.torso = {0.0f, 0.5f, 0.0f};
    currentBodyParts.leftArm = {-0.6f, 0.8f, 0.0f};
    currentBodyParts.rightArm = {0.6f, 0.8f, 0.0f};
    currentBodyParts.leftLeg = {-0.3f, -0.5f, 0.0f};
    currentBodyParts.rightLeg = {0.3f, -0.5f, 0.0f};
    
    // Apply animation based on current state
    if (currentState == "idle") {
        applyBreathing();
    } else if (currentState == "walk") {
        applyWalkCycle();
        applyBreathing();
    } else if (currentState == "run") {
        applyRunCycle();
        applyBreathing();
    } else if (currentState == "sprint") {
        applySprintCycle();
        applyBreathing();
    } else if (currentState == "jump") {
        applyJumpCycle();
    } else if (currentState == "doublejump") {
        applyDoubleJumpCycle();
    } else if (currentState == "punch") {
        applyPunchAnimation();
    } else if (currentState == "kick") {
        applyKickAnimation();
    } else if (currentState == "combo1") {
        applyCombo1Animation();
    } else if (currentState == "combo2") {
        applyCombo2Animation();
    } else if (currentState == "combo3") {
        applyCombo3Animation();
    }
}

void SimpleAnimationController::applyBreathing() {
    float breathe = sin(animationTime * breathingRate * 2.0f) * 0.02f;
    currentBodyParts.head.y += breathe;
    currentBodyParts.torso.y += breathe * 0.5f;
}

void SimpleAnimationController::applyWalkCycle() {
    float walkCycle = animationTime * walkSpeed;
    float legSwing = sin(walkCycle) * 0.3f;
    float armSwing = sin(walkCycle) * 0.2f;
    float verticalBob = abs(sin(walkCycle * 2.0f)) * 0.05f;
    
    // Leg movement (alternating)
    currentBodyParts.leftLeg.z = legSwing;
    currentBodyParts.rightLeg.z = -legSwing;
    
    // Arm movement (opposite to legs)
    currentBodyParts.leftArm.z = -armSwing * 0.5f;
    currentBodyParts.rightArm.z = armSwing * 0.5f;
    
    // Vertical bob
    currentBodyParts.head.y += verticalBob;
    currentBodyParts.torso.y += verticalBob * 0.8f;
}

void SimpleAnimationController::applyRunCycle() {
    float runCycle = animationTime * runSpeed;
    float legSwing = sin(runCycle) * 0.5f;
    float armSwing = sin(runCycle) * 0.4f;
    float verticalBob = abs(sin(runCycle * 2.0f)) * 0.08f;
    float forwardLean = 0.1f;
    
    // More pronounced leg movement
    currentBodyParts.leftLeg.z = legSwing;
    currentBodyParts.rightLeg.z = -legSwing;
    
    // More pronounced arm movement
    currentBodyParts.leftArm.z = -armSwing * 0.8f;
    currentBodyParts.rightArm.z = armSwing * 0.8f;
    
    // Vertical bob and forward lean
    currentBodyParts.head.y += verticalBob;
    currentBodyParts.head.z += forwardLean;
    currentBodyParts.torso.y += verticalBob * 0.8f;
    currentBodyParts.torso.z += forwardLean * 0.5f;
}

void SimpleAnimationController::applySprintCycle() {
    float sprintCycle = animationTime * sprintSpeed;
    float legSwing = sin(sprintCycle) * 0.7f;
    float armSwing = sin(sprintCycle) * 0.6f;
    float verticalBob = abs(sin(sprintCycle * 2.0f)) * 0.12f;
    float forwardLean = 0.2f;
    
    // Maximum leg movement
    currentBodyParts.leftLeg.z = legSwing;
    currentBodyParts.rightLeg.z = -legSwing;
    
    // Maximum arm movement
    currentBodyParts.leftArm.z = -armSwing;
    currentBodyParts.rightArm.z = armSwing;
    
    // Maximum vertical bob and forward lean
    currentBodyParts.head.y += verticalBob;
    currentBodyParts.head.z += forwardLean;
    currentBodyParts.torso.y += verticalBob * 0.8f;
    currentBodyParts.torso.z += forwardLean * 0.75f;
}

void SimpleAnimationController::applyJumpCycle() {
    // Simple jump animation - could be enhanced with jump phases
    float jumpHeight = sin(animationTime * 3.0f) * 0.5f;
    if (jumpHeight < 0) jumpHeight = 0;
    
    currentBodyParts.head.y += jumpHeight;
    currentBodyParts.torso.y += jumpHeight * 0.8f;
    currentBodyParts.leftLeg.y += jumpHeight * 0.6f;
    currentBodyParts.rightLeg.y += jumpHeight * 0.6f;
    
    // Arms raised during jump
    currentBodyParts.leftArm.y += jumpHeight * 0.3f;
    currentBodyParts.rightArm.y += jumpHeight * 0.3f;
}

void SimpleAnimationController::applyDoubleJumpCycle() {
    // Fast spinning double jump animation
    float spinCycle = animationTime * 8.0f;
    float jumpHeight = sin(animationTime * 4.0f) * 0.7f;
    if (jumpHeight < 0) jumpHeight = 0;
    
    // Spinning effect
    float spin = sin(spinCycle) * 0.4f;
    currentBodyParts.leftArm.x += spin;
    currentBodyParts.rightArm.x -= spin;
    currentBodyParts.leftArm.z += cos(spinCycle) * 0.4f;
    currentBodyParts.rightArm.z -= cos(spinCycle) * 0.4f;
    
    // Higher jump
    currentBodyParts.head.y += jumpHeight;
    currentBodyParts.torso.y += jumpHeight * 0.9f;
    currentBodyParts.leftLeg.y += jumpHeight * 0.7f;
    currentBodyParts.rightLeg.y += jumpHeight * 0.7f;
}

void SimpleAnimationController::applyPunchAnimation() {
    float punchCycle = animationTime * 10.0f;
    float punchExtend = sin(punchCycle) * 0.5f;
    if (punchExtend < 0) punchExtend = 0;
    
    // Right arm punches forward
    currentBodyParts.rightArm.z += punchExtend;
    currentBodyParts.rightArm.x += punchExtend * 0.2f;
    
    // Body lean into punch
    currentBodyParts.torso.z += punchExtend * 0.1f;
    currentBodyParts.head.z += punchExtend * 0.05f;
    
    // Left arm back for balance
    currentBodyParts.leftArm.z -= punchExtend * 0.3f;
}

void SimpleAnimationController::applyKickAnimation() {
    float kickCycle = animationTime * 8.0f;
    float kickExtend = sin(kickCycle) * 0.8f;
    if (kickExtend < 0) kickExtend = 0;
    
    // Right leg kicks forward
    currentBodyParts.rightLeg.z += kickExtend;
    currentBodyParts.rightLeg.y += kickExtend * 0.3f;
    
    // Body lean back for balance
    currentBodyParts.torso.z -= kickExtend * 0.1f;
    currentBodyParts.head.z -= kickExtend * 0.05f;
    
    // Arms out for balance
    currentBodyParts.leftArm.x -= kickExtend * 0.2f;
    currentBodyParts.rightArm.x += kickExtend * 0.2f;
}

void SimpleAnimationController::applyCombo1Animation() {
    float comboCycle = animationTime * 12.0f;
    float comboIntensity = sin(comboCycle) * 0.6f;
    if (comboIntensity < 0) comboIntensity = 0;
    
    // Alternating arm strikes
    currentBodyParts.leftArm.z += comboIntensity;
    currentBodyParts.rightArm.z += sin(comboCycle + M_PI) * 0.4f;
    
    // Dynamic body movement
    currentBodyParts.torso.x += sin(comboCycle * 0.5f) * 0.1f;
    currentBodyParts.head.x += sin(comboCycle * 0.5f) * 0.05f;
}

void SimpleAnimationController::applyCombo2Animation() {
    float comboCycle = animationTime * 15.0f;
    float comboIntensity = sin(comboCycle) * 0.8f;
    if (comboIntensity < 0) comboIntensity = 0;
    
    // Rapid strikes with both arms
    currentBodyParts.leftArm.z += sin(comboCycle) * 0.5f;
    currentBodyParts.rightArm.z += sin(comboCycle + M_PI/2) * 0.5f;
    
    // More aggressive body movement
    currentBodyParts.torso.z += comboIntensity * 0.2f;
    currentBodyParts.torso.x += sin(comboCycle * 0.8f) * 0.15f;
    
    // Head movement
    currentBodyParts.head.z += comboIntensity * 0.1f;
    currentBodyParts.head.x += sin(comboCycle * 0.8f) * 0.08f;
}

void SimpleAnimationController::applyCombo3Animation() {
    float comboCycle = animationTime * 20.0f;
    float ultimateIntensity = sin(comboCycle) * 1.0f;
    if (ultimateIntensity < 0) ultimateIntensity = 0;
    
    // Ultimate combo - full body involvement
    currentBodyParts.leftArm.z += sin(comboCycle) * 0.7f;
    currentBodyParts.rightArm.z += sin(comboCycle + M_PI/3) * 0.7f;
    currentBodyParts.leftLeg.z += sin(comboCycle + M_PI/6) * 0.4f;
    currentBodyParts.rightLeg.z += sin(comboCycle + M_PI*2/3) * 0.4f;
    
    // Maximum aggression body movement
    currentBodyParts.torso.z += ultimateIntensity * 0.3f;
    currentBodyParts.torso.x += sin(comboCycle * 1.2f) * 0.2f;
    currentBodyParts.torso.y += abs(sin(comboCycle * 2.0f)) * 0.1f;
    
    // Dynamic head movement
    currentBodyParts.head.z += ultimateIntensity * 0.15f;
    currentBodyParts.head.x += sin(comboCycle * 1.2f) * 0.1f;
    currentBodyParts.head.y += abs(sin(comboCycle * 2.0f)) * 0.05f;
}
