#pragma once

#include <string>
#include <unordered_map>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct BodyParts {
    struct { float x, y, z; } head;
    struct { float x, y, z; } torso;
    struct { float x, y, z; } leftArm;
    struct { float x, y, z; } rightArm;
    struct { float x, y, z; } leftLeg;
    struct { float x, y, z; } rightLeg;
};

class SimpleAnimationController {
public:
    SimpleAnimationController();
    ~SimpleAnimationController();
    
    void setState(const std::string& state);
    void update(float deltaTime);
    
    BodyParts getBodyParts() const { return currentBodyParts; }
    std::string getCurrentState() const { return currentState; }
    float getEnergyLevel() const { return energyLevel; }
    
private:
    void updateBodyParts();
    void applyBreathing();
    void applyWalkCycle();
    void applyRunCycle();
    void applySprintCycle();
    void applyJumpCycle();
    void applyDoubleJumpCycle();
    void applyPunchAnimation();
    void applyKickAnimation();
    void applyCombo1Animation();
    void applyCombo2Animation();
    void applyCombo3Animation();
    
    std::string currentState;
    float animationTime;
    float energyLevel;
    BodyParts currentBodyParts;
    
    // Animation parameters
    float walkSpeed;
    float runSpeed;
    float sprintSpeed;
    float breathingRate;
};
