#include "../include/CharacterMesh.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CharacterMesh::CharacterMesh() : worldTransform(1.0f), isInitialized(false) {}

CharacterMesh::~CharacterMesh() {
    cleanup();
}

void CharacterMesh::createHumanoidModel() {
    bodyParts.clear();
    
    // Create body parts
    createHead();
    createTorso();
    createArms();
    createLegs();
    
    initializeBuffers();
    isInitialized = true;
}

void CharacterMesh::createHead() {
    BodyPart head("head", glm::vec3(0.9f, 0.7f, 0.6f)); // Skin tone
    head.localPosition = glm::vec3(0.0f, 1.6f, 0.0f);   // Above torso
    
    // Create sphere-like head using cube approximation
    createCube(head, glm::vec3(0.3f, 0.3f, 0.25f));
    
    bodyParts.push_back(head);
}

void CharacterMesh::createTorso() {
    BodyPart torso("torso", glm::vec3(0.2f, 0.4f, 0.8f)); // Blue shirt
    torso.localPosition = glm::vec3(0.0f, 0.8f, 0.0f);     // Center body
    
    // Rectangular torso
    createCube(torso, glm::vec3(0.5f, 0.8f, 0.3f));
    
    bodyParts.push_back(torso);
}

void CharacterMesh::createArms() {
    // Left arm
    BodyPart leftUpperArm("left_upper_arm", glm::vec3(0.9f, 0.7f, 0.6f)); // Skin
    leftUpperArm.localPosition = glm::vec3(-0.4f, 1.2f, 0.0f);
    createCube(leftUpperArm, glm::vec3(0.15f, 0.4f, 0.15f));
    bodyParts.push_back(leftUpperArm);
    
    BodyPart leftLowerArm("left_lower_arm", glm::vec3(0.9f, 0.7f, 0.6f));
    leftLowerArm.localPosition = glm::vec3(-0.4f, 0.6f, 0.0f);
    createCube(leftLowerArm, glm::vec3(0.12f, 0.4f, 0.12f));
    bodyParts.push_back(leftLowerArm);
    
    // Right arm
    BodyPart rightUpperArm("right_upper_arm", glm::vec3(0.9f, 0.7f, 0.6f));
    rightUpperArm.localPosition = glm::vec3(0.4f, 1.2f, 0.0f);
    createCube(rightUpperArm, glm::vec3(0.15f, 0.4f, 0.15f));
    bodyParts.push_back(rightUpperArm);
    
    BodyPart rightLowerArm("right_lower_arm", glm::vec3(0.9f, 0.7f, 0.6f));
    rightLowerArm.localPosition = glm::vec3(0.4f, 0.6f, 0.0f);
    createCube(rightLowerArm, glm::vec3(0.12f, 0.4f, 0.12f));
    bodyParts.push_back(rightLowerArm);
}

void CharacterMesh::createLegs() {
    // Left leg
    BodyPart leftUpperLeg("left_upper_leg", glm::vec3(0.3f, 0.2f, 0.6f)); // Dark blue pants
    leftUpperLeg.localPosition = glm::vec3(-0.15f, 0.0f, 0.0f);
    createCube(leftUpperLeg, glm::vec3(0.18f, 0.5f, 0.18f));
    bodyParts.push_back(leftUpperLeg);
    
    BodyPart leftLowerLeg("left_lower_leg", glm::vec3(0.9f, 0.7f, 0.6f)); // Skin
    leftLowerLeg.localPosition = glm::vec3(-0.15f, -0.5f, 0.0f);
    createCube(leftLowerLeg, glm::vec3(0.15f, 0.5f, 0.15f));
    bodyParts.push_back(leftLowerLeg);
    
    BodyPart leftFoot("left_foot", glm::vec3(0.2f, 0.1f, 0.1f)); // Dark shoes
    leftFoot.localPosition = glm::vec3(-0.15f, -1.0f, 0.1f);
    createCube(leftFoot, glm::vec3(0.18f, 0.1f, 0.3f));
    bodyParts.push_back(leftFoot);
    
    // Right leg
    BodyPart rightUpperLeg("right_upper_leg", glm::vec3(0.3f, 0.2f, 0.6f));
    rightUpperLeg.localPosition = glm::vec3(0.15f, 0.0f, 0.0f);
    createCube(rightUpperLeg, glm::vec3(0.18f, 0.5f, 0.18f));
    bodyParts.push_back(rightUpperLeg);
    
    BodyPart rightLowerLeg("right_lower_leg", glm::vec3(0.9f, 0.7f, 0.6f));
    rightLowerLeg.localPosition = glm::vec3(0.15f, -0.5f, 0.0f);
    createCube(rightLowerLeg, glm::vec3(0.15f, 0.5f, 0.15f));
    bodyParts.push_back(rightLowerLeg);
    
    BodyPart rightFoot("right_foot", glm::vec3(0.2f, 0.1f, 0.1f));
    rightFoot.localPosition = glm::vec3(0.15f, -1.0f, 0.1f);
    createCube(rightFoot, glm::vec3(0.18f, 0.1f, 0.3f));
    bodyParts.push_back(rightFoot);
}

void CharacterMesh::createCube(BodyPart& part, glm::vec3 size, glm::vec3 offset) {
    part.vertices.clear();
    part.indices.clear();
    
    glm::vec3 min = offset - size * 0.5f;
    glm::vec3 max = offset + size * 0.5f;
    
    // Cube vertices with normals and colors
    std::vector<glm::vec3> positions = {
        // Front face
        {min.x, min.y, max.z}, {max.x, min.y, max.z}, {max.x, max.y, max.z}, {min.x, max.y, max.z},
        // Back face  
        {max.x, min.y, min.z}, {min.x, min.y, min.z}, {min.x, max.y, min.z}, {max.x, max.y, min.z},
        // Left face
        {min.x, min.y, min.z}, {min.x, min.y, max.z}, {min.x, max.y, max.z}, {min.x, max.y, min.z},
        // Right face
        {max.x, min.y, max.z}, {max.x, min.y, min.z}, {max.x, max.y, min.z}, {max.x, max.y, max.z},
        // Bottom face
        {min.x, min.y, min.z}, {max.x, min.y, min.z}, {max.x, min.y, max.z}, {min.x, min.y, max.z},
        // Top face
        {min.x, max.y, max.z}, {max.x, max.y, max.z}, {max.x, max.y, min.z}, {min.x, max.y, min.z}
    };
    
    std::vector<glm::vec3> normals = {
        // Front, Back, Left, Right, Bottom, Top
        {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1},     // Front
        {0,0,-1}, {0,0,-1}, {0,0,-1}, {0,0,-1}, // Back  
        {-1,0,0}, {-1,0,0}, {-1,0,0}, {-1,0,0}, // Left
        {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},     // Right
        {0,-1,0}, {0,-1,0}, {0,-1,0}, {0,-1,0}, // Bottom
        {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0}      // Top
    };
    
    std::vector<glm::vec2> texCoords = {
        {0,0}, {1,0}, {1,1}, {0,1}, // Front
        {0,0}, {1,0}, {1,1}, {0,1}, // Back
        {0,0}, {1,0}, {1,1}, {0,1}, // Left  
        {0,0}, {1,0}, {1,1}, {0,1}, // Right
        {0,0}, {1,0}, {1,1}, {0,1}, // Bottom
        {0,0}, {1,0}, {1,1}, {0,1}  // Top
    };
    
    for (int i = 0; i < positions.size(); ++i) {
        part.vertices.emplace_back(positions[i], normals[i], texCoords[i], part.color);
    }
    
    // Cube indices
    std::vector<unsigned int> indices = {
        0,1,2, 2,3,0,       // Front
        4,5,6, 6,7,4,       // Back
        8,9,10, 10,11,8,    // Left
        12,13,14, 14,15,12, // Right
        16,17,18, 18,19,16, // Bottom
        20,21,22, 22,23,20  // Top
    };
    
    part.indices = indices;
}

void CharacterMesh::initializeBuffers() {
    for (auto& part : bodyParts) {
        setupBodyPartBuffers(part);
    }
}

void CharacterMesh::setupBodyPartBuffers(BodyPart& part) {
    // Generate buffers
    glGenVertexArrays(1, &part.VAO);
    glGenBuffers(1, &part.VBO);
    glGenBuffers(1, &part.EBO);
    
    glBindVertexArray(part.VAO);
    
    // Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, part.VBO);
    glBufferData(GL_ARRAY_BUFFER, part.vertices.size() * sizeof(Vertex), 
                 part.vertices.data(), GL_STATIC_DRAW);
    
    // Upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, part.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, part.indices.size() * sizeof(unsigned int),
                 part.indices.data(), GL_STATIC_DRAW);
    
    // Only set up position attribute to match existing shader
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void CharacterMesh::render(GLuint shaderProgram, const glm::mat4& view, const glm::mat4& projection, 
                          const glm::vec3& position, float rotation) {
    if (!isInitialized) return;
    
    glUseProgram(shaderProgram);
    
    // Set uniform matrices
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint colorLoc = glGetUniformLocation(shaderProgram, "color");
    GLint pulseLoc = glGetUniformLocation(shaderProgram, "pulse");
    
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
    glUniform1f(pulseLoc, 1.0f); // No pulse for character
    
    // Character world transform
    glm::mat4 charTransform = glm::mat4(1.0f);
    charTransform = glm::translate(charTransform, position);
    charTransform = glm::rotate(charTransform, rotation, glm::vec3(0, 1, 0));
    
    // Render each body part
    for (const auto& part : bodyParts) {
        glm::mat4 partTransform = charTransform;
        partTransform = glm::translate(partTransform, part.localPosition);
        
        // Apply local rotations if any
        if (glm::length(part.localRotation) > 0.01f) {
            partTransform = glm::rotate(partTransform, part.localRotation.x, glm::vec3(1, 0, 0));
            partTransform = glm::rotate(partTransform, part.localRotation.y, glm::vec3(0, 1, 0));
            partTransform = glm::rotate(partTransform, part.localRotation.z, glm::vec3(0, 0, 1));
        }
        
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &partTransform[0][0]);
        glUniform3fv(colorLoc, 1, &part.color[0]);
        
        glBindVertexArray(part.VAO);
        glDrawElements(GL_TRIANGLES, part.indices.size(), GL_UNSIGNED_INT, 0);
    }
    
    glBindVertexArray(0);
}

void CharacterMesh::setBodyPartTransform(const std::string& partName, glm::vec3 position, glm::vec3 rotation) {
    auto* part = getBodyPart(partName);
    if (part) {
        part->localPosition = position;
        part->localRotation = rotation;
    }
}

void CharacterMesh::setBodyPartColor(const std::string& partName, glm::vec3 color) {
    auto* part = getBodyPart(partName);
    if (part) {
        part->color = color;
    }
}

CharacterMesh::BodyPart* CharacterMesh::getBodyPart(const std::string& name) {
    for (auto& part : bodyParts) {
        if (part.name == name) {
            return &part;
        }
    }
    return nullptr;
}

void CharacterMesh::cleanup() {
    for (auto& part : bodyParts) {
        if (part.VAO != 0) {
            glDeleteVertexArrays(1, &part.VAO);
            glDeleteBuffers(1, &part.VBO);
            glDeleteBuffers(1, &part.EBO);
        }
    }
    bodyParts.clear();
    isInitialized = false;
}

// Character Animation Controller Implementation
CharacterAnimationController::CharacterAnimationController(CharacterMesh* characterMesh)
    : mesh(characterMesh), currentState(IDLE), animationTime(0.0f), blendWeight(1.0f) {}

void CharacterAnimationController::update(float deltaTime) {
    animationTime += deltaTime;
    
    switch (currentState) {
        case IDLE:
            applyIdleAnimation(animationTime);
            break;
        case IDLE_BORED:
            applyIdleBoredAnimation(animationTime);
            break;
        case WALKING:
            applyWalkAnimation(animationTime);
            break;
        case RUNNING:
            applyRunAnimation(animationTime);
            break;
        case JUMPING:
            applyJumpAnimation(animationTime);
            break;
        case COMBAT_IDLE:
            applyCombatIdleAnimation(animationTime);
            break;
        case ATTACKING:
            applyAttackAnimation(animationTime);
            break;
        case PARRYING:
            applyParryAnimation(animationTime);
            break;
        case SLIDING:
            applySlidingAnimation(animationTime);
            break;
        case WALL_RUNNING:
            applyWallRunAnimation(animationTime);
            break;
        default:
            applyIdleAnimation(animationTime);
            break;
    }
}

void CharacterAnimationController::setState(AnimationState state) {
    if (currentState != state) {
        currentState = state;
        animationTime = 0.0f;
    }
}

void CharacterAnimationController::applyIdleAnimation(float time) {
    float breathe = sin(time * 2.0f) * 0.02f;
    
    // Subtle breathing animation
    mesh->setBodyPartTransform("torso", glm::vec3(0.0f, 0.8f + breathe, 0.0f), glm::vec3(0.0f));
    mesh->setBodyPartTransform("head", glm::vec3(0.0f, 1.6f + breathe, 0.0f), glm::vec3(0.0f));
}

void CharacterAnimationController::applyIdleBoredAnimation(float time) {
    float sway = sin(time * 0.8f) * 0.05f;
    float headTilt = sin(time * 0.6f) * 0.1f;
    
    // Swaying and head tilting when bored
    mesh->setBodyPartTransform("torso", glm::vec3(sway, 0.8f, 0.0f), glm::vec3(0.0f));
    mesh->setBodyPartTransform("head", glm::vec3(sway, 1.6f, 0.0f), glm::vec3(0.0f, 0.0f, headTilt));
}

void CharacterAnimationController::applyWalkAnimation(float time) {
    float walkCycle = time * 4.0f;
    float armSwing = sin(walkCycle) * 0.3f;
    float legSwing = sin(walkCycle) * 0.2f;
    
    // Walking arm and leg movement
    mesh->setBodyPartTransform("left_upper_arm", glm::vec3(-0.4f, 1.2f, 0.0f), glm::vec3(armSwing, 0.0f, 0.0f));
    mesh->setBodyPartTransform("right_upper_arm", glm::vec3(0.4f, 1.2f, 0.0f), glm::vec3(-armSwing, 0.0f, 0.0f));
    mesh->setBodyPartTransform("left_upper_leg", glm::vec3(-0.15f, 0.0f, 0.0f), glm::vec3(legSwing, 0.0f, 0.0f));
    mesh->setBodyPartTransform("right_upper_leg", glm::vec3(0.15f, 0.0f, 0.0f), glm::vec3(-legSwing, 0.0f, 0.0f));
}

void CharacterAnimationController::applyRunAnimation(float time) {
    float runCycle = time * 6.0f;
    float armSwing = sin(runCycle) * 0.5f;
    float legSwing = sin(runCycle) * 0.4f;
    float bounce = abs(sin(runCycle)) * 0.1f;
    
    // Running with more pronounced movement and bounce
    mesh->setBodyPartTransform("torso", glm::vec3(0.0f, 0.8f + bounce, 0.0f), glm::vec3(0.0f));
    mesh->setBodyPartTransform("head", glm::vec3(0.0f, 1.6f + bounce, 0.0f), glm::vec3(0.0f));
    mesh->setBodyPartTransform("left_upper_arm", glm::vec3(-0.4f, 1.2f, 0.0f), glm::vec3(armSwing, 0.0f, 0.0f));
    mesh->setBodyPartTransform("right_upper_arm", glm::vec3(0.4f, 1.2f, 0.0f), glm::vec3(-armSwing, 0.0f, 0.0f));
    mesh->setBodyPartTransform("left_upper_leg", glm::vec3(-0.15f, 0.0f, 0.0f), glm::vec3(legSwing, 0.0f, 0.0f));
    mesh->setBodyPartTransform("right_upper_leg", glm::vec3(0.15f, 0.0f, 0.0f), glm::vec3(-legSwing, 0.0f, 0.0f));
}

void CharacterAnimationController::applyJumpAnimation(float time) {
    // Arms raised up during jump
    mesh->setBodyPartTransform("left_upper_arm", glm::vec3(-0.4f, 1.2f, 0.0f), glm::vec3(-0.5f, 0.0f, -0.3f));
    mesh->setBodyPartTransform("right_upper_arm", glm::vec3(0.4f, 1.2f, 0.0f), glm::vec3(-0.5f, 0.0f, 0.3f));
    mesh->setBodyPartTransform("left_upper_leg", glm::vec3(-0.15f, 0.0f, 0.0f), glm::vec3(-0.3f, 0.0f, 0.0f));
    mesh->setBodyPartTransform("right_upper_leg", glm::vec3(0.15f, 0.0f, 0.0f), glm::vec3(-0.3f, 0.0f, 0.0f));
}

void CharacterAnimationController::applyCombatIdleAnimation(float time) {
    float tense = sin(time * 4.0f) * 0.01f;
    
    // Tense combat stance
    mesh->setBodyPartTransform("torso", glm::vec3(0.0f, 0.8f + tense, 0.0f), glm::vec3(0.1f, 0.0f, 0.0f));
    mesh->setBodyPartTransform("left_upper_arm", glm::vec3(-0.4f, 1.2f, 0.0f), glm::vec3(-0.2f, 0.0f, -0.2f));
    mesh->setBodyPartTransform("right_upper_arm", glm::vec3(0.4f, 1.2f, 0.0f), glm::vec3(-0.2f, 0.0f, 0.2f));
}

void CharacterAnimationController::applyAttackAnimation(float time) {
    float attackProgress = std::min(1.0f, time * 3.0f);
    float swing = sin(attackProgress * M_PI) * 0.8f;
    
    // Attack swing animation
    mesh->setBodyPartTransform("right_upper_arm", glm::vec3(0.4f, 1.2f, 0.0f), glm::vec3(-swing, 0.0f, swing));
    mesh->setBodyPartTransform("torso", glm::vec3(0.0f, 0.8f, 0.0f), glm::vec3(0.0f, swing * 0.3f, 0.0f));
}

void CharacterAnimationController::applyParryAnimation(float time) {
    // Defensive posture with arms in front
    mesh->setBodyPartTransform("left_upper_arm", glm::vec3(-0.4f, 1.2f, 0.0f), glm::vec3(-0.8f, 0.0f, -0.6f));
    mesh->setBodyPartTransform("right_upper_arm", glm::vec3(0.4f, 1.2f, 0.0f), glm::vec3(-0.8f, 0.0f, 0.6f));
    mesh->setBodyPartTransform("torso", glm::vec3(0.0f, 0.8f, 0.0f), glm::vec3(-0.2f, 0.0f, 0.0f));
}

void CharacterAnimationController::applySlidingAnimation(float time) {
    // Low sliding pose
    mesh->setBodyPartTransform("torso", glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.3f, 0.0f, 0.0f));
    mesh->setBodyPartTransform("head", glm::vec3(0.0f, 1.3f, 0.0f), glm::vec3(0.3f, 0.0f, 0.0f));
    mesh->setBodyPartTransform("left_upper_leg", glm::vec3(-0.15f, -0.2f, 0.0f), glm::vec3(0.8f, 0.0f, 0.0f));
    mesh->setBodyPartTransform("right_upper_leg", glm::vec3(0.15f, -0.2f, 0.0f), glm::vec3(0.8f, 0.0f, 0.0f));
}

void CharacterAnimationController::applyWallRunAnimation(float time) {
    float runCycle = time * 8.0f;
    float armSwing = sin(runCycle) * 0.4f;
    
    // Wall running with angled body
    mesh->setBodyPartTransform("torso", glm::vec3(0.0f, 0.8f, 0.0f), glm::vec3(0.0f, 0.0f, 0.3f));
    mesh->setBodyPartTransform("head", glm::vec3(0.0f, 1.6f, 0.0f), glm::vec3(0.0f, 0.0f, 0.3f));
    mesh->setBodyPartTransform("left_upper_arm", glm::vec3(-0.4f, 1.2f, 0.0f), glm::vec3(armSwing, 0.0f, 0.0f));
    mesh->setBodyPartTransform("right_upper_arm", glm::vec3(0.4f, 1.2f, 0.0f), glm::vec3(-armSwing, 0.0f, 0.0f));
}
