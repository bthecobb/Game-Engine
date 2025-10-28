#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glad/glad.h>
#include <vector>
#include <string>

// Basic 3D Character Mesh System
class CharacterMesh {
public:
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 texCoords;
        glm::vec3 color;
        
        Vertex(glm::vec3 pos, glm::vec3 norm, glm::vec2 tex, glm::vec3 col = glm::vec3(1.0f))
            : position(pos), normal(norm), texCoords(tex), color(col) {}
    };

    struct BodyPart {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        GLuint VAO, VBO, EBO;
        glm::vec3 color;
        glm::vec3 localPosition;    // Offset from character center
        glm::vec3 localRotation;    // Local rotation angles
        std::string name;
        
        BodyPart(const std::string& partName, glm::vec3 col = glm::vec3(0.8f, 0.6f, 0.4f))
            : name(partName), color(col), localPosition(0.0f), localRotation(0.0f), VAO(0), VBO(0), EBO(0) {}
    };

private:
    std::vector<BodyPart> bodyParts;
    glm::mat4 worldTransform;
    bool isInitialized;

public:
    CharacterMesh();
    ~CharacterMesh();
    
    // Setup functions
    void createHumanoidModel();
    void initializeBuffers();
    void cleanup();
    
    // Rendering
    void render(GLuint shaderProgram, const glm::mat4& view, const glm::mat4& projection, const glm::vec3& position, float rotation = 0.0f);
    
    // Animation support
    void setBodyPartTransform(const std::string& partName, glm::vec3 position, glm::vec3 rotation);
    void setBodyPartColor(const std::string& partName, glm::vec3 color);
    
    // Utility
    BodyPart* getBodyPart(const std::string& name);
    void setWorldTransform(const glm::mat4& transform) { worldTransform = transform; }

private:
    // Model creation helpers
    void createHead();
    void createTorso();
    void createArms();
    void createLegs();
    void createCube(BodyPart& part, glm::vec3 size, glm::vec3 offset = glm::vec3(0.0f));
    void createCylinder(BodyPart& part, float radius, float height, int segments = 12);
    void setupBodyPartBuffers(BodyPart& part);
};

// Character Animation Controller for visual feedback
class CharacterAnimationController {
public:
    enum AnimationState {
        IDLE,
        IDLE_BORED,
        WALKING,
        RUNNING,
        SPRINTING,
        JUMPING,
        AIRBORNE,
        FALLING,
        COMBAT_IDLE,
        ATTACKING,
        PARRYING,
        SLIDING,
        WALL_RUNNING
    };

private:
    CharacterMesh* mesh;
    AnimationState currentState;
    float animationTime;
    float blendWeight;
    
public:
    CharacterAnimationController(CharacterMesh* characterMesh);
    
    void update(float deltaTime);
    void setState(AnimationState state);
    void setBlendWeight(float weight) { blendWeight = weight; }
    
private:
    void applyIdleAnimation(float time);
    void applyIdleBoredAnimation(float time);
    void applyWalkAnimation(float time);
    void applyRunAnimation(float time);
    void applyJumpAnimation(float time);
    void applyCombatIdleAnimation(float time);
    void applyAttackAnimation(float time);
    void applyParryAnimation(float time);
    void applySlidingAnimation(float time);
    void applyWallRunAnimation(float time);
};
