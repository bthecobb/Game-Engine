#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Player.h"

class CharacterRenderer {
public:
    CharacterRenderer();
    ~CharacterRenderer();
    
    void initialize();
    void render(const Player* player, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix);
    void cleanup();
    
private:
    // Rendering data
    unsigned int m_VAO, m_VBO, m_EBO;
    unsigned int m_shaderProgram;
    
    // Character geometry
    struct CharacterVertex {
        glm::vec3 position;
        glm::vec3 color;
    };
    
    // Shader management
    bool createShaders();
    unsigned int compileShader(const char* source, GLenum type);
    unsigned int createShaderProgram(const char* vertexSource, const char* fragmentSource);
    
    // Geometry creation
    void createCharacterMesh();
    void updateCharacterAnimation(const Player* player);
    
    // Animation state
    float m_animationTime;
    float m_lastSpeed;
    MovementState m_lastState;
    
    // Character properties
    glm::vec3 m_characterColor;
    float m_characterSize;
    bool m_isAnimating;
};
