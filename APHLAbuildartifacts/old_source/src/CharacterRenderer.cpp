#include "CharacterRenderer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include "ShaderRegistry.h"

using namespace CudaGame::Rendering;

CharacterRenderer::CharacterRenderer() 
    : m_VAO(0), m_VBO(0), m_EBO(0), m_shaderProgram(0),
      m_animationTime(0.0f), m_lastSpeed(0.0f), m_lastState(MovementState::IDLE),
      m_characterColor(0.2f, 0.6f, 1.0f), m_characterSize(1.0f), m_isAnimating(false) {
}

CharacterRenderer::~CharacterRenderer() {
    cleanup();
}

void CharacterRenderer::initialize() {
    createShaders();
    createCharacterMesh();
    std::cout << "CharacterRenderer initialized" << std::endl;
}

void CharacterRenderer::render(const Player* player, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix) {
    if (!player || m_shaderProgram == 0) return;
    
    glUseProgram(m_shaderProgram);
    
    // Update animation based on player state
    updateCharacterAnimation(player);
    
    // Create model matrix from player position
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    glm::vec3 playerPos = player->getPosition();
    modelMatrix = glm::translate(modelMatrix, playerPos);
    
    // Scale the character
    modelMatrix = glm::scale(modelMatrix, glm::vec3(m_characterSize));
    
    // Add slight rotation based on movement direction
    glm::vec3 velocity = player->getVelocity();
    float moveDirection = atan2(velocity.z, velocity.x);
    if (glm::length(glm::vec2(velocity.x, velocity.z)) > 0.1f) {
        modelMatrix = glm::rotate(modelMatrix, moveDirection, glm::vec3(0, 1, 0));
    }
    
    // Set uniforms
    unsigned int modelLoc = glGetUniformLocation(m_shaderProgram, "model");
    unsigned int viewLoc = glGetUniformLocation(m_shaderProgram, "view");
    unsigned int projLoc = glGetUniformLocation(m_shaderProgram, "projection");
    unsigned int animLoc = glGetUniformLocation(m_shaderProgram, "animationOffset");
    unsigned int speedLoc = glGetUniformLocation(m_shaderProgram, "speedMultiplier");
    unsigned int glowLoc = glGetUniformLocation(m_shaderProgram, "glowIntensity");
    unsigned int timeLoc = glGetUniformLocation(m_shaderProgram, "time");
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &modelMatrix[0][0]);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &viewMatrix[0][0]);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projectionMatrix[0][0]);
    glUniform1f(animLoc, m_animationTime);
    
    // Speed-based effects
    float speed = player->getCurrentSpeed();
    float speedNormalized = std::min(speed / 30.0f, 1.0f); // Normalize to 0-1
    glUniform1f(speedLoc, speedNormalized);
    glUniform1f(glowLoc, speedNormalized * 0.5f);
    glUniform1f(timeLoc, m_animationTime);
    
    // Render the character (6 cubes * 36 triangles each = 216 triangles)
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, 216, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void CharacterRenderer::cleanup() {
    if (m_VAO) {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
    if (m_VBO) {
        glDeleteBuffers(1, &m_VBO);
        m_VBO = 0;
    }
    if (m_EBO) {
        glDeleteBuffers(1, &m_EBO);
        m_EBO = 0;
    }
    if (m_shaderProgram) {
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }
}

bool CharacterRenderer::createShaders() {
    ShaderRegistry& shaderRegistry = ShaderRegistry::getInstance();
    
    if (!shaderRegistry.initialize()) {
        std::cerr << "ShaderRegistry initialization failed in CharacterRenderer." << std::endl;
        return false;
    }
    
    const std::string& vertexSource = shaderRegistry.getShaderSource(ShaderRegistry::ShaderID::CHARACTER_RENDERER_VERTEX);
    const std::string& fragmentSource = shaderRegistry.getShaderSource(ShaderRegistry::ShaderID::CHARACTER_RENDERER_FRAGMENT);
    
    m_shaderProgram = createShaderProgram(vertexSource.c_str(), fragmentSource.c_str());
    return m_shaderProgram != 0;
}

unsigned int CharacterRenderer::compileShader(const char* source, GLenum type) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    // Check for compilation errors
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Character shader compilation failed: " << infoLog << std::endl;
        return 0;
    }
    
    return shader;
}

unsigned int CharacterRenderer::createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    unsigned int vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    unsigned int fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);
    
    if (vertexShader == 0 || fragmentShader == 0) {
        return 0;
    }
    
    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Character shader program linking failed: " << infoLog << std::endl;
        return 0;
    }
    
    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

void CharacterRenderer::createCharacterMesh() {
    // Create a simple humanoid character using colored cubes
    std::vector<CharacterVertex> vertices;
    std::vector<unsigned int> indices;
    
    // Character colors
    glm::vec3 bodyColor(0.2f, 0.6f, 1.0f);   // Blue body
    glm::vec3 headColor(0.9f, 0.7f, 0.6f);   // Skin tone head
    glm::vec3 limbColor(0.1f, 0.3f, 0.8f);   // Darker blue limbs
    
    // Helper function to add a cube
    auto addCube = [&](glm::vec3 center, glm::vec3 size, glm::vec3 color) {
        int baseIndex = vertices.size();
        
        // Define cube vertices relative to center
        std::vector<glm::vec3> cubeVertices = {
            center + glm::vec3(-size.x, -size.y, -size.z) * 0.5f,
            center + glm::vec3( size.x, -size.y, -size.z) * 0.5f,
            center + glm::vec3( size.x,  size.y, -size.z) * 0.5f,
            center + glm::vec3(-size.x,  size.y, -size.z) * 0.5f,
            center + glm::vec3(-size.x, -size.y,  size.z) * 0.5f,
            center + glm::vec3( size.x, -size.y,  size.z) * 0.5f,
            center + glm::vec3( size.x,  size.y,  size.z) * 0.5f,
            center + glm::vec3(-size.x,  size.y,  size.z) * 0.5f
        };
        
        // Add vertices
        for (const auto& vertex : cubeVertices) {
            vertices.push_back({vertex, color});
        }
        
        // Add indices for cube faces
        std::vector<unsigned int> cubeIndices = {
            0, 1, 2, 2, 3, 0, // Back
            4, 5, 6, 6, 7, 4, // Front
            0, 1, 5, 5, 4, 0, // Bottom
            2, 3, 7, 7, 6, 2, // Top
            0, 3, 7, 7, 4, 0, // Left
            1, 2, 6, 6, 5, 1  // Right
        };
        
        for (unsigned int index : cubeIndices) {
            indices.push_back(baseIndex + index);
        }
    };
    
    // Build character (standing upright)
    // Head
    addCube(glm::vec3(0.0f, 0.7f, 0.0f), glm::vec3(0.3f, 0.3f, 0.3f), headColor);
    
    // Body
    addCube(glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(0.4f, 0.6f, 0.2f), bodyColor);
    
    // Arms
    addCube(glm::vec3(-0.35f, 0.2f, 0.0f), glm::vec3(0.15f, 0.4f, 0.15f), limbColor);
    addCube(glm::vec3(0.35f, 0.2f, 0.0f), glm::vec3(0.15f, 0.4f, 0.15f), limbColor);
    
    // Legs
    addCube(glm::vec3(-0.15f, -0.3f, 0.0f), glm::vec3(0.15f, 0.5f, 0.15f), limbColor);
    addCube(glm::vec3(0.15f, -0.3f, 0.0f), glm::vec3(0.15f, 0.5f, 0.15f), limbColor);
    
    // Create OpenGL buffers
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);
    
    glBindVertexArray(m_VAO);
    
    // Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(CharacterVertex), vertices.data(), GL_STATIC_DRAW);
    
    // Upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CharacterVertex), (void*)offsetof(CharacterVertex, position));
    glEnableVertexAttribArray(0);
    
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CharacterVertex), (void*)offsetof(CharacterVertex, color));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    std::cout << "Character mesh created with " << vertices.size() << " vertices and " << indices.size() << " indices" << std::endl;
}

void CharacterRenderer::updateCharacterAnimation(const Player* player) {
    float currentSpeed = player->getCurrentSpeed();
    MovementState currentState = player->getMovementState();
    
    // Update animation time based on movement
    if (currentSpeed > 0.1f) {
        m_animationTime += 0.016f * (1.0f + currentSpeed * 0.1f); // 60 FPS base
        m_isAnimating = true;
    } else {
        m_isAnimating = false;
    }
    
    // Update character color based on state
    switch (currentState) {
        case MovementState::IDLE:
            m_characterColor = glm::vec3(0.2f, 0.6f, 1.0f); // Blue
            break;
        case MovementState::WALKING:
            m_characterColor = glm::vec3(0.3f, 0.7f, 1.0f); // Light blue
            break;
        case MovementState::RUNNING:
            m_characterColor = glm::vec3(0.1f, 0.8f, 0.3f); // Green
            break;
        case MovementState::SPRINTING:
            m_characterColor = glm::vec3(1.0f, 0.5f, 0.1f); // Orange
            break;
        case MovementState::JUMPING:
            m_characterColor = glm::vec3(1.0f, 1.0f, 0.2f); // Yellow
            break;
        case MovementState::DASHING:
            m_characterColor = glm::vec3(1.0f, 0.2f, 0.8f); // Pink
            break;
        case MovementState::WALL_RUNNING:
            m_characterColor = glm::vec3(0.8f, 0.2f, 1.0f); // Purple
            break;
    }
    
    m_lastSpeed = currentSpeed;
    m_lastState = currentState;
}
