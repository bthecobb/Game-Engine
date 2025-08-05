#include "InteractiveObject.h"
#include <glad/glad.h>
#include <iostream>
#include <cmath>
#include <vector>

// Object shaders (simpler than character shaders)
const char* objectVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float damageFlash;

out vec3 vertexColor;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    
    // Add damage flash effect
    vec3 flashColor = mix(aColor, vec3(1.0, 0.2, 0.2), damageFlash);
    vertexColor = flashColor;
}
)";

const char* objectFragmentShaderSource = R"(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

uniform float alpha;

void main() {
    FragColor = vec4(vertexColor, alpha);
}
)";

InteractiveObject::InteractiveObject(ObjectType type, glm::vec3 position, glm::vec3 size) 
    : m_type(type), m_state(ObjectState::INTACT), m_damageFlashTimer(0.0f), 
      m_destructionTimer(0.0f), m_VAO(0), m_VBO(0), m_EBO(0), m_shaderProgram(0) {
    
    transform.position = position;
    transform.scale = size;
    
    // Set visibility for all views by default
    for (int i = 0; i < 4; i++) {
        activeFromView[i] = true;
        collidableFromView[i] = true;
    }
    
    initializeObject();
    createShaders();
    createMesh();
}

InteractiveObject::~InteractiveObject() {
    if (m_VAO) glDeleteVertexArrays(1, &m_VAO);
    if (m_VBO) glDeleteBuffers(1, &m_VBO);
    if (m_EBO) glDeleteBuffers(1, &m_EBO);
    if (m_shaderProgram) glDeleteProgram(m_shaderProgram);
}

void InteractiveObject::initializeObject() {
    // Set default properties based on object type
    switch (m_type) {
        case ObjectType::DESTRUCTIBLE_BOX:
            m_health = 50.0f;
            m_maxHealth = 50.0f;
            m_canMove = false;
            m_canDestroy = true;
            m_mass = 5.0f;
            m_friction = 0.8f;
            m_restitution = 0.3f;
            m_isKinematic = false;
            m_baseColor = glm::vec3(0.6f, 0.4f, 0.2f); // Brown wood
            break;
            
        case ObjectType::MOVEABLE_CRATE:
            m_health = 100.0f;
            m_maxHealth = 100.0f;
            m_canMove = true;
            m_canDestroy = true;
            m_mass = 10.0f;
            m_friction = 0.6f;
            m_restitution = 0.2f;
            m_isKinematic = true;
            m_baseColor = glm::vec3(0.4f, 0.3f, 0.2f); // Dark brown
            break;
            
        case ObjectType::EXPLOSIVE_BARREL:
            m_health = 30.0f;
            m_maxHealth = 30.0f;
            m_canMove = true;
            m_canDestroy = true;
            m_mass = 8.0f;
            m_friction = 0.5f;
            m_restitution = 0.4f;
            m_isKinematic = true;
            m_baseColor = glm::vec3(0.8f, 0.2f, 0.2f); // Red barrel
            break;
            
        case ObjectType::BOUNCE_PAD:
            m_health = 1000.0f; // Very durable
            m_maxHealth = 1000.0f;
            m_canMove = false;
            m_canDestroy = false;
            m_mass = 50.0f;
            m_friction = 0.9f;
            m_restitution = 2.0f; // Super bouncy
            m_isKinematic = false;
            m_baseColor = glm::vec3(0.2f, 0.8f, 0.2f); // Green bounce pad
            break;
            
        case ObjectType::SPEED_BOOST:
            m_health = 1.0f; // One-time use
            m_maxHealth = 1.0f;
            m_canMove = false;
            m_canDestroy = true;
            m_mass = 1.0f;
            m_friction = 0.0f;
            m_restitution = 0.0f;
            m_isKinematic = false;
            m_baseColor = glm::vec3(1.0f, 1.0f, 0.2f); // Yellow speed boost
            break;
            
        case ObjectType::COLLECTIBLE:
            m_health = 1.0f; // One-time collect
            m_maxHealth = 1.0f;
            m_canMove = false;
            m_canDestroy = true;
            m_mass = 0.5f;
            m_friction = 0.0f;
            m_restitution = 0.0f;
            m_isKinematic = false;
            m_baseColor = glm::vec3(0.2f, 0.2f, 1.0f); // Blue collectible
            break;
    }
    
    m_currentColor = m_baseColor;
}

void InteractiveObject::update(float deltaTime) {
    if (m_state == ObjectState::DESTROYED) return;
    
    updatePhysics(deltaTime);
    updateVisuals(deltaTime);
    
    // Type-specific updates
    switch (m_type) {
        case ObjectType::DESTRUCTIBLE_BOX:
            updateDestructibleBox(deltaTime);
            break;
        case ObjectType::MOVEABLE_CRATE:
            updateMoveableCrate(deltaTime);
            break;
        case ObjectType::EXPLOSIVE_BARREL:
            updateExplosiveBarrel(deltaTime);
            break;
        case ObjectType::BOUNCE_PAD:
            updateBouncePad(deltaTime);
            break;
        case ObjectType::SPEED_BOOST:
            updateSpeedBoost(deltaTime);
            break;
        case ObjectType::COLLECTIBLE:
            updateCollectible(deltaTime);
            break;
    }
}

void InteractiveObject::render() {
    if (m_state == ObjectState::DESTROYED || m_shaderProgram == 0) return;
    
    glUseProgram(m_shaderProgram);
    
    // Create model matrix
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    modelMatrix = glm::translate(modelMatrix, transform.position);
    modelMatrix = glm::scale(modelMatrix, transform.scale);
    
    // Set uniforms (we'll need to get view/projection from somewhere)
    // For now, just render the basic shape
    unsigned int modelLoc = glGetUniformLocation(m_shaderProgram, "model");
    unsigned int flashLoc = glGetUniformLocation(m_shaderProgram, "damageFlash");
    unsigned int alphaLoc = glGetUniformLocation(m_shaderProgram, "alpha");
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &modelMatrix[0][0]);
    glUniform1f(flashLoc, m_damageFlashTimer > 0.0f ? 0.5f : 0.0f);
    
    // Set alpha based on state
    float alpha = 1.0f;
    if (m_state == ObjectState::EXPLODING) {
        alpha = 1.0f - (m_destructionTimer / 1.0f); // Fade out over 1 second
    }
    glUniform1f(alphaLoc, alpha);
    
    // Render the object
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void InteractiveObject::onPlayerCollision(const glm::vec3& playerVelocity, float playerSpeed) {
    switch (m_type) {
        case ObjectType::MOVEABLE_CRATE:
        case ObjectType::EXPLOSIVE_BARREL:
            if (playerSpeed > 15.0f) { // High speed collision
                applyForce(glm::normalize(playerVelocity) * playerSpeed * 2.0f);
                takeDamage(playerSpeed * 0.5f);
            }
            break;
            
        case ObjectType::DESTRUCTIBLE_BOX:
            if (playerSpeed > 20.0f) { // Only break on very high speed
                takeDamage(playerSpeed);
            }
            break;
            
        case ObjectType::BOUNCE_PAD:
            // Player will bounce - this is handled in the player physics
            std::cout << "Player hit bounce pad!" << std::endl;
            break;
            
        case ObjectType::SPEED_BOOST:
            std::cout << "Player got speed boost!" << std::endl;
            destroy(); // Consume the speed boost
            break;
            
        case ObjectType::COLLECTIBLE:
            std::cout << "Player collected item!" << std::endl;
            destroy(); // Collect the item
            break;
    }
}

void InteractiveObject::onPlayerAttack(float attackPower, const glm::vec3& attackDirection) {
    if (!m_canDestroy) return;
    
    takeDamage(attackPower);
    
    if (m_canMove) {
        applyForce(attackDirection * attackPower * 0.5f);
    }
}

void InteractiveObject::takeDamage(float damage) {
    if (m_state == ObjectState::DESTROYED) return;
    
    m_health -= damage;
    m_damageFlashTimer = 0.2f; // Flash red for 0.2 seconds
    
    if (m_health <= 0.0f) {
        destroy();
    } else if (m_health < m_maxHealth * 0.5f) {
        m_state = ObjectState::DAMAGED;
    }
    
    updateColor();
}

void InteractiveObject::destroy() {
    if (m_type == ObjectType::EXPLOSIVE_BARREL && m_state != ObjectState::EXPLODING) {
        m_state = ObjectState::EXPLODING;
        m_destructionTimer = 1.0f; // Explosion animation time
        std::cout << "Barrel exploding!" << std::endl;
        // TODO: Create explosion particles
    } else {
        m_state = ObjectState::DESTROYED;
        std::cout << "Object destroyed!" << std::endl;
        // TODO: Create destruction particles
    }
}

void InteractiveObject::applyForce(const glm::vec3& force) {
    if (!m_isKinematic || m_mass <= 0.0f) return;
    
    m_velocity += force / m_mass;
    m_state = ObjectState::MOVING;
}

void InteractiveObject::updatePhysics(float deltaTime) {
    if (!m_isKinematic || m_state == ObjectState::DESTROYED) return;
    
    // Apply gravity
    m_velocity.y -= 9.8f * deltaTime;
    
    // Apply friction
    if (glm::length(m_velocity) > 0.1f) {
        glm::vec3 friction = -glm::normalize(m_velocity) * m_friction * deltaTime;
        if (glm::length(friction) < glm::length(m_velocity)) {
            m_velocity += friction;
        } else {
            m_velocity = glm::vec3(0.0f);
            if (m_state == ObjectState::MOVING) {
                m_state = ObjectState::INTACT;
            }
        }
    }
    
    // Update position
    transform.position += m_velocity * deltaTime;
    
    // Simple ground collision
    if (transform.position.y < transform.scale.y * 0.5f) {
        transform.position.y = transform.scale.y * 0.5f;
        if (m_velocity.y < 0.0f) {
            m_velocity.y = -m_velocity.y * m_restitution;
        }
    }
}

void InteractiveObject::updateVisuals(float deltaTime) {
    // Update damage flash
    if (m_damageFlashTimer > 0.0f) {
        m_damageFlashTimer -= deltaTime;
    }
    
    // Update destruction timer
    if (m_state == ObjectState::EXPLODING) {
        m_destructionTimer -= deltaTime;
        if (m_destructionTimer <= 0.0f) {
            m_state = ObjectState::DESTROYED;
        }
    }
    
    updateColor();
}

void InteractiveObject::updateColor() {
    switch (m_state) {
        case ObjectState::INTACT:
            m_currentColor = m_baseColor;
            break;
        case ObjectState::DAMAGED:
            m_currentColor = m_baseColor * 0.7f; // Darker when damaged
            break;
        case ObjectState::MOVING:
            m_currentColor = m_baseColor * 1.2f; // Brighter when moving
            break;
        case ObjectState::EXPLODING:
            m_currentColor = glm::vec3(1.0f, 0.5f, 0.0f); // Orange explosion
            break;
        case ObjectState::DESTROYED:
            m_currentColor = glm::vec3(0.3f, 0.3f, 0.3f); // Gray debris
            break;
    }
}

void InteractiveObject::createMesh() {
    // Create a simple cube (similar to platforms)
    float vertices[] = {
        // Positions              // Colors
        -0.5f, -0.5f, -0.5f,     m_currentColor.r, m_currentColor.g, m_currentColor.b,
         0.5f, -0.5f, -0.5f,     m_currentColor.r, m_currentColor.g, m_currentColor.b,
         0.5f,  0.5f, -0.5f,     m_currentColor.r, m_currentColor.g, m_currentColor.b,
        -0.5f,  0.5f, -0.5f,     m_currentColor.r, m_currentColor.g, m_currentColor.b,
        -0.5f, -0.5f,  0.5f,     m_currentColor.r, m_currentColor.g, m_currentColor.b,
         0.5f, -0.5f,  0.5f,     m_currentColor.r, m_currentColor.g, m_currentColor.b,
         0.5f,  0.5f,  0.5f,     m_currentColor.r, m_currentColor.g, m_currentColor.b,
        -0.5f,  0.5f,  0.5f,     m_currentColor.r, m_currentColor.g, m_currentColor.b
    };
    
    unsigned int indices[] = {
        0, 1, 2, 2, 3, 0, // Back
        4, 5, 6, 6, 7, 4, // Front
        0, 1, 5, 5, 4, 0, // Bottom
        2, 3, 7, 7, 6, 2, // Top
        0, 3, 7, 7, 4, 0, // Left
        1, 2, 6, 6, 5, 1  // Right
    };
    
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);
    
    glBindVertexArray(m_VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void InteractiveObject::createShaders() {
    // Compile vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &objectVertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    
    // Compile fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &objectFragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    // Create shader program
    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vertexShader);
    glAttachShader(m_shaderProgram, fragmentShader);
    glLinkProgram(m_shaderProgram);
    
    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

// Type-specific update methods
void InteractiveObject::updateDestructibleBox(float deltaTime) {
    // Boxes just sit there until destroyed
    if (m_health < m_maxHealth * 0.3f) {
        // Wobble when heavily damaged
        transform.rotation.z = sin(m_damageFlashTimer * 20.0f) * 0.1f;
    }
}

void InteractiveObject::updateMoveableCrate(float deltaTime) {
    // Crates can be pushed around
    // Add slight rotation when moving
    if (m_state == ObjectState::MOVING && glm::length(m_velocity) > 0.1f) {
        transform.rotation.y += glm::length(m_velocity) * deltaTime;
    }
}

void InteractiveObject::updateExplosiveBarrel(float deltaTime) {
    // Barrels get more dangerous as they're damaged
    if (m_state == ObjectState::DAMAGED) {
        // Pulse red when damaged
        float pulse = sin(deltaTime * 10.0f) * 0.2f + 0.8f;
        m_currentColor = glm::vec3(pulse, 0.2f, 0.2f);
    }
}

void InteractiveObject::updateBouncePad(float deltaTime) {
    // Bounce pads pulse with energy
    float pulse = sin(deltaTime * 3.0f) * 0.2f + 0.8f;
    m_currentColor = m_baseColor * pulse;
}

void InteractiveObject::updateSpeedBoost(float deltaTime) {
    // Speed boosts rotate and pulse
    transform.rotation.y += deltaTime * 2.0f;
    float pulse = sin(deltaTime * 5.0f) * 0.3f + 0.7f;
    m_currentColor = m_baseColor * pulse;
}

void InteractiveObject::updateCollectible(float deltaTime) {
    // Collectibles float and rotate
    transform.position.y += sin(deltaTime * 2.0f) * 0.01f;
    transform.rotation.y += deltaTime * 1.5f;
    float pulse = sin(deltaTime * 4.0f) * 0.2f + 0.8f;
    m_currentColor = m_baseColor * pulse;
}
