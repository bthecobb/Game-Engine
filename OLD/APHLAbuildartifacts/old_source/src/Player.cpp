#include "Player.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "ShaderRegistry.h"

using namespace CudaGame::Rendering;

Player::Player() {
    transform.position = glm::vec3(0.0f, 2.0f, 0.0f);
    transform.scale = glm::vec3(1.0f);
    transform.rotation = glm::vec3(0.0f);
    
    m_velocity = glm::vec3(0.0f);
    m_movementState = MovementState::IDLE;
    
    initializeBodyParts();
    loadAnimations();
    createCharacterShaders();
    createParticleShaders();
    
    m_particles.resize(100);
    
    std::cout << "Enhanced Player initialized!" << std::endl;
}

void Player::initializeBodyParts() {
    m_bodyParts.resize(6);
    
    // HEAD (0)
    {
        std::vector<float> headVertices = {
            -0.25f, -0.25f, -0.25f,  0.0f, -1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
             0.25f, -0.25f, -0.25f,  0.0f, -1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
             0.25f,  0.25f, -0.25f,  0.0f,  1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
            -0.25f,  0.25f, -0.25f,  0.0f,  1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
            -0.25f, -0.25f,  0.25f,  0.0f, -1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
             0.25f, -0.25f,  0.25f,  0.0f, -1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
             0.25f,  0.25f,  0.25f,  0.0f,  1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
            -0.25f,  0.25f,  0.25f,  0.0f,  1.0f, 0.0f,  0.9f, 0.8f, 0.7f
        };
        std::vector<unsigned int> headIndices = {
            0,1,2, 2,3,0, 4,5,6, 6,7,4, 0,1,5, 5,4,0,
            2,3,7, 7,6,2, 0,3,7, 7,4,0, 1,2,6, 6,5,1
        };
        createBodyPartMesh(m_bodyParts[0], headVertices, headIndices);
        m_bodyParts[0].position = glm::vec3(0.0f, 1.5f, 0.0f);
        m_bodyParts[0].scale = glm::vec3(0.8f, 0.8f, 0.8f);
        m_bodyParts[0].color = glm::vec3(0.9f, 0.8f, 0.7f);
    }
    
    // TORSO (1)
    {
        std::vector<float> torsoVertices = {
            -0.4f, -0.6f, -0.2f,  0.0f, -1.0f, 0.0f,  0.2f, 0.3f, 0.8f,
             0.4f, -0.6f, -0.2f,  0.0f, -1.0f, 0.0f,  0.2f, 0.3f, 0.8f,
             0.4f,  0.6f, -0.2f,  0.0f,  1.0f, 0.0f,  0.2f, 0.3f, 0.8f,
            -0.4f,  0.6f, -0.2f,  0.0f,  1.0f, 0.0f,  0.2f, 0.3f, 0.8f,
            -0.4f, -0.6f,  0.2f,  0.0f, -1.0f, 0.0f,  0.2f, 0.3f, 0.8f,
             0.4f, -0.6f,  0.2f,  0.0f, -1.0f, 0.0f,  0.2f, 0.3f, 0.8f,
             0.4f,  0.6f,  0.2f,  0.0f,  1.0f, 0.0f,  0.2f, 0.3f, 0.8f,
            -0.4f,  0.6f,  0.2f,  0.0f,  1.0f, 0.0f,  0.2f, 0.3f, 0.8f
        };
        std::vector<unsigned int> torsoIndices = {
            0,1,2, 2,3,0, 4,5,6, 6,7,4, 0,1,5, 5,4,0,
            2,3,7, 7,6,2, 0,3,7, 7,4,0, 1,2,6, 6,5,1
        };
        createBodyPartMesh(m_bodyParts[1], torsoVertices, torsoIndices);
        m_bodyParts[1].position = glm::vec3(0.0f, 0.5f, 0.0f);
        m_bodyParts[1].scale = glm::vec3(1.0f, 1.2f, 0.8f);
        m_bodyParts[1].color = glm::vec3(0.2f, 0.3f, 0.8f);
    }
    
    // Initialize other body parts (arms and legs) similarly...
    // Simplified for initial testing
    for (int i = 2; i < 6; i++) {
        std::vector<float> limbVertices = {
            -0.15f, -0.4f, -0.15f,  0.0f, -1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
             0.15f, -0.4f, -0.15f,  0.0f, -1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
             0.15f,  0.4f, -0.15f,  0.0f,  1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
            -0.15f,  0.4f, -0.15f,  0.0f,  1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
            -0.15f, -0.4f,  0.15f,  0.0f, -1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
             0.15f, -0.4f,  0.15f,  0.0f, -1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
             0.15f,  0.4f,  0.15f,  0.0f,  1.0f, 0.0f,  0.9f, 0.8f, 0.7f,
            -0.15f,  0.4f,  0.15f,  0.0f,  1.0f, 0.0f,  0.9f, 0.8f, 0.7f
        };
        std::vector<unsigned int> limbIndices = {
            0,1,2, 2,3,0, 4,5,6, 6,7,4, 0,1,5, 5,4,0,
            2,3,7, 7,6,2, 0,3,7, 7,4,0, 1,2,6, 6,5,1
        };
        createBodyPartMesh(m_bodyParts[i], limbVertices, limbIndices);
        
        // Position limbs
        switch(i) {
            case 2: // Left arm
                m_bodyParts[i].position = glm::vec3(-0.6f, 0.8f, 0.0f);
                break;
            case 3: // Right arm
                m_bodyParts[i].position = glm::vec3(0.6f, 0.8f, 0.0f);
                break;
            case 4: // Left leg
                m_bodyParts[i].position = glm::vec3(-0.2f, -0.8f, 0.0f);
                m_bodyParts[i].color = glm::vec3(0.1f, 0.1f, 0.3f);
                break;
            case 5: // Right leg
                m_bodyParts[i].position = glm::vec3(0.2f, -0.8f, 0.0f);
                m_bodyParts[i].color = glm::vec3(0.1f, 0.1f, 0.3f);
                break;
        }
        
        m_bodyParts[i].scale = glm::vec3(0.6f, 1.0f, 0.6f);
    }
}

void Player::createBodyPartMesh(BodyPart& part, const std::vector<float>& vertices, const std::vector<unsigned int>& indices) {
    glGenVertexArrays(1, &part.VAO);
    glGenBuffers(1, &part.VBO);
    glGenBuffers(1, &part.EBO);
    
    glBindVertexArray(part.VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, part.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, part.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Color attribute
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    glBindVertexArray(0);
    
    part.indexCount = indices.size();
}

void Player::loadAnimations() {
    m_animations.resize(12);
    
    // IDLE ANIMATION
    Animation& idle = m_animations[static_cast<int>(AnimationState::IDLE_ANIM)];
    idle.duration = 2.0f;
    idle.looping = true;
    
    AnimationKeyframe frame1, frame2;
    frame1.time = 0.0f;
    frame2.time = 1.0f;
    
    for (int i = 0; i < 6; i++) {
        frame1.bodyPartPositions.push_back(m_bodyParts[i].position);
        frame1.bodyPartRotations.push_back(glm::vec3(0.0f));
        frame2.bodyPartPositions.push_back(m_bodyParts[i].position + glm::vec3(0.0f, 0.02f, 0.0f));
        frame2.bodyPartRotations.push_back(glm::vec3(0.0f, 0.0f, sin(1.0f) * 0.05f));
    }
    
    idle.keyframes = {frame1, frame2};
    
    // Copy idle for other animations initially
    for (int i = 1; i < 12; i++) {
        m_animations[i] = idle;
        m_animations[i].duration *= (i == 2 ? 0.7f : i == 3 ? 0.5f : 1.0f);
    }
}

void Player::createCharacterShaders() {
    ShaderRegistry& shaderRegistry = ShaderRegistry::getInstance();

    if (!shaderRegistry.initialize()) {
        std::cerr << "ShaderRegistry initialization failed." << std::endl;
        return;
    }

    const std::string& vertexSource = shaderRegistry.getShaderSource(ShaderRegistry::ShaderID::PLAYER_CHARACTER_VERTEX);
    const std::string& fragmentSource = shaderRegistry.getShaderSource(ShaderRegistry::ShaderID::PLAYER_CHARACTER_FRAGMENT);

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const char* vertexSourcePtr = vertexSource.c_str();
    glShaderSource(vertexShader, 1, &vertexSourcePtr, nullptr);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fragmentSourcePtr = fragmentSource.c_str();
    glShaderSource(fragmentShader, 1, &fragmentSourcePtr, nullptr);
    glCompileShader(fragmentShader);

    m_characterShaderProgram = glCreateProgram();
    glAttachShader(m_characterShaderProgram, vertexShader);
    glAttachShader(m_characterShaderProgram, fragmentShader);
    glLinkProgram(m_characterShaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Player::createParticleShaders() {
    ShaderRegistry& shaderRegistry = ShaderRegistry::getInstance();

    const std::string& vertexSource = shaderRegistry.getShaderSource(ShaderRegistry::ShaderID::PLAYER_PARTICLE_VERTEX);
    const std::string& fragmentSource = shaderRegistry.getShaderSource(ShaderRegistry::ShaderID::PLAYER_PARTICLE_FRAGMENT);

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const char* vertexSourcePtr = vertexSource.c_str();
    glShaderSource(vertexShader, 1, &vertexSourcePtr, nullptr);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fragmentSourcePtr = fragmentSource.c_str();
    glShaderSource(fragmentShader, 1, &fragmentSourcePtr, nullptr);
    glCompileShader(fragmentShader);

    m_particleShaderProgram = glCreateProgram();
    glAttachShader(m_particleShaderProgram, vertexShader);
    glAttachShader(m_particleShaderProgram, fragmentShader);
    glLinkProgram(m_particleShaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Player::update(float deltaTime) {
    // Update core systems
    updateMovement(deltaTime);
    updatePhysics(deltaTime);
    updateDashing(deltaTime);
    updateWallRunning(deltaTime);
    updateCombat(deltaTime);
    
    // Update visual systems
    updateAnimations(deltaTime);
    updateRhythmFeedback(deltaTime);
    updateParticles(deltaTime);
    updateBodyParts(deltaTime);
    
    // Update movement state
    float speed = getCurrentSpeed();
    if (speed < 0.1f) {
        m_movementState = MovementState::IDLE;
    } else if (speed < m_baseSpeed * 1.5f) {
        m_movementState = MovementState::WALKING;
    } else if (speed < m_baseSpeed * 2.5f) {
        m_movementState = MovementState::RUNNING;
    } else {
        m_movementState = MovementState::SPRINTING;
    }
    
    if (!m_isGrounded) m_movementState = MovementState::JUMPING;
    if (m_isDashing) m_movementState = MovementState::DASHING;
    if (m_isWallRunning) m_movementState = MovementState::WALL_RUNNING;
    
    // Add particle trails during high-speed movement
    if (getCurrentSpeed() > 20.0f) {
        addParticleTrail();
    }
}

void Player::updateAnimations(float deltaTime) {
    m_animationTime += deltaTime * m_animationSpeed;
    
    // Determine animation state
    AnimationState newState = AnimationState::IDLE_ANIM;
    
    if (m_isDashing) {
        newState = AnimationState::DASH_POSE;
    } else if (m_isWallRunning) {
        newState = AnimationState::WALL_RUN_POSE;
    } else if (!m_isGrounded) {
        newState = AnimationState::JUMP_PEAK;
    } else if (m_combatState == CombatState::ATTACKING) {
        newState = AnimationState::ATTACK_LIGHT;
    } else if (m_combatState == CombatState::BLOCKING) {
        newState = AnimationState::BLOCK_POSE;
    } else {
        float speed = getCurrentSpeed();
        if (speed > 30.0f) newState = AnimationState::SPRINT_CYCLE;
        else if (speed > 15.0f) newState = AnimationState::RUN_CYCLE;
        else if (speed > 2.0f) newState = AnimationState::WALK_CYCLE;
    }
    
    if (newState != m_currentAnimationState) {
        m_currentAnimationState = newState;
        m_animationTime = 0.0f;
    }
}

void Player::updateRhythmFeedback(float deltaTime) {
    m_beatTimer += deltaTime * 2.0f; // 120 BPM simulation
    
    if (m_beatTimer >= 1.0f) {
        m_beatTimer = 0.0f;
        m_isOnBeat = true;
        createParticleEffects(transform.position + glm::vec3(0.0f, 1.0f, 0.0f), 5, 0.2f);
    } else if (m_beatTimer > 0.1f) {
        m_isOnBeat = false;
    }
}

void Player::addParticleTrail() {
    if (m_particleCooldown > 0.0f) return;
    
    m_particleCooldown = 0.05f;
    
    for (auto& particle : m_particles) {
        if (!particle.active) {
            particle.position = transform.position;
            particle.velocity = -m_velocity * 0.3f;
            particle.color = glm::vec3(0.3f, 0.7f, 1.0f);
            particle.life = 1.0f;
            particle.maxLife = 1.0f;
            particle.size = 3.0f;
            particle.active = true;
            break;
        }
    }
}

void Player::createParticleEffects(glm::vec3 position, int count, float size) {
    int created = 0;
    for (auto& particle : m_particles) {
        if (!particle.active && created < count) {
            particle.position = position;
            particle.velocity = glm::vec3(
                (rand() % 200 - 100) * 0.01f,
                (rand() % 100) * 0.02f,
                (rand() % 200 - 100) * 0.01f
            );
            particle.color = glm::vec3(1.0f, 0.8f, 0.2f);
            particle.life = 0.5f;
            particle.maxLife = 0.5f;
            particle.size = size * 5.0f;
            particle.active = true;
            created++;
        }
    }
}

void Player::updateParticles(float deltaTime) {
    m_particleCooldown = std::max(0.0f, m_particleCooldown - deltaTime);
    
    for (auto& particle : m_particles) {
        if (particle.active) {
            particle.life -= deltaTime;
            particle.position += particle.velocity * deltaTime;
            particle.velocity.y -= 9.8f * deltaTime;
            particle.velocity *= 0.98f;
            
            float lifeRatio = particle.life / particle.maxLife;
            particle.color *= lifeRatio;
            particle.size *= lifeRatio;
            
            if (particle.life <= 0.0f) {
                particle.active = false;
            }
        }
    }
}

void Player::updateBodyParts(float deltaTime) {
    for (auto& part : m_bodyParts) {
        if (m_isOnBeat) {
            part.scale *= 1.05f;
        } else {
            part.scale = glm::mix(part.scale, glm::vec3(1.0f), deltaTime * 5.0f);
        }
    }
}

void Player::render() {
    if (m_characterShaderProgram == 0) return;
    
    glUseProgram(m_characterShaderProgram);
    
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, transform.position);
    model = glm::scale(model, transform.scale);
    
    unsigned int modelLoc = glGetUniformLocation(m_characterShaderProgram, "model");
    unsigned int rhythmPulseLoc = glGetUniformLocation(m_characterShaderProgram, "rhythmPulse");
    unsigned int animationBendLoc = glGetUniformLocation(m_characterShaderProgram, "animationBend");
    unsigned int rhythmIntensityLoc = glGetUniformLocation(m_characterShaderProgram, "rhythmIntensity");
    unsigned int timeLoc = glGetUniformLocation(m_characterShaderProgram, "time");
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model[0][0]);
    glUniform1f(rhythmPulseLoc, m_isOnBeat ? 1.0f : 0.0f);
    glUniform1f(animationBendLoc, m_animationTime);
    glUniform1f(rhythmIntensityLoc, m_isOnBeat ? 1.0f : 0.2f);
    glUniform1f(timeLoc, m_animationTime);
    
    // Render body parts
    for (const auto& part : m_bodyParts) {
        if (part.VAO != 0) {
            renderBodyPart(part, model);
        }
    }
    
    renderParticles();
}

void Player::renderBodyPart(const BodyPart& part, const glm::mat4& parentTransform) {
    glm::mat4 partModel = parentTransform;
    partModel = glm::translate(partModel, part.position);
    partModel = glm::rotate(partModel, part.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    partModel = glm::rotate(partModel, part.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    partModel = glm::rotate(partModel, part.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    partModel = glm::scale(partModel, part.scale);
    
    unsigned int modelLoc = glGetUniformLocation(m_characterShaderProgram, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &partModel[0][0]);
    
    glBindVertexArray(part.VAO);
    glDrawElements(GL_TRIANGLES, part.indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Player::renderParticles() {
    if (m_particleShaderProgram == 0) return;
    
    glUseProgram(m_particleShaderProgram);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    
    unsigned int timeLoc = glGetUniformLocation(m_particleShaderProgram, "time");
    glUniform1f(timeLoc, m_animationTime);
    
    std::vector<float> particleData;
    for (const auto& particle : m_particles) {
        if (particle.active) {
            particleData.insert(particleData.end(), {
                particle.position.x, particle.position.y, particle.position.z,
                particle.color.r, particle.color.g, particle.color.b,
                particle.size,
                particle.life / particle.maxLife
            });
        }
    }
    
    if (!particleData.empty()) {
        unsigned int particleVAO, particleVBO;
        glGenVertexArrays(1, &particleVAO);
        glGenBuffers(1, &particleVBO);
        
        glBindVertexArray(particleVAO);
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
        glBufferData(GL_ARRAY_BUFFER, particleData.size() * sizeof(float), particleData.data(), GL_DYNAMIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(7 * sizeof(float)));
        glEnableVertexAttribArray(3);
        
        glDrawArrays(GL_POINTS, 0, particleData.size() / 8);
        
        glDeleteVertexArrays(1, &particleVAO);
        glDeleteBuffers(1, &particleVBO);
    }
    
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

// Stub implementations for existing Player methods
void Player::handleInput(const bool keys[1024], float deltaTime) {
    glm::vec2 inputDirection = getMovementInput(keys);
    if (glm::length(inputDirection) > 0.0f) {
        move(inputDirection, deltaTime);
    }
    
    if (keys[GLFW_KEY_SPACE]) jump();
    if (keys[GLFW_KEY_LEFT_SHIFT]) dash();
    if (keys[GLFW_KEY_LEFT_CONTROL]) attack(false);
    if (keys[GLFW_KEY_LEFT_ALT]) attack(true);
    if (keys[GLFW_KEY_B]) block();
}

void Player::handleMouseInput(double mouseX, double mouseY) {
    // Mouse input handling
}

void Player::move(glm::vec2 inputDirection, float deltaTime) {
    if (!m_gameWorld) return;
    
    glm::vec2 worldDir = worldToViewDirection(inputDirection, m_gameWorld->getCurrentView());
    buildMomentum(worldDir, deltaTime);
}

void Player::jump() {
    if (m_isGrounded && !m_isDashing) {
        m_velocity.y = m_jumpForce;
        m_isGrounded = false;
        m_canDoubleJump = true;
    } else if (m_canDoubleJump && !m_isGrounded) {
        m_velocity.y = m_jumpForce * 0.8f;
        m_canDoubleJump = false;
    }
}

void Player::dash() {
    if (m_dashCooldownTimer <= 0.0f && !m_isDashing) {
        m_isDashing = true;
        m_dashTimer = m_dashDuration;
        m_dashCooldownTimer = m_dashCooldown;
        
        glm::vec3 dashDirection = glm::vec3(0, 0, -1);
        if (glm::length(m_velocity) > 0.1f) {
            dashDirection = glm::normalize(glm::vec3(m_velocity.x, 0, m_velocity.z));
        }
        
        m_velocity.x = dashDirection.x * m_dashForce;
        m_velocity.z = dashDirection.z * m_dashForce;
    }
}

void Player::wallRun(glm::vec3 wallNormal, float deltaTime) {
    if (!m_isWallRunning) {
        m_isWallRunning = true;
        m_wallRunTimer = m_wallRunDuration;
        m_wallNormal = wallNormal;
    }
    
    glm::vec3 wallDirection = glm::normalize(glm::cross(wallNormal, glm::vec3(0, 1, 0)));
    m_velocity.x = wallDirection.x * m_wallRunSpeed;
    m_velocity.z = wallDirection.z * m_wallRunSpeed;
    m_velocity.y = 0;
}

void Player::attack(bool heavy) {
    if (m_attackTimer <= 0.0f) {
        m_attackTimer = m_attackCooldown;
        m_combatState = CombatState::ATTACKING;
        
        if (heavy) {
            playAnimation(AnimationState::ATTACK_HEAVY);
            createParticleEffects(transform.position + glm::vec3(0.0f, 1.0f, 0.0f), 8, 0.3f);
        } else {
            playAnimation(AnimationState::ATTACK_LIGHT);
            createParticleEffects(transform.position + glm::vec3(0.0f, 1.0f, 0.0f), 5, 0.2f);
        }
    }
}

void Player::block() {
    m_isBlocking = true;
    m_combatState = CombatState::BLOCKING;
    playAnimation(AnimationState::BLOCK_POSE);
}

void Player::playAnimation(AnimationState animState) {
    if (animState != m_currentAnimationState) {
        m_currentAnimationState = animState;
        m_animationTime = 0.0f;
    }
}

void Player::preserveMomentumOnRotation() {
    std::cout << "Preserving momentum through rotation" << std::endl;
}

// Stub implementations for core Player methods
void Player::updateMovement(float deltaTime) {
    if (m_isGrounded && !m_isDashing) {
        applyFriction(deltaTime);
    }
}

void Player::updatePhysics(float deltaTime) {
    if (!m_isGrounded && !m_isWallRunning && !m_isDashing) {
        applyGravity(deltaTime);
    }
    
    if (m_gameWorld) {
        glm::vec3 size = transform.scale;
        m_velocity = m_gameWorld->resolveCollision(transform.position, m_velocity, size);
    }
    
    transform.position += m_velocity * deltaTime;
    
    transform.position.x = glm::clamp(transform.position.x, -50.0f, 50.0f);
    transform.position.z = glm::clamp(transform.position.z, -50.0f, 50.0f);
    
    if (transform.position.y < 0.0f) {
        transform.position.y = 0.0f;
        m_velocity.y = 0.0f;
        m_isGrounded = true;
    }
}

void Player::updateDashing(float deltaTime) {
    if (m_isDashing) {
        m_dashTimer -= deltaTime;
        if (m_dashTimer <= 0.0f) {
            m_isDashing = false;
        }
    }
    
    if (m_dashCooldownTimer > 0.0f) {
        m_dashCooldownTimer -= deltaTime;
    }
}

void Player::updateWallRunning(float deltaTime) {
    if (m_isWallRunning) {
        m_wallRunTimer -= deltaTime;
        if (m_wallRunTimer <= 0.0f) {
            m_isWallRunning = false;
        }
    }
}

void Player::updateCombat(float deltaTime) {
    if (m_attackTimer > 0.0f) {
        m_attackTimer -= deltaTime;
    } else {
        m_combatState = CombatState::NEUTRAL;
    }
    
    m_isBlocking = false;
}

bool Player::checkGrounding() {
    return transform.position.y <= 0.1f;
}

glm::vec3 Player::checkWallCollision() {
    return glm::vec3(0.0f);
}

void Player::applyFriction(float deltaTime) {
    float friction = m_deceleration * deltaTime;
    glm::vec3 horizontalVel = glm::vec3(m_velocity.x, 0, m_velocity.z);
    float speed = glm::length(horizontalVel);
    
    if (speed > 0.0f) {
        glm::vec3 frictionForce = -glm::normalize(horizontalVel) * friction;
        if (glm::length(frictionForce) * deltaTime > speed) {
            m_velocity.x = 0;
            m_velocity.z = 0;
        } else {
            m_velocity.x += frictionForce.x;
            m_velocity.z += frictionForce.z;
        }
    }
}

void Player::applyGravity(float deltaTime) {
    m_velocity.y -= m_gravity * deltaTime;
    m_velocity.y = glm::max(m_velocity.y, -50.0f);
}

glm::vec2 Player::getMovementInput(const bool keys[1024]) {
    glm::vec2 input(0.0f);
    
    if (keys[GLFW_KEY_W] || keys[GLFW_KEY_UP]) input.y += 1.0f;
    if (keys[GLFW_KEY_S] || keys[GLFW_KEY_DOWN]) input.y -= 1.0f;
    if (keys[GLFW_KEY_A] || keys[GLFW_KEY_LEFT]) input.x -= 1.0f;
    if (keys[GLFW_KEY_D] || keys[GLFW_KEY_RIGHT]) input.x += 1.0f;
    
    if (glm::length(input) > 1.0f) {
        input = glm::normalize(input);
    }
    
    return input;
}

glm::vec2 Player::worldToViewDirection(glm::vec2 input, ViewDirection currentView) {
    switch (currentView) {
        case ViewDirection::FRONT:
            return input;
        case ViewDirection::RIGHT:
            return glm::vec2(-input.y, input.x);
        case ViewDirection::BACK:
            return glm::vec2(-input.x, -input.y);
        case ViewDirection::LEFT:
            return glm::vec2(input.y, -input.x);
        default:
            return input;
    }
}

void Player::buildMomentum(glm::vec2 inputDirection, float deltaTime) {
    float acceleration = m_isGrounded ? m_acceleration : m_airAcceleration;
    
    glm::vec3 desiredVel = glm::vec3(inputDirection.x, 0, inputDirection.y) * m_maxSpeed;
    glm::vec3 currentHorizontalVel = glm::vec3(m_velocity.x, 0, m_velocity.z);
    
    glm::vec3 velocityDiff = desiredVel - currentHorizontalVel;
    float diffLength = glm::length(velocityDiff);
    
    if (diffLength > 0.0f) {
        glm::vec3 accelerationVector = glm::normalize(velocityDiff) * acceleration * deltaTime;
        
        if (glm::length(accelerationVector) > diffLength) {
            accelerationVector = velocityDiff;
        }
        
        m_velocity.x += accelerationVector.x;
        m_velocity.z += accelerationVector.z;
    }
    
    glm::vec3 newHorizontalVel = glm::vec3(m_velocity.x, 0, m_velocity.z);
    if (glm::length(newHorizontalVel) > m_maxSpeed) {
        newHorizontalVel = glm::normalize(newHorizontalVel) * m_maxSpeed;
        m_velocity.x = newHorizontalVel.x;
        m_velocity.z = newHorizontalVel.z;
    }
}
