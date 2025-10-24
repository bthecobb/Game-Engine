#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <unordered_set>
#include <algorithm>

// OpenGL includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Animation system
#include "SimpleAnimationController.h"

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Combat system
enum class CombatState {
    NONE,
    PUNCH,
    KICK,
    COMBO_1,
    COMBO_2,
    COMBO_3
};

// Game objects
struct GameObject {
    glm::vec3 position;
    glm::vec3 size;
    glm::vec3 color;
    bool isCollidable;
    
    GameObject(glm::vec3 pos, glm::vec3 sz, glm::vec3 col, bool collides = true)
        : position(pos), size(sz), color(col), isCollidable(collides) {}
};

struct Player {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    SimpleAnimationController* animController;
    float speed;
    float jumpForce;
    bool onGround;
    bool canDoubleJump;
    bool hasDoubleJumped;
    
    // Combat system
    CombatState combatState;
    float combatTimer;
    float comboWindow;
    int comboCount;
    
    Player() : position(0.0f, 0.0f, 0.0f), velocity(0.0f), size(0.8f, 1.8f, 0.8f), 
               speed(8.0f), jumpForce(12.0f), onGround(true), canDoubleJump(true), hasDoubleJumped(false),
               combatState(CombatState::NONE), combatTimer(0.0f), comboWindow(0.8f), comboCount(0) {}
};

// Global variables
Player player;
std::vector<GameObject> obstacles;
std::vector<GameObject> collectibles;
GLFWwindow* window;
int windowWidth = 1400, windowHeight = 900;
bool keys[1024] = {false};
bool keysPressed[1024] = {false}; // For single key press detection
float deltaTime = 0.0f;
float lastFrame = 0.0f;
int frameCount = 0;
float fpsTimer = 0.0f;
float currentFPS = 0.0f;

// Camera - pulled back and higher
glm::vec3 cameraPos = glm::vec3(0.0f, 8.0f, 15.0f);
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

// OpenGL objects
GLuint shaderProgram;
GLuint cubeVAO, cubeVBO;

// Vertex shader source
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

// Fragment shader source
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

uniform vec3 color;
uniform float alpha;

void main() {
    FragColor = vec4(color, alpha);
}
)";

// Cube vertices
float cubeVertices[] = {
    // Front face
    -0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,
    // Back face
    -0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    // Left face
    -0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    // Right face
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    // Bottom face
    -0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,
    // Top face
    -0.5f,  0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f, -0.5f
};

// Function prototypes
bool initOpenGL();
void setupShaders();
void setupGeometry();
void processInput();
void updatePlayer();
void updateCamera();
void updateCombat();
bool checkCollision(const glm::vec3& pos1, const glm::vec3& size1, const glm::vec3& pos2, const glm::vec3& size2);
void resolveCollisions();
void renderCube(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color, float alpha = 1.0f);
void renderScene();
void setupWorld();
void cleanup();
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

// Collision detection
bool checkCollision(const glm::vec3& pos1, const glm::vec3& size1, const glm::vec3& pos2, const glm::vec3& size2) {
    return (pos1.x - size1.x/2 < pos2.x + size2.x/2 && pos1.x + size1.x/2 > pos2.x - size2.x/2 &&
            pos1.y - size1.y/2 < pos2.y + size2.y/2 && pos1.y + size1.y/2 > pos2.y - size2.y/2 &&
            pos1.z - size1.z/2 < pos2.z + size2.z/2 && pos1.z + size1.z/2 > pos2.z - size2.z/2);
}

void resolveCollisions() {
    glm::vec3 newPos = player.position + player.velocity * deltaTime;
    
    // Check collisions with obstacles
    for (const auto& obstacle : obstacles) {
        if (obstacle.isCollidable && checkCollision(newPos, player.size, obstacle.position, obstacle.size)) {
            // Simple collision resolution - stop movement in collision direction
            glm::vec3 overlap = glm::abs(newPos - obstacle.position) - (player.size + obstacle.size) * 0.5f;
            
            if (overlap.x > overlap.z) {
                player.velocity.z = 0.0f;
                if (newPos.z > obstacle.position.z) {
                    newPos.z = obstacle.position.z + obstacle.size.z/2 + player.size.z/2;
                } else {
                    newPos.z = obstacle.position.z - obstacle.size.z/2 - player.size.z/2;
                }
            } else {
                player.velocity.x = 0.0f;
                if (newPos.x > obstacle.position.x) {
                    newPos.x = obstacle.position.x + obstacle.size.x/2 + player.size.x/2;
                } else {
                    newPos.x = obstacle.position.x - obstacle.size.x/2 - player.size.x/2;
                }
            }
        }
    }
    
    // Check collectibles
    for (auto it = collectibles.begin(); it != collectibles.end();) {
        if (checkCollision(newPos, player.size, it->position, it->size)) {
            std::cout << "🌟 Collected item at (" << it->position.x << ", " << it->position.z << ")!\n";
            it = collectibles.erase(it);
        } else {
            ++it;
        }
    }
    
    player.position = newPos;
}

void updateCombat() {
    if (player.combatState != CombatState::NONE) {
        player.combatTimer -= deltaTime;
        
        if (player.combatTimer <= 0.0f) {
            player.combatState = CombatState::NONE;
            player.comboCount = 0;
        }
        
        // Update animation controller with combat state
        switch (player.combatState) {
            case CombatState::PUNCH:
                player.animController->setState("punch");
                break;
            case CombatState::KICK:
                player.animController->setState("kick");
                break;
            case CombatState::COMBO_1:
                player.animController->setState("combo1");
                break;
            case CombatState::COMBO_2:
                player.animController->setState("combo2");
                break;
            case CombatState::COMBO_3:
                player.animController->setState("combo3");
                break;
            default:
                break;
        }
    }
}

void processInput() {
    // Reset velocity (but keep Y for gravity)
    float currentY = player.velocity.y;
    player.velocity = glm::vec3(0.0f, currentY, 0.0f);
    
    // Movement input - increased speed
    glm::vec3 inputDir(0.0f);
    
    if (keys[GLFW_KEY_W] || keys[GLFW_KEY_UP]) inputDir.z -= 1.0f;
    if (keys[GLFW_KEY_S] || keys[GLFW_KEY_DOWN]) inputDir.z += 1.0f;
    if (keys[GLFW_KEY_A] || keys[GLFW_KEY_LEFT]) inputDir.x -= 1.0f;
    if (keys[GLFW_KEY_D] || keys[GLFW_KEY_RIGHT]) inputDir.x += 1.0f;
    
    // Normalize diagonal movement
    if (glm::length(inputDir) > 0.0f) {
        inputDir = glm::normalize(inputDir);
    }
    
    // Speed modifiers - increased base speeds
    float currentSpeed = player.speed;
    bool isRunning = keys[GLFW_KEY_LEFT_SHIFT];
    bool isSprinting = keys[GLFW_KEY_LEFT_CONTROL];
    
    if (isSprinting) {
        currentSpeed *= 3.0f; // Increased from 2.5f
    } else if (isRunning) {
        currentSpeed *= 2.0f; // Increased from 1.5f
    }
    
    // Apply horizontal movement
    player.velocity.x = inputDir.x * currentSpeed;
    player.velocity.z = inputDir.z * currentSpeed;
    
    // Jump mechanics - single press detection
    if (keysPressed[GLFW_KEY_SPACE]) {
        if (player.onGround) {
            // First jump
            player.velocity.y = player.jumpForce;
            player.onGround = false;
            player.canDoubleJump = true;
            player.hasDoubleJumped = false;
            player.animController->setState("jump");
            std::cout << "🦘 Jump!\n";
        } else if (player.canDoubleJump && !player.hasDoubleJumped) {
            // Double jump - faster and higher
            player.velocity.y = player.jumpForce * 1.3f;
            player.hasDoubleJumped = true;
            player.canDoubleJump = false;
            player.animController->setState("doublejump");
            std::cout << "🚀 Double Jump!\n";
        }
    }
    
    // Combat moves - single press detection
    if (keysPressed[GLFW_KEY_Q] && player.combatState == CombatState::NONE) {
        player.combatState = CombatState::PUNCH;
        player.combatTimer = 0.5f;
        player.comboCount = 1;
        std::cout << "👊 Punch!\n";
    } else if (keysPressed[GLFW_KEY_E] && player.combatState == CombatState::NONE) {
        player.combatState = CombatState::KICK;
        player.combatTimer = 0.6f;
        player.comboCount = 1;
        std::cout << "🦵 Kick!\n";
    }
    
    // Combo system
    if (keysPressed[GLFW_KEY_Q] && player.combatState != CombatState::NONE && player.combatTimer > 0.2f) {
        if (player.comboCount == 1) {
            player.combatState = CombatState::COMBO_1;
            player.combatTimer = 0.7f;
            player.comboCount = 2;
            std::cout << "💥 Combo 1!\n";
        } else if (player.comboCount == 2) {
            player.combatState = CombatState::COMBO_2;
            player.combatTimer = 0.8f;
            player.comboCount = 3;
            std::cout << "⚡ Combo 2!\n";
        }
    }
    
    if (keysPressed[GLFW_KEY_E] && player.combatState != CombatState::NONE && player.combatTimer > 0.2f) {
        if (player.comboCount >= 2) {
            player.combatState = CombatState::COMBO_3;
            player.combatTimer = 0.9f;
            player.comboCount = 4;
            std::cout << "🌟 Ultimate Combo!\n";
        }
    }
    
    // Update animation based on movement (only if not in combat)
    if (player.combatState == CombatState::NONE) {
        float velocityMagnitude = glm::length(glm::vec2(player.velocity.x, player.velocity.z));
        
        if (!player.onGround) {
            // Keep jump animation
        } else if (velocityMagnitude < 0.1f) {
            player.animController->setState("idle");
        } else if (velocityMagnitude < player.speed * 1.5f) {
            player.animController->setState("walk");
        } else if (velocityMagnitude < player.speed * 2.5f) {
            player.animController->setState("run");
        } else {
            player.animController->setState("sprint");
        }
    }
    
    // Clear single press flags
    for (int i = 0; i < 1024; i++) {
        keysPressed[i] = false;
    }
}

void updatePlayer() {
    // Apply gravity - stronger gravity for more responsive jumps
    if (!player.onGround) {
        player.velocity.y -= 15.0f * deltaTime; // Increased from 9.8f
    }
    
    // Ground collision
    if (player.position.y <= 0.0f) {
        player.position.y = 0.0f;
        player.velocity.y = 0.0f;
        player.onGround = true;
        player.canDoubleJump = true;
        player.hasDoubleJumped = false;
    }
    
    resolveCollisions();
    updateCombat();
    player.animController->update(deltaTime);
}

void updateCamera() {
    // Follow player with smooth camera - pulled back more
    glm::vec3 targetPos = player.position + glm::vec3(0.0f, 8.0f, 15.0f);
    cameraPos = glm::mix(cameraPos, targetPos, deltaTime * 3.0f); // Faster camera follow
    cameraTarget = glm::mix(cameraTarget, player.position + glm::vec3(0.0f, 2.0f, 0.0f), deltaTime * 4.0f);
}

void renderCube(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color, float alpha) {
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = glm::scale(model, size);
    
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, glm::value_ptr(color));
    glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), alpha);
    
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

void renderScene() {
    // Clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Use shader program
    glUseProgram(shaderProgram);
    
    // Set matrices
    glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(50.0f), (float)windowWidth / windowHeight, 0.1f, 200.0f);
    
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    glBindVertexArray(cubeVAO);
    
    // Render ground - expanded
    renderCube(glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(100.0f, 1.0f, 100.0f), glm::vec3(0.2f, 0.6f, 0.2f));
    
    // Render obstacles
    for (const auto& obstacle : obstacles) {
        renderCube(obstacle.position, obstacle.size, obstacle.color);
    }
    
    // Render collectibles with pulsing effect
    float pulseScale = 1.0f + 0.3f * sin(glfwGetTime() * 5.0f);
    for (const auto& collectible : collectibles) {
        renderCube(collectible.position, collectible.size * pulseScale, collectible.color, 0.8f);
    }
    
    // Render animated player with combat effects
    auto bodyParts = player.animController->getBodyParts();
    
    // Combat glow effect
    glm::vec3 combatGlow(1.0f, 1.0f, 1.0f);
    float glowIntensity = 1.0f;
    if (player.combatState != CombatState::NONE) {
        glowIntensity = 1.0f + 0.5f * sin(glfwGetTime() * 20.0f);
        combatGlow = glm::vec3(1.0f, 0.5f, 0.0f); // Orange glow for combat
    }
    
    // Head (Red with combat glow)
    renderCube(glm::vec3(player.position.x + bodyParts.head.x, 
                        player.position.y + bodyParts.head.y, 
                        player.position.z + bodyParts.head.z), 
               glm::vec3(0.4f, 0.4f, 0.4f), glm::vec3(1.0f, 0.3f, 0.3f) * combatGlow * glowIntensity);
    
    // Torso (Blue with combat glow)
    renderCube(glm::vec3(player.position.x + bodyParts.torso.x, 
                        player.position.y + bodyParts.torso.y, 
                        player.position.z + bodyParts.torso.z), 
               glm::vec3(0.6f, 0.8f, 0.3f), glm::vec3(0.3f, 0.3f, 1.0f) * combatGlow * glowIntensity);
    
    // Arms (Green with enhanced combat glow)
    float armGlow = (player.combatState == CombatState::PUNCH) ? glowIntensity * 1.5f : glowIntensity;
    renderCube(glm::vec3(player.position.x + bodyParts.leftArm.x, 
                        player.position.y + bodyParts.leftArm.y, 
                        player.position.z + bodyParts.leftArm.z), 
               glm::vec3(0.2f, 0.6f, 0.2f), glm::vec3(0.3f, 1.0f, 0.3f) * combatGlow * armGlow);
    
    renderCube(glm::vec3(player.position.x + bodyParts.rightArm.x, 
                        player.position.y + bodyParts.rightArm.y, 
                        player.position.z + bodyParts.rightArm.z), 
               glm::vec3(0.2f, 0.6f, 0.2f), glm::vec3(0.3f, 1.0f, 0.3f) * combatGlow * armGlow);
    
    // Legs (Yellow with enhanced kick glow)
    float legGlow = (player.combatState == CombatState::KICK) ? glowIntensity * 1.5f : glowIntensity;
    renderCube(glm::vec3(player.position.x + bodyParts.leftLeg.x, 
                        player.position.y + bodyParts.leftLeg.y, 
                        player.position.z + bodyParts.leftLeg.z), 
               glm::vec3(0.25f, 0.8f, 0.25f), glm::vec3(1.0f, 1.0f, 0.3f) * combatGlow * legGlow);
    
    renderCube(glm::vec3(player.position.x + bodyParts.rightLeg.x, 
                        player.position.y + bodyParts.rightLeg.y, 
                        player.position.z + bodyParts.rightLeg.z), 
               glm::vec3(0.25f, 0.8f, 0.25f), glm::vec3(1.0f, 1.0f, 0.3f) * combatGlow * legGlow);
    
    // Energy bar
    float energy = player.animController->getEnergyLevel();
    renderCube(glm::vec3(player.position.x, player.position.y + 2.5f, player.position.z), 
               glm::vec3(energy * 2.5f, 0.1f, 0.1f), glm::vec3(1.0f, 1.0f, 1.0f));
    
    // Combo indicator
    if (player.comboCount > 0) {
        for (int i = 0; i < player.comboCount; i++) {
            renderCube(glm::vec3(player.position.x - 1.0f + i * 0.3f, player.position.y + 3.0f, player.position.z), 
                      glm::vec3(0.2f, 0.1f, 0.1f), glm::vec3(1.0f, 0.0f, 0.5f), 0.8f);
        }
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        keys[key] = true;
        keysPressed[key] = true;
    } else if (action == GLFW_RELEASE) {
        keys[key] = false;
    }
    
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        // Reset player position
        player.position = glm::vec3(0.0f, 0.0f, 0.0f);
        player.velocity = glm::vec3(0.0f);
        player.combatState = CombatState::NONE;
        player.comboCount = 0;
        player.onGround = true;
        std::cout << "🔄 Player position reset!\n";
    }
}

void setupWorld() {
    // Create obstacles
    obstacles.clear();
    
    // Expanded walls around the area
    obstacles.emplace_back(glm::vec3(0.0f, 2.0f, -40.0f), glm::vec3(80.0f, 4.0f, 4.0f), glm::vec3(0.6f, 0.3f, 0.3f));
    obstacles.emplace_back(glm::vec3(0.0f, 2.0f, 40.0f), glm::vec3(80.0f, 4.0f, 4.0f), glm::vec3(0.6f, 0.3f, 0.3f));
    obstacles.emplace_back(glm::vec3(-40.0f, 2.0f, 0.0f), glm::vec3(4.0f, 4.0f, 80.0f), glm::vec3(0.6f, 0.3f, 0.3f));
    obstacles.emplace_back(glm::vec3(40.0f, 2.0f, 0.0f), glm::vec3(4.0f, 4.0f, 80.0f), glm::vec3(0.6f, 0.3f, 0.3f));
    
    // Interior obstacles - more spread out
    obstacles.emplace_back(glm::vec3(10.0f, 1.5f, 10.0f), glm::vec3(3.0f, 3.0f, 3.0f), glm::vec3(0.8f, 0.4f, 0.2f));
    obstacles.emplace_back(glm::vec3(-15.0f, 1.5f, -8.0f), glm::vec3(4.0f, 3.0f, 2.0f), glm::vec3(0.4f, 0.8f, 0.2f));
    obstacles.emplace_back(glm::vec3(18.0f, 2.0f, -18.0f), glm::vec3(2.0f, 4.0f, 2.0f), glm::vec3(0.2f, 0.4f, 0.8f));
    obstacles.emplace_back(glm::vec3(-12.0f, 1.0f, 20.0f), glm::vec3(6.0f, 2.0f, 3.0f), glm::vec3(0.6f, 0.6f, 0.2f));
    obstacles.emplace_back(glm::vec3(25.0f, 1.5f, 5.0f), glm::vec3(3.0f, 3.0f, 5.0f), glm::vec3(0.7f, 0.2f, 0.7f));
    obstacles.emplace_back(glm::vec3(-25.0f, 2.0f, -25.0f), glm::vec3(4.0f, 4.0f, 4.0f), glm::vec3(0.3f, 0.7f, 0.9f));
    
    // Create collectibles - more spread out
    collectibles.clear();
    collectibles.emplace_back(glm::vec3(8.0f, 1.0f, -12.0f), glm::vec3(0.7f, 0.7f, 0.7f), glm::vec3(1.0f, 1.0f, 0.0f), false);
    collectibles.emplace_back(glm::vec3(-20.0f, 1.0f, 15.0f), glm::vec3(0.7f, 0.7f, 0.7f), glm::vec3(1.0f, 0.0f, 1.0f), false);
    collectibles.emplace_back(glm::vec3(30.0f, 1.0f, -30.0f), glm::vec3(0.7f, 0.7f, 0.7f), glm::vec3(0.0f, 1.0f, 1.0f), false);
    collectibles.emplace_back(glm::vec3(-18.0f, 1.0f, -35.0f), glm::vec3(0.7f, 0.7f, 0.7f), glm::vec3(1.0f, 0.5f, 0.0f), false);
    collectibles.emplace_back(glm::vec3(0.0f, 1.0f, 25.0f), glm::vec3(0.7f, 0.7f, 0.7f), glm::vec3(0.5f, 1.0f, 0.5f), false);
    collectibles.emplace_back(glm::vec3(22.0f, 1.0f, 18.0f), glm::vec3(0.7f, 0.7f, 0.7f), glm::vec3(1.0f, 0.2f, 0.8f), false);
}

bool initOpenGL() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // MSAA
    
    window = glfwCreateWindow(windowWidth, windowHeight, "Enhanced Interactive Demo - Combat & Double Jump", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSwapInterval(1); // VSync for 60 FPS
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return false;
    }
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.05f, 0.05f, 0.15f, 1.0f); // Darker background
    
    return true;
}

void setupShaders() {
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    
    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    // Create shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void setupGeometry() {
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

void cleanup() {
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteBuffers(1, &cubeVBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
}

int main() {
    std::cout << "🎮 Enhanced Interactive Animation Demo\n";
    std::cout << "=====================================\n";
    std::cout << "Movement:\n";
    std::cout << "  WASD/Arrow Keys - Move (faster!)\n";
    std::cout << "  Shift - Run (2x speed)\n";
    std::cout << "  Ctrl - Sprint (3x speed)\n";
    std::cout << "Jump:\n";
    std::cout << "  Space - Jump (press again for double jump!)\n";
    std::cout << "Combat:\n";
    std::cout << "  Q - Punch (start combos)\n";
    std::cout << "  E - Kick (finish combos)\n";
    std::cout << "  Q->Q->Q - Punch combo chain\n";
    std::cout << "  Q->Q->E - Ultimate combo finisher\n";
    std::cout << "Other:\n";
    std::cout << "  R - Reset position\n";
    std::cout << "  ESC - Exit\n";
    std::cout << "=====================================\n\n";
    
    if (!initOpenGL()) {
        return -1;
    }
    
    setupShaders();
    setupGeometry();
    setupWorld();
    
    // Initialize player
    player.animController = new SimpleAnimationController();
    player.position = glm::vec3(0.0f, 0.0f, 0.0f);
    
    auto lastTime = std::chrono::high_resolution_clock::now();
    
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        // Cap delta time to prevent large jumps
        deltaTime = std::min(deltaTime, 1.0f / 30.0f);
        
        // FPS counter
        frameCount++;
        fpsTimer += deltaTime;
        if (fpsTimer >= 1.0f) {
            currentFPS = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0.0f;
            
            // Update window title with FPS
            std::string title = "Enhanced Interactive Demo - FPS: " + std::to_string((int)currentFPS) + 
                              " | Combo: " + std::to_string(player.comboCount);
            glfwSetWindowTitle(window, title.c_str());
        }
        
        glfwPollEvents();
        processInput();
        updatePlayer();
        updateCamera();
        renderScene();
        glfwSwapBuffers(window);
    }
    
    delete player.animController;
    cleanup();
    
    std::cout << "\n🎯 Enhanced demo completed successfully!\n";
    return 0;
}
