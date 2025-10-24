#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>

// OpenGL includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Enhanced systems
#include "Enhanced3DCameraSystem.h"
#include "SimpleAnimationController.h"

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Window dimensions
const int WINDOW_WIDTH = 1920;
const int WINDOW_HEIGHT = 1080;

// Enhanced game states
enum GameState {
    GAME_MENU,
    GAME_PLAYING,
    GAME_PAUSED,
    GAME_LEVEL_COMPLETE,
    GAME_BOSS_FIGHT,
    GAME_GAME_OVER
};

// Combat states remain the same as in original
enum class CombatState {
    NONE, PUNCH, KICK, COMBO_1, COMBO_2, COMBO_3,
    SWORD_SLASH_1, SWORD_SLASH_2, SWORD_THRUST,
    STAFF_SPIN, STAFF_CAST, STAFF_SLAM,
    HAMMER_SWING, HAMMER_OVERHEAD, HAMMER_GROUND_POUND,
    GRAB, GRAB_THROW, PARRY, COUNTER_ATTACK, SLIDE, SLIDE_LEAP
};

enum ComboState {
    COMBO_NONE, COMBO_DASH, COMBO_LIGHT_1, COMBO_LIGHT_2, 
    COMBO_LIGHT_3, COMBO_LAUNCHER, COMBO_AIR, COMBO_WEAPON_DIVE
};

enum class WeaponType {
    NONE, SWORD, STAFF, HAMMER
};

// Enhanced Player struct with grappling hook integration
struct EnhancedPlayer {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    SimpleAnimationController* animController;
    float speed;
    float sprintMultiplier;
    float jumpForce;
    bool onGround;
    bool canDoubleJump;
    bool hasDoubleJumped;
    
    // Wall running
    bool isWallRunning;
    float wallRunTime;
    float maxWallRunTime;
    glm::vec3 wallNormal;
    glm::vec3 wallRunDirection;
    float wallRunSpeed;
    
    // Combat system
    CombatState combatState;
    ComboState comboState;
    float combatTimer;
    float comboWindow;
    int comboCount;
    float attackRange;
    
    // Movement enhancements
    bool sprintToggled;
    bool isDashing;
    float dashTimer;
    float dashSpeed;
    glm::vec3 dashDirection;
    
    // Health and weapons
    float health;
    float maxHealth;
    WeaponType currentWeapon;
    std::vector<WeaponType> inventory;
    
    // Enhanced movement
    bool isSliding;
    bool isGrabbing;
    bool isParrying;
    float parryTimer;
    float parryWindow;
    
    // Grappling hook integration
    GrapplingHook grapplingHook;
    
    // Enhanced tracking distances for moves
    float lightAttackRange;
    float heavyAttackRange;
    float grabRange;
    float dashRange;
    
    EnhancedPlayer() : 
        position(0.0f, 1.0f, 0.0f), velocity(0.0f), size(0.8f, 1.8f, 0.8f),
        speed(18.0f), sprintMultiplier(2.5f), jumpForce(20.0f), onGround(true),
        canDoubleJump(true), hasDoubleJumped(false), isWallRunning(false),
        wallRunTime(0.0f), maxWallRunTime(3.0f), wallRunSpeed(15.0f),
        combatState(CombatState::NONE), comboState(COMBO_NONE), combatTimer(0.0f),
        comboWindow(0.8f), comboCount(0), attackRange(4.0f),
        sprintToggled(false), isDashing(false), dashTimer(0.0f), dashSpeed(35.0f),
        health(100.0f), maxHealth(100.0f), currentWeapon(WeaponType::NONE),
        isSliding(false), isGrabbing(false), isParrying(false), 
        parryTimer(0.0f), parryWindow(0.3f),
        lightAttackRange(3.5f), heavyAttackRange(5.0f), grabRange(3.0f), dashRange(8.0f) {
        
        inventory.push_back(WeaponType::NONE);
        inventory.push_back(WeaponType::SWORD);
        inventory.push_back(WeaponType::STAFF);
        inventory.push_back(WeaponType::HAMMER);
    }
};

// Enhanced Enemy with better AI
struct EnhancedEnemy {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    float speed;
    float detectionRange;
    float attackRange;
    bool isAggressive;
    float health;
    float maxHealth;
    float damage;
    float attackCooldown;
    bool isDead;
    
    // Enhanced AI
    enum class AIState {
        PATROL, CHASE, ATTACK, STUNNED, RETREATING
    };
    AIState aiState;
    glm::vec3 lastKnownPlayerPos;
    float alertLevel;
    float retreatTimer;
    
    // Vision cone for better detection
    float visionAngle;  // Degrees
    glm::vec3 facingDirection;
    
    EnhancedEnemy(glm::vec3 pos) : 
        position(pos), velocity(0.0f), size(1.0f, 2.0f, 1.0f),
        speed(10.0f), detectionRange(25.0f), attackRange(3.5f),
        isAggressive(false), health(75.0f), maxHealth(75.0f),
        damage(15.0f), attackCooldown(0.0f), isDead(false),
        aiState(AIState::PATROL), alertLevel(0.0f), retreatTimer(0.0f),
        visionAngle(60.0f), facingDirection(1.0f, 0.0f, 0.0f) {}
    
    bool canSeePlayer(const glm::vec3& playerPos) {
        glm::vec3 toPlayer = playerPos - position;
        float distance = glm::length(toPlayer);
        
        if (distance > detectionRange) return false;
        
        // Check if player is within vision cone
        glm::vec3 toPlayerNorm = glm::normalize(toPlayer);
        float dotProduct = glm::dot(facingDirection, toPlayerNorm);
        float angleToPlayer = glm::degrees(acos(dotProduct));
        
        return angleToPlayer <= visionAngle * 0.5f;
    }
    
    void updateAI(float deltaTime, const glm::vec3& playerPos) {
        // Basic AI state machine
        switch (aiState) {
            case AIState::PATROL:
                if (canSeePlayer(playerPos)) {
                    aiState = AIState::CHASE;
                    lastKnownPlayerPos = playerPos;
                    alertLevel = 1.0f;
                }
                break;
                
            case AIState::CHASE:
                lastKnownPlayerPos = playerPos;
                if (glm::length(playerPos - position) <= attackRange) {
                    aiState = AIState::ATTACK;
                } else if (glm::length(playerPos - position) > detectionRange * 1.5f) {
                    aiState = AIState::PATROL;
                    alertLevel = 0.0f;
                }
                break;
                
            case AIState::ATTACK:
                if (glm::length(playerPos - position) > attackRange * 1.2f) {
                    aiState = AIState::CHASE;
                }
                break;
                
            case AIState::STUNNED:
                // Handle stun recovery
                break;
                
            case AIState::RETREATING:
                retreatTimer -= deltaTime;
                if (retreatTimer <= 0.0f) {
                    aiState = AIState::PATROL;
                }
                break;
        }
    }
};

// Global game objects
GameState gameState = GAME_MENU;
EnhancedPlayer player;
Enhanced3DCamera camera;
EnhancedControllerInput controller;
LevelProgression levelProgression;
std::vector<EnhancedEnemy> enemies;
std::vector<glm::vec3> hookPickups;

// Timing
auto lastTime = std::chrono::high_resolution_clock::now();
float deltaTime = 0.0f;

// OpenGL objects
GLuint shaderProgram;
GLuint VAO, VBO;

// Shader source code
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 color;

out vec3 fragColor;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    fragColor = color;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(fragColor, 1.0);
}
)";

// Function declarations
void initOpenGL();
void setupShaders();
void processInput(GLFWwindow* window);
void updateGame(float dt);
void renderGame();
void renderUI();
void updatePlayer(float dt);
void updateEnemies(float dt);
void updateCamera(float dt);
void handleCollisions();
void renderCube(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color);
void renderHookPickups();
void renderGrapplingHook();
void spawnEnemiesForSegment();

// Shader compilation
GLuint compileShader(const char* source, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "Shader compilation error: " << infoLog << std::endl;
    }
    
    return shader;
}

void setupShaders() {
    GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
    
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "Shader program linking error: " << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void initOpenGL() {
    // Cube vertices
    float vertices[] = {
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
    
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    setupShaders();
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
}

void processInput(GLFWwindow* window) {
    // Update controller input
    controller.update(window);
    
    // Handle camera control with right stick
    camera.processController(controller.current.rightStickX, 
                           controller.current.rightStickY, deltaTime);
    
    // Handle keyboard camera control
    camera.processKeyboard(window, deltaTime);
    
    // Player movement with left stick
    glm::vec3 moveDirection = glm::vec3(0.0f);
    
    if (std::abs(controller.current.leftStickX) > 0.1f || 
        std::abs(controller.current.leftStickY) > 0.1f) {
        moveDirection.x = controller.current.leftStickX;
        moveDirection.z = -controller.current.leftStickY; // Inverted for forward movement
    }
    
    // Keyboard movement (fallback)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) moveDirection.z = -1.0f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) moveDirection.z = 1.0f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) moveDirection.x = -1.0f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) moveDirection.x = 1.0f;
    
    // Apply movement
    if (glm::length(moveDirection) > 0.1f) {
        moveDirection = glm::normalize(moveDirection);
        float currentSpeed = player.speed;
        
        // Sprint with controller left stick click or keyboard shift
        if (controller.current.leftStick || glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            currentSpeed *= player.sprintMultiplier;
        }
        
        player.velocity.x = moveDirection.x * currentSpeed;
        player.velocity.z = moveDirection.z * currentSpeed;
    } else {
        // Apply friction when not moving
        player.velocity.x *= 0.8f;
        player.velocity.z *= 0.8f;
    }
    
    // Jump with controller A button or spacebar
    if ((controller.buttonAJustPressed() || glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) 
        && (player.onGround || (!player.hasDoubleJumped && player.canDoubleJump))) {
        
        if (player.onGround) {
            player.velocity.y = player.jumpForce;
            player.onGround = false;
        } else if (!player.hasDoubleJumped) {
            player.velocity.y = player.jumpForce * 0.8f;
            player.hasDoubleJumped = true;
        }
    }
    
    // Dash with controller B button or C key
    if ((controller.buttonBJustPressed() || glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) 
        && !player.isDashing) {
        
        player.isDashing = true;
        player.dashTimer = 0.3f;
        
        if (glm::length(moveDirection) > 0.1f) {
            player.dashDirection = moveDirection;
        } else {
            player.dashDirection = glm::vec3(0.0f, 0.0f, -1.0f); // Default forward
        }
        
        // Camera shake on dash
        camera.startShake(2.0f, 0.2f);
    }
    
    // Grappling hook with controller Y button or G key
    if ((controller.buttonYJustPressed() || glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) 
        && player.grapplingHook.hookCharges > 0) {
        
        // Launch hook in camera forward direction
        glm::vec3 hookDirection = camera.front;
        player.grapplingHook.launch(player.position, hookDirection);
        
        std::cout << "Grappling hook launched! Charges remaining: " 
                  << player.grapplingHook.hookCharges << std::endl;
    }
    
    // Combat with controller X button or mouse
    if (controller.buttonXJustPressed() || glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        if (player.combatState == CombatState::NONE) {
            player.combatState = CombatState::PUNCH;
            player.combatTimer = 0.5f;
            
            std::cout << "Light attack!" << std::endl;
        }
    }
    
    // Heavy attack with right trigger or right mouse button
    if (controller.current.rightTrigger > 0.5f || glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        if (player.combatState == CombatState::NONE) {
            player.combatState = CombatState::KICK;
            player.combatTimer = 0.8f;
            
            std::cout << "Heavy attack!" << std::endl;
        }
    }
    
    // Parry with controller left bumper or Q key
    if ((controller.leftBumperJustPressed() || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) 
        && !player.isParrying) {
        
        player.isParrying = true;
        player.parryTimer = player.parryWindow;
        
        std::cout << "Parry activated!" << std::endl;
    }
    
    // Camera mode switching
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        camera.setCameraMode(Enhanced3DCamera::CameraMode::PLAYER_FOLLOW);
    } else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        camera.setCameraMode(Enhanced3DCamera::CameraMode::FREE_LOOK);
    } else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        camera.setCameraMode(Enhanced3DCamera::CameraMode::COMBAT_FOCUS);
    }
    
    // Game state changes
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        gameState = (gameState == GAME_PAUSED) ? GAME_PLAYING : GAME_PAUSED;
    }
}

void updateGame(float dt) {
    deltaTime = dt;
    
    if (gameState != GAME_PLAYING) return;
    
    updatePlayer(dt);
    updateEnemies(dt);
    updateCamera(dt);
    handleCollisions();
    
    // Update level progression
    levelProgression.update(player.position);
    
    // Check for level completion
    if (levelProgression.levelCompleted) {
        gameState = GAME_LEVEL_COMPLETE;
        std::cout << "Level completed!" << std::endl;
    }
    
    // Spawn enemies for current segment
    spawnEnemiesForSegment();
}

void updatePlayer(float dt) {
    // Update grappling hook
    player.grapplingHook.update(dt, player.position, player.velocity);
    
    // Update combat timers
    if (player.combatTimer > 0.0f) {
        player.combatTimer -= dt;
        if (player.combatTimer <= 0.0f) {
            player.combatState = CombatState::NONE;
        }
    }
    
    // Update parry timer
    if (player.parryTimer > 0.0f) {
        player.parryTimer -= dt;
        if (player.parryTimer <= 0.0f) {
            player.isParrying = false;
        }
    }
    
    // Update dash
    if (player.isDashing) {
        player.dashTimer -= dt;
        if (player.dashTimer > 0.0f) {
            player.velocity.x = player.dashDirection.x * player.dashSpeed;
            player.velocity.z = player.dashDirection.z * player.dashSpeed;
        } else {
            player.isDashing = false;
        }
    }
    
    // Apply gravity
    if (!player.onGround && !player.grapplingHook.isSwinging) {
        player.velocity.y -= 35.0f * dt; // Gravity
    }
    
    // Update position
    player.position += player.velocity * dt;
    
    // Ground collision
    if (player.position.y <= 1.0f) {
        player.position.y = 1.0f;
        player.velocity.y = 0.0f;
        player.onGround = true;
        player.hasDoubleJumped = false;
    } else {
        player.onGround = false;
    }
    
    // Check for hook pickups
    for (auto it = hookPickups.begin(); it != hookPickups.end(); ) {
        if (glm::length(player.position - *it) < 2.0f) {
            player.grapplingHook.addCharge();
            std::cout << "Grappling hook pickup! Charges: " 
                      << player.grapplingHook.hookCharges << std::endl;
            it = hookPickups.erase(it);
        } else {
            ++it;
        }
    }
}

void updateEnemies(float dt) {
    for (auto& enemy : enemies) {
        if (enemy.isDead) continue;
        
        enemy.updateAI(dt, player.position);
        
        // Move enemy based on AI state
        switch (enemy.aiState) {
            case EnhancedEnemy::AIState::CHASE:
                {
                    glm::vec3 toPlayer = glm::normalize(player.position - enemy.position);
                    enemy.position += toPlayer * enemy.speed * dt;
                    enemy.facingDirection = toPlayer;
                }
                break;
                
            case EnhancedEnemy::AIState::ATTACK:
                // Handle attack logic
                if (enemy.attackCooldown <= 0.0f) {
                    // Attack player if in range
                    if (glm::length(player.position - enemy.position) <= enemy.attackRange) {
                        // Damage player (simplified)
                        player.health -= enemy.damage * dt;
                        enemy.attackCooldown = 2.0f;
                        
                        // Camera shake on player hit
                        camera.startShake(3.0f, 0.3f);
                    }
                }
                break;
                
            case EnhancedEnemy::AIState::RETREATING:
                {
                    glm::vec3 awayFromPlayer = glm::normalize(enemy.position - player.position);
                    enemy.position += awayFromPlayer * enemy.speed * 0.5f * dt;
                }
                break;
        }
        
        if (enemy.attackCooldown > 0.0f) {
            enemy.attackCooldown -= dt;
        }
    }
}

void updateCamera(float dt) {
    camera.update(dt, player.position, player.velocity, player.isWallRunning);
}

void handleCollisions() {
    // Check player-enemy combat collisions
    if (player.combatState != CombatState::NONE) {
        for (auto& enemy : enemies) {
            if (enemy.isDead) continue;
            
            float distance = glm::length(player.position - enemy.position);
            float attackRange = (player.combatState == CombatState::KICK) ? 
                               player.heavyAttackRange : player.lightAttackRange;
            
            if (distance <= attackRange) {
                // Deal damage to enemy
                float damage = (player.combatState == CombatState::KICK) ? 30.0f : 20.0f;
                enemy.health -= damage;
                
                if (enemy.health <= 0.0f) {
                    enemy.isDead = true;
                    std::cout << "Enemy defeated!" << std::endl;
                }
                
                // Camera shake on successful hit
                camera.startShake(1.5f, 0.1f);
            }
        }
    }
}

void spawnEnemiesForSegment() {
    auto* currentSegment = levelProgression.getCurrentSegment();
    if (!currentSegment) return;
    
    // Spawn enemies based on current segment
    static bool segmentEnemiesSpawned = false;
    if (!segmentEnemiesSpawned) {
        for (const auto& spawnPos : currentSegment->enemySpawns) {
            enemies.emplace_back(spawnPos);
        }
        
        // Add hook pickups
        for (const auto& pickupPos : currentSegment->hookPickups) {
            hookPickups.push_back(pickupPos);
        }
        
        segmentEnemiesSpawned = true;
    }
    
    // Reset for next segment
    if (currentSegment->isCompleted) {
        segmentEnemiesSpawned = false;
    }
}

void renderCube(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color) {
    glUseProgram(shaderProgram);
    
    // Create model matrix
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = glm::scale(model, size);
    
    // Set uniforms
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(camera.getViewMatrix()));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(camera.getProjectionMatrix((float)WINDOW_WIDTH / WINDOW_HEIGHT)));
    glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, glm::value_ptr(color));
    
    // Render cube
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

void renderHookPickups() {
    for (const auto& pickup : hookPickups) {
        renderCube(pickup, glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1.0f, 1.0f, 0.0f)); // Yellow
    }
}

void renderGrapplingHook() {
    if (player.grapplingHook.isActive) {
        // Render hook (simplified as a small cube)
        renderCube(player.grapplingHook.hookPosition, 
                  glm::vec3(0.2f, 0.2f, 0.2f), 
                  glm::vec3(0.8f, 0.8f, 0.8f)); // Gray
        
        // Render chain (simplified - could be improved with line rendering)
        glm::vec3 chainStart = player.position + glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 chainEnd = player.grapplingHook.hookPosition;
        glm::vec3 chainDir = chainEnd - chainStart;
        float chainLength = glm::length(chainDir);
        
        // Render chain segments
        int segments = static_cast<int>(chainLength / 2.0f);
        for (int i = 0; i < segments; i++) {
            float t = static_cast<float>(i) / segments;
            glm::vec3 segmentPos = chainStart + chainDir * t;
            renderCube(segmentPos, glm::vec3(0.1f, 0.1f, 0.1f), glm::vec3(0.6f, 0.6f, 0.6f));
        }
    }
}

void renderGame() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    
    // Render ground
    renderCube(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(200.0f, 1.0f, 200.0f), glm::vec3(0.3f, 0.3f, 0.3f));
    
    // Render player
    glm::vec3 playerColor = player.isParrying ? glm::vec3(0.0f, 1.0f, 1.0f) : glm::vec3(1.0f, 0.5f, 0.0f);
    renderCube(player.position, player.size, playerColor);
    
    // Render enemies
    for (const auto& enemy : enemies) {
        if (enemy.isDead) continue;
        
        glm::vec3 enemyColor = glm::vec3(1.0f, 0.0f, 0.0f);
        if (enemy.aiState == EnhancedEnemy::AIState::CHASE) {
            enemyColor = glm::vec3(1.0f, 0.5f, 0.0f); // Orange when chasing
        } else if (enemy.aiState == EnhancedEnemy::AIState::ATTACK) {
            enemyColor = glm::vec3(1.0f, 0.0f, 0.5f); // Pink when attacking
        }
        
        renderCube(enemy.position, enemy.size, enemyColor);
    }
    
    // Render hook pickups
    renderHookPickups();
    
    // Render grappling hook
    renderGrapplingHook();
    
    // Render level progression markers
    auto* currentSegment = levelProgression.getCurrentSegment();
    if (currentSegment) {
        // Render start and end markers
        renderCube(currentSegment->startPosition + glm::vec3(0.0f, 2.0f, 0.0f), 
                  glm::vec3(1.0f, 4.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f)); // Green start marker
        renderCube(currentSegment->endPosition + glm::vec3(0.0f, 2.0f, 0.0f), 
                  glm::vec3(1.0f, 4.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // Red end marker
    }
}

void renderUI() {
    // This would typically use a UI library like Dear ImGui
    // For now, we'll output to console
    
    static float uiUpdateTimer = 0.0f;
    uiUpdateTimer += deltaTime;
    
    if (uiUpdateTimer > 1.0f) { // Update UI every second
        system("cls"); // Clear console (Windows)
        
        std::cout << "=== Enhanced Rhythm Arena 3D ===" << std::endl;
        std::cout << "Health: " << player.health << "/" << player.maxHealth << std::endl;
        std::cout << "Grappling Hook Charges: " << player.grapplingHook.hookCharges << "/" << player.grapplingHook.maxCharges << std::endl;
        std::cout << "Position: (" << player.position.x << ", " << player.position.y << ", " << player.position.z << ")" << std::endl;
        
        auto* segment = levelProgression.getCurrentSegment();
        if (segment) {
            std::cout << "Level Progress: " << (int)(segment->progressPercentage * 100) << "%" << std::endl;
            std::cout << "Current Segment: " << levelProgression.currentSegment + 1 << "/" << levelProgression.segments.size() << std::endl;
        }
        
        std::cout << "\nControls:" << std::endl;
        std::cout << "Left Stick/WASD: Move" << std::endl;
        std::cout << "Right Stick: Camera" << std::endl;
        std::cout << "A/Space: Jump" << std::endl;
        std::cout << "B/C: Dash" << std::endl;
        std::cout << "X/LMB: Light Attack" << std::endl;
        std::cout << "RT/RMB: Heavy Attack" << std::endl;
        std::cout << "LB/Q: Parry" << std::endl;
        std::cout << "Y/G: Grappling Hook" << std::endl;
        std::cout << "Left Stick Click/Shift: Sprint" << std::endl;
        std::cout << "1/2/3: Camera Modes" << std::endl;
        std::cout << "ESC: Pause" << std::endl;
        
        uiUpdateTimer = 0.0f;
    }
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Create window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Enhanced Rhythm Arena 3D", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    
    // Initialize OpenGL
    initOpenGL();
    
    // Initialize game state
    gameState = GAME_PLAYING;
    
    // Game loop
    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        // Limit delta time to prevent large jumps
        deltaTime = std::min(deltaTime, 0.016f);
        
        processInput(window);
        updateGame(deltaTime);
        renderGame();
        renderUI();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    
    glfwTerminate();
    return 0;
}
