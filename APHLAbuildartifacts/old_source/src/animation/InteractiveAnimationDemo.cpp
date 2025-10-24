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
    bool onGround;
    
    Player() : position(0.0f, 0.0f, 0.0f), velocity(0.0f), size(0.8f, 1.8f, 0.8f), 
               speed(5.0f), onGround(true) {}
};

// Global variables
Player player;
std::vector<GameObject> obstacles;
std::vector<GameObject> collectibles;
GLFWwindow* window;
int windowWidth = 1200, windowHeight = 800;
bool keys[1024] = {false};
float deltaTime = 0.0f;
float lastFrame = 0.0f;
int frameCount = 0;
float fpsTimer = 0.0f;
float currentFPS = 0.0f;

// Camera
glm::vec3 cameraPos = glm::vec3(0.0f, 5.0f, 10.0f);
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

void processInput() {
    // Reset velocity
    player.velocity = glm::vec3(0.0f);
    
    // Movement input
    glm::vec3 inputDir(0.0f);
    
    if (keys[GLFW_KEY_W] || keys[GLFW_KEY_UP]) inputDir.z -= 1.0f;
    if (keys[GLFW_KEY_S] || keys[GLFW_KEY_DOWN]) inputDir.z += 1.0f;
    if (keys[GLFW_KEY_A] || keys[GLFW_KEY_LEFT]) inputDir.x -= 1.0f;
    if (keys[GLFW_KEY_D] || keys[GLFW_KEY_RIGHT]) inputDir.x += 1.0f;
    
    // Normalize diagonal movement
    if (glm::length(inputDir) > 0.0f) {
        inputDir = glm::normalize(inputDir);
    }
    
    // Speed modifiers
    float currentSpeed = player.speed;
    bool isRunning = keys[GLFW_KEY_LEFT_SHIFT];
    bool isSprinting = keys[GLFW_KEY_LEFT_CONTROL];
    
    if (isSprinting) {
        currentSpeed *= 2.5f;
    } else if (isRunning) {
        currentSpeed *= 1.5f;
    }
    
    player.velocity = inputDir * currentSpeed;
    
    // Update animation based on movement
    float velocityMagnitude = glm::length(player.velocity);
    
    if (velocityMagnitude < 0.1f) {
        player.animController->setState("idle");
    } else if (velocityMagnitude < player.speed * 1.2f) {
        player.animController->setState("walk");
    } else if (velocityMagnitude < player.speed * 2.0f) {
        player.animController->setState("run");
    } else {
        player.animController->setState("sprint");
    }
}

void updatePlayer() {
    // Apply gravity (simple)
    if (!player.onGround) {
        player.velocity.y -= 9.8f * deltaTime;
    }
    
    // Ground collision
    if (player.position.y <= 0.0f) {
        player.position.y = 0.0f;
        player.velocity.y = 0.0f;
        player.onGround = true;
    }
    
    // Jump
    if (keys[GLFW_KEY_SPACE] && player.onGround) {
        player.velocity.y = 8.0f;
        player.onGround = false;
        player.animController->setState("jump");
    }
    
    resolveCollisions();
    player.animController->update(deltaTime);
}

void updateCamera() {
    // Follow player with smooth camera
    glm::vec3 targetPos = player.position + glm::vec3(0.0f, 3.0f, 8.0f);
    cameraPos = glm::mix(cameraPos, targetPos, deltaTime * 2.0f);
    cameraTarget = glm::mix(cameraTarget, player.position + glm::vec3(0.0f, 1.0f, 0.0f), deltaTime * 3.0f);
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
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)windowWidth / windowHeight, 0.1f, 100.0f);
    
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    glBindVertexArray(cubeVAO);
    
    // Render ground
    renderCube(glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(50.0f, 1.0f, 50.0f), glm::vec3(0.2f, 0.6f, 0.2f));
    
    // Render obstacles
    for (const auto& obstacle : obstacles) {
        renderCube(obstacle.position, obstacle.size, obstacle.color);
    }
    
    // Render collectibles with pulsing effect
    float pulseScale = 1.0f + 0.2f * sin(glfwGetTime() * 4.0f);
    for (const auto& collectible : collectibles) {
        renderCube(collectible.position, collectible.size * pulseScale, collectible.color, 0.8f);
    }
    
    // Render animated player
    auto bodyParts = player.animController->getBodyParts();
    
    // Head (Red)
    renderCube(glm::vec3(player.position.x + bodyParts.head.x, 
                        player.position.y + bodyParts.head.y, 
                        player.position.z + bodyParts.head.z), 
               glm::vec3(0.4f, 0.4f, 0.4f), glm::vec3(1.0f, 0.3f, 0.3f));
    
    // Torso (Blue)
    renderCube(glm::vec3(player.position.x + bodyParts.torso.x, 
                        player.position.y + bodyParts.torso.y, 
                        player.position.z + bodyParts.torso.z), 
               glm::vec3(0.6f, 0.8f, 0.3f), glm::vec3(0.3f, 0.3f, 1.0f));
    
    // Arms (Green)
    renderCube(glm::vec3(player.position.x + bodyParts.leftArm.x, 
                        player.position.y + bodyParts.leftArm.y, 
                        player.position.z + bodyParts.leftArm.z), 
               glm::vec3(0.2f, 0.6f, 0.2f), glm::vec3(0.3f, 1.0f, 0.3f));
    
    renderCube(glm::vec3(player.position.x + bodyParts.rightArm.x, 
                        player.position.y + bodyParts.rightArm.y, 
                        player.position.z + bodyParts.rightArm.z), 
               glm::vec3(0.2f, 0.6f, 0.2f), glm::vec3(0.3f, 1.0f, 0.3f));
    
    // Legs (Yellow)
    renderCube(glm::vec3(player.position.x + bodyParts.leftLeg.x, 
                        player.position.y + bodyParts.leftLeg.y, 
                        player.position.z + bodyParts.leftLeg.z), 
               glm::vec3(0.25f, 0.8f, 0.25f), glm::vec3(1.0f, 1.0f, 0.3f));
    
    renderCube(glm::vec3(player.position.x + bodyParts.rightLeg.x, 
                        player.position.y + bodyParts.rightLeg.y, 
                        player.position.z + bodyParts.rightLeg.z), 
               glm::vec3(0.25f, 0.8f, 0.25f), glm::vec3(1.0f, 1.0f, 0.3f));
    
    // Energy bar
    float energy = player.animController->getEnergyLevel();
    renderCube(glm::vec3(player.position.x, player.position.y + 2.2f, player.position.z), 
               glm::vec3(energy * 2.0f, 0.1f, 0.1f), glm::vec3(1.0f, 1.0f, 1.0f));
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        keys[key] = true;
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
        std::cout << "🔄 Player position reset!\n";
    }
}

void setupWorld() {
    // Create obstacles
    obstacles.clear();
    
    // Walls around the area
    obstacles.emplace_back(glm::vec3(0.0f, 1.0f, -20.0f), glm::vec3(40.0f, 2.0f, 2.0f), glm::vec3(0.6f, 0.3f, 0.3f));
    obstacles.emplace_back(glm::vec3(0.0f, 1.0f, 20.0f), glm::vec3(40.0f, 2.0f, 2.0f), glm::vec3(0.6f, 0.3f, 0.3f));
    obstacles.emplace_back(glm::vec3(-20.0f, 1.0f, 0.0f), glm::vec3(2.0f, 2.0f, 40.0f), glm::vec3(0.6f, 0.3f, 0.3f));
    obstacles.emplace_back(glm::vec3(20.0f, 1.0f, 0.0f), glm::vec3(2.0f, 2.0f, 40.0f), glm::vec3(0.6f, 0.3f, 0.3f));
    
    // Interior obstacles
    obstacles.emplace_back(glm::vec3(5.0f, 1.0f, 5.0f), glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.8f, 0.4f, 0.2f));
    obstacles.emplace_back(glm::vec3(-7.0f, 1.0f, -3.0f), glm::vec3(3.0f, 2.0f, 1.0f), glm::vec3(0.4f, 0.8f, 0.2f));
    obstacles.emplace_back(glm::vec3(8.0f, 1.0f, -8.0f), glm::vec3(1.5f, 3.0f, 1.5f), glm::vec3(0.2f, 0.4f, 0.8f));
    obstacles.emplace_back(glm::vec3(-5.0f, 1.0f, 10.0f), glm::vec3(4.0f, 1.0f, 2.0f), glm::vec3(0.6f, 0.6f, 0.2f));
    
    // Create collectibles
    collectibles.clear();
    collectibles.emplace_back(glm::vec3(3.0f, 1.0f, -5.0f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1.0f, 1.0f, 0.0f), false);
    collectibles.emplace_back(glm::vec3(-10.0f, 1.0f, 7.0f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1.0f, 0.0f, 1.0f), false);
    collectibles.emplace_back(glm::vec3(12.0f, 1.0f, -12.0f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0.0f, 1.0f, 1.0f), false);
    collectibles.emplace_back(glm::vec3(-8.0f, 1.0f, -15.0f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1.0f, 0.5f, 0.0f), false);
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
    
    window = glfwCreateWindow(windowWidth, windowHeight, "Interactive Animation Demo - 60 FPS", nullptr, nullptr);
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
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    
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
    std::cout << "🎮 Interactive Animation Demo\n";
    std::cout << "=============================\n";
    std::cout << "Controls:\n";
    std::cout << "  WASD/Arrow Keys - Move\n";
    std::cout << "  Shift - Run\n";
    std::cout << "  Ctrl - Sprint\n";
    std::cout << "  Space - Jump\n";
    std::cout << "  R - Reset position\n";
    std::cout << "  ESC - Exit\n";
    std::cout << "=============================\n\n";
    
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
            std::string title = "Interactive Animation Demo - FPS: " + std::to_string((int)currentFPS);
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
    
    std::cout << "\n🎯 Demo completed successfully!\n";
    return 0;
}
