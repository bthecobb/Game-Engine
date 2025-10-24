#include <iostream>
#include <chrono>

// OpenGL includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Enhanced systems
#include "Enhanced3DCameraSystem.h"

// Window dimensions
const int WINDOW_WIDTH = 1920;
const int WINDOW_HEIGHT = 1080;

// Global objects
Enhanced3DCamera camera;
EnhancedControllerInput controller;
glm::vec3 playerPos(0.0f, 1.0f, 0.0f);
glm::vec3 playerVel(0.0f);

// Timing
auto lastTime = std::chrono::high_resolution_clock::now();
float deltaTime = 0.0f;

// OpenGL objects
GLuint shaderProgram;
GLuint VAO, VBO;

// Simple shader
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
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void initOpenGL() {
    // Cube vertices
    float vertices[] = {
        // Front face
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,
        // Back face
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,
         0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
        // Additional faces...
        -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f
    };
    
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    setupShaders();
    glEnable(GL_DEPTH_TEST);
}

void renderCube(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color) {
    glUseProgram(shaderProgram);
    
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = glm::scale(model, size);
    
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(camera.getViewMatrix()));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(camera.getProjectionMatrix((float)WINDOW_WIDTH / WINDOW_HEIGHT)));
    glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, glm::value_ptr(color));
    
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

void processInput(GLFWwindow* window) {
    controller.update(window);
    
    // Handle camera control
    camera.processController(controller.current.rightStickX, 
                           controller.current.rightStickY, deltaTime);
    camera.processKeyboard(window, deltaTime);
    
    // Player movement
    glm::vec3 moveDirection = glm::vec3(0.0f);
    
    if (std::abs(controller.current.leftStickX) > 0.1f || 
        std::abs(controller.current.leftStickY) > 0.1f) {
        moveDirection.x = controller.current.leftStickX;
        moveDirection.z = -controller.current.leftStickY;
    }
    
    // Keyboard movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) moveDirection.z = -1.0f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) moveDirection.z = 1.0f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) moveDirection.x = -1.0f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) moveDirection.x = 1.0f;
    
    // Apply movement
    if (glm::length(moveDirection) > 0.1f) {
        moveDirection = glm::normalize(moveDirection);
        float speed = 15.0f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            speed *= 2.0f;
        }
        playerVel.x = moveDirection.x * speed;
        playerVel.z = moveDirection.z * speed;
    } else {
        playerVel.x *= 0.8f;
        playerVel.z *= 0.8f;
    }
    
    // Jump
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && playerPos.y <= 1.1f) {
        playerVel.y = 15.0f;
    }
    
    // Camera mode switching
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        camera.setCameraMode(Enhanced3DCamera::CameraMode::PLAYER_FOLLOW);
        std::cout << "Camera Mode: PLAYER_FOLLOW" << std::endl;
    } else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        camera.setCameraMode(Enhanced3DCamera::CameraMode::FREE_LOOK);
        std::cout << "Camera Mode: FREE_LOOK" << std::endl;
    } else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        camera.setCameraMode(Enhanced3DCamera::CameraMode::COMBAT_FOCUS);
        std::cout << "Camera Mode: COMBAT_FOCUS" << std::endl;
    }
}

void updateGame(float dt) {
    // Apply gravity
    if (playerPos.y > 1.0f) {
        playerVel.y -= 30.0f * dt;
    }
    
    // Update position
    playerPos += playerVel * dt;
    
    // Ground collision
    if (playerPos.y <= 1.0f) {
        playerPos.y = 1.0f;
        playerVel.y = 0.0f;
    }
    
    // Update camera
    camera.update(dt, playerPos, playerVel, false);
}

void renderScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    
    // Render ground
    renderCube(glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(50.0f, 1.0f, 50.0f), glm::vec3(0.3f, 0.3f, 0.3f));
    
    // Render player
    renderCube(playerPos, glm::vec3(0.8f, 1.8f, 0.8f), glm::vec3(1.0f, 0.5f, 0.0f));
    
    // Render some reference objects
    renderCube(glm::vec3(10.0f, 1.0f, 10.0f), glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    renderCube(glm::vec3(-10.0f, 1.0f, 10.0f), glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    renderCube(glm::vec3(10.0f, 1.0f, -10.0f), glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    renderCube(glm::vec3(-10.0f, 1.0f, -10.0f), glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(1.0f, 1.0f, 0.0f));
}

void renderUI() {
    static float uiUpdateTimer = 0.0f;
    uiUpdateTimer += deltaTime;
    
    if (uiUpdateTimer > 0.5f) {
        system("cls");
        
        std::cout << "=== Camera Test Demo ===" << std::endl;
        std::cout << "Player Position: (" << playerPos.x << ", " << playerPos.y << ", " << playerPos.z << ")" << std::endl;
        std::cout << "Camera Position: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")" << std::endl;
        
        std::string currentMode;
        switch(camera.currentMode) {
            case Enhanced3DCamera::CameraMode::PLAYER_FOLLOW: currentMode = "PLAYER_FOLLOW"; break;
            case Enhanced3DCamera::CameraMode::FREE_LOOK: currentMode = "FREE_LOOK"; break;
            case Enhanced3DCamera::CameraMode::COMBAT_FOCUS: currentMode = "COMBAT_FOCUS"; break;
            default: currentMode = "UNKNOWN"; break;
        }
        std::cout << "Camera Mode: " << currentMode << std::endl;
        
        std::cout << "\nControls:" << std::endl;
        std::cout << "WASD/Left Stick: Move Player" << std::endl;
        std::cout << "Right Stick: Camera Control" << std::endl;
        std::cout << "Space: Jump" << std::endl;
        std::cout << "Shift: Sprint" << std::endl;
        std::cout << "1: Player Follow Mode" << std::endl;
        std::cout << "2: Free Look Mode" << std::endl;
        std::cout << "3: Combat Focus Mode" << std::endl;
        std::cout << "ESC: Exit" << std::endl;
        
        uiUpdateTimer = 0.0f;
    }
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Camera Test Demo", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    
    initOpenGL();
    
    std::cout << "Camera Test Demo Started!" << std::endl;
    std::cout << "Use 1, 2, 3 to switch camera modes" << std::endl;
    
    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        deltaTime = std::min(deltaTime, 0.016f);
        
        processInput(window);
        
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }
        
        updateGame(deltaTime);
        renderScene();
        renderUI();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    
    glfwTerminate();
    return 0;
}
