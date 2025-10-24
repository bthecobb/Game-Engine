#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Simple vector3 implementation for demo
struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator*(float scale) const { return Vec3(x * scale, y * scale, z * scale); }
    glm::vec3 toGLM() const { return glm::vec3(x, y, z); }
};

struct Vec2 {
    float x, y;
    Vec2(float x = 0, float y = 0) : x(x), y(y) {}
    float length() const { return sqrt(x*x + y*y); }
};

// Simple animation states
enum class SimpleAnimationType {
    IDLE,
    WALK,
    RUN,
    SPRINT,
    JUMP
};

// Simple animation frame
struct SimpleAnimationFrame {
    Vec3 headPos{0, 1.5f, 0};
    Vec3 torsoPos{0, 0.5f, 0};
    Vec3 leftArmPos{-0.6f, 0.8f, 0};
    Vec3 rightArmPos{0.6f, 0.8f, 0};
    Vec3 leftLegPos{-0.2f, -0.8f, 0};
    Vec3 rightLegPos{0.2f, -0.8f, 0};
    
    float energyLevel = 0.5f;
    float animationTime = 0.0f;
};

class SimpleAnimationController {
private:
    SimpleAnimationType m_currentType = SimpleAnimationType::IDLE;
    float m_animationTime = 0.0f;
    float m_speed = 0.0f;
    Vec2 m_direction{0, 0};
    
public:
    void update(float deltaTime, const Vec2& inputDirection, float speed) {
        m_animationTime += deltaTime;
        m_direction = inputDirection;
        m_speed = speed;
        
        // Simple state selection
        if (speed < 0.5f) {
            m_currentType = SimpleAnimationType::IDLE;
        } else if (speed < 5.0f) {
            m_currentType = SimpleAnimationType::WALK;
        } else if (speed < 15.0f) {
            m_currentType = SimpleAnimationType::RUN;
        } else {
            m_currentType = SimpleAnimationType::SPRINT;
        }
    }
    
    SimpleAnimationFrame getCurrentFrame() const {
        SimpleAnimationFrame frame;
        frame.animationTime = m_animationTime;
        
        switch (m_currentType) {
            case SimpleAnimationType::IDLE:
                generateIdleFrame(frame);
                break;
            case SimpleAnimationType::WALK:
                generateWalkFrame(frame);
                break;
            case SimpleAnimationType::RUN:
                generateRunFrame(frame);
                break;
            case SimpleAnimationType::SPRINT:
                generateSprintFrame(frame);
                break;
            case SimpleAnimationType::JUMP:
                generateJumpFrame(frame);
                break;
        }
        
        return frame;
    }
    
    std::string getCurrentAnimationName() const {
        switch (m_currentType) {
            case SimpleAnimationType::IDLE: return "Idle";
            case SimpleAnimationType::WALK: return "Walk";
            case SimpleAnimationType::RUN: return "Run";
            case SimpleAnimationType::SPRINT: return "Sprint";
            case SimpleAnimationType::JUMP: return "Jump";
            default: return "Unknown";
        }
    }
    
private:
    void generateIdleFrame(SimpleAnimationFrame& frame) const {
        // Subtle breathing and sway
        float breathPhase = m_animationTime * 3.0f;
        float swayPhase = m_animationTime * 1.5f;
        
        float breathOffset = sin(breathPhase) * 0.01f;
        float swayOffset = sin(swayPhase) * 0.005f;
        
        frame.headPos = Vec3(swayOffset, 1.5f + breathOffset * 0.5f, 0);
        frame.torsoPos = Vec3(swayOffset, 0.5f + breathOffset, 0);
        frame.leftArmPos = Vec3(-0.6f + swayOffset, 0.8f, 0);
        frame.rightArmPos = Vec3(0.6f + swayOffset, 0.8f, 0);
        frame.leftLegPos = Vec3(-0.2f + swayOffset, -0.8f, 0);
        frame.rightLegPos = Vec3(0.2f + swayOffset, -0.8f, 0);
        
        frame.energyLevel = 0.2f;
    }
    
    void generateWalkFrame(SimpleAnimationFrame& frame) const {
        float cycleTime = fmod(m_animationTime, 1.0f);
        float cyclePhase = cycleTime * 2.0f * M_PI;
        
        float verticalBob = sin(cyclePhase * 2.0f) * 0.05f;
        float legSwing = sin(cyclePhase) * 0.3f;
        float armSwing = sin(cyclePhase + M_PI) * 0.2f;
        
        frame.headPos = Vec3(0, 1.5f + verticalBob * 0.5f, 0);
        frame.torsoPos = Vec3(0, 0.5f + verticalBob, 0);
        
        frame.leftLegPos = Vec3(-0.2f, -0.8f, legSwing);
        frame.rightLegPos = Vec3(0.2f, -0.8f, -legSwing);
        
        frame.leftArmPos = Vec3(-0.6f, 0.8f, armSwing);
        frame.rightArmPos = Vec3(0.6f, 0.8f, -armSwing);
        
        frame.energyLevel = 0.6f;
    }
    
    void generateRunFrame(SimpleAnimationFrame& frame) const {
        float cycleTime = fmod(m_animationTime, 0.6f);
        float cyclePhase = cycleTime * 2.0f * M_PI / 0.6f;
        
        float verticalBounce = sin(cyclePhase * 2.0f) * 0.1f;
        float legStride = sin(cyclePhase) * 0.5f;
        float armPump = sin(cyclePhase + M_PI) * 0.4f;
        
        // Forward lean
        frame.headPos = Vec3(0, 1.5f + verticalBounce * 0.7f, 0.1f);
        frame.torsoPos = Vec3(0, 0.5f + verticalBounce, 0.05f);
        
        frame.leftLegPos = Vec3(-0.2f, -0.8f, legStride);
        frame.rightLegPos = Vec3(0.2f, -0.8f, -legStride);
        
        frame.leftArmPos = Vec3(-0.6f, 0.8f, armPump * 0.2f);
        frame.rightArmPos = Vec3(0.6f, 0.8f, -armPump * 0.2f);
        
        frame.energyLevel = 0.8f;
    }
    
    void generateSprintFrame(SimpleAnimationFrame& frame) const {
        float cycleTime = fmod(m_animationTime, 0.4f);
        float cyclePhase = cycleTime * 2.0f * M_PI / 0.4f;
        
        float verticalBound = sin(cyclePhase * 2.0f) * 0.15f;
        float legDrive = sin(cyclePhase) * 0.7f;
        float armDrive = sin(cyclePhase + M_PI) * 0.6f;
        
        // Aggressive forward lean
        frame.headPos = Vec3(0, 1.5f + verticalBound * 0.8f, 0.2f);
        frame.torsoPos = Vec3(0, 0.5f + verticalBound, 0.15f);
        
        frame.leftLegPos = Vec3(-0.2f, -0.8f, legDrive);
        frame.rightLegPos = Vec3(0.2f, -0.8f, -legDrive);
        
        frame.leftArmPos = Vec3(-0.6f, 0.8f, armDrive * 0.3f);
        frame.rightArmPos = Vec3(0.6f, 0.8f, -armDrive * 0.3f);
        
        frame.energyLevel = 1.0f;
    }
    
    void generateJumpFrame(SimpleAnimationFrame& frame) const {
        // Extended pose for jumping
        frame.headPos = Vec3(0, 1.7f, 0);
        frame.torsoPos = Vec3(0, 0.8f, 0);
        frame.leftArmPos = Vec3(-0.6f, 1.2f, 0);
        frame.rightArmPos = Vec3(0.6f, 1.2f, 0);
        frame.leftLegPos = Vec3(-0.2f, -0.4f, 0);
        frame.rightLegPos = Vec3(0.2f, -0.4f, 0);
        
        frame.energyLevel = 1.0f;
    }
};

// OpenGL Renderer for the character
class CharacterRenderer {
private:
    GLuint VAO, VBO;
    GLuint shaderProgram;
    glm::mat4 view, projection;
    
public:
    bool initialize() {
        // Create and compile shaders
        const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 color;
        out vec3 vertColor;
        
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            vertColor = color;
        }
        )";
        
        const char* fragmentShaderSource = R"(
        #version 330 core
        in vec3 vertColor;
        out vec4 FragColor;
        
        void main() {
            FragColor = vec4(vertColor, 1.0);
        }
        )";
        
        // Compile vertex shader
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        
        // Check for compilation errors
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
            return false;
        }
        
        // Compile fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
            return false;
        }
        
        // Create shader program
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "Shader program linking failed: " << infoLog << std::endl;
            return false;
        }
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        
        // Set up cube vertices for body parts
        float vertices[] = {
            // Front face
            -0.1f, -0.1f,  0.1f,
             0.1f, -0.1f,  0.1f,
             0.1f,  0.1f,  0.1f,
             0.1f,  0.1f,  0.1f,
            -0.1f,  0.1f,  0.1f,
            -0.1f, -0.1f,  0.1f,
            
            // Back face
            -0.1f, -0.1f, -0.1f,
             0.1f, -0.1f, -0.1f,
             0.1f,  0.1f, -0.1f,
             0.1f,  0.1f, -0.1f,
            -0.1f,  0.1f, -0.1f,
            -0.1f, -0.1f, -0.1f,
            
            // Left face
            -0.1f,  0.1f,  0.1f,
            -0.1f,  0.1f, -0.1f,
            -0.1f, -0.1f, -0.1f,
            -0.1f, -0.1f, -0.1f,
            -0.1f, -0.1f,  0.1f,
            -0.1f,  0.1f,  0.1f,
            
            // Right face
             0.1f,  0.1f,  0.1f,
             0.1f,  0.1f, -0.1f,
             0.1f, -0.1f, -0.1f,
             0.1f, -0.1f, -0.1f,
             0.1f, -0.1f,  0.1f,
             0.1f,  0.1f,  0.1f,
            
            // Bottom face
            -0.1f, -0.1f, -0.1f,
             0.1f, -0.1f, -0.1f,
             0.1f, -0.1f,  0.1f,
             0.1f, -0.1f,  0.1f,
            -0.1f, -0.1f,  0.1f,
            -0.1f, -0.1f, -0.1f,
            
            // Top face
            -0.1f,  0.1f, -0.1f,
             0.1f,  0.1f, -0.1f,
             0.1f,  0.1f,  0.1f,
             0.1f,  0.1f,  0.1f,
            -0.1f,  0.1f,  0.1f,
            -0.1f,  0.1f, -0.1f
        };
        
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Set up view and projection matrices
        view = glm::lookAt(glm::vec3(3.0f, 2.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        
        return true;
    }
    
    void renderCharacter(const SimpleAnimationFrame& frame) {
        glUseProgram(shaderProgram);
        
        // Set view and projection matrices
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        glBindVertexArray(VAO);
        
        // Render each body part with different colors and scales
        
        // Head (red)
        renderBodyPart(frame.headPos.toGLM(), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.2f, 1.2f, 1.2f));
        
        // Torso (blue)
        renderBodyPart(frame.torsoPos.toGLM(), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.5f, 2.0f, 1.0f));
        
        // Left arm (green)
        renderBodyPart(frame.leftArmPos.toGLM(), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.5f, 1.0f, 0.5f));
        
        // Right arm (green)
        renderBodyPart(frame.rightArmPos.toGLM(), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.5f, 1.0f, 0.5f));
        
        // Left leg (yellow)
        renderBodyPart(frame.leftLegPos.toGLM(), glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(0.6f, 1.5f, 0.6f));
        
        // Right leg (yellow)
        renderBodyPart(frame.rightLegPos.toGLM(), glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(0.6f, 1.5f, 0.6f));
        
        // Energy indicator (white, intensity based on energy level)
        glm::vec3 energyColor = glm::vec3(frame.energyLevel, frame.energyLevel, frame.energyLevel);
        glm::vec3 energyPos = frame.headPos.toGLM() + glm::vec3(0, 0.3f, 0);
        renderBodyPart(energyPos, energyColor, glm::vec3(0.3f, 0.1f, 0.1f));
    }
    
private:
    void renderBodyPart(const glm::vec3& position, const glm::vec3& color, const glm::vec3& scale) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model = glm::scale(model, scale);
        
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, glm::value_ptr(color));
        
        glDrawArrays(GL_TRIANGLES, 0, 36);
    }
    
public:
    ~CharacterRenderer() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteProgram(shaderProgram);
    }
};

class VisualAnimationDemo {
private:
    GLFWwindow* m_window;
    SimpleAnimationController m_controller;
    CharacterRenderer m_renderer;
    bool m_isRunning = false;
    float m_demoTime = 0.0f;
    
public:
    bool initialize() {
        // Initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }
        
        // Configure GLFW
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        // Create window
        m_window = glfwCreateWindow(800, 600, "Visual Animation Demo - Character Locomotion", NULL, NULL);
        if (!m_window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(m_window);
        
        // Initialize GLAD
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            return false;
        }
        
        // Configure OpenGL
        glViewport(0, 0, 800, 600);
        glEnable(GL_DEPTH_TEST);
        
        // Set callbacks
        glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int width, int height) {
            glViewport(0, 0, width, height);
        });
        
        // Initialize renderer
        if (!m_renderer.initialize()) {
            return false;
        }
        
        std::cout << "🎮 Visual Animation Demo Initialized!" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Window: 800x600 OpenGL Renderer" << std::endl;
        std::cout << "Controls: Watch the character animate automatically" << std::endl;
        std::cout << "Colors: Red=Head, Blue=Torso, Green=Arms, Yellow=Legs" << std::endl;
        std::cout << "White bar above head = Energy level" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        return true;
    }
    
    void run() {
        m_isRunning = true;
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        
        while (!glfwWindowShouldClose(m_window) && m_isRunning) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;
            
            deltaTime = std::min(deltaTime, 1.0f / 30.0f); // Cap at 30 FPS minimum
            
            update(deltaTime);
            render();
            
            frameCount++;
            
            // Print status every 3 seconds
            if (frameCount % 180 == 0) {
                printStatus();
                changeScenario(frameCount / 180);
            }
            
            glfwSwapBuffers(m_window);
            glfwPollEvents();
        }
        
        std::cout << "\n🎬 Visual Demo completed!" << std::endl;
        printSummary();
    }
    
private:
    void update(float deltaTime) {
        m_demoTime += deltaTime;
        
        // Simulate different movement patterns
        Vec2 input = simulateInput();
        float speed = calculateSpeed(input);
        
        m_controller.update(deltaTime, input, speed);
    }
    
    void render() {
        // Clear the screen
        glClearColor(0.1f, 0.1f, 0.2f, 1.0f); // Dark blue background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Get current animation frame
        SimpleAnimationFrame frame = m_controller.getCurrentFrame();
        
        // Render the character
        m_renderer.renderCharacter(frame);
    }
    
    Vec2 simulateInput() {
        float cycleTime = fmod(m_demoTime, 12.0f); // 12-second cycles
        
        if (cycleTime < 3.0f) {
            // Idle
            return Vec2(0, 0);
        } else if (cycleTime < 6.0f) {
            // Walking forward
            return Vec2(0, 1);
        } else if (cycleTime < 9.0f) {
            // Running in circles
            float angle = cycleTime * 2.0f * M_PI;
            return Vec2(sin(angle), cos(angle));
        } else {
            // Sprinting forward
            return Vec2(0, 1);
        }
    }
    
    float calculateSpeed(const Vec2& input) {
        float cycleTime = fmod(m_demoTime, 12.0f);
        
        if (cycleTime < 3.0f) {
            return 0.0f; // Idle
        } else if (cycleTime < 6.0f) {
            return 2.0f; // Walk speed
        } else if (cycleTime < 9.0f) {
            return 8.0f; // Run speed
        } else {
            return 20.0f; // Sprint speed
        }
    }
    
    void changeScenario(int scenarioIndex) {
        switch (scenarioIndex % 4) {
            case 0:
                std::cout << "\n🚶 Scenario: Idle Breathing Demo" << std::endl;
                break;
            case 1:
                std::cout << "\n🚶‍♂️ Scenario: Walking Gait Cycle" << std::endl;
                break;
            case 2:
                std::cout << "\n🏃‍♂️ Scenario: Running with Bob & Lean" << std::endl;
                break;
            case 3:
                std::cout << "\n⚡ Scenario: Sprint with Aggressive Lean" << std::endl;
                break;
        }
    }
    
    void printStatus() {
        SimpleAnimationFrame frame = m_controller.getCurrentFrame();
        
        std::cout << "\n📊 Visual Status (t=" << std::fixed << std::setprecision(1) << m_demoTime << "s)" << std::endl;
        std::cout << "   Animation: " << m_controller.getCurrentAnimationName() << std::endl;
        std::cout << "   Energy: " << std::setprecision(2) << frame.energyLevel << std::endl;
        std::cout << "   Head Position: (" << frame.headPos.x << ", " << frame.headPos.y << ", " << frame.headPos.z << ")" << std::endl;
        std::cout << "   Torso Position: (" << frame.torsoPos.x << ", " << frame.torsoPos.y << ", " << frame.torsoPos.z << ")" << std::endl;
        std::cout << "   Left Leg Z: " << frame.leftLegPos.z << " | Right Leg Z: " << frame.rightLegPos.z << std::endl;
    }
    
    void printSummary() {
        std::cout << "\n📈 Visual Demo Summary" << std::endl;
        std::cout << "======================" << std::endl;
        std::cout << "Total runtime: " << m_demoTime << " seconds" << std::endl;
        std::cout << "Visual features demonstrated:" << std::endl;
        std::cout << "  ✅ Real-time 3D character rendering" << std::endl;
        std::cout << "  ✅ Animated body part positioning" << std::endl;
        std::cout << "  ✅ Color-coded body parts" << std::endl;
        std::cout << "  ✅ Energy level visualization" << std::endl;
        std::cout << "  ✅ Smooth animation transitions" << std::endl;
        std::cout << "  ✅ OpenGL-based rendering pipeline" << std::endl;
    }
    
public:
    ~VisualAnimationDemo() {
        glfwTerminate();
    }
};

int main() {
    std::cout << "🎮 Visual Animation System Demo" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "Real-time 3D character animation with OpenGL rendering" << std::endl;
    std::cout << "===============================" << std::endl;
    
    VisualAnimationDemo demo;
    
    if (!demo.initialize()) {
        std::cerr << "❌ Failed to initialize visual demo" << std::endl;
        return 1;
    }
    
    try {
        demo.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 Thank you for watching the Visual Animation Demo!" << std::endl;
    return 0;
}
