#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <memory>
#include <chrono>
#include "ParticleSystem.cuh"
#include "GameWorld.h"
#include "Player.h"
#include "TestLevel.h"
#include "RhythmSystem.h"
#include "CharacterRenderer.h"

class GameEngine {
public:
    GameEngine();
    ~GameEngine();

    bool initialize();
    void run();
    void shutdown();

private:
    GLFWwindow* m_window;
    std::unique_ptr<ParticleSystem> m_particleSystem;
    std::unique_ptr<GameWorld> m_gameWorld;
    std::unique_ptr<Player> m_player;
    std::unique_ptr<TestLevel> m_testLevel;
    std::unique_ptr<RhythmSystem> m_rhythmSystem;
    std::unique_ptr<CharacterRenderer> m_characterRenderer;
    
    // Window properties
    static const int WINDOW_WIDTH = 1200;
    static const int WINDOW_HEIGHT = 800;
    
    // Timing
    std::chrono::high_resolution_clock::time_point m_lastTime;
    float m_deltaTime;
    
    // Input handling
    bool m_keys[1024];
    double m_mouseX, m_mouseY;
    bool m_mousePressed;
    
    bool initializeWindow();
    bool initializeOpenGL();
    void processInput();
    void update();
    void render();
    void calculateDeltaTime();
    
    // Callbacks
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    
    // Shader management
    unsigned int m_shaderProgram;
    bool loadShaders();
    unsigned int compileShader(const char* source, GLenum type);
    unsigned int createShaderProgram(const char* vertexSource, const char* fragmentSource);
};
