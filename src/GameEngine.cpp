#include "GameEngine.h"
#include <iostream>
#include <cstring>

// Vertex shader source
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;

out vec3 vertexColor;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    gl_PointSize = aSize;
    vertexColor = aColor;
}
)";

// Fragment shader source
const char* fragmentShaderSource = R"(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5)
        discard;
    
    float alpha = 1.0 - length(coord) * 2.0;
    FragColor = vec4(vertexColor, alpha);
}
)";

GameEngine::GameEngine() 
    : m_window(nullptr), m_shaderProgram(0), m_deltaTime(0.0f), 
      m_mouseX(0.0), m_mouseY(0.0), m_mousePressed(false) {
    memset(m_keys, 0, sizeof(m_keys));
}

GameEngine::~GameEngine() {
    shutdown();
}

bool GameEngine::initialize() {
    if (!initializeWindow()) {
        return false;
    }
    
    if (!initializeOpenGL()) {
        return false;
    }
    
    if (!loadShaders()) {
        return false;
    }
    
    // Initialize game systems
    m_gameWorld = std::make_unique<GameWorld>();
    m_gameWorld->initialize();
    
    m_player = std::make_unique<Player>();
    m_player->setGameWorld(m_gameWorld.get());
    
    // Create and populate test level
    m_testLevel = std::make_unique<TestLevel>();
    m_testLevel->createLevel(m_gameWorld.get());
    
    // Initialize rhythm system
    m_rhythmSystem = std::make_unique<RhythmSystem>(140.0f); // 140 BPM
    m_rhythmSystem->initialize();
    
    // Initialize character renderer
    m_characterRenderer = std::make_unique<CharacterRenderer>();
    m_characterRenderer->initialize();
    
    m_particleSystem = std::make_unique<ParticleSystem>(10000);
    m_particleSystem->initialize();
    
    // Set up timing
    m_lastTime = std::chrono::high_resolution_clock::now();
    
    // Enable point size modification in shaders
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    return true;
}

bool GameEngine::initializeWindow() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // Set OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create window
    m_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA Particle Game", nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(m_window);
    
    // Set callbacks
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetMouseButtonCallback(m_window, mouseButtonCallback);
    glfwSetCursorPosCallback(m_window, cursorPosCallback);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    
    return true;
}

bool GameEngine::initializeOpenGL() {
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }
    
    // Set viewport
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    
    // Print OpenGL info
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    
    return true;
}

bool GameEngine::loadShaders() {
    m_shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    return m_shaderProgram != 0;
}

unsigned int GameEngine::compileShader(const char* source, GLenum type) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    // Check for compilation errors
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        return 0;
    }
    
    return shader;
}

unsigned int GameEngine::createShaderProgram(const char* vertexSource, const char* fragmentSource) {
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
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        return 0;
    }
    
    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

void GameEngine::run() {
    while (!glfwWindowShouldClose(m_window)) {
        calculateDeltaTime();
        processInput();
        update();
        render();
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

void GameEngine::calculateDeltaTime() {
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - m_lastTime);
    m_deltaTime = duration.count() / 1000000.0f;
    m_lastTime = currentTime;
}

void GameEngine::processInput() {
    if (m_keys[GLFW_KEY_ESCAPE]) {
        glfwSetWindowShouldClose(m_window, true);
    }
    
    // Handle world rotation
    static bool qPressed = false, ePressed = false;
    if (m_keys[GLFW_KEY_Q] && !qPressed) {
        m_gameWorld->rotateWorld(false); // Counter-clockwise
        m_player->preserveMomentumOnRotation();
        qPressed = true;
    } else if (!m_keys[GLFW_KEY_Q]) {
        qPressed = false;
    }
    
    if (m_keys[GLFW_KEY_E] && !ePressed) {
        m_gameWorld->rotateWorld(true); // Clockwise
        m_player->preserveMomentumOnRotation();
        ePressed = true;
    } else if (!m_keys[GLFW_KEY_E]) {
        ePressed = false;
    }
    
    // Handle player input
    m_player->handleInput(m_keys, m_deltaTime);
    m_player->handleMouseInput(m_mouseX, m_mouseY);
    
    // Handle mouse for particles (testing)
    if (m_mousePressed) {
        float x = (m_mouseX / WINDOW_WIDTH) * 2.0f - 1.0f;
        float y = -((m_mouseY / WINDOW_HEIGHT) * 2.0f - 1.0f);
        m_particleSystem->addParticles(make_float2(x, y), 50);
    }
}

void GameEngine::update() {
    m_rhythmSystem->update(m_deltaTime);
    m_gameWorld->update(m_deltaTime);
    m_player->update(m_deltaTime);
    m_particleSystem->update(m_deltaTime);
}

void GameEngine::render() {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST);
    
    // Get view and projection matrices from game world
    glm::mat4 view = m_gameWorld->getViewMatrix();
    glm::mat4 projection = m_gameWorld->getProjectionMatrix();
    
    // Render world geometry first
    m_gameWorld->render();
    
    // Render the player character
    m_characterRenderer->render(m_player.get(), view, projection);
    
    // Render particles on top
    glUseProgram(m_shaderProgram);
    m_particleSystem->render();
}

void GameEngine::shutdown() {
    if (m_shaderProgram) {
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }
    
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    
    glfwTerminate();
}

// Callback functions
void GameEngine::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    GameEngine* engine = static_cast<GameEngine*>(glfwGetWindowUserPointer(window));
    
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) {
            engine->m_keys[key] = true;
        } else if (action == GLFW_RELEASE) {
            engine->m_keys[key] = false;
        }
    }
}

void GameEngine::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    GameEngine* engine = static_cast<GameEngine*>(glfwGetWindowUserPointer(window));
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        engine->m_mousePressed = (action == GLFW_PRESS);
    }
}

void GameEngine::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    GameEngine* engine = static_cast<GameEngine*>(glfwGetWindowUserPointer(window));
    engine->m_mouseX = xpos;
    engine->m_mouseY = ypos;
}

void GameEngine::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
