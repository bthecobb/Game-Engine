#include "GameEngine.h"
#include <iostream>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Core/Coordinator.h"
#include "Physics/PhysicsComponents.h"
#include "Physics/PhysicsSystem.h"
#include "Physics/CudaPhysicsSystem.h"
#include "Physics/WallRunningSystem.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/RenderSystem.h"
#include "Rendering/CudaRenderingSystem.h"
#include "Rendering/LightingSystem.h"
#include "../include_refactored/Rendering/Camera.h"
#include "Animation/AnimationSystem.h"
#include "Animation/IK.h"
#include "Animation/IKSystem.h"
#include "Combat/CombatSystem.h"
#include "GameFeel/GameFeelSystem.h"
#include "Particles/ParticleSystem.h"
#include "Particles/ParticleComponents.h"
#include "Rhythm/RhythmSystem.h"
#include "Audio/AudioComponents.h"
#include "Gameplay/EnemyComponents.h"

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
      m_mouseX(0.0), m_mouseY(0.0), m_mousePressed(false),
      m_fps(0.0f), m_frameCount(0), m_playerPosition(0.0f), m_enemyCount(0) {
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
    
    // Initialize UI Renderer
    m_uiRenderer = std::make_unique<CudaGame::UI::UIRenderer>();
    if (!m_uiRenderer->Initialize()) {
        std::cerr << "Failed to initialize UI Renderer" << std::endl;
        return false;
    }
    m_uiRenderer->SetViewportSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    
    // Set up timing
    m_lastTime = std::chrono::high_resolution_clock::now();
    m_fpsTimer = std::chrono::high_resolution_clock::now();
    
// Create Coordinator and register systems
    auto& coordinator = CudaGame::Core::Coordinator::GetInstance();
    coordinator.Initialize();

    // Register ALL components
    coordinator.RegisterComponent<CudaGame::Physics::RigidbodyComponent>();
    coordinator.RegisterComponent<CudaGame::Physics::ColliderComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::MeshComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::TransformComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::MaterialComponent>();
    coordinator.RegisterComponent<CudaGame::Rendering::LightComponent>();
    coordinator.RegisterComponent<CudaGame::Animation::AnimationComponent>();
    coordinator.RegisterComponent<CudaGame::Animation::IKComponent>();
coordinator.RegisterComponent<CudaGame::Combat::CombatComponent>();
    coordinator.RegisterComponent<CudaGame::Particles::ParticleSystemComponent>();

    // Register ALL systems
    auto animSystem = coordinator.RegisterSystem<CudaGame::Animation::AnimationSystem>();
    auto ikSystem = coordinator.RegisterSystem<CudaGame::Animation::IKSystem>();
    auto combatSystem = coordinator.RegisterSystem<CudaGame::Combat::CombatSystem>();
    auto gameFeelSystem = coordinator.RegisterSystem<CudaGame::GameFeel::GameFeelSystem>();
    auto renderSystem = coordinator.RegisterSystem<CudaGame::Rendering::RenderSystem>();
    auto lightingSystem = coordinator.RegisterSystem<CudaGame::Rendering::LightingSystem>();
    auto physicsSystem = coordinator.RegisterSystem<CudaGame::Physics::PhysicsSystem>();
    auto wallRunSystem = coordinator.RegisterSystem<CudaGame::Physics::WallRunningSystem>();
    auto particleSystemECS = coordinator.RegisterSystem<CudaGame::Particles::ParticleSystem>();
    auto rhythmSystemECS = coordinator.RegisterSystem<CudaGame::Rhythm::RhythmSystem>();

    // Initialize ALL systems
    animSystem->Initialize();
    ikSystem->Initialize();
    combatSystem->Initialize();
    gameFeelSystem->Initialize();
    renderSystem->Initialize();
    lightingSystem->Initialize();
    physicsSystem->Initialize();
    wallRunSystem->Initialize();
    particleSystemECS->Initialize();
    rhythmSystemECS->Initialize();

    // Enable point size modification in shaders
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // CREATE CAMERA FOR DEFERRED RENDERING
m_camera = std::make_unique<CudaGame::Rendering::Camera>(CudaGame::Rendering::ProjectionType::PERSPECTIVE);
    m_camera->SetPosition(glm::vec3(0.0f, 10.0f, 15.0f));
    m_camera->LookAt(glm::vec3(0.0f, 0.0f, 0.0f));
    m_camera->SetPerspective(45.0f, (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 100.0f);
    renderSystem->SetMainCamera(m_camera.get());
    
    // CREATE 3D ENTITIES FOR RENDERING
    CreateDemo3DEntities(coordinator);
    
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
    
    // Calculate FPS
    m_frameCount++;
    auto fpsDuration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - m_fpsTimer);
    if (fpsDuration.count() >= 1000) { // Update FPS every second
        m_fps = m_frameCount * 1000.0f / fpsDuration.count();
        m_frameCount = 0;
        m_fpsTimer = currentTime;
    }
}

void GameEngine::processInput() {
    if (m_keys[GLFW_KEY_ESCAPE]) {
        glfwSetWindowShouldClose(m_window, true);
    }
    
    if (m_keys[GLFW_KEY_F4] && !f4Pressed) {
        auto& coordinator = CudaGame::Core::Coordinator::GetInstance();
        auto renderSystem = coordinator.GetSystem<CudaGame::Rendering::RenderSystem>();
        renderSystem->CycleDebugMode();
        f4Pressed = true;
    } else if (!m_keys[GLFW_KEY_F4]) {
        f4Pressed = false;
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
    // Update camera
    m_camera->UpdateMatrices();

    // Get coordinator and update all ECS systems
    auto& coordinator = CudaGame::Core::Coordinator::GetInstance();
    coordinator.UpdateSystems(m_deltaTime);
    
    // Update legacy systems
    m_rhythmSystem->update(m_deltaTime);
    m_gameWorld->update(m_deltaTime);
    m_player->update(m_deltaTime);
    m_particleSystem->update(m_deltaTime);
    
    // Update debug info
    m_playerPosition = m_player->getPosition();
    
    // Count enemies by checking for EnemyAIComponent
    m_enemyCount = 0;
    for (CudaGame::Core::Entity entity = 0; entity < 1000; ++entity) {
        if (coordinator.HasComponent<CudaGame::Gameplay::EnemyAIComponent>(entity)) {
            m_enemyCount++;
        }
    }
}

void GameEngine::render() {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST);
    
    // USE THE PROPER 3D CAMERA MATRICES
    glm::mat4 view = m_camera->GetViewMatrix();
    glm::mat4 projection = m_camera->GetProjectionMatrix();
    
    // Get coordinator for system access
    auto& coordinator = CudaGame::Core::Coordinator::GetInstance();
    
    // Update camera for all rendering systems
    auto renderSystem = coordinator.GetSystem<CudaGame::Rendering::RenderSystem>();
    auto lightingSystem = coordinator.GetSystem<CudaGame::Rendering::LightingSystem>();
    
    // Update lighting system for shadow maps
    lightingSystem->Update(0.0f);
    
    // Main render pass using 3D deferred rendering
    renderSystem->Render(m_player.get());
    
    // Render a simple debug HUD
    glDisable(GL_DEPTH_TEST);
    renderDebugHUD();
    glEnable(GL_DEPTH_TEST);
}

void GameEngine::renderDebugHUD() {
    if (m_uiRenderer) {
        m_uiRenderer->BeginFrame();
        m_uiRenderer->DrawDebugInfo(m_fps, m_playerPosition, m_enemyCount);
        m_uiRenderer->EndFrame();
    }
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

void GameEngine::CreateDemo3DEntities(CudaGame::Core::Coordinator& coordinator) {
    // Create a ground plane
    auto ground = coordinator.CreateEntity();
    coordinator.AddComponent(ground, CudaGame::Rendering::TransformComponent{ glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(0.0f), glm::vec3(10.0f, 0.1f, 10.0f) });
    coordinator.AddComponent(ground, CudaGame::Rendering::MeshComponent{ "player_cube" });
    coordinator.AddComponent(ground, CudaGame::Rendering::MaterialComponent{ {0.5f, 0.5f, 0.5f} });
    coordinator.AddComponent(ground, CudaGame::Physics::RigidbodyComponent{});
    coordinator.AddComponent(ground, CudaGame::Physics::ColliderComponent{ CudaGame::Physics::ColliderShape::BOX, {5.0f, 0.05f, 5.0f} });

    // Create a few dynamic cubes
    for (int i = 0; i < 5; ++i) {
        auto cube = coordinator.CreateEntity();
        coordinator.AddComponent(cube, CudaGame::Rendering::TransformComponent{ glm::vec3(-4.0f + i * 2.0f, 2.0f, 0.0f), glm::vec3(0.0f), glm::vec3(1.0f) });
        coordinator.AddComponent(cube, CudaGame::Rendering::MeshComponent{ "player_cube" });
        coordinator.AddComponent(cube, CudaGame::Rendering::MaterialComponent{ {0.8f, 0.2f, 0.2f} });
        CudaGame::Physics::RigidbodyComponent rb;
        rb.setMass(1.0f);
        coordinator.AddComponent(cube, rb);
        coordinator.AddComponent(cube, CudaGame::Physics::ColliderComponent{ CudaGame::Physics::ColliderShape::BOX, {0.5f, 0.5f, 0.5f} });
    }
    
    // Create some static wall structures
    for (int i = 0; i < 4; ++i) {
        auto wall = coordinator.CreateEntity();
        glm::vec3 position;
        glm::vec3 scale;
        
        // Position walls around the scene
        if (i == 0) { position = glm::vec3(-5.0f, 1.0f, 0.0f); scale = glm::vec3(0.5f, 3.0f, 10.0f); }
        else if (i == 1) { position = glm::vec3(5.0f, 1.0f, 0.0f); scale = glm::vec3(0.5f, 3.0f, 10.0f); }
        else if (i == 2) { position = glm::vec3(0.0f, 1.0f, -5.0f); scale = glm::vec3(10.0f, 3.0f, 0.5f); }
        else { position = glm::vec3(0.0f, 1.0f, 5.0f); scale = glm::vec3(10.0f, 3.0f, 0.5f); }
        
        coordinator.AddComponent(wall, CudaGame::Rendering::TransformComponent{ position, glm::vec3(0.0f), scale });
        coordinator.AddComponent(wall, CudaGame::Rendering::MeshComponent{ "player_cube" });
        coordinator.AddComponent(wall, CudaGame::Rendering::MaterialComponent{ {0.3f, 0.3f, 0.8f} });
        coordinator.AddComponent(wall, CudaGame::Physics::RigidbodyComponent{});
        coordinator.AddComponent(wall, CudaGame::Physics::ColliderComponent{ CudaGame::Physics::ColliderShape::BOX, scale * 0.5f });
    }
    
    // Create some metallic spheres (rendered as cubes for now)
    for (int i = 0; i < 3; ++i) {
        auto sphere = coordinator.CreateEntity();
        float x = -2.0f + i * 2.0f;
        coordinator.AddComponent(sphere, CudaGame::Rendering::TransformComponent{ glm::vec3(x, 3.5f, 2.0f), glm::vec3(0.0f), glm::vec3(0.7f) });
        coordinator.AddComponent(sphere, CudaGame::Rendering::MeshComponent{ "player_cube" });
        CudaGame::Rendering::MaterialComponent mat;
        mat.albedo = glm::vec3(0.9f, 0.9f, 0.9f);
        mat.metallic = 0.9f;
        mat.roughness = 0.1f;
        mat.ao = 1.0f;
        coordinator.AddComponent(sphere, mat);
        CudaGame::Physics::RigidbodyComponent rb;
        rb.setMass(0.5f);
        coordinator.AddComponent(sphere, rb);
        coordinator.AddComponent(sphere, CudaGame::Physics::ColliderComponent{ CudaGame::Physics::ColliderShape::SPHERE, {0.35f, 0.35f, 0.35f} });
    }
}
