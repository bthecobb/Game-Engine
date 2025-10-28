#include "TestLevel.h"
#include <glad/glad.h>
#include <iostream>

// Platform implementation
Platform::Platform(glm::vec3 position, glm::vec3 size, bool visibleViews[4]) {
    transform.position = position;
    transform.scale = size;
    
    // Set visibility per view
    for (int i = 0; i < 4; i++) {
        activeFromView[i] = visibleViews[i];
        collidableFromView[i] = visibleViews[i]; // Solid when visible
    }
    
    // Color based on visibility pattern for debugging
    if (visibleViews[0] && visibleViews[1] && visibleViews[2] && visibleViews[3]) {
        m_color = glm::vec3(1.0f, 1.0f, 1.0f); // White - visible from all views
    } else if (visibleViews[0] && visibleViews[2]) {
        m_color = glm::vec3(1.0f, 0.0f, 0.0f); // Red - front/back only
    } else if (visibleViews[1] && visibleViews[3]) {
        m_color = glm::vec3(0.0f, 1.0f, 0.0f); // Green - left/right only
    } else {
        m_color = glm::vec3(0.0f, 0.0f, 1.0f); // Blue - specific view only
    }
    
    setupMesh();
}

void Platform::update(float deltaTime) {
    // Platforms are static for now
}

void Platform::render() {
    // Get shader uniforms (we'll need to pass view/projection matrices)
    // For now, render with basic OpenGL immediate mode style
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Platform::setupMesh() {
    // Create a simple cube
    float vertices[] = {
        // Positions
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f
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
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

// Wall implementation (similar to Platform)
Wall::Wall(glm::vec3 position, glm::vec3 size, bool visibleViews[4]) {
    transform.position = position;
    transform.scale = size;
    
    for (int i = 0; i < 4; i++) {
        activeFromView[i] = visibleViews[i];
        collidableFromView[i] = visibleViews[i];
    }
    
    m_color = glm::vec3(0.5f, 0.5f, 0.5f); // Gray for walls
    setupMesh();
}

void Wall::update(float deltaTime) {
    // Static walls
}

void Wall::render() {
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Wall::setupMesh() {
    // Same cube setup as Platform
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f
    };
    
    unsigned int indices[] = {
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 1, 5, 5, 4, 0,
        2, 3, 7, 7, 6, 2,
        0, 3, 7, 7, 4, 0,
        1, 2, 6, 6, 5, 1
    };
    
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);
    
    glBindVertexArray(m_VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

// TestLevel implementation
TestLevel::TestLevel() {
    std::cout << "TestLevel constructor called" << std::endl;
}

void TestLevel::createLevel(GameWorld* gameWorld) {
    std::cout << "Creating test level with dimensional geometry..." << std::endl;
    
    createDimensionalPlatforms(gameWorld);
    createWallRunSections(gameWorld);
    createSecretAreas(gameWorld);
    
    std::cout << "Test level created!" << std::endl;
}

void TestLevel::createDimensionalPlatforms(GameWorld* gameWorld) {
    // Ground platform (visible from all views)
    bool allViews[4] = {true, true, true, true};
    addPlatform(gameWorld, glm::vec3(0, -1, 0), glm::vec3(20, 1, 20), allViews);
    
    // Platforms that only exist in certain views
    bool frontBack[4] = {true, false, true, false}; // Front and Back only
    bool leftRight[4] = {false, true, false, true}; // Left and Right only
    bool frontOnly[4] = {true, false, false, false}; // Front view only
    
    // Create a "path" that changes based on view
    addPlatform(gameWorld, glm::vec3(-5, 2, 0), glm::vec3(3, 0.5f, 3), frontBack);
    addPlatform(gameWorld, glm::vec3(0, 2, -5), glm::vec3(3, 0.5f, 3), leftRight);
    addPlatform(gameWorld, glm::vec3(5, 2, 0), glm::vec3(3, 0.5f, 3), frontBack);
    addPlatform(gameWorld, glm::vec3(0, 4, 5), glm::vec3(3, 0.5f, 3), frontOnly);
    
    // Jumping platforms at different heights
    addPlatform(gameWorld, glm::vec3(-8, 1, -8), glm::vec3(2, 0.5f, 2), allViews);
    addPlatform(gameWorld, glm::vec3(-6, 3, -6), glm::vec3(2, 0.5f, 2), frontBack);
    addPlatform(gameWorld, glm::vec3(-4, 5, -4), glm::vec3(2, 0.5f, 2), leftRight);
}

void TestLevel::createWallRunSections(GameWorld* gameWorld) {
    // Walls for wall-running (tall and visible from multiple views)
    bool allViews[4] = {true, true, true, true};
    bool frontBack[4] = {true, false, true, false};
    
    // Wall run corridor
    addWall(gameWorld, glm::vec3(-10, 3, 10), glm::vec3(1, 6, 10), allViews);
    addWall(gameWorld, glm::vec3(10, 3, 10), glm::vec3(1, 6, 10), allViews);
    
    // Dimensional walls that appear/disappear
    addWall(gameWorld, glm::vec3(0, 3, 15), glm::vec3(8, 6, 1), frontBack);
}

void TestLevel::createSecretAreas(GameWorld* gameWorld) {
    // Hidden platforms only visible from specific angles
    bool rightOnly[4] = {false, true, false, false};
    bool backOnly[4] = {false, false, true, false};
    bool leftOnly[4] = {false, false, false, true};
    
    // Secret platform cluster (only visible from right view)
    addPlatform(gameWorld, glm::vec3(15, 3, 0), glm::vec3(2, 0.5f, 2), rightOnly);
    addPlatform(gameWorld, glm::vec3(15, 5, 2), glm::vec3(2, 0.5f, 2), rightOnly);
    addPlatform(gameWorld, glm::vec3(15, 7, 4), glm::vec3(2, 0.5f, 2), rightOnly);
    
    // Secret passage (only from back view)
    addPlatform(gameWorld, glm::vec3(0, 1, 20), glm::vec3(4, 0.5f, 4), backOnly);
    addWall(gameWorld, glm::vec3(-2, 3, 20), glm::vec3(1, 6, 4), backOnly);
    addWall(gameWorld, glm::vec3(2, 3, 20), glm::vec3(1, 6, 4), backOnly);
    
    // Reward area (only from left view)
    addPlatform(gameWorld, glm::vec3(-15, 8, 0), glm::vec3(6, 0.5f, 6), leftOnly);
}

void TestLevel::addPlatform(GameWorld* gameWorld, glm::vec3 pos, glm::vec3 size, bool views[4]) {
    auto platform = std::make_unique<Platform>(pos, size, views);
    gameWorld->addGameObject(std::move(platform));
}

void TestLevel::addWall(GameWorld* gameWorld, glm::vec3 pos, glm::vec3 size, bool views[4]) {
    auto wall = std::make_unique<Wall>(pos, size, views);
    gameWorld->addGameObject(std::move(wall));
}
