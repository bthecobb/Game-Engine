#include "GameWorld.h"
#include <iostream>
#include <algorithm>

// Transform3D implementation
glm::mat4 Transform3D::getMatrix() const {
    glm::mat4 translate = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 rotate = glm::rotate(glm::mat4(1.0f), rotation.z, glm::vec3(0, 0, 1));
    rotate = glm::rotate(rotate, rotation.y, glm::vec3(0, 1, 0));
    rotate = glm::rotate(rotate, rotation.x, glm::vec3(1, 0, 0));
    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
    return translate * rotate * scaleMatrix;
}

// GameObject implementation
bool GameObject::isVisibleFromView(ViewDirection view) const {
    return activeFromView[static_cast<int>(view)];
}

bool GameObject::isCollidableFromView(ViewDirection view) const {
    return collidableFromView[static_cast<int>(view)];
}

// GameWorld implementation
GameWorld::GameWorld() 
    : m_currentView(ViewDirection::FRONT), m_rotationProgress(0.0f), m_isRotating(false),
      m_cameraPosition(0.0f, 0.0f, 10.0f), m_cameraTarget(0.0f, 0.0f, 0.0f),
      m_orthoSize(10.0f), m_nearPlane(0.1f), m_farPlane(100.0f) {
}

GameWorld::~GameWorld() = default;

void GameWorld::initialize() {
    std::cout << "GameWorld initialized with dimensional system" << std::endl;
    updateCamera();
}

void GameWorld::update(float deltaTime) {
    // Update rotation animation
    if (m_isRotating) {
        m_rotationProgress += deltaTime * 4.0f; // Rotation takes 0.25 seconds
        if (m_rotationProgress >= 1.0f) {
            m_rotationProgress = 1.0f;
            m_isRotating = false;
        }
        updateCamera();
    }
    
    // Update all game objects
    for (auto& object : m_gameObjects) {
        if (object->isVisibleFromView(m_currentView)) {
            object->update(deltaTime);
        }
    }
}

void GameWorld::render() {
    // Render all visible objects
    for (auto& object : m_gameObjects) {
        if (object->isVisibleFromView(m_currentView)) {
            object->render();
        }
    }
}

void GameWorld::rotateWorld(bool clockwise) {
    if (m_isRotating) return; // Already rotating
    
    // Calculate next view
    int currentIndex = static_cast<int>(m_currentView);
    int nextIndex = clockwise ? (currentIndex + 1) % 4 : (currentIndex + 3) % 4;
    m_currentView = static_cast<ViewDirection>(nextIndex);
    
    // Start rotation animation
    m_isRotating = true;
    m_rotationProgress = 0.0f;
    
    std::cout << "Rotating world to view: " << nextIndex << std::endl;
}

glm::mat4 GameWorld::getViewMatrix() const {
    glm::vec3 eye = m_cameraPosition;
    glm::vec3 center = m_cameraTarget;
    glm::vec3 up = getViewUp(m_currentView);
    
    // Apply rotation animation if in progress
    if (m_isRotating) {
        // Interpolate between current and target camera positions
        float t = glm::smoothstep(0.0f, 1.0f, m_rotationProgress);
        // Add smooth rotation interpolation here
    }
    
    return glm::lookAt(eye, center, up);
}

glm::mat4 GameWorld::getProjectionMatrix() const {
    // Orthographic projection for 2.5D gameplay
    float aspect = 1200.0f / 800.0f; // Window aspect ratio
    return glm::ortho(-m_orthoSize * aspect, m_orthoSize * aspect, 
                     -m_orthoSize, m_orthoSize, m_nearPlane, m_farPlane);
}

void GameWorld::addGameObject(std::unique_ptr<GameObject> object) {
    m_gameObjects.push_back(std::move(object));
}

std::vector<GameObject*> GameWorld::getVisibleObjects() const {
    std::vector<GameObject*> visibleObjects;
    for (const auto& object : m_gameObjects) {
        if (object->isVisibleFromView(m_currentView)) {
            visibleObjects.push_back(object.get());
        }
    }
    return visibleObjects;
}

std::vector<GameObject*> GameWorld::getCollidableObjects() const {
    std::vector<GameObject*> collidableObjects;
    for (const auto& object : m_gameObjects) {
        if (object->isCollidableFromView(m_currentView)) {
            collidableObjects.push_back(object.get());
        }
    }
    return collidableObjects;
}

bool GameWorld::checkCollision(const glm::vec3& position, const glm::vec3& size) const {
    // Simple AABB collision detection
    auto collidableObjects = getCollidableObjects();
    for (const auto* object : collidableObjects) {
        // Implement AABB vs AABB collision
        // This is a simplified version - you'd want proper collision detection
        glm::vec3 objPos = object->transform.position;
        glm::vec3 objSize = object->transform.scale;
        
        if (position.x - size.x/2 < objPos.x + objSize.x/2 &&
            position.x + size.x/2 > objPos.x - objSize.x/2 &&
            position.y - size.y/2 < objPos.y + objSize.y/2 &&
            position.y + size.y/2 > objPos.y - objSize.y/2) {
            return true;
        }
    }
    return false;
}

glm::vec3 GameWorld::resolveCollision(const glm::vec3& position, const glm::vec3& velocity, const glm::vec3& size) const {
    // Basic collision resolution
    glm::vec3 resolvedVelocity = velocity;
    
    // Check if the next position would cause a collision
    glm::vec3 nextPosition = position + velocity * 0.016f; // Assume 60fps
    if (checkCollision(nextPosition, size)) {
        // Simple approach: zero out velocity component causing collision
        resolvedVelocity = glm::vec3(0.0f);
    }
    
    return resolvedVelocity;
}

void GameWorld::updateCamera() {
    glm::vec3 direction = getViewDirection(m_currentView);
    glm::vec3 up = getViewUp(m_currentView);
    
    // Position camera based on current view
    m_cameraPosition = -direction * 15.0f; // Distance from origin
    m_cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
}

glm::vec3 GameWorld::getViewDirection(ViewDirection view) const {
    switch (view) {
        case ViewDirection::FRONT: return glm::vec3(0, 0, -1); // Look down -Z
        case ViewDirection::RIGHT: return glm::vec3(-1, 0, 0); // Look down -X  
        case ViewDirection::BACK:  return glm::vec3(0, 0, 1);  // Look down +Z
        case ViewDirection::LEFT:  return glm::vec3(1, 0, 0);  // Look down +X
        default: return glm::vec3(0, 0, -1);
    }
}

glm::vec3 GameWorld::getViewUp(ViewDirection view) const {
    // Y is always up in our coordinate system
    return glm::vec3(0, 1, 0);
}
