#pragma once

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum class ViewDirection {
    FRONT = 0,  // Looking down -Z axis (XY plane)
    RIGHT = 1,  // Looking down -X axis (ZY plane) 
    BACK = 2,   // Looking down +Z axis (XY plane, flipped)
    LEFT = 3    // Looking down +X axis (ZY plane, flipped)
};

struct Transform3D {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
    
    glm::mat4 getMatrix() const;
};

struct GameObject {
    Transform3D transform;
    bool activeFromView[4] = {true, true, true, true}; // Visible from which views
    bool collidableFromView[4] = {true, true, true, true}; // Solid from which views
    
    virtual ~GameObject() = default;
    virtual void update(float deltaTime) = 0;
    virtual void render() = 0;
    virtual bool isVisibleFromView(ViewDirection view) const;
    virtual bool isCollidableFromView(ViewDirection view) const;
};

class GameWorld {
public:
    GameWorld();
    ~GameWorld();
    
    // World management
    void initialize();
    void update(float deltaTime);
    void render();
    
    // Dimension system
    void rotateWorld(bool clockwise);
    ViewDirection getCurrentView() const { return m_currentView; }
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    
    // Object management
    void addGameObject(std::unique_ptr<GameObject> object);
    std::vector<GameObject*> getVisibleObjects() const;
    std::vector<GameObject*> getCollidableObjects() const;
    
    // Physics queries
    bool checkCollision(const glm::vec3& position, const glm::vec3& size) const;
    glm::vec3 resolveCollision(const glm::vec3& position, const glm::vec3& velocity, const glm::vec3& size) const;

private:
    ViewDirection m_currentView;
    float m_rotationProgress; // 0.0 to 1.0 during rotation animation
    bool m_isRotating;
    
    std::vector<std::unique_ptr<GameObject>> m_gameObjects;
    
    // Camera/projection settings
    glm::vec3 m_cameraPosition;
    glm::vec3 m_cameraTarget;
    float m_orthoSize;
    float m_nearPlane, m_farPlane;
    
    void updateCamera();
    glm::vec3 getViewDirection(ViewDirection view) const;
    glm::vec3 getViewUp(ViewDirection view) const;
};
