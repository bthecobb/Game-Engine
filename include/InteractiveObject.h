#pragma once

#include "GameWorld.h"
#include <glm/glm.hpp>

enum class ObjectType {
    DESTRUCTIBLE_BOX,
    MOVEABLE_CRATE,
    EXPLOSIVE_BARREL,
    BOUNCE_PAD,
    SPEED_BOOST,
    COLLECTIBLE
};

enum class ObjectState {
    INTACT,
    DAMAGED,
    DESTROYED,
    MOVING,
    EXPLODING
};

class InteractiveObject : public GameObject {
public:
    InteractiveObject(ObjectType type, glm::vec3 position, glm::vec3 size);
    ~InteractiveObject() override;
    
    // GameObject interface
    void update(float deltaTime) override;
    void render() override;
    
    // Interaction methods
    virtual void onPlayerCollision(const glm::vec3& playerVelocity, float playerSpeed);
    virtual void onPlayerAttack(float attackPower, const glm::vec3& attackDirection);
    virtual void takeDamage(float damage);
    virtual void destroy();
    
    // Physics
    void applyForce(const glm::vec3& force);
    void setVelocity(const glm::vec3& velocity) { m_velocity = velocity; }
    glm::vec3 getVelocity() const { return m_velocity; }
    
    // Getters
    ObjectType getType() const { return m_type; }
    ObjectState getState() const { return m_state; }
    float getHealth() const { return m_health; }
    bool isDestroyed() const { return m_state == ObjectState::DESTROYED; }
    bool canMove() const { return m_canMove; }
    bool canDestroy() const { return m_canDestroy; }
    
    // Setters
    void setHealth(float health) { m_health = health; }
    
private:
    // Object properties
    ObjectType m_type;
    ObjectState m_state;
    float m_health;
    float m_maxHealth;
    bool m_canMove;
    bool m_canDestroy;
    
    // Physics properties
    glm::vec3 m_velocity{0.0f};
    float m_mass;
    float m_friction;
    float m_restitution; // Bounciness
    bool m_isKinematic; // Affected by physics
    
    // Visual properties
    glm::vec3 m_baseColor;
    glm::vec3 m_currentColor;
    float m_damageFlashTimer;
    float m_destructionTimer;
    
    // Rendering
    unsigned int m_VAO, m_VBO, m_EBO;
    unsigned int m_shaderProgram;
    
    // Internal methods
    void initializeObject();
    void updatePhysics(float deltaTime);
    void updateVisuals(float deltaTime);
    void createMesh();
    void createShaders();
    void updateColor();
    
    // Type-specific behaviors
    void updateDestructibleBox(float deltaTime);
    void updateMoveableCrate(float deltaTime);
    void updateExplosiveBarrel(float deltaTime);
    void updateBouncePad(float deltaTime);
    void updateSpeedBoost(float deltaTime);
    void updateCollectible(float deltaTime);
};
