#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>

namespace CudaGame {
namespace Physics {

// Rigidbody component for dynamic physics simulation
struct RigidbodyComponent {
    glm::vec3 velocity{0.0f};
    glm::vec3 acceleration{0.0f};
    glm::vec3 forceAccumulator{0.0f}; // Accumulated forces for this frame
    
    float mass = 1.0f;
    float inverseMass = 1.0f; // Pre-calculated for efficiency
    float restitution = 0.8f; // Bounciness
    float friction = 0.1f;
    bool isKinematic = false; // Kinematic objects are not affected by forces

    void setMass(float newMass) {
        mass = newMass;
        if (mass != 0.0f) {
            inverseMass = 1.0f / mass;
        } else {
            inverseMass = 0.0f;
        }
    }

    void addForce(const glm::vec3& force) {
        forceAccumulator += force;
    }

    void clearAccumulator() {
        forceAccumulator = glm::vec3(0.0f);
    }

    glm::vec3 getVelocity() const {
        return velocity;
    }

    void setVelocity(const glm::vec3& newVelocity) {
        velocity = newVelocity;
    }

    float getMass() const {
        return mass;
    }
};

// Collider component for collision detection
enum class ColliderShape { BOX, SPHERE, CAPSULE };

struct ColliderComponent {
    ColliderShape shape = ColliderShape::BOX;
    glm::vec3 size{1.0f, 1.0f, 1.0f}; // Size for box colliders
    
    // For BOX colliders
    glm::vec3 halfExtents{0.5f, 0.5f, 0.5f};
    
    // For SPHERE colliders
    float radius = 0.5f;
    
    // For CAPSULE colliders
    float capsuleHeight = 1.0f;
    float capsuleRadius = 0.5f;
    
    // General properties
    glm::vec3 offset{0.0f}; // Offset from the entity's transform
    bool isTrigger = false; // Trigger colliders detect overlaps but don't resolve collisions
    
    // Collision layer/mask for filtering interactions
    uint32_t collisionLayer = 0; 
    uint32_t collisionMask = 0;  
};


} // namespace Physics
} // namespace CudaGame
