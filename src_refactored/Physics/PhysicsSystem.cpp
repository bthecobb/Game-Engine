#include "Physics/PhysicsSystem.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderComponents.h"
#include <iostream>
#include <chrono>

namespace CudaGame {
namespace Physics {

PhysicsSystem::PhysicsSystem() {}

PhysicsSystem::~PhysicsSystem() {
    Shutdown();
}

bool PhysicsSystem::Initialize() {
    std::cout << "[PhysicsSystem] Initializing physics simulation..." << std::endl;
    return true;
}

void PhysicsSystem::Shutdown() {
    std::cout << "[PhysicsSystem] Shutting down physics simulation." << std::endl;
}

void PhysicsSystem::Update(float deltaTime) {
    m_stats = {}; // Reset stats for this frame
    auto startTime = std::chrono::high_resolution_clock::now();

    m_accumulator += deltaTime;

    while (m_accumulator >= m_fixedTimeStep) {
        FixedUpdate(m_fixedTimeStep);
        m_accumulator -= m_fixedTimeStep;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    m_stats.simulationTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
}

void PhysicsSystem::FixedUpdate(float deltaTime) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();

    // 1. Apply forces and integrate velocities
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<RigidbodyComponent>(entity)) {
            auto& rigidbody = coordinator.GetComponent<RigidbodyComponent>(entity);
            if (!rigidbody.isKinematic) {
                // Apply gravity
                rigidbody.addForce(m_gravity * rigidbody.getMass());

                // Verlet integration
                glm::vec3 acceleration = rigidbody.forceAccumulator * rigidbody.inverseMass;
                rigidbody.velocity += acceleration * deltaTime;
                rigidbody.clearAccumulator();
            }
        }
    }

    // 2. Collision detection
    DetectCollisions();

    // 3. Collision resolution
    ResolveCollisions();

    // 4. Integrate positions
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<RigidbodyComponent>(entity) && coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
            auto& rigidbody = coordinator.GetComponent<RigidbodyComponent>(entity);
            auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
            transform.position += rigidbody.velocity * deltaTime;
        }
    }
}

void PhysicsSystem::DetectCollisions() {
    m_contacts.clear();
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();

    // Simple O(n^2) check for now
    for (auto const& entityA : mEntities) {
        for (auto const& entityB : mEntities) {
            if (entityA >= entityB) continue; // Avoid self-collision and duplicate checks

            if (coordinator.HasComponent<ColliderComponent>(entityA) && coordinator.HasComponent<ColliderComponent>(entityB)) {
                ContactPoint contact;
                if (TestEntityCollision(entityA, entityB, contact)) {
                    m_contacts.push_back(contact);
                }
            }
        }
    }
}

void PhysicsSystem::ResolveCollisions() {
    for (const auto& contact : m_contacts) {
        ResolveContact(contact);
    }
}

bool PhysicsSystem::TestEntityCollision(Core::Entity entityA, Core::Entity entityB, ContactPoint& contact) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    auto const& transformA = coordinator.GetComponent<Rendering::TransformComponent>(entityA);
    auto const& colliderA = coordinator.GetComponent<ColliderComponent>(entityA);
    auto const& transformB = coordinator.GetComponent<Rendering::TransformComponent>(entityB);
    auto const& colliderB = coordinator.GetComponent<ColliderComponent>(entityB);

    // For now, only AABB vs AABB
    AABB aabbA = { transformA.position + colliderA.offset - colliderA.halfExtents, transformA.position + colliderA.offset + colliderA.halfExtents };
    AABB aabbB = { transformB.position + colliderB.offset - colliderB.halfExtents, transformB.position + colliderB.offset + colliderB.halfExtents };

    if (CollisionDetection::testAABBvsAABB(aabbA, aabbB)) {
        contact.entityA = entityA;
        contact.entityB = entityB;
        // More contact info can be calculated here
        return true;
    }

    return false;
}

void PhysicsSystem::ResolveContact(const ContactPoint& contact) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    
    // Check if both entities have rigidbodies
    bool hasRbA = coordinator.HasComponent<RigidbodyComponent>(contact.entityA);
    bool hasRbB = coordinator.HasComponent<RigidbodyComponent>(contact.entityB);
    
    // Skip if neither entity has a rigidbody (two static colliders)
    if (!hasRbA && !hasRbB) return;
    
    auto& transA = coordinator.GetComponent<Rendering::TransformComponent>(contact.entityA);
    auto& transB = coordinator.GetComponent<Rendering::TransformComponent>(contact.entityB);

    // Get velocities (static objects have zero velocity)
    glm::vec3 velA = hasRbA ? coordinator.GetComponent<RigidbodyComponent>(contact.entityA).velocity : glm::vec3(0.0f);
    glm::vec3 velB = hasRbB ? coordinator.GetComponent<RigidbodyComponent>(contact.entityB).velocity : glm::vec3(0.0f);
    
    // Basic impulse resolution
    glm::vec3 relativeVelocity = velB - velA;
    glm::vec3 collisionNormal = glm::normalize(transB.position - transA.position);

    float velAlongNormal = glm::dot(relativeVelocity, collisionNormal);

    // Do not resolve if velocities are separating
    if (velAlongNormal > 0) return;

    // Get restitution (static objects have restitution of 0.5)
    float eA = hasRbA ? coordinator.GetComponent<RigidbodyComponent>(contact.entityA).restitution : 0.5f;
    float eB = hasRbB ? coordinator.GetComponent<RigidbodyComponent>(contact.entityB).restitution : 0.5f;
    float e = std::min(eA, eB);

    // Calculate impulse
    float invMassA = hasRbA ? coordinator.GetComponent<RigidbodyComponent>(contact.entityA).inverseMass : 0.0f;
    float invMassB = hasRbB ? coordinator.GetComponent<RigidbodyComponent>(contact.entityB).inverseMass : 0.0f;
    
    float j = -(1 + e) * velAlongNormal;
    j /= invMassA + invMassB;

    glm::vec3 impulse = j * collisionNormal;
    
    // Apply impulse only to entities with rigidbodies
    if (hasRbA) {
        auto& rbA = coordinator.GetComponent<RigidbodyComponent>(contact.entityA);
        rbA.velocity -= invMassA * impulse;
    }
    if (hasRbB) {
        auto& rbB = coordinator.GetComponent<RigidbodyComponent>(contact.entityB);
        rbB.velocity += invMassB * impulse;
    }

    // Positional correction to avoid sinking
    const float percent = 0.2f; // 20% to 80%
    const float slop = 0.01f; // 0.01 to 0.1
    AABB aabbA = { transA.position - coordinator.GetComponent<ColliderComponent>(contact.entityA).halfExtents, transA.position + coordinator.GetComponent<ColliderComponent>(contact.entityA).halfExtents };
    AABB aabbB = { transB.position - coordinator.GetComponent<ColliderComponent>(contact.entityB).halfExtents, transB.position + coordinator.GetComponent<ColliderComponent>(contact.entityB).halfExtents };
    glm::vec3 penetration = glm::min(aabbA.max - aabbB.min, aabbB.max - aabbA.min);
    glm::vec3 correction = (std::max)(penetration.x, (std::max)(penetration.y, penetration.z)) * percent * collisionNormal;
    
    // Apply correction only to entities with rigidbodies
    if (hasRbA) {
        transA.position -= invMassA * correction;
    }
    if (hasRbB) {
        transB.position += invMassB * correction;
    }
}

} // namespace Physics
} // namespace CudaGame
