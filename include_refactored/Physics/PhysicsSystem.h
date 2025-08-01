#pragma once

#include "Core/System.h"
#include "Physics/PhysicsComponents.h"
#include "Physics/CollisionDetection.h"
#include <vector>
#include <functional>

namespace CudaGame {
namespace Physics {

// Collision callback function type
using CollisionCallback = std::function<void(Core::Entity entityA, Core::Entity entityB, const ContactPoint& contact)>;

class PhysicsSystem : public Core::System {
public:
    PhysicsSystem();
    ~PhysicsSystem();

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // World settings
    void SetGravity(const glm::vec3& gravity) { m_gravity = gravity; }
    const glm::vec3& GetGravity() const { return m_gravity; }
    
    void SetDamping(float linearDamping, float angularDamping) {
        m_linearDamping = linearDamping;
        m_angularDamping = angularDamping;
    }

    // Collision callbacks
    void RegisterCollisionCallback(CollisionCallback callback);
    void RegisterTriggerCallback(CollisionCallback callback);

    // Force application
    void ApplyForce(Core::Entity entity, const glm::vec3& force);
    void ApplyImpulse(Core::Entity entity, const glm::vec3& impulse);
    void ApplyForceAtPosition(Core::Entity entity, const glm::vec3& force, const glm::vec3& position);

    // Raycasting
    bool Raycast(const CollisionDetection::Ray& ray, CollisionDetection::RaycastHit& hit);
    std::vector<CollisionDetection::RaycastHit> RaycastAll(const CollisionDetection::Ray& ray);

    // Overlap testing
    bool OverlapSphere(const glm::vec3& center, float radius, std::vector<Core::Entity>& overlappingEntities);
    bool OverlapBox(const glm::vec3& center, const glm::vec3& halfExtents, std::vector<Core::Entity>& overlappingEntities);

    // Debug visualization
    void SetDebugVisualization(bool enable) { m_debugVisualization = enable; }
    void DrawDebugColliders();

    // Physics material system
    struct PhysicsMaterial {
        float friction = 0.5f;
        float restitution = 0.5f;
        float density = 1.0f;
    };
    
    void RegisterPhysicsMaterial(const std::string& name, const PhysicsMaterial& material);
    PhysicsMaterial* GetPhysicsMaterial(const std::string& name);

    // Performance settings
    void SetMaxSubsteps(int maxSubsteps) { m_maxSubsteps = maxSubsteps; }
    void SetFixedTimeStep(float timeStep) { m_fixedTimeStep = timeStep; }
    
    // Statistics
    struct PhysicsStats {
        int activeRigidbodies = 0;
        int collisionTests = 0;
        int collisionsDetected = 0;
        float simulationTime = 0.0f;
    };
    
    const PhysicsStats& GetStats() const { return m_stats; }

private:
    // World settings
    glm::vec3 m_gravity{0.0f, -9.81f, 0.0f};
    float m_linearDamping = 0.1f;
    float m_angularDamping = 0.1f;
    
    // Time stepping
    float m_fixedTimeStep = 1.0f / 60.0f;
    int m_maxSubsteps = 3;
    float m_accumulator = 0.0f;
    
    // Collision detection
    std::vector<ContactPoint> m_contacts;
    std::vector<ContactPoint> m_triggerContacts;
    
    // Callbacks
    std::vector<CollisionCallback> m_collisionCallbacks;
    std::vector<CollisionCallback> m_triggerCallbacks;
    
    // Physics materials
    std::unordered_map<std::string, PhysicsMaterial> m_physicsMaterials;
    
    // Debug visualization
    bool m_debugVisualization = false;
    
    // Statistics
    PhysicsStats m_stats;
    
    // Internal simulation methods
    void FixedUpdate(float deltaTime);
    void IntegrateVelocities(float deltaTime);
    void DetectCollisions();
    void ResolveCollisions();
    void IntegratePositions(float deltaTime);
    
    // Collision detection helpers
    void BroadPhaseCollisionDetection();
    void NarrowPhaseCollisionDetection();
    bool TestEntityCollision(Core::Entity entityA, Core::Entity entityB, ContactPoint& contact);
    
    // Collision resolution
    void ResolveContact(const ContactPoint& contact);
    void SeparateEntities(const ContactPoint& contact);
    
    // Spatial partitioning for broad phase collision detection
    struct SpatialGrid {
        static const int GRID_SIZE = 32;
        std::vector<Core::Entity> cells[GRID_SIZE][GRID_SIZE][GRID_SIZE];
        glm::vec3 cellSize{10.0f};
        glm::vec3 worldMin{-160.0f, -160.0f, -160.0f};
        
        void clear();
        void insert(Core::Entity entity, const AABB& bounds);
        void query(const AABB& bounds, std::vector<Core::Entity>& results);
        glm::ivec3 worldToGrid(const glm::vec3& worldPos);
    };
    
    SpatialGrid m_spatialGrid;
    
    // Helper functions
    AABB GetEntityAABB(Core::Entity entity);
    Sphere GetEntitySphere(Core::Entity entity);
    Capsule GetEntityCapsule(Core::Entity entity);
    
    // Integration methods
    void EulerIntegration(RigidbodyComponent& rigidbody, const glm::vec3& position, float deltaTime);
    void VerletIntegration(RigidbodyComponent& rigidbody, const glm::vec3& position, float deltaTime);
    
    // Constraint solving (for advanced physics)
    void SolveConstraints();
    
    // Character controller specific methods
    void UpdateCharacterControllers(float deltaTime);
    glm::vec3 ResolveCharacterMovement(Core::Entity entity, const glm::vec3& desiredMovement);
};

} // namespace Physics
} // namespace CudaGame
