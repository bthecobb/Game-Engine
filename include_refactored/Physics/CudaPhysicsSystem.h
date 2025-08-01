#pragma once

#include "Core/System.h"
#include "Physics/PhysicsComponents.h"
#include "Physics/CudaPhysicsData.h"
#include <glm/glm.hpp>
#include <memory>

namespace CudaGame {
namespace Physics {

// Manages GPU-accelerated physics simulations using CUDA
class CudaPhysicsSystem : public Core::System {
public:
    CudaPhysicsSystem();
    ~CudaPhysicsSystem() override;

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Registering physics objects with the CUDA system
    void RegisterEntity(Core::Entity entity, const RigidbodyComponent& rb, const ColliderComponent& collider);
    void UnregisterEntity(Core::Entity entity);

    // Simulation control
    void SetGravity(const glm::vec3& gravity);
    void SetSubstepCount(int count);

    // Raycasting and collision queries (to be implemented)
    bool Raycast(const glm::vec3& origin, const glm::vec3& direction, float maxDistance);

private:
    void SynchronizeWithECS();

    // Pointer to the CUDA-specific data and implementation (PIMPL idiom)
    std::unique_ptr<CudaPhysicsData> m_cudaData;

    // Physics simulation parameters
    glm::vec3 m_gravity = {0.0f, -9.81f, 0.0f};
    int m_substepCount = 4;
    float m_fixedTimeStep = 1.0f / 60.0f;
    float m_accumulator = 0.0f;
};

} // namespace Physics
} // namespace CudaGame
