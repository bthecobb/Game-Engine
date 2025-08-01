#include "Physics/CudaPhysicsSystem.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderComponents.h"
#include <iostream>
#include <algorithm>

// In a real implementation, we would include CUDA headers here
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

namespace CudaGame {
namespace Physics {


CudaPhysicsSystem::CudaPhysicsSystem() : m_cudaData(std::make_unique<CudaPhysicsData>()) {
    // Physics runs early in the frame
}

CudaPhysicsSystem::~CudaPhysicsSystem() {
    Shutdown();
}

bool CudaPhysicsSystem::Initialize() {
    std::cout << "[CudaPhysicsSystem] Initializing GPU-accelerated physics..." << std::endl;
    
    // TODO: Initialize CUDA context and allocate GPU memory
    // cudaError_t result = cudaSetDevice(0);
    // if (result != cudaSuccess) {
    //     std::cerr << "[CudaPhysicsSystem] Failed to set CUDA device!" << std::endl;
    //     return false;
    // }
    
    // Allocate GPU memory for physics data
    size_t maxEntities = m_cudaData->maxEntities;
    
    // TODO: Allocate actual GPU memory
    // cudaMalloc(&m_cudaData->d_positions, maxEntities * sizeof(float3));
    // cudaMalloc(&m_cudaData->d_velocities, maxEntities * sizeof(float3));
    // cudaMalloc(&m_cudaData->d_masses, maxEntities * sizeof(float));
    // cudaMalloc(&m_cudaData->d_forces, maxEntities * sizeof(float3));
    
    // For now, simulate allocation
    m_cudaData->d_positions = (void*)1;
    m_cudaData->d_velocities = (void*)2;
    m_cudaData->d_masses = (void*)3;
    m_cudaData->d_forces = (void*)4;
    
    m_cudaData->isInitialized = true;
    
    std::cout << "[CudaPhysicsSystem] GPU physics system initialized successfully." << std::endl;
    return true;
}

void CudaPhysicsSystem::Shutdown() {
    if (!m_cudaData->isInitialized) return;
    
    std::cout << "[CudaPhysicsSystem] Shutting down GPU physics..." << std::endl;
    
    // TODO: Free GPU memory
    // cudaFree(m_cudaData->d_positions);
    // cudaFree(m_cudaData->d_velocities);
    // cudaFree(m_cudaData->d_masses);
    // cudaFree(m_cudaData->d_forces);
    
    m_cudaData->isInitialized = false;
    
    std::cout << "[CudaPhysicsSystem] GPU physics shutdown complete." << std::endl;
}

void CudaPhysicsSystem::Update(float deltaTime) {
    if (!m_cudaData->isInitialized || m_cudaData->currentEntityCount == 0) {
        return;
    }
    
    // Fixed timestep physics integration
    m_accumulator += deltaTime;
    
    while (m_accumulator >= m_fixedTimeStep) {
        // Run physics simulation substeps on GPU
        for (int substep = 0; substep < m_substepCount; ++substep) {
            float substepDt = m_fixedTimeStep / m_substepCount;
            
            // TODO: Launch CUDA kernels for physics integration
            // LaunchIntegrateForces(substepDt);
            // LaunchUpdatePositions(substepDt);
            // LaunchDetectCollisions();
            // LaunchResolveCollisions();
            
            // Simulate physics work
            std::cout << "[CudaPhysicsSystem] Processing " << m_cudaData->currentEntityCount 
                      << " physics entities on GPU (substep " << substep + 1 << "/" << m_substepCount << ")" << std::endl;
        }
        
        m_accumulator -= m_fixedTimeStep;
    }
    
    // Synchronize results back to ECS
    SynchronizeWithECS();
}

void CudaPhysicsSystem::RegisterEntity(Core::Entity entity, const RigidbodyComponent& rb, const ColliderComponent& collider) {
    if (m_cudaData->currentEntityCount >= m_cudaData->maxEntities) {
        std::cerr << "[CudaPhysicsSystem] Maximum entity count reached!" << std::endl;
        return;
    }
    
    // Check if entity is already registered
    auto it = std::find(m_cudaData->entities.begin(), m_cudaData->entities.end(), entity);
    if (it != m_cudaData->entities.end()) {
        std::cout << "[CudaPhysicsSystem] Entity " << entity << " already registered." << std::endl;
        return;
    }
    
    // Add entity to CUDA physics simulation
    m_cudaData->entities.push_back(entity);
    m_cudaData->rigidbodies.push_back(rb);
    m_cudaData->colliders.push_back(collider);
    m_cudaData->currentEntityCount++;
    
    // TODO: Upload entity data to GPU
    // UpdateEntityDataOnGPU(m_cudaData->currentEntityCount - 1, rb, collider);
    
    std::cout << "[CudaPhysicsSystem] Registered entity " << entity << " for GPU physics simulation." << std::endl;
}

void CudaPhysicsSystem::UnregisterEntity(Core::Entity entity) {
    auto it = std::find(m_cudaData->entities.begin(), m_cudaData->entities.end(), entity);
    if (it == m_cudaData->entities.end()) {
        return;
    }
    
    size_t index = std::distance(m_cudaData->entities.begin(), it);
    
    // Remove entity from lists (swap and pop for O(1) removal)
    if (index < m_cudaData->currentEntityCount - 1) {
        std::swap(m_cudaData->entities[index], m_cudaData->entities[m_cudaData->currentEntityCount - 1]);
        std::swap(m_cudaData->rigidbodies[index], m_cudaData->rigidbodies[m_cudaData->currentEntityCount - 1]);
        std::swap(m_cudaData->colliders[index], m_cudaData->colliders[m_cudaData->currentEntityCount - 1]);
    }
    
    m_cudaData->entities.pop_back();
    m_cudaData->rigidbodies.pop_back();
    m_cudaData->colliders.pop_back();
    m_cudaData->currentEntityCount--;
    
    std::cout << "[CudaPhysicsSystem] Unregistered entity " << entity << " from GPU physics simulation." << std::endl;
}

void CudaPhysicsSystem::SetGravity(const glm::vec3& gravity) {
    m_gravity = gravity;
    // TODO: Update gravity on GPU
    std::cout << "[CudaPhysicsSystem] Updated gravity to (" << gravity.x << ", " << gravity.y << ", " << gravity.z << ")" << std::endl;
}

void CudaPhysicsSystem::SetSubstepCount(int count) {
    m_substepCount = std::max(1, count);
    std::cout << "[CudaPhysicsSystem] Set substep count to " << m_substepCount << std::endl;
}

bool CudaPhysicsSystem::Raycast(const glm::vec3& origin, const glm::vec3& direction, float maxDistance) {
    // TODO: Implement GPU-accelerated raycasting
    // This would involve launching a CUDA kernel to test ray against all colliders in parallel
    return false;
}

void CudaPhysicsSystem::SynchronizeWithECS() {
    // Simple CPU-based gravity simulation for demo purposes
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    
    for (size_t i = 0; i < m_cudaData->currentEntityCount; ++i) {
        Core::Entity entity = m_cudaData->entities[i];
        
        // Apply basic gravity to objects with mass > 0
        if (coordinator.HasComponent<CudaGame::Physics::RigidbodyComponent>(entity) && 
            coordinator.HasComponent<CudaGame::Rendering::TransformComponent>(entity)) {
            
            auto& rb = coordinator.GetComponent<CudaGame::Physics::RigidbodyComponent>(entity);
            auto& transform = coordinator.GetComponent<CudaGame::Rendering::TransformComponent>(entity);
            
            if (rb.getMass() > 0.0f) {
                // Simple gravity integration (Euler method)
                float dt = m_fixedTimeStep / m_substepCount;
                
                // Apply gravity acceleration
                glm::vec3 gravityAccel = m_gravity; // Default is (0, -9.81, 0)
                glm::vec3 velocity = rb.getVelocity();
                velocity += gravityAccel * dt;
                rb.setVelocity(velocity);
                
                // Update position
                transform.position += velocity * dt;
                
                // Simple ground collision (y = 0.5 to account for sphere radius)
                if (transform.position.y < 0.5f) {
                    transform.position.y = 0.5f;
                    velocity.y = -velocity.y * 0.6f; // Bounce with damping
                    rb.setVelocity(velocity);
                }
            }
        }
    }
}

} // namespace Physics
} // namespace CudaGame
