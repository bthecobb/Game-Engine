#pragma once

#include "Core/ECS_Types.h"
#include "Physics/PhysicsComponents.h"
#include <vector>

namespace CudaGame {
namespace Physics {

// CUDA-specific data structure (PIMPL pattern to hide CUDA dependencies)
struct CudaPhysicsData {
    // GPU memory pointers
    void* d_positions = nullptr;
    void* d_velocities = nullptr;
    void* d_masses = nullptr;
    void* d_forces = nullptr;
    
    // Entity mapping
    std::vector<Core::Entity> entities;
    std::vector<RigidbodyComponent> rigidbodies;
    std::vector<ColliderComponent> colliders;
    
    size_t maxEntities = 10000;
    size_t currentEntityCount = 0;
    
    bool isInitialized = false;
};

} // namespace Physics
} // namespace CudaGame

