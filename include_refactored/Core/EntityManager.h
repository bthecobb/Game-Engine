#pragma once

#include "ECS_Types.h"
#include <array>
#include <queue>
#include <cassert>

namespace CudaGame {
namespace Core {

// The EntityManager is responsible for distributing entity IDs and keeping track of which IDs are in use.
class EntityManager {
public:
    EntityManager();
    
    Entity CreateEntity();
    void DestroyEntity(Entity entity);
    void SetSignature(Entity entity, Signature signature);
    Signature GetSignature(Entity entity);
    
    // For debugging and statistics
    uint32_t GetLivingEntityCount() const { return mLivingEntityCount; }
    bool IsEntityAlive(Entity entity) const;
    
private:
    // Queue of unused entity IDs
    std::queue<Entity> mAvailableEntities{};
    
    // Array of signatures where the index corresponds to the entity ID
    std::array<Signature, MAX_ENTITIES> mSignatures{};
    
    // Total living entities - used to keep limits on how many exist
    uint32_t mLivingEntityCount{};
};

} // namespace Core
} // namespace CudaGame
