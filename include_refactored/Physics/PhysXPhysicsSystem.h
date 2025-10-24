#pragma once

#include "Core/System.h"
#include <PxPhysicsAPI.h>
#include <memory>
#include <unordered_map>

namespace CudaGame {
namespace Physics {

class PhysXPhysicsSystem : public Core::System {
public:
    PhysXPhysicsSystem();
    ~PhysXPhysicsSystem();

    bool Initialize();
    void Shutdown();
    void Update(float deltaTime);
    
    // Alias for test framework
    void Cleanup() { Shutdown(); }

    // PhysX scene access
    physx::PxScene* GetScene() { return m_pxScene; }
    physx::PxPhysics* GetPhysics() { return m_pxPhysics; }

private:
    // PhysX core objects
    physx::PxDefaultAllocator m_pxAllocator;
    physx::PxDefaultErrorCallback m_pxErrorCallback;
    physx::PxFoundation* m_pxFoundation = nullptr;
    physx::PxPhysics* m_pxPhysics = nullptr;
    physx::PxDefaultCpuDispatcher* m_pxDispatcher = nullptr;
    physx::PxScene* m_pxScene = nullptr;
    physx::PxMaterial* m_pxDefaultMaterial = nullptr;
    physx::PxPvd* m_pxPvd = nullptr;

    // Entity to PhysX actor mapping
    std::unordered_map<Core::Entity, physx::PxRigidActor*> m_entityToActor;
    std::unordered_map<physx::PxRigidActor*, Core::Entity> m_actorToEntity;

    // Helper methods
    void CreatePhysXActor(Core::Entity entity);
    void RemovePhysXActor(Core::Entity entity);
    void SyncTransformFromPhysX(Core::Entity entity, physx::PxRigidActor* actor);
    void SyncTransformToPhysX(Core::Entity entity, physx::PxRigidActor* actor);
};

} // namespace Physics
} // namespace CudaGame
