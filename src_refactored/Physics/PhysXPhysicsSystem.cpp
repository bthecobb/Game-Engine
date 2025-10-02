#include "Physics/PhysXPhysicsSystem.h"
#include "Core/Coordinator.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/RenderComponents.h"
#include <iostream>

namespace CudaGame {
namespace Physics {

PhysXPhysicsSystem::PhysXPhysicsSystem() {}

PhysXPhysicsSystem::~PhysXPhysicsSystem() {
    Shutdown();
}

bool PhysXPhysicsSystem::Initialize() {
    std::cout << "[PhysXPhysicsSystem] Initializing PhysX physics engine..." << std::endl;

    m_pxFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, m_pxAllocator, m_pxErrorCallback);
    if (!m_pxFoundation) {
        std::cerr << "Failed to create PhysX Foundation!" << std::endl;
        return false;
    }

    // PVD disabled for now to avoid linking issues
    // m_pxPvd = PxCreatePvd(*m_pxFoundation);
    // physx::PxPvdTransport* transport = physx::PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10);
    // m_pxPvd->connect(*transport, physx::PxPvdInstrumentationFlag::eALL);

    m_pxPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *m_pxFoundation, physx::PxTolerancesScale(), true, nullptr);
    if (!m_pxPhysics) {
        std::cerr << "Failed to create PhysX Physics!" << std::endl;
        return false;
    }

    m_pxDispatcher = physx::PxDefaultCpuDispatcherCreate(2);
    m_pxDefaultMaterial = m_pxPhysics->createMaterial(0.5f, 0.5f, 0.6f);

    physx::PxSceneDesc sceneDesc(m_pxPhysics->getTolerancesScale());
    sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = m_pxDispatcher;
    sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;

    m_pxScene = m_pxPhysics->createScene(sceneDesc);

    std::cout << "[PhysXPhysicsSystem] Initialized. Managing " << mEntities.size() << " entities." << std::endl;
    return true;
}

void PhysXPhysicsSystem::Shutdown() {
    if (m_pxScene) {
        m_pxScene->release();
        m_pxScene = nullptr;
    }
    if (m_pxDispatcher) {
        m_pxDispatcher->release();
        m_pxDispatcher = nullptr;
    }
    if (m_pxPhysics) {
        m_pxPhysics->release();
        m_pxPhysics = nullptr;
    }
    // if (m_pxPvd) {
    //     physx::PxPvdTransport* transport = m_pxPvd->getTransport();
    //     m_pxPvd->release();
    //     m_pxPvd = nullptr;
    //     if (transport) transport->release();
    // }
    if (m_pxFoundation) {
        m_pxFoundation->release();
        m_pxFoundation = nullptr;
    }
}

void PhysXPhysicsSystem::Update(float deltaTime) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    
    // 1. Actor Creation: Check for new entities that need PhysX actors
    for (auto const& entity : mEntities) {
        if (m_entityToActor.find(entity) == m_entityToActor.end()) {
            // Entity doesn't have a PhysX actor yet, check if it has physics components
            if (coordinator.HasComponent<Physics::RigidbodyComponent>(entity) || 
                coordinator.HasComponent<Physics::ColliderComponent>(entity)) {
                CreatePhysXActor(entity);
                std::cout << "[DEBUG] Created PhysX actor for entity " << entity << std::endl;
            }
        }
    }
    
    // 2. Actor Destruction: Check for actors whose entities are no longer in the system
    std::vector<Core::Entity> toRemove;
    for (auto const& [entity, actor] : m_entityToActor) {
        if (mEntities.find(entity) == mEntities.end()) {
            toRemove.push_back(entity);
        }
    }
    for (auto entity : toRemove) {
        std::cout << "[DEBUG] Removing PhysX actor for entity " << entity << std::endl;
        RemovePhysXActor(entity);
    }
    
    // 2.5. Sync transforms and velocities TO PhysX before simulation
    for (auto const& [entity, actor] : m_entityToActor) {
        if (coordinator.HasComponent<Physics::RigidbodyComponent>(entity)) {
            auto& rigidbody = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
            
            // Only apply to dynamic actors
            if (actor->is<physx::PxRigidDynamic>() && !rigidbody.isKinematic) {
                physx::PxRigidDynamic* dynamicActor = static_cast<physx::PxRigidDynamic*>(actor);
                
                // Sync transform position to PhysX (respect game logic updates)
                SyncTransformToPhysX(entity, actor);
                
                // Apply velocities
                dynamicActor->setLinearVelocity(physx::PxVec3(rigidbody.velocity.x, rigidbody.velocity.y, rigidbody.velocity.z));
            }
        }
    }
    
    // 3. Physics Simulation
    m_pxScene->simulate(deltaTime);
    m_pxScene->fetchResults(true);

    // 4. Transform Synchronization: Update entity transforms with PhysX results
    // Handle transform synchronization with priority system:
    // - Kinematic entities: Game logic controls position
    // - Dynamic entities: PhysX controls position unless overridden
    // - Static entities: No updates needed
    for (auto const& [entity, actor] : m_entityToActor) {
        if (coordinator.HasComponent<Physics::RigidbodyComponent>(entity)) {
            auto& rigidbody = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
            
            if (rigidbody.isKinematic) {
                // Kinematic bodies: game logic controls position
                // We already synced to PhysX earlier, nothing to do here
                continue;
            }
            
            // Handle dynamic bodies
            if (actor->is<physx::PxRigidDynamic>()) {
                physx::PxRigidDynamic* dynamicActor = static_cast<physx::PxRigidDynamic*>(actor);
                
                // Check if game logic requested position override
                if (rigidbody.overridePhysicsTransform) {
                    // Game logic takes priority this frame
                    rigidbody.overridePhysicsTransform = false;
                    SyncTransformToPhysX(entity, actor);
                } else {
                    // Physics simulation results take priority
                    SyncTransformFromPhysX(entity, actor);
                    
                    // Update velocities from simulation
                    const physx::PxVec3& vel = dynamicActor->getLinearVelocity();
                    rigidbody.velocity = glm::vec3(vel.x, vel.y, vel.z);
                }
            }
        }
    }
}

void PhysXPhysicsSystem::CreatePhysXActor(Core::Entity entity) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    if (coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);

        physx::PxTransform pxTransform(transform.position.x, transform.position.y, transform.position.z);
        
        // Check if entity has rigidbody component to determine static vs dynamic
        physx::PxRigidActor* actor = nullptr;
        if (coordinator.HasComponent<Physics::RigidbodyComponent>(entity)) {
            auto& rigidbody = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
            if (rigidbody.isKinematic) {
                actor = m_pxPhysics->createRigidStatic(pxTransform);
            } else {
                physx::PxRigidDynamic* dynamicActor = m_pxPhysics->createRigidDynamic(pxTransform);
                dynamicActor->setMass(rigidbody.mass);
                actor = dynamicActor;
            }
        } else {
            // No rigidbody component, create static actor
            actor = m_pxPhysics->createRigidStatic(pxTransform);
        }
        
        // Create shape based on collider component
        if (coordinator.HasComponent<Physics::ColliderComponent>(entity)) {
            auto& collider = coordinator.GetComponent<Physics::ColliderComponent>(entity);
            physx::PxShape* shape = nullptr;
            
            if (collider.shape == Physics::ColliderShape::BOX) {
                physx::PxBoxGeometry boxGeom(collider.halfExtents.x, collider.halfExtents.y, collider.halfExtents.z);
                shape = m_pxPhysics->createShape(boxGeom, *m_pxDefaultMaterial);
            } else if (collider.shape == Physics::ColliderShape::SPHERE) {
                physx::PxSphereGeometry sphereGeom(collider.radius);
                shape = m_pxPhysics->createShape(sphereGeom, *m_pxDefaultMaterial);
            } else if (collider.shape == Physics::ColliderShape::CAPSULE) {
                physx::PxCapsuleGeometry capsuleGeom(collider.radius, collider.halfExtents.y);
                shape = m_pxPhysics->createShape(capsuleGeom, *m_pxDefaultMaterial);
            }
            
            if (shape) {
                actor->attachShape(*shape);
                shape->release();
            }
        }
        
        m_pxScene->addActor(*actor);
        m_entityToActor[entity] = actor;
        m_actorToEntity[actor] = entity;
    }
}

void PhysXPhysicsSystem::RemovePhysXActor(Core::Entity entity) {
    auto it = m_entityToActor.find(entity);
    if (it != m_entityToActor.end()) {
        physx::PxRigidActor* actor = it->second;
        m_pxScene->removeActor(*actor);
        actor->release();
        m_entityToActor.erase(it);
    }
}

void PhysXPhysicsSystem::SyncTransformFromPhysX(Core::Entity entity, physx::PxRigidActor* actor) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    if (coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        const physx::PxTransform& pxTransform = actor->getGlobalPose();
        glm::vec3 newPosition = glm::vec3(pxTransform.p.x, pxTransform.p.y, pxTransform.p.z);
        
        // Debug: Log player entity position every 100 frames (assume entity 0 is player)
        static int syncFrameCount = 0;
        syncFrameCount++;
        if (entity == 0 && syncFrameCount % 100 == 0) {
            std::cout << "[DEBUG] Player entity " << entity << " position: (" 
                      << newPosition.x << ", " << newPosition.y << ", " << newPosition.z 
                      << ") | PhysX actor position: (" 
                      << pxTransform.p.x << ", " << pxTransform.p.y << ", " << pxTransform.p.z << ")" 
                      << std::endl;
        }
        
        transform.position = newPosition;
    }
}

void PhysXPhysicsSystem::SyncTransformToPhysX(Core::Entity entity, physx::PxRigidActor* actor) {
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    if (coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
        auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        actor->setGlobalPose(physx::PxTransform(transform.position.x, transform.position.y, transform.position.z));
    }
}

} // namespace Physics
} // namespace CudaGame

