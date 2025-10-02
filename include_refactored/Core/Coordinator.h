#pragma once

#include "ECS_Types.h"
#include "EntityManager.h"
#include "ComponentManager.h"
#include "SystemManager.h"
#include <memory>

namespace CudaGame {
namespace Core {

// The Coordinator is the main entry point to the ECS. It delegates all the work to the managers.
class Coordinator {
public:
    static Coordinator& GetInstance() {
        static Coordinator instance;
        return instance;
    }
    
    Coordinator() {
        // Create pointers to each manager
        mEntityManager = std::make_unique<EntityManager>();
        mComponentManager = std::make_unique<ComponentManager>();
        mSystemManager = std::make_unique<SystemManager>();
    }
    
    void Initialize() {
        // Initialize managers if needed
    }

    void Cleanup() {
        // Clear systems and components for test runs
        mSystemManager = std::make_unique<SystemManager>();
        mComponentManager = std::make_unique<ComponentManager>();
        mEntityManager = std::make_unique<EntityManager>();
    }

    // Entity methods
    Entity CreateEntity() {
        return mEntityManager->CreateEntity();
    }

    void DestroyEntity(Entity entity) {
        mEntityManager->DestroyEntity(entity);
        mComponentManager->EntityDestroyed(entity);
        mSystemManager->EntityDestroyed(entity);
    }

    // Component methods
    template<typename T>
    void RegisterComponent() {
        mComponentManager->RegisterComponent<T>();
    }

    template<typename T>
    void AddComponent(Entity entity, T component) {
        mComponentManager->AddComponent<T>(entity, component);

        auto signature = mEntityManager->GetSignature(entity);
        signature.set(mComponentManager->GetComponentType<T>(), true);
        mEntityManager->SetSignature(entity, signature);

        mSystemManager->EntitySignatureChanged(entity, signature);
    }

    template<typename T>
    void RemoveComponent(Entity entity) {
        mComponentManager->RemoveComponent<T>(entity);

        auto signature = mEntityManager->GetSignature(entity);
        signature.set(mComponentManager->GetComponentType<T>(), false);
        mEntityManager->SetSignature(entity, signature);

        mSystemManager->EntitySignatureChanged(entity, signature);
    }

    template<typename T>
    T& GetComponent(Entity entity) {
        return mComponentManager->GetComponent<T>(entity);
    }
    
    template<typename T>
    bool HasComponent(Entity entity) {
        auto signature = mEntityManager->GetSignature(entity);
        return signature.test(mComponentManager->GetComponentType<T>());
    }

    template<typename T>
    ComponentType GetComponentType() {
        return mComponentManager->GetComponentType<T>();
    }

// System methods
    // Assets directory macro (string literal) will be provided via CMake
#ifndef ASSET_DIR
#define ASSET_DIR ""
#endif

// Note: This macro can be used at runtime for asset loading
    template<typename T>
    std::shared_ptr<T> RegisterSystem() {
        return mSystemManager->RegisterSystem<T>();
    }

    template<typename T>
    void SetSystemSignature(Signature signature) {
        mSystemManager->SetSystemSignature<T>(signature);
    }

    template<typename T>
    std::shared_ptr<T> GetSystem() {
        return mSystemManager->GetSystem<T>();
    }

    void UpdateSystems(float deltaTime) {
        mSystemManager->UpdateAllSystems(deltaTime);
    }

    void LateUpdateSystems(float deltaTime) {
        mSystemManager->LateUpdateAllSystems(deltaTime);
    }

    // Getters for managers
    EntityManager* GetEntityManager() { return mEntityManager.get(); }
    ComponentManager* GetComponentManager() { return mComponentManager.get(); }
    SystemManager* GetSystemManager() { return mSystemManager.get(); }

private:
    std::unique_ptr<EntityManager> mEntityManager;
    std::unique_ptr<ComponentManager> mComponentManager;
    std::unique_ptr<SystemManager> mSystemManager;
};

} // namespace Core
} // namespace CudaGame
