#pragma once

#include "ECS_Types.h"
#include "System.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <set>
#include <typeinfo>
#include <cassert>

namespace CudaGame {
namespace Core {

// The SystemManager manages all systems and their update order
class SystemManager {
public:
    template<typename T>
    std::shared_ptr<T> RegisterSystem() {
        const char* typeName = typeid(T).name();

        assert(mSystems.find(typeName) == mSystems.end() && "Registering system more than once.");

        // Create a pointer to the system and return it so it can be used externally
        auto system = std::make_shared<T>();
        mSystems.insert({typeName, system});
        
        return system;
    }

    template<typename T>
    void SetSystemSignature(Signature signature) {
        const char* typeName = typeid(T).name();

        assert(mSystems.find(typeName) != mSystems.end() && "System used before registered.");

        // Set the signature for this system
        mSignatures.insert({typeName, signature});
    }

    void EntityDestroyed(Entity entity) {
        // Erase a destroyed entity from all system lists
        for (auto const& pair : mSystems) {
            auto const& system = pair.second;
            system->mEntities.erase(entity);
        }
    }

    void EntitySignatureChanged(Entity entity, Signature entitySignature) {
        // Notify each system that an entity's signature changed
        for (auto const& pair : mSystems) {
            auto const& type = pair.first;
            auto const& system = pair.second;
            auto const& systemSignature = mSignatures[type];

            // Entity signature matches system signature - insert into set
            if ((entitySignature & systemSignature) == systemSignature) {
                system->mEntities.insert(entity);
            }
            // Entity signature does not match system signature - erase from set
            else {
                system->mEntities.erase(entity);
            }
        }
    }

    // Initialize all systems
    bool InitializeAllSystems() {
        for (auto const& pair : mSystems) {
            auto const& system = pair.second;
            if (!system->Initialize()) {
                return false;
            }
        }
        return true;
    }

    // Shutdown all systems
    void ShutdownAllSystems() {
        for (auto const& pair : mSystems) {
            auto const& system = pair.second;
            system->Shutdown();
        }
    }

    // Update all systems in priority order
    void UpdateAllSystems(float deltaTime) {
        for (auto const& pair : mSystems) {
            auto const& system = pair.second;
            if (system->IsEnabled()) {
                system->Update(deltaTime);
            }
        }
    }

    // Late update all systems in priority order
    void LateUpdateAllSystems(float deltaTime) {
        
    }

    template<typename T>
    std::shared_ptr<T> GetSystem() {
        const char* typeName = typeid(T).name();

        auto it = mSystems.find(typeName);
        if (it != mSystems.end()) {
            return std::static_pointer_cast<T>(it->second);
        }
        return nullptr;
    }

    // Get system statistics
    size_t GetSystemCount() const { return mSystems.size(); }
    
    // Debug: Get all system names and entity counts
    std::vector<std::pair<std::string, size_t>> GetSystemDebugInfo() const {
        std::vector<std::pair<std::string, size_t>> info;
        for (auto const& pair : mSystems) {
            // info.emplace_back(pair.first, pair.second->GetEntityCount());
        }
        return info;
    }

private:
    // Map from system type string pointer to a signature
    std::unordered_map<const char*, Signature> mSignatures{};

    // Map from system type string pointer to a system pointer
    std::unordered_map<const char*, std::shared_ptr<System>> mSystems{};
};

} // namespace Core
} // namespace CudaGame
