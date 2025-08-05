#pragma once

#include "ECS_Types.h"
#include <set>

namespace CudaGame {
namespace Core {

// Base class for all engine systems
class System {
public:
    virtual ~System() = default;

    // System lifecycle
    virtual bool Initialize() = 0;
    virtual void Shutdown() = 0;
    virtual void Update(float deltaTime) = 0;

    // Called when an entity is added to this system
    virtual void OnEntityAdded(Entity entity) {}
    
    // Called when an entity is removed from this system
    virtual void OnEntityRemoved(Entity entity) {}

    // System enabled/disabled state
    void SetEnabled(bool enabled) { m_enabled = enabled; }
    bool IsEnabled() const { return m_enabled; }

    // Get the signature for entities this system should process
    Signature GetSignature() const { return m_signature; }

    // Entity set that this system processes
    std::set<Entity> mEntities;

protected:
    // Systems should set their signature during construction
    void SetSignature(const Signature& signature) { m_signature = signature; }

private:
    bool m_enabled = true;
    Signature m_signature;
};

} // namespace Core
} // namespace CudaGame
