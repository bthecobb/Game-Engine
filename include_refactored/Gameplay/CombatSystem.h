#pragma once

#include "Core/System.h"
#include "Gameplay/CombatComponents.h"
#include "Gameplay/PlayerComponents.h"
#include "Physics/PhysicsComponents.h"

namespace CudaGame {
namespace Gameplay {

/**
 * [AAA Pattern] Combat System
 * Handles weapon logic, attack inputs, cooldowns, and hit resolution.
 */
class CombatSystem : public Core::System {
public:
    CombatSystem();
    ~CombatSystem();
    
    bool Initialize() override;
    void Update(float deltaTime) override;
    void Shutdown() override;
    
    // Commands
    void Attack(Core::Entity attackerID);
    
private:
    // Internal Logic
    void ProcessInput(Core::Entity entity, const PlayerInputComponent& input, CombatComponent& combat);
    void UpdateWeapon(Core::Entity weaponID, WeaponComponent& weapon, float deltaTime);
    void SyncWeaponTransform(Core::Entity ownerID, Core::Entity weaponID);
    
    // Hit Resolution
    void PerformHitScan(Core::Entity attacker, const WeaponComponent& weapon, const Physics::RigidbodyComponent& rb);
};

} // namespace Gameplay
} // namespace CudaGame
