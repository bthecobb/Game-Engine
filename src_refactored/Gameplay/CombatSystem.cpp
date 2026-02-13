#include "Gameplay/CombatSystem.h"
#include "Core/Coordinator.h"
#include "Gameplay/CharacterFactory.h" // For definitions if needed, but CombatSystem uses Components
#include <iostream>
#include <glm/glm.hpp>
#include "Rendering/RenderComponents.h"
#include "Gameplay/AnimationControllerComponent.h"

// GLFW Input Constants
#define GLFW_MOUSE_BUTTON_LEFT 0

namespace CudaGame {
namespace Gameplay {

CombatSystem::CombatSystem() {
}

CombatSystem::~CombatSystem() {
    Shutdown();
}

bool CombatSystem::Initialize() {
    std::cout << "[CombatSystem] Initialized" << std::endl;
    return true;
}

void CombatSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // 1. Process Input for Players
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<CombatComponent>(entity) && 
            coordinator.HasComponent<PlayerInputComponent>(entity)) {
            
            auto& combat = coordinator.GetComponent<CombatComponent>(entity);
            const auto& input = coordinator.GetComponent<PlayerInputComponent>(entity);
            
            ProcessInput(entity, input, combat);
        }
    }
    
    // 2. Update Weapons (Cooldowns, Reloads)
    // We should iterate over Weapon Entities, but mEntities only contains those with Signature.
    // If CombatSystem Signature includes WeaponComponent, we'd see them.
    // Or we iterate inventory from CombatComponent.
    // Let's iterate ALL entities for WeaponComponent manual check (inefficient but safe for now)
    // Better: Register System for WeaponComponent too? Core::System supports one signature.
    // We can assume CombatSystem manages Characters. Weapons are managed via Character update loop here.
    
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<CombatComponent>(entity)) {
            auto& combat = coordinator.GetComponent<CombatComponent>(entity);
            if (coordinator.HasComponent<WeaponComponent>(combat.activeWeaponEntity)) {
                auto& weapon = coordinator.GetComponent<WeaponComponent>(combat.activeWeaponEntity);
                UpdateWeapon(combat.activeWeaponEntity, weapon, deltaTime);
                SyncWeaponTransform(entity, combat.activeWeaponEntity);
            }
        }
    }
}

void CombatSystem::ProcessInput(Core::Entity entity, const PlayerInputComponent& input, CombatComponent& combat) {
    if (input.mouseButtons[GLFW_MOUSE_BUTTON_LEFT]) {
        Attack(entity);
    }
}

void CombatSystem::Attack(Core::Entity attackerID) {
    auto& coordinator = Core::Coordinator::GetInstance();
    auto& combat = coordinator.GetComponent<CombatComponent>(attackerID);
    
    if (combat.activeWeaponEntity == 0) return;
    
    auto& weapon = coordinator.GetComponent<WeaponComponent>(combat.activeWeaponEntity);
    
    // Check Cooldown
    if (weapon.cooldownTimer > 0) return;
    
    // Trigger Attack
    weapon.cooldownTimer = 1.0f; // Default, should read from WeaponDefinition
    weapon.isAttacking = true;
    
    // Trigger Animation (via AnimationController)
    if (coordinator.HasComponent<AnimationControllerComponent>(attackerID)) {
        auto& animCtrl = coordinator.GetComponent<AnimationControllerComponent>(attackerID);
        animCtrl.Trigger("IsAttacking");
    }
    
    std::cout << "[CombatSystem] Entity " << attackerID << " attacking with Weapon " << combat.activeWeaponEntity << std::endl;
    
    // Perform Hit Resolution (Immediate for prototype)
    // Ideally this happens on a specific animation frame event.
    if (coordinator.HasComponent<Physics::RigidbodyComponent>(attackerID)) {
        PerformHitScan(attackerID, weapon, coordinator.GetComponent<Physics::RigidbodyComponent>(attackerID));
    }
}

void CombatSystem::UpdateWeapon(Core::Entity weaponID, WeaponComponent& weapon, float deltaTime) {
    if (weapon.cooldownTimer > 0) {
        weapon.cooldownTimer -= deltaTime;
    }
    if (weapon.isAttacking && weapon.cooldownTimer <= 0.5f) { // Arbitrary end of attack
        weapon.isAttacking = false;
    }
}

void CombatSystem::SyncWeaponTransform(Core::Entity ownerID, Core::Entity weaponID) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    const auto& ownerTransform = coordinator.GetComponent<Rendering::TransformComponent>(ownerID);
    auto& weaponTransform = coordinator.GetComponent<Rendering::TransformComponent>(weaponID);
    
    // Simple parenting offset (Right Hand)
    // glm::vec3 offset = glm::vec3(0.5f, -0.2f, 0.5f); // Relative to center
    // Ideally: Bone Transform.
    // For now: Just match position + forward offset
    
    // If we have AnimationComponent, we can try to get bone matrix? 
    // Too complex for Step 2. Just strict offset.
    
    weaponTransform.position = ownerTransform.position + glm::vec3(0.5f, 1.0f, 0.5f); 
    // weaponTransform.rotation = ownerTransform.rotation;
}

void CombatSystem::PerformHitScan(Core::Entity attacker, const WeaponComponent& weapon, const Physics::RigidbodyComponent& rb) {
    // Simple Box Check for Demo
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Get Attacker Pos
    const auto& attackerTransform = coordinator.GetComponent<Rendering::TransformComponent>(attacker);
    glm::vec3 origin = attackerTransform.position + glm::vec3(0, 1, 0);
    glm::vec3 forward = glm::vec3(0, 0, 1); // Assuming Z-forward if no rotation stored
    
    // Iterate entities to find targets (Naive)
    for (Core::Entity target = 0; target < 1000; ++target) {
        if (target == attacker || target == weapon.owner) continue;
        
        if (coordinator.HasComponent<CombatComponent>(target) && 
            coordinator.HasComponent<Rendering::TransformComponent>(target)) {
            
            const auto& targetTransform = coordinator.GetComponent<Rendering::TransformComponent>(target);
            float dist = glm::length(origin - targetTransform.position);
            
            if (dist < 2.0f) { // Melee Range
                std::cout << "[CombatSystem] HIT! Entity " << target << " took damage." << std::endl;
                
                // Construct Damage Info?
                // For now just log
                auto& targetCombat = coordinator.GetComponent<CombatComponent>(target);
                targetCombat.health -= 10.0f;
                if (targetCombat.health <= 0) {
                    std::cout << "[CombatSystem] Entity " << target << " DIED." << std::endl;
                    targetCombat.isDead = true;
                    // Respawn or Destroy?
                }
            }
        }
    }
}

void CombatSystem::Shutdown() {
    std::cout << "[CombatSystem] Shut down" << std::endl;
}

} // namespace Gameplay
} // namespace CudaGame
