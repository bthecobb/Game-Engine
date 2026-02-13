#pragma once

#include "Gameplay/CharacterResources.h"
#include "Gameplay/PlayerComponents.h"
#include <string>
#include <vector>

namespace CudaGame {
namespace Gameplay {

enum class DamageType {
    PHYSICAL,
    ENERGY,
    FIRE,
    ICE
};

/**
 * [AAA Pattern] Weapon Definition (Static Data)
 * Flyweight pattern: Loaded once, referenced by many.
 * Defines the properties of a weapon archetype.
 */
struct WeaponDefinition {
    ResourceID name;
    WeaponType type = WeaponType::SWORD;
    
    // Visuals
    ResourceID meshID;
    ResourceID attackAnimationID;
    ResourceID equipAnimationID;
    
    // Combat Stats
    float damage = 10.0f;
    float range = 2.0f;
    float attackRate = 1.0f;     // Attacks per second
    float reloadTime = 2.0f;
    int maxAmmo = 0;             // 0 = Infinite
    
    // Hitbox / Physics
    float hitRadius = 0.5f;      // For melee casting
    
    // Audio (IDs)
    ResourceID sfxAttack;
    ResourceID sfxHit;
};

/**
 * [AAA Pattern] Weapon Component (Runtime State)
 * Attached to the Weapon Entity (which is child of Character).
 * Tracks the dynamic state of a specific weapon instance.
 */
struct WeaponComponent {
    ResourceID definitionID; // Reference to static data
    
    Core::Entity owner = 0;  // Who holds this weapon
    
    // State
    float cooldownTimer = 0.0f;
    float reloadTimer = 0.0f;
    int currentAmmo = 0;
    bool isAttacking = false;
    
    // Hit Registration
    // Used to prevent multi-hits in a single swing if using simple overlap
    std::vector<Core::Entity> hitEntities; 
};

/**
 * [AAA Pattern] Combat Component (Character State)
 * Attached to the Character Entity.
 * Manages health, faction, and active weapon.
 */
struct CombatComponent {
    float health = 100.0f;
    float maxHealth = 100.0f;
    bool isDead = false;
    
    std::string faction = "Neutral"; // "Player", "Enemy"
    
    // Loadout
    Core::Entity activeWeaponEntity = 0;
    // Potentially a list of weapon entities in inventory
    std::vector<Core::Entity> inventory;
};

/**
 * [AAA Pattern] Hitbox Component
 * Attached to specific bones/entities to receive damage.
 * Allows localized damage (Headshot multiplier).
 */
struct HitboxComponent {
    float damageMultiplier = 1.0f; // 2.0 for Head
    Core::Entity parentRoot = 0;   // The main character entity that takes the damage
};

} // namespace Gameplay
} // namespace CudaGame
