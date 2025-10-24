#pragma once

#include "Core/System.h"
#include <glm/glm.hpp>
#include <vector>
#include <memory>

namespace CudaGame {
namespace Combat {

// Forward declarations
class Entity;
class WeaponSystem;

// Combat states for precise frame-based combat
enum class CombatState {
    NONE,
    PUNCH,
    KICK,
    COMBO_1,
    COMBO_2,
    COMBO_3,
    // Weapon-specific states
    SWORD_SLASH_1,
    SWORD_SLASH_2,
    SWORD_THRUST,
    STAFF_SPIN,
    STAFF_CAST,
    STAFF_SLAM,
    HAMMER_SWING,
    HAMMER_OVERHEAD,
    HAMMER_GROUND_POUND,
    // Enhanced combat states
    GRAB,
    GRAB_THROW,
    PARRY,
    COUNTER_ATTACK,
    SLIDE,
    SLIDE_LEAP
};

enum class ComboState {
    Idle,           // Not performing any move
    Startup,        // Move is in startup frames
    Active,         // Move is in active frames (can hit)
    Recovery,       // Move is in recovery frames
    Cancelled       // Move was cancelled into another move
};

enum class WeaponType {
    NONE,
    SWORD,
    STAFF,
    HAMMER
};

// Combat component for entities
struct CombatComponent {
    CombatState combatState = CombatState::NONE;
    ComboState comboState = ComboState::Idle;
    float combatTimer = 0.0f;
    float comboWindow = 0.8f;
    int comboCount = 0;
    float attackRange = 3.0f;
    float damage = 10.0f;
    float health = 100.0f;
    float maxHealth = 100.0f;
    bool isBlocking = false;
    bool isParrying = false;
    float parryTimer = 0.0f;
    float parryWindow = 0.3f;
    bool justParried = false;
    float damageMultiplier = 1.0f;
    int perfectParries = 0;
    int stunComboCount = 0;
};

// Weapon data structure
struct WeaponData {
    WeaponType type;
    float damage;
    float range;
    float speed;
    glm::vec3 color;
    float comboWindow;
    int maxComboChain;
    
    WeaponData(WeaponType t = WeaponType::NONE);
};

// Combat system class
class CombatSystem : public Core::System {
public:
    CombatSystem();
    ~CombatSystem();
    
    // Override Core::System methods
    bool Initialize() override { return true; }
    void Shutdown() override {}
    void Update(float deltaTime) override { update(deltaTime); }
    
    // Configuration method for integration
    void Configure() {}
    
    // Core combat functions
    void update(float deltaTime);
    void processCombatInput(uint32_t entityId, int inputFlags);
    bool attemptAttack(uint32_t attackerId, uint32_t targetId);
    bool attemptParry(uint32_t entityId);
    bool attemptGrab(uint32_t grabberId, uint32_t targetId);
    
    // Combo system
    void updateComboSystem(uint32_t entityId, float deltaTime);
    bool canPerformCombo(uint32_t entityId, ComboState desiredCombo);
    void executeCombo(uint32_t entityId, ComboState combo);
    void resetCombo(uint32_t entityId);
    
    // Weapon system integration
    void equipWeapon(uint32_t entityId, WeaponType weapon);
    void switchWeapon(uint32_t entityId, WeaponType newWeapon);
    WeaponData* getWeaponData(WeaponType type);
    
    // Damage and health
    void dealDamage(uint32_t attackerId, uint32_t targetId, float damage);
    void heal(uint32_t entityId, float amount);
    bool isEntityDead(uint32_t entityId);
    
    // Combat state queries
    bool isInCombat(uint32_t entityId);
    float getRemainingCombatTime(uint32_t entityId);
    int getCurrentComboCount(uint32_t entityId);
    
    // Rhythm integration
    void setRhythmTiming(float beatWindow, float perfectWindow);
    float calculateRhythmMultiplier(float timingAccuracy);
    
private:
    std::vector<WeaponData> m_weaponDatabase;
    float m_beatWindow = 0.1f;
    float m_perfectWindow = 0.05f;
    
    // Internal helper functions
    void initializeWeaponDatabase();
    bool validateCombatAction(uint32_t entityId, CombatState action);
    void applyHitStop(uint32_t entityId, float duration);
    void triggerScreenShake(float intensity);
};

// Combat events for other systems
struct CombatEvent {
    enum Type {
        ATTACK_STARTED,
        ATTACK_HIT,
        ATTACK_MISSED,
        COMBO_EXECUTED,
        PARRY_SUCCESS,
        ENTITY_DIED
    };
    
    Type type;
    uint32_t attackerId;
    uint32_t targetId;
    float damage;
    glm::vec3 position;
    glm::vec3 direction;
};

} // namespace Combat
} // namespace CudaGame
