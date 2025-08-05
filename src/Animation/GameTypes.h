#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <string>

// Forward declarations
class SimpleAnimationController;
class CharacterMesh;
class CharacterAnimationController;
class Enhanced3DCamera;
class EnhancedControllerInput;

// Game states
enum GameState {
    GAME_PLAYING,
    GAME_DEATH,
    GAME_GAME_OVER,
    GAME_RESTART
};

// Combat states
enum class CombatState {
    NONE,
    PUNCH,
    KICK,
    COMBO_1,
    COMBO_2,
    COMBO_3,
    SWORD_SLASH_1,
    SWORD_SLASH_2,
    SWORD_THRUST,
    STAFF_SPIN,
    STAFF_CAST,
    STAFF_SLAM,
    HAMMER_SWING,
    HAMMER_OVERHEAD,
    HAMMER_GROUND_POUND,
    GRAB,
    GRAB_THROW,
    PARRY,
    COUNTER_ATTACK,
    SLIDE,
    SLIDE_LEAP
};

enum ComboState {
    COMBO_NONE,
    COMBO_DASH,
    COMBO_LIGHT_1,
    COMBO_LIGHT_2,
    COMBO_LIGHT_3,
    COMBO_LAUNCHER,
    COMBO_AIR,
    COMBO_WEAPON_DIVE
};

enum class WeaponType {
    NONE,
    SWORD,
    STAFF,
    HAMMER
};

// Controller state structure
struct ControllerState {
    bool buttonA, buttonB, buttonX, buttonY;
    bool leftBumper, rightBumper;
    bool leftTrigger, rightTrigger;
    bool dpadUp, dpadDown, dpadLeft, dpadRight;
    bool leftStick, rightStick;
    bool start, back;
    float leftStickX, leftStickY;
    float rightStickX, rightStickY;
    float leftTriggerValue, rightTriggerValue;
};

// Player structure
struct Player {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    float rotation;
    float speed;
    float jumpForce;
    float health;
    float maxHealth;
    bool onGround;
    bool isDead;
    bool canDoubleJump;
    bool hasDoubleJumped;
    bool isDashing;
    bool isDiving;
    bool isFlipping;
    bool isWallRunning;
    bool isWeaponDiving;
    bool isChargingWeaponDive;
    bool sprintToggled;
    float flipRotation;
    float dashTimer;
    float deathTimer;
    float sprintMultiplier;
    float dashSpeed;
    float diveSpeed;
    float wallRunSpeed;
    float wallRunGravity;
    float wallRunTime;
    float maxWallRunTime;
    float weaponDiveChargeTime;
    float maxChargeTime;
    int wallRunPillarIndex;
    bool canWallJump;
    bool hasLandedFromDive;
    bool canAirDash;
    bool isAirDashing;
    float airDashTimer;
    glm::vec3 dashDirection;
    glm::vec3 diveDirection;
    glm::vec3 wallNormal;
    glm::vec3 wallRunDirection;
    glm::vec3 spawnPosition;
    std::vector<int> drawnEnemyIndices;
    
    SimpleAnimationController* animController;
    CharacterMesh* characterMesh;
    CharacterAnimationController* characterAnimController;
    
    CombatState combatState;
    ComboState comboState;
    float combatTimer;
    float comboWindow;
    int comboCount;
    
    struct Weapon {
        WeaponType type;
        float damage;
        float range;
        glm::vec3 color;
        struct WeaponAnimation {
            float animationTime;
            float totalDuration;
            glm::vec3 rotationAxis;
            float rotationAngle;
            glm::vec3 positionOffset;
            CombatState comboState;
            bool isActive;
        } currentAnimation;
        
        Weapon(WeaponType t = WeaponType::NONE);
    } currentWeapon;
    
    struct RhythmState {
        float accuracy;
        float lastHitTime;
        int perfectHits;
        int goodHits;
        int misses;
        float comboMultiplier;
    } rhythm;
    
    float damageMultiplier;
    float globalTimeScale;
    bool hasInvincibility;
    
    WeaponType previousWeapon;
    WeaponType targetWeapon;
    bool isSwitchingWeapon;
    float weaponSwitchTimer;
    float weaponSwitchDuration;
    float weaponHolsterProgress;
    glm::vec3 weaponSwitchOffset;
    float weaponSwitchRotation;
    
    std::vector<struct ZoneReward> activeRewards;
};

// Enemy structure
struct Enemy {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    float speed;
    float detectionRange;
    float attackRange;
    float health;
    float maxHealth;
    float damage;
    bool isDead;
    bool isAggressive;
    bool isLaunched;
    bool isFlying;
    bool isStunned;
    float deathTimer;
    float launchTimer;
    float flyTimer;
    float stunTimer;
    float damageFlash;
    float attackCooldown;
    glm::vec3 flyVelocity;
    glm::vec3 patrolStart;
    glm::vec3 patrolEnd;
    glm::vec3 patrolTarget;
    
    enum RewardState {
        NO_REWARD,
        CARRYING_REWARD,
        ABOUT_TO_SPARK,
        SPARKING,
        REWARD_ACTIVATED
    } rewardState;
    
    struct ZoneReward carriedReward;
    float rewardGlowIntensity;
    glm::vec3 rewardAura;
    float rewardPulseTimer;
    float sparkBuildupTimer;
    float sparkTimer;
    float sparkDuration;
    bool isAboutToSpark;
    std::vector<glm::vec3> sparkParticles;
    int rewardOrbitCount;
    float rewardOrbitRadius;
    
    enum EnemyType {
        BASIC_MELEE,
        HEAVY_TANK,
        FAST_RUSHER,
        RANGED_SHOOTER,
        FLYING_DRONE,
        SHIELD_BEARER,
        MINI_BOSS_PHASE_SHIFTER,
        MINI_BOSS_RHYTHM_THIEF,
        MINI_BOSS_GRAVITY_LORD,
        MINI_BOSS_TIME_WEAVER,
        MINI_BOSS_COMBO_BREAKER,
        MINI_BOSS_WALL_CRAWLER,
        MINI_BOSS_ENERGY_VAMPIRE,
        MINI_BOSS_MIRROR_FIGHTER,
        MINI_BOSS_FINAL_CONDUCTOR
    } enemyType;
    
    std::unique_ptr<struct MiniBossData> miniBossData;
    
    float wingFlapTimer;
    float wingFlapSpeed;
    glm::vec3 wingRotation;
    float hoverBounceTimer;
    float hoverBounceAmplitude;
    float baseHeight;
    
    Enemy(glm::vec3 pos, EnemyType type);
};

// Giant pillar structure
struct GiantPillar {
    glm::vec3 position;
    glm::vec3 size;
    glm::vec3 color;
    float radius;
    bool isCylindrical;
    bool isWallRunnable;
    std::vector<glm::vec3> wallRunSurfaces;
    
    GiantPillar(glm::vec3 pos, glm::vec3 sz, bool cylindrical = true);
    glm::vec3 getClosestSurface(glm::vec3 playerPos, glm::vec3& outNormal) const;
};
