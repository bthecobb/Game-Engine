#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>

// OpenGL includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Animation system
#include "SimpleAnimationController.h"
#include "../include/CharacterMesh.h"
#include "Enhanced3DCameraSystem.h"
#include "AutomatedGameTest.h"

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Window dimensions
const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;

// Enhanced game states
enum GameState {
    GAME_PLAYING,
    GAME_DEATH,
    GAME_GAME_OVER,
    GAME_RESTART
};

// Enhanced combat system
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

// Zone-based reward system
enum class ZoneRewardType {
    SLOW_MOTION,        // Bullet-time effect
    COMBO_BONUS,        // +1 to combo multiplier
    STAMINA_REFUND,     // Restore stamina/health
    DAMAGE_BOOST,       // Temporary damage increase
    INVINCIBILITY,      // Brief invulnerability
    RHYTHM_PERFECT,     // Perfect rhythm timing for duration
    SPEED_BOOST,        // Movement speed increase
    WALL_RUN_EXTENDED,  // Longer wall run duration
    DOUBLE_JUMP_REFRESH // Restore air abilities
};

struct ZoneReward {
    ZoneRewardType type;
    float duration;
    float magnitude;
    bool isActive;
    float timer;
    glm::vec3 color;  // Visual indicator color
    std::string name;
    
    ZoneReward(ZoneRewardType t = ZoneRewardType::SLOW_MOTION) : type(t), isActive(false), timer(0.0f) {
        switch(t) {
            case ZoneRewardType::SLOW_MOTION:
                duration = 3.0f; magnitude = 0.3f; color = glm::vec3(0.3f, 0.3f, 1.0f); name = "Slow Motion"; break;
            case ZoneRewardType::COMBO_BONUS:
                duration = 10.0f; magnitude = 1.0f; color = glm::vec3(1.0f, 0.8f, 0.0f); name = "Combo Boost"; break;
            case ZoneRewardType::STAMINA_REFUND:
                duration = 0.1f; magnitude = 50.0f; color = glm::vec3(0.0f, 1.0f, 0.3f); name = "Health Restore"; break;
            case ZoneRewardType::DAMAGE_BOOST:
                duration = 8.0f; magnitude = 1.5f; color = glm::vec3(1.0f, 0.3f, 0.0f); name = "Damage Boost"; break;
            case ZoneRewardType::INVINCIBILITY:
                duration = 2.0f; magnitude = 1.0f; color = glm::vec3(1.0f, 1.0f, 1.0f); name = "Invincible"; break;
            case ZoneRewardType::RHYTHM_PERFECT:
                duration = 6.0f; magnitude = 1.0f; color = glm::vec3(1.0f, 0.0f, 1.0f); name = "Perfect Rhythm"; break;
            case ZoneRewardType::SPEED_BOOST:
                duration = 7.0f; magnitude = 1.8f; color = glm::vec3(0.0f, 1.0f, 1.0f); name = "Speed Boost"; break;
            case ZoneRewardType::WALL_RUN_EXTENDED:
                duration = 5.0f; magnitude = 2.0f; color = glm::vec3(0.6f, 0.4f, 0.8f); name = "Wall Master"; break;
            case ZoneRewardType::DOUBLE_JUMP_REFRESH:
                duration = 0.1f; magnitude = 1.0f; color = glm::vec3(0.8f, 1.0f, 0.2f); name = "Air Refresh"; break;
        }
    }
};

struct RewardZone {
    glm::vec3 position;
    glm::vec3 size;
    ZoneReward reward;
    bool isTriggered;
    float cooldownTimer;
    float cooldownDuration;
    bool requiresMovementEntry;  // Must enter zone with velocity
    float minVelocityRequired;
    glm::vec3 entryDirection;    // Required movement direction
    
    RewardZone(glm::vec3 pos, glm::vec3 sz, ZoneRewardType rewardType) : 
        position(pos), size(sz), reward(rewardType), isTriggered(false), 
        cooldownTimer(0.0f), cooldownDuration(15.0f), 
        requiresMovementEntry(true), minVelocityRequired(10.0f), entryDirection(0.0f) {}
};

// Mini-boss specific data
struct MiniBossData {
    int uniqueId;                    // 1-9 unique identifier
    bool hasBeenDefeated;
    ZoneReward stolenReward;         // Reward they carry
    float specialAbilityTimer;
    bool isUsingSpecialAbility;
    
    // Phase Shifter (Boss #1)
    int currentPhase;                // 0-3 for different "dimensions"
    float phaseTimer;
    
    // Rhythm Thief (Boss #2)
    float stolenRhythmAccuracy;
    
    // Gravity Lord (Boss #3)
    glm::vec3 gravityDirection;
    float gravityMagnitude;
    
    // Time Weaver (Boss #4)
    float timeScale;                 // Affects movement/animation speed
    
    // Combo Breaker (Boss #5)
    int disruptedCombos;
    
    // Wall Crawler (Boss #6)
    bool isOnWall;
    glm::vec3 wallSurface;
    
    // Energy Vampire (Boss #7)
    float drainedEnergy;
    
    // Mirror Fighter (Boss #8)
    std::vector<CombatState> mirroredMoves;
    float mirrorDelay;
    
    // Final Conductor (Boss #9)
    float conductorBeat;
    bool isUltimateAttack;
    
    MiniBossData(int id) : uniqueId(id), hasBeenDefeated(false), 
        stolenReward(static_cast<ZoneRewardType>(rand() % 9)),
        specialAbilityTimer(0.0f), isUsingSpecialAbility(false),
        currentPhase(0), phaseTimer(0.0f), stolenRhythmAccuracy(0.0f),
        gravityDirection(0.0f, -1.0f, 0.0f), gravityMagnitude(9.8f),
        timeScale(1.0f), disruptedCombos(0), isOnWall(false), wallSurface(0.0f),
        drainedEnergy(0.0f), mirrorDelay(1.5f), conductorBeat(0.0f), isUltimateAttack(false) {}
};

enum class WeaponType {
    NONE,
    SWORD,
    STAFF,
    HAMMER
};

// Weapon animation data
struct WeaponAnimation {
    float animationTime;
    float totalDuration;
    glm::vec3 rotationAxis;
    float rotationAngle;
    glm::vec3 positionOffset;
    bool isActive;
    CombatState comboState;
    
    WeaponAnimation() : animationTime(0.0f), totalDuration(0.0f), 
                       rotationAxis(0.0f), rotationAngle(0.0f), 
                       positionOffset(0.0f), isActive(false), comboState(CombatState::NONE) {}
};

struct Weapon {
    WeaponType type;
    float damage;
    float range;
    float speed;
    glm::vec3 color;
    WeaponAnimation currentAnimation;
    
    Weapon(WeaponType t = WeaponType::NONE) : type(t) {
        switch(t) {
            case WeaponType::SWORD:
                damage = 25.0f;
                range = 4.0f; // Increased range
                speed = 1.2f;
                color = glm::vec3(0.7f, 0.7f, 0.9f);
                break;
            case WeaponType::STAFF:
                damage = 15.0f;
                range = 5.5f; // Increased range
                speed = 0.8f;
                color = glm::vec3(0.5f, 0.3f, 0.8f);
                break;
            case WeaponType::HAMMER:
                damage = 40.0f;
                range = 3.0f; // Increased range
                speed = 0.6f;
                color = glm::vec3(0.8f, 0.4f, 0.2f);
                break;
            default:
                damage = 10.0f;
                range = 2.5f; // Increased range
                speed = 1.5f;
                color = glm::vec3(0.5f, 0.5f, 0.5f);
                break;
        }
    }
};

struct RhythmState {
    float accuracy;
    float lastHitTime;
    int perfectHits;
    int goodHits;
    int missedHits;
    float beatWindow;
    
    RhythmState() : accuracy(0.0f), lastHitTime(0.0f), 
                   perfectHits(0), goodHits(0), missedHits(0),
                   beatWindow(0.1f) {}
    
    float calculateMultiplier() {
        if (accuracy > 0.9f) return 2.0f;  // Perfect timing
        if (accuracy > 0.7f) return 1.5f;  // Good timing
        if (accuracy > 0.5f) return 1.0f;  // OK timing
        return 0.5f;  // Poor timing
    }
};

// Game structures
struct Player {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    SimpleAnimationController* animController;
    
    // 3D Character Mesh System
    CharacterMesh* characterMesh;
    CharacterAnimationController* characterAnimController;
    float speed;
    float sprintMultiplier;
    float jumpForce;
    bool onGround;
    bool canDoubleJump;
    bool hasDoubleJumped;
    bool isWallRunning;
    float wallRunTime;
    float maxWallRunTime;
    glm::vec3 wallNormal;
    glm::vec3 wallRunDirection;
    float wallRunSpeed;
    int wallRunPillarIndex;
    float wallRunHeight; // Current height on the wall
    float wallRunGravity; // Reduced gravity while wall-running
    
    // Wall jump mechanics
    bool canWallJump;
    float wallJumpForce;
    glm::vec3 wallJumpDirection;
    float wallJumpTimer;
    float rotation; // Y-axis rotation
    
    // Double jump flip
    bool isFlipping;
    float flipRotation;
    float flipSpeed;
    
    // Enhanced Combat
    CombatState combatState;
    ComboState comboState;
    float combatTimer;
    float comboWindow;
    int comboCount;
    float attackRange;
    
    // Enhanced jump and dive system
    bool canDive;
    bool isDiving;
    bool hasLandedFromDive; // New: tracks if player just landed from dive
    bool isWeaponDiving;
    bool isChargingWeaponDive;
    float diveSpeed;
    float diveBounceHeight;
    glm::vec3 diveDirection;
    bool isFloating;
    float floatTimer;
    float floatDuration;
    
    // Weapon dive charge (triggered after dive landing)
    float weaponDiveChargeTime;
    float maxChargeTime;
    std::vector<int> drawnEnemyIndices;
    float landingChargeWindow; // Time window to start charging after landing
    
    // Movement enhancements
    bool sprintToggled;
    bool isDashing;
    float dashTimer;
    float dashSpeed;
    glm::vec3 dashDirection;
    bool canAirDash;
    bool isAirDashing;
    float airDashTimer;
    
    // Health and weapons
    float health;
    float maxHealth;
    Weapon currentWeapon;
    std::vector<WeaponType> inventory;
    
    // Death state
    bool isDead;
    float deathTimer;
    float respawnTimer;
    glm::vec3 spawnPosition;
    
    // Rhythm
    RhythmState rhythm;
    
    // Enhanced combat system
    bool isSliding;
    float slideSpeed;
    float slideDuration;
    float slideTimer;
    bool canSlideJump;
    float slideJumpForce;
    
    // Grab system
    bool isGrabbing;
    int grabbedEnemyIndex;
    float grabTimer;
    float grabDuration;
    float grabRange;
    
    // Parry system
    bool isParrying;
    float parryTimer;
    float parryWindow;
    bool justParried;
    float parryCounterWindow;
    
    // Progressive enhancement system
    int perfectParries;
    int stunComboCount;
    float damageMultiplier;
    float healthScaling;
    float maxHealthBonus;
    
    // Weapon switching animation system
    bool isSwitchingWeapon;
    float weaponSwitchTimer;
    float weaponSwitchDuration;
    WeaponType previousWeapon;
    WeaponType targetWeapon;
    float weaponHolsterProgress; // 0 = fully holstered, 1 = fully drawn
    glm::vec3 weaponSwitchOffset;
    float weaponSwitchRotation;
    
    // Zone reward system
    std::vector<ZoneReward> activeRewards;
    int totalBossesDefeated;
    int score;
    float globalTimeScale;  // For time manipulation effects
    bool hasInvincibility;
    
    // Enhanced Animation System
    enum AnimationState {
        ANIM_IDLE,
        ANIM_IDLE_BORED,     // After 5+ seconds of no activity
        ANIM_WALKING,
        ANIM_RUNNING,
        ANIM_SPRINTING,
        ANIM_JUMPING,
        ANIM_AIRBORNE,       // In air pose
        ANIM_FALLING,
        ANIM_LANDING,
        ANIM_DIVING,
        ANIM_WALL_RUNNING,
        ANIM_SLIDING,
        ANIM_COMBAT_IDLE,
        ANIM_ATTACKING,
        ANIM_PARRYING,
        ANIM_GRABBING,
        ANIM_STUNNED
    };
    
    AnimationState currentAnimation;
    AnimationState previousAnimation;
    float animationTimer;
    float idleTimer;                // Time since last activity
    float animationBlendWeight;     // For smooth transitions
    bool hasMovementInput;
    bool hasRecentActivity;
    
    // Tighter movement controls
    float acceleration;
    float deceleration;
    float airControl;
    float groundFriction;
    glm::vec3 targetVelocity;
    float movementPrecision;
    
    Player() : position(10.0f, 1.0f, 10.0f), velocity(0.0f), size(0.8f, 1.8f, 0.8f),
               speed(15.0f), sprintMultiplier(4.0f), jumpForce(18.0f), onGround(true),
               canDoubleJump(true), hasDoubleJumped(false), isWallRunning(false), 
               wallRunTime(0.0f), maxWallRunTime(3.0f), wallNormal(0.0f), 
               wallRunDirection(0.0f), wallRunSpeed(12.0f), wallRunPillarIndex(-1),
               wallRunHeight(0.0f), wallRunGravity(5.0f),
               canWallJump(false), wallJumpForce(25.0f), wallJumpDirection(0.0f), wallJumpTimer(0.0f),
               rotation(0.0f),
               isFlipping(false), flipRotation(0.0f), flipSpeed(15.0f),
               combatState(CombatState::NONE), comboState(COMBO_NONE), combatTimer(0.0f), 
               comboWindow(0.8f), comboCount(0), attackRange(3.0f),
               canDive(false), isDiving(false), hasLandedFromDive(false), isWeaponDiving(false),
               isChargingWeaponDive(false), diveSpeed(40.0f), diveBounceHeight(35.0f),
               isFloating(false), floatTimer(0.0f), floatDuration(1.5f),
               weaponDiveChargeTime(0.0f), maxChargeTime(2.0f), landingChargeWindow(1.0f),
               sprintToggled(false), isDashing(false), dashTimer(0.0f), dashSpeed(30.0f),
               canAirDash(true), isAirDashing(false), airDashTimer(0.0f),
               health(100.0f), maxHealth(100.0f), currentWeapon(WeaponType::NONE),
               isDead(false), deathTimer(0.0f), respawnTimer(0.0f), spawnPosition(10, 1, 10),
               isSliding(false), slideSpeed(25.0f), slideDuration(1.0f), slideTimer(0.0f),
               canSlideJump(true), slideJumpForce(22.0f),
               isGrabbing(false), grabbedEnemyIndex(-1), grabTimer(0.0f), grabDuration(1.5f), grabRange(2.5f),
               isParrying(false), parryTimer(0.0f), parryWindow(0.3f), justParried(false), parryCounterWindow(0.0f),
               perfectParries(0), stunComboCount(0), damageMultiplier(1.0f), healthScaling(1.0f), maxHealthBonus(0.0f),
               isSwitchingWeapon(false), weaponSwitchTimer(0.0f), weaponSwitchDuration(0.8f),
               previousWeapon(WeaponType::NONE), targetWeapon(WeaponType::NONE),
               weaponHolsterProgress(1.0f), weaponSwitchOffset(0.0f), weaponSwitchRotation(0.0f),
               totalBossesDefeated(0), score(0), globalTimeScale(1.0f), hasInvincibility(false),
               // Enhanced animation system initialization
               currentAnimation(ANIM_IDLE), previousAnimation(ANIM_IDLE), animationTimer(0.0f),
               idleTimer(0.0f), animationBlendWeight(1.0f), hasMovementInput(false), hasRecentActivity(false),
               // Tighter movement controls
               acceleration(45.0f), deceleration(25.0f), airControl(0.8f), groundFriction(0.85f),
               targetVelocity(0.0f), movementPrecision(0.95f) {
        inventory.push_back(WeaponType::NONE);
        inventory.push_back(WeaponType::SWORD);
        inventory.push_back(WeaponType::STAFF);
        inventory.push_back(WeaponType::HAMMER);
        
        // Initialize 3D Character Mesh - will be done after OpenGL is ready
        characterMesh = nullptr;
        characterAnimController = nullptr;
    }
};

enum class EnemyType {
    BASIC_GRUNT,
    FAST_SCOUT,
    HEAVY_BRUTE,
    MAGIC_CASTER,
    FLYING_DRONE,
    // Mini-boss types with unique mechanics
    MINI_BOSS_PHASE_SHIFTER,    // #1: Phases between dimensions
    MINI_BOSS_RHYTHM_THIEF,     // #2: Steals player's rhythm accuracy
    MINI_BOSS_GRAVITY_LORD,     // #3: Controls gravity in arena sections
    MINI_BOSS_TIME_WEAVER,      // #4: Slows/speeds time in zones
    MINI_BOSS_COMBO_BREAKER,    // #5: Disrupts combo chains
    MINI_BOSS_WALL_CRAWLER,     // #6: Uses walls and ceiling movement
    MINI_BOSS_ENERGY_VAMPIRE,   // #7: Drains player stamina/health
    MINI_BOSS_MIRROR_FIGHTER,   // #8: Copies player moves with delay
    MINI_BOSS_FINAL_CONDUCTOR   // #9: The ultimate rhythm master
};

struct Enemy {
    EnemyType enemyType;
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    float speed;
    float detectionRange;
    float attackRange;
    bool isAggressive;
    float health;
    float maxHealth;
    float damage;
    float attackCooldown;
    bool isDead;
    float deathTimer;
    
    // Patrol AI
    glm::vec3 patrolStart;
    glm::vec3 patrolEnd;
    glm::vec3 patrolTarget;
    float patrolSpeed;
    bool isPatrolling;
    float patrolWaitTimer;
    float patrolWaitDuration;
    
    // Visual feedback
    float damageFlash;
    bool isLaunched;
    float launchTimer;
    bool isFlying;
    float flyTimer;
    glm::vec3 flyVelocity;
    
    // Stun system
    bool isStunned;
    float stunTimer;
    float stunDuration;
    bool isGrabbed;
    float grabResistance;
    
    // Flying animation for FLYING_DRONE type
    float wingFlapTimer;
    float wingFlapSpeed;
    float hoverBounceTimer;
    float hoverBounceAmplitude;
    float baseHeight;
    glm::vec3 wingRotation; // Wing rotation angles
    
    // Mini-boss data (only used for mini-boss enemy types)
    std::unique_ptr<MiniBossData> miniBossData;
    
    // Zone reward carrying system
    enum RewardCarryState {
        NO_REWARD,           // Enemy has no reward
        CARRYING_REWARD,     // Enemy is carrying a zone reward
        ABOUT_TO_SPARK,      // Enemy is about to activate reward (pre-spark animation)
        SPARKING,            // Enemy is in sparking animation
        REWARD_ACTIVATED     // Reward has been activated/stolen
    };
    
    RewardCarryState rewardState;
    ZoneReward carriedReward;
    float rewardGlowIntensity;
    float rewardPulseTimer;
    float sparkTimer;
    float sparkDuration;
    glm::vec3 rewardAura;        // Color aura around enemy
    std::vector<glm::vec3> sparkParticles;  // Particle positions for spark effect
    float rewardOrbitRadius;     // Radius for orbiting reward indicators
    float rewardOrbitSpeed;      // Speed of reward orbit animation
    int rewardOrbitCount;        // Number of orbiting elements
    bool isAboutToSpark;         // Flag for pre-spark state
    float sparkBuildupTimer;     // Timer for spark buildup effect
    
    Enemy(glm::vec3 pos, EnemyType type = EnemyType::BASIC_GRUNT) : 
        enemyType(type), position(pos), velocity(0.0f), 
        isAggressive(false), isDead(false), deathTimer(0.0f), 
        patrolSpeed(4.0f), isPatrolling(true),
        patrolWaitTimer(0.0f), patrolWaitDuration(2.0f),
        damageFlash(0.0f), isLaunched(false), launchTimer(0.0f),
        isFlying(false), flyTimer(0.0f), flyVelocity(0.0f),
        isStunned(false), stunTimer(0.0f), stunDuration(1.5f), isGrabbed(false), grabResistance(1.0f),
        wingFlapTimer(0.0f), wingFlapSpeed(8.0f), hoverBounceTimer(0.0f), 
        hoverBounceAmplitude(0.3f), baseHeight(0.0f), wingRotation(0.0f),
        rewardState(NO_REWARD), carriedReward(), rewardGlowIntensity(0.0f),
        rewardPulseTimer(0.0f), sparkTimer(0.0f), sparkDuration(2.0f),
        rewardAura(0.0f), rewardOrbitRadius(2.0f), rewardOrbitSpeed(3.0f),
        rewardOrbitCount(3), isAboutToSpark(false), sparkBuildupTimer(0.0f) {
        
        // Configure based on enemy type
        switch(type) {
            case EnemyType::BASIC_GRUNT:
                size = glm::vec3(1.0f, 2.0f, 1.0f);
                speed = 8.0f;
                detectionRange = 20.0f;
                attackRange = 3.0f;
                health = maxHealth = 75.0f;
                damage = 10.0f;
                attackCooldown = 0.0f;
                break;
            case EnemyType::FAST_SCOUT:
                size = glm::vec3(0.8f, 1.6f, 0.8f);
                speed = 15.0f;
                detectionRange = 25.0f;
                attackRange = 2.5f;
                health = maxHealth = 50.0f;
                damage = 8.0f;
                attackCooldown = 0.0f;
                patrolSpeed = 8.0f;
                break;
            case EnemyType::HEAVY_BRUTE:
                size = glm::vec3(1.5f, 2.5f, 1.5f);
                speed = 5.0f;
                detectionRange = 15.0f;
                attackRange = 4.0f;
                health = maxHealth = 150.0f;
                damage = 25.0f;
                attackCooldown = 0.0f;
                patrolSpeed = 2.0f;
                break;
            case EnemyType::MAGIC_CASTER:
                size = glm::vec3(0.9f, 1.9f, 0.9f);
                speed = 6.0f;
                detectionRange = 30.0f;
                attackRange = 8.0f;
                health = maxHealth = 60.0f;
                damage = 15.0f;
                attackCooldown = 0.0f;
                patrolSpeed = 3.0f;
                break;
            case EnemyType::FLYING_DRONE:
                size = glm::vec3(1.2f, 0.8f, 1.2f);
                speed = 12.0f;
                detectionRange = 35.0f;
                attackRange = 6.0f;
                health = maxHealth = 40.0f;
                damage = 12.0f;
                attackCooldown = 0.0f;
                isFlying = true;
                flyVelocity = glm::vec3(0, 5.0f, 0); // Start flying
                baseHeight = pos.y + 3.0f; // Hover 3 units above spawn
                wingFlapSpeed = 12.0f; // Faster wing flapping for drones
                hoverBounceAmplitude = 0.5f; // More pronounced bouncing
                break;
                
            // Mini-boss configurations
            case EnemyType::MINI_BOSS_PHASE_SHIFTER:
                size = glm::vec3(2.0f, 3.0f, 2.0f);
                speed = 12.0f;
                detectionRange = 50.0f;
                attackRange = 8.0f;
                health = maxHealth = 300.0f;
                damage = 35.0f;
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(1);
                break;
                
            case EnemyType::MINI_BOSS_RHYTHM_THIEF:
                size = glm::vec3(1.8f, 2.8f, 1.8f);
                speed = 14.0f;
                detectionRange = 45.0f;
                attackRange = 6.0f;
                health = maxHealth = 250.0f;
                damage = 30.0f;
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(2);
                break;
                
            case EnemyType::MINI_BOSS_GRAVITY_LORD:
                size = glm::vec3(2.2f, 3.2f, 2.2f);
                speed = 8.0f;
                detectionRange = 60.0f;
                attackRange = 12.0f;
                health = maxHealth = 400.0f;
                damage = 40.0f;
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(3);
                break;
                
            case EnemyType::MINI_BOSS_TIME_WEAVER:
                size = glm::vec3(1.6f, 2.6f, 1.6f);
                speed = 16.0f;
                detectionRange = 40.0f;
                attackRange = 7.0f;
                health = maxHealth = 275.0f;
                damage = 32.0f;
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(4);
                break;
                
            case EnemyType::MINI_BOSS_COMBO_BREAKER:
                size = glm::vec3(1.9f, 2.9f, 1.9f);
                speed = 10.0f;
                detectionRange = 35.0f;
                attackRange = 9.0f;
                health = maxHealth = 320.0f;
                damage = 38.0f;
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(5);
                break;
                
            case EnemyType::MINI_BOSS_WALL_CRAWLER:
                size = glm::vec3(1.7f, 2.7f, 1.7f);
                speed = 18.0f;
                detectionRange = 55.0f;
                attackRange = 5.0f;
                health = maxHealth = 260.0f;
                damage = 28.0f;
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(6);
                break;
                
            case EnemyType::MINI_BOSS_ENERGY_VAMPIRE:
                size = glm::vec3(2.1f, 3.1f, 2.1f);
                speed = 9.0f;
                detectionRange = 50.0f;
                attackRange = 10.0f;
                health = maxHealth = 350.0f;
                damage = 25.0f; // Lower direct damage, but drains energy
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(7);
                break;
                
            case EnemyType::MINI_BOSS_MIRROR_FIGHTER:
                size = glm::vec3(1.8f, 2.8f, 1.8f);
                speed = 15.0f;
                detectionRange = 45.0f;
                attackRange = 6.0f;
                health = maxHealth = 300.0f;
                damage = 35.0f;
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(8);
                break;
                
            case EnemyType::MINI_BOSS_FINAL_CONDUCTOR:
                size = glm::vec3(2.5f, 3.5f, 2.5f);
                speed = 11.0f;
                detectionRange = 70.0f;
                attackRange = 15.0f;
                health = maxHealth = 500.0f;
                damage = 50.0f;
                attackCooldown = 0.0f;
                miniBossData = std::make_unique<MiniBossData>(9);
                break;
        }
        
        // Set patrol path in a cross pattern
        patrolStart = pos;
        float direction = (rand() % 4) * 90.0f * M_PI / 180.0f;
        float distance = 8.0f + (rand() % 8);
        patrolEnd = pos + glm::vec3(
            cos(direction) * distance,
            0,
            sin(direction) * distance
        );
        patrolTarget = patrolEnd;
    }
};

struct Wall {
    glm::vec3 position;
    glm::vec3 size;
    glm::vec3 color;
    bool isWallRunnable;
    
    Wall(glm::vec3 pos, glm::vec3 sz, glm::vec3 col, bool runnable = false)
        : position(pos), size(sz), color(col), isWallRunnable(runnable) {}
};

// Giant pillars for wall-running
struct GiantPillar {
    glm::vec3 position;
    glm::vec3 size;
    glm::vec3 color;
    float radius; // For cylindrical pillars
    bool isCylindrical;
    bool isWallRunnable;
    std::vector<glm::vec3> wallRunSurfaces; // Multiple surfaces for complex pillars
    
    GiantPillar(glm::vec3 pos, glm::vec3 sz, bool cylindrical = true) 
        : position(pos), size(sz), isCylindrical(cylindrical), isWallRunnable(true) {
        radius = cylindrical ? std::max(sz.x, sz.z) / 2.0f : 0.0f;
        color = glm::vec3(0.6f, 0.5f, 0.4f); // Stone-like color
        
        // Add wall-run surfaces (4 main sides for rectangular, 8 for cylindrical)
        if (cylindrical) {
            for (int i = 0; i < 8; i++) {
                float angle = (i * 45.0f) * M_PI / 180.0f;
                glm::vec3 surface = pos + glm::vec3(cos(angle) * radius, 0, sin(angle) * radius);
                wallRunSurfaces.push_back(surface);
            }
        } else {
            // Rectangular pillar - 4 sides
            wallRunSurfaces.push_back(pos + glm::vec3(sz.x/2, 0, 0));   // Right
            wallRunSurfaces.push_back(pos + glm::vec3(-sz.x/2, 0, 0));  // Left
            wallRunSurfaces.push_back(pos + glm::vec3(0, 0, sz.z/2));   // Front
            wallRunSurfaces.push_back(pos + glm::vec3(0, 0, -sz.z/2));  // Back
        }
    }
    
    // Get the closest wall-runnable surface to a position
    glm::vec3 getClosestSurface(glm::vec3 playerPos, glm::vec3& outNormal) const {
        glm::vec3 closestSurface = wallRunSurfaces[0];
        float closestDist = glm::distance(playerPos, closestSurface);
        
        for (const auto& surface : wallRunSurfaces) {
            float dist = glm::distance(playerPos, surface);
            if (dist < closestDist) {
                closestDist = dist;
                closestSurface = surface;
            }
        }
        
        // Calculate wall normal (pointing outward from pillar center)
        outNormal = glm::normalize(closestSurface - position);
        return closestSurface;
    }
};

struct Platform {
    glm::vec3 position;
    glm::vec3 size;
    glm::vec3 color;
    
    Platform(glm::vec3 pos, glm::vec3 sz, glm::vec3 col)
        : position(pos), size(sz), color(col) {}
};

struct Collectible {
    glm::vec3 position;
    glm::vec3 size;
    glm::vec3 color;
    bool collected;
    
    Collectible(glm::vec3 pos, glm::vec3 sz, glm::vec3 col)
        : position(pos), size(sz), color(col), collected(false) {}
};

// Damage numbers for visual feedback
struct DamageNumber {
    glm::vec3 position;
    float value;
    float lifetime;
    glm::vec3 color;
};

// Ripple effect for landing impacts
struct RippleEffect {
    glm::vec3 position;
    float radius;
    float maxRadius;
    float lifetime;
    glm::vec3 color;
    float spawnTime;
    
    RippleEffect(glm::vec3 pos, float maxR, glm::vec3 col) :
        position(pos), radius(0.0f), maxRadius(maxR), lifetime(1.0f), color(col), spawnTime(glfwGetTime()) {}
};

// 3D Triangle particle explosion
struct ParticleTriangle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 color;
    float lifetime;
    float size;
    float rotation;
    float rotationSpeed;
    float spawnTime;
    
    ParticleTriangle(glm::vec3 pos, glm::vec3 vel, glm::vec3 col) :
        position(pos), velocity(vel), color(col), lifetime(2.0f), 
        size(0.3f), rotation(0.0f), rotationSpeed((rand() % 200 - 100) / 10.0f), spawnTime(glfwGetTime()) {}
};

// Arena bounds with connected landscapes
struct ArenaBounds {
    glm::vec3 center;
    float radius;
    float wallHeight;
    float fallResetY;
    std::vector<glm::vec3> connectedAreas;
};

// Boundary system for arena limits
struct ArenaBoundary {
    glm::vec3 position;
    glm::vec3 normal;  // Direction the boundary faces (inward)
    glm::vec3 size;    // Size of the boundary wall
    float repelForce;
    bool isVisible;
    
    ArenaBoundary(glm::vec3 pos, glm::vec3 norm, glm::vec3 sz) : 
        position(pos), normal(glm::normalize(norm)), size(sz), 
        repelForce(30.0f), isVisible(true) {}
    
    // Check if player is colliding with boundary
    bool checkCollision(const glm::vec3& playerPos, float playerRadius) const {
        // Simple plane-based collision for boundaries
        float distToPlane = glm::dot(playerPos - position, normal);
        return distToPlane < playerRadius;
    }
    
    // Get repel direction and force
    glm::vec3 getRepelVector(const glm::vec3& playerPos, bool hasCombo) const {
        float basePower = hasCombo ? repelForce * 2.0f : repelForce;
        return normal * basePower;
    }
};

// Debug system
struct DebugInfo {
    bool enabled = true;
    bool verbose = false;
    glm::vec3 lastCameraPos;
    glm::vec3 lastCameraTarget;
    glm::vec3 lastPlayerPos;
    std::string lastEvent;
    float timeSinceLastEvent = 0.0f;
    int framesSinceLastRender = 0;
    bool cameraStuck = false;
    float cameraDistanceFromPlayer = 0.0f;
    
    void logEvent(const std::string& event) {
        lastEvent = event;
        timeSinceLastEvent = 0.0f;
        if (enabled) {
            std::cout << "[DEBUG] " << event << std::endl;
        }
    }
    
    void checkCameraHealth(const glm::vec3& camPos, const glm::vec3& camTarget, const glm::vec3& playerPos) {
        // Check for NaN or infinity
        if (std::isnan(camPos.x) || std::isnan(camPos.y) || std::isnan(camPos.z) ||
            std::isinf(camPos.x) || std::isinf(camPos.y) || std::isinf(camPos.z)) {
            logEvent("ERROR: Camera position is NaN or Infinity!");
            cameraStuck = true;
        }
        
        // Check camera distance from player
        cameraDistanceFromPlayer = glm::length(camPos - playerPos);
        if (cameraDistanceFromPlayer > 100.0f) {
            logEvent("WARNING: Camera very far from player: " + std::to_string(cameraDistanceFromPlayer));
        }
        
        // Check if camera moved
        float moveDist = glm::length(camPos - lastCameraPos);
        if (moveDist < 0.001f && framesSinceLastRender > 60) {
            if (!cameraStuck) {
                logEvent("WARNING: Camera appears stuck!");
                cameraStuck = true;
            }
        } else {
            cameraStuck = false;
        }
        
        lastCameraPos = camPos;
        lastCameraTarget = camTarget;
        lastPlayerPos = playerPos;
    }
    
    void update(float deltaTime) {
        timeSinceLastEvent += deltaTime;
        framesSinceLastRender++;
    }
    
    void onRender() {
        framesSinceLastRender = 0;
    }
    
    void printStatus() {
        if (verbose && enabled) {
            std::cout << "[DEBUG STATUS] Camera: (" 
                      << lastCameraPos.x << ", " << lastCameraPos.y << ", " << lastCameraPos.z << ")"
                      << " Player: (" 
                      << lastPlayerPos.x << ", " << lastPlayerPos.y << ", " << lastPlayerPos.z << ")"
                      << " Dist: " << cameraDistanceFromPlayer << std::endl;
        }
    }
};

// Global game objects
DebugInfo debugInfo;
Player player;
std::vector<Platform> platforms;
std::vector<Wall> walls;
std::vector<Enemy> enemies;
std::vector<Collectible> collectibles;
std::vector<DamageNumber> damageNumbers;
std::vector<RippleEffect> rippleEffects;
std::vector<ParticleTriangle> explosionParticles;
std::vector<GiantPillar> giantPillars;
std::vector<RewardZone> rewardZones;
std::vector<ArenaBoundary> arenaBoundaries;
GLFWwindow* window;
bool keys[1024] = {false};
bool keysPressed[1024] = {false}; // For single key press
float deltaTime = 0.0f;
float lastFrame = 0.0f;
float beatTimer = 0.0f;
float beatInterval = 60.0f / 140.0f; // 140 BPM - can be changed per level
float currentBPM = 140.0f;
bool onBeat = false;
int score = 0;

// Enhanced game state
GameState gameState = GAME_PLAYING;
int targetedEnemyIndex = -1;
float cameraShake = 0.0f;
ArenaBounds arena = { glm::vec3(0, 0, 0), 150.0f, 0.6f, -25.0f }; // Expanded to 150 units radius
float deathScreenTimer = 0.0f;
int playerLives = 3;

// Enhanced Controller support
int controllerID = -1;
bool useController = false;
float controllerDeadzone = 0.15f; // Optimized deadzone
float controllerSensitivity = 1.0f;
bool controllerInvertY = true; // Enable Y-axis inversion for correct movement

// Standard controller button mapping
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

ControllerState currentController;
ControllerState lastController;
float controllerInputCooldown = 0.0f;
float lastControllerInputTime = 0.0f;

// Rhythm visualization
bool showRhythmFlow = false;
float beatPulse = 0.0f;
float nextBeatTime = 0.0f;
float rhythmAccuracy = 0.0f;

// Enhanced Camera System
Enhanced3DCamera* gameCamera = nullptr;
EnhancedControllerInput* controllerInput = nullptr;

// Automated Testing System
AutomatedGameTest* automatedTest = nullptr;
bool isAutomatedTestMode = false;

// Legacy camera variables for compatibility
glm::vec3 cameraPos;
glm::vec3 cameraTarget;
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

// OpenGL objects
GLuint shaderProgram;
GLuint cubeVAO, cubeVBO;

// Shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

uniform vec3 color;
uniform float pulse;

void main() {
    vec3 finalColor = color * (0.8 + 0.2 * pulse);
    FragColor = vec4(finalColor, 1.0);
}
)";

// Cube vertices
float cubeVertices[] = {
    // Front face
    -0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,
    // Back face
    -0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    // Left face
    -0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    // Right face
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    // Bottom face
    -0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,
    // Top face
    -0.5f,  0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f, -0.5f
};

// Function prototypes
bool initOpenGL();
void setupShaders();
void setupGeometry();
void setupWorld();
void processInput();
void updatePlayer();
void updatePlayerAnimations(float deltaTime);
void updatePlayerMovement(float deltaTime);
void updateEnemies();
void updateCamera();
void updateRhythm();
void updateCombat();
void updateGameState();
void checkRhythmTiming();
void checkBoundaryCollisions();
void handleCombatHit();
void updateBPM(float newBPM);
void spawnDamageNumber(glm::vec3 position, float damage, glm::vec3 color);
void spawnRippleEffect(glm::vec3 position, float radius, glm::vec3 color);
void spawnTriangleExplosion(glm::vec3 position, int numParticles);
void weaponDiveExplosion();
void updateTargeting();
int findBestLockOnTarget();
int getNextLockOnTarget(int currentTarget, bool cycleForward);
void resetGame();
void airDashLauncher();
void updateController();
glm::vec2 getControllerInput();
void startWeaponCombo(WeaponType weaponType, int comboStep);
void updateWeaponAnimation(float deltaTime);
void performSwordCombo(int step);
void performStaffSpell();
void performHammerGroundPound();
bool checkCollision(const glm::vec3& pos1, const glm::vec3& size1, const glm::vec3& pos2, const glm::vec3& size2);
void renderCube(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color, const glm::mat4& rotation = glm::mat4(1.0f));
void renderPlayer();
void renderWeapon();
void renderRhythmUI();
void renderRippleEffects();
void renderParticleExplosions();
void renderChargingVisuals();
void renderLandingEffects();
void renderScene();
void cleanup();
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void applyControllerFiltering();
bool isControllerButtonPressed(int buttonIndex);
bool wasControllerButtonPressed(int buttonIndex);
bool isControllerButtonJustPressed(int buttonIndex);
void updateWeaponSwitching(float deltaTime);
void startWeaponSwitch(WeaponType newWeapon);
void updateFlyingEnemyAnimations(float deltaTime);
void renderFlyingEnemyWings(const Enemy& enemy);

// Zone reward system functions
void assignRandomRewards();
void updateEnemyRewardStates(float deltaTime);
void checkZoneRewardActivation();
void activateZoneReward(ZoneReward& reward);
void updatePlayerRewards(float deltaTime);
void renderRewardVisuals();
void renderEnemyRewardIndicators(const Enemy& enemy);
void stealEnemyReward(int enemyIndex);

// Enhanced targeting system with multi-part enemy support
void updateTargeting() {
    // Don't override targeting if lock-on is active
    if (gameCamera && gameCamera->isLockOnActive()) {
        // Just validate current target is still alive
        if (targetedEnemyIndex >= 0 && targetedEnemyIndex < enemies.size() && enemies[targetedEnemyIndex].isDead) {
            // Target died, clear lock-on
            gameCamera->clearLockOnTarget();
            targetedEnemyIndex = -1;
        }
        return;
    }
    
    targetedEnemyIndex = -1;
    if (enemies.empty()) return;
    
    float closestDistance = FLT_MAX;
    glm::vec3 playerForward(sin(player.rotation), 0, -cos(player.rotation));
    
    for (int i = 0; i < enemies.size(); ++i) {
        if (enemies[i].isDead) continue;
        
        glm::vec3 toEnemy = enemies[i].position - player.position;
        float distance = glm::length(toEnemy);
        
        if (distance < 15.0f) {
            toEnemy = glm::normalize(toEnemy);
            float dot = glm::dot(playerForward, toEnemy);
            
            if (dot > 0.5f && distance < closestDistance) {
                closestDistance = distance;
                targetedEnemyIndex = i;
            }
        }
    }
}

// Enhanced lock-on target selection
int findBestLockOnTarget() {
    if (enemies.empty()) return -1;
    
    float bestScore = -1.0f;
    int bestIndex = -1;
    glm::vec3 playerForward(sin(player.rotation), 0, -cos(player.rotation));
    
    for (int i = 0; i < enemies.size(); ++i) {
        if (enemies[i].isDead) continue;
        
        glm::vec3 toEnemy = enemies[i].position - player.position;
        float distance = glm::length(toEnemy);
        
        // Extended range for lock-on compared to regular targeting
        if (distance < 30.0f) {
            toEnemy = glm::normalize(toEnemy);
            float dot = glm::dot(playerForward, toEnemy);
            
            // Calculate targeting score
            float score = 0.0f;
            
            // Base score from direction and distance
            if (dot > 0.2f) { // More lenient than regular targeting
                score = dot * (1.0f - (distance / 30.0f));
                
                // Bonus for enemies carrying rewards
                if (enemies[i].rewardState == Enemy::CARRYING_REWARD) {
                    score += 0.3f;
                }
                
                // Bonus for mini-bosses
                if (enemies[i].miniBossData) {
                    score += 0.2f;
                }
                
                // Slight bonus for current target to reduce flicking
                if (i == targetedEnemyIndex) {
                    score += 0.1f;
                }
                
                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = i;
                }
            }
        }
    }
    
    return bestIndex;
}

// Cycle through available lock-on targets
int getNextLockOnTarget(int currentTarget, bool cycleForward) {
    if (enemies.empty()) return -1;
    
    // Build list of valid targets within range
    std::vector<int> validTargets;
    for (int i = 0; i < enemies.size(); ++i) {
        if (enemies[i].isDead) continue;
        
        float distance = glm::length(enemies[i].position - player.position);
        if (distance < 30.0f) {
            validTargets.push_back(i);
        }
    }
    
    if (validTargets.empty()) return -1;
    if (validTargets.size() == 1) return validTargets[0];
    
    // Find current target in valid list
    int currentIdx = -1;
    for (int i = 0; i < validTargets.size(); ++i) {
        if (validTargets[i] == currentTarget) {
            currentIdx = i;
            break;
        }
    }
    
    // If current target not in list, return first valid target
    if (currentIdx == -1) return validTargets[0];
    
    // Cycle to next/previous target
    if (cycleForward) {
        currentIdx = (currentIdx + 1) % validTargets.size();
    } else {
        currentIdx = (currentIdx - 1 + validTargets.size()) % validTargets.size();
    }
    
    return validTargets[currentIdx];
}

void spawnDamageNumber(glm::vec3 position, float damage, glm::vec3 color) {
    DamageNumber dmg;
    dmg.position = position + glm::vec3(0, 2.0f, 0);
    dmg.value = damage;
    dmg.lifetime = 1.0f;
    dmg.color = color;
    damageNumbers.push_back(dmg);
}

void spawnRippleEffect(glm::vec3 position, float radius, glm::vec3 color) {
    RippleEffect ripple(position, radius, color);
    rippleEffects.push_back(ripple);
}

void spawnTriangleExplosion(glm::vec3 position, int numParticles) {
    for (int i = 0; i < numParticles; i++) {
        // Create triangular explosion pattern
        float angle = (i / float(numParticles)) * 2.0f * M_PI;
        float radius = 2.0f + (rand() % 100) / 50.0f;
        float height = (rand() % 200 - 100) / 50.0f;
        
        glm::vec3 velocity = glm::vec3(
            cos(angle) * radius,
            height + 5.0f,
            sin(angle) * radius
        );
        
        glm::vec3 color = glm::vec3(
            0.8f + (rand() % 40) / 100.0f,
            0.3f + (rand() % 40) / 100.0f,
            0.1f + (rand() % 20) / 100.0f
        );
        
        explosionParticles.emplace_back(position, velocity, color);
    }
    
    std::cout << "TRIANGLE EXPLOSION!" << std::endl;
}

void weaponDiveExplosion() {
    if (!player.isChargingWeaponDive) return;
    
    float chargeRatio = player.weaponDiveChargeTime / player.maxChargeTime;
    float explosionRadius = 5.0f + (chargeRatio * 10.0f);
    float explosionDamage = 80.0f + (chargeRatio * 120.0f);
    int particleCount = 20 + (int)(chargeRatio * 30);
    
    // Create massive explosion
    spawnTriangleExplosion(player.position, particleCount);
    spawnRippleEffect(player.position, explosionRadius, glm::vec3(1, 0.5f, 0));
    cameraShake = 2.0f + chargeRatio;
    
    // Damage all drawn enemies
    for (int index : player.drawnEnemyIndices) {
        if (index >= 0 && index < enemies.size() && !enemies[index].isDead) {
            enemies[index].health -= explosionDamage;
            enemies[index].velocity.y = 30.0f + chargeRatio * 20.0f;
            
            spawnDamageNumber(enemies[index].position, explosionDamage, 
                             glm::vec3(1, 0.8f, 0));
            
            if (enemies[index].health <= 0) {
                enemies[index].isDead = true;
                enemies[index].deathTimer = 2.0f;
                score += (int)(3000 * chargeRatio);
            }
        }
    }
    
    // Reset weapon dive state
    player.isChargingWeaponDive = false;
    player.weaponDiveChargeTime = 0.0f;
    player.drawnEnemyIndices.clear();
    
    std::cout << "WEAPON DIVE EXPLOSION! Damage: " << explosionDamage << std::endl;
}

// Weapon switching animation system
void startWeaponSwitch(WeaponType newWeapon) {
    if (player.isSwitchingWeapon || player.currentWeapon.type == newWeapon) return;
    
    player.isSwitchingWeapon = true;
    player.weaponSwitchTimer = 0.0f;
    player.previousWeapon = player.currentWeapon.type;
    player.targetWeapon = newWeapon;
    player.weaponHolsterProgress = 1.0f; // Start with weapon drawn
    
    // Set weapon-specific switch duration and style
    switch(newWeapon) {
        case WeaponType::SWORD:
            player.weaponSwitchDuration = 0.6f; // Quick draw
            break;
        case WeaponType::STAFF:
            player.weaponSwitchDuration = 0.9f; // Ceremonial unsheathing
            break;
        case WeaponType::HAMMER:
            player.weaponSwitchDuration = 1.2f; // Heavy weapon, slower
            break;
        default:
            player.weaponSwitchDuration = 0.4f; // Fists, fastest
            break;
    }
    
    std::cout << "Switching to weapon: " << (int)newWeapon << std::endl;
}

void updateWeaponSwitching(float deltaTime) {
    if (!player.isSwitchingWeapon) return;
    
    player.weaponSwitchTimer += deltaTime;
    float progress = player.weaponSwitchTimer / player.weaponSwitchDuration;
    
    if (progress >= 1.0f) {
        // Complete the weapon switch
        player.isSwitchingWeapon = false;
        player.currentWeapon = Weapon(player.targetWeapon);
        player.weaponHolsterProgress = 1.0f;
        player.weaponSwitchOffset = glm::vec3(0.0f);
        player.weaponSwitchRotation = 0.0f;
        std::cout << "Weapon switch complete!" << std::endl;
        return;
    }
    
    // Animate weapon switching with unique animations per weapon
    float halfPoint = 0.5f;
    
    if (progress < halfPoint) {
        // Holstering phase (0 to 0.5)
        float holsterProgress = progress / halfPoint;
        player.weaponHolsterProgress = 1.0f - holsterProgress;
        
        // Weapon-specific holster animations
        switch(player.previousWeapon) {
            case WeaponType::SWORD:
                // Sheath sword smoothly to right hip
                player.weaponSwitchOffset = glm::vec3(
                    holsterProgress * 0.8f,
                    -holsterProgress * 0.5f,
                    -holsterProgress * 0.3f
                );
                player.weaponSwitchRotation = holsterProgress * 45.0f;
                break;
            case WeaponType::STAFF:
                // Staff goes to back with spinning motion
                player.weaponSwitchOffset = glm::vec3(
                    0.0f,
                    holsterProgress * 1.2f,
                    -holsterProgress * 0.8f
                );
                player.weaponSwitchRotation = holsterProgress * 180.0f;
                break;
            case WeaponType::HAMMER:
                // Heavy hammer to back with gravity-assisted motion
                player.weaponSwitchOffset = glm::vec3(
                    -holsterProgress * 0.6f,
                    holsterProgress * 0.8f - holsterProgress * holsterProgress * 1.5f,
                    -holsterProgress * 1.0f
                );
                player.weaponSwitchRotation = holsterProgress * 90.0f;
                break;
            default:
                player.weaponSwitchOffset = glm::vec3(0.0f);
                break;
        }
    } else {
        // Drawing phase (0.5 to 1.0)
        float drawProgress = (progress - halfPoint) / halfPoint;
        player.weaponHolsterProgress = drawProgress;
        
        // Switch to target weapon at halfway point
        if (player.currentWeapon.type != player.targetWeapon) {
            player.currentWeapon = Weapon(player.targetWeapon);
        }
        
        // Weapon-specific draw animations
        float swordEase, staffEase, hammerEase;
        switch(player.targetWeapon) {
            case WeaponType::SWORD:
                // Quick sword draw with flourish
                swordEase = 1.0f - (1.0f - drawProgress) * (1.0f - drawProgress); // Ease out
                player.weaponSwitchOffset = glm::vec3(
                    (1.0f - swordEase) * 0.8f,
                    (1.0f - swordEase) * -0.5f + swordEase * 0.2f, // Small upward flourish
                    (1.0f - swordEase) * -0.3f
                );
                player.weaponSwitchRotation = (1.0f - swordEase) * 45.0f + swordEase * 15.0f * sin(drawProgress * M_PI * 2);
                break;
            case WeaponType::STAFF:
                // Staff emerges with magical energy
                staffEase = drawProgress * drawProgress; // Ease in
                player.weaponSwitchOffset = glm::vec3(
                    sin(drawProgress * M_PI * 4) * 0.1f, // Slight wobble
                    (1.0f - staffEase) * 1.2f,
                    (1.0f - staffEase) * -0.8f
                );
                player.weaponSwitchRotation = (1.0f - staffEase) * 180.0f + sin(drawProgress * M_PI * 6) * 10.0f;
                break;
            case WeaponType::HAMMER:
                // Heavy hammer with momentum
                hammerEase = drawProgress < 0.7f ? 
                    (drawProgress / 0.7f) * (drawProgress / 0.7f) : // Slow start
                    1.0f - ((1.0f - drawProgress) / 0.3f) * ((1.0f - drawProgress) / 0.3f); // Fast finish
                player.weaponSwitchOffset = glm::vec3(
                    (1.0f - hammerEase) * -0.6f,
                    (1.0f - hammerEase) * 0.8f - (1.0f - hammerEase) * (1.0f - hammerEase) * 1.5f + hammerEase * 0.3f,
                    (1.0f - hammerEase) * -1.0f
                );
                player.weaponSwitchRotation = (1.0f - hammerEase) * 90.0f;
                break;
            default:
                // Fists - simple fade in
                player.weaponSwitchOffset = glm::vec3(0.0f);
                player.weaponSwitchRotation = 0.0f;
                break;
        }
    }
}

// Flying enemy animation system
void updateFlyingEnemyAnimations(float deltaTime) {
    for (auto& enemy : enemies) {
        if (enemy.isDead || enemy.enemyType != EnemyType::FLYING_DRONE) continue;
        
        // Update wing flapping animation
        enemy.wingFlapTimer += deltaTime * enemy.wingFlapSpeed;
        
        // Calculate wing rotation based on flap timer
        float flapCycle = sin(enemy.wingFlapTimer);
        enemy.wingRotation.x = flapCycle * 45.0f; // Primary wing flap
        enemy.wingRotation.y = cos(enemy.wingFlapTimer * 0.7f) * 15.0f; // Secondary motion
        enemy.wingRotation.z = sin(enemy.wingFlapTimer * 1.3f) * 10.0f; // Tertiary twist
        
        // Update hover bouncing
        enemy.hoverBounceTimer += deltaTime * 2.0f; // Slower bounce than wing flap
        float bounceOffset = sin(enemy.hoverBounceTimer) * enemy.hoverBounceAmplitude;
        
        // Apply hover height with bouncing
        if (enemy.isFlying) {
            float targetY = enemy.baseHeight + bounceOffset;
            float heightDiff = targetY - enemy.position.y;
            
            // Smooth hover height adjustment
            enemy.velocity.y = heightDiff * 5.0f; // Spring-like behavior
            
            // Add slight forward momentum based on wing flap
            glm::vec3 forward = glm::normalize(player.position - enemy.position);
            forward.y = 0; // Keep horizontal
            if (glm::length(forward) > 0.1f) {
                float flapPush = abs(flapCycle) * 0.5f;
                enemy.velocity.x += forward.x * flapPush * deltaTime;
                enemy.velocity.z += forward.z * flapPush * deltaTime;
            }
            
            // Wing-based speed boost when aggressive
            if (enemy.isAggressive) {
                enemy.wingFlapSpeed = 15.0f; // Faster wing beats when chasing
                enemy.hoverBounceAmplitude = 0.2f; // Less bouncing, more focused
            } else {
                enemy.wingFlapSpeed = 8.0f; // Normal wing beats
                enemy.hoverBounceAmplitude = 0.5f; // More relaxed bouncing
            }
        }
    }
}

// Render flying enemy wings
void renderFlyingEnemyWings(const Enemy& enemy) {
    if (enemy.enemyType != EnemyType::FLYING_DRONE || enemy.isDead) return;
    
    glm::vec3 enemyPos = enemy.position;
    
    // Left wing
    glm::mat4 leftWingModel = glm::mat4(1.0f);
    leftWingModel = glm::translate(leftWingModel, enemyPos + glm::vec3(-0.8f, 0.2f, 0.0f));
    leftWingModel = glm::rotate(leftWingModel, glm::radians(enemy.wingRotation.x), glm::vec3(1, 0, 0));
    leftWingModel = glm::rotate(leftWingModel, glm::radians(enemy.wingRotation.y + 15.0f), glm::vec3(0, 1, 0));
    leftWingModel = glm::rotate(leftWingModel, glm::radians(enemy.wingRotation.z), glm::vec3(0, 0, 1));
    leftWingModel = glm::scale(leftWingModel, glm::vec3(0.6f, 0.1f, 0.4f));
    
    // Right wing
    glm::mat4 rightWingModel = glm::mat4(1.0f);
    rightWingModel = glm::translate(rightWingModel, enemyPos + glm::vec3(0.8f, 0.2f, 0.0f));
    rightWingModel = glm::rotate(rightWingModel, glm::radians(-enemy.wingRotation.x), glm::vec3(1, 0, 0));
    rightWingModel = glm::rotate(rightWingModel, glm::radians(-enemy.wingRotation.y - 15.0f), glm::vec3(0, 1, 0));
    rightWingModel = glm::rotate(rightWingModel, glm::radians(-enemy.wingRotation.z), glm::vec3(0, 0, 1));
    rightWingModel = glm::scale(rightWingModel, glm::vec3(0.6f, 0.1f, 0.4f));
    
    // Wing color based on flap intensity
    float flapIntensity = abs(sin(enemy.wingFlapTimer)) * 0.3f + 0.4f;
    glm::vec3 wingColor = glm::vec3(0.3f + flapIntensity, 0.6f + flapIntensity * 0.3f, 0.8f + flapIntensity * 0.2f);
    
    // Render both wings
    renderCube(glm::vec3(0.0f), glm::vec3(1.0f), wingColor, leftWingModel);
    renderCube(glm::vec3(0.0f), glm::vec3(1.0f), wingColor, rightWingModel);
    
    // Add wing trail particles when flapping fast
    if (enemy.isAggressive && abs(sin(enemy.wingFlapTimer)) > 0.8f) {
        // Spawn wing trail particles occasionally
        static float trailTimer = 0.0f;
        trailTimer += deltaTime;
        if (trailTimer > 0.1f) {
            glm::vec3 leftWingTip = enemyPos + glm::vec3(-1.2f, 0.2f, 0.0f);
            glm::vec3 rightWingTip = enemyPos + glm::vec3(1.2f, 0.2f, 0.0f);
            
            // Create small wing particle effects
            explosionParticles.emplace_back(leftWingTip, glm::vec3(
                (rand() % 100 - 50) / 50.0f,
                (rand() % 50) / 50.0f,
                (rand() % 100 - 50) / 50.0f
            ), wingColor * 0.7f);
            
            explosionParticles.emplace_back(rightWingTip, glm::vec3(
                (rand() % 100 - 50) / 50.0f,
                (rand() % 50) / 50.0f,
                (rand() % 100 - 50) / 50.0f
            ), wingColor * 0.7f);
            
            trailTimer = 0.0f;
        }
    }
}

void airDashLauncher() {
    if (!player.canAirDash || player.onGround) return;
    
    player.isAirDashing = true;
    player.airDashTimer = 0.3f;
    player.canAirDash = false;
    
    // Launch nearby enemies
    for (auto& enemy : enemies) {
        if (enemy.isDead) continue;
        
        float dist = glm::length(enemy.position - player.position);
        if (dist < 8.0f) {
            glm::vec3 launchDir = glm::normalize(enemy.position - player.position);
            enemy.isFlying = true;
            enemy.flyTimer = 3.0f;
            enemy.flyVelocity = launchDir * 20.0f + glm::vec3(0, 15.0f, 0);
            enemy.health -= 40.0f;
            
            spawnDamageNumber(enemy.position, 40.0f, glm::vec3(1, 0.8f, 0));
            spawnRippleEffect(player.position, 8.0f, glm::vec3(1, 1, 0));
            
            std::cout << "AIR DASH LAUNCHER! Enemy launched!" << std::endl;
        }
    }
    
    cameraShake = 0.8f;
    std::cout << "AIR DASH LAUNCHER ACTIVATED!" << std::endl;
}

void updateController() {
    // Save previous state
    lastController = currentController;
    
    // Reset current state
    memset(&currentController, 0, sizeof(ControllerState));
    
    // Check for connected controllers
    for (int i = GLFW_JOYSTICK_1; i <= GLFW_JOYSTICK_LAST; i++) {
        if (glfwJoystickPresent(i)) {
            controllerID = i;
            useController = true;
            
            // Get axes (sticks and triggers)
            int axisCount;
            const float* axes = glfwGetJoystickAxes(controllerID, &axisCount);
            
            if (axisCount >= 6) {
                // Left stick
                currentController.leftStickX = axes[0];
                currentController.leftStickY = axes[1];
                
                // Right stick
                currentController.rightStickX = axes[2];
                currentController.rightStickY = axes[3];
                
                // Triggers
                currentController.leftTriggerValue = axes[4];
                currentController.rightTriggerValue = axes[5];
                
                // Apply deadzones and filtering
                applyControllerFiltering();
            }
            
            // Get buttons
            int buttonCount;
            const unsigned char* buttons = glfwGetJoystickButtons(controllerID, &buttonCount);
            
            if (buttonCount >= 14) {
                // Standard Xbox/PlayStation mapping
                currentController.buttonA = buttons[0] == GLFW_PRESS;       // A/Cross
                currentController.buttonB = buttons[1] == GLFW_PRESS;       // B/Circle
                currentController.buttonX = buttons[2] == GLFW_PRESS;       // X/Square
                currentController.buttonY = buttons[3] == GLFW_PRESS;       // Y/Triangle
                currentController.leftBumper = buttons[4] == GLFW_PRESS;    // LB/L1
                currentController.rightBumper = buttons[5] == GLFW_PRESS;   // RB/R1
                currentController.back = buttons[6] == GLFW_PRESS;          // Back/Share
                currentController.start = buttons[7] == GLFW_PRESS;         // Start/Options
                currentController.leftStick = buttons[8] == GLFW_PRESS;     // LS/L3
                currentController.rightStick = buttons[9] == GLFW_PRESS;    // RS/R3
                
                // Triggers as buttons
                currentController.leftTrigger = currentController.leftTriggerValue > 0.5f;
                currentController.rightTrigger = currentController.rightTriggerValue > 0.5f;
                
                // D-pad
                if (buttonCount >= 18) {
                    currentController.dpadUp = buttons[10] == GLFW_PRESS;
                    currentController.dpadRight = buttons[11] == GLFW_PRESS;
                    currentController.dpadDown = buttons[12] == GLFW_PRESS;
                    currentController.dpadLeft = buttons[13] == GLFW_PRESS;
                }
            }
            break;
        }
    }
    
    if (!glfwJoystickPresent(controllerID)) {
        useController = false;
        controllerID = -1;
    }
}

void applyControllerFiltering() {
    // Apply deadzone to left stick
    float leftMagnitude = sqrt(currentController.leftStickX * currentController.leftStickX + 
                              currentController.leftStickY * currentController.leftStickY);
    if (leftMagnitude < controllerDeadzone) {
        currentController.leftStickX = 0.0f;
        currentController.leftStickY = 0.0f;
    } else {
        // Scale past deadzone
        float scale = (leftMagnitude - controllerDeadzone) / (1.0f - controllerDeadzone);
        currentController.leftStickX = (currentController.leftStickX / leftMagnitude) * scale * controllerSensitivity;
        currentController.leftStickY = (currentController.leftStickY / leftMagnitude) * scale * controllerSensitivity;
        
        // Apply Y inversion if enabled
        if (controllerInvertY) {
            currentController.leftStickY = -currentController.leftStickY;
        }
    }
    
    // Apply deadzone to right stick
    float rightMagnitude = sqrt(currentController.rightStickX * currentController.rightStickX + 
                               currentController.rightStickY * currentController.rightStickY);
    if (rightMagnitude < controllerDeadzone) {
        currentController.rightStickX = 0.0f;
        currentController.rightStickY = 0.0f;
    } else {
        float scale = (rightMagnitude - controllerDeadzone) / (1.0f - controllerDeadzone);
        currentController.rightStickX = (currentController.rightStickX / rightMagnitude) * scale * controllerSensitivity;
        currentController.rightStickY = (currentController.rightStickY / rightMagnitude) * scale * controllerSensitivity;
        
        if (controllerInvertY) {
            currentController.rightStickY = -currentController.rightStickY;
        }
    }
    
    // Clamp trigger values
    currentController.leftTriggerValue = std::max(0.0f, std::min(1.0f, currentController.leftTriggerValue));
    currentController.rightTriggerValue = std::max(0.0f, std::min(1.0f, currentController.rightTriggerValue));
}

glm::vec2 getControllerInput() {
    if (!useController) {
        return glm::vec2(0.0f);
    }
    return glm::vec2(currentController.leftStickX, currentController.leftStickY);
}

bool isControllerButtonPressed(int buttonIndex) {
    if (!useController) return false;
    
    switch(buttonIndex) {
        case 0: return currentController.buttonA;
        case 1: return currentController.buttonB;
        case 2: return currentController.buttonX;
        case 3: return currentController.buttonY;
        case 4: return currentController.leftBumper;
        case 5: return currentController.rightBumper;
        case 6: return currentController.leftTrigger;
        case 7: return currentController.rightTrigger;
        case 8: return currentController.dpadUp;
        case 9: return currentController.dpadDown;
        case 10: return currentController.dpadLeft;
        case 11: return currentController.dpadRight;
        case 12: return currentController.start;
        case 13: return currentController.back;
        case 14: return currentController.leftStick;
        case 15: return currentController.rightStick;
    }
    return false;
}

bool isControllerButtonJustPressed(int buttonIndex) {
    return isControllerButtonPressed(buttonIndex) && !wasControllerButtonPressed(buttonIndex);
}

bool wasControllerButtonPressed(int buttonIndex) {
    switch(buttonIndex) {
        case 0: return lastController.buttonA;
        case 1: return lastController.buttonB;
        case 2: return lastController.buttonX;
        case 3: return lastController.buttonY;
        case 4: return lastController.leftBumper;
        case 5: return lastController.rightBumper;
        case 6: return lastController.leftTrigger;
        case 7: return lastController.rightTrigger;
        case 8: return lastController.dpadUp;
        case 9: return lastController.dpadDown;
        case 10: return lastController.dpadLeft;
        case 11: return lastController.dpadRight;
        case 12: return lastController.start;
        case 13: return lastController.back;
        case 14: return lastController.leftStick;
        case 15: return lastController.rightStick;
    }
    return false;
}

void startWeaponCombo(WeaponType weaponType, int comboStep) {
    WeaponAnimation& anim = player.currentWeapon.currentAnimation;
    anim.isActive = true;
    anim.animationTime = 0.0f;
    
    switch(weaponType) {
        case WeaponType::SWORD:
            performSwordCombo(comboStep);
            break;
        case WeaponType::STAFF:
            if (comboStep == 3) performStaffSpell();
            else {
                anim.comboState = (comboStep == 1) ? CombatState::STAFF_SPIN : CombatState::STAFF_SLAM;
                anim.totalDuration = 0.8f;
                anim.rotationAxis = glm::vec3(0, 1, 0);
                anim.rotationAngle = (comboStep == 1) ? 360.0f : 180.0f;
            }
            break;
        case WeaponType::HAMMER:
            if (comboStep == 3) performHammerGroundPound();
            else {
                anim.comboState = (comboStep == 1) ? CombatState::HAMMER_SWING : CombatState::HAMMER_OVERHEAD;
                anim.totalDuration = 1.0f;
                anim.rotationAxis = glm::vec3(1, 0, 0);
                anim.rotationAngle = (comboStep == 1) ? 90.0f : 180.0f;
            }
            break;
        default:
            anim.comboState = CombatState::PUNCH;
            anim.totalDuration = 0.5f;
            break;
    }
}

void performSwordCombo(int step) {
    WeaponAnimation& anim = player.currentWeapon.currentAnimation;
    anim.totalDuration = 0.6f;
    
    switch(step) {
        case 1:
            anim.comboState = CombatState::SWORD_SLASH_1;
            anim.rotationAxis = glm::vec3(0, 0, 1);
            anim.rotationAngle = 120.0f;
            anim.positionOffset = glm::vec3(0.5f, 0, 0);
            break;
        case 2:
            anim.comboState = CombatState::SWORD_SLASH_2;
            anim.rotationAxis = glm::vec3(0, 0, -1);
            anim.rotationAngle = 120.0f;
            anim.positionOffset = glm::vec3(-0.5f, 0, 0);
            break;
        case 3:
            anim.comboState = CombatState::SWORD_THRUST;
            anim.rotationAxis = glm::vec3(1, 0, 0);
            anim.rotationAngle = 45.0f;
            anim.positionOffset = glm::vec3(0, 0, 1.5f);
            anim.totalDuration = 0.8f;
            break;
    }
}

void performStaffSpell() {
    WeaponAnimation& anim = player.currentWeapon.currentAnimation;
    anim.comboState = CombatState::STAFF_CAST;
    anim.totalDuration = 1.5f;
    anim.rotationAxis = glm::vec3(0, 1, 0);
    anim.rotationAngle = 720.0f; // Two full spins
    anim.positionOffset = glm::vec3(0, 2.0f, 0);
    
    // Create magical effects
    spawnRippleEffect(player.position + glm::vec3(0, 1, 0), 8.0f, glm::vec3(0.5f, 0.3f, 0.8f));
    spawnTriangleExplosion(player.position + glm::vec3(0, 2, 0), 15);
    cameraShake = 0.4f;
    
    // Damage nearby enemies with magic
    for (auto& enemy : enemies) {
        if (!enemy.isDead) {
            float distance = glm::length(enemy.position - player.position);
            if (distance < 10.0f) {
                float magicDamage = 35.0f * (1.0f - distance / 10.0f);
                enemy.health -= magicDamage;
                spawnDamageNumber(enemy.position, magicDamage, glm::vec3(0.8f, 0.3f, 1.0f));
                if (enemy.health <= 0) {
                    enemy.isDead = true;
                    enemy.deathTimer = 2.0f;
                    score += 500;
                }
            }
        }
    }
    
    std::cout << "STAFF SPELL CAST!" << std::endl;
}

void performHammerGroundPound() {
    WeaponAnimation& anim = player.currentWeapon.currentAnimation;
    anim.comboState = CombatState::HAMMER_GROUND_POUND;
    anim.totalDuration = 1.2f;
    anim.rotationAxis = glm::vec3(1, 0, 0);
    anim.rotationAngle = 270.0f;
    anim.positionOffset = glm::vec3(0, -1.0f, 0);
    
    // Player flips forward during ground pound
    player.isFlipping = true;
    player.flipRotation = 0.0f;
    player.flipSpeed = 20.0f;
    
    // Create powerful ground impact
    spawnRippleEffect(player.position, 12.0f, glm::vec3(0.8f, 0.4f, 0.2f));
    spawnTriangleExplosion(player.position, 25);
    cameraShake = 0.8f;
    
    // Damage and knock back all nearby enemies
    for (auto& enemy : enemies) {
        if (!enemy.isDead) {
            float distance = glm::length(enemy.position - player.position);
            if (distance < 12.0f) {
                float hammerDamage = 60.0f * (1.0f - distance / 12.0f);
                enemy.health -= hammerDamage;
                enemy.velocity.y = 20.0f; // Launch enemies
                enemy.isLaunched = true;
                enemy.launchTimer = 1.5f;
                
                spawnDamageNumber(enemy.position, hammerDamage, glm::vec3(1.0f, 0.6f, 0.2f));
                if (enemy.health <= 0) {
                    enemy.isDead = true;
                    enemy.deathTimer = 2.0f;
                    score += 750;
                }
            }
        }
    }
    
    std::cout << "HAMMER GROUND POUND!" << std::endl;
}

void updateWeaponAnimation(float deltaTime) {
    WeaponAnimation& anim = player.currentWeapon.currentAnimation;
    
    if (anim.isActive) {
        anim.animationTime += deltaTime;
        
        if (anim.animationTime >= anim.totalDuration) {
            anim.isActive = false;
            anim.animationTime = 0.0f;
            anim.comboState = CombatState::NONE;
            
            // Reset player flip if it was a ground pound
            if (anim.comboState == CombatState::HAMMER_GROUND_POUND) {
                player.isFlipping = false;
                player.flipRotation = 0.0f;
            }
        }
    }
}

void resetGame() {
    player.position = glm::vec3(10.0f, 1.0f, 10.0f);  // Spawn away from center pillar
    player.velocity = glm::vec3(0.0f);
    player.rotation = 0.0f;
    player.combatState = CombatState::NONE;
    player.comboState = COMBO_NONE;
    player.comboCount = 0;
    player.isFlipping = false;
    player.flipRotation = 0.0f;
    player.health = player.maxHealth;
    player.isDead = false;
    player.sprintToggled = false;
    player.isDashing = false;
    player.isDiving = false;
    player.isWeaponDiving = false;
    player.rhythm = RhythmState();
    player.currentWeapon.currentAnimation.isActive = false;
    
    // Reset camera to follow player at spawn position
    if (gameCamera) {
        gameCamera->resetToPlayer(player.position);
        gameCamera->clearLockOnTarget();
    }
    
    gameState = GAME_PLAYING;
    targetedEnemyIndex = -1;
    cameraShake = 0.0f;
    deathScreenTimer = 0.0f;
    playerLives = 3;
    score = 0;
    
    damageNumbers.clear();
    
    for (auto& collectible : collectibles) {
        collectible.collected = false;
    }
    setupWorld(); // Reset enemies too
    std::cout << "Game reset!\n";
}

// Zone reward system implementation
void assignRandomRewards() {
    // Randomly assign zone rewards to 30% of enemies
    for (auto& enemy : enemies) {
        if (rand() % 100 < 30) { // 30% chance
            enemy.rewardState = Enemy::CARRYING_REWARD;
            enemy.carriedReward = ZoneReward(static_cast<ZoneRewardType>(rand() % 9));
            enemy.rewardGlowIntensity = 1.0f;
            enemy.rewardAura = enemy.carriedReward.color;
            
            // Mini-bosses always carry premium rewards
            if (enemy.miniBossData) {
                enemy.carriedReward = enemy.miniBossData->stolenReward;
                enemy.rewardOrbitCount = 5; // More orbiting elements for bosses
                enemy.rewardOrbitRadius = 3.0f;
            }
        }
    }
    std::cout << "Assigned zone rewards to enemies!" << std::endl;
}

void updateEnemyRewardStates(float deltaTime) {
    for (auto& enemy : enemies) {
        if (enemy.isDead || enemy.rewardState == Enemy::NO_REWARD) continue;
        
        // Update pulse and glow effects
        enemy.rewardPulseTimer += deltaTime * 2.0f;
        enemy.rewardGlowIntensity = 0.7f + 0.3f * sin(enemy.rewardPulseTimer);
        
        switch (enemy.rewardState) {
            case Enemy::CARRYING_REWARD:
                // Check if enemy should start sparking (low health or player nearby)
                if (enemy.health < enemy.maxHealth * 0.3f || 
                    glm::length(player.position - enemy.position) < 5.0f) {
                    enemy.rewardState = Enemy::ABOUT_TO_SPARK;
                    enemy.isAboutToSpark = true;
                    enemy.sparkBuildupTimer = 0.0f;
                }
                break;
                
            case Enemy::ABOUT_TO_SPARK:
                enemy.sparkBuildupTimer += deltaTime;
                enemy.rewardGlowIntensity *= (1.0f + enemy.sparkBuildupTimer * 0.5f);
                
                // Generate pre-spark particles
                if (static_cast<int>(enemy.sparkBuildupTimer * 10) % 3 == 0) {
                    glm::vec3 sparkPos = enemy.position + glm::vec3(
                        (rand() % 200 - 100) / 100.0f,
                        (rand() % 200 - 100) / 100.0f + 1.0f,
                        (rand() % 200 - 100) / 100.0f
                    );
                    enemy.sparkParticles.push_back(sparkPos);
                }
                
                if (enemy.sparkBuildupTimer > 2.0f) {
                    enemy.rewardState = Enemy::SPARKING;
                    enemy.sparkTimer = 0.0f;
                }
                break;
                
            case Enemy::SPARKING:
                enemy.sparkTimer += deltaTime;
                enemy.rewardGlowIntensity = 2.0f + sin(enemy.sparkTimer * 10.0f);
                
                // Generate intense spark particles
                for (int i = 0; i < 3; i++) {
                    glm::vec3 sparkPos = enemy.position + glm::vec3(
                        (rand() % 300 - 150) / 100.0f,
                        (rand() % 300 - 150) / 100.0f + 2.0f,
                        (rand() % 300 - 150) / 100.0f
                    );
                    enemy.sparkParticles.push_back(sparkPos);
                }
                
                if (enemy.sparkTimer > enemy.sparkDuration) {
                    // Activate the reward
                    activateZoneReward(enemy.carriedReward);
                    enemy.rewardState = Enemy::REWARD_ACTIVATED;
                    enemy.sparkParticles.clear();
                    
                    // Create activation effect
                    spawnRippleEffect(enemy.position, 10.0f, enemy.rewardAura);
                    spawnTriangleExplosion(enemy.position, 20);
                    std::cout << "Enemy activated reward: " << enemy.carriedReward.name << std::endl;
                }
                break;
                
            case Enemy::REWARD_ACTIVATED:
                // Fade out effects after activation
                enemy.rewardGlowIntensity *= 0.95f;
                if (enemy.rewardGlowIntensity < 0.1f) {
                    enemy.rewardState = Enemy::NO_REWARD;
                }
                break;
        }
        
        // Clean up old spark particles
        if (enemy.sparkParticles.size() > 50) {
            enemy.sparkParticles.erase(enemy.sparkParticles.begin(), 
                                     enemy.sparkParticles.begin() + 10);
        }
    }
}

void checkZoneRewardActivation() {
    for (auto& zone : rewardZones) {
        if (zone.isTriggered) {
            zone.cooldownTimer -= deltaTime;
            if (zone.cooldownTimer <= 0.0f) {
                zone.isTriggered = false;
            }
            continue;
        }
        
        // Check if player is in zone
        bool inZone = checkCollision(player.position, player.size, zone.position, zone.size);
        
        if (inZone) {
            bool canActivate = true;
            
            if (zone.requiresMovementEntry) {
                float playerSpeed = glm::length(player.velocity);
                if (playerSpeed < zone.minVelocityRequired) {
                    canActivate = false;
                }
                
                // Check movement direction if specified
                if (glm::length(zone.entryDirection) > 0.1f) {
                    glm::vec3 playerDir = glm::normalize(player.velocity);
                    float dot = glm::dot(playerDir, zone.entryDirection);
                    if (dot < 0.7f) canActivate = false;
                }
            }
            
            if (canActivate) {
                activateZoneReward(zone.reward);
                zone.isTriggered = true;
                zone.cooldownTimer = zone.cooldownDuration;
                
                spawnRippleEffect(zone.position, 15.0f, zone.reward.color);
                std::cout << "Zone reward activated: " << zone.reward.name << std::endl;
            }
        }
    }
}

void activateZoneReward(ZoneReward& reward) {
    reward.isActive = true;
    reward.timer = reward.duration;
    player.activeRewards.push_back(reward);
    
    // Apply immediate effects
    switch (reward.type) {
        case ZoneRewardType::STAMINA_REFUND:
            player.health = std::min(player.maxHealth, player.health + reward.magnitude);
            break;
        case ZoneRewardType::DOUBLE_JUMP_REFRESH:
            player.canDoubleJump = true;
            player.hasDoubleJumped = false;
            player.canAirDash = true;
            break;
        case ZoneRewardType::INVINCIBILITY:
            player.hasInvincibility = true;
            break;
    }
    
    player.score += 1000; // Bonus points for activating rewards
}

void updatePlayerRewards(float deltaTime) {
    player.globalTimeScale = 1.0f; // Reset time scale
    player.hasInvincibility = false;
    
    // Update active rewards
    for (auto it = player.activeRewards.begin(); it != player.activeRewards.end();) {
        it->timer -= deltaTime;
        
        if (it->timer <= 0.0f) {
            // Reward expired
            it = player.activeRewards.erase(it);
            continue;
        }
        
        // Apply reward effects
        switch (it->type) {
            case ZoneRewardType::SLOW_MOTION:
                player.globalTimeScale = it->magnitude;
                break;
            case ZoneRewardType::COMBO_BONUS:
                player.damageMultiplier = 1.0f + it->magnitude;
                break;
            case ZoneRewardType::DAMAGE_BOOST:
                player.damageMultiplier = it->magnitude;
                break;
            case ZoneRewardType::INVINCIBILITY:
                player.hasInvincibility = true;
                break;
            case ZoneRewardType::RHYTHM_PERFECT:
                player.rhythm.accuracy = 1.0f;
                break;
            case ZoneRewardType::SPEED_BOOST:
                player.speed = 15.0f * it->magnitude;
                player.sprintMultiplier = 4.0f * it->magnitude;
                break;
            case ZoneRewardType::WALL_RUN_EXTENDED:
                player.maxWallRunTime = 3.0f * it->magnitude;
                break;
        }
        
        ++it;
    }
}

void stealEnemyReward(int enemyIndex) {
    if (enemyIndex < 0 || enemyIndex >= enemies.size()) return;
    
    Enemy& enemy = enemies[enemyIndex];
    if (enemy.rewardState != Enemy::CARRYING_REWARD && 
        enemy.rewardState != Enemy::ABOUT_TO_SPARK) return;
    
    // Steal the reward
    activateZoneReward(enemy.carriedReward);
    enemy.rewardState = Enemy::NO_REWARD;
    enemy.sparkParticles.clear();
    
    // Bonus points for stealing
    player.score += 2000;
    
    // Create steal effect
    spawnRippleEffect(enemy.position, 8.0f, enemy.rewardAura);
    spawnTriangleExplosion(enemy.position, 15);
    
    std::cout << "Stole reward: " << enemy.carriedReward.name << " from enemy!" << std::endl;
}

// Collision detection
bool checkCollision(const glm::vec3& pos1, const glm::vec3& size1, const glm::vec3& pos2, const glm::vec3& size2) {
    return (pos1.x - size1.x/2 < pos2.x + size2.x/2 && pos1.x + size1.x/2 > pos2.x - size2.x/2 &&
            pos1.y - size1.y/2 < pos2.y + size2.y/2 && pos1.y + size1.y/2 > pos2.y - size2.y/2 &&
            pos1.z - size1.z/2 < pos2.z + size2.z/2 && pos1.z + size1.z/2 > pos2.z - size2.z/2);
}

void processInput() {
    /* COMPLETE ENHANCED CONTROL LAYOUT:
     * KEYBOARD         | CONTROLLER      | ACTION
     * -----------------|-----------------|------------------
     * WASD             | Left Stick      | Movement
     * Space            | A Button        | Jump/Wall-Jump
     * X                | X Button        | Dive (when airborne)
     * Y                | Y Button        | Weapon Dive Charge
     * F                | Right Bumper    | Dash Attack
     * G                | Left Bumper     | Air Dash Launcher
     * Q                | Right Trigger   | Light Attack Combo
     * E                | Left Trigger    | Kick/Launcher
     * C                | Right Stick     | Slide (on ground with speed)
     * V                | Back Button     | Grab Enemy
     * B                | B Button        | Parry (hold to block)
     * Left Shift       | Left Stick      | Sprint Toggle
     * 1-4              | D-Pad           | Weapon Switch
     *                  | D-Pad Up        | Sword (2)
     *                  | D-Pad Down      | Hammer (4)
     *                  | D-Pad Left      | Fists (1)
     *                  | D-Pad Right     | Staff (3)
     *
     * ENHANCED MECHANICS:
     * • Slide + Jump = Long Jump with momentum
     * • Grab + Attack = Throw enemy with stun
     * • Parry + Attack = Counter attack with bonus damage
     * • Perfect parries/stun combos = Progressive enhancement
     * • Damage multiplier and health bonus system
     */
    
    // Update controller input
    updateController();
    
    if (gameState == GAME_GAME_OVER) {
        // Death state input
        if (keysPressed[GLFW_KEY_R]) {
            gameState = GAME_RESTART;
        }
        return;
    }
    
    if (gameState != GAME_PLAYING) return;
    
    // Update targeting
    updateTargeting();
    
    // Reset horizontal velocity if not dashing
    if (!player.isDashing && !player.isDiving && !player.isWeaponDiving && !player.isChargingWeaponDive) {
        float currentY = player.velocity.y;
        player.velocity = glm::vec3(0.0f, currentY, 0.0f);
    }
    
    // Movement input - keyboard + controller
    glm::vec3 inputDir(0.0f);
    if (keys[GLFW_KEY_W]) inputDir.z -= 1.0f;
    if (keys[GLFW_KEY_S]) inputDir.z += 1.0f;
    if (keys[GLFW_KEY_A]) inputDir.x -= 1.0f;
    if (keys[GLFW_KEY_D]) inputDir.x += 1.0f;
    
    // Add controller input (Y needs to be negated for correct forward/backward movement)
    glm::vec2 controllerInput = getControllerInput();
    inputDir.x += controllerInput.x;
    inputDir.z -= controllerInput.y; // Negate Y so stick up = forward (negative Z)
    
    // Sprint toggle - Left Shift/Left Stick Click
    bool sprintPressed = keysPressed[GLFW_KEY_LEFT_SHIFT] || isControllerButtonJustPressed(14); // LS button
    if (sprintPressed) {
        player.sprintToggled = !player.sprintToggled;
        std::cout << "Sprint " << (player.sprintToggled ? "ON" : "OFF") << std::endl;
    }
    
    // Handle dash
    if (player.isDashing) {
        player.dashTimer -= deltaTime;
        if (player.dashTimer <= 0) {
            player.isDashing = false;
            player.comboState = COMBO_DASH;
            player.comboWindow = 0.8f;
        } else {
            player.velocity = player.dashDirection * player.dashSpeed;
        }
    } else if (glm::length(inputDir) > 0.0f && !player.isDiving && !player.isWeaponDiving) {
        inputDir = glm::normalize(inputDir);
        
        // Face target or movement direction relative to camera
        if (targetedEnemyIndex >= 0) {
            // Speed calculation with sprint toggle
            float currentSpeed = player.speed;
            if (player.sprintToggled) {
                currentSpeed *= player.sprintMultiplier;
            }
            
            glm::vec3 toEnemy = enemies[targetedEnemyIndex].position - player.position;
            toEnemy.y = 0;
            if (glm::length(toEnemy) > 0.1f) {
                player.rotation = atan2(toEnemy.x, -toEnemy.z);
            }
            
            // Apply movement
            player.velocity.x = inputDir.x * currentSpeed;
            player.velocity.z = inputDir.z * currentSpeed;
        } else {
            // Speed calculation with sprint toggle
            float currentSpeed = player.speed;
            if (player.sprintToggled) {
                currentSpeed *= player.sprintMultiplier;
            }
            
            // Calculate movement direction relative to camera
            glm::vec3 cameraForward = (gameCamera) ? -gameCamera->front : glm::vec3(0.0f, 0.0f, -1.0f);
            glm::vec3 cameraRight = (gameCamera) ? gameCamera->right : glm::vec3(1.0f, 0.0f, 0.0f);
            cameraForward.y = 0.0f;
            cameraRight.y = 0.0f;
            cameraForward = glm::normalize(cameraForward);
            cameraRight = glm::normalize(cameraRight);
            
            // Transform input direction relative to camera
            glm::vec3 worldMovement = cameraForward * -inputDir.z + cameraRight * inputDir.x;
            worldMovement.y = 0;
            
            if (glm::length(worldMovement) > 0.1f) {
                worldMovement = glm::normalize(worldMovement);
                player.rotation = atan2(worldMovement.x, -worldMovement.z);
                
                // Apply movement with proper speed
                player.velocity.x = worldMovement.x * currentSpeed;
                player.velocity.z = worldMovement.z * currentSpeed;
            }
        }
    }
    
    // Enhanced jump mechanics with separate jump and dive controls
    // Jump: Space/A Button
    bool jumpPressed = keysPressed[GLFW_KEY_SPACE] || isControllerButtonJustPressed(0); // A button
    
    if (jumpPressed) {
        if (player.onGround) {
            // Ground jump
            player.velocity.y = player.jumpForce;
            player.onGround = false;
            player.canDoubleJump = true;
            player.hasDoubleJumped = false;
            player.isFlipping = false;
            player.flipRotation = 0.0f;
            player.hasLandedFromDive = false; // Reset dive landing flag
            player.animController->setState("jump");
            std::cout << "Jump!" << std::endl;
            
            // Debug logging for jump
            if (debugInfo.debugEnabled) {
                std::cout << "[DEBUG] Jump Event:" << std::endl;
                std::cout << "  Player Position: (" << player.position.x << ", " << player.position.y << ", " << player.position.z << ")" << std::endl;
                std::cout << "  Player Velocity: (" << player.velocity.x << ", " << player.velocity.y << ", " << player.velocity.z << ")" << std::endl;
                std::cout << "  Jump Force Applied: " << player.jumpForce << std::endl;
                if (gameCamera) {
                    std::cout << "  Camera Position: (" << gameCamera->position.x << ", " << gameCamera->position.y << ", " << gameCamera->position.z << ")" << std::endl;
                    std::cout << "  Camera Target: (" << gameCamera->target.x << ", " << gameCamera->target.y << ", " << gameCamera->target.z << ")" << std::endl;
                }
            }
        } else if (player.isWallRunning && player.canWallJump) {
            // Wall-jump with directional control
            glm::vec3 jumpDirection = player.wallNormal;
            
            // Allow player to aim the wall-jump in a ~90 degree arc
            glm::vec3 inputDir(0.0f);
            if (keys[GLFW_KEY_W]) inputDir.z -= 1.0f;
            if (keys[GLFW_KEY_S]) inputDir.z += 1.0f;
            if (keys[GLFW_KEY_A]) inputDir.x -= 1.0f;
            if (keys[GLFW_KEY_D]) inputDir.x += 1.0f;
            
            // Add controller input for jump direction
            glm::vec2 controllerInput = getControllerInput();
            inputDir.x += controllerInput.x;
            inputDir.z -= controllerInput.y; // Negate Y so stick up = forward (negative Z)
            
            if (glm::length(inputDir) > 0.1f) {
                inputDir = glm::normalize(inputDir);
                
                // Calculate the desired jump direction within 90 degree cone
                glm::vec3 forward = -player.wallNormal; // Away from wall
                glm::vec3 right = glm::cross(forward, glm::vec3(0, 1, 0));
                
                // Mix the wall normal with player input (limit to 90 degrees)
                glm::vec3 aimDirection = forward + (right * inputDir.x * 0.8f) + (glm::cross(right, glm::vec3(0, 1, 0)) * inputDir.z * 0.5f);
                aimDirection = glm::normalize(aimDirection);
                
                // Ensure the jump direction is within reasonable bounds
                float angleFromNormal = glm::dot(aimDirection, player.wallNormal);
                if (angleFromNormal > -0.3f) { // Limit to ~90 degrees
                    jumpDirection = aimDirection;
                }
            }
            
            // Perform the wall-jump
            player.velocity = jumpDirection * player.wallJumpForce;
            player.velocity.y = player.jumpForce * 1.1f; // Add upward component
            
            // End wall-running
            player.isWallRunning = false;
            player.wallRunPillarIndex = -1;
            player.canWallJump = false;
            player.canDoubleJump = true;
            
            // Visual feedback
            spawnRippleEffect(player.position, 4.0f, glm::vec3(0.2f, 0.8f, 1.0f));
            cameraShake = 0.3f;
            
            player.animController->setState("jump");
            std::cout << "WALL-JUMP! Leaping in aimed direction!" << std::endl;
        } else if (player.canDoubleJump && !player.hasDoubleJumped) {
            // Double jump
            player.velocity.y = player.jumpForce * 1.2f;
            player.hasDoubleJumped = true;
            player.canDoubleJump = false;
            player.isFlipping = true;
            player.flipRotation = 0.0f;
            player.animController->setState("doublejump");
            std::cout << "Double Jump!" << std::endl;
            
            // Debug logging for double jump
            if (debugInfo.debugEnabled) {
                std::cout << "[DEBUG] Double Jump Event:" << std::endl;
                std::cout << "  Player Position: (" << player.position.x << ", " << player.position.y << ", " << player.position.z << ")" << std::endl;
                std::cout << "  Player Velocity Before: (" << player.velocity.x << ", " << player.velocity.y << ", " << player.velocity.z << ")" << std::endl;
                std::cout << "  Double Jump Force Applied: " << (player.jumpForce * 1.2f) << std::endl;
                if (gameCamera) {
                    std::cout << "  Camera Position: (" << gameCamera->position.x << ", " << gameCamera->position.y << ", " << gameCamera->position.z << ")" << std::endl;
                    std::cout << "  Camera Target: (" << gameCamera->target.x << ", " << gameCamera->target.y << ", " << gameCamera->target.z << ")" << std::endl;
                    std::cout << "  Camera Distance from Player: " << glm::length(gameCamera->position - player.position) << std::endl;
                }
            }
        }
    }
    
    // Dive: X key/X Button (separate from jump)
    bool divePressed = keysPressed[GLFW_KEY_X] || isControllerButtonJustPressed(2); // X button
    
    if (divePressed && !player.onGround && !player.isDiving) {
        // Start dive
        player.isDiving = true;
        player.diveDirection = glm::normalize(glm::vec3(player.velocity.x, -1.0f, player.velocity.z));
        player.velocity = player.diveDirection * player.diveSpeed;
        player.animController->setState("dive");
        std::cout << "DIVING!" << std::endl;
    }
    
    // Dash attack - F key/Right Bumper
    bool dashPressed = keysPressed[GLFW_KEY_F] || isControllerButtonJustPressed(5); // RB button
    
    if (dashPressed && !player.isDashing) {
        if (targetedEnemyIndex >= 0) {
            glm::vec3 toEnemy = enemies[targetedEnemyIndex].position - player.position;
            toEnemy.y = 0;
            if (glm::length(toEnemy) > 0.1f) {
                player.dashDirection = glm::normalize(toEnemy);
                player.isDashing = true;
                player.dashTimer = 0.25f;
                player.comboWindow = 0.8f;
                cameraShake = 0.1f;
                std::cout << "Dash to enemy!" << std::endl;
            }
        } else if (glm::length(inputDir) > 0.1f) {
            player.dashDirection = inputDir;
            player.isDashing = true;
            player.dashTimer = 0.25f;
            player.comboWindow = 0.8f;
        }
    }
    
    // Enhanced weapon dive (X while airborne) - Hold to charge, release to explode
    bool xPressed = keys[GLFW_KEY_X] || isControllerButtonPressed(2);
    bool xJustPressed = keysPressed[GLFW_KEY_X] || isControllerButtonJustPressed(2);
    
    if (!player.onGround) {
        if (xPressed && !player.isChargingWeaponDive && !player.isWeaponDiving) {
            // Start charging weapon dive
            player.isChargingWeaponDive = true;
            player.weaponDiveChargeTime = 0.0f;
            player.drawnEnemyIndices.clear();
            std::cout << "CHARGING WEAPON DIVE..." << std::endl;
        }
        
        if (player.isChargingWeaponDive) {
            if (xPressed) {
                // Continue charging
                player.weaponDiveChargeTime += deltaTime;
                
                // Draw enemies towards player
                for (int i = 0; i < enemies.size(); i++) {
                    if (enemies[i].isDead) continue;
                    
                    float dist = glm::length(enemies[i].position - player.position);
                    float drawRadius = 8.0f + (player.weaponDiveChargeTime / player.maxChargeTime) * 12.0f;
                    
                    if (dist < drawRadius) {
                        glm::vec3 pullDir = glm::normalize(player.position - enemies[i].position);
                        float pullForce = 15.0f + (player.weaponDiveChargeTime / player.maxChargeTime) * 25.0f;
                        
                        enemies[i].velocity += pullDir * pullForce * deltaTime;
                        
                        // Add to drawn enemies list if not already there
                        if (std::find(player.drawnEnemyIndices.begin(), 
                                     player.drawnEnemyIndices.end(), i) == player.drawnEnemyIndices.end()) {
                            player.drawnEnemyIndices.push_back(i);
                        }
                    }
                }
                
                // Clamp charge time
                if (player.weaponDiveChargeTime > player.maxChargeTime) {
                    player.weaponDiveChargeTime = player.maxChargeTime;
                }
            } else {
                // Released X - trigger explosion!
                weaponDiveExplosion();
            }
        } else if (xJustPressed && !player.isWeaponDiving) {
            // Quick tap for regular weapon dive
            player.isWeaponDiving = true;
            glm::vec3 targetDir = targetedEnemyIndex >= 0 ? 
                enemies[targetedEnemyIndex].position - player.position :
                glm::vec3(0, -1, 0);
            player.diveDirection = glm::normalize(targetDir + glm::vec3(0, -2, 0));
            player.velocity = player.diveDirection * 60.0f;
            cameraShake = 0.5f;
            std::cout << "WEAPON DIVE!" << std::endl;
        }
    }
    
    // Air dash launcher - G key/Left Bumper
    bool airDashPressed = keysPressed[GLFW_KEY_G] || isControllerButtonJustPressed(4); // LB button
    
    if (airDashPressed && !player.onGround && player.canAirDash) {
        airDashLauncher();
    }
    
    // Update combo timer
    if (player.comboWindow > 0) {
        player.comboWindow -= deltaTime;
        if (player.comboWindow <= 0) {
            player.comboState = COMBO_NONE;
            player.comboCount = 0;
        }
    }
    
    // Declare attack input variables early for use in enhanced mechanics
    bool lightAttackPressed = keysPressed[GLFW_KEY_Q] || (currentController.rightTrigger && !lastController.rightTrigger); // RT
    bool kickPressed = keysPressed[GLFW_KEY_E] || (currentController.leftTrigger && !lastController.leftTrigger); // LT
    
    // === NEW ENHANCED MECHANICS ===
    
    // Slide mechanic - C key/B button
    bool slidePressed = keysPressed[GLFW_KEY_C] || isControllerButtonJustPressed(1); // B/Circle button
    
    if (slidePressed && player.onGround && !player.isSliding && glm::length(glm::vec2(player.velocity.x, player.velocity.z)) > 5.0f) {
        player.isSliding = true;
        player.slideTimer = player.slideDuration;
        player.combatState = CombatState::SLIDE;
        player.combatTimer = player.slideDuration;
        player.animController->setState("slide");
        
        // Boost forward momentum for slide
        glm::vec3 slideDir = glm::normalize(glm::vec3(player.velocity.x, 0, player.velocity.z));
        player.velocity.x = slideDir.x * player.slideSpeed;
        player.velocity.z = slideDir.z * player.slideSpeed;
        
        std::cout << "SLIDING!" << std::endl;
    }
    
    // Slide jump for long jump - Space while sliding
    if (player.isSliding && jumpPressed && player.canSlideJump) {
        player.velocity.y = player.slideJumpForce;
        player.isSliding = false;
        player.slideTimer = 0.0f;
        player.combatState = CombatState::SLIDE_LEAP;
        player.combatTimer = 0.8f;
        player.onGround = false;
        player.animController->setState("slidejump");
        
        // Maintain forward momentum in air
        glm::vec3 slideDir = glm::normalize(glm::vec3(player.velocity.x, 0, player.velocity.z));
        player.velocity.x = slideDir.x * (player.slideSpeed * 0.8f);
        player.velocity.z = slideDir.z * (player.slideSpeed * 0.8f);
        
        std::cout << "SLIDE LEAP! Long jump activated!" << std::endl;
    }
    
    // Grab mechanic - V key/Back button
    bool grabPressed = keysPressed[GLFW_KEY_V] || isControllerButtonJustPressed(6); // Back/Select button
    
    if (grabPressed && !player.isGrabbing && targetedEnemyIndex >= 0) {
        float distToTarget = glm::length(enemies[targetedEnemyIndex].position - player.position);
        if (distToTarget <= player.grabRange && !enemies[targetedEnemyIndex].isDead && !enemies[targetedEnemyIndex].isStunned) {
            player.isGrabbing = true;
            player.grabbedEnemyIndex = targetedEnemyIndex;
            player.grabTimer = player.grabDuration;
            player.combatState = CombatState::GRAB;
            player.combatTimer = player.grabDuration;
            
            enemies[targetedEnemyIndex].isGrabbed = true;
            enemies[targetedEnemyIndex].velocity = glm::vec3(0);
            
            std::cout << "GRABBED ENEMY!" << std::endl;
        }
    }
    
    // Grab throw - Any attack button while grabbing
    if (player.isGrabbing && (lightAttackPressed || kickPressed)) {
        if (player.grabbedEnemyIndex >= 0 && player.grabbedEnemyIndex < enemies.size()) {
            Enemy& grabbedEnemy = enemies[player.grabbedEnemyIndex];
            
            // Throw enemy in facing direction
            glm::vec3 throwDir(sin(player.rotation), 0.5f, -cos(player.rotation));
            grabbedEnemy.velocity = throwDir * 25.0f;
            grabbedEnemy.isGrabbed = false;
            grabbedEnemy.isStunned = true;
            grabbedEnemy.stunTimer = grabbedEnemy.stunDuration;
            grabbedEnemy.health -= 60.0f * player.damageMultiplier;
            
            player.isGrabbing = false;
            player.combatState = CombatState::GRAB_THROW;
            player.combatTimer = 0.6f;
            
            // Increase stun combo count for progressive enhancement
            player.stunComboCount++;
            if (player.stunComboCount >= 3) {
                player.damageMultiplier += 0.2f;
                player.maxHealthBonus += 20.0f;
                player.maxHealth = 100.0f + player.maxHealthBonus;
                player.health = std::min(player.health + 15.0f, player.maxHealth);
                player.stunComboCount = 0;
                std::cout << "STUN COMBO BONUS! Damage: " << player.damageMultiplier << "x, Max Health: " << player.maxHealth << std::endl;
            }
            
            spawnDamageNumber(grabbedEnemy.position, 60.0f * player.damageMultiplier, glm::vec3(1, 0.8f, 0.2f));
            spawnRippleEffect(grabbedEnemy.position, 4.0f, glm::vec3(1, 0.5f, 0));
            cameraShake = 0.7f;
            
            std::cout << "GRAB THROW! Damage: " << (60.0f * player.damageMultiplier) << std::endl;
        }
    }
    
    // Parry mechanic - B key/RB button (hold to parry)
    bool parryPressed = keys[GLFW_KEY_B] || currentController.rightBumper; // RB button for parry
    
    if (parryPressed && !player.isParrying) {
        player.isParrying = true;
        player.parryTimer = player.parryWindow;
        player.combatState = CombatState::PARRY;
        player.combatTimer = player.parryWindow;
        player.animController->setState("parry");
        std::cout << "PARRYING!" << std::endl;
    } else if (!parryPressed && player.isParrying) {
        player.isParrying = false;
        player.parryTimer = 0.0f;
    }
    
    // Parry counter attack window
    if (player.justParried && player.parryCounterWindow > 0) {
        player.parryCounterWindow -= deltaTime;
        if (lightAttackPressed || kickPressed) {
            // Perfect counter attack with damage bonus
            player.combatState = CombatState::COUNTER_ATTACK;
            player.combatTimer = 0.8f;
            player.justParried = false;
            player.parryCounterWindow = 0.0f;
            
            // Enhanced damage and stun
            if (targetedEnemyIndex >= 0 && !enemies[targetedEnemyIndex].isDead) {
                float counterDamage = 80.0f * player.damageMultiplier;
                enemies[targetedEnemyIndex].health -= counterDamage;
                enemies[targetedEnemyIndex].isStunned = true;
                enemies[targetedEnemyIndex].stunTimer = 2.0f;
                enemies[targetedEnemyIndex].velocity.y = 20.0f;
                
                // Perfect parry bonus
                player.perfectParries++;
                if (player.perfectParries >= 5) {
                    player.damageMultiplier += 0.3f;
                    player.maxHealthBonus += 25.0f;
                    player.maxHealth = 100.0f + player.maxHealthBonus;
                    player.health = std::min(player.health + 20.0f, player.maxHealth);
                    player.perfectParries = 0;
                    std::cout << "PERFECT PARRY MASTER! Damage: " << player.damageMultiplier << "x, Max Health: " << player.maxHealth << std::endl;
                }
                
                spawnDamageNumber(enemies[targetedEnemyIndex].position, counterDamage, glm::vec3(1, 1, 0.2f));
                spawnRippleEffect(player.position, 6.0f, glm::vec3(1, 1, 0));
                cameraShake = 1.0f;
                
                std::cout << "PERFECT COUNTER! Damage: " << counterDamage << std::endl;
            }
        }
        
        if (player.parryCounterWindow <= 0) {
            player.justParried = false;
        }
    }
    
    // === END NEW ENHANCED MECHANICS ===
    
    // Enhanced Camera lock-on toggle - T key/Right Stick Click (improved controller logic)
    bool lockOnPressed = keysPressed[GLFW_KEY_T] || isControllerButtonJustPressed(9); // RS/R3 button
    
    if (lockOnPressed) {
        if (gameCamera) {
            if (gameCamera->isLockOnActive()) {
                // Clear lock-on
                gameCamera->clearLockOnTarget();
                targetedEnemyIndex = -1;
                std::cout << "Lock-on DISABLED" << std::endl;
            } else {
                // Find best target for lock-on (not just current target)
                int bestTargetIndex = findBestLockOnTarget();
                if (bestTargetIndex >= 0) {
                    // Enable lock-on to best target
                    gameCamera->setLockOnTarget(enemies[bestTargetIndex].position);
                    targetedEnemyIndex = bestTargetIndex; // Update targeting to match lock-on
                    
                    // Display lock-on info with reward status
                    std::string rewardInfo = "";
                    if (enemies[bestTargetIndex].rewardState == Enemy::CARRYING_REWARD) {
                        rewardInfo = " [REWARD: " + enemies[bestTargetIndex].carriedReward.name + "]";
                    }
                    std::cout << "Lock-on ENABLED - Target: Enemy " << bestTargetIndex << rewardInfo << std::endl;
                } else {
                    std::cout << "No valid target for lock-on" << std::endl;
                }
            }
        }
    }
    
    // Lock-on cycling - Q/E for keyboard (when locked on), L1/R1 for controller
    bool cycleForwardPressed = (gameCamera && gameCamera->isLockOnActive() && keysPressed[GLFW_KEY_E]) || isControllerButtonJustPressed(5); // RB
    bool cycleBackwardPressed = (gameCamera && gameCamera->isLockOnActive() && keysPressed[GLFW_KEY_Q]) || isControllerButtonJustPressed(4); // LB
    
    if (gameCamera && gameCamera->isLockOnActive() && targetedEnemyIndex >= 0) {
        if (cycleForwardPressed || cycleBackwardPressed) {
            int newTarget = getNextLockOnTarget(targetedEnemyIndex, cycleForwardPressed);
            if (newTarget >= 0 && newTarget != targetedEnemyIndex) {
                targetedEnemyIndex = newTarget;
                gameCamera->setLockOnTarget(enemies[newTarget].position);
                
                // Display new target info
                std::string rewardInfo = "";
                if (enemies[newTarget].rewardState == Enemy::CARRYING_REWARD) {
                    rewardInfo = " [REWARD: " + enemies[newTarget].carriedReward.name + "]";
                }
                std::cout << "Lock-on SWITCHED - Target: Enemy " << newTarget << rewardInfo << std::endl;
            }
        }
    }
    
    // Weapon switching with animations - Number keys/D-Pad
    if ((keysPressed[GLFW_KEY_1] || isControllerButtonJustPressed(10)) && player.inventory.size() > 0) { // D-Pad Left
        startWeaponSwitch(player.inventory[0]);
    } else if ((keysPressed[GLFW_KEY_2] || isControllerButtonJustPressed(8)) && player.inventory.size() > 1) { // D-Pad Up
        startWeaponSwitch(player.inventory[1]);
    } else if ((keysPressed[GLFW_KEY_3] || isControllerButtonJustPressed(11)) && player.inventory.size() > 2) { // D-Pad Right
        startWeaponSwitch(player.inventory[2]);
    } else if ((keysPressed[GLFW_KEY_4] || isControllerButtonJustPressed(9)) && player.inventory.size() > 3) { // D-Pad Down
        startWeaponSwitch(player.inventory[3]);
    }
    
    // Combat moves with enhanced weapon-specific combos - Q key/Right Trigger
    if (lightAttackPressed) {
        checkRhythmTiming();
        
        // Combo progression
        if (player.comboState == COMBO_DASH) {
            player.comboState = COMBO_LIGHT_1;
            player.comboCount = 1;
        } else if (player.comboState == COMBO_LIGHT_1) {
            player.comboState = COMBO_LIGHT_2;
            player.comboCount = 2;
        } else if (player.comboState == COMBO_LIGHT_2) {
            player.comboState = COMBO_LIGHT_3;
            player.comboCount = 3;
        } else {
            player.comboState = COMBO_LIGHT_1;
            player.comboCount = 1;
        }
        
        // Start weapon-specific combo animation
        startWeaponCombo(player.currentWeapon.type, player.comboCount);
        
        player.combatState = player.currentWeapon.currentAnimation.comboState;
        player.combatTimer = 0.5f;
        player.comboWindow = 0.8f;
        handleCombatHit();
        
        // Special weapon messages
        if (player.currentWeapon.type == WeaponType::SWORD && player.comboCount == 3) {
            std::cout << "SWORD THRUST COMBO!" << std::endl;
        }
    }
    
    // Kick/Launcher - E key/Left Trigger
    if (kickPressed) {
        checkRhythmTiming();
        
        if (player.comboState == COMBO_LIGHT_3) {
            player.comboState = COMBO_LAUNCHER;
            player.combatState = CombatState::KICK;
            player.combatTimer = 0.6f;
            player.comboCount++;
            
            // Launch targeted enemy
            if (targetedEnemyIndex >= 0 && !enemies[targetedEnemyIndex].isDead) {
                enemies[targetedEnemyIndex].velocity.y = 15.0f;
                enemies[targetedEnemyIndex].isLaunched = true;
                enemies[targetedEnemyIndex].launchTimer = 1.0f;
                cameraShake = 0.3f;
                std::cout << "LAUNCHER!" << std::endl;
            }
        } else {
            player.combatState = CombatState::KICK;
            player.combatTimer = 0.6f;
        }
        
        player.comboWindow = 0.8f;
        handleCombatHit();
    }
    
    // Combo system with rhythm synchronization
    if (keysPressed[GLFW_KEY_Q] && player.combatState != CombatState::NONE && player.combatTimer > 0.2f) {
        checkRhythmTiming();
        if (player.comboCount == 1) {
            player.combatState = CombatState::COMBO_1;
            player.combatTimer = 0.7f / player.currentWeapon.speed;
            player.comboCount = 2;
            handleCombatHit();
        } else if (player.comboCount == 2) {
            player.combatState = CombatState::COMBO_2;
            player.combatTimer = 0.8f / player.currentWeapon.speed;
            player.comboCount = 3;
            handleCombatHit();
        }
    }
    
    if (keysPressed[GLFW_KEY_E] && player.combatState != CombatState::NONE && player.combatTimer > 0.2f) {
        checkRhythmTiming();
        if (player.comboCount >= 2) {
            player.combatState = CombatState::COMBO_3;
            player.combatTimer = 0.9f / player.currentWeapon.speed;
            player.comboCount = 4;
            handleCombatHit();
        }
    }
    
    // Update animation based on movement (only if not in combat)
    if (player.combatState == CombatState::NONE) {
        float velocityMagnitude = glm::length(glm::vec2(player.velocity.x, player.velocity.z));
        if (!player.onGround && !player.isWallRunning && !player.isFlipping) {
            player.animController->setState("jump");
        } else if (player.isWallRunning) {
            player.animController->setState("wallrun");
        } else if (velocityMagnitude < 0.1f) {
            player.animController->setState("idle");
        } else if (velocityMagnitude < player.speed * 1.5f) {
            player.animController->setState("walk");
        } else if (velocityMagnitude < player.speed * 3.0f) {
            player.animController->setState("run");
        } else {
            player.animController->setState("sprint");
        }
    }
    
    // Clear single press flags
    for (int i = 0; i < 1024; i++) {
        keysPressed[i] = false;
    }
}

void updateCombat() {
    if (player.combatState != CombatState::NONE) {
        player.combatTimer -= deltaTime;
        
        if (player.combatTimer <= 0.0f) {
            player.combatState = CombatState::NONE;
            player.comboCount = 0;
        }
        
        // Update animation controller with combat state
        switch (player.combatState) {
            case CombatState::PUNCH:
                player.animController->setState("punch");
                break;
            case CombatState::KICK:
                player.animController->setState("kick");
                break;
            case CombatState::COMBO_1:
                player.animController->setState("combo1");
                break;
            case CombatState::COMBO_2:
                player.animController->setState("combo2");
                break;
            case CombatState::COMBO_3:
                player.animController->setState("combo3");
                break;
            default:
                break;
        }
    }
}

// Enhanced Player Animation System
void updatePlayerAnimations(float deltaTime) {
    Player::AnimationState newAnimation = player.currentAnimation;
    
    // Update animation timer
    player.animationTimer += deltaTime;
    
    // Check for movement input
    player.hasMovementInput = (glm::length(glm::vec2(player.velocity.x, player.velocity.z)) > 0.5f);
    
    // Update idle timer
    if (player.hasMovementInput || player.combatState != CombatState::NONE || 
        player.isDashing || player.isDiving || player.isWallRunning) {
        player.idleTimer = 0.0f;
        player.hasRecentActivity = true;
    } else {
        player.idleTimer += deltaTime;
        if (player.idleTimer > 1.0f) {
            player.hasRecentActivity = false;
        }
    }
    
    // Determine animation state based on player status
    if (player.isDead) {
        newAnimation = Player::ANIM_STUNNED;
    } else if (player.combatState != CombatState::NONE) {
        if (player.isParrying) {
            newAnimation = Player::ANIM_PARRYING;
        } else if (player.isGrabbing) {
            newAnimation = Player::ANIM_GRABBING;
        } else {
            newAnimation = Player::ANIM_ATTACKING;
        }
    } else if (player.isDiving || player.isWeaponDiving) {
        newAnimation = Player::ANIM_DIVING;
    } else if (player.isWallRunning) {
        newAnimation = Player::ANIM_WALL_RUNNING;
    } else if (player.isSliding) {
        newAnimation = Player::ANIM_SLIDING;
    } else if (!player.onGround) {
        if (player.velocity.y > 5.0f) {
            newAnimation = Player::ANIM_JUMPING;
        } else if (player.velocity.y < -5.0f) {
            newAnimation = Player::ANIM_FALLING;
        } else {
            newAnimation = Player::ANIM_AIRBORNE;
        }
    } else if (player.hasMovementInput) {
        float speed = glm::length(glm::vec2(player.velocity.x, player.velocity.z));
        if (speed > player.speed * 2.5f) {
            newAnimation = Player::ANIM_SPRINTING;
        } else if (speed > player.speed * 0.8f) {
            newAnimation = Player::ANIM_RUNNING;
        } else {
            newAnimation = Player::ANIM_WALKING;
        }
    } else {
        // Idle states
        if (player.idleTimer > 5.0f) {
            newAnimation = Player::ANIM_IDLE_BORED;
        } else if (targetedEnemyIndex >= 0) {
            newAnimation = Player::ANIM_COMBAT_IDLE;
        } else {
            newAnimation = Player::ANIM_IDLE;
        }
    }
    
    // Handle animation transitions
    if (newAnimation != player.currentAnimation) {
        player.previousAnimation = player.currentAnimation;
        player.currentAnimation = newAnimation;
        player.animationTimer = 0.0f;
        player.animationBlendWeight = 0.0f;
        
        // Set animation state in controller
        switch (newAnimation) {
            case Player::ANIM_IDLE:
                player.animController->setState("idle");
                player.characterAnimController->setState(CharacterAnimationController::IDLE);
                break;
            case Player::ANIM_IDLE_BORED:
                player.animController->setState("idle_bored");
                player.characterAnimController->setState(CharacterAnimationController::IDLE_BORED);
                std::cout << "Player bored - showing idle animation" << std::endl;
                break;
            case Player::ANIM_WALKING:
                player.animController->setState("walk");
                player.characterAnimController->setState(CharacterAnimationController::WALKING);
                break;
            case Player::ANIM_RUNNING:
                player.animController->setState("run");
                player.characterAnimController->setState(CharacterAnimationController::RUNNING);
                break;
            case Player::ANIM_SPRINTING:
                player.animController->setState("sprint");
                player.characterAnimController->setState(CharacterAnimationController::SPRINTING);
                break;
            case Player::ANIM_JUMPING:
                player.animController->setState("jump");
                player.characterAnimController->setState(CharacterAnimationController::JUMPING);
                break;
            case Player::ANIM_AIRBORNE:
                player.animController->setState("airborne");
                player.characterAnimController->setState(CharacterAnimationController::AIRBORNE);
                break;
            case Player::ANIM_FALLING:
                player.animController->setState("fall");
                player.characterAnimController->setState(CharacterAnimationController::FALLING);
                break;
            case Player::ANIM_DIVING:
                player.animController->setState("dive");
                player.characterAnimController->setState(CharacterAnimationController::JUMPING);
                break;
            case Player::ANIM_WALL_RUNNING:
                player.animController->setState("wallrun");
                player.characterAnimController->setState(CharacterAnimationController::WALL_RUNNING);
                break;
            case Player::ANIM_SLIDING:
                player.animController->setState("slide");
                player.characterAnimController->setState(CharacterAnimationController::SLIDING);
                break;
            case Player::ANIM_COMBAT_IDLE:
                player.animController->setState("combat_idle");
                player.characterAnimController->setState(CharacterAnimationController::COMBAT_IDLE);
                break;
            case Player::ANIM_ATTACKING:
                player.animController->setState("attack");
                player.characterAnimController->setState(CharacterAnimationController::ATTACKING);
                break;
            case Player::ANIM_PARRYING:
                player.animController->setState("parry");
                player.characterAnimController->setState(CharacterAnimationController::PARRYING);
                break;
            case Player::ANIM_GRABBING:
                player.animController->setState("grab");
                player.characterAnimController->setState(CharacterAnimationController::ATTACKING);
                break;
            default:
                player.animController->setState("idle");
                player.characterAnimController->setState(CharacterAnimationController::IDLE);
                break;
        }
    }
    
    // Update blend weight for smooth transitions
    if (player.animationBlendWeight < 1.0f) {
        player.animationBlendWeight = std::min(1.0f, player.animationBlendWeight + deltaTime * 5.0f);
    }
}

// Enhanced Player Movement System with Tighter Controls
void updatePlayerMovement(float deltaTime) {
    // Apply ground friction for tighter controls
    if (player.onGround && !player.isDashing && !player.isSliding) {
        player.velocity.x *= player.groundFriction;
        player.velocity.z *= player.groundFriction;
    }
    
    // Apply air control
    if (!player.onGround && player.hasMovementInput) {
        glm::vec3 airInfluence = player.targetVelocity * player.airControl * deltaTime;
        player.velocity.x += airInfluence.x;
        player.velocity.z += airInfluence.z;
    }
    
    // Smooth movement precision - reduce small movements
    if (glm::length(glm::vec2(player.velocity.x, player.velocity.z)) < 0.1f) {
        player.velocity.x = 0.0f;
        player.velocity.z = 0.0f;
    }
}

void updatePlayer() {
    // Update enhanced systems
    updatePlayerAnimations(deltaTime);
    updatePlayerMovement(deltaTime);
    
    // Update 3D character mesh animations
    if (player.characterAnimController) {
        player.characterAnimController->update(deltaTime);
    }
    
    // === UPDATE ENHANCED MECHANICS ===
    
    // Update slide timer
    if (player.isSliding) {
        player.slideTimer -= deltaTime;
        if (player.slideTimer <= 0.0f) {
            player.isSliding = false;
        }
    }
    
    // Update grab timer
    if (player.isGrabbing) {
        player.grabTimer -= deltaTime;
        if (player.grabTimer <= 0 || player.grabbedEnemyIndex < 0 || 
            player.grabbedEnemyIndex >= enemies.size() || enemies[player.grabbedEnemyIndex].isDead) {
            
            // Release grabbed enemy
            if (player.grabbedEnemyIndex >= 0 && player.grabbedEnemyIndex < enemies.size()) {
                enemies[player.grabbedEnemyIndex].isGrabbed = false;
            }
            
            player.isGrabbing = false;
            player.grabbedEnemyIndex = -1;
            player.grabTimer = 0.0f;
        } else {
            // Keep grabbed enemy close
            Enemy& grabbedEnemy = enemies[player.grabbedEnemyIndex];
            glm::vec3 targetPos = player.position + glm::vec3(sin(player.rotation), 0, -cos(player.rotation)) * 1.5f;
            grabbedEnemy.position = targetPos;
            grabbedEnemy.velocity = glm::vec3(0);
        }
    }
    
    // Update parry timer
    if (player.isParrying) {
        player.parryTimer -= deltaTime;
        if (player.parryTimer <= 0) {
            player.isParrying = false;
            player.parryTimer = 0.0f;
        }
    }
    
    // Update air dash timer
    if (player.isAirDashing) {
        player.airDashTimer -= deltaTime;
        if (player.airDashTimer <= 0) {
            player.isAirDashing = false;
        }
    }
    
    // Update floating state
    if (player.isFloating) {
        player.floatTimer -= deltaTime;
        if (player.floatTimer <= 0) {
            player.isFloating = false;
            player.canDive = true; // Can dive again after floating
        } else {
            // Reduced gravity while floating
            player.velocity.y -= 8.0f * deltaTime;
        }
    }
    
    // Check boundary collisions
    checkBoundaryCollisions();
    
    // Apply gravity
    if (!player.onGround && !player.isWallRunning) {
        player.velocity.y -= 50.0f * deltaTime;
    }
    
    // Update flip rotation for double jump
    if (player.isFlipping) {
        player.flipRotation += player.flipSpeed * deltaTime;
        if (player.flipRotation >= 2.0f * M_PI) {
            player.isFlipping = false;
            player.flipRotation = 0.0f;
        }
    }
    
    // Update position
    glm::vec3 newPos = player.position + player.velocity * deltaTime;
    
    // Arena bounds check - can jump over but fall recovery
    glm::vec3 toCenter = newPos - arena.center;
    toCenter.y = 0;
    float distFromCenter = glm::length(toCenter);
    
    // Fall recovery check
    if (newPos.y < arena.fallResetY) {
        // Player fell out of level - respawn
        player.health -= 25.0f;
        player.position = player.spawnPosition;
        player.velocity = glm::vec3(0);
        std::cout << "Fell out of bounds! Health: " << player.health << std::endl;
        
        if (player.health <= 0) {
            gameState = GAME_DEATH;
            player.isDead = true;
            player.deathTimer = 3.0f;
        }
        return;
    }
    
    // Check for dive landing
    if ((player.isDiving || player.isWeaponDiving) && player.onGround) {
        bool wasWeaponDiving = player.isWeaponDiving;
        
        if (wasWeaponDiving) {
            // Massive weapon dive impact with enhanced ripple
            cameraShake = 1.5f;
            spawnRippleEffect(player.position, 12.0f, glm::vec3(1, 0.2f, 0));
            
            for (auto& enemy : enemies) {
                float dist = glm::length(enemy.position - player.position);
                if (dist < 8.0f && !enemy.isDead) {
                    enemy.health -= 120.0f;
                    enemy.velocity.y = 25.0f;
                    enemy.isLaunched = true;
                    enemy.launchTimer = 2.0f;
                    spawnDamageNumber(enemy.position, 120.0f, glm::vec3(1, 0.5f, 0));
                    std::cout << "WEAPON DIVE IMPACT! 120 damage!" << std::endl;
                    
                    if (enemy.health <= 0) {
                        enemy.isDead = true;
                        enemy.deathTimer = 2.0f;
                        score += 2500;
                    }
                }
            }
        } else if (player.isDiving) {
            // Enhanced dive bounce with more recoil and floating
            player.velocity.y = player.diveBounceHeight;
            player.onGround = false;
            player.isFloating = true;
            player.floatTimer = player.floatDuration;
            player.canAirDash = true; // Reset air dash on bounce
            
            // Enhanced ripple effect on landing
            spawnRippleEffect(player.position, 6.0f, glm::vec3(0.3f, 0.8f, 1.0f));
            cameraShake = 0.8f;
            std::cout << "ENHANCED DIVE BOUNCE! Now floating..." << std::endl;
            
            // Enhanced damage with ripple effect
            for (auto& enemy : enemies) {
                float dist = glm::length(enemy.position - player.position);
                if (dist < 6.0f && !enemy.isDead) {
                    enemy.health -= 45.0f;
                    enemy.velocity.y = 15.0f;
                    enemy.isLaunched = true;
                    enemy.launchTimer = 1.5f;
                    spawnDamageNumber(enemy.position, 45.0f, glm::vec3(0.3f, 0.8f, 1.0f));
                    std::cout << "Enhanced dive ripple damage!" << std::endl;
                    
                    if (enemy.health <= 0) {
                        enemy.isDead = true;
                        enemy.deathTimer = 2.0f;
                        score += 1500;
                    }
                }
            }
        }
        
        player.isDiving = false;
        player.isWeaponDiving = false;
    }
    
    // Ground collision
    player.onGround = false;
    for (const auto& platform : platforms) {
        if (newPos.x + player.size.x/2 > platform.position.x - platform.size.x/2 &&
            newPos.x - player.size.x/2 < platform.position.x + platform.size.x/2 &&
            newPos.z + player.size.z/2 > platform.position.z - platform.size.z/2 &&
            newPos.z - player.size.z/2 < platform.position.z + platform.size.z/2) {
            
            float platformTop = platform.position.y + platform.size.y/2;
            if (player.position.y >= platformTop && newPos.y - player.size.y/2 <= platformTop) {
                newPos.y = platformTop + player.size.y/2;
                player.velocity.y = 0.0f;
                player.onGround = true;
                player.canDoubleJump = true;
                player.hasDoubleJumped = false;
                player.isFlipping = false;
                player.flipRotation = 0.0f;
                player.isWallRunning = false;
                player.canAirDash = true; // Reset air dash on ground
                if (!player.isFloating) {
                    player.canDive = false;
                }
            }
        }
    }
    
    // Enhanced wall collision for regular walls
    for (const auto& wall : walls) {
        if (checkCollision(newPos, player.size, wall.position, wall.size)) {
            if (wall.isWallRunnable && !player.onGround && glm::length(glm::vec2(player.velocity.x, player.velocity.z)) > player.speed * 2.0f) {
                // Enable wall running
                player.isWallRunning = true;
                player.velocity.y = std::max(player.velocity.y, 0.0f);
                player.wallRunTime += deltaTime;
                
                // Determine wall normal
                glm::vec3 toWall = wall.position - player.position;
                if (abs(toWall.x) > abs(toWall.z)) {
                    player.wallNormal = glm::vec3(-glm::sign(toWall.x), 0.0f, 0.0f);
                } else {
                    player.wallNormal = glm::vec3(0.0f, 0.0f, -glm::sign(toWall.z));
                }
                
                // Slide along wall
                newPos = player.position;
                if (player.wallRunTime > 3.0f) {
                    player.isWallRunning = false;
                    player.wallRunTime = 0.0f;
                }
            } else {
                // Normal collision
                glm::vec3 overlap = glm::abs(newPos - wall.position) - (player.size + wall.size) * 0.5f;
                if (overlap.x > overlap.z) {
                    player.velocity.z = 0.0f;
                    newPos.z = player.position.z;
                } else {
                    player.velocity.x = 0.0f;
                    newPos.x = player.position.x;
                }
            }
        }
    }
    
    // Giant Pillar Wall-Running System
    if (!player.isWallRunning) { // Only check pillars if not already wall-running on regular walls
        for (int i = 0; i < giantPillars.size(); i++) {
            const auto& pillar = giantPillars[i];
            float distToPillar = glm::distance(glm::vec2(newPos.x, newPos.z), glm::vec2(pillar.position.x, pillar.position.z));
            
            // Check if player is close enough to start wall-running (within pillar radius + 2 units)
            float activationRadius = pillar.radius + 2.0f;
            if (!pillar.isCylindrical) {
                activationRadius = std::max(pillar.size.x, pillar.size.z) / 2.0f + 2.0f;
            }
            
            if (distToPillar <= activationRadius && !player.onGround) {
                // Check if player is moving fast enough and performing a combo attack to trigger wall-run
                bool hasComboVelocity = player.comboState != COMBO_NONE || player.combatState != CombatState::NONE;
                bool hasSpeed = glm::length(glm::vec2(player.velocity.x, player.velocity.z)) > player.speed * 1.5f;
                
                if (hasComboVelocity && hasSpeed) {
                    // Activate wall-running on this pillar!
                    player.isWallRunning = true;
                    player.wallRunPillarIndex = i;
                    player.wallRunTime = 0.0f;
                    player.wallRunHeight = newPos.y - pillar.position.y;
                    player.canWallJump = true;
                    
                    // Get the closest surface and normal
                    glm::vec3 closestSurface = pillar.getClosestSurface(newPos, player.wallNormal);
                    
                    // Calculate wall-run direction (tangent to the pillar)
                    glm::vec3 toPillar = glm::normalize(glm::vec3(pillar.position.x - newPos.x, 0, pillar.position.z - newPos.z));
                    player.wallRunDirection = glm::cross(player.wallNormal, glm::vec3(0, 1, 0));
                    
                    // Choose direction based on player's current velocity
                    glm::vec3 velocityDir = glm::normalize(glm::vec3(player.velocity.x, 0, player.velocity.z));
                    if (glm::dot(player.wallRunDirection, velocityDir) < 0) {
                        player.wallRunDirection = -player.wallRunDirection;
                    }
                    
                    std::cout << "WALL-RUNNING ACTIVATED on pillar " << i << "!" << std::endl;
                    break;
                }
            }
        }
    }
    
    // Update wall-running physics
    if (player.isWallRunning) {
        player.wallRunTime += deltaTime;
        
        if (player.wallRunPillarIndex >= 0 && player.wallRunPillarIndex < giantPillars.size()) {
            const auto& pillar = giantPillars[player.wallRunPillarIndex];
            
            // Apply reduced gravity while wall-running
            player.velocity.y -= player.wallRunGravity * deltaTime;
            
            // Move along the wall surface
            glm::vec3 wallRunMovement = player.wallRunDirection * player.wallRunSpeed * deltaTime;
            newPos.x += wallRunMovement.x;
            newPos.z += wallRunMovement.z;
            
            // Keep player at correct distance from pillar
            glm::vec2 toPillar2D = glm::vec2(pillar.position.x - newPos.x, pillar.position.z - newPos.z);
            float distFromCenter = glm::length(toPillar2D);
            glm::vec3 toPillar = glm::vec3(toPillar2D.x, 0.0f, toPillar2D.y);
            float targetDist = pillar.radius + 1.0f;
            if (!pillar.isCylindrical) {
                targetDist = std::max(pillar.size.x, pillar.size.z) / 2.0f + 1.0f;
            }
            
            if (distFromCenter > 0.1f) {
                glm::vec3 normalizedToPillar = glm::normalize(toPillar);
                newPos.x = pillar.position.x - normalizedToPillar.x * targetDist;
                newPos.z = pillar.position.z - normalizedToPillar.z * targetDist;
            }
            
            // End wall-running conditions
            if (player.wallRunTime > player.maxWallRunTime || 
                newPos.y < pillar.position.y - 2.0f || 
                newPos.y > pillar.position.y + pillar.size.y + 2.0f) {
                player.isWallRunning = false;
                player.wallRunPillarIndex = -1;
                player.canWallJump = false;
                std::cout << "Wall-run ended naturally" << std::endl;
            }
        } else {
            // Invalid pillar index, end wall-running
            player.isWallRunning = false;
            player.wallRunPillarIndex = -1;
        }
    }
    
    if (!player.isWallRunning) {
        player.wallRunTime = 0.0f;
        player.wallRunPillarIndex = -1;
    }
    
    // Check collectibles
    for (auto& collectible : collectibles) {
        if (!collectible.collected && checkCollision(newPos, player.size, collectible.position, collectible.size)) {
            collectible.collected = true;
            score += 100;
            if (onBeat) score += 50; // Bonus for collecting on beat
            std::cout << "Score: " << score << " (Collected on beat!)" << std::endl;
        }
    }
    
    player.position = newPos;
    
    // Death check
    if (player.health <= 0 && !player.isDead) {
        gameState = GAME_DEATH;
        player.isDead = true;
        player.deathTimer = 3.0f;
        std::cout << "PLAYER DIED!" << std::endl;
        
        // Debug logging for death
        if (debugInfo.debugEnabled) {
            debugInfo.logEvent("PLAYER DEATH EVENT");
            std::cout << "[DEBUG] Player Death Details:" << std::endl;
            std::cout << "  Final Position: (" << player.position.x << ", " << player.position.y << ", " << player.position.z << ")" << std::endl;
            std::cout << "  Final Velocity: (" << player.velocity.x << ", " << player.velocity.y << ", " << player.velocity.z << ")" << std::endl;
            std::cout << "  Was Jumping: " << (!player.onGround ? "Yes" : "No") << std::endl;
            std::cout << "  Was Double Jumping: " << (player.hasDoubleJumped ? "Yes" : "No") << std::endl;
            std::cout << "  Was Diving: " << (player.isDiving ? "Yes" : "No") << std::endl;
            std::cout << "  Was Wall Running: " << (player.isWallRunning ? "Yes" : "No") << std::endl;
            if (gameCamera) {
                std::cout << "  Camera Position: (" << gameCamera->position.x << ", " << gameCamera->position.y << ", " << gameCamera->position.z << ")" << std::endl;
                std::cout << "  Camera Target: (" << gameCamera->target.x << ", " << gameCamera->target.y << ", " << gameCamera->target.z << ")" << std::endl;
                std::cout << "  Camera Distance: " << glm::length(gameCamera->position - player.position) << std::endl;
            }
        }
    }
    
    player.animController->update(deltaTime);
    updateCombat();
}

void checkBoundaryCollisions() {
    bool hasCombo = (player.comboCount > 0 || player.combatState != CombatState::NONE);
    
    for (const auto& boundary : arenaBoundaries) {
        if (boundary.checkCollision(player.position, 1.0f)) {
            // Apply repel force
            glm::vec3 repelVector = boundary.getRepelVector(player.position, hasCombo);
            player.velocity += repelVector;
            
            // Visual feedback
            spawnRippleEffect(player.position, 5.0f, hasCombo ? glm::vec3(1.0f, 0.5f, 0.0f) : glm::vec3(0.5f, 0.5f, 1.0f));
            
            if (hasCombo) {
                // Combo allows stronger repel and potential dash to enemy
                cameraShake = 0.3f;
                
                // Find nearest enemy to dash towards
                if (targetedEnemyIndex >= 0 && targetedEnemyIndex < enemies.size()) {
                    glm::vec3 toEnemy = enemies[targetedEnemyIndex].position - player.position;
                    toEnemy.y = 0;
                    if (glm::length(toEnemy) > 0.1f) {
                        player.dashDirection = glm::normalize(toEnemy);
                        player.isDashing = true;
                        player.dashTimer = 0.3f;
                        std::cout << "BOUNDARY COMBO DASH!" << std::endl;
                    }
                }
                
                std::cout << "BOUNDARY REPEL - Combo boost!" << std::endl;
            } else {
                std::cout << "Hit boundary - use combos for stronger repel!" << std::endl;
            }
        }
    }
}

void updateGameState() {
    switch (gameState) {
        case GAME_DEATH:
            deathScreenTimer += deltaTime;
            player.deathTimer -= deltaTime;
            
            if (player.deathTimer <= 0) {
                playerLives--;
                if (playerLives <= 0) {
                    gameState = GAME_GAME_OVER;
                    std::cout << "GAME OVER! Press R to restart" << std::endl;
                } else {
                    // Respawn
                    player.health = player.maxHealth;
                    player.position = player.spawnPosition;
                    player.velocity = glm::vec3(0);
                    player.isDead = false;
                    
                    // Reset camera to follow player at spawn position
                    if (gameCamera) {
                        gameCamera->resetToPlayer(player.position);
                        gameCamera->clearLockOnTarget();
                    }
                    targetedEnemyIndex = -1;
                    
                    gameState = GAME_PLAYING;
                    std::cout << "Respawned! Lives remaining: " << playerLives << std::endl;
                }
            }
            break;
            
        case GAME_RESTART:
            resetGame();
            gameState = GAME_PLAYING;
            break;
    }
    
    // Update camera shake
    if (cameraShake > 0) {
        cameraShake -= deltaTime * 3.0f;
        if (cameraShake < 0) cameraShake = 0;
    }
    
    // Update damage numbers
    for (auto& dmg : damageNumbers) {
        dmg.lifetime -= deltaTime;
        dmg.position.y += deltaTime * 2.0f;
    }
    
    damageNumbers.erase(
        std::remove_if(damageNumbers.begin(), damageNumbers.end(),
            [](const DamageNumber& d) { return d.lifetime <= 0; }),
        damageNumbers.end()
    );
    
    // Update ripple effects
    for (auto& ripple : rippleEffects) {
        ripple.lifetime -= deltaTime;
        ripple.radius += (ripple.maxRadius / 1.0f) * deltaTime; // Expand over 1 second
        if (ripple.radius > ripple.maxRadius) {
            ripple.radius = ripple.maxRadius;
        }
    }
    
    rippleEffects.erase(
        std::remove_if(rippleEffects.begin(), rippleEffects.end(),
            [](const RippleEffect& r) { return r.lifetime <= 0; }),
        rippleEffects.end()
    );
    
    // Update explosion particles
    for (auto& particle : explosionParticles) {
        particle.lifetime -= deltaTime;
        particle.position += particle.velocity * deltaTime;
        particle.velocity.y -= 20.0f * deltaTime; // Gravity
        particle.rotation += particle.rotationSpeed * deltaTime;
        particle.size *= 0.98f; // Shrink over time
    }
    
    explosionParticles.erase(
        std::remove_if(explosionParticles.begin(), explosionParticles.end(),
            [](const ParticleTriangle& p) { return p.lifetime <= 0; }),
        explosionParticles.end()
    );
}

void updateEnemies() {
    for (auto& enemy : enemies) {
        if (enemy.isDead) {
            enemy.deathTimer -= deltaTime;
            continue;
        }
        
        // === UPDATE ENHANCED ENEMY MECHANICS ===
        
        // Update stun timer
        if (enemy.isStunned) {
            enemy.stunTimer -= deltaTime;
            if (enemy.stunTimer <= 0) {
                enemy.isStunned = false;
                enemy.stunTimer = 0.0f;
            }
        }
        
        // Update visual effects
        if (enemy.damageFlash > 0) {
            enemy.damageFlash -= deltaTime * 4.0f;
        }
        
        if (enemy.isLaunched) {
            enemy.launchTimer -= deltaTime;
            if (enemy.launchTimer <= 0) {
                enemy.isLaunched = false;
            }
        }
        
        // Update flying state
        if (enemy.isFlying) {
            enemy.flyTimer -= deltaTime;
            enemy.position += enemy.flyVelocity * deltaTime;
            enemy.flyVelocity.y -= 15.0f * deltaTime; // Gravity for flying enemies
            
            // Landing check for flying enemies
            if (enemy.position.y <= 0.5f || enemy.flyTimer <= 0) {
                enemy.position.y = 0.5f;
                enemy.isFlying = false;
                enemy.flyVelocity = glm::vec3(0);
                
                // Small landing impact
                spawnRippleEffect(enemy.position, 2.0f, glm::vec3(0.8f, 0.2f, 0.2f));
                std::cout << "Enemy landed!" << std::endl;
            }
        }
        
        // Update attack cooldown
        if (enemy.attackCooldown > 0) {
            enemy.attackCooldown -= deltaTime;
        }
        
        glm::vec3 toPlayer = player.position - enemy.position;
        float distance = glm::length(toPlayer);
        
        if (distance < enemy.detectionRange && !player.isDead) {
            // Enemy detected player - switch to aggressive mode
            enemy.isAggressive = true;
            enemy.isPatrolling = false;
            
            // Move towards player
            glm::vec3 direction = glm::normalize(toPlayer);
            float speedMultiplier = 1.0f;
            
            // Speed up if player is sprinting
            if (player.sprintToggled) {
                speedMultiplier = 1.5f;
            }
            
            // Rhythm-based speed boost
            if (onBeat) {
                speedMultiplier *= 1.3f;
            }
            
            enemy.velocity = direction * enemy.speed * speedMultiplier;
            enemy.position += enemy.velocity * deltaTime;
            
            // Attack if in range (but not if stunned or grabbed)
            if (distance < enemy.attackRange && enemy.attackCooldown <= 0 && !enemy.isStunned && !enemy.isGrabbed) {
                
                // Check if player is parrying
                if (player.isParrying && player.parryTimer > 0) {
                    // Successful parry!
                    enemy.isStunned = true;
                    enemy.stunTimer = enemy.stunDuration;
                    enemy.attackCooldown = 3.0f; // Longer cooldown after being parried
                    
                    player.justParried = true;
                    player.parryCounterWindow = 0.8f; // Window for counter attack
                    player.isParrying = false;
                    player.parryTimer = 0.0f;
                    
                    // Visual feedback
                    spawnRippleEffect(player.position, 4.0f, glm::vec3(0, 1, 1));
                    cameraShake = 0.4f;
                    
                    std::cout << "SUCCESSFUL PARRY! Enemy stunned!" << std::endl;
                } else {
                    // Normal attack
                    float finalDamage = enemy.damage;
                    
                    // Reduce damage if player is sliding (partial dodge)
                    if (player.isSliding) {
                        finalDamage *= 0.3f;
                        std::cout << "Sliding dodge! Reduced damage!" << std::endl;
                    }
                    
                    player.health -= finalDamage;
                    enemy.attackCooldown = 1.5f;
                    std::cout << "Player hit! Health: " << player.health << std::endl;
                }
            }
        } else {
            // Patrol behavior
            enemy.isAggressive = false;
            
            if (enemy.isPatrolling) {
                if (enemy.patrolWaitTimer > 0) {
                    enemy.patrolWaitTimer -= deltaTime;
                } else {
                    glm::vec3 toTarget = enemy.patrolTarget - enemy.position;
                    float distToTarget = glm::length(toTarget);
                    
                    if (distToTarget < 1.0f) {
                        // Reached target, switch to other end
                        enemy.patrolTarget = (enemy.patrolTarget == enemy.patrolEnd) ? 
                            enemy.patrolStart : enemy.patrolEnd;
                        enemy.patrolWaitTimer = enemy.patrolWaitDuration;
                    } else {
                        // Move toward target
                        glm::vec3 direction = glm::normalize(toTarget);
                        enemy.velocity = direction * enemy.patrolSpeed;
                        enemy.position += enemy.velocity * deltaTime;
                    }
                }
            }
        }
        
        // Simple gravity for enemies
        enemy.position.y = 0.5f; // Keep on ground for now
    }
    
    // Remove dead enemies after timer
    enemies.erase(
        std::remove_if(enemies.begin(), enemies.end(),
            [](const Enemy& e) { return e.isDead && e.deathTimer <= 0; }),
        enemies.end()
    );
}

void updateCamera(float deltaTime, float rightStickX, float rightStickY) {
    if (gameCamera) {
        // Debug info update
        debugInfo.update(deltaTime);
        
        // Update controller input
        if (controllerInput) {
            controllerInput->update(window);
        }
        
        // Process camera rotation input
        gameCamera->processController(rightStickX, rightStickY, deltaTime);
        
        // Calculate player speed for dynamic distance
        float playerSpeed = glm::length(player.velocity);
        
        // Determine if player is in combat
        bool inCombat = (player.combatState != CombatState::NONE) || 
                       (targetedEnemyIndex >= 0 && targetedEnemyIndex < enemies.size());
        
        // Set combat mode for distance adjustment
        gameCamera->setCombatMode(inCombat);
        
        // Update dynamic FOV based on speed and combat
        gameCamera->updateDynamicFOV(playerSpeed, inCombat);
        
        // Handle camera shake from game events
        if (cameraShake > 0) {
            gameCamera->startShake(cameraShake, 0.3f);
            cameraShake = 0.0f; // Reset shake trigger
        }
        
        // Get target position for lock-on if active
        glm::vec3 targetPosition(0.0f);
        bool hasValidTarget = (targetedEnemyIndex >= 0 && targetedEnemyIndex < enemies.size() && !enemies[targetedEnemyIndex].isDead);
        
        if (gameCamera->isLockOnActive()) {
            if (hasValidTarget) {
                targetPosition = enemies[targetedEnemyIndex].position;
                // Update the lock-on target position dynamically
                gameCamera->setLockOnTarget(targetPosition);
            } else {
                // Target died or is invalid, clear lock-on
                gameCamera->clearLockOnTarget();
                std::cout << "Lock-on cleared - target lost" << std::endl;
            }
        }
        
        // Update camera with player position, velocity, and target
        gameCamera->update(deltaTime, player.position, player.velocity, player.isWallRunning, targetPosition);
        
        // Update legacy camera variables for compatibility with existing rendering code
        cameraPos = gameCamera->position;
        cameraTarget = gameCamera->target;
        
        // Debug camera health check
        debugInfo.checkCameraHealth(cameraPos, cameraTarget, player.position);
        
        // Verbose debug output
        if (debugInfo.verbose && debugInfo.debugEnabled) {
            debugInfo.printStatus();
        }
        cameraUp = gameCamera->up;
    }
}

void updateBPM(float newBPM) {
    currentBPM = newBPM;
    beatInterval = 60.0f / currentBPM;
}

void checkRhythmTiming() {
    float timeToBeat = fmod(beatTimer, beatInterval);
    float distanceToBeat = std::min(timeToBeat, beatInterval - timeToBeat);
    
    if (distanceToBeat < player.rhythm.beatWindow * 0.5f) {
        // Perfect timing
        player.rhythm.accuracy = 1.0f;
        player.rhythm.perfectHits++;
        rhythmAccuracy = 1.0f;
    } else if (distanceToBeat < player.rhythm.beatWindow) {
        // Good timing
        player.rhythm.accuracy = 0.7f;
        player.rhythm.goodHits++;
        rhythmAccuracy = 0.7f;
    } else {
        // Poor timing
        player.rhythm.accuracy = 0.3f;
        player.rhythm.missedHits++;
        rhythmAccuracy = 0.3f;
    }
    
    player.rhythm.lastHitTime = glfwGetTime();
}

void handleCombatHit() {
    if (player.combatState != CombatState::NONE && player.combatTimer > 0.3f) {
        float attackRange = player.currentWeapon.range + player.attackRange;
        float damage = player.currentWeapon.damage;
        
        // Apply rhythm multiplier
        damage *= player.rhythm.calculateMultiplier();
        
        // Combo damage scaling
        if (player.comboCount > 0) {
            damage *= (1.0f + 0.3f * player.comboCount);
        }
        
        // Prioritize targeted enemy
        if (targetedEnemyIndex >= 0 && !enemies[targetedEnemyIndex].isDead) {
            Enemy& enemy = enemies[targetedEnemyIndex];
            float dist = glm::length(enemy.position - player.position);
            if (dist <= attackRange) {
                enemy.health -= damage;
                enemy.damageFlash = 1.0f;
                
                // Visual feedback
                glm::vec3 dmgColor = player.rhythm.accuracy > 0.8f ? 
                    glm::vec3(1, 1, 0) : glm::vec3(1, 1, 1);
                spawnDamageNumber(enemy.position, damage, dmgColor);
                
                std::cout << "Hit! " << damage << " damage! Combo x" << player.comboCount << std::endl;
                
                // Check for reward stealing on critical hits
                if (player.rhythm.accuracy > 0.8f && enemy.rewardState != Enemy::NO_REWARD) {
                    stealEnemyReward(targetedEnemyIndex);
                }
                
                if (enemy.health <= 0) {
                    enemy.isDead = true;
                    enemy.deathTimer = 2.0f;
                    score += 1000 * player.comboCount;
                    
                    // Bonus score for defeating reward-carrying enemies
                    if (enemy.rewardState != Enemy::NO_REWARD) {
                        score += 5000;
                        std::cout << "Defeated reward-carrying enemy! Bonus points!" << std::endl;
                    }
                    
                    targetedEnemyIndex = -1;
                }
            }
        }
    }
}

void updateRhythm() {
    beatTimer += deltaTime;
    nextBeatTime = beatInterval - fmod(beatTimer, beatInterval);
    
    if (beatTimer >= beatInterval) {
        beatTimer -= beatInterval;
        onBeat = true;
        beatPulse = 1.0f;
    } else {
        onBeat = false;
        beatPulse *= 0.95f; // Decay
    }
    
    // Update rhythm accuracy display
    rhythmAccuracy *= 0.98f; // Slowly decay for visual effect
}

void renderCube(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color, const glm::mat4& rotationMat) {
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = model * rotationMat;
    model = glm::scale(model, size);
    
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, glm::value_ptr(color));
    
    float pulse = onBeat ? 1.0f : 0.0f;
    glUniform1f(glGetUniformLocation(shaderProgram, "pulse"), pulse);
    
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

void renderWeapon() {
    if (player.currentWeapon.type == WeaponType::NONE) return;
    
    // Calculate weapon position relative to player
    glm::mat4 playerRotation = glm::rotate(glm::mat4(1.0f), player.rotation, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 weaponOffset(0.8f, 0.0f, 0.2f);
    
    // Apply weapon switching animation
    if (player.isSwitchingWeapon) {
        weaponOffset += player.weaponSwitchOffset;
        playerRotation = glm::rotate(playerRotation, glm::radians(player.weaponSwitchRotation), glm::vec3(0, 1, 0));
        
        // Scale weapon based on holster progress during switching
        float switchScale = player.weaponHolsterProgress;
        weaponOffset *= switchScale;
    }
    
    // Apply weapon animation if active
    WeaponAnimation& anim = player.currentWeapon.currentAnimation;
    glm::mat4 weaponAnimRotation = glm::mat4(1.0f);
    
    if (anim.isActive && anim.totalDuration > 0.0f) {
        float progress = anim.animationTime / anim.totalDuration;
        progress = std::min(progress, 1.0f);
        
        // Apply animation rotation
        float currentAngle = anim.rotationAngle * progress;
        weaponAnimRotation = glm::rotate(weaponAnimRotation, glm::radians(currentAngle), anim.rotationAxis);
        
        // Apply animation position offset
        weaponOffset += anim.positionOffset * progress;
        
        // Special effects for different animations
        if (anim.comboState == CombatState::SWORD_THRUST && progress > 0.7f) {
            weaponOffset.z += 0.5f; // Extend forward during thrust
        }
        if (anim.comboState == CombatState::STAFF_CAST) {
            weaponOffset.y += sin(progress * 6.28f * 2) * 0.3f; // Float during cast
        }
        if (anim.comboState == CombatState::HAMMER_GROUND_POUND && progress > 0.5f) {
            weaponOffset.y -= (progress - 0.5f) * 2.0f; // Slam down
        }
    } else if (player.combatState != CombatState::NONE) {
        // Legacy animation for non-weapon-specific attacks
        float swingAngle = sin(player.combatTimer * 10.0f) * 0.5f;
        weaponOffset.x += sin(swingAngle) * 0.3f;
        weaponOffset.z += cos(swingAngle) * 0.3f;
    }
    
    glm::vec3 weaponPos = player.position + glm::vec3(playerRotation * glm::vec4(weaponOffset, 1.0f));
    
    // Weapon size based on type
    glm::vec3 weaponSize;
    switch(player.currentWeapon.type) {
        case WeaponType::SWORD:
            weaponSize = glm::vec3(0.1f, 1.2f, 0.2f);
            break;
        case WeaponType::STAFF:
            weaponSize = glm::vec3(0.15f, 1.8f, 0.15f);
            break;
        case WeaponType::HAMMER:
            weaponSize = glm::vec3(0.4f, 1.0f, 0.3f);
            break;
        default:
            weaponSize = glm::vec3(0.2f, 0.8f, 0.2f);
    }
    
    // Add rhythm glow to weapon
    glm::vec3 weaponColor = player.currentWeapon.color;
    if (player.rhythm.accuracy > 0.7f) {
        weaponColor *= 1.0f + rhythmAccuracy * 0.5f;
    }
    
    // Apply animation glow for special attacks
    // anim is already declared above, just reuse it
    if (anim.isActive) {
        if (anim.comboState == CombatState::STAFF_CAST) {
            weaponColor *= glm::vec3(1.5f, 1.2f, 2.0f); // Magic glow
        } else if (anim.comboState == CombatState::HAMMER_GROUND_POUND) {
            weaponColor *= glm::vec3(2.0f, 1.5f, 1.0f); // Power glow
        } else if (anim.comboState == CombatState::SWORD_THRUST) {
            weaponColor *= glm::vec3(1.8f, 1.8f, 1.2f); // Blade glow
        }
    }
    
    // Combine player rotation with weapon animation
    glm::mat4 finalRotation = playerRotation * weaponAnimRotation;
    renderCube(weaponPos, weaponSize, weaponColor, finalRotation);
}

void renderRhythmUI() {
    if (!showRhythmFlow) return;
    
    // Render beat indicator at bottom of screen
    float beatIndicatorSize = 0.2f + beatPulse * 0.3f;
    glm::vec3 beatPos(0.0f, -8.0f, -15.0f);
    glm::vec3 beatColor = onBeat ? glm::vec3(1.0f, 1.0f, 0.0f) : glm::vec3(0.3f, 0.3f, 0.3f);
    
    // Show upcoming beat
    float predictiveScale = 1.0f - (nextBeatTime / beatInterval);
    beatColor *= predictiveScale;
    
    renderCube(cameraPos + beatPos, glm::vec3(beatIndicatorSize), beatColor);
    
    // Render rhythm accuracy indicator
    if (rhythmAccuracy > 0.1f) {
        glm::vec3 accuracyPos(0.0f, -7.0f, -15.0f);
        glm::vec3 accuracyColor;
        
        if (rhythmAccuracy > 0.9f) {
            accuracyColor = glm::vec3(0.0f, 1.0f, 0.0f); // Perfect - Green
        } else if (rhythmAccuracy > 0.7f) {
            accuracyColor = glm::vec3(1.0f, 1.0f, 0.0f); // Good - Yellow
        } else {
            accuracyColor = glm::vec3(1.0f, 0.0f, 0.0f); // Miss - Red
        }
        
        renderCube(cameraPos + accuracyPos, glm::vec3(rhythmAccuracy * 0.5f, 0.1f, 0.1f), accuracyColor);
    }
    
    // Health bar
    float healthPercent = player.health / player.maxHealth;
    glm::vec3 healthPos(-5.0f, 8.0f, -15.0f);
    glm::vec3 healthColor = healthPercent > 0.5f ? glm::vec3(0.0f, 1.0f, 0.0f) : 
                           healthPercent > 0.25f ? glm::vec3(1.0f, 1.0f, 0.0f) : 
                           glm::vec3(1.0f, 0.0f, 0.0f);
    
    renderCube(cameraPos + healthPos, glm::vec3(healthPercent * 5.0f, 0.3f, 0.1f), healthColor);
}

void renderRippleEffects() {
    float currentTime = glfwGetTime();
    
    for (auto it = rippleEffects.begin(); it != rippleEffects.end();) {
        float age = currentTime - it->spawnTime;
        float lifetime = 2.0f; // Ripple lasts 2 seconds
        
        if (age > lifetime) {
            it = rippleEffects.erase(it);
        } else {
            float progress = age / lifetime;
            float currentRadius = it->maxRadius * progress;
            float alpha = 1.0f - progress; // Fade out
            
            // Render as expanding rings
            for (int i = 0; i < 3; i++) {
                float ringOffset = i * 0.5f;
                float ringRadius = currentRadius + ringOffset;
                if (ringRadius <= it->maxRadius) {
                    glm::vec3 ringColor = it->color * alpha * (1.0f - i * 0.3f);
                    
                    // Render ring as thin cube
                    renderCube(it->position, glm::vec3(ringRadius * 2.0f, 0.1f, ringRadius * 2.0f), ringColor);
                }
            }
            ++it;
        }
    }
}

void renderParticleExplosions() {
    float currentTime = glfwGetTime();
    
    for (auto it = explosionParticles.begin(); it != explosionParticles.end();) {
        float age = currentTime - it->spawnTime;
        
        if (age > it->lifetime) {
            it = explosionParticles.erase(it);
        } else {
            float progress = age / it->lifetime;
            glm::vec3 currentPos = it->position + it->velocity * age;
            
            // Apply gravity
            currentPos.y -= 0.5f * 9.8f * age * age;
            
            float alpha = 1.0f - progress;
            glm::vec3 color = it->color * alpha;
            
            // Render triangle as small rotating cube
            glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), age * 10.0f, glm::vec3(1.0f, 1.0f, 1.0f));
            renderCube(currentPos, glm::vec3(0.2f), color, rotation);
            
            ++it;
        }
    }
}

void renderChargingVisuals() {
    if (player.isChargingWeaponDive) {
        float chargeProgress = player.weaponDiveChargeTime / player.maxChargeTime;
        
        // Pulsing energy effect around player
        float pulseIntensity = 1.0f + sin(glfwGetTime() * 15.0f) * 0.5f;
        glm::vec3 chargeColor = glm::vec3(1.0f, 0.5f, 0.0f) * pulseIntensity * chargeProgress;
        
        // Render expanding charge sphere
        float sphereSize = 2.0f + chargeProgress * 3.0f;
        renderCube(player.position, glm::vec3(sphereSize), chargeColor * 0.3f);
        
        // Render energy lines to drawn enemies
        for (int enemyIndex : player.drawnEnemyIndices) {
            if (enemyIndex < enemies.size() && !enemies[enemyIndex].isDead) {
                glm::vec3 enemyPos = enemies[enemyIndex].position;
                glm::vec3 direction = glm::normalize(enemyPos - player.position);
                
                // Render energy beam as series of cubes
                float distance = glm::length(enemyPos - player.position);
                int beamSegments = (int)(distance / 0.5f);
                
                for (int i = 0; i < beamSegments; i++) {
                    float t = (float)i / beamSegments;
                    glm::vec3 beamPos = player.position + direction * distance * t;
                    
                    glm::vec3 beamColor = chargeColor * (1.0f - t * 0.5f);
                    renderCube(beamPos, glm::vec3(0.1f, 0.1f, 0.5f), beamColor);
                }
            }
        }
    }
}

void renderLandingEffects() {
    // This would render landing impact effects
    // For now, we'll use the existing ripple system
    // The actual landing effects are triggered in the game logic
}

// Render reward zone visuals
void renderRewardVisuals() {
    // Render reward zones
    for (const auto& zone : rewardZones) {
        glm::vec3 zoneColor = zone.reward.color;
        if (zone.isTriggered) {
            zoneColor *= 0.5f; // Dimmed when on cooldown
        } else {
            // Pulsing effect for active zones
            float pulse = 0.7f + 0.3f * sin(glfwGetTime() * 3.0f);
            zoneColor *= pulse;
        }
        
        // Render zone boundary
        renderCube(zone.position, zone.size, zoneColor * 0.3f);
        
        // Render zone indicator
        glm::vec3 indicatorPos = zone.position + glm::vec3(0, zone.size.y * 0.6f, 0);
        renderCube(indicatorPos, glm::vec3(0.5f), zoneColor);
    }
    
    // Render active player rewards as UI elements
    glm::vec3 cameraPosForUI = (gameCamera) ? gameCamera->position : cameraPos;
    for (size_t i = 0; i < player.activeRewards.size() && i < 5; ++i) {
        const auto& reward = player.activeRewards[i];
        glm::vec3 rewardPos = cameraPosForUI + glm::vec3(
            -4.0f + i * 1.0f,  // Spread horizontally
            3.0f,              // Above player
            -8.0f              // In front of camera
        );
        
        // Pulsing reward indicator
        float pulse = 0.8f + 0.2f * sin(glfwGetTime() * 5.0f);
        glm::vec3 rewardColor = reward.color * pulse;
        
        // Timer visualization - shrinking size
        float sizeScale = reward.timer / reward.duration;
        renderCube(rewardPos, glm::vec3(0.3f * sizeScale), rewardColor);
        
        // Orbiting elements around reward
        for (int j = 0; j < 3; ++j) {
            float angle = glfwGetTime() * 2.0f + (j * 2.0f * M_PI / 3.0f);
            glm::vec3 orbitPos = rewardPos + glm::vec3(
                cos(angle) * 0.5f,
                sin(angle * 2.0f) * 0.2f,
                sin(angle) * 0.5f
            );
            renderCube(orbitPos, glm::vec3(0.1f), rewardColor * 0.7f);
        }
    }
}

// Render enemy reward indicators
void renderEnemyRewardIndicators(const Enemy& enemy) {
    if (enemy.rewardState == Enemy::NO_REWARD || enemy.isDead) return;
    
    glm::vec3 enemyPos = enemy.position;
    glm::vec3 rewardColor = enemy.rewardAura;
    
    switch (enemy.rewardState) {
        case Enemy::CARRYING_REWARD: {
            // Gentle glow around enemy
            glm::vec3 glowPos = enemyPos + glm::vec3(0, enemy.size.y * 0.7f, 0);
            float glowSize = 1.0f + 0.3f * sin(enemy.rewardPulseTimer);
            renderCube(glowPos, glm::vec3(glowSize) * 0.8f, rewardColor * 0.4f);
            
            // Orbiting reward indicators
            for (int i = 0; i < enemy.rewardOrbitCount; ++i) {
                float angle = enemy.rewardPulseTimer * enemy.rewardOrbitSpeed + 
                             (i * 2.0f * M_PI / enemy.rewardOrbitCount);
                glm::vec3 orbitPos = enemyPos + glm::vec3(
                    cos(angle) * enemy.rewardOrbitRadius,
                    enemy.size.y * 0.5f + sin(angle * 2.0f) * 0.5f,
                    sin(angle) * enemy.rewardOrbitRadius
                );
                renderCube(orbitPos, glm::vec3(0.2f), rewardColor * enemy.rewardGlowIntensity);
            }
            break;
        }
        
        case Enemy::ABOUT_TO_SPARK: {
            // Intensifying glow with buildup particles
            glm::vec3 glowPos = enemyPos + glm::vec3(0, enemy.size.y * 0.7f, 0);
            float glowSize = 1.2f + 0.5f * sin(enemy.rewardPulseTimer * 2.0f);
            renderCube(glowPos, glm::vec3(glowSize), rewardColor * enemy.rewardGlowIntensity * 0.6f);
            
            // Render buildup particles
            for (const auto& particle : enemy.sparkParticles) {
                float particleIntensity = 0.5f + 0.5f * sin(glfwGetTime() * 8.0f);
                renderCube(particle, glm::vec3(0.1f), rewardColor * particleIntensity);
            }
            
            // Warning pulse
            float warningPulse = sin(enemy.sparkBuildupTimer * 5.0f);
            if (warningPulse > 0.5f) {
                renderCube(enemyPos, enemy.size * 1.1f, glm::vec3(1, 0.3f, 0.3f) * 0.3f);
            }
            break;
        }
        
        case Enemy::SPARKING: {
            // Intense sparking effect
            glm::vec3 glowPos = enemyPos + glm::vec3(0, enemy.size.y * 0.7f, 0);
            float sparkIntensity = 2.0f + sin(enemy.sparkTimer * 15.0f);
            renderCube(glowPos, glm::vec3(2.0f), rewardColor * sparkIntensity * 0.8f);
            
            // Intense spark particles
            for (const auto& particle : enemy.sparkParticles) {
                float particleIntensity = 1.0f + sin(glfwGetTime() * 12.0f + 
                                                   glm::length(particle) * 2.0f);
                renderCube(particle, glm::vec3(0.15f), rewardColor * particleIntensity);
            }
            
            // Lightning-like effects
            for (int i = 0; i < 5; ++i) {
                float angle = (i / 5.0f) * 2.0f * M_PI + enemy.sparkTimer * 3.0f;
                glm::vec3 lightningPos = enemyPos + glm::vec3(
                    cos(angle) * 2.0f,
                    sin(enemy.sparkTimer * 8.0f) * 1.5f + enemy.size.y,
                    sin(angle) * 2.0f
                );
                renderCube(lightningPos, glm::vec3(0.05f, 0.8f, 0.05f), 
                         glm::vec3(1, 1, 1) * sparkIntensity);
            }
            break;
        }
        
        case Enemy::REWARD_ACTIVATED: {
            // Fading afterglow
            glm::vec3 glowPos = enemyPos + glm::vec3(0, enemy.size.y * 0.7f, 0);
            float fadeGlow = enemy.rewardGlowIntensity * 0.5f;
            renderCube(glowPos, glm::vec3(1.5f), rewardColor * fadeGlow);
            break;
        }
    }
    
    // Special rendering for mini-bosses
    if (enemy.miniBossData && enemy.rewardState != Enemy::NO_REWARD) {
        // Crown-like indicator above mini-boss
        glm::vec3 crownPos = enemyPos + glm::vec3(0, enemy.size.y + 1.5f, 0);
        float crownPulse = 0.8f + 0.2f * sin(glfwGetTime() * 4.0f);
        renderCube(crownPos, glm::vec3(0.8f, 0.3f, 0.8f), 
                  glm::vec3(1, 0.8f, 0) * crownPulse);
        
        // Boss aura ring
        for (int i = 0; i < 8; ++i) {
            float angle = (i / 8.0f) * 2.0f * M_PI + glfwGetTime();
            glm::vec3 auraPos = enemyPos + glm::vec3(
                cos(angle) * 4.0f,
                0.5f,
                sin(angle) * 4.0f
            );
            renderCube(auraPos, glm::vec3(0.3f), rewardColor * 0.6f);
        }
    }
}

void renderPlayer(GLuint shaderProgram, const glm::mat4& view, const glm::mat4& projection) {
    // Calculate player rotation
    float playerRotation = player.rotation;
    if (player.isFlipping) {
        // Add flip rotation effect (could be enhanced further)
        playerRotation += player.flipRotation;
    }
    
    // Set body part colors based on rhythm and combat state
    float glowIntensity = 1.0f + (onBeat ? 0.5f : 0.0f);
    if (player.combatState != CombatState::NONE) {
        glowIntensity = 1.0f + 0.5f * sin(glfwGetTime() * 20.0f);
    }
    
    // Apply colors to body parts for visual feedback
    if (player.characterMesh) {
        glm::vec3 baseColor = glm::vec3(0.8f, 0.6f, 0.4f); // Humanoid skin tone
        glm::vec3 glowColor = baseColor * glowIntensity;
        
        // Combat glow modification
        if (player.combatState != CombatState::NONE) {
            glm::vec3 combatGlow = glm::vec3(1.0f, 0.5f, 0.0f); // Orange glow
            glowColor *= combatGlow;
        }
        
        // Apply color to all body parts with correct names
        player.characterMesh->setBodyPartColor("head", glm::vec3(1.0f, 0.3f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("torso", glm::vec3(0.3f, 0.3f, 1.0f) * glowColor);
        player.characterMesh->setBodyPartColor("left_upper_arm", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("left_lower_arm", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("right_upper_arm", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("right_lower_arm", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("left_upper_leg", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("left_lower_leg", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("left_foot", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("right_upper_leg", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("right_lower_leg", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        player.characterMesh->setBodyPartColor("right_foot", glm::vec3(0.3f, 1.0f, 0.3f) * glowColor);
        
        // Render the 3D character mesh with correct parameters
        player.characterMesh->render(shaderProgram, view, projection, player.position, playerRotation);
    }
}

void renderScene() {
    // Debug - mark render frame
    if (debugInfo.debugEnabled) {
        debugInfo.onRender();
    }
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glUseProgram(shaderProgram);
    
    // Set view and projection matrices using enhanced camera
    glm::mat4 view = (gameCamera) ? gameCamera->getViewMatrix() : glm::lookAt(cameraPos, cameraTarget, cameraUp);
    glm::mat4 projection = (gameCamera) ? gameCamera->getProjectionMatrix((float)WINDOW_WIDTH / WINDOW_HEIGHT) : 
                          glm::perspective(glm::radians(60.0f), (float)WINDOW_WIDTH / WINDOW_HEIGHT, 0.1f, 1000.0f);
    
    // Debug - check view matrix for invalid values
    if (debugInfo.debugEnabled && debugInfo.verbose) {
        // Check if view matrix has NaN or infinity values
        bool viewMatrixValid = true;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float val = view[i][j];
                if (std::isnan(val) || std::isinf(val)) {
                    viewMatrixValid = false;
                    break;
                }
            }
        }
        
        if (!viewMatrixValid) {
            debugInfo.logEvent("ERROR: View matrix contains invalid values!");
            std::cout << "[DEBUG] View matrix is invalid! Camera may be corrupted." << std::endl;
        }
    }
    
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    glBindVertexArray(cubeVAO);
    
    // Render platforms
    for (const auto& platform : platforms) {
        renderCube(platform.position, platform.size, platform.color);
    }
    
    // Render walls
    for (const auto& wall : walls) {
        renderCube(wall.position, wall.size, wall.color);
    }
    
    // Render enemies with enhanced targeting indicators
    for (int i = 0; i < enemies.size(); ++i) {
        const auto& enemy = enemies[i];
        
        if (enemy.isDead) {
            // Death animation - shrink and fade
            float deathScale = enemy.deathTimer / 2.0f;
            glm::vec3 enemyColor = glm::vec3(0.3f, 0.0f, 0.0f) * deathScale;
            renderCube(enemy.position, enemy.size * deathScale, enemyColor);
        } else {
            glm::vec3 enemyColor = enemy.isAggressive ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.8f, 0.2f, 0.2f);
            
            // Health indicator
            float healthPercent = enemy.health / enemy.maxHealth;
            if (healthPercent < 1.0f) {
                glm::vec3 healthBarPos = enemy.position + glm::vec3(0.0f, 2.5f, 0.0f);
                glm::vec3 healthColor = healthPercent > 0.5f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                renderCube(healthBarPos, glm::vec3(healthPercent, 0.1f, 0.1f), healthColor);
            }
            
            renderCube(enemy.position, enemy.size, enemyColor);
            renderEnemyRewardIndicators(enemy); // Render enemy reward visuals
            
            // Render wings for flying drones
            if (enemy.enemyType == EnemyType::FLYING_DRONE) {
                renderFlyingEnemyWings(enemy);
            }
        }
    }
    
    // Render enhanced targeting indicators separately (after all enemies)
    for (int i = 0; i < enemies.size(); ++i) {
        const auto& enemy = enemies[i];
        if (enemy.isDead) continue;
        
        bool isTargeted = (i == targetedEnemyIndex);
        bool isLockedOn = (gameCamera && gameCamera->isLockOnActive() && isTargeted);
        
        if (isTargeted) {
            // Determine indicator color based on enemy state
            glm::vec3 indicatorColor;
            float indicatorIntensity = 1.0f;
            
            if (isLockedOn) {
                // Lock-on indicator - different colors based on reward status
                if (enemy.rewardState == Enemy::CARRYING_REWARD) {
                    // Gold/yellow for reward carriers
                    indicatorColor = glm::vec3(1.0f, 0.8f, 0.0f);
                    indicatorIntensity = 1.5f + 0.5f * sin(glfwGetTime() * 8.0f);
                } else if (enemy.miniBossData) {
                    // Purple for mini-bosses
                    indicatorColor = glm::vec3(0.8f, 0.2f, 1.0f);
                    indicatorIntensity = 1.3f + 0.3f * sin(glfwGetTime() * 6.0f);
                } else {
                    // Bright cyan for lock-on
                    indicatorColor = glm::vec3(0.0f, 1.0f, 1.0f);
                    indicatorIntensity = 1.2f + 0.4f * sin(glfwGetTime() * 10.0f);
                }
            } else {
                // Regular targeting indicator
                if (enemy.rewardState == Enemy::CARRYING_REWARD) {
                    // Orange for reward carriers
                    indicatorColor = glm::vec3(1.0f, 0.6f, 0.0f);
                } else {
                    // White for regular targets
                    indicatorColor = glm::vec3(1.0f, 1.0f, 1.0f);
                }
                indicatorIntensity = 0.8f + 0.3f * sin(glfwGetTime() * 5.0f);
            }
            
            // Render targeting indicator ring
            glm::vec3 ringPos = enemy.position + glm::vec3(0.0f, enemy.size.y + 0.5f, 0.0f);
            float ringSize = 0.8f + (isLockedOn ? 0.4f : 0.2f);
            
            // Multiple ring segments for better visibility
            for (int j = 0; j < 12; ++j) {
                float angle = (j / 12.0f) * 2.0f * M_PI + (isLockedOn ? glfwGetTime() * 3.0f : 0.0f);
                glm::vec3 segmentPos = ringPos + glm::vec3(
                    cos(angle) * ringSize,
                    0.0f,
                    sin(angle) * ringSize
                );
                renderCube(segmentPos, glm::vec3(0.15f, 0.15f, 0.15f), 
                          indicatorColor * indicatorIntensity);
            }
            
            // Additional lock-on visual effects
            if (isLockedOn) {
                // Vertical beam indicator
                glm::vec3 beamPos = enemy.position + glm::vec3(0.0f, enemy.size.y + 3.0f, 0.0f);
                float beamIntensity = 0.7f + 0.3f * sin(glfwGetTime() * 8.0f);
                renderCube(beamPos, glm::vec3(0.1f, 2.0f, 0.1f), 
                          indicatorColor * beamIntensity);
                
                // Corner brackets for lock-on frame
                float bracketSize = enemy.size.x * 0.7f;
                for (int corner = 0; corner < 4; ++corner) {
                    float x = (corner & 1) ? bracketSize : -bracketSize;
                    float z = (corner & 2) ? bracketSize : -bracketSize;
                    glm::vec3 cornerPos = enemy.position + glm::vec3(x, enemy.size.y * 0.5f, z);
                    renderCube(cornerPos, glm::vec3(0.3f, 0.1f, 0.3f), 
                              indicatorColor * indicatorIntensity);
                }
            }
        }
    }
    
    // Render giant pillars for wall-running
    for (const auto& pillar : giantPillars) {
        renderCube(pillar.position, pillar.size, pillar.color);
    }

    // Render collectibles with pulsing effect
    float pulseScale = 1.0f + 0.3f * sin(glfwGetTime() * 5.0f);
    for (const auto& collectible : collectibles) {
        if (!collectible.collected) {
            renderCube(collectible.position, collectible.size * pulseScale, collectible.color);
        }
    }

    // Render visual effects
    renderRippleEffects();
    renderParticleExplosions();
    renderChargingVisuals();
    renderLandingEffects();
    
    // Render wall-running indicator
    if (player.isWallRunning && player.wallRunPillarIndex >= 0) {
        const auto& pillar = giantPillars[player.wallRunPillarIndex];
        // Render a glowing line from player to pillar showing wall-run connection
        glm::vec3 connectionColor = glm::vec3(0.2f, 0.8f, 1.0f) * (1.0f + (float)sin(glfwGetTime() * 8.0f) * 0.3f);
        
        // Draw connection line as series of small cubes
        glm::vec3 tempNormal;
        glm::vec3 closestSurface = pillar.getClosestSurface(player.position, tempNormal);
        glm::vec3 toPillar = closestSurface - player.position;
        int segments = 8;
        for (int i = 0; i < segments; i++) {
            float t = (float)i / segments;
            glm::vec3 linePos = player.position + toPillar * t;
            renderCube(linePos, glm::vec3(0.1f), connectionColor * (1.0f - t * 0.5f));
        }
        
        // Highlight the active pillar
        glm::vec3 pillarGlow = pillar.color * (1.5f + (float)sin(glfwGetTime() * 10.0f) * 0.5f);
        renderCube(pillar.position, pillar.size * 1.05f, pillarGlow);
    }

    // Render player
    renderPlayer(shaderProgram, view, projection);
    renderWeapon();
    
    // Render UI elements
renderRhythmUI();
renderRewardVisuals();
}

void setupWorld() {
    // Clear all game objects
    platforms.clear();
    walls.clear();
    enemies.clear();
    collectibles.clear();
    giantPillars.clear();
    arenaBoundaries.clear();
    
    // === 3D CHARACTER ACTION ARENA DESIGN ===
    
    // Central combat arena - circular design for 360-degree movement
    platforms.emplace_back(
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(80.0f, 1.0f, 80.0f),  // Main circular arena
        glm::vec3(0.4f, 0.4f, 0.5f)  // Dark gray-blue
    );
    
    // Inner raised ring for height variation
    for (int i = 0; i < 8; i++) {
        float angle = (i / 8.0f) * 2.0f * M_PI;
        float radius = 35.0f;
        platforms.emplace_back(
            glm::vec3(cos(angle) * radius, 5.0f, sin(angle) * radius),
            glm::vec3(12.0f, 1.0f, 12.0f),
            glm::vec3(0.6f, 0.5f, 0.7f)
        );
    }
    
    // Outer combat platforms - staggered heights
    for (int i = 0; i < 12; i++) {
        float angle = (i / 12.0f) * 2.0f * M_PI;
        float radius = 60.0f;
        float height = 3.0f + (i % 3) * 4.0f;  // Varying heights
        platforms.emplace_back(
            glm::vec3(cos(angle) * radius, height, sin(angle) * radius),
            glm::vec3(15.0f, 1.0f, 15.0f),
            glm::vec3(0.5f, 0.6f + 0.1f * (i % 3), 0.8f)
        );
    }
    
    // Quadrant arenas - larger combat zones in each direction
    float quadDist = 120.0f;
    glm::vec3 quadrantColors[4] = {
        glm::vec3(0.8f, 0.4f, 0.4f),  // Red quadrant
        glm::vec3(0.4f, 0.8f, 0.4f),  // Green quadrant
        glm::vec3(0.4f, 0.4f, 0.8f),  // Blue quadrant
        glm::vec3(0.8f, 0.8f, 0.4f)   // Yellow quadrant
    };
    
    for (int i = 0; i < 4; i++) {
        float angle = (i / 4.0f) * 2.0f * M_PI + M_PI/4;
        // Main quadrant platform
        platforms.emplace_back(
            glm::vec3(cos(angle) * quadDist, 0.0f, sin(angle) * quadDist),
            glm::vec3(50.0f, 1.0f, 50.0f),
            quadrantColors[i]
        );
        
        // Raised center platform in each quadrant
        platforms.emplace_back(
            glm::vec3(cos(angle) * quadDist, 8.0f, sin(angle) * quadDist),
            glm::vec3(25.0f, 1.0f, 25.0f),
            quadrantColors[i] * 0.7f
        );
    }
    
    // Connecting bridges between quadrants and center
    for (int i = 0; i < 4; i++) {
        float angle = (i / 4.0f) * 2.0f * M_PI + M_PI/4;
        glm::vec3 dir = glm::vec3(cos(angle), 0.0f, sin(angle));
        
        // Bridge segments
        for (int j = 1; j <= 3; j++) {
            platforms.emplace_back(
                dir * (30.0f + j * 20.0f) + glm::vec3(0.0f, 2.0f, 0.0f),
                glm::vec3(10.0f, 0.5f, 10.0f),
                glm::vec3(0.6f, 0.6f, 0.7f)
            );
        }
    }
    
    // === CREATE ARENA BOUNDARIES - Octagonal for better 3D movement ===
    float boundaryRadius = 180.0f;
    float boundaryHeight = 40.0f;
    
    for (int i = 0; i < 8; i++) {
        float angle = (i / 8.0f) * 2.0f * M_PI;
        float nextAngle = ((i + 1) / 8.0f) * 2.0f * M_PI;
        
        glm::vec3 pos = glm::vec3(
            (cos(angle) + cos(nextAngle)) * 0.5f * boundaryRadius,
            boundaryHeight/2,
            (sin(angle) + sin(nextAngle)) * 0.5f * boundaryRadius
        );
        
        glm::vec3 normal = -glm::normalize(glm::vec3(pos.x, 0.0f, pos.z));
        float wallWidth = boundaryRadius * 2.0f * sin(M_PI/8);
        
        arenaBoundaries.emplace_back(
            pos,
            normal,
            glm::vec3(wallWidth, boundaryHeight, 5.0f)
        );
        
        // Visible wall representation
        walls.emplace_back(
            pos,
            glm::vec3(wallWidth, boundaryHeight, 5.0f),
            glm::vec3(0.2f, 0.2f, 0.3f),
            false
        );
    }
    
    // === WALL-RUNNABLE STRUCTURES ===
    
    // Central pillars for vertical movement
    giantPillars.emplace_back(
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(15.0f, 40.0f, 15.0f),
        true  // cylindrical
    );
    
    // Ring of pillars around center
    for (int i = 0; i < 6; i++) {
        float angle = (i / 6.0f) * 2.0f * M_PI;
        float radius = 45.0f;
        giantPillars.emplace_back(
            glm::vec3(cos(angle) * radius, 0.0f, sin(angle) * radius),
            glm::vec3(10.0f, 30.0f, 10.0f),
            true
        );
    }
    
    // Quadrant pillars
    for (int i = 0; i < 4; i++) {
        float angle = (i / 4.0f) * 2.0f * M_PI + M_PI/4;
        giantPillars.emplace_back(
            glm::vec3(cos(angle) * 120.0f, 0.0f, sin(angle) * 120.0f),
            glm::vec3(20.0f, 50.0f, 20.0f),
            true
        );
    }
    
    // Wall-runnable walls in strategic locations
    for (int i = 0; i < 8; i++) {
        float angle = (i / 8.0f) * 2.0f * M_PI + M_PI/16;
        float radius = 90.0f;
        
        walls.emplace_back(
            glm::vec3(cos(angle) * radius, 10.0f, sin(angle) * radius),
            glm::vec3(4.0f, 20.0f, 30.0f),
            glm::vec3(0.5f, 0.5f, 0.8f),
            true  // wall-runnable
        );
    }
    
    // === 3D ARENA ENEMY PLACEMENT ===
    
    // Center arena enemies - mixed types
    for (int i = 0; i < 6; i++) {
        float angle = (i / 6.0f) * 2.0f * M_PI;
        float radius = 20.0f + (i % 2) * 10.0f;
        EnemyType centerType = static_cast<EnemyType>(i % 5);
        enemies.emplace_back(glm::vec3(cos(angle) * radius, 1.0f, sin(angle) * radius), centerType);
    }
    
    // Ring enemies on elevated platforms
    for (int i = 0; i < 12; i++) {
        float angle = (i / 12.0f) * 2.0f * M_PI;
        float radius = 60.0f;
        float height = 4.0f + (i % 3) * 4.0f;
        
        if (i % 3 == 0) {
            enemies.emplace_back(glm::vec3(cos(angle) * radius, height, sin(angle) * radius), EnemyType::FLYING_DRONE);
        } else if (i % 3 == 1) {
            enemies.emplace_back(glm::vec3(cos(angle) * radius, height, sin(angle) * radius), EnemyType::FAST_SCOUT);
        } else {
            enemies.emplace_back(glm::vec3(cos(angle) * radius, height, sin(angle) * radius), EnemyType::MAGIC_CASTER);
        }
    }
    
    // Quadrant enemies - tougher enemies in each zone
    for (int i = 0; i < 4; i++) {
        float angle = (i / 4.0f) * 2.0f * M_PI + M_PI/4;
        glm::vec3 quadCenter = glm::vec3(cos(angle) * 120.0f, 1.0f, sin(angle) * 120.0f);
        
        // Heavy brute in center of quadrant
        enemies.emplace_back(quadCenter, EnemyType::HEAVY_BRUTE);
        
        // Supporting enemies around
        for (int j = 0; j < 4; j++) {
            float subAngle = (j / 4.0f) * 2.0f * M_PI;
            glm::vec3 offset = glm::vec3(cos(subAngle) * 15.0f, 0.0f, sin(subAngle) * 15.0f);
            EnemyType supportType = (j % 2 == 0) ? EnemyType::BASIC_GRUNT : EnemyType::FAST_SCOUT;
            enemies.emplace_back(quadCenter + offset + glm::vec3(0.0f, 1.0f, 0.0f), supportType);
        }
    }
    
    // === 3D ARENA COLLECTIBLES ===
    
    // Center ring collectibles
    for (int i = 0; i < 8; i++) {
        float angle = (i / 8.0f) * 2.0f * M_PI;
        float radius = 25.0f;
        collectibles.emplace_back(
            glm::vec3(cos(angle) * radius, 2.0f, sin(angle) * radius),
            glm::vec3(0.5f, 0.5f, 0.5f),
            glm::vec3(1.0f, 1.0f, 0.0f)  // Yellow
        );
    }
    
    // Elevated platform collectibles
    for (int i = 0; i < 12; i++) {
        float angle = (i / 12.0f) * 2.0f * M_PI;
        float radius = 60.0f;
        float height = 5.0f + (i % 3) * 4.0f;
        collectibles.emplace_back(
            glm::vec3(cos(angle) * radius, height, sin(angle) * radius),
            glm::vec3(0.6f, 0.6f, 0.6f),
            glm::vec3(0.5f, 0.5f, 1.0f)  // Blue
        );
    }
    
    // Quadrant special collectibles
    for (int i = 0; i < 4; i++) {
        float angle = (i / 4.0f) * 2.0f * M_PI + M_PI/4;
        glm::vec3 quadCenter = glm::vec3(cos(angle) * 120.0f, 10.0f, sin(angle) * 120.0f);
        collectibles.emplace_back(
            quadCenter,
            glm::vec3(1.0f, 1.0f, 1.0f),
            glm::vec3(1.0f, 0.0f, 1.0f)  // Magenta - special
        );
    }
    
    // Outer ring exploration rewards
    for (int i = 0; i < 16; i++) {
        float angle = (i / 16.0f) * 2.0f * M_PI;
        float radius = 150.0f;
        collectibles.emplace_back(
            glm::vec3(cos(angle) * radius, 3.0f, sin(angle) * radius),
            glm::vec3(0.4f, 0.4f, 0.4f),
            glm::vec3(0.8f, 0.8f, 0.8f)  // White
        );
    }
    
    // === 3D ARENA REWARD ZONES ===
    rewardZones.clear();
    
    // Center speed boost zone
    rewardZones.emplace_back(
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(10.0f, 3.0f, 10.0f),
        ZoneRewardType::SPEED_BOOST
    );
    
    // Slow motion zones near enemy clusters
    for (int i = 0; i < 4; i++) {
        float angle = (i / 4.0f) * 2.0f * M_PI;
        rewardZones.emplace_back(
            glm::vec3(cos(angle) * 50.0f, 1.0f, sin(angle) * 50.0f),
            glm::vec3(8.0f, 3.0f, 8.0f),
            ZoneRewardType::SPEED_BOOST
        );
    }
    
    // Slow motion zones in quadrants for tactical combat
    for (int i = 0; i < 4; i++) {
        float angle = (i / 4.0f) * 2.0f * M_PI + M_PI/4;
        rewardZones.emplace_back(
            glm::vec3(cos(angle) * 120.0f, 1.0f, sin(angle) * 120.0f),
            glm::vec3(12.0f, 3.0f, 12.0f),
            ZoneRewardType::SLOW_MOTION
        );
    }
    
    // Health restore zones at inner ring
    for (int i = 0; i < 4; i++) {
        float angle = (i / 4.0f) * 2.0f * M_PI + M_PI/8;
        rewardZones.emplace_back(
            glm::vec3(cos(angle) * 35.0f, 6.0f, sin(angle) * 35.0f),
            glm::vec3(6.0f, 3.0f, 6.0f),
            ZoneRewardType::STAMINA_REFUND
        );
    }
    
    // Combat bonus zones on elevated platforms
    for (int i = 0; i < 6; i++) {
        float angle = (i / 6.0f) * 2.0f * M_PI;
        float radius = 60.0f;
        rewardZones.emplace_back(
            glm::vec3(cos(angle) * radius, 8.0f, sin(angle) * radius),
            glm::vec3(8.0f, 3.0f, 8.0f),
            ZoneRewardType::COMBO_BONUS
        );
    }
    
    // Wall run extension zones near wall-run structures
    for (int i = 0; i < 8; i++) {
        float angle = (i / 8.0f) * 2.0f * M_PI + M_PI/16;
        float radius = 90.0f;
        rewardZones.emplace_back(
            glm::vec3(cos(angle) * radius, 1.0f, sin(angle) * radius),
            glm::vec3(6.0f, 3.0f, 6.0f),
            ZoneRewardType::WALL_RUN_EXTENDED
        );
    }
    
    // Assign random rewards to enemies
    assignRandomRewards();
}

bool initOpenGL() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Rhythm Arena - High Speed Combat", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return false;
    }
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.15f, 1.0f);
    
    return true;
}

void setupShaders() {
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    
    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    // Create shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void setupGeometry() {
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        keys[key] = true;
        keysPressed[key] = true;
    } else if (action == GLFW_RELEASE) {
        keys[key] = false;
    }
    
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        player.position = glm::vec3(0.0f, 1.0f, 0.0f);
        player.velocity = glm::vec3(0.0f);
        player.rotation = 0.0f;
        player.combatState = CombatState::NONE;
        player.comboCount = 0;
        player.isFlipping = false;
        player.flipRotation = 0.0f;
        player.health = player.maxHealth;
        player.rhythm = RhythmState();
        score = 0;
        for (auto& collectible : collectibles) {
            collectible.collected = false;
        }
        setupWorld(); // Reset enemies too
        std::cout << "Game reset!\n";
    }
    
    // Toggle rhythm visualization
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        showRhythmFlow = !showRhythmFlow;
        std::cout << "Rhythm visualization: " << (showRhythmFlow ? "ON" : "OFF") << std::endl;
    }
    
    // Change BPM for testing
    if (key == GLFW_KEY_MINUS && action == GLFW_PRESS) {
        updateBPM(currentBPM - 10);
        std::cout << "BPM: " << currentBPM << std::endl;
    }
    if (key == GLFW_KEY_EQUAL && action == GLFW_PRESS) {
        updateBPM(currentBPM + 10);
        std::cout << "BPM: " << currentBPM << std::endl;
    }
    
    // Debug controls
    if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
        debugInfo.debugEnabled = !debugInfo.debugEnabled;
        std::cout << "Debug mode: " << (debugInfo.debugEnabled ? "ENABLED" : "DISABLED") << std::endl;
    }
    
    if (key == GLFW_KEY_F2 && action == GLFW_PRESS && debugInfo.debugEnabled) {
        debugInfo.verbose = !debugInfo.verbose;
        std::cout << "Verbose debug: " << (debugInfo.verbose ? "ON" : "OFF") << std::endl;
    }
    
    if (key == GLFW_KEY_F3 && action == GLFW_PRESS && debugInfo.debugEnabled) {
        // Manual camera reset
        if (gameCamera) {
            gameCamera->resetToPlayer(player.position);
            gameCamera->clearLockOnTarget();
            targetedEnemyIndex = -1;
            std::cout << "[DEBUG] Camera manually reset to player position" << std::endl;
        }
    }
    
    // Automated testing controls
    if (key == GLFW_KEY_F5 && action == GLFW_PRESS) {
        if (!isAutomatedTestMode) {
            std::cout << "\n[TEST] Starting automated testing mode..." << std::endl;
            isAutomatedTestMode = true;
            if (automatedTest) {
                automatedTest->startTesting();
            }
        } else {
            std::cout << "[TEST] Stopping automated testing mode..." << std::endl;
            isAutomatedTestMode = false;
            if (automatedTest) {
                automatedTest->stopTesting();
            }
        }
    }
}

void cleanup() {
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteBuffers(1, &cubeVBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
}

int main() {
    std::cout << "=== Enhanced Rhythm Arena Demo ===\n";
    std::cout << "KEYBOARD CONTROLS:\n";
    std::cout << "  WASD - Move (character rotates)\n";
    std::cout << "  Left Shift - Toggle sprint\n";
    std::cout << "  Space - Jump (double jump / wall-jump with WASD aiming)\n";
    std::cout << "  X - Dive (separate from jump)\n";
    std::cout << "  Y - Weapon dive charge (hold airborne, release for explosion)\n";
    std::cout << "  F - Dash attack (starts combos)\n";
    std::cout << "  G - Air dash launcher (airborne, launches enemies)\n";
    std::cout << "  Q - Light attack (build combos)\n";
    std::cout << "  E - Heavy attack / Launcher\n";
    std::cout << "  C - Slide (on ground with speed, jump for long jump)\n";
    std::cout << "  V - Grab enemy (throw with attack buttons)\n";
    std::cout << "  B - Parry (hold to block, counter with attacks)\n";
    std::cout << "  T - Camera lock-on toggle (smart targeting)\n";
    std::cout << "  1-4 - Switch weapons (Fists/Sword/Staff/Hammer)\n";
    std::cout << "\nCONTROLLER SUPPORT (Xbox/PlayStation):\n";
    std::cout << "  Left Stick - Movement\n";
    std::cout << "  Left Stick Click - Sprint toggle\n";
    std::cout << "  A/Cross - Jump\n";
    std::cout << "  X/Square - Dive\n";
    std::cout << "  Y/Triangle - Weapon dive charge\n";
    std::cout << "  Right Bumper - Parry (hold to block) / Dash attack\n";
    std::cout << "  Left Bumper - Air dash launcher\n";
    std::cout << "  Right Trigger - Light attack\n";
    std::cout << "  Left Trigger - Heavy attack/Launcher\n";
    std::cout << "  Right Stick Click - Camera lock-on toggle\n";
    std::cout << "  Back/Select - Grab enemy\n";
    std::cout << "  B/Circle - Slide\n";
    std::cout << "  D-Pad - Weapon switching\n";
    std::cout << "\nOTHER CONTROLS:\n";
    std::cout << "  Tab - Toggle rhythm visualization\n";
    std::cout << "  -/+ - Decrease/Increase BPM\n";
    std::cout << "  R - Reset game / Restart when game over\n";
    std::cout << "  ESC - Exit\n";
    std::cout << "\nDEBUG CONTROLS:\n";
    std::cout << "  F1 - Toggle debug mode\n";
    std::cout << "  F2 - Toggle verbose debug (when debug enabled)\n";
    std::cout << "  F3 - Manual camera reset (when debug enabled)\n";
    std::cout << "  F5 - Start/Stop automated testing\n";
    std::cout << "\nEnhanced Features:\n";
    std::cout << "  🆕 GIANT PILLARS with wall-running system!\n";
    std::cout << "  🆕 Wall-jump with 90° directional aiming!\n";
    std::cout << "  🆕 Combo-triggered wall-running activation!\n";
    std::cout << "  🆕 Death state with lives system\n";
    std::cout << "  🆕 Sprint toggle instead of hold\n";
    std::cout << "  🆕 Dive mechanics with enhanced bounce and floating\n";
    std::cout << "  🆕 HOLD X weapon dive with enemy draw and explosion!\n";
    std::cout << "  🆕 3D Triangle particle explosions\n";
    std::cout << "  🆕 Controller support (Xbox/PlayStation)\n";
    std::cout << "  🆕 Enhanced combo system with dash-light-launcher\n";
    std::cout << "  🆕 Enemy patrol AI when not aggressive\n";
    std::cout << "  🆕 Fall recovery system with health penalty\n";
    std::cout << "  🆕 Auto-targeting system for nearest enemy\n";
    std::cout << "  🆕 Increased weapon ranges\n";
    std::cout << "  🆕 Visual damage numbers\n";
    std::cout << "  🆕 Camera shake effects\n";
    std::cout << "  🆕 SLIDE mechanic with long jump capability!\n";
    std::cout << "  🆕 GRAB system - grab and throw enemies!\n";
    std::cout << "  🆕 PARRY system - perfect timing counter attacks!\n";
    std::cout << "  🆕 Progressive enhancement system!\n";
    std::cout << "  🆕 Damage multiplier and health scaling!\n";
    std::cout << "  🆕 Enemy stun mechanics and grab resistance!\n";
    std::cout << "  🆕 ENHANCED CAMERA LOCK-ON system:\n";
    std::cout << "     - Smart targeting (prioritizes reward carriers & bosses)\n";
    std::cout << "     - Color-coded indicators (Gold=Reward, Purple=Boss, Cyan=Locked)\n";
    std::cout << "     - Right stick click for controller support\n";
    std::cout << "  🆕 Enhanced camera with collision detection and dynamic FOV!\n";
    std::cout << "  ✅ Enemy health and damage system\n";
    std::cout << "  ✅ Player health (enemies attack back!)\n";
    std::cout << "  ✅ Weapon system (4 weapons total)\n";
    std::cout << "  ✅ Rhythm-based damage multipliers\n";
    std::cout << "  ✅ Visual rhythm flow indicators\n";
    std::cout << "  ✅ Dynamic BPM per level\n";
    std::cout << "  ✅ Perfect/Good/Miss timing system\n";
    std::cout << "=======================================\n\n";
    
    if (!initOpenGL()) {
        return -1;
    }
    
    setupShaders();
    setupGeometry();
    setupWorld();
    
    // Initialize player
    player.animController = new SimpleAnimationController();
    
    // Initialize 3D Character Mesh after OpenGL is ready
    player.characterMesh = new CharacterMesh();
    player.characterMesh->createHumanoidModel();
    player.characterMesh->initializeBuffers();
    player.characterAnimController = new CharacterAnimationController(player.characterMesh);
    
    // Initialize Enhanced3DCamera system
    gameCamera = new Enhanced3DCamera(
        player.position + glm::vec3(0.0f, 15.0f, 25.0f),  // Initial position
        player.position + glm::vec3(0.0f, 1.0f, 0.0f),    // Look target
        glm::vec3(0.0f, 1.0f, 0.0f)                       // Up vector
    );
    
    // Initialize enhanced controller input
    controllerInput = new EnhancedControllerInput();
    
    // Initialize automated testing system
    automatedTest = new AutomatedGameTest();
    automatedTest->initialize(&gameState);
    
    // Create all test scenarios
    automatedTest->createBasicMovementTest();
    automatedTest->createJumpMechanicsTest();
    automatedTest->createCombatComboTest();
    automatedTest->createWallRunningTest();
    automatedTest->createDeathRecoveryTest();
    automatedTest->createCameraLockOnTest();
    automatedTest->createWeaponSwitchingTest();
    automatedTest->createStressTest();
    
    // Configure camera for the game
    gameCamera->followPlayer = true;
    gameCamera->isDynamicDistance = true;
    gameCamera->enableCollisionDetection = true;
    gameCamera->setCameraMode(Enhanced3DCamera::CameraMode::PLAYER_FOLLOW);
    
    // Set initial camera position for compatibility
    cameraPos = gameCamera->position;
    cameraTarget = gameCamera->target;
    cameraUp = gameCamera->up;
    
    // Main game loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        deltaTime = std::min(deltaTime, 0.05f); // Cap delta time
        
        glfwPollEvents();
        
        // Update automated testing if active
        if (isAutomatedTestMode && automatedTest) {
            automatedTest->update(deltaTime);
        }
        
        processInput();
        updatePlayer();
        updateEnemies();
        updateFlyingEnemyAnimations(deltaTime);
        updateCamera(deltaTime, currentController.rightStickX, currentController.rightStickY);
        updateRhythm();
        updateWeaponAnimation(deltaTime);
        updateWeaponSwitching(deltaTime);
        updateGameState();
        
        // Update reward systems
        updateEnemyRewardStates(deltaTime);
        checkZoneRewardActivation();
        updatePlayerRewards(deltaTime);
        renderScene();
        glfwSwapBuffers(window);
    }
    
    delete player.animController;
    delete gameCamera;
    delete controllerInput;
    delete automatedTest;
    cleanup();
    
    return 0;
}

