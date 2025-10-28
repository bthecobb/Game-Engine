#pragma once

#include "GameWorld.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <memory>

enum class MovementState {
    IDLE,
    WALKING,
    RUNNING,
    SPRINTING,
    JUMPING,
    WALL_RUNNING,
    DASHING
};

enum class AnimationState {
    IDLE_ANIM,
    WALK_CYCLE,
    RUN_CYCLE,
    SPRINT_CYCLE,
    JUMP_START,
    JUMP_PEAK,
    JUMP_LAND,
    DASH_POSE,
    WALL_RUN_POSE,
    ATTACK_LIGHT,
    ATTACK_HEAVY,
    BLOCK_POSE
};

enum class CombatState {
    NEUTRAL,
    ATTACKING,
    BLOCKING,
    STUNNED
};

// Character body part for detailed modeling
struct BodyPart {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
    glm::vec3 color{1.0f};
    unsigned int VAO = 0;
    unsigned int VBO = 0;
    unsigned int EBO = 0;
    int indexCount = 0;
};

// Player particle for trails and effects (renamed to avoid conflict)
struct PlayerParticle {
    glm::vec3 position{0.0f};
    glm::vec3 velocity{0.0f};
    glm::vec3 color{1.0f};
    float life = 1.0f;
    float maxLife = 1.0f;
    float size = 1.0f;
    bool active = false;
};

// Animation keyframe for smooth transitions
struct AnimationKeyframe {
    float time;
    std::vector<glm::vec3> bodyPartPositions;
    std::vector<glm::vec3> bodyPartRotations;
};

// Animation sequence
struct Animation {
    std::vector<AnimationKeyframe> keyframes;
    float duration;
    bool looping = true;
};

class Player : public GameObject {
public:
    Player();
    ~Player() override = default;
    
    // GameObject interface
    void update(float deltaTime) override;
    void render() override;
    
    // Input handling
    void handleInput(const bool keys[1024], float deltaTime);
    void handleMouseInput(double mouseX, double mouseY);
    
    // Animation and Visual Effects
    void loadAnimations();
    void playAnimation(AnimationState animState);
    void addParticleTrail();
    void createParticleEffects(glm::vec3 position, int count, float size);
    
    // Rhythm Visuals
    void updateRhythmFeedback(float deltaTime);
    
    // Movement
    void move(glm::vec2 inputDirection, float deltaTime);
    void jump();
    void dash();
    void wallRun(glm::vec3 wallNormal, float deltaTime);
    
    // More details
    void updateBodyParts(float deltaTime);
    
    // Combat
    void attack(bool heavy = false);
    void block();
    
    // Getters
    glm::vec3 getPosition() const { return transform.position; }
    glm::vec3 getVelocity() const { return m_velocity; }
    float getCurrentSpeed() const { return glm::length(m_velocity); }
    MovementState getMovementState() const { return m_movementState; }
    AnimationState getCurrentAnimationState() const { return m_currentAnimationState; }
    CombatState getCombatState() const { return m_combatState; }
    bool isGrounded() const { return m_isGrounded; }
    bool isOnBeat() const { return m_isOnBeat; }
    
    // Setters
    void setGameWorld(GameWorld* world) { m_gameWorld = world; }
    
    // Momentum preservation for world rotation
    void preserveMomentumOnRotation();

private:
    // Movement properties
    glm::vec3 m_velocity{0.0f};
    MovementState m_movementState = MovementState::IDLE;
    
    // Speed parameters
    float m_baseSpeed = 10.0f;
    float m_maxSpeed = 50.0f;
    float m_acceleration = 30.0f;
    float m_deceleration = 20.0f;
    float m_airAcceleration = 15.0f;
    
    // Jump parameters
    float m_jumpForce = 20.0f;
    float m_gravity = 40.0f;
    bool m_isGrounded = false;
    bool m_canDoubleJump = true;
    
    // Dash parameters
    float m_dashForce = 80.0f;
    float m_dashDuration = 0.2f;
    float m_dashCooldown = 1.0f;
    float m_dashTimer = 0.0f;
    float m_dashCooldownTimer = 0.0f;
    bool m_isDashing = false;
    
    // Wall running
    float m_wallRunSpeed = 25.0f;
    float m_wallRunDuration = 2.0f;
    float m_wallRunTimer = 0.0f;
    bool m_isWallRunning = false;
    glm::vec3 m_wallNormal{0.0f};
    
    // Combat
    float m_attackCooldown = 0.5f;
    float m_attackTimer = 0.0f;
    bool m_isBlocking = false;
    
    // Rhythm feedback
    float m_beatTimer = 0.0f;
    bool m_isOnBeat = false;
    
    // Animations
    AnimationState m_currentAnimationState = AnimationState::IDLE_ANIM;
    CombatState m_combatState = CombatState::NEUTRAL;
    std::vector<Animation> m_animations;
    std::vector<BodyPart> m_bodyParts;
    
    // Particle effects
    std::vector<PlayerParticle> m_particles;
    float m_particleCooldown = 0.0f;
    
    // References
    GameWorld* m_gameWorld = nullptr;
    
    // Internal methods
    void updateMovement(float deltaTime);
    void updatePhysics(float deltaTime);
    void updateGrounding();
    void updateWallRunning(float deltaTime);
    void updateDashing(float deltaTime);
    void updateCombat(float deltaTime);
    
    // Physics helpers
    bool checkGrounding();
    glm::vec3 checkWallCollision();
    void applyFriction(float deltaTime);
    void applyGravity(float deltaTime);
    
    // Movement helpers
    glm::vec2 getMovementInput(const bool keys[1024]);
    glm::vec2 worldToViewDirection(glm::vec2 input, ViewDirection currentView);
    void buildMomentum(glm::vec2 inputDirection, float deltaTime);
    
    // Visual system helpers
    void initializeBodyParts();
    void createBodyPartMesh(BodyPart& part, const std::vector<float>& vertices, const std::vector<unsigned int>& indices);
    void renderBodyPart(const BodyPart& part, const glm::mat4& parentTransform);
    void updateParticles(float deltaTime);
    void renderParticles();
    void updateAnimations(float deltaTime);
    glm::vec3 interpolateBodyPartPosition(const Animation& anim, float time, int partIndex);
    glm::vec3 interpolateBodyPartRotation(const Animation& anim, float time, int partIndex);
    
    // Shader and rendering
    unsigned int m_characterShaderProgram = 0;
    unsigned int m_particleShaderProgram = 0;
    void createCharacterShaders();
    void createParticleShaders();
    
    // Animation timing
    float m_animationTime = 0.0f;
    float m_animationSpeed = 1.0f;
};
