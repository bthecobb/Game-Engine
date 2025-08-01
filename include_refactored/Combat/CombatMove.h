#pragma once

#include "Physics/CollisionDetection.h"
#include "Animation/AnimationSystem.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>

namespace CudaGame {
namespace Combat {

// Defines the properties of a single frame of an attack animation
struct FrameData {
    int frameNumber;
    bool isStartup = false;
    bool isActive = false;
    bool isRecovery = false;
    bool canCancel = false;         // Can this frame be cancelled into another move?
    bool canBuffer = false;         // Can the next move be buffered during this frame?
    float damage = 0.0f;
    float stun = 0.0f;              // How much stun this frame inflicts
    float knockback = 0.0f;         // How much knockback this frame inflicts
    glm::vec3 knockbackDirection{0.0f, 1.0f, 0.0f};
    bool isLauncher = false;        // Does this frame launch the opponent?
    float hitstop = 0.0f;           // Duration of hit-stop effect on hit
    std::string sfxOnHit;
    std::string vfxOnHit;
};

// Defines a hitbox for collision detection
struct Hitbox {
    Physics::OBB box;
    bool isActive = false;
    float damageMultiplier = 1.0f;
    int boneAttachment = -1; // -1 for no attachment
};

// Defines a single combat move (e.g., a punch, kick, or sword slash)
class Move {
public:
    Move(const std::string& name);
    ~Move() = default;

    const std::string& GetName() const { return m_name; }
    
    // Animation and timing
    Animation::AnimationState GetAnimationState() const { return m_animationState; }
    int GetTotalFrames() const { return m_totalFrames; }
    const FrameData& GetFrameData(int frame) const;
    const FrameData& GetFrameDataAtTime(float animationTime) const;
    
    // Hitboxes
    const std::vector<Hitbox>& GetHitboxes() const { return m_hitboxes; }
    void UpdateHitboxes(float animationTime, glm::mat4 entityTransform);

    // Move properties
    float GetBaseDamage() const { return m_baseDamage; }
    bool IsCancellable() const { return m_isCancellable; }

    // Rhythm integration
    float GetRhythmWindowStart() const { return m_rhythmWindowStart; }
    float GetRhythmWindowEnd() const { return m_rhythmWindowEnd; }
    
    // For data-driven setup
    void SetAnimation(Animation::AnimationState animState, int totalFrames);
    void AddFrameData(const FrameData& frameData);
    void AddHitbox(const Hitbox& hitbox);
    void SetRhythmWindow(float start, float end);

private:
    std::string m_name;
    Animation::AnimationState m_animationState;
    int m_totalFrames = 0;
    float m_baseDamage = 10.0f;
    bool m_isCancellable = false;
    
    std::vector<FrameData> m_frameData;
    std::vector<Hitbox> m_hitboxes;
    
    float m_rhythmWindowStart = 0.0f;
    float m_rhythmWindowEnd = 1.0f;
};

// Defines a sequence of moves that form a combo
class Combo {
public:
    Combo(const std::string& name);
    ~Combo() = default;
    
    const std::string& GetName() const { return m_name; }
    
    // Combo structure
    struct ComboNode {
        std::string moveName;
        float cancelWindowStart = 0.0f;
        float cancelWindowEnd = 1.0f;
        std::vector<std::string> nextMoves; // Possible follow-up moves
    };
    
    void AddNode(const ComboNode& node);
    const ComboNode* GetNode(const std::string& moveName) const;
    const ComboNode* GetStartingNode() const;
    void SetStartingMove(const std::string& moveName);

private:
    std::string m_name;
    std::vector<ComboNode> m_nodes;
    std::string m_startingMove;
};

} // namespace Combat
} // namespace CudaGame
