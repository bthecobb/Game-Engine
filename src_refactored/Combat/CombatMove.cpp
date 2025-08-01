#include "Combat/CombatMove.h"
#include <stdexcept>
#include <algorithm>

namespace CudaGame {
namespace Combat {

// Move implementation
Move::Move(const std::string& name)
    : m_name(name) {}

const FrameData& Move::GetFrameData(int frame) const {
    if (frame < 0 || frame >= static_cast<int>(m_frameData.size())) {
        throw std::out_of_range("Frame index out of range");
    }
    return m_frameData[frame];
}

const FrameData& Move::GetFrameDataAtTime(float animationTime) const {
    int currentFrame = static_cast<int>(animationTime * 60.0f); // Assuming 60 FPS
    return GetFrameData(currentFrame);
}

void Move::UpdateHitboxes(float animationTime, glm::mat4 entityTransform) {
    // Calculate current frame based on animation time
    int currentFrame = static_cast<int>(animationTime * 60.0f); // Assuming 60 FPS
    
    // Update hitbox active states based on frame data
    for (auto& hitbox : m_hitboxes) {
        hitbox.isActive = false; // Default to inactive
        
        if (currentFrame >= 0 && currentFrame < static_cast<int>(m_frameData.size())) {
            const FrameData& frameData = m_frameData[currentFrame];
            if (frameData.isActive) {
                hitbox.isActive = true;
                
                // Transform hitbox based on entity transform
                hitbox.box.center = glm::vec3(entityTransform * glm::vec4(hitbox.box.center, 1.0f));
                
                // Apply entity rotation to hitbox orientation
                glm::mat3 rotationMatrix = glm::mat3(entityTransform);
                for (int i = 0; i < 3; ++i) {
                    hitbox.box.axes[i] = rotationMatrix * hitbox.box.axes[i];
                }
            }
        }
    }
}

void Move::SetAnimation(Animation::AnimationState animState, int totalFrames) {
    m_animationState = animState;
    m_totalFrames = totalFrames;
    m_frameData.resize(totalFrames);
}

void Move::AddFrameData(const FrameData& frameData) {
    if (frameData.frameNumber >= 0 && frameData.frameNumber < static_cast<int>(m_frameData.size())) {
        m_frameData[frameData.frameNumber] = frameData;
    }
}

void Move::AddHitbox(const Hitbox& hitbox) {
    m_hitboxes.push_back(hitbox);
}

void Move::SetRhythmWindow(float start, float end) {
    m_rhythmWindowStart = start;
    m_rhythmWindowEnd = end;
}

// Combo implementation
Combo::Combo(const std::string& name)
    : m_name(name) {}

void Combo::AddNode(const ComboNode& node) {
    m_nodes.push_back(node);
    
    // Set the first node as starting move if none is set
    if (m_startingMove.empty()) {
        m_startingMove = node.moveName;
    }
}

const Combo::ComboNode* Combo::GetNode(const std::string& moveName) const {
    auto it = std::find_if(m_nodes.begin(), m_nodes.end(),
        [&moveName](const ComboNode& node) {
            return node.moveName == moveName;
        });
    
    return (it != m_nodes.end()) ? &(*it) : nullptr;
}

const Combo::ComboNode* Combo::GetStartingNode() const {
    return GetNode(m_startingMove);
}

void Combo::SetStartingMove(const std::string& moveName) {
    m_startingMove = moveName;
}

} // namespace Combat
} // namespace CudaGame
