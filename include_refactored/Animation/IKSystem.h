#pragma once

#include "Animation/IK.h"
#include "Core/System.h"
#include <memory>
#include <unordered_map>

namespace CudaGame {

// Forward declarations
namespace Core {
    class Coordinator;
}

namespace Animation {

// Configuration for procedural foot placement
struct FootPlacementSettings {
    float maxStepHeight = 0.5f;
    float footRadius = 0.15f;
    float pelvisOffset = 0.05f;
    float footIKWeight = 1.0f;
    bool enableFootIK = true;
    bool enablePelvisAdjustment = true;
};

// Configuration for procedural hand placement
struct HandPlacementSettings {
    float reachDistance = 1.5f;
    float handRadius = 0.1f;
    float handIKWeight = 1.0f;
    bool enableHandIK = true;
    bool enableLookAt = true;
};

// Manages all IK operations in the game
class IKSystem : public Core::System {
public:
    IKSystem();
    ~IKSystem() override = default;

    bool Initialize() override;
    void Update(float deltaTime) override;
    void Shutdown() override;

    // Entity management
    void RegisterSkeleton(uint32_t entityId, std::shared_ptr<Skeleton> skeleton);
    void UnregisterSkeleton(uint32_t entityId);
    
    // IK component management
    void AddIKComponent(uint32_t entityId, const IKComponent& ikComponent);
    IKComponent* GetIKComponent(uint32_t entityId);
    
    // Procedural animation helpers
    void EnableFootPlacement(uint32_t entityId, const FootPlacementSettings& settings);
    void EnableHandPlacement(uint32_t entityId, const HandPlacementSettings& settings);
    void SetGroundHeight(uint32_t entityId, float height);
    void SetLookAtTarget(uint32_t entityId, const glm::vec3& target);
    
    // Manual IK target setting
    void SetIKTarget(uint32_t entityId, const std::string& chainName, const glm::vec3& target);
    void ClearIKTarget(uint32_t entityId, const std::string& chainName);
    
    // Debugging
    void EnableDebugVisualization(bool enable) { m_debugVisualization = enable; }
    void DrawDebugInfo() const;

private:
    // Internal data structures
    std::unordered_map<uint32_t, std::shared_ptr<Skeleton>> m_skeletonRegistry;
    std::unordered_map<uint32_t, IKComponent> m_ikComponents;
    std::unordered_map<uint32_t, FootPlacementSettings> m_footPlacementSettings;
    std::unordered_map<uint32_t, HandPlacementSettings> m_handPlacementSettings;
    std::unordered_map<uint32_t, float> m_groundHeights;
    std::unordered_map<uint32_t, glm::vec3> m_lookAtTargets;
    
    bool m_debugVisualization = false;
    
    // Coordinator reference
    Core::Coordinator* m_coordinator = nullptr;
    
    // Helper functions
    void UpdateFootPlacement(uint32_t entityId, float deltaTime);
    void UpdateHandPlacement(uint32_t entityId, float deltaTime);
    void UpdateLookAt(uint32_t entityId, float deltaTime);
    void SolveIKChain(uint32_t entityId, const IKChain& chain);
    
    // Ground detection (placeholder)
    float DetectGroundHeight(const glm::vec3& position) const;
    bool IsValidFootPlacement(const glm::vec3& position) const;
};

} // namespace Animation
} // namespace CudaGame
