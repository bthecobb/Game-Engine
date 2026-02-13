#pragma once

#include "Core/System.h"
#include "Gameplay/CharacterResources.h"
#include "Gameplay/CombatComponents.h"
#include "Core/Coordinator.h"
#include <unordered_map>
#include <string>

namespace CudaGame {

namespace Animation {
    class Skeleton;
}

namespace Rendering {
    class DX12RenderPipeline;
}

namespace Gameplay {

/**
 * [AAA Pattern] Character Factory
 * Centralized service to spawn entities based on data-driven profiles.
 * Manages resource caching (AnimationSets, Profiles) to avoid redundant loading.
 */
class CharacterFactory {
public:
    CharacterFactory();
    ~CharacterFactory();
    
    // Lifecycle
    bool Initialize();
    void Shutdown();
    
    // Dependencies
    void SetRenderPipeline(Rendering::DX12RenderPipeline* pipeline) { m_renderPipeline = pipeline; }
    
    // Asset Management
    void RegisterProfile(const std::string& profileName, const CharacterProfile& profile);
    void RegisterAnimationSet(const std::string& setName, const AnimationSet& animSet);
    void RegisterSkeleton(const std::string& skeletonID, std::shared_ptr<Animation::Skeleton> skeleton);
    void RegisterWeaponDefinition(const std::string& weaponID, const WeaponDefinition& def);
    
    // Spawning
    Core::Entity SpawnCharacter(const std::string& profileName, const glm::vec3& position);
    // SetupPlayer removed
    // void SetupPlayer(Core::Entity playerEntity);
    
private:
    Rendering::DX12RenderPipeline* m_renderPipeline = nullptr;
    std::unordered_map<ResourceID, CharacterProfile> m_profiles;
    std::unordered_map<ResourceID, AnimationSet> m_animationSets;
    std::unordered_map<ResourceID, std::shared_ptr<Animation::Skeleton>> m_skeletons;
    std::unordered_map<ResourceID, WeaponDefinition> m_weapons;
    
    // Helper to assemble ECS components
    void AssembleCharacter(Core::Entity entity, const CharacterProfile& profile);
    void SpawnWeapon(Core::Entity owner, const std::string& weaponID);
};

} // namespace Gameplay
} // namespace CudaGame
