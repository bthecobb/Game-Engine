#pragma once

#include "Core/System.h"
#include "Core/Coordinator.h"
#include "LevelComponents.h"
#include "PlayerComponents.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/RenderComponents.h"

namespace CudaGame {
namespace Gameplay {

class LevelSystem : public Core::System {
public:
    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Level creation methods
    void CreateTestLevel();
    void CreateDimensionalPlatforms();
    void CreateWallRunSections();
    void CreateEnemies();
    void CreateCollectibles();

private:
    void UpdateDimensionalVisibility(float deltaTime);
    void UpdateWorldRotation(float deltaTime);
    void UpdateCollectibles(float deltaTime);
    void UpdateInteractables(float deltaTime);
    
    // Entity creation helpers
    Core::Entity CreatePlatform(const glm::vec3& position, const glm::vec3& scale, 
                                bool visibleViews[4], const glm::vec3& color = glm::vec3(1.0f));
    
    Core::Entity CreateWall(const glm::vec3& position, const glm::vec3& scale,
                           bool visibleViews[4], const glm::vec3& normal = glm::vec3(0.0f, 1.0f, 0.0f));
    
    Core::Entity CreateEnemy(const glm::vec3& position);
    
    Core::Entity CreateCollectible(const glm::vec3& position, 
                                  CollectibleComponent::CollectibleType type, float value);
    
    // World management
    ViewDirection currentView = ViewDirection::FRONT;
    bool isRotating = false;
    float rotationProgress = 0.0f;
    
    void RotateWorld(bool clockwise);
    void UpdateVisibilityForView(ViewDirection newView);
};

} // namespace Gameplay
} // namespace CudaGame
