#include "Gameplay/LevelSystem.h"
#include "Gameplay/EnemyComponents.h"
#include "Core/Coordinator.h"
#include <iostream>
#include <cmath>

namespace CudaGame {
namespace Gameplay {

bool LevelSystem::Initialize() {
    std::cout << "LevelSystem initialized" << std::endl;
    CreateTestLevel();
    return true;
}

void LevelSystem::Shutdown() {
    std::cout << "LevelSystem shut down" << std::endl;
}

void LevelSystem::Update(float deltaTime) {
    UpdateDimensionalVisibility(deltaTime);
    UpdateWorldRotation(deltaTime);
    UpdateCollectibles(deltaTime);
    UpdateInteractables(deltaTime);
}

void LevelSystem::CreateTestLevel() {
    std::cout << "Creating test level with dimensional geometry..." << std::endl;
    
    CreateDimensionalPlatforms();
    CreateWallRunSections();
    CreateEnemies();
    CreateCollectibles();
    
    std::cout << "Test level created!" << std::endl;
}

void LevelSystem::CreateDimensionalPlatforms() {
    // Ground platform (visible from all views)
    bool allViews[4] = {true, true, true, true};
    CreatePlatform(glm::vec3(0, -1, 0), glm::vec3(20, 1, 20), allViews, glm::vec3(0.8f, 0.8f, 0.8f));
    
    // Platforms that only exist in certain views
    bool frontBack[4] = {true, false, true, false}; // Front and Back only
    bool leftRight[4] = {false, true, false, true}; // Left and Right only
    bool frontOnly[4] = {true, false, false, false}; // Front view only
    
    // Create a "path" that changes based on view
    CreatePlatform(glm::vec3(-5, 2, 0), glm::vec3(3, 0.5f, 3), frontBack, glm::vec3(1.0f, 0.0f, 0.0f));
    CreatePlatform(glm::vec3(0, 2, -5), glm::vec3(3, 0.5f, 3), leftRight, glm::vec3(0.0f, 1.0f, 0.0f));
    CreatePlatform(glm::vec3(5, 2, 0), glm::vec3(3, 0.5f, 3), frontBack, glm::vec3(1.0f, 0.0f, 0.0f));
    CreatePlatform(glm::vec3(0, 4, 5), glm::vec3(3, 0.5f, 3), frontOnly, glm::vec3(0.0f, 0.0f, 1.0f));
    
    // Jumping platforms at different heights
    CreatePlatform(glm::vec3(-8, 1, -8), glm::vec3(2, 0.5f, 2), allViews);
    CreatePlatform(glm::vec3(-6, 3, -6), glm::vec3(2, 0.5f, 2), frontBack);
    CreatePlatform(glm::vec3(-4, 5, -4), glm::vec3(2, 0.5f, 2), leftRight);
}

void LevelSystem::CreateWallRunSections() {
    // Walls for wall-running (tall and visible from multiple views)
    bool allViews[4] = {true, true, true, true};
    bool frontBack[4] = {true, false, true, false};
    
    // Wall run corridor
    CreateWall(glm::vec3(-10, 3, 10), glm::vec3(1, 6, 10), allViews, glm::vec3(1.0f, 0.0f, 0.0f));
    CreateWall(glm::vec3(10, 3, 10), glm::vec3(1, 6, 10), allViews, glm::vec3(-1.0f, 0.0f, 0.0f));
    
    // Dimensional walls that appear/disappear
    CreateWall(glm::vec3(0, 3, 15), glm::vec3(8, 6, 1), frontBack, glm::vec3(0.0f, 0.0f, -1.0f));
}

void LevelSystem::CreateEnemies() {
    // Create some enemies at strategic positions
    CreateEnemy(glm::vec3(8, 0, 8));
    CreateEnemy(glm::vec3(-8, 0, -8));
    CreateEnemy(glm::vec3(0, 0, 12));
}

void LevelSystem::CreateCollectibles() {
    // Create health packs and other collectibles
    CreateCollectible(glm::vec3(3, 1, 3), CollectibleComponent::CollectibleType::HEALTH_PACK, 25.0f);
    CreateCollectible(glm::vec3(-3, 1, -3), CollectibleComponent::CollectibleType::WEAPON_UPGRADE, 1.0f);
    CreateCollectible(glm::vec3(0, 3, -8), CollectibleComponent::CollectibleType::ABILITY_ORB, 1.0f);
}

Core::Entity LevelSystem::CreatePlatform(const glm::vec3& position, const glm::vec3& scale, 
                                        bool visibleViews[4], const glm::vec3& color) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    Core::Entity platform = coordinator.CreateEntity();
    
    // Transform component
    Rendering::TransformComponent transform;
    transform.position = position;
    transform.scale = scale;
    coordinator.AddComponent(platform, transform);
    
    // Mesh component
    Rendering::MeshComponent mesh;
    mesh.modelPath = "cube"; // Basic cube mesh
    coordinator.AddComponent(platform, mesh);
    
    // Material component  
    Rendering::MaterialComponent material;
    material.albedo = color;
    coordinator.AddComponent(platform, material);
    
    // Dimensional visibility
    DimensionalVisibilityComponent visibility;
    for (int i = 0; i < 4; i++) {
        visibility.activeFromView[i] = visibleViews[i];
        visibility.collidableFromView[i] = visibleViews[i];
    }
    coordinator.AddComponent(platform, visibility);
    
    // Platform component
    PlatformComponent platformComp;
    platformComp.color = color;
    coordinator.AddComponent(platform, platformComp);
    
    // Physics collider
    Physics::ColliderComponent collider;
    collider.shape = Physics::ColliderShape::BOX;
    collider.size = scale;
    coordinator.AddComponent(platform, collider);
    
    return platform;
}

Core::Entity LevelSystem::CreateWall(const glm::vec3& position, const glm::vec3& scale,
                                    bool visibleViews[4], const glm::vec3& normal) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    Core::Entity wall = coordinator.CreateEntity();
    
    // Transform component
    Rendering::TransformComponent transform;
    transform.position = position;
    transform.scale = scale;
    coordinator.AddComponent(wall, transform);
    
    // Mesh component
    Rendering::MeshComponent mesh;
    mesh.modelPath = "cube";
    coordinator.AddComponent(wall, mesh);
    
    // Material component
    Rendering::MaterialComponent material;
    material.albedo = glm::vec3(0.5f, 0.5f, 0.5f); // Gray walls
    coordinator.AddComponent(wall, material);
    
    // Dimensional visibility
    DimensionalVisibilityComponent visibility;
    for (int i = 0; i < 4; i++) {
        visibility.activeFromView[i] = visibleViews[i];
        visibility.collidableFromView[i] = visibleViews[i];
    }
    coordinator.AddComponent(wall, visibility);
    
    // Wall component
    WallComponent wallComp;
    wallComp.normal = normal;
    wallComp.canWallRun = true;
    coordinator.AddComponent(wall, wallComp);
    
    // Physics collider
    Physics::ColliderComponent collider;
    collider.shape = Physics::ColliderShape::BOX;
    collider.size = scale;
    coordinator.AddComponent(wall, collider);
    
    return wall;
}

Core::Entity LevelSystem::CreateEnemy(const glm::vec3& position) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    Core::Entity enemy = coordinator.CreateEntity();
    
    // Transform component
    Rendering::TransformComponent transform;
    transform.position = position;
    transform.scale = glm::vec3(1.0f, 2.0f, 1.0f);
    coordinator.AddComponent(enemy, transform);
    
    // Mesh component
    Rendering::MeshComponent mesh;
    mesh.modelPath = "cube"; // Simple cube for now
    coordinator.AddComponent(enemy, mesh);
    
    // Material component
    Rendering::MaterialComponent material;
    material.albedo = glm::vec3(1.0f, 0.0f, 0.0f); // Red enemies
    coordinator.AddComponent(enemy, material);
    
    // Enemy AI component
    EnemyAIComponent ai;
    ai.aiState = AIState::PATROL;
    ai.detectionRange = 15.0f;
    ai.attackRange = 3.0f;
    coordinator.AddComponent(enemy, ai);
    
    // Enemy combat component
    EnemyCombatComponent combat;
    combat.health = 75.0f;
    combat.maxHealth = 75.0f;
    combat.damage = 15.0f;
    coordinator.AddComponent(enemy, combat);
    
    // Enemy movement component
    EnemyMovementComponent movement;
    movement.speed = 8.0f;
    coordinator.AddComponent(enemy, movement);
    
    // Physics components
    Physics::RigidbodyComponent rigidbody;
    rigidbody.mass = 70.0f;
    coordinator.AddComponent(enemy, rigidbody);
    
    Physics::ColliderComponent collider;
    collider.shape = Physics::ColliderShape::BOX;
    collider.size = glm::vec3(1.0f, 2.0f, 1.0f);
    coordinator.AddComponent(enemy, collider);
    
    return enemy;
}

Core::Entity LevelSystem::CreateCollectible(const glm::vec3& position, 
                                           CollectibleComponent::CollectibleType type, float value) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    Core::Entity collectible = coordinator.CreateEntity();
    
    // Transform component
    Rendering::TransformComponent transform;
    transform.position = position;
    transform.scale = glm::vec3(0.5f, 0.5f, 0.5f);
    coordinator.AddComponent(collectible, transform);
    
    // Mesh component
    Rendering::MeshComponent mesh;
    mesh.modelPath = "sphere"; // Use sphere for collectibles
    coordinator.AddComponent(collectible, mesh);
    
    // Material component
    Rendering::MaterialComponent material;
    switch (type) {
        case CollectibleComponent::CollectibleType::HEALTH_PACK:
            material.albedo = glm::vec3(0.0f, 1.0f, 0.0f); // Green
            break;
        case CollectibleComponent::CollectibleType::WEAPON_UPGRADE:
            material.albedo = glm::vec3(0.0f, 0.0f, 1.0f); // Blue
            break;
        case CollectibleComponent::CollectibleType::ABILITY_ORB:
            material.albedo = glm::vec3(1.0f, 1.0f, 0.0f); // Yellow
            break;
        default:
            material.albedo = glm::vec3(1.0f, 1.0f, 1.0f); // White
            break;
    }
    coordinator.AddComponent(collectible, material);
    
    // Collectible component
    CollectibleComponent collectibleComp;
    collectibleComp.type = type;
    collectibleComp.value = value;
    coordinator.AddComponent(collectible, collectibleComp);
    
    return collectible;
}

void LevelSystem::UpdateDimensionalVisibility(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Update visibility based on current view
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<DimensionalVisibilityComponent>(entity) &&
            coordinator.HasComponent<Rendering::MeshComponent>(entity)) {
            auto& visibility = coordinator.GetComponent<DimensionalVisibilityComponent>(entity);
            auto& mesh = coordinator.GetComponent<Rendering::MeshComponent>(entity);
            
            // Set visibility based on current view
            mesh.isVisible = visibility.activeFromView[static_cast<int>(currentView)];
        }
    }
}

void LevelSystem::UpdateWorldRotation(float deltaTime) {
    // Handle world rotation input and animation
    // This would typically be triggered by player input
}

void LevelSystem::UpdateCollectibles(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<CollectibleComponent>(entity) &&
            coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
            auto& collectible = coordinator.GetComponent<CollectibleComponent>(entity);
            auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
            
            if (!collectible.isCollected) {
                // Bob up and down
                float bobOffset = sin(deltaTime * collectible.bobSpeed) * collectible.bobHeight;
                transform.position.y += bobOffset * deltaTime;
                
                // Rotate
                transform.rotation.y += collectible.rotationSpeed * deltaTime;
            }
        }
    }
}

void LevelSystem::UpdateInteractables(float deltaTime) {
    // Update interactive objects like doors, switches, etc.
    auto& coordinator = Core::Coordinator::GetInstance();
    
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<InteractableComponent>(entity)) {
            auto& interactable = coordinator.GetComponent<InteractableComponent>(entity);
            // Handle interaction logic here
        }
    }
}

} // namespace Gameplay
} // namespace CudaGame
