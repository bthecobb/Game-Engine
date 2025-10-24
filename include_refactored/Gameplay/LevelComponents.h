#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>

namespace CudaGame {
namespace Gameplay {

enum class ViewDirection {
    FRONT = 0,  // Looking down -Z axis (XY plane)
    RIGHT = 1,  // Looking down -X axis (ZY plane) 
    BACK = 2,   // Looking down +Z axis (XY plane, flipped)
    LEFT = 3    // Looking down +X axis (ZY plane, flipped)
};

// Component for objects that exist only in certain dimensional views
struct DimensionalVisibilityComponent {
    bool activeFromView[4] = {true, true, true, true}; // Visible from which views
    bool collidableFromView[4] = {true, true, true, true}; // Solid from which views
};

// World rotation/dimension switching component
struct WorldRotationComponent {
    ViewDirection currentView = ViewDirection::FRONT;
    float rotationProgress = 0.0f; // 0.0 to 1.0 during rotation animation
    bool isRotating = false;
    float rotationSpeed = 2.0f; // radians per second
};

// Platform component for level geometry
struct PlatformComponent {
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    bool isStatic = true;
    bool canWallRun = false;
};

// Wall component for wall-running surfaces
struct WallComponent {
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    bool canWallRun = true;
    float wallRunFriction = 0.1f;
    glm::vec3 color{0.5f, 0.5f, 0.5f};
};

// Collectible item component
struct CollectibleComponent {
    enum class CollectibleType {
        HEALTH_PACK,
        WEAPON_UPGRADE,
        ABILITY_ORB,
        KEY_ITEM
    };
    
    CollectibleType type = CollectibleType::HEALTH_PACK;
    float value = 25.0f; // health restored, damage bonus, etc.
    bool isCollected = false;
    float bobSpeed = 2.0f;
    float bobHeight = 0.5f;
    float rotationSpeed = 1.0f;
};

// Interactive object component
struct InteractableComponent {
    enum class InteractionType {
        DOOR,
        SWITCH,
        CHEST,
        TERMINAL
    };
    
    InteractionType type = InteractionType::SWITCH;
    bool isActivated = false;
    float interactionRange = 3.0f;
    bool requiresKey = false;
    int keyItemId = 0;
    std::string interactionText = "Press E to interact";
};

} // namespace Gameplay
} // namespace CudaGame
