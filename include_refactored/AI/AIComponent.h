#pragma once

#include "AI/BehaviorTree.h"
#include <memory>
#include <glm/glm.hpp>

namespace CudaGame {
namespace AI {

/**
 * [AAA Pattern] AI Blackboard / Behavior State
 * Holds the Behavior Tree instance for an entity.
 */
struct AIComponent {
    std::shared_ptr<Node> rootNode;
    
    // Blackboard data (shared memory for tree nodes)
    std::unordered_map<std::string, Core::Entity> targetEntities;
    std::unordered_map<std::string, glm::vec3> targetLocations;
    
    // Helper
    Core::Entity GetTarget(const std::string& key) {
        if (targetEntities.count(key)) return targetEntities[key];
        return 0;
    }
    
    void SetTarget(const std::string& key, Core::Entity target) {
        targetEntities[key] = target;
    }
};

} // namespace AI
} // namespace CudaGame
