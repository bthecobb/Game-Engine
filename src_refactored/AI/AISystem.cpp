#include "AI/AISystem.h"
#include "Core/Coordinator.h"
#include <iostream>

namespace CudaGame {
namespace AI {

AISystem::AISystem() {
}

AISystem::~AISystem() {
    Shutdown();
}

bool AISystem::Initialize() {
    std::cout << "[AISystem] Initialized" << std::endl;
    return true;
}

void AISystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    for (auto const& entity : mEntities) {
        if (!coordinator.HasComponent<AIComponent>(entity)) continue;
        
        auto& ai = coordinator.GetComponent<AIComponent>(entity);
        
        if (ai.rootNode) {
            // Tick behavior tree
            ai.rootNode->Tick(entity, deltaTime);
        }
    }
}

void AISystem::Shutdown() {
    // Cleanup if needed
}

} // namespace AI
} // namespace CudaGame
