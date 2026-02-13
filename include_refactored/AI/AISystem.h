#pragma once

#include "Core/System.h"
#include "AI/AIComponent.h"
#include "Gameplay/PlayerComponents.h" // To drive input

namespace CudaGame {
namespace AI {

/**
 * [AAA Pattern] AI System
 * Executes the Behavior Trees for all AI entities.
 * Bridges the gap between Decision (AI) and Action (InputComponent).
 */
class AISystem : public Core::System {
public:
    AISystem();
    ~AISystem();
    
    bool Initialize() override;
    void Update(float deltaTime) override;
    void Shutdown() override;
};

} // namespace AI
} // namespace CudaGame
