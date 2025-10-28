#include "Core/Coordinator.h"
#include "Animation/AnimationSystem.h"
#include "Audio/AudioSystem.h"
#include "Combat/CombatSystem.h"
#include "GameFeel/GameFeelSystem.h"
#include "Particles/ParticleSystem.h"
#include "Physics/PhysicsSystem.h"
#include "Rendering/RenderSystem.h"
#include "Rhythm/RhythmSystem.h"
#include <iostream>

int main() {
    std::cout << "Starting Full Game Demo..." << std::endl;

    // Coordinator setup
    Coordinator coordinator;
    coordinator.Init();
    
    // Initialize Systems
    auto& animSystem = coordinator.RegisterSystem<AnimationSystem>();
    auto& audioSystem = coordinator.RegisterSystem<AudioSystem>();
    auto& combatSystem = coordinator.RegisterSystem<CombatSystem>();
    auto& gameFeelSystem = coordinator.RegisterSystem<GameFeelSystem>();
    auto& particleSystem = coordinator.RegisterSystem<ParticleSystem>();
    auto& physicsSystem = coordinator.RegisterSystem<PhysicsSystem>();
    auto& renderSystem = coordinator.RegisterSystem<RenderSystem>();
    auto& rhythmSystem = coordinator.RegisterSystem<RhythmSystem>();

    // Configure Systems
    animSystem.Configure();
    audioSystem.Configure();
    combatSystem.Configure();
    gameFeelSystem.Configure();
    particleSystem.Configure();
    physicsSystem.Configure();
    renderSystem.Configure();
    rhythmSystem.Configure();

    // Load Resources and Scenes
    std::cout << "Loading assets and initializing scenes..." << std::endl;
    animSystem.LoadResources();
    audioSystem.LoadAudio();
    combatSystem.LoadCombos();
    particleSystem.Initialize();
    renderSystem.LoadShaders();

    // Main Game Loop
    bool running = true;
    while (running) {
        // Update systems
        animSystem.Update(0.016f); // Assuming 60fps
        audioSystem.Update(0.016f);
        combatSystem.Update(0.016f);
        gameFeelSystem.Update(0.016f);
        particleSystem.Update(0.016f);
        physicsSystem.Update(0.016f);
        renderSystem.Render();
        rhythmSystem.Update(0.016f);

        // Temp: exit condition after a loop
        // In practice, we would handle actual game events and controls
        running = false;
    }

    std::cout << "Demo completed!" << std::endl;
    return 0;
}
