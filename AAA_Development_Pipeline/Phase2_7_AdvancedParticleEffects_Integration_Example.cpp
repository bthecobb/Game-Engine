/*
 * Phase 2.7: Advanced Particle Effects Integration Example
 * 
 * This file demonstrates how to use the ParticleSystem to create stunning,
 * high-performance visual effects for combat, movement, and environments.
 */

#include "../include_refactored/Particles/ParticleSystem.h"
#include "../include_refactored/Combat/CombatSystem.h"
#include "../include_refactored/Core/Coordinator.h"

using namespace CudaGame;

class ParticleEffectsIntegrationDemo {
private:
    Core::Coordinator* m_coordinator;
    std::shared_ptr<Particles::ParticleSystem> m_particleSystem;
    std::shared_ptr<Combat::CombatSystem> m_combatSystem;

public:
    bool Initialize() {
        m_coordinator = Core::Coordinator::GetInstance();
        
        // Initialize systems
        m_particleSystem = std::make_shared<Particles::ParticleSystem>();
        m_combatSystem = std::make_shared<Combat::CombatSystem>();
        
        // Register systems with coordinator
        m_coordinator->RegisterSystem<Particles::ParticleSystem>(m_particleSystem);
        m_coordinator->RegisterSystem<Combat::CombatSystem>(m_combatSystem);
        
        // Configure particle system
        ConfigureParticleSystem();
        
        // Set up integration with combat system
        SetupCombatIntegration();
        
        return true;
    }
    
    void ConfigureParticleSystem() {
        // Enable high-performance features
        m_particleSystem->SetGlobalParticleLimit(20000);
        m_particleSystem->SetGPUAcceleration(true);
        m_particleSystem->SetLODEnabled(true);
        m_particleSystem->SetCullingEnabled(true);
        m_particleSystem->SetDebugVisualization(true);
    }
    
    void SetupCombatIntegration() {
        // Create particle effects for combat events
        m_combatSystem->RegisterCombatEventCallback([this](const Combat::CombatEvent& event) {
            switch (event.type) {
                case Combat::CombatEvent::Type::ATTACK_HIT:
                    // Create a blood splash effect on hit
                    m_particleSystem->CreateEffectFromPreset("Blood", event.position);
                    break;
                case Combat::CombatEvent::Type::COMBO_EXECUTED:
                    // Create a fiery trail for combo finishers
                    m_particleSystem->CreateEffectFromPreset("FireTrail", event.position);
                    break;
                case Combat::CombatEvent::Type::PARRY_SUCCESS:
                    // Create a spark effect for successful parries
                    m_particleSystem->CreateEffectFromPreset("Sparks", event.position);
                    break;
                case Combat::CombatEvent::Type::ENTITY_DIED:
                    // Create a dramatic soul-sucking effect on death
                    m_particleSystem->CreateEffectFromPreset("SoulVortex", event.position);
                    break;
                default: break;
            }
        });
    }
    
    void Update(float deltaTime) {
        // Update all systems
        m_particleSystem->Update(deltaTime);
        m_combatSystem->Update(deltaTime);
    }
    
    void DemonstrateParticleFeatures() {
        std::cout << "\n=== Advanced Particle Effects System Features ===\n";
        std::cout << "✓ CUDA-accelerated GPU simulation for high performance\n";
        std::cout << "✓ Data-driven effect presets for easy customization\n";
        std::cout << "✓ Advanced rendering: billboards, stretched billboards, trails, meshes\n";
        std::cout << "✓ Physics integration: gravity, drag, bounce, and collisions\n";
        std::cout << "✓ Animation system: texture animation, rotation, noise-based movement\n";
        std::cout << "✓ Environmental effects: force fields (wind, vortex, turbulence)\n";
        std::cout << "✓ Performance optimization: LOD, culling, particle pooling\n";
        std::cout << "✓ Seamless integration with combat and game feel systems\n";
    }
};

// Example usage function
void RunParticleEffectsDemo() {
    ParticleEffectsIntegrationDemo demo;
    
    if (demo.Initialize()) {
        std::cout << "Advanced Particle Effects System initialized successfully!\n";
        
        // Demonstrate features
        demo.DemonstrateParticleFeatures();
        
        // Simulate some updates
        for (int i = 0; i < 5; ++i) {
            demo.Update(0.016f); // ~60 FPS
        }
        
        std::cout << "\nParticle effects demo completed!\n";
    } else {
        std::cout << "Failed to initialize particle effects demo.\n";
    }
}

