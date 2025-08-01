#pragma once

#include "Core/System.h"
#include "Particles/ParticleComponents.h"
#include "Rendering/RenderComponents.h"
#include "Physics/CollisionDetection.h"
#include <glm/glm.hpp>
#include <functional>
#include <unordered_map>
#include <memory>
#include <random>

namespace CudaGame {

// Forward declarations
namespace Physics {
    class PhysicsSystem;
}

namespace Particles {

// Particle effect presets for common effects
struct ParticleEffectPreset {
    std::string name;
    EmissionProperties emission;
    RenderProperties rendering;
    PhysicsProperties physics;
    AnimationProperties animation;
};

// GPU memory management for CUDA acceleration
struct GPUParticleData {
    float* positions = nullptr;
    float* velocities = nullptr;
    float* colors = nullptr;
    float* sizes = nullptr;
    float* lifetimes = nullptr;
    float* ages = nullptr;
    
    int maxParticles = 0;
    bool allocated = false;
    
    void Allocate(int maxCount);
    void Deallocate();
    void CopyToGPU(const std::vector<Particle>& particles);
    void CopyFromGPU(std::vector<Particle>& particles);
};

class ParticleSystem : public Core::System {
public:
    ParticleSystem();
    ~ParticleSystem();

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Particle system management
    Core::Entity CreateParticleSystem(const ParticleEffectPreset& preset);
    void DestroyParticleSystem(Core::Entity entity);
    
    // Particle emission control
    void PlayParticleSystem(Core::Entity entity);
    void StopParticleSystem(Core::Entity entity);
    void PauseParticleSystem(Core::Entity entity);
    void BurstParticles(Core::Entity entity, int count);
    
    // Effect presets management
    void RegisterEffectPreset(const std::string& name, const ParticleEffectPreset& preset);
    ParticleEffectPreset* GetEffectPreset(const std::string& name);
    Core::Entity CreateEffectFromPreset(const std::string& presetName, const glm::vec3& position);
    
    // Performance settings
    void SetGlobalParticleLimit(int limit) { m_globalParticleLimit = limit; }
    void SetLODEnabled(bool enabled) { m_lodEnabled = enabled; }
    void SetCullingEnabled(bool enabled) { m_cullingEnabled = enabled; }
    void SetGPUAcceleration(bool enabled) { m_useGPUAcceleration = enabled; }
    
    // Force field management
    void RegisterForceField(Core::Entity entity);
    void UnregisterForceField(Core::Entity entity);
    
    // Collision system integration
    void SetCollisionSystem(std::shared_ptr<Physics::PhysicsSystem> physicsSystem);
    
    // Statistics and debugging
    struct SystemStats {
        int totalParticleSystems = 0;
        int activeParticleSystems = 0;
        int totalParticles = 0;
        int activeParticles = 0;
        float averageSimulationTime = 0.0f;
        float averageRenderTime = 0.0f;
        int particlesEmittedThisFrame = 0;
        int particlesCulledThisFrame = 0;
    };
    
    const SystemStats& GetStats() const { return m_stats; }
    void SetDebugVisualization(bool enabled) { m_debugVisualization = enabled; }
    void DrawDebugInfo();
    
    // Callback system for particle events
    using ParticleSpawnCallback = std::function<void(Core::Entity system, Particle& particle)>;
    using ParticleDeathCallback = std::function<void(Core::Entity system, const Particle& particle)>;
    using ParticleCollisionCallback = std::function<void(Core::Entity system, Particle& particle, const Physics::ContactPoint& contact)>;
    
    void RegisterParticleSpawnCallback(ParticleSpawnCallback callback);
    void RegisterParticleDeathCallback(ParticleDeathCallback callback);
    void RegisterParticleCollisionCallback(ParticleCollisionCallback callback);

private:
    // System configuration
    int m_globalParticleLimit = 10000;
    bool m_lodEnabled = true;
    bool m_cullingEnabled = true;
    bool m_useGPUAcceleration = false;
    bool m_debugVisualization = false;
    
    // Camera reference for culling and LOD
    glm::vec3 m_cameraPosition{0.0f};
    glm::vec3 m_cameraForward{0.0f, 0.0f, -1.0f};
    
    // Effect presets
    std::unordered_map<std::string, ParticleEffectPreset> m_effectPresets;
    
    // Force fields
    std::vector<Core::Entity> m_forceFields;
    
    // Physics system integration
    std::shared_ptr<Physics::PhysicsSystem> m_physicsSystem;
    
    // GPU acceleration
    std::unordered_map<Core::Entity, std::unique_ptr<GPUParticleData>> m_gpuData;
    
    // Statistics
    SystemStats m_stats;
    
    // Random number generation
    std::mt19937 m_randomEngine;
    std::uniform_real_distribution<float> m_uniformDist;
    
    // Callbacks
    std::vector<ParticleSpawnCallback> m_spawnCallbacks;
    std::vector<ParticleDeathCallback> m_deathCallbacks;
    std::vector<ParticleCollisionCallback> m_collisionCallbacks;
    
    // Core simulation methods
    void UpdateParticleSystem(Core::Entity entity, ParticleSystemComponent& system, 
                            const Rendering::TransformComponent& transform, float deltaTime);
    
    // Emission
    void UpdateEmission(ParticleSystemComponent& system, const glm::vec3& systemPosition, float deltaTime);
    void EmitParticle(ParticleSystemComponent& system, const glm::vec3& systemPosition);
    glm::vec3 GetEmissionPosition(const EmissionProperties& emission, const glm::vec3& systemPosition);
    glm::vec3 GetEmissionVelocity(const EmissionProperties& emission);
    
    // Simulation
    void SimulateParticles(ParticleSystemComponent& system, float deltaTime);
    void UpdateParticle(Particle& particle, const PhysicsProperties& physics, float deltaTime);
    void ApplyForceFields(Particle& particle, const glm::vec3& systemPosition);
    
    // Collision detection
    void HandleParticleCollisions(ParticleSystemComponent& system);
    bool TestParticleCollision(const Particle& particle, const PhysicsProperties& physics, Physics::ContactPoint& outContact);
    
    // Animation and interpolation
    void UpdateParticleAnimation(Particle& particle, const AnimationProperties& animation, float deltaTime);
    void InterpolateParticleProperties(Particle& particle);
    
    // Performance optimization
    void ApplyLevelOfDetail(ParticleSystemComponent& system, const glm::vec3& systemPosition);
    void CullParticles(ParticleSystemComponent& system, const glm::vec3& systemPosition);
    bool IsParticleVisible(const Particle& particle, const glm::vec3& systemPosition);
    
    // GPU acceleration methods
    void InitializeGPUData(Core::Entity entity, ParticleSystemComponent& system);
    void UpdateGPUSimulation(Core::Entity entity, ParticleSystemComponent& system, float deltaTime);
    void SynchronizeGPUData(Core::Entity entity, ParticleSystemComponent& system);
    
    // Utility methods
    float GetRandomFloat(float min = 0.0f, float max = 1.0f);
    glm::vec3 GetRandomDirection();
    glm::vec3 GetRandomPositionInShape(const EmissionProperties& emission);
    
    // Force field calculations
    glm::vec3 CalculateWindForce(const ParticleForceFieldComponent& field, const glm::vec3& particlePos, float deltaTime);
    glm::vec3 CalculateVortexForce(const ParticleForceFieldComponent& field, const glm::vec3& particlePos, const glm::vec3& fieldPos);
    glm::vec3 CalculateMagnetForce(const ParticleForceFieldComponent& field, const glm::vec3& particlePos, const glm::vec3& fieldPos);
    glm::vec3 CalculateTurbulenceForce(const ParticleForceFieldComponent& field, const glm::vec3& particlePos, float time);
    glm::vec3 CalculateExplosionForce(const ParticleForceFieldComponent& field, const glm::vec3& particlePos, const glm::vec3& fieldPos);
    glm::vec3 CalculateGravityWellForce(const ParticleForceFieldComponent& field, const glm::vec3& particlePos, const glm::vec3& fieldPos);
    
    // Noise generation for procedural effects
    float PerlinNoise(const glm::vec3& position, float frequency);
    float SimplexNoise(const glm::vec3& position, float frequency);
    
    // Debug visualization
    void DrawParticleSystem(const ParticleSystemComponent& system, const glm::vec3& systemPosition);
    void DrawForceField(const ParticleForceFieldComponent& field, const glm::vec3& fieldPosition);
    void DrawEmissionShape(const EmissionProperties& emission, const glm::vec3& systemPosition);
    
    // Default effect presets initialization
    void InitializeDefaultPresets();
    void CreateSmokePreset();
    void CreateFirePreset();
    void CreateSparkPreset();
    void CreateMagicPreset();
    void CreateExplosionPreset();
    void CreateBloodPreset();
    void CreateDustPreset();
    void CreateWaterPreset();
};

} // namespace Particles
} // namespace CudaGame
