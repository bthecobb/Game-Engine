#include "Particles/ParticleSystem.h"
#include <iostream>

namespace CudaGame {
namespace Particles {

ParticleSystem::ParticleSystem() : m_randomEngine(std::random_device{}()), m_uniformDist(0.0f, 1.0f) {
    // Particle system updates after physics, before rendering
}

ParticleSystem::~ParticleSystem() {
    Shutdown();
}

bool ParticleSystem::Initialize() {
    std::cout << "[ParticleSystem] Initialized. Managing " << mEntities.size() << " entities." << std::endl;
    return true;
}

void ParticleSystem::Shutdown() {
    std::cout << "[ParticleSystem] Shutting down particle effects system." << std::endl;
}

void ParticleSystem::Update(float deltaTime) {
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Update Camera Position for Sorting/LOD
    // (In a real system, we'd get the main camera entity)
    // For now, assume origin or wait for specific SetCamera call
    
    for (auto const& entity : mEntities) {
        auto& system = coordinator.GetComponent<ParticleSystemComponent>(entity);
        const auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
        
        UpdateParticleSystem(entity, system, transform, deltaTime);
    }
}

void ParticleSystem::UpdateParticleSystem(Core::Entity entity, ParticleSystemComponent& system, 
                                        const Rendering::TransformComponent& transform, float deltaTime) {
    if (!system.isPlaying) return;
    
    // 1. Update Lifetime
    system.systemAge += deltaTime;
    if (!system.isLooping && system.systemAge >= system.systemLifetime) {
        system.isPlaying = false;
        return;
    }
    
    // 2. Emission
    UpdateEmission(system, transform.position, deltaTime);
    
    // 3. Simulation
    if (system.useGPUSimulation) {
        UpdateGPUSimulation(entity, system, deltaTime);
    } else {
        SimulateParticles(system, deltaTime);
    }
    
    // 4. Update Stats
    system.stats.particlesActiveThisFrame = system.activeParticles;
}

void ParticleSystem::UpdateEmission(ParticleSystemComponent& system, const glm::vec3& systemPosition, float deltaTime) {
    // Continuous Emission
    if (system.emission.continuous) {
        float emissionInterval = 1.0f / system.emission.emissionRate;
        system.emissionTimer += deltaTime;
        
        while (system.emissionTimer >= emissionInterval) {
            EmitParticle(system, systemPosition);
            system.emissionTimer -= emissionInterval;
            system.stats.particlesEmittedThisFrame++;
        }
    }
    
    // Burst Emission (Simplified)
    // In a real system, we'd track bursts in a list
}

void ParticleSystem::EmitParticle(ParticleSystemComponent& system, const glm::vec3& systemPosition) {
    Particle* p = system.GetFreeParticle();
    if (!p) return;
    
    // Init Properties
    p->position = GetEmissionPosition(system.emission, systemPosition);
    p->velocity = GetEmissionVelocity(system.emission);
    p->acceleration = glm::vec3(0.0f);
    
    p->lifetime = system.emission.particleLifetime + GetRandomFloat(-0.5f, 0.5f) * system.emission.lifetimeVariation;
    p->age = 0.0f;
    p->normalizedAge = 0.0f;
    p->isActive = true;
    
    p->size = 1.0f; // Simplified
    p->color = glm::vec4(1.0f); // Simplified
}

void ParticleSystem::SimulateParticles(ParticleSystemComponent& system, float deltaTime) {
    for (int i = 0; i < system.maxParticles; ++i) {
        auto& p = system.particles[i];
        if (!p.isActive) continue;
        
        // 1. Update Age
        p.age += deltaTime;
        if (p.age >= p.lifetime) {
            system.ReturnParticle(i);
            continue;
        }
        p.normalizedAge = p.age / p.lifetime;
        
        // 2. Physics
        // Apply Gravity
        p.velocity += system.physics.gravity * deltaTime;
        
        // Apply Drag
        if (system.physics.drag > 0.0f) {
            p.velocity *= (1.0f - system.physics.drag * deltaTime);
        }
        
        // Integrate Position
        p.position += p.velocity * deltaTime;
        
        // 3. Visuals (Interpolation)
        if (system.physics.colorOverLifetime) {
            p.color = glm::mix(p.startColor, p.endColor, p.normalizedAge);
        }
        
        if (system.physics.sizeOverLifetime) {
            p.size = glm::mix(p.startSize, p.endSize, p.normalizedAge);
        }
    }
}

void ParticleSystem::InitializeDefaultPresets() {
    std::cout << "[ParticleSystem] Creating default effect presets..." << std::endl;
    CreateSmokePreset();
    CreateFirePreset();
    CreateSparkPreset();
    CreateMagicPreset();
    CreateExplosionPreset();
    CreateBloodPreset();
    CreateDustPreset();
    CreateWaterPreset();
}

void ParticleSystem::CreateSmokePreset() {
    ParticleEffectPreset smoke;
    smoke.name = "Smoke";
    // ... (full preset implementation would be here)
    RegisterEffectPreset(smoke.name, smoke);
}

void ParticleSystem::CreateFirePreset() {
    ParticleEffectPreset preset;
    preset.name = "Fire";
    preset.emission.shape = EmissionProperties::EmissionShape::SPHERE;
    preset.emission.emissionRate = 80;
    // Additional properties setup...
    RegisterEffectPreset(preset.name, preset);
}

void ParticleSystem::CreateSparkPreset() {
    ParticleEffectPreset preset;
    preset.name = "Spark";
    preset.emission.shape = EmissionProperties::EmissionShape::POINT;
    preset.emission.emissionRate = 100;
    // Additional properties setup...
    RegisterEffectPreset(preset.name, preset);
}

void ParticleSystem::CreateMagicPreset() {
    ParticleEffectPreset preset;
    preset.name = "Magic";
    preset.emission.shape = EmissionProperties::EmissionShape::SPHERE;
    preset.emission.emissionRate = 60;
    // Additional properties setup...
    RegisterEffectPreset(preset.name, preset);
}

void ParticleSystem::CreateExplosionPreset() {
    ParticleEffectPreset preset;
    preset.name = "Explosion";
    preset.emission.shape = EmissionProperties::EmissionShape::SPHERE;
    preset.emission.emissionRate = 150;
    // Additional properties setup...
    RegisterEffectPreset(preset.name, preset);
}

void ParticleSystem::CreateBloodPreset() {
    ParticleEffectPreset preset;
    preset.name = "Blood";
    preset.emission.shape = EmissionProperties::EmissionShape::CONE;
    preset.emission.emissionRate = 70;
    // Additional properties setup...
    RegisterEffectPreset(preset.name, preset);
}

void ParticleSystem::CreateDustPreset() {
    ParticleEffectPreset preset;
    preset.name = "Dust";
    preset.emission.shape = EmissionProperties::EmissionShape::BOX;
    preset.emission.emissionRate = 40;
    // Additional properties setup...
    RegisterEffectPreset(preset.name, preset);
}

void ParticleSystem::CreateWaterPreset() {
    ParticleEffectPreset preset;
    preset.name = "Water";
    preset.emission.shape = EmissionProperties::EmissionShape::BOX;
    preset.emission.emissionRate = 30;
    // Additional properties setup...
    RegisterEffectPreset(preset.name, preset);
}

void ParticleSystem::RegisterEffectPreset(const std::string& name, const ParticleEffectPreset& preset) {
    m_effectPresets[name] = preset;
    std::cout << "Particle effect preset '" << name << "' registered." << std::endl;
}

// Helper Implementations
glm::vec3 ParticleSystem::GetEmissionPosition(const EmissionProperties& emission, const glm::vec3& systemPosition) {
    // Simple sphere/point emission for now
    if (emission.shape == EmissionProperties::EmissionShape::POINT) {
        return systemPosition;
    }
    
    // Sphere
    glm::vec3 randomDir = GetRandomDirection();
    float radius = GetRandomFloat(0.0f, emission.emissionRadius);
    return systemPosition + randomDir * radius;
}

glm::vec3 ParticleSystem::GetEmissionVelocity(const EmissionProperties& emission) {
    glm::vec3 baseDir = emission.velocityDirection;
    // Add variation (Simplified)
    glm::vec3 randomDir = GetRandomDirection();
    glm::vec3 finalDir = glm::normalize(baseDir + randomDir * emission.velocityVariation);
    return finalDir * emission.velocityMagnitude;
}

float ParticleSystem::GetRandomFloat(float min, float max) {
    return std::uniform_real_distribution<float>(min, max)(m_randomEngine);
}

glm::vec3 ParticleSystem::GetRandomDirection() {
    float theta = GetRandomFloat(0.0f, 6.28318f);
    float z = GetRandomFloat(-1.0f, 1.0f);
    float temp = sqrt(1.0f - z * z);
    return glm::vec3(temp * cos(theta), temp * sin(theta), z);
}

// Unused Placeholders
void ParticleSystem::UpdateGPUSimulation(Core::Entity entity, ParticleSystemComponent& system, float deltaTime) {}
void ParticleSystem::DrawParticleSystem(const ParticleSystemComponent& system, const glm::vec3& systemPosition) {}

// Force Field Stubs
void ParticleSystem::RegisterForceField(Core::Entity entity) {}
void ParticleSystem::UnregisterForceField(Core::Entity entity) {}
Core::Entity ParticleSystem::CreateParticleSystem(const ParticleEffectPreset& preset) {
    auto& coordinator = Core::Coordinator::GetInstance();
    Core::Entity entity = coordinator.CreateEntity();
    
    ParticleSystemComponent psc;
    psc.maxParticles = 500; // Default limit
    psc.Initialize();
    
    psc.emission = preset.emission;
    psc.rendering = preset.rendering;
    psc.physics = preset.physics;
    psc.animation = preset.animation;
    
    // Copy name for debugging
    // psc.name = preset.name;
    
    coordinator.AddComponent(entity, psc);
    
    Rendering::TransformComponent transform;
    transform.scale = glm::vec3(1.0f);
    coordinator.AddComponent(entity, transform);
    
    return entity;
}

Core::Entity ParticleSystem::CreateEffectFromPreset(const std::string& presetName, const glm::vec3& position) {
    if (m_effectPresets.find(presetName) == m_effectPresets.end()) {
        std::cerr << "[ParticleSystem] Preset not found: " << presetName << std::endl;
        return 0; // Invalid entity
    }
    
    Core::Entity entity = CreateParticleSystem(m_effectPresets[presetName]);
    auto& coordinator = Core::Coordinator::GetInstance();
    auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
    transform.position = position;
    
    return entity;
}
void ParticleSystem::SetCollisionSystem(std::shared_ptr<Physics::PhysicsSystem> physicsSystem) { m_physicsSystem = physicsSystem; }
void ParticleSystem::DrawDebugInfo() {}
void ParticleSystem::RegisterParticleSpawnCallback(ParticleSpawnCallback callback) {}
void ParticleSystem::RegisterParticleDeathCallback(ParticleDeathCallback callback) {}
void ParticleSystem::RegisterParticleCollisionCallback(ParticleCollisionCallback callback) {}

} // namespace CudaGame
