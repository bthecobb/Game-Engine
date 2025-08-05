#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>
#include <vector>
#include <string>

namespace CudaGame {
namespace Particles {

// Individual particle data structure
struct Particle {
    glm::vec3 position{0.0f};
    glm::vec3 velocity{0.0f};
    glm::vec3 acceleration{0.0f};
    
    glm::vec4 color{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 startColor{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 endColor{1.0f, 1.0f, 1.0f, 0.0f};
    
    float size = 1.0f;
    float startSize = 1.0f;
    float endSize = 0.0f;
    
    float lifetime = 1.0f;
    float age = 0.0f;
    float normalizedAge = 0.0f; // age / lifetime (0.0 to 1.0)
    
    float rotation = 0.0f;
    float angularVelocity = 0.0f;
    
    bool isActive = false;
    
    // Custom data for specialized effects
    glm::vec3 customData{0.0f};
    int customInt = 0;
};

// Particle emission properties
struct EmissionProperties {
    float emissionRate = 10.0f;        // Particles per second
    int burstCount = 0;                // Particles to emit in burst
    float burstTime = 0.0f;            // Time between bursts
    bool continuous = true;            // Continuous emission vs burst
    
    // Emission shape
    enum class EmissionShape {
        POINT,
        CIRCLE,
        SPHERE,
        BOX,
        CONE,
        MESH
    } shape = EmissionShape::POINT;
    
    glm::vec3 emissionBoxSize{1.0f};
    float emissionRadius = 1.0f;
    float emissionConeAngle = 45.0f;
    
    // Velocity properties
    glm::vec3 velocityDirection{0.0f, 1.0f, 0.0f};
    float velocityMagnitude = 5.0f;
    float velocityVariation = 0.2f;    // Random variation factor
    
    // Lifetime properties
    float particleLifetime = 2.0f;
    float lifetimeVariation = 0.5f;
};

// Particle rendering properties
struct RenderProperties {
    enum class RenderMode {
        BILLBOARD,          // Always face camera
        STRETCHED_BILLBOARD, // Stretch along velocity
        MESH,              // 3D mesh particles
        TRAIL              // Trail renderer
    } renderMode = RenderMode::BILLBOARD;
    
    enum class BlendMode {
        ALPHA,
        ADDITIVE,
        MULTIPLY,
        SUBTRACT
    } blendMode = BlendMode::ALPHA;
    
    std::string texturePath;
    uint32_t textureId = 0;
    
    // Billboard properties
    bool velocityStretching = false;
    float stretchFactor = 1.0f;
    
    // Trail properties
    int trailSegments = 10;
    float trailWidth = 0.1f;
    
    // Lighting properties
    bool receiveLighting = false;
    bool castShadows = false;
    float emissionIntensity = 0.0f;   // For emissive particles
};

// Physics properties for particles
struct PhysicsProperties {
    glm::vec3 gravity{0.0f, -9.81f, 0.0f};
    float drag = 0.0f;                // Air resistance
    float bounce = 0.0f;              // Collision restitution
    
    // Force fields
    bool affectedByWind = false;
    bool affectedByMagnetism = false;
    bool affectedByTurbulence = false;
    
    // Collision properties
    bool collisionEnabled = false;
    float collisionRadius = 0.1f;
    uint32_t collisionMask = 0xFFFFFFFF;
    
    // Size/lifetime modifiers
    bool sizeOverLifetime = true;
    bool colorOverLifetime = true;
    bool velocityOverLifetime = false;
};

// Animation properties for advanced effects
struct AnimationProperties {
    // Texture animation
    bool animateTexture = false;
    int textureFrames = 1;
    int textureColumns = 1;
    int textureRows = 1;
    float animationSpeed = 1.0f;
    bool loopAnimation = true;
    
    // Rotation animation
    bool animateRotation = false;
    float rotationSpeed = 0.0f;
    float rotationVariation = 0.0f;
    
    // Noise-based animation
    bool useNoise = false;
    float noiseStrength = 1.0f;
    float noiseFrequency = 1.0f;
    glm::vec3 noiseOffset{0.0f};
};

// Main particle system component
struct ParticleSystemComponent {
    std::vector<Particle> particles;
    int maxParticles = 1000;
    int activeParticles = 0;
    
    // System properties
    bool isPlaying = true;
    bool isLooping = true;
    float systemLifetime = 5.0f;
    float systemAge = 0.0f;
    
    // Emission control
    float emissionTimer = 0.0f;
    float burstTimer = 0.0f;
    int particlesToEmit = 0;
    
    // Performance settings
    bool useGPUSimulation = false;     // Enable CUDA acceleration
    bool useLOD = true;                // Level of detail based on distance
    float lodDistance = 50.0f;
    float cullingDistance = 100.0f;
    
    // Pooling for performance
    std::vector<int> freeParticleIndices;
    
    // Component references
    EmissionProperties emission;
    RenderProperties rendering;
    PhysicsProperties physics;
    AnimationProperties animation;
    
    // Statistics
    struct Stats {
        int particlesEmittedThisFrame = 0;
        int particlesActiveThisFrame = 0;
        float averageParticleLifetime = 0.0f;
        float systemPerformanceMs = 0.0f;
    } stats;
    
    // Initialization
    void Initialize() {
        particles.resize(maxParticles);
        freeParticleIndices.reserve(maxParticles);
        
        // Initialize free particle pool
        for (int i = 0; i < maxParticles; ++i) {
            freeParticleIndices.push_back(i);
        }
    }
    
    // Get next available particle
    Particle* GetFreeParticle() {
        if (freeParticleIndices.empty()) {
            return nullptr;
        }
        
        int index = freeParticleIndices.back();
        freeParticleIndices.pop_back();
        
        particles[index].isActive = true;
        activeParticles++;
        
        return &particles[index];
    }
    
    // Return particle to pool
    void ReturnParticle(int index) {
        if (index >= 0 && index < maxParticles && particles[index].isActive) {
            particles[index].isActive = false;
            freeParticleIndices.push_back(index);
            activeParticles--;
        }
    }
};

// Particle force field component for environmental effects
struct ParticleForceFieldComponent {
    enum class ForceType {
        WIND,
        VORTEX,
        MAGNET,
        TURBULENCE,
        EXPLOSION,
        GRAVITY_WELL
    } type = ForceType::WIND;
    
    glm::vec3 direction{0.0f, 0.0f, 1.0f};
    float strength = 1.0f;
    float radius = 10.0f;
    float falloff = 1.0f;              // How quickly force diminishes with distance
    
    // Wind-specific
    float windVariation = 0.1f;
    glm::vec3 windNoise{0.0f};
    
    // Vortex-specific
    glm::vec3 vortexAxis{0.0f, 1.0f, 0.0f};
    float vortexSpeed = 1.0f;
    
    // Explosion-specific
    bool isOneShot = false;
    float explosionForce = 10.0f;
    
    bool isActive = true;
};

} // namespace Particles
} // namespace CudaGame
