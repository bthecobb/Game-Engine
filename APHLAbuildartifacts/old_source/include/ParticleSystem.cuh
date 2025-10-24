#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

struct Particle {
    float2 position;
    float2 velocity;
    float3 color;
    float life;
    float size;
};

class ParticleSystem {
public:
    ParticleSystem(int maxParticles);
    ~ParticleSystem();

    void initialize();
    void update(float deltaTime);
    void render();
    void addParticles(float2 position, int count);
    
    // CUDA functions
    void updateParticlesCUDA(float deltaTime);
    
    // Getters
    int getParticleCount() const { return m_activeParticles; }
    const std::vector<Particle>& getParticles() const { return m_particles; }

private:
    int m_maxParticles;
    int m_activeParticles;
    
    // CPU data
    std::vector<Particle> m_particles;
    
    // GPU data
    Particle* d_particles;
    
    // OpenGL buffers
    unsigned int m_VAO, m_VBO;
    
    void setupOpenGL();
    void cleanup();
};

// CUDA kernel declarations
__global__ void updateParticlesKernel(Particle* particles, int numParticles, float deltaTime);
__global__ void initializeParticlesKernel(Particle* particles, int numParticles, float2 position);
