#include "ParticleSystem.cuh"
#include <glad/glad.h>
#include <iostream>
#include <random>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        } \
    } while(0)

// CUDA kernels
__global__ void updateParticlesKernel(Particle* particles, int numParticles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numParticles) return;
    
    Particle& p = particles[idx];
    
    if (p.life <= 0.0f) return;
    
    // Update position
    p.position.x += p.velocity.x * deltaTime;
    p.position.y += p.velocity.y * deltaTime;
    
    // Apply gravity
    p.velocity.y -= 9.8f * deltaTime;
    
    // Apply air resistance
    p.velocity.x *= 0.999f;
    p.velocity.y *= 0.999f;
    
    // Update life
    p.life -= deltaTime;
    
    // Fade color based on life
    float lifeFactor = p.life / 5.0f; // Assuming max life is 5 seconds
    p.color.x = fminf(1.0f, lifeFactor);
    p.color.y = fminf(1.0f, lifeFactor * 0.8f);
    p.color.z = fminf(1.0f, lifeFactor * 0.6f);
    
    // Boundary checking
    if (p.position.x < -1.0f || p.position.x > 1.0f) {
        p.velocity.x *= -0.8f;
        p.position.x = fmaxf(-1.0f, fminf(1.0f, p.position.x));
    }
    if (p.position.y < -1.0f) {
        p.velocity.y *= -0.8f;
        p.position.y = -1.0f;
    }
}

__global__ void initializeParticlesKernel(Particle* particles, int numParticles, float2 position) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numParticles) return;
    
    Particle& p = particles[idx];
    
    // Use thread index for pseudo-random generation
    unsigned int seed = idx + blockIdx.x * 1234 + threadIdx.x * 5678;
    
    // Simple linear congruential generator
    seed = (seed * 1664525u + 1013904223u);
    float randX = ((seed >> 16) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
    
    seed = (seed * 1664525u + 1013904223u);
    float randY = ((seed >> 16) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
    
    seed = (seed * 1664525u + 1013904223u);
    float speed = ((seed >> 16) & 0xFFFF) / 65535.0f * 10.0f + 2.0f;
    
    p.position = position;
    p.velocity.x = randX * speed;
    p.velocity.y = randY * speed;
    p.color = make_float3(1.0f, 0.8f, 0.6f);
    p.life = 5.0f;
    p.size = 2.0f;
}

// ParticleSystem implementation
ParticleSystem::ParticleSystem(int maxParticles) 
    : m_maxParticles(maxParticles), m_activeParticles(0), d_particles(nullptr), m_VAO(0), m_VBO(0) {
    m_particles.resize(maxParticles);
}

ParticleSystem::~ParticleSystem() {
    cleanup();
}

void ParticleSystem::initialize() {
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_particles, m_maxParticles * sizeof(Particle)));
    
    setupOpenGL();
    
    std::cout << "ParticleSystem initialized with " << m_maxParticles << " max particles" << std::endl;
}

void ParticleSystem::setupOpenGL() {
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    
    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    
    // Allocate buffer for particle data
    glBufferData(GL_ARRAY_BUFFER, m_maxParticles * sizeof(Particle), nullptr, GL_DYNAMIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, position));
    glEnableVertexAttribArray(0);
    
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, color));
    glEnableVertexAttribArray(1);
    
    // Size attribute
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, size));
    glEnableVertexAttribArray(2);
    
    glBindVertexArray(0);
}

void ParticleSystem::update(float deltaTime) {
    if (m_activeParticles == 0) return;
    
    updateParticlesCUDA(deltaTime);
    
    // Copy data back from GPU
    CUDA_CHECK(cudaMemcpy(m_particles.data(), d_particles, m_activeParticles * sizeof(Particle), cudaMemcpyDeviceToHost));
    
    // Remove dead particles
    int writeIndex = 0;
    for (int readIndex = 0; readIndex < m_activeParticles; readIndex++) {
        if (m_particles[readIndex].life > 0.0f) {
            if (writeIndex != readIndex) {
                m_particles[writeIndex] = m_particles[readIndex];
            }
            writeIndex++;
        }
    }
    m_activeParticles = writeIndex;
}

void ParticleSystem::updateParticlesCUDA(float deltaTime) {
    if (m_activeParticles == 0) return;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (m_activeParticles + threadsPerBlock - 1) / threadsPerBlock;
    
    updateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, m_activeParticles, deltaTime);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ParticleSystem::render() {
    if (m_activeParticles == 0) return;
    
    // Update VBO with current particle data
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_activeParticles * sizeof(Particle), m_particles.data());
    
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_POINTS, 0, m_activeParticles);
    glBindVertexArray(0);
}

void ParticleSystem::addParticles(float2 position, int count) {
    if (m_activeParticles + count > m_maxParticles) {
        count = m_maxParticles - m_activeParticles;
    }
    
    if (count <= 0) return;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    // Initialize new particles on GPU
    initializeParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles + m_activeParticles, count, position);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    m_activeParticles += count;
}

void ParticleSystem::cleanup() {
    if (d_particles) {
        cudaFree(d_particles);
        d_particles = nullptr;
    }
    
    if (m_VBO) {
        glDeleteBuffers(1, &m_VBO);
        m_VBO = 0;
    }
    
    if (m_VAO) {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
}
