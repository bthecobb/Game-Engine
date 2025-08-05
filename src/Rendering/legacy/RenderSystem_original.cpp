#include "Rendering/RenderSystem.h"
#include "Rendering/ShaderProgram.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/Mesh.h"
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>

namespace CudaGame {
namespace Rendering {

RenderSystem::RenderSystem() : m_mainCamera(nullptr) {}

RenderSystem::~RenderSystem() {}

bool RenderSystem::Initialize() {
    std::cout << "[RenderSystem] Initializing deferred rendering pipeline..." << std::endl;

    // Load shaders
    m_geometryPassShader = std::make_shared<ShaderProgram>();
    if (!m_geometryPassShader->LoadFromFiles(ASSET_DIR "/shaders/deferred_geometry.vert", ASSET_DIR "/shaders/deferred_geometry.frag")) {
        std::cerr << "[RenderSystem] Failed to load geometry shader." << std::endl;
        return false;
    }

    m_lightingPassShader = std::make_shared<ShaderProgram>();
    if (!m_lightingPassShader->LoadFromFiles(ASSET_DIR "/shaders/deferred_lighting.vert", ASSET_DIR "/shaders/deferred_lighting.frag")) {
        std::cerr << "[RenderSystem] Failed to load lighting shader." << std::endl;
        return false;
    }

    m_shadowShader = std::make_shared<ShaderProgram>();
    if (!m_shadowShader->LoadFromFiles(ASSET_DIR "/shaders/shadow_mapping.vert", ASSET_DIR "/shaders/shadow_mapping.frag")) {
        std::cerr << "[RenderSystem] Failed to load shadow shader." << std::endl;
        return false;
    }
    
    // Create dummy white texture for unbound material textures
    glGenTextures(1, &m_dummyTexture);
    glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
    unsigned char whitePixel[4] = {255, 255, 255, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, whitePixel);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Create G-buffer
    m_gBuffer = std::make_shared<Framebuffer>();
    if (!m_gBuffer->Initialize(1280, 720)) {
        std::cerr << "[RenderSystem] Failed to initialize G-buffer." << std::endl;
        return false;
    }

    // Create shadow map framebuffer
    m_shadowMapFBO = std::make_shared<Framebuffer>();
    if (!m_shadowMapFBO->Initialize(2048, 2048)) {
        std::cerr << "[RenderSystem] Failed to initialize shadow map." << std::endl;
        return false;
    }

    // Create fullscreen quad for lighting pass
    CreateFullscreenQuad();
    
    // Create simple cube for basic rendering
    CreateSimpleCube();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    
    // Initialize light space matrix
    m_lightSpaceMatrix = glm::mat4(1.0f);
    
    std::cout << "[RenderSystem] Deferred rendering pipeline initialized successfully." << std::endl;
    return true;
}

void RenderSystem::Shutdown() {
    std::cout << "[RenderSystem] Shutting down..." << std::endl;
}

void RenderSystem::Update(float deltaTime) {
    // Debug: Log entity count
    static int frameCount = 0;
    if (frameCount % 60 == 0) {
        std::cout << "[RenderSystem] Update - Processing " << mEntities.size() << " entities" << std::endl;
        Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
        int renderableEntities = 0;
        for (auto const& entity : mEntities) {
            if (coordinator.HasComponent<Rendering::TransformComponent>(entity) &&
                coordinator.HasComponent<Rendering::MeshComponent>(entity)) {
                renderableEntities++;
            }
        }
        std::cout << "[RenderSystem] Renderable entities: " << renderableEntities << std::endl;
    }
    frameCount++;
    
    // Call Render to actually perform rendering
    Render();
}

void RenderSystem::Render() {
    if (!m_mainCamera) return;

    // 1. Shadow Pass
    ShadowPass();

    // 2. Geometry Pass
    GeometryPass();

    // 3. Lighting Pass
    LightingPass();
}

void RenderSystem::ShadowPass() {
    m_shadowShader->Use();
    m_shadowMapFBO->Bind();
    glViewport(0, 0, 2048, 2048);  // Set viewport to shadow map size
    glClear(GL_DEPTH_BUFFER_BIT);

    // Set up light space matrix for directional light
    glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
    glm::vec3 lightPos = -lightDir * 20.0f; // Position light far away for directional effect
    glm::mat4 lightProjection = glm::ortho(-25.0f, 25.0f, -25.0f, 25.0f, 1.0f, 50.0f);
    glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    m_lightSpaceMatrix = lightProjection * lightView;
    
    m_shadowShader->SetMat4("lightSpaceMatrix", m_lightSpaceMatrix);

    // Render scene from light's perspective
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<Rendering::TransformComponent>(entity) &&
            coordinator.HasComponent<Rendering::MeshComponent>(entity)) {

            auto const& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
            auto const& meshComponent = coordinator.GetComponent<Rendering::MeshComponent>(entity);

            // Set model matrix for shadow pass
            glm::mat4 modelMatrix = transform.getMatrix();
            m_shadowShader->SetMat4("model", modelMatrix);
            
            // Use simple cube for player_cube, Model class for everything else
            if (meshComponent.modelPath == "player_cube") {
                RenderSimpleCube();
            } else {
                Model model(meshComponent.modelPath);
                model.Draw(*m_shadowShader);
            }
        }
    }

    m_shadowMapFBO->Unbind();
}

void RenderSystem::GeometryPass() {
    if (!m_geometryPassShader) {
        std::cerr << "[RenderSystem] GeometryPass: Shader not loaded, skipping" << std::endl;
        return;
    }
    
    m_gBuffer->Bind();
    glViewport(0, 0, 1280, 720);  // Set viewport to G-buffer size
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    m_geometryPassShader->Use();
    m_geometryPassShader->SetMat4("projection", m_mainCamera->GetProjectionMatrix());
    m_geometryPassShader->SetMat4("view", m_mainCamera->GetViewMatrix());

    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<Rendering::TransformComponent>(entity) && 
            coordinator.HasComponent<Rendering::MeshComponent>(entity)) {
            
            auto const& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
            auto const& meshComponent = coordinator.GetComponent<Rendering::MeshComponent>(entity);

            glm::mat4 modelMatrix = transform.getMatrix();
            m_geometryPassShader->SetMat4("model", modelMatrix);
            
            // Set additional required uniforms
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(modelMatrix)));
            m_geometryPassShader->SetMat3("normalMatrix", normalMatrix);
            
            // Set the light space matrix from shadow pass
            m_geometryPassShader->SetMat4("lightSpaceMatrix", m_lightSpaceMatrix);
            
            // Set material properties if available
            if (coordinator.HasComponent<Rendering::MaterialComponent>(entity)) {
                auto const& material = coordinator.GetComponent<Rendering::MaterialComponent>(entity);
                m_geometryPassShader->SetVec3("albedo", material.albedo);
                m_geometryPassShader->SetFloat("metallic", material.metallic);
                m_geometryPassShader->SetFloat("roughness", material.roughness);
                m_geometryPassShader->SetFloat("ao", material.ao);
            } else {
                // Default material
                m_geometryPassShader->SetVec3("albedo", glm::vec3(0.7f, 0.7f, 0.7f));
                m_geometryPassShader->SetFloat("metallic", 0.0f);
                m_geometryPassShader->SetFloat("roughness", 0.5f);
                m_geometryPassShader->SetFloat("ao", 1.0f);
            }
            
            // Bind dummy textures for material maps
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
            m_geometryPassShader->SetInt("albedoMap", 0);
            
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
            m_geometryPassShader->SetInt("normalMap", 1);
            
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
            m_geometryPassShader->SetInt("metallicMap", 2);
            
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
            m_geometryPassShader->SetInt("roughnessMap", 3);
            
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, m_dummyTexture);
            m_geometryPassShader->SetInt("aoMap", 4);
            
            // Use simple cube for player_cube, Model class for everything else
            if (meshComponent.modelPath == "player_cube") {
                RenderSimpleCube();
            } else {
                Model model(meshComponent.modelPath);
                model.Draw(*m_geometryPassShader);
            }
        }
    }

    m_gBuffer->Unbind();
}

void RenderSystem::CreateSimpleCube() {
    // Cube with position, normal, texcoord, and tangent (matching deferred_geometry.vert)
    float cubeVertices[] = {
        // positions          // normals           // texture coords // tangents
        // Back face
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        // Front face
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 1.0f, -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
        // Left face
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,  0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        // Right face
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,  0.0f, 0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,  0.0f, 0.0f, -1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, -1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, -1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 0.0f, -1.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,  0.0f, 0.0f, -1.0f,
        // Bottom face
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f,
        // Top face
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f
    };
    
    glGenVertexArrays(1, &m_cubeVAO);
    glGenBuffers(1, &m_cubeVBO);

    glBindBuffer(GL_ARRAY_BUFFER, m_cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

    glBindVertexArray(m_cubeVAO);
    // Position attribute (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Normal attribute (location = 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Texture coordinate attribute (location = 2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    // Tangent attribute (location = 3)
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);
}

void RenderSystem::RenderSimpleCube() {
    glBindVertexArray(m_cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

void RenderSystem::LightingPass() {
    // Bind default framebuffer for final output
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 1280, 720);  // Reset viewport to window size
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    
    m_lightingPassShader->Use();

    // Bind G-buffer textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_gBuffer->GetColorAttachment(0));
    m_lightingPassShader->SetInt("gPosition", 0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_gBuffer->GetColorAttachment(1));
    m_lightingPassShader->SetInt("gNormal", 1);
    
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_gBuffer->GetColorAttachment(2));
    m_lightingPassShader->SetInt("gAlbedoSpec", 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, m_gBuffer->GetColorAttachment(3));
    m_lightingPassShader->SetInt("gMetallicRoughnessAO", 3);

    // Bind shadow map
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, m_shadowMapFBO->GetDepthAttachment());
    m_lightingPassShader->SetInt("shadowMap", 4);

    // Set light uniforms - add a default light
    m_lightingPassShader->SetVec3("light.position", glm::vec3(10.0f, 10.0f, 10.0f));
    m_lightingPassShader->SetVec3("light.direction", glm::vec3(-1.0f, -1.0f, -1.0f));
    m_lightingPassShader->SetVec3("light.color", glm::vec3(1.0f, 1.0f, 1.0f));
    m_lightingPassShader->SetFloat("light.intensity", 1.0f);
    m_lightingPassShader->SetFloat("light.radius", 100.0f);
    m_lightingPassShader->SetInt("light.type", 0); // directional light
    
    // Set view position
    m_lightingPassShader->SetVec3("viewPos", m_mainCamera->GetPosition());
    
    // Set light space matrix from shadow pass
    m_lightingPassShader->SetMat4("lightSpaceMatrix", m_lightSpaceMatrix);

    // Disable depth test for fullscreen quad
    glDisable(GL_DEPTH_TEST);
    
    // Draw fullscreen quad
    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    
    // Re-enable depth test
    glEnable(GL_DEPTH_TEST);
}

void RenderSystem::CreateFullscreenQuad() {
    float quadVertices[] = {
        // positions        // texture Coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };
    // screen quad VAO
    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);
    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
}

} // namespace Rendering
} // namespace CudaGame

