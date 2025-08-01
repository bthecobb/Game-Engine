#include "Rendering/RenderSystem.h"
#include "Rendering/ShaderProgram.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/Mesh.h"
#include "../../include/Player.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/common.hpp>
#include <iostream>

namespace CudaGame {
namespace Rendering {

RenderSystem::RenderSystem() : m_mainCamera(nullptr) {}

RenderSystem::~RenderSystem() {}

bool RenderSystem::Initialize() {
    std::cout << "[RenderSystem] Initializing deferred rendering pipeline..." << std::endl;
    
    // Log startup GL information
    std::cout << "{ \"startup\": { "
              << "\"GL_VERSION\": \"" << glGetString(GL_VERSION) << "\", "
              << "\"GL_RENDERER\": \"" << glGetString(GL_RENDERER) << "\" } }" << std::endl;

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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    // Get actual window dimensions
    int width, height;
    GLFWwindow* window = glfwGetCurrentContext();
    if (window) {
        glfwGetFramebufferSize(window, &width, &height);
    } else {
        // Default to 1920x1080 if no context
        width = 1920;
        height = 1080;
    }
    
    std::cout << "[RenderSystem] Initializing G-buffer with window size: " << width << "x" << height << std::endl;
    
    // Create G-buffer with actual window dimensions
    m_gBuffer = std::make_shared<Framebuffer>();
    if (!m_gBuffer->Initialize(width, height)) {
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
    Render(nullptr);
}

void RenderSystem::Render(const Player* player) {
    if (!m_mainCamera) return;

    // Start frame logging
    LogFrameStart();

    // Get actual window dimensions
    int width, height;
    GLFWwindow* window = glfwGetCurrentContext();
    if (window) {
        glfwGetFramebufferSize(window, &width, &height);
    } else {
        width = 1920;
        height = 1080;
    }

    // Disable scissor and stencil tests to rule them out
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    
    LogGLError("BeforeRenderPasses");

    // 1. Shadow Pass
    ShadowPass();
    LogGLError("AfterShadowPass");

    // 2. Geometry Pass
    GeometryPass();
    LogGLError("AfterGeometryPass"); 

    // 3. Lighting Pass
    LightingPass();
    LogGLError("AfterLightingPass");
    
    // 4. Render Character (if provided)
    if (player && m_mainCamera) {
        RenderSimpleCharacter(player, m_mainCamera->GetViewMatrix(), m_mainCamera->GetProjectionMatrix());
        LogGLError("AfterCharacterRender");
    }
    
    DumpGLState("EndOfFrame");
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
    // Get actual window dimensions
    int windowWidth, windowHeight;
    GLFWwindow* window = glfwGetCurrentContext();
    if (window) {
        glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
    } else {
        windowWidth = 1920;
        windowHeight = 1080;
    }
    if (!m_geometryPassShader) {
        std::cerr << "[RenderSystem] GeometryPass: Shader not loaded, skipping" << std::endl;
        return;
    }
    
    // Log pass start
    LogPassStart("GeometryPass", 1, windowWidth, windowHeight); // Use placeholder FBO ID
    
    static int frameCount = 0;
    static int entityCount = 0;
    frameCount++;
    if (frameCount % 60 == 0) {  // Log every 60 frames
        std::cout << "[RenderSystem] Geometry Pass: Rendering " << entityCount << " entities" << std::endl;
        entityCount = 0;
    }
    
    m_gBuffer->Bind();
    glViewport(0, 0, windowWidth, windowHeight);  // Use actual window dimensions
    
    // Ensure we're writing to all color attachments (AFTER binding framebuffer)
    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
    glDrawBuffers(4, drawBuffers);
    
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
            
            // Bind dummy textures to prevent shader errors
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
            
            // Use simple cube for player_cube
            if (meshComponent.modelPath == "player_cube") {
                LogDrawCall("GeometryPass", 1, m_cubeVAO, "GL_TRIANGLES", 36); // Use placeholder shader ID
                RenderSimpleCube();
                entityCount++;
                m_drawCallCount++;
                m_triangleCount += 12; // 36 vertices / 3 = 12 triangles
            } else {
                // Log model draw (approximate triangle count)
                LogDrawCall("GeometryPass", 1, 0, "MODEL_DRAW", -1); // Use placeholder shader ID
                Model model(meshComponent.modelPath);
                model.Draw(*m_geometryPassShader);
                entityCount++;
                m_drawCallCount++;
                m_triangleCount += 100; // Rough estimate for model
            }
        }
    }

    LogPassEnd("GeometryPass", m_drawCallCount, m_triangleCount);
    // Unbind G-buffer by binding default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RenderSystem::LightingPass() {
    // Get actual window dimensions
    int windowWidth, windowHeight;
    GLFWwindow* window = glfwGetCurrentContext();
    if (window) {
        glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
    } else {
        windowWidth = 1920;
        windowHeight = 1080;
    }
    if (!m_lightingPassShader) {
        std::cerr << "[RenderSystem] LightingPass: Shader not loaded, skipping" << std::endl;
        return;
    }
    
    LogPassStart("LightingPass", 0, windowWidth, windowHeight); // Log default framebuffer
    
    // Render to default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, windowWidth, windowHeight);  // Use actual window dimensions
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    m_lightingPassShader->Use();
    
    // Bind G-buffer textures
    glActiveTexture(GL_TEXTURE0);
    uint32_t positionTex = m_gBuffer->GetColorAttachment(0);
    glBindTexture(GL_TEXTURE_2D, positionTex);
    LogTextureBinding("LightingPass", 0, positionTex, "Position");
    m_lightingPassShader->SetInt("gPosition", 0);
    
    glActiveTexture(GL_TEXTURE1);
    uint32_t normalTex = m_gBuffer->GetColorAttachment(1);
    glBindTexture(GL_TEXTURE_2D, normalTex);
    LogTextureBinding("LightingPass", 1, normalTex, "Normal");
    m_lightingPassShader->SetInt("gNormal", 1);
    
    glActiveTexture(GL_TEXTURE2);
    uint32_t albedoTex = m_gBuffer->GetColorAttachment(2);
    glBindTexture(GL_TEXTURE_2D, albedoTex);
    LogTextureBinding("LightingPass", 2, albedoTex, "Albedo");
    m_lightingPassShader->SetInt("gAlbedoSpec", 2);
    
    glActiveTexture(GL_TEXTURE3);
    uint32_t metallicTex = m_gBuffer->GetColorAttachment(3);
    glBindTexture(GL_TEXTURE_2D, metallicTex);
    LogTextureBinding("LightingPass", 3, metallicTex, "MetallicRoughnessAO");
    m_lightingPassShader->SetInt("gMetallicRoughness", 3);
    
    // Bind shadow map (if available)
    glActiveTexture(GL_TEXTURE4);
    uint32_t shadowTex = m_shadowMapFBO ? m_shadowMapFBO->GetDepthAttachment() : 0;
    if (shadowTex > 0) {
        glBindTexture(GL_TEXTURE_2D, shadowTex);
        LogTextureBinding("LightingPass", 4, shadowTex, "ShadowMap");
    } else {
        // Bind a default texture or use 0
        glBindTexture(GL_TEXTURE_2D, 0);
        LogTextureBinding("LightingPass", 4, 0, "ShadowMap_Default");
    }
    m_lightingPassShader->SetInt("shadowMap", 4);
    
    // Set light and camera uniforms
    // Use actual camera position
    if (m_mainCamera) {
        m_lightingPassShader->SetVec3("viewPos", m_mainCamera->GetPosition());
    } else {
        m_lightingPassShader->SetVec3("viewPos", glm::vec3(0.0f, 0.0f, 3.0f));
    }
    
    // Set a directional light
    m_lightingPassShader->SetVec3("light.direction", glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f)));
    m_lightingPassShader->SetVec3("light.color", glm::vec3(1.0f, 1.0f, 1.0f));
    m_lightingPassShader->SetFloat("light.intensity", 1.0f);
    m_lightingPassShader->SetInt("light.type", 0); // 0 = directional light
    // For directional lights, position represents the light direction at infinite distance
    glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
    m_lightingPassShader->SetVec3("light.position", -lightDir * 1000.0f);
    
    // Set the light space matrix for shadow mapping
    m_lightingPassShader->SetMat4("lightSpaceMatrix", m_lightSpaceMatrix);
    
    // Set debug mode uniform
    m_lightingPassShader->SetInt("debugMode", m_debugMode);
    
    // Set depth scale for interactive debug visualization
    m_lightingPassShader->SetFloat("depthScale", m_depthScale);
    
    // Render fullscreen quad
    RenderFullscreenQuad();

    // Log final draw call of the lighting pass
    LogDrawCall("LightingPass", 1, m_quadVAO, "GL_TRIANGLE_STRIP", 4);

    LogPassEnd("LightingPass", 1, 2); // Typically 1 draw call & 2 triangles for strip
}

void RenderSystem::CreateFullscreenQuad() {
    float quadVertices[] = {
        // positions        // texCoords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };
    
    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);
    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindVertexArray(0);
}

void RenderSystem::RenderFullscreenQuad() {
    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}


void RenderSystem::CreateSimpleCube() {
    // Cube vertices with position, normal, texture coordinates, and tangent
    // Format: pos(3), normal(3), texCoord(2), tangent(3) = 11 floats per vertex
    float vertices[] = {
        // Front face
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        
        // Back face
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f, -1.0f, 0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f, -1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f, -1.0f, 0.0f, 0.0f,
        
        // Left face
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,  0.0f, 0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 0.0f, -1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, -1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, -1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,  0.0f, 0.0f, -1.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,  0.0f, 0.0f, -1.0f,
        
        // Right face
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,  0.0f, 0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,  0.0f, 0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,  0.0f, 0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        
        // Bottom face
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f,
        
        // Top face
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,  1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f
    };
    
    glGenVertexArrays(1, &m_cubeVAO);
    glGenBuffers(1, &m_cubeVBO);
    
    glBindVertexArray(m_cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
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

void RenderSystem::CycleDebugMode() {
    m_debugMode = (m_debugMode + 1) % 5;  // 0-4: normal, position, normal, albedo, metallic/roughness
    const char* debugModeNames[] = {"Normal Lighting", "Position Buffer", "Normal Buffer", "Albedo Buffer", "Metallic/Roughness/AO Buffer"};
    std::cout << "[RenderSystem] Debug mode switched to: " << debugModeNames[m_debugMode] << std::endl;
}


void RenderSystem::RenderSimpleCharacter(const Player* player, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix) {
    if (!player) return;

    // For now, render player as a simple colored cube at player position
    // This is a temporary implementation until we integrate the full character rendering
    
    // Use the geometry pass shader for simple rendering
    if (!m_geometryPassShader) return;
    
    m_geometryPassShader->Use();
    
    // Set matrices
    m_geometryPassShader->SetMat4("view", viewMatrix);
    m_geometryPassShader->SetMat4("projection", projectionMatrix);
    
    // Create model matrix from player position
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    glm::vec3 playerPos = player->getPosition();
    modelMatrix = glm::translate(modelMatrix, playerPos);
    
    // Make the player cube slightly larger and different color
    modelMatrix = glm::scale(modelMatrix, glm::vec3(1.2f));
    
    m_geometryPassShader->SetMat4("model", modelMatrix);
    
    // Set normal matrix
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(modelMatrix)));
    m_geometryPassShader->SetMat3("normalMatrix", normalMatrix);
    
    // Set light space matrix
    m_geometryPassShader->SetMat4("lightSpaceMatrix", m_lightSpaceMatrix);
    
    // Bind dummy textures
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
    
    // Set player material - bright color to distinguish from other objects
    glm::vec3 playerColor = glm::vec3(0.2f, 0.8f, 0.3f); // Green for visibility
    m_geometryPassShader->SetVec3("albedo", playerColor);
    m_geometryPassShader->SetFloat("metallic", 0.2f);
    m_geometryPassShader->SetFloat("roughness", 0.8f);
    m_geometryPassShader->SetFloat("ao", 1.0f);
    
    // Render the cube
    RenderSimpleCube();
    
    // Log the character draw call
    LogDrawCall("CharacterRender", 1, m_cubeVAO, "GL_TRIANGLES", 36);
}

void RenderSystem::AdjustDepthScale(float multiplier) {
    m_depthScale *= multiplier;
    m_depthScale = glm::clamp(m_depthScale, 0.1f, 1000.0f);
    std::cout << "[Debug] Depth Scale: " << m_depthScale << std::endl;
}

// SetMainCamera is already defined in the header as an inline function

// Diagnostic logging methods
void RenderSystem::LogFrameStart() {
    m_frameID++;
    m_drawCallCount = 0;
    m_triangleCount = 0;
    std::cout << "{ \"frame\":" << m_frameID << " }" << std::endl;
}

void RenderSystem::LogPassStart(const std::string& passName, uint32_t fbo, int width, int height) {
    std::cout << "{ \"frame\":" << m_frameID 
              << ",\"pass\":\"" << passName << "_Start\""
              << ",\"FBO\":" << fbo
              << ",\"viewport\":\"" << width << "x" << height << "\" }" << std::endl;
}

void RenderSystem::LogPassEnd(const std::string& passName, int drawCalls, int triangles) {
    std::cout << "{ \"frame\":" << m_frameID 
              << ",\"pass\":\"" << passName << "_End\""
              << ",\"drawCalls\":" << drawCalls
              << ",\"triangles\":" << triangles << " }" << std::endl;
}

void RenderSystem::LogDrawCall(const std::string& pass, uint32_t shader, uint32_t vao, const std::string& primitive, int count) {
    std::cout << "{ \"frame\":" << m_frameID 
              << ",\"pass\":\"" << pass << "_Draw\""
              << ",\"shader\":" << shader
              << ",\"VAO\":" << vao
              << ",\"primitive\":\"" << primitive << "\""
              << ",\"count\":" << count;
    
    // Add GL state info
    GLboolean depthTest;
    glGetBooleanv(GL_DEPTH_TEST, &depthTest);
    GLint cullFaceMode;
    glGetIntegerv(GL_CULL_FACE_MODE, &cullFaceMode);
    
    std::cout << ",\"depthTest\":" << (depthTest ? "\"ON\"" : "\"OFF\"")
              << ",\"cullMode\":" << cullFaceMode << " }" << std::endl;
}

void RenderSystem::LogTextureBinding(const std::string& pass, int unit, uint32_t textureID, const std::string& format) {
    std::cout << "{ \"frame\":" << m_frameID 
              << ",\"pass\":\"" << pass << "_TextureBind\""
              << ",\"unit\":" << unit
              << ",\"tex\":" << textureID
              << ",\"format\":\"" << format << "\" }" << std::endl;
}

void RenderSystem::LogGLError(const std::string& location) {
    GLenum error;
    bool hasError = false;
    
    // Check for multiple errors in a loop
    while ((error = glGetError()) != GL_NO_ERROR) {
        hasError = true;
        const char* errorString = "UNKNOWN_ERROR";
        
        // Translate error code to human-readable string
        switch (error) {
            case GL_INVALID_ENUM: errorString = "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE: errorString = "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: errorString = "GL_INVALID_OPERATION"; break;
            case GL_OUT_OF_MEMORY: errorString = "GL_OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: errorString = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
            default: break;
        }
        
        // Log the error in JSON format
        std::cout << "{ \"frame\":" << m_frameID << ",\"GLError\":\"" << location 
                  << "\",\"code\":" << error << ",\"name\":\"" << errorString << "\" }" << std::endl;
    }
    
    // If any error occurred, dump full GL state for diagnosis
    if (hasError) {
        DumpGLState(location + "_ErrorState");
    }
}

void RenderSystem::DumpGLState(const std::string& location) {
    GLint currentFBO, currentProgram, viewport[4];
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFBO);
    glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgram);
    glGetIntegerv(GL_VIEWPORT, viewport);
    
    std::cout << "{ \"frame\":" << m_frameID 
              << ",\"GLState\":\"" << location << "\""
              << ",\"FBO\":" << currentFBO
              << ",\"program\":" << currentProgram
              << ",\"viewport\":[" << viewport[0] << "," << viewport[1] << "," << viewport[2] << "," << viewport[3] << "]";
    
    // If we have a bound framebuffer, check its completeness and attachments
    if (currentFBO != 0) {
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        const char* statusString = "UNKNOWN";
        
        switch (status) {
            case GL_FRAMEBUFFER_COMPLETE: statusString = "GL_FRAMEBUFFER_COMPLETE"; break;
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: statusString = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT"; break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: statusString = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"; break;
            case GL_FRAMEBUFFER_UNSUPPORTED: statusString = "GL_FRAMEBUFFER_UNSUPPORTED"; break;
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: statusString = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER"; break;
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: statusString = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER"; break;
            default: break;
        }
        
        std::cout << ",\"framebufferStatus\":\"" << statusString << "\"";
        
        // Check color attachments
        std::cout << ",\"attachments\":{";
        for (int i = 0; i < 4; i++) {
            GLint type, name;
            glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, 
                                                 GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
            if (type == GL_TEXTURE) {
                glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, 
                                                     GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
                if (i > 0) std::cout << ",";
                std::cout << "\"COLOR" << i << "\":" << name;
            }
        }
        
        // Check depth attachment
        GLint depthType, depthName;
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
                                             GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &depthType);
        if (depthType == GL_TEXTURE) {
            glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
                                                 GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &depthName);
            std::cout << ",\"DEPTH\":" << depthName;
        }
        
        std::cout << "}";
    }
    
    std::cout << " }" << std::endl;
}

} // namespace Rendering
} // namespace CudaGame
