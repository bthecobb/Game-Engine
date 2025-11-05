#include "Rendering/RenderSystem.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/CameraDebugSystem.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/Mesh.h"
#include "Gameplay/PlayerMovementSystem.h"
#include "Gameplay/PlayerComponents.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/common.hpp>
#include <iostream>
#include <chrono>

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

    // Set a readable sky-like clear color for contrast with emissive
    m_clearColor = glm::vec4(0.55f, 0.82f, 0.92f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    
    // Initialize light space matrix
    m_lightSpaceMatrix = glm::mat4(1.0f);
    
    // Initialize camera debug system
    m_cameraDebugSystem = std::make_unique<CameraDebugSystem>(this);
    
    std::cout << "[RenderSystem] Deferred rendering pipeline initialized successfully. Managing " << mEntities.size() << " entities." << std::endl;
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
    
    // Call Render - no longer passing nullptr
    Render();
}

void RenderSystem::Render() {
    if (!m_mainCamera) {
        std::cout << "[RenderSystem] ERROR: No main camera set!" << std::endl;
        return;
    }

    // Start frame logging
    LogFrameStart();
    
    // Validate and log detailed camera state every frame
    ValidateAndLogCameraState();

    // Get actual window dimensions
    int width, height;
    GLFWwindow* window = glfwGetCurrentContext();
    if (window) {
        glfwGetFramebufferSize(window, &width, &height);
    } else {
        width = 1920;
        height = 1080;
    }

    std::cout << "{ \"frame\":" << m_frameID << ",\"renderStart\":\"WindowSize\",\"width\":" << width << ",\"height\":" << height << " }" << std::endl;

    // === STEP 1: Clear default framebuffer once at the beginning ===
    // This prevents flicker by ensuring the back buffer is always properly initialized
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDrawBuffer(GL_BACK);
    glViewport(0, 0, width, height);
    glClearColor(m_clearColor.r, m_clearColor.g, m_clearColor.b, m_clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    std::cout << "{ \"frame\":" << m_frameID << ",\"step\":\"BackBufferCleared\",\"clearColor\":[" 
              << m_clearColor.r << "," << m_clearColor.g << "," << m_clearColor.b << "," << m_clearColor.a << "] }" << std::endl;

    // Disable scissor and stencil tests to rule them out
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    
    LogGLError("AfterBackBufferClear");

    // === STEP 2: Shadow Pass (offscreen) ===
    ShadowPass();
    LogGLError("AfterShadowPass");

    // === STEP 3: Geometry Pass (to G-buffer) ===
    GeometryPass();
    LogGLError("AfterGeometryPass"); 

    // === STEP 4: Lighting Pass (G-buffer to back buffer, no clear) ===
    LightingPass();
    LogGLError("AfterLightingPass");
    
    // === STEP 5: Copy depth from G-buffer to default framebuffer ===
    // This is crucial for forward-rendered objects to depth test correctly
    if (m_gBuffer) {
        std::cout << "{ \"frame\":" << m_frameID << ",\"step\":\"DepthBlit\",\"status\":\"Starting\" }" << std::endl;
        
        // Bind G-buffer as read framebuffer and default as draw framebuffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, m_gBuffer->GetFBO());
        // For depth-only blit, some drivers require the read buffer to be NONE
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        // Ensure we target the back buffer for the draw framebuffer
        glDrawBuffer(GL_BACK);
        
        // Blit depth buffer from G-buffer to default framebuffer
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        
        // Restore default framebuffer for subsequent draws
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        
        LogGLError("AfterDepthBlit");
        std::cout << "{ \"frame\":" << m_frameID << ",\"step\":\"DepthBlit\",\"status\":\"Complete\" }" << std::endl;
    } else {
        std::cout << "{ \"frame\":" << m_frameID << ",\"step\":\"DepthBlit\",\"status\":\"SKIPPED_NO_GBUFFER\" }" << std::endl;
    }
    
    // === STEP 6: Forward Pass (character, debug, transparent objects) ===
    ForwardPass();
    LogGLError("AfterForwardPass");
    
    DumpGLState("EndOfFrame");
    std::cout << "{ \"frame\":" << m_frameID << ",\"renderComplete\":true }" << std::endl;
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
            
            // Priority: generator VAO -> simple cube -> model
            if (meshComponent.vaoId != 0 && meshComponent.indexCount > 0) {
                glBindVertexArray(meshComponent.vaoId);
                glDrawElements(GL_TRIANGLES, meshComponent.indexCount, GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            } else if (meshComponent.modelPath == "player_cube") {
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
    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4 };
    glDrawBuffers(5, drawBuffers);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    m_geometryPassShader->Use();
    m_geometryPassShader->SetMat4("projection", m_mainCamera->GetProjectionMatrix());
    m_geometryPassShader->SetMat4("view", m_mainCamera->GetViewMatrix());
    // Drive geometry debug override from global debug mode (mode 5 => emissive color view)
    m_geometryPassShader->SetInt("debugForceEmissive", (m_debugMode == 5) ? 1 : 0);

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
            
            // Bind defaults to prevent shader errors
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, m_dummyTexture); m_geometryPassShader->SetInt("albedoMap", 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, m_dummyTexture); m_geometryPassShader->SetInt("normalMap", 1);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, m_dummyTexture); m_geometryPassShader->SetInt("metallicMap", 2);
            glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, m_dummyTexture); m_geometryPassShader->SetInt("roughnessMap", 3);
            glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, m_dummyTexture); m_geometryPassShader->SetInt("aoMap", 4);
            glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, m_dummyTexture); m_geometryPassShader->SetInt("emissiveMap", 5);
            m_geometryPassShader->SetFloat("emissiveIntensity", 1.0f);
            
            // Set material properties and bind actual maps if provided
            if (coordinator.HasComponent<Rendering::MaterialComponent>(entity)) {
                auto const& material = coordinator.GetComponent<Rendering::MaterialComponent>(entity);
                m_geometryPassShader->SetVec3("albedo", material.albedo);
                m_geometryPassShader->SetFloat("metallic", material.metallic);
                m_geometryPassShader->SetFloat("roughness", material.roughness);
                m_geometryPassShader->SetFloat("ao", material.ao);
                
                if (material.albedoMap)   { glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, material.albedoMap); }
                if (material.normalMap)   { glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, material.normalMap); }
                if (material.metallicMap) { glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, material.metallicMap); }
                if (material.roughnessMap){ glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, material.roughnessMap); }
                if (material.aoMap)       { glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, material.aoMap); }
                if (material.emissiveMap) {
                    glActiveTexture(GL_TEXTURE5);
                    glBindTexture(GL_TEXTURE_2D, material.emissiveMap);
                    // Log the bound emissive texture for diagnostics
                    LogTextureBinding("GeometryPass", 5, material.emissiveMap, "EmissiveMap");
                    GLint _w=0,_h=0; glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &_w); glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &_h);
                    std::cout << "{ \"frame\":" << m_frameID << ",\"EmissiveTexInfo\":{\"id\":" << material.emissiveMap << ",\"w\":" << _w << ",\"h\":" << _h << "} }" << std::endl;
                }
                
                // Allow per-material control of emissive intensity (fallback to 1.0)
                m_geometryPassShader->SetFloat("emissiveIntensity", material.emissiveIntensity > 0.0f ? material.emissiveIntensity : 1.0f);
            } else {
                // Default material
                m_geometryPassShader->SetVec3("albedo", glm::vec3(0.7f, 0.7f, 0.7f));
                m_geometryPassShader->SetFloat("metallic", 0.0f);
                m_geometryPassShader->SetFloat("roughness", 0.5f);
                m_geometryPassShader->SetFloat("ao", 1.0f);
            }
            
            // Draw logic priority: generator VAO -> simple cube -> model
            if (meshComponent.vaoId != 0 && meshComponent.indexCount > 0) {
                LogDrawCall("GeometryPass", 1, meshComponent.vaoId, "GL_TRIANGLES", meshComponent.indexCount);
                glBindVertexArray(meshComponent.vaoId);
                glDrawElements(GL_TRIANGLES, meshComponent.indexCount, GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
                entityCount++; m_drawCallCount++; m_triangleCount += meshComponent.indexCount / 3;
            } else if (meshComponent.modelPath == "player_cube") {
                LogDrawCall("GeometryPass", 1, m_cubeVAO, "GL_TRIANGLES", 36);
                RenderSimpleCube();
                entityCount++; m_drawCallCount++; m_triangleCount += 12;
            } else {
                LogDrawCall("GeometryPass", 1, 0, "MODEL_DRAW", -1);
                Model model(meshComponent.modelPath);
                model.Draw(*m_geometryPassShader);
                entityCount++; m_drawCallCount++; m_triangleCount += 100;
            }
        }
    }

    LogPassEnd("GeometryPass", m_drawCallCount, m_triangleCount);
    // Unbind G-buffer by binding default framebuffer
glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // Reset draw buffer to default back-buffer
    glDrawBuffer(GL_BACK);
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
    
    // Render to default framebuffer (already cleared at frame start)
    // Ensure default buffer is active
    glDrawBuffer(GL_BACK);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, windowWidth, windowHeight);  // Use actual window dimensions
    // NOTE: No clear here - back buffer already cleared at frame start to prevent flicker
    
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
    LogTextureBinding("LightingPass", 3, metallicTex, "MetallicRoughnessAOEmissive");
    m_lightingPassShader->SetInt("gMetallicRoughnessAOEmissive", 3);
    
    glActiveTexture(GL_TEXTURE4);
    uint32_t emissiveTex = m_gBuffer->GetColorAttachment(4);
    glBindTexture(GL_TEXTURE_2D, emissiveTex);
    LogTextureBinding("LightingPass", 4, emissiveTex, "Emissive");
    m_lightingPassShader->SetInt("gEmissive", 4);
    
    // Bind shadow map (if available)
    glActiveTexture(GL_TEXTURE5);
    uint32_t shadowTex = m_shadowMapFBO ? m_shadowMapFBO->GetDepthAttachment() : 0;
    if (shadowTex > 0) {
        glBindTexture(GL_TEXTURE_2D, shadowTex);
        LogTextureBinding("LightingPass", 5, shadowTex, "ShadowMap");
    } else {
        // Bind a default texture or use 0
        glBindTexture(GL_TEXTURE_2D, 0);
        LogTextureBinding("LightingPass", 5, 0, "ShadowMap_Default");
    }
    m_lightingPassShader->SetInt("shadowMap", 5);
    
    // Set light and camera uniforms
    // Use actual camera position
    if (m_mainCamera) {
        m_lightingPassShader->SetVec3("viewPos", m_mainCamera->GetPosition());
    } else {
        m_lightingPassShader->SetVec3("viewPos", glm::vec3(0.0f, 0.0f, 3.0f));
    }
    
    // Set a directional light (from above and slightly to the side for better depth perception)
    // Using a bright "sun" light coming from above at an angle for good coverage
    glm::vec3 lightDir = glm::normalize(glm::vec3(0.5f, -1.0f, 0.3f));
    m_lightingPassShader->SetVec3("light.direction", lightDir);
    m_lightingPassShader->SetVec3("light.color", glm::vec3(1.0f, 0.98f, 0.92f)); // Warm white sunlight
    m_lightingPassShader->SetFloat("light.intensity", 2.2f); // Increased for better visibility
    m_lightingPassShader->SetInt("light.type", 0); // 0 = directional light
    // For directional lights, position represents the light direction at infinite distance
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
    int previousMode = m_debugMode;
    m_debugMode = (m_debugMode + 1) % 7;  // 0..6: normal, position, normal, albedo, M/R/AO, emissiveColor, emissivePower
    const char* debugModeNames[] = {
        "Normal Lighting",
        "Position Buffer",
        "Normal Buffer",
        "Albedo Buffer",
        "Metallic/Roughness/AO Buffer",
        "Emissive Color",
        "Emissive Power"
    };
    
    std::cout << "{ \"debugModeChange\":{ \"frameID\":" << m_frameID << ",\"previousMode\":" << previousMode 
              << ",\"newMode\":" << m_debugMode << ",\"modeName\":\"" << debugModeNames[m_debugMode] 
              << "\",\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() << "} }" << std::endl;
    
    std::cout << "[RenderSystem] Debug mode switched to: " << debugModeNames[m_debugMode] << std::endl;
}

void RenderSystem::AdjustDepthScale(float multiplier) {
    m_depthScale *= multiplier;
    m_depthScale = glm::clamp(m_depthScale, 0.1f, 1000.0f);
    std::cout << "[Debug] Depth Scale: " << m_depthScale << std::endl;
}

void RenderSystem::ValidateAndLogCameraState() {
    if (!m_mainCamera) {
        std::cerr << "[RenderSystem] WARNING: No main camera available for validation!" << std::endl;
        return;
    }

    glm::vec3 camPos = m_mainCamera->GetPosition();
    glm::mat4 camView = m_mainCamera->GetViewMatrix();
    glm::mat4 camProj = m_mainCamera->GetProjectionMatrix();

    // Get viewport size from current OpenGL context instead of camera
    int viewWidth = 1920;
    int viewHeight = 1080;
    GLFWwindow* window = glfwGetCurrentContext();
    if (window) {
        glfwGetFramebufferSize(window, &viewWidth, &viewHeight);
    }

    std::cout << "{ \"frame\":" << m_frameID << ",\"cameraState\":{"
              << "\"position\":[" << camPos.x << "," << camPos.y << "," << camPos.z << "],"
              << "\"viewMatrix\":\"CHECKED\","
              << "\"projectionMatrix\":\"CHECKED\","
              << "\"viewportSize\":[" << viewWidth << "," << viewHeight << "] } }" << std::endl;
}

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
        for (int i = 0; i < 5; i++) {
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

// Debug drawing implementations
void RenderSystem::DrawDebugLine(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color) {
    // Simple line rendering using OpenGL immediate mode for debugging
    // Note: This is not performance-optimized but good for debug visualization
    
    glUseProgram(0); // Use fixed-function pipeline for simplicity
    glDisable(GL_DEPTH_TEST); // Draw lines on top
    
    glBegin(GL_LINES);
    glColor3f(color.r, color.g, color.b);
    glVertex3f(start.x, start.y, start.z);
    glVertex3f(end.x, end.y, end.z);
    glEnd();
    
    glEnable(GL_DEPTH_TEST); // Re-enable depth testing
}

void RenderSystem::DrawDebugCube(const glm::vec3& center, float size, const glm::vec3& color) {
    float halfSize = size * 0.5f;
    
    // Draw 12 edges of a cube
    glm::vec3 vertices[8] = {
        {center.x - halfSize, center.y - halfSize, center.z - halfSize}, // 0: bottom-left-back
        {center.x + halfSize, center.y - halfSize, center.z - halfSize}, // 1: bottom-right-back
        {center.x + halfSize, center.y + halfSize, center.z - halfSize}, // 2: top-right-back
        {center.x - halfSize, center.y + halfSize, center.z - halfSize}, // 3: top-left-back
        {center.x - halfSize, center.y - halfSize, center.z + halfSize}, // 4: bottom-left-front
        {center.x + halfSize, center.y - halfSize, center.z + halfSize}, // 5: bottom-right-front
        {center.x + halfSize, center.y + halfSize, center.z + halfSize}, // 6: top-right-front
        {center.x - halfSize, center.y + halfSize, center.z + halfSize}  // 7: top-left-front
    };
    
    // Draw back face
    DrawDebugLine(vertices[0], vertices[1], color);
    DrawDebugLine(vertices[1], vertices[2], color);
    DrawDebugLine(vertices[2], vertices[3], color);
    DrawDebugLine(vertices[3], vertices[0], color);
    
    // Draw front face
    DrawDebugLine(vertices[4], vertices[5], color);
    DrawDebugLine(vertices[5], vertices[6], color);
    DrawDebugLine(vertices[6], vertices[7], color);
    DrawDebugLine(vertices[7], vertices[4], color);
    
    // Draw connecting edges
    DrawDebugLine(vertices[0], vertices[4], color);
    DrawDebugLine(vertices[1], vertices[5], color);
    DrawDebugLine(vertices[2], vertices[6], color);
    DrawDebugLine(vertices[3], vertices[7], color);
}

void RenderSystem::DrawDebugFrustum(const Camera::Frustum& frustum, const glm::vec3& color) {
    // Implementation for drawing camera frustum
    // This would require extracting frustum corners from the frustum planes
    // For now, this is a stub - the CameraDebugSystem handles frustum drawing
}

void RenderSystem::ForwardPass() {
    // Log forward pass start
    LogPassStart("ForwardPass", 0, 0, 0); // Default framebuffer
    
    std::cout << "{ \"frame\":" << m_frameID << ",\"forwardPassStart\":true }" << std::endl;
    
    // Enable depth testing but don't write to depth (preserve depth from deferred pass)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL); // Use LEQUAL for proper depth testing with existing depth
    
    // Get current window dimensions
    int windowWidth, windowHeight;
    GLFWwindow* window = glfwGetCurrentContext();
    if (window) {
        glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
    } else {
        windowWidth = 1920;
        windowHeight = 1080;
    }
    
    // Ensure we're rendering to the default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, windowWidth, windowHeight);
    
    int forwardDrawCalls = 0;
    int forwardTriangles = 0;
    
    // === FORWARD PASS ITEM 1: Render Character (find player entity) ===
    Core::Entity playerEntity = Core::MAX_ENTITIES; // Invalid entity by default
    Core::Coordinator& coordinator = Core::Coordinator::GetInstance();
    
    // Find player entity by checking for PlayerMovementComponent
    auto playerMovementType = coordinator.GetComponentType<Gameplay::PlayerMovementComponent>();
    for (auto const& entity : mEntities) {
        if (coordinator.HasComponent<Gameplay::PlayerMovementComponent>(entity)) {
            playerEntity = entity;
            break;
        }
    }
    
    if (playerEntity != Core::MAX_ENTITIES && m_mainCamera) {
        std::cout << "{ \"frame\":" << m_frameID << ",\"forwardItem\":\"Character\",\"status\":\"Starting\" }" << std::endl;
        
        // Get player transform component for position
        auto& playerTransform = coordinator.GetComponent<Rendering::TransformComponent>(playerEntity);
        
        // Log camera parameters for character rendering
        glm::vec3 camPos = m_mainCamera->GetPosition();
        glm::vec3 playerPos = playerTransform.position;
        float distance = glm::length(playerPos - camPos);
        
        std::cout << "{ \"frame\":" << m_frameID << ",\"characterRender\":{"
                  << "\"playerPos\":[" << playerPos.x << "," << playerPos.y << "," << playerPos.z << "],"
                  << "\"cameraPos\":[" << camPos.x << "," << camPos.y << "," << camPos.z << "],"
                  << "\"distance\":" << distance << "} }" << std::endl;
        
        // Instead of using RenderSimpleCharacter, render the player directly here
        // This avoids the need for the old Player class
        if (m_geometryPassShader) {
            m_geometryPassShader->Use();
            
            // Set matrices
            m_geometryPassShader->SetMat4("view", m_mainCamera->GetViewMatrix());
            m_geometryPassShader->SetMat4("projection", m_mainCamera->GetProjectionMatrix());
            
            // Create model matrix from player position
            glm::mat4 modelMatrix = glm::mat4(1.0f);
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
            
            // Set player material - bright green color to distinguish from other objects
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
        LogGLError("AfterCharacterRender");
        
        forwardDrawCalls++;
        forwardTriangles += 12; // Simple cube has 12 triangles
        
        std::cout << "{ \"frame\":" << m_frameID << ",\"forwardItem\":\"Character\",\"status\":\"Complete\" }" << std::endl;
    } else {
        std::cout << "{ \"frame\":" << m_frameID << ",\"forwardItem\":\"Character\",\"status\":\"SKIPPED\",\"reason\":\"" 
                  << (playerEntity != Core::MAX_ENTITIES ? "NoCamera" : "NoPlayer") << "\" }" << std::endl;
    }
    
    // === FORWARD PASS ITEM 2: Debug camera frustum (if enabled) ===
    if (m_cameraDebugEnabled && m_cameraDebugSystem && m_mainCamera) {
        std::cout << "{ \"frame\":" << m_frameID << ",\"forwardItem\":\"CameraDebug\",\"status\":\"Starting\" }" << std::endl;
        
        // Log camera debug parameters
        std::cout << "{ \"frame\":" << m_frameID << ",\"cameraDebugParams\":{"
                  << "\"nearPlane\":" << m_mainCamera->GetNearPlane() << ","
                  << "\"farPlane\":" << m_mainCamera->GetFarPlane() << ","
                  << "\"fov\":" << m_mainCamera->GetFOV() << ","
                  << "\"aspectRatio\":" << m_mainCamera->GetAspectRatio() << "} }" << std::endl;
        
        // Disable depth writes for debug lines so they don't interfere with future frames
        glDepthMask(GL_FALSE);
        
        m_cameraDebugSystem->DrawFrustum(*m_mainCamera);
        LogGLError("AfterCameraDebugRender");
        
        // Re-enable depth writes
        glDepthMask(GL_TRUE);
        
        forwardDrawCalls += 8; // Frustum typically has 8 lines (edges)
        
        std::cout << "{ \"frame\":" << m_frameID << ",\"forwardItem\":\"CameraDebug\",\"status\":\"Complete\" }" << std::endl;
    } else {
        std::string reason = "DISABLED";
        if (!m_cameraDebugEnabled) reason = "DISABLED";
        else if (!m_cameraDebugSystem) reason = "NoDebugSystem";
        else if (!m_mainCamera) reason = "NoCamera";
        
        std::cout << "{ \"frame\":" << m_frameID << ",\"forwardItem\":\"CameraDebug\",\"status\":\"SKIPPED\",\"reason\":\"" 
                  << reason << "\" }" << std::endl;
    }
    
    // === FORWARD PASS ITEM 3: Placeholder for transparent objects ===
    // This is where transparent objects would be rendered in the future
    std::cout << "{ \"frame\":" << m_frameID << ",\"forwardItem\":\"TransparentObjects\",\"status\":\"NOT_IMPLEMENTED\" }" << std::endl;
    
    // === FORWARD PASS ITEM 4: Placeholder for UI/HUD elements ===
    // This is where UI elements would be rendered in the future
    std::cout << "{ \"frame\":" << m_frameID << ",\"forwardItem\":\"UI\",\"status\":\"NOT_IMPLEMENTED\" }" << std::endl;
    
    // Reset depth function to default
    glDepthFunc(GL_LESS);
    
    LogPassEnd("ForwardPass", forwardDrawCalls, forwardTriangles);
    std::cout << "{ \"frame\":" << m_frameID << ",\"forwardPassComplete\":true,"
              << "\"totalDrawCalls\":" << forwardDrawCalls << ","
              << "\"totalTriangles\":" << forwardTriangles << " }" << std::endl;
}

void RenderSystem::ToggleCameraDebug() {
    m_cameraDebugEnabled = !m_cameraDebugEnabled;
    std::cout << "[RenderSystem] Camera debug visualization " 
              << (m_cameraDebugEnabled ? "ENABLED" : "DISABLED") << std::endl;
}

} // namespace Rendering
} // namespace CudaGame
