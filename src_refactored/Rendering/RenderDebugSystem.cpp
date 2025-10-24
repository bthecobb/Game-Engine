#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Rendering/RenderDebugSystem.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Framebuffer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace CudaGame {
namespace Rendering {

RenderDebugSystem::RenderDebugSystem()
    : m_currentMode(DebugVisualizationMode::NONE)
    , m_frameTimeIndex(0)
    , m_avgFrameTime(0.0f)
    , m_minFrameTime(FLT_MAX)
    , m_maxFrameTime(0.0f)
    , m_shaderHotReload(false)
    , m_showStatistics(true)
    , m_showPerformanceWarnings(true)
    , m_enableGLDebugOutput(true)
    , m_fullscreenQuadVAO(0)
    , m_fullscreenQuadVBO(0)
    , m_debugLineVAO(0)
    , m_debugLineVBO(0) {
    
    // Initialize frame time history
    std::fill(std::begin(m_frameTimeHistory), std::end(m_frameTimeHistory), 0.0f);
}

RenderDebugSystem::~RenderDebugSystem() {
    Shutdown();
}

bool RenderDebugSystem::Initialize() {
    std::cout << "[RenderDebugSystem] Initializing debug rendering system..." << std::endl;

    // Enable OpenGL debug output if available
    if (m_enableGLDebugOutput) {
        GLint flags;
        glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
        if (flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
            glEnable(GL_DEBUG_OUTPUT);
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback(GLDebugCallback, this);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
            std::cout << "[RenderDebugSystem] OpenGL debug output enabled" << std::endl;
        }
    }

    // Create debug shaders
    CreateDebugShaders();
    
    // Create debug meshes
    CreateDebugMeshes();

    std::cout << "[RenderDebugSystem] Debug system initialized successfully" << std::endl;
    return true;
}

void RenderDebugSystem::Update(float deltaTime) {
    // Update is handled through manual BeginFrame/EndFrame calls
}

void RenderDebugSystem::Shutdown() {
    // Clean up VAOs and VBOs
    if (m_fullscreenQuadVAO) {
        glDeleteVertexArrays(1, &m_fullscreenQuadVAO);
        m_fullscreenQuadVAO = 0;
    }
    if (m_fullscreenQuadVBO) {
        glDeleteBuffers(1, &m_fullscreenQuadVBO);
        m_fullscreenQuadVBO = 0;
    }
    if (m_debugLineVAO) {
        glDeleteVertexArrays(1, &m_debugLineVAO);
        m_debugLineVAO = 0;
    }
    if (m_debugLineVBO) {
        glDeleteBuffers(1, &m_debugLineVBO);
        m_debugLineVBO = 0;
    }
}

void RenderDebugSystem::CreateDebugShaders() {
    // Create debug texture shader
    m_debugTextureShader = std::make_shared<ShaderProgram>();
    if (!m_debugTextureShader->LoadFromFiles(
        ASSET_DIR "/shaders/debug_texture.vert",
        ASSET_DIR "/shaders/debug_texture.frag")) {
        std::cerr << "[RenderDebugSystem] Failed to load debug texture shader" << std::endl;
    }

    // Create depth visualization shader
    const char* depthVertSrc = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 1.0);
            TexCoord = aTexCoord;
        }
    )";

    const char* depthFragSrc = R"(
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;
        uniform sampler2D depthTexture;
        uniform float near = 0.1;
        uniform float far = 100.0;
        
        float LinearizeDepth(float depth) {
            float z = depth * 2.0 - 1.0;
            return (2.0 * near * far) / (far + near - z * (far - near));
        }
        
        void main() {
            float depthValue = texture(depthTexture, TexCoord).r;
            float linearDepth = LinearizeDepth(depthValue) / far;
            FragColor = vec4(vec3(linearDepth), 1.0);
        }
    )";

    m_depthShader = std::make_shared<ShaderProgram>();
    m_depthShader->LoadFromSource(depthVertSrc, depthFragSrc);

    // Create wireframe shader
    const char* wireframeVertSrc = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
    )";

    const char* wireframeFragSrc = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec3 wireColor = vec3(0.0, 1.0, 0.0);
        void main() {
            FragColor = vec4(wireColor, 1.0);
        }
    )";

    m_wireframeShader = std::make_shared<ShaderProgram>();
    m_wireframeShader->LoadFromSource(wireframeVertSrc, wireframeFragSrc);

    // Create debug line shader
    const char* lineVertSrc = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        out vec3 vertexColor;
        uniform mat4 viewProjection;
        void main() {
            gl_Position = viewProjection * vec4(aPos, 1.0);
            vertexColor = aColor;
        }
    )";

    const char* lineFragSrc = R"(
        #version 330 core
        in vec3 vertexColor;
        out vec4 FragColor;
        void main() {
            FragColor = vec4(vertexColor, 1.0);
        }
    )";

    m_debugLineShader = std::make_shared<ShaderProgram>();
    m_debugLineShader->LoadFromSource(lineVertSrc, lineFragSrc);
}

void RenderDebugSystem::CreateDebugMeshes() {
    // Create fullscreen quad for texture visualization
    float quadVertices[] = {
        // positions        // texture coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f
    };

    glGenVertexArrays(1, &m_fullscreenQuadVAO);
    glGenBuffers(1, &m_fullscreenQuadVBO);

    glBindVertexArray(m_fullscreenQuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_fullscreenQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Create debug line VAO/VBO
    glGenVertexArrays(1, &m_debugLineVAO);
    glGenBuffers(1, &m_debugLineVBO);

    glBindVertexArray(m_debugLineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_debugLineVBO);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void RenderDebugSystem::BeginFrame() {
    // Reset statistics for new frame
    m_lastFrameStats = m_statistics;
    m_statistics = RenderStatistics();
    m_performanceWarnings.clear();
}

void RenderDebugSystem::EndFrame() {
    // Update frame time history
    UpdateFrameTimeHistory(m_statistics.frameTime);
    
    // Check for performance issues
    CheckPerformanceIssues();
    
    // Render debug overlay if enabled
    if (m_currentMode != DebugVisualizationMode::NONE) {
        RenderDebugOverlay();
    }
    
    // Render statistics if enabled
    if (m_showStatistics) {
        RenderStatisticsOverlay();
    }
}

void RenderDebugSystem::UpdateStatistics(const RenderStatistics& stats) {
    m_statistics = stats;
}

void RenderDebugSystem::SetVisualizationMode(DebugVisualizationMode mode) {
    m_currentMode = mode;
    
    // Apply specific settings based on mode
    switch (mode) {
        case DebugVisualizationMode::WIREFRAME:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            break;
        default:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            break;
    }
    
    std::cout << "[RenderDebugSystem] Visualization mode changed to: " 
              << GetVisualizationModeName(mode) << std::endl;
}

void RenderDebugSystem::CycleVisualizationMode() {
    int currentModeInt = static_cast<int>(m_currentMode);
    currentModeInt = (currentModeInt + 1) % 11; // Total number of modes
    SetVisualizationMode(static_cast<DebugVisualizationMode>(currentModeInt));
}

void RenderDebugSystem::RenderGBufferVisualization(Framebuffer* gBuffer) {
    if (!gBuffer || !m_debugTextureShader) return;

    // Save current viewport
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    // Render G-buffer textures in quadrants
    int halfWidth = viewport[2] / 2;
    int halfHeight = viewport[3] / 2;

    m_debugTextureShader->Use();

    // Top-left: Position
    glViewport(0, halfHeight, halfWidth, halfHeight);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gBuffer->GetColorAttachment(0));
    m_debugTextureShader->SetInt("debugTexture", 0);
    RenderFullscreenQuad();

    // Top-right: Normal
    glViewport(halfWidth, halfHeight, halfWidth, halfHeight);
    glBindTexture(GL_TEXTURE_2D, gBuffer->GetColorAttachment(1));
    RenderFullscreenQuad();

    // Bottom-left: Albedo
    glViewport(0, 0, halfWidth, halfHeight);
    glBindTexture(GL_TEXTURE_2D, gBuffer->GetColorAttachment(2));
    RenderFullscreenQuad();

    // Bottom-right: Specular
    glViewport(halfWidth, 0, halfWidth, halfHeight);
    glBindTexture(GL_TEXTURE_2D, gBuffer->GetColorAttachment(3));
    RenderFullscreenQuad();

    // Restore viewport
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}

void RenderDebugSystem::RenderDepthBufferVisualization() {
    if (!m_depthShader) return;

    m_depthShader->Use();
    
    // Bind depth texture from current framebuffer
    GLint currentFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFBO);
    
    if (currentFBO != 0) {
        GLint depthTexture;
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                              GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &depthTexture);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        m_depthShader->SetInt("depthTexture", 0);
    }
    
    RenderFullscreenQuad();
}

void RenderDebugSystem::RenderWireframeMode(bool enable) {
    if (enable) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDisable(GL_CULL_FACE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_CULL_FACE);
    }
}

void RenderDebugSystem::RenderFullscreenQuad() {
    if (m_fullscreenQuadVAO == 0) return;
    
    glBindVertexArray(m_fullscreenQuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void RenderDebugSystem::ValidateFramebuffer(const std::string& context) {
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[RenderDebugSystem] Framebuffer incomplete in " << context << ": ";
        switch (status) {
            case GL_FRAMEBUFFER_UNDEFINED:
                std::cerr << "UNDEFINED" << std::endl;
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                std::cerr << "INCOMPLETE_ATTACHMENT" << std::endl;
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                std::cerr << "INCOMPLETE_MISSING_ATTACHMENT" << std::endl;
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                std::cerr << "INCOMPLETE_DRAW_BUFFER" << std::endl;
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                std::cerr << "INCOMPLETE_READ_BUFFER" << std::endl;
                break;
            case GL_FRAMEBUFFER_UNSUPPORTED:
                std::cerr << "UNSUPPORTED" << std::endl;
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                std::cerr << "INCOMPLETE_MULTISAMPLE" << std::endl;
                break;
            default:
                std::cerr << "UNKNOWN (" << status << ")" << std::endl;
                break;
        }
    }
}

void RenderDebugSystem::CheckGLError(const std::string& context) {
    GLenum error;
    bool hasError = false;
    while ((error = glGetError()) != GL_NO_ERROR) {
        hasError = true;
        std::cerr << "[RenderDebugSystem] OpenGL error in " << context << ": ";
        switch (error) {
            case GL_INVALID_ENUM:
                std::cerr << "INVALID_ENUM" << std::endl;
                break;
            case GL_INVALID_VALUE:
                std::cerr << "INVALID_VALUE" << std::endl;
                break;
            case GL_INVALID_OPERATION:
                std::cerr << "INVALID_OPERATION" << std::endl;
                break;
            case GL_STACK_OVERFLOW:
                std::cerr << "STACK_OVERFLOW" << std::endl;
                break;
            case GL_STACK_UNDERFLOW:
                std::cerr << "STACK_UNDERFLOW" << std::endl;
                break;
            case GL_OUT_OF_MEMORY:
                std::cerr << "OUT_OF_MEMORY" << std::endl;
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                std::cerr << "INVALID_FRAMEBUFFER_OPERATION" << std::endl;
                break;
            default:
                std::cerr << "UNKNOWN (" << error << ")" << std::endl;
                break;
        }
    }
    
    if (hasError) {
        // Log current GL state for debugging
        LogGLState(context);
    }
}

void RenderDebugSystem::LogGLState(const std::string& context) {
    std::cout << "[RenderDebugSystem] OpenGL state at " << context << ":" << std::endl;
    
    GLint intVal[4];
    GLfloat floatVal[4];
    GLboolean boolVal;
    
    // Viewport
    glGetIntegerv(GL_VIEWPORT, intVal);
    std::cout << "  Viewport: " << intVal[0] << ", " << intVal[1] 
              << ", " << intVal[2] << ", " << intVal[3] << std::endl;
    
    // Framebuffer binding
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &intVal[0]);
    std::cout << "  Framebuffer binding: " << intVal[0] << std::endl;
    
    // Depth test
    glGetBooleanv(GL_DEPTH_TEST, &boolVal);
    std::cout << "  Depth test: " << (boolVal ? "enabled" : "disabled") << std::endl;
    
    // Blend
    glGetBooleanv(GL_BLEND, &boolVal);
    std::cout << "  Blend: " << (boolVal ? "enabled" : "disabled") << std::endl;
    
    // Cull face
    glGetBooleanv(GL_CULL_FACE, &boolVal);
    std::cout << "  Cull face: " << (boolVal ? "enabled" : "disabled") << std::endl;
    
    // Clear color
    glGetFloatv(GL_COLOR_CLEAR_VALUE, floatVal);
    std::cout << "  Clear color: " << floatVal[0] << ", " << floatVal[1] 
              << ", " << floatVal[2] << ", " << floatVal[3] << std::endl;
}

void RenderDebugSystem::DrawDebugLine(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color) {
    m_debugLineVertices.push_back(start.x);
    m_debugLineVertices.push_back(start.y);
    m_debugLineVertices.push_back(start.z);
    m_debugLineVertices.push_back(color.x);
    m_debugLineVertices.push_back(color.y);
    m_debugLineVertices.push_back(color.z);
    
    m_debugLineVertices.push_back(end.x);
    m_debugLineVertices.push_back(end.y);
    m_debugLineVertices.push_back(end.z);
    m_debugLineVertices.push_back(color.x);
    m_debugLineVertices.push_back(color.y);
    m_debugLineVertices.push_back(color.z);
}

void RenderDebugSystem::DrawDebugBox(const glm::vec3& min, const glm::vec3& max, const glm::vec3& color) {
    // Draw 12 edges of the box
    glm::vec3 corners[8] = {
        glm::vec3(min.x, min.y, min.z),
        glm::vec3(max.x, min.y, min.z),
        glm::vec3(max.x, max.y, min.z),
        glm::vec3(min.x, max.y, min.z),
        glm::vec3(min.x, min.y, max.z),
        glm::vec3(max.x, min.y, max.z),
        glm::vec3(max.x, max.y, max.z),
        glm::vec3(min.x, max.y, max.z)
    };
    
    // Bottom face
    DrawDebugLine(corners[0], corners[1], color);
    DrawDebugLine(corners[1], corners[2], color);
    DrawDebugLine(corners[2], corners[3], color);
    DrawDebugLine(corners[3], corners[0], color);
    
    // Top face
    DrawDebugLine(corners[4], corners[5], color);
    DrawDebugLine(corners[5], corners[6], color);
    DrawDebugLine(corners[6], corners[7], color);
    DrawDebugLine(corners[7], corners[4], color);
    
    // Vertical edges
    DrawDebugLine(corners[0], corners[4], color);
    DrawDebugLine(corners[1], corners[5], color);
    DrawDebugLine(corners[2], corners[6], color);
    DrawDebugLine(corners[3], corners[7], color);
}

void RenderDebugSystem::CheckPerformanceIssues() {
    // Check for high draw call count
    if (m_statistics.drawCalls > 1000) {
        LogPerformanceWarning("High draw call count: " + std::to_string(m_statistics.drawCalls));
    }
    
    // Check for excessive triangle count
    if (m_statistics.trianglesRendered > 10000000) {
        LogPerformanceWarning("High triangle count: " + std::to_string(m_statistics.trianglesRendered));
    }
    
    // Check for frame time spikes
    if (m_statistics.frameTime > 33.33f) { // Below 30 FPS
        LogPerformanceWarning("Frame time spike: " + std::to_string(m_statistics.frameTime) + "ms");
    }
    
    // Check for excessive texture binds
    if (m_statistics.textureBinds > 500) {
        LogPerformanceWarning("Excessive texture binds: " + std::to_string(m_statistics.textureBinds));
    }
    
    // Check for shader switching
    if (m_statistics.shaderSwitches > 100) {
        LogPerformanceWarning("Excessive shader switches: " + std::to_string(m_statistics.shaderSwitches));
    }
}

void RenderDebugSystem::LogPerformanceWarning(const std::string& warning) {
    m_performanceWarnings.push_back(warning);
    if (m_showPerformanceWarnings) {
        std::cout << "[RenderDebugSystem] Performance Warning: " << warning << std::endl;
    }
}

void RenderDebugSystem::UpdateFrameTimeHistory(float frameTime) {
    m_frameTimeHistory[m_frameTimeIndex] = frameTime;
    m_frameTimeIndex = (m_frameTimeIndex + 1) % FRAME_TIME_HISTORY_SIZE;
    
    // Update statistics
    m_avgFrameTime = 0.0f;
    m_minFrameTime = FLT_MAX;
    m_maxFrameTime = 0.0f;
    
    for (int i = 0; i < FRAME_TIME_HISTORY_SIZE; ++i) {
        float time = m_frameTimeHistory[i];
        m_avgFrameTime += time;
        m_minFrameTime = std::min(m_minFrameTime, time);
        m_maxFrameTime = std::max(m_maxFrameTime, time);
    }
    m_avgFrameTime /= FRAME_TIME_HISTORY_SIZE;
}

std::string RenderDebugSystem::GetVisualizationModeName(DebugVisualizationMode mode) {
    switch (mode) {
        case DebugVisualizationMode::NONE: return "None";
        case DebugVisualizationMode::WIREFRAME: return "Wireframe";
        case DebugVisualizationMode::NORMALS: return "Normals";
        case DebugVisualizationMode::DEPTH_BUFFER: return "Depth Buffer";
        case DebugVisualizationMode::GBUFFER_POSITION: return "G-Buffer Position";
        case DebugVisualizationMode::GBUFFER_NORMAL: return "G-Buffer Normal";
        case DebugVisualizationMode::GBUFFER_ALBEDO: return "G-Buffer Albedo";
        case DebugVisualizationMode::GBUFFER_SPECULAR: return "G-Buffer Specular";
        case DebugVisualizationMode::SHADOW_MAP: return "Shadow Map";
        case DebugVisualizationMode::OVERDRAW: return "Overdraw";
        case DebugVisualizationMode::FRUSTUM_CULLING: return "Frustum Culling";
        default: return "Unknown";
    }
}

void RenderDebugSystem::RenderStatisticsOverlay() {
    // This would typically render text overlay with statistics
    // For now, just log to console periodically
    static int frameCounter = 0;
    if (++frameCounter % 60 == 0) {
        std::cout << "[RenderDebugSystem] Frame Statistics:" << std::endl;
        std::cout << "  FPS: " << (1000.0f / m_avgFrameTime) << std::endl;
        std::cout << "  Frame Time: " << m_avgFrameTime << "ms (min: " << m_minFrameTime 
                  << "ms, max: " << m_maxFrameTime << "ms)" << std::endl;
        std::cout << "  Draw Calls: " << m_statistics.drawCalls << std::endl;
        std::cout << "  Triangles: " << m_statistics.trianglesRendered << std::endl;
        std::cout << "  Texture Binds: " << m_statistics.textureBinds << std::endl;
        std::cout << "  Shader Switches: " << m_statistics.shaderSwitches << std::endl;
        
        if (!m_performanceWarnings.empty()) {
            std::cout << "  Warnings:" << std::endl;
            for (const auto& warning : m_performanceWarnings) {
                std::cout << "    - " << warning << std::endl;
            }
        }
    }
}

void RenderDebugSystem::RenderDebugOverlay() {
    // Render based on current visualization mode
    switch (m_currentMode) {
        case DebugVisualizationMode::WIREFRAME:
            RenderWireframeMode(true);
            break;
        case DebugVisualizationMode::DEPTH_BUFFER:
            RenderDepthBufferVisualization();
            break;
        // Add other visualization modes as needed
        default:
            break;
    }
}

void APIENTRY RenderDebugSystem::GLDebugCallback(GLenum source, GLenum type, GLuint id,
                                                 GLenum severity, GLsizei length,
                                                 const GLchar* message, const void* userParam) {
    // Ignore non-significant error codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    std::string sourceStr;
    switch (source) {
        case GL_DEBUG_SOURCE_API: sourceStr = "API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM: sourceStr = "Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: sourceStr = "Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY: sourceStr = "Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION: sourceStr = "Application"; break;
        case GL_DEBUG_SOURCE_OTHER: sourceStr = "Other"; break;
    }

    std::string typeStr;
    switch (type) {
        case GL_DEBUG_TYPE_ERROR: typeStr = "Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typeStr = "Deprecated"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: typeStr = "Undefined Behavior"; break;
        case GL_DEBUG_TYPE_PORTABILITY: typeStr = "Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE: typeStr = "Performance"; break;
        case GL_DEBUG_TYPE_MARKER: typeStr = "Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP: typeStr = "Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP: typeStr = "Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER: typeStr = "Other"; break;
    }

    std::string severityStr;
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH: severityStr = "HIGH"; break;
        case GL_DEBUG_SEVERITY_MEDIUM: severityStr = "MEDIUM"; break;
        case GL_DEBUG_SEVERITY_LOW: severityStr = "LOW"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: severityStr = "NOTIFICATION"; break;
    }

    if (severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM) {
        std::cerr << "[GL Debug " << severityStr << "] " << sourceStr << " - " << typeStr 
                  << " (" << id << "): " << message << std::endl;
    }
}

void RenderDebugSystem::ValidateShaderProgram(GLuint program, const std::string& name) {
    GLint success;
    glValidateProgram(program);
    glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
    
    if (!success) {
        GLchar infoLog[1024];
        glGetProgramInfoLog(program, 1024, nullptr, infoLog);
        std::cerr << "[RenderDebugSystem] Shader program '" << name 
                  << "' validation failed: " << infoLog << std::endl;
    }
}

void RenderDebugSystem::DumpFramebufferToFile(GLuint fbo, const std::string& filename) {
    // Save current framebuffer binding
    GLint currentFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFBO);
    
    // Bind the framebuffer to dump
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    
    // Get framebuffer dimensions
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int width = viewport[2];
    int height = viewport[3];
    
    // Allocate buffer for pixel data
    std::vector<unsigned char> pixels(width * height * 3);
    
    // Read pixels
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    // Write to file (simple PPM format)
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file << "P6\n" << width << " " << height << "\n255\n";
        
        // Flip vertically (OpenGL reads from bottom-left)
        for (int y = height - 1; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                file.write(reinterpret_cast<char*>(&pixels[idx]), 3);
            }
        }
        
        file.close();
        std::cout << "[RenderDebugSystem] Framebuffer dumped to " << filename << std::endl;
    } else {
        std::cerr << "[RenderDebugSystem] Failed to open file " << filename << std::endl;
    }
    
    // Restore previous framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, currentFBO);
}

void RenderDebugSystem::RenderImGuiDebugWindow() {
    // Placeholder for ImGui integration
    // This would render an ImGui window with debug controls and visualizations
}

} // namespace Rendering
} // namespace CudaGame
