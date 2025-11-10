#include "UIRenderer.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include <cstring>

// Undefine Windows DrawText macro to avoid conflicts
#ifdef DrawText
#undef DrawText
#endif

namespace CudaGame {
namespace UI {

UIRenderer::UIRenderer() :
    m_VAO(0), m_VBO(0), m_shaderProgram(0), m_fontTexture(0),
    m_viewportWidth(800), m_viewportHeight(600) {}

UIRenderer::~UIRenderer() {
    Shutdown();
}

bool UIRenderer::Initialize() {
    // Create shaders and font texture
    if (!CreateShaders() || !CreateFontTexture()) {
        Shutdown();
        return false;
    }

    // Create VAO and VBO for UI rendering
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);

    glBindVertexArray(m_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, 1024 * sizeof(UIVertex), nullptr, GL_DYNAMIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(UIVertex), (void*)offsetof(UIVertex, position));
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(UIVertex), (void*)offsetof(UIVertex, texCoord));
    glEnableVertexAttribArray(1);

    // Color attribute
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(UIVertex), (void*)offsetof(UIVertex, color));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    return true;
}

void UIRenderer::Shutdown() {
    if (m_shaderProgram) {
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }

    if (m_fontTexture) {
        glDeleteTextures(1, &m_fontTexture);
        m_fontTexture = 0;
    }

    if (m_VAO) {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }

    if (m_VBO) {
        glDeleteBuffers(1, &m_VBO);
        m_VBO = 0;
    }
}

void UIRenderer::BeginFrame() {
    m_vertices.clear();
    m_indices.clear();
}

void UIRenderer::EndFrame() {
    if (m_vertices.empty()) return;
    
    // Upload vertex data to GPU and draw
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_vertices.size() * sizeof(UIVertex), m_vertices.data());

    glUseProgram(m_shaderProgram);
    glBindVertexArray(m_VAO);

    // Set projection matrix
    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(m_viewportWidth),
                                      static_cast<float>(m_viewportHeight), 0.0f);
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "projection"),
                       1, GL_FALSE, glm::value_ptr(projection));

    // Bind font texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_fontTexture);
    glUniform1i(glGetUniformLocation(m_shaderProgram, "fontTexture"), 0);
    glUniform1i(glGetUniformLocation(m_shaderProgram, "useTexture"), 1);
    
    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Draw all vertices as triangles
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_vertices.size()));
    
    glDisable(GL_BLEND);
    glBindVertexArray(0);
    glUseProgram(0);
}

void UIRenderer::SetViewportSize(int width, int height) {
    m_viewportWidth = width;
    m_viewportHeight = height;
}

void UIRenderer::RenderText(const std::string& text, float x, float y, float scale, glm::vec3 color) {
    const float CHAR_WIDTH = 8.0f * scale;
    const float CHAR_HEIGHT = 8.0f * scale;
    const float FONT_TEX_WIDTH = 128.0f;
    const float FONT_TEX_HEIGHT = 64.0f;
    const float CHAR_TEX_WIDTH = 8.0f / FONT_TEX_WIDTH;
    const float CHAR_TEX_HEIGHT = 8.0f / FONT_TEX_HEIGHT;
    
    float currentX = x;
    
    for (char c : text) {
        if (c == ' ') {
            currentX += CHAR_WIDTH;
            continue;
        }
        
        // Simple character mapping (assumes ASCII 32-127)
        int charIndex = (int)c - 32;
        if (charIndex < 0 || charIndex > 95) charIndex = 0;
        
        // Calculate texture coordinates
        int row = charIndex / 16;
        int col = charIndex % 16;
        
        float u0 = col * CHAR_TEX_WIDTH;
        float v0 = row * CHAR_TEX_HEIGHT;
        float u1 = u0 + CHAR_TEX_WIDTH;
        float v1 = v0 + CHAR_TEX_HEIGHT;
        
        // Add character quad
        AddQuad(currentX, y, CHAR_WIDTH, CHAR_HEIGHT, glm::vec4(color, 1.0f), u0, v0, u1, v1);
        
        currentX += CHAR_WIDTH;
    }
}

void UIRenderer::DrawRect(float x, float y, float width, float height, glm::vec4 color) {
    AddQuad(x, y, width, height, color);
}

void UIRenderer::DrawFilledRect(float x, float y, float width, float height, glm::vec4 color) {
    AddQuad(x, y, width, height, color);
}

void UIRenderer::DrawHealthBar(float x, float y, float width, float height, float health, float maxHealth) {
    // Background
    DrawFilledRect(x, y, width, height, glm::vec4(0.2f, 0.2f, 0.2f, 0.8f));
    
    // Health fill
    if (health > 0 && maxHealth > 0) {
        float healthPercent = health / maxHealth;
        glm::vec4 healthColor = glm::vec4(1.0f - healthPercent, healthPercent, 0.0f, 1.0f); // Red to green
        DrawFilledRect(x + 2, y + 2, (width - 4) * healthPercent, height - 4, healthColor);
    }
    
    // Border
    DrawRect(x, y, width, height, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
}

void UIRenderer::DrawDebugInfo(float fps, const glm::vec3& playerPos, int enemyCount) {
    const float LINE_HEIGHT = 20.0f;
    float yPos = 10.0f;
    
    // FPS Counter
    std::string fpsText = "FPS: " + std::to_string((int)fps);
    RenderText(fpsText, 10.0f, yPos, 1.0f, glm::vec3(1.0f));
    yPos += LINE_HEIGHT;
    
    // Player Position
    std::string posText = "Pos: (" + 
        std::to_string((int)playerPos.x) + ", " + 
        std::to_string((int)playerPos.y) + ", " + 
        std::to_string((int)playerPos.z) + ")";
    RenderText(posText, 10.0f, yPos, 1.0f, glm::vec3(0.8f, 0.8f, 1.0f));
    yPos += LINE_HEIGHT;
    
    // Enemy Count
    std::string enemyText = "Enemies: " + std::to_string(enemyCount);
    RenderText(enemyText, 10.0f, yPos, 1.0f, glm::vec3(1.0f, 0.8f, 0.8f));
}

void UIRenderer::AddQuad(float x, float y, float width, float height, const glm::vec4& color,
                         float u0, float v0, float u1, float v1) {
    if (m_vertices.size() + 6 >= 1024) { // Limited to 1024 vertices for now
        FlushBatch();
    }
    
    glm::vec2 pos[] = {
        glm::vec2(x, y),
        glm::vec2(x + width, y),
        glm::vec2(x + width, y + height),
        glm::vec2(x, y + height)
    };

    glm::vec2 tex[] = {
        glm::vec2(u0, v0),
        glm::vec2(u1, v0),
        glm::vec2(u1, v1),
        glm::vec2(u0, v1)
    };

    m_vertices.push_back({pos[0], tex[0], color});
    m_vertices.push_back({pos[1], tex[1], color});
    m_vertices.push_back({pos[2], tex[2], color});

    m_vertices.push_back({pos[0], tex[0], color});
    m_vertices.push_back({pos[2], tex[2], color});
    m_vertices.push_back({pos[3], tex[3], color});
}

void UIRenderer::FlushBatch() {
    // Actual draw calls to render all UI elements
    if (m_vertices.empty()) return;

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_vertices.size() * sizeof(UIVertex), m_vertices.data());

    glUseProgram(m_shaderProgram);
    glBindVertexArray(m_VAO);

    // Change this to GL_TRIANGLES if the batch is using triangles
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_vertices.size()));

    glBindVertexArray(0);
    glUseProgram(0);

    m_vertices.clear();
}

bool UIRenderer::CreateShaders() {
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        layout (location = 2) in vec4 aColor;
        
        out vec2 TexCoord;
        out vec4 Color;
        
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * vec4(aPos.x, aPos.y, 0.0, 1.0);
            TexCoord = aTexCoord;
            Color = aColor;
        }
    )";
    
    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec2 TexCoord;
        in vec4 Color;
        out vec4 FragColor;
        
        uniform sampler2D fontTexture;
        uniform int useTexture;
        
        void main() {
            if (useTexture == 1) {
                vec4 sampled = vec4(1.0, 1.0, 1.0, texture(fontTexture, TexCoord).r);
                FragColor = Color * sampled;
            } else {
                FragColor = Color;
            }
        }
    )";
    
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for vertex shader compile errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return false;
    }
    
    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    // Check for fragment shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        return false;
    }
    
    // Link shaders
    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vertexShader);
    glAttachShader(m_shaderProgram, fragmentShader);
    glLinkProgram(m_shaderProgram);
    
    // Check for linking errors
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        return false;
    }
    
    // Delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return true;
}

bool UIRenderer::CreateFontTexture() {
    // Create a simple 8x8 font texture for ASCII characters 32-127
    const int FONT_WIDTH = 128;  // 16 chars * 8 pixels
    const int FONT_HEIGHT = 64;  // 8 rows * 8 pixels
    const int CHAR_WIDTH = 8;
    const int CHAR_HEIGHT = 8;
    
    // Simple bitmap font data - very basic 8x8 characters
    unsigned char fontData[FONT_WIDTH * FONT_HEIGHT];
    memset(fontData, 0, sizeof(fontData));
    
    // Create simple patterns for some basic characters
    // This is a very simplified font - in a real implementation you'd load from a file
    
    // For now, create a simple white square for all characters
    // You can expand this to include actual font data
    for (int i = 0; i < FONT_WIDTH * FONT_HEIGHT; i++) {
        fontData[i] = 255; // Full white for now - makes text visible
    }
    
    glGenTextures(1, &m_fontTexture);
    glBindTexture(GL_TEXTURE_2D, m_fontTexture);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, FONT_WIDTH, FONT_HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE, fontData);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    
    return true;
}

} // namespace UI
} // namespace CudaGame

