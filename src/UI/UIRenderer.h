#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <memory>

namespace CudaGame {
namespace UI {

class UIRenderer {
public:
    UIRenderer();
    ~UIRenderer();
    
    bool Initialize();
    void Shutdown();
    
    // Begin a new UI frame
    void BeginFrame();
    
    // End the UI frame and render
    void EndFrame();
    
    // Draw text
    void DrawText(const std::string& text, float x, float y, float scale = 1.0f, glm::vec3 color = glm::vec3(1.0f));
    
    // Draw a rectangle
    void DrawRect(float x, float y, float width, float height, glm::vec4 color = glm::vec4(1.0f));
    
    // Draw a filled rectangle
    void DrawFilledRect(float x, float y, float width, float height, glm::vec4 color = glm::vec4(1.0f));
    
    // Draw a health bar
    void DrawHealthBar(float x, float y, float width, float height, float health, float maxHealth);
    
    // Draw debug info
    void DrawDebugInfo(float fps, const glm::vec3& playerPos, int enemyCount);
    
    // Set viewport size (call when window resizes)
    void SetViewportSize(int width, int height);

private:
    struct UIVertex {
        glm::vec2 position;
        glm::vec2 texCoord;
        glm::vec4 color;
    };
    
    GLuint m_shaderProgram;
    GLuint m_VAO, m_VBO, m_EBO;
    GLuint m_fontTexture;
    
    std::vector<UIVertex> m_vertices;
    std::vector<unsigned int> m_indices;
    
    int m_viewportWidth;
    int m_viewportHeight;
    
    // Create basic UI shaders
    bool CreateShaders();
    
    // Create font texture (basic ASCII bitmap font)
    bool CreateFontTexture();
    
    // Add a quad to the vertex buffer
    void AddQuad(float x, float y, float width, float height, const glm::vec4& color, 
                 float u0 = 0.0f, float v0 = 0.0f, float u1 = 1.0f, float v1 = 1.0f);
    
    // Flush the vertex buffer to GPU
    void FlushBatch();
};

} // namespace UI
} // namespace CudaGame
