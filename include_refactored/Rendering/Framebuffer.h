#pragma once

#include <cstdint>
#include <vector>

namespace CudaGame {
namespace Rendering {

// Mock framebuffer class for deferred rendering pipeline
class Framebuffer {
public:
    Framebuffer();
    ~Framebuffer();
    
    // Initialize framebuffer with specified dimensions
    bool Initialize(uint32_t width, uint32_t height);
    
    // Bind this framebuffer for rendering
    void Bind();
    
    // Unbind framebuffer (bind default framebuffer)
    void Unbind();
    
    // Get framebuffer object ID
    uint32_t GetFBO() const { return m_fbo; }
    
    // Get color attachment texture IDs
    uint32_t GetColorAttachment(int index = 0) const;
    uint32_t GetDepthAttachment() const { return m_depthTexture; }
    
    // Get framebuffer dimensions
    uint32_t GetWidth() const { return m_width; }
    uint32_t GetHeight() const { return m_height; }

private:
    uint32_t m_fbo = 0;
    uint32_t m_depthTexture = 0;
    std::vector<uint32_t> m_colorTextures;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
};

} // namespace Rendering
} // namespace CudaGame
