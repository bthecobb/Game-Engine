#pragma once

#include <memory>
#include <glm/glm.hpp>
#include "Rendering/BackendResources.h"

namespace CudaGame {
namespace Rendering {

class Framebuffer;

// Minimal render backend interface (Phase 1 -> extended with resource creation)
class IRenderBackend {
public:
    virtual ~IRenderBackend() = default;

    // Initialize backend (create device/context if needed)
    virtual bool Initialize() = 0;

    // Begin a new frame on the default framebuffer
    virtual void BeginFrame(const glm::vec4& clearColor, int width, int height) = 0;

    // Bind/unbind a framebuffer for offscreen passes
    virtual void BindFramebuffer(Framebuffer* fb) = 0;
    virtual void UnbindFramebuffer() = 0;

    // Blit depth from an offscreen framebuffer to default framebuffer
    virtual void BlitDepth(Framebuffer* fb, int width, int height) = 0;

    // Resource creation (stubs for now; implemented in GL, will map to DX12 later)
    virtual bool CreateTexture(const TextureDesc& desc, TextureHandle& outHandle) = 0;
    virtual void DestroyTexture(TextureHandle& handle) = 0;
};

} // namespace Rendering
} // namespace CudaGame
