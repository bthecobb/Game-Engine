#pragma once

#include "Rendering/RenderBackend.h"
#include <glad/glad.h>

namespace CudaGame {
namespace Rendering {

class Framebuffer;

class GLRenderBackend : public IRenderBackend {
public:
    bool Initialize() override { return gladLoadGL() != 0; }

    void BeginFrame(const glm::vec4& clearColor, int width, int height) override {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDrawBuffer(GL_BACK);
        glViewport(0, 0, width, height);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);
        glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void BindFramebuffer(Framebuffer* fb) override;
    void UnbindFramebuffer() override {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void BlitDepth(Framebuffer* fb, int width, int height) override;

    // Resource creation
    bool CreateTexture(const TextureDesc& desc, TextureHandle& outHandle) override;
    void DestroyTexture(TextureHandle& handle) override;
};

} // namespace Rendering
} // namespace CudaGame