#include "Rendering/Backends/GLRenderBackend.h"
#include "Rendering/Framebuffer.h"
#include "Rendering/BackendResources.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

namespace CudaGame {
namespace Rendering {

static void MapGLFormat(TextureFormat fmt, GLint& internalFormat, GLenum& format, GLenum& type, bool& isDepth) {
    isDepth = false;
    switch (fmt) {
        case TextureFormat::RGBA8:
            internalFormat = GL_RGBA8; format = GL_RGBA; type = GL_UNSIGNED_BYTE; break;
        case TextureFormat::RGB8:
            internalFormat = GL_RGB8; format = GL_RGB; type = GL_UNSIGNED_BYTE; break;
        case TextureFormat::RG16F:
            internalFormat = GL_RG16F; format = GL_RG; type = GL_FLOAT; break;
        case TextureFormat::RGB16F:
            internalFormat = GL_RGB16F; format = GL_RGB; type = GL_FLOAT; break;
        case TextureFormat::RGB32F:
            internalFormat = GL_RGB32F; format = GL_RGB; type = GL_FLOAT; break;
        case TextureFormat::DEPTH24:
            internalFormat = GL_DEPTH_COMPONENT24; format = GL_DEPTH_COMPONENT; type = GL_UNSIGNED_INT; isDepth = true; break;
        case TextureFormat::DEPTH32F:
            internalFormat = GL_DEPTH_COMPONENT32F; format = GL_DEPTH_COMPONENT; type = GL_FLOAT; isDepth = true; break;
        default:
            internalFormat = GL_RGBA8; format = GL_RGBA; type = GL_UNSIGNED_BYTE; break;
    }
}

void GLRenderBackend::BindFramebuffer(Framebuffer* fb) {
    if (!fb) { glBindFramebuffer(GL_FRAMEBUFFER, 0); return; }
    fb->Bind();
}

void GLRenderBackend::BlitDepth(Framebuffer* fb, int width, int height) {
    if (!fb) return;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb->GetFBO());
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glDrawBuffer(GL_BACK);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool GLRenderBackend::CreateTexture(const TextureDesc& desc, TextureHandle& outHandle) {
    if (desc.width == 0 || desc.height == 0) return false;

    GLuint tex = 0;
    glGenTextures(1, &tex);
    if (tex == 0) return false;

    GLint internalFormat = GL_RGBA8; GLenum format = GL_RGBA; GLenum type = GL_UNSIGNED_BYTE; bool isDepth = false;
    MapGLFormat(desc.format, internalFormat, format, type, isDepth);

    glBindTexture(GL_TEXTURE_2D, tex);

    // Prefer immutable storage when possible
    if (glTexStorage2D) {
        glTexStorage2D(GL_TEXTURE_2D, desc.mipLevels, internalFormat, desc.width, desc.height);
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, desc.width, desc.height, 0, format, type, nullptr);
    }

    // Basic sampler state (can be refined per-usage later)
    GLenum minFilter = (desc.mipLevels > 1) ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (desc.mipLevels > 1 && !glTexStorage2D) {
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    // Optional debug label if KHR_debug is available
    if (desc.debugName) {
#if defined(GL_KHR_debug)
        glObjectLabel(GL_TEXTURE, tex, -1, desc.debugName);
#endif
    }

    outHandle.value = static_cast<uint64_t>(tex);
    return true;
}

void GLRenderBackend::DestroyTexture(TextureHandle& handle) {
    if (!handle.IsValid()) return;
    GLuint tex = static_cast<GLuint>(handle.value);
    glDeleteTextures(1, &tex);
    handle.value = 0;
}

} // namespace Rendering
} // namespace CudaGame
