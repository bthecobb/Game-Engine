#pragma once

#include <cstdint>

namespace CudaGame {
namespace Rendering {

// Usage is a bitmask to allow combinations (e.g., ColorAttachment | ShaderResource)
enum class TextureUsage : uint32_t {
    None           = 0,
    ColorAttachment= 1u << 0,
    DepthStencil   = 1u << 1,
    ShaderResource = 1u << 2,
};

inline TextureUsage operator|(TextureUsage a, TextureUsage b) {
    return static_cast<TextureUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline TextureUsage& operator|=(TextureUsage& a, TextureUsage b) {
    a = a | b; return a;
}
inline bool HasUsage(TextureUsage u, TextureUsage flag) {
    return (static_cast<uint32_t>(u) & static_cast<uint32_t>(flag)) != 0;
}

// API-agnostic texture formats chosen to map cleanly to GL/DX12/VK
// Keep this minimal; extend as needed
enum class TextureFormat : uint8_t {
    RGBA8,
    RGB8,
    RG16F,
    RGB16F,
    RGB32F,
    DEPTH24,
    DEPTH32F,
};

struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;      // immutable textures preferred
    uint32_t samples = 1;        // MSAA samples (1 = no MSAA)
    TextureFormat format = TextureFormat::RGBA8;
    TextureUsage usage = TextureUsage::ShaderResource;
    const char* debugName = nullptr; // optional
};

// Opaque handle that wraps underlying API resource
struct TextureHandle {
    uint64_t value = 0; // GL: GLuint, DX12: pointer/index; keep 64-bit for flexibility
    bool IsValid() const { return value != 0; }
};

} // namespace Rendering
} // namespace CudaGame
