#pragma once

#ifdef _WIN32
#include <d3d12.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

namespace CudaGame {
namespace Rendering {

// Shader stages
enum class ShaderStage : uint8_t {
    Vertex,
    Pixel,
    Geometry,
    Hull,
    Domain,
    Compute
};

// Shader compilation flags
enum class ShaderCompileFlags : uint32_t {
    None = 0,
    Debug = 1 << 0,           // Enable debug info (/Zi)
    SkipOptimization = 1 << 1, // Skip optimization (/Od)
    WarningsAsErrors = 1 << 2, // Treat warnings as errors (/WX)
    EnableStrictMode = 1 << 3, // Enable strict mode
    IEEE = 1 << 4              // IEEE strictness
};

inline ShaderCompileFlags operator|(ShaderCompileFlags a, ShaderCompileFlags b) {
    return static_cast<ShaderCompileFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool HasFlag(ShaderCompileFlags flags, ShaderCompileFlags flag) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
}

// Shader bytecode container
struct ShaderBytecode {
    std::vector<uint8_t> data;
    ShaderStage stage;
    std::wstring entryPoint;
    std::wstring profile; // e.g., "vs_6_0", "ps_6_0"
    
    bool IsValid() const { return !data.empty(); }
    const void* GetBufferPointer() const { return data.data(); }
    size_t GetBufferSize() const { return data.size(); }
};

// Shader source info for hot-reloading
struct ShaderSourceInfo {
    std::filesystem::path filePath;
    std::filesystem::file_time_type lastWriteTime;
    std::vector<std::filesystem::path> includes; // Track dependencies
};

// AAA-grade shader manager with compilation, caching, and hot-reload
class ShaderManager {
public:
    ShaderManager();
    ~ShaderManager();

    // Initialize DXC compiler
    bool Initialize();
    void Shutdown();

    // Compile shader from file
    bool CompileFromFile(
        const std::wstring& filePath,
        const std::wstring& entryPoint,
        ShaderStage stage,
        ShaderCompileFlags flags,
        ShaderBytecode& outBytecode,
        std::string& outErrorMsg
    );

    // Compile shader from source string (for runtime-generated shaders)
    bool CompileFromSource(
        const std::wstring& source,
        const std::wstring& sourceName,
        const std::wstring& entryPoint,
        ShaderStage stage,
        ShaderCompileFlags flags,
        ShaderBytecode& outBytecode,
        std::string& outErrorMsg
    );

    // Get cached shader (returns nullptr if not found)
    const ShaderBytecode* GetCachedShader(const std::wstring& key) const;

    // Cache management
    void CacheShader(const std::wstring& key, const ShaderBytecode& bytecode);
    void ClearCache();
    size_t GetCacheSize() const { return m_shaderCache.size(); }

    // Hot-reload support (AAA workflow feature)
    void EnableHotReload(bool enable) { m_hotReloadEnabled = enable; }
    bool IsHotReloadEnabled() const { return m_hotReloadEnabled; }
    void CheckForModifiedShaders(); // Call per frame to detect changes
    void RegisterShaderForHotReload(const std::wstring& key, const std::filesystem::path& filePath);

    // Set include directories for shader includes
    void AddIncludeDirectory(const std::filesystem::path& dir);
    void ClearIncludeDirectories();

    // Utility: Generate cache key from shader parameters
    static std::wstring GenerateCacheKey(
        const std::wstring& filePath,
        const std::wstring& entryPoint,
        ShaderStage stage,
        ShaderCompileFlags flags
    );

private:
    template<typename T>
    using ComPtr = Microsoft::WRL::ComPtr<T>;

    // DXC compiler interface (modern HLSL compiler, not legacy FXC)
    ComPtr<IDxcUtils> m_dxcUtils;
    ComPtr<IDxcCompiler3> m_dxcCompiler;
    ComPtr<IDxcIncludeHandler> m_includeHandler;

    // Shader cache (key = GenerateCacheKey result)
    std::unordered_map<std::wstring, ShaderBytecode> m_shaderCache;

    // Hot-reload tracking
    std::unordered_map<std::wstring, ShaderSourceInfo> m_sourceTracking;
    bool m_hotReloadEnabled = false;

    // Include directories
    std::vector<std::filesystem::path> m_includeDirectories;

    bool m_initialized = false;

    // Helper: Convert ShaderStage to HLSL profile (e.g., Vertex -> "vs_6_0")
    std::wstring GetShaderProfile(ShaderStage stage, const wchar_t* shaderModel = L"6_0") const;

    // Helper: Build DXC arguments
    std::vector<LPCWSTR> BuildCompilerArguments(
        const std::wstring& entryPoint,
        const std::wstring& profile,
        ShaderCompileFlags flags
    );
};

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
