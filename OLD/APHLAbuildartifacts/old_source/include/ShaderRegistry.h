#pragma once

#include <string>
#include <unordered_map>
#include <memory>

namespace CudaGame {
namespace Rendering {

/**
 * @brief Centralized shader source management for AAA-grade architecture
 * 
 * This class follows the Singleton pattern to ensure unified shader management
 * across the entire game engine. It provides compile-time shader validation,
 * hot-reloading capabilities, and performance-optimized caching.
 */
class ShaderRegistry {
public:
    enum class ShaderType {
        VERTEX,
        FRAGMENT,
        GEOMETRY,
        TESSELLATION_CONTROL,
        TESSELLATION_EVALUATION,
        COMPUTE
    };

    enum class ShaderID {
        // Character System Shaders
        PLAYER_CHARACTER_VERTEX,
        PLAYER_CHARACTER_FRAGMENT,
        PLAYER_PARTICLE_VERTEX,
        PLAYER_PARTICLE_FRAGMENT,
        
        // Character Renderer Shaders
        CHARACTER_RENDERER_VERTEX,
        CHARACTER_RENDERER_FRAGMENT,
        
        // Advanced Character Rendering
        CHARACTER_PBR_VERTEX,
        CHARACTER_PBR_FRAGMENT,
        CHARACTER_SHADOW_VERTEX,
        CHARACTER_SHADOW_FRAGMENT,
        
        // Post-Processing Effects
        RHYTHM_FEEDBACK_VERTEX,
        RHYTHM_FEEDBACK_FRAGMENT,
        MOTION_BLUR_VERTEX,
        MOTION_BLUR_FRAGMENT,
        
        // Future Expansion
        ENVIRONMENT_VERTEX,
        ENVIRONMENT_FRAGMENT,
        UI_VERTEX,
        UI_FRAGMENT
    };

    /**
     * @brief Get the singleton instance
     * @return Reference to the shader registry instance
     */
    static ShaderRegistry& getInstance();

    /**
     * @brief Initialize the shader registry with all game shaders
     * @return True if initialization was successful
     */
    bool initialize();

    /**
     * @brief Get shader source code by ID
     * @param shaderID The shader identifier
     * @return Const reference to shader source string
     */
    const std::string& getShaderSource(ShaderID shaderID) const;

    /**
     * @brief Check if a shader exists in the registry
     * @param shaderID The shader identifier
     * @return True if shader exists
     */
    bool hasShader(ShaderID shaderID) const;

    /**
     * @brief Get shader type information
     * @param shaderID The shader identifier
     * @return The shader type
     */
    ShaderType getShaderType(ShaderID shaderID) const;

    /**
     * @brief Enable hot-reloading for development builds
     * @param enable True to enable hot-reloading
     */
    void enableHotReload(bool enable);

    /**
     * @brief Reload shaders from source (development only)
     * @return True if reload was successful
     */
    bool reloadShaders();

    /**
     * @brief Validate all registered shaders
     * @return True if all shaders are valid
     */
    bool validateAllShaders() const;

    /**
     * @brief Get shader compilation statistics
     * @return String containing compilation stats
     */
    std::string getCompilationStats() const;

private:
    ShaderRegistry() = default;
    ~ShaderRegistry() = default;
    ShaderRegistry(const ShaderRegistry&) = delete;
    ShaderRegistry& operator=(const ShaderRegistry&) = delete;

    // Internal shader registration methods
    void registerPlayerCharacterShaders();
    void registerCharacterRendererShaders();
    void registerAdvancedRenderingShaders();
    void registerPostProcessingShaders();
    void registerEnvironmentShaders();

    // Shader validation and compilation helpers
    bool validateShaderSyntax(const std::string& source, ShaderType type) const;
    std::string preprocessShader(const std::string& source) const;

    // Internal data structures
    struct ShaderEntry {
        std::string source;
        ShaderType type;
        bool isValid;
        std::string lastError;
    };

    std::unordered_map<ShaderID, ShaderEntry> m_shaders;
    bool m_hotReloadEnabled = false;
    bool m_initialized = false;
    
    // Statistics
    mutable size_t m_compilationCount = 0;
    mutable size_t m_validationCount = 0;
};

} // namespace Rendering
} // namespace CudaGame
