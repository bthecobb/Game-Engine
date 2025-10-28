#include "ShaderRegistry.h"
#include <iostream>
#include <sstream>
#include <regex>

namespace CudaGame {
namespace Rendering {

// Singleton instance access
ShaderRegistry& ShaderRegistry::getInstance() {
    static ShaderRegistry instance;
    return instance;
}

bool ShaderRegistry::initialize() {
    if (m_initialized) {
        return true;
    }

    try {
        // Register all shader systems
        registerPlayerCharacterShaders();
        registerCharacterRendererShaders();
        registerAdvancedRenderingShaders();
        registerPostProcessingShaders();
        registerEnvironmentShaders();

        // Validate all registered shaders
        if (!validateAllShaders()) {
            std::cerr << "ShaderRegistry: Shader validation failed during initialization" << std::endl;
            return false;
        }

        m_initialized = true;
        std::cout << "ShaderRegistry: Successfully initialized with " << m_shaders.size() << " shaders" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ShaderRegistry: Initialization failed - " << e.what() << std::endl;
        return false;
    }
}

const std::string& ShaderRegistry::getShaderSource(ShaderID shaderID) const {
    static const std::string emptyString;
    
    auto it = m_shaders.find(shaderID);
    if (it != m_shaders.end()) {
        m_compilationCount++;
        return it->second.source;
    }
    
    std::cerr << "ShaderRegistry: Shader not found for ID: " << static_cast<int>(shaderID) << std::endl;
    return emptyString;
}

bool ShaderRegistry::hasShader(ShaderID shaderID) const {
    return m_shaders.find(shaderID) != m_shaders.end();
}

ShaderRegistry::ShaderType ShaderRegistry::getShaderType(ShaderID shaderID) const {
    auto it = m_shaders.find(shaderID);
    if (it != m_shaders.end()) {
        return it->second.type;
    }
    return ShaderType::VERTEX; // Default fallback
}

void ShaderRegistry::enableHotReload(bool enable) {
    m_hotReloadEnabled = enable;
    if (enable) {
        std::cout << "ShaderRegistry: Hot-reload enabled for development" << std::endl;
    }
}

bool ShaderRegistry::reloadShaders() {
    if (!m_hotReloadEnabled) {
        return false;
    }
    
    // In a full implementation, this would reload from files
    std::cout << "ShaderRegistry: Hot-reload not implemented in this version" << std::endl;
    return true;
}

bool ShaderRegistry::validateAllShaders() const {
    m_validationCount++;
    
    for (const auto& pair : m_shaders) {
        if (!pair.second.isValid) {
            std::cerr << "ShaderRegistry: Invalid shader found - ID: " << static_cast<int>(pair.first) << std::endl;
            return false;
        }
    }
    
    return true;
}

std::string ShaderRegistry::getCompilationStats() const {
    std::stringstream ss;
    ss << "ShaderRegistry Statistics:\n";
    ss << "  Total Shaders: " << m_shaders.size() << "\n";
    ss << "  Compilation Requests: " << m_compilationCount << "\n";
    ss << "  Validation Requests: " << m_validationCount << "\n";
    ss << "  Hot-reload Enabled: " << (m_hotReloadEnabled ? "Yes" : "No") << "\n";
    return ss.str();
}

void ShaderRegistry::registerPlayerCharacterShaders() {
    // Enhanced Player character vertex shader with advanced features
    m_shaders[ShaderID::PLAYER_CHARACTER_VERTEX] = {
        R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float rhythmPulse;
uniform float animationBend;
uniform float time;

out vec3 FragPos;
out vec3 Normal;
out vec3 vertexColor;
out float rhythmIntensity;

void main() {
    // Apply rhythm-based scaling
    vec3 scaledPos = aPos * (1.0 + rhythmPulse * 0.1);
    
    // Apply animation bending for organic movement
    scaledPos.x += sin(animationBend + aPos.y * 2.0) * 0.05;
    scaledPos.z += cos(animationBend + aPos.y * 1.5) * 0.03;
    
    // Apply time-based micro-movements for life-like feel
    scaledPos.y += sin(time * 3.0 + aPos.x * 10.0) * 0.01;
    
    vec4 worldPos = model * vec4(scaledPos, 1.0);
    
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    Normal = mat3(transpose(inverse(model))) * aNormal;
    vertexColor = aColor;
    rhythmIntensity = rhythmPulse;
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    // Enhanced Player character fragment shader with PBR-like lighting
    m_shaders[ShaderID::PLAYER_CHARACTER_FRAGMENT] = {
        R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec3 vertexColor;
in float rhythmIntensity;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;
uniform float time;

void main() {
    // Enhanced ambient lighting
    float ambientStrength = 0.3 + rhythmIntensity * 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    // Advanced diffuse lighting with normal mapping simulation
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Enhanced specular with variable roughness
    float specularStrength = 0.5 + rhythmIntensity * 0.2;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    // Dynamic rhythm-based glow
    vec3 rhythmGlow = vec3(0.2, 0.5, 1.0) * rhythmIntensity * 0.4;
    rhythmGlow += vec3(sin(time * 2.0), cos(time * 1.5), sin(time * 3.0)) * 0.1 * rhythmIntensity;
    
    // Combine all lighting components
    vec3 result = (ambient + diffuse + specular) * vertexColor + rhythmGlow;
    
    // Apply tone mapping for HDR-like appearance
    result = result / (result + vec3(1.0));
    
    FragColor = vec4(result, 1.0);
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };

    // Enhanced particle system vertex shader
    m_shaders[ShaderID::PLAYER_PARTICLE_VERTEX] = {
        R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;
layout (location = 3) in float aLife;

uniform mat4 view;
uniform mat4 projection;
uniform float time;

out vec3 particleColor;
out float particleLife;

void main() {
    // Apply time-based particle movement variations
    vec3 pos = aPos;
    pos.y += sin(time * 5.0 + aPos.x * 10.0) * 0.02 * aLife;
    pos.x += cos(time * 3.0 + aPos.z * 8.0) * 0.01 * aLife;
    
    gl_Position = projection * view * vec4(pos, 1.0);
    
    // Dynamic particle size based on life and distance
    float dynamicSize = aSize * aLife * (1.0 + sin(time * 4.0) * 0.2);
    gl_PointSize = dynamicSize;
    
    particleColor = aColor;
    particleLife = aLife;
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    // Enhanced particle fragment shader with advanced blending
    m_shaders[ShaderID::PLAYER_PARTICLE_FRAGMENT] = {
        R"(
#version 330 core
in vec3 particleColor;
in float particleLife;

out vec4 FragColor;

uniform float time;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Soft particle edges with life-based falloff
    if (dist > 0.5) discard;
    
    // Create smooth circular particle with soft edges
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    alpha *= particleLife;
    
    // Add subtle sparkle effect
    float sparkle = sin(time * 8.0 + dist * 20.0) * 0.3 + 0.7;
    
    // Enhanced color with energy-based intensity
    vec3 finalColor = particleColor * sparkle * (0.8 + particleLife * 0.4);
    
    FragColor = vec4(finalColor, alpha);
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };
}

void ShaderRegistry::registerCharacterRendererShaders() {
    // Character renderer vertex shader (simplified for basic rendering)
    m_shaders[ShaderID::CHARACTER_RENDERER_VERTEX] = {
        R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float animationOffset;
uniform float speedMultiplier;
uniform float time;

out vec3 vertexColor;
out float animationFactor;

void main() {
    vec3 pos = aPos;
    
    // Animation system for character renderer
    if (abs(aPos.y) > 0.1) { // Don't animate the base/feet
        pos.x += sin(animationOffset * 10.0 + time) * 0.1 * speedMultiplier;
        pos.z += cos(animationOffset * 8.0 + time) * 0.05 * speedMultiplier;
        pos.y += sin(animationOffset * 12.0 + time) * 0.02 * speedMultiplier;
    }
    
    gl_Position = projection * view * model * vec4(pos, 1.0);
    vertexColor = aColor;
    animationFactor = speedMultiplier;
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    // Character renderer fragment shader with enhanced effects
    m_shaders[ShaderID::CHARACTER_RENDERER_FRAGMENT] = {
        R"(
#version 330 core
in vec3 vertexColor;
in float animationFactor;

out vec4 FragColor;

uniform float glowIntensity;
uniform float time;

void main() {
    // Base color with animation-based modifications
    vec3 baseColor = vertexColor;
    
    // Add movement-based glow effect
    vec3 glowColor = vec3(0.3, 0.7, 1.0) * glowIntensity;
    glowColor += vec3(sin(time * 2.0), cos(time * 1.5), sin(time * 2.5)) * 0.1 * animationFactor;
    
    // Combine base color with glow
    vec3 finalColor = baseColor + glowColor * 0.4;
    
    // Apply subtle pulsing effect during movement
    float pulse = 1.0 + sin(time * 4.0) * 0.1 * animationFactor;
    finalColor *= pulse;
    
    // Ensure color doesn't exceed reasonable bounds
    finalColor = min(finalColor, vec3(1.5));
    
    FragColor = vec4(finalColor, 1.0);
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };
}

void ShaderRegistry::registerAdvancedRenderingShaders() {
    // PBR Character rendering (future implementation)
    m_shaders[ShaderID::CHARACTER_PBR_VERTEX] = {
        R"(
#version 330 core
// PBR Character Vertex Shader - Future Implementation
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aTangent;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    m_shaders[ShaderID::CHARACTER_PBR_FRAGMENT] = {
        R"(
#version 330 core
// PBR Character Fragment Shader - Future Implementation
out vec4 FragColor;

void main() {
    FragColor = vec4(0.5, 0.5, 0.5, 1.0); // Placeholder
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };

    // Shadow mapping shaders (future implementation)
    m_shaders[ShaderID::CHARACTER_SHADOW_VERTEX] = {
        R"(
#version 330 core
// Shadow Mapping Vertex Shader - Future Implementation
layout (location = 0) in vec3 aPos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main() {
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    m_shaders[ShaderID::CHARACTER_SHADOW_FRAGMENT] = {
        R"(
#version 330 core
// Shadow Mapping Fragment Shader - Future Implementation
void main() {
    // Depth values are automatically written to gl_FragDepth
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };
}

void ShaderRegistry::registerPostProcessingShaders() {
    // Rhythm feedback post-processing
    m_shaders[ShaderID::RHYTHM_FEEDBACK_VERTEX] = {
        R"(
#version 330 core
// Rhythm Feedback Post-Processing Vertex Shader
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    m_shaders[ShaderID::RHYTHM_FEEDBACK_FRAGMENT] = {
        R"(
#version 330 core
// Rhythm Feedback Post-Processing Fragment Shader
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform float rhythmIntensity;
uniform float time;

void main() {
    vec3 color = texture(screenTexture, TexCoord).rgb;
    
    // Apply rhythm-based screen effects
    float pulse = sin(time * 4.0) * 0.5 + 0.5;
    color += vec3(0.1, 0.3, 0.8) * rhythmIntensity * pulse * 0.2;
    
    FragColor = vec4(color, 1.0);
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };

    // Motion blur post-processing (future)
    m_shaders[ShaderID::MOTION_BLUR_VERTEX] = {
        R"(
#version 330 core
// Motion Blur Vertex Shader - Future Implementation
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    m_shaders[ShaderID::MOTION_BLUR_FRAGMENT] = {
        R"(
#version 330 core
// Motion Blur Fragment Shader - Future Implementation
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;

void main() {
    FragColor = texture(screenTexture, TexCoord);
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };
}

void ShaderRegistry::registerEnvironmentShaders() {
    // Environment rendering shaders (future)
    m_shaders[ShaderID::ENVIRONMENT_VERTEX] = {
        R"(
#version 330 core
// Environment Vertex Shader - Future Implementation
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    m_shaders[ShaderID::ENVIRONMENT_FRAGMENT] = {
        R"(
#version 330 core
// Environment Fragment Shader - Future Implementation
out vec4 FragColor;

void main() {
    FragColor = vec4(0.1, 0.2, 0.3, 1.0); // Placeholder
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };

    // UI rendering shaders (future)
    m_shaders[ShaderID::UI_VERTEX] = {
        R"(
#version 330 core
// UI Vertex Shader - Future Implementation
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
        )",
        ShaderType::VERTEX,
        true,
        ""
    };

    m_shaders[ShaderID::UI_FRAGMENT] = {
        R"(
#version 330 core
// UI Fragment Shader - Future Implementation
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uiTexture;
uniform vec4 uiColor;

void main() {
    FragColor = texture(uiTexture, TexCoord) * uiColor;
}
        )",
        ShaderType::FRAGMENT,
        true,
        ""
    };
}

bool ShaderRegistry::validateShaderSyntax(const std::string& source, ShaderType type) const {
    // Basic syntax validation - in a full implementation, this would use
    // actual OpenGL shader compilation or a dedicated GLSL parser
    
    // Check for required GLSL version
    if (source.find("#version") == std::string::npos) {
        return false;
    }
    
    // Check for main function
    if (source.find("void main()") == std::string::npos) {
        return false;
    }
    
    // Type-specific validation
    switch (type) {
        case ShaderType::VERTEX:
            return source.find("gl_Position") != std::string::npos;
        case ShaderType::FRAGMENT:
            return source.find("FragColor") != std::string::npos || 
                   source.find("gl_FragColor") != std::string::npos;
        default:
            return true;
    }
}

std::string ShaderRegistry::preprocessShader(const std::string& source) const {
    // Basic preprocessing - in a full implementation, this would handle
    // #include directives, conditional compilation, etc.
    return source;
}

} // namespace Rendering
} // namespace CudaGame
