#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

namespace CudaGame {
namespace Rendering {

class ShaderProgram {
public:
    ShaderProgram();
    ~ShaderProgram();

    // Shader compilation and linking
    bool LoadFromFiles(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath = "");
    bool LoadFromSource(const std::string& vertexSource, const std::string& fragmentSource, const std::string& geometrySource = "");
    
    // Shader usage
    void Use() const;
    bool IsValid() const { return m_programId != 0; }
    uint32_t GetProgramId() const { return m_programId; }

    // Uniform setters
    void SetBool(const std::string& name, bool value);
    void SetInt(const std::string& name, int value);
    void SetFloat(const std::string& name, float value);
    void SetVec2(const std::string& name, const glm::vec2& value);
    void SetVec3(const std::string& name, const glm::vec3& value);
    void SetVec4(const std::string& name, const glm::vec4& value);
    void SetMat3(const std::string& name, const glm::mat3& value);
    void SetMat4(const std::string& name, const glm::mat4& value);

    // Texture binding
    void SetTexture2D(const std::string& name, uint32_t textureId, int textureUnit = 0);

private:
    uint32_t m_programId = 0;
    mutable std::unordered_map<std::string, int> m_uniformLocationCache;

    // Helper functions
    uint32_t CompileShader(const std::string& source, uint32_t type);
    bool LinkProgram(uint32_t vertexShader, uint32_t fragmentShader, uint32_t geometryShader = 0);
    int GetUniformLocation(const std::string& name) const;
    std::string ReadFile(const std::string& filePath);
    void CheckCompileErrors(uint32_t shader, const std::string& type);
};

} // namespace Rendering
} // namespace CudaGame
