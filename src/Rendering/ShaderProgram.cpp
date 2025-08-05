#include "Rendering/ShaderProgram.h"
#include <glad/glad.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace CudaGame {
namespace Rendering {

ShaderProgram::ShaderProgram() : m_programId(0) {}

ShaderProgram::~ShaderProgram() {
    if (m_programId != 0) {
        glDeleteProgram(m_programId);
    }
}

bool ShaderProgram::LoadFromFiles(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath) {
    std::string vertexSource = ReadFile(vertexPath);
    std::string fragmentSource = ReadFile(fragmentPath);
    std::string geometrySource = geometryPath.empty() ? "" : ReadFile(geometryPath);

    if (vertexSource.empty() || fragmentSource.empty()) {
        std::cerr << "[ShaderProgram] ERROR: Failed to read shader files." << std::endl;
        std::cerr << "Vertex path: " << vertexPath << " - Empty: " << vertexSource.empty() << std::endl;
        std::cerr << "Fragment path: " << fragmentPath << " - Empty: " << fragmentSource.empty() << std::endl;
        return false;
    }

    return LoadFromSource(vertexSource, fragmentSource, geometrySource);
}

bool ShaderProgram::LoadFromSource(const std::string& vertexSource, const std::string& fragmentSource, const std::string& geometrySource) {
    uint32_t vertexShader = CompileShader(vertexSource, GL_VERTEX_SHADER);
    if (!vertexShader) return false;

    uint32_t fragmentShader = CompileShader(fragmentSource, GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        glDeleteShader(vertexShader);
        return false;
    }

    uint32_t geometryShader = 0;
    if (!geometrySource.empty()) {
        geometryShader = CompileShader(geometrySource, GL_GEOMETRY_SHADER);
        if (!geometryShader) {
            glDeleteShader(vertexShader);
            glDeleteShader(fragmentShader);
            return false;
        }
    }

    if (!LinkProgram(vertexShader, fragmentShader, geometryShader)) {
        return false;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    if (geometryShader != 0) {
        glDeleteShader(geometryShader);
    }
    
    std::cout << "[ShaderProgram] Shader program " << m_programId << " loaded successfully." << std::endl;
    return true;
}

void ShaderProgram::Use() const {
    glUseProgram(m_programId);
}

void ShaderProgram::SetBool(const std::string& name, bool value) {
    glUniform1i(GetUniformLocation(name), (int)value);
}

void ShaderProgram::SetInt(const std::string& name, int value) {
    glUniform1i(GetUniformLocation(name), value);
}

void ShaderProgram::SetFloat(const std::string& name, float value) {
    glUniform1f(GetUniformLocation(name), value);
}

void ShaderProgram::SetVec2(const std::string& name, const glm::vec2& value) {
    glUniform2fv(GetUniformLocation(name), 1, &value[0]);
}

void ShaderProgram::SetVec3(const std::string& name, const glm::vec3& value) {
    glUniform3fv(GetUniformLocation(name), 1, &value[0]);
}

void ShaderProgram::SetVec4(const std::string& name, const glm::vec4& value) {
    glUniform4fv(GetUniformLocation(name), 1, &value[0]);
}

void ShaderProgram::SetMat3(const std::string& name, const glm::mat3& value) {
    glUniformMatrix3fv(GetUniformLocation(name), 1, GL_FALSE, &value[0][0]);
}

void ShaderProgram::SetMat4(const std::string& name, const glm::mat4& value) {
    glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, &value[0][0]);
}

void ShaderProgram::SetTexture2D(const std::string& name, uint32_t textureId, int textureUnit) {
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D, textureId);
    SetInt(name, textureUnit);
}

uint32_t ShaderProgram::CompileShader(const std::string& source, uint32_t type) {
    if(source.empty()) return 0;
    
    const char* src = source.c_str();
    uint32_t shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    CheckCompileErrors(shader, (type == GL_VERTEX_SHADER ? "VERTEX" : (type == GL_FRAGMENT_SHADER ? "FRAGMENT" : "GEOMETRY")));
    
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool ShaderProgram::LinkProgram(uint32_t vertexShader, uint32_t fragmentShader, uint32_t geometryShader) {
    m_programId = glCreateProgram();
    glAttachShader(m_programId, vertexShader);
    glAttachShader(m_programId, fragmentShader);
    if (geometryShader != 0) {
        glAttachShader(m_programId, geometryShader);
    }
    glLinkProgram(m_programId);
    CheckCompileErrors(m_programId, "PROGRAM");

    int success;
    glGetProgramiv(m_programId, GL_LINK_STATUS, &success);
    if (!success) {
        glDeleteProgram(m_programId);
        m_programId = 0;
        return false;
    }
    return true;
}

int ShaderProgram::GetUniformLocation(const std::string& name) const {
    auto it = m_uniformLocationCache.find(name);
    if (it != m_uniformLocationCache.end()) {
        return it->second;
    }

    int location = glGetUniformLocation(m_programId, name.c_str());
    //if (location == -1) {
    //   std::cerr << "[ShaderProgram] Warning: uniform '" << name << "' not found." << std::endl;
    //}
    m_uniformLocationCache[name] = location;
    return location;
}

std::string ShaderProgram::ReadFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[ShaderProgram] Failed to open file: " << filePath << std::endl;
        std::cerr << "Current directory: " << std::filesystem::current_path() << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}

void ShaderProgram::CheckCompileErrors(uint32_t shader, const std::string& type) {
    int success;
    char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "[ShaderProgram] ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "[ShaderProgram] ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}

} // namespace Rendering
} // namespace CudaGame
